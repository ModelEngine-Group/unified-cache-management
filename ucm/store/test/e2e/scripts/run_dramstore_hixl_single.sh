#!/usr/bin/env bash
# Single-node DramStore bench over hixl (HCCS) on one Ascend A2 box.
#
# One drampool daemon (NPU POOL_DEVICE) + one DramStore client (NPU
# CLIENT_DEVICE), both 127.0.0.1, hixl transport. The client's KV cache is
# NPU device memory (aclrtMalloc); host-memory transfers (reply/pool) go
# through the host-sync overlay's queued TransferSync.
#
# Run inside the cann-8.5.1 container (hixl 8.5.0 installed) on the build host:
#   bash ucm/store/test/e2e/scripts/run_dramstore_hixl_single.sh
#   POOL_DEVICE=4 CLIENT_DEVICE=5 REQUEST_SIZE=4 bash ucm/store/test/e2e/scripts/run_dramstore_hixl_single.sh
#
# Prereq: source the cann set_env before running (ASCEND_HOME_PATH etc.):
#   source /usr/local/Ascend/cann-8.5.1/set_env.sh
set -euo pipefail
# export ASCEND_SLOG_PRINT_TO_STDOUT=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# repo root: ucm/store/test/e2e/scripts -> 5 levels up
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
[[ -f "$REPO_ROOT/CMakeLists.txt" && -d "$REPO_ROOT/ucm" ]] || {
    echo "[err] repo root not found from $SCRIPT_DIR (resolved $REPO_ROOT)" >&2
    exit 1
}
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build}"

POOL_DEVICE="${POOL_DEVICE:-4}"
CLIENT_DEVICE="${CLIENT_DEVICE:-5}"
POOL_GB="${POOL_GB:-1}"
SHARD_SIZE="${SHARD_SIZE:-4096}"
REQUEST_SIZE="${REQUEST_SIZE:-1}"          # blocks per dump; 4 verified on A2
BATCH_NUMBER="${BATCH_NUMBER:-32}"
DRAMPOL_PORT="${DRAMPOL_PORT:-9000}"
DRAMPOL_ONE_SIDED="${DRAMPOL_ONE_SIDED:-4501}"
CLIENT_CTL_PORT="${CLIENT_CTL_PORT:-4702}"
CLIENT_MGR_PORT="${CLIENT_MGR_PORT:-4502}"

# cann env (best-effort: honor an already-sourced env)
if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    CANN_SET_ENV="${CANN_SET_ENV:-/usr/local/Ascend/cann-8.5.1/set_env.sh}"
    [[ -f "$CANN_SET_ENV" ]] && source "$CANN_SET_ENV"
fi

STAGE_DIR="${STAGE_DIR:-/tmp}"
DRAMPOOL="$BUILD_DIR/ucm/store/dram/drampool"
DRAMSTORE_SO="$BUILD_DIR/ucm/store/dram/libdramstore.so"
P2P_SO="$BUILD_DIR/ucm/transport/p2p/libucm_p2p_transport.so"
UCMPIPELINE_SO_DIR="$BUILD_DIR/ucm/store/pipeline"
BENCH="$REPO_ROOT/ucm/store/test/e2e/dramstore_bench_test.py"

echo "[hixl-single] drampool=device$POOL_DEVICE client=device$CLIENT_DEVICE hixl/HCCS 127.0.0.1"

# --- build if any artifact is missing ---
need_build=0
for f in "$DRAMPOOL" "$DRAMSTORE_SO" "$P2P_SO" "$BENCH"; do
    [[ -e "$f" ]] || { need_build=1; break; }
done
if [[ ! -d "$UCMPIPELINE_SO_DIR" || -z "$(ls -A "$UCMPIPELINE_SO_DIR"/*.so 2>/dev/null)" ]]; then
    need_build=1
fi
if [[ "$need_build" -eq 1 ]]; then
    echo "[hixl-single] building (RUNTIME_ENVIRONMENT=ascend BUILD_UCM_DRAMPOOL=ON)..."
    cmake -B "$BUILD_DIR" -S "$REPO_ROOT" \
        -DRUNTIME_ENVIRONMENT=ascend -DBUILD_UCM_DRAMPOOL=ON -DCMAKE_BUILD_TYPE=Release >/dev/null
    cmake --build "$BUILD_DIR" -j"$(nproc)" \
        -t dramstore ucm_p2p_transport ucmpipelinestore drampool
fi

# --- stage runtime artifacts to one dir (bench looks for libdramstore.so next to the pybind .so) ---
mkdir -p "$STAGE_DIR"
cp -f "$DRAMSTORE_SO" "$STAGE_DIR/"
cp -f "$P2P_SO" "$STAGE_DIR/"
cp -f "$UCMPIPELINE_SO_DIR"/ucmpipelinestore*.so "$STAGE_DIR/"
cp -f "$DRAMPOOL" "$STAGE_DIR/drampool"
cp -f "$BENCH" "$STAGE_DIR/dramstore_bench_test.py"

# --- drampool yaml: drampool + single client (worker) endpoints ---
YAML="$STAGE_DIR/drampool_hixl_single.yaml"
cat > "$YAML" <<YAML
transport:
  device_ids:
    - ${POOL_DEVICE}
  endpoints:
    - two_sided: "127.0.0.1:${DRAMPOL_PORT}"
      one_sided: "127.0.0.1:${DRAMPOL_ONE_SIDED}"
    - two_sided: "127.0.0.1:${CLIENT_CTL_PORT}"
      one_sided: "127.0.0.1:${CLIENT_MGR_PORT}"
    - two_sided: "127.0.0.1:$((CLIENT_CTL_PORT + 1))"
      one_sided: "127.0.0.1:$((CLIENT_MGR_PORT + 1))"
health:
  port: 0
queue:
  request_depth: 65536
  completion_depth: 65536
request_receiver:
  idle_wait_us: 100
poller:
  pending_depth: 64
flag_buffer:
  capacity_mb: 64
  slot_size_bytes: 64
gc:
  enabled: true
  interval_ms: 1000
metadata:
  periodic_eviction_policy: TTL
  deep_eviction_policy: POSITION
  lease_time_ms: 5000
  default_evict_ratio: 0.0
  evict_period_ms: 31536000000
operation:
  timeout_ms: 5000
logger:
  level: info
  dir: ${STAGE_DIR}/dp_logs
  max_files: 10
  max_size_mb: 5
YAML

export LD_LIBRARY_PATH="$STAGE_DIR:${LD_LIBRARY_PATH:-}"

# --- cleanup any stale drampool (use -x, NOT -f: -f matches this script's own cmdline) ---
pkill -9 -x drampool 2>/dev/null || true
sleep 1
rm -f "$STAGE_DIR/dp0.log"
mkdir -p "$STAGE_DIR/dp_logs"

# --- start drampool (hixl, POOL_DEVICE) ---
echo "[hixl-single] starting drampool on device${POOL_DEVICE}..."
nohup "$STAGE_DIR/drampool" \
    --addr "127.0.0.1:${DRAMPOL_PORT}" --nics "davinci${POOL_DEVICE}" \
    --pool-size-gb "$POOL_GB" \
    --kvcache-block-sizes "$SHARD_SIZE" --config "$YAML" \
    > "$STAGE_DIR/dp0.log" 2>&1 &
disown
DRAMPOOL_PID=$!

# --- wait for DramPool ready (or early exit) ---
ready=0
for _ in $(seq 1 30); do
    grep -q "DramPool service ready" "$STAGE_DIR/dp0.log" 2>/dev/null && { ready=1; break; }
    grep -q "EXIT=" "$STAGE_DIR/dp0.log" 2>/dev/null && { echo "[err] drampool exited early"; tail -20 "$STAGE_DIR/dp0.log"; exit 1; }
    sleep 1
done
if [[ "$ready" -ne 1 ]]; then
    echo "[err] drampool not ready in 30s"; tail -20 "$STAGE_DIR/dp0.log"; exit 1
fi
echo "[hixl-single] drampool ready (pid=$DRAMPOOL_PID)"
sleep 4
# --- run the bench (single-peer: scheduler=worker, client on CLIENT_DEVICE) ---
echo "[hixl-single] running bench: client=device${CLIENT_DEVICE} request_size=${REQUEST_SIZE} batches=${BATCH_NUMBER}"
set +e
python3 "$STAGE_DIR/dramstore_bench_test.py" \
    --so-dir "$STAGE_DIR" \
    --kv-cache-memory-type device \
    --local-host 127.0.0.1 --device-id "$CLIENT_DEVICE" \
    --node-control-endpoints "127.0.0.1:${DRAMPOL_PORT}" \
    --node-transport-manager-ids "127.0.0.1:${DRAMPOL_ONE_SIDED}" \
    --batch-number "$BATCH_NUMBER" --tensor-size "$SHARD_SIZE" \
    --layer-size 61 --chunk-size 1 --request-size "$REQUEST_SIZE" \
    2>&1 | tee "$STAGE_DIR/bench_out.txt"
BENCH_RC=${PIPESTATUS[0]}
set -e

# --- cleanup ---
pkill -9 -x drampool 2>/dev/null || true

echo
if [[ "$BENCH_RC" -eq 0 ]]; then
    echo "RESULT: PASS (rc=0)"
else
    echo "RESULT: FAIL (rc=$BENCH_RC)"
    echo "--- drampool errors ---"
    grep -iE "503900|103901|peer unusable|operation failed|connect failed" "$STAGE_DIR/dp0.log" 2>/dev/null | tail -10 || true
fi
exit "$BENCH_RC"
