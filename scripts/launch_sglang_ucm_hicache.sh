#!/usr/bin/env bash
# Single-host Linux + NVIDIA launcher for SGLang HiCache with UCM storage.
# Edit the local configuration block before use.
#
# Prerequisites: SGLang 0.5.9 and UCM must be installed in PYTHON's
# environment. STORAGE_DIR must be on persistent storage with enough space.
# Extra SGLang arguments may be supplied after --. HiCache mode and its
# storage configuration are managed here and cannot be overridden.

set -Eeuo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    sed -n '2,10p' "$0"
    exit 0
fi
[[ $(uname -s) == Linux ]] || { echo 'Run on a Linux GPU host.' >&2; exit 1; }

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# ---------------------------------------------------------------------------
# Local configuration
# ---------------------------------------------------------------------------
PYTHON=python3
MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct
TP_SIZE=1
PAGE_SIZE=128
SGLANG_HOST=127.0.0.1
SGLANG_PORT=7800
START_TIMEOUT=600
STORAGE_DIR=/home/storage
LOG_DIR="$ROOT_DIR/logs/sglang-hicache-$(date +%Y%m%d-%H%M%S)-$$"
# ---------------------------------------------------------------------------

[[ ${1:-} != -- ]] || shift
for executable in "$PYTHON" setsid curl; do
    command -v "$executable" >/dev/null || {
        echo "Missing executable: $executable" >&2
        exit 1
    }
done
for number in "$TP_SIZE" "$PAGE_SIZE" "$SGLANG_PORT" "$START_TIMEOUT"; do
    [[ $number =~ ^[1-9][0-9]*$ ]] || {
        echo 'TP_SIZE, PAGE_SIZE, port, and timeout must be positive integers.' >&2
        exit 1
    }
done
((SGLANG_PORT < 65536)) || { echo 'SGLANG_PORT must be below 65536.' >&2; exit 1; }

for argument in "$@"; do
    case "$argument" in
        --model-path|--model-path=*|--tensor-parallel-size|--tensor-parallel-size=*|--page-size|--page-size=*|--host|--host=*|--port|--port=*|--enable-hierarchical-cache|--hicache-mem-layout|--hicache-mem-layout=*|--hicache-write-policy|--hicache-write-policy=*|--hicache-storage-backend|--hicache-storage-backend=*|--hicache-storage-prefetch-policy|--hicache-storage-prefetch-policy=*|--hicache-storage-backend-extra-config|--hicache-storage-backend-extra-config=*)
            echo "Conflicting or managed argument: $argument" >&2
            exit 1
            ;;
    esac
done

mkdir -p "$STORAGE_DIR" "$LOG_DIR"
[[ -d $STORAGE_DIR && -w $STORAGE_DIR ]] || {
    echo "UCM HiCache storage directory is not writable: $STORAGE_DIR" >&2
    exit 1
}
LOG_DIR=$(cd "$LOG_DIR" && pwd)

"$PYTHON" - "$TP_SIZE" "$SGLANG_PORT" "$STORAGE_DIR" "$LOG_DIR/hicache-config.json" <<'PY'
import importlib.metadata
import json
import re
import socket
import sys
from pathlib import Path

import torch

tp_size, port = int(sys.argv[1]), int(sys.argv[2])
storage_dir = Path(sys.argv[3]).resolve()
config_path = Path(sys.argv[4])

assert torch.cuda.is_available() and torch.version.hip is None, (
    'Requires NVIDIA CUDA GPUs'
)
assert torch.cuda.device_count() >= tp_size, 'Insufficient visible GPUs for TP_SIZE'
assert 0 < port < 65536, f'Invalid port: {port}'
with socket.socket() as sock:
    sock.bind(('0.0.0.0', port))

version = importlib.metadata.version('sglang')
assert re.match(r'^0\.5\.9(?:$|[.+-])', version), (
    f'This HiCache adapter recipe targets SGLang 0.5.9; found {version}'
)

# Importing this module checks that the UCM HiCache adapter and its SGLang API
# dependencies are available before the server starts.
from ucm.integration.sglang.unifiedcache_store import UnifiedCacheStore  # noqa: F401

config = {
    'backend_name': 'unifiedcache',
    'module_path': 'ucm.integration.sglang.unifiedcache_store',
    'class_name': 'UnifiedCacheStore',
    'interface_v1': 1,
    'kv_connector_extra_config': {
        'ucm_connector_name': 'UcmPipelineStore',
        'ucm_connector_config': {
            'storage_backends': str(storage_dir),
        },
    },
}
config_path.write_text(json.dumps(config), encoding='utf-8')
PY

"$PYTHON" -m sglang.launch_server --help >"$LOG_DIR/sglang-help.log" 2>&1
for option in \
    --tensor-parallel-size \
    --page-size \
    --enable-hierarchical-cache \
    --hicache-mem-layout \
    --hicache-write-policy \
    --hicache-storage-backend \
    --hicache-storage-prefetch-policy \
    --hicache-storage-backend-extra-config; do
    grep -q -- "$option" "$LOG_DIR/sglang-help.log" || {
        echo "SGLang CLI is missing $option; see $LOG_DIR/sglang-help.log" >&2
        exit 1
    }
done

pid=''
cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n $pid ]]; then
        kill -TERM -- "-$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
    echo "Stopped SGLang. Log: $LOG_DIR/sglang.log"
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

setsid "$PYTHON" -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --tensor-parallel-size "$TP_SIZE" \
    --page-size "$PAGE_SIZE" \
    --trust-remote-code \
    --host "$SGLANG_HOST" \
    --port "$SGLANG_PORT" \
    --enable-hierarchical-cache \
    --hicache-mem-layout page_first \
    --hicache-write-policy write_through \
    --hicache-storage-backend dynamic \
    --hicache-storage-prefetch-policy wait_complete \
    --hicache-storage-backend-extra-config "$(cat "$LOG_DIR/hicache-config.json")" \
    "$@" >"$LOG_DIR/sglang.log" 2>&1 &
pid=$!
echo "Started SGLang HiCache (PID $pid), log: $LOG_DIR/sglang.log"
echo "Model: $MODEL_PATH; UCM storage: $STORAGE_DIR"

health_host=$SGLANG_HOST
[[ $health_host != 0.0.0.0 ]] || health_host=127.0.0.1
deadline=$((SECONDS + START_TIMEOUT))
until curl --noproxy '*' -fsS --max-time 2 \
    "http://$health_host:$SGLANG_PORT/health" >/dev/null 2>&1; do
    if ! kill -0 "$pid" 2>/dev/null; then
        echo 'SGLang exited; recent log:' >&2
        tail -n 80 "$LOG_DIR/sglang.log" >&2
        exit 1
    fi
    if ((SECONDS >= deadline)); then
        echo "Timed out waiting for SGLang; log: $LOG_DIR/sglang.log" >&2
        exit 1
    fi
    sleep 2
done

echo "SGLang HiCache ready: http://$health_host:$SGLANG_PORT"
wait "$pid"
