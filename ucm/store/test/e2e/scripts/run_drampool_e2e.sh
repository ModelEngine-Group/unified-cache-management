#!/usr/bin/env bash

# Exit immediately when a command fails.
set -e

# ================================================================
# 1. Parameters and configuration
# ================================================================
# --- 1.1 Global environment variables ---
# export UC_LOGGER_LEVEL=debug
# export ASCEND_SLOG_PRINT_TO_STDOUT=1
# export ASCEND_RT_VISIBLE_DEVICES=9,10,11,12

# Preload HIXL from the standard CANN installation.
HIXL_SO="/usr/local/Ascend/cann/lib64/libcann_hixl.so"
export LD_PRELOAD="${HIXL_SO}"

# --- 1.2 DramPool process configuration ---
DRAMPOOL_BIN="./build/ucm/store/dram/drampool"
DRAMPOOL_LOG="drampool_server.log"

# DramPool command-line parameters.
DRAMPOOL_CONFIG="examples/drampool.yaml"
DRAMPOOL_ADDR="127.0.0.1:9000"
DRAMPOOL_NICS="eth0"
DRAMPOOL_POOL_SIZE_GB="1"
DRAMPOOL_KVCACHE_BLOCK_SIZES=(131072 16384)

# Assemble the DramPool argument array.
DRAMPOOL_ARGS=(
  --config "${DRAMPOOL_CONFIG}"
  --addr "${DRAMPOOL_ADDR}"
  --nics "${DRAMPOOL_NICS}"
  --pool-size-gb "${DRAMPOOL_POOL_SIZE_GB}"
  --kvcache-block-sizes "${DRAMPOOL_KVCACHE_BLOCK_SIZES[@]}"
)

# --- 1.3 Python inference process configuration ---
PYTHON_SCRIPT="examples/offline_inference.py"
INFERENCE_LOG="offline_inference.log"

# ================================================================
# 2. Process cleanup
# ================================================================
cleanup() {
    echo ""
    echo "[INFO] Cleaning up background processes..."
    if [ -n "${DRAMPOOL_PID}" ] && kill -0 "${DRAMPOOL_PID}" 2>/dev/null; then
        echo "[INFO] Stopping DramPool service (PID: ${DRAMPOOL_PID})..."
        kill -9 "${DRAMPOOL_PID}" 2>/dev/null || true
    fi
    echo "[INFO] All tasks have completed or been terminated."
    exit 0
}

# Stop the background DramPool process when the script exits.
trap cleanup SIGINT SIGTERM EXIT

# ================================================================
# 3. Run the workload
# ================================================================

# --- Process 1: DramPool service ---
echo "[INFO] Starting DramPool service..."
"${DRAMPOOL_BIN}" "${DRAMPOOL_ARGS[@]}" > "${DRAMPOOL_LOG}" 2>&1 &

DRAMPOOL_PID=$!
echo "[INFO] DramPool started in the background (PID: ${DRAMPOOL_PID})"
echo "[INFO] DramPool log: ${DRAMPOOL_LOG}"

# --- Process 2: Python inference workload ---
echo "[INFO] Starting the Python offline inference script..."
echo "[INFO] Command: python3 ${PYTHON_SCRIPT}"
echo "[INFO] Python inference log: ${INFERENCE_LOG}"

# Run the Python workload synchronously.
python3 "${PYTHON_SCRIPT}" > "${INFERENCE_LOG}" 2>&1

echo "[INFO] Python inference workload completed."
