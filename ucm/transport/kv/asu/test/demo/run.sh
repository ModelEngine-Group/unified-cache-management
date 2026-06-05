#!/bin/bash
# aicpu_send_with_provider test script
# Environment: 2x NPU (0 2), internal RoCE network
# RoCE IP: rank0=192.168.190.170, rank1=192.168.190.172

SCRIPT_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
source /usr/local/Ascend/cann/set_env.sh

CANN_DIR="${ASCEND_HOME_PATH}"
KERNEL_JSON="${CANN_DIR}/opp/built-in/op_impl/aicpu/config/libcann_hixl_kernel.json"
BUILD_DIR="/home/lx/test/master/unified-cache-management/build/ucm/transport/kv/asu"
IP_MAP="/tmp/npu_ip_map.txt"

BINARY="${BUILD_DIR}/aicpu_send_with_provider"
LOG_DIR="${SCRIPT_DIR}/logs"
PORT=16666

cleanup() {
    rm -f /tmp/r0.bin /tmp/r1.bin /tmp/hixl.done /tmp/npu_ip_map.txt
}
trap cleanup EXIT
cleanup

mkdir -p "${LOG_DIR}"

echo "=== Generating NPU IP map ==="
DEV_COUNT=$(npu-smi info -l 2>/dev/null | grep "Total Count" | awk '{print $NF}')
if [ -z "${DEV_COUNT}" ]; then
    DEV_COUNT=8
fi
for i in $(seq 0 $((DEV_COUNT - 1))); do
    IP=$(hccn_tool -i ${i} -ip -g 2>/dev/null | grep "ipaddr" | awk -F: '{print $2}')
    if [ -n "${IP}" ]; then
        echo "${IP} ${i}" >> "${IP_MAP}"
    fi
done
echo "IP map:"
cat "${IP_MAP}"

echo "=== Starting rank 1 (receiver) ==="
(
    export ASCEND_GLOBAL_LOG_LEVEL=1
    export ASCEND_SLOG_PRINT_TO_STDOUT=0

    "${BINARY}" \
        --rank=1 --logic-dev=2 --phy-dev=2 --ip=192.168.190.172 \
        --message="Hello World!" \
        --local-file=/tmp/r1.bin --peer-file=/tmp/r0.bin \
        --done-file=/tmp/hixl.done \
        --kernel-json="${KERNEL_JSON}" \
        --ip-map="${IP_MAP}"
) &
PID_RANK1=$!

sleep 1

echo "=== Starting rank 0 (sender) ==="
(
    export ASCEND_GLOBAL_LOG_LEVEL=1
    export ASCEND_SLOG_PRINT_TO_STDOUT=0

    "${BINARY}" \
        --rank=0 --logic-dev=0 --phy-dev=0 --ip=192.168.190.170 \
        --message="Hello World!" \
        --local-file=/tmp/r0.bin --peer-file=/tmp/r1.bin \
        --done-file=/tmp/hixl.done \
        --kernel-json="${KERNEL_JSON}" \
        --ip-map="${IP_MAP}"
)
RET_RANK0=$?

echo "=== Waiting for rank 1 ==="
wait ${PID_RANK1}
RET_RANK1=$?

echo ""
if [ ${RET_RANK0} -eq 0 ] && [ ${RET_RANK1} -eq 0 ]; then
    echo "PASS: rank0 and rank1 both exited normally"
else
    echo "FAIL: rank0=${RET_RANK0}, rank1=${RET_RANK1}"
fi

echo ""
echo "=== Collecting logs ==="
cat /root/ascend/log/debug/device-0/* > "${LOG_DIR}/provider_dev0.log" 2>/dev/null || true
cat /root/ascend/log/debug/device-2/* > "${LOG_DIR}/provider_dev2.log" 2>/dev/null || true
echo "Logs saved to ${LOG_DIR}/"

exit $(( RET_RANK0 + RET_RANK1 ))