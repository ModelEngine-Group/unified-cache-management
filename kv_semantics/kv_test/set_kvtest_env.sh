SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)
BUILD_DIR="${BUILD_DIR:-${PROJECT_ROOT}/build-kv-test}"

export KV_TEST_CONFIG="${SCRIPT_DIR}/asu_kv_test.conf"  # 配置文件路径
export PATH="${PATH}:${BUILD_DIR}/ucm/transport/kv/kv-test"  # 可执行文件路径
# export KV_LOGGER_LEVEL=debug
# export ASU_TRACE=1  # SubBatch切分检查工具
# export KV_TEST_FAKE_BACKEND_TRACE=/path/of/unified-cache-management/fb.trace
