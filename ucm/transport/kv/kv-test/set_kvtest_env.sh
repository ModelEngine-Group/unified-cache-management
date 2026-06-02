SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)
BUILD_DIR="${BUILD_DIR:-${PROJECT_ROOT}/build-kv-test}"

export KV_TEST_CONFIG="${SCRIPT_DIR}/asu_kv_test.conf"
export PATH="${PATH}:${BUILD_DIR}/ucm/transport/kv/kv-test"
