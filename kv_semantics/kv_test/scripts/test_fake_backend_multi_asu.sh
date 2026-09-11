#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/fake_backend_common.sh"

KV_TEST=$(find_kv_test_bin)
WORK_DIR=$(make_temp_dir)
echo "script artifacts: ${WORK_DIR}"

CONFIG="${WORK_DIR}/kv-test.conf"
STORE="${WORK_DIR}/store"
OUTPUT="${WORK_DIR}/output"
LOG="${WORK_DIR}/command.log"

write_fake_backend_config "${CONFIG}" "${STORE}" "${OUTPUT}" "1,2"

run_success "${LOG}" "${KV_TEST}" store --configpath "${CONFIG}" --prefix m --key-start 0 --key-end 63 --check
run_success "${LOG}" "${KV_TEST}" exist --configpath "${CONFIG}" --prefix m --key-start 0 --key-end 63
assert_contains "${LOG}" "total=64"
assert_contains "${LOG}" "exists=64"
assert_contains "${LOG}" "missing=0"

assert_dir_has_bins "${STORE}/asu-1"
assert_dir_has_bins "${STORE}/asu-2"

print_success "fake_backend multi ASU store namespace passed"
