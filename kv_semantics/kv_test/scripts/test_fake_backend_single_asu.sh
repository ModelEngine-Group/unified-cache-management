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

write_fake_backend_config "${CONFIG}" "${STORE}" "${OUTPUT}" "1"

run_success "${LOG}" "${KV_TEST}" store --configpath "${CONFIG}" --key fb1 --check
run_success "${LOG}" "${KV_TEST}" exist --configpath "${CONFIG}" --key fb1
assert_contains "${LOG}" "result=exists"

run_success "${LOG}" "${KV_TEST}" retrieve --configpath "${CONFIG}" --key fb1 --check
run_success "${LOG}" "${KV_TEST}" delete --configpath "${CONFIG}" --key fb1 --check

run_success "${LOG}" "${KV_TEST}" exist --configpath "${CONFIG}" --key fb1
assert_contains "${LOG}" "result=missing"

print_success "fake_backend single ASU flow passed"
