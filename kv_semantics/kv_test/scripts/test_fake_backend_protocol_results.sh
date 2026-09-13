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
TRACE="${WORK_DIR}/fake-backend.trace"

write_fake_backend_config "${CONFIG}" "${STORE}" "${OUTPUT}" "1"
export KV_TEST_FAKE_BACKEND_TRACE="${TRACE}"

run_success "${LOG}" "${KV_TEST}" store --configpath "${CONFIG}" --key pexist

>"${TRACE}"
run_success "${LOG}" "${KV_TEST}" exist --configpath "${CONFIG}" --key pexist
assert_contains "${LOG}" "result=exists"
assert_contains "${TRACE}" "opcode=Exist"
assert_contains "${TRACE}" "status=0x000"
assert_contains "${TRACE}" "result_buffer=0"

>"${TRACE}"
run_success "${LOG}" "${KV_TEST}" exist --configpath "${CONFIG}" --keys pexist,pmiss
assert_contains "${LOG}" "total=2"
assert_contains "${LOG}" "exists=1"
assert_contains "${LOG}" "missing=1"
assert_contains "${TRACE}" "opcode=Exist"
assert_contains "${TRACE}" "status=0x732"
assert_contains "${TRACE}" "result_buffer=1"

>"${TRACE}"
run_failure "${LOG}" "${KV_TEST}" batch-retrieve --configpath "${CONFIG}" --keys pexist,pmiss --batch-size 2
assert_contains "${TRACE}" "opcode=BatchRetrieve"
assert_contains "${TRACE}" "status=0x732"
assert_contains "${TRACE}" "result_buffer=1"

>"${TRACE}"
run_success "${LOG}" "${KV_TEST}" delete --configpath "${CONFIG}" --keys pexist,pmiss
assert_contains "${TRACE}" "opcode=Delete"
assert_contains "${TRACE}" "status=0x000"
assert_contains "${TRACE}" "result_buffer=0"

print_success "fake_backend protocol result handling passed"
