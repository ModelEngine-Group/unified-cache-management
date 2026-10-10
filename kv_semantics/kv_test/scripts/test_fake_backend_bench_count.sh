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

run_success "${LOG}" "${KV_TEST}" bench batch-store --configpath "${CONFIG}" \
    --count 5 --batch-size 2 --concurrency 2 --warmup 0
assert_contains "${LOG}" "entries=5"
assert_contains "${LOG}" "bytes=20480"
assert_contains "${LOG}" "operations=3"
assert_contains "${LOG}" "io_count=5"

print_success "fake_backend bench count passed"
