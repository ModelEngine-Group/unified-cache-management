#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
KV_TEST_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
PROJECT_ROOT=$(cd "${KV_TEST_DIR}/../../../.." && pwd)
ANSI_GREEN=$'\033[32m'
ANSI_RED=$'\033[31m'
ANSI_RESET=$'\033[0m'

print_success() {
    echo "${ANSI_GREEN}$*${ANSI_RESET}"
}

print_error() {
    echo "${ANSI_RED}$*${ANSI_RESET}" >&2
}

find_kv_test_bin() {
    if [[ -n "${KV_TEST_BIN:-}" ]]; then
        printf '%s\n' "${KV_TEST_BIN}"
        return
    fi

    if command -v kv-test >/dev/null 2>&1; then
        command -v kv-test
        return
    fi

    if command -v kv-test.exe >/dev/null 2>&1; then
        command -v kv-test.exe
        return
    fi

    local candidate
    for candidate in \
        "${PROJECT_ROOT}/kv-test" \
        "${PROJECT_ROOT}/kv-test.exe" \
        "${PROJECT_ROOT}/build-kv-test/ucm/transport/kv/kv-test/kv-test" \
        "${PROJECT_ROOT}/build-kv-test/ucm/transport/kv/kv-test/kv-test.exe" \
        "${PROJECT_ROOT}/build/ucm/transport/kv/kv-test/kv-test" \
        "${PROJECT_ROOT}/build/ucm/transport/kv/kv-test/kv-test.exe"; do
        if [[ -x "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return
        fi
    done

    echo "kv-test binary not found. Set KV_TEST_BIN or add kv-test to PATH." >&2
    return 1
}

make_temp_dir() {
    local base_dir="${KV_TEST_SCRIPT_LOG_DIR:-${PROJECT_ROOT}/kv-test-output/fake-backend-scripts}"
    local run_id
    run_id=$(date +"%Y%m%d-%H%M%S")
    local dir="${base_dir}/run-${run_id}-$$"
    mkdir -p "${dir}"
    printf '%s\n' "${dir}"
}

write_fake_backend_config() {
    local config_path=$1
    local store_path=$2
    local output_path=$3
    local asu_ids=$4
    local view_path="${config_path}.view"

    {
        echo "viewEpoch=1"
        echo "viewId=1"
        echo "asuIds=${asu_ids}"
        local id
        IFS=',' read -ra ids <<< "${asu_ids}"
        for id in "${ids[@]}"; do
            echo "asu_info.${id}=protocol=TCP,local.comm_id=127.0.0.1,port=$((19000 + id))"
        done
    } > "${view_path}"

    {
        echo "client_id=kv-test-fake-backend-script"
        echo "wait_timeout_ms=5000"
        echo
        echo "fake_backend.path=${store_path}"
        echo "fake_backend.latency_ms=1"
        echo
        echo "view.config_path=${view_path}"
        echo "hash_table.type=RING_HASH"
        echo "ring_hash.virtual_node_count=128"
        echo "transport.asuIds=${asu_ids}"
        echo "transport.provider_type=FAKE"
        IFS=',' read -ra ids <<< "${asu_ids}"
        for id in "${ids[@]}"; do
            echo "asu_info.${id}=protocol=TCP,local.comm_id=127.0.0.1,port=$((19000 + id))"
        done
        echo
        echo "kv.key_prefix=k"
        echo "kv.seed=20260530"
        echo "kv.value_size=4096"
        echo "kv.count=16"
        echo
        echo "limits.memory_max_bytes=4294967296"
        echo
        echo "bench.io_size=4096"
        echo "bench.concurrency=1"
        echo "bench.duration_sec=1"
        echo "bench.io_count=0"
        echo "bench.warmup_sec=0"
        echo "bench.read_ratio=50"
        echo "bench.write_ratio=50"
        echo "bench.batch_size=16"
        echo
        echo "output.path=${output_path}"
        echo "output.realtime_file_max_bytes=104857600"
    } > "${config_path}"
}

run_success() {
    local log_file=$1
    shift
    local timeout_sec="${KV_TEST_COMMAND_TIMEOUT_SEC:-30}"

    echo "+ $*"
    set +e
    if command -v timeout >/dev/null 2>&1; then
        timeout "${timeout_sec}" "$@" 2>&1 | tee "${log_file}"
    else
        "$@" 2>&1 | tee "${log_file}"
    fi
    local exit_code=${PIPESTATUS[0]}
    set -e
    if [[ ${exit_code} -ne 0 ]]; then
        print_error "command failed with exit code ${exit_code}: $*"
        return "${exit_code}"
    fi
}

run_failure() {
    local log_file=$1
    shift
    local timeout_sec="${KV_TEST_COMMAND_TIMEOUT_SEC:-30}"

    echo "+ $*"
    set +e
    if command -v timeout >/dev/null 2>&1; then
        timeout "${timeout_sec}" "$@" 2>&1 | tee "${log_file}"
    else
        "$@" 2>&1 | tee "${log_file}"
    fi
    local exit_code=${PIPESTATUS[0]}
    set -e
    if [[ ${exit_code} -eq 124 ]]; then
        print_error "command timed out after ${timeout_sec}s: $*"
        return 1
    fi
    if [[ ${exit_code} -eq 0 ]]; then
        print_error "expected command to fail, but it succeeded: $*"
        return 1
    fi
}

assert_contains() {
    local file=$1
    local pattern=$2

    if ! grep -Fq "${pattern}" "${file}"; then
        print_error "missing expected pattern '${pattern}' in ${file}"
        return 1
    fi
}

assert_dir_has_bins() {
    local dir=$1

    if [[ ! -d "${dir}" ]]; then
        print_error "missing directory: ${dir}"
        return 1
    fi

    if ! find "${dir}" -maxdepth 1 -type f -name '*.bin' | grep -q .; then
        print_error "directory has no .bin files: ${dir}"
        return 1
    fi
}
