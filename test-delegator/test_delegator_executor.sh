#!/usr/bin/env bash

set -Eeo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build}"
BUILD_TYPE="${BUILD_TYPE:-Debug}"
JOBS="${JOBS:-$(nproc)}"
GTEST_FILTER="${GTEST_FILTER:-ExecutorTest.*}"
ASCEND_ENV_SCRIPT="${ASCEND_ENV_SCRIPT:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
DOWNLOAD_DEPENDENCE="${DOWNLOAD_DEPENDENCE:-ON}"

if [[ -f "${ASCEND_ENV_SCRIPT}" ]]; then
    # Ascend's environment script may reference variables that are not defined yet.
    set +u
    # shellcheck disable=SC1090
    source "${ASCEND_ENV_SCRIPT}"
    set -u
elif [[ -z "${ASCEND_HOME_PATH:-}" && -z "${ASCEND_TOOLKIT_HOME:-}" ]]; then
    echo "ERROR: Ascend environment script not found: ${ASCEND_ENV_SCRIPT}" >&2
    echo "Set ASCEND_ENV_SCRIPT, ASCEND_HOME_PATH, or ASCEND_TOOLKIT_HOME." >&2
    exit 1
else
    set -u
fi

echo "[1/3] Configuring delegator executor tests"
cmake \
    -S "${PROJECT_ROOT}" \
    -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DBUILD_UNIT_TESTS=ON \
    -DBUILD_UCM_STORE=ON \
    -DBUILD_UCM_ASU=ON \
    -DBUILD_UCM_DRAMPOOL=OFF \
    -DBUILD_UCM_SPARSE=OFF \
    -DBUILD_UCM_MINDIE=OFF \
    -DRUNTIME_ENVIRONMENT=ascend \
    -DDOWNLOAD_DEPENDENCE="${DOWNLOAD_DEPENDENCE}"

echo "[2/3] Building ucmstore.test"
cmake --build "${BUILD_DIR}" --target ucmstore.test --parallel "${JOBS}"

TEST_BINARY="${BUILD_DIR}/ucm/store/test/ucmstore.test"
if [[ ! -x "${TEST_BINARY}" ]]; then
    echo "ERROR: test binary was not generated: ${TEST_BINARY}" >&2
    exit 1
fi

echo "[3/3] Running ${GTEST_FILTER}"
"${TEST_BINARY}" \
    --gtest_filter="${GTEST_FILTER}" \
    --gtest_color=yes \
    "$@"
