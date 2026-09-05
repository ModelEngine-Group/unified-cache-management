#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)
BUILD_DIR="${BUILD_DIR:-${PROJECT_ROOT}/build-kv-test}"
BUILD_KV_CLIENT_PROVIDER_AICPU="${BUILD_KV_CLIENT_PROVIDER_AICPU:-OFF}"
BUILD_KV_CLIENT_PROVIDER_FAKE="${BUILD_KV_CLIENT_PROVIDER_FAKE:-ON}"
BUILD_KV_CLIENT_PROVIDER_AIV="${BUILD_KV_CLIENT_PROVIDER_AIV:-OFF}"

# ASU_AIV_PROVIDER_ROOT may be exported by the caller when
# BUILD_KV_CLIENT_PROVIDER_AIV=ON and libumc.a is not in the default CMake search path.
cmake -S "${PROJECT_ROOT}" -B "${BUILD_DIR}" \
    -DBUILD_KV_CLIENT=ON \
    -DBUILD_UCM_STORE=OFF \
    -DBUILD_UNIT_TESTS=OFF \
    -DRUNTIME_ENVIRONMENT=ascend \
    -DBUILD_KV_CLIENT_PROVIDER_AICPU="${BUILD_KV_CLIENT_PROVIDER_AICPU}" \
    -DBUILD_KV_CLIENT_PROVIDER_FAKE="${BUILD_KV_CLIENT_PROVIDER_FAKE}" \
    -DBUILD_KV_CLIENT_PROVIDER_AIV="${BUILD_KV_CLIENT_PROVIDER_AIV}"

cmake --build "${BUILD_DIR}" --target kv_transport kv_client
cmake --build "${BUILD_DIR}" --target kv-test
