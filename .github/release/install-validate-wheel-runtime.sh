#!/usr/bin/env bash
# Install one audited backend Wheel in a pinned Runtime on its native architecture.
# The Wheel directory must include wheel-result.json from the same build.

set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: install-validate-wheel-runtime.sh WHEEL_DIRECTORY" >&2
  exit 2
fi
: "${RUNTIME_IMAGE:?RUNTIME_IMAGE must name the pinned Runtime}"
: "${RUNTIME_PLATFORM:?RUNTIME_PLATFORM must name the native Docker platform}"
: "${BACKEND_DISTRIBUTION:?BACKEND_DISTRIBUTION must be set}"
: "${PLATFORM_ARG:?PLATFORM_ARG must be set}"

wheel_directory="$1"
release_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
result="${wheel_directory}/wheel-result.json"
wheel="$(find "${wheel_directory}" -maxdepth 1 -type f -name '*.whl' -print -quit)"
test -n "${wheel}"
test "$(jq -r '.distribution' "${result}")" = "${BACKEND_DISTRIBUTION}"
wheel="$(realpath "${wheel}")"
wheel_name="$(basename "${wheel}")"
external_libraries="$(jq -c '.external_libraries' "${result}")"
deferred_external_libraries="$(jq -c '.deferred_external_libraries' "${result}")"
runtime_requirements="$(jq -ce '.dependencies | select(type == "array")' "${result}")"
wheel_glibc_floor="$(jq -er '.glibc_floor | select(type == "string")' "${result}")"
ucm_version="$(jq -er '.version' "${result}")"

docker pull --platform "${RUNTIME_PLATFORM}" "${RUNTIME_IMAGE}"
docker run --rm --platform "${RUNTIME_PLATFORM}" --entrypoint bash \
  -e DEFERRED_EXTERNAL_LIBRARIES="${deferred_external_libraries}" \
  -e PLATFORM="${PLATFORM_ARG}" \
  -e EXPECTED_UCM_VERSION="${ucm_version}" \
  -e EXPECTED_BACKEND_DISTRIBUTION="${BACKEND_DISTRIBUTION}" \
  -e EXPECTED_EXTERNAL_LIBRARIES="${external_libraries}" \
  -e EXPECTED_RUNTIME_REQUIREMENTS="${runtime_requirements}" \
  -e EXPECTED_WHEEL_GLIBC_FLOOR="${wheel_glibc_floor}" \
  -e UCM_WHEEL="/tmp/${wheel_name}" \
  -v "${wheel}:/tmp/${wheel_name}:ro" \
  -v "${release_directory}/validate_wheel_runtime.py:/tmp/validate_wheel_runtime.py:ro" \
  "${RUNTIME_IMAGE}" -c '
    set -euo pipefail
    ldd_output="$(ldd --version 2>&1)"
    actual_glibc="$(python3 -c "import re, sys; match = re.search(r\"[0-9]+(?:\\.[0-9]+)+\", sys.stdin.read()); print(match.group(0) if match else \"\")" \
      <<<"${ldd_output}")"
    test -n "${actual_glibc}"
    required_glibc="${EXPECTED_WHEEL_GLIBC_FLOOR#GLIBC_}"
    python3 -c "import sys; parse=lambda value: tuple(map(int, value.split(\".\"))); sys.exit(0 if parse(sys.argv[1]) >= parse(sys.argv[2]) else 1)" \
      "${actual_glibc}" "${required_glibc}"
    python3 -m venv /tmp/ucm-wheel-venv
    . /tmp/ucm-wheel-venv/bin/activate
    python -m pip install --disable-pip-version-check --no-cache-dir \
      --index-url https://pypi.org/simple "${UCM_WHEEL}"
    python /tmp/validate_wheel_runtime.py
  '
