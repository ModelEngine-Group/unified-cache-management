#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: retry-registry-command.sh LOG_PATH COMMAND [ARG ...]" >&2
  exit 2
fi

log_path="$1"
shift
read -r -a retry_delays <<<"${UCM_REGISTRY_RETRY_DELAYS:-5 15 30 60}"
for delay in "${retry_delays[@]}"; do
  if [[ ! "${delay}" =~ ^[0-9]+$ ]]; then
    echo "registry retry delays must be non-negative integers" >&2
    exit 2
  fi
done
max_attempts=$(( ${#retry_delays[@]} + 1 ))
mkdir -p "$(dirname "${log_path}")"

for ((attempt = 1; attempt <= max_attempts; attempt++)); do
  if "$@" 2>&1 | tee "${log_path}"; then
    exit 0
  else
    pipeline_status=("${PIPESTATUS[@]}")
  fi
  command_status="${pipeline_status[0]}"
  tee_status="${pipeline_status[1]}"

  if [ "${tee_status}" -ne 0 ]; then
    echo "Registry command retry log could not be written" >&2
    exit "${tee_status}"
  fi

  if ! grep -Eiq \
    'TOOMANYREQUESTS|too many requests|retry-after|HTTP 429|(^|[^0-9])429([^0-9]|$)' \
    "${log_path}"; then
    echo "Registry command failed with a non-retryable error" >&2
    exit "${command_status}"
  fi
  if [ "${attempt}" -eq "${max_attempts}" ]; then
    echo "Registry command failed after ${attempt} attempts" >&2
    exit "${command_status}"
  fi

  sleep_seconds="${retry_delays[$((attempt - 1))]}"
  echo "Registry command was rate-limited; retrying in ${sleep_seconds}s" >&2
  sleep "${sleep_seconds}"
done
