#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: retry-registry-command.sh LOG_PATH [--retry-transport] [--rate-limit-marker PATH [--rate-limit-scope TEXT]] COMMAND [ARG ...]" >&2
  exit 2
fi

log_path="$1"
shift
retry_transport=false
if [ "${1:-}" = "--retry-transport" ]; then
  retry_transport=true
  shift
fi
rate_limit_marker=""
rate_limit_scope=""
if [ "${1:-}" = "--rate-limit-marker" ]; then
  if [ "$#" -lt 3 ]; then
    echo "--rate-limit-marker requires MARKER_PATH and COMMAND" >&2
    exit 2
  fi
  rate_limit_marker="$2"
  shift 2
  if [ "${1:-}" = "--rate-limit-scope" ]; then
    if [ "$#" -lt 3 ]; then
      echo "--rate-limit-scope requires TEXT and COMMAND" >&2
      exit 2
    fi
    rate_limit_scope="$2"
    shift 2
  fi
fi
if [ -n "${rate_limit_marker}" ] && [ -e "${rate_limit_marker}" ]; then
  echo "registry rate-limit marker already exists" >&2
  exit 2
fi
if [ "$#" -lt 1 ]; then
  echo "registry retry command is required" >&2
  exit 2
fi
read -r -a retry_delays <<<"${UCM_REGISTRY_RETRY_DELAYS:-5 15 30 60}"
for delay in "${retry_delays[@]}"; do
  if [[ ! "${delay}" =~ ^[0-9]+$ ]]; then
    echo "registry retry delays must be non-negative integers" >&2
    exit 2
  fi
done
max_attempts=$(( ${#retry_delays[@]} + 1 ))
mkdir -p "$(dirname "${log_path}")"
rate_limit_pattern='TOOMANYREQUESTS|too many requests|retry-after|HTTP 429|(^|[^[:alnum:]])429([^[:alnum:]]|$)'
scoped_rate_limit_pattern='TOOMANYREQUESTS|too many requests|retry-after|HTTP 429|status[^0-9]*429|429 Too Many Requests'
transport_pattern='connection reset( by peer)?|(^|[^[:alnum:]_])EOF([^[:alnum:]_]|$)'

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

  retry_reason=""
  if grep -Eiq "${rate_limit_pattern}" "${log_path}"; then
    retry_reason="rate-limit"
  elif [ "${retry_transport}" = true ] && \
       grep -Eiq "${transport_pattern}" "${log_path}"; then
    retry_reason="transport"
  else
    echo "Registry command failed with a non-retryable error" >&2
    exit "${command_status}"
  fi
  if [ "${attempt}" -eq "${max_attempts}" ]; then
    if [ "${retry_reason}" = "rate-limit" ] && \
       [ -n "${rate_limit_marker}" ]; then
      marker_allowed=true
      if [ -n "${rate_limit_scope}" ] && \
         ! grep -Fi -- "${rate_limit_scope}" "${log_path}" \
           | grep -Ei "${scoped_rate_limit_pattern}" >/dev/null; then
        marker_allowed=false
      fi
      if [ "${marker_allowed}" = true ]; then
        if ! printf '%s\n' "rate-limit-exhausted" >"${rate_limit_marker}"; then
          echo "Registry rate-limit marker could not be written" >&2
          exit 2
        fi
      fi
    fi
    echo "Registry command failed after ${attempt} attempts" >&2
    exit "${command_status}"
  fi

  sleep_seconds="${retry_delays[$((attempt - 1))]}"
  if [ "${retry_reason}" = "rate-limit" ]; then
    echo "Registry command was rate-limited; retrying in ${sleep_seconds}s" >&2
  else
    echo "Registry command hit a transient transport error; retrying in ${sleep_seconds}s" >&2
  fi
  sleep "${sleep_seconds}"
done
