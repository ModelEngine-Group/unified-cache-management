#!/usr/bin/env bash
# Single-host Linux + NVIDIA launcher for SGLang 0.5.20 and newer compatible releases
# with the real UCM L1 external-cache linker. Edit the local configuration block
# before use.
#
# Prerequisites: install this UCM checkout, install the matching SGLang package,
# and configure a usable Store in UCM_CONFIG_FILE. Older SGLang releases also
# need ucm_patch.pth; PR #40595 and newer have native linker support.
# Extra SGLang arguments may be supplied after --. Managed linker and network
# options are rejected so the process cannot silently start a different mode.

set -Eeuo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    sed -n '2,10p' "$0"
    exit 0
fi
[[ $(uname -s) == Linux ]] || { echo 'Run on a Linux GPU host.' >&2; exit 1; }

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# ---------------------------------------------------------------------------
# Local configuration
# ---------------------------------------------------------------------------
PYTHON=python3
MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct
TP_SIZE=1
SGLANG_HOST=127.0.0.1
SGLANG_PORT=30000
MAX_TOTAL_TOKENS=4096
MAX_RUNNING_REQUESTS=1
MEM_FRACTION_STATIC=0.7
START_TIMEOUT=600
UCM_CONFIG_FILE="$ROOT_DIR/examples/ucm_config_example.yaml"
LOG_DIR="$ROOT_DIR/logs/ucm-linker-$(date +%Y%m%d-%H%M%S)-$$"
# ---------------------------------------------------------------------------

[[ ${1:-} != -- ]] || shift
for executable in "$PYTHON" setsid curl; do
    command -v "$executable" >/dev/null || {
        echo "Missing executable: $executable" >&2
        exit 1
    }
done
SGLANG_VERSION=$("$PYTHON" -c \
    'import importlib.metadata; print(importlib.metadata.version("sglang"))')
for number in "$TP_SIZE" "$SGLANG_PORT" "$MAX_TOTAL_TOKENS" \
    "$MAX_RUNNING_REQUESTS" "$START_TIMEOUT"; do
    [[ $number =~ ^[1-9][0-9]*$ ]] || {
        echo 'TP_SIZE, ports, limits, and timeout must be positive integers.' >&2
        exit 1
    }
done
[[ -f $UCM_CONFIG_FILE ]] || {
    echo "UCM configuration file does not exist: $UCM_CONFIG_FILE" >&2
    exit 1
}
for argument in "$@"; do
    case "$argument" in
        --model-path|--model-path=*|--tp-size|--tp-size=*|--host|--host=*|--port|--port=*|--enable-hierarchical-cache|--enable-lmcache|--enable-unified-cache-external-linker|--unified-cache-external-linker-backend|--unified-cache-external-linker-backend=*|--unified-cache-external-linker-config|--unified-cache-external-linker-config=*)
            echo "Conflicting or managed argument: $argument" >&2
            exit 1
            ;;
    esac
done

mkdir -p "$LOG_DIR"
LOG_DIR=$(cd "$LOG_DIR" && pwd)

"$PYTHON" - "$TP_SIZE" "$SGLANG_PORT" "$UCM_CONFIG_FILE" <<'PY'
import socket
import sys
from pathlib import Path

import torch
import yaml

assert torch.cuda.is_available() and torch.version.hip is None, 'Requires NVIDIA CUDA GPUs'
assert torch.cuda.device_count() >= int(sys.argv[1]), 'Insufficient visible GPUs for TP_SIZE'
port = int(sys.argv[2])
assert 0 < port < 65536, f'Invalid port: {port}'
with socket.socket() as sock:
    sock.bind(('0.0.0.0', port))

document = yaml.safe_load(Path(sys.argv[3]).read_text(encoding='utf-8')) or {}
connectors = document.get('ucm_connectors') if isinstance(document, dict) else None
assert isinstance(connectors, list) and len(connectors) == 1, (
    'UCM_CONFIG_FILE must contain exactly one ucm_connectors entry'
)
connector = connectors[0]
assert isinstance(connector, dict) and connector.get('ucm_connector_name'), (
    'UCM connector entry is missing ucm_connector_name'
)
assert isinstance(connector.get('ucm_connector_config'), dict), (
    'UCM connector entry is missing ucm_connector_config'
)
PY

export ENABLE_UCM_PATCH=1
export UCM_SGLANG_VERSION="$SGLANG_VERSION"
UCM_LINKER_CONFIG=$("$PYTHON" - "$UCM_CONFIG_FILE" <<'PY'
import json
import sys

print(json.dumps({
    "linker": "UcmDirectLinker",
    "linker_module_path": "ucm.integration.sglang.ucm_direct_linker",
    "linker_extra_config": {"config_file": sys.argv[1]},
}))
PY
)

if ! "$PYTHON" - >"$LOG_DIR/patch-check.log" 2>&1 <<'PY'
import sglang
import importlib.metadata
from ucm.integration.sglang.patch.apply_patch import (
    _has_native_external_linker,
    patch_package_for,
)
from sglang.srt.mem_cache import registry

version = importlib.metadata.version('sglang')
if _has_native_external_linker():
    print(f'SGLang {version} has native linker support')
else:
    patch_package = patch_package_for(version)
    if patch_package is None:
        raise RuntimeError(
            f'SGLang {version} has no UCM compatibility patch and lacks the '
            'native linker interface.'
        )
    factory_name = ('_create_unified_radix_cache' if patch_package == 'v0520'
                    else 'create_unified_radix_cache')
    factory = getattr(registry, factory_name, None)
    if not getattr(factory, '_ucm_patched', False):
        raise RuntimeError(
            f'SGLang {version} requires the UCM compatibility patch. '
            'Install ucm_patch.pth and set ENABLE_UCM_PATCH=1.'
        )
    print(f'UCM compatibility patch active for SGLang {version}')
PY
then
    echo "SGLang linker check failed; see $LOG_DIR/patch-check.log" >&2
    tail -n 80 "$LOG_DIR/patch-check.log" >&2
    exit 1
fi
if ! "$PYTHON" -m sglang.launch_server --help >"$LOG_DIR/sglang-help.log" 2>&1; then
    echo "SGLang CLI check failed; see $LOG_DIR/sglang-help.log" >&2
    tail -n 80 "$LOG_DIR/sglang-help.log" >&2
    exit 1
fi
grep -q -- '--unified-cache-external-linker-config' "$LOG_DIR/sglang-help.log" && \
    grep -q -- 'dynamic' "$LOG_DIR/sglang-help.log" || {
    echo "SGLang external linker CLI is unavailable; see $LOG_DIR/patch-check.log and $LOG_DIR/sglang-help.log" >&2
    exit 1
}

pid=''
cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n $pid ]]; then
        kill -TERM -- "-$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
    echo "Stopped SGLang. Log: $LOG_DIR/sglang.log"
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

setsid "$PYTHON" -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --tp-size "$TP_SIZE" \
    --page-size 64 \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --max-total-tokens "$MAX_TOTAL_TOKENS" \
    --max-running-requests "$MAX_RUNNING_REQUESTS" \
    --enable-cache-report \
    --enable-unified-cache-external-linker \
    --unified-cache-external-linker-backend dynamic \
    --unified-cache-external-linker-config "$UCM_LINKER_CONFIG" \
    --host "$SGLANG_HOST" \
    --port "$SGLANG_PORT" \
    "$@" >"$LOG_DIR/sglang.log" 2>&1 &
pid=$!
echo "Started SGLang (PID $pid), log: $LOG_DIR/sglang.log"
echo "UCM config: $UCM_CONFIG_FILE"

health_host=$SGLANG_HOST
[[ $health_host != 0.0.0.0 ]] || health_host=127.0.0.1
deadline=$((SECONDS + START_TIMEOUT))
until curl --noproxy '*' -fsS --max-time 2 \
    "http://$health_host:$SGLANG_PORT/health" >/dev/null 2>&1; do
    if ! kill -0 "$pid" 2>/dev/null; then
        echo 'SGLang exited; recent log:' >&2
        tail -n 80 "$LOG_DIR/sglang.log" >&2
        exit 1
    fi
    if ((SECONDS >= deadline)); then
        echo "Timed out waiting for SGLang; log: $LOG_DIR/sglang.log" >&2
        exit 1
    fi
    sleep 2
done

echo "SGLang ready: http://$health_host:$SGLANG_PORT"
echo "UCM linker and Store adapter are active."
wait "$pid"
