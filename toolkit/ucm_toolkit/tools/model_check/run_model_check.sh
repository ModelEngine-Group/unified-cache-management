#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Run the UCM model-compatibility check on one model, in the fixed case order
# the README calls the standard flow: layout-only first, then a full dump/load
# in bulk mode, then the same in layerwise mode.
#
#   ./run_model_check.sh /path/to/model-dir [options]
#
# The model directory needs only `config.json` (plus tokenizer files for
# multimodal models). No checkpoint weights are read.
#
# Options (all optional):
#   --platform P        auto | ascend | ascend-sim | cuda | cuda-sim | cpu
#                       default: auto (detected from the installed stack)
#   --ascend-soc S      ascend-sim: assert the installed wheel's SoC, e.g. A2
#   --cuda-capability C cuda-sim: the capability to simulate, e.g. 9.0 (required there)
#   --tokens N          synthetic request length (default 1024)
#   --block-size N      vLLM KV block size in tokens (default 128)
#   --tokens-per-chunk N  connector-v2 tokens per UCM record (default: connector's own)
#   --connector v2|legacy  which UCMConnector facade (default v2)
#   --device-id ID      accelerator id for the on-device launchers (default 0)
#   --additional-config JSON   passed to vLLM, e.g. GLM's C8 switches
#   --cases LIST        comma-separated subset of: layout,bulk,layerwise
#                       default: layout,bulk,layerwise
#   --out DIR           where logs and stores go (default ./model-check-runs)
#   --keep-store        keep each case's store directory (default: removed first)
#
# Exit code is 0 only when every selected case passed.

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# <repo>/toolkit/ucm_toolkit/tools/model_check/run_model_check.sh
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"

MODEL=""
PLATFORM="auto"
ASCEND_SOC=""
CUDA_CAPABILITY=""
TOKENS="1024"
BLOCK_SIZE="128"
TOKENS_PER_CHUNK=""
CONNECTOR="v2"
DEVICE_ID="0"
ADDITIONAL_CONFIG=""
CASES="layout,bulk,layerwise"
OUT_DIR=""
KEEP_STORE="0"

die() {
    printf '%s\n' "ERROR: $*" >&2
    exit 2
}

usage() {
    sed -n '3,/^set -uo/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//; $d'
}

while [ $# -gt 0 ]; do
    case "$1" in
        -h | --help)
            usage
            exit 0
            ;;
        --platform)
            PLATFORM="${2:-}"
            shift 2
            ;;
        --ascend-soc)
            ASCEND_SOC="${2:-}"
            shift 2
            ;;
        --cuda-capability)
            CUDA_CAPABILITY="${2:-}"
            shift 2
            ;;
        --tokens)
            TOKENS="${2:-}"
            shift 2
            ;;
        --block-size)
            BLOCK_SIZE="${2:-}"
            shift 2
            ;;
        --tokens-per-chunk)
            TOKENS_PER_CHUNK="${2:-}"
            shift 2
            ;;
        --connector)
            CONNECTOR="${2:-}"
            shift 2
            ;;
        --device-id)
            DEVICE_ID="${2:-}"
            shift 2
            ;;
        --additional-config)
            ADDITIONAL_CONFIG="${2:-}"
            shift 2
            ;;
        --cases)
            CASES="${2:-}"
            shift 2
            ;;
        --out)
            OUT_DIR="${2:-}"
            shift 2
            ;;
        --keep-store)
            KEEP_STORE="1"
            shift
            ;;
        -*)
            die "unknown option: $1 (try --help)"
            ;;
        *)
            [ -z "$MODEL" ] || die "more than one model directory given: $MODEL and $1"
            MODEL="$1"
            shift
            ;;
    esac
done

[ -n "$MODEL" ] || {
    usage
    exit 2
}
[ -d "$MODEL" ] || die "model directory not found: $MODEL"
[ -f "$MODEL/config.json" ] || die "no config.json in $MODEL"

case "$CONNECTOR" in
    v2) CONNECTOR_MODULE="ucm.integration.vllm.v2.ucm_connector" ;;
    legacy) CONNECTOR_MODULE="ucm.integration.vllm.ucm_connector" ;;
    *) die "--connector must be v2 or legacy, got: $CONNECTOR" ;;
esac

MODEL_NAME="$(basename -- "$MODEL")"
RUN_STAMP="$(date +%Y%m%d-%H%M%S)"
[ -n "$OUT_DIR" ] || OUT_DIR="${PWD}/model-check-runs/${MODEL_NAME}-${RUN_STAMP}"
mkdir -p "$OUT_DIR" || die "cannot create output directory: $OUT_DIR"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

# UCM's C++ core is installed inside the ucm/ tree; add those directories while
# keeping the platform's own library paths.
UCM_LIB_DIRS="$(find "${REPO_ROOT}/ucm" -name '*.so' -printf '%h\n' 2> /dev/null | sort -u | paste -sd:)"
if [ -n "$UCM_LIB_DIRS" ]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${UCM_LIB_DIRS}"
else
    printf '%s\n' "WARNING: no .so found under ${REPO_ROOT}/ucm -- was UCM built?" >&2
fi

TOOLKIT_CMD=""
if command -v ucm-toolkit > /dev/null 2>&1; then
    TOOLKIT_CMD="ucm-toolkit"
else
    # Not installed as a console script: run the CLI module from the checkout.
    export PYTHONPATH="${REPO_ROOT}/toolkit:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
    TOOLKIT_CMD="python -m ucm_toolkit.cli"
    printf '%s\n' "note: ucm-toolkit not on PATH, using: ${TOOLKIT_CMD}"
fi

# ---------------------------------------------------------------------------
# Preflight: report what will actually be exercised
# ---------------------------------------------------------------------------

printf '%s\n' "=== model-check run ==================================================="
printf 'model        : %s\n' "$MODEL"
printf 'repo         : %s (%s)\n' "$REPO_ROOT" "$(git -C "$REPO_ROOT" rev-parse --short HEAD 2> /dev/null || echo 'not a git checkout')"
printf 'platform     : %s\n' "$PLATFORM"
printf 'connector    : %s (%s)\n' "$CONNECTOR" "$CONNECTOR_MODULE"
printf 'tokens       : %s   block-size: %s\n' "$TOKENS" "$BLOCK_SIZE"
printf 'cases        : %s\n' "$CASES"
printf 'output       : %s\n' "$OUT_DIR"

python - <<'PREFLIGHT' 2>&1 | sed 's/^/preflight    : /'
import importlib.util as u

for name, hint in (
    ("torch", "required"),
    ("ucm", "build it: unset PLATFORM; pip install -v -e . --no-build-isolation"),
    ("wrapt", "pip install wrapt  (several vLLM images ship without it)"),
    ("vllm", "required"),
    ("vllm_ascend", "only needed for the ascend / ascend-sim launchers"),
):
    found = u.find_spec(name) is not None
    print(f"{name:12} {'ok' if found else 'MISSING -- ' + hint}")
PREFLIGHT
printf '%s\n' "======================================================================"

# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

run_case() {
    case_name="$1"
    shift
    log="${OUT_DIR}/${case_name}.log"
    store="${OUT_DIR}/store-${case_name}"
    [ "$KEEP_STORE" = "1" ] || rm -rf "$store"
    mkdir -p "$store"

    set -- "$@" \
        --model "$MODEL" \
        --tokens "$TOKENS" \
        --block-size "$BLOCK_SIZE" \
        --storage-backends "$store" \
        --connector-module-path "$CONNECTOR_MODULE" \
        --device-id "$DEVICE_ID"
    [ "$PLATFORM" = "auto" ] || set -- "$@" --platform "$PLATFORM"
    [ -z "$ASCEND_SOC" ] || set -- "$@" --ascend-soc "$ASCEND_SOC"
    [ -z "$CUDA_CAPABILITY" ] || set -- "$@" --cuda-capability "$CUDA_CAPABILITY"
    [ -z "$TOKENS_PER_CHUNK" ] || set -- "$@" --tokens-per-chunk "$TOKENS_PER_CHUNK"
    [ -z "$ADDITIONAL_CONFIG" ] || set -- "$@" --additional-config "$ADDITIONAL_CONFIG"

    printf '\n--- %s ---\n' "$case_name"
    printf 'command: %s run model-check %s\n' "$TOOLKIT_CMD" "$*"
    printf 'log    : %s\n' "$log"

    # shellcheck disable=SC2086
    $TOOLKIT_CMD run model-check "$@" > "$log" 2>&1
    status=$?

    # The checker prints its verdicts with a stable prefix.
    grep -E '^\[ucm-kv-check\] (PASS|PARTIALLY|black-box|strict|hit check|transfer-range|native oracle)' "$log" |
        sed 's/^/  /' | tail -25

    # A case counts as passed only when the checker itself printed PASS. The
    # exit code alone is not enough: a wrapper that never reached the checker
    # would exit 0 too, and reporting that as a pass is exactly the mistake
    # these checks exist to prevent.
    if ! grep -qE '^\[ucm-kv-check\] PASS' "$log"; then
        if [ "$status" -eq 0 ]; then
            printf '  FAILED: exited 0 but printed no PASS verdict.\n'
            printf '  The checker was probably never reached -- see %s\n' "$log"
        fi
        status=1
    fi
    if [ "$status" -ne 0 ]; then
        printf '  FAILED (exit %s). Last lines:\n' "$status"
        grep -E 'Error|error:|Assertion|Traceback|Unsupported|^error:' "$log" |
            tail -5 | sed 's/^/    /'
    fi
    return "$status"
}

overall=0
summary=""

case ",${CASES}," in
    *,layout,*)
        run_case "layout-only" --layout-only
        rc=$?
        [ "$rc" -eq 0 ] || overall=1
        summary="${summary}layout-only:$([ "$rc" -eq 0 ] && echo PASS || echo FAIL) "
        ;;
esac

case ",${CASES}," in
    *,bulk,*)
        run_case "bulk" --no-layerwise
        rc=$?
        [ "$rc" -eq 0 ] || overall=1
        summary="${summary}bulk:$([ "$rc" -eq 0 ] && echo PASS || echo FAIL) "
        ;;
esac

case ",${CASES}," in
    *,layerwise,*)
        run_case "layerwise" --layerwise
        rc=$?
        [ "$rc" -eq 0 ] || overall=1
        summary="${summary}layerwise:$([ "$rc" -eq 0 ] && echo PASS || echo FAIL) "
        ;;
esac

printf '\n=== summary ==========================================================\n'
printf 'model   : %s\n' "$MODEL_NAME"
printf 'result  : %s\n' "${summary:-no case selected}"
printf 'logs    : %s\n' "$OUT_DIR"
printf '%s\n' "A PASS covers layout parsing and byte-level dump/load in host memory."
printf '%s\n' "It does not cover device copy kernels, event ordering or performance;"
printf '%s\n' "read each case's PARTIALLY VERIFIED lines for the groups without an"
printf '%s\n' "independent required-restore set."
printf '%s\n' "======================================================================"
exit "$overall"
