#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PIP_BIN="${PIP_BIN:-$PYTHON_BIN -m pip}"

# detect ABI from torch
ABI="$("$PYTHON_BIN" - <<'PY'
import sys
try:
    import torch
    print(int(torch._C._GLIBCXX_USE_CXX11_ABI))
except Exception as e:
    print(f"[install_ucm] ERROR: cannot detect ABI via torch: {e}", file=sys.stderr)
    sys.exit(2)
PY
)"

echo "[install_ucm] Detected torch CXX11 ABI flag: $ABI"

WHEEL=( "$SCRIPT_DIR"/ucm_whls/ucm_abi"${ABI}"*.whl )

echo "[install_ucm] Installing: ${WHEEL[0]}"
$PIP_BIN install --no-deps --force-reinstall "${WHEEL[0]}"

WHEEL=( "$SCRIPT_DIR"/uc_hash_ext*.whl )
echo "[install_uc_hash_ext] Installing: ${WHEEL[0]}"
$PIP_BIN install --no-deps --force-reinstall "${WHEEL[0]}"

MINDIE_DIR="$($PYTHON_BIN - <<'PY'
import importlib.util, os, sys

spec = importlib.util.find_spec("mindie_llm")
if spec is None:
    sys.stderr.write("ERROR: mindie_llm is not importable in this Python env. Please install/activate the correct env.\n")
    sys.exit(1)

if spec.submodule_search_locations:
    print(next(iter(spec.submodule_search_locations)))
else:
    print(os.path.dirname(spec.origin))
PY
)"

install -m 0644 "$SCRIPT_DIR/uc_utils.py"               "$MINDIE_DIR/text_generator/mempool/uc_utils.py"
install -m 0644 "$SCRIPT_DIR/unifiedcache_mempool.py"   "$MINDIE_DIR/text_generator/mempool/unifiedcache_mempool.py"
install -m 0644 "$SCRIPT_DIR/prefix_cache_plugin.py"    "$MINDIE_DIR/text_generator/plugins/prefix_cache/prefix_cache_plugin.py"
echo "Installed MindIE-LLM files to: $MINDIE_DIR"
