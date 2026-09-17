#!/usr/bin/env bash
set -euo pipefail

# Build the ucm-toolkit wheel from the toolkit/ subproject.
cd "$(dirname "$0")/../toolkit"

python -m pip install build
python -m build --wheel
