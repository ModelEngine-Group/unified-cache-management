#!/usr/bin/env bash


check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "❓❓$1 is not installed, please run:"
        echo "# Install lint deps"
        echo "pip install -r requirements-lint.txt"
        echo "# (optional) Enable git commit pre check"
        echo "pre-commit install"
        echo ""
        echo "See step by step contribution guide:"
        echo "Unifiedcache Official Website"
        exit 1
    fi
}

check_command pre-commit

# TODO: cleanup SC exclude
export SHELLCHECK_OPTS="--exclude=SC2046,SC2006,SC2086"
if [[ "$1" != 'ci' ]]; then
    pre-commit run --all-files
else
    pre-commit run --all-files --hook-stage manual
fi