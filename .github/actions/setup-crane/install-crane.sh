#!/usr/bin/env bash
set -euo pipefail

# Shared by the composite action and conditional registry-retention setup.
if [ "$#" -ne 1 ] || [ -z "$1" ]; then
  echo "Usage: install-crane.sh INSTALL_DIRECTORY" >&2
  exit 1
fi

mkdir -p "$1"
curl -fsSL --retry 3 \
  https://github.com/google/go-containerregistry/releases/download/v0.20.3/go-containerregistry_Linux_x86_64.tar.gz \
  | tar -xz -C "$1" crane
