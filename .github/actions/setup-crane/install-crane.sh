#!/usr/bin/env bash
set -euo pipefail

# Shared by the composite action and conditional registry-retention setup.
if [ "$#" -ne 1 ] || [ -z "$1" ]; then
  echo "Usage: install-crane.sh INSTALL_DIRECTORY" >&2
  exit 1
fi

mkdir -p "$1"
case "$(uname -m)" in
  x86_64) arch=x86_64 ;;
  aarch64|arm64) arch=arm64 ;;
  *) echo "Unsupported crane architecture" >&2; exit 1 ;;
esac
curl -fsSL --retry 3 \
  "https://github.com/google/go-containerregistry/releases/download/v0.20.3/go-containerregistry_Linux_${arch}.tar.gz" \
  | tar -xz -C "$1" crane
