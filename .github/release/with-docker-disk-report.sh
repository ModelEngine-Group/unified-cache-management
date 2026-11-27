#!/usr/bin/env bash
# Report the filesystem that actually holds Docker layers, including failed imports.
set -eu
root="$(docker info --format '{{.DockerRootDir}}')" || root=""
report() {
  if [[ -n "$root" ]]; then
    df -B1 "$root" || true
    df -i "$root" || true
  fi
}
report
trap report EXIT
"$@"
