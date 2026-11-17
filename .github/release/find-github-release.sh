#!/usr/bin/env bash
set -euo pipefail

# Print zero or one Release as a JSON array. Ambiguity exits 3; lookup errors exit 1.
if [ "$#" -ne 2 ] || [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: find-github-release.sh OWNER/REPOSITORY TAG" >&2
  exit 1
fi
repository="$1"
tag="$2"

if ! releases="$(gh api --paginate --slurp \
  "/repos/${repository}/releases?per_page=100" \
  | jq -cse --arg tag "${tag}" '
      if length != 1 or (.[0] | type) != "array" then
        error("expected one paginated Release response")
      else .[0] end
      | if all(.[]; type == "array") then . else
          error("expected Release pages")
        end
      | [.[][] | select(
          (.tag_name == $tag) or
          (.draft == true and .name == $tag and
           ((.tag_name // "") | startswith("untagged-"))))]
    ')"; then
  echo "Failed to look up Tag ${tag} Releases in ${repository}" >&2
  exit 1
fi

count="$(jq 'length' <<<"${releases}")"
if [ "${count}" -gt 1 ]; then
  echo "Tag ${tag} resolves to ${count} Releases; refusing ambiguity" >&2
  exit 3
fi
printf '%s\n' "${releases}"
