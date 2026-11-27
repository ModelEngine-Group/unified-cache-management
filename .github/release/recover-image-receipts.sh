#!/usr/bin/env bash
# Completed Tag runs can contribute pins only for their exact source and frozen plan.
set -euo pipefail
plan_dir="$1"
runs="$(gh api --paginate --method GET \
  "repos/${GH_REPO}/actions/workflows/release-tag.yml/runs" \
  -f event=push -f head_sha="${SOURCE_SHA}" -F per_page=100 \
  --jq '.workflow_runs[] | select(.status == "completed") | [.id,.head_branch,.head_repository.full_name] | @tsv')"
while IFS=$'\t' read -r id tag repo; do
  [[ "$repo" == "$GH_REPO" && "$tag" == "$RELEASE_TAG" && "$id" != "$GITHUB_RUN_ID" ]] || continue
  pattern="ucm-image-*-receipt-*-attempt-*-run-${id}"
  receipts="$(gh api --paginate "repos/${GH_REPO}/actions/runs/${id}/artifacts?per_page=100" \
    --jq '.artifacts[] | select(.expired == false) | .name | select(test("^ucm-image-(index|member)-receipt-.*-attempt-[0-9]+-run-[0-9]+$"))')"
  if [[ -n "$receipts" ]]; then
    gh run download "$id" -R "$GH_REPO" --pattern "$pattern" --dir "${plan_dir}/prior-image-receipts/${id}"
  fi
done <<< "$runs"
