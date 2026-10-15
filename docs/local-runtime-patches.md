# Local runtime patches on UCM v0.8.0

`v0.8.0-patch` starts from official UCM v0.8.0 commit
`3e781c1c90f3a8ce4e6453f63c03360dfbcdeb5d`. It contains the common runtime
synchronization fix without adding Early prefetch. The same fix is included in
`codex/ucm-deferred-pin-wait-v0.8.0`, alongside the Early-only query-range fix.

## Sparse MLA layer-load synchronization

Enable the packaged patch before Python imports vLLM:

```bash
export ENABLE_UCM_PATCH=1
export GLM51_LAYERWISE_SYNC_FIX=1
```

Install UCM from the chosen branch using the repository's normal installation
instructions. The existing `ucm_patch.pth` entry loads the patch through
`apply_all_patches()`. The experiment's `sitecustomize.py` and observation hooks
are not required. When the flag is absent or not exactly `1`, this fix is not
installed. Existing experiment hooks use the same installation markers, so
loading both implementations does not wrap methods twice.

For sparse `MultiHeadLatentAttentionWrapper` instances with an indexer and an
active UCM layerwise connector, the patch waits before the original forward
reads indexer cache data. Each batch resets the set of completed layer waits;
the original main-attention entry's repeated wait is skipped by layer ID. This
also avoids submitting the next layer's load twice. Original computation,
post-attention cache saving, and connector load-failure handling are retained.

The patch is a formatted port of the same `runtime_hooks/layerwise_sync.py`
recorded in all three source experiments:

- `experiment_layerwise_sync_static_20261015T033640Z` (frozen source analysis)
- `experiment_n40_prefetch_repeats_20261012T112647Z`
- `experiment_swe_full3379_b8192_20261013T101025Z`

Original hook SHA-256:
`6ef7af01d5e43d7a9dd71100f212e9baf884029a43fc5a9858464407b618b549`.
The implementation is in
`ucm/integration/vllm/patch/layerwise_sync_patch.py`; installation is integrated
into UCM instead of requiring an experiment-local `PYTHONPATH` directory.

The supplied evidence concerns the inspected NVIDIA sparse MLA path and its
specific GLM/vLLM configurations. It does not establish a need for this patch
on every model, vLLM version, or Ascend path. Historical ON/OFF prefetch groups
both enabled this fix; their results do not measure the fix's own performance
benefit. New hardware inference or benchmark results are not implied by this
source integration.

## Early-only READY prefix query range

On `codex/ucm-deferred-pin-wait-v0.8.0`, `EarlyPipelineStore.lookup_on_prefix`
counts protected hits from the actual query slice. A waiting-request retry can
resume near the previous hit boundary instead of the HBM boundary; counting
from HBM again overstates the remaining protected hits. The local fix locates
the query in the prepared block list, bounds the result by the protected hit
range and query length, and stops at a mismatched block.

This change and its waiting-retry/SWE-boundary regression cases come from the
experiment UCM checkout at `692c3d231861be577d0283f222f638360202571e` plus its
local changes in `ucm/store/pipeline/early_connector.py` and
`test/test_early_ready_reuse.py`. Official v0.8.0 has no `EarlyPipelineStore`, so
this fix is only present on the Early branch.

## Focused offline validation

The synchronization unit tests exercise ordering before the indexer, original
forward/save behavior, repeated waits, batch resets, failed waits, inactive
paths, existing hook coexistence, and opt-in routing. They use stand-in vLLM
interfaces and do not require model weights or a GPU:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest --confcutdir=test/suites/Unit \
  -q test/suites/Unit/test_layerwise_sync_patch.py \
  test/suites/Unit/test_apply_patch_version.py
```

The Early branch additionally includes offline prefix-range unit tests and
the original integration regressions in `test/test_early_ready_reuse.py`.
The latter require a compatible vLLM installation and built UCM extensions.

On 2026-10-15, the focused synchronization/version checks passed all 63 cases,
and the Early prefix-range checks passed all 22 cases. The original Early
integration suite and model inference were not rerun: the earlier migration
environment and the experiment runtime's vLLM/native dependencies were not
available in this publishing environment. These offline results do not replace
the historical hardware evidence or establish complete end-to-end validation.
