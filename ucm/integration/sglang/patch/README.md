# SGLang L1 external-linker patch

SGLang versions containing [PR #40595](https://github.com/sgl-project/sglang/pull/40595)
already support out-of-tree linkers natively and do not need the runtime patch.
For older releases, install UCM with `ucm_patch.pth` and set
`ENABLE_UCM_PATCH=1` before importing SGLang. SGLang 0.5.20 uses its dedicated
patch; 0.5.21 and 0.5.22 share the `v0521` patch. Version selection follows
the explicit patch-required version map, like the vLLM integration. Unsupported
versions log a warning and receive no runtime patch. They can still use their
existing backends, including HiCache; skipping a patch does not add the dynamic
linker API to an older release. Like vLLM, version selection strips rc, post and
local build suffixes: `0.5.21rc1`, `0.5.21.post1` and `0.5.21+build` select the
`0.5.21` patch. Patch application still validates the framework extension points.
`UCM_SGLANG_VERSION` may pin the exact installed version, including any suffixes;
this check does not normalize the version, and a mismatch remains an error.
Like vLLM, the import hook is installed by `ucm_patch.pth`, while
`apply_all_patches()` checks `ENABLE_UCM_PATCH` and selects the versioned patch.

After installing UCM and its native libraries, create a configuration with one
Store entry. The following example uses Cache and Posix without capacity GC:

```yaml
# /path/to/ucm-sglang.yaml
ucm_connectors:
  - ucm_connector_name: UcmPipelineStore
    ucm_connector_config:
      store_pipeline: "Cache|Posix"
      storage_backends: /mnt/ucm-cache
      posix_capacity_gb: 0
      sglang_load_prefetch_layers: 2
```

Create the backend directory, then start SGLang directly. Replace the model,
configuration and storage paths with paths available to the cache-owning workers:

```bash
mkdir -p /mnt/ucm-cache
# Required for the 0.5.20/0.5.21/0.5.22 compatibility patches.
# Native releases may leave ENABLE_UCM_PATCH unset.
export ENABLE_UCM_PATCH=1
export UNIFIEDCACHE_CONFIG_FILE=/path/to/ucm-sglang.yaml

python -m sglang.launch_server \
--model-path /path/to/model \
--tensor-parallel-size 1 \
--page-size 64 \
--enable-unified-cache-external-linker \
--unified-cache-external-linker-backend dynamic \
--unified-cache-external-linker-config '{
  "linker": "UcmDirectLinker",
  "linker_module_path": "ucm.integration.sglang.ucm_direct_linker"
}'
```

The JSON config is validated during CLI parsing. The compatibility patch logs
a warning if a config is supplied without
`--enable-unified-cache-external-linker`; the config does not enable the linker
by itself. Config validation still applies in that case. The linker class is
imported and constructed in the cache-owning worker. Alternatively, set
`linker_extra_config.config_file` in the JSON object instead of exporting
`UNIFIEDCACHE_CONFIG_FILE`. SGLang 0.5.20 also keeps
its legacy `ucm` backend selector. This release backport uses the new config
flag for `dynamic` only; Mooncake and MORI retain their existing configuration
flags.

The shared config parser, CLI hook, and dynamic linker factory live in `v0520`.
The `v0521` package reuses them for SGLang 0.5.21 and 0.5.22, with a registry
hook for the public factory signature. It also adds the shared
`LayerWiseLoadCounter` on 0.5.21; 0.5.22 already provides that class. Both
versions share the CLI and linker factory backports in `v0520`.
`UcmDirectLinker` implements the linker load completion contract. For native
releases, the import hook leaves SGLang unchanged. A failure applying a patch to
a supported release remains an error, rather than continuing with a partial patch.

SGLang 0.5.20 and 0.5.21 use the older load-completion API. A Store load
failure can therefore surface through the layer wait instead of the newer
request-level load-failure path available from 0.5.22 onward.

Mooncake and Mori retain their upstream selection, initialization, and default
behavior. The UCM store adapter in `ucm_direct_store_adapter.py` reads the single
`ucm_connectors` entry from the configured YAML file and creates one UCM Store
per linker-selected device-pool group. Each Store gets a backend subdirectory
named from its physical pools (for example, `kv`, `swa`, or
`deepseek_v4_c128`). If the same physical pool set is split into multiple
Stores, a layer-range or layout suffix keeps their directories separate. Model
identity remains part of each block ID, so different models do not hit the
same cached prompt even when they share a backend directory. DSA KV and
INDEXER data share one composite block when they belong to the same pool group.

The SGLang UCM configuration may set `sglang_load_prefetch_layers` under
`ucm_connector_config`. The default is `2`: the linker submits the current
logical layer and two future layers, then waits and reports completion one
layer at a time. Use `0` for the previous submit-and-wait behavior. Like the
vLLM HLA prefetch setting, the value is converted to an integer; invalid values
log a warning and fall back to `2`. Values outside `0` to `8` are clamped to
that range. The setting is consumed by the SGLang adapter and is not passed to
the Store backends.

The offload worker submits queued radix-tree nodes asynchronously, polls
in-flight Store tasks with `Check()`, and calls `Wait()` to obtain final status
and release task handles. Results are reported in node submission order, even
if a later node finishes first. There is no SGLang offload concurrency setting.
With `Cache|Posix`, this reflects the Cache task's completion; the Cache
backend's Posix write can still finish later.

For capacity GC, set `posix_capacity_gb` to the total budget in GiB. The adapter
splits it between groups by `capacity_pages * block_bytes`, rounding to integer
GiB and distributing the remainder so the allocations sum to the configured
budget. A group allocated `0` has capacity GC disabled; its storage usage is not
bounded by that allocation. Posix validates the minimum positive capacity.
Zero estimated bytes across all groups is invalid metadata and raises an error;
it does not fall back to assigning the entire budget to every Store.
Only DP0 / attention-DP0 / TP0 is nominated to run GC, subject to
`posix_gc_enable` (default `true`) and Posix's directory lease. The adapter enables
`posix_hotness_enable` only on that rank for groups with positive capacity.

Runtime patch modules are Python files; no diff files are loaded.

## Validation

The CPU preparation, patch-selection and single-worker offload tests use fake
framework/native dependencies and require NumPy:

```bash
python -m unittest discover -s test -p test_sglang_direct_preparation.py
python -m unittest discover -s test -p test_sglang_patch_selection.py
python -m unittest discover -s test -p test_sglang_offload_workers.py
```

To check address order, sizes, offsets, strides and page capacity against the
installed SGLang `DevicePoolEntry`, run the layout contract test in a Linux
environment with UCM, SGLang, PyTorch and the repository's test dependencies
installed (see `test/requirements.txt`). The test uses CPU tensors and does not
load a model or allocate GPU tensors:

```bash
ENABLE_UCM_PATCH=1 python -m pytest \
  test/suites/Unit/test_sglang_linker_layout_contract.py \
  --stage=0 --feature=sglang_linker --platform=cpu
```

This contract test covers packed composite pools, multiple buffers per logical
layer, sparse layer mappings, unpacked pools and both page-row conventions.
Run it on each supported SGLang release; fake-framework tests alone do not
validate compatibility with the installed framework.
