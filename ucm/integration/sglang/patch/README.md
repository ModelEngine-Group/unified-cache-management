# SGLang L1 external-linker patch

SGLang versions containing [PR #40595](https://github.com/sgl-project/sglang/pull/40595)
already support out-of-tree linkers natively and do not need the runtime patch.
For older releases, install UCM with `ucm_patch.pth` and set
`ENABLE_UCM_PATCH=1` before importing SGLang. SGLang 0.5.20 uses its dedicated
patch; 0.5.21 and 0.5.22 share the `v0521` patch. Version selection follows
the explicit patch-required version map, like the vLLM integration. Other
versions receive no runtime patch; the launcher verifies their native linker
support before starting. `UCM_SGLANG_VERSION` may pin an expected version.
Like vLLM, the import hook is installed by `ucm_patch.pth`, while
`apply_all_patches()` checks `ENABLE_UCM_PATCH` and selects the versioned patch.

Select UCM through the dynamic linker configuration:

```bash
--enable-unified-cache-external-linker \
--unified-cache-external-linker-backend dynamic \
--unified-cache-external-linker-config '{
  "linker": "UcmDirectLinker",
  "linker_module_path": "ucm.integration.sglang.ucm_direct_linker",
  "linker_extra_config": {"config_file": "/path/to/ucm-config.yaml"}
}'
```

The JSON config is validated during CLI parsing. The linker class is imported
and constructed in the cache-owning worker. The `config_file` option replaces
the need for `UNIFIEDCACHE_CONFIG_FILE` in this path. SGLang 0.5.20 also keeps
its legacy `ucm` backend selector. This release backport uses the new config
flag for `dynamic` only; Mooncake and MORI retain their existing configuration
flags.

The shared config parser, CLI hook, and dynamic linker factory live in `v0520`.
The `v0521` package reuses them for SGLang 0.5.21 and 0.5.22, with a registry
hook for the public factory signature. It also adds the shared
`LayerWiseLoadCounter` on 0.5.21; 0.5.22 already provides that class. Both
versions share the CLI and linker factory backports in `v0520`.
`UcmDirectLinker` implements the linker load completion contract. For native
releases, the import hook leaves SGLang
unchanged.

SGLang 0.5.20 and 0.5.21 use the older load-completion API. A Store load
failure can therefore surface through the layer wait instead of the newer
request-level load-failure path available from 0.5.22 onward.

Mooncake and Mori retain their upstream selection, initialization, and default
behavior. The UCM store adapter in `ucm_direct_store_adapter.py` reads the single
`ucm_connectors` entry from the configured YAML file and creates one UCM Store
per linker-selected device-pool group. It creates one backend subdirectory per
Store, named from
the physical pool names in that Store (for example, `kv`, `swa`, or
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

Edit the local configuration block in
`scripts/launch_sglang_ucm_linker.sh`, then run it to start SGLang with the
UCM backend and a real Store. The launcher exports the patch and UCM
configuration variables, accepts either SGLang's native linker factory or an
active compatibility patch, and writes the service and patch checks to its log
directory. Native SGLang releases do not require `ucm_patch.pth`.

Runtime patch modules are Python files; no diff files are loaded.
