# Pipeline Store

Use `Cache|Posix` when KV blocks should survive a serving-process restart on a
local disk or mounted filesystem, while recently accessed blocks remain in host
memory. `Cache` owns device-to-host transfers and the host buffer; `Posix` owns
files, lookup, and filesystem I/O. The engine still owns its device KV cache.

Select a registered pipeline through `UcmPipelineStore`; see the [configuration reference](../../reference/config-parameters.md) for available names.

## Configure Cache and Posix

Start from [Installation](../../user-guide/quick_start/index.md), or
[build from source](../build_from_source.md) in the target
engine environment. This pipeline requires the `ucmpipelinestore` extension,
`libcachestore.so`, and `libposixstore.so`. A successful package import alone does
not show that the selected pipeline can initialize.

Create a dedicated writable directory, mount it into every process that needs
access, and save this as a UCM YAML file:

```yaml
ucm_connectors:
  - ucm_connector_name: "UcmPipelineStore"
    ucm_connector_config:
      store_pipeline: "Cache|Posix"
      store_health:
        enabled: true
      storage_backends: "/mnt/ucm-cache"
      io_direct: false
      # cache_buffer_capacity_gb: 256
      # posix_capacity_gb: 1024
      use_gdr: false
enable_event_sync: true
use_layerwise: true
enable_record_traces: false
use_lite: false
persist_token_threshold: 0
```

Connect the file through `kv_connector_extra_config.UCM_CONFIG_FILE` using the
[vLLM quickstart](../../user-guide/quick_start/index.md#vllm), or its
[Ascend counterpart](../../user-guide/quick_start/index.md#vllm-ascend). The connector
supplies device IDs, block sizes, and tensor layout; do not copy these values
from a different model's Store test.

The sample retains the original `io_direct: false` setting without overriding the I/O engine.
`io_direct` defaults to `true` in the native Cache and Posix stages. Switch to
`posix_io_engine: aio` only with `io_direct: true` and a filesystem that supports
the resulting aligned I/O. Put `timeout_ms` inside `ucm_connector_config`; it is
a Store task timeout, not an HTTP request timeout.

## Budget host memory

The capacity fields remain commented out, so runtime defaults apply. The commented 256 is not the effective capacity for every model.
Cache Store requires enough space for at least
`max(1024, 2 * cache_load_exclusive_buffer_number)` shards; the default exclusive
buffer count is 1024. Initialization reports the minimum required GiB when the
chosen capacity is too small.

| Allocation path without an explicit capacity | Effective default |
| --- | --- |
| Native Cache with shared buffers | 256 GiB |
| Native Cache with `share_buffer_enable: false` | 32 GiB per worker |
| Current vLLM connector with shared buffers | 128 GiB |

The vLLM connector defaults `share_buffer_enable` to whether the model uses MLA.
A positive `cache_buffer_capacity_gb` overrides the native defaults. Shared
buffers need sufficient `/dev/shm`; unshared workers allocate separately, so
multiply their budget by the number of workers on the host. Model weights,
device KV cache, and other host allocations remain additional costs.

## Storage capacity and health

`posix_capacity_gb: 0` disables capacity-driven garbage collection. A positive
value supplies a capacity budget; it does not reserve filesystem space. The
vLLM connector assigns Posix GC to its DP0 scheduler. When multiple serving
instances share a directory, account for their combined data and retain the
Posix coordination settings unless you have validated a different ownership
arrangement.

Pipeline health checking is enabled by default, with a 10-second interval,
3-second timeout, 8-sample window, and failure threshold of 2. Posix implements
a write/read/compare/remove probe. A successful probe establishes filesystem
health; only request-level hits and completed loads establish KV reuse. See
[Health metrics](../../user-guide/observability/health-metrics.md).

## Pipeline parallelism

With `pipeline_parallel_size > 1` the layer-wise connector gives every PP stage
its own store instead of sharing one namespace:

| | Example (PP=2, backend `/data/ucm`, engine id `3f9c1a`, layers 8) |
| --- | --- |
| Storage path | `/data/ucm/pp_0_l0-4`, `/data/ucm/pp_1_l4-4` |
| Cache id | `3f9c1a_pp_0_l0-4`, `3f9c1a_pp_1_l4-4` |

The stage directory is created before the Posix store opens, and the block key
format is unchanged, so keys stay comparable across stages.

**The layer allocation is part of the namespace.** A stage directory names the
rank together with the layers that stage owns (`pp_<rank>_l<first>-<count>`),
where the allocation comes from `VLLM_PP_LAYER_PARTITION`. This is not cosmetic:
keying only on the rank makes a re-partitioned restart read the wrong layers.
With 12 layers, `4,4,4` gives stage 1 layers 4-7 and `2,4,6` gives stage 1
layers 2-5. Stage 1 keeps four layers in both runs, so `block_size` and
`shard_size` still match, the old files still load, and shard 0 — layer 4
before, layer 2 after — would silently feed layer 2 the cached KV of layer 4.
Including the allocation in the directory (and in the Cache id) makes the
old data unreachable instead. Every worker publishes its allocation through
the handshake so the scheduler names the directory identically.

**Budget split.** On a node holding several stages, `cache_buffer_capacity_gb` is
divided by the number of local stages, and `posix_capacity_gb` is divided by the
full `pipeline_parallel_size`. A split that would round down to 0 aborts startup
rather than silently falling back to a native default.

**Geometry comes from the handshake.** Each worker publishes the real block and
shard sizes of its stage, and the scheduler builds the matching stores from that
metadata. The previous "per-layer size × total model layers" estimate is gone
because it sized a stage's block by layers the stage does not hold.

**Shard indices are stage-local.** A stage stores only its own layers, so its
shard indices restart at 0. With PP=2 over 32 layers, layer 16 is shard 0 of
stage 1, not shard 16.

**A block is a hit only when every stage has it.** Lookup runs against each
stage store and takes the shortest common prefix: PP0 may report ten blocks
while PP1 has only four, in which case four is the hit and the remaining six are
recomputed by vLLM. A block is never handed to the load path unless all stages
can serve it, so a failed or still-in-flight stage dump, a restarted stage, or a
stage-local GC eviction simply shortens the hit instead of producing a load
error. The cost is one lookup round trip per stage — the later stages keep
Posix-only stores, so a stage on another node pays a network round trip on the
scheduler's critical path, once per batch of candidate blocks rather than once
per block.

**Stage queries run in parallel.** PP0 is queried first because a full miss there
settles the lookup without touching the remote stages; the remaining stages are then
queried concurrently on a connector-owned thread pool sized to the stage count. The
pool lives for the whole process, so the metrics registry sees a fixed number of
threads that does not grow with the number of lookups.

**`kv_load_failure_policy="recompute"` is still recommended.** The intersection
removes the PP0-versus-later-stage mismatch, but one race remains: a stage store
can evict a block after the lookup reported it and before the load runs. With
the default `"fail"` that race ends the request; with `"recompute"` it degrades
to partial reuse. A stage load error is still reported through
`get_block_ids_with_load_errors()` as the safety net.

**Scope.** PP is supported by the layer-wise connector on a shared Posix backend
reachable from every stage. Non-Posix backends and the direct connector are out
of scope, and context parallelism combined with PP is rejected at startup.

**Upgrading from a flat layout.** Data written by an earlier version sits
directly under the configured root rather than under a stage directory. Data
written by the first PP version sits under `pp_<rank>` without the layer
suffix. Neither layout is found after the upgrade, and the new stage namespaces
do not reclaim them. Delete both once you have confirmed nothing else still
reads those directories. Because changing `VLLM_PP_LAYER_PARTITION` produces a
new directory set, an intentional repartition leaves the previous directories
behind as well; remove them when the disk budget matters.

## Other registered pipelines

| Requirement | Selection |
| --- | --- |
| 3FS client I/O | [`Cache|Ds3fs`](ds3fs.md) |
| Smaller stored BF16 payload, with a quality tradeoff | [`Cache|Compress|Posix`](compress.md) |
| Shared Mooncake memory, optionally backed by files | [`Mooncake` or `Mooncake|Posix`](mooncake.md) |

SGLang already manages a host cache, so its adapter selects the `Posix` stage
directly. Use its [quickstart](../../user-guide/quick_start/index.md#sglang) rather than
passing the vLLM YAML unchanged.

## Original serving and request commands

Save the YAML above as `/etc/ucm/ucm.yaml` in the matching engine environment, then use the original Qwen2.5-14B-Instruct serving parameters below. `--no-enable-prefix-caching` is retained for the original storage test; remove it for normal deployments that also use native prefix caching.

```bash
export ENABLE_UCM_PATCH=1
vllm serve Qwen/Qwen2.5-14B-Instruct \
--max-model-len 20000 \
--tensor-parallel-size 2 \
--gpu_memory_utilization 0.87 \
--block_size 128 \
--trust-remote-code \
--port 7800 \
--enforce-eager \
--no-enable-prefix-caching \
--kv-transfer-config \
'{
    "kv_connector": "UCMConnector",
    "kv_role": "kv_both",
    "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
    "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/etc/ucm/ucm.yaml"}
}'
```

After readiness, run the original request workload from another terminal. These are workload inputs, not reproduced performance results:

```bash
vllm bench serve \
--backend vllm \
--model Qwen/Qwen2.5-14B-Instruct \
--host 127.0.0.1 \
--port 7800 \
--dataset-name random \
--num-prompts 12 \
--random-input-len 16000 \
--random-output-len 2 \
--request-rate inf \
--seed 123456 \
--percentile-metrics "ttft,tpot,itl,e2el" \
--metric-percentiles "90,99" \
--ignore-eos
```

## Verify writes and reads

Use the [shared verification procedure](../../user-guide/observability/verify-cache.md) to save, restart and replay. Check Posix reads and errors for filesystem access. If host cache satisfies the load, a dedicated diagnostic run can use `cache_load_backend_only: true` to inspect the backend path.


## Measure the storage benefit { #historical-performance-report }

Compare uncached prefill, storage replay, and warm-memory replay separately.
Record external-hit tokens, Posix bytes read, host memory, TTFT, and throughput
with the same model and input set. These measurements answer whether storage
I/O is cheaper than recomputation for your workload.

[Implementation and extension](../extending-store.md#backend-entrypoints).
