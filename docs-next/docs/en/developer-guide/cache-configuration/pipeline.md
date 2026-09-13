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

## Other registered pipelines

| Requirement | Selection |
| --- | --- |
| Existing vLLM NFS connector configuration | [NFS Store](nfs.md) |
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
