# Compressed KV storage

Use `Cache|Compress|Posix` to evaluate smaller filesystem KV payloads when
storage I/O is a bottleneck and your application can tolerate changes to cached
values. Cache Store transfers device tensors to host buffers, Compress encodes
the host data, and Posix stores it. Loads reverse that path.

The active compression mode is **lossy BF16 encoding**. Its codec clamps the
exponent range and retains fewer mantissa bits; it may fall back to an FP8-like
encoding to meet the size budget. It does not promise bit-identical KV recovery
or unchanged model quality. It also does not change the engine's allocated
device KV-cache dtype or reduce the model's device-memory allocation.

## Check whether the codec fits your workload

The runtime currently accepts only `data_type: 0` (BF16). This parameter tells
the codec how to interpret bytes; it does not convert the engine's actual KV
values. Confirm the effective engine KV dtype before enabling the pipeline.
FP16, FP8, quantized, and mixed-type KV layouts cannot be made compatible by
setting this field to zero.

Only two `compress_ratio` values are accepted:

| Value | Behavior |
| --- | --- |
| `16` | Stores the encoded payload at half the original shard size before filesystem alignment. |
| `32` | Passes data through the Compress stage without compression. |

Other ratios listed in internal enum definitions are not accepted by the
active stage. Use [Cache|Posix](pipeline.md) for the uncompressed baseline.

## Prepare a separate storage namespace

Start from [Installation](../../user-guide/quick_start/index.md), or
[build from source](../build_from_source.md) for the target
engine environment. The registered pipeline loads `libcachestore.so`,
`libcompressor.so`, and `libposixstore.so` through `ucmpipelinestore`.
The compressor target is included in the Store CMake build.

Use a new directory for compressed data. Do not point compressed and
uncompressed deployments at the same files, or reuse a directory after changing
codec settings or tensor geometry. The pipeline derives stored block sizes from
the ratio, so identical filenames do not imply compatible payload layouts.

## Configure a BF16 trial

```yaml
ucm_connectors:
  - ucm_connector_name: "UcmPipelineStore"
    ucm_connector_config:
      store_pipeline: "Cache|Compress|Posix"
      storage_backends: "/mnt/ucm-compressed-bf16"
      compress_ratio: 16
      data_type: 0
      decompress_thread_num: 48
      io_direct: true
      cache_buffer_capacity_gb: 64
      posix_io_engine: "aio"

use_layerwise: true
enable_record_traces: false
```

The example retains a 64 GiB cache, 48 decompression threads, and direct AIO. These are explicit settings from the original example, not runtime defaults.

Supply this file using the [vLLM quickstart](../../user-guide/quick_start/index.md#vllm)
or [vLLM-Ascend quickstart](../../user-guide/quick_start/index.md#vllm-ascend). Verify
the engine's BF16 KV dtype in addition to these Store settings.
`data_type` defaults to an invalid sentinel, so it must be explicit.
`decompress_thread_num` defaults to 6; `stream_number` defaults to 8 and the
compressor uses half that count for its dump workers. Cache transfer streams
are configured separately with `cache_stream_number`.

Before the first write, inspect the effective shard size. The builder computes
stored shard bytes as:

```text
floor((shard_size * compress_ratio / 32) / 4096) * 4096
```

For ratio 16, require a nonzero half-shard size that is already a multiple of
4096; otherwise the builder rounds the storage size down. This is a data-layout
constraint even with buffered I/O. The current connector does not establish
that constraint for every possible model layout, so a successful server startup
alone is insufficient. The host buffer must also satisfy
[Pipeline memory sizing](pipeline.md#budget-host-memory).

## Original serving and request commands

Save the YAML above as `/etc/ucm/ucm.yaml` in the matching engine environment, then use the original Qwen2.5-14B-Instruct serving parameters below. `--no-enable-prefix-caching` is retained for the original storage test; remove it for normal deployments that also use native prefix caching.

```bash
export ENABLE_UCM_PATCH=1
vllm serve Qwen/Qwen2.5-14B-Instruct \
--max-model-len 32000 \
--tensor-parallel-size 4 \
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

Use the [external-cache verification](../../user-guide/observability/verify-cache.md) with this server. Input plus generated tokens must fit the 32000-token context; the original benchmark's 32000 input plus 2 output tokens exceeds that limit and is not included as a runnable command.

## Verify storage and answer quality

Use a fixed representative prompt set and a separate baseline directory.
Validate three things independently:

1. **Write and read:** complete a first request, retain the compressed files,
   restart the serving process, and replay identical inputs. Require external
   hits and Posix reads so that the codec's load path is exercised.
2. **Stored size:** compare completed cache-file sizes for the same full blocks
   against `Cache|Posix`. Exclude temporary files and filesystem allocation
   overhead when checking the encoded payload ratio.
3. **Model quality:** compare application-relevant outputs and quality metrics
   with the uncompressed run under the same decoding settings. Choose acceptable
   quality limits before evaluating speed.

The same-process replay can be served from uncompressed host/device memory,
which does not test decompression. Follow the
[external-cache procedure](../../user-guide/quick_start/index.md#vllm-verify-the-service-and-external-cache)
and report cold-read TTFT, CPU consumption, storage bytes, and quality together.
Halving a payload does not imply halving end-to-end latency.

## Diagnose a failed trial

| Observation | Next check |
| --- | --- |
| Invalid compression dtype or ratio | Actual BF16 KV dtype and the accepted values above. |
| Native library load failure | That all three stages were built for the target environment. |
| Unexpected file size or corrupt replay | Shard alignment and an empty namespace dedicated to this codec/layout. |
| `COMPRESS DUMP FAILED` or `COMPRESS LOAD FAILED` | The complete codec/backend error logs; do not count that run as a successful cache test. |
| Latency increases despite smaller files | Compression/decompression CPU work versus saved storage transfer time. |
| Quality regression | Compare against uncompressed KV and evaluate whether the application's tolerance permits this codec. |

Current codec error paths include logging and continuing at shard level. Treat
an error-free request and verified outputs as part of acceptance; task completion
by itself is not proof that every shard decoded correctly. Resolve failures
before using the compressed directory for production traffic.

[Implementation and extension](../extending-store.md#backend-entrypoints).
