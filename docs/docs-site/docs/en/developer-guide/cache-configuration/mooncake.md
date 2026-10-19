# Mooncake Store

Use the Mooncake pipeline to put reusable KV blocks in a shared memory service.
UCM connects to an existing Mooncake master and transfers data through its
client. Add Posix when KV blocks also need a filesystem backing tier.

| Pipeline | Read and write behavior |
| --- | --- |
| `Mooncake` | Reads and writes Mooncake; there is no UCM filesystem backing tier. |
| `Mooncake|Posix` | Reads Mooncake first and sends misses to Posix; dumps also write to Posix. |

This differs from [Cache|Posix](pipeline.md), whose first stage is a local host
buffer. A live Mooncake service can retain memory objects when an inference
process restarts, but that is not a guarantee of durable storage across service
failure. The persistence boundary of `Mooncake|Posix` is its configured
filesystem and completed backing writes.

## Prerequisites

The native UCM Mooncake target in this source tree requires Ascend ACL headers,
`libascendcl`, and `libmooncake_store`. Its CMake file skips the target when any
of these are missing. Treat this as an Ascend integration; a different
Mooncake transport name does not make this UCM target a CUDA backend.

Build UCM in the matching [Ascend environment](../build_from_source.md#vllm-ascend-ascend-platform)
after preparing Mooncake. CMake searches the toolkit's `include` and `lib64`
directories under `/usr/local/Ascend/ascend-toolkit/latest`, and the Mooncake
library under `/usr/local/lib`. `MOONCAKE_STORE_INCLUDE_DIR` selects the
Mooncake header tree; its current fallback is
`/vllm-workspace/Mooncake/mooncake-store/include`. Use the header revision that
matches the linked client library.

Before serving, confirm the installed `ucm/store/mooncakestore/libmooncakestore.so`
and its dependencies. Start a Mooncake master using your deployment's service
configuration, verify connectivity from every serving process, and prepare a
writable Posix directory if selecting the backing tier. UCM does not launch
the master for you.

## Start Mooncake master

On the service host with matching Mooncake binaries installed, run:

```bash
mooncake_master --port 50088 --eviction_high_watermark_ratio 0.9 --eviction_ratio 0.1 --default_kv_lease_ttl 60000
```

Keep it running and set `master_server_address` below to the reachable host IP and port `50088`. Reuse an existing master if available. Check the installed `mooncake_master --help` for capacity, lease and eviction settings; client and server versions must be compatible.

## Configuration for Prefix Caching

The following is an initial configuration for a host with an Ascend device and
a reachable Mooncake master. Replace both addresses and the storage directory:

```yaml
ucm_connectors:
  - ucm_connector_name: "UcmPipelineStore"
    ucm_connector_config:
      store_pipeline: "Mooncake|Posix"
      store_health:
        enabled: true
      local_hostname: "127.0.0.1"
      master_server_address: "127.0.0.1:50088"
      metadata_server: "P2PHANDSHAKE"
      protocol: "ascend"
      global_segment_size_gb: 30
      replica_num: 1
      storage_backends: "/mnt/ucm-mooncake"
      io_direct: true
      posix_io_engine: "aio"
      share_buffer_capacity_gb: 64

enable_event_sync: true
use_layerwise: false
```

`local_hostname` is required and must identify the serving host appropriately
for the transport. Loopback addresses only apply when all relevant services
share that host/network namespace. Keep `enable_event_sync: true`: the native
dump path waits on the prerequisite event before accessing newly computed KV.
Use the [vLLM-Ascend quickstart](../../user-guide/quick_start/index.md#vllm-ascend) to
supply this file through `UCM_CONFIG_FILE` and start the model server.

Both configurations use `UcmPipelineStore`, as registered by the V1 factory.

### Mooncake without filesystem backing

```yaml
ucm_connectors:
  - ucm_connector_name: "UcmPipelineStore"
    ucm_connector_config:
      store_pipeline: "Mooncake"
      store_health:
        enabled: true
      local_hostname: "127.0.0.1"
      master_server_address: "127.0.0.1:50088"
      metadata_server: "P2PHANDSHAKE"
      protocol: "ascend"
      global_segment_size_gb: 30
      replica_num: 1
      share_buffer_capacity_gb: 64

enable_event_sync: true
use_layerwise: false
```

## Budget memory and concurrency

The configuration retains the original 30 GiB global segment and 64 GiB shared buffer. These are distinct allocations. The local buffer is not overridden and defaults to 1 GiB. Positive `_gb` settings take precedence over the corresponding byte-valued settings.

`stream_number` defaults to 4 and must be between 1 and 32. The private host
buffer pool is derived from stream count and tensor size, unless
`host_buf_pool_size` is explicitly set. `replica_num` defaults to 1 and must be
positive. Replica count is not a persistence policy.

The vLLM adapter and Mooncake stage have different shared-buffer settings: `share_buffer_capacity_gb` belongs to Mooncake. When the adapter enables shared buffers, its preflight reads `cache_buffer_capacity_gb` and uses 128 GiB if it is omitted. This does not size the Mooncake buffer. Check shared memory against the effective startup configuration; the example does not override that preflight capacity.

## Original GLM deployment commands

The original GLM-4.7-W8A8 example uses two Ascend nodes, each with eight NPUs, TP8 and one local DP process (DP2 overall). Keep the original 32768-token context and other model-specific settings. Prepare the matching engine, model and driver environment; these scripts are not a compatibility claim for newer vLLM-Ascend releases. Save either UCM configuration above as `/etc/ucm/ucm.yaml` on both nodes, and replace `x.x.x.x` with the first node's reachable IP. Its `local_hostname` must identify each node and `master_server_address` must reach the master.

### Node 1

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:/vllm-workspace/vllm
export PYTHONHASHSEED=0
export ACL_OP_INIT_MODE=1
export HCCL_RDMA_TIMEOUT=17
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000
export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_MLAPO=1
export HCCL_INTRA_PCIE_ENABLE=0
export HCCL_INTRA_ROCE_ENABLE=1
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export VLLM_SERVER_DEV_MODE=1
export VLLM_USE_DEEP_GEMM=0
export VLLM_LOGGING_LEVEL=INFO
export ENABLE_UCM_PATCH=1
export ENABLE_SPARSE=FALSE
export VLLM_HASH_ATTENTION=0
export VLLM_CPU_AFFINITY=0
export UC_LOGGER_LEVEL=info

vllm serve /model/GLM-4.7-W8A8/ \
    --max-model-len 32768 \
    --tensor-parallel-size 8 \
    --data-parallel-size 2 \
    --data-parallel-size-local 1 \
    --data-parallel-start-rank 0 \
    --data-parallel-address x.x.x.x \
    --data-parallel-rpc-port 13389 \
    --pipeline-parallel-size 1 \
    --gpu-memory-utilization 0.88 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 7800 \
    --block-size 128 \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 20 \
    --seed 1024 \
    --quantization ascend \
    --served-model-name GLM-4.7-W8A8 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --enable-expert-parallel \
    --additional-config '{"ascend_scheduler_config":{"enabled":false}}' \
    --kv-transfer-config '{
        "kv_connector":"UCMConnector",
        "kv_connector_module_path":"ucm.integration.vllm.ucm_connector",
        "kv_role":"kv_both",
        "kv_connector_extra_config":{
            "UCM_CONFIG_FILE":"/etc/ucm/ucm.yaml"
        }
    }'
```

### Node 2

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:/vllm-workspace/vllm
export PYTHONHASHSEED=0
export ACL_OP_INIT_MODE=1
export HCCL_RDMA_TIMEOUT=17
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000
export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_MLAPO=1
export HCCL_INTRA_PCIE_ENABLE=0
export HCCL_INTRA_ROCE_ENABLE=1
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export VLLM_SERVER_DEV_MODE=1
export VLLM_USE_DEEP_GEMM=0
export VLLM_LOGGING_LEVEL=INFO
export ENABLE_UCM_PATCH=1
export ENABLE_SPARSE=FALSE
export VLLM_HASH_ATTENTION=0
export VLLM_CPU_AFFINITY=0
export UC_LOGGER_LEVEL=info

vllm serve /model/GLM-4.7-W8A8/ \
    --max-model-len 32768 \
    --tensor-parallel-size 8 \
    --data-parallel-size 2 \
    --data-parallel-size-local 1 \
    --data-parallel-start-rank 1 \
    --data-parallel-address x.x.x.x \
    --data-parallel-rpc-port 13389 \
    --pipeline-parallel-size 1 \
    --gpu-memory-utilization 0.88 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 7800 \
    --headless \
    --block-size 128 \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 20 \
    --seed 1024 \
    --quantization ascend \
    --served-model-name GLM-4.7-W8A8 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --enable-expert-parallel \
    --additional-config '{"ascend_scheduler_config":{"enabled":false}}' \
    --kv-transfer-config '{
        "kv_connector":"UCMConnector",
        "kv_connector_module_path":"ucm.integration.vllm.ucm_connector",
        "kv_role":"kv_both",
        "kv_connector_extra_config":{
            "UCM_CONFIG_FILE":"/etc/ucm/ucm.yaml"
        }
    }'
```

After readiness, run the original request workload from another terminal. These are workload inputs, not reproduced performance results:

```bash
vllm bench serve \
--backend vllm \
--model GLM-4.7-W8A8 \
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

## Verify each tier separately

1. Record the model revision, dtype, parallel layout, and Mooncake namespace.
   Run a fresh-prefix request and wait for dumps to complete.
2. Replay through a restarted inference process while leaving the Mooncake
   service running. Require external hits and Mooncake read/hit metrics.
3. For `Mooncake|Posix`, confirm backing files and completed Posix dumps. To
   exercise backing reads, evict only the test objects from Mooncake in an
   isolated test environment, retain the files, and replay again.
4. Require Posix reads and Mooncake backend-load metrics for that last case.
   A Mooncake hit alone does not verify the filesystem tier.

See [Metrics](../../user-guide/observability/metrics.md) for load-hit, load-miss, backend,
and byte counters, and [Health metrics](../../user-guide/observability/health-metrics.md)
for the active probes. Mooncake health checks perform a small write/read/remove
operation; they do not replace the two-tier request test.

## Diagnose the failing boundary

| Symptom | Next check |
| --- | --- |
| Native library missing or unresolved symbols | The optional CMake target, Ascend libraries, and matching Mooncake headers/library. |
| Setup or lookup cannot reach the service | Master address, local hostname, transport configuration, and the service logs. |
| Posix hits are expected but absent | Backing writes completed, matching storage paths, and Posix health. |
| Queue rejections or growing load latency | Queue and stage metrics, host buffer demand, service capacity, and network traffic. |

Keep performance comparisons specific to the tier hit. Report memory-service
hits and filesystem hits separately; no speedup is assumed by enabling
this pipeline.

[Implementation and extension](../extending-store.md#backend-entrypoints).
