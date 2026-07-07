# Mooncake Store

For Mooncake introduction and **deployment**, please refer to [the official documentation](https://github.com/kvcache-ai/Mooncake).

**Mooncake Store** integrates [Mooncake] — a high-performance distributed KV cache transfer engine — into UCM's storage pipeline. It uses RDMA (or Ascend HCCL) to move KV cache blocks directly between device memory and a distributed shared memory pool across nodes, delivering significantly lower latency than disk-based backends.

Two pipeline modes are supported:

- **`Mooncake`** (standalone): KV blocks are written to / read from the Mooncake distributed memory pool only. No persistent fallback.
- **`Mooncake|Posix`** (chained): Mooncake acts as the high-speed DRAM tier; Posix provides a persistent SSD fallback. This is the recommended configuration for production use.

## Performance

### Overview

The following are the multi-concurrency performance test results of UCM in the Prefix Cache scenario, showing the performance improvements when using Mooncake Store as the KV cache backend.

The following table shows the results on the **GLM4.7-W8A8** model:

| **Input** | **Output** | **Hit Rate** | **Concurrent** | **Requests** | **Full Compute TTFT (s)** | **Full Compute TPOT (s)** | **Full Compute E2E (s)** | **UCM+Mooncake TTFT (s)** | **UCM+Mooncake TPOT (s)** | **UCM+Mooncake E2E (s)** | **TTFT Speedup** |
| --------: | ---------: | -----------: | -------------: | -----------: | ------------------------: | ------------------------: | -----------------------: | ------------------------: | ------------------------: | -----------------------: | :--------------- |
|     2 000 |        512 |          50% |             64 |          200 |                     4.164 |                     0.076 |                   42.808 |                     3.342 |                     0.074 |                   41.311 | **+20%**         |
|     4 000 |        512 |          50% |             64 |          100 |                     9.828 |                     0.091 |                   56.322 |                     7.089 |                     0.084 |                   50.193 | **+28%**         |
|     8 000 |        512 |          50% |             32 |           50 |                    11.518 |                     0.090 |                   57.568 |                     8.532 |                     0.081 |                   48.910 | **+26%**         |
|    16 000 |        512 |          50% |             16 |           25 |                    12.889 |                     0.087 |                   57.434 |                     8.415 |                     0.081 |                   49.820 | **+35%**         |
|    32 000 |        512 |          50% |              8 |           13 |                    16.705 |                     0.100 |                   67.686 |                    11.481 |                     0.086 |                   55.271 | **+31%**         |
|    64 000 |        512 |          50% |              4 |            7 |                    27.141 |                     0.114 |                   85.421 |                    18.623 |                     0.101 |                   70.430 | **+31%**         |
|   128 000 |        512 |          50% |              2 |            4 |                    56.878 |                     0.120 |                  118.100 |                    39.876 |                     0.123 |                  102.517 | **+30%**         |
|     2 000 |        512 |          90% |             64 |          200 |                     4.164 |                     0.076 |                   42.808 |                     2.353 |                     0.065 |                   35.647 | **+43%**         |
|     4 000 |        512 |          90% |             64 |          100 |                     9.828 |                     0.091 |                   56.322 |                     3.240 |                     0.068 |                   37.843 | **+67%**         |
|     8 000 |        512 |          90% |             32 |           50 |                    11.518 |                     0.090 |                   57.568 |                     3.289 |                     0.062 |                   35.149 | **+71%**         |
|    16 000 |        512 |          90% |             16 |           25 |                    12.889 |                     0.087 |                   57.434 |                     3.623 |                     0.060 |                   34.167 | **+72%**         |
|    32 000 |        512 |          90% |              8 |           13 |                    16.705 |                     0.100 |                   67.686 |                     4.421 |                     0.068 |                   38.356 | **+74%**         |
|    64 000 |        512 |          90% |              4 |            7 |                    27.141 |                     0.114 |                   85.421 |                     6.548 |                     0.088 |                   51.265 | **+76%**         |
|   128 000 |        512 |          90% |              2 |            4 |                    56.878 |                     0.120 |                  118.100 |                    13.528 |                     0.121 |                   75.386 | **+76%**         |

## Features

The Mooncake connector supports the following functionalities:

- `dump`: Offload KV cache blocks from device (HBM/DRAM) to the Mooncake distributed memory pool.
- `load`: Retrieve KV cache blocks from the Mooncake distributed memory pool back to device memory.
- `lookup`: Check whether KV blocks are present in the Mooncake store by block hash.
- `wait`: Block until all in-flight dump or load operations have completed.

## Prerequisites

Before starting, ensure the Mooncake master service is running:

```bash
mooncake_master --port 50088 --eviction_high_watermark_ratio 0.9 --eviction_ratio 0.1 --default_kv_lease_ttl 60000
```

| Flag | Description |
|------|-------------|
| `--port` | Port on which the master service listens |
| `--eviction_high_watermark_ratio` | Cache pool fill ratio that triggers GC (e.g. `0.9` = 90%) |
| `--eviction_ratio` | Fraction of the cache to evict per GC cycle (e.g. `0.1` = 10%) |
| `--default_kv_lease_ttl` | Lease TTL for KV entries in milliseconds |

## Configuration for Prefix Caching

Modify the UCM configuration file to enable the Mooncake connector.
You may directly edit the example file at:

`unified-cache-management/examples/ucm_mooncake_config.yaml`

### Recommended Configuration (`Mooncake|Posix` pipeline)

```yaml
ucm_connectors:
  - ucm_connector_name: "UcmPipelineStore"
    ucm_connector_config:
      store_pipeline: "Mooncake|Posix"
      local_hostname: "127.0.0.1"
      master_server_address: "127.0.0.1:50088"
      metadata_server: "P2PHANDSHAKE"
      protocol: "ascend"
      global_segment_size_gb: 30
      replica_num: 1
      storage_backends: "/mnt/test"
      io_direct: true
      posix_io_engine: "aio"
      share_buffer_capacity_gb: 64

enable_event_sync: true
use_layerwise: false
```

### Standalone Configuration (`Mooncake` only)

```yaml
ucm_connectors:
  - ucm_connector_name: "UcmPipelineStore"
    ucm_connector_config:
      store_pipeline: "Mooncake"
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

### Required Parameters

* **`ucm_connector_name`**:  
  Must be set to `"UcmPipelineStore"`.

* **`store_pipeline`**:  
  Use `"Mooncake|Posix"` (recommended) or `"Mooncake"` (standalone).

* **`local_hostname`** *(required)*:  
  The IP address of the local node. Must not include a port number.  
  **⚠️ Replace `"127.0.0.1"` with the actual IP of this node.**

* **`master_server_address`** *(required)*:  
  Address of the Mooncake master service in `host:port` format.  
  Default: `"127.0.0.1:50088"`.

* **`protocol`** *(required)*:  
  Transport protocol used by Mooncake. Use `"ascend"` for Ascend NPU environments, `"tcp"` for CPU-only testing, or `"rdma"` for InfiniBand/RoCE environments.

* **`storage_backends`** *(required for `Mooncake|Posix`)*:  
  Posix storage path for the persistent SSD tier.  
  **⚠️ Replace `"/mnt/test"` with your actual storage path.**

### Optional Parameters

* **`metadata_server`** *(optional, default: `"P2PHANDSHAKE"`)*:  
  Metadata server address. Use `"P2PHANDSHAKE"` for peer-to-peer handshake mode (no separate metadata server needed).

* **`global_segment_size_gb`** *(optional)*:  
  Size of the global distributed memory segment in GiB.  
  Alternatively, set `global_segment_size` in bytes. Default: ~30 GiB.

* **`local_buffer_size_gb`** *(optional)*:  
  Size of the local transfer buffer in GiB.  
  Alternatively, set `local_buffer_size` in bytes. Default: 1 GiB.

* **`replica_num`** *(optional, default: `1`)*:  
  Number of replicas for each KV block in the Mooncake pool. Must be ≥ 1.

* **`device_name`** *(optional, default: `""`)*:  
  Name of the RDMA or Ascend device to use (e.g., `"mlx5_0"` for RDMA). Leave empty to use the default device.

* **`stream_number`** *(optional, default: `4`)*:  
  Number of concurrent transfer streams.

* **`io_direct`** *(optional, default: `false`)*:  
  Whether to enable direct I/O for the Posix tier (only applicable in `Mooncake|Posix` pipeline).

* **`posix_io_engine`** *(optional, default: `"sync"`)*:  
  I/O engine for the Posix tier. Options: `"sync"`, `"aio"`. Recommended: `"aio"`.

* **`timeout_ms`** *(optional, default: `0` — no timeout)*:  
  Timeout in milliseconds for transfer operations. `0` means wait indefinitely.

* **`enable_event_sync`** *(optional, default: `false`, top-level config key)*:  
  When set to `true`, UCM inserts a device event synchronization barrier before each dump, ensuring NPU/GPU compute finishes before Mooncake reads device memory. **Required for correct behavior in Ascend NPU environments.**

## Launching Inference

This section uses **GLM4.7-W8A8 with DP2 TP8 (two-node)** as an example, providing the complete launch commands.

**⚠️ Replace `141.111.32.88` (master node IP), `141.111.32.82` (worker node IP), the model path, and the UCM config file path with your actual values.**

On the master node:

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
            "UCM_CONFIG_FILE":"/unified-cache-management/examples/ucm_mooncake_config.yaml"
        }
    }'
```

On the worker node:

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
            "UCM_CONFIG_FILE":"/unified-cache-management/examples/ucm_mooncake_config.yaml"
        }
    }'
```

If you see the following log on the master node, the vLLM server has started successfully with the UCM Mooncake connector:

```bash
INFO:     Started server process [1049932]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

## Evaluating UCM Prefix Caching Performance

After launching the vLLM server with `UCMConnector` enabled, the easiest way to observe the prefix caching effect is to run the built-in `vllm bench` CLI. Execute the following command **twice** in a separate terminal on the master node to clearly see the improvement.

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

### After the first execution

The `vllm bench` terminal prints the benchmark result:

```
---------------Time to First Token----------------
Mean TTFT (ms):                           XXXX.XX
```

Inspecting the vLLM server logs reveals entries like:

```
INFO ucm_connector.py:228: request_id: xxx, total_blocks_num: 125, hit hbm: 0, hit external: 0
```

This indicates that for the first inference request, UCM did not hit any cached KV blocks. As a result, the full 16K-token prefill must be computed, leading to a relatively large TTFT.

### After the second execution

Running the same benchmark again produces a significantly reduced TTFT.

The vLLM server logs now contain similar entries:

```
INFO ucm_connector.py:228: request_id: xxx, total_blocks_num: 125, hit hbm: 0, hit external: 125
```

This indicates that during the second request, UCM successfully retrieved all 125 cached KV blocks from the Mooncake distributed memory pool, yielding a substantial improvement in TTFT.

### Log Message Structure

> If you want to view detailed transfer information, set the environment variable `UC_LOGGER_LEVEL` to `debug`.

**Dump path logs:**

```text
[UC][D] Mooncake task(<task_id>) start running, wait <wait_ms>ms.
[UC][D] Mooncake task(<task_id>) exists check: <missing>/<total> keys missing.
[UC][D] Mooncake task(<task_id>) dump done, cost=<cost_ms>ms.
```

| Component   | Description                                                                 |
|-------------|-----------------------------------------------------------------------------|
| `task_id`   | Unique identifier for the task                                              |
| `wait_ms`   | Time the task spent waiting in the queue before starting (milliseconds)     |
| `missing`   | Number of KV blocks not yet present in the Mooncake pool (will be written)  |
| `total`     | Total number of KV blocks in this dump task                                 |
| `cost_ms`   | Time taken to complete the dump operation (milliseconds)                    |

**Load path logs:**

```text
[UC][D] Mooncake task(<task_id>) all hit(<shard_count>), wait=<wait_ms>ms, cost=<cost_ms>ms.
```
Indicates all KV blocks were found in the Mooncake pool — no fallback to the Posix backend was needed.

```text
[UC][D] Mooncake task(<task_id>) dispatch shards(<total>, miss=<miss_count>), wait=<wait_ms>ms, cost=<cost_ms>ms.
```
Indicates a partial miss: `miss_count` shards were not found in the Mooncake pool and have been dispatched to the Posix backend for retrieval.

| Component    | Description                                                                      |
|--------------|----------------------------------------------------------------------------------|
| `task_id`    | Unique identifier for the task                                                   |
| `shard_count`| Total number of KV block shards in the load task                                 |
| `total`      | Total number of KV block shards in the load task                                 |
| `miss_count` | Number of shards not found in Mooncake, falling back to the Posix backend        |
| `wait_ms`    | Time the task spent waiting in the queue before dispatch (milliseconds)          |
| `cost_ms`    | Time taken for the Mooncake lookup and dispatch phase (milliseconds)             |
