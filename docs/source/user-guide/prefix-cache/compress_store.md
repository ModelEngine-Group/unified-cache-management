# UCM prefix-KVcache Compress User Guide

This document provides a complete usage example, configuration instructions, performance reference, and best practices for the UCM Storage Compression Module (Compress Store). Based on KVfold compression algorithm, this module implements compression/decompression capabilities for KV cache tensors, and can be seamlessly integrated into the existing UCM storage pipeline. By reducing disk I/O overhead, it significantly optimizes Time-To-First-Token (TTFT) in SSD cache hit scenarios.

　

## 1. Core Features

The Compress Store supports the following core capabilities:
- Full KV cache compression/decompression pipeline based on KVfold coding, with current version supporting 2.0x compression for BF16 tensors
- Multi-threaded parallel decompression capability, with flexibly configurable thread counts to adapt to different hardware environments and business scenarios
- Seamless integration with the existing UCM storage pipeline, one-click enable/disable of compression function via Pipeline configuration
- Full parametric YAML configuration support, flexible tuning of compression parameters to adapt to different deployment environments
- Compatible with both layer-wise and block-wise cache modes, adapting to different inference business scenarios

　

## 2. Configuration Guide
Modify the UCM configuration file to enable the compression module through Pipeline configuration and adjust relevant parameters. You can directly modify based on the sample configuration file: <br>
`unified-cache-management/examples/ucm_config_example.yaml`

### 2.1 Full Configuration Example
```
ucm_connectors:
  - ucm_connector_name: "UcmPipelineStore"
    ucm_connector_config:
      # Mandatory: Enable compression pipeline, fixed configuration as Cache|Compress|Posix
      store_pipeline: "Cache|Compress|Posix"
      # Storage path, supports local path or NFS mounted path
      storage_backends: "/home/lyj/temp/kv"
      # Compression ratio configuration: 16=2.0x compression, 32=1.0x no compression, other values not supported yet
      compress_ratio: 16
      # Data type configuration: 0=BF16, 100=INVALID, other values not supported yet
      data_type: 0
      # Number of threads for parallel decompression
      decompress_thread_num: 48
      # Whether to enable direct I/O
      io_direct: true
      # Cache buffer capacity in GB
      cache_buffer_capacity_gb: 64
      # POSIX IO engine
      posix_io_engine: "aio"

# Global configuration
use_layerwise: true
enable_record_traces: false
```

### 2.2 Mandatory Parameters

| Parameter Name | Configuration Description |
| :------------- | :------------------------ |
| `store_pipeline` | Fixed configuration as `Cache\|Compress\|Posix` to enable the compression pipeline. The compression function will not take effect without this configuration. |

### 2.3 Compression-Specific Optional Parameters
| Parameter Name | Default Value | Configuration Description and Notes |
| :------------- | :------------ | :----------------------------------- |
| `compress_ratio` | 16 | Compression ratio configuration.<br>16 = 2.0x lossless compression (recommended);<br>32 = 1.0x no compression;<br>Other values are not supported yet. |
| `data_type` | 0 | Tensor data type configuration.<br>0 = BF16 (the only supported type currently);<br>100 = INVALID;<br>Other values are not supported yet. |
| `decompress_thread_num` | 6 | Number of threads for parallel decompression.<br>Default value can be used when `use_layerwise: false`;<br>Recommended to increase to 40+ when `use_layerwise: true`, which needs to be tuned according to server hardware configuration through stress testing. |

　

## 3. Inference Service Startup Guide

This document deploys an OpenAI-compatible online inference service based on vLLM + UCM compression module, and the configuration method is fully compatible with the existing UCM process.

### 3.1 Startup Command Example
Take starting the Qwen/Qwen2.5-14B-Instruct model as an example, the complete startup command is as follows:
```
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
    "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/vllm-workspace/unified-cache-management/examples/ucm_config_example.yaml"}
}'
```
### 3.2 Notes
<div style="padding-left:0">
1. Please replace the UCM_CONFIG_FILE path with the actual configuration file path on your machine.<br>
2. The tensor parallelism --tensor-parallel-size must match the hardware environment.<br>
3. --max-model-len can be adjusted according to the context length supported by the model.
</div>

### 3.3 Startup Success Indicator
The following log indicates that the service is started successfully and the compression module is loaded and working properly:
```
[UC][I] Using UCM with config: {'ucm_connectors': [{'ucm_connector_name': 'UcmPipelineStore', 'ucm_connector_config': {'store_pipeline': 'Cache|Compress|Posix', 'storage_backends': './kv', 'compress_ratio': 16, 'data_type': 0, 'decompress_thread_num': 48, 'io_direct': True, 'cache_buffer_capacity_gb': 64, 'posix_io_engine': 'aio'}}], 'use_layerwise': True, 'enable_record_traces': False}
```

　

## 4. Compression Effect Verification

After the service is started, you can verify the performance benefits of the compression module through the built-in stress testing tool vllm bench of vLLM. To accurately verify the real performance benefits of the compression module in SSD cache hit scenarios, interference from DRAM memory cache must be excluded to ensure that the tested KV cache is only read from SSD. Strictly follow the test process below.

### 4.1 Stress Test Command Example
```
vllm bench serve \
--backend vllm \
--model Qwen/Qwen2.5-14B-Instruct \
--host 127.0.0.1 \
--port 7800 \
--dataset-name random \
--num-prompts 16 \
--random-input-len 32000 \
--random-output-len 2 \
--request-rate inf \
--seed 123456 \
--percentile-metrics "ttft,tpot,itl,e2el" \
--metric-percentiles "90,99" \
--ignore-eos
```
### 4.2 Result Verification
**1. First Execution (Cold Start No Cache, Baseline Test)**

First restart the vLLM service to clear all historical caches and ensure the service is in a pure cold start state. After executing the stress test command, no historical KV cache can be reused, and all prefix tokens need to be fully prefilled and calculated.

- The Mean TTFT output by the stress test is the baseline value without cache and compression optimization.
- The service log will show `hit hbm: 0, hit external: 0`, with no SSD cache hit.
- Observe the `COMPRESS DUMP` related compression dump logs; the KV cache generated by this request has been compressed and persisted to SSD.

**2. Before Second Execution: Restart Service to Clear DRAM Cache**

You must restart the vLLM service to completely clear the DRAM memory cache in the process, ensuring that subsequent requests cannot hit the cache from memory and can only read the compressed KV cache that has been dumped to SSD.

This step is the core control variable operation of the test, which avoids memory cache interference on test results and verifies the pure performance benefit brought by the compression module under SSD cache hit scenarios.

**3. Second Execution (SSD Cache Hit, Compression Effect Verification)**

After restarting the service, immediately execute the exact same stress test command as the first run.

- The Mean TTFT output by the stress test will be significantly reduced, representing the end-to-end latency optimization benefit brought by the compression module.
- The service log will show `hit hbm: 0, hit external: 125` (full SSD cache hit) with no memory cache hit.
- Observe the `COMPRESS LOAD` related decompression and loading logs to verify the full workflow of the compression module works properly.

### 4.3 Detailed Log Viewing
Set the environment variable UC_LOGGER_LEVEL=debug to print detailed logs of the entire compression/decompression process. The key log formats are as follows:

**Compression Dump Related Logs:**
```
[UC][D] COMPRESS DUMP START | task_id: xx
[UC][D] COMPRESS DUMP | shard: xx, done, compressed_size: xx
[UC][D] COMPRESS DUMP END | task_id: xx
```

**Decompression Load Related Logs:**

```
[UC][D] COMPRESS LOAD START | task_id: xx
[UC][D] COMPRESS LOAD | shard: xx, done, decompressed_size: xx
[UC][D] COMPRESS LOAD END | task_id: xx
```

　

## 5. Performance Test

Prefix-KVcache compression technique can reduce the IO pressure of SSD and optimize Time-To-First-Token (TTFT) in SSD cache hit scenarios. Here's a test result:

### 5.1 Test Environment

- Test Model: Qwen2.5-14B-Instruct
- Hardware Environment: Kunpeng 920 5250 + 4 × Ascend 910B4 (tensor parallel = 4)
- Test Configuration: DRAM cache disabled, tested under local SSD hit rates of 50% / 80% / 100%
- Test Modes: Layer-wise mode (use_layerwise: true), Block-wise mode (use_layerwise: false)


### 5.2 Test result

#### Layer_wise (use_layerwise: true) 

| input tokens | output tokens | batch size | odirect | TTFT/ms, 50% hit, w/o compression | TTFT/ms, 50% hit, with compression | TTFT reduction | TTFT/ms, 80% hit, w/o compression | TTFT/ms, 80% hit, with compression | TTFT reduction | TTFT/ms, 100% hit, w/o compression | TTFT/ms, 100% hit, with compression | TTFT reduction |
| ------------ | ------------- | ---------- | ------- | --------------------------------- | ---------------------------------- | -------------- | --------------------------------- | ---------------------------------- | -------------- | ---------------------------------- | ----------------------------------- | -------------- |
| 4000         | 1             | 1          | TRUE    | 357.99                            | 341.97                             | 4.47%          | 207.02                            | 225.74                             | -9.04%         | 208.37                             | 193.22                              | 7.27%          |
| 8000         | 1             | 1          | TRUE    | 531.6                             | 578.86                             | -8.89%         | 304.06                            | 321.29                             | -5.67%         | 350.01                             | 331.22                              | 5.37%          |
| 16000        | 1             | 1          | TRUE    | 1236.44                           | 1284.19                            | -3.86%         | 800.25                            | 720.4                              | 9.98%          | 611.74                             | 532.57                              | 12.94%         |
| 32000        | 1             | 1          | TRUE    | 3278.62                           | 3152.37                            | 3.85%          | 2077.46                           | 1807.59                            | 12.99%         | 1204.62                            | 959.57                              | 20.34%         |
| 4000         | 1             | 8          | TRUE    | 1339.06                           | 1396.03                            | -4.25%         | 868.46                            | 839.52                             | 3.33%          | 1056.99                            | 829.89                              | 21.49%         |
| 8000         | 1             | 8          | TRUE    | 2353.23                           | 2373.6                             | -0.87%         | 1436.33                           | 1356.39                            | 5.57%          | 2024.04                            | 1421.53                             | 29.77%         |
| 16000        | 1             | 8          | TRUE    | 5311.28                           | 5548.89                            | -4.47%         | 3433.49                           | 2829.16                            | 17.60%         | 3808.22                            | 2906.72                             | 23.67%         |
| 32000        | 1             | 8          | TRUE    | 13796.04                          | 13660.42                           | 0.98%          | 8353.08                           | 7851.27                            | 6.01%          | 6939.55                            | 5274.13                             | 24.00%         |
| 4000         | 1             | 16         | TRUE    | 2280.48                           | 2343.98                            | -2.78%         | 1428.27                           | 1232.16                            | 13.73%         | 2154.65                            | 1342.75                             | 37.68%         |
| 8000         | 1             | 16         | TRUE    | 4414.05                           | 4436.29                            | -0.50%         | 2639.71                           | 2388.07                            | 9.53%          | 4170.57                            | 2646                                | 36.56%         |
| 16000        | 1             | 16         | TRUE    | 10144.97                          | 9960.42                            | 1.82%          | 6167.56                           | 5084.08                            | 17.57%         | 7170.8                             | 4005.21                             | 44.15%         |
| 32000        | 1             | 16         | TRUE    | 25967.02                          | 24714.92                           | 4.82%          | 15550.23                          | 12950.47                           | 16.72%         | 12261.77                           | 7504.62                             | 38.80%         |

#### Block_wise (use_layerwise: false) 

| input tokens | output tokens | batch size | odirect | TTFT/ms, 50% hit, w/o compression | TTFT/ms, 50% hit, with compression | TTFT reduction | TTFT/ms, 80% hit, w/o compression | TTFT/ms, 80% hit, with compression | TTFT reduction | TTFT/ms, 100% hit, w/o compression | TTFT/ms, 100% hit, with compression | TTFT reduction |
| ------------ | ------------- | ---------- | ------- | --------------------------------- | ---------------------------------- | -------------- | --------------------------------- | ---------------------------------- | -------------- | ---------------------------------- | ----------------------------------- | -------------- |
| 4000         | 1             | 1          | TRUE    | 423.87                            | 406.85                             | 4.02%          | 307.49                            | 253.9                              | 17.43%         | 260.84                             | 198.13                              | 24.04%         |
| 8000         | 1             | 1          | TRUE    | 688.38                            | 665.83                             | 3.28%          | 507.52                            | 419.18                             | 17.41%         | 460.22                             | 352.85                              | 23.33%         |
| 16000        | 1             | 1          | TRUE    | 1528.11                           | 1479.79                            | 3.16%          | 1083.55                           | 874.61                             | 19.28%         | 617.35                             | 446.64                              | 27.65%         |
| 32000        | 1             | 1          | TRUE    | 3678.05                           | 3426.42                            | 6.84%          | 2327.42                           | 2039.48                            | 12.37%         | 1387.85                            | 832.33                              | 40.03%         |
| 4000         | 1             | 8          | TRUE    | 1801.28                           | 1673.41                            | 7.10%          | 1261.3                            | 1025.12                            | 18.73%         | 1117.65                            | 745.11                              | 33.33%         |
| 8000         | 1             | 8          | TRUE    | 3262.38                           | 2952.3                             | 9.50%          | 2344.01                           | 1880.82                            | 19.76%         | 2381.43                            | 1314.63                             | 44.80%         |
| 16000        | 1             | 8          | TRUE    | 6982.66                           | 6343.26                            | 9.16%          | 4897.66                           | 3862.16                            | 21.14%         | 4157.12                            | 2060.17                             | 50.44%         |
| 32000        | 1             | 8          | TRUE    | 16615.58                          | 14960.37                           | 9.96%          | 10797.82                          | 8443.65                            | 21.80%         | 7872.14                            | 3896.58                             | 50.50%         |
| 4000         | 1             | 16         | TRUE    | 3181.24                           | 2928.33                            | 7.95%          | 2228.06                           | 1775.39                            | 20.32%         | 2402.1                             | 1152.3                              | 52.03%         |
| 8000         | 1             | 16         | TRUE    | 6082.47                           | 5437.54                            | 10.60%         | 4213.28                           | 3197.17                            | 24.12%         | 4381.51                            | 2386.19                             | 45.54%         |
| 16000        | 1             | 16         | TRUE    | 12832.2                           | 11805.53                           | 8.00%          | 8489.46                           | 6721.34                            | 20.83%         | 6231.51                            | 3555.04                             | 42.95%         |
| 32000        | 1             | 16         | TRUE    | 29728.23                          | 28228.75                           | 5.04%          | 18745.21                          | 15293.23                           | 18.42%         | 13077.52                           | 6831.96                             | 47.76%         |

## 6. Scope of Application

This section specifies the mandatory prerequisites, recommended scenarios, and not recommended scenarios for the compression function, to help you determine whether to enable it and how to configure it for optimal benefits.

### 6.1 Mandatory Prerequisites

The compression function can only be enabled and work stably when all of the following conditions are met:

- Software Stack Constraint: Must be used based on the UCM unified cache management framework integrated with the vLLM inference engine; the mandatory parameter store_pipeline: "Cache|Compress|Posix" must be set in the configuration to enable the compression pipeline. The compression function will not take effect without this configuration.
- Data Type Constraint: The current version only supports KV cache tensors in BF16 format (corresponding to configuration data_type: 0). Other data types such as FP16, FP8, INT8, and INT4 are not supported yet.
- Compression Ratio Constraint: The current version only supports 2.0x lossless compression (corresponding to configuration compress_ratio: 16). compress_ratio: 32 is 1.0x no compression mode, other values are not supported yet.
- Storage Backend Constraint: Must be used with Posix-compatible local file system, SSD, or NFS-mounted storage, compatible with UCM's Posix IO engine (e.g., aio). Cannot be used alone for pure HBM cache scenarios.
- Hardware Compatibility: Verified and adapted for Kunpeng 920 architecture CPU + Ascend 910B4 NPU deployment environment; also compatible with x86 architecture CPU + NVIDIA GPU general inference environment, requiring sufficient CPU cores to support multi-threaded parallel decompression.

　

### 6.2 Recommended Scenarios

The core scenarios where significant performance benefits can be obtained after enabling the compression function are as follows:

- Ultra-long Context Inference Scenarios
  Inference scenarios with input token length ≥16K, especially 32K ultra-long context. Disk I/O is the core bottleneck of TTFT in this scenario. 2.0x compression can directly reduce I/O data volume by 50%, bringing up to 40%+ TTFT reduction in full SSD hit scenarios, and up to 50%+ optimization benefits in block-wise mode.
- High-concurrency Online Service Scenarios
  Online inference services with concurrency ≥8, especially high-load scenarios with concurrency ≥16. I/O requests are intensive under high concurrency, compression can significantly reduce disk bandwidth pressure. The maximum TTFT reduction is 38.8% in 16-concurrency, 32K context scenario in layer-wise mode, and up to 47.76% in block-wise mode.
- High SSD Cache Hit Rate Scenarios
  Business scenarios with prefix cache hit rate ≥80% (especially 100% full hit), which is the core benefit scenario of the compression feature. For example, RAG scenarios, multi-turn dialogue scenarios, fixed template inference scenarios, etc. The frequently reused prefix KV cache can greatly reduce loading time after compression, which is directly converted into TTFT optimization.
- Storage Resource Constrained Scenarios
  Deployment environments where GPU/NPU HBM is tight and a large amount of KV cache needs to be offloaded to disk. 2.0x compression can directly reduce disk storage space occupation by 50%, while reducing I/O overhead, supporting longer context windows or higher concurrency with the same storage resources.
- Layer-wise Cache Long Sequence Scenarios
  Scenarios where layer-wise cache mode (use_layerwise: true) is enabled and input length ≥16K. With decompress_thread_num adjusted to 40+, the parallel advantage of multi-threaded decompression can be fully utilized, significantly alleviating the I/O bottleneck of layer-by-layer loading, and obtaining stable TTFT optimization benefits.

### 6.3 Not Recommended Scenarios

Enabling the compression function in the following scenarios has no benefit, or even negative performance optimization, so it is not recommended to turn it on:

- Short Context and Low Concurrency Scenarios
  Light-load scenarios with input token length ≤8K and concurrency ≤2. The proportion of I/O overhead is low in this scenario, and the CPU overhead of decompression may exceed the benefits brought by I/O reduction. Some cases in this scenario show negative TTFT optimization in actual measurement.
- Low SSD Cache Hit Rate Scenarios
  Scenarios with SSD cache hit rate <50%. Most of the KV cache needs to be recalculated and generated, the compression feature cannot play a role, and the additional compression calculation overhead will bring meaningless performance loss.
- CPU Computing Power/Core Constrained Environment
  Deployment environment with insufficient CPU cores or tight computing power. Multi-threaded decompression will seize the CPU resources of the inference service, leading to increased inference scheduling delay, and thus end-to-end performance degradation.
- Memory Bandwidth Constrained Environment
  Deployment environment with insufficient server memory bandwidth. Memory copy operations during compression/decompression will exacerbate the memory bandwidth bottleneck, failing to achieve expected performance benefits, and even performance regression.
- Non-BF16 Precision Inference Scenarios
  Model inference scenarios using other precisions such as FP16, INT8, FP8, etc. The current version does not support compression of non-BF16 tensors, and the feature cannot be enabled.
- Pure HBM Cache Scenarios
  Scenarios where KV cache is completely stored in GPU/NPU HBM/DRAM without disk offload requirements. The compression feature is designed for disk I/O scenarios, enabling it in pure HBM scenarios has no benefit, but will increase unnecessary computing overhead.