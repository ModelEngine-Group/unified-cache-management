# UCM Prefix KV Cache Compress Store User Guide

This document provides a usage example, configuration instructions, performance reference, and deployment guidance for the UCM Compress Store. Based on KVfold coding, Compress Store compresses BF16 KV cache tensors before they are written to a Posix-compatible backend and decompresses them when they are loaded. Reducing the amount of storage I/O can improve Time To First Token (TTFT) when external-cache loading is on the critical path.

　

## 1. Core Features

The Compress Store supports the following core capabilities:
- Two fixed-budget BF16 codec paths: R160 (`compress_ratio: 20`, nominal 1.60x) and R200 (`compress_ratio: 16`, 2.00x)
- R160 prioritizes precision and uses a shard-level high-precision path with a quantized fallback when the fixed payload budget cannot be met; R200 prioritizes a smaller payload
- Multi-threaded parallel decompression capability, with flexibly configurable thread counts to adapt to different hardware environments and business scenarios
- Seamless integration with the existing UCM storage pipeline, one-click enable/disable of compression function via Pipeline configuration
- Full parametric YAML configuration support, flexible tuning of compression parameters to adapt to different deployment environments
- Compatible with both layer-wise and block-wise cache modes, adapting to different inference business scenarios

Both R160 and R200 are lossy codecs. They restore BF16-formatted data but do not guarantee bit-exact equality with the input. Validate model accuracy on representative workloads before enabling either codec in production.

　

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
      storage_backends: "/mnt/kv"
      # BF16 codec: 20=R160 (nominal 1.60x), 16=R200 (2.00x), 32=no compression
      compress_ratio: 20
      # Data type configuration: 0=BF16, 100=INVALID, other values not supported yet
      data_type: 0
      # Number of threads for parallel decompression
      decompress_thread_num: 24
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
| Parameter Name | Supported/Recommended Values | Configuration Description and Notes |
| :------------- | :--------------------------- | :----------------------------------- |
| `compress_ratio` | 20 / 16 / 32 | BF16 codec selection.<br>20 = R160, nominal 1.60x, higher-precision option;<br>16 = R200, 2.00x, higher-compression option;<br>32 = no compression;<br>Other values are not supported. |
| `data_type` | 0 | Tensor data type configuration.<br>0 = BF16 (the only supported type currently);<br>100 = INVALID;<br>Other values are not supported yet. |
| `decompress_thread_num` | 24 / 36 / 48 | Number of parallel decompression workers.<br>R160: 24 workers are recommended as the starting point.<br>R200: 48 workers are recommended as the starting point.<br>36 workers can be used as an intermediate tuning point. The optimum still depends on shard size, CPU topology, storage bandwidth, and request concurrency. |

For Posix direct I/O, the pipeline rounds the nominal compressed shard size down to a 4 KiB boundary. Therefore, the effective R160 ratio can be slightly higher than 1.60x for shard sizes whose nominal 5/8 payload is not already 4 KiB aligned.

　

## 3. Inference Service Startup Guide

This document deploys an OpenAI-compatible online inference service based on vLLM + UCM compression module, and the configuration method is fully compatible with the existing UCM process.

### 3.1 Startup Command Example
Take starting the Qwen/Qwen3-32B model as an example, the complete startup command is as follows:
```
vllm serve Qwen/Qwen3-32B \
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
[UC][I] Using UCM with config: {'ucm_connectors': [{'ucm_connector_name': 'UcmPipelineStore', 'ucm_connector_config': {'store_pipeline': 'Cache|Compress|Posix', 'storage_backends': './kv', 'compress_ratio': 20, 'data_type': 0, 'decompress_thread_num': 24, 'io_direct': True, 'cache_buffer_capacity_gb': 64, 'posix_io_engine': 'aio'}}], 'use_layerwise': True, 'enable_record_traces': False}
```

　

## 4. Compression Effect Verification

Use `vllm bench serve` to compare Compress Store with an uncompressed baseline. A cold-start run and an external-cache-hit run measure different amounts of Prefill work and cannot, by themselves, isolate compression benefit. Compare the two pipelines at the same external-cache hit rate, request set, and service configuration.

### 4.1 Stress Test Command Example
```
vllm bench serve \
--backend vllm \
--model Qwen/Qwen3-32B \
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
**1. Measure the uncompressed baseline**

Configure `store_pipeline: "Cache|Posix"` and use a dedicated storage directory. Run the workload once to populate external cache, restart the service to clear process-local DRAM cache while preserving the storage directory, and then run the same workload again for measurement.

- Confirm that the service log reports no HBM/DRAM cache hit and the expected number of external-cache hits.
- Record TTFT and the effective hit rate from the measured run.

**2. Measure R160 or R200**

Configure `store_pipeline: "Cache|Compress|Posix"`, select `compress_ratio: 20` for R160 or `compress_ratio: 16` for R200, and use a separate empty storage directory. Repeat the same populate, restart, and measured-run sequence.

- Keep the model, input data, request order, concurrency, I/O engine, direct-I/O setting, CPU affinity, and all non-compression settings unchanged.
- Confirm that the measured baseline and compressed runs have the same external-cache hit rate.
- Observe `COMPRESS DUMP` during population and `COMPRESS LOAD` during the measured run.

**3. Calculate the benefit**

Calculate TTFT reduction using the same statistic from both measured runs:

```text
TTFT reduction = (baseline TTFT - compressed TTFT) / baseline TTFT × 100%
```

A positive value indicates lower TTFT with compression. Repeat each case enough times to report stable median and tail statistics; compression may regress workloads where decompression overhead exceeds the saved I/O time.

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

The following results compare `Cache|Posix` with R160 or R200 `Cache|Compress|Posix`. Positive reduction means that compression has a lower TTFT. The R160 and R200 results use different models and hardware environments, so each codec must be compared only with its own baseline; the two result sets are not a direct codec-to-codec comparison. These measurements are workload-specific and should not be treated as a guarantee for other models, storage devices, or CPU configurations.

### 5.1 R160 Test Environment

- Model: Qwen3-32B, tensor parallel size 4
- Hardware: Kunpeng 920 7280Z + 4 × Ascend 910
- Codec: R160 (`compress_ratio: 20`)
- Mode: Layer-wise (`use_layerwise: true`)
- Decompression workers: 24
- I/O: AIO with `io_direct: true`
- Output length: 1 token
- External-cache hit rates: 50%, 80%, and 100%

### 5.2 R160 Layer-wise Results

| Input tokens | Concurrency | 50% hit baseline (ms) | 50% hit R160 (ms) | Reduction | 80% hit baseline (ms) | 80% hit R160 (ms) | Reduction | 100% hit baseline (ms) | 100% hit R160 (ms) | Reduction |
| -----------: | ----------: | --------------------: | -----------------: | --------: | --------------------: | -----------------: | --------: | ---------------------: | ------------------: | --------: |
| 4,000  | 16 | 1,875.32  | 1,878.24  | -0.16% | 1,197.64  | 1,007.30  | 15.89% | 1,547.42  | 1,041.17 | 32.72% |
| 8,000  | 16 | 3,668.65  | 3,665.55  | 0.08%  | 2,135.45  | 1,836.74  | 13.99% | 2,995.06  | 1,995.93 | 33.36% |
| 16,000 | 16 | 8,766.18  | 8,718.02  | 0.55%  | 4,089.70  | 3,860.12  | 5.61%  | 5,024.22  | 3,797.71 | 24.41% |
| 32,000 | 16 | 23,164.46 | 23,026.22 | 0.60%  | 10,711.75 | 10,445.64 | 2.48%  | 12,793.07 | 5,895.86 | 53.91% |
| 4,000  | 8  | 1,100.31  | 1,092.09  | 0.75%  | 575.48    | 506.30    | 12.02% | 683.50    | 553.64   | 19.00% |
| 8,000  | 8  | 1,951.69  | 1,953.97  | -0.12% | 1,199.81  | 1,127.93  | 5.99%  | 1,247.67  | 1,120.91 | 10.16% |
| 16,000 | 8  | 4,623.99  | 4,636.65  | -0.27% | 2,586.86  | 2,192.05  | 15.26% | 2,335.95  | 1,529.98 | 34.50% |
| 32,000 | 8  | 12,378.65 | 12,287.68 | 0.73%  | 5,861.73  | 5,691.73  | 2.90%  | 5,759.41  | 3,040.24 | 47.21% |
| 4,000  | 1  | 251.33    | 247.71    | 1.44%  | 142.78    | 134.00    | 6.15%  | 141.95    | 120.48   | 15.13% |
| 8,000  | 1  | 468.39    | 467.32    | 0.23%  | 244.48    | 230.73    | 5.62%  | 230.92    | 202.88   | 12.14% |
| 16,000 | 1  | 1,081.37  | 1,083.62  | -0.21% | 552.84    | 517.03    | 6.48%  | 462.39    | 371.86   | 19.58% |
| 32,000 | 1  | 2,918.42  | 2,831.99  | 2.96%  | 1,815.68  | 1,628.04  | 10.33% | 802.02    | 632.25   | 21.17% |

Across these 12 cases, the arithmetic mean TTFT reduction is 0.55% at 50% hit, 8.56% at 80% hit, and 26.94% at 100% hit. The result shows that R160 primarily benefits workloads with a high external-cache hit rate; at 50% hit, the I/O reduction and decompression overhead are approximately balanced.

### 5.3 R200 Test Environment

- Model: Qwen2.5-14B-Instruct, tensor parallel size 4
- Hardware: Kunpeng 920 5250 + 4 × Ascend 910B4
- Codec: R200 (`compress_ratio: 16`)
- Modes: Layer-wise (`use_layerwise: true`) and Block-wise (`use_layerwise: false`)
- I/O: `io_direct: true`
- Output length: 1 token
- External-cache hit rates: 50%, 80%, and 100%

### 5.4 R200 Layer-wise Results

| Input tokens | Concurrency | 50% hit baseline (ms) | 50% hit R200 (ms) | Reduction | 80% hit baseline (ms) | 80% hit R200 (ms) | Reduction | 100% hit baseline (ms) | 100% hit R200 (ms) | Reduction |
| -----------: | ----------: | --------------------: | -----------------: | --------: | --------------------: | -----------------: | --------: | ---------------------: | ------------------: | --------: |
| 4,000  | 1  | 357.99    | 341.97    | 4.47%  | 207.02    | 225.74    | -9.04% | 208.37    | 193.22   | 7.27%  |
| 8,000  | 1  | 531.60    | 578.86    | -8.89% | 304.06    | 321.29    | -5.67% | 350.01    | 331.22   | 5.37%  |
| 16,000 | 1  | 1,236.44  | 1,284.19  | -3.86% | 800.25    | 720.40    | 9.98%  | 611.74    | 532.57   | 12.94% |
| 32,000 | 1  | 3,278.62  | 3,152.37  | 3.85%  | 2,077.46  | 1,807.59  | 12.99% | 1,204.62  | 959.57   | 20.34% |
| 4,000  | 8  | 1,339.06  | 1,396.03  | -4.25% | 868.46    | 839.52    | 3.33%  | 1,056.99  | 829.89   | 21.49% |
| 8,000  | 8  | 2,353.23  | 2,373.60  | -0.87% | 1,436.33  | 1,356.39  | 5.57%  | 2,024.04  | 1,421.53 | 29.77% |
| 16,000 | 8  | 5,311.28  | 5,548.89  | -4.47% | 3,433.49  | 2,829.16  | 17.60% | 3,808.22  | 2,906.72 | 23.67% |
| 32,000 | 8  | 13,796.04 | 13,660.42 | 0.98%  | 8,353.08  | 7,851.27  | 6.01%  | 6,939.55  | 5,274.13 | 24.00% |
| 4,000  | 16 | 2,280.48  | 2,343.98  | -2.78% | 1,428.27  | 1,232.16  | 13.73% | 2,154.65  | 1,342.75 | 37.68% |
| 8,000  | 16 | 4,414.05  | 4,436.29  | -0.50% | 2,639.71  | 2,388.07  | 9.53%  | 4,170.57  | 2,646.00 | 36.56% |
| 16,000 | 16 | 10,144.97 | 9,960.42  | 1.82%  | 6,167.56  | 5,084.08  | 17.57% | 7,170.80  | 4,005.21 | 44.15% |
| 32,000 | 16 | 25,967.02 | 24,714.92 | 4.82%  | 15,550.23 | 12,950.47 | 16.72% | 12,261.77 | 7,504.62 | 38.80% |

### 5.5 R200 Block-wise Results

| Input tokens | Concurrency | 50% hit baseline (ms) | 50% hit R200 (ms) | Reduction | 80% hit baseline (ms) | 80% hit R200 (ms) | Reduction | 100% hit baseline (ms) | 100% hit R200 (ms) | Reduction |
| -----------: | ----------: | --------------------: | -----------------: | --------: | --------------------: | -----------------: | --------: | ---------------------: | ------------------: | --------: |
| 4,000  | 1  | 423.87    | 406.85    | 4.02%  | 307.49    | 253.90    | 17.43% | 260.84    | 198.13   | 24.04% |
| 8,000  | 1  | 688.38    | 665.83    | 3.28%  | 507.52    | 419.18    | 17.41% | 460.22    | 352.85   | 23.33% |
| 16,000 | 1  | 1,528.11  | 1,479.79  | 3.16%  | 1,083.55  | 874.61    | 19.28% | 617.35    | 446.64   | 27.65% |
| 32,000 | 1  | 3,678.05  | 3,426.42  | 6.84%  | 2,327.42  | 2,039.48  | 12.37% | 1,387.85  | 832.33   | 40.03% |
| 4,000  | 8  | 1,801.28  | 1,673.41  | 7.10%  | 1,261.30  | 1,025.12  | 18.73% | 1,117.65  | 745.11   | 33.33% |
| 8,000  | 8  | 3,262.38  | 2,952.30  | 9.50%  | 2,344.01  | 1,880.82  | 19.76% | 2,381.43  | 1,314.63 | 44.80% |
| 16,000 | 8  | 6,982.66  | 6,343.26  | 9.16%  | 4,897.66  | 3,862.16  | 21.14% | 4,157.12  | 2,060.17 | 50.44% |
| 32,000 | 8  | 16,615.58 | 14,960.37 | 9.96%  | 10,797.82 | 8,443.65  | 21.80% | 7,872.14  | 3,896.58 | 50.50% |
| 4,000  | 16 | 3,181.24  | 2,928.33  | 7.95%  | 2,228.06  | 1,775.39  | 20.32% | 2,402.10  | 1,152.30 | 52.03% |
| 8,000  | 16 | 6,082.47  | 5,437.54  | 10.60% | 4,213.28  | 3,197.17  | 24.12% | 4,381.51  | 2,386.19 | 45.54% |
| 16,000 | 16 | 12,832.20 | 11,805.53 | 8.00%  | 8,489.46  | 6,721.34  | 20.83% | 6,231.51  | 3,555.04 | 42.95% |
| 32,000 | 16 | 29,728.23 | 28,228.75 | 5.04%  | 18,745.21 | 15,293.23 | 18.42% | 13,077.52 | 6,831.96 | 47.76% |

## 6. Scope of Application

This section specifies the mandatory prerequisites, recommended scenarios, and not recommended scenarios for the compression function, to help you determine whether to enable it and how to configure it for optimal benefits.

### 6.1 Mandatory Prerequisites

The compression function can only be enabled when all of the following conditions are met:

- Software stack: Use `UcmPipelineStore` with `store_pipeline: "Cache|Compress|Posix"`.
- Data type: The current codec implementation supports BF16 (`data_type: 0`) only.
- Codec selection: Use `compress_ratio: 20` for R160 or `compress_ratio: 16` for R200. Both modes are lossy; `compress_ratio: 32` bypasses compression.
- Storage backend: Use a Posix-compatible local file system, SSD, or mounted network file system. Compression does not benefit a pure HBM-only cache path.
- Accuracy validation: Evaluate task-level accuracy and generated output quality with representative model inputs before production deployment. R160 normally retains more BF16 information than R200, but neither mode is bit-exact.
- Cache compatibility: Use a separate storage directory, or clear existing persisted cache data, when changing `compress_ratio` or codec versions.
- Hardware validation: The R160 performance data in this document was collected on Kunpeng 920 7280Z and Ascend 910. Performance and optimum thread count must be revalidated on other platforms.

　

### 6.2 Recommended Scenarios

Compress Store is most suitable when external-storage I/O contributes materially to TTFT:

- High external-cache hit rates, especially 80% to 100%. In the R160 measurements above, the average TTFT reduction was 8.56% at 80% hit and 26.94% at 100% hit.
- Long-context or high-concurrency workloads that generate enough storage traffic for byte reduction to offset CPU decompression and scheduling overhead.
- Storage-capacity or storage-bandwidth constrained deployments. Nominal payload reduction is 37.5% for R160 and 50% for R200, before accounting for 4 KiB alignment.
- Workloads that prefer higher numerical fidelity can start with R160. Workloads that are more I/O constrained and tolerate greater precision loss can evaluate R200.
- Layer-wise loading on servers with enough CPU capacity to run decompression in parallel with the rest of the inference pipeline.

### 6.3 Not Recommended Scenarios

Do not enable Compress Store by default in the following situations without additional validation:

- Low external-cache hit rates. In the measured R160 cases, the average reduction at 50% hit was only 0.55%, and individual cases showed small regressions.
- CPU-core or memory-bandwidth constrained deployments where decompression competes with inference scheduling and other host-side work.
- Workloads whose storage path is already faster than the decompressor, or where storage loading is not on the TTFT critical path.
- Non-BF16 KV cache formats, which are not supported by the current codecs.
- Accuracy-sensitive workloads that have not completed R160/R200 quality evaluation.
- Pure HBM/DRAM cache scenarios without external-storage loading.
