# NFS Store

This document provides a usage example and configuration guide for the **NFS Connector**. This connector enables offloading of KV cache from GPU HBM to SSD or Local Disk, helping reduce memory pressure and support larger models or batch sizes.

## Performance

### Overview
The following are the multi-concurrency performance test results of UCM in the Prefix Cache scenario under a CUDA environment, showing the performance improvements of UCM on two different models.
During the tests, HBM cache was disabled, and KV Cache was retrieved and matched only from SSD.

In the QwQ-32B model, the test used one H20 server with 2 GPUs.
In the DeepSeek-V3 model, the test used two H20 servers with 16 GPUs.

Here, Full Compute refers to pure VLLM inference, while Disk80% indicates that after UCM pooling, the SSD hit rate of the KV cache is 80%.

The following table shows the results on the QwQ-32B model:
|      **QwQ-32B** |                |                     |                |              |
| ---------------: | -------------: | ------------------: | -------------: | :----------- |
| **Input length** | **Concurrent** | **Full Compute(s)** | **Disk80%(s)** | **Speedup**  |
|            2 000 |              1 |              0.5311 |         0.2053 | **+158.7 %** |
|            4 000 |              1 |              1.0269 |         0.3415 | **+200.7 %** |
|            8 000 |              1 |              2.0902 |         0.6429 | **+225.1 %** |
|           16 000 |              1 |              4.4852 |         1.3598 | **+229.8 %** |
|           32 000 |              1 |             10.2037 |         3.0713 | **+232.2 %** |
|            2 000 |              2 |              0.7938 |         0.3039 | **+161.2 %** |
|            4 000 |              2 |              1.5383 |         0.4968 | **+209.6 %** |
|            8 000 |              2 |              3.1323 |         0.9544 | **+228.2 %** |
|           16 000 |              2 |              6.7984 |         2.0149 | **+237.4 %** |
|           32 000 |              2 |             15.3395 |         4.5619 | **+236.3 %** |
|            2 000 |              4 |              1.6572 |         0.5998 | **+176.3 %** |
|            4 000 |              4 |              2.8173 |         1.2657 | **+122.6 %** |
|            8 000 |              4 |              5.2643 |         1.9829 | **+165.5 %** |
|           16 000 |              4 |             11.3651 |         3.9776 | **+185.7 %** |
|           32 000 |              4 |             25.6718 |         8.2881 | **+209.7 %** |
|            2 000 |              8 |              2.8559 |         1.2250 | **+133.1 %** |
|            4 000 |              8 |              5.0003 |         2.0995 | **+138.2 %** |
|            8 000 |              8 |              9.5365 |         3.6584 | **+160.7 %** |
|           16 000 |              8 |             20.3839 |         6.8949 | **+195.6 %** |
|           32 000 |              8 |             46.2107 |        14.8704 | **+210.8 %** |

The following table shows the results on the DeepSeek-V3 model:
|  **DeepSeek-V3** |                |                     |                |              |
| ---------------: | -------------: | ------------------: | -------------: | :----------- |
| **Input length** | **Concurrent** | **Full Compute(s)** | **Disk80%(s)** | **Speedup**  |
|            2 000 |              1 |             0.66971 |        0.33960 | **+97.2 %**  |
|            4 000 |              1 |             1.73146 |        0.48720 | **+255.4 %** |
|            8 000 |              1 |             3.33155 |        0.86782 | **+283.9 %** |
|           16 000 |              1 |             6.71235 |        2.09067 | **+221.1 %** |
|           32 000 |              1 |            14.16003 |        4.26111 | **+232.3 %** |
|            2 000 |              2 |             0.94628 |        0.50635 | **+86.9 %**  |
|            4 000 |              2 |             2.56590 |        0.71750 | **+257.6 %** |
|            8 000 |              2 |             4.98428 |        1.32238 | **+276.9 %** |
|           16 000 |              2 |            10.08294 |        3.10009 | **+225.2 %** |
|           32 000 |              2 |            21.11799 |        6.35784 | **+232.2 %** |
|            2 000 |              4 |             2.86674 |        0.84273 | **+240.2 %** |
|            4 000 |              4 |             5.42761 |        1.35695 | **+300.0 %** |
|            8 000 |              4 |            10.90076 |        3.02942 | **+259.8 %** |
|           16 000 |              4 |            22.43841 |        6.59230 | **+240.4 %** |
|           32 000 |              4 |            43.29353 |       14.51481 | **+198.3 %** |
|            2 000 |              8 |             5.69329 |        1.82275 | **+212.3 %** |
|            4 000 |              8 |            11.80801 |        3.36708 | **+250.7 %** |
|            8 000 |              8 |            23.93016 |        7.01634 | **+241.1 %** |
|           16 000 |              8 |            42.04222 |       14.78947 | **+184.3 %** |
|           32 000 |              8 |            78.55850 |       35.63042 | **+120.5 %** |

## Features

The NFS connector supports the following functionalities:

- `dump`: Offload KV cache blocks from HBM to SSD or Local Disk.
- `load`: Load KV cache blocks from SSD or Local Disk back to HBM.
- `lookup`: Look up KV blocks stored in SSD or Local Disk by block hash.
- `wait`: Ensure that all dump or load operations have completed.

## Configure UCM for Prefix Caching

Modify the UCM configuration file to specify which UCM connector to use and where KV blocks should be stored.  
You may directly edit the example file at:

`unified-cache-management/examples/ucm_config_example.yaml`

A minimal configuration looks like this:

```yaml
ucm_connectors:
  - ucm_connector_name: "UcmNfsStore"
    ucm_connector_config:
      storage_backends: "/mnt/test"
      io_direct: false

load_only_first_rank: false
```
### Required Parameters

* **ucm_connector_name**:  
  Specifies `UcmNfsStore` as the UCM connector.

* **storage_backends**:  
  Directory used for storing KV blocks. Can be a local path or an NFS-mounted path.  
  **⚠️ Replace `"/mnt/test"` with your actual storage directory.**

### Optional Parameters

* **io_direct** (optional, default: `false`):  
  Whether to enable direct I/O.

* **stream_number** *(optional, default: 8)*  
  Number of concurrent streams used for data transfer.

* **timeout_ms** *(optional, default: 30000)*  
  Timeout in milliseconds for external interfaces.

* **buffer_number** *(optional, default: 4096)*  
  The number of intermediate buffers for data transfer.

* **shard_data_dir** *(optional, default: true)*   
  Whether files are spread across subdirectories or stored in a single directory.

### Must-be-Set Parameters

* **load_only_first_rank** (must be `false`):  
  This feature is currently disabled.

## Launching Inference

In this guide, we describe **online inference** using vLLM with the UCM connector, deployed as an OpenAI-compatible server. For best performance with UCM, it is recommended to set `block_size` to 128.

To start the vLLM server with the Qwen/Qwen2.5-14B-Instruct model, run:

```bash
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
    "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/vllm-workspace/unified-cache-management/examples/ucm_config_example.yaml"}
}'
```

**⚠️ Make sure to replace `"/vllm-workspace/unified-cache-management/examples/ucm_config_example.yaml"` with your actual config file path.**

If you see log as below:

```bash
INFO:     Started server process [1049932]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

Congratulations, you have successfully started the vLLM server with UCM connector!

## Evaluating UCM Prefix Caching Performance
After launching the vLLM server with `UCMConnector` enabled, the easiest way to observe the prefix caching effect is to run the built-in `vllm bench` CLI. Executing the following command **twice** in a separate terminal shows the improvement clearly.

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

### After the first execution
The `vllm bench` terminal prints the benchmark result:

```
---------------Time to First Token----------------
Mean TTFT (ms):                           15323.87
```

Inspecting the vLLM server logs reveals entries like:

```
INFO ucm_connector.py:228: request_id: xxx, total_blocks_num: 125, hit hbm: 0, hit external: 0
```

This indicates that for the first inference request, UCM did not hit any cached KV blocks. As a result, the full 16K-token prefill must be computed, leading to a relatively large TTFT.

### After the second execution
Running the same benchmark again produces:

```
---------------Time to First Token----------------
Mean TTFT (ms):                            3183.97
```

The vLLM server logs now contain similar entries:

```
INFO ucm_connector.py:228: request_id: xxx, total_blocks_num: 125, hit hbm: 0, hit external: 125
```

This indicates that during the second request, UCM successfully retrieved all 125 cached KV blocks from the storage backend. Leveraging the fully cached prefix significantly reduces the initial latency observed by the model, yielding an approximate **5× improvement in TTFT** compared to the initial run.

### Log Message Structure
> If you want to view detailed transfer information, set the environment variable `UC_LOGGER_LEVEL` to `debug`.
```text
[UC][D] Task(<task_id>,<direction>,<task_count>,<size>) finished, costs=<time>s, bw={speed}GB/s
```
| Component    | Description                                                                 |
|--------------|-----------------------------------------------------------------------------|
| `task_id`    | Unique identifier for the task                                              |
| `direction`  | `PC::D2S`: Dump to Storage (Device → SSD)<br>`PC::S2D`: Load from Storage (SSD → Device) |
| `task_count` | Number of tasks executed in this operation                                  |
| `size`       | Total size of data transferred in bytes (across all tasks)                  |
| `time`       | Time taken for the complete operation in seconds                            |
| `speed`      | Task transfer speed between Device and Storage                              |