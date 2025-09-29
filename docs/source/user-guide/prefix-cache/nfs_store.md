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
- `commit`: Mark cache operations as complete and ready for reuse.

## Configuration

To use the NFS connector, you need to configure the `connector_config` dictionary in your model's launch configuration.

### Required Parameters

- `storage_backends` *(required)*:  
  The `storage_backends` directory can either be a local folder or an NFS-mounted directory backed by an SSD driver
- `transferStreamNumber`*(optional)*: 
  This parameter specifies the number of worker threads. The default is 32, but it can be adjusted as needed. A value of 16 or 32 is recommended.


### Example:

```python
kv_connector_extra_config={"ucm_connector_name": "UcmNfsStore", "ucm_connector_config":{"storage_backends": "/mnt/test1", "transferStreamNumber": 32}}
```

## Launching Inference

### Offline Inference

To start **offline inference** with the NFS connector，modify the script `examples/offline_inference.py` to include the `kv_connector_extra_config` for NFS connector usage:

```python
# In examples/offline_inference.py
ktc = KVTransferConfig(
    ...
    kv_connector_extra_config={"ucm_connector_name": "UcmNfsStore", "ucm_connector_config":{"storage_backends": "/mnt/test1", "transferStreamNumber": 32}}
)
```

Then run the script as follows:

```bash
cd examples/
export PYTHONHASHSEED=123456
python offline_inference.py
```

### Online Inference

For **online inference** , vLLM with our connector can also be deployed as a server that implements the OpenAI API protocol. Run the following command to start the vLLM server with the Qwen/Qwen2.5-14B-Instruct model:

```bash
export PYTHONHASHSEED=123456
vllm serve /home/models/Qwen2.5-14B-Instruct \
--max-model-len 20000 \
--tensor-parallel-size 2 \
--gpu_memory_utilization 0.87 \
--trust-remote-code \
--port 7800 \
--kv-transfer-config \
'{
    "kv_connector": "UnifiedCacheConnectorV1",
    "kv_connector_module_path": "ucm.integration.vllm.uc_connector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "ucm_connector_name": "UcmNfsStore",
        "ucm_connector_config": {
            "storage_backends": "/mnt/test",
            "transferStreamNumber":32
        }
    }
}'
```

If you see log as below:

```bash
INFO:     Started server process [1049932]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

Congratulations, you have successfully started the vLLM server with NFS Connector!

After successfully started the vLLM server，You can interact with the API as following:

```bash
curl http://localhost:7800/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/home/models/Qwen2.5-14B-Instruct",
        "prompt": "Shanghai is a",
        "max_tokens": 7,
        "temperature": 0
    }'
```
To quickly experience the NFS Connector's effect:

1. Start the service with:  
   `--no-enable-prefix-caching`  
2. Send the same request (exceed 128 tokens) twice consecutively
3. Remember to enable prefix caching (do not add `--no-enable-prefix-caching`) in production environments.
### Log Message Structure
```text
[UCMNFSSTORE] [I] Task(<task_id>,<direction>,<task_count>,<size>) finished, elapsed <time>s
```
| Component    | Description                                                                 |
|--------------|-----------------------------------------------------------------------------|
| `task_id`    | Unique identifier for the task                                              |
| `direction`  | `D2S`: Dump to Storage (Device → SSD)<br>`S2D`: Load from Storage (SSD → Device) |
| `task_count` | Number of tasks executed in this operation                                  |
| `size`       | Total size of data transferred in **bytes** (across all tasks)              |
| `time`       | Time taken for the complete operation in seconds                            |