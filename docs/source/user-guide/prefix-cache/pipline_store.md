# PipelineStore

**PipelineStore** is a composite store built by **chaining multiple Store implementations** together to form a data transfer pipeline.  

Currently, the pipeline supports a chain composed of **Cache Store** and **Posix Store**.

In this chained pipeline:
- **Cache Store** handles data transfer between the **Device and Host**.
- Once the data flows from the Device to the Host, **Posix Store** is responsible for transferring the data between the **Host and POSIX-compliant persistent storage**, such as local disks, SSDs, or remote NFS (including NFS over RDMA) mount points.

At present, only this Store chain is supported.  
Additional Store implementations will be developed in the future and **chained** into the pipeline to enable more flexible and extensible transfer paths.


## Performance

### Overview
The following are the multi-concurrency performance test results of UCM in the Prefix Cache scenario under a CUDA environment, showing the performance improvements of UCM.
During the tests, HBM cache was disabled, and KV Cache was retrieved and matched only from SSD. 

Here, Full Compute refers to pure VLLM inference, while SSD80% indicates that after UCM pooling, the SSD hit rate of the KV cache is 80%.

The following table shows the results on the QwQ-32B model(**4 x H100 GPUs**):
|      **QwQ-32B** |                |                      |                |               |
| ---------------: | -------------: | -------------------: | -------------: | :------------ |
| **Input length** | **Concurrent** | **Full Compute (ms)** | **SSD80% (ms)** | **Speedup (%)** |
|            4 000 |              1 |              223.05 |         156.54 | **+42.5%**   |
|            8 000 |              1 |              350.47 |         228.27 | **+53.5%**   |
|           16 000 |              1 |              708.94 |         349.17 | **+103.0%**  |
|           32 000 |              1 |             1512.04 |         635.18 | **+138.0%**  |
|            4 000 |              8 |              908.52 |         625.92 | **+45.1%**   |
|            8 000 |              8 |             1578.72 |         955.25 | **+65.3%**   |
|           16 000 |              8 |             3139.03 |        1647.72 | **+90.5%**   |
|           32 000 |              8 |             6735.25 |        3025.23 | **+122.6%**  |
|            4 000 |             16 |             1509.79 |         919.53 | **+64.2%**   |
|            8 000 |             16 |             2602.34 |        1480.30 | **+75.8%**   |
|           16 000 |             16 |             5732.49 |        2393.54 | **+139.5%**  |
|           32 000 |             16 |            11891.61 |        4790.00 | **+148.3%**  |


The following table shows the results on the DeepSeek-R1-awq model (**8 × H100 GPUs**):
|**DeepSeek-R1-awq**|                |                      |                |               |
| -----------------:| -------------: | -------------------: | -------------: | :------------ |
| **Input length**  | **Concurrent** | **Full Compute (ms)** | **SSD80% (ms)** | **Speedup (%)** |
|             4 000 |              1 |               429.30 |        261.34 | **+64.3%**   |
|             8 000 |              1 |               762.23 |        363.37 | **+109.8%**  |
|            16 000 |              1 |              1426.06 |        586.17 | **+143.3%**  |
|            32 000 |              1 |              3086.85 |       1073.25 | **+187.6%**  |
|             4 000 |              8 |              1823.55 |       1017.72 | **+79.2%**   |
|             8 000 |              8 |              3214.76 |       1511.16 | **+112.7%**  |
|            16 000 |              8 |              6417.81 |       2596.70 | **+147.2%**  |
|            32 000 |              8 |             14278.00 |       5111.67 | **+179.3%**  |
|             4 000 |             16 |              3205.22 |       1534.00 | **+108.9%**  |
|             8 000 |             16 |              5813.09 |       2208.60 | **+163.2%**  |
|            16 000 |             16 |             11752.48 |       4000.46 | **+193.8%**  |
|            32 000 |             16 |             38643.73 |      19910.41 | **+94.1%**   |



## Configuration for Prefix Caching

Modify the UCM configuration file to specify which UCM connector to use and where KV blocks should be stored.  
You may directly edit the example file at:

`unified-cache-management/examples/ucm_config_example.yaml`

A minimal configuration looks like this:

```yaml
ucm_connectors:
  - ucm_connector_name: "UcmPipelineStore"
    ucm_connector_config:
      store_pipeline: "Cache|Posix"
      storage_backends: "/mnt/test"

load_only_first_rank: false
```

### Required Parameters

* **ucm_connector_name**:  
  Specifies `UcmPipelineStore` as the UCM connector.

* **store_pipeline: "Cache|Posix"**  
  Specifies a pipeline built by **chaining the Cache Store and the Posix Store**.  
  In this chained pipeline, the Cache Store handles data transfer between the **Device and Host**,  
  and once the data reaches the Host, the Posix Store transfers it between the **Host and POSIX-compliant persistent storage**.

  The pipeline must be registered in advance in  
  `unified-cache-management/ucm/store/pipeline/connector.py` under `PIPELINE_REGISTRY`.

  Currently, **only this Store chain is supported**.


* **storage_backends**:  
  Directory used for storing KV blocks. Can be a local path or an NFS-mounted path.  
  **⚠️ Replace `"/mnt/test"` with your actual storage directory.**

### Optional Parameters

* **io_direct** (optional, default: `false`):  
  Whether to enable direct I/O.

* **stream_number** *(optional, default: 8)*  
  Number of concurrent streams used for data transfer.

* **waiting_queue_depth** *(optional, default: 1024)*  
  Depth of the waiting queue for transfer tasks.  

* **running_queue_depth** *(optional, default: 32768)*  
  Depth of the running queue for transfer tasks.  

* **timeout_ms** *(optional, default: 30000)*  
  Timeout in milliseconds for external interfaces.

* **buffer_size** *(optional, default: 64GB)*  
  Amount of dram pinned memory used by a single worker process.

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
Mean TTFT (ms):                           15001.64
```

Inspecting the vLLM server logs reveals entries like:

```
INFO ucm_connector.py:317: request_id: xxx, total_blocks_num: 125, hit hbm: 0, hit external: 0
```

This indicates that for the first inference request, UCM did not hit any cached KV blocks. As a result, the full 16K-token prefill must be computed, leading to a relatively large TTFT.

### After the second execution
Running the same benchmark again produces:

```
---------------Time to First Token----------------
Mean TTFT (ms):                            2874.21
```

The vLLM server logs now contain similar entries:

```
INFO ucm_connector.py:317: request_id: xxx, total_blocks_num: 125, hit hbm: 0, hit external: 125
```

This indicates that during the second request, UCM successfully retrieved all 125 cached KV blocks from the storage backend. Leveraging the fully cached prefix significantly reduces the initial latency observed by the model, yielding an approximate **5× improvement in TTFT** compared to the initial run.

### Log Message Structure
> If you want to view detailed transfer information, set the environment variable `UC_LOGGER_LEVEL` to `debug`.

You may see the following typical log messages in the logs.

```text
[UC][D] Cache task({task_id},{operation},{subtask_number},{size}) dispatching. [PID,TID]
```
This log indicates that the **Cache Store** has received a **load or dump task**
| Component    | Description                                                                 |
|--------------|-----------------------------------------------------------------------------|
| `task_id`    | Unique identifier for the Cache Ctore task                                              |
| `operation`  | `DUMP`: Dump to Host(Device → Host) <br>`LOAD`: Load from Host (Host → Device) |
| `subtask_number` | Number of subtasks executed in this operation                                  |
| `size`       | Total size of data transferred in bytes (across all tasks)                  |

```test
[UC][D] Cache task({task_id},{operation},{subtask_number},{size}) finished, cost {time}ms. [PID,TID]
```
This log indicates that a load or dump task in the **Cache Store** has completed, along with its execution time **in ms**.

```test
[UC][D] Posix task({task_id},{operation},{subtask_number},{size}) dispatching. [PID,TID]
```
This log indicates that the **Posix Store** has received a **load or dump task**
| Component    | Description                                                                 |
|--------------|-----------------------------------------------------------------------------|
| `task_id`    | Unique identifier for the Posix Store task                                              |
| `operation` | `Cache2Backend`: Dump data from Cache Store to Posix Store.<br>`Backend2Cache`: Load data from Posix Store back to Cache Store. |
| `subtask_number` | Number of subtasks executed in this operation                                  |
| `size`       | Total size of data transferred in bytes (across all tasks)                  |

```test
[UC][D] Posix task({task_id},{operation},{subtask_number},{size}) finished, cost {time}ms. [PID,TID]
```
This log indicates that a load or dump task in the **Posix Store** has completed, along with its execution time in **in ms**.