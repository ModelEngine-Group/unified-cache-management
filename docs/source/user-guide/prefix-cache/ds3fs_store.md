# Ds3fs Store

For 3FS introduction and **deployment**, please refer to [the official documentation](https://github.com/deepseek-ai/3FS).  
**⚠️ Recommended 3FS source code compilation and installation path: /usr/local**

**Ds3fs Store** is chained together with **Cache Store** through **PipelineStore** to form a data transfer pipeline.

In this chained pipeline:
- **Cache Store** handles data transfer between the **Device and Host**.
- Once the data flows from the Device to the Host, **Ds3fs Store** is responsible for transferring the data between the **Host and 3FS**.


## Performance

### Overview
This document uses **two nodes** to deploy 3FS service. Specifically, **MGMTD, META, and Storage1 are deployed on the same node, while Storage2 is deployed on another node**.
Each Storage node aggregates **two NVME disks**.

Based on this deployment approach, the following are the multi-concurrency performance test results of UCM in the Prefix Cache scenario under a CUDA environment, showing the performance improvements of UCM.
During the tests, HBM cache was disabled, and KV Cache was retrieved and matched only from Ds3fs storage.

Here, Full Compute refers to pure VLLM inference, while Ds3fs80% indicates that after UCM pooling, the Ds3fs hit rate of the KV cache is 80%.

The following table shows the results on the QwQ-32B model(**2 x H100 GPUs**):
|      **QwQ-32B** |                |                      |                |               |
| ---------------: | -------------: | -------------------: | -------------: | :------------ |
| **Input length** | **Concurrent** | **Full Compute (ms)** | **Ds3fs80% (ms)** | **Speedup (%)** |
|            4 000 |              1 |              268.28 |         116.19 | **+131%**   |
|            8 000 |              1 |              533.62 |         222.95 | **+139%**   |
|           16 000 |              1 |             1144.92 |         493.03 | **+132%**   |
|           32 000 |              1 |             2544.43 |        1032.93 | **+146%**   |
|            4 000 |              2 |              402.22 |         231.55 | **+74%**   |
|            8 000 |              2 |              800.22 |         461.20 | **+74%**   |
|           16 000 |              2 |             1656.04 |         760.39 | **+118%**   |
|           32 000 |              2 |             3691.84 |        1490.49 | **+148%**   |
|            4 000 |              4 |              776.34 |         453.37 | **+71%**   |
|            8 000 |              4 |             1596.00 |         757.66 | **+111%**   |
|           16 000 |              4 |             3413.29 |        1489.32 | **+129%**   |
|           32 000 |              4 |             6922.88 |        3022.35 | **+129%**   |
|            4 000 |              8 |             1616.54 |         805.95 | **+101%**   |
|            8 000 |              8 |             3357.11 |        1528.08 | **+120%**   |
|           16 000 |              8 |             5970.69 |        2707.25 | **+121%**   |
|           32 000 |              8 |            12114.44 |        5578.02 | **+117%**   |


The following table shows the results on the DeepSeek-R1-awq model (**8 × H100 GPUs**):
|**DeepSeek-R1-awq**|                |                      |                |               |
| -----------------:| -------------: | -------------------: | -------------: | :------------ |
| **Input length**  | **Concurrent** | **Full Compute (ms)** | **Ds3fs80% (ms)** | **Speedup (%)** |
|             4 000 |              1 |               376.96 |        246.79 | **+53%**   |
|             8 000 |              1 |               699.60 |        354.76 | **+97%**   |
|            16 000 |              1 |              1389.29 |        595.04 | **+133%**  |
|            32 000 |              1 |              2937.73 |        987.10 | **+198%**  |
|             4 000 |              2 |               567.43 |        417.65 | **+36%**   |
|             8 000 |              2 |              1000.06 |        581.51 | **+72%**   |
|            16 000 |              2 |              1989.25 |        990.39 | **+101%**  |
|            32 000 |              2 |              4261.70 |       1688.59 | **+152%**  |
|             4 000 |              4 |              1038.78 |        742.64 | **+40%**   |
|             8 000 |              4 |              1993.80 |       1023.65 | **+95%**   |
|            16 000 |              4 |              4066.21 |       1893.01 | **+115%**  |
|            32 000 |              4 |              8210.61 |       3588.30 | **+129%**  |
|             4 000 |              8 |              2098.07 |       1232.56 | **+70%**   |
|             8 000 |              8 |              4157.40 |       2001.83 | **+108%**  |
|            16 000 |              8 |              7270.95 |       3493.18 | **+108%**  |
|            32 000 |              8 |             14542.17 |       6913.94 | **+110%**  |



## Configuration for Prefix Caching

Modify the UCM configuration file to specify which UCM connector to use and where KV blocks should be stored.
You may directly edit the example file at:

`unified-cache-management/examples/ucm_config_example.yaml`

A minimal configuration looks like this:

```yaml
ucm_connectors:
  - ucm_connector_name: "UcmPipelineStore"
    ucm_connector_config:
      store_pipeline: "Cache|Ds3fs"
      storage_backends: "/mount/3fs"
```

### Required Parameters

* **ucm_connector_name**:
  Specifies `UcmPipelineStore` as the UCM connector.

* **store_pipeline: "Cache|Ds3fs"**
  Specifies a pipeline built by **chaining the Cache Store and the Ds3fs Store**.
  In this chained pipeline, the Cache Store handles data transfer between the **Device and Host**,
  and once the data reaches the Host, the Ds3fs Store transfers it between the **Host and 3FS**.

  The pipeline must be registered in advance in
  `unified-cache-management/ucm/store/pipeline/connector.py` under `PIPELINE_REGISTRY`.

  Currently, **only this Store chain is supported**.


* **storage_backends**:
  3FS storage path used for storing KV blocks.  
  **⚠️ Replace `"/mount/3fs"` with your actual 3FS mount path.**

### Optional Parameters

* **io_direct** (optional, default: `true`):
  Whether to enable direct I/O.

* **stream_number** *(optional, default: 32)*
  Number of concurrent streams used for data transfer.

* **waiting_queue_depth** *(optional, default: 1024)*
  Depth of the waiting queue for transfer tasks.

* **running_queue_depth** *(optional, default: 32768)*
  Depth of the running queue for transfer tasks.

* **timeout_ms** *(optional, default: 30000)*
  Timeout in milliseconds for external interfaces.

* **buffer_size** *(optional, default: 64GB)*
  Amount of dram pinned memory used by a single worker process.


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
Mean TTFT (ms):                           7270.95
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
Mean TTFT (ms):                            1560.84
```

The vLLM server logs now contain similar entries:

```
INFO ucm_connector.py:317: request_id: xxx, total_blocks_num: 125, hit hbm: 0, hit external: 125
```

This indicates that during the second request, UCM successfully retrieved all 125 cached KV blocks from the Ds3fs storage backend. Leveraging the fully cached prefix significantly reduces the initial latency observed by the model, yielding an approximate **5× improvement in TTFT** compared to the initial run.

### Log Message Structure
> If you want to view detailed transfer information, set the environment variable `UC_LOGGER_LEVEL` to `debug`.

You may see the following typical log messages in the logs.

```text
[UC][D] Cache task({task_id},{operation},{subtask_number},{size}) dispatching. [PID,TID]
```
This log indicates that the **Cache Store** has received a **load or dump task**
| Component    | Description                                                                 |
|--------------|-----------------------------------------------------------------------------|
| `task_id`    | Unique identifier for the Cache Store task                                              |
| `operation`  | `DUMP`: Dump to Host(Device → Host) <br>`LOAD`: Load from Host (Host → Device) |
| `subtask_number` | Number of subtasks executed in this operation                                  |
| `size`       | Total size of data transferred in bytes (across all tasks)                  |

```text
[UC][D] Cache task({task_id},{operation},{subtask_number},{size}) finished, cost {time}ms. [PID,TID]
```
This log indicates that a load or dump task in the **Cache Store** has completed, along with its execution time **in ms**.

```text
[UC][D] Ds3fs task({task_id},{operation},{subtask_number},{size}) dispatching. [PID,TID]
```
This log indicates that the **Ds3fs Store** has received a **load or dump task**
| Component    | Description                                                                 |
|--------------|-----------------------------------------------------------------------------|
| `task_id`    | Unique identifier for the Ds3fs Store task                                              |
| `operation` | `Cache2Backend`: Dump data from Cache Store to Ds3fs Store.<br>`Backend2Cache`: Load data from Ds3fs Store back to Cache Store. |
| `subtask_number` | Number of subtasks executed in this operation                                  |
| `size`       | Total size of data transferred in bytes (across all tasks)                  |

```text
[UC][D] Ds3fs task({task_id},{operation},{subtask_number},{size}) finished, cost {time}ms. [PID,TID]
```
This log indicates that a load or dump task in the **Ds3fs Store** has completed, along with its execution time in **in ms**.
