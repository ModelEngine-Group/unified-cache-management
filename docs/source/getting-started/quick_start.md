# Quickstart
## Prerequisites

- OS: Linux
- Python: 3.12
- GPU: NVIDIA compute capability 8.0+ (e.g., L20, L40, H20)
- CUDA: CUDA Version 12.8
- vLLM: v0.9.2

## Installation
Before you start with UCM, please make sure that you have installed UCM correctly by following the [Installation](./installation_gpu.md) guide.

## Features Overview

UCM supports two key features: **Prefix Cache** and **GSA Sparsity**. 

Each feature supports both **Offline Inference** and **Online API** modes. 

For quick start, just follow the [usage](#usage) guide below to launch your own inference experience;

For further research, click on the links blow to see more details of each feature:
- [Prefix Cache](../user-guide/prefix-cache/base.md)
- [GSA Sparsity](../user-guide/sparse-attention/gsa.md)

## Usage

<details open>
<summary><b>Offline Inference</b></summary>

You can use our official offline example script to run offline inference as following commands:

```bash
cd examples/
python offline_inference.py
```

</details>

<details>
<summary><b>OpenAI-Compatible Online API</b></summary>

For online inference , vLLM with our connector can also be deployed as a server that implements the OpenAI API protocol.

First, specify the python hash seed by:
```bash
export PYTHONHASHSEED=123456
```

Run the following command to start the vLLM server with the Qwen/Qwen2.5-14B-Instruct model:

```bash
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
        "ucm_connector_name": "UcmDramStore",
        "ucm_connector_config": {
            "max_cache_size": 5368709120,
            "kv_block_size": 262144
        }
    }
}'
```

If you see log as below:

```bash
INFO:     Started server process [32890]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

Congratulations, you have successfully started the vLLM server with UCM!

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
</details>

