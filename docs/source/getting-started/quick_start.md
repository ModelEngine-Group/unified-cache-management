# Quickstart
## Prerequisites

### GPU
- OS: Linux
- Python: 3.12
- GPU: NVIDIA compute capability 8.0+ (e.g., L20, L40, H20)
- CUDA: CUDA Version 12.8
- vLLM: v0.9.2

### NPU
- OS: Linux
- Python: >= 3.9, < 3.12
- NPU: Atlas 800 A2/A3 series
- CANN: CANN Version 8.1.RC1
- vLLM: v0.9.2
- vLLM Ascend: v0.9.2rc1

## Installation
Before you start with UCM, please make sure that you have installed UCM correctly by following the [GPU Installation](./installation_gpu.md) guide or [NPU Installation](./installation_npu.md) guide.

## Features Overview

UCM supports two key features: **Prefix Cache** and **Sparse attention**. 

Each feature supports both **Offline Inference** and **Online API** modes. 

For quick start, just follow the [usage](./quick_start.md) guide below to launch your own inference experience;

For further research on Prefix Cache, more details are available via the link below:
- [Prefix Cache](../user-guide/prefix-cache/index.md)

Various Sparse Attention features are now available, try GSA Sparsity via the link below:
- [GSA Sparsity](../user-guide/sparse-attention/gsa.md)

## Usage

<details open>
<summary><b>Offline Inference</b></summary>

You can use our official offline example script to run offline inference as following commands:

```bash
cd examples/
# Change the model path to your own model path
export MODEL_PATH=/home/models/Qwen2.5-14B-Instruct
python offline_inference.py
```

</details>

<details open>
<summary><b>OpenAI-Compatible Online API</b></summary>

For online inference , vLLM with our connector can also be deployed as a server that implements the OpenAI API protocol.

First, specify the python hash seed by:
```bash
export PYTHONHASHSEED=123456
```

Run the following command to start the vLLM server with the Qwen/Qwen2.5-14B-Instruct model:

```bash
# Change the model path to your own model path
export MODEL_PATH=/home/models/Qwen2.5-14B-Instruct
vllm serve ${MODEL_PATH} \
--served-model-name vllm_cpu_offload \
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
            "storage_backends": "/home/test"
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
        "model": "vllm_cpu_offload",
        "prompt": "Shanghai is a",
        "max_tokens": 7,
        "temperature": 0
    }'
```
</details>

Note: If you want to disable vLLM prefix cache to test the cache ability of UCM, you can add `--no-enable-prefix-caching` to the command line.