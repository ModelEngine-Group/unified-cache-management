# XpYd

## Overview
This example demonstrates how to run unified-cache-management with disaggregated prefill using NFS connector on with multiple prefiller + multiple decoder instances.

## Prerequisites
- UCM: Installed with reference to the Installation documentation.
- Hardware: At least 4 GPUs (At least 2 GPUs for prefiller + 2 for decoder in 2d2p setup)

## Start disaggregated service
For illustration purposes, let us assume that the model used is Qwen2.5-7B-Instruct.
### Run prefill servers
Prefiller1 Launch Command:
```bash
export PYTHONHASHSEED=123456
CUDA_VISIBLE_DEVICES=0 vllm serve /home/models/Qwen2.5-7B-Instruct \
--max-model-len 20000 \
--tensor-parallel-size 1 \
--gpu_memory_utilization 0.87 \
--trust-remote-code \
--enforce-eager \
--no-enable-prefix-caching \
--port 7800 \
--block-size 128 \
--kv-transfer-config \
'{
    "kv_connector": "UnifiedCacheConnectorV1",
    "kv_connector_module_path": "ucm.integration.vllm.uc_connector",
    "kv_role": "kv_producer",
    "kv_connector_extra_config": {
        "ucm_connector_name": "UcmNfsStore",
        "ucm_connector_config": {
            "storage_backends": "/mnt/test1",
            "transferStreamNumber":32
        }
    }
}'
```

Prefiller2 Launch Command:
```bash
export PYTHONHASHSEED=123456
CUDA_VISIBLE_DEVICES=1 vllm serve /home/models/Qwen2.5-7B-Instruct \
--max-model-len 20000 \
--tensor-parallel-size 1 \
--gpu_memory_utilization 0.87 \
--trust-remote-code \
--enforce-eager \
--no-enable-prefix-caching \
--port 7801 \
--block-size 128 \
--kv-transfer-config \
'{
    "kv_connector": "UnifiedCacheConnectorV1",
    "kv_connector_module_path": "ucm.integration.vllm.uc_connector",
    "kv_role": "kv_producer",
    "kv_connector_extra_config": {
        "ucm_connector_name": "UcmNfsStore",
        "ucm_connector_config": {
            "storage_backends": "/mnt/test1",
            "transferStreamNumber":32
        }
    }
}'
```

### Run decode servers
Decoder1 Launch Command:
```bash
export PYTHONHASHSEED=123456
CUDA_VISIBLE_DEVICES=2 vllm serve /home/models/Qwen2.5-7B-Instruct \
--max-model-len 20000 \
--tensor-parallel-size 1 \
--gpu_memory_utilization 0.87 \
--trust-remote-code \
--enforce-eager \
--no-enable-prefix-caching \
--port 7802 \
--block-size 128 \
--kv-transfer-config \
'{
    "kv_connector": "UnifiedCacheConnectorV1",
    "kv_connector_module_path": "ucm.integration.vllm.uc_connector",
    "kv_role": "kv_consumer",
    "kv_connector_extra_config": {
        "ucm_connector_name": "UcmNfsStore",
        "ucm_connector_config": {
            "storage_backends": "/mnt/test1",
            "transferStreamNumber":32
        }
    }
}'
```
Decoder2 Launch Command:
```bash
export PYTHONHASHSEED=123456
CUDA_VISIBLE_DEVICES=3 vllm serve /home/models/Qwen2.5-7B-Instruct \
--max-model-len 20000 \
--tensor-parallel-size 1 \
--gpu_memory_utilization 0.87 \
--trust-remote-code \
--enforce-eager \
--no-enable-prefix-caching \
--port 7803 \
--block-size 128 \
--kv-transfer-config \
'{
    "kv_connector": "UnifiedCacheConnectorV1",
    "kv_connector_module_path": "ucm.integration.vllm.uc_connector",
    "kv_role": "kv_consumer",
    "kv_connector_extra_config": {
        "ucm_connector_name": "UcmNfsStore",
        "ucm_connector_config": {
            "storage_backends": "/mnt/test1",
            "transferStreamNumber":32
        }
    }
}'
```

### Run proxy server
Make sure prefill nodes and decode nodes can connect to each other. the number of prefill/decode hosts should be equal to the number of prefill/decode ports.
```bash
cd vllm-workspace/unified-cache-management/ucm/pd
python3 toy_proxy_server.py --pd-disaggregation --host localhost --port 7805 --prefiller-hosts <prefill-node-ip-1> <prefill-node-ip-2> --prefiller-port 7800 7801 --decoder-hosts <decoder-node-ip-1> <decoder-node-ip-2> --decoder-ports 7802 7803
```

## Testing and Benchmarking
### Basic Test
After running all servers , you can test with a simple curl command:
```bash
curl http://localhost:7805/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/home/models/Qwen2.5-7B-Instruct",
        "prompt": "What date is today?",
        "max_tokens": 20,
        "temperature": 0
    }'
```
### Benchmark Test
Use the benchmark scripts provided by vLLM.
```bash
cd /vllm-workspace/vllm/benchmarks
python3 benchmark_serving.py \
    --backend vllm \
    --dataset-name random \
    --random-input-len 4096 \
    --random-output-len 100 \
    --num-prompts 10 \
    --ignore-eos \
    --model /home/models/Qwen2.5-7B-Instruct \
    --tokenizer /home/models/Qwen2.5-7B-Instruct \
    --host localhost \
    --port 7805 \
    --endpoint /v1/completions \
    --request-rate 1
```
