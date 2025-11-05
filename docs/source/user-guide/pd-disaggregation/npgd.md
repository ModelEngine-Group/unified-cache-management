# 1p1d with different platforms

## Overview
This document demonstrates how to run unified-cache-management with disaggregated prefill using NFS connector on different platforms, with a setup of one prefiller node and one decoder node.

If you need additional nodes to support your PD-disaggregation system, please refer to the [XpYd](./xpyd.md) documentation. 

When deploying your disaggregated PD system, please ensure the following needs:
- Environment Variable: Using  `ASCEND_RT_VISIBLE_DEVICES` instead of `CUDA_VISIBLE_DEVICES` to specify visible devices when starting service on Ascend platform.
- Data Type Consistency: All vLLM service instances must be configured with the same data type (`dtype`).

## Prerequisites
- UCM: Installed with reference to the Installation documentation.
- Hardware: At least 1 GPU and 1 NPU

## Start disaggregated service
For illustration purposes, let us assume that the model used is Qwen2.5-7B-Instruct and the prefill platform is ascend while decode platform is cuda.

### Run prefill server
Prefiller Launch Command:
```bash
export PYTHONHASHSEED=123456
export ASCEND_RT_VISIBLE_DEVICES=0
vllm serve /home/models/Qwen2.5-7B-Instruct \
--max-model-len 20000 \
--tensor-parallel-size 1 \
--gpu_memory_utilization 0.87 \
--trust-remote-code \
--enforce-eager \
--no-enable-prefix-caching \
--port 7800 \
--block-size 128 \
--dtype bfloat16 \
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

### Run decode server
Decoder Launch Command:
```bash
export PYTHONHASHSEED=123456
export CUDA_VISIBLE_DEVICES=0 
vllm serve /home/models/Qwen2.5-7B-Instruct \
--max-model-len 20000 \
--tensor-parallel-size 1 \
--gpu_memory_utilization 0.87 \
--trust-remote-code \
--enforce-eager \
--no-enable-prefix-caching \
--port 7801 \
--block-size 128 \
--dtype bfloat16 \
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
Make sure prefill nodes and decode nodes can connect to each other.
```bash
cd vllm-workspace/unified-cache-management/ucm/pd
python3 toy_proxy_server.py --host localhost --port 7802 --prefiller-host <prefill-node-ip> --prefiller-port 7800 --decoder-host <decode-node-ip> --decoder-port 7801
```

## Testing and Benchmarking
### Basic Test
After running all servers , you can test with a simple curl command:
```bash
curl http://localhost:7802/v1/completions \
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
vllm bench serve \
    --backend vllm \
    --dataset-name random \
    --random-input-len 4096 \
    --random-output-len 100 \
    --num-prompts 10 \
    --ignore-eos \
    --model /home/models/Qwen2.5-7B-Instruct \
    --tokenizer /home/models/Qwen2.5-7B-Instruct \
    --host localhost \
    --port 7802 \
    --endpoint /v1/completions \
    --request-rate 1
```
