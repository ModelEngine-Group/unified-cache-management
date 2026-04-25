# Centralized PD Disaggregation

## Overview
PD disaggregation can be implemented in two architectures: centralized and distributed. This document demonstrates how to run centralized PD disaggregation using UCM. In the centralized implementation, Prefill instances store KV Cache to a storage device accessible by all compute nodes via UCM, and Decode instances load KV Cache from that storage device to the GPU via UCM. No P2P communication is required between Prefill and Decode instances.

## 1p1d

This example demonstrates how to run unified-cache-management with disaggregated prefill using PipelineStore on a single node with a 1 prefiller + 1 decoder setup.

### Prerequisites
- UCM: Installed with reference to the Installation documentation.
- Hardware: At least 2 GPUs or 2 NPUs
- File System: When Prefill and Decode instances run on different nodes, all nodes must mount the same shared file system (e.g., NFS)

### Prepare UCM Configuration File

Create a UCM configuration file (e.g., `ucm_config_example.yaml`) with PipelineStore:

```yaml
ucm_connectors:
  - ucm_connector_name: "UcmPipelineStore"
    ucm_connector_config:
      store_pipeline: "Cache|Posix"
      storage_backends: "/mnt/test1"
      cache_buffer_capacity_gb: 32
enable_event_sync: true
use_layerwise: false
```

Key configuration parameters:
- **storage_backends**: The shared storage directory accessible from all nodes (e.g., NFS-mounted path).

> **Note**: For more configuration options, refer to [UCM PipelineStore Documentation](https://ucm.readthedocs.io/en/latest/user-guide/prefix-cache/pipeline_store.html).

### Start disaggregated service
For illustration purposes, let us take GPU as an example and assume the model used is Qwen2.5-7B-Instruct.Using ASCEND_RT_VISIBLE_DEVICES instead of CUDA_VISIBLE_DEVICES to specify visible devices when starting service on Ascend platform.

#### Run prefill server
Prefiller Launch Command:
```bash
export CUDA_VISIBLE_DEVICES=0 
vllm serve /home/models/Qwen2.5-7B-Instruct \
--max-model-len 20000 \
--tensor-parallel-size 1 \
--gpu_memory_utilization 0.87 \
--trust-remote-code \
--port 7800 \
--block-size 128 \
--kv-transfer-config \
'{
    "kv_connector": "UCMConnector",
    "kv_role": "kv_both",
    "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
    "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/vllm-workspace/unified-cache-management/examples/ucm_config_example.yaml"}
}'
```

#### Run decode server
Decoder Launch Command:
```bash
export CUDA_VISIBLE_DEVICES=0 
vllm serve /home/models/Qwen2.5-7B-Instruct \
--max-model-len 20000 \
--tensor-parallel-size 1 \
--gpu_memory_utilization 0.87 \
--trust-remote-code \
--port 7801 \
--block-size 128 \
--kv-transfer-config \
'{
    "kv_connector": "UCMConnector",
    "kv_role": "kv_both",
    "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
    "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/vllm-workspace/unified-cache-management/examples/ucm_config_example.yaml"}
}'
```

#### Run proxy server
Make sure prefill nodes and decode nodes can connect to each other.
```bash
cd /vllm-workspace/unified-cache-management/ucm/pd
python3 toy_proxy_server.py --pd-disaggregation --host localhost --port 7802 --prefiller-host <prefill-node-ip> --prefiller-port 7800 --decoder-host <decode-node-ip> --decoder-port 7801
```

### Testing and Benchmarking
#### Basic Test
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
#### Benchmark Test
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

## 1p1d with Different Platforms

This document demonstrates how to run unified-cache-management with disaggregated prefill using PipelineStore on different platforms, with a setup of one prefiller node and one decoder node.

If you need additional nodes to support your PD-disaggregation system, please refer to the [XpYd](#xpYd) documentation. 

When deploying your disaggregated PD system, please ensure the following needs:
- Environment Variable: Using  `ASCEND_RT_VISIBLE_DEVICES` instead of `CUDA_VISIBLE_DEVICES` to specify visible devices when starting service on Ascend platform.
- Data Type Consistency: All vLLM service instances must be configured with the same data type (`dtype`).

### Prerequisites
- UCM: Installed with reference to the Installation documentation.
- Hardware: At least 1 GPU and 1 NPU
- File System: When Prefill and Decode instances run on different nodes, all nodes must mount the same shared file system (e.g., NFS)

### Start disaggregated service
For illustration purposes, let us assume that the model used is Qwen2.5-7B-Instruct and the prefill platform is ascend while decode platform is cuda.

#### Run prefill server
Prefiller Launch Command:
```bash
export ASCEND_RT_VISIBLE_DEVICES=0
vllm serve /home/models/Qwen2.5-7B-Instruct \
--max-model-len 20000 \
--tensor-parallel-size 1 \
--gpu_memory_utilization 0.87 \
--trust-remote-code \
--port 7800 \
--block-size 128 \
--dtype bfloat16 \
--kv-transfer-config \
'{
    "kv_connector": "UCMConnector",
    "kv_role": "kv_both",
    "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
    "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/vllm-workspace/unified-cache-management/examples/ucm_config_example.yaml"}
}'
```

#### Run decode server
Decoder Launch Command:
```bash
export CUDA_VISIBLE_DEVICES=0 
vllm serve /home/models/Qwen2.5-7B-Instruct \
--max-model-len 20000 \
--tensor-parallel-size 1 \
--gpu_memory_utilization 0.87 \
--trust-remote-code \
--port 7801 \
--block-size 128 \
--dtype bfloat16 \
--kv-transfer-config \
'{
    "kv_connector": "UCMConnector",
    "kv_role": "kv_both",
    "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
    "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/vllm-workspace/unified-cache-management/examples/ucm_config_example.yaml"}
}'
```

#### Run proxy server
Make sure prefill nodes and decode nodes can connect to each other.
```bash
cd /vllm-workspace/unified-cache-management/ucm/pd
python3 toy_proxy_server.py --host localhost --port 7802 --prefiller-host <prefill-node-ip> --prefiller-port 7800 --decoder-host <decode-node-ip> --decoder-port 7801
```

### Testing and Benchmarking
#### Basic Test
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
#### Benchmark Test
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

## XpYd

This example demonstrates how to run unified-cache-management with disaggregated prefill using PipelineStore on with multiple prefiller + multiple decoder instances.

### Prerequisites
- UCM: Installed with reference to the Installation documentation.
- Hardware: At least 4 GPUs (At least 2 GPUs for prefiller + 2 for decoder in 2d2p setup or 2 NPUs for prefiller + 2 for decoder in 2d2p setup)
- File System: When Prefill and Decode instances run on different nodes, all nodes must mount the same shared file system (e.g., NFS)

### Start disaggregated service
For illustration purposes, let us take GPU as an example and assume the model used is Qwen2.5-7B-Instruct.Using ASCEND_RT_VISIBLE_DEVICES instead of CUDA_VISIBLE_DEVICES to specify visible devices when starting service on Ascend platform.

#### Run prefill servers
Prefiller1 Launch Command:
```bash
export CUDA_VISIBLE_DEVICES=0 
vllm serve /home/models/Qwen2.5-7B-Instruct \
--max-model-len 20000 \
--tensor-parallel-size 1 \
--gpu_memory_utilization 0.87 \
--trust-remote-code \
--port 7800 \
--block-size 128 \
--kv-transfer-config \
'{
    "kv_connector": "UCMConnector",
    "kv_role": "kv_both",
    "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
    "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/vllm-workspace/unified-cache-management/examples/ucm_config_example.yaml"}
}'
```

Prefiller2 Launch Command:
```bash
export CUDA_VISIBLE_DEVICES=1 
vllm serve /home/models/Qwen2.5-7B-Instruct \
--max-model-len 20000 \
--tensor-parallel-size 1 \
--gpu_memory_utilization 0.87 \
--trust-remote-code \
--port 7801 \
--block-size 128 \
--kv-transfer-config \
'{
    "kv_connector": "UCMConnector",
    "kv_role": "kv_both",
    "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
    "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/vllm-workspace/unified-cache-management/examples/ucm_config_example.yaml"}
}'
```

#### Run decode servers
Decoder1 Launch Command:
```bash
export PYTHONHASHSEED=123456
export CUDA_VISIBLE_DEVICES=2 
vllm serve /home/models/Qwen2.5-7B-Instruct \
--max-model-len 20000 \
--tensor-parallel-size 1 \
--gpu_memory_utilization 0.87 \
--trust-remote-code \
--enforce-eager \
--port 7802 \
--block-size 128 \
--kv-transfer-config \
'{
    "kv_connector": "UCMConnector",
    "kv_role": "kv_both",
    "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
    "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/vllm-workspace/unified-cache-management/examples/ucm_config_example.yaml"}
}'
```
Decoder2 Launch Command:
```bash
export PYTHONHASHSEED=123456
export CUDA_VISIBLE_DEVICES=3 
vllm serve /home/models/Qwen2.5-7B-Instruct \
--max-model-len 20000 \
--tensor-parallel-size 1 \
--gpu_memory_utilization 0.87 \
--trust-remote-code \
--enforce-eager \
--port 7803 \
--block-size 128 \
--kv-transfer-config \
'{
    "kv_connector": "UCMConnector",
    "kv_role": "kv_both",
    "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
    "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/vllm-workspace/unified-cache-management/examples/ucm_config_example.yaml"}
}'
```

#### Run proxy server
Make sure prefill nodes and decode nodes can connect to each other. the number of prefill/decode hosts should be equal to the number of prefill/decode ports.
```bash
cd /vllm-workspace/unified-cache-management/ucm/pd
python3 toy_proxy_server.py --pd-disaggregation --host localhost --port 7805 --prefiller-hosts <prefill-node-ip-1> <prefill-node-ip-2> --prefiller-port 7800 7801 --decoder-hosts <decoder-node-ip-1> <decoder-node-ip-2> --decoder-ports 7802 7803
```

### Testing and Benchmarking
#### Basic Test
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
#### Benchmark Test
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
    --port 7805 \
    --endpoint /v1/completions \
    --request-rate 1
```