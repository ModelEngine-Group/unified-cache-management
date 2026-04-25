# Large-Scale Expert Parallelism PD Disaggregation

This section demonstrates PD disaggregation for MoE models with large-scale Expert Parallelism. MoE models require enabling data parallelism to distribute expert weights across multiple nodes.

## Prerequisites

* vLLM-ascend: Refer to the [vLLM-ascend installation documentation](https://docs.vllm.ai/projects/ascend/en/latest/installation.html#set-up-using-docker).
* UCM: Installed with reference to the Installation documentation.
* Mooncake: Mooncake is the serving platform for Kimi, a leading LLM service provided by Moonshot AI. Installation and Compilation Guide:[kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake?tab=readme-ov-file#build-and-use-binaries). Starting from vLLM-ascend version 0.11.0, Mooncake is pre-installed in the official vLLM-ascend image, so manual installation is no longer required.

## Deployment Configuration

- **Prefill Instance**: 4 nodes (192.168.10.1-4), DP4TP8 (4 DP processes, each with TP8)
- **Decode Instance**: 4 nodes (192.168.10.5-8), DP8TP4 (8 DP processes, each with TP4)
- **Total**: 8 Atlas 800T A2 servers with 8 Ascend 910B3 NPU cards each
- **Storage**: 8 servers connected to AI storage device A800 via CE8875 switch

> **Note**: For external load balancing data parallelism, refer to the vLLM official documentation: [Data Parallel Deployment of external Load Balancing](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/#external-load-balancing).

## Deployment Steps

### Step 1: Run Mooncake Master Service

Run Mooncake master on any node (e.g., 192.168.10.1):

```bash
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
mooncake_master --port 50088 \
    --eviction_high_watermark_ratio 0.9 \
    --eviction_ratio 0.1 \
    --default_kv_lease_ttl 11000
```

Prepare `mooncake.json` on all 8 nodes:

```json
{
    "metadata_server": "P2PHANDSHAKE",
    "protocol": "ascend",
    "device_name": "",
    "master_server_address": "192.168.10.1:50088",
    "global_segment_size": "1GB"
}
```

### Step 2: Run Prefill Service (DP4TP8)

First, prepare a UCM configuration file (`ucm_config_example.yaml`) for prefix cache on Prefill nodes (192.168.10.1-4):

```yaml
ucm_connectors:
  - ucm_connector_name: "UcmPipelineStore"
    ucm_connector_config:
      store_pipeline: "Cache|Posix"
      storage_backends: "/mnt/test1"
      cache_buffer_capacity_gb: 64
enable_event_sync: true
use_layerwise: true
```

Prepare `prefill.sh` on Prefill nodes (192.168.10.1-4):

```bash
#!/bin/sh

export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH
export PYTHONHASHSEED=0
export PYTHONPATH=$PYTHONPATH:/vllm-workspace/vllm
export MOONCAKE_CONFIG_PATH="./mooncake.json"

device_list=$1
local_ip=$2
nic_name=$3
server_port=$4
tp_size=$5
dp_size=$6
dp_rank=$7
dp_address=$8
dp_rpc_port=$9
mooncake_port=${10}

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export HCCL_BUFFSIZE=256
export ASCEND_RT_VISIBLE_DEVICES=$device_list

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export VLLM_USE_MODELSCOPE="True"

vllm serve /models/GLM-5.1-w4a8 \
    --host 0.0.0.0 \
    --port $server_port \
    --data-parallel-size $dp_size \
    --data-parallel-address $dp_address \
    --data-parallel-rpc-port $dp_rpc_port \
    --data-parallel-rank $dp_rank \
    --tensor-parallel-size $tp_size \
    --enable-expert-parallel \
    --seed 1024 \
    --max-model-len 17000 \
    --max-num-batched-tokens 8000 \
    --trust-remote-code \
    --max-num-seqs 4 \
    --gpu-memory-utilization 0.92 \
    --quantization ascend \
    --enforce-eager \
    --additional-config '{"enable_weight_nz_layout":true,"enable_prefill_optimizations":true}' \
    --kv-transfer-config \
    '{
        "kv_connector": "MultiConnector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {
            "connectors": [
                {
                    "kv_connector": "MooncakeConnectorV1",
                    "kv_role": "kv_producer",
                    "kv_port": '$mooncake_port',
                    "kv_connector_extra_config": {
                        "prefill": {"dp_size": '$dp_size', "tp_size": '$tp_size'},
                        "decode": {"dp_size": 8, "tp_size": 4}
                    }
                },
                {
                    "kv_connector": "UCMConnector",
                    "kv_role": "kv_both",
                    "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
                    "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/path/to/ucm_config_example.yaml"}
                }
            ]
        }
    }' 2>&1 | tee "prefiller_dp_$dp_rank.log"
```

Prepare `run_multi_dp.sh` for Prefill nodes:

```bash
#!/bin/bash

local_ip="xxxx"           # IP of current node (192.168.10.1/2/3/4)
nic_name="xxxx"           # Network interface name corresponding to local_ip
tp_size=8
dp_size=4                 # Total DP engines for Prefill
dp_size_local=1           # 1 DP process per node (TP8 uses all 8 cards)
dp_rank_start=xxxx        # 0 for node1, 1 for node2, 2 for node3, 3 for node4
dp_address="192.168.10.1" # Master node for DP communication
dp_rpc_port=13395
server_port=9000
mooncake_port=20001
template_path="./prefill.sh"
cards_per_node=8

cards_per_process=$((cards_per_node / dp_size_local))

for ((i=0; i<dp_size_local; i++)); do
  dp_rank=$((dp_rank_start + i))
  server_port=$((server_port + i))
  mooncake_port=$((mooncake_port + i * tp_size))
  
  start_card=$((i * cards_per_process))
  device_list=$(seq -s, $start_card $((start_card + cards_per_process - 1)))
  
  bash $template_path $device_list $local_ip $nic_name $server_port $tp_size $dp_size $dp_rank $dp_address $dp_rpc_port $mooncake_port &
done

wait
```

Execute `run_multi_dp.sh` on each Prefill node (192.168.10.1-4) with appropriate `local_ip` and `dp_rank_start`:
- 192.168.10.1: `dp_rank_start=0`
- 192.168.10.2: `dp_rank_start=1`
- 192.168.10.3: `dp_rank_start=2`
- 192.168.10.4: `dp_rank_start=3`

### Step 3: Run Decode Service (DP8TP4)

Prepare `decode.sh` on Decode nodes (192.168.10.5-8):

```bash
#!/bin/sh

export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH
export PYTHONHASHSEED=0
export PYTHONPATH=$PYTHONPATH:/vllm-workspace/vllm
export MOONCAKE_CONFIG_PATH="./mooncake.json"

device_list=$1
local_ip=$2
nic_name=$3
server_port=$4
tp_size=$5
dp_size=$6
dp_rank=$7
dp_address=$8
dp_rpc_port=$9
mooncake_port=${10}

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export HCCL_BUFFSIZE=256
export ASCEND_RT_VISIBLE_DEVICES=$device_list

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export VLLM_USE_MODELSCOPE="True"

vllm serve /models/GLM-5.1-w4a8 \
    --host 0.0.0.0 \
    --port $server_port \
    --data-parallel-size $dp_size \
    --data-parallel-address $dp_address \
    --data-parallel-rpc-port $dp_rpc_port \
    --data-parallel-rank $dp_rank \
    --tensor-parallel-size $tp_size \
    --enable-expert-parallel \
    --seed 1024 \
    --max-model-len 17000 \
    --max-num-batched-tokens 8000 \
    --trust-remote-code \
    --max-num-seqs 4 \
    --gpu-memory-utilization 0.92 \
    --quantization ascend \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --kv-transfer-config \
    '{
        "kv_connector": "MooncakeConnectorV1",
        "kv_role": "kv_consumer",
        "kv_port": '$mooncake_port',
        "kv_connector_extra_config": {
            "prefill": {"dp_size": 4, "tp_size": 8},
            "decode": {"dp_size": '$dp_size', "tp_size": '$tp_size'}
        }
    }' 2>&1 | tee "decoder_dp_$dp_rank.log"
```

Prepare `run_multi_dp.sh` for Decode nodes:

```bash
#!/bin/bash

local_ip="xxxx"           # IP of current node (192.168.10.5/6/7/8)
nic_name="xxxx"           # Network interface name corresponding to local_ip
tp_size=4
dp_size=8                 # Total DP engines for Decode
dp_size_local=2           # 2 DP processes per node (TP4 uses 4 cards each)
dp_rank_start=xxxx        # 0 for node5, 2 for node6, 4 for node7, 6 for node8
dp_address="192.168.10.5" # Master node for DP communication
dp_rpc_port=13395
server_port=9000
mooncake_port=20001
template_path="./decode.sh"
cards_per_node=8

cards_per_process=$((cards_per_node / dp_size_local))

for ((i=0; i<dp_size_local; i++)); do
  dp_rank=$((dp_rank_start + i))
  server_port=$((server_port + i))
  mooncake_port=$((mooncake_port + i * tp_size))
  
  start_card=$((i * cards_per_process))
  device_list=$(seq -s, $start_card $((start_card + cards_per_process - 1)))
  
  bash $template_path $device_list $local_ip $nic_name $server_port $tp_size $dp_size $dp_rank $dp_address $dp_rpc_port $mooncake_port &
done

wait
```

Execute `run_multi_dp.sh` on each Decode node (192.168.10.5-8) with appropriate `local_ip` and `dp_rank_start`:
- 192.168.10.5: `dp_rank_start=0`
- 192.168.10.6: `dp_rank_start=2`
- 192.168.10.7: `dp_rank_start=4`
- 192.168.10.8: `dp_rank_start=6`

### Step 4: Run Load Balancing Service

```bash
python /vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
    --port 7850 \
    --host 0.0.0.0 \
    --prefiller-hosts 192.168.10.1 192.168.10.2 192.168.10.3 192.168.10.4 \
    --prefiller-ports 9000 9000 9000 9000 \
    --decoder-hosts 192.168.10.5 192.168.10.5 192.168.10.6 192.168.10.6 192.168.10.7 192.168.10.7 192.168.10.8 192.168.10.8 \
    --decoder-ports 9000 9001 9000 9001 9000 9001 9000 9001
```

## Benchmark Results

The following benchmark demonstrates UCM prefix cache effectiveness in large-scale Expert Parallelism PD disaggregation scenarios.

### Test Configuration

- Total requests: 128
- Request concurrency: 128
- Constraint: Total requests kept within Prefill instance's available HBM capacity for KV cache storage

### KV Cache Pre-seeding Procedure

Before each test, KV cache must be pre-seeded with a prefix ratio of **0.8**:

1. **Pre-seed Phase**: Send 128 requests with input length = `target_input_length × 0.8` and output length = 1 to establish the KV cache prefix
2. **Test Phase**: Send 128 requests with full target input length and output length = 1000

Example for 32K input scenario:
- Pre-seed: 128 requests with 25600 (32K × 0.8) input tokens + 1 output token
- Test: 128 requests with 32000 input tokens + 1000 output tokens

This procedure ensures the prefix portion (80% of input) is cached before measuring performance, simulating real-world prefix reuse scenarios.

### Test Commands

```bash
# Step 1: Pre-seed KV cache (25600 = 32000 * 0.8)
vllm bench serve \
    --backend vllm \
    --model /models/GLM-5.1-w4a8 \
    --host 192.168.10.1 \
    --port 7850 \
    --seed 123456 \
    --dataset-name random \
    --num-prompts 128 \
    --random-input-len 25600 \
    --random-output-len 1 \
    --request-rate inf \
    --ignore-eos

# Step 2: Run performance test
vllm bench serve \
    --backend vllm \
    --model /models/GLM-5.1-w4a8 \
    --host 192.168.10.1 \
    --port 7850 \
    --seed 123456 \
    --dataset-name random \
    --num-prompts 128 \
    --random-input-len 32000 \
    --random-output-len 1000 \
    --request-rate inf \
    --ignore-eos
```

### Test Scenarios

| Scenario | Description |
|----------|-------------|
| **Recalculation** | Baseline without UCM, HBM prefix cache disabled (full recomputation) |
| **HBM PC** | Without UCM, HBM prefix cache enabled |
| **UCM PC** | With UCM prefix cache enabled |

### Performance Results

<table>
  <thead>
    <tr>
      <th rowspan="2">Input Length</th>
      <th rowspan="2">Output Length</th>
      <th colspan="3">Recalculation</th>
      <th colspan="3">HBM PC</th>
      <th colspan="3">UCM PC</th>
    </tr>
    <tr>
      <th>TTFT (ms)</th>
      <th>TPOT (ms)</th>
      <th>E2EL (ms)</th>
      <th>TTFT (ms)</th>
      <th>TPOT (ms)</th>
      <th>E2EL (ms)</th>
      <th>TTFT (ms)</th>
      <th>TPOT (ms)</th>
      <th>E2EL (ms)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>32K</strong></td>
      <td><strong>1K</strong></td>
      <td>140730</td>
      <td>64</td>
      <td>173820</td>
      <td>108879</td>
      <td>65</td>
      <td>142228</td>
      <td>51861</td>
      <td>66</td>
      <td>85615</td>
    </tr>
    <tr>
      <td><strong>64K</strong></td>
      <td><strong>1K</strong></td>
      <td>181864</td>
      <td>64</td>
      <td>214988</td>
      <td>144444</td>
      <td>65</td>
      <td>177561</td>
      <td>69718</td>
      <td>66</td>
      <td>103752</td>
    </tr>
    <tr>
      <td><strong>128K</strong></td>
      <td><strong>1K</strong></td>
      <td>268016</td>
      <td>65</td>
      <td>301648</td>
      <td>267680</td>
      <td>65</td>
      <td>301135</td>
      <td>105083</td>
      <td>66</td>
      <td>138946</td>
    </tr>
  </tbody>
</table>

> **Note**: Due to data parallelism, requests during the test phase may not be routed to the same DP process that was used for KV cache pre-seeding. As a result, HBM PC achieves an actual cache hit rate lower than the intended 0.8. UCM addresses this limitation by storing all KV cache in shared external storage, ensuring that requests can hit cached data regardless of which DP process handles them. This guarantees a true cache hit rate equal to the pre-seeding ratio of 0.8, significantly reducing TTFT compared to HBM PC. The improved TTFT effectively increases Prefill instance throughput, thereby boosting the overall system throughput.