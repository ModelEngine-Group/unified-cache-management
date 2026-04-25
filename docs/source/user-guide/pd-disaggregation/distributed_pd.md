
# Distributed PD Disaggregation on Ascend

## Overview

This document demonstrates how to implement P2P distributed PD disaggregation on Ascend using UCM and Mooncake. In this architecture, UCM is used to implement prefix cache on Prefill nodes for KV cache reuse, while Mooncake handles the transfer of KV cache from Prefill instances to Decode instances.

## Prerequisites

* vLLM-ascend: Refer to the [vLLM-ascend installation documentation](https://docs.vllm.ai/projects/ascend/en/latest/installation.html#set-up-using-docker).
* UCM: Installed with reference to the Installation documentation.
* Mooncake: Mooncake is the serving platform for Kimi, a leading LLM service provided by Moonshot AI. Installation and Compilation Guide:[kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake?tab=readme-ov-file#build-and-use-binaries). Starting from vLLM-ascend version 0.11.0, Mooncake is pre-installed in the official vLLM-ascend image, so manual installation is no longer required.


## PD Disaggregation for MoE Models

Running MoE models typically requires enabling data parallelism. This section demonstrates a 1P1D example with external load balancing for data parallelism. For questions regarding external load balancing data parallelism, please refer to the vLLM official documentation: [Data Parallel Deployment of external Load Balancing](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/#external-load-balancing). In this demonstration, both the Prefill instance and Decode instance use DP4TP4 as their parallelism strategy. Assume a total of 4 A2 servers with IPs 192.168.10.1-192.168.10.4.

**Step 1: Run the Mooncake Master Service**

First, run the Mooncake master service on any node. Here we select 192.168.10.1 and execute the following script:

```shell
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
mooncake_master --port 50088 --eviction_high_watermark_ratio 0.9 --eviction_ratio 0.1 --default_kv_lease_ttl 11000
```

Additionally, prepare a mooncake.json file on each node, which will be used when starting the vLLM service.

```json
{
    "metadata_server": "P2PHANDSHAKE",
    "protocol": "ascend",
    "device_name": "",
    "master_server_address": "192.168.10.1:50088",
    "global_segment_size": "1GB"
}
```

Also prepare a UCM configuration file (`ucm_config_example.yaml`) for prefix cache on Prefill nodes:

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

Key configuration parameters:
- **storage_backends**: The shared storage directory accessible from all nodes (e.g., NFS-mounted path or 3FS).

> **Note**: For more configuration options, refer to [UCM PipelineStore Documentation](https://ucm.readthedocs.io/en/latest/user-guide/prefix-cache/pipeline_store.html).

**Step 2: Run the Prefill Service**

First, prepare a `prefill.sh` file:

```shell
# prefill.sh
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

# basic configuration for HCCL and connection
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export HCCL_BUFFSIZE=256
export ASCEND_RT_VISIBLE_DEVICES=$device_list

# pytorch_npu settings and vllm settings
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export VLLM_USE_MODELSCOPE="True"

vllm serve /models/Qwen3-235B-A22B-W8A8 \
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
                        "prefill": {
                            "dp_size": '$dp_size',
                            "tp_size": '$tp_size'
                        },
                        "decode": {
                            "dp_size": '$dp_size',
                            "tp_size": '$tp_size'
                        }
                    }
                },
                {
                    "kv_connector": "UCMConnector",
                    "kv_role": "kv_both",
                    "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
                    "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/vllm-workspace/unified-cache-management/examples/ucm_config_example.yaml"}
                }
            ]
        }
    }' 2>&1 | tee "prefiller_dp_$dp_rank.log"
```

Then execute the following script on 192.168.10.1 and 192.168.10.2 respectively to start the DP4TP4 Prefill instance. After script execution, 4 DP processes will be started, with 2 on each node, collectively forming one Prefill instance. You need to modify the following parameters:

* `local_ip`: IP address of the current node
* `nic_name`: The network interface name corresponding to local_ip
* `dp_address`: Fixed to 192.168.10.1
* `dp_rank_start`: The global rank of the first DP process on the current node. Use 0 for 192.168.10.1, and 2 for 192.168.10.2

```shell
# run_multi_dp.sh
#!/bin/bash

# ==========================================
# Configuration parameters
# ==========================================
local_ip="xxxx"
nic_name="xxxx"
tp_size=4
dp_size=4           # total number of DP engines for decode/prefill
dp_size_local=2     # number of DP engines on the current node
dp_rank_start=xxxx     # starting DP rank for the current node
dp_address="xxxx"   # master node IP for DP communication
dp_rpc_port=13395       # port used for DP communication
server_port=9000    # starting port for all DP groups on the current node
mooncake_port=20001
template_path="./prefill.sh"
cards_per_node=8   # total number of NPU cards per machine (8 for 8-card machines, 16 for 16-card machines)


# Calculate the number of cards allocated to each process
cards_per_process=$((cards_per_node / dp_size_local))
echo "Total cards on current node: $cards_per_node, Number of processes to start: $dp_size_local, Cards per process: $cards_per_process"
echo "Starting $dp_size_local DP engine processes..."

# ==========================================
# Start processes
# ==========================================
pids=()

for ((i=0; i<dp_size_local; i++)); do
  dp_rank=$((dp_rank_start + i))
  dp_rank_local=$i
  server_port=$((server_port + i))
  mooncake_port=$((mooncake_port + i * tp_size))

  start_card=$((i * cards_per_process))
  device_list=$(seq -s, $start_card $((start_card + cards_per_process - 1)))

  command="bash $template_path $device_list $local_ip $nic_name $server_port $tp_size $dp_size $dp_rank $dp_address $dp_rpc_port $mooncake_port"
  
  echo "Starting process $i (rank: $dp_rank, port: $server_port)..."
  
  eval "$command" &
  pids+=($!)
done

# ==========================================
# Wait for all processes to complete
# ==========================================
echo "All processes started, waiting for completion..."

for pid in "${pids[@]}"; do
  wait "$pid"
  if [ $? -ne 0 ]; then
      echo "Warning: Process $pid exited abnormally"
  fi
done

echo "All DP engine processes have completed."
```

**Step 3: Run the Decode Service**

First, prepare a decode.sh file:

```shell
# decode.sh
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

# basic configuration for HCCL and connection
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export HCCL_BUFFSIZE=256
export ASCEND_RT_VISIBLE_DEVICES=$device_list

# pytorch_npu settings and vllm settings
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export VLLM_USE_MODELSCOPE="True"

vllm serve /models/Qwen3-235B-A22B-W8A8 \
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
            "prefill": {
                "dp_size": '$dp_size',
                "tp_size": '$tp_size'
            },
            "decode": {
                "dp_size": '$dp_size',
                "tp_size": '$tp_size'
            }
        }
    }' 2>&1 | tee "decoder_dp_$dp_rank.log"
```

Then run the `run_multi_dp.sh` script from Step 2 on 192.168.10.3 and 192.168.10.4 respectively to start the DP4TP4 Decode instance. Modify the following parameters:

* `local_ip`: IP address of the current node
* `nic_name`: The network interface name corresponding to local_ip
* `dp_address`: Fixed to 192.168.10.3
* `dp_rank_start`: The global rank of the first DP process on the current node. Use 0 for 192.168.10.3, and 2 for 192.168.10.4
* `template_path`: "./decode.sh"

**Step 4: Run the Load Balancing Service**

Execute the following command on any node to run the load balancing service:

```shell
python /vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
  --port 7850 \
  --host 0.0.0.0 \
  --prefiller-hosts 192.168.10.1 192.168.10.1 192.168.10.2 192.168.10.2 \
  --prefiller-ports 9000  9001 9000 9001 \
  --decoder-hosts 192.168.10.3 192.168.10.3 192.168.10.4 192.168.10.4 \
  --decoder-ports 9000 9001 9000 9001 \
```

**Step 5: Performance Testing**

You can use the native vLLM benchmark to test the performance of the PD disaggregation service. Assuming the load balancing service is running on 192.168.10.1, execute the following command:

```shell
vllm bench serve \
    --backend vllm \
    --model /models/Qwen3-235B-A22B-W8A8 \
    --host 192.168.10.1 \
    --port 7850 \
    --seed 123456  \
    --dataset-name random \
    --num-prompts 10 \
    --random-input-len 8000 \
    --random-output-len 1000 \
    --request-rate inf \
    --ignore-eos 
```

## PD Disaggregation for Dense Models

For Dense models, running external load balancing data parallelism is simpler than for MoE models, as each DP is an independent instance.

**Step 1: Run the Mooncake Master Service**

First, run the Mooncake master service on any node. Using 192.168.10.1 as an example, execute the following script:

```shell
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
mooncake_master --port 50088 --eviction_high_watermark_ratio 0.9 --eviction_ratio 0.1 --default_kv_lease_ttl 11000
```

Additionally, prepare a mooncake.json file on each node, which will be used when starting the vLLM service.

```json
{
    "metadata_server": "P2PHANDSHAKE",
    "protocol": "ascend",
    "device_name": "",
    "master_server_address": "192.168.10.1:50088",
    "global_segment_size": "1GB"
}
```

Also prepare a UCM configuration file (`ucm_config_example.yaml`) for prefix cache on Prefill nodes:

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

Key configuration parameters:
- **storage_backends**: The shared storage directory accessible from all nodes (e.g., NFS-mounted path or 3FS).

> **Note**: For more configuration options, refer to [UCM PipelineStore Documentation](https://ucm.readthedocs.io/en/latest/user-guide/prefix-cache/pipeline_store.html).

**Step 2: Run the Prefill Service**

Execute the following script to run a DP1TP4 Prefill instance. Multiple executions will start multiple Prefill instances. When starting multiple Prefill instances on the same node, ensure that the `server_port` and `mooncake_port` are different for each instance.

```shell
#!/bin/sh
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH
export PYTHONHASHSEED=0
export PYTHONPATH=$PYTHONPATH:/vllm-workspace/vllm
export MOONCAKE_CONFIG_PATH="./mooncake.json"

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

# pytorch_npu settings and vllm settings
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export VLLM_USE_MODELSCOPE="True"

dp_size=1
tp_size=4
server_port=9000
mooncake_port=20001

vllm serve /models/QwQ-32B \
    --host 0.0.0.0 \
    --port $server_port \
    --data-parallel-size $dp_size \
    --tensor-parallel-size $tp_size \
    --no-enable-expert-parallel \
    --seed 1024 \
    --max-model-len 17000 \
    --max-num-batched-tokens 8000 \
    --trust-remote-code \
    --max-num-seqs 4 \
    --gpu-memory-utilization 0.92 \
    --quantization None \
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
                        "prefill": {
                            "dp_size": '$dp_size',
                            "tp_size": '$tp_size'
                        },
                        "decode": {
                            "dp_size": '$dp_size',
                            "tp_size": '$tp_size'
                        }
                    }
                },
                {
                    "kv_connector": "UCMConnector",
                    "kv_role": "kv_both",
                    "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
                    "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/vllm-workspace/unified-cache-management/examples/ucm_config_example.yaml"}
                }
            ]
        }
    }' 2>&1 | tee "prefiller.log"
```

**Step 3: Run the Decode Service**

Execute the following script to run a DP1TP4 Decode instance. When starting multiple Decode instances on the same node, ensure that the `server_port` and `mooncake_port` are different for each instance.

```shell
#!/bin/sh
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH
export PYTHONHASHSEED=0
export PYTHONPATH=$PYTHONPATH:/vllm-workspace/vllm
export MOONCAKE_CONFIG_PATH="./mooncake.json"

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

# pytorch_npu settings and vllm settings
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export VLLM_USE_MODELSCOPE="True"

dp_size=1
tp_size=4
server_port=9000
mooncake_port=20001

vllm serve /models/QwQ-32B \
    --host 0.0.0.0 \
    --port $server_port \
    --data-parallel-size $dp_size \
    --tensor-parallel-size $tp_size \
    --no-enable-expert-parallel \
    --seed 1024 \
    --max-model-len 17000 \
    --max-num-batched-tokens 8000 \
    --trust-remote-code \
    --max-num-seqs 4 \
    --gpu-memory-utilization 0.92 \
    --quantization None \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --kv-transfer-config \
    '{
        "kv_connector": "MooncakeConnectorV1",
        "kv_role": "kv_consumer",
        "kv_port": '$mooncake_port',
        "kv_connector_extra_config": {
            "prefill": {
                "dp_size": '$dp_size',
                "tp_size": '$tp_size'
            },
            "decode": {
                "dp_size": '$dp_size',
                "tp_size": '$tp_size'
            }
        }
    }' 2>&1 | tee "decoder.log"
```

**Step 4: Run the Load Balancing Service**

Assuming two Prefill instances have been started on 192.168.10.1 with ports 9000 and 9001 respectively, and two Decode instances have been started on 192.168.10.2 with ports 9000 and 9001, execute the following command to run the load balancing service:

```shell
python /vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
  --port 7850 \
  --host 0.0.0.0 \
  --prefiller-hosts 192.168.10.1 192.168.10.1 \
  --prefiller-ports 9000  9001 \
  --decoder-hosts 192.168.10.2 192.168.10.2 \
  --decoder-ports 9000 9001 \
```

**Step 5: Performance Testing**

You can use the native vLLM benchmark to test the performance of the PD disaggregation service. Assuming the load balancing service is running on 192.168.10.1, execute the following command:

```shell
vllm bench serve \
    --backend vllm \
    --model /models/QwQ-32B \
    --host 192.168.10.1 \
    --port 7850 \
    --seed 123456  \
    --dataset-name random \
    --num-prompts 10 \
    --random-input-len 8000 \
    --random-output-len 1000 \
    --request-rate inf \
    --ignore-eos 
```