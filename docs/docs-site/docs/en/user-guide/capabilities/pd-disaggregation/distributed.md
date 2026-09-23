# Transport with UCM

This guide covers the transport-connector handoff path in [PD Disaggregation](index.md). The manual examples use Ascend; the Helm configuration also applies to CUDA with the corresponding model profile.

The transport path shown here composes two connectors on prefill:
Mooncake sends the request's KV to decode, while UCM loads and saves reusable
prefix blocks in external storage. Decode runs the transport consumer alone.
This allows prefix reuse on prefill without making decode read the UCM store.

See [PD integration](../../../developer-guide/pd-integration.md) for request ordering and initialization identifiers.


## Manual deployment on Ascend

This path restores the original Ascend deployment scripts. It is independent of Helm: prepare matching vLLM-Ascend and Mooncake binaries, the model weights, collective networking, and shared storage before starting. Paths under `/vllm-workspace` refer to your installed source checkouts. These version-dependent scripts have not been rerun on accelerator hardware in this documentation update.

### MoE: DP4TP4 on four A2 nodes

Use Qwen3-235B-A22B-W8A8. Prefill runs on 192.168.10.1–2 and Decode on 192.168.10.3–4, with two DP processes per eight-NPU node.

### Start Mooncake and prepare configuration

Run the master on 192.168.10.1. Save `mooncake.json` in the working directory on every engine node, and save the UCM YAML as `/etc/ucm/pd.yaml` on Prefill nodes.

```bash
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
mooncake_master --port 50088 --eviction_high_watermark_ratio 0.9 --eviction_ratio 0.1 --default_kv_lease_ttl 11000
```

```json
{
    "metadata_server": "P2PHANDSHAKE",
    "protocol": "ascend",
    "device_name": "",
    "master_server_address": "192.168.10.1:50088",
    "global_segment_size": "1GB"
}
```

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

### Prefill script

Save as `prefill.sh` on every Prefill node.

```bash
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

export ENABLE_UCM_PATCH=1
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
                    "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/etc/ucm/pd.yaml"}
                }
            ]
        }
    }' 2>&1 | tee "prefiller_dp_$dp_rank.log"
```

### Decode script

Save as `decode.sh` on every Decode node. Decode uses the Mooncake consumer and does not load the UCM YAML.

```bash
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

export ENABLE_UCM_PATCH=1
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

### Launch DP processes

Save the launcher as `run_multi_dp.sh`. Replace `local_ip`, `nic_name`, `dp_rank_start` and `dp_address` for each node. Keep the original DP/TP sizes and port layout.

```bash
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

On Prefill nodes .1/.2 use `dp_address=192.168.10.1`, ranks 0/2 and `template_path="./prefill.sh"`. On Decode nodes .3/.4 use `dp_address=192.168.10.3`, ranks 0/2 and `template_path="./decode.sh"`. Both launchers keep DP4, TP4 and two local processes. Run from the directory containing the scripts and `mooncake.json`:

```bash
bash run_multi_dp.sh
```

### Start the proxy and send requests

```bash
python /vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
  --port 7850 \
  --host 0.0.0.0 \
  --prefiller-hosts 192.168.10.1 192.168.10.1 192.168.10.2 192.168.10.2 \
  --prefiller-ports 9000  9001 9000 9001 \
  --decoder-hosts 192.168.10.3 192.168.10.3 192.168.10.4 192.168.10.4 \
  --decoder-ports 9000 9001 9000 9001 \
```

The original workload invokes the proxy as follows; it is a test command, not a measured result of this update.

```bash
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

### Dense model: QwQ-32B, DP1TP4

Use the same master, `mooncake.json` and 64 GiB UCM YAML. Run Prefill and Decode on separate nodes with devices 0–3; each uses HTTP port 9000 and Mooncake port 20001 on its own node. For additional instances on one node, assign disjoint devices and unique ports.

#### Prefill

```bash
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

export ENABLE_UCM_PATCH=1
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
                    "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/etc/ucm/pd.yaml"}
                }
            ]
        }
    }' 2>&1 | tee "prefiller.log"
```

#### Decode

```bash
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

export ENABLE_UCM_PATCH=1
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

#### Proxy and request workload

```bash
python /vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
  --port 7850 \
  --host 0.0.0.0 \
  --prefiller-hosts 192.168.10.1 192.168.10.1 \
  --prefiller-ports 9000  9001 \
  --decoder-hosts 192.168.10.2 192.168.10.2 \
  --decoder-ports 9000 9001 \
```

```bash
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

## Helm deployment

The following Chart path is independent of the manual scripts above. Its bundled model profiles describe their own topology and parameters.

### Choose a profile

Start from `models/ascend/values-qwen3-0p6b-1p1-1d1.yaml` or `models/cuda/values-qwen3-0p6b-1p1-1d1.yaml` in the unpacked Chart, matching the target platform.
It defines one prefill and one decode role, a Mooncake master, and routing
resources. Use the [Helm deployment guide](../../frameworks/kubernetes/deploy.md) to prepare
site values, render, and install.

The profile is a configuration example. Replace the engine image using
[Installation](../../quick_start/index.md), mount the intended model, and set
resources, storage, networking, and scheduler values for the target cluster.
The profile's placeholder StorageClass must be replaced. Its host mounts and
RDMA resource names also require actual cluster support.

### Keep the configuration responsibilities separate

The following fields live under `servingEngineSpec.modelSpec`:

| Field | Responsibility |
| --- | --- |
| `roles[]` | P/D replica counts, workers, device resources, and model arguments |
| `pd.prefill`, `pd.decode` | Identify the role names used by routing |
| `pd.kvTransfer.connector` | Select the engine transport |
| `pd.kvTransfer.routerType` | Select the corresponding router protocol |
| `pd.kvTransfer.identity` | Reserve engine IDs and transport port ranges |
| `unifiedcacheConfig` | Enable and configure UCM on prefill |
| `storage.unifiedcacheStorage` | Supply UCM mounts; mount paths populate `storage_backends` |

For `MooncakeConnectorV1` or `MooncakeHybridConnector`, the matching router type
is `mooncake`. The Chart composes `MultiConnector` on prefill when UCM is
effective, and retains a transport-only consumer on decode. Do not put a
second `--kv-transfer-config` into `roles[].vllmArgs`: the Chart owns that flag.

`NixlConnector` uses router type `nixl`, but this Chart currently rejects its
combination with effective UCM configuration. Selecting an accepted transport
name does not establish that the chosen Ascend image implements it; verify the
image's connector support before changing the bundled Mooncake profile.

Disabling `unifiedcacheConfig.enabled` removes the UCM connector and managed
cache mounts while retaining the PD transport. The image's `ENABLE_UCM_PATCH`
environment variable is independently configured. Record it explicitly in
baseline comparisons.

### Validate in three stages

**Serving and transfer.** Check prefill/decode readiness and routing objects,
then send a cold request through the pre-installed kthena-router gateway.
Inspect the selected roles and transport errors. The release's engine Service
selects both roles for monitoring and is not the PD client endpoint.

**External reuse.** Populate the UCM store with a fixed prompt, verify completed
writes, restart serving processes while preserving storage, and replay that
prompt. Inspect prefill's UCM hit tokens and successful load activity separately
from engine-memory hits. Decode need not show UCM hits in this topology.

**Performance.** Replay the same traffic with UCM disabled, UCM cold, and UCM
externally warm. Keep transport, P/D counts, output lengths, and generation
settings fixed. Collect failures, answer correctness, client TTFT, TPOT,
throughput, and store/transport timings.

### Diagnose by boundary

| Symptom | First evidence to inspect |
| --- | --- |
| Roles are not Ready | Device allocation, model mounts, runtime compatibility, collective initialization |
| Engines work separately but gateway requests fail | `ModelRoute`, role labels, router protocol, transport address/port reachability |
| Cold PD works but repeated prompts have no UCM hits | Prefill connector composition, persistence threshold, cache contents and key compatibility |
| Hits increase but TTFT does not improve | Successful load latency, prefill compute saved, transfer time, router/engine queueing |

For multi-node role instances or MoE models, continue with
[Scaling PD Deployments](large-scale-ep.md). Avoid changing the model,
parallel layout, and cache backend in one step; each changes a different part
of the evidence needed to explain a result.
