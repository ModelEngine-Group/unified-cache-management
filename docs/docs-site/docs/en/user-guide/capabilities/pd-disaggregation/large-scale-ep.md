# Scaling PD Deployments

This guide extends [Transport with UCM](distributed.md) to larger deployments. Establish a working P/D group first; the KV handoff continues to use the transport connector.

Scaling a PD deployment involves two separate choices: how many independent
prefill/decode instances to run, and how each instance distributes its model
across devices and nodes. UCM provides prefix reuse within the selected engine
integration. It does not choose the model's expert partition or coordinate
collective communication.

This guide gives the manual deployment steps and a separate Chart topology path,
followed by the checks needed to assess performance.

## Manual deployment

This path restores the original Ascend deployment scripts. It is independent of Helm: prepare matching vLLM-Ascend and Mooncake binaries, the model weights, collective networking, and shared storage before starting. Paths under `/vllm-workspace` refer to your installed source checkouts. These version-dependent scripts have not been rerun on accelerator hardware in this documentation update.

The original GLM-5.1-w4a8 deployment uses eight Atlas 800T A2 nodes, each with eight Ascend 910B3 NPUs: Prefill is DP4TP8 on 192.168.10.1–4; Decode is DP8TP4 on 192.168.10.5–8. The original storage setup used A800 through CE8875; provide the shared filesystem at `/mnt/test1`.

### Start Mooncake and prepare configuration

Run the master on 192.168.10.1. Save `mooncake.json` in the working directory on every engine node, and save the UCM YAML as `/etc/ucm/pd.yaml` on Prefill nodes.

```bash
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
mooncake_master --port 50088 \
    --eviction_high_watermark_ratio 0.9 \
    --eviction_ratio 0.1 \
    --default_kv_lease_ttl 11000
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

export ENABLE_UCM_PATCH=1
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
                    "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/etc/ucm/pd.yaml"}
                }
            ]
        }
    }' 2>&1 | tee "prefiller_dp_$dp_rank.log"
```

### Decode script

Save as `decode.sh` on every Decode node. Decode uses the Mooncake consumer and does not load the UCM YAML.

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

export ENABLE_UCM_PATCH=1
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

### Launch DP processes

Save the launcher as `run_multi_dp.sh`. Replace `local_ip`, `nic_name`, `dp_rank_start` and `dp_address` for each node. Keep the original DP/TP sizes and port layout.

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

Prefill ranks on .1/.2/.3/.4 are 0/1/2/3. On Decode nodes save the following launcher instead; .5/.6/.7/.8 use starting ranks 0/2/4/6.

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

```bash
bash run_multi_dp.sh
```

### Start the proxy and send requests

```bash
python /vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
    --port 7850 \
    --host 0.0.0.0 \
    --prefiller-hosts 192.168.10.1 192.168.10.2 192.168.10.3 192.168.10.4 \
    --prefiller-ports 9000 9000 9000 9000 \
    --decoder-hosts 192.168.10.5 192.168.10.5 192.168.10.6 192.168.10.6 192.168.10.7 192.168.10.7 192.168.10.8 192.168.10.8 \
    --decoder-ports 9000 9001 9000 9001 9000 9001 9000 9001
```

Use the same completions request format as the shared-store guide, with the proxy port and GLM model name:

```bash
curl http://localhost:7850/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/models/GLM-5.1-w4a8",
        "prompt": "What date is today?",
        "max_tokens": 20,
        "temperature": 0
    }'
```

## Helm deployment

The following Chart path is independent of the manual scripts above. Its bundled model profiles describe their own topology and parameters.

### Describe the topology before changing arguments

Three replica settings have different meanings:

| Setting | Meaning |
| --- | --- |
| `modelSpec.replicas` | Number of independent serving groups |
| `roles[].replicas` | Number of logical instances of that role within each group |
| `roles[].workerReplicas` | Additional worker Pods belonging to each logical instance |

Within one serving group, the engine Pod count is the sum of
`role replicas × (1 + workerReplicas)` across its roles. Device counts then
follow each Pod's resource requests. These counts do not include the router,
Mooncake master, or other control-plane Pods.

For example, the bundled `values-qwen3-0p6b-2p2-2d2.yaml` profile has two
prefill instances and two decode instances, each with one worker. It therefore
requests eight engine Pods per serving group. It is not a four-Pod profile.
Default hostname anti-affinity also affects the number of schedulable nodes.
Inspect the rendered placement rules instead of inferring node count from the
profile filename.

### Assign DP, TP, and EP to the engine

Tensor parallelism, data parallelism, and expert parallelism are engine
execution settings. Put supported model/parallel arguments in each role's
`vllmArgs`, and verify that the engine version accepts the intended combination.
When using multi-node DP, align global and local DP sizes with the number of
Pods and devices actually allocated to the role instance.

The Chart owns HTTP binding, DP address/rank coordination, served model name,
and KV transfer arguments. Do not duplicate those flags or paste a separate
multi-process launch script into the profile. Use the existing entrypoint to
make the rendered configuration correspond to the process that starts.

The repository includes Ascend multi-node profiles for DeepSeek-V3.1 and
Qwen3-235B, but those files define single-role serving layouts. They are useful
for checking model argument structure; they are not ready-made EP PD recipes.
A model-specific EP deployment still needs a validated engine configuration,
collective-network setup, and sufficient memory on the target hardware.

Changing P and D to different TP or EP layouts also changes the transfer
compatibility question. Validate the selected transport's supported layout
conversion. For shared-store PD, the ordinary UCM key includes TP size and
rank, so a new layout cannot be assumed to reuse an old layout's blocks.

### Select the HTTP serving mode

The Chart's `dataParallelMode` controls how multi-node role instances expose
HTTP endpoints:

- `standard`: one entry HTTP endpoint with headless workers. `ModelServer`
  selects entry Pods.
- `hybrid`: entry and worker nodes of multi-node roles expose HTTP, and the
  router selects those endpoints. This requires PD, an enabled router, and at
  least one role with workers.

The Chart injects the corresponding vLLM LB arguments. Do not independently
set `--data-parallel-hybrid-lb`, `--headless`, or managed rank flags. Check the
final command and `ModelServer.workloadSelector` together; a healthy worker
outside the selector does not receive router traffic.

### Reserve the transfer and cache resources

Each logical P/D instance receives a distinct engine identity. Its entry and
workers resolve that same identity from the serving-group and role labels.
Mooncake `instanceStride` must cover the configured DP, TP, PP, and context
parallel port span. The Chart checks the conservative bound and rejects port
ranges beyond 65535.

Treat collective networking, KV transfer, and external storage as separate
consumers of host and network resources. For UCM, budget host buffers per
process and provision storage reachable by the prefill instances that should
share prefixes. Adding decode replicas does not add UCM readers in the current
Chart's transport-only decode path.

Use [Kubernetes deployment](../../frameworks/kubernetes/deploy.md) to render
site values and inspect the resulting `ModelServing`, `ModelServer`, and
`ModelRoute`. A render confirms the configuration contract; it does not test
collectives, transfer throughput, or expert placement on hardware.

### Validate one change at a time

1. Establish a cold-request baseline on one P and one D with correct output.
2. Add the required workers and engine parallelism without changing UCM's
   backend or workload. Confirm rank initialization and request transfer.
3. Validate prefill-side external reuse after restarting engine processes while
   retaining storage. Check UCM hits and successful loads separately from HBM
   hits.
4. Increase role replicas or serving groups. Confirm routing distribution,
   identity uniqueness, and storage access for every new prefill instance.
5. Apply EP-specific tuning only after the previous layout works, preserving
   enough results to attribute any difference to that change.

Choose an input/output length distribution and arrival schedule representative
of the intended workload. Report completed requests, correctness failures,
client TTFT, TPOT, throughput, and resource consumption. A mean latency gain
with more failures or fewer output tokens is not an equivalent comparison.

### Interpret the result

A high prefill UCM hit rate can coexist with poor end-to-end TTFT when requests
queue at the router, storage reads are slow, or transfer waits for decode.
Likewise, increasing expert-parallel capacity can improve compute utilization
while increasing communication cost. Use per-stage timing and endpoint metrics
to identify which stage changed; do not attribute every speedup to UCM.
