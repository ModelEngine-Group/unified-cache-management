# 传输连接器与 UCM 组合

本页介绍 [PD 分离部署](index.md)中的传输连接器交接方案。手工示例使用 Ascend；Helm 配置也适用于 CUDA，需选用对应平台的模型配置。

这里的传输路径在 Prefill 侧组合两个 connector：Mooncake 将当前请求的 KV 发送给 Decode，UCM 从外部存储加载并保存可复用的前缀块。Decode 只运行传输 consumer。这样，Prefill 可以复用前缀，而不要求 Decode 读取 UCM 存储。

请求顺序与初始化标识的原理见[PD 集成原理](../../../developer-guide/pd-integration.md)。


## Ascend 手工部署

这条路径恢复原 Ascend 部署脚本，与 Helm 独立。启动前准备匹配的 vLLM-Ascend、Mooncake、模型权重、集合通信网络及共享存储。`/vllm-workspace` 下的路径对应部署环境中的源码目录。这些脚本依赖对应引擎版本，本次文档恢复未在加速设备上重跑。

### MoE：四台 A2，DP4TP4

模型为 Qwen3-235B-A22B-W8A8。Prefill 使用 192.168.10.1–2，Decode 使用 192.168.10.3–4；每台八卡节点启动两个 DP 进程。

### 启动 Mooncake 并准备配置

在 192.168.10.1 启动 master。每个引擎节点的工作目录保存 `mooncake.json`，Prefill 节点将 UCM YAML 保存为 `/etc/ucm/pd.yaml`。

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

### Prefill 脚本

在每个 Prefill 节点保存为 `prefill.sh`。

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

### Decode 脚本

在每个 Decode 节点保存为 `decode.sh`。Decode 使用 Mooncake consumer，不加载 UCM YAML。

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

### 启动 DP 进程

将启动器保存为 `run_multi_dp.sh`，按节点填写 `local_ip`、`nic_name`、`dp_rank_start` 和 `dp_address`，保持原 DP/TP 数量及端口布局。

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

Prefill 的 .1/.2 节点使用 `dp_address=192.168.10.1`、起始 rank 0/2、`template_path="./prefill.sh"`。Decode 的 .3/.4 节点改为 `dp_address=192.168.10.3`、起始 rank 0/2、`template_path="./decode.sh"`。两侧均保持 DP4、TP4、每节点两个进程。在脚本和 `mooncake.json` 所在目录执行：

```bash
bash run_multi_dp.sh
```

### 启动代理并发送请求

```bash
python /vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
  --port 7850 \
  --host 0.0.0.0 \
  --prefiller-hosts 192.168.10.1 192.168.10.1 192.168.10.2 192.168.10.2 \
  --prefiller-ports 9000  9001 9000 9001 \
  --decoder-hosts 192.168.10.3 192.168.10.3 192.168.10.4 192.168.10.4 \
  --decoder-ports 9000 9001 9000 9001 \
```

原工作负载通过下列命令调用代理；这是测试命令，不是本次更新测得的结果。

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

### Dense 模型：QwQ-32B，DP1TP4

复用上面的 master、`mooncake.json` 和 64 GiB UCM YAML。Prefill、Decode 分别在独立节点使用设备 0–3；各自节点的 HTTP 端口为 9000，Mooncake 端口为 20001。同节点增加实例时须分配不重叠的设备和独立端口。

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

#### 代理与请求负载

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

## Helm 部署

下面的 Chart 路径独立于上面的手工脚本，内置模型配置使用各自的拓扑与参数。

### 选择部署配置

根据目标平台，从解压后 Chart 中的 `models/ascend/values-qwen3-0p6b-1p1-1d1.yaml` 或 `models/cuda/values-qwen3-0p6b-1p1-1d1.yaml` 开始。它定义一个 Prefill 角色、一个 Decode 角色、Mooncake master 和路由资源。按 [Helm 部署](../../frameworks/kubernetes/deploy.md)准备集群和站点 values，完成渲染和安装。

该文件是配置示例。需要根据[安装](../../quick_start/index.md)替换引擎镜像，挂载目标模型，并为目标集群设置资源、存储、网络和调度器参数。必须替换示例中的 StorageClass 占位值；主机挂载和 RDMA 资源名也需要实际集群支持。

### 明确各配置项的职责

以下字段位于 `servingEngineSpec.modelSpec` 下：

| 字段 | 职责 |
| --- | --- |
| `roles[]` | P/D 副本数、worker、设备资源和模型参数 |
| `pd.prefill`, `pd.decode` | 指定路由使用的角色名 |
| `pd.kvTransfer.connector` | 选择引擎传输连接器 |
| `pd.kvTransfer.routerType` | 选择对应路由协议 |
| `pd.kvTransfer.identity` | 预留 engine ID 和传输端口范围 |
| `unifiedcacheConfig` | 在 Prefill 侧启用并配置 UCM |
| `storage.unifiedcacheStorage` | 提供 UCM 挂载；挂载路径用于填充 `storage_backends` |

`MooncakeConnectorV1` 和 `MooncakeHybridConnector` 对应的路由器类型是 `mooncake`。UCM 配置有效时，Chart 在 Prefill 侧组合 `MultiConnector`，Decode 侧仍只保留传输 consumer。不要在 `roles[].vllmArgs` 中再添加一份 `--kv-transfer-config`，该参数由 Chart 管理。

`NixlConnector` 使用 `nixl` 路由器类型，但当前 Chart 不允许它与有效 UCM 配置组合。传输名称通过 Chart 校验，不代表所选 Ascend 镜像实现了该连接器；修改内置 Mooncake 配置前，先验证镜像的 connector 支持情况。

禁用 `unifiedcacheConfig.enabled` 会移除 UCM connector 及其管理的缓存挂载，同时保留 PD 传输。镜像的 `ENABLE_UCM_PATCH` 环境变量独立配置，进行基线比较时应显式记录。

### 分三个阶段验证

**服务与传输。**检查 Prefill/Decode 就绪状态和路由对象，再通过已安装的 kthena-router 网关发送冷请求。检查选中的角色和传输错误。Release 的引擎 Service 同时选择两个角色，用于监控，不是 PD 客户端入口。

**外部复用。**使用固定提示词填充 UCM 存储，确认写入完成，保留存储并重启服务进程，再重放该提示词。分别检查 Prefill 的 UCM 命中 token、成功加载活动和引擎内存命中。在这一拓扑中，Decode 不需要出现 UCM 命中。

**性能。**分别在禁用 UCM、UCM 冷缓存和外部缓存已预热的状态下重放同一批流量。保持传输配置、P/D 数量、输出长度和生成设置不变。采集失败、输出正确性、客户端 TTFT、TPOT、吞吐以及存储与传输耗时。

### 按职责边界排查

| 症状 | 首先检查的证据 |
| --- | --- |
| 角色未达到 Ready | 设备分配、模型挂载、运行时兼容性、集合通信初始化 |
| 引擎独立运行正常，但网关请求失败 | `ModelRoute`、角色标签、路由协议、传输地址与端口可达性 |
| 冷 PD 请求正常，但重复提示词没有 UCM 命中 | Prefill connector 组合、持久化阈值、缓存内容和 key 兼容性 |
| 命中增加，但 TTFT 没有改善 | 成功加载的延迟、节省的 Prefill 计算量、传输时间、路由器与引擎排队 |

多节点角色实例或 MoE 模型可继续阅读[PD 部署扩容](large-scale-ep.md)。不要在同一步同时更换模型、并行布局和缓存后端，因为解释结果时，它们分别需要不同的验证证据。
