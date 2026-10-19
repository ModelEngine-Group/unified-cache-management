# PD 部署扩容

本页在[传输连接器与 UCM 组合](distributed.md)的基础上扩展部署规模。请先建立一组正常工作的 P/D 实例；扩容后仍由传输连接器交接 KV。

扩展 PD 分离部署包含两个独立选择：运行多少个独立 Prefill/Decode 实例，以及每个实例如何将模型分布到设备和节点上。UCM 在所选引擎集成中提供前缀复用，不负责选择模型的专家划分或协调集合通信。

本页分别给出手工部署步骤和 Chart 拓扑配置，再说明评估性能所需的检查。

## 手工部署

这条路径恢复原 Ascend 部署脚本，与 Helm 独立。启动前准备匹配的 vLLM-Ascend、Mooncake、模型权重、集合通信网络及共享存储。`/vllm-workspace` 下的路径对应部署环境中的源码目录。这些脚本依赖对应引擎版本，本次文档恢复未在加速设备上重跑。

原 GLM-5.1-w4a8 部署使用八台 Atlas 800T A2，每台八张 Ascend 910B3：Prefill 在 192.168.10.1–4 上采用 DP4TP8，Decode 在 192.168.10.5–8 上采用 DP8TP4。原存储环境经 CE8875 连接 A800；在 `/mnt/test1` 提供共享文件系统。

### 启动 Mooncake 并准备配置

在 192.168.10.1 启动 master。每个引擎节点的工作目录保存 `mooncake.json`，Prefill 节点将 UCM YAML 保存为 `/etc/ucm/pd.yaml`。

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

### Prefill 脚本

在每个 Prefill 节点保存为 `prefill.sh`。

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

### Decode 脚本

在每个 Decode 节点保存为 `decode.sh`。Decode 使用 Mooncake consumer，不加载 UCM YAML。

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

### 启动 DP 进程

将启动器保存为 `run_multi_dp.sh`，按节点填写 `local_ip`、`nic_name`、`dp_rank_start` 和 `dp_address`，保持原 DP/TP 数量及端口布局。

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

Prefill 的 .1/.2/.3/.4 节点 rank 分别为 0/1/2/3。Decode 节点使用下面的启动器；.5/.6/.7/.8 起始 rank 分别为 0/2/4/6。

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

### 启动代理并发送请求

```bash
python /vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
    --port 7850 \
    --host 0.0.0.0 \
    --prefiller-hosts 192.168.10.1 192.168.10.2 192.168.10.3 192.168.10.4 \
    --prefiller-ports 9000 9000 9000 9000 \
    --decoder-hosts 192.168.10.5 192.168.10.5 192.168.10.6 192.168.10.6 192.168.10.7 192.168.10.7 192.168.10.8 192.168.10.8 \
    --decoder-ports 9000 9001 9000 9001 9000 9001 9000 9001
```

沿用共享存储指南的 completions 请求格式，使用此处的代理端口和 GLM 模型名：

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

## Helm 部署

下面的 Chart 路径独立于上面的手工脚本，内置模型配置使用各自的拓扑与参数。

### 修改参数前先描述拓扑

三个副本设置的含义不同：

| 设置 | 含义 |
| --- | --- |
| `modelSpec.replicas` | 独立 serving group 的数量 |
| `roles[].replicas` | 每个 group 内该角色的逻辑实例数量 |
| `roles[].workerReplicas` | 每个逻辑实例额外包含的 worker Pod 数量 |

一个 serving group 的引擎 Pod 总数，等于各角色 `role replicas × (1 + workerReplicas)` 之和。设备数量再由每个 Pod 的资源请求决定。这里不包含路由器、Mooncake master 或其他控制面 Pod。

例如，内置 `values-qwen3-0p6b-2p2-2d2.yaml` 定义两个 Prefill 实例和两个 Decode 实例，每个实例带一个 worker，因此每个 serving group 请求八个引擎 Pod，而不是四个。默认按 hostname 设置的反亲和也会影响所需的可调度节点数量。应查看渲染后的放置规则，不要从配置文件名推断节点数。

### 由引擎管理 DP、TP 和 EP

张量并行、数据并行和专家并行都是引擎执行设置。将引擎支持的模型与并行参数放到各角色的 `vllmArgs` 中，并确认该引擎版本接受目标组合。多节点 DP 的全局与本地 DP 大小，必须与实际分配给角色实例的 Pod 和设备数匹配。

Chart 管理 HTTP 绑定、DP 地址与 rank 协调、对外模型名以及 KV 传输参数。不要重复添加这些参数，也不要把另一套多进程启动脚本粘贴到配置里。沿用现有入口，确保渲染配置与实际启动的进程对应。

仓库包含 DeepSeek-V3.1 和 Qwen3-235B 的 Ascend 多节点配置，但这些文件定义的是单角色服务布局。它们可以用于参考模型参数结构，不是开箱即用的 EP PD 配方。面向具体模型的 EP 部署，仍然需要验证过的引擎配置、集合通信网络和足够的目标硬件内存。

把 P 和 D 改为不同的 TP 或 EP 布局，也会改变传输兼容性要求。需要验证所选传输连接器支持的布局转换。对于共享存储 PD 分离部署，常规 UCM key 包含 TP 大小和 rank，不能假定新布局可以复用旧布局的块。

### 选择 HTTP 服务模式

Chart 的 `dataParallelMode` 控制多节点角色实例如何暴露 HTTP 端点：

- `standard`：一个入口 HTTP 端点，worker 为 headless；`ModelServer` 选择入口 Pod。
- `hybrid`：多节点角色的入口和 worker 节点都暴露 HTTP，路由器选择这些端点。该模式要求启用 PD 和路由器，并且至少一个角色包含 worker。

Chart 会注入对应的 vLLM LB 参数。不要独立设置 `--data-parallel-hybrid-lb`、`--headless` 或 Chart 管理的 rank 参数。应同时核对最终命令和 `ModelServer.workloadSelector`；健康但未被 selector 选中的 worker 不会接收到路由器流量。

### 预留传输与缓存资源

每个逻辑 P/D 实例获得独立的引擎标识，其入口和 worker 根据 serving-group 与角色标签解析出同一标识。Mooncake 的 `instanceStride` 必须覆盖配置中 DP、TP、PP 和上下文并行所需的端口跨度。Chart 会检查保守边界，并拒绝超过 65535 的端口范围。

集合通信网络、KV 传输和外部存储应分别计算主机及网络资源开销。对于 UCM，需要按进程规划主机缓冲，并为需要共享前缀的 Prefill 实例提供可达存储。当前 Chart 的 Decode 路径仅使用传输 consumer，因此增加 Decode 副本不会增加 UCM 读取者。

按 [Kubernetes 部署](../../frameworks/kubernetes/deploy.md)渲染站点 values，检查生成的 `ModelServing`、`ModelServer` 和 `ModelRoute`。渲染只能验证配置约束，不会在硬件上测试集合通信、传输吞吐或专家放置。

### 每次验证一个变化

1. 在一个 P 和一个 D 上建立输出正确的冷请求基线。
2. 增加所需 worker 和引擎并行配置，保持 UCM 后端与负载不变，确认 rank 初始化和请求传输。
3. 保留存储并重启引擎进程后，验证 Prefill 侧外部复用；将 UCM 命中和成功加载与 HBM 命中分开检查。
4. 增加角色副本或 serving group，确认每个新 Prefill 实例的路由分布、标识唯一性和存储访问。
5. 前一布局正常后，再进行 EP 专项调优；保留足够结果，将差异归因到当前变化。

输入/输出长度分布和请求到达计划应能代表目标负载。报告完成请求数、正确性失败、客户端 TTFT、TPOT、吞吐和资源消耗。如果平均延迟下降的同时失败更多，或者输出 token 更少，就不能视为等价比较。

### 解读结果

Prefill UCM 命中率很高，端到端 TTFT 仍可能很差，原因可能是路由器排队、存储读取慢，或传输在等待 Decode。同样，增加专家并行容量可能提高计算利用率，却增加通信开销。应结合分阶段计时和端点指标确定哪个阶段发生了变化，不能把所有加速都归因于 UCM。
