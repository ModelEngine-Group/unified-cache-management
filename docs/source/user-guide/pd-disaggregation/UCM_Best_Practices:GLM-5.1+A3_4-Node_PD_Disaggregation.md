# 概述

**[UCM](https://github.com/ModelEngine-Group/unified-cache-management)**:统一缓存管理器（Unified Cache Management, UCM）的核心原理是持久化 LLM 的 KVCache，通过外部KVCache池缓存更多的可命中的前缀，降低TTFT以及提高吞吐。

**GLM-5.1** 是智谱旗舰模型，代码能力大大增强，**长程任务**显著提升，能够在单次任务中持续、自主地工作长达 8 小时，完成从规划、执行到迭代优化的完整闭环，交付工程级成果。使用UCM作为vLLM KV Pool后端，可以实现高效的KV Cache存储和请求间复用，从而大幅优化长文本、高复用（prefix cache）场景下的TTFT和吞吐水平。

## 一、性能测试总结

测试设计：

1. 输入输出长度：64K/1k，128K/1k
2. TPOT时延要求：卡30ms
3. prefix cache命中率设置：90%
4. kv cache前缀缓存种类：取并发值的1/2，模拟用户agent调用时，同时发送的多个请求之间有高度相似的请求前缀。
5. 测试并发量：10，20，30
6. 请求数量：并发量*4

在4机A3集群（每机8×910C，每张910C为双910B合封）上，基于GLM5.1 W8A8量化模型、PD分离架构（2 Prefill + 2 Decode），对ucm前缀缓存进行了4组场景的性能测试。测试数据如下：

| 场景         | InputLen | OutputLen | 前缀数量 | 请求数量 | 并发 | 实际并发 | RR | 有无缓存 | TTFT平均(ms） | TTFT降幅 | TPOT平均(ms) | 吞吐(token/s) | 吞吐提升 |
| -------------- | ---------- | ----------- | ---------- | ---------- | ------ | ---------- | ---- | ---------- | --------------- | ---------- | -------------- | --------------- | ---------- |
| 64K × 20CC  |    65536 |      1024 |       10 |       80 |   20 | 17.64    |  0 | 无       | 36,786        | /        | 20.2         | 314.1         | /        |
|              |    65536 |      1024 |       10 |       80 |   20 | 18.80    |  0 |      0.9 | 7,854         |    78.6% | 23.5         | 604.4         |    92.4% |
| 64K × 30CC  |    65536 |      1024 |       15 |      120 |   30 | 26.22    |  0 | 无       | 63,493        | /        | 20.5         | 318.1         | /        |
|              |    65536 |      1024 |       15 |      120 |   30 | 27.63    |  0 |      0.9 | 11,372        |    82.1% | 25.0         | 766.7         |   141.0% |
| 128K × 10CC |   131072 |      1024 |        5 |       40 |   10 | 9.18     |  0 | 无       | 46,967        | /        | 20.2         | 139.1         | /        |
|              |   131072 |      1024 |        5 |       40 |   10 | 9.20     |  0 |      0.9 | 15,942        |    66.1% | 20.5         | 255.2         |    83.5% |
| 128K × 20CC |   131072 |      1024 |       10 |       80 |   20 | 17.86    |  0 | 无       | 107,334       | /        | 19.7         | 143.5         | /        |
|              |   131072 |      1024 |       10 |       80 |   20 | 17.97    |  0 |      0.9 | 51,370        |    52.1% | 20.3         | 255.2         |    77.8% |

### 结论：

1、64K 上下文：提升并发可放大缓存收益，20 并发升至 30 并发后吞吐提升至 141%。
2、20 并发下，64K 为甜点长度，TTFT下降78.6%，吞吐提升92.4%。当请求长度超过 HBM 容量阈值（约235,008 / (20/8)=94K Tokens）后，受限于 HBM 空间不足，TTFT 降幅与吞吐提升会出现衰减。部署建议：在固定 20 并发的场景下，建议请求长度严格控制在 94K Tokens 以内，以保障最佳性能。
3、128K 超长上下文：并发 8 为甜点并发，超过该阈值后缓存收益递减。在并发 10 时，TTFT 依然能下降 66.1%，吞吐提升 83.5%；当并发从 10 升至 20 时，TTFT 降幅与吞吐提升同步下降。部署建议：推理 128K 超长上下文请求时，建议并发数不超过 8，以避免触发底层抢占机制。

注释：当前 Decode 实例单 DP 的 GPU KV Cache Size 为 235,008 Tokens。每个 DP 仅能承载单并发 128K + 1K 的请求，8 个 DP 实例的绝对并发上限为 8。在 vLLM 推理时，必须保证一个 Batch 内所有请求的 KV Cache 能被 HBM 完全容纳。若显存空间不足，系统会触发请求抢占机制：强制释放其他请求占用的显存空间；当这些请求再次被调度时，又需要重新进行 Decode 计算。如此往复的抢占与重算开销，导致了高并发下的吞吐性能衰减。此外，当 Decode 实例的请求量过多导致排队时，会直接增加请求的等待时间，进而导致首 Token 延迟（TTFT）性能衰减。因此，超出硬件物理边界的对比数据，无法真实反映缓存机制本身的性能表现。

## 二、测试环境概览

| 项目        | 规格                                                     |
| ----------- | -------------------------------------------------------- |
| 集群规模    | 4机A3（每机8×910C，每张910C为两张910B双芯合封）         |
| 缓存         | HBM+SSD：OceanDisk 1300（prefill节点共享4T），无DRAM                                  |
| 模型        | GLM-5.1-w8a8（W8A8量化，GLM5.1）                   |
| 框架版本    | vllm-ascend 0.18.0rc1 + ucm v0.17.0                      |
| PD分离架构  | 1x2 Prefill + 1x2 Decode                                     |
| Prefill配置 | DP=4, TP=8, EP=32            |
| Decode配置  | DP=8, TP=4, EP=32 |
| KV传输      | P节点：MultiConnector（Mooncake + UCM 双通道）  D节点： MooncakeConnectorV1               |

## 三、部署指南：

### 

### 1.拉取vllm-ascend官方镜像

```
docker pull quay.io/ascend/vllm-ascend:v0.18.0rc1-a3
```

容器启动脚本：

```
#!/bin/bash
IMAGES_ID="$1"
NAME="$2"

# 检查参数数量（需要2个：镜像ID和容器名）
if [ $# -ne 2 ]; then
    echo "error: 需要传入2个参数，格式：$0 <镜像ID> <容器名>"
    exit 1
fi

# 检查镜像是否存在
if ! docker images --format "{{.ID}}" | grep -q "^${IMAGES_ID:0:12}$"; then
    echo "error: 镜像ID $IMAGES_ID 不存在"
    exit 1
fi

# 启动容器（变量加引号，避免特殊字符问题）
docker run --name "${NAME}" -it -d --net=host --shm-size=500g \
    --privileged=true \
    -w /home \
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --device=/dev/devmm_svm \
    --entrypoint=bash \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /usr/local/sbin:/usr/local/sbin \
    -v /etc/hccn.conf:/etc/hccn.conf \
    -v /home:/home \
    -v /mnt:/mnt \
    -v /tmp:/tmp \
    -v /data:/data \
    -v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
    -e http_proxy="$http_proxy" \
    -e https_proxy="$https_proxy" \
    "${IMAGES_ID}"
#执行以下命令创建容器
#bash start-docker.sh ac1c767e5aa2 glm5-v0.18.0rc1-a3
#执行以下命令进入创建完成的容器
#docker exec -it glm5-v0.18.0rc1-a3 bash
```

### 2.安装UCM

进入容器后

```
git clone --depth 1 --branch develop https://github.com/ModelEngine-Group/unified-cache-management.git
cd unified-cache-management
export PLATFORM=ascend-a3
pip install -v -e . --no-build-isolation
cd ..
```

**注意:** Atlas A2 系列, 参数PLATFORM=ascend.
`通过环境变量导入补丁使能UCM`

```
export ENABLE_UCM_PATCH=1
```

### 3.UCM脚本配置

修改 UCM 配置文件`ucm_config_example.yaml`，指定要使用的 UCM 连接器以及 KV 数据块的存储位置。该文件可放在任意目录下，只要在拉起服务的脚本中配置对应路径即可

```
ucm_connectors:
  - ucm_connector_name: "UcmPipelineStore"
    ucm_connector_config:
      store_pipeline: "Cache|Posix"
      storage_backends: "/mnt/test"  
      io_direct: true
      posix_io_engine: "aio"
      cache_buffer_capacity_gb: 96 #参考下面推荐配置
      posix_capacity_gb: 1024  
enable_event_sync: true
use_layerwise: true                   
enable_record_traces: false
use_lite: false
persist_token_threshold: 0
```

#### **参数说明：**

##### 必需参数

* **ucm_connector_name**：指定UcmPipelineStore UCM 连接器。
* **store_pipeline：“Cache|Posix”：**指定一个由缓存存储和 Posix 存储链接而成的管道。
* **storage_backends**：用于存储键值块的目录。可以是本地路径或 NFS 挂载路径。理论上缓存容量上限取决于存储目录的可用空间⚠️  **请替换"/mnt/test"为您的实际存储目录。**

##### 可选参数

* **io_direct***（可选，默认值false）*：
  是否启用直接 I/O。对于 Posix 存储，启用后在读写磁盘文件时会绕过操作系统页面缓存。
* **cache_buffer_capacity_gb ***（可选，默认值：256）*
  GQA模型（Qwen3,GLM4.7等）：默认每张卡32GB，如配置就是每张卡占的DRAM内存
  MLA模型（DeepSeek V3/R1, GLM-5等）：192/单机dp数量
  DeepSeek V4:如单台A3采用TP8DP2，推荐为48。单台A2采用TP8DP2，推荐为96
* **posix_capacity_gb ***（可选，默认值：0）*
  POSIX 存储的最大容量（以 GB 为单位）。
  当设置为大于 0 的值时，将启用垃圾回收 (GC)，并在达到阈值时回收磁盘空间。
  当设置为 0（默认值）时，不应用容量限制或 GC。
* **posix_io_engine ***（可选，默认值：“psync”）*
  Posix 存储的 I/O 引擎类型。支持的值："psync"（pread/pwrite）、"aio"（libaio）。
* **enable_event_sync **​*（可选，默认值：true）*
  是否启用事件同步。
* **use_layerwise **​*（可选，默认值：true）*
  是否使用逐层加载/保存 KV 缓存块。
* **enable_record_traces **​*（可选，默认值：false）*
  是否记录请求跟踪信息。
  启用后，将记录每个请求的跟踪信息（时间戳、输入长度、输出长度、哈希 ID）。
* **use_lite **​*（可选，默认值：false）*
  是否使用 UCM Lite 连接器。
* **persist_token_threshold **​*（可选，默认值：0）*
  键值持久化的最小令牌阈值。

更多配置可以参考UCM官方文档：

1、[https://ucm.readthedocs.io/en/latest/](https://ucm.readthedocs.io/en/latest/ "Welcome to Unified Cache Manager — Unified Cache Manager")
2、[https://docs.vllm.ai/projects/ascend/en/latest/user\_guide/feature\_guide/ucm\_deployment.html](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/ucm_deployment.html)
3、[https://deepwiki.com/ModelEngine-Group/unified-cache-management/1-overview](https://deepwiki.com/ModelEngine-Group/unified-cache-management/1-overview "deepwiki.com")

### 4.PD分离部署指南

#### 选择一个节点，启动mooncake主服务：

```
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
mooncake_master --port 50088 --eviction_high_watermark_ratio 0.9 --eviction_ratio 0.1 --default_kv_lease_ttl 11000
```

每个节点也需要一个`ucm_config_example.yaml`（参考UCM脚本配置章节）和`mooncake.json`（服务启动脚本时通过MOONCAKE_CONFIG_PATH环境变量导入）文件。如下（本文采用100.100.123.166作为mooncake主服务节点）：

```
{
    "metadata_server": "P2PHANDSHAKE",
    "protocol": "ascend",
    "device_name": "",
    "master_server_address": "100.100.123.166:50088",
    "global_segment_size": "1GB"
}
```

#### 所有节点准备launch_online_dp.py脚本

```
import argparse
import multiprocessing
import os
import subprocess
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dp-size",
        type=int,
        required=True,
        help="Data parallel size."
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallel size."
    )
    parser.add_argument(
        "--dp-size-local",
        type=int,
        default=-1,
        help="Local data parallel size."
    )
    parser.add_argument(
        "--dp-rank-start",
        type=int,
        default=0,
        help="Starting rank for data parallel."
    )
    parser.add_argument(
        "--dp-address",
        type=str,
        required=True,
        help="IP address for data parallel master node."
    )
    parser.add_argument(
        "--dp-rpc-port",
        type=str,
        default=12345,
        help="Port for data parallel master node."
    )
    parser.add_argument(
        "--vllm-start-port",
        type=int,
        default=9000,
        help="Starting port for the engine."
    )
    return parser.parse_args()

args = parse_args()
dp_size = args.dp_size
tp_size = args.tp_size
dp_size_local = args.dp_size_local
if dp_size_local == -1:
    dp_size_local = dp_size
dp_rank_start = args.dp_rank_start
dp_address = args.dp_address
dp_rpc_port = args.dp_rpc_port
vllm_start_port = args.vllm_start_port

def run_command(visible_devices, dp_rank, vllm_engine_port):
    command = [
        "bash",
        "./run_dp_template.sh",
        visible_devices,
        str(vllm_engine_port),
        str(dp_size),
        str(dp_rank),
        dp_address,
        dp_rpc_port,
        str(tp_size),
    ]
    subprocess.run(command, check=True)

if __name__ == "__main__":
    template_path = "./run_dp_template.sh"
    if not os.path.exists(template_path):
        print(f"Template file {template_path} does not exist.")
        sys.exit(1)

    processes = []
    num_cards = dp_size_local * tp_size
    for i in range(dp_size_local):
        dp_rank = dp_rank_start + i
        vllm_engine_port = vllm_start_port + i
        visible_devices = ",".join(str(x) for x in range(i * tp_size, (i + 1) * tp_size))
        process = multiprocessing.Process(target=run_command,
                                        args=(visible_devices, dp_rank,
                                                vllm_engine_port))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()


```

#### 各个节点的启动脚本

P0节点：

```
nic_name="eth-bond4" # change to your own nic name
local_ip="100.100.123.166" # change to your own ip
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export PYTHONHASHSEED=0
export PYTHONPATH=$PYTHONPATH:/vllm-workspace/vllm
export MOONCAKE_CONFIG_PATH="./mooncake.json"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=256

export ASCEND_AGGREGATE_ENABLE=1
export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export VLLM_NIXL_ABORT_REQUEST_TIMEOUT=300000
export ENABLE_UCM_PATCH=1
export ASCEND_RT_VISIBLE_DEVICES=$1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

vllm serve /data/0416-GLM-5.1-w8a8/ \
    --host 0.0.0.0 \
    --port $2 \
    --data-parallel-size $3 \
    --data-parallel-rank $4 \
    --data-parallel-address $5 \
    --data-parallel-rpc-port $6 \
    --tensor-parallel-size $7 \
    --enable-expert-parallel \
    --speculative-config '{"num_speculative_tokens": 3, "method":"deepseek_mtp"}' \
    --seed 1024 \
    --served-model-name glm-5.1 \
    --max-model-len 202752 \
    --additional-config '{"enable_npugraph_ex": true, "fuse_muls_add":true,"multistream_overlap_shared_expert":true,"recompute_scheduler_enable" : true}' \
    --max-num-batched-tokens 4096 \
    --trust-remote-code \
    --max-num-seqs 64 \
    --quantization ascend \
    --gpu-memory-utilization 0.95 \
    --enforce-eager \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --kv-transfer-config \
    '{
        "kv_connector": "MultiConnector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {
            "connectors": [
                {
                    "kv_connector": "MooncakeConnectorV1",
                    "kv_role": "kv_producer",
                    "kv_port": '30000',
                    "kv_connector_extra_config": {
                        "prefill": {
                            "dp_size": 4,
                            "tp_size": 8
                        },
                        "decode": {
                            "dp_size": 8,
                            "tp_size": 4
                        }
                    }
                },
                {
                    "kv_connector": "UCMConnector",
                    "kv_role": "kv_both",
                    "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
                    "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/mnt/cephfs/huangchengmin/ucm/ucm_config_example.yaml"}
                }
            ]
        }
    }'

```

**需要根据环境做相应修改的包括：**

* local_ip：本机IP地址
* nic_name：本地网卡名称
* 权重路径

D0节点：

```
nic_name="bond0" # change to your own nic name
local_ip="100.100.123.148" # change to your own ip
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export HCCL_OP_EXPANSION_MODE="AIV"
export MOONCAKE_CONFIG_PATH="./mooncake.json"

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name

export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=256


export ASCEND_AGGREGATE_ENABLE=1
export ASCEND_TRANSPORT_PRINT=1
export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export VLLM_NIXL_ABORT_REQUEST_TIMEOUT=300000

export TASK_QUEUE_ENABLE=1

export ASCEND_RT_VISIBLE_DEVICES=$1


export VLLM_ASCEND_ENABLE_MLAPO=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

vllm serve /data/0416-GLM-5.1-w8a8/ \
    --host 0.0.0.0 \
    --port $2 \
    --data-parallel-size $3 \
    --data-parallel-rank $4 \
    --data-parallel-address $5 \
    --data-parallel-rpc-port $6 \
    --tensor-parallel-size $7 \
    --enable-expert-parallel \
    --speculative-config '{"num_speculative_tokens": 3,  "method":"deepseek_mtp"}' \
    --seed 1024 \
    --served-model-name glm-5.1 \
    --max-model-len 202752 \
    --max-num-batched-tokens 32 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY", "cudagraph_capture_sizes":[4, 8, 12, 16,20,24,28, 32]}' \
    --additional-config '{"enable_npugraph_ex": true, "fuse_muls_add":true,"multistream_overlap_shared_expert":true,"recompute_scheduler_enable" : true}' \
    --trust-remote-code \
    --max-num-seqs 8 \
    --gpu-memory-utilization 0.92 \
    --async-scheduling \
    --quantization ascend \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --kv-transfer-config \
    '{"kv_connector": "MooncakeConnectorV1",
    "kv_role": "kv_consumer",
    "kv_port": "30100",
    "engine_id": "1",
    "kv_connector_extra_config": {
                "use_ascend_direct": true,
                "prefill": {
                        "dp_size": 4,
                        "tp_size": 8
                },
                "decode": {
                        "dp_size": 8,
                        "tp_size": 4
                }
        }
    }'

```

**需要根据环境做相应修改的包括：**

* local_ip：本机IP地址
* nic_name：本地网卡名称
* 权重路径

使用下列指令分别拉起各个节点的服务：

```
p0:nohup python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 2 --dp-rank-start 0 --dp-address 100.100.123.166 --dp-rpc-port 10521 --vllm-start-port 6789 > online_dp_launch.log 2>&1 &
p1:nohup python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 2 --dp-rank-start 2 --dp-address 100.100.123.166 --dp-rpc-port 10521 --vllm-start-port 6789 > online_dp_launch.log 2>&1 &
d0: nohup python launch_online_dp.py --dp-size 8 --tp-size 4 --dp-size-local 4 --dp-rank-start 0 --dp-address 100.100.123.148 --dp-rpc-port 10523 --vllm-start-port 6721 > online_dp_launch.log 2>&1 &
d1:nohup python launch_online_dp.py --dp-size 8 --tp-size 4 --dp-size-local 4 --dp-rank-start 4 --dp-address 100.100.123.148 --dp-rpc-port 10523 --vllm-start-port 6721 > online_dp_launch.log 2>&1 &
```

#### 在主节点准备proxy脚本load_balance_proxy_server_example.py：

从vllm-ascend仓库中获取proxy脚本[load\_balance\_proxy\_server\_example.py](https://link.gitcode.com/?target=https%3A%2F%2Fgithub.com%2Fvllm-project%2Fvllm-ascend%2Fblob%2Fmain%2Fexamples%2Fdisaggregated_prefill_v1%2Fload_balance_proxy_server_example.py&from=https%3A%2F%2Fgitcode.com%2FAscend%2Fmemcache%2Fwiki%2FMMC%25E6%259C%2580%25E4%25BD%25B3%25E5%25AE%259E%25E8%25B7%25B5%25E2%2580%2594GLM-5.1%2BA3-4%25E6%259C%25BAPD%25E5%2588%2586%25E7%25A6%25BB%25EF%25BC%2588%25E6%259C%2580%25E4%25BD%25B3%25E6%2580%25A7%25E8%2583%25BD%25E7%2589%2588%25EF%25BC%2589.md&lang=zh&theme=white)

使用如下脚本启动转发（需要修改对应ip、port为服务实际启动时所设置的ip、port）：

```
unset http_proxy
unset https_proxy
python load_balance_proxy_server_example.py \
    --port 1999 \
    --host 0.0.0.0 \
    --prefiller-hosts \
       "100.100.123.166" \
       "100.100.123.166" \
       "100.100.123.165" \
       "100.100.123.165" \
    --prefiller-ports \
       6789 6790\
       6789 6790\
    --decoder-hosts \
      "100.100.123.148" \
      "100.100.123.148" \
      "100.100.123.148" \
      "100.100.123.148" \
      "100.100.123.146" \
      "100.100.123.146" \
      "100.100.123.146" \
      "100.100.123.146" \
    --decoder-ports \
      6721 6722 6723 6724\
      6721 6722 6723 6724

```

## 四、测试指南

### 1.测试工具

安装以下工具https://github.com/rayn-zzz/aisbench_auto_tools_prefix

### 2.无前缀缓存命中测试

测试指令如下：

```
python3 aisbench_test.py --input_len 65560 --output_len 1024 --data_num 80 --concurrency 20 --repeat_rate 0 
python3 aisbench_test.py --input_len 65560 --output_len 1024 --data_num 120 --concurrency 30 --repeat_rate 0
python3 aisbench_test.py --input_len 131072 --output_len 1024 --data_num 40 --concurrency 10 --repeat_rate 0 
python3 aisbench_test.py --input_len 131072 --output_len 1024 --data_num 80 --concurrency 20 --repeat_rate 0
```

### 3.90%前缀缓存命中测试

测试命令如下：

```
python3 aisbench_test.py --input_len 65536 --output_len 1024 --data_num 80 --concurrency 20  --dataset_type prefix_cache --prefix_num 10 --repeat_rate 0.9 --prefix_test --dp 4
python3 aisbench_test.py --input_len 65560 --output_len 1024 --data_num 120 --concurrency 30  --dataset_type prefix_cache --prefix_num 15 --repeat_rate 0.9 --prefix_test --dp 4
python3 aisbench_test.py --input_len 131072 --output_len 1024 --data_num 40 --concurrency 10  --dataset_type prefix_cache --prefix_num 5 --repeat_rate 0.9 --prefix_test --dp 4
python3 aisbench_test.py --input_len 131072 --output_len 1024 --data_num 80 --concurrency 20  --dataset_type prefix_cache --prefix_num 10 --repeat_rate 0.9 --prefix_test --dp 4
```

注：为了保证每次测试数据之间没有缓存影响，需要清空缓存，使用命令 find /mnt/test -type f -delete，/mnt/test为上面config中配置的UCM缓存地址

