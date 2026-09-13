# Mooncake Store

Mooncake 流水线将可复用 KV 块放入共享内存服务。UCM 连接已有 Mooncake master，通过客户端传输数据。如果 KV 块还需要文件系统后备存储层，可以再组合 Posix。

| 流水线 | 读写行为 |
| --- | --- |
| `Mooncake` | 读写 Mooncake，不包含 UCM 文件系统后备存储层。 |
| `Mooncake|Posix` | 优先读取 Mooncake，未命中时交给 Posix；dump 同时写入 Posix。 |

这与首阶段为本地主机缓冲的 [Cache|Posix](pipeline.md) 不同。推理进程重启后，仍在运行的 Mooncake 服务可以保留内存对象，但这不保证服务故障后数据仍然持久化。`Mooncake|Posix` 的持久化范围取决于配置的文件系统，以及已经完成的后备存储写入。

## 前提条件

当前源码树中的原生 UCM Mooncake 目标需要 Ascend ACL 头文件、`libascendcl` 和 `libmooncake_store`。缺少其中任何一项，CMake 都会跳过该目标。这是一条 Ascend 集成路径；更换 Mooncake 传输名称不会让这个 UCM 目标成为 CUDA 后端。

准备 Mooncake 后，在匹配的 [Ascend 环境](../build_from_source.md#vllm-ascend-ascend-platform)中构建 UCM。CMake 在 `/usr/local/Ascend/ascend-toolkit/latest` 下的 `include` 和 `lib64` 目录搜索 toolkit 依赖，并在 `/usr/local/lib` 搜索 Mooncake 库。`MOONCAKE_STORE_INCLUDE_DIR` 指定 Mooncake 头文件树，当前回退路径为 `/vllm-workspace/Mooncake/mooncake-store/include`。头文件修订必须与链接的客户端库匹配。

启动服务前，确认已安装 `ucm/store/mooncakestore/libmooncakestore.so` 及其依赖。按部署环境的服务配置启动 Mooncake master，验证每个服务进程到 master 的连通性；选择后备存储层时，还需准备可写的 Posix 目录。UCM 不会代为启动 master。

## 启动 Mooncake master

在已经安装匹配 Mooncake 二进制的服务主机上执行：

```bash
mooncake_master --port 50088 --eviction_high_watermark_ratio 0.9 --eviction_ratio 0.1 --default_kv_lease_ttl 60000
```

保持服务运行，将下面的 `master_server_address` 改为这台主机可访问的 IP 与 `50088` 端口。已有 master 时直接使用其地址。容量、租约和回收参数以所安装版本的 `mooncake_master --help` 为准；客户端和服务端需使用兼容版本。

## 配置前缀缓存

下面是用于带 Ascend 设备、且能访问 Mooncake master 的主机的初始配置。请替换两个地址和存储目录：

```yaml
ucm_connectors:
  - ucm_connector_name: "UcmPipelineStore"
    ucm_connector_config:
      store_pipeline: "Mooncake|Posix"
      store_health:
        enabled: true
      local_hostname: "127.0.0.1"
      master_server_address: "127.0.0.1:50088"
      metadata_server: "P2PHANDSHAKE"
      protocol: "ascend"
      global_segment_size_gb: 30
      replica_num: 1
      storage_backends: "/mnt/ucm-mooncake"
      io_direct: true
      posix_io_engine: "aio"
      share_buffer_capacity_gb: 64

enable_event_sync: true
use_layerwise: false
```

`local_hostname` 为必填项，需要按传输方式正确标识服务主机。只有相关服务都在同一主机或网络命名空间内时，才适合使用回环地址。保留 `enable_event_sync: true`：原生 dump 路径在访问新计算的 KV 之前，会等待前置事件。按 [vLLM-Ascend 快速开始](../../user-guide/quick_start/index.md#vllm-ascend)，通过 `UCM_CONFIG_FILE` 传入此文件并启动模型服务。

两套配置都使用 V1 工厂注册的 `UcmPipelineStore`。

### 仅使用 Mooncake 内存池

```yaml
ucm_connectors:
  - ucm_connector_name: "UcmPipelineStore"
    ucm_connector_config:
      store_pipeline: "Mooncake"
      store_health:
        enabled: true
      local_hostname: "127.0.0.1"
      master_server_address: "127.0.0.1:50088"
      metadata_server: "P2PHANDSHAKE"
      protocol: "ascend"
      global_segment_size_gb: 30
      replica_num: 1
      share_buffer_capacity_gb: 64

enable_event_sync: true
use_layerwise: false
```

## 规划内存与并发

配置沿用原示例的 30 GiB 全局段和 64 GiB 共享缓冲，两者是不同的分配。未覆盖本地缓冲容量，其原生默认值为 1 GiB。正数 `_gb` 设置优先于对应的字节值设置。

`stream_number` 默认是 4，取值必须在 1 到 32 之间。如果没有显式设置 `host_buf_pool_size`，私有主机缓冲池大小由 stream 数量和张量大小推导。`replica_num` 默认是 1，必须为正数；副本数量不是持久化策略。

vLLM 适配层和 Mooncake 阶段使用不同的共享缓冲设置：`share_buffer_capacity_gb` 属于 Mooncake。对于适配层启用共享缓冲的模型，当前预检查仍读取 `cache_buffer_capacity_gb`，省略时会使用 128 GiB 进行检查；这不决定 Mooncake 缓冲大小。部署前按实际生效配置核对共享内存。示例不额外覆盖该检查容量。

## 原 GLM 部署命令

原 GLM-4.7-W8A8 示例使用两台 Ascend 节点，每台八张 NPU，TP8、每节点一个 DP 进程，总 DP2。保持原 32768-token 上下文及模型特定设置。先准备匹配的引擎、模型和驱动环境；这些脚本不代表兼容更新的 vLLM-Ascend。两侧将上方任选一套 UCM 配置保存为 `/etc/ucm/ucm.yaml`，将 `x.x.x.x` 换成首节点的可访问 IP。`local_hostname` 应标识各自节点，`master_server_address` 应能访问 master。

### 节点 1

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:/vllm-workspace/vllm
export PYTHONHASHSEED=0
export ACL_OP_INIT_MODE=1
export HCCL_RDMA_TIMEOUT=17
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000
export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_MLAPO=1
export HCCL_INTRA_PCIE_ENABLE=0
export HCCL_INTRA_ROCE_ENABLE=1
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export VLLM_SERVER_DEV_MODE=1
export VLLM_USE_DEEP_GEMM=0
export VLLM_LOGGING_LEVEL=INFO
export ENABLE_UCM_PATCH=1
export ENABLE_SPARSE=FALSE
export VLLM_HASH_ATTENTION=0
export VLLM_CPU_AFFINITY=0
export UC_LOGGER_LEVEL=info

vllm serve /model/GLM-4.7-W8A8/ \
    --max-model-len 32768 \
    --tensor-parallel-size 8 \
    --data-parallel-size 2 \
    --data-parallel-size-local 1 \
    --data-parallel-start-rank 0 \
    --data-parallel-address x.x.x.x \
    --data-parallel-rpc-port 13389 \
    --pipeline-parallel-size 1 \
    --gpu-memory-utilization 0.88 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 7800 \
    --block-size 128 \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 20 \
    --seed 1024 \
    --quantization ascend \
    --served-model-name GLM-4.7-W8A8 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --enable-expert-parallel \
    --additional-config '{"ascend_scheduler_config":{"enabled":false}}' \
    --kv-transfer-config '{
        "kv_connector":"UCMConnector",
        "kv_connector_module_path":"ucm.integration.vllm.ucm_connector",
        "kv_role":"kv_both",
        "kv_connector_extra_config":{
            "UCM_CONFIG_FILE":"/etc/ucm/ucm.yaml"
        }
    }'
```

### 节点 2

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:/vllm-workspace/vllm
export PYTHONHASHSEED=0
export ACL_OP_INIT_MODE=1
export HCCL_RDMA_TIMEOUT=17
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000
export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_MLAPO=1
export HCCL_INTRA_PCIE_ENABLE=0
export HCCL_INTRA_ROCE_ENABLE=1
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export VLLM_SERVER_DEV_MODE=1
export VLLM_USE_DEEP_GEMM=0
export VLLM_LOGGING_LEVEL=INFO
export ENABLE_UCM_PATCH=1
export ENABLE_SPARSE=FALSE
export VLLM_HASH_ATTENTION=0
export VLLM_CPU_AFFINITY=0
export UC_LOGGER_LEVEL=info

vllm serve /model/GLM-4.7-W8A8/ \
    --max-model-len 32768 \
    --tensor-parallel-size 8 \
    --data-parallel-size 2 \
    --data-parallel-size-local 1 \
    --data-parallel-start-rank 1 \
    --data-parallel-address x.x.x.x \
    --data-parallel-rpc-port 13389 \
    --pipeline-parallel-size 1 \
    --gpu-memory-utilization 0.88 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 7800 \
    --headless \
    --block-size 128 \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 20 \
    --seed 1024 \
    --quantization ascend \
    --served-model-name GLM-4.7-W8A8 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --enable-expert-parallel \
    --additional-config '{"ascend_scheduler_config":{"enabled":false}}' \
    --kv-transfer-config '{
        "kv_connector":"UCMConnector",
        "kv_connector_module_path":"ucm.integration.vllm.ucm_connector",
        "kv_role":"kv_both",
        "kv_connector_extra_config":{
            "UCM_CONFIG_FILE":"/etc/ucm/ucm.yaml"
        }
    }'
```

服务就绪后，在另一终端运行原请求负载。这里只恢复测试输入，不表示复现了原性能结果：

```bash
vllm bench serve \
--backend vllm \
--model GLM-4.7-W8A8 \
--host 127.0.0.1 \
--port 7800 \
--dataset-name random \
--num-prompts 12 \
--random-input-len 16000 \
--random-output-len 2 \
--request-rate inf \
--seed 123456 \
--percentile-metrics "ttft,tpot,itl,e2el" \
--metric-percentiles "90,99" \
--ignore-eos
```

## 分别验证每一层

1. 记录模型修订、dtype、并行布局和 Mooncake 命名空间，运行新前缀请求并等待 dump 完成。
2. 保持 Mooncake 服务运行，通过重启后的推理进程重放。必须观察到外部命中和 Mooncake 读取、命中指标。
3. 对于 `Mooncake|Posix`，确认后备文件存在且 Posix dump 已完成。要测试后备读取，在隔离测试环境中只驱逐 Mooncake 内的测试对象，保留文件后再次重放。
4. 最后一种情况必须观察到 Posix 读取和 Mooncake backend-load 指标。仅 Mooncake 命中不能证明文件系统层正常。

加载命中、未命中、后端和字节计数器见[指标](../../user-guide/observability/metrics.md)，主动探针见[健康指标](../../user-guide/observability/health-metrics.md)。Mooncake 健康检查执行小规模写入、读取、删除操作，不能替代双层请求测试。

## 定位故障所在环节

| 症状 | 下一步检查 |
| --- | --- |
| 缺少原生库或存在未解析符号 | 可选 CMake 目标、Ascend 库，以及匹配的 Mooncake 头文件和库。 |
| Setup 或 lookup 无法访问服务 | Master 地址、local hostname、传输配置和服务日志。 |
| 预期有 Posix 命中，但实际没有 | 后备写入是否完成、存储路径是否匹配，以及 Posix 健康状态。 |
| 队列拒绝或加载延迟持续增加 | 队列与阶段指标、主机缓冲需求、服务容量和网络流量。 |

性能比较需要区分命中的具体层。分别报告内存服务命中和文件系统命中；不能因为启用了这条流水线就预设会有加速。

[实现与扩展说明](../extending-store.md#backend-entrypoints)。
