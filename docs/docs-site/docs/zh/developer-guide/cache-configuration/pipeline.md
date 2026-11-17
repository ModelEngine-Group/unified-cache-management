# Pipeline Store

需要让 KV 块在推理进程重启后继续从本地磁盘或挂载文件系统复用，同时将近期访问的块保存在主机内存中时，可以选择 `Cache|Posix`。`Cache` 负责设备与主机之间的数据搬运及主机缓冲区，`Posix` 负责文件、查询和文件系统 I/O；设备上的 KV Cache 仍由推理引擎管理。

通过 `UcmPipelineStore` 选择已注册的管线名称，完整可选值见[配置参考](../../reference/config-parameters.md)。

## 配置 Cache 与 Posix

先从[安装页面](../../user-guide/quick_start/index.md)选择制品，或在目标引擎环境中[从源码构建](../build_from_source.md)。这条管线需要 `ucmpipelinestore` 扩展、`libcachestore.so` 和 `libposixstore.so`。仅能导入 Python 包，还不能说明所选管线能够初始化。

准备独立且可写的目录，并将其挂载到需要访问它的进程中。将下面的配置保存为 UCM YAML 文件：

```yaml
ucm_connectors:
  - ucm_connector_name: "UcmPipelineStore"
    ucm_connector_config:
      store_pipeline: "Cache|Posix"
      store_health:
        enabled: true
      storage_backends: "/mnt/ucm-cache"
      io_direct: false
      # cache_buffer_capacity_gb: 256
      # posix_capacity_gb: 1024
      use_gdr: false
enable_event_sync: true
use_layerwise: true
enable_record_traces: false
use_lite: false
persist_token_threshold: 0
```

按 [vLLM 快速开始](../../user-guide/quick_start/index.md#vllm)或 [Ascend 快速开始](../../user-guide/quick_start/index.md#vllm-ascend)，通过 `kv_connector_extra_config.UCM_CONFIG_FILE` 指向该文件。Connector 会提供设备 ID、块大小和张量布局，不要复制其他模型 Store 测试中的这些数值。

示例沿用原配置的 `io_direct: false`，不额外覆盖 I/O 引擎。原生 Cache 和 Posix 阶段的 `io_direct` 默认值为 `true`。只有在设置 `io_direct: true`，且文件系统支持对应的对齐 I/O 时，才切换到 `posix_io_engine: aio`。`timeout_ms` 应放在 `ucm_connector_config` 内，它是 Store 任务超时，不是 HTTP 请求超时。

## 规划主机内存 { #budget-host-memory }

示例中的容量字段保持注释状态，因此使用运行时默认值；注释中的 256 不是对所有模型生效的固定容量。Cache Store 至少需要容纳 `max(1024, 2 * cache_load_exclusive_buffer_number)` 个 shard，独占缓冲数量默认是 1024。如果容量太小，初始化错误会给出至少需要多少 GiB。

| 未显式设置容量时的分配路径 | 实际默认值 |
| --- | --- |
| 原生 Cache，启用共享缓冲 | 256 GiB |
| 原生 Cache，`share_buffer_enable: false` | 每个 worker 32 GiB |
| 当前 vLLM Connector，启用共享缓冲 | 128 GiB |

vLLM Connector 根据模型是否使用 MLA 决定 `share_buffer_enable` 的默认值。正数 `cache_buffer_capacity_gb` 会覆盖原生默认值。共享缓冲需要足够的 `/dev/shm`；非共享 worker 分别分配内存，因此要按同一主机上的 worker 数量汇总预算。模型权重、设备 KV Cache 和其他主机内存开销需要另外计入。

## 存储容量与健康检查

`posix_capacity_gb: 0` 表示不启用基于容量的垃圾回收。正数值提供容量预算，但不会预留文件系统空间。vLLM Connector 将 Posix GC 交给 DP0 scheduler。多个推理实例共用目录时，需要统计它们共同写入的数据；没有验证其他回收归属方案之前，应保留 Posix 的协调设置。

管线健康检查默认开启，检查间隔为 10 秒、超时为 3 秒、窗口为 8 个样本、失败阈值为 2。Posix 探针实际执行写入、读取、比较和删除。探针成功说明文件系统健康，只有请求级命中和完成的加载才能说明 KV 被复用。详见[健康指标](../../user-guide/observability/health-metrics.md)。

## 流水线并行

当 `pipeline_parallel_size > 1` 时，分层 connector 会为每个 PP stage 分配独立的
store，而不是共用一个命名空间：

| | 示例（PP=2，后端 `/data/ucm`，engine id `3f9c1a`，共 8 层） |
| --- | --- |
| 存储路径 | `/data/ucm/pp_0_l0-4`、`/data/ucm/pp_1_l4-4` |
| Cache id | `3f9c1a_pp_0_l0-4`、`3f9c1a_pp_1_l4-4` |

stage 目录在 Posix store 打开之前创建，block key 的格式保持不变，因此各 stage
之间的 key 仍可直接比较。

**层分配是命名空间的一部分。** stage 目录同时编码 rank 与该 stage 负责的层
（`pp_<rank>_l<起始层>-<层数>`），层分配来自 `VLLM_PP_LAYER_PARTITION`。这不是
命名风格问题：只按 rank 命名时，重新分层后重启会读到错误的层。以 12 层为例，
`4,4,4` 下 stage 1 持有第 4~7 层，`2,4,6` 下 stage 1 持有第 2~5 层。两次运行中
stage 1 都是 4 层，`block_size` 与 `shard_size` 完全一致，旧文件可以正常打开，
于是 shard 0 从"第 4 层"变成"第 2 层"——第 2 层会静默读到第 4 层的 KV。把层分配
写进目录（以及 Cache id）后，旧数据直接不可达，从而避免这种错配。每个 worker 都
会通过握手发布自己的层分配，调度侧据此生成完全一致的目录名。

**预算拆分。** 一个节点上存在多个 stage 时，`cache_buffer_capacity_gb` 按节点内的
stage 数均分，`posix_capacity_gb` 按完整的 `pipeline_parallel_size` 均分。若均分结果
向下取整为 0，启动会直接报错，而不是静默回退到原生默认值。

**几何信息来自握手。** 每个 worker 通过握手发布本 stage 真实的 block 与 shard
大小，调度侧据此建立对应的 store。原先"单层大小 × 模型总层数"的估算方式已移除，
因为它会用该 stage 并不持有的层来放大 block 大小。

**shard 索引按 stage 局部编号。** 每个 stage 只保存自己的层，因此 shard 索引从 0
重新开始。例如 32 层、PP=2 时，第 16 层是 stage 1 的第 0 个 shard，而不是
第 16 个。

**只有所有 stage 都有该 block 才算命中。** 查找会对每个 stage 的 store 各做一次，
取最短公共前缀：PP0 命中 10 块而 PP1 只有 4 块时，命中就是 4 块，其余 6 块由 vLLM
重算。只有所有 stage 都能提供时才会被放进加载计划，因此某个 stage 的 dump 失败、
仍在进行中、stage 重启、或该 stage 自己的 GC 回收，都只会让命中变短，而不会产生
加载错误。代价是每个 stage 一次查找往返——后续 stage 保留纯 Posix store，跨节点时
这次查询会在调度关键路径上产生一次网络往返，但按"批"发生而不是按"块"。

**各 stage 的查询并行执行。** PP0 先查，因为完全未命中时它能单独定论，无需触碰远端
stage；其余 stage 再由一个 connector 自有的线程池并发查询，线程数等于 stage 数。该线程
池与进程同生命周期，因此指标注册表看到的线程数是固定的，不随查询次数增长。

**仍建议设置 `kv_load_failure_policy="recompute"`。** 交集消除了 PP0 与后续 stage
的不一致，但仍存在一个竞态：某个 stage 的 store 可能在命中判定之后、加载执行之前
把该 block 回收。vLLM 默认 `"fail"` 会让这种竞态直接结束请求，`"recompute"` 则退化
为部分复用。stage 级加载错误仍会通过 `get_block_ids_with_load_errors()` 上报，作为
兜底。

**适用范围。** 该能力仅覆盖分层 connector + 各 stage 均可访问的共享 Posix 后端。
非 Posix 后端与直连（非分层）connector 不在支持范围内，上下文并行与 PP 组合会在
启动时被拒绝。

**从平铺布局升级。** 旧版本写入的数据直接位于配置的根目录下，而不是 stage 子目录；
最早一版 PP 则写在 `pp_<rank>`（不带层后缀）下。升级后这两种布局都不会被命中，
新的 stage 命名空间也不会回收它们。确认没有其他进程仍在使用后，请手动清理。
另外，修改 `VLLM_PP_LAYER_PARTITION` 会生成一套新的目录，旧的 stage 目录同样会残留，
磁盘预算紧张时需要一并清理。

## 其他已注册管线

| 需求 | 选择 |
| --- | --- |
| 使用 3FS 客户端 I/O | [`Cache|Ds3fs`](ds3fs.md) |
| 缩小 BF16 存储载荷，并接受精度权衡 | [`Cache|Compress|Posix`](compress.md) |
| 使用共享 Mooncake 内存，可选文件后端 | [`Mooncake` 或 `Mooncake|Posix`](mooncake.md) |

SGLang 已经管理主机缓存，因此其适配器直接选择 `Posix` 阶段。请使用 [SGLang 快速开始](../../user-guide/quick_start/index.md#sglang)，不要直接套用 vLLM YAML。

## 原服务启动与请求命令

在匹配的引擎环境中将上面的 YAML 保存为 `/etc/ucm/ucm.yaml`，使用下面原 Qwen2.5-14B-Instruct 启动参数。`--no-enable-prefix-caching` 为原存储测试保留；正常部署需要同时使用原生前缀缓存时移除此参数。

```bash
export ENABLE_UCM_PATCH=1
vllm serve Qwen/Qwen2.5-14B-Instruct \
--max-model-len 20000 \
--tensor-parallel-size 2 \
--gpu_memory_utilization 0.87 \
--block_size 128 \
--trust-remote-code \
--port 7800 \
--enforce-eager \
--no-enable-prefix-caching \
--kv-transfer-config \
'{
    "kv_connector": "UCMConnector",
    "kv_role": "kv_both",
    "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
    "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/etc/ucm/ucm.yaml"}
}'
```

服务就绪后，在另一终端运行原请求负载。这里只恢复测试输入，不表示复现了原性能结果：

```bash
vllm bench serve \
--backend vllm \
--model Qwen/Qwen2.5-14B-Instruct \
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

## 验证写入与读取

按[公共验证步骤](../../user-guide/observability/verify-cache.md)完成保存、重启和重放。文件系统路径还需确认 Posix 读取及相关错误；如果加载由主机缓存满足，可以在专用诊断运行中设置 `cache_load_backend_only: true`，检查后端加载路径。


## 测量存储收益 { #historical-performance-report }

分别比较无缓存 prefill、存储重放和内存热缓存重放。保持模型与输入集相同，记录外部命中 token、Posix 读取字节数、主机内存、TTFT 和吞吐，判断当前负载下存储 I/O 是否比重新计算更划算。

[实现与扩展说明](../extending-store.md#backend-entrypoints)。
