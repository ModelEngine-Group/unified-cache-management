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

## 其他已注册管线

| 需求 | 选择 |
| --- | --- |
| 保留已有 vLLM NFS Connector 配置 | [NFS Store](nfs.md) |
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
