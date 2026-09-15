# KV 压缩存储

当存储 I/O 成为瓶颈，并且应用能够容忍缓存值变化时，可以用 `Cache|Compress|Posix` 评估更小的文件系统 KV 载荷。Cache Store 将设备张量传入主机缓冲，Compress 编码主机数据，再由 Posix 存储；加载时按相反方向执行。

当前启用的压缩模式是**有损 BF16 编码**。编解码器限制指数范围并保留更少的尾数位，也可能回退到类似 FP8 的编码以满足大小预算。因此，它不保证逐位还原 KV，也不保证模型质量不变。它不会改变引擎已分配的设备 KV Cache dtype，也不会减少模型的设备内存分配。

## 确认编解码器适合当前负载

运行时目前只接受 `data_type: 0`，即 BF16。该参数告诉编解码器如何解释字节，不会转换引擎实际的 KV 值。启用此流水线前，先确认引擎实际生效的 KV dtype。FP16、FP8、量化或混合类型的 KV 布局，不能通过把此字段设为零变得兼容。

当前只接受两个 `compress_ratio` 值：

| 值 | 行为 |
| --- | --- |
| `16` | 在文件系统对齐之前，将编码载荷存为原始分片大小的一半。 |
| `32` | 数据直接通过 Compress 阶段，不进行压缩。 |

内部枚举定义中的其他压缩比不被当前阶段接受。使用 [Cache|Posix](pipeline.md)建立未压缩基线。

## 准备独立存储命名空间

从[安装](../../user-guide/quick_start/index.md)开始，或针对目标引擎环境[从源码构建](../build_from_source.md)。注册的流水线通过 `ucmpipelinestore` 加载 `libcachestore.so`、`libcompressor.so` 和 `libposixstore.so`。Store 的 CMake 构建包含 compressor 目标。

为压缩数据使用新目录。不要让压缩部署和未压缩部署指向相同文件，也不要在更换编解码器设置或张量几何后复用旧目录。流水线根据压缩比推导存储块大小，因此文件名相同不代表载荷布局兼容。

## 配置 BF16 试验

```yaml
ucm_connectors:
  - ucm_connector_name: "UcmPipelineStore"
    ucm_connector_config:
      store_pipeline: "Cache|Compress|Posix"
      storage_backends: "/mnt/ucm-compressed-bf16"
      compress_ratio: 16
      data_type: 0
      decompress_thread_num: 48
      io_direct: true
      cache_buffer_capacity_gb: 64
      posix_io_engine: "aio"

use_layerwise: true
enable_record_traces: false
```

示例沿用 64 GiB 缓存、48 个解压线程和 Direct I/O + AIO。这些是原示例的显式设置，不是运行时默认值。

按 [vLLM 快速开始](../../user-guide/quick_start/index.md#vllm)或 [vLLM-Ascend 快速开始](../../user-guide/quick_start/index.md#vllm-ascend)传入此文件。除这些 Store 设置外，还必须确认引擎使用 BF16 KV dtype。`data_type` 默认为无效哨兵值，因此必须显式指定。`decompress_thread_num` 默认是 6；`stream_number` 默认是 8，压缩器使用该数量的一半作为 dump worker 数。Cache 传输流通过 `cache_stream_number` 单独配置。

首次写入前，检查实际生效的分片大小。构建器按以下公式计算存储分片字节数：

```text
floor((shard_size * compress_ratio / 32) / 4096) * 4096
```

压缩比为 16 时，分片大小的一半必须非零，且已经是 4096 的整数倍；否则构建器会向下取整存储大小。即使用 buffered I/O，这也是数据布局约束。当前 connector 不会为每一种可能的模型布局确认这一条件，因此仅服务启动成功还不够。主机缓冲还必须满足[流水线内存规划](pipeline.md#budget-host-memory)的要求。

## 原服务启动与请求命令

在匹配的引擎环境中将上面的 YAML 保存为 `/etc/ucm/ucm.yaml`，使用下面原 Qwen2.5-14B-Instruct 启动参数。`--no-enable-prefix-caching` 为原存储测试保留；正常部署需要同时使用原生前缀缓存时移除此参数。

```bash
export ENABLE_UCM_PATCH=1
vllm serve Qwen/Qwen2.5-14B-Instruct \
--max-model-len 32000 \
--tensor-parallel-size 4 \
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

通过[外部缓存验证](../../user-guide/observability/verify-cache.md)检查此服务。输入与生成 token 总数必须在 32000-token 上下文内；原 benchmark 的 32000 输入加 2 输出超过这个上限，因此不作为可执行命令恢复。

## 验证存储和输出质量

使用固定且有代表性的提示词集，并为基线准备独立目录。分别验证以下三点：

1. **写入与读取：**完成首次请求，保留压缩文件，重启服务进程，再重放相同输入。必须观察到外部命中和 Posix 读取，确保执行了编解码器的加载路径。
2. **存储大小：**对同一批完整块，将已完成写入的缓存文件大小与 `Cache|Posix` 比较。检查编码载荷比例时，排除临时文件和文件系统分配开销。
3. **模型质量：**在相同解码设置下，将应用关注的输出和质量指标与未压缩运行比较。评估速度前，先确定可接受的质量范围。

同进程重放可能由未压缩的主机或设备内存提供数据，不能测试解压。按[外部缓存验证流程](../../user-guide/quick_start/index.md#vllm-verify-the-service-and-external-cache)执行，并同时报告冷读取 TTFT、CPU 消耗、存储字节数和质量。载荷减半不等于端到端延迟减半。

## 排查试验失败

| 现象 | 下一步检查 |
| --- | --- |
| 压缩 dtype 或 ratio 无效 | 实际 BF16 KV dtype，以及上面列出的允许值。 |
| 原生库加载失败 | 三个阶段是否都为目标环境完成构建。 |
| 文件大小异常或重放数据损坏 | 分片对齐，以及是否使用仅供当前编解码器和布局使用的空命名空间。 |
| `COMPRESS DUMP FAILED` 或 `COMPRESS LOAD FAILED` | 完整的编解码器与后端错误日志；不能将该次运行计为成功的缓存测试。 |
| 文件变小但延迟增加 | 压缩与解压的 CPU 开销，相对于节省的存储传输时间是否划算。 |
| 质量下降 | 与未压缩 KV 比较，评估应用的容忍范围是否允许使用该编解码器。 |

当前编解码器的部分错误路径会在分片级记录日志后继续执行。验收需要同时包含无错误的请求和已验证的输出；仅任务完成不能证明每个分片都正确解码。解决这些失败后，再将压缩目录用于生产流量。

[实现与扩展说明](../extending-store.md#backend-entrypoints)。
