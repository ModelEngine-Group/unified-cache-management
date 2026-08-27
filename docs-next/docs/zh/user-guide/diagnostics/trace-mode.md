# UCM Trace 模式

UCM Trace 模式是一种轻量级的诊断和评估模式，在推理过程中记录每个请求的跟踪信息，**不**执行任何实际的 KV cache dump/load 操作。Trace 模式允许您收集真实的请求流量数据，并在提交完整的 UCM 存储部署之前模拟 UCM 可以提供的理论 KV cache 命中率。

建议首先使用 Trace 模式收集命中率统计信息，并与相关项目成员确认是否采用 UCM。

## 概述

通过在 UCM 配置文件中将两个选项设置为 `true` 来启用 Trace 模式：

| 选项 | 默认值 | 描述 |
| :----- | :------ | :---------- |
| `enable_record_traces` | `false` | 记录每个请求的跟踪（timestamp, input_length, output_length, hash_ids）。每个 hash_id 占用 32 字节。 |
| `use_lite` | `false` | 切换到 **UCM Lite Connector**，它与 Fake Store 一起工作，跳过所有实际的 KV dump/load 操作。 |

当两者都启用时，`UCMConnector` 内部实例化 `UCMLiteConnector` 而不是常规的存储后端 connector。Lite connector：

- 计算真实 UCM 部署将使用的相同块哈希 ID。
- 在首次查找时为每个请求记录跟踪记录。
- 返回 `0` 外部命中令牌（没有真实的 store 可以查找），因此推理正确性不受影响，不持久化 KV 数据。
- 将所有 KV 传输钩子（`start_load_kv`、`save_kv_layer`、`wait_for_save` 等）实现为空操作。

### 记录的跟踪格式

每个请求在首次查找时产生两条日志行：

```text
[UC][I] timestamp: 1234567.890123, request_id: req-42, input_length: 8192, output_length: 128, ucm_block_ids: ['a1b2...', 'c3d4...', ...]
[UC][I] request_id: req-42, hash_time_ms: 0.512, print_time_ms: 0.034
```

| 字段 | 描述 |
| :---- | :---------- |
| `timestamp` | 查找时的 `time.perf_counter()` 值，用于在分析期间保持请求顺序。 |
| `request_id` | vLLM 请求标识符（存在于 Lite connector 跟踪中）。 |
| `input_length` | 请求中的输入令牌数（`request.num_tokens`）。 |
| `output_length` | 请求的最大输出令牌数（`request.max_tokens`）。 |
| `ucm_block_ids` | 十六进制编码的块哈希 ID 列表。每个块对应 `block_size` 个令牌；每个哈希为 32 字节。 |

## 配置

您可以从 `unified-cache-management/examples/ucm_config_example.yaml` 的示例文件开始

通过将两个选项设置为 `true` 来启用 Trace 模式：

```yaml
enable_record_traces: true
use_lite: true
```

### 日志配置（可选）

跟踪行可能很大，因为每个 hash_id 为 32 字节，长请求可能包含许多块 ID。在启动服务之前调整以下环境变量：

| 环境变量 | 默认值 | 描述 |
| :------------------- | :------ | :---------- |
| `UCM_LOG_PATH` | `log` | 每进程日志文件的目录（例如 `ucm-<pid>.log`）。 |
| `UCM_LOG_MAX_FILES` | `10` | 每进程保留的轮转日志文件的最大数量。 |
| `UCM_LOG_MAX_SIZE` | `5` | 轮转前每个日志文件的最大大小（**MiB**）。为长请求记录跟踪时显著增加此值。 |
| `UCM_LOG_LEVEL` | `info` | 日志级别。跟踪在 `INFO` 级别发出，因此保持为 `info`（或更低以获得额外的调试输出）。 |

跟踪收集运行的示例：

```bash
export UCM_LOG_PATH=/workspace/ucm-trace-logs
export UCM_LOG_MAX_SIZE=256      # 每文件 256 MiB
export UCM_LOG_MAX_FILES=50     # 每进程保留最多 50 个轮转文件
export UCM_LOG_LEVEL=info
```

## 启动推理服务

Trace 模式作为 OpenAI 兼容的 vLLM 服务器部署。以与正常 UCM 部署相同的方式启动它——唯一的区别是 UCM 配置文件内容。

以 Qwen/Qwen2.5-14B-Instruct 模型为例：

```bash
export ENABLE_UCM_PATCH=1
vllm serve Qwen/Qwen2.5-14B-Instruct \
  --max-model-len 32000 \
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
      "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/workspace/unified-cache-management/examples/ucm_config_example.yaml"}
  }'
```

**将 `UCM_CONFIG_FILE` 路径替换为您机器上跟踪模式配置文件的实际路径。**

服务启动时，以下日志确认 Lite connector 处于活动状态：

```text
[UC][I] Init UCMLiteConnector.
```

您现在可以向服务器发送生产等效的流量。每个请求将在 UCM 日志目录中生成一条跟踪记录。不会转储或加载 KV cache。

## 跟踪分析

收集跟踪后，运行 `benchmarks/auto_trace_analysis.py` 以模拟理论 KV cache 命中率。脚本解析跟踪行**加上** vLLM/UCM 在启动时发出的 `available kv cache memory`（或 `current kv cache memory`）和 `tensor_parallel_size` 值，然后模拟 LRU 多级缓存（HBM → DRAM → FS）以估计 UCM 将实现的命中率。

```bash
python benchmarks/auto_trace_analysis.py \
  --service-url <ip:port of vllm service> \
  --log-dir <path to log folder> \
  --block-kv-cache-size <bytes_per_block> \
  --is-mla <true|false> \
  --dram-pool-size-gb <dram_gb> \
  --fs-pool-size-gb <fs_gb>
```

### 必需参数

| 参数 | 描述 |
| :------- | :---------- |
| `--service-url` | vLLM `/metrics` 端点（Prometheus）。设置后，工具获取服务的实际前缀缓存命中率以进行比较。 |
| `--log-dir` | 包含 UCM 日志文件的目录（递归扫描 `*.log`、`*.log.*`、`*.log.gz`）。必须包括 vLLM 的启动日志，以便可以解析可用的 KV cache 内存和张量并行大小。 |
| `--block-kv-cache-size` | 单个 KV cache 块的字节大小。使用 KV Cache 大小计算器确定模型的此值。 |
| `--is-mla` | 模型是否使用多潜在注意力（`true`/`false`）。 |
| `--dram-pool-size-gb` | 模拟的 DRAM（主机内存）池大小（GiB）。 |
| `--fs-pool-size-gb` | 模拟的文件系统（SSD/NFS）池大小（GiB）。 |

## 分析报告

分析生成包括以下内容的综合报告：

- **总请求计数和令牌计数**：工作负载摘要
- **命中率场景**：理论最大值、仅 HBM、HBM+DRAM 和 HBM+DRAM+FS 命中率
- **请求生命周期**：请求的块保持可重用的时间长度（平均值、P90、P95）

这些指标帮助您评估 UCM 是否适合您的工作负载，以及为每个存储层分配多少容量。

有关完整的文档，请参阅原始 UCM 文档中的详细 Trace 模式指南。