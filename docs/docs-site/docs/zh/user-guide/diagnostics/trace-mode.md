# UCM Trace 模式

UCM Trace 模式是一种轻量级的诊断和评估模式，在推理过程中记录每个请求的跟踪信息，**不**执行任何实际的 KV cache dump/load 操作。Trace 模式允许您收集真实的请求流量数据，并在提交完整的 UCM 存储部署之前模拟 UCM 可以提供的理论 KV cache 命中率。


## 概述

设置 `use_lite: true` 后，UCM 根据模型的 KV cache 布局选择 Lite connector：标准模型使用 `UCMLiteConnector`，混合线性注意力模型使用 `UCMHLALiteConnector`，FAWA 模型使用 `UCMFAWALiteConnector`。

| 选项 | 默认值 | 描述 |
| :----- | :------ | :---------- |
| `use_lite` | `false` | 切换到 **UCM Lite Connector**，它与 Fake Store 一起工作，跳过所有实际的 KV dump/load 操作。 |

启用后只记录请求与块标识，不执行真实 KV 读写，也不减少本次推理计算。分析结果用于估计复用机会，实际存储收益需另行测量。

### 记录的跟踪格式

Lite connector 在启动时输出一条 `UCMTraceMeta:` 日志，记录模型类型、块大小和缓存拓扑。采集时应同时保留这条日志、vLLM 启动日志和请求跟踪；分析器通过这些信息选择模拟引擎并确定缓存容量。

```text
UCMTraceMeta: type=standard, is_mla=False, vllm_hash_block_size=128, trace_hash_block_size=512, hbm_block_data_size=<bytes>
UCMTrace: timestamp: 1234567.890123, request_id: req-42, input_length: 8192, output_length: 128, block_hashes: ['a1b2...', 'c3d4...', ...]
```

`standard`、`mamba` 和 `fawa` 模型的拓扑字段不同。请求日志包含以下字段：

| 字段 | 描述 |
| :---- | :---------- |
| `timestamp` | 查找时的 `time.perf_counter()` 值，用于保持请求顺序。 |
| `request_id` | vLLM 请求标识符。 |
| `input_length` | 输入 token 数（`request.num_tokens`）。 |
| `output_length` | 最大输出 token 数（`request.max_tokens`）。 |
| `block_hashes` | 十六进制编码的块哈希列表，粒度由 `trace_hash_block_size` 指定。 |

标准 Lite connector 在请求至少包含一个完整跟踪块后记录请求。耗时诊断可能另占一条日志。

## 配置

您可以从 `unified-cache-management/examples/ucm_config_example.yaml` 的示例文件开始

使用以下配置启用 Trace 模式：

```yaml
use_lite: true
```

### 日志配置（可选）

长请求可能生成较大的块哈希列表。在启动服务之前调整以下环境变量：

| 环境变量 | 默认值 | 描述 |
| :------------------- | :------ | :---------- |
| `UCM_LOG_PATH` | `log` | 每进程日志文件的目录（例如 `ucm-<pid>.log`）。 |
| `UCM_LOG_MAX_FILES` | `10` | 每进程保留的轮转日志文件的最大数量。 |
| `UCM_LOG_MAX_SIZE` | `5` | 轮转前每个日志文件的最大大小（**MiB**）。为长请求记录跟踪时显著增加此值。 |
| `UCM_LOG_LEVEL` | `info` | 日志级别。跟踪在 `INFO` 级别发出，因此保持为 `info`（或更低以获得额外的调试输出）。 |

跟踪收集运行的示例：

```bash
export UCM_LOG_PATH=/workspace/ucm-trace-logs
export UCM_LOG_MAX_SIZE=256      # 256 MiB per file
export UCM_LOG_MAX_FILES=50     # keep up to 50 rotated files per process
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

对于标准模型，以下启动日志确认 Lite connector 已启用：

```text
[UC][I] Init UCMLiteConnector.
```

您现在可以向服务器发送生产等效的流量。每个请求将在 UCM 日志目录中生成一条跟踪记录。不会转储或加载 KV cache。

## 跟踪分析

收集跟踪后，运行 `benchmarks/auto_trace_analysis.py` 估算 KV cache 命中率。脚本从 `UCMTraceMeta:` 和 vLLM 启动信息中读取模型拓扑及块大小，自动选择 standard、mamba 或 fawa 模拟引擎，模拟 HBM → DRAM → FS 多级缓存。

```bash
python benchmarks/auto_trace_analysis.py \
  --log-dir /workspace/ucm-trace-logs \
  --dram-pool-size-gb <dram_gb> \
  --fs-pool-size-gb <fs_gb>
```

### 必需参数

| 参数 | 描述 |
| :------- | :---------- |
| `--log-dir` | 递归扫描 `*.log`、`*.log.*` 和 `*.log.gz` 的日志目录，必须包含 `UCMTraceMeta:` 和 vLLM 启动日志。 |
| `--dram-pool-size-gb` | 模拟的 DRAM 池容量，单位为 GiB。 |
| `--fs-pool-size-gb` | 模拟的文件系统池容量，单位为 GiB。 |

### 可选参数

| 参数 | 描述 |
| :------- | :---------- |
| `--service-url` | vLLM 服务地址。通过 `/metrics` 获取服务实际的前缀缓存命中率，用于对比。 |
| `--num-nodes` | 物理节点数，默认为 `1`。 |
| `--unified-memory-pool` | 模拟跨节点共享的 DRAM 池，而非各节点独立的池。 |
| `--output` | 将分析摘要写入 JSON 文件。 |
| `--trace-output` | 将解析后的跟踪记录写入 JSON Lines 文件。 |

当前分析器不再接受 `--block-kv-cache-size` 和 `--is-mla`。缺少 `UCMTraceMeta:` 的旧日志需要使用匹配的 Lite connector 重新采集。

## 分析报告

分析生成包括以下内容的综合报告：

- **总请求计数和令牌计数**：工作负载摘要
- **命中率场景**：理论最大值、仅 HBM、HBM+DRAM 和 HBM+DRAM+FS 命中率
- **请求生命周期**：请求的块保持可重用的时间长度（平均值、P90、P95）

这些指标帮助您评估 UCM 是否适合您的工作负载，以及为每个存储层分配多少容量。
