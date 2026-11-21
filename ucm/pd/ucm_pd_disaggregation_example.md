# UCM P/D 分离示例与压测指南（Mooncake 直传和 UCM 中转）

一次压测用两个脚本。从 UCM 仓库根目录执行下面的命令。

| 脚本 | 什么时候运行 | 输入和输出 |
|---|---|---|
| [ucm_pd_disaggregation_example.py](ucm_pd_disaggregation_example.py) | 压测前启动，压测期间持续运行 | 先把请求发给 Prefill（P），再发给 Decode（D）。启用计时后，后台读取引擎日志，在 proxy 日志中为每条请求写一行 `pd_request_timing`。 |
| [summarize_benchmark.py](summarize_benchmark.py) | 客户端跑完一轮后、下一轮开始前运行 | 读取本轮 proxy 日志，统计 QPS、输出吞吐和延迟分位数，把 `pd_benchmark_summary` 写入终端和本轮日志。`--archive` 可同时归档。 |

以下示例假设 P、D 引擎已启动，`P_HOST`、`D_HOST` 分别是它们的地址，HTTP 端口分别是 8100、8200。把示例共享目录换成 proxy 实际可读的**本轮引擎日志**目录，保留 glob 外层的引号。proxy 会从已有文件末尾开始读，也会发现新建的匹配文件。每个引擎事件只提供一份日志，避免把同一事件的控制台副本和归档文件都读进来。单独部署时复制上表两个脚本即可。代理脚本内置在线计时和离线逐请求对比，整轮统计脚本可独立运行。

## Mooncake P/D 直传

本节适用于 vLLM Ascend 的 `MooncakeConnectorV1` pull 模式。P 完成后返回远端块等交接参数，proxy 默认将这些参数转发给 D。

### 启动代理，查看逐请求耗时

先校准 P、D 主机的时钟。P 日志需要能看到 INFO 消息 `Delaying free of … blocks for request …`，D 日志需要能看到 scheduler 的 DEBUG 消息 `Finished recving KV transfer for request …`。两条日志的时间戳都需要毫秒或更细的精度。

支持内置 JSON 日志的 vLLM（如 v0.32.0）可以在两端引擎环境设置 `TZ=UTC`，在 P 的 `vllm serve` 命令追加 `--logging-config.formatter json`，在 D 追加 `--logging-config.formatter json --logging-config.log_level DEBUG`。解析器按 UTC 读取内置 JSON 的 `asctime`。D 的 CLI 选项会开启全局 DEBUG；已有自定义文本日志配置的部署也可只把 `vllm.v1.core.sched.scheduler` 设为 DEBUG，让 handler 允许该级别，并在 P、D 的 formatter 中把 `%(created).6f` 放在 `%(message)s` 前。旧版 vLLM 可通过已有的 `VLLM_LOGGING_CONFIG_PATH` 加载此配置。[vLLM 日志选项](https://docs.vllm.ai/en/stable/examples/features/logging_configuration/)

```bash
python ucm/pd/ucm_pd_disaggregation_example.py \
  --port 9000 --workers 2 \
  --prefiller-hosts "$P_HOST" --prefiller-ports 8100 \
  --decoder-hosts "$D_HOST" --decoder-ports 8200 \
  --timing-backend mooncake \
  --timing-p-logs '/shared/mooncake/p/*.log' \
  --timing-d-logs '/shared/mooncake/d/*.log' \
  --proxy-log mooncake-proxy.log
```

压测客户端向 proxy 的 `/v1/completions` 或 `/v1/chat/completions` 发请求。例如，查看 ID 为 `pd-0001` 的样本：

```bash
rg -F 'pd_request_timing request_id=pd-0001 ' mooncake-proxy.log
```

这一行的 `kv_handoff_ms` 从 P 记录保留待传块开始，到 D scheduler 收到各 worker 聚合后的接收完成通知为止。

### 汇总并归档 Mooncake 压测

假设本轮有 1000 个可计时的逻辑请求，在另一个终端执行：

```bash
python ucm/pd/summarize_benchmark.py mooncake-proxy.log \
  --expected-requests 1000 --archive mooncake-round-01.log
```

终端和 `mooncake-round-01.log` 都会出现 `pd_benchmark_summary backend=mooncake`。proxy 继续运行，下一条日志会写入新建的 `mooncake-proxy.log`。

## UCM P/D 共享存储中转

本节适用于 P 将 KV 写入 UCM 共享存储、D 通过 UCM 加载 KV 的部署。启动 proxy 时加 **`--ucm-pd`**，这样即使 P 返回空交接参数，proxy 发给 D 的请求仍带 `do_remote_prefill=true`。按 **D 实际采用的加载模式**选择下面一条启动命令；`--timing-backend` 只选择日志解析方式。

### D 异步加载时启动代理

P 的 UCM INFO 日志需要包含 `ucm_pd_prefill_done` 和 `ts_ns`。D 复用 vLLM scheduler 的 `Finished recving KV transfer` 日志。在支持内置 JSON 的版本，D 引擎用 `TZ=UTC` 和 `--logging-config.formatter json --logging-config.log_level DEBUG` 启动；已有 epoch 文本日志的部署可以沿用上面的文本配置。校准 P、D 主机时钟后，启动 proxy：

```bash
python ucm/pd/ucm_pd_disaggregation_example.py \
  --ucm-pd --port 9000 --workers 2 \
  --prefiller-hosts "$P_HOST" --prefiller-ports 8100 \
  --decoder-hosts "$D_HOST" --decoder-ports 8200 \
  --timing-backend ucm-async \
  --timing-p-logs '/shared/ucm-async/p/*.log' \
  --timing-d-logs '/shared/ucm-async/d/*.log' \
  --proxy-log ucm-async-proxy.log
```

`kv_handoff_ms` 从 P 记录 prefill 完成计到 D scheduler 收到接收完成通知。与 Mooncake 直传相比，两者都包含 worker 向 D scheduler 回报的时间。

### D 同步加载时启动代理

P、D 两端只需 UCM INFO 日志。`ucm_pd_prefill_done` 和 `ucm_pd_sync_load_done` 都自带 `ts_ns`，仍需校准主机时钟。下面假设 D 端本轮需要 **rank 0 和 1** 参与加载，请按实际部署替换：

```bash
python ucm/pd/ucm_pd_disaggregation_example.py \
  --ucm-pd --port 9000 --workers 2 \
  --prefiller-hosts "$P_HOST" --prefiller-ports 8100 \
  --decoder-hosts "$D_HOST" --decoder-ports 8200 \
  --timing-backend ucm-sync --timing-sync-ranks 0 1 \
  --timing-p-logs '/shared/ucm-sync/p/*.log' \
  --timing-d-logs '/shared/ucm-sync/d/*.log' \
  --proxy-log ucm-sync-proxy.log
```

同步模式取最后一个必需 rank 成功完成本地加载的时间作为终点。比较 Mooncake 和 UCM 同步结果时，要考虑 Mooncake 还包含 D scheduler 收到回报的时间。

### 查看请求，汇总并归档 UCM 压测

异步模式读 `ucm-async-proxy.log`：

```bash
rg -F 'pd_request_timing request_id=pd-0001 ' ucm-async-proxy.log
python ucm/pd/summarize_benchmark.py ucm-async-proxy.log \
  --expected-requests 1000 --archive ucm-async-round-01.log
```

同步模式读 `ucm-sync-proxy.log`：

```bash
rg -F 'pd_request_timing request_id=pd-0001 ' ucm-sync-proxy.log
python ucm/pd/summarize_benchmark.py ucm-sync-proxy.log \
  --expected-requests 1000 --archive ucm-sync-round-01.log
```

## 对比两轮结果

两轮重放相同样本、保持相同并发和缓存初始状态。每轮给每个请求唯一的 `X-Request-Id`，同一样本跨轮复用 ID。下面发一条可用 `pd-0001` 查询的流式请求。在 proxy 所在主机执行，替换模型名和提示词；验证 KV 交接时选覆盖多个块并满足 UCM 持久化阈值的样本。

```bash
curl -N http://127.0.0.1:9000/v1/completions \
  -H 'Content-Type: application/json' -H 'X-Request-Id: pd-0001' \
  -d '{
    "model": "YOUR_SERVED_MODEL",
    "prompt": "替换为本轮的测试提示词",
    "n": 1, "max_tokens": 16, "temperature": 0, "stream": true,
    "return_token_ids": true, "stream_options": {"include_usage": true}
  }'
```

例如，对比 Mooncake 和 UCM 异步的整轮结果：

```bash
rg '^pd_benchmark_summary ' mooncake-round-01.log ucm-async-round-01.log
```

### 整轮指标（`pd_benchmark_summary`）

| 字段 | 怎么读 |
|---|---|
| `qps`、`output_tps` | 成功逻辑请求数、成功请求输出 token 总数分别除以 `duration_s`。窗口从同一台 proxy 上最早的请求开始，到最晚的请求收尾，包含失败请求和轮内空隙。后台等待引擎日志的时间在窗口之外。 |
| `kv_handoff_p50_ms`、`kv_handoff_p95_ms`、`kv_handoff_p99_ms` | 成功请求中，`kv_status=observed` 的 KV 交接耗时分位数。先比较 `kv_handoff_n` 和成功数，确认两组样本的覆盖情况。 |
| `ttft_p*_ms`、`tpot_p*_ms` | 成功请求的 TTFT、TPOT 各自的 P50、P95、P99，按排序后的有效值线性插值。`ttft_n`、`tpot_n` 各自给出样本数，缺失值不参与计算。 |
| `requests`、`successful`、`failed`、`output_tokens_n` | 分别是请求总数、成功数、失败数和成功请求中有可靠输出计数的样本数。有成功请求缺少计数时，总输出吞吐显示 `NA`，`known_output_tokens` 仍给出已知部分。 |

### 逐请求指标（`pd_request_timing`）

| 字段 | 怎么读 |
|---|---|
| `kv_handoff_ms`、`kv_status` | 取对应模式的 P、D 引擎事件之差。`observed` 表示收到了预期事件；确认完整 KV 交接还需核对 D 的外部命中量、成功加载量和重算量。 |
| `ttft_ms`、`p_response_to_first_token_ms` | 终点都是 proxy 收到首个输出 token，起点分别是请求进入 proxy、proxy 收到 P 的完整 HTTP 响应。空角色、心跳和 usage 事件不会触发。非流式请求的首 token 耗时显示 `NA`。 |
| `prefill_http_ms`、`decode_http_ms` | proxy 观测到的 P、D 两段 HTTP 耗时，包含网络和排队。P 计到完整响应，包含内部重试退避；D 计到最后一批响应字节，多次 HTTP 尝试分别累计。 |
| `p_response_to_d_send_ms`、`d_send_to_first_token_ms` | 首次 P 返回到首次 D 派发，以及首次 D 派发到首个输出 token。 |
| `generation_ms`、`tpot_ms`、`decode_tps` | `generation_ms` 是首末输出 token 到达的跨度。单序列 TPOT = 跨度 /（输出 token 数 − 1），decode TPS = 1000 / TPOT，需要至少两个 token 和可区分的到达时刻。 |
| `e2e_ms`、`output_tps` | 请求进入 proxy 到转发收尾的时间，以及**本条请求**的输出 token 数 / 该时间（秒）。 |
| `prompt_tokens`、`prefill_cached_tokens`、`output_tokens` | 从引擎 usage 或输出 token IDs 计数，`count_source` 标明输出计数来源。 |
| `status`、`http_retries`、`recomputes` | 请求结果、HTTP 重试次数和重新派发 P/D 的次数。批量或多候选输出、重算、重试、缺事件或缺 rank 等情况无法确认单次完整交接时，`kv_handoff_ms` 显示 `NA`。 |

### 收尾与复核

- **一轮一个活动日志。** 用 `--proxy-log` 写 proxy 日志，避免再用 `tee` 或 shell 重定向写同一文件。预热样本在正式压测前单独归档；等本轮汇总命令结束再开始下一轮。
- **区分两个等待时间。** proxy 在后台等待迟到的引擎事件，`--timing-log-wait` 默认 30 秒；统计脚本等待逐请求行写齐，`--wait-seconds` 默认 60 秒。`--expected-requests` 填本轮实际可计时的逻辑请求数，包含失败请求。
- **归档在同一文件系统内进行。** 单独写 `--archive` 会生成带时间戳的归档名；省略该选项会把汇总行追加到当前日志。超时、计数不符、日志明确报告丢记录、同一后端的请求 ID 重复或归档名已存在时，脚本会报错并保留原日志。
- **旧日志的时间窗口。** 没有 `proxy_start_ns`、`proxy_end_ns` 的日志仍可计算分位数；若客户端实测整轮耗时为 42.5 秒，可传 `--duration-s 42.5` 来计算 QPS 和输出吞吐。两轮应使用相同的窗口定义。

### 用 `compare-handoff` 按请求对比两轮交接耗时

在线结果按打印时已到达的日志计算。要复核迟到的完成或失败事件，等两轮的 P、D **引擎日志**收齐后运行代理脚本的 `compare-handoff` 子命令。它只用 Python 标准库，将 CSV 写到标准输出。例如，对比 Mooncake 和 UCM 异步加载：

```bash
python ucm/pd/ucm_pd_disaggregation_example.py compare-handoff \
  --mooncake-p /shared/mooncake/p/*.log --mooncake-d /shared/mooncake/d/*.log \
  --ucm-p /shared/ucm-async/p/*.log --ucm-d /shared/ucm-async/d/*.log \
  > handoff.csv
```

对比 UCM 同步加载时，将 `--ucm-p`、`--ucm-d` 指向同步压测的引擎日志，并追加 `--ucm-sync-ranks 0 1`（替换成实际参与加载的 rank）。这些参数接收文件列表，示例由 shell 展开通配符。CSV 按请求列出两边耗时、`ucm_minus_mooncake_ms` 和配对状态；缺事件、缺 rank 或重试无法唯一配对时，对应差值留空。

两轮重放时复用相同的 `X-Request-Id`。单条 Chat 和 Completion 的引擎 ID 分别按 `chatcmpl-<ID>` 和 `cmpl-<ID>-0` 配对。默认去掉 vLLM 加的八位随机后缀；引擎关闭 ID 随机化时，启动代理用 `--timing-exact-ids`，离线对比用 `--exact-ids`。

## 同步 vLLM Ascend 代理

代理基于 vLLM Ascend 的 `8cd3c3762`。脚本中 `UCM extensions` 到 `vLLM Ascend proxy` 两个注释之间集中放置了 D 请求构造、P/D 计时、流事件解析、后台日志汇总和 `compare-handoff`。后面的代码按上游顺序组织。升级时先查看上游脚本从该基线到目标版本的增量，在已检出目标版本的 vLLM Ascend 仓库中执行

```bash
git diff 8cd3c3762 HEAD -- examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py
```

若上游改动涉及以下位置，合并时要一起检查相应行为。

| 上游位置 | 本地接入点 |
|---|---|
| `parse_args`、日志和进程生命周期 | `parse_pd_options` 注册本地选项。启用计时时由持有调度器的进程启动和关闭唯一的日志收集器。指定 `--proxy-log` 时各写日志的进程异步落盘，支持压测轮次间轮转。 |
| `SharedProxyScheduler.begin_request` / `finish_request` | 随原有调度调用登记请求和提交计时结果。注册发生在发出 P 请求之前，请求处理路径只入队，后台线程读取新增的引擎日志。 |
| `assign_instances` / `reassign_instances` | `prefill_for_decode` 记录本次 P 返回时刻，`build_decode_request` 构造对应的 D 请求。重算沿用逻辑请求 ID，并使用本次 P 的交接参数。 |
| `stream_service_response` / `handle_completions_impl` | 在 HTTP 读取处给字节打时间戳，按完整 SSE 事件识别首 token、回写 usage 和处理重算。一批字节可能包含多个事件，一个事件也可能跨多次读取。重算前关闭旧 D 流。 |

同步后在 UCM 仓库运行 `python -m pytest --noconftest -q test/pd`，验证默认 Mooncake 转发、UCM D 请求、流式重算、单/多 worker 汇总和离线对比，再更新脚本头部及本节的基线提交。
