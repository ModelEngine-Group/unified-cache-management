# UCM P/D 分离示例与压测指南（Mooncake 直传和 UCM 中转）

一次压测用两个脚本。从 UCM 仓库根目录执行下面的命令。

| 脚本 | 什么时候运行 | 输入和输出 |
|---|---|---|
| [ucm_pd_disaggregation_example.py](ucm_pd_disaggregation_example.py) | 压测前启动，压测期间持续运行 | 先把请求发给 Prefill（P），再发给 Decode（D）。启用计时后，后台读取引擎日志，在 proxy 日志中为每条请求写一行 `pd_request_timing`。 |
| [summarize_benchmark.py](summarize_benchmark.py) | 客户端跑完一轮后、下一轮开始前运行 | 读取本轮 proxy 日志，统计 QPS、输出吞吐和延迟分位数，把 `pd_benchmark_summary` 写入终端和本轮日志。`--archive` 可同时归档。 |

## 启动前准备

下面用两台 Ascend 主机各起一个实例，P 和 D 都用本机 0 号卡，TP=1、DP=1，服务名为 `pd-demo`。已部署的引擎可以保留自己的模型、设备、并行和网络参数，按下表补齐日志设置并重启引擎。新部署先准备兼容的 vLLM、vLLM Ascend 和相应 connector。UCM 引擎须安装包含 `ucm_pd_prefill_done` 和 `ucm_pd_sync_load_done` 打点的版本。proxy 可独立部署，安装 `httpx`、`fastapi<0.124.0` 和 `uvicorn`。离线对比与整轮汇总只需要 Python 标准库。

| 模式 | P 端计时配置与事件 | D 端计时配置与事件 | proxy 计时方式 |
|---|---|---|---|
| Mooncake pull | `VLLM_LOGGING_LEVEL=INFO`（默认），UTC JSON，记录 `Delaying free of … blocks for request …` | `VLLM_LOGGING_LEVEL=DEBUG`，UTC JSON，记录 `Finished recving KV transfer for request …` | `--timing-backend mooncake` |
| UCM 请求级异步加载 | `UCM_LOG_LEVEL=info`（默认），`ucm_pd_prefill_done` | `VLLM_LOGGING_LEVEL=DEBUG`，UTC JSON，记录 `Finished recving KV transfer for request …` | `--ucm-pd --timing-backend ucm-async` |
| UCM 同步加载 | `UCM_LOG_LEVEL=info`，`ucm_pd_prefill_done` | `UCM_LOG_LEVEL=info`，每个参与 rank 的 `ucm_pd_sync_load_done` | `--ucm-pd --timing-backend ucm-sync --timing-sync-ranks 0`（本例） |

1. 在引擎所在容器里设置环境变量。示例基于 vLLM 0.32.0 的内置日志配置，`VLLM_LOGGING_LEVEL` 控制 vLLM 日志，`UCM_LOG_LEVEL` 控制 UCM 日志。vLLM 默认 `INFO`，UCM 默认 `info`。凡使用 vLLM 事件的一端，示例都加 `--logging-config.formatter json` 和 `TZ=UTC`，proxy 按 UTC 解析 JSON 的毫秒时间戳。UCM 打点自带 `ts_ns`。所有跨机 KV 计时都要校准 P、D 时钟，可用 `chronyc tracking` 查看各主机偏差。proxy 自测的 TTFT、TPOT 和 P 返回到首 token 耗时使用同一台 proxy 的单调时钟。[vLLM 0.32.0 日志选项](https://docs.vllm.ai/en/v0.32.0/examples/features/logging_configuration/)
2. 将 `/shared` 替换为引擎可写、proxy 可读的共享日志挂载点。下面把每个引擎及其子进程的标准输出和标准错误重定向到一个 `engine-*.log`。示例用主机名区分文件，同机多实例还应加实例或端口号。proxy 仅读取这些文件。若已有日志采集器，保留原始 JSON 行或含 `ts_ns` 的原始 UCM 消息，避免加前缀破坏 JSON，或重复采集控制台、UCM 进程日志和归档副本。
3. P、D 分别就绪后，在 proxy 主机执行下面的检查，替换示例地址。随后启动对应模式的 proxy，再发请求。后台收集器从已有文件末尾开始读，历史日志可用文末的 `compare-handoff` 复核。

```bash
export P_HOST=192.0.2.10 D_HOST=192.0.2.20
curl -fsS "http://$P_HOST:8100/health"
curl -fsS "http://$D_HOST:8200/health"
```

**已有自定义日志或旧版 vLLM。** 用 `vllm serve --help=all` 检查镜像是否支持 `--logging-config`。使用内置配置时，也可在 D 端命令追加 `--logging-config.log_level DEBUG` 代替设置环境变量。若使用自定义配置，在对应引擎启动前执行 `export VLLM_LOGGING_CONFIG_PATH=/etc/vllm/logging.json`（换成实际文件），省略内置 JSON 和级别参数。vLLM 采用文件内的 logger、handler 和 formatter。在需要 D 端完成通知的模式，把 `vllm.v1.core.sched.scheduler` 设为 `DEBUG`，它写入的 handler 也要放行 `DEBUG`，其他 logger 可维持 `INFO`。文本 formatter 可写成 `%(created).6f %(levelname)s %(name)s %(message)s`，proxy 会读取消息前的 Unix 秒数及小数。默认人类可读的秒级时间戳不足以计算交接耗时。

## Mooncake P/D 直传

本节适用于 vLLM Ascend 的 `MooncakeConnectorV1` pull 模式。P 完成后返回远端块等交接参数，proxy 默认将这些参数转发给 D。

### 在 P 主机启动 Prefill

以下两段分别在各自的引擎容器执行。替换模型路径、示例 IP 和网卡名，`VLLM_HOST_IP` 要能被对端访问。`--port` 是 proxy 访问的 HTTP 端口，`kv_port` 是 Mooncake 握手的基准端口。网络应放行实际拓扑所需的端口。扩大 TP、DP 时，两端 JSON 的 `prefill` 和 `decode` 都要描述同一套实际拓扑。这两个计时事件由 vLLM 写出，使用 Mooncake 默认的传输库日志级别即可。[Ascend Mooncake 部署示例](https://github.com/vllm-project/vllm-ascend/blob/8cd3c3762/docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md)

```bash
export ASCEND_RT_VISIBLE_DEVICES=0 VLLM_HOST_IP=192.0.2.10
export HCCL_IF_IP="$VLLM_HOST_IP" GLOO_SOCKET_IFNAME=eth0 TP_SOCKET_IFNAME=eth0
export TZ=UTC VLLM_LOGGING_LEVEL=INFO
mkdir -p /shared/mooncake/p
vllm serve /models/Qwen2.5-7B-Instruct --served-model-name pd-demo \
  --host 0.0.0.0 --port 8100 --tensor-parallel-size 1 --block-size 128 \
  --logging-config.formatter json \
  --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_producer","kv_port":36000,"kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":1},"decode":{"dp_size":1,"tp_size":1}}}' \
  >> "/shared/mooncake/p/engine-$(hostname).log" 2>&1
```

### 在 D 主机启动 Decode

```bash
export ASCEND_RT_VISIBLE_DEVICES=0 VLLM_HOST_IP=192.0.2.20
export HCCL_IF_IP="$VLLM_HOST_IP" GLOO_SOCKET_IFNAME=eth0 TP_SOCKET_IFNAME=eth0
export TZ=UTC VLLM_LOGGING_LEVEL=DEBUG
mkdir -p /shared/mooncake/d
vllm serve /models/Qwen2.5-7B-Instruct --served-model-name pd-demo \
  --host 0.0.0.0 --port 8200 --tensor-parallel-size 1 --block-size 128 \
  --logging-config.formatter json \
  --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_consumer","kv_port":36200,"kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":1},"decode":{"dp_size":1,"tp_size":1}}}' \
  >> "/shared/mooncake/d/engine-$(hostname).log" 2>&1
```

### 启动代理，查看逐请求耗时

```bash
python ucm/pd/ucm_pd_disaggregation_example.py \
  --port 9000 --workers 2 \
  --prefiller-hosts "$P_HOST" --prefiller-ports 8100 \
  --decoder-hosts "$D_HOST" --decoder-ports 8200 \
  --timing-backend mooncake \
  --timing-p-logs '/shared/mooncake/p/engine-*.log' \
  --timing-d-logs '/shared/mooncake/d/engine-*.log' \
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

### 准备 UCM 共享存储与两端公共配置

示例使用普通注意力模型、相同的模型路径、tokenizer、dtype、TP 和块大小。P、D 的 `/mnt/ucm-shared` 必须挂载同一份可读写的 KV 存储。`/shared` 单独存日志。将下面的 YAML 分别保存为两台主机**本地**的 `/etc/ucm/pd.yaml`。P 保持 `use_request_async_load: false`。D 做异步实验时仅把该字段改为 `true`，做同步实验时用 `false`。变更后重启对应引擎。示例显式选择整块直接加载（`use_layerwise: false`）。若实际模型选择的 connector 报 `Request-async loading is not supported`，应按它实际采用的同步路径采集。[共享存储部署说明](../../docs/docs-site/docs/zh/user-guide/capabilities/pd-disaggregation/centralized.md)

```yaml
ucm_connectors:
  - ucm_connector_name: "UcmPipelineStore"
    ucm_connector_config:
      store_pipeline: "Cache|Posix"
      storage_backends: "/mnt/ucm-shared"
      cache_buffer_capacity_gb: 32
enable_event_sync: true
use_layerwise: false
use_request_async_load: false
```

在 **P、D 两端各自的启动 shell** 先执行以下公共配置。`ENABLE_UCM_PATCH` 启用 UCM 的 vLLM 适配，`UCM_LOG_LEVEL=info` 放行 UCM 交接打点。`UCM_CONFIG_FILE` 通过 connector 的 JSON 参数传递。两端都用 `kv_both`，proxy 会按请求阶段为 P 设置 `do_remote_decode`、为 D 设置 `do_remote_prefill`。

```bash
export ASCEND_RT_VISIBLE_DEVICES=0 ENABLE_UCM_PATCH=1 UCM_LOG_LEVEL=info
UCM_KV='{"kv_connector":"UCMConnector","kv_role":"kv_both","kv_connector_module_path":"ucm.integration.vllm.ucm_connector","kv_connector_extra_config":{"UCM_CONFIG_FILE":"/etc/ucm/pd.yaml"}}'
```

示例收集控制台日志，可以沿用 `UCM_LOG_TO_FILE` 和 `UCM_LOG_PATH` 的默认值。如果直接收集 UCM 的进程日志，可设 `UCM_LOG_TO_FILE=1` 和 `UCM_LOG_PATH=<本实例的共享目录>`，在对应的 proxy 参数中只匹配 `ucm-*.log`。其中的 `ts_ns` 可用于 UCM 的 P 完成事件和 D 同步完成事件。异步 D 仍应读取含 scheduler 完成事件和可解析时间戳的 vLLM 日志。

### 在 P 主机启动 Prefill（两种模式共用）

P 的完成日志自带 `ts_ns`，vLLM 使用默认 INFO 即可。将 `RUN` 设为本轮的 `ucm-async` 或 `ucm-sync`，让两轮重放相同请求 ID 时各有一份 P 日志。

```bash
RUN=ucm-async  # 同步实验改为 ucm-sync
mkdir -p "/shared/$RUN/p"
vllm serve /models/Qwen2.5-7B-Instruct --served-model-name pd-demo \
  --host 0.0.0.0 --port 8100 --tensor-parallel-size 1 --block-size 128 \
  --kv-transfer-config "$UCM_KV" \
  >> "/shared/$RUN/p/engine-$(hostname).log" 2>&1
```

### 启动异步加载的 D 引擎与 proxy

在 D 主机确认 YAML 中 `use_request_async_load: true`。D 端计时终点来自 vLLM scheduler，给 D 进程设置 `VLLM_LOGGING_LEVEL=DEBUG` 并输出 UTC JSON。

```bash
export TZ=UTC VLLM_LOGGING_LEVEL=DEBUG
mkdir -p /shared/ucm-async/d
vllm serve /models/Qwen2.5-7B-Instruct --served-model-name pd-demo \
  --host 0.0.0.0 --port 8200 --tensor-parallel-size 1 --block-size 128 \
  --logging-config.formatter json --kv-transfer-config "$UCM_KV" \
  >> "/shared/ucm-async/d/engine-$(hostname).log" 2>&1
```

随后在 proxy 主机启动代理。

```bash
python ucm/pd/ucm_pd_disaggregation_example.py \
  --ucm-pd --port 9000 --workers 2 \
  --prefiller-hosts "$P_HOST" --prefiller-ports 8100 \
  --decoder-hosts "$D_HOST" --decoder-ports 8200 \
  --timing-backend ucm-async \
  --timing-p-logs '/shared/ucm-async/p/engine-*.log' \
  --timing-d-logs '/shared/ucm-async/d/engine-*.log' \
  --proxy-log ucm-async-proxy.log
```

`kv_handoff_ms` 从 P 记录 prefill 完成计到 D scheduler 收到接收完成通知。与 Mooncake 直传相比，两者都包含 worker 向 D scheduler 回报的时间。

### 启动同步加载的 D 引擎与 proxy

在 D 主机确认 YAML 中 `use_request_async_load: false`。D 每个成功完成本地加载的 rank 都用 UCM INFO 打印 `rank` 和 `ts_ns`。P 和 D 的这两条 UCM 消息都自带时间戳，因此本模式可直接使用控制台文本格式。若从异步实验切换，重新执行公共配置，并把 vLLM 日志级别切回 INFO。

```bash
export VLLM_LOGGING_LEVEL=INFO
mkdir -p /shared/ucm-sync/d
vllm serve /models/Qwen2.5-7B-Instruct --served-model-name pd-demo \
  --host 0.0.0.0 --port 8200 --tensor-parallel-size 1 --block-size 128 \
  --kv-transfer-config "$UCM_KV" \
  >> "/shared/ucm-sync/d/engine-$(hostname).log" 2>&1
```

在 proxy 主机运行。`--timing-sync-ranks` 要列出该请求在 D 端参与加载的全部全局 rank。本例为 `0`，TP=2 且由 rank 0、1 加载时填 `0 1`。同时确保 proxy 能读取这些 rank 各自的日志。

```bash
python ucm/pd/ucm_pd_disaggregation_example.py \
  --ucm-pd --port 9000 --workers 2 \
  --prefiller-hosts "$P_HOST" --prefiller-ports 8100 \
  --decoder-hosts "$D_HOST" --decoder-ports 8200 \
  --timing-backend ucm-sync --timing-sync-ranks 0 \
  --timing-p-logs '/shared/ucm-sync/p/engine-*.log' \
  --timing-d-logs '/shared/ucm-sync/d/engine-*.log' \
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

两轮重放相同样本、保持相同并发和缓存初始状态。每轮给每个请求唯一的 `X-Request-Id`，同一样本跨轮复用 ID。在 proxy 主机先发一条流式请求，选择能覆盖多个 KV 块、满足 UCM 持久化阈值且 D 无本地 HBM 命中的提示词。下面用重复文本生成一条长提示词，正式对比时换成压测集中的真实输入。

```bash
python3 - <<'PY' | curl -sS -N http://127.0.0.1:9000/v1/completions \
  -H 'Content-Type: application/json' -H 'X-Request-Id: pd-0001' \
  --data-binary @-
import json
print(json.dumps({
    "model": "pd-demo", "prompt": "KV handoff test. " * 256,
    "n": 1, "max_tokens": 16, "temperature": 0, "stream": True,
    "return_token_ids": True, "stream_options": {"include_usage": True},
}))
PY
```

在 proxy 主机按请求 ID 核对 P、D 和 proxy 三份日志。将 `RUN` 设为实际启动的模式。Completion 的引擎请求 ID 会包含 `cmpl-` 前缀及后缀，所以引擎侧搜索客户端 ID 子串。

```bash
RUN=mooncake  # 或 ucm-async、ucm-sync
rg -F 'pd-0001' /shared/"$RUN"/p/engine-*.log | rg 'Delaying free of|ucm_pd_prefill_done'
rg -F 'pd-0001' /shared/"$RUN"/d/engine-*.log | rg 'Finished recving KV transfer|ucm_pd_sync_load_done'
rg -F 'pd_request_timing request_id=pd-0001 ' "$RUN-proxy.log"
```

`kv_status=observed` 时，本次流式请求的 `kv_handoff_ms` 和 `p_response_to_first_token_ms` 应能在同一行读到。前者用引擎事件时间，后者用 proxy 收到 P 响应和 D 首个输出 token 的时间。UCM 的 P 打点发生在 prefill 正常完成时，P 端 dump 可能仍在进行。要判断多少 KV 真正经共享存储交给 D，应同时核对 D 的外部命中、成功加载块数与重算情况。

| 日志现象 | 核对位置 |
|---|---|
| 有响应但没有 `pd_request_timing` | 核对 proxy 的 `--timing-backend`、`--proxy-log` 和请求 ID。若正在等待迟到的事件，可在 `--timing-log-wait` 窗口结束后复查。 |
| `missing_prefill` 或 `missing_decode` | 按表查对应端原始事件和共享文件权限。Mooncake P 须有待传块，UCM D 须有需要从外部加载的块。D 若走异步路径，检查 scheduler 和 handler 是否放行 DEBUG。若 proxy 报时间戳解析错误，检查 JSON 行首、UTC 或文本 formatter 的 `%(created).6f`。 |
| `rank_set_mismatch` | 查同一请求实际出现的 `rank=`、D worker 的加载错误及其日志文件。缺任一预期 rank 都无法判定全部 KV 就绪。 |
| `clock_order` 或交接耗时异常偏大 | 核对 P、D 的时钟同步状态和 JSON 时间戳的时区。两端时钟有偏差时，正值也可能失真。 |
| `ambiguous_attempts` | 查请求重算、HTTP 重试、同轮复用 ID，以及共享目录内是否收进了同一事件的多个副本。 |

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
  --mooncake-p /shared/mooncake/p/engine-*.log --mooncake-d /shared/mooncake/d/engine-*.log \
  --ucm-p /shared/ucm-async/p/engine-*.log --ucm-d /shared/ucm-async/d/engine-*.log \
  > handoff.csv
```

对比 UCM 同步加载时，将 `--ucm-p`、`--ucm-d` 指向同步压测的引擎日志，并追加 `--ucm-sync-ranks 0`（与本轮在线的 rank 列表一致）。这些参数接收文件列表，示例由 shell 展开通配符。每个输入文件只应包含本轮样本。引擎常驻且跨轮复用 ID 时，按轮归档引擎日志或按时间窗口切出该轮记录。CSV 按请求列出两边耗时、`ucm_minus_mooncake_ms` 和配对状态；缺事件、缺 rank 或重试无法唯一配对时，对应差值留空。

两轮重放时复用相同的 `X-Request-Id`。单条 Chat 和 Completion 的引擎 ID 分别按 `chatcmpl-<ID>` 和 `cmpl-<ID>-0` 配对。默认配置会去掉 vLLM 加的八位随机后缀。如果 P、D 已设置 `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1`，启动代理用 `--timing-exact-ids`，离线对比用 `--exact-ids`。

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
