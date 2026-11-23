# P/D 分离部署与压测指南

Prefill（P）计算提示词的 KV 缓存，Decode（D）用它继续生成 token。两种部署都由 proxy 先调用 P、再调用 D，区别在于 **D 从哪里取得 KV**。

| 交接方式 | KV 怎么到 D | proxy 做什么 |
| --- | --- | --- |
| UCM 共享存储 | P 将提示词的 KV 块写入共享缓存。D 按提示词前缀查找并加载可见的块，缺失的部分可能需要重算。 | 将原始请求发给 D，并设置 `do_remote_prefill=true`。 |
| Mooncake 直传（pull） | P 暂时保留待传的 KV 块。D 根据 P 返回的远端引擎和块信息，通过 Mooncake 从 P 拉取所需的 KV。 | 把 P 返回的 `kv_transfer_params` 转交给 D。 |

## 启动压测 proxy

压测时，两种方式都传入 P、D 引擎日志。proxy 默认使用 Mooncake。传入 `--ucm-config` 后，proxy 使用 UCM，并从 D 的 YAML 选择同步或异步计时。两种方式都能统计 KV 交接耗时、首 token 延迟（TTFT）、后续每个 token 的平均耗时（TPOT），以及整轮的 QPS 和输出吞吐。

在 proxy 主机的 UCM 仓库根目录执行，将示例 IP、端口和日志路径换成部署环境的值。P、D 的引擎日志都要对 proxy 可读。跨机计算交接耗时前，先校准两台引擎主机的时钟，并确认服务就绪。

```bash
curl -fsS http://192.0.2.10:8100/health
curl -fsS http://192.0.2.20:8200/health
```

### 在 P、D 端开启计时日志

`--prefiller-logs` 和 `--decoder-logs` 告诉 proxy 去哪里读日志。日志级别和格式要在 **Prefill 和 Decode 各自的启动环境**设置。按下表选一行，调整后重启对应的引擎。表中 vLLM 参数使用 v0.32.0 的内置日志配置。

| 交接方式 | Prefill（P）启动设置 | Decode（D）启动设置 |
| --- | --- | --- |
| UCM 同步（默认） | `UCM_LOG_LEVEL=info`（默认值） | `UCM_LOG_LEVEL=info`（默认值）。两端的 UCM 事件自带时间戳，使用默认级别即可。 |
| UCM 请求级异步 | `UCM_LOG_LEVEL=info`（默认值） | 启动前执行 `export TZ=UTC VLLM_LOGGING_LEVEL=DEBUG`，并给 `vllm serve` 加 `--logging-config.formatter json`。 |
| Mooncake pull | 启动前执行 `export TZ=UTC VLLM_LOGGING_LEVEL=INFO`，并给 `vllm serve` 加 `--logging-config.formatter json`。 | 启动前执行 `export TZ=UTC VLLM_LOGGING_LEVEL=DEBUG`，并给 `vllm serve` 加 `--logging-config.formatter json`。 |

UCM 日志默认为 INFO。如果部署已把级别调到 WARNING 或 ERROR，在需要 UCM 事件的一端执行 `export UCM_LOG_LEVEL=info` 后再启动引擎。

如果使用自定义 vLLM 日志配置，在配置文件里设置级别和格式。D 端需要 DEBUG 时，确保 `vllm.v1.core.sched.scheduler` 和对应的 handler 都允许 DEBUG。可沿用 vLLM 的 UTC JSON 格式，或在文本日志的消息前加 `%(created).6f`，让 proxy 能读取带小数的 Unix 时间戳。

**把日志写到 proxy 可读的位置。** 若已有日志文件，直接将实际路径传给 proxy。若引擎只输出到终端，以 UCM 同步为例，先在 P、D 分别创建共享目录，再把以下重定向加在各自原有启动命令的末尾。

| 所在端 | 创建目录 | 追加到引擎启动命令末尾 |
| --- | --- | --- |
| P | `mkdir -p /shared/ucm-sync/p` | `>> /shared/ucm-sync/p/engine-prefill.log 2>&1` |
| D | `mkdir -p /shared/ucm-sync/d` | `>> /shared/ucm-sync/d/engine-decode.log 2>&1` |

多实例部署时，为每个实例使用不同的 `engine-*.log` 文件名。Mooncake 示例使用 `/shared/mooncake/`，相应调整两端的输出目录。

### UCM 共享存储

**沿用 D 正在使用的 UCM YAML。** 在 D 的 `--kv-transfer-config` 中，`kv_connector_extra_config.UCM_CONFIG_FILE` 指向它读取的文件。让 proxy 能读到同一份文件，把 **proxy 一侧的路径**传给 `--ucm-config`。独立部署的 proxy 需要安装 PyYAML。

P 的 `ucm_pd_prefill_done` 是交接计时的起点。D 的 YAML 中，`use_request_async_load` 未填写或为 `false` 时走同步计时，proxy 收齐 D 的 worker 回报后，取最后一个实际加载完成的时间。设为 `true` 时走异步计时，终点是 D scheduler 记录 `Finished recving KV transfer` 的时间。

```bash
python ucm/pd/ucm_pd_disaggregation_example.py \
  --port 9000 --workers 2 \
  --prefiller-hosts 192.0.2.10 --prefiller-ports 8100 \
  --decoder-hosts 192.0.2.20 --decoder-ports 8200 \
  --ucm-config /etc/ucm/pd.yaml \
  --prefiller-logs '/shared/ucm-sync/p/engine-*.log' \
  --decoder-logs '/shared/ucm-sync/d/engine-*.log' \
  --proxy-log proxy.log
```

| 参数 | 填什么 | proxy 怎么用 |
| --- | --- | --- |
| `--ucm-config` | D 正在使用的 UCM YAML 在 proxy 上的路径 | **加上此参数走 UCM 共享存储中转，省略时默认走 Mooncake 直传。** 走 UCM 时，proxy 读取 YAML 中的 `use_request_async_load` 选择计时方式。 |
| `--prefiller-logs` | P 引擎写出的日志文件 | 读取每条请求的 prefill 完成时间，作为 KV 交接的起点。 |
| `--decoder-logs` | D 引擎写出的日志文件 | 读取同一请求的 KV 加载完成事件，确定交接终点。同步模式要覆盖该 D 副本的所有 worker。 |
| `--proxy-log` | proxy 要写入的日志文件。示例中的 `proxy.log` 位于启动目录 | 将逐请求统计写入该文件。每轮结束后，把它传给 `summarize_benchmark.py` 汇总。 |

P、D 日志都填 **proxy 能读到的路径**。只有一份日志时可直接填文件名。有多份时，保留示例中的单引号，让 proxy 用 `engine-*.log` 匹配该目录下的引擎日志。

支持 `workers` 日志字段的 D 会自动报告 worker 数，启动 proxy 时可以省略 `--timing-sync-ranks`。命令中的 `--workers 2` 控制 proxy 自身的服务进程数。

### Mooncake 直传（pull）

下面以 vLLM Ascend 的 `MooncakeConnectorV1` pull 模式为例。P、D 启动时使用 UTC 时间和 JSON 日志。P 用 vLLM INFO 记录为请求保留 KV 块的时刻（`Delaying free of … blocks`），D 用 vLLM DEBUG 记录 scheduler 收到传输完成通知的时刻（`Finished recving KV transfer`）。proxy 用这两个时刻计算交接耗时。

```bash
python ucm/pd/ucm_pd_disaggregation_example.py \
  --port 9000 --workers 2 \
  --prefiller-hosts 192.0.2.10 --prefiller-ports 8100 \
  --decoder-hosts 192.0.2.20 --decoder-ports 8200 \
  --prefiller-logs '/shared/mooncake/p/engine-*.log' \
  --decoder-logs '/shared/mooncake/d/engine-*.log' \
  --proxy-log proxy.log
```

## 发起压测并汇总结果

上面的 proxy 默认监听本机。客户端可请求 `http://127.0.0.1:9000/v1/chat/completions` 或 `/v1/completions`。客户端在另一台机器上时，在 proxy 启动命令中用 `--host` 指定 proxy 主机可供客户端访问的 IP。

要同时得到 TTFT、TPOT 和输出吞吐，每个请求发一条输入，并在请求体中加入以下字段，以便 proxy 记录首 token 到达时间和输出 token 数。

```json
{"n": 1, "stream": true, "stream_options": {"include_usage": true}}
```

TPOT 需要至少输出两个 token，且 proxy 能区分首、末 token 的到达时间。

每轮压测结束后，在 proxy 主机另开一个终端运行下面的命令。如果有单独的预热轮，预热结束时也归档一次。

```bash
python ucm/pd/summarize_benchmark.py proxy.log --archive
```

脚本在终端打印 `pd_benchmark_summary`，并将本轮日志和汇总存入同目录下带时间戳的归档文件。等命令结束再发下一轮请求，proxy 会自动写入新的 `proxy.log`。

| 想看什么 | 看哪些字段 |
| --- | --- |
| 本轮处理了多少请求 | `requests`、`successful`、`failed`。QPS 是成功请求数除以本轮耗时。 |
| 输出吞吐是否完整 | `output_tps` 的单位是 token/s。`output_tokens_n` 应等于 `successful`。有成功请求缺少 token 计数时，输出吞吐显示 `NA`。 |
| 延迟分布覆盖多少请求 | KV 交接、TTFT、TPOT 各有 P50、P95、P99。`kv_handoff_n`、`ttft_n`、`tpot_n` 分别是参与计算的有效样本数。 |

**检查某条请求。** 如果客户端发送了 `X-Request-Id: pd-0001`，用下面的命令搜索各轮归档。输出的文件名会标明请求出现在哪一轮。

```bash
rg -H -F 'pd_request_timing request_id=pd-0001 ' proxy.*.log
```

当 `kv_status=observed` 时，`kv_handoff_ms` 是这条请求的交接耗时。若显示 `NA`，可从 `kv_status` 定位原因。

- 遇到 `missing_decode`，先确认该请求有需要交接的 KV，再查 D 的日志。UCM 同步加载查 `ucm_pd_sync_load_done`，UCM 异步加载和 Mooncake pull 查 `Finished recving KV transfer`。
- `rank_set_mismatch` 表示同步加载的 worker 回报没有配齐。检查 D 的日志路径是否覆盖服务该请求的全部 worker。
- `no_external_load` 表示所有 worker 都报告本次无需加载，此请求没有可统计的 KV 交接耗时。

## 其他启动方式

- **旧版 UCM 同步日志。** D 的 `ucm_pd_sync_load_done` 没有 `workers` 字段时，在 proxy 命令里加 `--timing-sync-ranks`，列出实际加载的 rank，例如 `--timing-sync-ranks 0 1`。省略时，这类样本会标记为 `missing_worker_count`。
- **D 直接使用内联配置，或启动时覆盖了 YAML 的开关。** 将 `--ucm-config /etc/ucm/pd.yaml` 替换为 `--ucm-pd --timing-backend ucm-sync` 或 `--ucm-pd --timing-backend ucm-async`，按 D 实际采用的模式选择。
- **只做请求转发。** UCM 共享存储使用下面的命令。Mooncake 直传时去掉 `--ucm-pd`。这两种启动方式都无需提供引擎日志。

```bash
python ucm/pd/ucm_pd_disaggregation_example.py \
  --ucm-pd --port 9000 --workers 2 \
  --prefiller-hosts 192.0.2.10 --prefiller-ports 8100 \
  --decoder-hosts 192.0.2.20 --decoder-ports 8200
```
