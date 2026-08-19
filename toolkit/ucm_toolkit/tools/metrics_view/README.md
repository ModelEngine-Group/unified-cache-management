# metrics-view

采集 Prometheus/OpenMetrics `/metrics` 样本到 SQLite，并在终端查询聚合后的 UCM/vLLM 指标。适合在没有 Prometheus/Grafana 的环境中观察分层 KV cache 命中率、带宽等指标。

← 返回 [UCM Toolkit 顶层文档](../../../README.md)

## 依赖

仅依赖 Python 标准库（`sqlite3` 内置）；采集需要可访问的 Prometheus/OpenMetrics `/metrics` HTTP 接口。

## 模型与分层 KV cache 命中率

要获取正确的分层 KV cache 命中率数据，需要根据模型设置参数。GQA/MHA 模型无需设置，直接使用默认值；MLA 模型使用 `--config-param tp_size=<实际TP>` 设置服务实际 TP。

| 模型类型 | 参数 |
| --- | --- |
| GQA/MHA | 使用默认值 |
| MLA | 传入 `--config-param tp_size=<实际TP>` |

列出内置配置：

```bash
ucm-toolkit run metrics-view list-configs
```

查询 metrics 有两种方式：**方式一** `check` 即时拉取一次快照，适合快速看总量；**方式二** `start`/`query` 后台持续采集到 SQLite，适合查看时间窗口内的变化趋势。

## 方式一：check —— 查看启动以来的统计

`check` 会直接拉取一次当前 `/metrics` 快照并输出总聚合值，获取的是从服务启动到现在的总计值。

输入示例，以 mla 模型 tp=8 为例：

```bash
ucm-toolkit run metrics-view check \
  --url http://127.0.0.1:35325/metrics \
  --config metrics_lite \
  --config-param tp_size=8
```

输出示例：

```text
metric                           values                                   unit
-------------------------------  ---------------------------------------  -----
total_requests                   requests=1.000
e2e_request_latency_s            p50=0.150 p90=0.270 p99=0.297 avg=0.060  s
ttft_s                           p50=0.150 p90=0.270 p99=0.297 avg=0.060  s
tpot_s                           p50=0.005 p90=0.009 p99=0.010 avg=0.000  s
hbm_hit_rate                     hit_rate=0.333                           ratio
cache_hit_rate                   hit_rate=0.333                           ratio
posix_hit_rate                   hit_rate=0.333                           ratio
cache_store_load_bandwidth_gbps  gbps=0.001                               GB/s
cache_store_dump_bandwidth_gbps  gbps=0.000                               GB/s
posix_store_load_bandwidth_gbps  gbps=2.949e-04                           GB/s
posix_store_dump_bandwidth_gbps  gbps=0.000                               GB/s
```

使用 `check` 的话 gbps 不具备参考性，因为逻辑是总数据传输量/时间，不是瞬时值。需要看到合理的带宽需要使用如下 `start` 的方式来后台持续拉取 metrics。

## 方式二：后台采集和查询

推荐使用 `start` / `stop` 方式后台采集 metrics。后台采集会把多次样本写入 SQLite，因此后续可以查询指定时间范围内的数据。

后台启动采集（采集的时候无需配置 config）：

```bash
ucm-toolkit run metrics-view start \
  --url http://prefill:8000/metrics \
  --url http://decode:8000/metrics \
  --interval 5s
```

`--url` 可以重复传入。每个样本都会增加 `url=<完整 metrics URL>` label，用于区分 PD 分离部署中的不同实例。某个 URL 临时抓取失败时，后台进程会记录错误，其他 URL 仍会正常采集。

查看或停止后台采集：

```bash
ucm-toolkit run metrics-view status
ucm-toolkit run metrics-view stop
```

按时间窗口查询。推荐使用 `--aggr-by`，例如 `--window 10m --aggr-by 1m` 会展示最新 10 分钟的数据，每分钟聚合一份结果。MLA 模型需要按实际 TP 覆盖配置参数：

```bash
ucm-toolkit run metrics-view query \
  --window 10m \
  --aggr-by 1m \
  --config metrics_lite \
  --config-param tp_size=8
```

`query` 也支持使用 `--tag` 按 Prometheus label 过滤：

```bash
ucm-toolkit run metrics-view query \
  --start-time 2026-06-25T10:00:00 \
  --window 10m \
  --aggr-by 1m \
  --tag url=http://prefill:8000/metrics \
  --tag model_name=qwen \
  --tag worker_id=0
```

默认数据库、PID 和日志分别为 `/tmp/ucm_metrics.db`、`/tmp/ucm_metrics.pid` 和 `/tmp/terminal_metrics.log`，因此切换工作目录后仍可以执行 `status` 和 `stop`。如需同时运行多个采集进程，必须分别指定不同的 `--db`、`--pid-file` 和 `--log-file`。

如果需要使用其它数据库文件，可以显式指定 `--db`：

```bash
ucm-toolkit run metrics-view query \
  --db /tmp/another_ucm_metrics.db \
  --window 10m \
  --aggr-by 1m
```

清空 metrics 数据库使用 `metrics-view` 自己的 `clean` 子命令：

```bash
ucm-toolkit run metrics-view clean
```
