# Terminal View Metrics

This tool scrapes Prometheus/OpenMetrics text endpoints into SQLite and queries
the stored raw samples from a terminal. It is intended for UCM/vLLM test
deployments that do not have Prometheus and Grafana available.

## Usage

List bundled configs:

```bash
python test/terminal_view_metrics/metrics_cli.py list-configs
```

Start a background collector:

```bash
python test/terminal_view_metrics/metrics_cli.py start \
  --url http://127.0.0.1:8000/metrics \
  --interval 5s
```

The collector stores all scraped metrics by default. Config files are used by
`query` to choose which stored metrics to display.

Stop the collector:

```bash
python test/terminal_view_metrics/metrics_cli.py stop
```

Run a query:

```bash
python test/terminal_view_metrics/metrics_cli.py query \
  --config metrics_lite \
  --window 10m \
  --aggr-by 1m
```

```text
bucket                                       metric                group                         values                         unit
------------------------------------------  --------------------  ----------------------------  -----------------------------  ----
2026-06-25 10:00:00..2026-06-25 10:01:00  ucm:load_bytes_total  model_name=qwen,worker_id=0    rate=8.420                     GB/s
2026-06-25 10:01:00..2026-06-25 10:02:00  ucm:load_duration     model_name=qwen,worker_id=0    avg=31.200 p50=22.000 p90=80  ms
```

The default database is `/tmp/ucm_metrics.db`. Use `--db` only when you want a
different file.

Query a fixed historical window by providing the window start time:

```bash
python test/terminal_view_metrics/metrics_cli.py query \
  --config ucm_overview \
  --start-time 2026-06-25T10:00:00 \
  --window 10m \
  --aggr-by 1m \
  --tag model_name=qwen \
  --tag worker_id=0
```

`--start-time` accepts epoch seconds, epoch milliseconds, or a local ISO timestamp.
With `--start-time`, `--window` means `[start-time, start-time + window]`.
`--aggr-by` is the recommended display mode: it renders one row per time
bucket, metric, and label group. `--tag KEY=VALUE` filters Prometheus labels
before aggregation, and can be repeated for multiple labels.

## Query model

Counter metrics use positive deltas inside each `--aggr-by` time bucket.
`op=rate` divides that delta by elapsed seconds.

Classic histogram metrics store raw `_bucket`, `_sum`, and `_count` samples.
Queries compute bucket deltas inside each `--aggr-by` time bucket and then
estimate quantiles with Prometheus-style linear interpolation.

## Config shape

Configs can be JSON or YAML. JSON has no extra dependency. A metric entry looks
like this for direct histogram handling:

```json
{
  "name": "ucm:load_duration",
  "type": "histogram",
  "avg": true,
  "quantiles": [0.5, 0.9, 0.99],
  "unit": "ms",
  "group_by": ["model_name", "worker_id"]
}
```

For Grafana-style panel values, use `type: "promql"` with a PromQL-like
expression:

```json
{
  "name": "Cache Lookup Hit Rate",
  "type": "promql",
  "expr": "sum by (worker_id) (rate(ucm:cache_lookup_hit_blocks_total[$__rate_interval])) / (sum by (worker_id) (rate(ucm:cache_lookup_hit_blocks_total[$__rate_interval])) + sum by (worker_id) (rate(ucm:cache_lookup_miss_blocks_total[$__rate_interval])))",
  "value": "hit_rate",
  "group_by": ["worker_id"]
}
```

The local evaluator supports the Prometheus/Grafana patterns used by the bundled
dashboards: `rate`, `increase`, `sum by (...)`, arithmetic, `histogram_quantile`,
`clamp_min`, direct gauge selectors, `$__rate_interval`, and `${perWorker:raw}`.
Bundled panel configs include `metrics_lite`, `ucm_grafana_panels`, and
`vllm_grafana_panels`. `metrics_lite` mirrors the data points tracked by the
dev-toolkit lightweight metrics collector.
