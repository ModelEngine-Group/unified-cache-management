# metrics-view

Collects Prometheus/OpenMetrics `/metrics` samples into SQLite and queries aggregated UCM/vLLM metrics in the terminal. Suitable for observing tiered KV cache hit rates, bandwidth, and other metrics in environments without Prometheus/Grafana.

← Back to [UCM Toolkit](../index.md)

## Dependencies

Only depends on the Python standard library (`sqlite3` built-in); collection requires an accessible Prometheus/OpenMetrics `/metrics` HTTP endpoint.

## Model and Tiered KV Cache Hit Rate

To obtain correct tiered KV cache hit rate data, parameters must be set according to the model. GQA/MHA models need no configuration and use default values; MLA models require `--config-param tp_size=<actual TP>` to set the service's actual TP.

| Model Type | Parameter |
| --- | --- |
| GQA/MHA | Use default values |
| MLA | Pass `--config-param tp_size=<actual TP>` |

List built-in configurations:

```bash
ucm-toolkit run metrics-view list-configs
```

Built-in configurations and their Grafana dashboard mappings:

| Metrics View Config | Grafana Dashboard |
| --- | --- |
| `metrics_lite` | Compact set of commonly used metrics |
| `vllm` | `examples/metrics/grafana_vllm.json` |
| `connector` | `examples/metrics/grafana_connector.json` |
| `store` | `examples/metrics/grafana_store.json` |

`vllm`, `connector`, and `store` display all data from the corresponding Grafana dashboard. Single-curve data uses the panel name; multi-curve data uses `<panel_name>: <curve_name>`.

There are two ways to query metrics: **Method 1** `check` pulls an instant snapshot, suitable for quick total values; **Method 2** `start`/`query` continuously collects to SQLite in the background, suitable for observing trends over a time window.

## Method 1: check — View Statistics Since Startup

`check` pulls a single current `/metrics` snapshot and outputs aggregate totals, representing values from service startup to now.

Example, using an MLA model with tp=8:

```bash
ucm-toolkit run metrics-view check \
  --url http://127.0.0.1:35325/metrics \
  --config metrics_lite \
  --config-param tp_size=8
```

Output example:

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

With `check`, the gbps values are not meaningful because the logic computes total data transferred / time, not instantaneous values. To see reasonable bandwidth, use the `start` method below for continuous background metrics collection.

## Method 2: Background Collection and Query

The `start` / `stop` method is recommended for background metrics collection. Background collection writes multiple samples to SQLite, enabling queries over specified time ranges.

Start background collection (no config needed during collection):

```bash
ucm-toolkit run metrics-view start \
  --url http://prefill:8000/metrics \
  --url http://decode:8000/metrics \
  --interval 5s
```

`--url` can be repeated. Each sample includes a `url=<full metrics URL>` label to distinguish different instances in PD disaggregated deployments. If a URL temporarily fails to scrape, the background process logs the error while other URLs continue collecting normally.

View or stop background collection:

```bash
ucm-toolkit run metrics-view status
ucm-toolkit run metrics-view stop
```

Query by time window. `--aggr-by` is recommended, e.g. `--window 10m --aggr-by 1m` shows the latest 10 minutes of data, aggregated per minute. MLA models require overriding the config parameter with the actual TP:

```bash
ucm-toolkit run metrics-view query \
  --window 10m \
  --aggr-by 1m \
  --config metrics_lite \
  --config-param tp_size=8
```

`query` also supports filtering by Prometheus label using `--tag`:

```bash
ucm-toolkit run metrics-view query \
  --start-time 2026-06-25T10:00:00 \
  --window 10m \
  --aggr-by 1m \
  --tag url=http://prefill:8000/metrics \
  --tag model_name=qwen \
  --tag worker_id=0
```

The default database, PID, and log files are `/tmp/ucm_metrics.db`, `/tmp/ucm_metrics.pid`, and `/tmp/terminal_metrics.log` respectively, so `status` and `stop` work even after changing directories. To run multiple collection processes simultaneously, each must specify different `--db`, `--pid-file`, and `--log-file`.

To use a different database file, specify `--db` explicitly:

```bash
ucm-toolkit run metrics-view query \
  --db /tmp/another_ucm_metrics.db \
  --window 10m \
  --aggr-by 1m
```

Clear the metrics database using the `clean` subcommand:

```bash
ucm-toolkit run metrics-view clean
```
