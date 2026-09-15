# Metrics development

Define the operation, unit and aggregation scope before adding a metric at the point where that operation occurs. A shared Dispatcher distributes native UCM statistics; the default path exports them through vLLM's KV Connector to the engine's `/metrics` endpoint.

## Default collection path

1. `UCMConnector._setup_ucm_metrics()` reads launch configuration. Without a custom catalog it uses built-in definitions. Inline `metrics_config` or `metrics_config_path` supplies a complete replacement.
2. `setup_ucm_metrics()` registers native Counters, Gauges and Histograms from definitions, including configured bucket boundaries.
3. Python or C++ updates the native statistics where operations occur.
4. `MetricsDispatcher` drains native increments into separate buffers for enabled consumers, preventing consumers from consuming one another's data.
5. When vLLM calls `get_kv_connector_stats()`, UCM retrieves the `vllm_connector` buffer and passes Connector statistics to vLLM's Prometheus bridge.

Default labels include engine labels and `worker_rank`. The optional `multiproc` consumer uses `PrometheusStatsLogger`, a periodic logger and `worker_id`. This is a separate path, not the default Connector registration/export process.

## Define a metric

Put runtime custom metrics in the deployed metrics YAML. For additions to the default set, update both `examples/metrics/metrics_configs.yaml` and `ucm/default_metrics_config.py`. The existing `test_default_metrics_config_matches_example_yaml` checks their consistency.

This fragment illustrates definitions and belongs inside a complete metrics configuration:

```yaml
counter:
  - name: "my_events_total"
    documentation: "Total number of events"

gauge:
  - name: "my_current_value"
    documentation: "Most recent value"
    multiprocess_mode: "livemostrecent"

histogram:
  - name: "my_stage_duration_ms"
    documentation: "Stage duration in milliseconds"
    buckets: [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]
```

Counters take positive increments, Gauges take current values and Histograms take observations. Configure buckets in ascending order; registration adds `+Inf` when needed. Keep a fixed event scope and unit: interface calls, transfer shards and user requests are different quantities.

## Update at the operation boundary

The caller measures `cost_ms` in this Python example:

```python
from ucm.shared.metrics import ucmmetrics

ucmmetrics.update_stats("my_stage_duration_ms", cost_ms)
ucmmetrics.update_stats({"my_events_total": 1.0})
```

Link the C++ target to `metrics` and use the API with UCM include paths configured:

```cmake
target_link_libraries(xxxstore PUBLIC storeintf metrics)
```

```cpp
#include "metrics_api.h"

UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("my_stage_duration_ms"), costMs);
UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("my_events_total"), 1.0);
```

`NAME_TO_METRIC_ID` caches the ID for hot paths. Producers do not need to create/register metrics again and must not independently drain global statistics around the Dispatcher.

Update a successful-completion metric only on success. Name submission metrics accordingly. Explain whether durations or bytes include errors and cancellations in the definition or [metric reference](../user-guide/observability/metrics-reference.md).

## Load the metric configuration

Copy the complete metrics catalog, add the definitions above, and point the serving configuration to it. The fragment below belongs at the root of the existing UCM YAML; it does not replace the Store configuration. Restart the engine to register the updated catalog before producing values.

```yaml
enable_metrics: true
metrics_config_path: /etc/ucm/metrics_configs.yaml
```

## Verify export and presentation

Load a configuration containing the definition, trigger the operation and check the actual exported name, unit, labels and delta. The default prefix is `ucm:`; consumer configuration can change names and scales. A metric may be absent or unchanged when its operation has not run or Connector statistics have not been collected.

For example, query the Counter rate:

```promql
rate(ucm:my_events_total[5m])
```

Use buckets for histogram quantiles and preserve `le` when aggregating. See [metric semantics](../user-guide/observability/metrics-reference.md) for other queries. For dashboards, update the appropriate `grafana_connector.json`, `grafana_store.json` or `grafana_vllm.json`, retaining model, instance and worker selectors.

Read `ucm/metrics_config.py`, `ucm/metrics_dispatcher.py`, `ucm/integration/vllm/ucm_connector.py` and `ucm/integration/vllm/metrics.py` in order. Collection setup belongs in the [user guide](../user-guide/observability/metrics.md).

### Dashboard query examples

The following use the Grafana model and interval variables from the original guide. For per-worker views, retain the dashboard's worker grouping in addition to `le`.

Counter throughput：

```promql
rate(ucm:my_events_total{model_name="$model_name"}[$__rate_interval])
```

Histogram average：

```promql
sum(rate(ucm:my_stage_duration_ms_sum{model_name="$model_name"}[$__rate_interval]))
/
sum(rate(ucm:my_stage_duration_ms_count{model_name="$model_name"}[$__rate_interval]))
```

Histogram quantile：

```promql
histogram_quantile(
  0.99,
  sum by (le) (
    rate(ucm:my_stage_duration_ms_bucket{model_name="$model_name"}[$__rate_interval])
  )
)
```
