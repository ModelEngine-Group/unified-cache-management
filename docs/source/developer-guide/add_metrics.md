# How to Add a New Metric

UCM metrics are defined in YAML, registered by `PrometheusStatsLogger`, updated
from Python or C++ code, and exported through vLLM's Prometheus `/metrics`
endpoint.

## Registration Model

The metrics config drives registration:

1. `PrometheusStatsLogger` reads the configured YAML file.
2. It creates `prometheus_client` Counter, Gauge, and Histogram objects with
   `model_name` and `worker_id` labels.
3. For histograms, it reads the Python Prometheus histogram bucket boundaries
   and passes those buckets to the C++ metrics library.
4. C++ stores counter, gauge, and histogram bucket deltas in per-thread double
   buffers.
5. The observability thread periodically calls `get_all_stats_and_clear()` and
   applies the deltas to `prometheus_client`.

Histograms are aggregated into buckets on the update path. They do not store raw
sample vectors, and there is no `histogram_max_length` setting. The C++ metrics
library appends a `+Inf` bucket during registration when needed.

## Step 1: Define the Metric in YAML

Add the metric to `examples/metrics/metrics_configs.yaml` or to the metrics
config used by your deployment.

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

Use these metric types as follows:

- Counter: pass positive increments.
- Gauge: pass the latest value.
- Histogram: pass one observation; UCM assigns it to the configured bucket.

Histogram buckets should be sorted in ascending order. You do not need to add
`+Inf` in YAML; it is added during registration.

## Step 2: Update the Metric

### Python

Import `ucmmetrics` and update the metric by name:

```python
from ucm.shared.metrics import ucmmetrics

ucmmetrics.update_stats("my_stage_duration_ms", cost_ms)
ucmmetrics.update_stats({"my_events_total": 1.0})
```

### C++

Link the `metrics` library in the target that emits the metric:

```cmake
target_link_libraries(xxxstore PUBLIC storeinfra metrics)
```

Include the metrics API and update the metric:

```c++
#include "metrics_api.h"

UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("my_stage_duration_ms"), costMs);
UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("my_events_total"), 1.0);
```

Use `NAME_TO_METRIC_ID("metric_name")` on C++ hot paths. It caches metric ID
resolution behind a function-local static object and avoids repeated metric name
hashing after the first successful resolution. The string overload is still
available for non-hot code paths.

Do not add separate `metrics_config` checks around metric updates. If a metric is
not registered, the cached ID path records the miss and returns quickly until the
registration epoch changes.

## Step 3: Add Dashboard Panels

If the metric should be visible in Grafana, add it to the dashboard that matches
its layer:

| Dashboard | Metric scope |
|-----------|--------------|
| `examples/metrics/grafana_connector.json` | Connector-level metrics. |
| `examples/metrics/grafana_pipeline_store.json` | Cache Store and Posix Store metrics. |
| `examples/metrics/grafana_layerwise.json` | Layerwise connector metrics. |

For CI dashboards, mirror the panel in the corresponding file under
`ci/grafana/`.

## Useful PromQL Patterns

Counter throughput:

```promql
rate(ucm:my_events_total{model_name="$model_name"}[$__rate_interval])
```

Histogram average:

```promql
sum(rate(ucm:my_stage_duration_ms_sum{model_name="$model_name"}[$__rate_interval]))
/
sum(rate(ucm:my_stage_duration_ms_count{model_name="$model_name"}[$__rate_interval]))
```

Histogram quantile:

```promql
histogram_quantile(
  0.99,
  sum by (le) (
    rate(ucm:my_stage_duration_ms_bucket{model_name="$model_name"}[$__rate_interval])
  )
)
```

When a dashboard supports the `View` selector, keep the existing
`${perWorker:raw}` grouping pattern so Aggregated and Per Worker modes continue
to work.

## Implementation Notes

- `ucmmetrics.set_up()` is called by `PrometheusStatsLogger`; the old
  `max_vector_len` argument is kept only for API compatibility and is ignored.
- `ucmmetrics.create_stats(name, metric_type, buckets)` is normally called by
  `PrometheusStatsLogger`, not by metric emitters.
- `get_all_stats_and_clear()` returns counter, gauge, and histogram deltas.
  Python applies those deltas to `prometheus_client`, which owns the cumulative
  Prometheus exposition and multiprocess files.
- Histogram deltas are returned to Python as `(bucket_counts, sum_delta)`.
