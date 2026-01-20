# How to Add A New Metric
UCM allows developers to add new metrics for monitoring service health status, and this doc provides the methods for adding new metrics.

## Getting Started
### Step 1: Define New Metrics in YAML
Prometheus provides three fundamental metric types: Counter, Gauge, and Histogram. UCM implements corresponding wrappers for each type. After defining new metric in yaml, it will be registered to Prometheus automatically by below function:
```python
def _register_metrics_by_type(self, metric_type):
        """
        Register metrics by different metric types.
        """
        metric_cls, default_kwargs = self.metric_type_config[metric_type]
        cfg_list = self.config.get(metric_type, [])

        for cfg in cfg_list:
            name = cfg.get("name")
            doc = cfg.get("documentation", "")
            # Prometheus metric name with prefix
            prometheus_name = f"{self.metric_prefix}{name}"
            ucmmetrics.create_stats(name, metric_type)

            metric_kwargs = {
                "name": prometheus_name,
                "documentation": doc,
                "labelnames": self.labelnames,
                **default_kwargs,
                **{k: v for k, v in cfg.items() if k in default_kwargs},
            }

            self.metric_mappings[name] = metric_cls(**metric_kwargs)
```

Example of yaml below:
```yaml
# Prometheus Metrics Configuration
# This file defines which metrics should be enabled and their configurations
log_interval: 5  # Interval in seconds for logging metrics

multiproc_dir: "/vllm-workspace"  # Directory for Prometheus multiprocess mode

metric_prefix: "ucm:" 

histogram_max_length: 10000  # Maximum length of the vector for each histogram metric

# Counter metrics configuration
# counter:
#   - name: "received_requests"
#     documentation: "Total number of requests sent to ucm"

# Gauge metrics configuration
# gauge:
#   - name: "lookup_hit_rate"
#     documentation: "Hit rate of ucm lookup requests since last log"
#     multiprocess_mode: "livemostrecent"

# Histogram metrics configuration
histogram:
  - name: "load_requests_num"
    documentation: "Number of requests loaded from ucm"
    buckets: [1, 5, 10, 20, 50, 100, 200, 500, 1000]
  - name: "d2s_bandwidth"
    documentation: "Band width of uc store task d2s, copy tensors from device to storage"
    buckets: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
  - name: "s2d_bandwidth"
    documentation: "Band width of uc store task s2d, copy tensors from storage to device"
    buckets: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
```
Please refer to the [example YAML](https://github.com/ModelEngine-Group/unified-cache-management/blob/develop/examples/metrics/metrics_configs.yaml) for more detailed information. 

### Step 2: Use Metrics APIs to Update Stats
After defining metrics in yaml, users only need to link metrics/import ucmmetrics and update them in suitable position, while `observability` component is responsible for fetching the stats.

:::::{tab-set}
:sync-group: install

::::{tab-item} Python side interfaces
:selected:
:sync: py
**Example:** Import the `ucmmetrics` and then use `update_stats` to update new metrics.
```python
# 1. Import ucmmetrics
from ucm.shared.metrics import ucmmetrics

# 2. Update a stat
ucmmetrics.update_stats(
  {"interval_lookup_hit_rates": external_hit_blocks / len(ucm_block_ids)},
)

# 2. Update stats
ucmmetrics.update_stats(
  {
      "load_requests_total": num_loaded_request,
      "load_blocks_total": num_loaded_block,
      "load_duration": load_end_time - load_start_time,
      "load_speed": load_speed,
  }
)
```
See more detailed example in [test case](https://github.com/ModelEngine-Group/unified-cache-management/tree/develop/ucm/shared/test/example).

::::

::::{tab-item} C++ side interfaces
:sync: cc

**Example:** UCM supports custom metrics by following steps:
- Step 1: linking the static library metrics
   ```c++
    target_link_libraries(xxxstore PUBLIC storeinfra metrics)
    ```
- Step 2: Update using function **UpdateStats**
```c++
// 1. Include metrics api head file
#include "metrics_api.h"

// 2. Update metrics defined in yaml
auto Epilog(const size_t ioSize) const noexcept
  {
      auto total = ioSize * number_;
      auto costs = NowTp() - startTp;
      auto bw = double(total) / costs / 1e9;
      switch (type)
      {
      case Type::DUMP:
          UC::Metrics::UpdateStats("d2s_bandwidth", bw);
          break;
      case Type::LOAD:
          UC::Metrics::UpdateStats("s2d_bandwidth", bw);
          break;
      default:
          break;
      }
      return fmt::format("Task({},{},{},{}) finished, costs={:.06f}s, bw={:.06f}GB/s.", id,
                          brief_, number_, total, costs, bw);
  }
```
See more detailed example in [test case](https://github.com/ModelEngine-Group/unified-cache-management/tree/develop/ucm/shared/test/case).
::::
:::::

## How to See New Metrics
After completing the above two steps, developers can view the newly added metrics via the /metrics endpoint.

Developers can also add a new panel in grafana.json to display the newly added metrics. Refer to [grafana example](https://github.com/ModelEngine-Group/unified-cache-management/tree/main/examples/metrics) for more information.
