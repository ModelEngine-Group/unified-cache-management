# Custom Metrics
UCM supports custom metrics with bidirectional updates from both Python and C++ runtimes. The unified monitoring interface provides the ability to mutate stats across language boundaries through a shared metrics registry.

## Architecture Overview
The metrics consists of these components below:
- **metrics** : Defined in cpp. Central stats registry that manages all metric lifecycle operations (creation, updates, queries)
- **observability.py** : Prometheus integration layer that handles metric exposition
- **metrics_config.yaml** : Declarative configuration that defines which custom metrics to register and their properties

## Getting Started
### Step 1: Define Metrics in YAML
Prometheus provides three fundamental metric types: Counter, Gauge, and Histogram. UCM implements corresponding wrappers for each type. After defining new metric in yaml, it will be registered to Promethues automatically by below function:
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
      # Parameters for constructing a metric
      metric_kwargs = {
          "name": prometheus_name,
          "documentation": doc,
          "labelnames": self.labelnames,
          **default_kwargs,
          **{k: v for k, v in cfg.items() if k in default_kwargs},
      }
      # Store metric mapping: metric_name -> Union[Counter, Gauge, Histogram]
      self.metric_mappings[name] = metric_cls(**metric_kwargs)
```

Example of yaml below:
```yaml
# Prometheus Metrics Configuration
# This file defines which metrics should be enabled and their configurations
log_interval: 5  # Interval in seconds for logging metrics

multiproc_dir: "/vllm-workspace"  # Directory for Prometheus multiprocess mode

metric_prefix: "ucm:" 

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
  - name: "load_requests_total"
    documentation: "Number of requests loaded from ucm"
    buckets: [1, 5, 10, 20, 50, 100, 200, 500, 1000]
  - name: "load_blocks_total"
    documentation: "Number of blocks loaded from ucm"
    buckets: [0, 50, 100, 150, 200, 250, 300, 350, 400, 550, 600, 750, 800, 850, 900, 950, 1000]
  - name: "load_duration"
    documentation: "Time to load from ucm (ms)"
    buckets: [0, 50, 100, 150, 200, 250, 300, 350, 400, 550, 600, 750, 800, 850, 900, 950, 1000]
  - name: "load_speed"
    documentation: "Speed of loading from ucm (GB/s)"
    buckets: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 50, 60, 70, 80, 90, 100]
  - name: "interval_lookup_hit_rates"
    documentation: "Hit rates of ucm lookup requests"
    buckets: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
  - name: "s2d_bandwidth"
    documentation: "Band width of uc store task s2d, copy tensors from storage to device"
    buckets: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
  - name: "d2s_bandwidth"
    documentation: "Band width of uc store task d2s, copy tensors from device to storage"
    buckets: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
```
Please refer to the [example YAML](https://github.com/ModelEngine-Group/unified-cache-management/blob/develop/examples/metrics/metrics_configs.yaml) for more detailed information. 

### Step 2: Use Metrics APIs to Update Stats
The metrics provides a unified interface for metric operations. After defining metrics in yaml, users only need to link metrics/import ucmmetrics and update them in suitable position, while the observability component is responsible for fetching the stats and pushing them to Prometheus.
:::::{tab-set}
:sync-group: install

::::{tab-item} Python side interfaces
:selected:
:sync: py
**Example:** Using built-in ConnStats
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

**Example:** Implementing custom stats in C++
UCM supports custom metrics by following steps:
- Step 1: linking the static library metrics
   ```c++
    target_link_libraries(xxxstore PUBLIC storeinfra monitor_static)
    ```
- 
- Step 3: Update using function **UpdateStats**
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
::::
:::::

