# Custom Metrics
UCM supports custom metrics with bidirectional updates from both Python and C++ runtimes. The unified monitoring interface provides the ability to mutate stats across language boundaries through a shared metrics registry.

## Architecture Overview
The metrics consists of these components below:
- **monitor** : Central stats registry that manages all metric lifecycle operations (registration, creation, updates, queries)
- **observability.py** : Prometheus integration layer that handles metric exposition
- **metrics_config.yaml** : Declarative configuration that defines which custom metrics to register and their properties

## Getting Started
### Define Metrics in YAML
Prometheus provides three fundamental metric types: Counter, Gauge, and Histogram. UCM implements corresponding wrappers for each type. The method for adding new metrics is as follows; please refer to the [example YAML](https://github.com/ModelEngine-Group/unified-cache-management/blob/develop/examples/metrics/metrics_configs.yaml) for more detailed information. 
```yaml
log_interval: 5  # Interval in seconds for logging metrics

prometheus:
  multiproc_dir: "/vllm-workspace"  # Directory for Prometheus multiprocess mode

  metric_prefix: "ucm:"
  
  # Enable/disable metrics by category
  enabled_metrics:
    counters: true
    gauges: true
    histograms: true
  
  # Counter metrics configuration
  counters:
    - name: "received_requests"
      documentation: "Total number of requests sent to ucm"
  
  # Gauge metrics configuration
  gauges:
    - name: "external_lookup_hit_rate"
      documentation: "Hit rate of ucm lookup requests"
      multiprocess_mode: "livemostrecent"
  
  # Histogram metrics configuration
  histograms:
    - name: "load_requests_num"
      documentation: "Number of requests loaded from ucm"
      buckets: [1, 5, 10, 20, 50, 100, 200, 500, 1000]
```

### Use Monitor APIs to Update Stats
The monitor provides a unified interface for metric operations. Users only need to create stats and update them, while the observability component is responsible for fetching the stats and pushing them to Prometheus.
:::::{tab-set}
:sync-group: install

::::{tab-item} Python side interfaces
:selected:
:sync: py
**Lifecycle Methods**
- `create_stats(name)`: Create and initialize a registered stats object.

**Operation Methods**
- `update_stats(name, dict)`: Update specific fields of a specific stats object.
- `get_stats(name)`: Retrieve current values of a specific stats object.
- `get_stats_and_clear(name)`:  Retrieve and reset a specific stats object.
- `get_all_stats_and_clear()`: Retrieve and reset all stats objects.
- `reset_stats(name)`: Reset a specific stats object to initial state.
- `reset_all()`: Reset all stats registered in monitor.

**Example:** Using built-in ConnStats
```python
from ucm.shared.metrics import ucmmonitor

ucmmonitor.create_stats("ConnStats") # Create a stats obj

# Update stats
ucmmonitor.update_stats(
    "ConnStats",
    {"interval_lookup_hit_rates": external_hit_blocks / len(ucm_block_ids)},
)

```
See more detailed example in [test case](https://github.com/ModelEngine-Group/unified-cache-management/tree/develop/ucm/shared/test/example).

::::

::::{tab-item} C++ side interfaces
:sync: cc
**Lifecycle Methods**
- `CreateStats(const std::string& name)`: Create and initialize a registered stats object.

**Operation Methods**
- `UpdateStats(const std::string& name, const std::unordered_map<std::string, double>& params)`: Update specific fields of a specific stats object.
- `ResetStats(const std::string& name)`: Retrieve current values of a specific stats object.
- `ResetAllStats()`: Retrieve and reset a specific stats object.
- `GetStats(const std::string& name)`: Retrieve and reset all stats objects.
- `GetStatsAndClear(const std::string& name)`: Reset a specific stats object to initial state.
- `GetAllStatsAndClear()`: Reset all stats registered in monitor.

**Example:** Implementing custom stats in C++
UCM supports custom metrics by following steps:
- Step 1: linking the static library monitor_static
   ```c++
    target_link_libraries(xxxstore PUBLIC storeinfra monitor_static)
    ```
- Step 2: Create stats object using function **CreateStats**
- Step 3: Update using function **UpdateStats**

See more detailed example in [test case](https://github.com/ModelEngine-Group/unified-cache-management/tree/develop/ucm/shared/test/case).

::::
:::::