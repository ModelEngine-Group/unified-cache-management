# 指标开发

新增指标时，先定义要观察的操作、单位和聚合范围，再在该操作实际发生的位置更新数据。UCM 的原生统计由统一 Dispatcher 分发，默认通过 vLLM KV Connector 导出到引擎的 `/metrics`。

## 默认采集链路

1. `UCMConnector._setup_ucm_metrics()` 读取启动配置。没有自定义目录时使用内置定义；内联 `metrics_config` 或 `metrics_config_path` 提供完整替代配置。
2. `setup_ucm_metrics()` 从定义中注册原生 Counter、Gauge 和 Histogram。Histogram 的桶边界来自指标配置。
3. Python 或 C++ 在操作发生时调用更新接口，原生库累积统计。
4. `MetricsDispatcher` 读取原生增量，分别保存在启用的 consumer 缓冲中，避免多个 consumer 互相取走数据。
5. vLLM 调用 `get_kv_connector_stats()`，UCM 取出 `vllm_connector` 缓冲，经 Connector 统计对象交给 vLLM 的 Prometheus 桥接层。

默认导出标签包含引擎标签和 `worker_rank`。另一个可选路径是 `multiproc` consumer 与 `PrometheusStatsLogger`，它使用周期 logger 和 `worker_id`；不要把该路径描述成默认 Connector 的注册与导出过程。

## 定义一个指标

运行时自定义指标写入实际使用的指标 YAML；向默认集合添加指标时，同步更新 `examples/metrics/metrics_configs.yaml` 和 `ucm/default_metrics_config.py`，保留两者一致。现有 `test_default_metrics_config_matches_example_yaml` 用例检查这一关系。

以下片段展示定义格式，需放入完整指标配置中：

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

Counter 更新正增量，Gauge 更新当前值，Histogram 更新一次观测。桶边界按升序配置，注册时按需补充 `+Inf`。为一个指标固定事件范围和单位；接口调用次数、传输分片和用户请求数不能互换。

## 在操作完成处更新

Python 示例中的 `cost_ms` 由调用方测量：

```python
from ucm.shared.metrics import ucmmetrics

ucmmetrics.update_stats("my_stage_duration_ms", cost_ms)
ucmmetrics.update_stats({"my_events_total": 1.0})
```

C++ target 链接 `metrics`，在已包含 UCM 头文件路径的构建中使用：

```cmake
target_link_libraries(xxxstore PUBLIC storeintf metrics)
```

```cpp
#include "metrics_api.h"

UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("my_stage_duration_ms"), costMs);
UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("my_events_total"), 1.0);
```

`NAME_TO_METRIC_ID` 缓存 ID，适合热路径。产生指标的代码不需要再次创建或注册指标，也不要直接读取并清空全局统计，以免绕过 Dispatcher。

若测量“成功完成”，应在成功路径更新；提交时计数应明确命名为提交。错误和中止是否纳入耗时、字节数，必须写入定义或[指标参考](../user-guide/observability/metrics-reference.md)。

## 加载指标配置

复制完整指标目录并加入上面的定义，再让服务读取该配置。以下片段放在现有 UCM YAML 根级，不替换 Store 配置。重启引擎注册更新后的指标目录，再触发对应操作。

```yaml
enable_metrics: true
metrics_config_path: /etc/ucm/metrics_configs.yaml
```

## 验证导出与展示

加载包含新定义的配置，触发对应操作，再在 `/metrics` 中检查实际名称、单位、标签和增量。默认前缀为 `ucm:`，consumer 配置可能改变名称和比例。没有发生操作或未采集 Connector 统计时，指标可能缺失或保持不变。

例如查看 Counter 的速率：

```promql
rate(ucm:my_events_total[5m])
```

为直方图计算分位数时使用 bucket，聚合保留 `le`。PromQL 的其他口径见[指标参考](../user-guide/observability/metrics-reference.md)。需要可视化时更新与职责对应的 `grafana_connector.json`、`grafana_store.json` 或 `grafana_vllm.json`，并检查已有模型、实例和 worker 选择器。

源码依次阅读 `ucm/metrics_config.py`、`ucm/metrics_dispatcher.py`、`ucm/integration/vllm/ucm_connector.py` 和 `ucm/integration/vllm/metrics.py`。指标采集部署见[用户指南](../user-guide/observability/metrics.md)。

### Dashboard 查询示例

以下沿用原指南的 Grafana 模型与时间窗口变量。按 worker 展示时，除 `le` 外还应保留 dashboard 的 worker 分组。

Counter 速率：

```promql
rate(ucm:my_events_total{model_name="$model_name"}[$__rate_interval])
```

Histogram 平均值：

```promql
sum(rate(ucm:my_stage_duration_ms_sum{model_name="$model_name"}[$__rate_interval]))
/
sum(rate(ucm:my_stage_duration_ms_count{model_name="$model_name"}[$__rate_interval]))
```

Histogram 分位数：

```promql
histogram_quantile(
  0.99,
  sum by (le) (
    rate(ucm:my_stage_duration_ms_bucket{model_name="$model_name"}[$__rate_interval])
  )
)
```
