# UCM Health Metrics

UCM Pipeline Store can enable health probes and circuit breaking for individual Store stages. Health metrics answer two different questions:

- How many health probes succeeded or failed during a time window?
- Is a Store currently blocked from accepting new requests by its circuit breaker?

Counters and Gauges represent these two kinds of information and require different aggregation methods. This guide describes the Posix Store and Mooncake Store probes, metric semantics, and recommended PromQL aggregation.

## 1. Health Probes

Each UCM connector instance independently probes and maintains a circuit breaker for every Store that supports health checks. A worker Store uses a numeric `worker_rank`, while a scheduler Store uses `worker_rank="scheduler"`. Although the scheduler is not a distributed rank, it owns a Store and is therefore included in the Store count.

Only the following remote Stores currently implement health checks and circuit breaking. Other Stores are unaffected.

### 1.1 Posix Store

A Posix health probe performs a complete small-file I/O operation on every health-check path: it creates a file, writes 4 KiB of test data, synchronizes the data when required, reads and verifies the data, and finally removes the file. A failure to open, read, write, synchronize, verify, or remove the file on any path causes the probe to fail.

The probe therefore validates the actual I/O path rather than only checking whether a directory exists. For a remote file system such as NFS, it can also detect mount, network, and remote-storage failures.

### 1.2 Mooncake Store

A Mooncake health probe uses a dedicated temporary key to perform a small Put, Get, content verification, and Remove sequence. The probe fails when the client is unavailable, an operation fails, or the returned content does not match.

## 2. Health Metrics

The default configuration contains six health metrics:

| Metric | Type | Meaning | Update |
| --- | --- | --- | --- |
| `ucm:posix_healthy_count_total` | Counter | Successful Posix health probes | Incremented by 1 after a successful Posix probe |
| `ucm:posix_unhealthy_count_total` | Counter | Failed or timed-out Posix health probes | Incremented by 1 after a failed Posix probe |
| `ucm:posix_store_health` | Gauge | Effective Posix circuit-breaker state: 1 is available and 0 is fused | Updated at startup and after every Posix probe |
| `ucm:mooncake_healthy_count_total` | Counter | Successful Mooncake health probes | Incremented by 1 after a successful Mooncake probe |
| `ucm:mooncake_unhealthy_count_total` | Counter | Failed or timed-out Mooncake health probes | Incremented by 1 after a failed Mooncake probe |
| `ucm:mooncake_store_health` | Gauge | Effective Mooncake circuit-breaker state: 1 is available and 0 is fused | Updated at startup and after every Mooncake probe |

Use the Gauge to determine whether a Store is currently fused. Use both the success and failure Counters to analyze probe quality over time. There is currently no dedicated Counter for fuse or recovery transitions.

Metric names distinguish Store types, while labels distinguish vLLM instances and UCM processes. Connector metrics carry `model_name`, `engine`, and `worker_rank`; Prometheus also adds `job` and `instance`. See [UCM Metrics Observability](metrics.md) for the complete label definitions.

### 2.1 Synchronization Delay in Connector Mode

The health threads continue to run inside UCM at their configured interval, but connector metrics are synchronized to `/metrics` only when vLLM calls `get_kv_connector_stats()`. With no inference requests, the health metrics in Prometheus do not update even if the background probe result has changed.

## 3. Recommended Aggregation

The examples below omit some selectors. In production, scope queries by at least `job`, `instance`, `model_name`, and `engine`, and filter `worker_rank` as needed.

- Scheduler only: `worker_rank="scheduler"`
- Workers only: `worker_rank!="scheduler"`
- No `worker_rank` filter: include the scheduler and all workers

### 3.1 View the Current State of Each Store

```promql
ucm:posix_store_health{
  job="vllm",
  instance="10.0.0.8:8000",
  model_name="Qwen3-32B"
}
```

A value of 1 means the Store corresponding to that `worker_rank` is available. A value of 0 means it is fused. This query is the most direct way to locate an unhealthy worker Store or scheduler Store.

### 3.2 Count Healthy and Fused Stores

Because the Gauge is either 0 or 1, use `sum` to count healthy Stores and `count - sum` to count fused Stores:

```promql
# Healthy Store count
sum by (job, instance, model_name, engine) (
  ucm:posix_store_health
)
```

```promql
# Fused Store count
clamp_min(
  count by (job, instance, model_name, engine) (
    ucm:posix_store_health
  )
  -
  sum by (job, instance, model_name, engine) (
    ucm:posix_store_health
  ),
  0
)
```

This is also the basic aggregation used by the health-state count panels in the UCM Overview dashboard. Calculate Posix and Mooncake counts separately so their states are not mixed.

#### Understanding the Store Count

| Deployment | Store count visible in metrics | Description |
| --- | ---: | --- |
| DP1, TP1 | 2 | One worker Store and one scheduler Store |
| DP1, multiple TP ranks | `TP + 1` | TP worker Stores and one scheduler Store |
| Multiple DP ranks | `DP × (TP + 1)` | Each DP rank creates its own worker Stores and scheduler Store |

For DeepSeek V4, each worker has two actual Stores, but their states are combined in the metrics and appear as one healthy or unhealthy value.

### 3.3 Calculate the Healthy Store Ratio

```promql
sum by (job, instance, model_name, engine) (
  ucm:posix_store_health
)
/
clamp_min(
  count by (job, instance, model_name, engine) (
    ucm:posix_store_health
  ),
  1
)
```

The numerator is the number of healthy Stores, and the denominator is the total number of reported Stores. Because `posix_store_health` is either 0 or 1, this expression is equivalent to applying `avg()` to the Gauge, but “healthy count divided by total count” makes the meaning explicit. For example, if two of eight Stores are fused, the result is 0.75.

Without a `worker_rank` filter, the scheduler Store is included in both the numerator and denominator. The two Stores in the DeepSeek V4/HMA/FAWA path are combined into one Gauge, so this query calculates the ratio of visible health states, not the exact health ratio of the underlying FA and WA Stores.

### 3.4 Calculate the Probe Failure Ratio

Aggregate successful and failed probe counts first, then calculate the ratio. Do not calculate a failure ratio for each Store and then take the arithmetic mean.

```promql
(
  sum by (job, instance, model_name, engine) (
    rate(ucm:posix_unhealthy_count_total[5m])
  )
  or
  0 * sum by (job, instance, model_name, engine) (
    rate({__name__=~"ucm:posix_(healthy|unhealthy)_count_total"}[5m])
  )
)
/
clamp_min(
  sum by (job, instance, model_name, engine) (
    rate({__name__=~"ucm:posix_(healthy|unhealthy)_count_total"}[5m])
  ),
  1e-12
)
```

This expression is weighted by the number of probes. The `or 0 * ...` term supplies zero before a failure series exists, preventing a healthy Store from displaying No data. To query Mooncake, replace the `posix` prefix with `mooncake`.

Use `increase()` to count failed probes over a time window:

```promql
(
  sum by (job, instance, model_name, engine) (
    increase(ucm:posix_unhealthy_count_total[15m])
  )
  or
  0 * sum by (job, instance, model_name, engine) (
    increase({__name__=~"ucm:posix_(healthy|unhealthy)_count_total"}[15m])
  )
)
```

`rate()` and `increase()` handle Counter resets caused by process restarts. Do not subtract raw Counter values or treat a Counter as the current health state.

The default probe interval is 10 seconds, but connector synchronization depends on requests. For low-traffic services, use a longer window, such as 5–15 minutes, to reduce fluctuations caused by delayed synchronization and small sample counts.

## 4. Multi-instance Aggregation

| Monitoring goal | Recommended | Not recommended |
| --- | --- | --- |
| Determine whether one Store is fused | Preserve `worker_rank` and inspect the Gauge | Sum the Gauge and interpret it as a Boolean |
| Determine whether any Store in an instance is fused | Apply `min` to the Gauge by instance | Apply `avg` and only check whether it is greater than zero |
| Count healthy/fused Stores in an instance | Use `sum` and `count - sum` | Accumulate Gauge states over time |
| Calculate the probe failure ratio over a window | Aggregate the Counter numerator and denominator, then divide | Calculate per-Store ratios and take their arithmetic mean |
| Count independent physical-backend failures | Deduplicate with backend identifiers, logs, or external monitoring | Directly sum failure Counters from all Stores |

To aggregate by cluster, node, or storage failure domain, add stable labels to the Prometheus target configuration:

```yaml
static_configs:
  - targets:
      - "10.0.0.8:8000"
    labels:
      cluster: "production-a"
      node: "inference-01"
      storage_domain: "posix-cluster-a"
```

Add these labels to `by (...)`. Aggregated results are meaningful only after confirming that the selected series belong to the same failure domain. UCM does not currently derive these labels from storage paths or endpoints.

## 5. Alerting Recommendations

### 5.1 A Posix Circuit Breaker Remains Fused

```promql
min by (job, instance, model_name, engine) (
  ucm:posix_store_health
) == 0
```

Configure a suitable `for` duration, such as 30 seconds, so a brief observation-side fluctuation does not immediately trigger a notification. The circuit breaker already filters individual probe failures through its sliding window, so the alert delay should not replace the breaker logic.

### 5.2 The Probe Failure Ratio Remains High

Use the failure ratio from section 3.4 and require enough failure samples in the window. For example, alert when the failure ratio exceeds 20% over 15 minutes and there are at least three failed probes. Tune the thresholds for the Store probe interval, Store count, and service tolerance.

## 6. Troubleshooting

When a Gauge is 0 or a failure Counter increases:

1. Locate the affected process by `instance`, `engine`, and `worker_rank`.
2. Search UCM logs for `Store health check` and `transitioned to UNHEALTHY/HEALTHY`.
3. For Posix, check mount state, directory permissions, free space, and read/write/remove operations.
4. For Mooncake, check the client, metadata/master services, network, and Put/Get/Remove path.
5. Confirm that requests are triggering connector metric synchronization and that the Prometheus target is UP.
6. After the backend recovers, verify consecutive successful probes and confirm that the Gauge returns to 1.

Import `examples/metrics/grafana_vllm.json` in Grafana to view healthy/fused Store counts and Posix/Mooncake probe trends in the bottom **Store Health Metrics** group.
