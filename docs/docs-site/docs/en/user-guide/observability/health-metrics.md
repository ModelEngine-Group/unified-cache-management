# Store health and circuit breaking

A responding inference server can still have an unusable cache backend. Store
health tells you whether a Pipeline stage is accepting new cache operations.
It does not establish that a particular KV block exists, that a model answer is
correct, or that a deployment meets its latency target.

## What happens when a backend fails

Pipeline wraps each loaded stage in a `HealthBreakerStore` when health checking
is enabled. The wrapper starts enabled and records a rolling window of probe
results. With the defaults, two failures in the window block the stage; recovery
requires a full window of eight successful results.

Posix and Mooncake also trigger circuit breaking on actual I/O failures,
independently of active probes. Recovery requires an elapsed cooldown; repeated
failures shortly after recovery extend it, while sustained health resets the backoff.
Logs report passive windows with failures and remaining cooldown when probes are healthy.

| Operation while blocked | Result |
| --- | --- |
| `Lookup` | A miss for every requested block |
| `LookupOnPrefix` / `LookupOnReverse` | No hit (`-1`) |
| `Prefetch` | No new prefetch is submitted |
| `Load` / `Dump` | `StoreUnhealthy` is returned |
| `Check` / `Wait` for an existing task | Forwarded to the underlying Store |

Blocking new work does not cancel an already-submitted transfer. Whether a cache
miss leads to recomputation or an error is decided by the integration and request
path; the breaker itself does not retry inference requests.

The base `StoreV1.CheckHealth()` returns success. Posix and Mooncake override it
with backend operations. An enabled wrapper around a stage without such an
override is therefore not an independent check of that stage's storage service.

## Set the probe policy

`store_health` belongs inside `ucm_connector_config`. For example, a
`Cache|Posix` configuration can explicitly set the default health policy:

```yaml
ucm_connectors:
  - ucm_connector_name: UcmPipelineStore
    ucm_connector_config:
      store_pipeline: "Cache|Posix"
      storage_backends: /mnt/ucm-cache
      store_health:
        enabled: true
        health_check_interval_s: 10
        health_check_timeout_s: 3
        health_window_size: 8
        failure_threshold: 2
enable_metrics: true
```

Keep the model, mount and cache settings appropriate to your deployment; see
[Pipeline Store](../../developer-guide/cache-configuration/pipeline.md).

| Field | Meaning |
| --- | --- |
| `enabled` | Create the health wrapper and probe thread; default `true` |
| `health_check_interval_s` | Target probe interval; default 10 seconds |
| `health_check_timeout_s` | Probe execution deadline; default 3 seconds |
| `health_window_size` | Number of recent results retained; default 8 |
| `failure_threshold` | Failures needed to block new operations; default 2 |

Numeric values must be positive, the failure threshold cannot exceed the window,
and the timeout must be shorter than the interval. The first probe is delayed by
one interval plus random jitter of up to another interval. The initial enabled
state is published before that first probe; it is not evidence of a successful I/O.

Setting `enabled: false` removes this protection. It does not fix a backend failure
or turn off the backend's own error handling.

## Understand what each probe exercises

**Posix** checks every path selected by its storage layout. It creates a temporary
file, writes 4096 bytes, reads and compares them, and removes the file. Buffered
I/O also calls `Sync`; Direct I/O uses the configured direct-open flag. A failed
open, transfer, sync, comparison or removal makes the probe fail. For an NFS-backed
path this exercises the mounted filesystem from that UCM process.

**Mooncake** writes an eight-byte test value under a dedicated key, retrieves and
compares it, then removes it. The implementation uses the real client on the
transfer path and the RPC client on the scheduler path. This checks the configured
client path; it is not a test of every model's KV layout or every remote replica.

Health checks produce their own small objects. Do not count those files or keys as
proof that a request saved reusable KV data.

## Inspect state before aggregating

The default vLLM connector export uses these metrics:

| Backend | State Gauge | Probe Counters |
| --- | --- | --- |
| Posix | `ucm:posix_store_health` | `ucm:posix_healthy_count_total`, `ucm:posix_unhealthy_count_total` |
| Mooncake | `ucm:mooncake_store_health` | `ucm:mooncake_healthy_count_total`, `ucm:mooncake_unhealthy_count_total` |

The Gauge is 1 while the wrapper accepts work and 0 while it is blocked. Counters
record probe outcomes, including timeouts, rather than breaker transitions.
A single successful probe need not change a blocked Gauge back to 1.

`ucm:posix_passive_failures_total` and `ucm:mooncake_passive_failures_total`
count observed I/O failures separately from active probes.

Start with individual series and their labels:

```promql
ucm:posix_store_health{job="vllm"}
```

Then inspect failures over a recent window:

```promql
increase(ucm:posix_unhealthy_count_total{job="vllm"}[5m])
```

These queries assume your scrape job is named `vllm`. Preserve `instance`, model,
engine and `worker_rank` labels when locating a failure. Scheduler observations use
`worker_rank="scheduler"`. Multiple processes can probe the same backend, and the
exported labels do not identify every physical mount or underlying pipeline
object; series counts are not counts of failed disks.

The native probe thread and Prometheus scrape run on different schedules. In the
vLLM connector path, native statistics reach the exporter through
`get_kv_connector_stats()`. Check that collection is advancing before interpreting
a repeated value or a missing series as the current backend state.

## Investigate and confirm recovery

1. Check the scrape target and identify the affected process from the labels.
2. Find `Store health check` failures and `transitioned to UNHEALTHY` in its log;
   the log includes the pipeline stage identifier, cause and cooldown.
3. For Posix, inspect that process's mount, permissions, available capacity and
   read/write/remove errors. For Mooncake, inspect its configured client and
   metadata/master connectivity and the reported operation error.
4. Restore the failed dependency, then watch successful probes replace the failing
   window. After cooldown, confirm `transitioned to HEALTHY` and the corresponding Gauge update.
5. Separately repeat the [external-cache verification](../quick_start/index.md#vllm-verify-the-service-and-external-cache).
   Recovery of a probe does not prove recovery of a particular request's cache.

For metric units and export paths, see [Metrics reference](metrics-reference.md).
The policy and operation behavior are defined in
[`StoreHealthConfig`](https://github.com/ModelEngine-Group/unified-cache-management/blob/a336d69bc03a550d44bee3df9da7664e9edfe3a7/ucm/store/pipeline/cc/store_health_config.h)
and [`HealthBreakerStore`](https://github.com/ModelEngine-Group/unified-cache-management/blob/a336d69bc03a550d44bee3df9da7664e9edfe3a7/ucm/store/pipeline/cc/health_breaker_store.cc).
