# Operations

First establish that requests complete, then verify external reuse, then investigate performance and failures. Service health describes the engine, a cache hit describes matching, and transfer/error metrics describe storage operations.

| Question | Guide |
| --- | --- |
| Is external reuse actually enabled? | [Verify external cache](verify-cache.md) |
| How do I collect and view metrics? | [Metrics setup](metrics.md) |
| Why has a backend stopped accepting cache work? | [Storage health](health-metrics.md) |
| How can I estimate prefix reuse opportunities? | [Trace mode](../diagnostics/trace-mode.md) |
| What should I check after an error or failed initialization? | [Troubleshooting](../../reference/troubleshooting.md) |
| What are the units, labels and denominators? | [Metric semantics](metrics-reference.md) |

Compare latency and throughput under the same workload. The [toolkit](../../toolkit/index.md) supports ad hoc metric and network inspection.
