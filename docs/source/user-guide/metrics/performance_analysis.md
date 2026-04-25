# Pipeline Store Performance Analysis (Cache | Posix)

This guide explains how to use the per-stage and layerwise metrics to
diagnose performance issues when running UCM with the
`pipelinestore("Cache|Posix")` backend, which is the most common
production configuration. It applies to both `UCMDirectConnector`
(non-layerwise) and `UCMLayerWiseConnector`.

It assumes you have already enabled the metrics configuration
(`metrics_config_path`) as described in `metrics.md`. Throughout this
document `ucm:` is the metric prefix; PromQL examples use the variable
`$model_name` (provided by the bundled Grafana dashboard).

---

## 1. Architecture and data flow

### 1.1 The four storage tiers

```
   ┌────────────────────────────────────────────────────────────┐
   │  vLLM Worker (forward pass)                                │
   │   ─ start_load_kv / wait_for_layer_load / save_kv_layer ─  │
   └────────────────────────────────────────────────────────────┘
                       │ store.load_data / dump_data
                       ▼
   ┌────────────────────────────────────────────────────────────┐
   │  CacheStore (in-memory tier, on host RAM)                  │
   │    LoadQueue  ─►  H2D  ─►  Device                          │
   │    DumpQueue  ◄─  D2H  ◄─  Device                          │
   │      │ cache miss / dump backflush                         │
   │      ▼                                                     │
   │  PosixStore (disk tier, /tmp, NFS, /mnt/...)               │
   │    TransQueue (worker pool)  ─►  S2H / H2S                 │
   └────────────────────────────────────────────────────────────┘
```

### 1.2 LOAD data flow (Cache miss case)

When a load request descends from Python all the way to disk, four
durations are captured separately. Knowing which one dominates is the
first step in any diagnosis.

```
 user calls store.load_data
          │
          ▼  ┐
 [waiting queue]                  pipeline_cache_load_wait_duration_ms
          │  ┘
          ▼  ┐
 DispatchOneTask                  pipeline_cache_load_dispatch_duration_ms
   │ buffer alloc                 (also: backend_submit_shards_total
   │ submit miss → Posix          /shards_total = miss ratio)
          │  ┘
          ▼
   ┌── Posix ──────────────────┐
   │ [posix waiting queue]     │  pipeline_posix_load_wait_duration_ms
   │ S2H worker reads disk     │  pipeline_posix_s2h_duration_ms
   │                           │  pipeline_posix_s2h_bandwidth_gbps
   └────────────┬──────────────┘
                ▼  ┐
 WaitBackendTaskReady             pipeline_cache_load_backend_wait_duration_ms
                │  ┘              (≈ posix wait + posix s2h)
                ▼  ┐
 H2D copy + stream sync           pipeline_cache_h2d_duration_ms
                │  ┘
                ▼
       waiter->Done() ─► epilog fires:
                                  pipeline_cache_load_duration_ms
                                  pipeline_cache_load_bandwidth_gbps
                                  pipeline_cache_load_blocks_total
```

For a Cache **hit**, the chain is shorter — there is no backend wait,
H2D runs straight from the in-memory buffer:

```
 wait → dispatch → H2D → done
```

So `pipeline_cache_load_backend_wait_duration_ms` near zero means
either every load hit Cache, or the buffer happened to already be
filled by a concurrent prefetch.

### 1.3 DUMP data flow

Dump is asymmetric: from the user's perspective `wait` returns as soon
as the Cache D2H copy and Posix submission complete. The actual disk
write happens later in `BackendDumpStage` and does **not** block the
caller.

```
 user calls store.dump_data
          │
          ▼  ┐
 [waiting queue]                  pipeline_cache_dump_wait_duration_ms
          │  ┘
          ▼  ┐
 D2H + stream sync                pipeline_cache_d2h_duration_ms
          │  ┘
          ▼  ┐
 backend_->Dump (submit only)     pipeline_cache_dump_backend_submit_duration_ms
          │  ┘
          ▼
 waiter->Done() ─► epilog fires:
                                  pipeline_cache_dump_duration_ms
                                  pipeline_cache_dump_bandwidth_gbps
                                  pipeline_cache_dump_blocks_total
          │
          ▼  (asynchronous, in BackendDumpStage thread)
   ┌── Posix ──────────────────┐
   │ H2S worker writes disk    │  pipeline_posix_h2s_duration_ms
   │                           │  pipeline_posix_h2s_bandwidth_gbps
   └───────────────────────────┘
```

The implication: **`pipeline_cache_dump_duration_ms` is what the user
felt; `pipeline_posix_h2s_duration_ms` is what the disk did.** They
can diverge by orders of magnitude when the Cache buffer is large
enough to absorb bursts. If they begin to track each other closely it
means the Cache buffer is full and back-pressure has propagated up to
the caller.

---

## 2. Critical metrics, ranked

When you sit down to diagnose a slow run, look at these in order. The
first one that's red usually points to the actual bottleneck.

| Rank | Metric | What it tells you |
|------|--------|-------------------|
| 1 | `ucm:pipeline_cache_lookup_hit_rate` (gauge) | Is the cache helping at all? |
| 2 | `ucm:layerwise_wait_blocking_ms` (layerwise only) | Is the load/forward overlap working? |
| 3 | `ucm:pipeline_cache_load_duration_ms` (avg or p99) | How slow are loads end-to-end? |
| 4 | `ucm:pipeline_cache_load_backend_wait_duration_ms` | If load is slow: is it Posix's fault? |
| 5 | `ucm:pipeline_posix_s2h_bandwidth_gbps` | If Posix is slow: is the disk the limit? |
| 6 | `ucm:pipeline_posix_load_wait_duration_ms` | Or is it queueing on the worker pool? |
| 7 | `ucm:layerwise_save_tail_total_ms` | Is dump tail eating budget? |
| 8 | `ucm:pipeline_cache_dump_duration_ms` vs `ucm:pipeline_posix_h2s_duration_ms` | Is the dump back-pressuring the caller? |

The rest are secondary signals used to confirm a hypothesis.

---

## 3. Diagnostic playbook

Each entry below is **symptoms → metric signature → likely cause →
tunables**. Numbers in brackets are illustrative thresholds — adjust
to your hardware.

### 3.1 Cache is barely helping (low hit rate)

**Symptoms.** Inference is no faster than running without UCM.
Recompute load is high.

**Metric signature.**
- `ucm:pipeline_cache_lookup_hit_rate` < 0.3
- `rate(ucm:pipeline_cache_load_backend_submit_shards_total) /
   rate(ucm:pipeline_cache_load_shards_total)` > 0.7
- `ucm:interval_lookup_hit_rates` (legacy, end-to-end) also low

**Likely causes.**
1. Cache buffer is too small — blocks evicted before reuse.
2. Workload diversity is too high — every prompt has unique prefixes.
3. Dumps aren't reaching Posix in time, so subsequent loads can't find
   the data — check `pipeline_posix_dump_failures_total` and the gap
   between `cache_dump_duration_ms` and `posix_h2s_duration_ms`.

**Tunables.**
- `cache_buffer_capacity_gb` ↑ (Cache config) — biggest win usually.
- `posix_capacity_gb` ↑ (give the disk tier room for the working set).
- Verify hash seed + tokenizer match between writes and reads (if you
  recently bumped the model, old cache won't be addressable).

### 3.2 Cache hit rate is OK but loads are still slow

**Symptoms.** Hit rate looks healthy but `load_duration` is high
relative to your model's forward time.

**Metric signature.** `pipeline_cache_load_duration_ms` is high.
Decompose by reading these in parallel:

| Component metric | Normal | High means |
|------------------|--------|------------|
| `pipeline_cache_load_wait_duration_ms` | < 1 ms | Load queue is saturated; too many concurrent requests |
| `pipeline_cache_load_dispatch_duration_ms` | < 1 ms | Dispatch is rarely the bottleneck; if high, suspect lock contention |
| `pipeline_cache_load_backend_wait_duration_ms` | depends on hit rate | Cache misses going to Posix — see §3.3 |
| `pipeline_cache_h2d_duration_ms` | bounded by PCIe bw | Rare; suspect device-side contention or wrong stream affinity |

**Tunables.**
- Wait high → `waiting_queue_depth` ↑ or reduce concurrency.
- Backend wait high → see §3.3.
- H2D high → check `cache_stream_number` and that pinned memory is
  actually pinned (compare H2D bandwidth to PCIe spec; pinned should
  hit > 20 GB/s on PCIe Gen4 x16).

### 3.3 Cache misses are slow because Posix is slow

**Symptoms.** `cache_load_backend_wait_duration_ms` dominates the load
chain.

**Metric signature.**
- `pipeline_cache_load_backend_wait_duration_ms` > `pipeline_cache_h2d_duration_ms`
- `pipeline_posix_s2h_bandwidth_gbps` well below disk spec
   (e.g. < 1 GB/s on an NVMe rated for 3 GB/s)
- AND/OR `pipeline_posix_load_wait_duration_ms` high (queue buildup)

**Decision split:**

| If … | Then root cause | Fix |
|------|-----------------|-----|
| Bandwidth low, wait low | Per-IO latency is bad — small IOs, no `O_DIRECT`, slow filesystem | Try `io_direct: true`, larger `tensor_size` / `shard_size` |
| Bandwidth low, wait high | Workers can't drain fast enough | `posix_data_trans_concurrency` ↑ |
| Bandwidth OK, wait high | Burst arrival exceeds steady throughput | Add Cache capacity to absorb bursts; throttle inbound rate |
| Bandwidth OK, wait low | Posix is fine; the issue is upstream | Re-check §3.2 components |

### 3.4 Dumps are silently overflowing (back-pressure)

**Symptoms.** Loads start failing or hit rate drops over time even
though the workload looks the same. New requests wait longer than
expected.

**Metric signature.** Two conditions together:
- `ucm:pipeline_cache_dump_duration_ms` (caller-felt) starts climbing
  toward `ucm:pipeline_posix_h2s_duration_ms` (disk-felt). When the
  Cache buffer is healthy these diverge sharply; when the buffer is
  saturated they converge.
- `ucm:pipeline_posix_h2s_bandwidth_gbps` ≪ Cache dump rate

**Likely cause.** The Posix tier cannot keep up with sustained dump
throughput. Cache buffer fills, BackendDumpStage blocks, dispatch
blocks on buffer allocation, caller blocks.

**Tunables.**
- Easy: `posix_data_trans_concurrency` ↑ (more disk workers).
- Better: switch `posix_io_engine` to `aio` for high-depth workloads.
- Capacity: `cache_buffer_capacity_gb` ↑ to widen the absorption
  window if dump rate is bursty.

### 3.5 Layerwise mode shows no speedup over non-layerwise

**Symptoms.** You enabled `use_layerwise: true` expecting overlap, but
end-to-end latency or throughput barely changed.

**Metric signature.** This is exactly what
`layerwise_wait_blocking_ms` is for. Compare it against the time
between waits:

| `wait_blocking_ms` | `inter_wait_interval_ms` | Diagnosis |
|--------------------|--------------------------|-----------|
| ≈ 0 | any | **Overlap working perfectly.** If you still want gains, the bottleneck is elsewhere (the forward pass itself). |
| > 0, < `inter_wait_interval` | > 0 | Partial overlap — load is slightly slower than forward. Reduce load_duration (see §3.2-3.3) and gain will appear. |
| ≈ `pipeline_cache_load_duration_ms` | small | **Pipeline degenerated to serial.** Forward is too fast to hide load. Likely you're decode-bound (one token per layer is fast, loads can't keep up) or Cache miss rate just spiked. |
| Large, growing | Stable | Backlog forming — submission is faster than completion. Check `next_layer_submit_ms` (should be < 1 ms; if not, store.load_data itself is slow). |

**`stalled_layers_total` rate** is a coarser version of the same
signal — easy to alert on. If you see a non-zero rate sustained, dig
into `wait_blocking_ms` distribution.

### 3.6 Layerwise mode: TTFT regression

**Symptoms.** First token is slower with layerwise than without.

**Metric signature.**
- `layerwise_first_layer_submit_ms` is high (rare; usually fast)
- OR `pipeline_cache_load_backend_wait_duration_ms` is high during
  the first batch (cold cache, first-layer load goes all the way to
  disk before forward can start)

**Why TTFT is special in layerwise.** In non-layerwise, all layer
loads complete before forward begins, so TTFT = max(forward,
load_total). In layerwise, only the first layer must complete before
forward of layer 0 begins, but if the first layer's load is a Cache
miss, you pay full Posix latency upfront.

**Tunables.**
- Pre-warm: use prefix prefetching, or batch eviction policies that
  keep first-layer blocks resident.
- Verify the first-layer shard is actually the first item submitted —
  out-of-order shard submission can defeat the optimization.

### 3.7 Layerwise mode: dump tail at end of forward

**Symptoms.** Each forward iteration has an extra few-ms tail you
can't account for. Throughput just below expected.

**Metric signature.**
- `layerwise_save_tail_total_ms` consistently > 0 (e.g. 5-50 ms)
- `layerwise_save_per_layer_wait_ms` skewed: most layers near 0,
  last few non-zero — i.e. dump is keeping up until the end, then
  the last few layer-dumps haven't finished

**Why.** Saves are submitted layer-by-layer during forward, but
`wait_for_save` at the end blocks for **all** of them. The tail is
the dumps that didn't finish before forward ended.

**Tunables.**
- Increase Cache dump throughput: `cache_stream_number` ↑.
- Increase Posix write throughput: `posix_data_trans_concurrency` ↑.
- If dump rate is the limit and can't be raised: accept the tail or
  reduce save load (e.g. only save final-layer KV for short prompts).

### 3.8 Non-layerwise mode: dump-bound iterations

**Symptoms.** Non-layerwise version of §3.4/§3.7. Each iteration
spends a noticeable chunk in `wait_for_save`.

**Metric signature.**
- Legacy `ucm:save_duration` is a meaningful fraction of one
  iteration's wall time
- `ucm:pipeline_cache_dump_duration_ms` ≈ `ucm:save_duration` (no
  surprise — both measure roughly the same thing here)
- Posix h2s bandwidth healthy, h2s duration low → Cache D2H is the
  bottleneck, not disk
- Posix h2s slow → disk is the bottleneck

The split between Cache D2H vs Posix H2S tells you **whether to
optimize the host→cache→disk pipeline at the device-copy stage or at
the disk-write stage**. Those are very different fixes.

### 3.9 Worker pool starvation (shared symptom)

**Symptoms.** Tail latencies are bad even when averages look fine.
Some requests are 10× slower than others.

**Metric signature.** Look at p99 vs avg of any duration metric — if
avg is fine but p99 is enormous, the workers are stuck on something:
- `pipeline_posix_load_wait_duration_ms` p99 → Posix workers blocked
   (slow disk IO, head-of-line blocking)
- `pipeline_cache_load_wait_duration_ms` p99 → Cache dispatcher
   blocked (rare — usually means deadlock or buffer exhaustion)

**Tunables.** Concurrency knobs (`*_concurrency` in config),
`cpu_affinity_cores` (avoid stepping on vLLM scheduler cores), and
`waiting_queue_depth` (drop loudly when full instead of growing
unbounded — the depth is also visible via the wait histogram).

---

## 4. Layerwise vs non-layerwise — what to look at

The pipeline-store metrics (`pipeline_*`) **apply identically to both
modes** — they live in the C++ layer and don't know which Python
connector called them. Use them the same way regardless of mode.

Where the analysis differs:

| Question | Non-layerwise | Layerwise |
|----------|---------------|-----------|
| "Is the load fast enough?" | Compare `load_duration` to forward time | Compare `wait_blocking_ms` to 0 |
| "What's the load latency?" | `load_duration` (= `start_load_kv` block) | `pipeline_cache_load_duration_ms` (per-layer task; sum × n_layers if you want total) |
| "Are saves blocking forward?" | Yes by design — `save_duration` is on the critical path | Only the tail is — `save_tail_total_ms` |
| "Where's TTFT going?" | Total load is on the critical path before forward | `layerwise_first_layer_submit_ms` + first layer's `pipeline_cache_load_*` chain |
| "Is the cache buffer big enough?" | Same indicator: hit rate, dump back-pressure (§3.4) |

A useful rule of thumb: in **layerwise** mode the most informative
single metric is `wait_blocking_ms` (overlap signal). In
**non-layerwise** mode it's `load_duration` plus the
`backend_wait_duration_ms` decomposition (locating the slow tier).

---

## 5. PromQL recipes

All examples assume the bundled Grafana dashboard's `$model_name`
template variable. Replace with a literal string for ad-hoc
querying.

**Cache hit rate (counter ratio, more accurate than the gauge for
historical analysis):**

```promql
rate(ucm:pipeline_cache_lookup_hit_blocks_total{model_name="$model_name"}[5m])
/
clamp_min(
    rate(ucm:pipeline_cache_lookup_hit_blocks_total{model_name="$model_name"}[5m])
  + rate(ucm:pipeline_cache_lookup_miss_blocks_total{model_name="$model_name"}[5m]),
  1)
```

**Shard-level miss ratio at load time (descended to backend):**

```promql
rate(ucm:pipeline_cache_load_backend_submit_shards_total{model_name="$model_name"}[5m])
/
clamp_min(rate(ucm:pipeline_cache_load_shards_total{model_name="$model_name"}[5m]), 1)
```

**P99 load duration per stage (decomposition of the load chain):**

```promql
histogram_quantile(0.99, sum by (le) (
  rate(ucm:pipeline_cache_load_wait_duration_ms_bucket{model_name="$model_name"}[5m])))

histogram_quantile(0.99, sum by (le) (
  rate(ucm:pipeline_cache_load_backend_wait_duration_ms_bucket{model_name="$model_name"}[5m])))

histogram_quantile(0.99, sum by (le) (
  rate(ucm:pipeline_cache_h2d_duration_ms_bucket{model_name="$model_name"}[5m])))
```

Sum these for an approximation of total load p99 (they're correlated,
so the sum overestimates — read trends, not absolute values).

**Average overlap loss (how much of forward time is wasted waiting on
loads):**

```promql
rate(ucm:layerwise_wait_blocking_ms_sum{model_name="$model_name"}[5m])
/
clamp_min(
  rate(ucm:layerwise_inter_wait_interval_ms_sum{model_name="$model_name"}[5m]),
  1)
```

Close to 0 = great overlap. Close to 1 = serial.

**Dump back-pressure ratio (caller-felt vs disk-felt):**

```promql
rate(ucm:pipeline_cache_dump_duration_ms_sum{model_name="$model_name"}[5m])
/
rate(ucm:pipeline_cache_dump_duration_ms_count{model_name="$model_name"}[5m])
```

vs.

```promql
rate(ucm:pipeline_posix_h2s_duration_ms_sum{model_name="$model_name"}[5m])
/
rate(ucm:pipeline_posix_h2s_duration_ms_count{model_name="$model_name"}[5m])
```

The first should be much smaller than the second. When they converge,
back-pressure has reached the caller (see §3.4).

**Posix worker pool utilization proxy:**

```promql
# average wait + IO per IO unit
(rate(ucm:pipeline_posix_load_wait_duration_ms_sum{model_name="$model_name"}[5m])
 + rate(ucm:pipeline_posix_s2h_duration_ms_sum{model_name="$model_name"}[5m]))
/
rate(ucm:pipeline_posix_s2h_duration_ms_count{model_name="$model_name"}[5m])
```

If wait dominates IO, increase `posix_data_trans_concurrency`.

---

## 6. Tunables, indexed by symptom

| Symptom (metric you saw) | First knob to try | Where it lives |
|--------------------------|-------------------|----------------|
| Low hit rate | `cache_buffer_capacity_gb` ↑ | Cache config |
| Cache dump back-pressure | `cache_buffer_capacity_gb` ↑, then `cache_stream_number` ↑ | Cache config |
| Posix wait high | `posix_data_trans_concurrency` ↑ | Posix config |
| Posix bandwidth low, latency low | `io_direct: true`, increase IO size via `tensor_size`/`shard_size` | Posix config |
| Posix bandwidth low, depth-limited | Switch `posix_io_engine: aio` | Posix config |
| Cache load wait high | `waiting_queue_depth` ↑ or reduce concurrency | Cache config |
| Layerwise no overlap | First confirm via `wait_blocking_ms`; then either reduce load latency (above knobs) or accept that workload is forward-bound | — |
| Layerwise save tail high | `cache_stream_number` ↑ (more dump streams), then `posix_data_trans_concurrency` ↑ | Cache + Posix |
| Bad p99, fine avg | `cpu_affinity_cores` (separate from vLLM cores) | Cache + Posix |

---

## 7. What this set of metrics does **not** tell you

For honesty, things the current instrumentation cannot resolve:

- **Per-block hot/cold distribution.** Hit/miss is aggregated; you
   can't see whether 5% of blocks account for 95% of hits.
- **Eviction events.** No counter for "blocks evicted from Cache" —
   sustained drops in hit rate are the indirect signal.
- **GPU compute time.** `inter_wait_interval_ms` includes save
   submission, not just forward. For tight forward-only timing, use
   vLLM's own metrics.
- **Network / RDMA tiers.** The Ds3fs and Mooncake backends are not
   instrumented by this set of metrics. If you switch off `Posix`,
   the Cache-side metrics still apply but the lower-tier ones become
   "no-op visible".
- **Per-request attribution.** All metrics are aggregated by
   `(model_name, worker_id)`. You cannot ask "which request was slow"
   from these metrics alone — combine with vLLM request logs or the
   `enable_record_traces` UCM option.
