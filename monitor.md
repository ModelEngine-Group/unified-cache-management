# vLLM Inference Duration Monitor

## 功能定位

`ucm/integration/vllm/inference_duration_monitor_connector.py` 是一个不执行 KV I/O 的 vLLM connector，用于采集：

- 新请求首次进入模型时的整体 forward 时延。
- 相邻 attention hook 之间的逐层 block-window 时延。
- 当前 forward 的 fake-hit token 数和 scheduler 调度信息。

该 connector 不解析 KV cache tensor 布局，不计算 KV bytes，也不计算带宽。`register_kv_caches()` 只初始化计时设备，因此不会受到不同模型 cache layout 或 block size 的影响。

## 配置

在 UCM 配置中启用：

```yaml
use_inference_duration_monitor: true
inference_duration_monitor_fake_hit_ratio: 0.5  # 可选，默认 0.0
```

`inference_duration_monitor_fake_hit_ratio` 用于模拟外部缓存命中率。`get_num_new_matched_tokens()` 将 fake-hit token 数返回给 vLLM，并设置 `need_load=False`。vLLM 会把这些 token 视为同步命中并跳过对应计算，但 connector 不会真正加载 KV，因此启用 fake hit 后模型输出不正确，仅适用于性能测试。

## 使用限制

初始化时会进行以下检查：

- `vllm_config.speculative_config` 必须为空。Speculative decoding 可能延迟 `wait_for_save()`，使整体和末层时延混入 draft model、采样及 bookkeeping。
- `vllm_config.cache_config.enable_prefix_caching` 必须为 false。关闭 HBM prefix caching 后，新请求的 `num_computed_tokens` 才只表示 connector 返回的 fake external hit。

条件不满足时 connector 会直接抛出 `ValueError`，避免生成语义不明确的数据。

## 采集范围

`InferenceDurationMonitorMetadata.should_collect_duration` 的条件为：

```python
new_reqs > 0 and scheduled_tokens > 0
```

因此只采集有新请求首次进入模型且实际调度了 token 的 forward，不采集：

- 纯 decode forward。
- 后续 chunked-prefill forward。
- `scheduled_tokens=0` 的空调度轮次。

新请求与已有请求组成混合 batch 时，时延代表整个混合 batch。可通过日志中的 `new_reqs`、`scheduled_reqs` 和 `scheduled_tokens` 识别。

## ID 语义

- `dp_rank`：当前 data-parallel engine。
- `scheduler_iteration_id`：当前 DP 内每次 `build_connector_meta()` 调用的序号，从 0 开始，包含空调度轮次。
- `forward_id`：当前 DP 内实际调度了 token 的 forward 序号，从 0 开始。`scheduled_tokens=0` 时为 `-1`。
- `worker_rank`：产生 worker metadata 的 model/global rank。

多个请求不会重置这些 ID；它们在对应 DP engine 的生命周期内持续递增。

## 计时语义

### 整体 forward

```text
forward = synchronize after wait_for_save
        - synchronize before start_load_kv
```

起点使用 `time.perf_counter()`，并在起止位置同步设备。它表示 connector hook 覆盖的整体 forward 阶段，不等同于单个 CUDA kernel 时间。

### 逐层 block window

`wait_for_layer_load(layer_name)` 在 attention 前记录 start event。`save_kv_layer()` 只确认该层 hook 已完成并保存 start event，不再记录未参与计算的 attention end event。

对非末层：

```text
block_layer:N = start_event(N+1) - start_event(N)
```

该窗口通常包含第 N 层 attention、MLP、Residual、LayerNorm，以及下一层 attention 前的准备工作。因此 `block_layer:N` 是层间窗口时间，不是纯 attention 时间，也不严格等于模型定义中第 N 层的全部计算时间。

末层使用 `wait_for_save()` 中记录的 `block_end_event`：

```text
last block window = block_end_event - last start_event
```

## 主要数据结构

### InferenceDurationMonitorMetadata

Scheduler 传给 worker 的 metadata 包含：

- `scheduled_reqs`
- `new_reqs`
- `new_reqs_with_computed_tokens`
- `scheduled_tokens`
- `total_num_computed_tokens`
- `fake_hit`
- `dp_rank`
- `scheduler_iteration_id`
- `forward_id`

关闭 HBM prefix caching 后，`total_num_computed_tokens` 等于当前 step 新请求的 fake external hit token 总数，并直接作为 `fake_hit` 传给 worker。

### DurationStats

每个 scope 保存：

- `count`
- `sum_ms`
- `min_ms`
- `max_ms`
- `avg_ms`

多个 TP worker 的数据通过 `aggregate()` 合并，因此 scheduler 输出的 avg/min/max 是当前 DP 内跨 worker 的聚合结果。

### InferenceDurationMonitorWorkerMetadata

Worker metadata 包含：

- `duration_stats`
- `worker_ranks`
- `fake_hit`
- `dp_rank`
- `scheduler_iteration_id`
- `forward_id`

只允许聚合 ID 完全一致的数据。

## Hook 流程

```text
build_connector_meta()
  └─ 生成调度 metadata 和 ID

start_load_kv()
  ├─ 判断是否采集
  ├─ 同步设备
  └─ 记录 forward 起点

wait_for_layer_load(N)
  └─ 记录 start_event(N)

save_kv_layer(N)
  └─ 将 start_event(N) 加入有序队列

wait_for_save()
  ├─ 记录末层 block_end_event
  ├─ 同步设备并计算整体 forward
  ├─ 计算所有 block_layer 时延
  └─ 构建 worker metadata

update_connector_output()
  └─ 输出跨 worker 聚合后的 forward 和逐层统计
```

如果执行路径绕过 KV connector layer hooks，connector 会记录 warning，整体 forward 仍可用，但逐层时延不可用。

## 日志格式

### Scheduler 调度日志

```text
Inference duration scheduler stats: dp_rank=0, scheduler_iteration_id=3, forward_id=3, rank=0, scheduled_reqs=1, new_reqs=1, scheduled_tokens=6554
```

### 整体 forward 聚合日志

```text
Inference duration aggregate: dp_rank=0, scheduler_iteration_id=3, forward_id=3, workers=8, scope=forward, fake_hit=58982, count=8, avg_ms=862.810, min_ms=862.354, max_ms=863.220
```

### 逐层聚合日志

```text
Inference duration aggregate: dp_rank=0, scheduler_iteration_id=3, forward_id=3, workers=8, scope=block_layer:0, fake_hit=58982, count=8, avg_ms=13.337, min_ms=13.201, max_ms=13.490
```

`fake_hit` 是请求级命中 token 数。汇总表会在每层行中重复显示它，便于关联，但它不是各层独立计算出来的命中量。

## 日志解析脚本

`parse_monitor_log.py` 按 `(dp_rank, scheduler_iteration_id)` 聚合日志，并使用 `forward_id` 选择实际 forward。

### 按多次 bench 分组

开始一次 bench 前执行：

```bash
python monitor_bench_session.py start vllm_server.log
```

随后使用任意方式发送请求。请求全部完成后执行：

```bash
python monitor_bench_session.py stop vllm_server.log
```

`stop` 会结束当前 bench 并立即打印该次 bench 的汇总。bench ID 自动生成，无需手动指定。每次独立 bench 都需要执行一组 `start` 和 `stop`，多次 bench 必须顺序执行。

所有 bench 完成后，可统一打印全部 bench 的汇总：

```bash
python parse_monitor_log.py vllm_server.log --all-benches
```

脚本会在服务日志旁生成 `.bench_active.json` 和 `.bench_runs.jsonl` 状态文件，通过每次 bench 对应的日志字节区间进行分组，不会修改或向 vLLM 服务发送请求。

### 直接解析 forward

```bash
# 每个 DP 最新一次被采集的 forward
python parse_monitor_log.py vllm_server.log

# 指定 forward ID
python parse_monitor_log.py vllm_server.log --forward-id 3

# 指定 DP
python parse_monitor_log.py vllm_server.log --dp-rank 1

# 输出全部 forward
python parse_monitor_log.py vllm_server.log --all

# 导出 CSV
python parse_monitor_log.py vllm_server.log --csv timings.csv

# 将汇总追加到文件
python parse_monitor_log.py vllm_server.log --output summary.log
```

示例输出：

```text
=== Inference Duration Summary (dp_rank 0, forward 3, scheduler iteration 3) ===
workers=8, layers=78, fake_hit=58982
forward: avg=862.810ms, min=862.354ms, max=863.220ms
scheduled_reqs=1, new_reqs=1, scheduled_tokens=6554

   layer |     avg_ms |     min_ms |     max_ms | hit_tokens
------------------------------------------------------------
       0 |     13.337 |     13.201 |     13.490 |      58982
       1 |      8.672 |      8.610 |      8.731 |      58982
=== End Summary ===
```

## 不执行的操作

该 connector 不会：

- 从外部存储加载 KV。
- 保存或传输 KV。
- 解析 KV cache tensor 大小或布局。
- 计算 KV bytes、KV 总量或所需带宽。
- 修改 KV block 分配。

它仅借用 KV connector 生命周期获得 scheduler、forward 和 attention 边界。
