# inference_duration_monitor_connector.py 功能说明

## 总览

`ucm/integration/vllm/inference_duration_monitor_connector.py` 实现了一个不做 KV cache 读写的 vLLM KV connector。它借用 vLLM KV connector 的调度、forward 和 layer hook 生命周期来采集推理耗时，主要用于监控 forward 阶段和每层完整 transformer block 的执行时间。

这个 connector 不会加载、保存或传输 KV cache。它的定位是性能探针，而不是缓存传输组件。

该 connector 支持通过配置 `inference_duration_monitor_fake_hit_ratio` 模拟外部缓存命中率。启用后，`get_num_new_matched_tokens()` 会向 vLLM 返回假的命中 token 数（`need_load=False`，视为同步命中），使 vLLM 直接扣减调度量、跳过对应 token 的计算，但 connector 不会真正执行 KV load。这用于在不依赖实际外部存储的情况下，测量不同命中率下每层的计算时间，进而推算 load 所需带宽。注意：启用 fake hit 后模型输出不正确，仅适用于耗时分析。

## 启用方式

入口在 `ucm/integration/vllm/ucm_connector.py`。

`UCMConnector.requires_piecewise_for_cudagraph()` 会调用 `inference_duration_monitor_enabled(extra_config)` 判断是否启用监控。

`UCMConnector.__init__()` 会读取:

```yaml
use_inference_duration_monitor: true
inference_duration_monitor_fake_hit_ratio: 0.5  # 可选, 0.0~1.0, 默认 0.0 (不 fake)
```

当 `use_inference_duration_monitor` 为 true 时，`UCMConnector` 会实例化 `UCMInferenceDurationMonitorConnector`，并直接返回，不再继续选择正常的 UCM/HMA/HLA connector。

`inference_duration_monitor_fake_hit_ratio` 控制模拟命中率。设为 0.0 时行为与不 fake 一致（不声明外部命中），设为 0.5 时每个请求 50% 的 token 被声明为外部缓存命中。该参数可通过 YAML 配置文件或 `kv_connector_extra_config` 直接传入，与 `use_inference_duration_monitor` 走同一条配置链路。

## 采集内容

该 connector 采集以下耗时和指标:

- `forward`: 从 `start_load_kv()` 到 `wait_for_save()` 的整体 forward 阶段耗时。
- `block_layer:<layer_idx>`: 从 `wait_for_layer_load(layer_name)` 到下一层 `wait_for_layer_load` 的完整 transformer block 耗时（attention + MLP + LayerNorm + Residual）。末层使用 `wait_for_save()` 中记录的 `block_end_event` 作为终点。
- 每层 KV cache 总大小（bytes/token）: 在 `register_kv_caches()` 时从 KV cache tensor 按层号累加得出，含同一层的所有 KV cache 条目（如 attn + indexer）。
- 所需带宽（GB/s）: worker 侧按本卡的 block 耗时分别计算并打印；scheduler 侧在 `update_connector_output()` 中使用跨 worker 聚合后的平均 block 耗时生成汇总表。两者都使用 `下一层 KV 总大小 × fake_hit / 本层 block 耗时` 计算。

forward 总耗时使用 `time.perf_counter()` 计算，并在开始和结束附近执行设备同步。

block 层耗时使用设备事件计算。CUDA 平台对应 `torch.cuda.Event(enable_timing=True)`，NPU 平台对应 `torch.npu.Event(enable_timing=True)`，具体抽象来自 `ucm/integration/vllm/device.py`。

这里采集的是 **per-layer block window**，不是纯 attention device time。每个窗口从当前层 attention 调用前开始，到下一层 attention 调用前结束，因此近似表示一个完整 transformer block 的计算窗口，可用于估算预取下一层 KV 的可用重叠时间。

scope 中的 `<layer_idx>` 是从层名中提取的层号（如 `model.layers.5.self_attn.attn` → `5`），通过 `_extract_layer_index()` 方法实现，与具体模型命名无关。

## 带宽计算原理

在 layerwise 加载模式下，每层 KV 的 load 可以与本层计算重叠，但必须在下一层 attention 开始前完成。因此：

```
可用掩盖窗口 = block_layer:N 的计算时间（本层 attention + MLP + 其他）
需要 load 的数据 = 下一层 (layer N+1) 的 KV 总大小
所需带宽 = 下一层 KV 总大小 / 本层 block 计算时间
```

即：`bandwidth = kv_bytes(N+1) × fake_hit / block_avg_ms(N)`

这衡量的是：layer N 的计算能否掩盖 layer N+1 的 load。如果存储实际带宽 ≥ 该值，load 能被计算掩盖。

## Hook 调用时机与计时关系

vLLM 在 piecewise 模式下（`requires_piecewise_for_cudagraph=True`），按层串行执行模型，每层的 attention 前后调用 connector hook。MLP/LayerNorm/Residual 不触发 hook，因为它们不读写 KV cache。

```
forward 开始
│
├── start_load_kv()  ◄── 记录 forward 起点 (time.perf_counter)
│       │
│       │  ┌─────────────────── layer 0 ────────────────────┐
│       │  │  LayerNorm (pre-attn for 0)                       │
│       │  │  wait_for_layer_load(0)  ◄── 记录 start_event_0   │ ─┐
│       │  │  Attention(0)                                     │  │
│       │  │  save_kv_layer(0)        ◄── 记录 end_event_0     │  │
│       │  │  Residual                                         │  │ block_0
│       │  │  LayerNorm (pre-MLP)                              │  │ (attention+
│       │  │  MLP(0)                                            │  │  MLP+其他)
│       │  │  Residual                                         │  │
│       │  └───────────────────────────────────────────────────┘  │
│       │  ┌─────────────────── layer 1 ────────────────────┐   │
│       │  │  LayerNorm (pre-attn for 1, 算在 block_0)         │   │
│       │  │  wait_for_layer_load(1)  ◄── 记录 start_event_1   │ ──┘ block_1
│       │  │  Attention(1)                                     │  │
│       │  │  save_kv_layer(1)        ◄── 记录 end_event_1     │  │
│       │  │  ...                                              │  │
│       │  └───────────────────────────────────────────────────┘  │
│       │                         ...                             │
│       │  ┌─────────────────── layer N ────────────────────┐   │
│       │  │  LayerNorm (pre-attn for N, 算在 block_{N-1})     │   │
│       │  │  wait_for_layer_load(N)  ◄── 记录 start_event_N   │ ──┘ block_N
│       │  │  Attention(N)                                     │  │
│       │  │  save_kv_layer(N)        ◄── 记录 end_event_N     │  │
│       │  │  ...                                              │  │
│       │  └───────────────────────────────────────────────────┘  │
│       │                                                         │
├── wait_for_save()  ◄── 记录 forward 终点 (time.perf_counter)    │
│       │                  + 记录 block_end_event (用于末层 block) │
│       │                                                         │
│       └── 计算:                                                  │
│              forward            = forward终点 - forward起点      │
│              block_layer:K     = start_event_{K+1} - start_event_K│
│              block_layer:last  = block_end_event - start_event_last│
│
forward 结束
```

各计时值的关系:

```
│  LayerNorm │                         │ Residual              │
│ (pre-attn  │                         │ LayerNorm (pre-MLP)   │
│  for N,    │                         │ MLP(N)                 │
│  不算在    │                         │ Residual               │
│  block:N)  │                         │ LayerNorm (pre-attn    │
│           │                         │  for N+1, 算在block:N) │
│◄──────────│─────────────────────────│────────────────────────►│
│◄──────────────────── block_layer:N ──────────────────────────►│
           start_N                     end_N                   start_{N+1}
```

- **block_layer:N** = `start_event_{N+1} - start_event_N`（attention + Residual + LayerNorm(pre-MLP) + MLP + Residual + LayerNorm(pre-attn for N+1)）

注意：LayerNorm(pre-attn for N) 在 `wait_for_layer_load(N)` 之前执行，不算在 block:N 里，而是算在 block:N-1 中。末层 block 使用 `wait_for_save()` 中记录的 `block_end_event` 作为终点。

`block_layer:N` 本质上是第 N 层的 forward 耗时。整体 `forward` 与各层 `block_layer` 的关系：

```
forward = embedding + Σ(block_layer:N) + LM_head + 其他开销
```

## 核心数据结构

### InferenceDurationMonitorMetadata

`InferenceDurationMonitorMetadata` 是 scheduler 传给 worker 的元数据，记录当前 step 的调度情况:

- `preempted_req_ids`: 被抢占的请求 ID。
- `scheduled_reqs`: 当前 step 调度的请求数。
- `new_reqs`: 当前 step 的新请求数。
- `new_reqs_with_computed_tokens`: 新请求中已有 computed tokens 的请求数。
- `scheduled_tokens`: 当前 step 调度的 token 总数。
- `total_num_computed_tokens`: 新请求累计 computed tokens 数。
- `fake_hit`: 当前 step 的 fake hit token 数（batch 内所有请求累加），从 scheduler 侧传到 worker 侧。
- `dp_rank`: 产生该 step 的 DP rank。
- `step_id`: 当前 DP 内单调递增的 scheduler step ID。

它的关键属性是:

```python
should_collect_duration = True
```

因此，connector 会在每个 step 都采集耗时，不再依赖本地前缀缓存命中。

### DurationStats

`DurationStats` 是可合并的耗时统计结构，单位是毫秒。

它维护:

- `count`: 样本数量。
- `sum_ms`: 总耗时。
- `min_ms`: 最小耗时。
- `max_ms`: 最大耗时。
- `avg_ms`: 平均耗时。

`observe()` 用于记录单次耗时，`aggregate()` 用于合并其他 worker 的统计结果。

### InferenceDurationMonitorWorkerMetadata

`InferenceDurationMonitorWorkerMetadata` 是 worker 侧输出的聚合元数据。

它包含:

- `duration_stats`: 按 scope 统计的耗时，例如 `forward` 或 `block_layer:<layer_idx>`。
- `worker_ranks`: 参与统计的 worker/model rank 集合。
- `fake_hit`: 当前 step 的 fake hit token 数，从 worker 侧传回 scheduler 侧，用于汇总表带宽计算。
- `dp_rank`: 当前 metadata 所属的 DP rank。
- `step_id`: 与 scheduler metadata 一致的 step ID。

它的 `aggregate()` 只允许合并相同 `dp_rank` 和 `step_id` 的数据，然后合并多个 worker 的 `DurationStats` 与 worker rank 集合，并传递 `fake_hit`。

## 主要执行流程

### 1. 初始化

`UCMInferenceDurationMonitorConnector.__init__()` 会记录:

- data parallel rank: `_dp_rank`
- model/global rank: `_model_rank`
- 模拟命中率: `_fake_hit_ratio` (从配置 `inference_duration_monitor_fake_hit_ratio` 读取, 默认 0.0)
- 每个请求的 computed token 记录: `_hbm_hit_tokens_by_request`
- 每层每个 KV cache 条目大小: `_kv_bytes_per_token` (在 `register_kv_caches` 中填充)
- 每层 KV cache 总大小: `_layer_total_bytes_per_token` (在 `register_kv_caches` 中按层号累加)
- 上一次 fake hit token 数: `_last_fake_hit` (在 `start_load_kv` 中从 metadata 读取)
- 当前 DP 的 scheduler step 计数: `_scheduler_step_id`
- 当前 worker forward 对应的 DP/step: `_current_dp_rank`、`_current_step_id`
- 设备抽象: `_device`
- 当前 forward 是否采集: `_collect_current_forward`
- forward 起点时间: `_inference_start_time`
- attention start/end event 缓存
- 当前 step 的耗时统计
- 待上报的 worker metadata

初始化日志为:

```text
Init UCMInferenceDurationMonitorConnector (no KV I/O, fake_hit_ratio=0.50).
```

### 2. register_kv_caches()

该方法创建设备抽象，并从传入的 KV cache tensor 中提取每层 KV 大小:

```python
self._device = self._create_device()
block_size = self._vllm_config.cache_config.block_size
for layer_name, kv_cache in kv_caches.items():
    bytes_per_token = kv_cache.numel() * kv_cache.element_size()
        // (num_blocks * block_size)
    self._kv_bytes_per_token[layer_name] = bytes_per_token
    layer_idx = self._extract_layer_index(layer_name)
    self._layer_total_bytes_per_token[layer_idx] += bytes_per_token
```

每个 token 的 KV 大小 = tensor 总字节数 / (block 数 × block_size)。该值是当前 TP worker 上的 KV 大小。

`_layer_total_bytes_per_token` 按层号累加同一层的所有 KV cache 条目（如 `attn` + `indexer.k_cache`），通过 `_extract_layer_index()` 从层名中提取层号（如 `model.layers.6.self_attn.attn` → `6`），与具体模型命名无关。

如果不同层的 KV 配置不同（如混合注意力模型），各层的 `total_bytes_per_token` 会不同。

启动时对每层输出汇总日志:

```text
KV cache total: layer_idx=0, total_bytes_per_token=1284 (1.25 KB/token)
KV cache total: layer_idx=1, total_bytes_per_token=1284 (1.25 KB/token)
KV cache total: layer_idx=3, total_bytes_per_token=1152 (1.12 KB/token)
```

如果当前平台不支持，会抛出:

```text
Unsupported device platform for inference duration monitoring.
```

### 3. get_block_size()

返回 vLLM cache config 中的 block size:

```python
return self._vllm_config.cache_config.block_size
```

该方法由 `UCMConnector._record_prefix_cache_token_metrics()` 通过 `self.connector.get_block_size()` 调用，用于计算 block 级别的指标统计。`self._vllm_config` 由基类 `KVConnectorBase_V1.__init__()` 设置。

### 4. get_num_new_matched_tokens()

该方法首先记录请求已有的本地 HBM 前缀缓存命中 token 数:

```python
local_hit = max(int(num_computed_tokens), 0)
self._hbm_hit_tokens_by_request[request.request_id] = min(local_hit, total)
```

然后根据配置的 `_fake_hit_ratio` 计算假外部命中 token 数:

```python
fake_hit = min(int(total * self._fake_hit_ratio), max(total - local_hit, 0))
self._last_fake_hit += fake_hit
```

`_last_fake_hit` 使用**累加**（`+=`）而非覆盖，因此同一 step 内多个新请求的 fake_hit 会被累加。在 `build_connector_meta()` 中传给 metadata 后重置为 0。

如果 `fake_hit > 0`，返回:

```python
return fake_hit, False
```

`need_load=False` 表示 vLLM 将 fake hit 视为**同步命中**，直接从 `num_scheduled_tokens` 中扣减 fake_hit 个 token，请求正常调度不会被挂起。vLLM 只计算 `total - fake_hit` 个 token。

per-layer hooks（`start_load_kv`、`wait_for_layer_load`、`save_kv_layer`、`wait_for_save`）的调用与 `need_load` 无关，只要 connector 配置了且绑定了 metadata 就会每层调用。因此 monitor 仍能采集每层耗时。

注意：vLLM 期望 fake_hit 个 token 的 KV 已在 block 中（同步命中），但 connector 不会 load 任何数据，因此这些 token 的 KV block 是未初始化数据，模型输出不正确。

不使用 `need_load=True` 的原因：`True` 会让 vLLM 将请求挂起到 `WAITING_FOR_REMOTE_KS` 状态，期望 connector 异步 load 后通过 `get_finished()` → `KVConnectorOutput.finished_recving` 信号通知完成。但 monitor 不做真正 load，不发送 `finished_recving`，导致请求无法推进。

如果 `fake_hit <= 0`（`_fake_hit_ratio` 为 0.0 或 local_hit 已覆盖全部 token），返回:

```python
return 0, False
```

此时行为与不 fake 一致，vLLM 会计算全部 token。

记录 `local_hit` 仍是为了后续 `build_connector_meta()` 的 fallback 使用。

### 5. build_connector_meta()

该方法在 scheduler 侧构建 `InferenceDurationMonitorMetadata`。

它会统计:

- 当前调度请求 ID。
- 新请求数量。
- 新请求中 computed tokens 大于 0 的数量。
- computed token 总数。
- 调度 token 总数。
- 被抢占请求 ID。
- fake_hit: 从 `_last_fake_hit`（累加值）传入 metadata，然后重置 `_last_fake_hit = 0`，供 worker 侧使用。
- dp_rank 和 step_id: 标识当前 DP 内的 scheduler step，并随 metadata 传到 worker。

同时会打印调度统计日志:

```text
Inference duration scheduler stats: dp_rank=..., step_id=..., rank=..., scheduled_reqs=..., new_reqs=..., scheduled_tokens=...
```

对于已完成请求，它会从 `_hbm_hit_tokens_by_request` 中删除对应记录，避免状态残留。

### 6. start_load_kv()

这是 forward 开始附近的 hook。

它从 vLLM 获取 connector metadata，读取 fake_hit 并判断采集开关:

```python
self._collect_current_forward = metadata.should_collect_duration
self._last_fake_hit = metadata.fake_hit
```

由于 `should_collect_duration` 已改为始终返回 `True`，connector 会在每个 step 都采集耗时。代码中仍保留了 `if not self._collect_current_forward: return` 分支，但在当前实现下不会触发。

如果需要采集，它会:

1. 清空上一轮 attention event 和统计状态。
2. 调用 `device.synchronize()`，确保之前设备上的异步任务完成。
3. 用 `time.perf_counter()` 记录 forward 起点。

### 7. wait_for_layer_load()

这是每层 attention 前的 hook。

当当前 forward 正在采集时，它会记录一个 timing event，并按 layer name 保存:

```python
self._active_attention_events[layer_name] = self._get_device().record_timing_event()
```

该 event 同时作为 `block_layer` 的起点。

如果记录 start event 失败，会打印 warning，但不会中断推理。

### 8. save_kv_layer()

这是每层 attention 后的 hook。

虽然方法名是 `save_kv_layer()`，但这里不会保存 KV。传入的 `kv_layer`、`attn_metadata` 和 `kwargs` 都会被丢弃。

该方法只做计时:

1. 取出对应 layer 的 start event。
2. 记录 end event。
3. 将 `(layer_name, start_event, end_event)` 放入 `_pending_attention_events`。

`_pending_attention_events` 按层执行顺序排列，`wait_for_save()` 中的 block 计算依赖此顺序。

当前 block 耗时计算使用相邻层的 start event；end event 用于确认该层 attention hook 已完整结束并把该层加入待处理列表，不单独生成纯 attention 耗时指标。

如果没有 start event 或 end event 记录失败，则跳过该层统计。

### 9. wait_for_save()

这是 forward 结束附近的 hook，也是单次采集的收口点。

如果没有 `_inference_start_time`，说明当前 step 没有采集，方法直接返回。

否则它会:

**步骤 1 - 记录 block 终点 event（在 synchronize 之前）:**

```python
block_end_event = device.record_timing_event()
```

该 event 用于计算末层 `block_layer` 的耗时。

**步骤 2 - 设备同步并计算 forward 总耗时:**

```python
device.synchronize()
elapsed_ms = (time.perf_counter() - self._inference_start_time) * 1000
self._observe_duration("forward", elapsed_ms)
```

**步骤 3 - 计算每层 block 耗时:**

```python
for i in range(num_events):
    layer_name, start_event, _ = self._pending_attention_events[i]
    if i < num_events - 1:
        next_start_event = self._pending_attention_events[i + 1][1]
        block_ms = device.elapsed_time_ms(start_event, next_start_event)
    else:
        block_ms = device.elapsed_time_ms(start_event, block_end_event)
    layer_idx = self._extract_layer_index(layer_name)
    scope_name = str(layer_idx) if layer_idx is not None else layer_name
    self._observe_duration(f"block_layer:{scope_name}", block_ms)
```

- 非末层: `block_layer:K = start_event_{K+1} - start_event_K`
- 末层: `block_layer:last = block_end_event - start_event_last`

如果本轮没有观察到任何 attention hook，会打印一次 warning:

```text
Inference duration monitor observed no attention hooks. Per-layer block-window timing is unavailable when the active model execution path bypasses KV connector layer hooks.
```

**步骤 4 - 构建并清理:**

构建 `InferenceDurationMonitorWorkerMetadata`，设置 `fake_hit=self._last_fake_hit`，清空本地状态。

### 10. build_connector_worker_meta()

该方法返回上一轮 `wait_for_save()` 生成的 worker metadata，并清空本地 pending 状态，避免重复上报。

### 11. update_connector_output()

该方法在 scheduler 侧读取 vLLM 聚合后的 `kv_connector_worker_meta`（跨当前 DP engine 全部 worker 聚合后的数据），并输出该 DP/step 的 `forward` 聚合日志。日志包含 `dp_rank` 和 `step_id`，用于与 worker 侧逐卡带宽日志关联。

当前实现只直接打印 `forward` scope：

```text
Inference duration aggregate: dp_rank=1, step_id=42, workers=8, scope=forward, count=8, avg_ms=..., min_ms=..., max_ms=...
```

逐层跨 worker 汇总表由 `parse_monitor_log.py` 根据相同 `(dp_rank, step_id)` 的逐卡 `KV bandwidth` 日志生成，而不是由 `update_connector_output()` 直接打印。

## 不执行 KV I/O 的方法

以下方法体现了该 connector 的 no-I/O 设计:

```python
get_block_size()             # 返回 cache_config block size, 不涉及 I/O
update_state_after_alloc()  # no-op
save_kv_layer()             # only records timing events
request_finished_all_groups() -> (False, None)
```

`get_num_new_matched_tokens()` 在 `fake_hit_ratio > 0` 时会返回非零值（`need_load=False`，同步命中），vLLM 直接扣减调度量，但 connector 不会真正执行 KV load。`wait_for_layer_load()` 只记录 timing event，不加载任何数据。

因此它不会:

- 从外部存储加载 KV cache。
- 向外部系统保存 KV cache。
- 改变 vLLM 的 KV block 分配。
- 在请求结束时释放外部 KV 资源。

注意：当 `fake_hit_ratio > 0` 时，vLLM 会跳过 fake hit token 的计算，但 connector 不会 load 对应的 KV 数据，因此这些 token 的 KV block 中是未初始化数据，模型输出不正确。

## 和 vLLM hook 的关系

该 connector 利用 vLLM KV connector 的以下生命周期方法:

- `build_connector_meta()`: scheduler 侧生成元数据，传递 fake_hit 到 worker。
- `get_block_size()`: 返回 block size, 供 `UCMConnector` 指标统计使用。
- `get_num_new_matched_tokens()`: 声明 fake 外部命中 token 数，累加到 `_last_fake_hit`。
- `register_kv_caches()`: 记录每层 KV 大小，按层号累加。
- `start_load_kv()`: forward 开始，记录 forward 起点，从 metadata 读取 fake_hit。
- `wait_for_layer_load()`: attention 前，记录 start event（block 的起点）。
- `save_kv_layer()`: attention 后，记录 end event。
- `wait_for_save()`: forward 结束，计算 forward 总耗时和 block 耗时，构建 worker metadata（含 fake_hit）。
- `build_connector_worker_meta()`: worker 侧上报统计。
- `update_connector_output()`: scheduler 侧聚合后生成汇总表。

hook 围绕 attention 调用，不围绕 MLP/LayerNorm/Residual 调用，因为这些操作不读写 KV cache。因此:
- `block_layer` 表示完整 transformer block 时间（attention + MLP + LayerNorm + Residual），即单层 forward 耗时

## 日志输出

该文件输出以下日志:

1. **启动时 - 初始化**（一次性）:

```text
Init UCMInferenceDurationMonitorConnector (no KV I/O, fake_hit_ratio=0.50).
```

2. **启动时 - 每层 KV cache 总大小**（一次性）:

```text
KV cache total: layer_idx=0, total_bytes_per_token=1284 (1.25 KB/token)
```

3. **Scheduler 侧调度统计**（每步）:

```text
Inference duration scheduler stats: dp_rank=1, step_id=42, rank=..., scheduled_reqs=..., new_reqs=..., scheduled_tokens=...
```

4. **Worker 侧逐卡带宽**（每步，每张卡分别输出）:

```text
KV bandwidth: dp_rank=1, step_id=42, worker_rank=6, layer_idx=6 (compute) -> layer_idx=7 (load), cur_kv_bytes_per_token=1284, next_kv_bytes_per_token=1284, fake_hit=14745, kv_total=18.93 MB, layer_avg_ms=1.534, required_bandwidth=11.07 GB/s
```

该日志反映当前 worker/卡的实际 block 耗时及对应带宽需求，因此多卡运行时会看到每张卡分别打印。

5. **汇总表**（由 `parse_monitor_log.py` 按 DP/step 聚合输出）:

```text
=== Inference Duration Summary (dp_rank 1, step 42) ===
workers=8, layers=78, fake_hit=14745
forward: avg=308.578ms, min=306.593ms, max=312.090ms

layer | block_avg_ms | -> load_layer | kv_bytes/token | kv_total_MB | bandwidth_GBps
    0 |      106.103 |            1 |           1284 |      18.93 |          0.16
    1 |        2.691 |            2 |           1284 |      18.93 |          6.31
    2 |        1.534 |            3 |           1284 |      18.93 |         11.07
  ...
   76 |        3.950 |           77 |           1152 |      16.99 |          4.30
   77 |        3.120 |            - |              0 |          - |           -

--- Summary ---
max_bandwidth: 11.07 GB/s (layer 2)
min_bandwidth: 2.88 GB/s (layer 3, excl. layer 0)
=== End Summary ===
```

各列含义:

| 列 | 含义 | 聚合方式 |
|---|---|---|
| `layer` | 本层（计算层） | - |
| `block_avg_ms` | 本层计算耗时 | 跨全部 worker 平均 |
| `-> load_layer` | 下一层（被 load 的层） | - |
| `kv_bytes/token` | 下一层的 KV 大小 | 单 worker（TP 分区） |
| `kv_total_MB` | 下一层 KV 总量 = kv_bytes × fake_hit | 单 worker |
| `bandwidth_GBps` | 所需带宽 = kv_total / 聚合后的 block_avg_ms | 单 worker 数据量、跨 worker 平均耗时 |

末层无下一层，不输出带宽。预热后数据为稳定值。

## 多 DP 日志解析

多 DP 场景下，每个 DP replica 有独立的 scheduler step 计数和请求 batch。监控日志使用 `(dp_rank, step_id)` 作为聚合键，`worker_rank` 用于在 DP 内识别并去重 TP worker，避免不同 DP 的 fake hit、窗口时间和带宽被混在一起。

`parse_monitor_log.py` 的默认行为是输出每个 DP 最新的有效 step：

```bash
python parse_monitor_log.py vllm_server.log
```

也可以筛选 DP 或 step：

```bash
python parse_monitor_log.py vllm_server.log --dp-rank 1
python parse_monitor_log.py vllm_server.log --step 42
python parse_monitor_log.py vllm_server.log --dp-rank 1 --step 42
python parse_monitor_log.py vllm_server.log --all
```

新格式日志中的 `--step` 表示明确的 step ID；旧格式日志没有 DP/step 标识，解析器仍兼容原有的 scheduler 行顺序分组，此时 `--step` 表示从 0 开始的列表索引。

## 总结

`inference_duration_monitor_connector.py` 是一个插在 vLLM KV connector 生命周期中的推理耗时监控器。它不参与 KV cache 传输，在配置启用后，每步都采集 forward 总耗时和相邻 attention hook 之间的 block window。多 DP 日志通过 `dp_rank`、`step_id` 和 `worker_rank` 精确关联，再由 `parse_monitor_log.py` 分 DP 聚合输出汇总表。

通过 `inference_duration_monitor_fake_hit_ratio` 配置可模拟不同外部缓存命中率（`need_load=False`，同步命中），使 vLLM 直接扣减调度量但不执行实际 load，从而测量不同命中率下的 KV 预取窗口。结合从 KV cache tensor 按层号累加的每层 KV 总大小，计算 layerwise load 所需带宽（下一层 KV / 当前预取窗口），判断 load 是否能被该窗口掩盖。注意启用 fake hit 后模型输出不正确，仅适用于耗时分析。
