# DramPool Metrics 埋点与跨进程回流设计方案

| 项 | 说明 |
|---|---|
| 范围 | DramPool 服务端守护进程（`ucm/store/dram/cc/drampool/`）**业务逻辑**埋点 + 快照日志输出；UCM/Scheduler 侧新增回流器组件。DramStore 客户端侧零改动，HealthServer 零改动 |
| 核心诉求 | 解决 DramPool 独立进程与 vLLM/UCM 跨进程指标统一暴露，对接现有 Prometheus 监控链路 |
| 设计原则 | **业务聚焦**：仅保留直接反映 DUMP/LOAD/LOOKUP 服务质量（请求量 / 成功率 / 时延 / 命中率）与资源水位的指标。接入层协议错误、内部调度诊断、低频异常分类、驱逐/GC 中间信号等**非必要指标此阶段不涉及**（相应路径维持 UC_WARN/UC_ERROR 日志） |
| 共享代码约束 | 不修改 `ucm/shared/pool`、`ucm/shared/infra/template` 等共享组件；仅新增对 `ucm/shared/metrics` 的链接依赖 |
| 指标规模 | **22 个** |
| 数据结构影响 | **零扩展**（`RequestTask`/`CompletionRecord` 均保持原样） |

---

## 1. 总体架构与数据链路

```
┌────────────────────── DramPool 进程 (C++) ──────────────────────┐
│ 业务打点(TaskWorker/CompletionPoller/GC)                         │
│   └─ UpdateStats → thread-local 双 buffer（UC::Metrics，无锁）   │
│        └─ Reporter 线程：10s 周期快照 → 增量累加为进程级累计值     │
│             └─ JSON 行序列化 → 有界队列 → MetricsFlush 线程       │
│                  └─ 异步 append 刷盘（JSON Lines，轮转+清理）     │
└──────────────────────────┬───────────────────────────────────┘
                           │ 本地 JSON Lines 日志文件
┌──────────────────────────▼───────────────────────────────────┐
│ UCM/Scheduler 侧回流器 (Python，推理进程内)                      │
│   Leader 选举(flock) → 尾部读取最新完整快照(64KiB 窗口)          │
│     → 累计转增量(Counter/Histogram delta，Gauge 直取)           │
│     → ucmmetrics → vllm_connector → Prometheus                  │
└──────────────────────────────────────────────────────────────┘
```

时序约定：DramPool 侧每 10s 产出一行"独立终点状态"快照（累计值）；回流器周期读取最新行、换算增量喂给 ucmmetrics。两侧时钟不要求同步——`ts` 仅用于排序与诊断，指标计算不依赖跨进程时钟。

---

## 2. 采集方案选型与框架约束

### 2.1 进程内采集选型（DramPool 侧）

| 候选 | 结论 | 理由 |
|---|---|---|
| Handler 方案（位置编码 + `/` 分割） | **弃用** | 字段位置编码易错位、维护成本高、多字段非强一致 |
| MetricsSlot 方案（全局固定数组 + 原子/互斥锁） | **弃用** | Histogram 更新存在锁竞争，且需重新引入整套注册/快照逻辑 |
| **UCM thread-local + 双 buffer（采用）** | ✅ | 线程完全隔离，读写通过原子索引切换，无锁竞争，天然适配高并发打点；复用现有 `UC::Metrics` 库 |

### 2.2 API（`ucm/shared/metrics/cc/api/metrics_api.h`）

```cpp
namespace UC::Metrics {
void SetUp(size_t maxVectorLen);                       // 进程级一次性初始化
void CreateStats(const std::string& name, const std::string& type);  // COUNTER/GAUGE/HISTOGRAM
void UpdateStats(const std::string& name, double value);             // 采样
std::tuple<counter表, gauge表, histogram原始值向量表> GetAllStatsAndClear();  // 读后即清
}
```

### 2.3 决定方案形态的五条框架语义（源自 `metrics.cc` 实现）

| # | 语义 | 对方案的约束 |
|---|---|---|
| C1 | **无标签维度**：内部是纯 `name → double` 扁平映射 | 带 `{op}`、`{slot_size}` 维度的指标必须拆成独立指标名；按 slot size 拆分的 Gauge 在启动期动态注册 |
| C2 | **未注册名字静默丢弃**（`UpdateStats` 直接 return） | 所有指标必须在启动期集中注册；单测需覆盖注册完备性 |
| C3 | **Histogram 存原始值向量**，上限 `maxVectorLen`，**满后丢弃后续观测**（保留先到的样本） | 观测点控制在请求粒度；向量长度可配置 |
| C4 | **`GetAllStatsAndClear()` 读后即清**：返回的是"距上次调用的增量" | **Reporter** 将周期增量累加至进程级累计值后再写 JSON（每行快照为累计终点状态）；增量换算由回流器在 UCM 侧完成 |
| C5 | 线程级双缓冲，首次 `UpdateStats` 的线程自动注册，无需手动登记 | 各 worker 线程直接调用即可 |

### 2.4 命名约定

- JSON 日志中指标名与 Prometheus 输出名分离：JSON 内为 `drampool_*`，回流器注册到 ucmmetrics 时统一加 `ucm:` 前缀（对齐 UCM 现有习惯）。
- 单位后缀进指标名：`_ms`、`_bytes_total`、`_entries_total`、`_ratio`。
- Counter 一律 `_total` 结尾；动态注册名中的 slot size 用原始字节数（如 `drampool_buffer_pool_usage_ratio_1048576`），保证名字合法且稳定。

---

## 3. 指标必要性审查与全集

### 3.1 判定标准（按优先级排序）

| # | 标准 | 含义 |
|---|---|---|
| **S0** | **业务直接相关** | 直接反映 DUMP/LOAD/LOOKUP 操作的服务质量（请求量、成功率、时延、命中率）或资源水位（池用量、元数据规模）。纯系统性/接入层/内部调度/低频异常分类指标**此阶段不涉及** |
| S1 | 不可推导 | 无法由其他指标经 PromQL 运算（rate 比值、差值）等价获得 |
| S2 | 可行动 | 指标异常时存在明确的运维/开发动作 |
| S3 | 业务失败分类完整 | 缺失会导致业务失败被误判（如"重复写"与"真失败"） |
| S4 | 开销合理 | 热路径 Histogram 数量从严控制 |


**第一类：可由其他指标推导（6 项，违反 S1）**

| 被裁指标 | 替代观测 |
|---|---|
| `dump/load_bandwidth_gbps`（Histogram ×2） | `rate(bytes) / (rate(duration_sum)/rate(duration_count)) / 1e6`（§4.2） |
| `dump/load_batch_entries`（Histogram ×2） | `rate(bytes)/rate(requests)` 近似；批量压力已由 prepare/transfer 时长分布体现 |
| `dump_ttl_ms`（Histogram） | TTL 为客户端入参透传，客户端侧可自行审计 |
| `interval_lookup_hit_rate`（Gauge） | hit/miss Counter 的 rate 比值精确推导（§4.4），窗口平滑优于单请求瞬时值 |

**第二类：非业务逻辑指标（16 项，违反 S0，此阶段不涉及）**

| 类别 | 被裁指标 | 说明 |
|---|---|---|
| 接入层协议/配置错误 | `requests_malformed_total`、`requests_unroutable_total` | 极低频，已有 UC_ERROR 日志；不影响业务成功率判断（客户端可见错误码） |
| 内部调度诊断 | `request_queue_wait_ms`、`request_queue_full_events_total` | 线程模型内部细节；排队影响最终由端到端 `response_rtt_ms` 分位数体现 |
| 低频异常分类 | `dump_duplicate_keys_total`、`load_len_mismatch_total` | 有日志可查；重复写不改变 dump/storebegin 失败语义（见 §3.3 B 组注） |
| 传输超时诊断 | `data_transfer_timeouts_total`、`response_transfer_timeouts_total` | 超时仅诊断性日志（源码注释确认传输仍持有 handle）；恢复或失败结果由 duration/failures 指标覆盖 |
| 内部池背压 | `flag_buffer_pool_full_total` | 影响（响应变慢/客户端超时）由 `response_rtt_ms` 间接体现 |
| 容量中间信号 | `evict_triggered_periodic/deep_total`、`evicted_entries_periodic/deep_total`、`buffer_alloc_failures_total` | 水位由 usage_ratio/metadata_entry_count 预警；最终分配失败的业务影响已被 `storebegin_failures` 计数覆盖 |
| GC 诊断 | `eviction_duration_ms`、`gc_cycles_total` | GC 线程低频内部任务，非业务信号 |

### 3.2 指标全集（22 个）

#### A. 请求量（3）

| 指标名 | 类型 | 记录形式 | 埋点位置 |
|---|---|---|---|
| `drampool_dump_requests_total` | Counter | 每请求 +1 | `task_worker.cc` `ProcessOneRequest()` DUMP 分支（L75） |
| `drampool_load_requests_total` | Counter | 每请求 +1 | 同上 LOAD 分支（L80） |
| `drampool_lookup_requests_total` | Counter | 每请求 +1 | 同上 LOOKUP 分支（L85） |

#### B. DUMP 业务（4）

| 指标名 | 类型 | 记录形式 | 埋点位置 |
|---|---|---|---|
| `drampool_dump_bytes_total` | Counter | 每请求 +Σ成功预留 entry 的 `len` | `task_worker.cc` `ProcessDump()` entries 循环内 StoreBegin 成功处累加（L128-146） |
| `drampool_dump_storebegin_failures_total` | Counter | 每次 StoreBegin 失败 +1（真失败；DuplicateKey 是正常结果不计入） | 同上 L133-140 |
| `drampool_dump_submit_failures_total` | Counter | 每请求传输提交失败 +1 | 同上 L159-168 |
| `drampool_dump_prepare_duration_ms` | Histogram | 每请求一次观测：函数入口 → ExecuteAsync 返回（含全部驱逐重试），`ScopedTimer` + 提交失败路径 `Disarm()`（§4.1） | 入口 L95 构造，L159-168 失败分支 Disarm |

> 注：重复 key entry 在循环内 `continue`（L129-132），不进入 StoreBegin 失败计数，故该指标天然只反映真失败，语义不受 duplicate_keys 裁剪影响。

#### C. LOAD 业务（4）

| 指标名 | 类型 | 记录形式 | 埋点位置 |
|---|---|---|---|
| `drampool_load_bytes_total` | Counter | 每请求 +Σ成功 LoadBegin 且长度合法 entry 的 `len` | `task_worker.cc` `ProcessLoad()` entries 循环（L184-227） |
| `drampool_load_miss_entries_total` | Counter | 每个 LoadBegin 失败 entry +1 | 同上 L187-192 |
| `drampool_load_submit_failures_total` | Counter | 每请求传输提交失败 +1 | 同上 L221-230 |
| `drampool_load_prepare_duration_ms` | Histogram | 每请求一次观测：入口 → ExecuteAsync 返回，`ScopedTimer` 同上 | 入口 L184 构造，L221-230 失败分支 Disarm |

> 注：`len_mismatch` 不单独计数，此类 entry 已计入 `load_miss_entries_total`（LoadBegin 返回失败），业务影响（未取到数据）语义一致。

#### D. LOOKUP 业务（2）

| 指标名 | 类型 | 记录形式 | 埋点位置 |
|---|---|---|---|
| `drampool_lookup_hit_entries_total` | Counter | 每个 Exists entry +1 | `task_worker.cc` `ProcessLookup()` 扫描循环（L274-278） |
| `drampool_lookup_miss_entries_total` | Counter | 每 batch 一次 +（`batch_size − hits`），循环外一次上报 | 同上 |

#### E. 传输与响应（7）

| 指标名 | 类型 | 记录形式 | 埋点位置 |
|---|---|---|---|
| `drampool_dump_transfer_duration_ms` | Histogram | 每请求一次观测：数据传输终态时刻 − `record.submit_ms` | `completion_poller.cc` `SettleDataTransfer()`（L260 起，统一出口，见 §4.2） |
| `drampool_load_transfer_duration_ms` | Histogram | 同上，按 opcode 二选一 | 同上 |
| `drampool_dump_transfer_failures_total` | Counter | 每个非 Completed 终态请求 +1 | 同上（`terminalStatus != Completed`） |
| `drampool_load_transfer_failures_total` | Counter | 同上 | 同上 |
| `drampool_response_rtt_ms` | Histogram | 每请求一次观测：响应写回 Completed 时刻 − 响应 submit_ms | `PollResponseTransfer()` Completed 分支（约 L255 后） |
| `drampool_response_transfer_failures_total` | Counter | 每请求响应写回失败 +1（GetStatus 异常或 Failed 终态） | 同上两个失败分支（L228-233、L248-252） |
| `drampool_response_submit_failures_total` | Counter | 每次 Pack 失败 / ExecuteAsync 失败 +1 | `SubmitResponse()` 两处错误分支（约 L187-191、L205-211） |

#### F. 资源水位（2 类，GCThreadLoop 每 `gcIntervalMs` 一轮采样，无热路径开销）

| 指标名 | 类型 | 记录形式 | 埋点位置 |
|---|---|---|---|
| `drampool_metadata_entry_count` | Gauge | 每轮覆盖写：`MetadataManager::GetKeyCnt()`（已存在，1024 分片求和，`metadata.h` L176-181） | `drampool_server.cc` `GCThreadLoop()`（L552-561），`PerformEvict()` 调用后采样 |
| `drampool_buffer_pool_usage_ratio_<slot_size>` | Gauge（**动态注册**，每个 block size 一个） | 每轮覆盖写：`BufferManager::GetUsedSlotRatio(size)`（新增，见 §4.3） | 同上，遍历 `g_config.poolBlockSizes` |

构成汇总：**5 个 Histogram**（全部 `_ms`，共用一组 duration 桶：queue/prepare/transfer/response RTT 中保留 prepare ×2 + transfer ×2 + RTT ×1）+ **15 个 Counter** + **2 类 Gauge**（其中 1 类动态注册）= 22。

---

## 4. 关键计算逻辑

### 4.1 prepare 耗时（ScopedTimer RAII）

`ScopedTimer` 在 `ProcessDump`/`ProcessLoad` 入口构造，仅**提交失败分支** `Disarm()`；空批次提前返回路径（无传输）视为 prepare 正常完成，正常析构观测。时钟用 `SteadyNowMs()`（`drampool_types.h` L48-53，steady_clock，不受系统跳变影响）。

### 4.2 数据传输时长（直测）与带宽（推导）

观测点（**统一出口，一处覆盖三条终态路径**——Completed、Failed、GetStatus 异常）：`completion_poller.cc` `SettleDataTransfer()` 末尾：

```cpp
if (record.submit_ms != 0) {
    const auto durationMs = SteadyNowMs() - record.submit_ms;
    const bool isDump = record.opcode == OpType::DUMP;
    MetricsObserve(isDump ? kDumpTransferDurationMs : kLoadTransferDurationMs,
                   static_cast<double>(durationMs));
    if (terminalStatus != transport::TransferStatus::Completed) {
        MetricsCount(isDump ? kDumpTransferFailures : kLoadTransferFailures, 1);
    }
}
```

带宽直方图，由 PromQL 推导（`_sum/_count` 为 histogram 附带序列）：

```promql
# 平均传输带宽 (GB/s)，以 DUMP 为例
rate(ucm:drampool_dump_bytes_total[5m])
  / (rate(ucm:drampool_dump_transfer_duration_ms_sum[5m])
     / rate(ucm:drampool_dump_transfer_duration_ms_count[5m]))
  / 1e6
```

- 时长复用现有 `record.submit_ms`（TaskWorker 提交传输时已写入，`task_worker.cc` L179），零额外字段成本。
- 空批次请求提前返回不进入数据传输轮询，天然不产生观测。

### 4.3 BufferManager 用量记账（drampool 模块内，不动共享 BufferPool）

`BufferManager::Allocate/Free` 是数据面缓冲池的**唯一出入口**（MetadataManager 的分配/释放全部经由它），在该层维护每尺寸原子计数：

```cpp
// buffer_manager.h
#include <atomic>
// 成员：
std::unordered_map<std::size_t, std::unique_ptr<std::atomic<std::size_t>>> usedSlots_;

// 构造函数建池时同步创建：usedSlots_.emplace(slotSize, std::make_unique<std::atomic<std::size_t>>(0));
// Allocate 成功后：++(*it->second)；Free 成功后：--(*it->second)（仅 pool->Free 返回 OK 时递减，
// 与池自身状态保持对账一致，不自行推断）
double GetUsedSlotRatio(std::size_t size) const;   // used / GetPool(size)->GetSlotCount()，池不存在返回 0
```

跨线程说明：Allocate/Free 会被 TaskWorker 线程（StoreBegin/LoadBegin/Delete 路径）与 GC 线程（EvictOneShard 路径）并发调用，故必须用 `std::atomic`。

flagBufferPool 是共享 `UC::BufferPool`（无用量查询接口），不做用量 Gauge，此阶段无背压指标（影响由 `response_rtt_ms` 间接体现）。

### 4.4 LOOKUP 命中率（Counter 直测）

`ProcessLookup()` 扫描循环内累计 `hits`；循环后一次上报：

```cpp
MetricsCount(kLookupHitEntries, hits);
MetricsCount(kLookupMissEntries, request.batch_size - hits);
```

命中率由 PromQL 推导（窗口平滑）：

```promql
sum(rate(ucm:drampool_lookup_hit_entries_total[5m]))
  / (sum(rate(ucm:drampool_lookup_hit_entries_total[5m]))
     + sum(rate(ucm:drampool_lookup_miss_entries_total[5m])))
```

---

## 5. DramPool 侧实现设计

### 5.1 新增 `drampool_metrics.h` / `drampool_metrics.cc`（注册器 + 采样辅助 + ScopedTimer + 名字常量）

```cpp
#pragma once
#include <string>
#include "metrics_api.h"   // ucm/shared/metrics/cc/api

namespace UC::DramPool {

// 指标名常量（节选，全部集中于此，禁止散落字符串字面量）
inline constexpr char kDumpRequestsTotal[]   = "drampool_dump_requests_total";
inline constexpr char kDumpPrepareDuration[] = "drampool_dump_prepare_duration_ms";
inline constexpr char kLookupHitEntries[]    = "drampool_lookup_hit_entries_total";
// ... 完整清单见 §3.3

inline void MetricsCount(const char* name, double delta)   { UC::Metrics::UpdateStats(name, delta); }
inline void MetricsSet(const char* name, double value)     { UC::Metrics::UpdateStats(name, value); }
inline void MetricsObserve(const char* name, double value) { UC::Metrics::UpdateStats(name, value); }

// RAII 耗时打点：构造计时、析构写 Histogram，零侵入业务逻辑；
// 失败路径调用 Disarm() 取消本次观测（用于"仅成功路径上报"的 prepare duration）
class ScopedTimer {
public:
    explicit ScopedTimer(const char* histName)
        : name_(histName), start_(SteadyNowMs()), armed_(true) {}
    void Disarm() { armed_ = false; }
    ~ScopedTimer() {
        if (armed_) MetricsObserve(name_, static_cast<double>(SteadyNowMs() - start_));
    }
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;
private:
    const char* name_;
    std::uint64_t start_;
    bool armed_;
};

// 启动期一次性注册（幂等，内部用 std::once_flag）：
//   1. UC::Metrics::SetUp(g_config.metricsHistogramVectorLen)
//   2. CreateStats 全部静态指标（COUNTER/GAUGE/HISTOGRAM）
//   3. 遍历 g_config.poolBlockSizes 动态注册 drampool_buffer_pool_usage_ratio_<size>
// 失败仅 UC_WARN，不阻塞 daemon 启动（metrics 故障不应拖垮数据面）
Status InitDramPoolMetrics();

}  // namespace UC::DramPool
```

### 5.2 数据结构：零扩展

`RequestTask` / `CompletionRecord` **均保持原样**（原计划的 `enqueue_ms`、`transfer_bytes` 分别随 queue_wait、bandwidth 指标裁剪而取消）。SPSC 队列布局与既有逻辑零影响。

### 5.3 BufferManager 用量记账

见 §4.3。改动仅限 `buffer_manager.h`（header-only，无 .cc）。

### 5.4 配置项

`drampool_config.h` `DramPoolConfig` 新增（5 个 Histogram 全部为 `_ms`，桶配置只需一组）：

```cpp
// Metrics snapshot (JSON Lines) + reflow pipeline.
bool metricsEnabled{true};
std::string metricsLogPath{"./drampool_metrics.log"};   // 与 UCM 侧回流器的 log_path 约定一致
std::uint32_t metricsSnapshotIntervalMs{10000};         // 快照周期
std::uint32_t metricsLogMaxBytes{64 * 1024 * 1024};     // 按大小轮转阈值
std::uint32_t metricsLogMaxFiles{5};                    // 轮转保留个数，超出删除最旧
std::uint32_t metricsHistogramVectorLen{10000};         // UC::Metrics::SetUp(maxVectorLen)
std::vector<double> metricsDurationBucketsMs{1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000};
```

`drampool_yaml_config.cc` `ParseYamlConfig()` 新增可选 `metrics:` 节（缺省用以上默认值）：

```yaml
# examples/drampool.yaml 追加
metrics:
  enabled: true
  log_path: "./drampool_metrics.log"
  snapshot_interval_ms: 10000
  log_max_bytes: 67108864
  log_max_files: 5
  histogram_vector_len: 10000
  duration_buckets_ms: [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
```

不新增 CLI 开关；`metrics.enabled=false` 时 InitDramPoolMetrics 与 Reporter 线程均不启动，打点辅助退化为空操作（旁路关闭零成本）。

### 5.5 CMake（`ucm/store/dram/CMakeLists.txt`）

两处改动（`BUILD_UCM_DRAMPOOL` 块内）：

1. `add_library(drampool_core STATIC ...)` 的显式源文件列表（L40-49）追加 `drampool_metrics.cc` 与 `drampool_metrics_reporter.cc`（该列表**非 GLOB**，漏加会直接链接失败——构建期防线）。
2. `target_link_libraries(drampool_core PUBLIC ...)`（L55-65）追加 `metrics`（`ucm/shared/metrics` 静态库，PUBLIC include 随链接传递）。

HealthServer 不做任何改动。

---

## 6. 快照与 JSON Lines 日志（新增 `drampool_metrics_reporter.h` / `.cc`）

### 6.1 周期快照机制（Reporter 线程）

```
MetricsReporter::Run()（独立线程，daemon Start 启动、Stop 退出）:
  while (!stopping_):
    sleep_for(metricsSnapshotIntervalMs)   // 可被 Stop 唤醒
    (counters, gauges, hists) = UC::Metrics::GetAllStatsAndClear()
    // —— 累计值维护（保证每行 JSON 为独立终点状态）——
    for (name, delta) in counters: counterTotals_[name] += delta
    for (name, v) in gauges:       gaugeLatest_[name] = v          // 覆盖最新采样
    for (name, values) in hists:   histTotals_[name] += (sum, count, 按桶分箱计数)
    line = RenderJsonLine(ts=NowEpochSec, interval, counterTotals_, gaugeLatest_, histTotals_)
    flushQueue_.TryPush(std::move(line))   // 有界；满则丢行 + UC_WARN（旁路非阻塞）
```

- **Histogram 在 C++ 侧分桶**（不输出原始值向量）：桶边界取 §5.4 `metricsDurationBucketsMs`（5 个 histogram 全部为 `_ms` 后缀，共用同一组桶；超出最大桶的值计入 `+Inf`）。原因：原始向量一行可达数百 KB，而桶计数后单行 < 4KB，且回流器可直接映射 Prometheus histogram。
- 10s 窗口内的 `maxVectorLen` 截断只影响该窗口桶计数偏差，不跨周期累积（每轮 GetAllStatsAndClear 后向量清空）。
- 内存开销：`maxVectorLen=10000` × 8B × 5 个 histogram × ~4 个打点线程 ≈ **1.6 MB**。
- 单行大小预算：5 个 histogram ×（~15 桶 + sum/count）+ ~17 个 counter/gauge ≈ **< 4KB**，远低于回流器 64KiB 读取窗口。

### 6.2 JSON Lines 行格式（一行 = 一个快照）

```json
{"ts": 1725868800.123, "interval_ms": 10000, "pid": 12345, "host": "node-a", "metrics": {
  "drampool_dump_requests_total":   {"type": "counter", "value": 12345},
  "drampool_dump_bytes_total":      {"type": "counter", "value": 9876543210},
  "drampool_metadata_entry_count":  {"type": "gauge",   "value": 10240},
  "drampool_buffer_pool_usage_ratio_1048576": {"type": "gauge", "value": 0.62},
  "drampool_dump_transfer_duration_ms": {
    "type": "histogram", "count": 12345, "sum": 45678.5,
    "buckets": {"1": 10200, "2": 12300, "5": 13100, "+Inf": 12345}
  }
}}
```

- 单行无内嵌换行（compact 序列化），文件尾始终以 `\n` 结束，保证回流器"完整行 = 完整快照"。
- 字段名直出，彻底摆脱位置编码与白名单耦合。

### 6.3 异步刷盘与轮转清理（MetricsFlush 线程）

```
MetricsFlush::Run():
  while (!stopping_):
    line = flushQueue_.TryPop(带超时，支持 Stop 唤醒)
    if line: AppendAndMaybeRotate(line)
```

- `AppendAndMaybeRotate`：以 append 模式写 `metricsLogPath`；写前检查当前文件大小 ≥ `metricsLogMaxBytes` → 关闭 → 依次 rename `.log.(k-1)→.log.k`、`.log→.log.1` → 重建 `.log` → 删除超出 `metricsLogMaxFiles` 的 `.log.k`。
- 崩溃/掉电容忍：JSON 行可能偶发半行（append 中断）——回流器按"忽略末尾半行"规则天然兼容（§7.2）。

### 6.4 旁路非阻塞与容错（贯穿全链路）

- 打点、快照、刷盘任何一步失败**仅 UC_WARN/UC_ERROR**，绝不阻塞或终止 DUMP/LOAD/LOOKUP 主流程。
- 文件打开失败：Flush 线程重试（每轮 WARN，不退出），业务打点不受影响。
- `metrics.enabled=false`：全链路旁路关闭。

---

## 7. UCM/Scheduler 侧回流器（Python，推理进程内）

### 7.1 Leader 选举

```python
lock_path = f"/tmp/ucm_metrics_reflow_{hashlib.sha1(f'{endpoint}|{log_path}'.encode()).hexdigest()[:16]}.lock"
fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
try:
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)   # 启动时尝试一次
except BlockingIOError:
    return   # 失败者静默退出，运行期不重选
```

- 锁文件名由 `endpoint + log_path` 哈希生成：同一（DramPool 实例, 日志文件）组合在多个推理实例场景下只有一个 Leader 独占回流职责，避免重复计数。
- 进程终止自动释放 flock，当前推理实例停止回流，不影响业务；运行期不重选。

### 7.2 高效读取（成本与日志文件大小无关）

```python
def read_latest_snapshot(path, window=64 * 1024):
    size = os.path.getsize(path)          # 文件缺失/为空 → 返回 None，跳过本轮
    with open(path, "rb") as f:
        f.seek(max(0, size - window))     # seek(EOF) 后从尾部向前读取 64KiB
        data = f.read()
    if size > window:
        data = data[data.find(b"\n") + 1:]   # 忽略首段半行（含末尾半行由取"最后完整行"规避）
    lines = [l for l in data.splitlines() if l.strip()]
    return json.loads(lines[-1])          # 最新完整 JSON 快照；解析失败 → 跳过本轮
```

- 只定位最新完整 JSON 快照，读取成本恒定 O(window)。
- 容错：文件缺失/为空/末尾半行/JSON 解析失败 → 跳过本轮，等待下一周期，**不更新 State**。

### 7.3 增量计算（累计转增量）

```python
def delta_or_reset(current, previous):
    # Counter/Histogram（sum、count、每个桶各自适用）
    if current >= previous:
        return current - previous
    return None   # current < previous 视为 drampool 重启/Reset → 丢弃本轮该指标，以 current 作为新 baseline
```

- **Gauge 直接获取最新值**，无增量语义。
- State 缺失（首次启动）：将当前累计值作为 **Baseline**，避免首次启动回灌历史全量。
- JSON 解析失败/State 损坏：不更新 State，降级为无状态模式（下一轮重新校准）。

### 7.4 状态持久化（先回流、后落盘）

```python
def save_state(state, path):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(state, f)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)                 # 原子替换
```

- 顺序保证：**先**将增量喂给 ucmmetrics，**后**原子更新 State——回流成功但落盘失败只会导致下一轮多计一次极小增量，绝不丢指标。
- State 文件路径：`<log_path>.state`（与日志文件同目录，按 `log_path` 区分实例）。

### 7.5 对接 ucmmetrics → vllm_connector → Prometheus

- 回流器以独立 Python 组件实现（建议模块 `ucm/metrics_reflow/`，含 `leader.py` / `tail_reader.py` / `reflow.py` / `state.py`），由 UCM Scheduler 在推理进程内启动并挂载，挂载参数为 `endpoint`（DramPool 地址）与 `log_path`（与 DramPool `metrics.log_path` 一致）。
- 指标注册：按 JSON 行内 `type` 动态注册到 ucmmetrics（名字加 `ucm:` 前缀）：
  - counter → 注册后每次喂增量（累计语义由 ucmmetrics/Prometheus 侧维护）；
  - gauge → 每次 set 最新值；
  - histogram → 喂桶计数（`le` 桶 + `_sum` + `_count`）；若 ucmmetrics 的 histogram 接口仅支持原始值 observe，则降级为自定义 Collector 直设桶计数（prometheus_client `HistogramMetricFamily` 语义），实现期二选一并固化。
- 暴露：沿用 vllm_connector 现有 `/metrics` 端点，Prometheus 侧无任何新增配置。

---

## 8. 测试方案

### 8.1 C++ 单测（`ucm/store/test/case/dram/drampool_metrics_test.cc`，gtest）

| 用例 | 验证点 |
|---|---|
| RegistrationCoverage | 遍历 `drampool_metrics.h` 全部名字常量 `CreateStats` 后 `UpdateStats`，`GetAllStatsAndClear` 必须能取到值——**防 C2 静默丢弃** |
| CounterAccumulates | 同名多次 UpdateStats，读出增量 = 各次之和 |
| GaugeOverwrites | 同名两次 UpdateStats，读出 = 最后一次 |
| HistogramOverflow | 观测数 > `metricsHistogramVectorLen` 后向量封顶（记录溢出行为符合 C3 预期） |
| ScopedTimerDisarm | 正常析构观测一次；Disarm 后析构不观测 |
| ReporterCumulativeSnapshot | 模拟两轮 UpdateStats + 两次快照，JSON 中 counter 为单调累计值、gauge 为最新值 |
| ReporterHistogramBuckets | 构造已知值集（含超出最大桶）→ duration 桶计数含 `+Inf`，sum/count 正确 |
| JsonLineFormat | 输出为单行合法 JSON、`type` 字段齐全、无内嵌换行 |
| FlushRotation | 写满 `metricsLogMaxBytes` → 轮转产生 `.log.1`；超出 `maxFiles` → 最旧被删 |
| BufferManagerUsedSlots | Allocate/Free 对账：分配 N 后 ratio = N/capacity，全部释放归零 |

### 8.2 Python 单测（回流器）

| 用例 | 验证点 |
|---|---|
| LeaderElection | flock 独占成功/第二次抢占失败静默退出；进程退出释放锁 |
| TailReader | 正常行、末尾半行、空文件、文件缺失、超窗口首行截断 |
| DeltaCalculation | 正常增量；current < previous 触发 reset（丢弃 + 新 baseline）；gauge 直取 |
| StatePersistence | tmp + os.replace 原子性；State 损坏 → 无状态降级；缺失 → baseline 模式 |
| ReflowToUcmmetrics | 注册/喂增量对接（mock ucmmetrics 接口） |

### 8.3 E2E（`ucm/store/test/e2e/scripts/run_drampool_e2e.sh`）

现有流程（起 drampool → offline_inference.py → 查询）结束后追加：

```bash
# 1) DramPool 侧：JSON Lines 日志产生快照行
sleep 12   # ≥ snapshot_interval，等待 Reporter 产出
test -s "${METRICS_LOG}" || { echo "metrics log missing"; exit 1; }
tail -n 1 "${METRICS_LOG}" | python3 -c \
  "import json,sys; m=json.load(sys.stdin)['metrics']; assert m['drampool_dump_requests_total']['value']>0"

# 2) 全链路（若回流器随推理进程挂载）：Prometheus 端点可见
curl -sf "http://127.0.0.1:${VLLM_METRICS_PORT}/metrics" | grep -q "^ucm:drampool_dump_requests_total"
```

### 8.4 手工验证清单

- Grafana/PromQL 冒烟：`rate(ucm:drampool_dump_bytes_total[1m])`、§4.2 带宽推导式、§4.4 命中率推导式、`histogram_quantile(0.99, rate(ucm:drampool_dump_transfer_duration_ms_bucket[5m]))`、`ucm:drampool_buffer_pool_usage_ratio_*`。
- 杀掉 drampool → 回流器检测 reset，不产生负增量；重启 drampool → 恢复回流。
- 双推理实例共享同一 log_path → 仅一个 Leader 回流，指标无重复计数。

---

## 9. 实施阶段划分

| 阶段 | 内容 | 涉及文件 |
|---|---|---|
| P1 基础设施 | CMake 链接 + `drampool_metrics.{h,cc}`（注册器/辅助/ScopedTimer/常量）+ 配置项与 YAML + §4.3 BufferManager 记账 | `dram/CMakeLists.txt`、`drampool_metrics.h/.cc`（新）、`drampool_config.h`、`drampool_yaml_config.cc`、`examples/drampool.yaml`、`buffer_manager.h` |
| P2 数据面埋点 | §3.3.A-F 全部采样点落位 | `task_worker.cc`、`completion_poller.cc`、`drampool_server.cc`（仅 GCThreadLoop 采样点） |
| P3 快照与日志 | Reporter + JSON Lines + Flush/轮转 | `drampool_metrics_reporter.h/.cc`（新）、`drampool_daemon.cc`（线程启停挂载） |
| P4 UCM 侧回流器 | Leader 选举、尾部读取、增量、State、ucmmetrics 对接 | `ucm/metrics_reflow/`（新 Python 模块）+ Scheduler 挂载点 |
| P5 测试与文档 | §8 单测 + E2E 断言 + `docs/source/developer-guide/add_metrics.md` 补充"跨进程 JSON Lines 回流"一节 | 测试文件（新）、`run_drampool_e2e.sh`、docs |

顺序依赖：P2 依赖 P1 名字常量；P3 依赖 P1 配置项；P4 与 P2/P3 可并行（接口契约 = §6.2 JSON 格式 + §5.4 配置项）；P5 贯穿。

---

## 10. 已明确的取舍与风险

| # | 事项 | 决策 / 缓解 |
|---|---|---|
| 1 | **业务聚焦原则（44 → 22）** | 仅保留 DUMP/LOAD/LOOKUP 服务质量 + 资源水位指标（§3.1 S0）；接入层协议错误、调度诊断、超时诊断、驱逐/GC 中间信号等 16 项非业务指标此阶段不涉及，相应路径维持 WARN/ERROR 日志。后续需要时按 §3.2 清单逐项恢复即可，埋点位置均已标注 |
| 2 | 可推导指标裁剪（6 项） | 带宽/批量/命中率均给出 PromQL 推导式（§4.2/§4.4）；若未来需要带宽**分位数**，恢复 2 个 histogram 即可 |
| 3 | 框架无标签维度 | op/slot-size 维度全部扁平化为独立指标名；动态注册仅限启动期 |
| 4 | Histogram 满后丢弃新样本（C3） | 观测点收敛在请求粒度；`metricsHistogramVectorLen` 可配置；截断只影响单窗口桶偏差；单测固化溢出行为 |
| 5 | 数据结构零扩展 | `RequestTask`/`CompletionRecord` 原样保留（queue_wait、bandwidth 裁剪的连锁收益），SPSC 队列布局零影响 |
| 6 | storebegin_failures 语义完整性 | DuplicateKey 是正常业务结果（循环内 continue，不进失败分支），裁掉 duplicate_keys 不改变该指标语义；len_mismatch entry 天然计入 load_miss_entries |
| 7 | 共享 BufferPool 无用量查询接口 | 数据池用量在 drampool 层 BufferManager 记账（§4.3）；flagBufferPool 此阶段无指标，影响由 response_rtt 体现 |
| 8 | 单行大小 vs 64KiB 读取窗口 | C++ 侧预分桶，5 个 histogram 单行 < 4KB；窗口大小可配置兜底 |
| 9 | 双侧"累计"语义分工 | C++ Reporter 维护进程级累计（每行 JSON 为终点状态）；Python 回流器负责累计→增量 + reset 检测（current < previous）；State 保证跨回流器重启连续性 |
| 10 | 超时场景可观测性 | timeouts 指标裁剪后，超时仅 WARN 日志；恢复/失败结果由 duration/failures 指标与 rtt 分位数覆盖 |
| 11 | `drampool_core` 源文件列表非 GLOB | 新 .cc 漏加直接链接失败（构建期暴露） |
| 12 | 范围隔离 | DramStore 零改动；HealthServer 零改动；共享组件零改动；数据结构零扩展；`metadata.cc` 无需修改（驱逐指标全部裁剪） |
| 13 | 跨进程时钟不一致 | `ts` 仅用于排序/诊断；增量计算基于指标值本身，不依赖跨进程时钟对齐 |
| 14 | 多推理实例共享同一日志 | Leader 选举保证单回流者；不同 `endpoint + log_path` 组合各自独立 Leader |
| 15 | 日志轮转与尾读竞态 | 回流器始终取"最新完整行"，轮转 rename 只影响旧文件；文件被删/缺失 → 跳过本轮 |
| 16 | 旁路非阻塞红线 | 打点/快照/刷盘/回流全链路失败仅 WARN，主流程（DUMP/LOAD/LOOKUP、推理）永不阻塞 |
