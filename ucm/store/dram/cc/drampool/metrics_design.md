# DramPool Metrics 埋点与跨进程回流设计方案

| 项 | 说明 |
|---|---|
| 范围 | DramPool 服务端守护进程（`ucm/store/dram/cc/drampool/`）**业务逻辑**埋点 + 快照日志输出；UCM/Scheduler 侧新增回流器组件。DramStore 客户端侧零改动，HealthServer 零改动 |
| 核心诉求 | 解决 DramPool 独立进程与 vLLM/UCM 跨进程指标统一暴露，对接现有 Prometheus 监控链路 |
| 设计原则 | **业务聚焦**：保留直接反映 DUMP/LOAD/LOOKUP 服务质量（请求量 / 成功率 / 时延 / 命中率）、**metadata 各阶段耗时**与资源水位的指标。接入层协议错误、内部调度诊断、低频异常分类等**非必要指标此阶段不涉及**（相应路径维持 UC_WARN/UC_ERROR 日志） |
| 共享代码约束 | 不修改 `ucm/shared/pool`、`ucm/shared/infra/template` 等共享组件；仅新增对 `ucm/shared/metrics` 的链接依赖 |
| 指标规模 | **38 个**（15 Histogram + 19 Counter + 4 类 Gauge） |
| 数据结构影响 | **零扩展**（`RequestTask`/`CompletionRecord` 均保持原样） |
| 时钟精度 | 新增 `SteadyNowUs()`（`drampool_types.h`）：metadata 单次操作为微秒级，毫秒精度时钟无法分辨；G 组计时内部以 µs 计、以 ms（double）入直方图 |

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
| C3 | **Histogram 存原始值向量**，上限 `maxVectorLen`，**满后丢弃后续观测**（保留先到的样本） | 业务级指标（A-F、H 组）控制在请求 / batch 粒度；G 组为 entry / 调用级（独立归因价值，§3.2.G），高吞吐下窗口截断偏差见 §10-4；向量长度可配置 |
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
| **S0** | **业务直接相关** | 直接反映 DUMP/LOAD/LOOKUP 操作的服务质量（请求量、成功率、时延、命中率）、**metadata 各阶段耗时**或资源水位（池用量、元数据规模）。纯系统性/接入层/内部调度/低频异常分类指标**此阶段不涉及** |
| S1 | 不可推导 | 无法由其他指标经 PromQL 运算（rate 比值、差值）等价获得 |
| S2 | 可行动 | 指标异常时存在明确的运维/开发动作 |
| S3 | 业务失败分类完整 | 缺失会导致业务失败被误判（如"重复写"与"真失败"） |
| S4 | 开销合理 | 热路径 Histogram 数量从严控制 |

**metadata 阶段耗时（G 组）的补充判定说明**：

- 用户的观测目标："metadata 各阶段用时 + 总用时，定位哪一阶段有问题或对性能影响大"。阶段拆分只保留**两个有独立归因价值的切面**：`allocate`（缓冲分配，含驱逐重试——内存压力信号）与 `shard register`（分片锁 + 驱逐策略——锁竞争信号）；驱逐再按**触发来源**拆为 `evict_sync`（业务线程内，直接拖慢 DUMP）与 `evict_gc`（后台线程，抢锁干扰）。其余单阶段原子操作（StoreEnd/LoadBegin/LoadEnd）无内部分阶段，仅观测总耗时。
- **本次增补中仍被裁掉的项**（留痕，恢复方式见 §10）：
  - `metadata_delete_duration_ms`：Delete 仅出现于失败清理（DUMP 提交失败回滚 / StoreEnd 失败清理）与驱逐逐 victim 释放两类非正常路径，前者低频且日志完备，后者已包含在 evict 直方图内——独立直方图不满足 S1/S2。
  - 驱逐内部 scan/release 二次拆分：来源拆分（sync/gc）优先级更高；若 sync 驱逐 P99 异常需进一步细挖，再恢复（观测点在 `EvictOneShard` 内部，位置已在 §3.2.G 标注）。
  - `Exist` 单次耗时直方图：LOOKUP 批量大（每 entry 一次观测会触发 C3 溢出），改由 `lookup_scan_duration_ms` 按 batch 粒度覆盖，单 key 均值可由 `rate(scan_sum)/rate(lookup_requests)/batch_size` 推导。


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

**第三类：失败类合并（6 项 → 3 个大类，按必要性审查）**

> 审查标准：失败类指标必须满足两条之一才单独设立——(a) **明确归因价值**：失败直接指向可行动根因（如缓冲分配失败 = 内存压力）；(b) **高发生概率**：真实负载下常见、需要告警基线。不满足者不单独设计，语义相近者合并为统一大类——按**管线阶段**归因保留定位能力，去掉 opcode 细分（低频事件下 opcode 维度无排查增益）。

| 处置 | 指标 | 理由 |
|---|---|---|
| **保留** | `dump_storebegin_failures_total` | 内存不足的**直接归因**指标：分配失败 → 扩容/驱逐策略调整，有明确行动入口（§3.2.B） |
| **保留** | `queue_request_full_total` / `queue_completion_full_total` / `queue_response_buffer_retry_total` | 三条阻塞链（§4.6）的**唯一量化信号**；B3 自旋停摆无日志兜底，无此指标则停摆不可观测，发生概率随负载压力显著上升 |
| **合并** | `dump_submit_failures` + `load_submit_failures` → **`drampool_submit_failures_total`** | 低频（传输子系统故障才出现）；合并后仍指向"传输提交失败"单一根因，排查动作与 opcode 无关 |
| **合并** | `dump_transfer_failures` + `load_transfer_failures` → **`drampool_transfer_failures_total`** | 低频（对端网络异常时出现）；排查动作统一为查连接/对端状态，opcode 细分无增益 |
| **合并** | `response_transfer_failures` + `response_submit_failures` → **`drampool_response_failures_total`** | 低频；合并后仍锁定"响应返回链路"（提交/写回）故障域，结合 WARN 日志定位环节 |

**第四类：buffer 池必要性审查（本次增补，新增 1 / 不设计留痕 4 类）**

> 池盘点：DramPool 共两类 buffer 池，均基于共享 `UC::BufferPool`（无用量查询接口，不可改）。**数据池** N 个按 block size 分池，由 drampool 层 `BufferManager` 持有，唯一出入口 `Allocate/Free`，跨 TaskWorker/GC 多线程并发；**flag 池** 1 个（`DramPoolServer::flagBufferPool_` 直接持有，独立 BufferRegion），仅 CompletionPoller 单线程访问，用于响应 flag 缓冲。

| 处置 | 指标 | 理由 |
|---|---|---|
| **新增** | `drampool_flag_pool_usage_ratio`（Gauge，F 组） | (a) B2 链（flag 池 NoSpace → `response_buffer_retry`）的**前兆信号**：水位逼近 1 时重试概率高；(b) §4.6.3 决策表"flag 池扩容"动作此前无水位指标可交叉验证，补缺口；(c) **高概率**——响应突发场景常见 |
| 不设计 | 数据池 alloc/free 计数（Counter ×2） | rate 差可推导（违反 S1），且水位已由 `buffer_pool_usage_ratio_<slot_size>` 直测覆盖 |
| 不设计 | 数据池 NoSpace 驱逐触发计数 | §3.1 第二类已裁（`buffer_alloc_failures_total`），业务影响由 `storebegin_failures_total` 覆盖 |
| 不设计 | size 未注册 / Free 失败计数 | bug 级低频路径，UC_ERROR 日志兜底 |
| 不设计 | 池字节绝对值 Gauge | ratio × capacity 可推导（违反 S1），ratio 已满足水位告警需求 |

### 3.2 指标全集（38 个）

> 功能描述统一为三段式：**测什么**（观测口径）→ **反映什么**（异常时指向的问题）→ **怎么用**（典型分析动作）。

#### A. 请求量（3）

| 指标名 | 类型 | 功能（测什么 / 反映什么 / 怎么用） | 埋点位置 |
|---|---|---|---|
| `drampool_dump_requests_total` | Counter | 累计接收并处理的 DUMP 请求数（每请求 +1，与 batch 内 entry 数无关）。反映写入负载规模；与 `dump_bytes_total` 配合可得平均批量，与 load/lookup 请求量对比反映负载构成 | `task_worker.cc` `ProcessOneRequest()` DUMP 分支（L75） |
| `drampool_load_requests_total` | Counter | 累计接收并处理的 LOAD 请求数。反映读取负载规模；读取侧所有 rate 类指标（miss、transfer 时长）的分母基准 | 同上 LOAD 分支（L80） |
| `drampool_lookup_requests_total` | Counter | 累计接收并处理的 LOOKUP 请求数。反映查询负载规模；也是 `lookup_scan_duration_ms` 均值换算"单请求查询成本"的分母 | 同上 LOOKUP 分支（L85） |

#### B. DUMP 业务（3）

| 指标名 | 类型 | 功能（测什么 / 反映什么 / 怎么用） | 埋点位置 |
|---|---|---|---|
| `drampool_dump_bytes_total` | Counter | 累计**成功预留缓冲**的 DUMP entry 字节总量（循环内 Σ `entry.len`；DuplicateKey 与分配失败的 entry 不计）。测 DUMP 吞吐规模（口径为"预留成功"，传输失败回滚的字节已计入本指标，不代表最终落池数据量）。用于吞吐趋势、带宽推导（§4.2）与容量规划 | `task_worker.cc` `ProcessDump()` entries 循环内 StoreBegin 成功处累加（L128-146） |
| `drampool_dump_storebegin_failures_total` | Counter | 累计 StoreBegin **真失败**的 entry 数（= 缓冲分配失败或分片注册失败；重复写同一 key 是幂等正常结果，不计入）。测 DUMP 写入失败的元数据侧原因。持续增长指向内存压力（分配失败）或元数据状态异常（注册失败），结合 G 组 allocate 直方图区分。**失败类必要性**：内存不足的直接归因指标（§3.1 第三类） | 同上 L133-140 |
| `drampool_dump_prepare_duration_ms` | Histogram | 每请求一次观测：DUMP 从开始处理到数据传输提交完成的**本地准备耗时**（逐 entry StoreBegin + 缓冲分配 + 可能的驱逐重试 + 提交），**不含**数据传输本身。测写入路径的服务端 CPU 侧开销。P99 高说明准备阶段拖慢写入，用 G 组指标按阶段归因（§4.5） | 入口 L95 构造 `ScopedTimer`，L159-168 失败分支 Disarm（§4.1） |

> 注：重复 key entry 在循环内 `continue`（L129-132），不进入 StoreBegin 失败计数，故该指标天然只反映真失败，语义不受 duplicate_keys 裁剪影响。

#### C. LOAD 业务（3）

| 指标名 | 类型 | 功能（测什么 / 反映什么 / 怎么用） | 埋点位置 |
|---|---|---|---|
| `drampool_load_bytes_total` | Counter | 累计**成功加引用**的 LOAD entry 字节总量（LoadBegin 成功且长度合法的 Σ `entry.len`）。测 LOAD 吞吐规模（口径为"加引用成功"，未命中/长度不匹配的 entry 不计）。用于读取吞吐趋势与读取带宽推导 | `task_worker.cc` `ProcessLoad()` entries 循环（L184-227） |
| `drampool_load_miss_entries_total` | Counter | 累计 LOAD 中**未取到数据**的 entry 数（key 不存在、状态非 READY、请求长度大于存储长度，均计入——对客户端都是"没取到"）。测读取未命中规模。突增说明客户端读取了已被驱逐/过期/尚未写完的数据；命中率 = 1 − miss/(miss + 成功 entry) | 同上 L187-192 |
| `drampool_load_prepare_duration_ms` | Histogram | 每请求一次观测：LOAD 从开始处理到传输提交完成的**本地准备耗时**（逐 entry LoadBegin + 长度校验 + 提交），**不含**传输本身。测读取路径服务端 CPU 侧开销；与 G 组 `loadbegin_duration_ms` 联动归因 | 入口 L184 构造，L221-230 失败分支 Disarm |

#### D. LOOKUP 业务（3）

| 指标名 | 类型 | 功能（测什么 / 反映什么 / 怎么用） | 埋点位置 |
|---|---|---|---|
| `drampool_lookup_hit_entries_total` | Counter | 累计 LOOKUP 中**存在且 READY**（命中并刷新 TTL lease）的 entry 数。测查询命中规模；命中率的分子（§4.4） | `task_worker.cc` `ProcessLookup()` 扫描循环（L274-278） |
| `drampool_lookup_miss_entries_total` | Counter | 累计 LOOKUP 中**不存在或非 READY** 的 entry 数（batch_size − hits，每 batch 一次上报）。测查询未命中规模；突增说明客户端查询了已被驱逐/过期的前缀，通常先于 LOAD miss 出现 | 同上 |
| `drampool_lookup_scan_duration_ms` | Histogram | 每 batch 一次观测：LOOKUP 元数据扫描总耗时（batch 内全部 `Exist` 调用 + 结果填充，**不含**响应写回）。LOOKUP 无数据传输，该耗时即 LOOKUP 的服务端主体处理时长；均值 ÷ 平均 batch_size = 单 key 查询成本，P99 高说明分片读锁竞争或 TTL lease 刷新开销大 | `ProcessLookup()` 扫描循环外层 `ScopedTimer`（L273 构造，L284 前析构） |

#### E. 传输与响应（6）

| 指标名 | 类型 | 功能（测什么 / 反映什么 / 怎么用） | 埋点位置 |
|---|---|---|---|
| `drampool_dump_transfer_duration_ms` | Histogram | 每请求一次观测：DUMP 数据传输从提交（`submit_ms`）到**终态**（Completed/Failed/GetStatus 异常）的异步耗时，涵盖网络与对端读取。测实际搬运能力；`rate(bytes)/均值` 即传输带宽（§4.2）。与 prepare 相加近似 DUMP 服务端端到端 | `completion_poller.cc` `SettleDataTransfer()` 统一出口（L260 起，§4.2） |
| `drampool_load_transfer_duration_ms` | Histogram | 同上，LOAD 侧（池→客户端写入方向的搬运时长） | 同上，按 opcode 二选一 |
| `drampool_transfer_failures_total` | Counter | 累计数据传输以**非 Completed 终态**结束的请求数（DUMP/LOAD 合并大类，§3.1 第三类；Failed 或 GetStatus 异常均计）。测传输失败率；增长需排查对端网络/连接状态（排查动作与 opcode 无关，低频事件不细分） | `SettleDataTransfer()` 统一出口（`terminalStatus != Completed`，DUMP/LOAD 同点合并计数） |
| `drampool_response_rtt_ms` | Histogram | 每请求一次观测：响应传输**提交成功**（`ExecuteAsync` 成功后写入 `record.submit_ms`）到**写回客户端内存完成**（Completed）的耗时。测结果返回阶段的传输时延；P99 高指向对端写入慢或写回链路异常（flag 池 NoSpace 重试等待与 Pack 耗时发生在提交前、不在本窗口内，该背压由 `response_buffer_retry_total` / `flag_pool_usage_ratio` / `completion_inflight` 覆盖） | `PollResponseTransfer()` Completed 分支（约 L255 后；起点 `record.submit_ms`，completion_poller.cc L215） |
| `drampool_response_failures_total` | Counter | 累计响应返回链路失败计数（**本地提交**失败 Pack/ExecuteAsync + **写回传输**失败 GetStatus 异常/Failed 终态，DUMP/LOAD 合并大类，§3.1 第三类；flag 池 NoSpace 属重试不算失败）。测结果返回通道健康度；增长锁定响应链路故障域，结合 WARN 日志定位提交/写回哪一环 | `SubmitResponse()` 两处错误分支（约 L187-191、L205-211）+ `PollResponseTransfer()` 两个失败分支（L228-233、L248-252），四点合并计数 |
| `drampool_submit_failures_total` | Counter | 累计数据传输**提交失败**的请求数（DUMP/LOAD 合并大类，§3.1 第三类；ExecuteAsync 失败或 handle 无效；LOAD 侧此时整批已 LoadEnd 释放引用）。测传输子系统提交路径健康度；增长指向传输子系统初始化/资源异常 | `task_worker.cc` `ProcessDump()` 提交失败分支（L159-168）+ `ProcessLoad()` 提交失败分支（L221-230），两点合并计数 |

#### F. 资源水位（3 类，数据池由 GCThreadLoop 每 `gcIntervalMs` 一轮采样、flag 池由 CompletionPoller 调用点记账，均无热路径开销）

| 指标名 | 类型 | 功能（测什么 / 反映什么 / 怎么用） | 埋点位置 |
|---|---|---|---|
| `drampool_metadata_entry_count` | Gauge | 当前池内缓存条目（block）总数（1024 分片求和）。测元数据规模容量水位。增长斜率反映写入/驱逐平衡；配合 usage_ratio 判断是否临近驱逐压力区 | `drampool_server.cc` `GCThreadLoop()`（L552-561），`PerformEvict()` 调用后采样 |
| `drampool_buffer_pool_usage_ratio_<slot_size>` | Gauge（**动态注册**，每个 block size 一个） | 各 block 尺寸池**已用槽位占比**（used / slot count）。测分尺寸内存水位。接近 1 预示该尺寸池即将触发驱逐重试（DUMP prepare 抖动前兆），是容量规划的第一信号 | 同上，遍历 `g_config.poolBlockSizes` |
| `drampool_flag_pool_usage_ratio` | Gauge | flag 响应缓冲池**已用槽位占比**（used / slot count）。测响应回填缓冲水位。逼近 1 是 B2 链（flag 池 NoSpace → `response_buffer_retry`）的**前兆信号**；为 §4.6.3 "flag 池扩容"决策提供交叉验证（该动作此前无水位指标可查） | `completion_poller.cc` 调用点记账（§4.3）：`SubmitResponse()` Allocate 成功（L164）+1、`ReleaseResponseBuffer` Free 成功（L38）−1，`PollPendingCompletions()` 每轮尾部 `MetricsSet` |

#### G. Metadata 阶段耗时（8，本次新增）

> 阶段层级与归因方法见 §4.5。观测点位于 `metadata.cc` 内部（模块边界测量），`ScopedTimer` RAII 零逻辑侵入；时钟 µs 精度、以 ms（double）入直方图。

| 指标名 | 类型 | 功能（测什么 / 反映什么 / 怎么用） | 埋点位置 |
|---|---|---|---|
| `drampool_metadata_storebegin_duration_ms` | Histogram | 每次 `MetadataManager::StoreBegin`（每 DUMP entry 一次）的**总耗时** = 缓冲分配（含驱逐重试）+ 分片注册。测 DUMP 写入路径元数据开销总视角。P99 抬高时按 allocate / register 两阶段直方图归因（§4.5 决策树） | `metadata.cc` `MetadataManager::StoreBegin()` 入口 `ScopedTimer`（覆盖成功/失败/重试全程） |
| `drampool_metadata_allocate_duration_ms` | Histogram | StoreBegin 内**缓冲分配阶段**耗时：首次 Allocate，NoSpace 时触发周期驱逐重试、再 NoSpace 触发深度驱逐重试，直到成功或失败（含驱逐耗时，与 evict_sync 嵌套）。测"拿到一块缓冲"的真实成本。高且 evict_sync 占比高 → 内存压力；高而 evict 占比低 → 共享 BufferPool Allocate 本身慢 | `metadata.cc` `MetadataManager::StoreBegin()` 分配段独立作用域 `ScopedTimer`（L180-189 对应段） |
| `drampool_metadata_shard_register_duration_ms` | Histogram | StoreBegin 内**分片注册阶段**耗时（`ShardMetadata::StoreBegin`：分片写锁 + 重复检查 + 双驱逐策略 AddKey + map emplace，失败含回滚 Free）。测元数据注册成本。高说明分片写锁竞争或驱逐策略数据结构（LRU/TTL 链表）操作开销大 | `metadata.cc` `shards_[idx]->StoreBegin()` 调用处 `ScopedTimer`（L196） |
| `drampool_metadata_evict_sync_duration_ms` | Histogram | **业务线程内同步驱逐**（StoreBegin 分配 NoSpace 触发，含周期/深度两级）单次耗时。测驱逐对 DUMP 请求的**直接拖慢**（在 TaskWorker 线程内执行，阻塞当前请求后续 entry）。增长 = 内存压力的直接信号，与 usage_ratio 交叉验证 | `metadata.cc` `MetadataManager::StoreBegin()` 两处 `EvictOneShard` 调用处 `ScopedTimer`（L183、L187） |
| `drampool_metadata_storeend_duration_ms` | Histogram | 每次 `StoreEnd`（DUMP 传输 Completed 后 entry INITIALIZED→READY）耗时，CompletionPoller 线程执行。测 DUMP 终态结算成本。高说明分片读锁竞争（与 GC/同步驱逐线程互扰）——GC 每秒全分片扫描会持有读锁 | `metadata.cc` `ShardMetadata::StoreEnd()` 函数体 `ScopedTimer`（覆盖全部调用路径） |
| `drampool_metadata_loadbegin_duration_ms` | Histogram | 每次 `LoadBegin`（查 key + TryIncRef 加引用 + 双驱逐策略 AccessKey）耗时。测 LOAD 读路径元数据开销。高说明读锁竞争或策略 AccessKey（LRU move-to-front / TTL 更新）开销大 | `metadata.cc` `ShardMetadata::LoadBegin()` 函数体 `ScopedTimer` |
| `drampool_metadata_loadend_duration_ms` | Histogram | 每次 `LoadEnd`（TryDecRef 减引用）耗时。出现在 poller 终态结算、len 不匹配回滚、提交失败回滚三类路径。测引用释放成本；单次应近常数，异常升高指向锁竞争 | `metadata.cc` `ShardMetadata::LoadEnd()` 函数体 `ScopedTimer` |
| `drampool_metadata_evict_gc_duration_ms` | Histogram | **后台 GC 每轮**全分片驱逐扫描总耗时（`PerformEvict` 整轮 = 1024 shard 之和，每 `gcIntervalMs`（默认 1s）一轮）。测 GC 对系统的持续开销：均值 ÷ `gcIntervalMs` = GC 线程占空比；高说明 GC 频繁持有分片锁，干扰业务线程（StoreEnd/LoadBegin 变慢的常见外因） | `drampool_server.cc` `GCThreadLoop()` L554 `PerformEvict()` 调用处 `ScopedTimer` |

#### H. 队列与阻塞（9，本次增补）

> 判定依据：两条 SPSC 队列与响应缓冲是请求端到端时延的关键路径（S0），排队/阻塞直接等价于客户端可感知延迟。线程衔接模型与归因方法见 §4.6。

| 指标名 | 类型 | 功能（测什么 / 反映什么 / 怎么用） | 埋点位置 |
|---|---|---|---|
| `drampool_queue_request_enqueued_total` | Counter | 累计**成功进入** requestQueue 的请求数（TryPush 成功 +1）。测接收侧入口流量（口径为"已入队待处理"，与 A 组"已开始处理"天然相差排队数）。入队速率 − 出队速率 = 当前排队数（§4.6 恒等推导） | `drampool_server.cc` `RequestReceiveLoop()` TryPush 成功后（L517） |
| `drampool_queue_request_dequeued_total` | Counter | 累计 TaskWorker 从 requestQueue **取出**的请求数（TryPop 成功 +1）。测消费速率；与 `enqueued_total` 差值 = requestQueue 当前排队数（SPSC 无丢弃路径，恒等成立） | `task_worker.cc` `Run()` TryPop 成功后（L49） |
| `drampool_queue_request_full_total` | Counter | 累计 requestQueue **满、TryPush 失败**的事件数（重试循环内每次失败 +1，同一请求多次重试按次计）。测接收线程被阻塞的强度：阻塞时长 ≈ full 事件数 × `requestReceiverIdleWaitUs`（默认 100µs）。增长 = TaskWorker 处理能力跟不上到达速率，或下游 completionQueue 反压传导（§4.6 阻塞链 B1） | `drampool_server.cc` TryPush 失败分支（L518-523，WARN 处） |
| `drampool_queue_request_enqueue_wait_ms` | Histogram | 每请求一次观测：从"Receiver 准备入队"到"TryPush 成功"的**入队前等待时长**（含满重试 sleep；未排队时 ≈ 0，队列容量 65536 下正常长期为 0）。测客户端可感知的接收背压延迟。P99 抬高必然伴随 `full_total` 增长，用分位数估计单请求接收延迟 | `drampool_server.cc` 入队重试循环外 `ScopedTimer`（L516 前构造，成功 break 后析构观测） |
| `drampool_queue_completion_enqueued_total` | Counter | 累计**成功进入** completionQueue 的完成记录数（每请求一条，`Push` 自旋至成功恒 +1）。测完成流生产速率；与 `dequeued_total` 差值 = completionQueue 当前排队数 | `task_worker.cc` `SubmitCompletion()` Push 成功后（L326） |
| `drampool_queue_completion_dequeued_total` | Counter | 累计 CompletionPoller 从 completionQueue **取出**的完成记录数（FillPendingWindow 拉取成功 +1）。测完成流消费速率；与 `enqueued_total` 差值 = completionQueue 当前排队数 | `completion_poller.cc` `FillPendingWindow()` TryPop 成功后（L71-72） |
| `drampool_queue_completion_full_total` | Counter | 累计 completionQueue **满、SubmitCompletion 被迫自旋等待**的事件数（Push 前以 TryPush 探测，失败 +1 后退回 `Push` 自旋——探测不改变任何既有行为）。测 TaskWorker **停摆**的位置与强度：此时既不取新请求也不响应停止指令，requestQueue 随之堆积并传导为接收背压（§4.6 阻塞链 B3）。增长 = Poller 消费能力不足（pending 窗口满）或传输终态/flag 池重试慢 | `task_worker.cc` `SubmitCompletion()` TryPush 探测失败分支（L320-328 微改造，见 §10-12） |
| `drampool_queue_completion_inflight` | Gauge | CompletionPoller pending 窗口内**在途完成记录**数（等数据传输终态 + 等响应提交 + 等响应写回，每轮覆盖写）。测完成链路第二级缓冲占用。持续逼近 `pollerPendingDepth`（默认 64）= 拉取停摆，与 `completion_full_total` 增长互相印证；≈ 0 而 full 增长则提示窗口/队列配置失配 | `completion_poller.cc` `PollPendingCompletions()` 每轮尾部 `MetricsSet(pending_.size())` |
| `drampool_queue_response_buffer_retry_total` | Counter | 累计响应 flag 缓冲池 **NoSpace、SubmitResponse 留 pending 下轮重试**的事件数。测响应缓冲供给不足造成的阻塞（不阻塞 Poller 线程，但阻塞该请求的响应提交，response_rtt 抬高）。增长 = 响应突发超过 slot 供给或写回慢未及时释放槽位；与 `response_rtt_ms` P99 联动 | `completion_poller.cc` `SubmitResponse()` NoSpace 分支（L166-172，WARN 处） |

构成汇总：**15 个 Histogram**（原 5 + G 组 8 + lookup_scan 1 + H 组 enqueue_wait 1，全部 `_ms` 后缀）+ **19 个 Counter**（22 − 失败类合并净减 3，§3.1 第三类）+ **4 类 Gauge**（其中 1 类动态注册）= **38**。

---

## 4. 关键计算逻辑

### 4.1 计时基础设施（ScopedTimer RAII + µs 精度时钟）

`ScopedTimer` 在 `ProcessDump`/`ProcessLoad` 入口构造，仅**提交失败分支** `Disarm()`；空批次提前返回路径（无传输）视为 prepare 正常完成，正常析构观测。

**时钟精度**：`SteadyNowMs()`（`drampool_types.h` L48-53）为毫秒精度，而 metadata 单次操作（register/LoadBegin/StoreEnd 等）为微秒量级，毫秒精度下观测值会退化为 0/1 两值直方图。为此 `drampool_types.h` 新增：

```cpp
inline std::uint64_t SteadyNowUs()
{
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(now).count());
}
```

- `ScopedTimer` 内部以 `SteadyNowUs()` 计时，析构时观测值换算为 **ms（double，µs 分辨率）** 写入直方图——指标名统一 `_ms` 后缀，Prometheus 侧单位一致。
- 现有 `SteadyNowMs()` 的三处用法（`record.submit_ms`、`OperationTimedOut`）不受影响，保持毫秒语义。

### 4.2 数据传输时长（直测）与带宽（推导）

观测点（**统一出口，一处覆盖三条终态路径**——Completed、Failed、GetStatus 异常）：`completion_poller.cc` `SettleDataTransfer()` 末尾：

```cpp
if (record.submit_ms != 0) {
    const auto durationMs = SteadyNowMs() - record.submit_ms;
    const bool isDump = record.opcode == OpType::DUMP;
    MetricsObserve(isDump ? kDumpTransferDurationMs : kLoadTransferDurationMs,
                   static_cast<double>(durationMs));
    if (terminalStatus != transport::TransferStatus::Completed) {
        MetricsCount(kTransferFailures, 1);   // DUMP/LOAD 合并大类（§3.1 第三类）
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

**flagBufferPool 记账**（F 组 `drampool_flag_pool_usage_ratio`）：flag 池同样无用量查询接口，由 `DramPoolServer::flagBufferPool_` 直接持有（独立 BufferRegion，与数据池无交集），**仅 CompletionPoller 单线程访问**（`SubmitResponse()` Allocate 与匿名命名空间 `ReleaseResponseBuffer()` Free 各一处）。记账无需共享池改动——在 poller 内维护 `std::size_t flagSlotsUsed_`（单线程访问，普通整型即可，无需 atomic）：Allocate 成功 +1、Free 返回 OK −1，`PollPendingCompletions()` 每轮尾部 `MetricsSet(kFlagPoolUsageRatio, flagSlotsUsed_ / slotCount)`。slot count 取 `flagBufferSlotCount` 配置值。

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

### 4.5 Metadata 阶段耗时：层级关系与归因方法（G 组）

#### 4.5.1 阶段嵌套层级

G 组指标与既有业务级指标的嵌套包含关系（父级耗时 = 子级耗时之和 + 少量胶水代码）：

```
DUMP 请求
├─ dump_prepare_duration_ms（请求级，业务总视角）
│  └─ Σ metadata_storebegin_duration_ms（entry 级，总用时）
│     ├─ metadata_allocate_duration_ms（阶段①：缓冲分配）
│     │  └─ metadata_evict_sync_duration_ms（NoSpace 时嵌套，同步驱逐）
│     └─ metadata_shard_register_duration_ms（阶段②：分片注册）
├─ dump_transfer_duration_ms（异步数据传输，poller 结算）
│  └─ Σ metadata_storeend_duration_ms（entry 级，INITIALIZED→READY）
LOAD 请求
├─ load_prepare_duration_ms（请求级）
│  └─ Σ metadata_loadbegin_duration_ms（entry 级）
├─ load_transfer_duration_ms
│  └─ Σ metadata_loadend_duration_ms（entry 级）
LOOKUP 请求（无数据传输）
└─ lookup_scan_duration_ms（batch 级，= Σ Exist + 结果填充）
后台 GC（每 gcIntervalMs 一轮）
└─ metadata_evict_gc_duration_ms（轮级，= 1024 shard 驱逐之和）
```

守恒校验：`rate(storebegin_sum) ≈ rate(allocate_sum) + rate(register_sum)`；偏差即胶水代码（entry 构造、TransferItem 填充）开销，正常应远小于任一阶段。

#### 4.5.2 归因决策树

```promql
# ① 定位"慢在哪个阶段"：storebegin P99 高时对比两阶段分位数
histogram_quantile(0.99, rate(ucm:drampool_metadata_allocate_duration_ms_bucket[5m]))
histogram_quantile(0.99, rate(ucm:drampool_metadata_shard_register_duration_ms_bucket[5m]))

# ② allocate 高 → 判断是否内存压力（驱逐占比）
rate(ucm:drampool_metadata_evict_sync_duration_ms_sum[5m])
  / rate(ucm:drampool_metadata_allocate_duration_ms_sum[5m])
# 占比高 → 内存压力，交叉验证 ucm:drampool_buffer_pool_usage_ratio_* 逼近 1
# 占比低 → 共享 BufferPool Allocate 本身慢（锁/伙伴算法问题）

# ③ GC 干扰评估：GC 线程占空比 = 每轮耗时均值 / gcIntervalMs
rate(ucm:drampool_metadata_evict_gc_duration_ms_sum[5m])
  / rate(ucm:drampool_metadata_evict_gc_duration_ms_count[5m])
  / (ucm:drampool_gc_interval_ms)          # gcIntervalMs 可另行静态配置暴露
# 占空比高 → GC 频繁持有分片锁，解释 storeend/loadbegin 直方图同步抬高

# ④ LOOKUP 单 key 成本
rate(ucm:drampool_lookup_scan_duration_ms_sum[5m])
  / rate(ucm:drampool_lookup_scan_duration_ms_count[5m])
  / (rate(ucm:drampool_lookup_hit_entries_total[5m])
     + rate(ucm:drampool_lookup_miss_entries_total[5m]))
```

| 症状组合 | 结论 | 动作方向 |
|---|---|---|
| allocate P99 高，evict_sync 占比高 | 内存压力触发驱逐重试 | 扩容 / 调整驱逐比例 / 排查突发写入 |
| allocate P99 高，evict_sync 占比低 | 共享 BufferPool 分配慢 | 排查池锁与伙伴分配实现 |
| register P99 高，GC 占空比高 | GC 扫描持锁干扰写锁 | 调大 gcIntervalMs 或降驱逐比例 |
| register P99 高，GC 占空比低 | 驱逐策略数据结构开销（LRU/TTL 链表） | 深挖策略实现（恢复 scan/release 细分，§3.1） |
| storeend/loadbegin 抬高，与 evict_gc 同步 | 后台驱逐读锁与业务读锁互扰 | 同上 |
| lookup_scan P99 高，loadbegin 正常 | Exist 路径特有问题（TTL lease 刷新写状态） | 排查 TryMarkHit 与 lease 逻辑 |

### 4.6 队列深度推导与阻塞归因（H 组）

#### 4.6.1 线程衔接模型

dramPool 三条常驻线程通过两条 SPSC 队列单向衔接，加上 Poller 内部的 pending_ 窗口，构成完整流水线：

```text
RequestReceiver 线程              TaskWorker 线程               CompletionPoller 线程
(TCP 收包 + Unpack)               (业务处理)                     (响应回填)
      │                                 │                               │
      │ TryPush   ┌──────────────┐      │ TryPop                         │
      ├──────────▶│ requestQueue │──────┼──────────────┐                 │
      │   失败则    │  SPSC 65536  │      │              ▼                 │
      │  sleep 重试 └──────────────┘      │        ┌──────────┐            │
      │ (B1 链)                          │        │ pending_ │            │
      │                                 │ Push*   │  deque   │──TryPop──▶ │
      │                                 ├────────▶└──────────┘            │
      │                                 │ (B3 链)    completionQueue       │
      │                                 │            SPSC 65536            │
      │                                 │                    │ flag 池     │
      │                                 │                    │ NoSpace 留  │
      │                                 │                    ▼ pending 重试 │
      │                                 │              (缓冲链 B2)          │
```

三条阻塞链的形态各不相同：

| 链 | 阻塞点 | 阻塞形态 | 直接后果 | 观测信号 |
|---|---|---|---|---|
| B1 | Receiver 在 requestQueue 满 | **主动 sleep 阻塞**（`requestReceiverIdleWaitUs`=100µs 重试） | TCP 收包停滞，客户端请求延迟上升 | `request_full_total` 增长 + `request_enqueue_wait_ms` 右移 |
| B3 | TaskWorker 在 completionQueue 满 | **自旋停摆**（`Push` 无限期自旋 yield） | 既不取新请求也不响应 stop，requestQueue 随之堆积反压 B1 | `completion_full_total` 增长 → 随后 `request_full_total` 连锁增长 |
| B2 | Poller 在 flag 缓冲池 NoSpace | **被动延迟**（单请求留 pending_ 下轮重试，不阻塞线程） | 单个响应延迟上升，pending_ 积压 | `response_buffer_retry_total` 增长 + `completion_inflight` Gauge 抬高 |

*：§10-12 微改造——SubmitCompletion 在 Push 前增加 TryPush 探测以获取 full 信号，探测失败仍退回 Push 自旋，不改变任何既有行为。

#### 4.6.2 队列深度恒等推导（免 size() 计数器）

SPSC 队列无丢弃路径（Push 必成功、TryPop 只在成功时计 dequeued），因此任意时刻：

```text
当前排队数 = 累计入队数 − 累计出队数
rate(enqueued_total) − rate(dequeued_total) = 稳态排队数（速率差）
```

| 队列 | 排队数 PromQL | inflow | outflow |
|---|---|---|---|
| requestQueue | `rate(ucm:drampool_queue_request_enqueued_total[5m]) - rate(ucm:drampool_queue_request_dequeued_total[5m])` | Receiver TryPush 后 | TaskWorker TryPop 后 |
| completionQueue | `rate(ucm:drampool_queue_completion_enqueued_total[5m]) - rate(ucm:drampool_queue_completion_dequeued_total[5m])` | TaskWorker SubmitCompletion 后 | Poller FillPendingWindow TryPop 后 |

- 速率差 > 0 且持续增长 → 消费端（TaskWorker / Poller）成为瓶颈；
- 速率差 ≈ 0 → 流水线吞吐均衡，队列仅起缓冲作用；
- 排队数无需在队列内部加锁维护 size 计数器（SpscRingQueue 零改动），两个 Counter 的 rate 差在 Prometheus 侧天然完成推导。

Poller 内部 pending_ 窗口深度（`completion_inflight` Gauge）是唯一例外的直测值：它是 deque 之后、响应回填之前的"第三级排队"，直接 `MetricsSet(pending_.size())`，上限即 `pollerPendingDepth`（默认 64）。

#### 4.6.3 阻塞位置归因决策表

| 症状组合 | 结论（阻塞位置） | 动作方向 |
|---|---|---|
| request_full 增长，completion_full 不变，TaskWorker CPU 正常 | B1：入队速率超 TaskWorker 消费能力 | 排查 TaskWorker 处理耗时（storebegin P99），必要时加 TaskWorker 数量评估 |
| completion_full 增长，且随后 request_full 连锁增长 | B3：Poller 消费不动，completionQueue 堆积反压至 TaskWorker 停摆 | 排查 Poller 的 SubmitResponse 耗时与 flag 池用量（B2 联动） |
| response_buffer_retry 增长，completion_inflight 贴近 pollerPendingDepth | B2：flag 缓冲池不足，响应回填延迟 | 扩大 flag 池容量，交叉验证 `flag_pool_usage_ratio` 逼近 1 |
| request_enqueue_wait_ms P99 抬高但 request_full 偶发 | B1 前兆：排队抖动（队列接近满的瞬时重试） | 观察 TaskWorker 消费速率与请求到达速率的差值趋势 |
| completionQueue 排队数（rate 差）持续逼近 completionQueueDepth，completion_full 开始零星增长 | B3 前兆：Poller 消费变慢，TaskWorker 已在自旋边缘 | 立即排查 Poller；此信号出现在 request_full 连锁增长之前，是 B3 早期预警 |
| 全链路三组 full/retry 均为 0，但吞吐低 | 阻塞不在队列链路 | 转向 §4.2 数据传输 / §4.5 metadata 阶段分析 |

#### 4.6.4 与 G/F 组的联动

- **B3 反压链**：completion_full + `metadata_allocate_duration_ms` P99 同升 → allocate 慢拖住 TaskWorker 出队，requestQueue 堆积（B1 是果不是因）；
- **B1 存量链**：request_full + `lookup_hit_entries_total` 高 → 命中请求处理快但批量到达，属正常突发；若同时 lookup 低则请求处理本身慢；
- **B2 与 F 组**：response_buffer_retry 与 `flag_pool_usage_ratio` 同升逼近 1 → flag 响应缓冲供给不足（flag 池为独立 BufferRegion，与数据池无关，不存在同池争抢），优先扩大 `flagBufferSlotCount`；若同时数据池 `buffer_pool_usage_ratio_*` 亦逼近 1，则反映整体内存压力或两类池的配额配置失配。

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
inline constexpr char kLookupScanDuration[]  = "drampool_lookup_scan_duration_ms";
// 失败类（§3.1 第三类：保留 1 + 合并大类 3）
inline constexpr char kStoreBeginFailures[] = "drampool_dump_storebegin_failures_total";
inline constexpr char kSubmitFailures[]     = "drampool_submit_failures_total";
inline constexpr char kTransferFailures[]   = "drampool_transfer_failures_total";
inline constexpr char kResponseFailures[]   = "drampool_response_failures_total";
// G 组：metadata 阶段耗时（§3.2.G）
inline constexpr char kMetadataStoreBeginDuration[]    = "drampool_metadata_storebegin_duration_ms";
inline constexpr char kMetadataAllocateDuration[]      = "drampool_metadata_allocate_duration_ms";
inline constexpr char kMetadataShardRegisterDuration[] = "drampool_metadata_shard_register_duration_ms";
inline constexpr char kMetadataEvictSyncDuration[]     = "drampool_metadata_evict_sync_duration_ms";
inline constexpr char kMetadataStoreEndDuration[]      = "drampool_metadata_storeend_duration_ms";
inline constexpr char kMetadataLoadBeginDuration[]     = "drampool_metadata_loadbegin_duration_ms";
inline constexpr char kMetadataLoadEndDuration[]       = "drampool_metadata_loadend_duration_ms";
inline constexpr char kMetadataEvictGcDuration[]       = "drampool_metadata_evict_gc_duration_ms";
// H 组：队列与阻塞（§3.2.H）
inline constexpr char kQueueRequestEnqueued[]      = "drampool_queue_request_enqueued_total";
inline constexpr char kQueueRequestDequeued[]      = "drampool_queue_request_dequeued_total";
inline constexpr char kQueueRequestFull[]          = "drampool_queue_request_full_total";
inline constexpr char kQueueRequestEnqueueWait[]   = "drampool_queue_request_enqueue_wait_ms";
inline constexpr char kQueueCompletionEnqueued[]   = "drampool_queue_completion_enqueued_total";
inline constexpr char kQueueCompletionDequeued[]   = "drampool_queue_completion_dequeued_total";
inline constexpr char kQueueCompletionFull[]       = "drampool_queue_completion_full_total";
inline constexpr char kQueueCompletionInflight[]   = "drampool_queue_completion_inflight";
inline constexpr char kQueueResponseBufferRetry[]  = "drampool_queue_response_buffer_retry_total";
// F 组：资源水位（§3.2.F；数据池 usage ratio 按启动期 poolBlockSizes 动态注册，无静态常量）
inline constexpr char kFlagPoolUsageRatio[] = "drampool_flag_pool_usage_ratio";
// ... 完整清单见 §3.2

inline void MetricsCount(const char* name, double delta)   { UC::Metrics::UpdateStats(name, delta); }
inline void MetricsSet(const char* name, double value)     { UC::Metrics::UpdateStats(name, value); }
inline void MetricsObserve(const char* name, double value) { UC::Metrics::UpdateStats(name, value); }

// RAII 耗时打点：构造计时、析构写 Histogram，零侵入业务逻辑；
// 失败路径调用 Disarm() 取消本次观测（用于"仅成功路径上报"的 prepare duration）
class ScopedTimer {
public:
    explicit ScopedTimer(const char* histName)
        : name_(histName), start_(SteadyNowUs()), armed_(true) {}
    void Disarm() { armed_ = false; }
    ~ScopedTimer() {
        if (armed_) MetricsObserve(name_, static_cast<double>(SteadyNowUs() - start_) / 1000.0);
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

`drampool_config.h` `DramPoolConfig` 新增（15 个 Histogram 全部为 `_ms`，桶配置只需一组）：

```cpp
// Metrics snapshot (JSON Lines) + reflow pipeline.
bool metricsEnabled{true};
std::string metricsLogPath{"./drampool_metrics.log"};   // 与 UCM 侧回流器的 log_path 约定一致
std::uint32_t metricsSnapshotIntervalMs{10000};         // 快照周期
std::uint32_t metricsLogMaxBytes{64 * 1024 * 1024};     // 按大小轮转阈值
std::uint32_t metricsLogMaxFiles{5};                    // 轮转保留个数，超出删除最旧
std::uint32_t metricsHistogramVectorLen{10000};         // UC::Metrics::SetUp(maxVectorLen)
std::vector<double> metricsDurationBucketsMs{0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000};
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
  duration_buckets_ms: [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
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
    flushQueue_.TryPush(std::move(line))   // 有界（容量 64 行 ≈ 10.7 分钟快照缓冲）；满则丢行 + UC_WARN + droppedLines_++（旁路非阻塞）
```

- **Histogram 在 C++ 侧分桶**（不输出原始值向量）：桶边界取 §5.4 `metricsDurationBucketsMs`（15 个 histogram 全部为 `_ms` 后缀，共用同一组 22 桶；超出最大桶的值计入 `+Inf`）。原因：原始向量一行可达数百 KB，而桶计数后单行 ≈ 9KB，且回流器可直接映射 Prometheus histogram。
- 10s 窗口内的 `maxVectorLen` 截断只影响该窗口桶计数偏差，不跨周期累积（每轮 GetAllStatsAndClear 后向量清空）。
- 内存开销：每个（打点线程, histogram 指标）组合各占一条 vector，上界 ≈ 15 × (N_worker + N_poller + N_gc + N_receiver) 条 × `maxVectorLen` × 8B——默认 `maxVectorLen=10000`、4 worker 时 15 × 7 × 10000 × 8B ≈ **8.4 MB**，随 worker 数线性增长（实际多数指标仅在部分线程打点，低于该上界；容量规划按上界预留）。
- 快照队列溢出可观测：flushQueue_ 满时丢行仅 UC_WARN、无回溯 → Reporter 维护本地 atomic 计数 `droppedLines_`（每丢一行 +1），随**下一成功行**的 JSON 顶层字段 `dropped_lines` 输出（§6.2），日志侧完整性可对账；该字段非 UC::Metrics 指标、不回流 ucmmetrics。
- 单行大小预算：15 个 histogram ×（22 桶 + sum/count）+ 23 个 counter/gauge（19 Counter + 4 类 Gauge）≈ **9KB**，低于回流器 64KiB 读取窗口（§10-8）。

### 6.2 JSON Lines 行格式（一行 = 一个快照）

```json
{"ts": 1725868800.123, "interval_ms": 10000, "pid": 12345, "dropped_lines": 0, "host": "node-a", "metrics": {
  "drampool_dump_requests_total":   {"type": "counter", "value": 12345},
  "drampool_dump_bytes_total":      {"type": "counter", "value": 9876543210},
  "drampool_metadata_entry_count":  {"type": "gauge",   "value": 10240},
  "drampool_buffer_pool_usage_ratio_1048576": {"type": "gauge", "value": 0.62},
  "drampool_queue_request_enqueued_total": {"type": "counter", "value": 12344},
  "drampool_queue_request_dequeued_total": {"type": "counter", "value": 12341},
  "drampool_queue_completion_inflight":    {"type": "gauge",   "value": 3},
  "drampool_flag_pool_usage_ratio":        {"type": "gauge",   "value": 0.25},
  "drampool_dump_transfer_duration_ms": {
    "type": "histogram", "count": 12345, "sum": 45678.5,
    "buckets": {"1": 10200, "2": 12300, "5": 13100, "+Inf": 12345}
  }
}}
```

- 单行无内嵌换行（compact 序列化），文件尾始终以 `\n` 结束，保证回流器"完整行 = 完整快照"。
- 字段名直出，彻底摆脱位置编码与白名单耦合。
- G 组 µs 级观测落在亚毫秒桶（`0.001`–`0.5`），桶键为原样字符串；回流器按 `le` 标签透传，无需感知桶集差异。

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
    enter_retry_backoff()   # 抢锁失败者进入低频重试（60s 间隔）继续尝试 flock，不永久退出
```

- 锁文件名由 `endpoint + log_path` 哈希生成：同一（DramPool 实例, 日志文件）组合在多个推理实例场景下只有一个 Leader 独占回流职责，避免重复计数。
- 进程终止自动释放 flock，当前推理实例停止回流，不影响业务；抢锁失败者以 60s 间隔低频重试而非静默永久退出——原 Leader 推理进程退出后失败者自动补位，避免回流长期断流（重试竞态无风险：flock 独占性保证任何时刻仍只有一个 Leader；补位后 State 缺失 → 按新 baseline 模式重新校准，§7.3）。

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
def delta_or_reset(current, previous, current_pid, previous_pid):
    # Counter/Histogram（sum、count、每个桶各自适用）
    if current_pid != previous_pid:
        return None   # pid 变化 = drampool 重启 → 直接判 reset，以 current 作为新 baseline
    if current >= previous:
        return current - previous
    return None   # 同 pid 下 current < previous：兜底判 reset（累计值异常回退）
```

- **Gauge 直接获取最新值**，无增量语义。
- **重置判定优先比对 JSON 行顶层 `pid`**（State 中同时持久化 pid）：pid 与上轮不同 → 直接判定 drampool 重启，全量指标以新 baseline 处理——比纯值比较更确定（值比较仅作同 pid 下的兜底，覆盖异常回退场景）。
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
| ScopedTimerUsResolution | µs 精度验证：极短作用域观测值 < 1ms 且 > 0（毫秒精度时钟会退化为 0/1，G 组失效） |
| MetadataDurationConservation | 单线程执行一次 StoreBegin：storebegin 观测值 ≥ allocate + register 之和（§4.5.1 守恒上界） |
| NestedEvictObservation | 构造 NoSpace 触发同步驱逐：allocate 观测值 ≥ evict_sync 观测值（嵌套包含） |
| LookupScanBatchGranularity | batch_size = N 的 LOOKUP：scan 直方图仅观测 1 次（batch 粒度，非 entry 粒度，规避 C3 溢出） |
| ReporterCumulativeSnapshot | 模拟两轮 UpdateStats + 两次快照，JSON 中 counter 为单调累计值、gauge 为最新值 |
| ReporterHistogramBuckets | 构造已知值集（含超出最大桶）→ duration 桶计数含 `+Inf`，sum/count 正确 |
| JsonLineFormat | 输出为单行合法 JSON、`type` 字段齐全、无内嵌换行 |
| FlushRotation | 写满 `metricsLogMaxBytes` → 轮转产生 `.log.1`；超出 `maxFiles` → 最旧被删 |
| BufferManagerUsedSlots | Allocate/Free 对账：分配 N 后 ratio = N/capacity，全部释放归零 |
| FlagPoolUsageRatioAccounting | flag 池单线程记账：Allocate 成功 +1 / Free 成功 −1，ratio = used/flagBufferSlotCount；NoSpace 重试路径（未分配）不影响计数 |
| QueueIdentityRateDiff | 并发压测下 `rate(enqueued) − rate(dequeued)` 与两队列实测排队数一致（SPSC 无丢弃，恒等成立，§4.6.2） |
| QueueFullCounting | 构造满队列：request/completion `full_total` 按次累计；completion 探测不改变 Push 语义（§4.6 B3 停摆可观测） |
| CompletionInflightGauge | pending 窗口写入/结算时 inflight = `pending_.size()`，覆盖写无残留 |
| SubmitCompletionProbeKeepsSemantics | TryPush 探测失败后退回 `Push`：完成记录数量、顺序、停止指令响应与改造前一致（§10-12） |

### 8.2 Python 单测（回流器）

| 用例 | 验证点 |
|---|---|
| LeaderElection | flock 独占成功/第二次抢占失败进入 60s 低频重试；进程退出释放锁，原 Leader 退出后失败者自动补位 |
| TailReader | 正常行、末尾半行、空文件、文件缺失、超窗口首行截断 |
| DeltaCalculation | 正常增量；pid 变化直接判重启（新 baseline）；同 pid 下 current < previous 兜底触发 reset（丢弃 + 新 baseline）；gauge 直取 |
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

- Grafana/PromQL 冒烟：`rate(ucm:drampool_dump_bytes_total[1m])`、§4.2 带宽推导式、§4.4 命中率推导式、`histogram_quantile(0.99, rate(ucm:drampool_dump_transfer_duration_ms_bucket[5m]))`、`ucm:drampool_buffer_pool_usage_ratio_*`、队列深度恒等式 `rate(ucm:drampool_queue_request_enqueued_total[1m]) − rate(ucm:drampool_queue_request_dequeued_total[1m])`（§4.6.2）、阻塞信号 `rate(ucm:drampool_queue_request_full_total[5m])`。
- 杀掉 drampool → 回流器检测 reset，不产生负增量；重启 drampool → 恢复回流。
- 双推理实例共享同一 log_path → 仅一个 Leader 回流，指标无重复计数。

---

## 9. 实施阶段划分

| 阶段 | 内容 | 涉及文件 |
|---|---|---|
| P1 基础设施 | CMake 链接 + `drampool_metrics.{h,cc}`（注册器/辅助/ScopedTimer/常量）+ 配置项与 YAML + §4.3 BufferManager 记账 | `dram/CMakeLists.txt`、`drampool_metrics.h/.cc`（新）、`drampool_config.h`、`drampool_yaml_config.cc`、`examples/drampool.yaml`、`buffer_manager.h` |
| P2 数据面埋点 | §3.2.A-H 全部采样点落位（业务级 A-F + metadata 阶段 G 组 + 队列与阻塞 H 组 + flag 池记账，§4.3） | `task_worker.cc`（A/B/C/D 组 + H 组 dequeue/completion 埋点）、`completion_poller.cc`（E 组 + H 组 inflight/retry 埋点 + flag 池记账）、`metadata.cc`（G 组 RAII 埋点，业务逻辑零改动）、`drampool_server.cc`（GCThreadLoop 采样点 + requestQueue H 组埋点，`RequestReceiveLoop` L516-523） |
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
| 4 | Histogram 满后丢弃新样本（C3） | 业务级指标（A-F、H 组）观测点收敛在请求 / batch 粒度（规避溢出）；**G 组为 entry / 调用级观测**（独立归因价值，§3.2.G），高吞吐下 10s 窗口内可能触发 `maxVectorLen=10000` 截断：溢出样本被静默丢弃（无溢出回调、不可直接观测），P99/均值仅反映窗口前段样本，偏差不跨周期累积。缓解：以 G 组为主要观测对象时调大 `metricsHistogramVectorLen` 或缩短 `snapshot_interval_ms`（§5.4）；截断行为由 §8.1 HistogramOverflow 用例固化验证 |
| 5 | 数据结构零扩展 | `RequestTask`/`CompletionRecord` 原样保留（queue_wait、bandwidth 裁剪的连锁收益），SPSC 队列布局零影响 |
| 6 | storebegin_failures 语义完整性 | DuplicateKey 是正常业务结果（循环内 continue，不进失败分支），裁掉 duplicate_keys 不改变该指标语义；len_mismatch entry 天然计入 load_miss_entries |
| 7 | 共享 BufferPool 无用量查询接口 | 数据池用量在 drampool 层 BufferManager 记账（§4.3）；flag 池在唯一使用方 CompletionPoller 的调用点记账（§4.3，单线程无需 atomic），输出 `drampool_flag_pool_usage_ratio`（§3.1 第四类） |
| 8 | 单行大小 vs 64KiB 读取窗口 | C++ 侧预分桶，15 个 histogram（含亚毫秒桶段）单行 ≈ 9KB（§6.1 预算）；窗口大小可配置兜底 |
| 9 | 双侧"累计"语义分工 | C++ Reporter 维护进程级累计（每行 JSON 为终点状态）；Python 回流器负责累计→增量 + reset 检测（pid 比对优先，同 pid 下 current < previous 兜底，§7.3）；State 保证跨回流器重启连续性 |
| 10 | 超时场景可观测性 | timeouts 指标裁剪后，超时仅 WARN 日志；恢复/失败结果由 duration/failures 指标与 rtt 分位数覆盖 |
| 11 | `drampool_core` 源文件列表非 GLOB | 新 .cc 漏加直接链接失败（构建期暴露） |
| 12 | 范围隔离 | DramStore 零改动；HealthServer 零改动；共享组件零改动（含 SpscRingQueue）；数据结构零扩展；`metadata.cc` 仅新增 G 组 RAII 埋点（ScopedTimer），业务逻辑零改动。**唯一微改造**：`task_worker.cc` `SubmitCompletion` 在 `Push` 前加 TryPush 探测获取 completion_full 信号（失败 +1 后仍退回 `Push` 自旋，行为语义不变，§3.2.H / §4.6.1） |
| 13 | 跨进程时钟不一致 | `ts` 仅用于排序/诊断；增量计算基于指标值本身，不依赖跨进程时钟对齐 |
| 14 | 多推理实例共享同一日志 | Leader 选举保证单回流者；不同 `endpoint + log_path` 组合各自独立 Leader |
| 15 | 日志轮转与尾读竞态 | 回流器始终取"最新完整行"，轮转 rename 只影响旧文件；文件被删/缺失 → 跳过本轮 |
| 16 | 旁路非阻塞红线 | 打点/快照/刷盘/回流全链路失败仅 WARN，主流程（DUMP/LOAD/LOOKUP、推理）永不阻塞 |
| 17 | `metadata_delete_duration_ms` 裁剪（本次增补审查） | Delete 仅出现于失败清理与驱逐释放两类非正常路径：前者低频且日志完备，后者耗时已含于 evict 直方图（不满足 S1/S2）。恢复：Delete 调用处加 ScopedTimer |
| 18 | 驱逐 scan/release 二次拆分裁剪（本次增补审查） | 来源拆分（evict_sync/evict_gc）优先满足归因需求；sync 驱逐 P99 异常需细挖时，在 `EvictOneShard` 内部恢复 scan/release 两个观测点 |
| 19 | `Exist` 单次直方图裁剪（本次增补审查） | LOOKUP 批量大，entry 级观测触发 C3 溢出；由 `lookup_scan_duration_ms` 按 batch 粒度覆盖，单 key 成本经 §4.5.2 ④ 推导 |
| 20 | 失败类指标合并（本次增补审查） | 6 个低频传输/提交/响应失败 Counter 按**管线阶段**合并为 3 个大类（submit / transfer / response_failures_total，§3.1 第三类）；保留 4 个有独立必要性的失败指标（`storebegin_failures` 内存压力归因 + 三条阻塞链 full/retry 信号）。恢复：按 §3.2.B/C/E 标注的埋点位置重新拆分 opcode 维度即可 |
| 21 | flushQueue 溢出丢行可观测（本次审查增补） | 快照队列有界，容量定为 64 行（≈ 10.7 分钟快照缓冲，I3 定值）；满则丢行仅 UC_WARN、无回溯 → Reporter 本地 atomic 计数，随下一成功行顶层 `dropped_lines` 字段输出（§6.1/§6.2），日志侧完整性可对账；该字段非 UC::Metrics 指标，不回流 ucmmetrics |
| 22 | drampool 重启判定（本次审查增补） | 回流器 reset 判定优先比对 JSON 行顶层 `pid`（State 同时持久化 pid）：pid 变化 → 直接判重启、全量重置 baseline；值比较 current < previous 仅作同 pid 兜底，消除累计值异常回退的误判空间（§7.3） |
| 23 | Leader 生命周期（本次审查增补） | flock 抢锁失败者由"静默永久退出"改为 60s 间隔低频重试：原 Leader 推理进程退出释放锁后自动补位，避免回流长期断流；flock 独占性保证补位竞态下仍单 Leader（§7.1） |
| 24 | 专项审查修正留痕 | 本方案经专项审查后合入修正：response_rtt 口径修正为"传输提交成功（ExecuteAsync 后）→ 写回完成"，flag 池排队在窗口外（§3.2.E）；ScopedTimer 升级 µs 精度时钟（§4.1/§5.1）；默认桶集补亚毫秒段对齐 22 桶（§5.4）；内存预算公式化并修正量级（§6.1）；§9-P2 补 requestQueue 埋点清单。经核对确认无需改动：flag 池单漏斗记账闭环、失败类无双计（`submit_ms` 守卫互斥 submit/transfer failures）、H 组 SPSC 恒等推导、B1/B2/B3 观测点选位、Gauge carry-forward、日志轮转与尾读竞态处理、38 指标构成数内部一致 |
