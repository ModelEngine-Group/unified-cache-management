# DramPool Metrics 埋点与跨进程回流设计方案

| 项 | 说明 |
|---|---|
| 范围 | DramPool 服务端守护进程（`ucm/store/dram/cc/drampool/`）**业务逻辑**埋点 + 快照日志输出；UCM/Scheduler 侧新增回流器组件。DramStore 客户端侧零改动，HealthServer 零改动 |
| 核心诉求 | 解决 DramPool 独立进程与 vLLM/UCM 跨进程指标统一暴露，对接现有 Prometheus 监控链路 |
| 设计原则 | **业务聚焦 + 批次观测**：以**批次（请求）粒度**为主线观测 DUMP/LOAD/LOOKUP 服务质量——每批次的**首条 entry 各阶段时长与总时长**、**整批总耗时**、**当前批次失败/命中实况**，辅以累计 Counter、队列阻塞信号与资源水位。接入层协议错误、内部调度诊断等非必要指标不涉及（相应路径维持 UC_WARN/UC_ERROR 日志） |
| 共享代码约束 | 不修改 `ucm/shared/pool`、`ucm/shared/infra/template` 等共享组件；仅新增对 `ucm/shared/metrics` 的链接依赖 |
| 指标规模 | **47 个**（19 Histogram + 20 Counter + 8 类 Gauge，其中 1 类动态注册） |
| 数据结构影响 | `CompletionRecord` **新增 `begin_us` 1 个字段**（批次出队时刻，µs；TaskWorker 在记录构造处写入，Poller 读取；上报后置 0 复用作"批次统计已上报"哨兵，§4.7.2）；`RequestTask` 原样 |
| 时钟精度 | `SteadyNowUs()`（`drampool_types.h`）：metadata 单次操作为微秒级，毫秒精度时钟无法分辨；G 组计时内部以 µs 计、以 ms（double）入直方图 |

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
| **UCM thread-local + 双 buffer（采用）** | ✅ | 线程完全隔离，读写通过原子索引切换，无锁竞争，天然适配高并发打点；复用现有 `UC::Metrics` 库；thread_local 语义恰好承载"批次首条门控"状态（§4.7） |

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
| C3 | **Histogram 存原始值向量**，上限 `maxVectorLen`，**满后丢弃后续观测**（保留先到的样本） | 观测点控制在**批次粒度**（每批次 1 次观测：首条 entry 各阶段 + 批次总耗时）与请求响应粒度；G 组 entry 级保留观测（storeend/loadend）亦为每 entry 一次；向量长度可配置 |
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
| **S0** | **业务直接相关** | 直接反映 DUMP/LOAD/LOOKUP 操作的服务质量（请求量、成功率、时延、命中率）、**批次/首条粒度耗时**或资源水位（池用量、元数据规模）。纯系统性/接入层/内部调度/低频异常分类指标**此阶段不涉及** |
| S1 | 不可推导 | 无法由其他指标经 PromQL 运算（rate 比值、差值）等价获得 |
| S2 | 可行动 | 指标异常时存在明确的运维/开发动作 |
| S3 | 业务失败分类完整 | 缺失会导致业务失败被误判（如"重复写"与"真失败"） |
| S4 | 开销合理 | 热路径 Histogram 数量从严控制 |

**metadata 阶段耗时（G 组）的补充判定说明**：

- 阶段拆分只保留**两个有独立归因价值的切面**：`allocate`（缓冲分配，含驱逐重试——内存压力信号）与 `shard register`（分片锁 + 驱逐策略——锁竞争信号）；驱逐再按**触发来源**拆为 `evict_sync`（业务线程内，直接拖慢 DUMP）与 `evict_gc`（后台线程，抢锁干扰）。其余单阶段原子操作（StoreEnd/LoadBegin/LoadEnd）无内部分阶段，仅观测总耗时。
- 仍被裁掉的项（留痕，恢复方式见 §10）：`metadata_delete_duration_ms`（非正常路径，日志完备）、驱逐 scan/release 二次拆分（来源拆分优先）、`Exist` 单次直方图（由 scan 批次粒度 + 首条直测覆盖）。

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
| 内部调度诊断 | `request_queue_wait_ms`、`request_queue_full_events_total` | 线程模型内部细节；排队影响最终由 H 组队列指标与端到端耗时体现 |
| 低频异常分类 | `dump_duplicate_keys_total`、`load_len_mismatch_total` | 有日志可查；重复写不改变失败语义；len 不匹配天然计入 load_miss_entries |
| 传输超时诊断 | `data_transfer_timeouts_total`、`response_transfer_timeouts_total` | 超时仅诊断性日志；恢复/失败结果由 duration/failures 指标覆盖 |
| 内部池背压 | `flag_buffer_pool_full_total` | 影响由 `response_buffer_retry_total`（B2 链唯一信号）与 `response_rtt_ms` 覆盖 |
| 容量中间信号 | `evict_triggered_periodic/deep_total`、`evicted_entries_periodic/deep_total`、`buffer_alloc_failures_total` | 水位由 usage_ratio/metadata_entry_count 预警；分配失败业务影响由 `dump_nospace_failures_total` 精确归因 |
| GC 诊断 | `eviction_duration_ms`、`gc_cycles_total` | GC 线程低频内部任务，非业务信号 |

**第三类：失败类审查（按必要性，本次重设计更新）**

> 审查标准：失败类指标必须满足两条之一才单独设立——(a) **明确归因价值**：失败直接指向可行动根因；(b) **高发生概率**：真实负载下常见、需要告警基线。不满足者合并为统一大类——按**管线阶段**归因保留定位能力。

| 处置 | 指标 | 理由 |
|---|---|---|
| **删除（本次重设计）** | ~~`drampool_dump_storebegin_failures_total`~~ | 其"缓冲分配失败"主因由 `dump_nospace_failures_total` **直测精确归因**（原指标混入注册失败、且分配失败 ≠ 都是 NoSpace）；批次失败实况由 `dump_batch_failed_entries/ratio` 以用户明确要求的"当前批次"口径覆盖 |
| **新增（本次重设计）** | `drampool_dump_nospace_failures_total` | NoSpace **直测点**：`MetadataManager::StoreBegin` 两次驱逐重试后仍 `Status::NoSpace()` 才计数（metadata.cc L186-193 显式判定点）——内存压力的精确归因，排除注册失败误报；高概率（内存压力场景必然出现） |
| **保留** | `queue_request_full_total` / `queue_completion_full_total` / `queue_response_buffer_retry_total` | 三条阻塞链（§4.6）的**唯一量化信号**；B3 自旋停摆无日志兜底，无此指标则停摆不可观测 |
| **合并** | `dump_submit_failures` + `load_submit_failures` → **`drampool_submit_failures_total`** | 低频（传输子系统故障才出现）；合并后仍指向"传输提交失败"单一根因 |
| **合并** | `dump_transfer_failures` + `load_transfer_failures` → **`drampool_transfer_failures_total`** | 低频（对端网络异常时出现）；排查动作统一为查连接/对端状态 |
| **合并** | `response_transfer_failures` + `response_submit_failures` → **`drampool_response_failures_total`** | 低频；合并后锁定"响应返回链路"故障域，结合 WARN 日志定位环节 |

**第四类：buffer 池必要性审查（新增 1 / 不设计留痕 4 类）**

> 池盘点：DramPool 共两类 buffer 池，均基于共享 `UC::BufferPool`（无用量查询接口，不可改）。**数据池** N 个按 block size 分池，由 drampool 层 `BufferManager` 持有，唯一出入口 `Allocate/Free`，跨 TaskWorker/GC 多线程并发；**flag 池** 1 个（`DramPoolServer::flagBufferPool_` 直接持有，独立 BufferRegion），仅 CompletionPoller 单线程访问。

| 处置 | 指标 | 理由 |
|---|---|---|
| **新增** | `drampool_flag_pool_usage_ratio`（Gauge，F 组） | (a) B2 链（flag 池 NoSpace → `response_buffer_retry`）的**前兆信号**；(b) §4.6.3 "flag 池扩容"动作的交叉验证；(c) **高概率**——响应突发场景常见 |
| 不设计 | 数据池 alloc/free 计数（Counter ×2） | rate 差可推导（违反 S1），水位已由 `buffer_pool_usage_ratio_<slot_size>` 直测覆盖 |
| 不设计 | 数据池 NoSpace 驱逐触发计数 | 业务影响由 `dump_nospace_failures_total` 精确归因覆盖 |
| 不设计 | size 未注册 / Free 失败计数 | bug 级低频路径，UC_ERROR 日志兜底 |
| 不设计 | 池字节绝对值 Gauge | ratio × capacity 可推导（违反 S1） |

**第五类：批次粒度观测审查（本次重设计核心）**

> 用户观测口径：每批次取**首条 entry** 的各阶段时长与总时长、**整批总耗时**、**当前批次**失败/命中实况。首条门控使 G 组样本量从 entry 数降为批次数（C3 溢出进一步缓解）；批次失败/命中采用 Gauge 覆盖写（与 Reporter 快照天然配合——每行 JSON 反映最近一批实况）。

| 处置 | 事项 | 理由 |
|---|---|---|
| **改造** | G 组 storebegin / allocate / register / evict_sync / loadbegin 5 个直方图 → **仅批次首条观测** | 观测点位置不变（metadata.cc 内部），仅观测条件化（§4.7.1 首条门控）；恢复 entry 级 = 去掉门控条件即可（§10 留痕） |
| **保留 entry 级** | storeend / loadend（G 组） | 观测线程为 CompletionPoller（跨批次逐 entry 结算），无"批次首条"概念；本身是"每 entry 结算成本"观测 |
| **保留轮级** | evict_gc | GC 轮粒度天然与批次无关 |
| **新增** | 批次失败 Gauge ×2（DUMP）+ 批次命中 Gauge ×2（LOOKUP） | 用户明确"当前批次/一批数据"口径；统计点 = record.results 定稿处（四条失败路径唯一汇合点，无重无漏） |
| **新增** | 批次总耗时 Hist ×3 | 用户明确"整批数据总耗时"；分位数**不可**由 prepare+transfer+rtt 三段速率均值推导（仅稳态近似）——直测需跨线程传递批次入口时刻 → `CompletionRecord` +1 字段（机制与既有 `submit_ms` 相同） |
| **新增** | `lookup_first_exist_duration_ms` | 三阶段首条观测的平行完整性；LOOKUP 无 metadata 拆分阶段，首条 Exist 即首条总时长 |

### 3.2 指标全集（47 个）

> 功能描述统一为三段式：**测什么**（观测口径）→ **反映什么**（异常时指向的问题）→ **怎么用**（典型分析动作）。
>
> 观测粒度标注：**[首条]** = 仅批次首条 entry 观测（样本量 = 批次数）；**[批次]** = 每批次一次；**[entry]** = 每 entry 一次；**[轮]** = GC 每轮一次；**[覆盖]** = Gauge 最新值覆盖写。

#### A. 请求量（3）

| 指标名 | 类型 | 功能（测什么 / 反映什么 / 怎么用） | 埋点位置 |
|---|---|---|---|
| `drampool_dump_requests_total` | Counter | 累计接收并处理的 DUMP 请求数（每请求 +1，与 batch 内 entry 数无关）。反映写入负载规模；与 `dump_bytes_total` 配合可得平均批量，与 load/lookup 请求量对比反映负载构成 | `task_worker.cc` `ProcessOneRequest()` DUMP 分支（L75） |
| `drampool_load_requests_total` | Counter | 累计接收并处理的 LOAD 请求数。反映读取负载规模；读取侧所有 rate 类指标（miss、transfer 时长）的分母基准 | 同上 LOAD 分支（L80） |
| `drampool_lookup_requests_total` | Counter | 累计接收并处理的 LOOKUP 请求数。反映查询负载规模；也是 `lookup_scan_duration_ms` 均值换算"单请求查询成本"的分母 | 同上 LOOKUP 分支（L85） |

#### B. DUMP 业务（5）

| 指标名 | 类型 | 功能（测什么 / 反映什么 / 怎么用） | 埋点位置 |
|---|---|---|---|
| `drampool_dump_bytes_total` | Counter | 累计**成功预留缓冲**的 DUMP entry 字节总量（循环内 Σ `entry.len`；DuplicateKey 与分配失败的 entry 不计）。测 DUMP 吞吐规模（口径为"预留成功"，传输失败回滚的字节已计入本指标）。用于吞吐趋势、带宽推导（§4.2）与容量规划 | `task_worker.cc` `ProcessDump()` entries 循环内 StoreBegin 成功处累加（L128-146） |
| `drampool_dump_nospace_failures_total` | Counter | 累计 StoreBegin 中**缓冲分配最终失败且原因为 NoSpace** 的 entry 数——两次驱逐重试后仍 `Status::NoSpace()` 才计数，注册失败不计。**内存压力的精确归因**：持续增长 = 数据池容量不足或驱逐策略不足以释放空间；与 `buffer_pool_usage_ratio_*` 交叉验证。行动入口：扩容 / 调整驱逐比例 | `metadata.cc` `MetadataManager::StoreBegin()` 第二次深度驱逐重试后、最终 `return Error` 前（L193 附近，`st == Status::NoSpace()` 判定点） |
| `drampool_dump_batch_failed_entries` | Gauge **[覆盖]** | **当前批次**（最近完成结算的 DUMP 请求）中结果为 Failed 的 entry 数（含 StoreBegin 失败短路标记与传输失败结算两类）。反映最近一批写入失败实况；与 `dump_batch_failure_ratio` 联动。失败持续 >0 时结合 `nospace_failures`（元数据侧）与 `transfer_failures`（传输侧）区分根因 | `completion_poller.cc` `SubmitResponse()` Pack 前（`record.results` 定稿处——StoreBegin 短路 / 提交失败 / 传输失败结算 / StoreEnd 失败四条路径的唯一汇合点），条件 `opcode == Dump` |
| `drampool_dump_batch_failure_ratio` | Gauge **[覆盖]** | 同一统计点的 `failed_entries / batch_size`（0~1）。测最近一批 DUMP 失败强度。告警阈值可直接作用于本指标；持续 >0 = 写入链路异常 | 同上，同一处计算后两次 `MetricsSet` |
| `drampool_dump_prepare_duration_ms` | Histogram **[批次]** | 每请求一次观测：DUMP 从开始处理到数据传输提交完成的**本地准备耗时**（逐 entry StoreBegin + 缓冲分配 + 可能的驱逐重试 + 提交），**不含**数据传输本身。测写入路径的服务端 CPU 侧开销。P99 高说明准备阶段拖慢写入，用 G 组指标按阶段归因（§4.5） | 入口 L95 构造 `ScopedTimer`，L159-168 失败分支 Disarm（§4.1） |

> 注：重复 key entry 在循环内 `continue`（L129-132），不进入失败计数口径；`nospace_failures` 在 metadata 层直测、天然不受 DuplicateKey 影响。批次失败 Gauge 统计 `record.results` 定稿值——失败短路标记（`mark_remaining_failed`）的剩余 entry 一并计入，反映批次最终实况。

#### C. LOAD 业务（4）

| 指标名 | 类型 | 功能（测什么 / 反映什么 / 怎么用） | 埋点位置 |
|---|---|---|---|
| `drampool_load_bytes_total` | Counter | 累计**成功加引用**的 LOAD entry 字节总量（LoadBegin 成功且长度合法的 Σ `entry.len`）。测 LOAD 吞吐规模（口径为"加引用成功"，未命中/长度不匹配的 entry 不计）。用于读取吞吐趋势与读取带宽推导 | `task_worker.cc` `ProcessLoad()` entries 循环（L184-227） |
| `drampool_load_miss_entries_total` | Counter | 累计 LOAD 中**未取到数据**的 entry 数——即登记本 miss：**LoadBegin 失败**（key 不存在或状态非 READY）与**请求长度大于存储长度**两类之和，对客户端都是"没取到"。测读取未命中规模。突增说明客户端读取了已被驱逐/过期/尚未写完的数据；命中率 = 1 − miss/(miss + 成功 entry) | 同上 L187-192 |
| `drampool_load_initialized_entries_total` | Counter | 累计 LoadBegin 时目标 entry 处于 **INITIALIZED**（DUMP 已分配缓冲、传输未完成 / 未 StoreEnd）而被拒的 entry 数——miss 的**子类归因**。反映"写读并发竞态窗口"：读取侧撞上写入在途（数据尚未可见）。增长说明客户端在 DUMP 完成前发起读取（时序问题）而非数据丢失 | `metadata.cc` `ShardMetadata::LoadBegin()` 内 `TryIncRef` 失败分支 + `existingEntry->status == INITIALIZED` 判定（L88-99 内，existingEntry 可达处；NotFound 路径不计） |
| `drampool_load_prepare_duration_ms` | Histogram **[批次]** | 每请求一次观测：LOAD 从开始处理到传输提交完成的**本地准备耗时**（逐 entry LoadBegin + 长度校验 + 提交），**不含**传输本身。测读取路径服务端 CPU 侧开销；与 G 组 `loadbegin_duration_ms` 联动归因 | 入口 L184 构造，L221-230 失败分支 Disarm |

#### D. LOOKUP 业务（6）

| 指标名 | 类型 | 功能（测什么 / 反映什么 / 怎么用） | 埋点位置 |
|---|---|---|---|
| `drampool_lookup_hit_entries_total` | Counter | 累计 LOOKUP 中**存在且 READY**（命中并刷新 TTL lease）的 entry 数。测累计查询命中规模；累计命中率的分子（§4.4 口径一）；与 miss 速率对比反映数据存留健康度 | `task_worker.cc` `ProcessLookup()` 扫描循环（L275-279，循环内累计、循环外一次上报） |
| `drampool_lookup_miss_entries_total` | Counter | 累计 LOOKUP 中**不存在或非 READY** 的 entry 数（`batch_size − hits`）。测累计未命中规模；突增 = 查询了被驱逐 / 过期的前缀，通常先于 LOAD miss 出现（驱逐过快的预警信号） | 同上（`batch_size − hits` 循环外一次上报） |
| `drampool_lookup_batch_hits` | Gauge **[覆盖]** | **当前批次**（最近完成响应提交的 LOOKUP 请求）中命中的 entry 数（`record.results` 中 `Exists` 计数）。反映最近一批查询命中实况；突降时结合 `miss_entries_total` 速率区分"负载前缀变化"与"数据异常丢失" | `completion_poller.cc` `SubmitResponse()` 入口批次统计点（`record.results` 定稿处、`begin_us` 哨兵一次性上报，§4.7.2），条件 `opcode == Lookup` |
| `drampool_lookup_batch_hit_ratio` | Gauge **[覆盖]** | 同一统计点的 `hits / batch_size`（0~1；batch_size 为 0 时置 0）。测最近一批命中率；告警阈值直接作用于本指标；持续低于累计命中率 = 最近负载命中恶化（容量不足 / 驱逐过快的前兆） | 同上，同一处计算后两次 `MetricsSet` |
| `drampool_lookup_first_exist_duration_ms` | Histogram **[首条]** | 每批次扫描循环**首个 entry** 的 `Exist` 单次耗时（key 定位 + 分片读锁 + `TryMarkHit` lease 刷新）。单 key 查询成本的**直测**（替代旧方案"scan 均值 ÷ batch_size"间接推导）；P99 高 = 读锁竞争或 lease 刷新写开销 | `task_worker.cc` `ProcessLookup()` 扫描循环 `index == 0` 的 `Exist` 调用处（L275-279 内，批次上下文可得，无需门控） |
| `drampool_lookup_scan_duration_ms` | Histogram **[批次]** | 每请求一次：LOOKUP 元数据扫描总耗时（Σ `Exist` + 结果填充，不含响应写回）。测 LOOKUP 服务端主体处理时长（LOOKUP 无数据传输）；P99 高 = 分片读锁竞争或 lease 刷新开销；均值 ÷ 平均 batch_size ≈ 单 key 查询成本（与首条直测互为印证） | 入口 L273 构造 `ScopedTimer`，L284 响应提交前析构 |

> 注：LOOKUP 无 metadata 拆分阶段与数据传输——**首条 Exist 耗时即"首条数据的总时长"**，scan 即批次主体耗时；整批总耗时见 I 组 `lookup_batch_total_duration_ms`。

#### E. 传输与响应（6，原样保留）

| 指标名 | 类型 | 功能（测什么 / 反映什么 / 怎么用） | 埋点位置 |
|---|---|---|---|
| `drampool_dump_transfer_duration_ms` | Histogram **[批次]** | DUMP 数据传输从提交（`submit_ms`，TaskWorker 写入）到**终态**（Completed / Failed / GetStatus 异常）的异步耗时，涵盖网络与对端读取。反映实际搬运能力；`rate(bytes)/均值` = 传输带宽（§4.2）；与 prepare 相加 ≈ DUMP 服务端处理主体 | `completion_poller.cc` `SettleDataTransfer()` 统一出口（L263 入口，一处覆盖 GetStatus 异常 / Failed / Completed 三条终态路径），`MetricsObserve(now − submit_ms)` |
| `drampool_load_transfer_duration_ms` | Histogram **[批次]** | 同上，LOAD 侧（池 → 客户端方向搬运时长）。反映读取侧搬运能力；带宽推导同上 | 同上，按 opcode 二选一 |
| `drampool_transfer_failures_total` | Counter | 数据传输以**非 Completed 终态**结束的请求数（DUMP/LOAD 合并大类；Failed 或 GetStatus 异常均计）。测传输失败强度；增长需排查对端网络 / 连接状态（与 opcode 无关，低频不细分） | 同上统一出口（`terminalStatus != Completed` 合并计数） |
| `drampool_response_rtt_ms` | Histogram **[批次]** | 响应从本地提交（响应写回传输发起，`submit_ms` 于 SubmitResponse 成功后更新，L215）到**写回客户端内存完成**的端到端耗时（含响应传输本身）。测客户端感知的"结果返回"时延；P99 高 = 对端写入慢（flag 池等待由 I 组批次总耗时覆盖，不在此重复计入） | `completion_poller.cc` `PollResponseTransfer()` 终态出口（L224-260，Completed / 异常 / Failed 三路径统一），`MetricsObserve(now − submit_ms)` |
| `drampool_response_failures_total` | Counter | 响应返回链路失败计数（**本地提交** Allocate 非 NoSpace 失败 / Pack 失败 / ExecuteAsync 失败 + **写回传输**失败，DUMP/LOAD 合并大类；flag 池 NoSpace 属重试不算失败）。测结果返回通道健康度；增长锁定响应链路故障域，结合 WARN/ERROR 日志定位提交 / 写回哪一环 | `SubmitResponse()` L174-177、L187-192、L205-212 + `PollResponseTransfer()` L228-233、L248-253 五点合并 |
| `drampool_submit_failures_total` | Counter | 数据传输**提交失败**的请求数（DUMP/LOAD 合并大类；ExecuteAsync 失败或 handle 无效；LOAD 侧整批已 LoadEnd 释放引用）。测传输子系统提交路径健康度；增长指向传输子系统初始化 / 资源异常，客户端整批失败 | `task_worker.cc` `ProcessDump()` L160-169 + `ProcessLoad()` L241-250 两点合并 |

#### F. 资源水位（3，原样保留；数据池由 GCThreadLoop 每 `gcIntervalMs` 一轮采样、flag 池由 CompletionPoller 调用点记账，均无热路径开销）

| 指标名 | 类型 | 功能（测什么 / 反映什么 / 怎么用） | 埋点位置 |
|---|---|---|---|
| `drampool_metadata_entry_count` | Gauge **[覆盖]** | 当前池内缓存条目（block）总数（1024 分片求和）。反映元数据规模容量水位；增长斜率反映写入 / 驱逐平衡，配合 usage_ratio 判断驱逐压力区 | `drampool_server.cc` `GCThreadLoop()`（L552-561），`MetricsSet(GetKeyCnt())`（接口已存在） |
| `drampool_buffer_pool_usage_ratio_<slot_size>` | Gauge **[覆盖]**（动态注册，每 block size 一个） | 各 block 尺寸数据池**已用槽位占比**（used / slot count）。反映分尺寸内存水位；逼近 1 = 该尺寸池即将触发驱逐重试（DUMP prepare 抖动与 `nospace_failures` 的前兆），容量规划第一信号 | `GCThreadLoop()` 遍历 `poolBlockSizes`（L552-561）；启动期按尺寸动态 `CreateStats` + `BufferManager.GetUsedSlotRatio()`（新增原子记账） |
| `drampool_flag_pool_usage_ratio` | Gauge **[覆盖]** | flag 响应缓冲池**已用槽位占比**（used / flagBufferSlotCount）。反映响应回填缓冲水位；逼近 1 = B2 链（flag 池 NoSpace → `response_buffer_retry`）前兆信号，且为"flag 池扩容"决策提供交叉验证（此前无水位指标可查） | `completion_poller.cc` 调用点记账（§4.3）：`SubmitResponse()` Allocate 成功（L164）+1、`ReleaseResponseBuffer()` Free 成功（L38）−1、`PollPendingCompletions()` 每轮尾部上报 |

#### G. Metadata 阶段耗时（8 = [首条]×5 + [entry]×2 + [轮]×1）

> **本次重设计**：storebegin / allocate / shard_register / evict_sync / loadbegin 5 个从"每 entry 观测"改为"**仅批次首条观测**"（§4.7.1 门控）——样本量从 entry 数降为批次数（C3 溢出进一步缓解），且与 B/C 组 prepare（[批次]）形成"总-分"同粒度对照。观测点位置不变（metadata.cc 内部）；恢复 entry 级 = 去掉门控条件（§10）。

| 指标名 | 类型 | 功能（测什么 / 反映什么 / 怎么用） | 埋点位置 |
|---|---|---|---|
| `drampool_metadata_storebegin_duration_ms` | Histogram **[首条]** | 批次首条 `MetadataManager::StoreBegin` 调用的**总耗时** = 缓冲分配（含驱逐重试）+ 分片注册。测 DUMP 写路径元数据开销总视角（首条口径）；P99 抬高时按 allocate / register 两阶段直方图归因（§4.5） | `metadata.cc` `MetadataManager::StoreBegin()` 入口（门控消费点，覆盖成功 / 失败 / DuplicateKey 全路径） |
| `drampool_metadata_allocate_duration_ms` | Histogram **[首条]** | 首条 StoreBegin 内**缓冲分配阶段**耗时（首次 Allocate → NoSpace 触发周期驱逐重试 → 再 NoSpace 触发深度驱逐重试，含驱逐耗时，与 evict_sync 嵌套）。测"拿到一块缓冲"的真实成本；高且 evict 占比高 = 内存压力；高而 evict 占比低 = 共享 BufferPool Allocate 本身慢 | `StoreBegin()` 分配段独立作用域（首次 `Allocate` 调用前构造，含两次驱逐重试全程） |
| `drampool_metadata_shard_register_duration_ms` | Histogram **[首条]** | 首条 StoreBegin 内**分片注册阶段**耗时（分片写锁 + 重复检查 + 双驱逐策略 AddKey + map emplace，失败含回滚 Free）。测元数据注册成本；高 = 分片写锁竞争或 LRU / TTL 链表操作开销大 | `shards_[idx]->StoreBegin()` 调用处（L196） |
| `drampool_metadata_evict_sync_duration_ms` | Histogram **[首条]** | 首条 StoreBegin 触发的**业务线程内同步驱逐**（周期 / 深度两级）单次耗时。测驱逐对 DUMP 请求的**直接拖慢**（TaskWorker 线程内执行，阻塞当前批次后续 entry）；增长 = 内存压力直接信号，与 `usage_ratio` 交叉验证 | `MetadataManager::StoreBegin()` 两处 `EvictOneShard` 调用处（L183 / L187），仅门控开放的首次 StoreBegin 内观测 |
| `drampool_metadata_storeend_duration_ms` | Histogram **[entry]** | 每次 `StoreEnd`（DUMP 传输 Completed 后 entry INITIALIZED→READY）耗时，CompletionPoller 线程执行。测 DUMP 终态结算的单 entry 成本（跨批次逐 entry 结算，无批次首条概念，保持 entry 级）；高 = 分片读锁竞争（GC 每秒全分片扫描持读锁） | `metadata.cc` `ShardMetadata::StoreEnd()` 函数体（覆盖全部调用路径） |
| `drampool_metadata_loadbegin_duration_ms` | Histogram **[首条]** | 批次首条 `LoadBegin`（查 key + TryIncRef 加引用 + 双驱逐策略 AccessKey）耗时。测 LOAD 读路径元数据开销（首条口径）；高 = 读锁竞争，或策略 AccessKey（LRU move-to-front / TTL 更新）开销大 | `metadata.cc` `ShardMetadata::LoadBegin()` 函数体（门控消费点） |
| `drampool_metadata_loadend_duration_ms` | Histogram **[entry]** | 每次 `LoadEnd`（TryDecRef 减引用）耗时（出现在 poller 终态结算、len 不匹配回滚、提交失败回滚三类路径）。测引用释放成本（单次应近常数）；异常升高 = 锁竞争 | `ShardMetadata::LoadEnd()` 函数体 |
| `drampool_metadata_evict_gc_duration_ms` | Histogram **[轮]** | 后台 GC 每轮全分片驱逐扫描总耗时（`PerformEvict` 整轮 = 1024 shard 之和，每 `gcIntervalMs`（默认 1s）一轮）。测 GC 对系统的持续开销；均值 ÷ `gcIntervalMs` = GC 线程占空比；高 = 频繁持锁干扰业务（storeend / loadbegin 变慢的常见外因） | `drampool_server.cc` `GCThreadLoop()` `PerformEvict()` 调用处（L554） |

#### H. 队列与阻塞（9，原样保留）

> 两条 SPSC 队列与响应缓冲是请求端到端时延的关键路径（S0），排队 / 阻塞直接等价于客户端可感知延迟。线程衔接模型与阻塞链归因见 §4.6。

| 指标名 | 类型 | 功能（测什么 / 反映什么 / 怎么用） | 埋点位置 |
|---|---|---|---|
| `drampool_queue_request_enqueued_total` | Counter | 累计**成功进入** requestQueue 的请求数（TryPush 成功 +1）。测接收侧入口流量（"已入队待处理"口径，与 A 组"已开始处理"相差排队数）；与 `dequeued` 差值 = requestQueue 当前排队数（§4.6 恒等推导） | `drampool_server.cc` `RequestReceiveLoop()` TryPush 成功后（L517） |
| `drampool_queue_request_dequeued_total` | Counter | 累计 TaskWorker 从 requestQueue **取出**的请求数（TryPop 成功 +1）。测 TaskWorker 消费速率；与 `enqueued` 组成 SPSC 无丢弃恒等式（排队数 = 入队 − 出队） | `task_worker.cc` `Run()` TryPop 成功后（L49） |
| `drampool_queue_request_full_total` | Counter | requestQueue **满、TryPush 失败**事件数（每次失败 +1 按次计）。测接收线程被阻塞强度（阻塞时长 ≈ full 数 × `requestReceiverIdleWaitUs` 100µs）；阻塞链 B1；高概率压力场景的核心告警信号 | `drampool_server.cc` TryPush 失败分支（L518-523 WARN 处） |
| `drampool_queue_request_enqueue_wait_ms` | Histogram **[批次]** | 每请求从准备入队到 TryPush 成功的**入队前等待时长**（含满重试 sleep，未排队 ≈ 0）。测客户端可感知的接收背压延迟；P99 抬高必伴随 `full_total` 增长 | `drampool_server.cc` 入队重试循环外 `ScopedTimer`（L516 前构造） |
| `drampool_queue_completion_enqueued_total` | Counter | 累计**成功进入** completionQueue 的完成记录数（每请求一条，Push 自旋至成功恒 +1）。测完成流生产速率；与 `dequeued` 差值 = completionQueue 当前排队数 | `task_worker.cc` `SubmitCompletion()` Push 成功后（L326） |
| `drampool_queue_completion_dequeued_total` | Counter | 累计 CompletionPoller 从 completionQueue **取出**的完成记录数（FillPendingWindow 拉取成功 +1）。测完成流消费速率；与 `enqueued` 组成恒等式 | `completion_poller.cc` `FillPendingWindow()` TryPop 成功后（L71-72） |
| `drampool_queue_completion_full_total` | Counter | completionQueue **满、SubmitCompletion 被迫自旋等待**事件数（Push 前 TryPush 探测，失败 +1 后退回 Push——探测不改行为）。测 TaskWorker **停摆**位置与强度（既不取新请求也不响应停止指令）；阻塞链 B3，**停摆无日志兜底，此指标是唯一观测手段** | `task_worker.cc` `SubmitCompletion()` TryPush 探测失败分支（L320-328 微改造，§10） |
| `drampool_queue_completion_inflight` | Gauge **[覆盖]** | CompletionPoller pending 窗口内在途完成记录数（等终态 / 等响应提交 / 等写回，每轮覆盖写）。测完成链路第二级缓冲占用；持续逼近 `pollerPendingDepth`（默认 64）= 拉取停摆，与 `completion_full` 互相印证 | `completion_poller.cc` `PollPendingCompletions()` 每轮尾部（L117 后），`MetricsSet(pending_.size())` |
| `drampool_queue_response_buffer_retry_total` | Counter | 响应 flag 缓冲池 **NoSpace、SubmitResponse 留 pending 下轮重试**事件数。测响应缓冲供给不足（不阻塞 Poller 线程，但阻塞该请求响应提交、推高 I 组批次总耗时）；缓冲链 B2，与 `flag_pool_usage_ratio` 联动 | `completion_poller.cc` `SubmitResponse()` NoSpace 分支（L166-172 WARN 处） |

#### I. 批次总耗时（3，本次新增）

> **口径**：请求从 TaskWorker **出队**（开始处理）到 CompletionPoller **响应提交完成**（`SubmitResponse()` 内 Pack 成功、响应写回传输已发起）的服务端端到端处理总耗时。**包含**：全部本地处理（prepare / scan）+ 数据传输终态等待 + Poller 结算 + flag 缓冲等待（NoSpace 重试期间计入）+ 响应提交失败前的全部耗时；**不含**：requestQueue 排队等待（H 组观测）与响应写回传输本身（`response_rtt_ms` 单独观测）。三者关系：批次总耗时 + response_rtt ≈ 客户端感知总时延（再加排队等待）。观测点与批次 Gauge 同点同频（每批次一次，§4.7.2）。

| 指标名 | 类型 | 功能（测什么 / 反映什么 / 怎么用） | 埋点位置 |
|---|---|---|---|
| `drampool_dump_batch_total_duration_ms` | Histogram **[批次]** | 每批 DUMP 从出队到响应提交完成的总耗时。测整批写入的服务端端到端时延（用户明确要求的"整批数据总耗时"）；与 `dump_prepare_duration_ms` 的差值 = 数据传输等待 + Poller 结算 + flag 等待；分位数不可由三段速率均值推导，故直测 | `SubmitResponse()` 入口批次统计点（§4.7.2），`now_us − record.begin_us`，条件 `opcode == Dump` |
| `drampool_load_batch_total_duration_ms` | Histogram **[批次]** | 同上，LOAD 侧。测整批读取的服务端端到端时延；差值归因同上 | 同上，条件 `opcode == Load` |
| `drampool_lookup_batch_total_duration_ms` | Histogram **[批次]** | 同上，LOOKUP 侧（无数据传输，总耗时 ≈ scan + Poller 调度 + flag 缓冲等待）。与 `lookup_scan_duration_ms` 的差值 = 轮询调度延迟与 flag 等待（后者伴随 `response_buffer_retry` 增长） | 同上，条件 `opcode == Lookup` |

> 注：接入层拒绝（参数非法 / opcode 非法 / 响应超 flag slot 上限的配置拒绝）不产生 CompletionRecord，不进入批次统计——与"第二类非业务指标不涉及"的口径一致（该路径有 UC_ERROR 日志兜底）。batch_size 为 0 的请求仍观测批次总耗时（流程完整），比率类置 0。

---

## 4. 关键计算逻辑与观测机制

### 4.1 ScopedTimer 与 Disarm 约定

- 所有耗时观测用 RAII `ScopedTimer`（构造记起点、析构观测），零逻辑侵入；内部以 `SteadyNowUs()` µs 计时、以 ms（double）入直方图。
- **仅成功路径观测**的指标（dump/load prepare）：失败 / 短路 / 无传输分支显式 `Disarm()`。prepare 口径 = "数据传输提交成功"（与 `submit_ms` 写入点对齐）；配置拒绝、全 entry 跳过 / 未命中、提交失败的请求不入 prepare 分布（耗时结构完全不同，混入会污染分位数）。
- 门控类观测（§4.7.1）以 `std::optional<ScopedTimer>` 条件构造实现"仅首条武装"。

### 4.2 吞吐与带宽推导

```promql
# DUMP 传输带宽 (GB/s)
rate(ucm:drampool_dump_bytes_total[5m])
  / (rate(ucm:drampool_dump_transfer_duration_ms_sum[5m]) / rate(ucm:drampool_dump_transfer_duration_ms_count[5m]))
  / 1e6
# LOAD 同理。GC 占空比 = rate(evict_gc_sum)/rate(evict_gc_count) / gcIntervalMs
```

### 4.3 flag 池记账（flag_pool_usage_ratio）

CompletionPoller 单线程访问 flag 池，用普通 `size_t` 记账（无需 atomic）：`SubmitResponse()` Allocate 成功 +1（L164）、`ReleaseResponseBuffer()` Free 成功 −1（L38）；`PollPendingCompletions()` 每轮尾部 `MetricsSet(used / slotCount)`。与 `response_buffer_retry_total`（B2 信号）构成"水位前兆 + 事件确认"组合。

### 4.4 命中率（两种口径）

```promql
# 口径一：累计命中率（Counter 精确推导，长周期趋势）
sum(rate(ucm:drampool_lookup_hit_entries_total[5m]))
  / (sum(rate(ucm:drampool_lookup_hit_entries_total[5m])) + sum(rate(ucm:drampool_lookup_miss_entries_total[5m])))
# 口径二：当前批次命中率（Gauge 直取，瞬时实况）
ucm:drampool_lookup_batch_hit_ratio
# LOAD 命中率（entry 维度）：1 − rate(load_miss_entries_total) / rate(load_requests_total 附近 entry 总数)，
#   精确式 = 1 − miss/(miss + 成功加引 entry 数)，成功 entry 数可由 load_bytes_total / 平均 entry 大小近似
```

### 4.5 metadata 阶段层级与归因（G 组）

```
DUMP 请求（I 组：dump_batch_total_duration_ms，批次端到端）
├─ dump_prepare_duration_ms [批次]（本地准备总视角）
│  └─ metadata_storebegin_duration_ms [首条]（首条 entry 元数据总用时）
│     ├─ metadata_allocate_duration_ms [首条]（阶段①：缓冲分配）
│     │  └─ metadata_evict_sync_duration_ms [首条]（NoSpace 嵌套，同步驱逐）
│     └─ metadata_shard_register_duration_ms [首条]（阶段②：分片注册）
├─ dump_transfer_duration_ms [批次]（异步数据传输，poller 结算）
│  └─ Σ metadata_storeend_duration_ms [entry]（INITIALIZED→READY）
LOAD 请求
├─ load_prepare_duration_ms [批次] → metadata_loadbegin_duration_ms [首条]
└─ load_transfer_duration_ms [批次] → Σ metadata_loadend_duration_ms [entry]
LOOKUP 请求
├─ lookup_first_exist_duration_ms [首条]（单 key 直测）
├─ lookup_scan_duration_ms [批次]（主体耗时）
└─ lookup_batch_total_duration_ms [批次]（端到端）
后台 GC（每 gcIntervalMs 一轮）
└─ metadata_evict_gc_duration_ms [轮]
```

守恒校验：`rate(storebegin_sum) ≈ rate(allocate_sum) + rate(register_sum)`（偏差 = 胶水开销，正常远小于任一阶段）。注意 [首条] 化后各分量为"同一首条 entry"的分-总关系，与 prepare（整批）之间为"首条 vs 整批"的近似对照——整批归因仍靠 prepare 与 I 组批次总耗时。

| 症状组合 | 结论 | 动作方向 |
|---|---|---|
| allocate P99 高，evict_sync 占比高 | 内存压力触发驱逐重试 | 扩容 / 调整驱逐比例 / 排查突发写入 |
| allocate P99 高，evict_sync 占比低 | 共享 BufferPool 分配慢 | 排查池锁与伙伴分配实现 |
| register P99 高，GC 占空比高 | GC 扫描持锁干扰写锁 | 调大 gcIntervalMs 或降驱逐比例 |
| register P99 高，GC 占空比低 | 驱逐策略数据结构开销（LRU/TTL 链表） | 深挖策略实现 |
| storeend / loadbegin 抬高，与 evict_gc 同步 | 后台驱逐读锁与业务读锁互扰 | 同上 |
| first_exist P99 高而 scan 均值正常 | 首条撞上冷分片 / lease 刷新写放大 | 排查 TryMarkHit 与 lease 逻辑 |

### 4.6 队列流水线与阻塞链（H 组）

```
RequestReceiver ─TryPush─▶ requestQueue ─TryPop─▶ TaskWorker ─Push─▶ completionQueue ─TryPop─▶ CompletionPoller
      ▲ B1：满→sleep 重试（TCP 收包停滞）                              ▲ B3：满→Push 自旋停摆                    │
                                                                                                              ▼
                                                                              pending_ 窗口（completion_inflight）─B2：flag 池
                                                                              NoSpace 留 pending 下轮重试（单批次响应延迟）─▶ 响应写回
```

| 链 | 阻塞位置 | 形态 | 观测信号 |
|---|---|---|---|
| B1 | Receiver：requestQueue 满 | 主动 sleep 阻塞（100µs 重试）→ TCP 收包停滞 | `request_full_total` ↑ + `request_enqueue_wait_ms` 右移 |
| B3 | TaskWorker：completionQueue 满 | Push 自旋停摆 → requestQueue 堆积反压 B1 | `completion_full_total` ↑ → `request_full_total` 连锁 ↑ |
| B2 | Poller：flag 池 NoSpace | 单批次留 pending 重试（不阻塞线程） | `response_buffer_retry_total` ↑ + `completion_inflight` 贴近 64 + `flag_pool_usage_ratio` 逼近 1 |

归因速查：`request_full` ↑ 而 `completion_full` 平 → B1（TaskWorker 消费不足）；`completion_full` ↑ 且 `request_full` 随后连锁 ↑ → B3（Poller 消费不足，反压传导）；`retry` ↑ + inflight 贴满 + flag 池水位逼近 1 → B2（flag 池扩容）；全为 0 但吞吐低 → 转向 §4.5 阶段归因与 I 组批次总耗时分析。

### 4.7 批次观测机制（本次重设计核心）

#### 4.7.1 首条门控（G 组 5 个 [首条] 指标）

**问题**：G 组观测点位于 metadata.cc 内部，被每 entry 调用，本身无批次上下文；"仅首条观测"需要跨层传递"本 entry 是否批次首条"。

**方案**：TaskWorker 线程批内顺序处理（线程内天然串行），用 **thread_local 门控标志 + take 语义**传递，数据结构零扩展：

```cpp
// task_worker.cc（每 worker 线程一份）
static thread_local bool tl_batch_first_pending = false;

Status TaskWorker::ProcessDump(...) {
    ...
    tl_batch_first_pending = true;               // 批次 entry 循环前置位（L120 循环前）
    for (index...) {
        runtime_.metadata.StoreBegin(key, entry); // 门控在 metadata 层消费
        ...
    }
}

// metadata.cc（观测点，take 语义：消费即复位）
const bool isFirst = tl_batch_first_pending;      // 读取即消费
tl_batch_first_pending = false;
std::optional<ScopedTimer> total, alloc, reg, evict;
if (isFirst) { total.emplace(kStorebeginDur); ... }  // 仅首条武装各阶段计时器
```

语义细节：

- **消费点**：`MetadataManager::StoreBegin` / `ShardMetadata::LoadBegin` 入口消费一次，`first` 值向内层 allocate / register / evict 观测点传递（函数内局部变量 / 调用参数），保证同一首条调用的分阶段观测同开同关。
- **首条定义**：批次内**首个实际执行的 StoreBegin / LoadBegin 调用**（含 DuplicateKey、含失败路径——门控在入口消费，与调用结果无关，忠实于"首条数据"口径）。
- **多 worker 隔离**：thread_local 每 worker 一份，批间顺序处理无跨批污染。
- **无需门控的观测**：B/C/D 组 prepare / scan / first_exist 均在 task_worker 批次函数体内（循环索引 / 单次构造天然批次粒度）；storeend / loadend（[entry]，Poller 逐 entry 结算无批次概念）与 evict_gc（[轮]）无条件观测。

#### 4.7.2 批次统计点与 `begin_us` 哨兵（I 组 3 个 Hist + 批次 Gauge ×4）

**问题**：批次总耗时起点在 TaskWorker（出队时刻），终点在 CompletionPoller（响应提交完成）——跨线程；且 `SubmitResponse()` 会被 flag 池 NoSpace 重试**重复进入**，批次统计必须只上报一次。

**方案**：`CompletionRecord` 新增 `begin_us`（µs 时间戳）兼作一次性哨兵：

```cpp
// drampool_types.h：唯一数据结构扩展
struct CompletionRecord {
    ...
    std::uint64_t submit_ms{0};
    std::uint64_t begin_us{0};   // 新增：批次出队时刻（µs）；Poller 上报后置 0 = "已上报"哨兵
};

// task_worker.cc：出队时刻以函数参数传递（签名 +1 参数，非数据结构）
Status TaskWorker::ProcessOneRequest(RequestTaskPtr task) {
    const auto beginUs = SteadyNowUs();          // 出队即取
    ... ProcessDump(*dump, peer, beginUs);       // → 三个记录构造点 + QueueResponse 均写入 record.begin_us
}

// completion_poller.cc：SubmitResponse 入口 = 全部 record 的唯一必经点、results 定稿处
bool CompletionPoller::SubmitResponse(CompletionRecord& record) {
    if (record.begin_us != 0) {                  // 哨兵：本批次尚未上报
        const auto elapsedMs = (SteadyNowUs() - record.begin_us) / 1000.0;
        switch (record.opcode) {                 // I 组：批次总耗时
            case Dump:   Observe(kDumpBatchTotalMs, elapsedMs);   break;
            case Load:   Observe(kLoadBatchTotalMs, elapsedMs);   break;
            case Lookup: Observe(kLookupBatchTotalMs, elapsedMs); break;
        }
        if (record.opcode == Lookup) {           // D 组批次 Gauge（DUMP 失败 Gauge 同理，条件 Dump）
            const auto hits = std::count(record.results.begin(), record.results.end(),
                                         static_cast<std::uint8_t>(LookupResult::Exists));
            MetricsSet(kLookupBatchHits, static_cast<double>(hits));
            MetricsSet(kLookupBatchHitRatio,
                       record.results.empty() ? 0.0 : static_cast<double>(hits) / record.results.size());
        }
        record.begin_us = 0;                     // 置哨兵：NoSpace 重试再入不重复上报
    }
    ...  // 既有 Allocate / Pack / ExecuteAsync 逻辑不变
}
```

**为何此点是唯一无重无漏的统计点**：

- **无漏**：所有请求终以 CompletionRecord 进入 completionQueue——传输路径（DUMP/LOAD 有传输）经 `SettleDataTransfer` 后转入 SubmitResponse 阶段；无传输路径（全跳过 / 未命中 / 提交失败 / LOOKUP）由 `QueueResponse` 直接以 SubmitResponse 阶段入队。四条失败路径（StoreBegin 短路 / 提交失败标记 / 传输失败结算 / StoreEnd 失败结算）均在入队前定稿 `record.results`。
- **无重**：`SubmitResponse` 仅在 flag 池 NoSpace 时以同阶段重入（L171，stage 不变返回 false）；成功则 stage 前移至 PollResponseTransfer，永久失败则 record 出窗——重入路径被 `begin_us == 0` 哨兵挡住。
- **覆盖完整终态**：统计在函数入口完成，之后的 Allocate 非 NoSpace 失败 / Pack 失败 / ExecuteAsync 失败不影响"批次已处理完"的事实（响应链路失败由 `response_failures_total` 独立归因）。

---

## 5. 实现设计

### 5.1 启动期注册（C2：未注册名字静默丢弃，必须集中注册）

启动路径（`drampool_launch_config.cc` 初始化序列，`Metrics::SetUp(maxVectorLen)` 之后）按 §3.2 A~I 组常量表逐一 `CreateStats`：46 个静态名 + 数据池按 `poolBlockSizes` 动态注册 `drampool_buffer_pool_usage_ratio_<slot_size>`（slot size 用原始字节数入名）。

### 5.2 数据结构与签名变更清单（最小化）

| 变更 | 位置 | 内容 |
|---|---|---|
| 数据结构 +1 字段（唯一） | `drampool_types.h` `CompletionRecord` | `std::uint64_t begin_us{0}`（§4.7.2） |
| 函数签名 +1 参数 ×4 | `task_worker.h/.cc` | `ProcessOneRequest` 内取 `beginUs`，向 `ProcessDump` / `ProcessLoad` / `ProcessLookup` / `QueueResponse` 传递并写入 `record.begin_us`（3 个直接构造点 + QueueResponse 构造点） |
| 微改造 ×1 | `task_worker.cc` `SubmitCompletion()` | Push 前 TryPush 探测（`completion_full_total` 计数，探测不改行为） |
| 新增记账 ×2 | `BufferManager`（原子 used 计数，`GetUsedSlotRatio()`）、`completion_poller.cc`（flag 池 used size_t） | F 组水位 |
| 新增时钟 ×1 | `drampool_types.h` | `SteadyNowUs()`（µs 精度，与 `SteadyNowMs()` 同型） |
| thread_local 门控 ×1 | `task_worker.cc` / `metadata.cc` | `tl_batch_first_pending`（§4.7.1） |

共享组件（`ucm/shared/pool`、`template/spsc_ring_queue.h` 等）零改动；仅新增 `ucm/shared/metrics` 链接依赖。

### 5.3 线程 × 指标归属矩阵（C5：首次 UpdateStats 的线程自动注册，无需手动登记线程）

| 线程 | 更新的指标 |
|---|---|
| RequestReceiver | H 组：request_enqueued / request_full / request_enqueue_wait_ms |
| TaskWorker（×N） | A 组 ×3；B 组 bytes；C 组 bytes / miss；D 组 hit / miss / first_exist；B/C 组 prepare；E 组 submit_failures；H 组 request_dequeued / completion_enqueued / completion_full；G 组门控置位 |
| CompletionPoller | B/D 批次 Gauge ×4；I 组 ×3；E 组 transfer ×2 / transfer_failures / response_rtt / response_failures；G 组 storeend / loadend；F 组 flag_pool_usage_ratio；H 组 completion_dequeued / completion_inflight / response_buffer_retry |
| GC 线程 | F 组 metadata_entry_count / buffer_pool_usage_ratio_*；G 组 evict_gc |

### 5.4 直方图桶集

19 个 Histogram 共用一组桶（`metricsDurationBucketsMs`）：亚毫秒段 `0.001–0.5`（覆盖 G 组 µs 级观测与 LOOKUP 批次）+ 毫秒段 `1–10000`（请求 / 传输 / 批次级）；向量长度（C3 上限）按"批次数 × 采样周期"配置，默认容纳 ≥2 个 Reporter 周期的批次量。

---

## 6. 快照输出（Reporter 线程）

- Reporter 每 10s 调用 `GetAllStatsAndClear()`（C4 读后即清），将增量**累加**为进程级累计值后序列化为一行 JSON（JSON Lines，轮转 + 清理）：`ts` + counter 表（累计值）+ gauge 表（最新值）+ histogram 表（count / sum，必要时含分位数）。
- **批次 Gauge 的快照语义**：窗口内可能覆盖多批，每行 JSON 反映**窗口内最近一批**实况——与"当前批次失败数 / 命中率"的观测意图一致；累计口径由 Counter（hit/miss/miss_entries）承载，两者互补。
- Histogram 增量在窗口间累加（count/sum 可加），分位数由回流器按桶重建或以 sum/count 均值近似。

## 7. UCM/Scheduler 侧回流器

与既有设计一致：Python 回流器运行于推理进程内，**Leader 选举（flock）**后尾部读取 DramPool JSON Lines 最新完整行（64KiB 窗口），Counter / Histogram 做周期 **delta 换算**（累计 → 增量速率），Gauge 直取（批次 Gauge 即"最近一批"实况），统一加 `ucm:` 前缀注册进 ucmmetrics → vllm_connector → Prometheus。两侧时钟不要求同步。

## 8. 测试方案

| # | 用例 | 断言 |
|---|---|---|
| 1 | 注册完备性 | 遍历常量指标名表逐一 `UpdateStats` 后 `GetAllStatsAndClear` 可见（C2 反向：未注册名丢弃） |
| 2 | 首条门控 | batch_size=N 的 DUMP / LOAD：G 组 5 个 [首条] Hist 每次 batch 恰 +1 样本；首条为 DuplicateKey / 失败路径时仍恰 +1；多 worker 并发无交叉污染 |
| 3 | 批次哨兵一次性 | flag 池 NoSpace 强制重试场景：批次 Gauge / I 组 Hist 仅 +1 样本；响应链路永久失败路径同样已上报 |
| 4 | 批次 Gauge 覆盖 | 连续两批不同失败数 / 命中数，快照反映最后一批；batch_size=0 比率置 0 |
| 5 | NoSpace 直测 | 构造数据池打满：`dump_nospace_failures_total` 增量 == 两次驱逐重试后仍 NoSpace 的 entry 数（注册失败 / DuplicateKey 不计入） |
| 6 | INITIALIZED 观测 | DUMP 在途（未 StoreEnd）时 LOAD 同 key：`load_initialized_entries_total` +1 且 miss +1；NotFound 仅 miss +1 |
| 7 | C3 溢出 | 观测数超 maxVectorLen 后先到样本保留、后续丢弃不崩溃 |
| 8 | 端到端 | 单批 DUMP→LOAD→LOOKUP 全链路：批次总耗时 > prepare/scan，批次 Gauge 与注入结果一致 |

## 9. 实施步骤（建议 PR 划分）

1. **PR-1 基础设施**：`SteadyNowUs()` + 启动期注册表 + Reporter/回流器骨架；A 组 + H 组 + F 组先行（无数据结构变更）。
2. **PR-2 批次观测**：`begin_us` 字段 + 签名传递 + `SubmitResponse` 统计点；I 组 ×3 + 批次 Gauge ×4 + 测试 3/4/8。
3. **PR-3 metadata 层**：G 组 ScopedTimer + 首条门控 + `nospace_failures` / `initialized_entries` 直测 + 测试 2/5/6。
4. **PR-4 收尾**：E/D 组埋点、`submit_completion` TryPush 探测微改造、lint（black/isort 仅 Python 侧）与全量测试。

## 10. 取舍留痕（裁剪 / 变更项与恢复方式）

| 类别 | 项 | 恢复方式 |
|---|---|---|
| 推导类（首轮裁剪，维持） | dump/load_bandwidth_gbps、batch_entries、dump_ttl_ms、interval_lookup_hit_rate | §4.2 / §4.4 公式 |
| 非业务类（16 项，维持不涉及） | 接入层协议错误、内部调度、低频异常分类、传输超时、GC 诊断等 | UC_WARN/UC_ERROR 日志兜底 |
| 失败类合并（维持） | submit / transfer / response_failures 三大类 | 需按 opcode 细分时在同一统计点按 opcode 拆名即可 |
| **本次删除** | ~~`drampool_dump_storebegin_failures_total`~~ | 在 `task_worker.cc` `ProcessDump()` `storeStatus.Failure()` 分支（L134-141）恢复 `MetricsCount`；若需"分配失败 vs 注册失败"细分，在 metadata 层 StoreBegin 返回前按 Status 类别分别计数 |
| **本次粒度改造** | G 组 5 个 [首条] 化（原 entry 级） | 去掉 §4.7.1 门控（观测点无条件武装 ScopedTimer）即恢复 entry 级 |
| **本次口径变更** | 批次失败 / 命中 Gauge（覆盖写） | 需累计口径时将同一统计点的 `MetricsSet` 改为 `MetricsCount`（位置不变，语义切换） |
| **本次新增直测** | `lookup_first_exist_duration_ms` | 删除后回退"scan 均值 ÷ 平均 batch_size"推导 |
| 维持 entry / 轮级 | storeend / loadend / evict_gc | 无变更 |

