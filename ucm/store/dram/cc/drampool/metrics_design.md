# DramPool Metrics 埋点与跨进程回流设计方案

| 项 | 说明 |
|---|---|
| 范围 | DramPool 服务端守护进程（`ucm/store/dram/cc/drampool/`）**业务逻辑**埋点 + 快照日志输出；UCM/Scheduler 侧新增回流器组件。DramStore 客户端侧零改动，HealthServer 零改动 |
| 核心诉求 | 解决 DramPool 独立进程与 vLLM/UCM 跨进程指标统一暴露，对接现有 Prometheus 监控链路 |
| 设计原则 | **业务聚焦 + 批次观测**：以**批次（请求）粒度**为主线观测 DUMP/LOAD/LOOKUP 服务质量——**整批总耗时**与**批次失败累计**，辅以累计 Counter、队列阻塞信号与资源水位。接入层协议错误、内部调度诊断等非必要指标不涉及（相应路径维持 UC_WARN/UC_ERROR 日志） |
| 共享代码约束 | 不修改 `ucm/shared/pool`、`ucm/shared/infra/template` 等共享组件；仅新增对 `ucm/shared/metrics` 的链接依赖 |
| 指标规模 | **32 个**（13 Histogram + 13 Counter + 6 类 Gauge，其中 1 类动态注册） |
| 数据结构影响 | `CompletionRecord` **新增 `begin_us` 1 个字段**（批次出队时刻，µs；TaskWorker 在记录构造处写入，Poller 读取；上报后置 0 复用作"批次统计已上报"哨兵，§4.7）；`RequestTask` 原样 |
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
| C3 | **Histogram 存原始值向量**，上限 `maxVectorLen`，**满后丢弃后续观测**（保留先到的样本） | 观测点控制在**批次粒度**（每批次 1 次观测：prepare / scan / 批次总耗时）与请求响应粒度；G 组 entry 级观测（storeend/loadend）为每 entry 一次、evict_gc 为每轮一次；向量长度可配置 |
| C4 | **`GetAllStatsAndClear()` 读后即清**：返回的是"距上次调用的增量" | **Reporter** 将周期增量累加至进程级累计值后再写 JSON（每行快照为累计终点状态）；增量换算由回流器在 UCM 侧完成 |
| C5 | 线程级双缓冲，首次 `UpdateStats` 的线程自动注册，无需手动登记 | 各 worker 线程直接调用即可 |

### 2.4 命名约定

- JSON 日志中指标名与 Prometheus 输出名分离：JSON 内为 `drampool_*`，回流器注册到 ucmmetrics 时统一加 `ucm:` 前缀（对齐 UCM 现有习惯）。
- 单位后缀进指标名：`_ms`、`_entries_total`、`_ratio`。
- Counter 一律 `_total` 结尾；动态注册名中的 slot size 用原始字节数（如 `drampool_buffer_pool_usage_ratio_1048576`），保证名字合法且稳定。

---

## 3. 指标必要性审查与全集

### 3.1 判定标准（按优先级排序）

| # | 标准 | 含义 |
|---|---|---|
| **S0** | **业务直接相关** | 直接反映 DUMP/LOAD/LOOKUP 操作的服务质量（请求量、成功率、时延、命中率）、**批次粒度耗时**或资源水位（池用量、元数据规模）。纯系统性/接入层/内部调度/低频异常分类指标**此阶段不涉及** |
| S1 | 不可推导 | 无法由其他指标经 PromQL 运算（rate 比值、差值）等价获得 |
| S2 | 可行动 | 指标异常时存在明确的运维/开发动作 |
| S3 | 业务失败分类完整 | 缺失会导致业务失败被误判（如"重复写"与"真失败"） |
| S4 | 开销合理 | 热路径 Histogram 数量从严控制 |

**metadata 耗时观测（G 组）的判定说明（二次修改后）**：

- 首条粒度的阶段拆分观测（storebegin / allocate / shard_register / evict_sync / loadbegin 5 个直方图）**已按需求删除**——批次内的阶段归因由 B/C 组 prepare（[批次] 整批视角）与 I 组批次总耗时承载；留痕与恢复方式见 §10。
- 保留的 3 个观测均与批次首条无关：storeend / loadend（[entry]，CompletionPoller 逐 entry 结算，无批次首条概念）与 evict_gc（[轮]，GC 线程轮粒度）。
- 仍被裁掉的项（留痕，恢复方式见 §10）：`metadata_delete_duration_ms`（非正常路径，日志完备）、驱逐 scan/release 二次拆分、`Exist` 单次直方图（由 scan 批次粒度覆盖，单 key 成本 ≈ scan 均值 ÷ 平均 batch_size）。

**第一类：可由其他指标推导（6 项，违反 S1）**

| 被裁指标 | 替代观测 |
|---|---|
| `dump/load_bandwidth_gbps`（Histogram ×2） | 原 `rate(bytes)` 带宽推导（四次修改后 bytes 指标已删除，带宽推导随之取消，§10 留痕） |
| `dump/load_batch_entries`（Histogram ×2） | 原 `rate(bytes)/rate(requests)` 近似（四次修改后 bytes 指标已删除）；批量压力由 prepare/transfer 时长分布体现 |
| `dump_ttl_ms`（Histogram） | TTL 为客户端入参透传，客户端侧可自行审计 |
| `interval_lookup_hit_rate`（Gauge） | 原 hit/miss Counter 的 rate 比值精确推导（§4.4；五次修改后 hit 指标已删除，命中率暂不可由指标推导，§10 留痕） |

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
| **删除（本次重设计）** | ~~`drampool_dump_storebegin_failures_total`~~ | 其"缓冲分配失败"主因由 `dump_nospace_failures_total` **直测精确归因**（原指标混入注册失败、且分配失败 ≠ 都是 NoSpace）；批次失败实况由 `dump_failed_entries_total`（累计，三次修改后）承载 |
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

> 用户观测口径：**整批总耗时**、**批次失败累计**。首条粒度观测（原 G 组 5 个 + `lookup_first_exist`）**已按需求二次删除**，首条门控机制随之移除；批次失败/命中 Gauge（覆盖写）**已按需求三次修改改为累计口径**——失败改 `dump_failed_entries_total`（Counter），命中 Gauge 与两个批次比率 Gauge 经重复性分析删除（与既有 Counter 重复，见下）。

| 处置 | 事项 | 理由 |
|---|---|---|
| **删除（二次修改）** | G 组 storebegin / allocate / register / evict_sync / loadbegin 5 个 [首条] 直方图 + `lookup_first_exist_duration_ms` | 按需求删除首条数据时间指标；thread_local 首条门控机制（原 §4.7.1）随之整体移除，TaskWorker / metadata 层零改动恢复原状；批次内阶段归因由 prepare（[批次]）与 I 组批次总耗时承载（§10 留痕） |
| **保留 entry 级** | storeend / loadend（G 组） | 观测线程为 CompletionPoller（跨批次逐 entry 结算），无"批次首条"概念；本身是"每 entry 结算成本"观测 |
| **保留轮级** | evict_gc | GC 轮粒度天然与批次无关 |
| **删除（三次修改）** | `lookup_batch_hits` / `lookup_batch_hit_ratio`（Gauge） | 累计化后与既有 Counter **完全重复**：hits ≡ `lookup_hit_entries_total`（同扫描循环累计）；累计命中率 = hit/(hit+miss) 已由 §4.4 公式推导；比率语义亦非 Counter 所能承载 |
| **新增（三次修改后：Gauge → Counter）** | `dump_failed_entries_total`（DUMP 失败 entry 累计，原批次失败 Gauge ×2 改造） | 统计点 = record.results 定稿处（四条失败路径唯一汇合点，无重无漏）；重复性分析：与 `nospace_failures`（仅 NoSpace 子类）、`transfer/submit_failures`（请求级）维度不同、无重复；比率 Gauge 因 Counter 无法承载 + 同点冗余删除 |
| **新增** | 批次总耗时 Hist ×3 | 用户明确"整批数据总耗时"；分位数**不可**由 prepare+transfer+rtt 三段速率均值推导（仅稳态近似）——直测需跨线程传递批次入口时刻 → `CompletionRecord` +1 字段（机制与既有 `submit_ms` 相同） |

### 3.2 指标全集（32 个）

> 功能描述统一为三段式：**测什么**（观测口径）→ **反映什么**（异常时指向的问题）→ **怎么用**（典型分析动作）。
>
> 观测粒度标注：**[批次]** = 每批次一次；**[entry]** = 每 entry 一次；**[轮]** = GC 每轮一次；**[覆盖]** = Gauge 最新值覆盖写。

#### A. 请求量（3）

| 指标名 | 类型 | 功能（测什么 / 反映什么 / 怎么用） | 埋点位置 |
|---|---|---|---|
| `drampool_dump_requests_total` | Counter | 累计接收并处理的 DUMP 请求数（每请求 +1，与 batch 内 entry 数无关）。反映写入负载规模；与 load/lookup 请求量对比反映负载构成 | `task_worker.cc` `ProcessOneRequest()` DUMP 分支（L75） |
| `drampool_load_requests_total` | Counter | 累计接收并处理的 LOAD 请求数。反映读取负载规模；读取侧所有 rate 类指标（miss、transfer 时长）的分母基准 | 同上 LOAD 分支（L80） |
| `drampool_lookup_requests_total` | Counter | 累计接收并处理的 LOOKUP 请求数。反映查询负载规模；也是 `lookup_scan_duration_ms` 均值换算"单请求查询成本"的分母 | 同上 LOOKUP 分支（L85） |

#### B. DUMP 业务（3）

| 指标名 | 类型 | 功能（测什么 / 反映什么 / 怎么用） | 埋点位置 |
|---|---|---|---|
| `drampool_dump_nospace_failures_total` | Counter | 累计 StoreBegin 中**缓冲分配最终失败且原因为 NoSpace** 的 entry 数——两次驱逐重试后仍 `Status::NoSpace()` 才计数，注册失败不计。**内存压力的精确归因**：持续增长 = 数据池容量不足或驱逐策略不足以释放空间；与 `buffer_pool_usage_ratio_*` 交叉验证。行动入口：扩容 / 调整驱逐比例 | `metadata.cc` `MetadataManager::StoreBegin()` 第二次深度驱逐重试后、最终 `return Error` 前（L193 附近，`st == Status::NoSpace()` 判定点） |
| `drampool_dump_failed_entries_total` | Counter | 累计**结果为 Failed** 的 DUMP entry 数（StoreBegin 失败短路标记 / 提交失败标记 / 传输失败结算 / StoreEnd 失败结算四条路径合并，`record.results` 定稿值）。测写入失败总规模；速率 >0 持续 = 写入链路异常，结合 `nospace_failures`（元数据侧）与 `transfer/submit_failures`（传输/提交侧）区分根因。重复性分析：与上述 Counter 维度不同、无重复 | `completion_poller.cc` `SubmitResponse()` Pack 前（`record.results` 定稿处——四条失败路径的唯一汇合点），哨兵块内对 Failed 结果计数 `MetricsCount(+N)`，条件 `opcode == Dump` |
| `drampool_dump_prepare_duration_ms` | Histogram **[批次]** | 每请求一次观测：DUMP 从开始处理到数据传输提交完成的**本地准备耗时**（逐 entry StoreBegin + 缓冲分配 + 可能的驱逐重试 + 提交），**不含**数据传输本身。测写入路径的服务端 CPU 侧开销。P99 高说明准备阶段拖慢写入，结合 `nospace_failures` 与 `buffer_pool_usage_ratio_*` 归因（§4.5） | 入口 L95 构造 `ScopedTimer`，L159-168 失败分支 Disarm（§4.1） |

> 注：重复 key entry 在循环内 `continue`（L129-132），不进入失败计数口径；`nospace_failures` 在 metadata 层直测、天然不受 DuplicateKey 影响。`failed_entries_total` 统计 `record.results` 定稿值——失败短路标记（`mark_remaining_failed`）的剩余 entry 一并计入，反映批次最终实况；NoSpace 重试再入时哨兵挡住重复计数。

#### C. LOAD 业务（2）

| 指标名 | 类型 | 功能（测什么 / 反映什么 / 怎么用） | 埋点位置 |
|---|---|---|---|
| `drampool_load_miss_entries_total` | Counter | 累计 LOAD 中**未取到数据**的 entry 数——即登记本 miss：**LoadBegin 失败**（key 不存在或状态非 READY）与**请求长度大于存储长度**两类之和，对客户端都是"没取到"。测读取未命中规模。突增说明客户端读取了已被驱逐/过期/尚未写完的数据；命中率精确式依赖成功 entry 直测（四次修改后 bytes 指标已删除，暂以 miss 绝对速率监控为主，§4.4） | 同上 L187-192 |
| `drampool_load_prepare_duration_ms` | Histogram **[批次]** | 每请求一次观测：LOAD 从开始处理到传输提交完成的**本地准备耗时**（逐 entry LoadBegin + 长度校验 + 提交），**不含**传输本身。测读取路径服务端 CPU 侧开销；异常时结合 miss 计数与 I 组批次总耗时归因 | 入口 L184 构造，L221-230 失败分支 Disarm |

#### D. LOOKUP 业务（2）

| 指标名 | 类型 | 功能（测什么 / 反映什么 / 怎么用） | 埋点位置 |
|---|---|---|---|
| `drampool_lookup_miss_entries_total` | Counter | 累计 LOOKUP 中**不存在或非 READY** 的 entry 数（`batch_size − hit 数`，hit 数仅循环内累计不上报——五次修改后 hit 指标已删除）。测累计未命中规模；突增 = 查询了被驱逐 / 过期的前缀，通常先于 LOAD miss 出现（驱逐过快的预警信号）；命中率暂不可由指标推导（§4.4 注） | `task_worker.cc` `ProcessLookup()` 扫描循环（L275-279，`batch_size − hit 数` 循环外一次上报） |
| `drampool_lookup_scan_duration_ms` | Histogram **[批次]** | 每请求一次：LOOKUP 元数据扫描总耗时（Σ `Exist` + 结果填充，不含响应写回）。测 LOOKUP 服务端主体处理时长（LOOKUP 无数据传输）；P99 高 = 分片读锁竞争或 lease 刷新开销；均值 ÷ 平均 batch_size ≈ 单 key 查询成本 | 入口 L273 构造 `ScopedTimer`，L284 响应提交前析构 |

> 注：LOOKUP 无 metadata 拆分阶段与数据传输——scan 即批次主体耗时，单 key 成本由 scan 均值 ÷ 平均 batch_size 近似；整批总耗时见 I 组 `lookup_batch_total_duration_ms`。批次命中实况 Gauge（`lookup_batch_hits` / `lookup_batch_hit_ratio`）已按三次修改删除、hit Counter 已按五次修改删除——查询命中规模与命中率暂无直测，以 miss 绝对速率与突增监控为主（§4.4 / §10 留痕）。

#### E. 传输与响应（6，原样保留）

| 指标名 | 类型 | 功能（测什么 / 反映什么 / 怎么用） | 埋点位置 |
|---|---|---|---|
| `drampool_dump_transfer_duration_ms` | Histogram **[批次]** | DUMP 数据传输从提交（`submit_ms`，TaskWorker 写入）到**终态**（Completed / Failed / GetStatus 异常）的异步耗时，涵盖网络与对端读取。反映实际搬运能力；与 prepare 相加 ≈ DUMP 服务端处理主体 | `completion_poller.cc` `SettleDataTransfer()` 统一出口（L263 入口，一处覆盖 GetStatus 异常 / Failed / Completed 三条终态路径），`MetricsObserve(now − submit_ms)` |
| `drampool_load_transfer_duration_ms` | Histogram **[批次]** | 同上，LOAD 侧（池 → 客户端方向搬运时长）。反映读取侧搬运能力 | 同上，按 opcode 二选一 |
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

#### G. Metadata 结算耗时（3 = [entry]×2 + [轮]×1）

> **本次二次修改**：原首条粒度的 storebegin / allocate / shard_register / evict_sync / loadbegin 5 个直方图已删除（§3.1 / §10）；保留的 3 个观测与批次首条无关——storeend / loadend 由 CompletionPoller 跨批次逐 entry 结算（无批次首条概念），evict_gc 为 GC 线程轮粒度。观测点位于 `metadata.cc` / `drampool_server.cc` 内部（模块边界测量），`ScopedTimer` RAII 零逻辑侵入；时钟 µs 精度（`SteadyNowUs()`）、以 ms（double）入直方图。

| 指标名 | 类型 | 功能（测什么 / 反映什么 / 怎么用） | 埋点位置 |
|---|---|---|---|
| `drampool_metadata_storeend_duration_ms` | Histogram **[entry]** | 每次 `StoreEnd`（DUMP 传输 Completed 后 entry INITIALIZED→READY）耗时，CompletionPoller 线程执行。测 DUMP 终态结算的单 entry 成本；高 = 分片读锁竞争（GC 每秒全分片扫描持读锁） | `metadata.cc` `ShardMetadata::StoreEnd()` 函数体（覆盖全部调用路径） |
| `drampool_metadata_loadend_duration_ms` | Histogram **[entry]** | 每次 `LoadEnd`（TryDecRef 减引用）耗时（出现在 poller 终态结算、len 不匹配回滚、提交失败回滚三类路径）。测引用释放成本（单次应近常数）；异常升高 = 锁竞争 | `ShardMetadata::LoadEnd()` 函数体 |
| `drampool_metadata_evict_gc_duration_ms` | Histogram **[轮]** | 后台 GC 每轮全分片驱逐扫描总耗时（`PerformEvict` 整轮 = 1024 shard 之和，每 `gcIntervalMs`（默认 1s）一轮）。测 GC 对系统的持续开销；均值 ÷ `gcIntervalMs` = GC 线程占空比；高 = 频繁持锁干扰业务（storeend / loadend 变慢的常见外因） | `drampool_server.cc` `GCThreadLoop()` `PerformEvict()` 调用处（L554） |

#### H. 队列与阻塞（7）

> 两条 SPSC 队列与响应缓冲是请求端到端时延的关键路径（S0），排队 / 阻塞直接等价于客户端可感知延迟。线程衔接模型与阻塞链归因见 §4.6。

| 指标名 | 类型 | 功能（测什么 / 反映什么 / 怎么用） | 埋点位置 |
|---|---|---|---|
| `drampool_queue_request_full_total` | Counter | requestQueue **满、TryPush 失败**事件数（每次失败 +1 按次计）。测接收线程被阻塞强度（阻塞时长 ≈ full 数 × `requestReceiverIdleWaitUs` 100µs）；阻塞链 B1；高概率压力场景的核心告警信号 | `drampool_server.cc` TryPush 失败分支（L518-523 WARN 处） |
| `drampool_queue_request_enqueue_wait_ms` | Histogram **[批次]** | 每请求从准备入队到 TryPush 成功的**入队前等待时长**（含满重试 sleep，未排队 ≈ 0）。测客户端可感知的接收背压延迟；P99 抬高必伴随 `full_total` 增长 | `drampool_server.cc` 入队重试循环外 `ScopedTimer`（L516 前构造） |
| `drampool_queue_request_size` | Gauge **[覆盖]**（六次修改：替代原 enqueued/dequeued ×2 Counter） | requestQueue **当前排队中的请求数**（TryPush 成功后与 TryPop 成功后覆盖写，最新值生效）。测接收侧待处理积压；持续增长 = TaskWorker 消费跟不上到达速率或下游反压传导（B1/B3 定位，§4.6） | `drampool_server.cc` `RequestReceiveLoop()` TryPush 成功后（L517）+ `task_worker.cc` `Run()` TryPop 成功后（L49）两点 `MetricsSet(队列当前长度)` |
| `drampool_queue_completion_full_total` | Counter | completionQueue **满、SubmitCompletion 被迫自旋等待**事件数（Push 前 TryPush 探测，失败 +1 后退回 Push——探测不改行为）。测 TaskWorker **停摆**位置与强度（既不取新请求也不响应停止指令）；阻塞链 B3，**停摆无日志兜底，此指标是唯一观测手段** | `task_worker.cc` `SubmitCompletion()` TryPush 探测失败分支（L320-328 微改造，§10） |
| `drampool_queue_completion_inflight` | Gauge **[覆盖]** | CompletionPoller pending 窗口内在途完成记录数（等终态 / 等响应提交 / 等写回，每轮覆盖写）。测完成链路第二级缓冲占用；持续逼近 `pollerPendingDepth`（默认 64）= 拉取停摆，与 `completion_full` 互相印证 | `completion_poller.cc` `PollPendingCompletions()` 每轮尾部（L117 后），`MetricsSet(pending_.size())` |
| `drampool_queue_completion_size` | Gauge **[覆盖]**（六次修改：替代原 enqueued/dequeued ×2 Counter） | completionQueue **当前排队的完成记录数**（Push 成功后与 TryPop 成功后覆盖写，最新值生效）。测完成流第二级缓冲积压；持续增长 = Poller 消费不足，随后 `request_full` 连锁 ↑（B3 反压传导的前置信号） | `task_worker.cc` `SubmitCompletion()` Push 成功后（L326）+ `completion_poller.cc` `FillPendingWindow()` TryPop 成功后（L71-72）两点 `MetricsSet(队列当前长度)` |
| `drampool_queue_response_buffer_retry_total` | Counter | 响应 flag 缓冲池 **NoSpace、SubmitResponse 留 pending 下轮重试**事件数。测响应缓冲供给不足（不阻塞 Poller 线程，但阻塞该请求响应提交、推高 I 组批次总耗时）；缓冲链 B2，与 `flag_pool_usage_ratio` 联动 | `completion_poller.cc` `SubmitResponse()` NoSpace 分支（L166-172 WARN 处） |

#### I. 批次总耗时（3，本次新增）

> **口径**：请求从 TaskWorker **出队**（开始处理）到 CompletionPoller **响应提交完成**（`SubmitResponse()` 内 Pack 成功、响应写回传输已发起）的服务端端到端处理总耗时。**包含**：全部本地处理（prepare / scan）+ 数据传输终态等待 + Poller 结算 + flag 缓冲等待（NoSpace 重试期间计入）+ 响应提交失败前的全部耗时；**不含**：requestQueue 排队等待（H 组观测）与响应写回传输本身（`response_rtt_ms` 单独观测）。三者关系：批次总耗时 + response_rtt ≈ 客户端感知总时延（再加排队等待）。观测点与 `dump_failed_entries_total` 同点同频（每批次一次，§4.7）。

| 指标名 | 类型 | 功能（测什么 / 反映什么 / 怎么用） | 埋点位置 |
|---|---|---|---|
| `drampool_dump_batch_total_duration_ms` | Histogram **[批次]** | 每批 DUMP 从出队到响应提交完成的总耗时。测整批写入的服务端端到端时延（用户明确要求的"整批数据总耗时"）；与 `dump_prepare_duration_ms` 的差值 = 数据传输等待 + Poller 结算 + flag 等待；分位数不可由三段速率均值推导，故直测 | `SubmitResponse()` 入口批次统计点（§4.7），`now_us − record.begin_us`，条件 `opcode == Dump` |
| `drampool_load_batch_total_duration_ms` | Histogram **[批次]** | 同上，LOAD 侧。测整批读取的服务端端到端时延；差值归因同上 | 同上，条件 `opcode == Load` |
| `drampool_lookup_batch_total_duration_ms` | Histogram **[批次]** | 同上，LOOKUP 侧（无数据传输，总耗时 ≈ scan + Poller 调度 + flag 缓冲等待）。与 `lookup_scan_duration_ms` 的差值 = 轮询调度延迟与 flag 等待（后者伴随 `response_buffer_retry` 增长） | 同上，条件 `opcode == Lookup` |

> 注：接入层拒绝（参数非法 / opcode 非法 / 响应超 flag slot 上限的配置拒绝）不产生 CompletionRecord，不进入批次统计——与"第二类非业务指标不涉及"的口径一致（该路径有 UC_ERROR 日志兜底）。batch_size 为 0 的请求仍观测批次总耗时（流程完整）。

---

## 4. 关键计算逻辑与观测机制

### 4.1 ScopedTimer 与 Disarm 约定

- 所有耗时观测用 RAII `ScopedTimer`（构造记起点、析构观测），零逻辑侵入；内部以 `SteadyNowUs()` µs 计时、以 ms（double）入直方图。
- **仅成功路径观测**的指标（dump/load prepare）：失败 / 短路 / 无传输分支显式 `Disarm()`。prepare 口径 = "数据传输提交成功"（与 `submit_ms` 写入点对齐）；配置拒绝、全 entry 跳过 / 未命中、提交失败的请求不入 prepare 分布（耗时结构完全不同，混入会污染分位数）。

### 4.2 GC 占空比推导

```promql
# GC 线程占空比 = rate(evict_gc_sum)/rate(evict_gc_count) / gcIntervalMs
```

> 四次修改后 `dump/load_bytes_total` 指标已删除，吞吐/带宽推导随之取消（§10 留痕）；传输侧观测由 E 组 `transfer_duration_ms` 直测承载。

### 4.3 flag 池记账（flag_pool_usage_ratio）

CompletionPoller 单线程访问 flag 池，用普通 `size_t` 记账（无需 atomic）：`SubmitResponse()` Allocate 成功 +1（L164）、`ReleaseResponseBuffer()` Free 成功 −1（L38）；`PollPendingCompletions()` 每轮尾部 `MetricsSet(used / slotCount)`。与 `response_buffer_retry_total`（B2 信号）构成"水位前兆 + 事件确认"组合。

### 4.4 命中率推导

```promql
# LOOKUP 命中率：五次修改后 hit 指标已删除（miss 保留，batch_size − hit 数内部累计），
#   查询命中/总量暂无直测——以 miss 绝对速率与突增监控为主（同 LOAD）
# LOAD 命中率（entry 维度）：精确式 = 1 − miss/(miss + 成功加引 entry 数)；
#   四次修改后 bytes 指标已删除，成功 entry 数暂无直测——以 miss 绝对速率与突增监控为主
```

> 三次修改后批次命中率 Gauge 已删除；五次修改后 LOOKUP hit 指标亦删除——两侧命中率均暂不可由指标推导，以 miss 绝对速率与突增监控为主（§10 留痕）。

### 4.5 阶段层级与归因（B/C/D + G + I 组）

```
DUMP 请求（I 组：dump_batch_total_duration_ms，批次端到端）
├─ dump_prepare_duration_ms [批次]（本地准备总视角：逐 entry StoreBegin + 分配 + 提交）
├─ dump_transfer_duration_ms [批次]（异步数据传输，poller 结算）
│  └─ Σ metadata_storeend_duration_ms [entry]（INITIALIZED→READY）
└─ flag 缓冲等待 + Poller 调度（B2 信号：response_buffer_retry）
LOAD 请求
├─ load_prepare_duration_ms [批次]（逐 entry LoadBegin + 长度校验 + 提交）
└─ load_transfer_duration_ms [批次] → Σ metadata_loadend_duration_ms [entry]
LOOKUP 请求（无数据传输）
├─ lookup_scan_duration_ms [批次]（主体耗时，单 key 成本 ≈ scan 均值 ÷ 平均 batch_size）
└─ lookup_batch_total_duration_ms [批次]（端到端）
后台 GC（每 gcIntervalMs 一轮）
└─ metadata_evict_gc_duration_ms [轮]
```

归因说明：批次内的元数据阶段细节（分配 / 注册 / 驱逐）不单独设指标——prepare（[批次] 整批）与 I 组批次总耗时的差值承载"传输等待 + Poller 结算 + flag 等待"归因；元数据侧异常通过 `nospace_failures`（内存压力直测）与 `usage_ratio` 水位交叉定位。

| 症状组合 | 结论 | 动作方向 |
|---|---|---|
| nospace_failures 增长 + usage_ratio 逼近 1 | 内存压力触发驱逐重试 | 扩容 / 调整驱逐比例 / 排查突发写入 |
| prepare P99 高而 transfer/批次总耗时正常 | 服务端 CPU 侧开销（metadata 操作 / 缓冲分配） | 结合 nospace_failures 与 GC 占空比定位 |
| storeend / loadend 抬高，与 evict_gc 同步 | 后台驱逐读锁与业务锁互扰 | 调大 gcIntervalMs 或降驱逐比例 |
| 批次总耗时 P99 高而 prepare/scan 正常 | 传输终态等待 / flag 缓冲等待 / Poller 调度延迟 | 查 `transfer_duration_ms` P99 与 `response_buffer_retry`、`flag_pool_usage_ratio` |

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
| B3 | TaskWorker：completionQueue 满 | Push 自旋停摆 → requestQueue 堆积反压 B1 | `completion_full_total` ↑ → `request_full_total` 连锁 ↑（前置信号：`completion_size` / `request_size` 持续增长——六次修改后直测，原恒等推导取消） |
| B2 | Poller：flag 池 NoSpace | 单批次留 pending 重试（不阻塞线程） | `response_buffer_retry_total` ↑ + `completion_inflight` 贴近 64 + `flag_pool_usage_ratio` 逼近 1 |

归因速查：`request_size` 持续增长而 `completion_full` 平 → B1（TaskWorker 消费不足）；`completion_full` ↑ 且 `request_full` 随后连锁 ↑（`completion_size` 先增） → B3（Poller 消费不足，反压传导）；`retry` ↑ + inflight 贴满 + flag 池水位逼近 1 → B2（flag 池扩容）；全为 0 但吞吐低 → 转向 §4.5 阶段归因与 I 组批次总耗时分析。

### 4.7 批次统计点与 `begin_us` 哨兵（I 组 3 个 Hist + 批次 Counter ×1，本次重设计核心）

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
        if (record.opcode == Dump) {             // B 组 failed_entries（三次修改后：批次 Gauge ×2 改 Counter）
            const auto failedN = std::count(record.results.begin(), record.results.end(),
                                            static_cast<std::uint8_t>(DumpLoadResult::Failed));
            MetricsCount(kDumpFailedEntries, static_cast<std::uint64_t>(failedN));
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

启动路径（`drampool_launch_config.cc` 初始化序列，`Metrics::SetUp(maxVectorLen)` 之后）按 §3.2 A~I 组常量表逐一 `CreateStats`：31 个静态名 + 数据池按 `poolBlockSizes` 动态注册 `drampool_buffer_pool_usage_ratio_<slot_size>`（slot size 用原始字节数入名）。

### 5.2 数据结构与签名变更清单（最小化）

| 变更 | 位置 | 内容 |
|---|---|---|
| 数据结构 +1 字段（唯一） | `drampool_types.h` `CompletionRecord` | `std::uint64_t begin_us{0}`（§4.7） |
| 函数签名 +1 参数 ×4 | `task_worker.h/.cc` | `ProcessOneRequest` 内取 `beginUs`，向 `ProcessDump` / `ProcessLoad` / `ProcessLookup` / `QueueResponse` 传递并写入 `record.begin_us`（3 个直接构造点 + QueueResponse 构造点） |
| 微改造 ×1 | `task_worker.cc` `SubmitCompletion()` | Push 前 TryPush 探测（`completion_full_total` 计数，探测不改行为） |
| 新增记账 ×2 | `BufferManager`（原子 used 计数，`GetUsedSlotRatio()`）、`completion_poller.cc`（flag 池 used size_t） | F 组水位 |
| 新增时钟 ×1 | `drampool_types.h` | `SteadyNowUs()`（µs 精度，与 `SteadyNowMs()` 同型） |

共享组件（`ucm/shared/pool`、`template/spsc_ring_queue.h` 等）零改动；仅新增 `ucm/shared/metrics` 链接依赖。

### 5.3 线程 × 指标归属矩阵（C5：首次 UpdateStats 的线程自动注册，无需手动登记线程）

| 线程 | 更新的指标 |
|---|---|
| RequestReceiver | H 组：request_full / request_enqueue_wait_ms / request_size（TryPush 成功后覆盖写） |
| TaskWorker（×N） | A 组 ×3；C 组 miss；D 组 miss；B/C 组 prepare；E 组 submit_failures；H 组 request_size（TryPop 后覆盖写）/ completion_full / completion_size（Push 后覆盖写） |
| CompletionPoller | B 组 failed_entries_total；I 组 ×3；E 组 transfer ×2 / transfer_failures / response_rtt / response_failures；G 组 storeend / loadend；F 组 flag_pool_usage_ratio；H 组 completion_size（TryPop 后覆盖写）/ completion_inflight / response_buffer_retry |
| GC 线程 | F 组 metadata_entry_count / buffer_pool_usage_ratio_*；G 组 evict_gc |

### 5.4 直方图桶集

13 个 Histogram 共用一组桶（`metricsDurationBucketsMs`）：亚毫秒段 `0.001–0.5`（覆盖 G 组 µs 级观测与 LOOKUP 批次）+ 毫秒段 `1–10000`（请求 / 传输 / 批次级）；向量长度（C3 上限）按"批次数 × 采样周期"配置，默认容纳 ≥2 个 Reporter 周期的批次量。

---

## 6. 快照输出（Reporter 线程）

- Reporter 每 10s 调用 `GetAllStatsAndClear()`（C4 读后即清），将增量**累加**为进程级累计值后序列化为一行 JSON（JSON Lines，轮转 + 清理）：`ts` + counter 表（累计值）+ gauge 表（最新值）+ histogram 表（count / sum，必要时含分位数）。
- **Gauge 的快照语义**：三次修改后批次实况 Gauge 已删除，保留的 Gauge 均为**池水位瞬时值**（F 组水位 ×3、H 组 completion_inflight）——每行 JSON 反映窗口内最后一次覆盖写的最新值，与"当前水位"观测意图一致；批次失败已由 Counter `dump_failed_entries_total` 累计承载，窗口间可加、无快照歧义。
- Histogram 增量在窗口间累加（count/sum 可加），分位数由回流器按桶重建或以 sum/count 均值近似。

## 7. UCM/Scheduler 侧回流器

与既有设计一致：Python 回流器运行于推理进程内，**Leader 选举（flock）**后尾部读取 DramPool JSON Lines 最新完整行（64KiB 窗口），Counter / Histogram 做周期 **delta 换算**（累计 → 增量速率），Gauge 直取（池水位最新值），统一加 `ucm:` 前缀注册进 ucmmetrics → vllm_connector → Prometheus。两侧时钟不要求同步。

## 8. 测试方案

| # | 用例 | 断言 |
|---|---|---|
| 1 | 注册完备性 | 遍历常量指标名表逐一 `UpdateStats` 后 `GetAllStatsAndClear` 可见（C2 反向：未注册名丢弃） |
| 2 | 批次哨兵一次性 | flag 池 NoSpace 强制重试场景：`failed_entries_total` / I 组 Hist 仅 +1 样本；响应链路永久失败路径同样已上报 |
| 3 | NoSpace 直测 | 构造数据池打满：`dump_nospace_failures_total` 增量 == 两次驱逐重试后仍 NoSpace 的 entry 数（注册失败 / DuplicateKey 不计入） |
| 4 | C3 溢出 | 观测数超 maxVectorLen 后先到样本保留、后续丢弃不崩溃 |
| 5 | 端到端 | 单批 DUMP→LOAD→LOOKUP 全链路：批次总耗时 > prepare/scan，`failed_entries_total` 与注入失败 entry 数一致（含失败短路标记的剩余 entry） |

## 9. 实施步骤（建议 PR 划分）

1. **PR-1 基础设施**：`SteadyNowUs()` + 启动期注册表 + Reporter/回流器骨架；A 组 + H 组 + F 组先行（无数据结构变更）。
2. **PR-2 批次观测**：`begin_us` 字段 + 签名传递 + `SubmitResponse` 统计点；I 组 ×3 + `dump_failed_entries_total` + 测试 2/5。
3. **PR-3 metadata 层**：G 组 ScopedTimer（storeend / loadend / evict_gc）+ `nospace_failures` 直测 + 测试 3。
4. **PR-4 收尾**：E/D 组埋点、`submit_completion` TryPush 探测微改造、lint（black/isort 仅 Python 侧）与全量测试。

## 10. 取舍留痕（裁剪 / 变更项与恢复方式）

| 类别 | 项 | 恢复方式 |
|---|---|---|
| 推导类（首轮裁剪，维持） | dump/load_bandwidth_gbps、batch_entries、dump_ttl_ms、interval_lookup_hit_rate | 带宽/批量推导已随四次修改取消（§3.1 注）；命中率推导已随五次修改取消（§4.4 注） |
| 非业务类（16 项，维持不涉及） | 接入层协议错误、内部调度、低频异常分类、传输超时、GC 诊断等 | UC_WARN/UC_ERROR 日志兜底 |
| 失败类合并（维持） | submit / transfer / response_failures 三大类 | 需按 opcode 细分时在同一统计点按 opcode 拆名即可 |
| **本次删除** | ~~`drampool_dump_storebegin_failures_total`~~ | 在 `task_worker.cc` `ProcessDump()` `storeStatus.Failure()` 分支（L134-141）恢复 `MetricsCount`；若需"分配失败 vs 注册失败"细分，在 metadata 层 StoreBegin 返回前按 Status 类别分别计数 |
| **二次删除（本修改）** | 6 个 [首条] 直方图：D 组 `lookup_first_exist_duration_ms` + G 组 storebegin / allocate / shard_register / evict_sync / loadbegin；thread_local 首条门控（`tl_batch_first_pending`，原 §4.7.1）随之整体移除 | 单 key 成本回退"scan 均值 ÷ 平均 batch_size"推导（§3.2 D 组注）；批次内阶段归因由 B/C 组 prepare（[批次]）与 I 组批次总耗时承载；entry 级阶段归因需恢复时，在 metadata.cc 对应观测点（StoreBegin / Allocate / ShardRegister / EvictSync / LoadBegin）以无条件 `ScopedTimer` 直测（无需门控） |
| **三次修改（本修改）** | 批次 Gauge ×4：`dump_batch_failed_entries` / `dump_batch_failure_ratio` / `lookup_batch_hits` / `lookup_batch_hit_ratio` 停用——`dump_failed_entries` 改造为 Counter `dump_failed_entries_total`（§4.7 同一统计点 `MetricsSet`→`MetricsCount`）；`lookup_batch_hits` 与 `lookup_hit_entries_total` 同点重复、两个比率 Gauge 可由 §4.4 公式推导且比率语义非 Counter 所能承载，均直接删除 | 恢复覆盖写实况：在 `SubmitResponse()` 哨兵块内（`record.results` 定稿处）重新 `MetricsSet` 对应 Gauge 名即可，统计点与哨兵机制原样保留 |
| **四次修改（本修改）** | 吞吐字节 Counter ×2：`dump_bytes_total` / `load_bytes_total` 删除——吞吐/带宽推导（原 §4.2）随之取消，LOAD 命中率精确式暂不可算（§4.4 注），"平均批量"近似（requests + bytes）随之取消；传输侧观测由 E 组 `transfer_duration_ms` 直测承载 | 恢复方式：在 `task_worker.cc` `ProcessDump()` entries 循环内 StoreBegin 成功处 / `ProcessLoad()` entries 循环内 LoadBegin 成功处累加 Σ `entry.len`，循环外一次 `MetricsCount`（§3.2 B/C 组原行） |
| **五次修改（本修改）** | 归因/分子 Counter ×2：`load_initialized_entries_total`（miss 的 INITIALIZED 子类归因，写读竞态窗口观测）与 `lookup_hit_entries_total`（LOOKUP 命中数，命中率分子）删除——LOAD 命中率精确式维持不可算（四次修改起），LOOKUP 命中率推导随之取消（§4.4 注），单 key 查询成本分母（hit + miss）暂不可由指标表达（§3.2 D 组注）；两侧均以 miss 绝对速率与突增监控为主 | 恢复方式：`load_initialized_entries_total` 在 `metadata.cc` `ShardMetadata::LoadBegin()` `TryIncRef` 失败分支 + `existingEntry->status == INITIALIZED` 判定处恢复 `MetricsCount`（§3.2 C 组原行）；`lookup_hit_entries_total` 在 `task_worker.cc` `ProcessLookup()` 扫描循环内恢复 hits 累计上报、循环外一次 `MetricsCount`（§3.2 D 组原行） |
| **六次修改（本修改）** | 队列累计 Counter ×4：`queue_request_enqueued/dequeued_total` 与 `queue_completion_enqueued/dequeued_total` 删除，替代为当前长度直测 Gauge ×2：`queue_request_size`（TryPush / TryPop 成功后覆盖写）与 `queue_completion_size`（Push / TryPop 成功后覆盖写）——SPSC 无丢弃恒等式（排队数 = 入队 − 出队，原 §4.6）改为直测，用户要求"只需统计当前队列中有多少"；入口/消费速率观测随之取消（如需恢复见恢复方式） | 恢复方式：原 4 个 Counter 统计点与本修改 Gauge 埋点完全同位（`drampool_server.cc` L517 / `task_worker.cc` L49、L326 / `completion_poller.cc` L71-72），将 `MetricsSet` 改回 `MetricsCount` 即恢复；排队数推导退化为速率差（enqueued − dequeued） |
| 维持 entry / 轮级 | storeend / loadend / evict_gc | 无变更 |

