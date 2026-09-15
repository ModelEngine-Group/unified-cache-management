# DramPool Metrics 指标速查手册（47 个）

> 权威规格（埋点实现细节、回流链路、配置、测试方案）见同目录 [metrics_design.md](metrics_design.md)；本文件为开发/运维速查摘要。
>
> 功能描述统一三段式：**测什么**（观测口径）→ **反映什么**（异常时指向的问题）→ **怎么用**（典型分析动作）。
>
> 指标规模：**47 个** = 19 Histogram（全部 `_ms` 后缀，µs 精度计时）+ 20 Counter（`_total` 后缀）+ 8 类 Gauge（1 类动态注册）。
>
> 观测粒度标注：**[首条]** = 仅批次首条 entry 观测（样本量 = 批次数）；**[批次]** = 每批次一次；**[entry]** = 每 entry 一次；**[轮]** = GC 每轮一次；**[覆盖]** = Gauge 最新值覆盖写（快照反映窗口内最近一批实况）。
>
> 本次重设计主线（批次观测，design §3.1 第五类）：
> - **DUMP**：新增批次失败实况 Gauge ×2（`batch_failed_entries` / `batch_failure_ratio`，覆盖写）与 NoSpace 直测（`nospace_failures_total`——两次驱逐重试后仍 NoSpace 才计）；**删除** `storebegin_failures_total`（NoSpace 直测精确归因 + 批次 Gauge 覆盖其业务价值）
> - **LOAD**：miss 口径明确 = **LoadBegin 失败 + 请求长度大于存储长度**两类之和（登记本 miss）；新增 INITIALIZED 子类归因（`load_initialized_entries_total`——写读竞态窗口观测）
> - **LOOKUP**：新增批次命中 Gauge ×2（`batch_hits` / `batch_hit_ratio`）+ 首条 Exist 直测（`first_exist_duration_ms`）
> - **整批总耗时**：新增 I 组 3 个 Histogram（TaskWorker 出队 → 响应提交完成，跨线程经 `CompletionRecord.begin_us`（µs，兼一次性上报哨兵）传递）
> - **G 组 5 个 [首条] 化**：storebegin / allocate / shard_register / evict_sync / loadbegin 从 entry 级改为仅批次首条观测（thread_local 门控，样本量 = 批次数，C3 溢出进一步缓解；storeend / loadend 保持 [entry]、evict_gc 保持 [轮]）

---

## 1. 指标总览

| 组 | 主题 | 数量 | Histogram | Counter | Gauge |
|---|---|---|---|---|---|
| A | 请求量 | 3 | — | 3 | — |
| B | DUMP 业务 | 5 | 1 | 2 | 2 |
| C | LOAD 业务 | 4 | 1 | 3 | — |
| D | LOOKUP 业务 | 6 | 2 | 2 | 2 |
| E | 传输与响应 | 6 | 3 | 3 | — |
| F | 资源水位 | 3 | — | — | 3 |
| G | Metadata 阶段耗时（[首条]×5 + [entry]×2 + [轮]×1） | 8 | 8 | — | — |
| H | 队列与阻塞 | 9 | 1 | 7 | 1 |
| I | 批次总耗时（本次新增） | 3 | 3 | — | — |
| **合计** | | **47** | **19** | **20** | **8** |

---

## 2. 分组明细

### A. 请求量（3）

| 指标名 | 类型 | 测什么 | 反映什么 | 怎么用 |
|---|---|---|---|---|
| `drampool_dump_requests_total` | Counter | 累计接收并处理的 DUMP 请求数（每请求 +1，与 batch 内 entry 数无关） | 写入负载规模 | 负载构成分析；与 `dump_bytes_total` 对比得平均批量；写入侧 rate 类指标的分母 |
| `drampool_load_requests_total` | Counter | 累计接收并处理的 LOAD 请求数 | 读取负载规模 | 读取侧所有 rate 类指标（miss、transfer 时长）的分母基准 |
| `drampool_lookup_requests_total` | Counter | 累计接收并处理的 LOOKUP 请求数 | 查询负载规模 | 换算"单请求查询成本"的分母 |

### B. DUMP 业务（5）

| 指标名 | 类型 | 测什么 | 反映什么 | 怎么用 |
|---|---|---|---|---|
| `drampool_dump_bytes_total` | Counter | 累计**成功预留缓冲**的 DUMP entry 字节总量（重复写与分配失败不计；传输失败回滚不回退本计数） | DUMP 吞吐规模（"预留成功"口径） | 吞吐趋势；带宽推导（§3）；容量规划 |
| `drampool_dump_nospace_failures_total` | Counter | StoreBegin 中**缓冲分配最终失败且原因为 NoSpace** 的 entry 数——两次驱逐重试后仍 NoSpace 才计，注册失败不计 | 内存压力的**精确归因**（排除注册失败误报） | 持续增长 = 数据池容量不足或驱逐策略失效；与 `buffer_pool_usage_ratio_*` 交叉验证；行动：扩容 / 调整驱逐比例 |
| `drampool_dump_batch_failed_entries` | Gauge **[覆盖]** | **当前批次**（最近完成结算的 DUMP 请求）中结果为 Failed 的 entry 数（含 StoreBegin 失败短路标记与传输失败结算两类） | 最近一批写入失败实况 | 失败持续 >0 时结合 `nospace_failures`（元数据侧）与 `transfer_failures`（传输侧）区分根因 |
| `drampool_dump_batch_failure_ratio` | Gauge **[覆盖]** | 同一统计点的 failed_entries / batch_size（0~1） | 最近一批 DUMP 失败强度 | 告警阈值直接作用于本指标；持续 >0 = 写入链路异常 |
| `drampool_dump_prepare_duration_ms` | Histogram **[批次]** | DUMP 入口 → 传输提交完成的**本地准备耗时**（逐 entry StoreBegin + 缓冲分配 + 驱逐重试 + 提交，**不含**数据传输；仅成功路径观测） | 写入路径服务端 CPU 侧开销 | P99 高时用 G 组按阶段归因（§4）；与 I 组批次总耗时的差值 = 传输等待 + Poller 结算 + flag 等待 |

> 注：重复 key entry 幂等 `continue`，不进入失败口径；批次失败 Gauge 统计 `record.results` 定稿值——失败短路标记（`mark_remaining_failed`）的剩余 entry 一并计入，反映批次最终实况。

### C. LOAD 业务（4）

| 指标名 | 类型 | 测什么 | 反映什么 | 怎么用 |
|---|---|---|---|---|
| `drampool_load_bytes_total` | Counter | 累计**成功加引用**的 LOAD entry 字节总量（未命中 / 长度不匹配不计） | LOAD 吞吐规模 | 读取吞吐趋势与读取带宽推导 |
| `drampool_load_miss_entries_total` | Counter | LOAD 中**未取到数据**的 entry 数——**登记本 miss**：LoadBegin 失败（key 不存在或状态非 READY）与**请求长度大于存储长度**两类之和，对客户端都是"没取到" | 读取未命中规模 | 命中率 = 1 − miss/(miss + 成功 entry)；突增 = 读取了被驱逐 / 过期 / 未写完的数据 |
| `drampool_load_initialized_entries_total` | Counter | LoadBegin 时目标 entry 处于 **INITIALIZED**（DUMP 已分配缓冲、传输未完成 / 未 StoreEnd）而被拒的 entry 数——miss 的**子类归因**（NotFound 路径不计） | **写读并发竞态窗口**：读取侧撞上写入在途 | 增长说明客户端在 DUMP 完成前发起读取（时序问题）而非数据丢失 |
| `drampool_load_prepare_duration_ms` | Histogram **[批次]** | LOAD 入口 → 传输提交完成的**本地准备耗时**（LoadBegin + 长度校验 + 提交，不含传输；仅成功路径观测） | 读取路径服务端 CPU 侧开销 | 与 G 组 `loadbegin_duration_ms` 联动归因 |

### D. LOOKUP 业务（6）

| 指标名 | 类型 | 测什么 | 反映什么 | 怎么用 |
|---|---|---|---|---|
| `drampool_lookup_hit_entries_total` | Counter | LOOKUP 中**存在且 READY**（命中并刷新 TTL lease）的 entry 数 | 查询命中规模 | 累计命中率的分子（§3 口径一） |
| `drampool_lookup_miss_entries_total` | Counter | LOOKUP 中**不存在或非 READY** 的 entry 数（batch_size − hits） | 查询未命中规模 | 突增 = 查询了被驱逐 / 过期的前缀，通常先于 LOAD miss 出现（驱逐过快预警） |
| `drampool_lookup_batch_hits` | Gauge **[覆盖]** | **当前批次**（最近完成响应提交的 LOOKUP 请求）中命中的 entry 数 | 最近一批查询命中实况 | 突降时结合 `miss_entries_total` 速率区分"负载前缀变化"与"数据异常丢失" |
| `drampool_lookup_batch_hit_ratio` | Gauge **[覆盖]** | 同一统计点的 hits / batch_size（0~1；batch_size 为 0 置 0） | 最近一批命中率 | 告警阈值直接作用于本指标；持续低于累计命中率 = 最近负载命中恶化（容量不足 / 驱逐过快前兆） |
| `drampool_lookup_first_exist_duration_ms` | Histogram **[首条]** | 每批次扫描循环**首个 entry** 的 `Exist` 单次耗时（key 定位 + 分片读锁 + TryMarkHit lease 刷新） | 单 key 查询成本的**直测**（替代 scan 均值间接推导） | P99 高 = 读锁竞争或 lease 刷新写开销 |
| `drampool_lookup_scan_duration_ms` | Histogram **[批次]** | 每 batch 一次：LOOKUP 元数据扫描总耗时（Σ `Exist` + 结果填充，不含响应写回） | LOOKUP 服务端主体处理时长（LOOKUP 无数据传输） | 均值 ÷ 平均 batch_size ≈ 单 key 查询成本（与首条直测互为印证） |

> 注：LOOKUP 无 metadata 拆分阶段与数据传输——**首条 Exist 耗时即"首条数据的总时长"**，scan 即批次主体耗时；整批总耗时见 I 组 `lookup_batch_total_duration_ms`。

### E. 传输与响应（6）

| 指标名 | 类型 | 测什么 | 反映什么 | 怎么用 |
|---|---|---|---|---|
| `drampool_dump_transfer_duration_ms` | Histogram **[批次]** | DUMP 传输从提交（`submit_ms`）到**终态**（Completed / Failed / GetStatus 异常）的异步耗时，涵盖网络与对端读取 | 实际搬运能力 | `rate(bytes)/均值` = 传输带宽（§3）；与 prepare 相加 ≈ DUMP 服务端处理主体 |
| `drampool_load_transfer_duration_ms` | Histogram **[批次]** | 同上，LOAD 侧（池 → 客户端方向搬运时长） | 同上 | 同上 |
| `drampool_transfer_failures_total` | Counter | 数据传输以**非 Completed 终态**结束的请求数（DUMP/LOAD 合并大类；Failed 或 GetStatus 异常均计） | 传输失败强度 | 增长需排查对端网络 / 连接状态（与 opcode 无关，低频不细分） |
| `drampool_response_rtt_ms` | Histogram **[批次]** | 响应从本地提交（响应写回传输发起）到**写回客户端内存完成**的端到端耗时（含响应传输本身） | 客户端感知的"结果返回"时延 | P99 高 = 对端写入慢；flag 池等待由 I 组批次总耗时覆盖，不在此重复计入 |
| `drampool_response_failures_total` | Counter | 响应返回链路失败计数（**本地提交** Allocate 非 NoSpace 失败 / Pack / ExecuteAsync 失败 + **写回传输**失败，DUMP/LOAD 合并大类；flag 池 NoSpace 属重试不算失败） | 结果返回通道健康度 | 增长锁定响应链路故障域，结合 WARN 日志定位提交 / 写回哪一环 |
| `drampool_submit_failures_total` | Counter | 数据传输**提交失败**的请求数（DUMP/LOAD 合并大类；ExecuteAsync 失败或 handle 无效；LOAD 侧整批已 LoadEnd 释放引用） | 传输子系统提交路径健康度 | 增长指向传输子系统初始化 / 资源异常，客户端整批失败 |

### F. 资源水位（3，数据池由 GCThreadLoop 每 `gcIntervalMs` 一轮采样、flag 池由 CompletionPoller 调用点记账，均无热路径开销）

| 指标名 | 类型 | 测什么 | 反映什么 | 怎么用 |
|---|---|---|---|---|
| `drampool_metadata_entry_count` | Gauge **[覆盖]** | 当前池内缓存条目（block）总数（1024 分片求和） | 元数据规模容量水位 | 增长斜率反映写入 / 驱逐平衡；配合 usage_ratio 判断驱逐压力区 |
| `drampool_buffer_pool_usage_ratio_<slot_size>` | Gauge **[覆盖]**（动态注册，每 block size 一个） | 各 block 尺寸数据池**已用槽位占比**（used / slot count） | 分尺寸内存水位 | 逼近 1 = 该尺寸池即将触发驱逐重试（DUMP prepare 抖动与 `nospace_failures` 的前兆），容量规划第一信号 |
| `drampool_flag_pool_usage_ratio` | Gauge **[覆盖]** | flag 响应缓冲池**已用槽位占比**（used / flagBufferSlotCount） | 响应回填缓冲水位；逼近 1 = B2 链（flag 池 NoSpace → `response_buffer_retry`）前兆信号 | 为"flag 池扩容"决策提供交叉验证（此前无水位指标可查） |

### G. Metadata 阶段耗时（8 = [首条]×5 + [entry]×2 + [轮]×1）

> 观测点位于 `metadata.cc` 内部（模块边界测量），`ScopedTimer` RAII 零逻辑侵入；时钟 µs 精度（`SteadyNowUs()`）、以 ms（double）入直方图。storebegin / allocate / shard_register / evict_sync / loadbegin 5 个**仅批次首条观测**（thread_local 门控，design §4.7.1）；恢复 entry 级 = 去掉门控条件。

| 指标名 | 类型 | 测什么 | 反映什么 | 怎么用 |
|---|---|---|---|---|
| `drampool_metadata_storebegin_duration_ms` | Histogram **[首条]** | 批次首条 `MetadataManager::StoreBegin` 调用的**总耗时** = 缓冲分配（含驱逐重试）+ 分片注册（覆盖成功 / 失败 / DuplicateKey 全路径） | DUMP 写路径元数据开销总视角（首条口径） | P99 抬高时按 allocate / register 两阶段直方图归因（§4） |
| `drampool_metadata_allocate_duration_ms` | Histogram **[首条]** | 首条 StoreBegin 内**缓冲分配阶段**耗时（首次 Allocate → NoSpace 周期驱逐重试 → 再 NoSpace 深度驱逐重试，含驱逐耗时，与 evict_sync 嵌套） | "拿到一块缓冲"的真实成本 | 高且 evict_sync 占比高 = 内存压力；高而 evict 占比低 = 共享 BufferPool Allocate 本身慢 |
| `drampool_metadata_shard_register_duration_ms` | Histogram **[首条]** | 首条 StoreBegin 内**分片注册阶段**耗时（分片写锁 + 重复检查 + 双驱逐策略 AddKey + map emplace，失败含回滚 Free） | 元数据注册成本 | 高 = 分片写锁竞争或 LRU / TTL 链表操作开销大 |
| `drampool_metadata_evict_sync_duration_ms` | Histogram **[首条]** | 首条 StoreBegin 触发的**业务线程内同步驱逐**（周期 / 深度两级）单次耗时 | 驱逐对 DUMP 的**直接拖慢**（TaskWorker 线程内执行，阻塞当前批次后续 entry） | 增长 = 内存压力直接信号，与 `usage_ratio` 交叉验证 |
| `drampool_metadata_storeend_duration_ms` | Histogram **[entry]** | 每次 `StoreEnd`（DUMP 传输 Completed 后 entry INITIALIZED→READY）耗时，CompletionPoller 线程执行 | DUMP 终态结算的单 entry 成本（跨批次逐 entry 结算，无批次首条概念） | 高 = 分片读锁竞争（GC 每秒全分片扫描持读锁） |
| `drampool_metadata_loadbegin_duration_ms` | Histogram **[首条]** | 批次首条 `LoadBegin`（查 key + TryIncRef 加引用 + 双驱逐策略 AccessKey）耗时 | LOAD 读路径元数据开销（首条口径） | 高 = 读锁竞争，或策略 AccessKey（LRU move-to-front / TTL 更新）开销大 |
| `drampool_metadata_loadend_duration_ms` | Histogram **[entry]** | 每次 `LoadEnd`（TryDecRef 减引用）耗时（poller 终态结算 / len 不匹配回滚 / 提交失败回滚三类路径） | 引用释放成本（单次应近常数） | 异常升高 = 锁竞争 |
| `drampool_metadata_evict_gc_duration_ms` | Histogram **[轮]** | 后台 GC 每轮全分片驱逐扫描总耗时（`PerformEvict` 整轮 = 1024 shard 之和，每 `gcIntervalMs`（默认 1s）一轮） | GC 对系统的持续开销 | 均值 ÷ `gcIntervalMs` = GC 线程占空比；高 = 频繁持锁干扰业务（storeend / loadbegin 变慢的常见外因） |

### H. 队列与阻塞（9）

> 两条 SPSC 队列与响应缓冲是请求端到端时延的关键路径（S0），排队 / 阻塞直接等价于客户端可感知延迟。线程衔接模型与阻塞链归因见 §4.2。

| 指标名 | 类型 | 测什么 | 反映什么 | 怎么用 |
|---|---|---|---|---|
| `drampool_queue_request_enqueued_total` | Counter | 累计**成功进入** requestQueue 的请求数（TryPush 成功 +1） | 接收侧入口流量（"已入队待处理"口径，与 A 组"已开始处理"相差排队数） | 与 `dequeued` 差值 = requestQueue 当前排队数（§3 恒等推导） |
| `drampool_queue_request_dequeued_total` | Counter | 累计 TaskWorker 从 requestQueue **取出**的请求数（TryPop 成功 +1） | TaskWorker 消费速率 | SPSC 无丢弃恒等式：排队数 = 入队 − 出队 |
| `drampool_queue_request_full_total` | Counter | requestQueue **满、TryPush 失败**事件数（每次失败 +1 按次计） | 接收线程被阻塞强度（阻塞时长 ≈ full 数 × `requestReceiverIdleWaitUs` 100µs）；阻塞链 B1 | 增长 = TaskWorker 消费跟不上到达速率或下游反压传导；高概率压力场景的核心告警信号 |
| `drampool_queue_request_enqueue_wait_ms` | Histogram **[批次]** | 每请求从准备入队到 TryPush 成功的**入队前等待时长**（含满重试 sleep，未排队 ≈ 0） | 客户端可感知的接收背压延迟 | P99 抬高必伴随 `full_total` 增长；分位数估计单请求接收延迟 |
| `drampool_queue_completion_enqueued_total` | Counter | 累计**成功进入** completionQueue 的完成记录数（每请求一条，Push 自旋至成功恒 +1） | 完成流生产速率 | 与 `dequeued` 差值 = completionQueue 当前排队数 |
| `drampool_queue_completion_dequeued_total` | Counter | 累计 CompletionPoller 从 completionQueue **取出**的完成记录数（FillPendingWindow 拉取成功 +1） | 完成流消费速率 | 与 `enqueued` 组成恒等式（排队数 = 入队 − 出队） |
| `drampool_queue_completion_full_total` | Counter | completionQueue **满、SubmitCompletion 被迫自旋等待**事件数（Push 前 TryPush 探测，失败 +1 后退回 Push——探测不改行为） | TaskWorker **停摆**位置与强度（既不取新请求也不响应停止指令）；阻塞链 B3，**停摆无日志兜底，此指标是唯一观测手段** | 增长 = Poller 消费能力不足（pending 窗口满）或传输终态 / flag 池重试慢；随后 `request_full` 连锁增长 |
| `drampool_queue_completion_inflight` | Gauge **[覆盖]** | CompletionPoller pending 窗口内在途完成记录数（等终态 / 等响应提交 / 等写回，每轮覆盖写） | 完成链路第二级缓冲占用 | 持续逼近 `pollerPendingDepth`（默认 64）= 拉取停摆；与 `completion_full` 互相印证 |
| `drampool_queue_response_buffer_retry_total` | Counter | 响应 flag 缓冲池 **NoSpace、SubmitResponse 留 pending 下轮重试**事件数 | 响应缓冲供给不足（不阻塞 Poller 线程，但阻塞该请求响应提交、推高 I 组批次总耗时）；缓冲链 B2 | 增长 = 响应突发超 slot 供给或写回慢未释放槽位；与 `flag_pool_usage_ratio`、I 组批次总耗时联动 |

### I. 批次总耗时（3，本次新增）

> **口径**：请求从 TaskWorker **出队**（开始处理）到 CompletionPoller **响应提交完成**（`SubmitResponse()` 内 Pack 成功、响应写回传输已发起）的服务端端到端处理总耗时。**包含**：全部本地处理（prepare / scan）+ 数据传输终态等待 + Poller 结算 + flag 缓冲等待（NoSpace 重试期间计入）+ 响应提交失败前的全部耗时；**不含**：requestQueue 排队等待（H 组观测）与响应写回传输本身（`response_rtt_ms` 单独观测）。三者关系：批次总耗时 + response_rtt ≈ 客户端感知总时延（再加排队等待）。观测点与批次 Gauge 同点同频（每批次一次，design §4.7.2）；跨线程起点经 `CompletionRecord.begin_us`（µs，兼一次性上报哨兵）传递。

| 指标名 | 类型 | 测什么 | 反映什么 | 怎么用 |
|---|---|---|---|---|
| `drampool_dump_batch_total_duration_ms` | Histogram **[批次]** | 每批 DUMP 从出队到响应提交完成的总耗时（整批数据总耗时） | 整批写入的服务端端到端时延 | 与 `dump_prepare_duration_ms` 的差值 = 数据传输等待 + Poller 结算 + flag 等待；分位数不可由三段速率均值推导，故直测 |
| `drampool_load_batch_total_duration_ms` | Histogram **[批次]** | 同上，LOAD 侧 | 整批读取的服务端端到端时延 | 差值归因同上 |
| `drampool_lookup_batch_total_duration_ms` | Histogram **[批次]** | 同上，LOOKUP 侧（无数据传输，总耗时 ≈ scan + Poller 调度 + flag 缓冲等待） | 整批查询的服务端端到端时延 | 与 `lookup_scan_duration_ms` 的差值 = 轮询调度延迟与 flag 等待（后者伴随 `response_buffer_retry` 增长） |

> 注：接入层拒绝（参数非法 / opcode 非法 / 响应超 flag slot 上限的配置拒绝）不产生 CompletionRecord，不进入批次统计（该路径有 UC_ERROR 日志兜底）。batch_size 为 0 的请求仍观测批次总耗时（流程完整），比率类置 0。

---

## 3. 关键推导公式（PromQL 速查）

```promql
# LOAD 命中率（entry 维度）：1 - miss/(miss+成功加引 entry)，成功 entry 数由 load_bytes_total/平均entry大小近似
1 - rate(ucm:drampool_load_miss_entries_total[5m])
    / (rate(ucm:drampool_load_bytes_total[5m]) / 平均entry大小 + rate(ucm:drampool_load_miss_entries_total[5m]))

# LOOKUP 命中率——口径一：累计命中率（Counter 精确推导，长周期趋势）
sum(rate(ucm:drampool_lookup_hit_entries_total[5m]))
  / (sum(rate(ucm:drampool_lookup_hit_entries_total[5m])) + sum(rate(ucm:drampool_lookup_miss_entries_total[5m])))
# 口径二：当前批次命中率（Gauge 直取，瞬时实况）
ucm:drampool_lookup_batch_hit_ratio

# 传输带宽 (GB/s)，以 DUMP 为例
rate(ucm:drampool_dump_bytes_total[5m])
  / (rate(ucm:drampool_dump_transfer_duration_ms_sum[5m]) / rate(ucm:drampool_dump_transfer_duration_ms_count[5m]))
  / 1e6

# GC 线程占空比
rate(ucm:drampool_metadata_evict_gc_duration_ms_sum[5m])
  / rate(ucm:drampool_metadata_evict_gc_duration_ms_count[5m]) / (gcIntervalMs秒数)

# LOOKUP 单 key 查询成本：scan 均值法（与首条直测 first_exist_duration_ms 互为印证）
rate(ucm:drampool_lookup_scan_duration_ms_sum[5m])
  / rate(ucm:drampool_lookup_scan_duration_ms_count[5m])
  / (rate(ucm:drampool_lookup_hit_entries_total[5m]) + rate(ucm:drampool_lookup_miss_entries_total[5m]))

# 守恒校验（[首条] 化后：各分量为"同一首条 entry"的分-总关系，偏差 = 胶水开销，正常远小于任一阶段）
rate(ucm:drampool_metadata_storebegin_duration_ms_sum[5m])
  ≈ rate(ucm:drampool_metadata_allocate_duration_ms_sum[5m])
  + rate(ucm:drampool_metadata_shard_register_duration_ms_sum[5m])

# 批次耗时分解（I 组 vs B/D 组 prepare/scan 差值 = 传输终态等待 + Poller 结算 + flag 缓冲等待）
rate(ucm:drampool_dump_batch_total_duration_ms_sum[5m]) / rate(ucm:drampool_dump_batch_total_duration_ms_count[5m])
  - rate(ucm:drampool_dump_prepare_duration_ms_sum[5m]) / rate(ucm:drampool_dump_prepare_duration_ms_count[5m])

# 队列当前排队数（SPSC 无丢弃 → 速率差恒等）
requestQueue 排队数：
  rate(ucm:drampool_queue_request_enqueued_total[5m]) - rate(ucm:drampool_queue_request_dequeued_total[5m])
completionQueue 排队数：
  rate(ucm:drampool_queue_completion_enqueued_total[5m]) - rate(ucm:drampool_queue_completion_dequeued_total[5m])
# 接收阻塞时长估计 ≈ rate(ucm:drampool_queue_request_full_total[1m]) × 100µs
```

---

## 4. 阶段层级与归因速查（B/C/D + G + I 组）

```
DUMP 请求
├─ dump_batch_total_duration_ms [批次]（I 组：批次端到端，出队→响应提交完成）
│  ├─ dump_prepare_duration_ms [批次]（本地准备总视角）
│  │  └─ metadata_storebegin_duration_ms [首条]（首条 entry 元数据总用时）
│  │     ├─ metadata_allocate_duration_ms [首条]（阶段①：缓冲分配）
│  │     │  └─ metadata_evict_sync_duration_ms [首条]（NoSpace 嵌套，同步驱逐）
│  │     └─ metadata_shard_register_duration_ms [首条]（阶段②：分片注册）
│  ├─ dump_transfer_duration_ms [批次]（异步数据传输，poller 结算）
│  │  └─ Σ metadata_storeend_duration_ms [entry]（INITIALIZED→READY）
│  └─ flag 缓冲等待 + Poller 调度（B2 信号：response_buffer_retry）
LOAD 请求
├─ load_batch_total_duration_ms [批次]（I 组）
│  ├─ load_prepare_duration_ms [批次] → metadata_loadbegin_duration_ms [首条]
│  └─ load_transfer_duration_ms [批次] → Σ metadata_loadend_duration_ms [entry]
LOOKUP 请求（无数据传输）
├─ lookup_batch_total_duration_ms [批次]（I 组：端到端）
├─ lookup_first_exist_duration_ms [首条]（单 key 直测）
└─ lookup_scan_duration_ms [批次]（主体耗时）
后台 GC（每 gcIntervalMs 一轮）
└─ metadata_evict_gc_duration_ms [轮]
```

| 症状组合 | 结论 | 动作方向 |
|---|---|---|
| allocate P99 高，evict_sync 占比高 | 内存压力触发驱逐重试 | 扩容 / 调整驱逐比例 / 排查突发写入 |
| allocate P99 高，evict_sync 占比低 | 共享 BufferPool 分配慢 | 排查池锁与伙伴分配实现 |
| register P99 高，GC 占空比高 | GC 扫描持锁干扰写锁 | 调大 gcIntervalMs 或降驱逐比例 |
| register P99 高，GC 占空比低 | 驱逐策略数据结构开销（LRU/TTL 链表） | 深挖策略实现 |
| storeend / loadbegin 抬高，与 evict_gc 同步 | 后台驱逐读锁与业务读锁互扰 | 同上 |
| first_exist P99 高而 scan 均值正常 | 首条撞上冷分片 / lease 刷新写放大 | 排查 TryMarkHit 与 lease 逻辑 |
| 批次总耗时 P99 高而 prepare/scan 正常 | 传输终态等待 / flag 缓冲等待 / Poller 调度延迟 | 查 `transfer_duration_ms` P99 与 `response_buffer_retry`、`flag_pool_usage_ratio` |

> 注意 [首条] 化后 G 组分量为"同一首条 entry"的分-总关系，与 prepare（整批）之间为"首条 vs 整批"的近似对照——整批归因靠 prepare 与 I 组批次总耗时。

### 4.2 队列流水线层级与阻塞链归因（H 组）

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

归因速查：request_full ↑ 而 completion_full 平 → B1（TaskWorker 消费不足）；completion_full ↑ 且 request_full 随后连锁 ↑ → B3（Poller 消费不足，反压传导）；retry ↑ + inflight 贴满 + `flag_pool_usage_ratio` 逼近 1 → B2（flag 池扩容，flag 池为独立 region 与数据池无关）；全为 0 但吞吐低 → 转向 §4 阶段归因与 I 组批次总耗时分析。

---

## 5. 时钟与精度说明

- `drampool_types.h` 新增 `SteadyNowUs()`（µs 精度）：metadata 单次操作为微秒量级，毫秒精度时钟下直方图会退化为 0/1 两值；I 组批次总耗时跨线程起点亦以 µs 记（`CompletionRecord.begin_us`）。
- 所有 Histogram 内部以 µs 计时、以 **ms（double，µs 分辨率）** 入直方图——指标名统一 `_ms` 后缀，Prometheus 侧单位一致。
- 桶集（design §5.4 `metricsDurationBucketsMs`）：亚毫秒段 `0.001–0.5`（覆盖 G 组 µs 级观测与 LOOKUP 批次）+ 毫秒段 `1–10000`（请求 / 传输 / 批次级），19 个 histogram 共用一组。
- `begin_us` 兼作一次性上报哨兵：`SubmitResponse()` 上报后置 0，flag 池 NoSpace 同阶段重入不重复统计（design §4.7.2）。
