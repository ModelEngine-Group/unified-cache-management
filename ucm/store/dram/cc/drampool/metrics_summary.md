# DramPool Metrics 指标速查手册（38 个）

> 权威规格（埋点实现细节、回流链路、配置、测试方案）见同目录 [metrics_design.md](metrics_design.md)；本文件为开发/运维速查摘要。
>
> 功能描述统一三段式：**测什么**（观测口径）→ **反映什么**（异常时指向的问题）→ **怎么用**（典型分析动作）。
>
> 指标规模：**38 个** = 15 Histogram（全部 `_ms` 后缀，µs 精度计时）+ 19 Counter（`_total` 后缀）+ 4 类 Gauge。
>
> 失败类指标按必要性审查（design §3.1 第三类）：保留内存不足归因（`storebegin_failures`）与三条阻塞链信号（`queue_*_full/retry`）4 个独立指标；6 个低频传输/提交/响应失败 Counter 按管线阶段合并为 3 个大类（`submit` / `transfer` / `response_failures_total`）。
>
> buffer 池按必要性审查（design §3.1 第四类）：**新增 1**（`flag_pool_usage_ratio`——B2 链前兆信号 + "flag 池扩容"决策交叉验证）；**不设计 4 类**（数据池 alloc/free 计数、NoSpace 驱逐触发计数、size 未注册 / Free 失败计数、字节绝对值——均可推导或有日志兜底或已裁）。

---

## 1. 指标总览

| 组 | 主题 | 数量 | Histogram | Counter | Gauge |
|---|---|---|---|---|---|
| A | 请求量 | 3 | — | 3 | — |
| B | DUMP 业务 | 3 | 1 | 2 | — |
| C | LOAD 业务 | 3 | 1 | 2 | — |
| D | LOOKUP 业务 | 3 | 1 | 2 | — |
| E | 传输与响应 | 6 | 3 | 3 | — |
| F | 资源水位（本次增补 flag 池） | 3 | — | — | 3 |
| G | Metadata 阶段耗时（本次新增） | 8 | 8 | — | — |
| H | 队列与阻塞（本次增补） | 9 | 1 | 7 | 1 |

---

## 2. 分组明细

### A. 请求量（3）

| 指标名 | 类型 | 测什么 | 反映什么 | 怎么用 |
|---|---|---|---|---|
| `drampool_dump_requests_total` | Counter | 累计接收并处理的 DUMP 请求数（每请求 +1，与 batch 内 entry 数无关） | 写入负载规模 | 负载构成分析；与 `dump_bytes_total` 对比得平均批量；写入侧 rate 类指标的分母 |
| `drampool_load_requests_total` | Counter | 累计接收并处理的 LOAD 请求数 | 读取负载规模 | 读取侧所有 rate 类指标（miss、transfer 时长）的分母基准 |
| `drampool_lookup_requests_total` | Counter | 累计接收并处理的 LOOKUP 请求数 | 查询负载规模 | 换算"单请求查询成本"的分母 |

### B. DUMP 业务（3）

| 指标名 | 类型 | 测什么 | 反映什么 | 怎么用 |
|---|---|---|---|---|
| `drampool_dump_bytes_total` | Counter | 累计**成功预留缓冲**的 DUMP entry 字节总量（重复写与分配失败不计；传输失败回滚不回退本计数） | DUMP 吞吐规模（"预留成功"口径） | 吞吐趋势；带宽推导 `rate(bytes)/transfer均值`（§4.2）；容量规划 |
| `drampool_dump_storebegin_failures_total` | Counter | StoreBegin **真失败** entry 数（缓冲分配失败或分片注册失败；重复写是幂等正常结果，不计入）。**内存不足的直接归因指标**（§3.1 第三类） | 内存压力（分配失败）或元数据状态异常（注册失败） | 持续增长时结合 G 组 `allocate` 直方图区分两类原因；分配失败 → 扩容/调整驱逐策略 |
| `drampool_dump_prepare_duration_ms` | Histogram | DUMP 入口 → 传输提交完成的**本地准备耗时**（逐 entry StoreBegin + 缓冲分配 + 驱逐重试 + 提交，**不含**数据传输） | 写入路径服务端 CPU 侧开销 | P99 高时用 G 组按阶段归因（§4.5 决策树） |

### C. LOAD 业务（3）

| 指标名 | 类型 | 测什么 | 反映什么 | 怎么用 |
|---|---|---|---|---|
| `drampool_load_bytes_total` | Counter | 累计**成功加引用**的 LOAD entry 字节总量（未命中 / 长度不匹配不计） | LOAD 吞吐规模 | 读取吞吐趋势与读取带宽推导 |
| `drampool_load_miss_entries_total` | Counter | LOAD 中**未取到数据**的 entry 数（key 不存在 / 状态非 READY / 长度不匹配均计入——对客户端都是"没取到"） | 读取未命中规模 | 突增 = 读取了被驱逐 / 过期 / 未写完的数据；命中率 = 1 − miss/(miss + 成功 entry) |
| `drampool_load_prepare_duration_ms` | Histogram | LOAD 入口 → 传输提交完成的**本地准备耗时**（LoadBegin + 长度校验 + 提交，不含传输） | 读取路径服务端 CPU 侧开销 | 与 G 组 `loadbegin_duration_ms` 联动归因 |

### D. LOOKUP 业务（3）

| 指标名 | 类型 | 测什么 | 反映什么 | 怎么用 |
|---|---|---|---|---|
| `drampool_lookup_hit_entries_total` | Counter | LOOKUP 中**存在且 READY**（命中并刷新 TTL lease）的 entry 数 | 查询命中规模 | 命中率的分子（§4.4） |
| `drampool_lookup_miss_entries_total` | Counter | LOOKUP 中**不存在或非 READY** 的 entry 数（batch_size − hits） | 查询未命中规模 | 突增 = 查询了被驱逐 / 过期的前缀，通常先于 LOAD miss 出现 |
| `drampool_lookup_scan_duration_ms` | Histogram | 每 batch 一次：LOOKUP 元数据扫描总耗时（Σ `Exist` + 结果填充，不含响应写回） | LOOKUP 服务端主体处理时长（LOOKUP 无数据传输）；P99 高 = 分片读锁竞争或 lease 刷新开销 | 均值 ÷ 平均 batch_size = 单 key 查询成本（§4.5.2 ④） |

### E. 传输与响应（6）

| 指标名 | 类型 | 测什么 | 反映什么 | 怎么用 |
|---|---|---|---|---|
| `drampool_dump_transfer_duration_ms` | Histogram | DUMP 传输从提交（`submit_ms`）到**终态**（Completed / Failed / GetStatus 异常）的异步耗时，涵盖网络与对端读取 | 实际搬运能力 | `rate(bytes)/均值` = 传输带宽；与 prepare 相加 ≈ DUMP 服务端端到端 |
| `drampool_load_transfer_duration_ms` | Histogram | 同上，LOAD 侧（池 → 客户端方向搬运时长） | 同上 | 同上 |
| `drampool_transfer_failures_total` | Counter | 数据传输以**非 Completed 终态**结束的请求数（DUMP/LOAD 合并大类，§3.1 第三类；Failed 或 GetStatus 异常均计） | 传输失败率 | 增长需排查对端网络 / 连接状态（与 opcode 无关，低频不细分） |
| `drampool_response_rtt_ms` | Histogram | 响应传输**提交成功**（ExecuteAsync 成功后写 `submit_ms`）到**写回客户端内存完成**（Completed）的耗时（flag 池排队与 Pack 在提交前、不在窗口内） | 结果返回阶段的传输时延 | P99 高 → 对端写入慢或写回链路异常；flag 池背压由 `response_buffer_retry_total` / `flag_pool_usage_ratio` / `completion_inflight` 覆盖 |
| `drampool_response_failures_total` | Counter | 响应返回链路失败计数（**本地提交** Pack/ExecuteAsync 失败 + **写回传输**失败，DUMP/LOAD 合并大类，§3.1 第三类；flag 池 NoSpace 属重试不算失败） | 结果返回通道健康度 | 增长锁定响应链路故障域，结合 WARN 日志定位提交 / 写回哪一环 |
| `drampool_submit_failures_total` | Counter | 数据传输**提交失败**的请求数（DUMP/LOAD 合并大类，§3.1 第三类；ExecuteAsync 失败或 handle 无效；LOAD 侧整批已 LoadEnd 释放引用） | 传输子系统提交路径健康度 | 增长指向传输子系统初始化 / 资源异常，客户端整批失败 |

### F. 资源水位（3，数据池由 GCThreadLoop 每 `gcIntervalMs` 一轮采样、flag 池由 CompletionPoller 调用点记账，均无热路径开销）

| 指标名 | 类型 | 测什么 | 反映什么 | 怎么用 |
|---|---|---|---|---|
| `drampool_metadata_entry_count` | Gauge | 当前池内缓存条目（block）总数（1024 分片求和） | 元数据规模容量水位 | 增长斜率反映写入 / 驱逐平衡；配合 usage_ratio 判断驱逐压力区 |
| `drampool_buffer_pool_usage_ratio_<slot_size>` | Gauge（动态注册，每 block size 一个） | 各 block 尺寸池**已用槽位占比**（used / slot count） | 分尺寸内存水位 | 逼近 1 = 该尺寸池即将触发驱逐重试（DUMP prepare 抖动前兆），容量规划第一信号 |
| `drampool_flag_pool_usage_ratio` | Gauge | flag 响应缓冲池**已用槽位占比**（used / flagBufferSlotCount） | 响应回填缓冲水位；逼近 1 = B2 链（flag 池 NoSpace → `response_buffer_retry`）前兆信号 | 为 §4.6.3 "flag 池扩容"决策提供交叉验证（此前无水位指标可查） |

### G. Metadata 阶段耗时（8，本次新增）

> 观测点位于 `metadata.cc` 内部（模块边界测量），`ScopedTimer` RAII 零逻辑侵入；时钟 µs 精度（`SteadyNowUs()`）、以 ms（double）入直方图。阶段层级与归因方法见 §4.5。

| 指标名 | 类型 | 测什么 | 反映什么 | 怎么用 |
|---|---|---|---|---|
| `drampool_metadata_storebegin_duration_ms` | Histogram | 每次 `MetadataManager::StoreBegin`（每 DUMP entry 一次）**总耗时** = 缓冲分配（含驱逐重试）+ 分片注册 | DUMP 写路径元数据开销总视角 | P99 抬高时按 allocate / register 两阶段直方图归因（§4.5.2） |
| `drampool_metadata_allocate_duration_ms` | Histogram | StoreBegin 内**缓冲分配阶段**耗时：首次 Allocate，NoSpace 触发周期驱逐重试、再 NoSpace 触发深度驱逐重试（含驱逐耗时，与 evict_sync 嵌套） | "拿到一块缓冲"的真实成本 | 高且 evict_sync 占比高 → 内存压力；高而 evict 占比低 → 共享 BufferPool Allocate 本身慢 |
| `drampool_metadata_shard_register_duration_ms` | Histogram | StoreBegin 内**分片注册阶段**耗时（分片写锁 + 重复检查 + 双驱逐策略 AddKey + map emplace，失败含回滚 Free） | 元数据注册成本 | 高 → 分片写锁竞争，或 LRU / TTL 链表操作开销大 |
| `drampool_metadata_evict_sync_duration_ms` | Histogram | **业务线程内同步驱逐**（StoreBegin 分配 NoSpace 触发，周期 / 深度两级）单次耗时 | 驱逐对 DUMP 请求的**直接拖慢**（TaskWorker 线程内执行，阻塞当前请求后续 entry） | 增长 = 内存压力直接信号，与 `usage_ratio` 交叉验证 |
| `drampool_metadata_storeend_duration_ms` | Histogram | 每次 `StoreEnd`（DUMP 传输 Completed 后 entry INITIALIZED→READY）耗时，CompletionPoller 线程执行 | DUMP 终态结算成本 | 高 → 分片读锁竞争（GC 每秒全分片扫描会持读锁） |
| `drampool_metadata_loadbegin_duration_ms` | Histogram | 每次 `LoadBegin`（查 key + TryIncRef 加引用 + 双驱逐策略 AccessKey）耗时 | LOAD 读路径元数据开销 | 高 → 读锁竞争，或策略 AccessKey（LRU move-to-front / TTL 更新）开销大 |
| `drampool_metadata_loadend_duration_ms` | Histogram | 每次 `LoadEnd`（TryDecRef 减引用）耗时（出现在 poller 终态结算、len 不匹配回滚、提交失败回滚三类路径） | 引用释放成本（单次应近常数） | 异常升高 → 锁竞争 |
| `drampool_metadata_evict_gc_duration_ms` | Histogram | **后台 GC 每轮**全分片驱逐扫描总耗时（`PerformEvict` 整轮 = 1024 shard 之和，每 `gcIntervalMs`（默认 1s）一轮） | GC 对系统的持续开销 | 均值 ÷ `gcIntervalMs` = GC 线程占空比；高 → GC 频繁持锁干扰业务（storeend / loadbegin 变慢的常见外因） |

### H. 队列与阻塞（9，本次增补）

> 两条 SPSC 队列与响应缓冲是请求端到端时延的关键路径（S0），排队 / 阻塞直接等价于客户端可感知延迟。线程衔接模型与阻塞链归因见 §4.2（design §4.6）。

| 指标名 | 类型 | 测什么 | 反映什么 | 怎么用 |
|---|---|---|---|---|
| `drampool_queue_request_enqueued_total` | Counter | 累计**成功进入** requestQueue 的请求数（TryPush 成功 +1） | 接收侧入口流量（"已入队待处理"口径，与 A 组"已开始处理"相差排队数） | 与 `dequeued` 差值 = requestQueue 当前排队数（§3 恒等推导） |
| `drampool_queue_request_dequeued_total` | Counter | 累计 TaskWorker 从 requestQueue **取出**的请求数（TryPop 成功 +1） | TaskWorker 消费速率 | SPSC 无丢弃恒等式：排队数 = 入队 − 出队 |
| `drampool_queue_request_full_total` | Counter | requestQueue **满、TryPush 失败**事件数（每次失败 +1 按次计） | 接收线程被阻塞强度（阻塞时长 ≈ full 数 × `requestReceiverIdleWaitUs` 100µs）；阻塞链 B1 | 增长 = TaskWorker 消费跟不上到达速率或下游反压传导；高概率压力场景的核心告警信号 |
| `drampool_queue_request_enqueue_wait_ms` | Histogram | 每请求从准备入队到 TryPush 成功的**入队前等待时长**（含满重试 sleep，未排队 ≈ 0） | 客户端可感知的接收背压延迟 | P99 抬高必伴随 `full_total` 增长；分位数估计单请求接收延迟 |
| `drampool_queue_completion_enqueued_total` | Counter | 累计**成功进入** completionQueue 的完成记录数（每请求一条，Push 自旋至成功恒 +1） | 完成流生产速率 | 与 `dequeued` 差值 = completionQueue 当前排队数 |
| `drampool_queue_completion_dequeued_total` | Counter | 累计 CompletionPoller 从 completionQueue **取出**的完成记录数（FillPendingWindow 拉取成功 +1） | 完成流消费速率 | 与 `enqueued` 组成恒等式（排队数 = 入队 − 出队） |
| `drampool_queue_completion_full_total` | Counter | completionQueue **满、SubmitCompletion 被迫自旋等待**事件数（Push 前 TryPush 探测，失败 +1 后退回 Push——探测不改行为） | TaskWorker **停摆**位置与强度（既不取新请求也不响应停止指令）；阻塞链 B3，**停摆无日志兜底，此指标是唯一观测手段** | 增长 = Poller 消费能力不足（pending 窗口满）或传输终态 / flag 池重试慢；随后 `request_full` 连锁增长 |
| `drampool_queue_completion_inflight` | Gauge | CompletionPoller pending 窗口内在途完成记录数（等终态 / 等响应提交 / 等写回，每轮覆盖写） | 完成链路第二级缓冲占用 | 持续逼近 `pollerPendingDepth`（默认 64）= 拉取停摆；与 `completion_full` 互相印证 |
| `drampool_queue_response_buffer_retry_total` | Counter | 响应 flag 缓冲池 **NoSpace、SubmitResponse 留 pending 下轮重试**事件数 | 响应缓冲供给不足（不阻塞 Poller 线程，但阻塞该请求响应提交、response_rtt 抬高）；缓冲链 B2 | 增长 = 响应突发超 slot 供给或写回慢未释放槽位；与 `response_rtt_ms` P99 联动（flag 池扩容） |

---

## 3. 关键推导公式（PromQL 速查）

```promql
# DUMP/LOAD 命中相关
命中率(LOAD)   = 1 - miss/(miss+成功)：
  1 - rate(ucm:drampool_load_miss_entries_total[5m])
      / (rate(ucm:drampool_load_bytes_total[5m]) / 平均entry大小 + rate(ucm:drampool_load_miss_entries_total[5m]))
# 更简：LOOKUP 命中率（精确，Counter 直测）
sum(rate(ucm:drampool_lookup_hit_entries_total[5m]))
  / (sum(rate(ucm:drampool_lookup_hit_entries_total[5m])) + sum(rate(ucm:drampool_lookup_miss_entries_total[5m])))

# 传输带宽 (GB/s)，以 DUMP 为例
rate(ucm:drampool_dump_bytes_total[5m])
  / (rate(ucm:drampool_dump_transfer_duration_ms_sum[5m]) / rate(ucm:drampool_dump_transfer_duration_ms_count[5m]))
  / 1e6

# GC 线程占空比
rate(ucm:drampool_metadata_evict_gc_duration_ms_sum[5m])
  / rate(ucm:drampool_metadata_evict_gc_duration_ms_count[5m]) / (gcIntervalMs秒数)

# LOOKUP 单 key 查询成本
rate(ucm:drampool_lookup_scan_duration_ms_sum[5m])
  / rate(ucm:drampool_lookup_scan_duration_ms_count[5m])
  / (rate(ucm:drampool_lookup_hit_entries_total[5m]) + rate(ucm:drampool_lookup_miss_entries_total[5m]))

# 守恒校验（偏差 = 胶水代码开销，正常远小于任一阶段）
rate(ucm:drampool_metadata_storebegin_duration_ms_sum[5m])
  ≈ rate(ucm:drampool_metadata_allocate_duration_ms_sum[5m])
  + rate(ucm:drampool_metadata_shard_register_duration_ms_sum[5m])

# 队列当前排队数（SPSC 无丢弃 → 速率差恒等，§4.6.2）
requestQueue 排队数：
  rate(ucm:drampool_queue_request_enqueued_total[5m]) - rate(ucm:drampool_queue_request_dequeued_total[5m])
completionQueue 排队数：
  rate(ucm:drampool_queue_completion_enqueued_total[5m]) - rate(ucm:drampool_queue_completion_dequeued_total[5m])
# 接收阻塞时长估计 ≈ rate(ucm:drampool_queue_request_full_total[1m]) × 100µs
```

---

## 4. 阶段层级与归因速查（G 组 + H 组）

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
├─ load_prepare_duration_ms → Σ metadata_loadbegin_duration_ms
└─ load_transfer_duration_ms → Σ metadata_loadend_duration_ms
LOOKUP 请求（无数据传输）
└─ lookup_scan_duration_ms（batch 级）
后台 GC（每 gcIntervalMs 一轮）
└─ metadata_evict_gc_duration_ms（轮级）
```

| 症状组合 | 结论 | 动作方向 |
|---|---|---|
| allocate P99 高，evict_sync 占比高 | 内存压力触发驱逐重试 | 扩容 / 调整驱逐比例 / 排查突发写入 |
| allocate P99 高，evict_sync 占比低 | 共享 BufferPool 分配慢 | 排查池锁与伙伴分配实现 |
| register P99 高，GC 占空比高 | GC 扫描持锁干扰写锁 | 调大 gcIntervalMs 或降驱逐比例 |
| register P99 高，GC 占空比低 | 驱逐策略数据结构开销（LRU/TTL 链表） | 深挖策略实现 |
| storeend / loadbegin 抬高，与 evict_gc 同步 | 后台驱逐读锁与业务读锁互扰 | 同上 |
| lookup_scan P99 高，loadbegin 正常 | Exist 路径特有问题（TTL lease 刷新写状态） | 排查 TryMarkHit 与 lease 逻辑 |

### 4.2 队列流水线层级与阻塞链归因（H 组，§4.6）

```
RequestReceiver ─TryPush─▶ requestQueue ─TryPop─▶ TaskWorker ─Push─▶ completionQueue ─TryPop─▶ CompletionPoller
      ▲ B1：满→sleep 重试（TCP 收包停滞）                              ▲ B3：满→Push 自旋停摆                    │
                                                                                                              ▼
                                                                              pending_ 窗口（completion_inflight）─B2：flag 池
                                                                              NoSpace 留 pending 下轮重试（单请求响应延迟）─▶ 响应写回
```

| 链 | 阻塞位置 | 形态 | 观测信号 |
|---|---|---|---|
| B1 | Receiver：requestQueue 满 | 主动 sleep 阻塞（100µs 重试）→ TCP 收包停滞 | `request_full_total` ↑ + `request_enqueue_wait_ms` 右移 |
| B3 | TaskWorker：completionQueue 满 | Push 自旋停摆 → requestQueue 堆积反压 B1 | `completion_full_total` ↑ → `request_full_total` 连锁 ↑ |
| B2 | Poller：flag 池 NoSpace | 单请求留 pending 重试（不阻塞线程） | `response_buffer_retry_total` ↑ + `completion_inflight` 贴近 64 |

归因速查：request_full ↑ 而 completion_full 平 → B1（TaskWorker 消费不足）；completion_full ↑ 且 request_full 随后连锁 ↑ → B3（Poller 消费不足，反压传导）；retry ↑ + inflight 贴满 + `flag_pool_usage_ratio` 逼近 1 → B2（flag 池扩容，flag 池为独立 region 与数据池无关）；全为 0 但吞吐低 → 转向传输/metadata 阶段分析。

---

## 5. 时钟与精度说明

- `drampool_types.h` 新增 `SteadyNowUs()`（µs 精度）：metadata 单次操作为微秒量级，毫秒精度时钟下直方图会退化为 0/1 两值。
- 所有 Histogram 内部以 µs 计时、以 **ms（double，µs 分辨率）** 入直方图——指标名统一 `_ms` 后缀，Prometheus 侧单位一致。
- 桶集（§5.4 `metricsDurationBucketsMs`）：亚毫秒段 `0.001–0.5`（覆盖 G 组 µs 级观测）+ 毫秒段 `1–10000`（请求 / 传输级），15 个 histogram 共用一组。
