# Early Prefetch：批次流水与原生引用交接

Early 在 UCM 内提前准备当前可用的 KV。Scheduler 继续使用原生 waiting /
skipped_waiting、block allocation 和 metadata；仍需要外部 KV 的 PENDING 返回 `(None, False)`。
若原生 HBM lookup 已覆盖所有完整 allocation block，则同步返回 `(0, False)`，
取消无收益的后台 job。取消仍经原有排空协议释放，不把尚未 READY 的 L1 当作命中；
保留已取消实例的 identity/hash 直到 finish，使 block allocation 重试无需重新扫描 tokens。
READY 通过保存的 block IDs 和受保护命中结果进入原生 token 计算：已核验共享域由单 owner 给出，其余配置继续汇总所有 rank 的最小命中。
不等待其他请求未来产出，不维护 producer prefix、持久化观察或 revalidation。

## 职责与复用

| 组件 | 职责与关系 |
|---|---|
| EarlyDirectConnector / EarlyLayerWiseConnector | 薄子类；继承原生初始化、metadata、逐层执行和 KV load，只接入请求生命周期与保存结果 |
| EarlyScheduler / EarlyWorker / Control | 请求实例 identity、异步提交、PENDING/READY/FAILED/CANCELLED、Stan HRRN、既有批次线程及传输；非 owner 模式保留 rank 汇总 |
| OwnerBroker : BrokerState | 在既有 broker/job table 上扩展原生共享对象握手、每 generation 单 owner、共享域额度、接管及排空确认 |
| OwnerWorker / Consumer / LoadConsistency | 组合现有 worker 和同一个原生 RankConsistencyManager；关联真实任务完成、消费范围及 generation，不创建另一套加载队列 |
| Early::DomainProvider | EarlyCacheStore 的只读能力接口；通过已创建 buffer 的映射取得内核对象身份和原生容量 |
| EarlyPipelineStore | 继承原生 builder、load/load_data 参数转换、任务对象和完成检查；组合 `_ReservedLoad` 路由同一个 C++ PipelineStore 的 Load/Wait，并观察成功完成 |
| EarlyCacheStore | 继承 CacheStore；使用父类已创建的 BufferManager、TransManager；为原生加载任务附加已有 Handle |
| ReservedPrefetch | 每批连续命中、reservation、定位合并、受保护 block 引用复用、有界位置提示和 worker 内的引用交接 |
| ReservedPrefetch::Batch / Early::Batch | Early 专用的有界推进能力；保存原生 task handle，复用同一 backend 的 Load、Check、Wait；不另建 I/O 线程 |
| BoundBatch | Python 能力对象；保持创建它的原生 Pipeline 与 health wrapper 存活，析构前排空已提交 I/O |
| EarlyBufferPins | 对同一个原生 buffer 批量查找；复用原生 Hash、bucket/节点锁和 Handle，不拥有共享状态 |
| PosixReadLease | 使用原生 SpaceLayout 定位 backend，在既有锁文件中保护 block；GC 仍通过原有删除 guard 与读者互斥 |

`_ReservedLoad` 不创建 Store、buffer 或线程。C++ EarlyLoad 绑定调用原生私有
MakeTaskDesc，通过原有 Self 入口及 health wrapper 提交；不复制参数解析。

## 调度匹配与消费确认

READY 已保存的 block IDs 和受保护命中数通过 Early 的 ContextVar / Store 子类
进入原生匹配方法；不再次 reserve、hash 或执行 L2 lookup。原生继续扣除 HBM 命中、
过滤已知缺失、处理 CP 和 full-hit recompute 并创建 metadata，返回 `(tokens, False)`。
这不是 HBM 命中伪装：外部 KV 的 H2D 仍通过原生 layerwise/direct load 完成。

owner generation 的 key/attempt/grant/expected 随原生 step metadata 传递。无需另发
broker 不使用的 owner handoff 消息；兼容的非 owner 路径仍保留有序 handoff 消息。
metadata 到达不释放保护，仍等每个必要消费者的实际成功 Wait 覆盖全部预期层、
block 和 chunk，再执行 retire / quiesced / owner_drained。

Consumer 按成功完成的 `(layer, block)` 增量扣除 remaining，并保留去重集合；
每次完成只处理本次任务涉及的层和本 generation 的预期 blocks，不重复检查前面
所有层。混合请求任务、重复 chunk 和重复 receipt 不多计；失败不扣额度，仍 fencing。
step 重新绑定相同计划时复用已建立的 expected 集合；变化的计划仍拒绝。

每个请求首次真正被调度时记录一次 `schedule_match` 汇总：generation key、DP、
owner、READY 接收时刻、首次 PENDING、PENDING 次数、匹配调用数和总耗时、
最后一次 HBM/external token 数以及原生 SCHEDULED 时刻。缺失时刻保留 None。
`match_ms` 包含原生 Python token/metadata 处理，不是 C++ reserve 时间；它可能覆盖
多次 allocation 重试，必须结合 match_calls 解读。日志不代表首 token 时间，
需通过 request/key 与客户端流式首 token 记录关联；不相减不同指标的分位数。

## Reservation 与可用前缀

候选段从 32 block 开始，连续命中时扩大到 64、128、256 block，随后保持上限。
按段获取 read lease，然后检查该段的 shard。只有全部 shard 已 READY
且持有 pin 的 block 才能直接作为 L1 命中，包括已经不在 L2 的 block。
其余 block 通过原生 LookupOnPrefix 判断当前 L2 可用性；遇到第一个两层均不能
满足的 block 后停止后续候选段。没有当前 KV 的部分由原生推理计算。

每段是独立 reservation，持有本段 lease。一次获取中，同一 backend 只打开一个
锁文件描述符。后段定位与前段读取可以并行，因此不修改已发布 lease；不再需要
Extend 或保存 layout 指针。获取中途失败时，RAII 释放本批已取得的锁，已发布
前批仍有自己的保护。锁文件不随请求删除，实际 GC owner 的 guard 与 unlink
保持同一互斥区间。read lease 不承诺尚未完成的 dump 已经持久化。

EarlyBufferPins 将该段的查询按原生 Hash 分组，同一 bucket 在本批只加锁和遍历
一次。分组数组是临时查询参数，不存储共享 buffer 布局。节点保护、容量限制与
状态判断继续使用原生策略。原生 TransBuffer 不增加 Early 查询方法；既有
friend 声明保留所有原生成员的 private 属性。

同 worker 内，同 block 的并发定位共享短期 ProbeFlight。领导者先完成自身全部
定位并发布结果，再等待其他领导者的结果，避免相反 block 顺序形成等待环。
等待不持有全局或 bucket 锁，沿用已有超时；异常也完成 promise 并唤醒加入者。
这只合并一次当前索引操作，不等待 producer 产生 KV，也不是跨 rank 的位置协议。

只对没有可复用保护的 block 执行批量查询。完整 READY 的 block 使用共享的
BlockPins 所有权，其原生 Handle 只取得一次；同一 worker 中后来的 reservation
可以共享它，不再逐 shard 查询、加锁或增加节点引用。原生节点仍因该组持有
Handle 而无法被淘汰，直到最后一份 BlockPins 所有权结束。readyBlocks_ 对保护组
只保存弱引用，不延长 pin 生命周期。
此机制只复用当前受保护的 READY block，不等待其他请求产出。
复用时仍调用原生 MarkAccessed 更新 CLOCK 热度。这是对已保护位置的原子访问
标记，不重新查 bucket、拿节点锁或增加节点引用，也不是完全没有逐 shard 内存操作。

上一 reservation 释放后，位置提示可以帮助新的请求跳过共享 bucket 查询。提示仅
保存节点序号，不保存裸指针或 pin；条目数不超过原生节点容量可容纳的 block 数。
使用提示时逐节点取得原生节点锁，在同一个临界区内检查容量范围、block identity、
shard index、READY 状态并取得引用。原生 Alloc 在同一节点锁内改写身份，且不能
回收仍有引用的节点。部分检查失败时 RAII 释放已取得的引用，再走正常批量查询。
这条路径仍有逐 shard 节点锁，只省去 bucket 查找；不能把旧位置本身当作可用结果。
提示按访问次序淘汰元数据，不影响 KV 回收、请求准入或 pin/lease 生命周期。
每个 worker 的提示独立；共享 L1 的权威身份与状态仍来自同一个原生共享 buffer。

批量 pin 可借用具有活跃 owner 的 LOADING 位置，不在 reservation 阶段分配新位置
或接管其 I/O。已经借用的位置在 owner 完成、释放后仍有引用保护，预取阶段无需
再次定位。真正不存在的位置不能作为可信的长期负查询结果：进入数据阶段后，
TryGet 必须重新确认其他 rank 是否已经创建同一 shard，再决定获取或分配。

total_hit_blocks 表示连续总命中；remaining_l2_work 是 reservation 时尚未 READY
的 shard 数，包含其他 owner 的在途工作，不等于本 rank 最终的 SSD 读取字节。
HRRN 保持 `1 + waiting_time / max(remaining_l2_work, 1)`。
批次推进时，已完成的本地读取与已观察到的 peer READY 从工作估计中扣除。
尚未观察到的 peer 完成可能仍被计入，因此它是保守估计；实际 SSD 读取另记。

当剩余工作为零时不进入数据队列，不调用 PrefetchReserved。零长度 miss 也满足
这个条件，因此分析日志时必须与正长度的 L1 命中分开。

## 分批搬运

```mermaid
sequenceDiagram
    participant W as EarlyWorker lookup executor
    participant D as EarlyWorker data thread
    participant R as ReservedPrefetch
    participant B as 原生共享或私有 Buffer
    participant P as 原生 Posix backend
    W->>R: ReservePrefix(第一候选批)
    R->>P: 本批 AcquireReadLease
    R->>R: 复用仍受保护的完整 READY block
    R->>B: 已释放位置提示：节点锁内核验身份与 READY 后取得引用
    R->>B: 对剩余 block 批量 pin READY 或活跃 LOADING 位置
    R->>P: LookupOnPrefix(未完整命中 L1 的 block)
    R-->>W: reservation、Batch 能力、命中量与剩余工作
    W->>D: 发布已确认批次
    par 继续确认后续连续前缀
        W->>R: 下一批 ReservePrefix
    and 在有界窗口内推进读取
        D->>R: Batch.Step(可用字节)
        R->>B: 复用已有 Handle；缺失位置用原生 TryGet 重新检查
        R->>P: 原生 Load(本进程 owner shard)
        R-->>D: 返回在途字节、剩余工作和提交能力
        D->>D: 选择其他请求的批次
        D->>R: Batch.Step(0) 检查完成
        R->>P: 原生 Check；已完成后 Wait 回收
        R->>B: MarkReady
    end
    Note over W,D: reserve 结束且本请求所有批次完成，才发本 rank READY
    Note over W: owner 模式仅此 worker 执行 reserve；否则仍收齐全部 rank READY
```

仅 owner 提交读取；READY shard 和其他 owner 的在途 shard 不重复读。
局部 Transfer 容器只保存已提交的原生 task handle。一次 Step 最多定位 64 shard，
窗口仍为 128 shard 的字节数，并由同 worker 的所有在途批次共同使用。健康的
未完成 I/O 不在 Step 内同步 Wait；原生没有完成回调时，数据线程以既有 1 ms
等待节奏调用 Check，不增加 lookup/data 线程或 Posix 并发。

数据队列每请求选取最早的可提交批次。每个有限轮次内，每请求取得一次有界
推进机会；该轮候选按 Stan HRRN 选择前四分之一，完成该轮后才接纳新到候选。
等待完成的批次仍在 active 中回收，不占用提交机会。整请求 PENDING 保持不变；owner 模式等 owner 的完整受保护结果，兼容路径等全部 rank。软化门控不属于此设计。

取消后停止新提交，已接受的任务由 Check/Wait 排空，再标记尚未完成的本地 owner
为 FAILED。非 owner 路径关闭时数据线程继续排空 active，未发布的 lookup 结果立即释放。owner 路径关闭执行下述 generation 排空握手；无法证明排空则停用共享域。
能力对象异常析构也先 Wait；这里沿用原生 Wait 返回前回收 I/O 的契约。
共享位置的失败仍会唤醒消费者进入可回退状态。

## 容量与保护

共享 BlockPins 持有一份字节额度；同 worker 的多个消费者共享该额度，最后一个
释放时返还。额度包含缺失位置，以保守约束将要保护的 buffer。上限来自同一原生
buffer 的 CanReserve 范围，排除原生 load-exclusive 区域；不在 Python 复制布局。
额度不足直接返回原生 NoSpace 并回退，不持有部分 pin 等待额度，避免整请求
READY 语义下发生容量等待死锁。这不是 H/R admission；owner 层另有限制排队任务和候选工作的安全配额。

这一底层额度仍按 worker 管理。owner 模式在它之上使用同一 broker 的共享域总预算；不能把每个 worker 的底层上限分别当作整份 L1 的可用额度。原生共享节点引用、CanReserve 与 TryGet 继续决定物理安全性。两种额度都是保守计费，不是物理占用采样。

## 原生加载交接

```mermaid
sequenceDiagram
    participant W as 本地 EarlyWorker
    participant E as EarlyPipelineStore / EarlyCacheStore
    participant R as Reservation
    participant Q as 原生 TransManager / LoadQueue
    participant H as 本 rank HBM
    W->>E: bind 本步 READY consumer 的本地 reservation IDs
    E->>R: 用 block identity 和 shard index 取得已持有 Handle
    R-->>E: 复制原生 Handle 引用
    E->>Q: 原生任务携带 Handle
    Q->>Q: 移入 ShardTask；缺失部分走原生 Get
    Q->>H: 原生 H2D / layerwise 执行
    H-->>Q: 原生 stream 完成
    Q->>Q: 释放 ShardTask / holder 引用
    W->>R: 本地引用已保护传输；owner 模式仍等全部消费者完成/终止确认
```

同一步相同 reservation ID 集合只建立一次 LoadPlan，索引 block 到对应的
reservation 及 block offset；不再为每个候选 reservation 遍历整个 load task。
每次 load 对实际请求的 block 做一次解析，连续 shard 直接复用已选中的组。
计划仅保存弱引用，不持有资源；READY 状态变化、release 或候选集合变化会使
计划失效，其中 READY/release 仅使包含该 reservation 的计划失效。无关请求
不会触发重建。Release 不扫描整个 READY/hint 表。交接仍检查本地 reservation 存在、未取消、已 READY、block identity
匹配及 shard 范围合法。另一个消费者持有相同 block 不构成本任务的访问授权。

原生 Handle 的复制通过既有节点引用计数取得保护，避免重新查找共享哈希桶。
加载队列接管该引用，继续使用原生 H2D 与完成清理；已有完整 block reservation
的 shard 无需再执行下一 shard 的 Prealloc。未交接的 shard 完整保留原生路径。
即使 reservation 在任务排队期间释放，任务引用仍保护数据直到传输完成。

所有地址与 Handle 只在持有它们的 worker 内使用。MLA shared L1 的创建/挂接、
unique_id、owner、物理范围和容量，以及非 MLA private L1 的语义均交给原生。
不同 rank 从自己的本地映射取得 Handle，并加载到各自 HBM。

Early 关闭继续选择原生工厂。没有预取结果或 FAILED/CANCELLED 时继续走原生
fallback。vLLM 与 LMCache 不增加 Early 状态机、队列或接口。

## 已核验共享域的单 owner 协议

`enable_early_prefetch` 关闭时不进入 Early 工厂。开启后，
`enable_early_owner` 默认为 true，但握手成功前仍使用既有非 owner 路径。
握手读取 **已创建的同一个** native buffer：通过 DataAt/MetaAt 所在 MAP_SHARED
映射核对设备号、inode、文件大小和映射内偏移，并核对当前路径仍指向该对象。
握手还包含 boot ID、IPC/mount namespace、native unique_id、节点容量、
block/shard bytes、tensor sizes、layerwise、CP/block 分组及原生 hash 指纹。
MLA block identity 来自 scheduler 的 rank 0 RequestHasher；worker 握手复用该
原生 hasher 类型与配置生成指纹，不读取仅 scheduler 初始化的 `_seed`，也不以
worker 自己包含 TP rank 的 hasher 指纹判断跨 rank 一致性。
不传播地址、不重建索引、不凭 UUID 或同名文件确认物理共享。

仅启用 MLA/shared L1、CP=1、PP=1、可证明布局一致的完整 DP/TP cohort。每个 DP 的
必要消费者须共享同一个对象；跨 DP 若指向同一对象，统一计费。完整 cohort 的
capability、协议版本、DP/rank 身份及心跳均须满足条件，且原广播模式任务已经
退出，才能开启 owner。缺少握手、private L1、非 MLA 或布局不兼容时记录原因，
保留原模式。启用后出现对象/进程 incarnation 改变则停用域，不能混入无配额广播。
这些检查是同一受信任引擎内的能力协商，不是面向恶意 worker 的认证协议。

broker 为请求实例生成的 UUID key 保存独立 attempt 和单调 grant。请求 ID
只用于原生 metadata 关联，不能授权释放。调度器发送序号防重放；进程 session
变化不允许接续旧任务。owner 从本 DP 的健康合格 rank 轮询选择；select_owner
是策略扩展点。非 owner 只收到 generation/owner 通知，不收到完整候选数组，
不执行 reserve、lease、候选扫描或预取。owner 使用既有四个 lookup worker、
一个 data worker 及原生 I/O 并发，继续自适应批次、reserve/read 重叠和跨请求
有限轮次 HRRN。无需跨 rank 的 handle、reservation ID、节点目录或读取分片。

owner reserve 完成且所有已发布批次 READY 后，broker 发布受保护的连续前缀。
Scheduler 复用已保存 IDs/hit，执行同一原生 HBM 扣除、缺失过滤、token 计算和
metadata 主流程。实际 allocation 前的重试可以刷新原生消费计划；首次 dispatch
后冻结该 generation 的预期 external blocks。失败恢复原生 lookup/load，而不
使用旧 READY 地址。没有运行中迁移或自动换 owner；需要重新处理时由原生重算。

## 共享额度

共享域按实际内核对象键统一记账，DP 间共享对象只计一份总量：

- 从 native 节点总数 `N`、非 load-exclusive 可预留数 `R` 推导保护上限。
  为每个成员的一项最大 native load 任务保留空间，合计 `L` 个 shard；为各
  MLA dump owner 的一项最大聚合 dump 任务保留空间，合计 `D` 个 shard。
  上限为 `max(0, min(R-D, N-L-D)) * shard_bytes`。load 每任务一个请求；
  dump 可聚合当前 scheduler batch，所以 dump 任务界限还乘 `max_num_seqs`。
  请求界限用锁定的 `max_model_len` 和原生 token/block 布局推导，direct 路径
  计入其所有 shard。每个 worker 只有一条原生 load dispatcher 和一条 dump
  dispatcher；已经完全提交的任务可以无需新增空间而排空。这个保留量确保
  所有尚在定位的任务都有到达末尾同步点的空间；dump 只能使用非 exclusive 区。
  不依赖重复 block 去重来缩小界限。实际 Load/Dump 提交前校验 shard 数，超出
  握手宣告的界限则停用域，不能悄悄突破证明。未启用 owner 时不施加该校验。
  上限为零时拒绝预取并回退原生，不切回无共享预算的广播模式。
  协议版本 2 在执行前按共享域内候选 block ID 并集预扣额度，包含尚未分配的位置。
  同一兼容布局下，相同 block 的多个 job/attempt 各持声明，物理额度只收一份；
  不同 DP 的任务仍独立路由，只有实际相同共享对象才共用额度。账本不是原生 pin，
  不传递本地指针。底层同 worker 的 BlockPins 保守额度及原生引用继续保留。
- 每个 owner 排队/执行 job 上限默认 4，可通过顶层 native 配置字典的
  `early_owner_max_jobs` 设置；候选工作上限 `early_owner_max_candidate_blocks`
  默认为 0，表示从共享域保护容量推导。两者限制的是本协议工作量，不恢复 H/R。
- 域 I/O 上限为 `min(保护上限, 成员数 * 128 * shard_bytes)`。按成员数向下取整
  成 shard 对齐的固定 worker 配额；每个 worker 所有 active 批次共享此配额。
  原生每 Step 64 shard / 每 worker 至多 128 shard 窗口继续生效。
- 额度不足时尝试其他合格 owner；仍不足则在任何 I/O 之前 FAILED/native
  fallback。没有持有部分额度等待余量，没有等待额度时持有 bucket/global 锁。
- 只有 owner READY 才能归还已截断尾部的声明：此时所有数据批已完成，Reserve
  已销毁首个 miss 之后的 BlockPins，没有后续 I/O。连续命中部分继续保护。
  零命中可归还全部内存声明，但 owner job/候选工作名额仍等待原释放协议结束。
- 命中部分不能因 metadata、取消或超时返还；必须收到该 attempt 的 owner_drained，
  证明 lookup、I/O 和消费引用已排空、reservation 已释放。只有该 block 的最后
  一个声明退出才归还物理额度。失败任务也保留声明直到 drain；域失联时不归还。

全候选预扣仍可能使超长短命中请求提前 fallback；HBM 命中部分也仍保守保护。
不能把候选并集额度峰值当作实测 pin 字节。未引入 unpinned READY 或部分成功语义。

## 全部消费完成后的保守交接

选择原生成功 Wait 作为可靠接管确认点。其之前 LoadQueue 已取得引用并完成
stream synchronize；不通过 metadata、入队、首层或首 chunk 推断保护接管。
消费状态按 `(key, attempt)` 保存每层已完成 block 集合及 active 原生任务数。
`LoadConsistency` 组合 **同一个** RankConsistencyManager，只给 submit_load
建立请求作用域，其余参数转换、提交、错误分类、等待与元数据反馈均调用原对象。
每个 receipt 绑定确实参与该次原生加载的请求，不能用相同前缀替另一个请求确认。

```mermaid
sequenceDiagram
    participant B as Broker
    participant O as 所选 owner
    participant C as 全部必要消费 rank
    participant Q as 各 rank 原生 LoadQueue
    B->>B: 候选并集预扣共享保护额度，选择 owner
    B->>O: owner_reserve(key, attempt, grant)
    B->>C: observe(key, attempt, owner)
    O->>O: reserve/read 批次流水，保留 pins/lease
    O-->>B: owner_answer(受保护连续 READY prefix)
    B->>B: 归还已丢弃尾部声明，保留命中声明
    B-->>C: 经 Scheduler/native metadata 绑定完整消费计划
    C->>Q: owner 用本地 handles，其他 rank 原生 Get
    Q->>Q: H2D；成功 Wait
    Q-->>C: 该任务完成（层与 block 范围）
    C->>C: 累积所有层/chunk；扣除原生 HBM 命中范围
    C-->>B: consumed（完整计划完成且 active=0）
    B->>C: retire(key, attempt)
    C-->>B: quiesced（禁止未来该 generation 加载且 active=0）
    B->>O: release_owner（全部消费者均 quiesced）
    O->>O: 停止新 I/O，排空已接受任务，释放 reservation
    O-->>B: owner_drained
    B->>B: 归还共享额度
    B->>C: forget；grant watermark 防止旧 metadata 复活
```

每个 rank 的必要 layers 来自本地原生 layer_ids，必要 blocks 来自 Scheduler
已经过滤且扣除 HBM 命中的完整计划。空计划可直接 consumed。正常 chunked
prefill 的后续 metadata 不覆盖先前未完成层/块；一次 last-layer callback 不
负责 owner 释放。只有 generation 级 broker retire 可以终止消费；原生 worker
只携 request ID 的旧 finish/preempt 通知不能终止复用该 ID 的新请求。forget 按
job 保存的 request ID 和当前映射值定向清理未调度取消的关联，不扫描全表，也
不能删除后来复用该 request ID 的映射。
retire 禁止未来加载，但必须等已经提交的原生任务成功完成才 quiesce。
已退休票据可能再次出现在 decode metadata，绑定允许；尝试再次加载会拒绝。

该策略不按层提前释放 owner pins；峰值寿命更长，由上述候选并集共享额度约束。
在最后一个消费 rank 延迟时，其他 owner 仍不能淘汰它需要的数据。原生引用自身
仍保护 H2D；保守策略在其完成后才释放 owner 保护，没有交接空档。

## 取消、关闭与恢复边界

PENDING 取消先停止新提交，已接受 I/O 沿用 Batch 的 Check/Wait 排空；owner
只有 lookup executor 结束、所有已启动 batch 完成后才 ack drained。READY
取消保留 pins，直到全部消费 rank quiesce。迟到/重复消息核验 key、attempt、
所属 DP 和发送角色；已完成旧 generation 不能影响新请求。正常 shutdown 尝试
同一 worker_retire → quiesced → drain 握手，在原生 timeout 范围内等候。

**原生共享节点 reference 没有 PID 所有权，spinlock 不是 robust 锁。**
控制超时不能证明进程死亡，更不能证明 H2D 已结束。owner/consumer 失联、
进程 session 变化、共享对象变化、失败的原生 Load/Wait 无法证明排空，或关闭
握手超时，都会 fence 整个 cohort：停止新的 Early/native load、传播引擎失败、
保留 pins 和计费，不猜测性释放或迁移。隔离对象保留到进程结束，避免析构触碰
可能由死亡进程持有的锁。恢复须在确认所有成员进程已退出后，通过原生机制以
新的 unique_id/共享对象重启整个 cohort；不在旧对象上原地恢复。

C++ Batch 仍遵守所选原生 backend 的 Load/Wait 契约：仅成功返回的 task 计为
已接受 I/O，Wait 回收后计 completed bytes。原生 AIO dump 提交/rename 时序
是独立风险；read lease 只防删除，不证明 KV 已完成写入，不恢复 producer 等待。

诊断按 key/attempt/DP/owner/consumer、batch/reservation 与 PID/task 关联；记录
队列、lease、定位、L2 查询、首末 I/O、READY、实际 Wait、保护释放及额度峰值。
`references_acquired_by_ns` 是 dispatch 结束的上界，非逐 shard 获取的精确时间。
取消时 wasted_read_upper_bound 是已完成读取的保守上界；submitted 与 completed
分开。没有记录不计零；跨进程阶段差值需要同 boot/time namespace 的时钟证据，
不以两个分位数相减推导阶段耗时。构建、测试和实测结果另见操作文档。

详细 reserve/prefetch/load-handle 阶段记录及逐 native task 提交/完成记录使用原生 DEBUG 级别；
格式化前检查是否启用。INFO 保留 job/batch 关联、额度、READY、消费完成和释放摘要。
`fast_path` 只计正命中且无需数据批的结果；owner 的 `outcome` 区分 zero_hit、
l1_ready、prefetched 和 failed。原生 Get 回退数不代表 SSD 读取数，实际读取需核对
load_l2_read_shards 与 Early 已提交/已完成字节，缺失记录不能记作零。
