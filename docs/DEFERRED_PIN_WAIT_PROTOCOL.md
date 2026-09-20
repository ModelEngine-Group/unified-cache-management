# 延后 pin 与可推进的容量等待

适用范围是协议 5 已确认共享同一原生 L1、block identity、布局及成员的 MLA owner 域。
原生 L1 是 CPU buffer，HBM 是各 rank 的设备 KV；不把二者混称。非 owner/private
路径保留 c50 行为，没有本轮共享域等待保证。

## 状态和所有权

| 状态 | 保留资源 | 可以推进的工作 | 转移条件 |
|---|---|---|---|
| 查询/资格等待 | identity、候选命中、read lease；无长期 L1 pin | 原生前台、其他预取 | HRRN 选中且安全完成顺序存在 |
| 激活 | 当前批原生 READY/LOADING 引用 | 后台查询与已有 I/O | 重新核验身份；L1-only 丢失可真 miss |
| 执行 | 实际目标引用、lease、原生 backend handle | 最多 64 shard/turn，128 shard 在途 | SSD 真完成发布节点 READY |
| 资源等待 | 已保护进度及真实在途 I/O；不再新增分配 | 其他任务、独立完成、前台消费释放 | 共享释放 epoch 改变后有限重试 |
| 完整 READY | 全部返回前缀持续受保护 | Scheduler 原生调度、各 rank H2D | 全消费者实际层/chunk 计划完成 |
| 退休/排空 | 仍在 I/O/H2D 使用的引用 | 仅回收，不新增 I/O | quiesced + owner_drained 后归还声明 |
| 隔离 | 无法证明完成的保护 | 停止该域新工作 | 整域退出并用新原生对象恢复 |

## 推进条件

域容量 C 为 floor(reservable_nodes / shards_per_block)，不含原生 load-exclusive 区。
声明 H[j] 只覆盖已选中、确认命中的批次；最大范围 M[j] 是完成当前请求所需的保守
唯一 block 集合，不是预先扣除的额度。候选资格通过如下安全性检查：

1. 假定加入当前批次，令 U 为所有仍持有声明的唯一 blocks 并集。
2. 若存在 j，使 `|U ∪ M[j]| <= C`，则它能取得完成所需余量；若 j 已 READY，
   它无需新增分配。模拟其消费并移除 j 的声明。
3. 重复到所有持有者均可完成；否则当前批次不获取保护，保留 PENDING。

重复前缀只按并集收费，删除一个请求后仍被其他请求声明的 block 不返还。
该检查是保守充分条件，可能延迟本可执行的候选；不能称为原有候选顺序完全不变。
已知真实最大前缀短于候选时收窄 M；未知超容量候选必须先查完无 pin 前缀。
真实超容量才允许发布有限安全前缀或零命中，不把一次 NoSpace 判成永久 miss。

物理节点可能仍被原生 load/dump、I/O 或消费者占用。已接受的原生任务具有完整
handles，c50 独立完成线程不依赖该等待者或 H2D FIFO 的缺失分配，因此可完成释放。
资格控制避免 Early 之间形成无安全完成顺序的部分占用。两项条件同时成立才使用
事件等待。原生正式加载未取得完整 handles 时仍有界拒绝；不修改 kv_load_failure_policy，
也不保证外部故障或未核验持有者总能恢复。

## 事件和失败

共享 epoch 在最后引用归零后递增。FUTEX_WAIT 对观察值作原子比较，避免释放先于
等待；Python release_serial 在同一 condition 内比较并等待，避免转发通知丢失。
通知只提示重试；TryGet 重新检查身份、状态、并发创建和物理容量。等待不持 bucket
锁、全局锁或 GIL。完成轮询保留 c50 既有节奏，容量重试不依赖高频轮询。

观察到 epoch 变化只开放一次有限分配机会；实际取得引用前仍处于资源等待，不恢复
后续查询。若后续批次正在激活而此前批次进入等待，归还该批尚未提交 I/O 的引用，
保留 identity/lease 后重新排队；正在执行的 I/O 继续独立回收。接受激活结果与资格
检查在同一 worker condition 内完成，不将迟到激活误发布成 READY。

资源等待通过带 generation/attempt/batch 的通知暂停 broker 的普通 PENDING 计时，
恢复后只扣除实际等待时长。此暂停不改变参与者 heartbeat、native I/O deadline 或
隔离判断；不能用固定时限把正常等待改成 miss，也不能用控制超时推断 I/O 已排空。

取消未启动批次直接释放 lease；正在 Activate 的调用结束后才报告 drained；已提交
I/O 继续持有 active Reservation，Check 真完成后唯一一次 Wait 回收。任务 latch、
控制超时或 TTL 不授权释放。未提交的失败 native 任务仍由 EarlyTaskPins RAII
标记失败并回滚。READY 的声明仅在全消费者 quiesce 和真实 drain 后删除。

## 接口和构建

新增接口均在 Early 类/绑定：ReserveDeferred、Batch.Activate/Defer/Limit/Cutoff、
ReleaseEpoch/WaitRelease、execution_request/grant、lookup_complete、resource_wait。
原生文件仅 trans_buffer.cc 增加同一映射内释放事件、初始化/挂接与最后引用通知；
没有改变任何原生私有成员访问属性，没有给 TransBuffer 增加 Early 方法。
BufferStrategy 使用现有共享映射中的事件指针；不是第二个 allocator/共享协议。

共享 header magic 3、owner VERSION 5 和扩展的 Early::Batch ABI 必须整套重建。
旧映射不能原地复用。测试/构建命令与证据见 DEFERRED_PIN_WAIT_VALIDATION.md。
