# Connector v2 / Store 接口交接

更新：2026-11-05。本文面向 Connector、Proxy 和 Store 开发者，记录当前实现、接口缺口和部署边界。

## 1. 快照与结论

| 项目 | 当前基线 |
| --- | --- |
| 实施仓库 / 分支 | `ModelEngine-Group/unified-cache-management` / `feature_ucm_v2` |
| 本文核对的代码提交 | `c00e75e54d3db7a24dc525de06cced9c07349b4f`，已推送 `qyh/feature_ucm_v2` |
| PR | [#1608](https://github.com/ModelEngine-Group/unified-cache-management/pull/1608)，目标主仓库 `feature_ucm_v2` |
| Connector 移植来源 | `dev_connector@485916d2e70e7d3640fa9574572b0f06935a4cb8` |
| CUDA MRV2 源码 | 本地 vLLM `5df63a7203cdb9328d824a27b682e77c7b9928be` |
| Ascend MRV2 源码 | 本地 vLLM-Ascend `ffca1c99bb28d9c92dd5dcf327f8b242404a5055` |

当前完成的是 Connector 到 native Store 的接口层实现，尚未形成可验收的存储闭环。
最先需要解决的是同 request/kind 多次 Prepare、Put 完整大小的准入来源、实际 Store 后端和同节点 shm 共享契约。
当前快照里的 CacheStore/PosixStore 主要入口仍返回 Unsupported，不能把其他分支已有的 shm 后端实现算成本分支已接通。

第一版限定同步预取、同步 Load 和同步 Dump；一个 DP 副本内的全部 TP/PP/PCP/DCP Worker 必须位于同一节点。
同节点并行的布局与聚合代码已实现，但没有设备或多 Worker 运行验证，不能等同于已验证支持。

详细布局和生命周期契约见 [Connector v2 与 native Store 接口](connector-v2-storage-groups.md)。

## 2. 两侧职责与现有接口

Scheduler 负责注册、查询、计划、Prepare、结果聚合和 Complete；Worker 只负责实际传输及终态回报。
Connector 负责设备布局、传输指针、字节大小和记录内 offsets。Store/Proxy 负责 shm、分层存储、准入、pin、路由及发布。

Scheduler 接口保持 [store_scheduler.h](../ucm/v2/proxy/store_scheduler.h) 的现有签名：

```python
on_new_request(request_context)
lookup_on_prefix(request_id, ucm_keys)
lookup_on_reverse(request_id, ucm_keys)
prepare_prefetch(request_id)
complete_prefetch(request_id, success)
prepare_get(request_id, ucm_keys)
complete_get(request_id)
prepare_put(request_id, ucm_keys)
complete_put(request_id, success)
on_request_finished(request_id)
```

Worker 对应 [store_worker.h](../ucm/v2/proxy/store_worker.h)：

```python
submit_prefetch(io_job_id, token)
submit_get(io_job_id, payloads, token, io_context)
submit_put(io_job_id, payloads, token, io_context)
wait(io_job_ids)
get_finished()
```

以上为逻辑签名。Python binding 的 keys 参数是连续的 16 字节 key 串，client 负责打包；Worker payload 目前仍经现有 BlockPayload binding 转换。
连续内存直接传参的优化由 Store/Proxy 后续处理，本轮未新增该 ABI，也不要求 Connector 再优化 Python 数值转换。

两类 Python client 只获取各自角色接口，但各自仍创建同一种 StoreAssembler。
native 工厂还没有只构建 Scheduler 或只构建 Worker 的角色参数。

## 3. 已实现的 Connector 行为

| 范围 | 当前代码行为 |
| --- | --- |
| 模块入口 | 显式使用 `ucm.integration.vllm.v2.ucm_connector`；不替换主线 v1 注册 |
| 模块职责 | spec/parser、实际 view、group 寻址、Store 布局分开；删除 Python 自有 proxy/开发 Store 和 UCMTransferBuilder |
| keys | Scheduler 统一生成完整 ucm_keys；前 15 字节为链式 MD5 前缀，尾字节为预留 2 bit / PP 3 bit / restore policy 3 bit；FA/WA/State 为 0/1/2 |
| PP 布局 | 各 PP stage 独立记录；UCMShard 只有 TP/DCP 坐标，记录大小及 offset 不包含其他 PP stage |
| 计划与 Prepare | 创建每个策略 plan 时分配主 job_id 并 Prepare；State running/checkpoint 已合为一个 Get plan、一次 Prepare，State dump 仍按边界独立准备；依赖下文未完成的 Store 追加能力 |
| token | AddressedToken 对业务是黑盒；Worker 使用 PreparedJob.ucm_keys 对齐 payload，原样携带整个 token，不按业务切分 token |
| 主子任务 | Scheduler 主 ID 为奇数，Worker layerwise 子 ID 为偶数；bulk/prefetch 用主 ID；保留每个子任务结果，按主 job 汇总 |
| 默认读路径 | 2026-11-06 未提交增量：明确 miss 直接 `(0, False)`，明确命中直接 `(N, False)`；仅需要预取或预取未结束时返回 `(None, False)`，预取后重新查询并进入同步 layerwise Get |
| 预取下发 | 预取请求独立写入 metadata，不依赖该请求已被调度进 forward；0-token step 也走 MRV2 Connector 回调 |
| 预取执行 | 同一 DP 内只有 `(TP=0, PP=0, PCP=0)` 提交并等待共享 shm 预取，其余 rank 回报 no-op 成功 |
| HBM Get | layerwise 在对应层使用前等待；所有 State 层在 start_load_kv 返回前等待；bulk 原地等待 |
| Dump | layerwise 按 layer ID 只提交一次，覆盖 MTP/speculative 重复访问；wait_for_save 补交未触发的层并同步排空 |
| offsets / sizes | Worker 传设备指针、每段大小、记录内 offset；完整 PP 记录大小经握手汇总到 Scheduler，不随 Submit 传递 |
| State | 同一快照分别恢复到 running 和 checkpoint 目标；保存只消费原生 boundary_state_offloads 的精确来源，不猜 block table 末尾 |
| event | 每个实际 Get/Put 提交前记录当前计算流 event，经 IoContext.prerequisite_handle 传入 Proxy；IO 明确终止后释放，未知终态保留 |
| 流顺序 | Get 前置 event 覆盖已提交的清零、CoW 和 Mamba pre-copy；Put 前置 event 覆盖此前源数据生产操作；记录失败先同步再提交 |
| 完成聚合 | 每个主 job 等全部配置 TP×PP×PCP rank，空 rank 也回报成功；DCP 是其中的子组，不再乘一次 DCP |
| Complete | update_connector_output 消费结果后，同一 request/kind 全部当前 job/rank 终态齐全，调用一次 request 级 Complete；Put success 为所有结果的合取 |
| 请求结束 | 引擎结束通知、最终 metadata 已处理、无 pending job 是分阶段推进的，最后才调用 Store.on_request_finished |
| 抢占和失败 | 抢占重建 RequestState/hash，旧 Put 等终态后按失败关闭；Get 失败回报 invalid blocks，阻止发布可能受污染的当前 Put |

普通内存抢占通常先由 build_connector_meta 感知；reset_prefix_cache 可在调度前抢占，因此 lookup 仍检查 num_preemptions。
RequestState 持有 vLLM Request 引用，保存进度读取 num_computed_tokens，跟随引擎可能的失败回滚。

元数据使用 vLLM 基类已有的 has/get/bind/clear。Worker 回调显式接收 metadata，begin_step 重置本轮 IO，clear 前检查任务已排空。
删除 _step_io_completed、metadata.no_forward 和重复 no-forward 收尾；正常及 no-forward 都由 MRV2 post_forward 统一调用 wait_for_save。

同步版本按当前窗口完成并发布，不要求等整个请求结束才 CompletePut；请求后续 chunked prefill 可继续产生新 job。
但同一步多个策略/State dump 边界就可能多次 Prepare，因此即使只做同步版，Store 的追加能力仍是联调前置条件。

## 4. Store / Proxy 尚未对齐的部分

### 4.1 阻塞：同一 request/kind 的多次 Prepare

现有 [StoreScheduler 契约](../ucm/v2/proxy/store_scheduler.h) 每个 request/kind 只允许一个未关闭窗口。
[StoreEntry](../ucm/v2/proxy/store_entry.cc) 检查 pinnedBlocks/reservedBlocks 非空后拒绝下一次 Prepare，返回 InvalidParam。
Connector 已按讨论假设 Store 将支持追加，当前实际行为是每个 plan 独立 Prepare：

```text
request R
  PrepareGet(R, FA keys)       → job 1 / token A
  PrepareGet(R, WA keys)       → job 3 / token B
  PrepareGet(R, State keys)    → job 5 / token C
    同次 Submit 包含 running/checkpoint 两份目标范围
  全部 job、全部 rank 终止     → CompleteGet(R)
```

上例反映当前 Connector 的分组方式，不是当前 native Store 已能执行的行为。当前第二次非空 Get Prepare 即可能失败。
2026-11-06 本地增量已将相同 State 源 keys 的 running/checkpoint 合为一次 Prepare，见 4.7。
InvalidParam 属于契约错误，client 不将它降级为可恢复 StoreIOError，因此不能指望自动按 miss 跑通。
Put 同样存在多策略、多 State 边界及未来跨 step 任务重叠的问题。

需要 Store/Proxy 确定并实现：

- 保留现有签名，允许同 request/kind 追加多次 Prepare；每次返回的 token 都能用于其对应 Submit。
- 现有同步版 request 级 Complete 暂时关闭当前全部同类 Prepare；目标是一次 Prepare 可独立 Complete，具体标识见 4.8。
- 不同窗口可能读取相同 key，多次 pin 的计数和释放必须平衡，不能只用 key 集合去重后漏放或提前释放。
- 当前 Complete 不含 job_id；Connector 的主 job_id 用于 Worker Submit 与结果关联，不是 native Scheduler 的窗口 ID。

同步版暂按这个边界对接。异步版若要完成一部分 plan 而不等待后续 plan，需要更细的 Complete 标识，
见 4.8；本轮不修改 native 或 client 的 Prepare/Complete 签名。

### 4.2 阻塞：PreparePut 时完整记录大小从哪里来

PreparePut(request_id, keys) 内部调用 Alloc，现有入参没有 sizes。
若后端需要按字节容量准入，不能用尚未收到的 sizes 做正确预留，也不能把 layerwise 当前层 sum(sizes) 当成整条记录大小。

需要确定完整大小是在初始化注册、请求上下文中提供，还是由后端延迟决定；这些只是待选方案，尚未实现。
Connector 在 Worker 计算 store_sizes_by_policy，经握手汇总为 Scheduler 的 `(pp_rank, restore_policy) → 对象大小`；
这两个值已编码在 key 尾字节。此前随 SubmitPut 传递的 total_size 来自 dev_connector 无 Alloc 的 Python 文件 Store，
native 没有使用者，已删除。交给 Store 的注册入口仍待 Store 侧确定。
增加 PreparedJob 或 token 字段不等于解决了 native Alloc 的大小来源。

### 4.3 阻塞：实际后端与同节点 shm 共享

本快照的 [CacheStore](../ucm/v2/store/cache/cache_store.h) 和 [PosixStore](../ucm/v2/store/posix/posix_store.h) 的 Setup/Load/Dump 等仍是 Unsupported 占位。
其他分支的 CacheStore v2 / shm 实现需要由 Store 侧合入或接到本分支，不能以当前配置示例作为可用后端。

Scheduler 和每个 Worker 是独立进程、独立 Assembler 实例。需要确认它们以相同节点/session 配置连接同一份 shm，
Prepare 预留的槽位与 Worker token 解析的槽位一致，同时不同 DP/服务实例不串用请求或资源。
token 必须可序列化，不携带 Scheduler 的裸虚拟地址；各 Worker 需在自己的地址空间解析同一份共享资源。

### 4.4 字节范围和发布语义：字段已有，实际执行待接通

[BlockPayload](../ucm/v2/common/type/block.h) 和 [Python binding](../ucm/v2/proxy/proxy.py.cc) 已有 offsets，需编译匹配版本的 native 模块。
Store 需要落实以下行为：

- 按 offsets/sizes 精确读写同 key 的非连续片段；不同层和 shard 可以分多次 Submit 提供。
- 当前 plan Prepare 包含其策略的全部参与 PP keys，Worker 只为本 PP keys 填范围，其他 keys 的 payload 为空。
- 空 pointers 必须是 no-op，不清空已有字节、大小或提前发布；当前仅 Proxy 校验 payload/token 的数量，不能据此证明后端行为。
- CompletePut(true) 在相关片段均成功后发布；false 丢弃本次写入，不破坏同 key 此前已发布的内容。
- 同 key 并发读取/覆盖、失败回滚和多个请求准入的具体隔离由后端保证，Connector 的跨 rank 成功聚合不等于存储事务。

### 4.5 event、完成与错误：透传已接，后端契约待确认

Get/Put 都传 IoContext.prerequisite_handle；StoreEntry 已转入 JobDescriptor.prerequisiteHandle。
Store 必须在真正访问设备范围之前等待 event，不能忽略该字段。
前置 event 只约束此前的计算，不保护未来对同一源范围的覆盖；同步版本在本 step 收尾前等待 IO 结束，跨 step pin 尚未实现。

GetFinished/Wait 共享消费完成结果，已被 Wait 消费的结果不会再次由 GetFinished 返回。
Connector 已缓存终态，后续 wait 不重复等同一事件。success=false 仍需表示 IO 已终止，Submit 返回失败需表示没有 IO 启动。
未知完成状态继续保留描述符和 event 并报错，不能提前 Complete、释放 pin 或复用 block。
native Status 经 binding 可能抛 UcmError；Connector 只把明确 IO/资源错误转为可恢复失败，契约错误继续抛出。

### 4.6 待对齐：同一 request 的多次 Lookup 与预取范围

`PreparePrefetch(request_id)` 不带 keys，依赖前一次 Lookup 在 Store 内保存的待预取查询。
当前 [Prefetchable 注释](../ucm/v2/store/prefetchable.h) 却要求 requestId 只关联一个固定 block 集合。
即使不考虑 PP，一个请求也会先查 FA，再查 WA/State；寻找共同恢复边界时还可能缩短范围并重复查询。
这是同步版本就必须解决的接口契约，不能只按未来异步优化处理。

建议与 Store 确认以下顺序语义，尚未实现到 native：

- 同一个已注册 request_id 可以顺序查询不同 keys，也允许某策略用更短的候选范围重新查询。
- Lookup 返回明确结果时，不占用待预取窗口；当前层命中无需预取，全部不可用则按 miss 处理。
- Lookup 返回 None 时，保存的是这一次查询中确需从下层补齐的范围。Connector 立即停止后续策略查询，调用 PreparePrefetch(request_id)。
- 从 None 到 CompletePrefetch 期间，范围固定，不能被其他不同 keys 的 Lookup 覆盖；当前 Connector 不要求同 request 并行多个预取窗口。
- CompletePrefetch 后可重新查 FA，再继续 WA/State。失败按 miss 退回计算，不得让请求永远返回 None；未知 IO 终态仍须保留引用并报错。
- 查询状态与 OnNewRequest/OnRequestFinished 的请求生命周期分开：一次查询完成不能注销整个请求；抢占和取消时需清理未使用的查询状态。

```text
Lookup(R, FA keys) = 命中
Lookup(R, State keys) = None   # Store 保存 State 的待预取范围
PreparePrefetch(R)            # 只准备这次 State 查询的待补齐部分
SubmitPrefetch → IO 终态 → CompletePrefetch(R, success)
Lookup(R, FA keys) → Lookup(R, State keys) → 汇总共同恢复边界
```

2026-11-06 未提交代码已删掉首次明确命中或 miss 后额外返回一次 None 的分支，
并删掉仅用于该延迟的 completed-request 集合；PreparePrefetch 明确失败也直接返回 0。
真实预取仍同步等待，此处不宣称完成了异步预取或 native 多查询支持。

### 4.7 需要固定：Prepare 结果与 opaque token

当前每个需要加载的策略 plan 单独 PrepareGet；不是一个 request 只 Prepare 一次：
FA、WA 各自一个 plan；State 的 running/checkpoint 是相同源 key、不同设备目标，已合入同一个 Get plan。
每个成功 Prepare 对应独立 Connector 主 job 和一份 AddressedToken，同一个 token 可被多个 rank、多个 layerwise Submit 使用。

2026-11-06 本地实现：对于同一份 State 源 keys，PrepareGet 只有一次，因为 Prepare 固定的是 Store 源，
不是设备目标。running/checkpoint 两个目标复用这一份 token，归到同一个主 job；
两份目标的所有必要层/片段、所有相关 rank 都终止后，该 Prepare 才能 CompleteGet。
plan 的 vllm_block_ids_by_group 保存 running 目标，state_checkpoint_block_ids_by_group 保存 checkpoint 目标。
Worker 为两份目标分别生成传输范围，Store client 按同一 key 把两份 pointers/sizes/offsets 合入同一个 payload；
bulk 一次 Submit 使用主 ID，layerwise 每层一次 Submit 使用对应子 ID，不会为两个目标并行复用同一个 IO ID。
因此 Get payload 允许同一源 offset 对应多个设备指针；Store 必须逐段恢复，不能按重复源 offset 去掉其中一个目标。
两个目标共用该次 IO 的前置 event；终态覆盖两份目标，任一目标失败按整个 State Get job 失败上报。
失败 block IDs 的收集也覆盖两份目标，但仍只上报原生 group 0。该增量未提交，也未做设备或端到端验证。

token 在 Python binding 适配层可见其 keys/payload 字段，仅用于序列化与还原；业务代码不解析 payload，
不读取 token.keys 决定 keys 或 offsets，也不根据 token 内容选择 shard。
payload 对齐使用单独保存的 Prepare 输入 PreparedJob.ucm_keys；这种机械复制不改变 token 黑盒定位。

需要 Store 保证或双方明确：

- 同一次 Prepare 返回的 token 在对应窗口 Complete 前稳定可用；后续追加 Prepare 不得使前一份 token 失效。
- token 可跨进程序列化，在 Worker 地址空间解析同一 shm/session，不包含仅 Scheduler 可用的裸指针。
- layerwise 和多 rank 会重复传递同一份 token，首次 Submit 不能消费掉窗口的全部资源；每个窗口的 pin/预留要覆盖全部相关 IO。
- 当前 payload 顺序来自 Prepare 输入 keys。若 Store 在 token 中重排、过滤或合并 keys，现有 binding 对齐会失效；必须保持一一对应，或另行提供明确映射，不能让业务代码猜 token。
- 空 payload 是 no-op。成功 Prepare 返回空 token 时是否仍打开窗口、是否必须 Complete，也要明确；当前 Connector 对每次成功 Prepare 都登记并最终 Complete。
- 相同 key 的多次 Prepare 可能属于不同目标或不同写入轮次，必须有正确的引用计数或窗口身份；不能只按 key 去重后共用一次释放。

### 4.8 异步 Dump 需要更细的 Complete 边界

当前 CompleteGet(request_id)、CompletePut(request_id, success) 不接受 keys、job 或 token，
Connector 只能在同 request/kind 所有 pending job、所有 rank 都终止后统一 Complete。
这是关闭当前窗口，不是等待整个推理请求结束；同步 chunked prefill 可在每轮排空后 Complete，再开下一轮。

跨 step 异步时，仅让 Store 支持追加 Prepare 还不够：

```text
step 1: PreparePut(R, keys A) → job 1，仍在传输
step 2: PreparePut(R, keys B) → job 3，仍在传输
step 3: job 1 全 rank 成功，job 3 未结束
        CompletePut(R, True) 不能只关闭 A，也不能释放 B 的预留
```

每轮轮询可以先记录 job 1 终态，但现有 request 级 Complete 无法独立发布/释放它。
持续追加未完成 job 还会延迟所有早期 job 的 Complete，不能把追加能力当成异步方案已完整。

2026-11-06 已确定的目标粒度是：一次 Prepare 可以单独 Complete，不等待同 request 下其他 Prepare。
用 opaque token 或 Store 可识别的窗口 ID 指定范围，具体接口参数仍待双方确定；
Connector 主 job_id 目前没有传入 Store Scheduler，仅增加 Complete 的 job_id 参数不足以建立对应关系。
整份 token 作为 Complete 参数仍是黑盒使用，不要求 Connector 理解其内部字段。
同 key 的不同 Prepare 也必须能区分，不能只用裸 keys 作为关闭标识。本轮没有新增任何 native ABI。

完成粒度还需区分：

- 子 IO 终态：一个层或一个 shard 的传输结束，可以释放它独占的 event/描述符；不代表整条记录可发布。
- Prepare job 完整：所有必要 Submit 已发出或明确跳过，全部相关 rank/子 IO 终态齐全；没有本地传输的 rank 才可回报 no-op 成功。
- Store Complete：关闭确定的预留/pin，并按写入结果发布或丢弃。不能因为一时没有 in-flight IO 就自动发布，后面可能还会 Submit 下一层。
- 请求结束：不再产生新计划，全部窗口和 IO 收尾后再 OnRequestFinished，与单个 job Complete 分开。

本轮确定以完整 Prepare job 为单位，不要求同一次 Prepare 内部分 keys 提前发布。
若以后增加更细的完成粒度，还需另行定义逐 key 的层/rank 覆盖范围和成功状态。

### 4.9 其余必须明确的生命周期契约

| 契约 | 对接时需要确认 |
| --- | --- |
| 写入可见性与失败隔离 | Lookup/Get 只能看到已发布的完整记录；某个写入轮次失败不能破坏同 key 的旧记录，也不能废弃其他独立窗口已经成功的记录 |
| Get 的释放 | CompleteGet 释放 pin，不承担 load 成功上报；失败仍要等真实 IO 终止后才能释放。多次 pin 相同 key 必须逐窗口平衡 |
| Prepare/Submit 拒绝 | 返回明确失败必须说明是否完全未打开窗口/启动 IO；部分成功要返回可清理的资源，不能只有一个失败异常而留下未知预留 |
| 超时与取消 | Wait 超时或取消请求已受理，不等于 IO 已停止；不能仅凭 Timeout/异常类型就释放引用或 Complete，必须有明确的终态/取消确认 |
| Complete 确认与重试 | 当前接口多为 void；若发布/释放会失败，需有确认语义。重复 Complete 是否幂等、旧 Complete 能否误关下一轮窗口，都需定义 |
| 完成队列与编号作用域 | Wait/GetFinished 共享消费终态；每个 Worker 的完成队列独立，允许各 rank 使用相同 job/subjob 编号。若后端改用共享队列，需要重新确定标识范围 |
| 设备源/目标生命周期 | event 只保证此前计算已完成，不阻止后续覆盖。异步 Dump 既要防 block 释放，也要防 SWA/State 原地改写；单纯 pin 物理 block 未必够，可能要冻结源、快照或在改写前等待 |
| 取消、抢占和请求 ID 复用 | 未终止 IO 必须排空或获得取消确认；旧轮次的失败/Complete 不能影响恢复后的新窗口，不能只靠 request_id 混在一起处理 |
| 多策略的一致恢复 | FA 可以提前形成较长前缀，但只有 WA/State 也存在的共同边界才能作为恢复命中；提前发布部分策略不能绕过联合 Lookup 判定 |

完整大小注册、同节点 shm/session、offsets、空 payload、event 等基础契约见 4.2–4.5。
本轮按 PP=1、TP/PCP/DCP 节点内讨论，不扩展跨节点设计；下节保留已有代码的并行边界说明。

## 5. TP / PCP / PP / DCP 跨机器边界

| 拓扑 | 当前处理 | 状态 |
| --- | --- | --- |
| 单节点 TP | 布局按有效 shard 划分，副本选择 writer；完成等待全部 TP rank | 代码已实现，未做设备/多 Worker 验证 |
| 单节点 PCP | 保留 PCP Worker 身份；握手仅 PCP rank0 代表各 TP 上报，IO 仍收齐所有 PCP rank | 代码已实现，未做多 Worker 验证；并非所有 cache 类型/CP 组合都已验收 |
| 单节点 PP | 各 PP key 独立、PP 数最多 8；Scheduler 查询和 Prepare 当前 plan 的全部参与 PP keys | 代码已实现，仍受上述 Store 阻塞项约束 |
| 单节点 DCP | 按原生 spec 的有效 block span 和 TP/PCP 坐标生成 DCP shard，不把 TP×DCP 当独立笛卡尔积 | 代码已实现，具体模型/布局限制仍按 parser；未做设备验证 |
| 跨节点 TP / PCP / PP / DCP | 配置检查 nnodes_within_dp != 1 时拒绝 | 当前不支持 |
| 多 DP 副本 | 单节点约束按每个 DP 副本计算 | 不是全局只能部署一台机器；不代表跨 DP 的共享命名和隔离已验证 |

跨机器的问题不限于 key 带不带 PP rank。当前只有一个 Scheduler 为当前 DP 的请求生成计划、执行 Prepare/Complete，
并假定它预留的 shm 能被该 DP 内所有 Worker 访问。远端节点的 Worker 无法直接使用本机 shm 槽位。

即使远端 Store 能 Lookup 到另一个 PP stage 的记录，也只证明数据存在，没有完成远端节点的准入、pin、预取落点和完成回收。
同理，把 TP/PCP rank 扩展到另一台机器也会破坏“单份 shm、零号 rank 统一预取”的假设。
event handle 也是 Worker 本地设备资源，不能作为跨节点可直接使用的句柄。

以后支持跨机器，需要另行设计节点级 Store 控制/路由，决定哪个节点执行 Prepare/Complete、token 指向哪个 shm，
如何按节点预取并聚合完成，以及部分节点失败后的资源回收。不能只删除配置 guard 或增加 key 字段。
PP rank 已编码进 key，UCMShard 继续只描述 stage 内 TP/DCP 分片；该拆分本身不解决跨节点控制面。

## 6. 还没有完成的 Connector / 引擎工作

- 跨 step 异步 Dump：wait_for_save 非阻塞轮询、跨 step 任务保留、源 block pin/延迟释放、结束和抢占排空尚未实现。
- 非阻塞预取：当前请求会返回 None 暂缓调度，但 Worker 提交预取后仍 wait，会阻塞该 Worker；不能称为真正异步预取。
- 跨 step 异步 HBM Load：当前 layerwise 等待只是分层同步依赖，返回的 load_kv_async 仍为 false。
- Ascend 延迟 Mamba copy 接管：当前不启用 after-load/逐层 copy 钩子，维持默认 pre-copy、双槽恢复、State 提前 wait。
- Load failure：单 group 沿用原生 policy，推荐显式配置 recompute，原生默认仍为 fail。
  2026-11-06 本地增量复用现有 patch/load_failure_patch.py 及统一加载入口（需启用 ENABLE_UCM_PATCH）：多 group 只允许 fail，
  Worker 仅上报 group 0 的 block IDs。外层多 group 检查只拒绝 recompute，内层继续复用已有
  group 0 解包 patch 和原生请求结束逻辑；没有新增独立安装入口或请求定位分支，不改引擎 checkout 文件。
  group 0 token 粒度及其他 group 的 prefix-cache 清理问题留待单独 review。
  本增量尚未提交或运行验证，不属于本文顶部的已推送快照；多 group 重算仍未实现。
- native 角色工厂拆分、连续内存 payload ABI：尚未实现；后者按讨论交给 Store/Proxy 处理。
- 真设备的 MTP 重复调用、State copy 与 slot 语义、CP 数据分片、请求取消/抢占以及多 Worker 结果聚合仍待联调验证。

## 7. 建议交接顺序与验证边界

| 顺序 | 主要责任 | 应先明确或完成的事项 |
| --- | --- | --- |
| 1 | Store / Proxy 与 Connector 共同确认 | 多 Prepare + request 级 Complete 的边界、重复 key pin 计数、完整大小的准入来源 |
| 2 | Store / Proxy | 接通真实 CacheStore/shm 和后端，确认跨进程 token、片段 offsets、空 payload、event 与终态保证 |
| 3 | 双方联调 | 先在单节点 TP1 跑通预取、重查、Get、Dump、Complete，再扩到单节点并行和 State/窗口策略 |
| 4 | Connector / 引擎 / Store | 再设计异步保存/预取的 block 生命周期；跨节点控制面作为独立后续工作 |

当前只完成了格式、静态语法和引用检查：codespell、Black、isort、Ruff、11 个模块 AST parse/compile、git diff --check。
按本轮约定，没有新增、迁移或运行测试，没有编译 native，也没有设备 IO、forward、分布式通信或端到端验证。
本文的“已实现”仅指当前代码存在对应路径，不引用 dev_connector 或其他历史分支的测试数量作为本分支证据。
