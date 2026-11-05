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
| 计划与 Prepare | 创建每个策略/State 目标 plan 时分配主 job_id 并 Prepare，plan 保存独立 PreparedJob 和 token；依赖下文未完成的 Store 追加能力 |
| token | AddressedToken 对业务是黑盒；Worker 使用 PreparedJob.ucm_keys 对齐 payload，原样携带整个 token，不按业务切分 token |
| 主子任务 | Scheduler 主 ID 为奇数，Worker layerwise 子 ID 为偶数；bulk/prefetch 用主 ID；保留每个子任务结果，按主 job 汇总 |
| 默认读路径 | 首次向引擎返回 `(None, False)`，先完成 Store 预取阶段，再查命中，返回 `(N, False)` 并进入同步 layerwise Get |
| 预取下发 | 预取请求独立写入 metadata，不依赖该请求已被调度进 forward；0-token step 也走 MRV2 Connector 回调 |
| 预取执行 | 同一 DP 内只有 `(TP=0, PP=0, PCP=0)` 提交并等待共享 shm 预取，其余 rank 回报 no-op 成功 |
| HBM Get | layerwise 在对应层使用前等待；所有 State 层在 start_load_kv 返回前等待；bulk 原地等待 |
| Dump | layerwise 按 layer ID 只提交一次，覆盖 MTP/speculative 重复访问；wait_for_save 补交未触发的层并同步排空 |
| offsets / sizes | Worker 传设备指针、每段大小、记录内 offset；Put 另传完整 PP 记录的 total_size，不用本层大小代替完整大小 |
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
但同一步多个策略/State 目标就可能多次 Prepare，因此即使只做同步版，Store 的追加能力仍是联调前置条件。

## 4. Store / Proxy 尚未对齐的部分

### 4.1 阻塞：同一 request/kind 的多次 Prepare

现有 [StoreScheduler 契约](../ucm/v2/proxy/store_scheduler.h) 每个 request/kind 只允许一个未关闭窗口。
[StoreEntry](../ucm/v2/proxy/store_entry.cc) 检查 pinnedBlocks/reservedBlocks 非空后拒绝下一次 Prepare，返回 InvalidParam。
Connector 已按讨论假设 Store 将支持追加，当前实际行为是每个 plan 独立 Prepare：

```text
request R
  PrepareGet(R, FA keys)       → job 1 / token A
  PrepareGet(R, WA keys)       → job 3 / token B
  PrepareGet(R, State running) → job 5 / token C
  PrepareGet(R, State checkpoint) → job 7 / token D
  全部 job、全部 rank 终止     → CompleteGet(R)
```

上例是需要的语义，不是当前 native Store 已能执行的行为。当前第二次非空 Get Prepare 即可能失败。
InvalidParam 属于契约错误，client 不将它降级为可恢复 StoreIOError，因此不能指望自动按 miss 跑通。
Put 同样存在多策略、多 State 边界及未来跨 step 任务重叠的问题。

需要 Store/Proxy 确定并实现：

- 保留现有签名，允许同 request/kind 追加多次 Prepare；每次返回的 token 都能用于其对应 Submit。
- 明确一次 request 级 Complete 对应本窗口内哪些 Prepare，完成后允许下一轮重新 Prepare。
- running/checkpoint 读取可能针对相同 key，多次 pin 的计数和释放必须平衡，不能只用 key 集合去重后漏放或提前释放。
- 当前 Complete 不含 job_id；Connector 的主 job_id 用于 Worker Submit 与结果关联，不是 native Scheduler 的窗口 ID。

若 Store 最终选择独立 job/window Complete，必须双方再明确新契约；本轮没有擅自改签名。

### 4.2 阻塞：PreparePut 时完整记录大小从哪里来

PreparePut(request_id, keys) 内部调用 Alloc，现有入参没有 sizes；完整 total_size 要到 Worker SubmitPut 才出现。
若后端需要按字节容量准入，不能用尚未收到的 sizes 做正确预留，也不能把 layerwise 当前层 sum(sizes) 当成整条记录大小。

需要确定完整大小是在初始化注册、请求上下文中提供，还是由后端延迟决定；这些只是待选方案，尚未实现。
当前 Connector 已计算 store_sizes_by_policy，并在每次 Put 的非空 payload 提供完整 PP 记录 total_size。
增加 PreparedJob 或 token 字段不等于解决了 native Alloc 的大小来源。

### 4.3 阻塞：实际后端与同节点 shm 共享

本快照的 [CacheStore](../ucm/v2/store/cache/cache_store.h) 和 [PosixStore](../ucm/v2/store/posix/posix_store.h) 的 Setup/Load/Dump 等仍是 Unsupported 占位。
其他分支的 CacheStore v2 / shm 实现需要由 Store 侧合入或接到本分支，不能以当前配置示例作为可用后端。

Scheduler 和每个 Worker 是独立进程、独立 Assembler 实例。需要确认它们以相同节点/session 配置连接同一份 shm，
Prepare 预留的槽位与 Worker token 解析的槽位一致，同时不同 DP/服务实例不串用请求或资源。
token 必须可序列化，不携带 Scheduler 的裸虚拟地址；各 Worker 需在自己的地址空间解析同一份共享资源。

### 4.4 字节范围和发布语义：字段已有，实际执行待接通

[BlockPayload](../ucm/v2/common/type/block.h) 和 [Python binding](../ucm/v2/proxy/proxy.py.cc) 已有 offsets、total_size，需编译匹配版本的 native 模块。
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

预取还需确认 request 级 Lookup 登记范围到 PreparePrefetch 的语义，以及跨 stage/策略的重复 Lookup 如何维护范围；PreparePrefetch 当前不显式带 keys。

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
- Load failure 的完整恢复：本地 vLLM 默认 kv_load_failure_policy 为 fail；显式 recompute 时单 group 可回滚，多 group 同步失败不支持自动恢复，会报错。
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
