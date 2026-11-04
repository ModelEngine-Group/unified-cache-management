# Connector v2：按恢复策略组织对象

2026-10-29 更新，位于 `refactor/connector-v2-storage-groups@5fafa6a6` 的未提交工作区。覆盖此前全局 piece 布局原型；主线 connector 与两个引擎源码未修改。

## 文件组织与调用流程

| 文件 | 职责 |
|---|---|
| `ucm_connector.py` | vLLM SPI 入口，根据角色把回调转交 scheduler 或 worker |
| `connector_scheduler.py` | 最小rank/fingerprint握手、请求状态、hash/lookup、load/dump 计划、调度 metadata、SaveJob 与跨 rank 保存结果聚合及 commit |
| `connector_worker.py` | KV 注册、PP metadata交换、直接构造 Worker 握手、UCMTransferBuilder、load/dump 提交与等待、SaveResult 与 worker metadata |
| `layout/store_layout.py` | UCMShard、分片/副本映射、PPStageMetadata和GroupStoreLayout，编译完整block/chunk的sizes与offsets |
| `runtime.py` | 两端共用的配置/fingerprint、UCMRuntimeContext、ParallelRank/ParallelTopology、仅含rank/fingerprint的UCMStorageHandshake、格式版本和固定策略tag编码 |

Lookup已合入Scheduler的 `_lookup_hit_tokens`，删除独立 `UCMLookup` 类和 `_lookup` 对象；独立 `UCMDispatcher` 及 dispatcher 对象也已删除，计划计算归 Scheduler 的 `_build_request_plan`；`UCMTransferBuilder` 保持独立类，集中在 worker 文件。删除独立的 `ucm_scheduler.py`、`ucm_lookup.py`、`ucm_kv_cache.py`、`ucm_worker.py` 和只有两个数据类的 `save.py`，不保留旧模块转发壳。公共 Layout、Runtime、Proxy/Store 与 key 编码继续按职责组织。`RequestSnapshot`已删除，计划方法直接接收RequestState和本轮SchedulerOutput的block tables；不修改请求状态，所有计划构造成功后Scheduler才统一推进。RequestState不再持有重复block tables。

公共分组直接使用`dict[RestorePolicy, tuple[UCMKVCacheGroupSpec, ...]]`。每种存在的策略对应一个完整对象，不再有StorageGroupLayout、storage_group_id、LookupEntry或编号映射。Worker的TransferBuilder初始化时按策略取得GroupStoreLayouts，预先累计各native group的起点和每种策略的对象总字节数。

握手只确认完整且不重复的rank集合与fingerprint。删除logical_spec、分组数量、重复版本和固定writer字符串确认；Scheduler使用已有_expected_ranks，不解释引擎外层字典key。查询策略目录由公共spec直接初始化，不从Worker收集。RequestHasher仍在握手完成后初始化，只保留返回的 `_chunk_hasher` 闭包；删除未消费的 `_request_hasher` 和 `hash_seed` 字段。

`ucm_chunk_id.py`已合入runtime；`make_common_ucm_chunk_ids()`取15字节common hash，`make_ucm_chunk_ids(common_ids, restore_policy)`追加固定1字节tag：FA=00、WA=01、State=02。缺少策略不重新编号。开发阶段没有上线数据兼容要求，本次保持版本6，不新增旧key隔离或迁移代码。

Scheduler 流程：收齐 Worker 握手后初始化 hasher → `get_num_new_matched_tokens` 调用内部查询方法 并记录请求状态 → `build_connector_meta` 调用内部 `_build_request_plan` 生成 load/dump plans，分配 SaveJob，再推进请求状态 → metadata 交给 Worker → `update_connector_output` 聚合每个 attempt 的 rank 终态；请求结束且结果齐全后 `commit(True)` 发布，失败且全部终态齐全后 `commit(False)` 废弃。

Worker 流程：`register_kv_caches` 构建本地 HBM Layout、交换 PP metadata、完成 StoreLayout 并创建 TransferBuilder → 返回握手 → 每步 `bind_connector_metadata` → `start_load_kv` 构造并提交 load，bulk 等待或按层等待 → `save_kv_layer` / `wait_for_save` 构造并提交 dump、等待终态 → `build_connector_worker_meta` 返回保存结果和加载失败请求 → 清理本步 metadata。Worker 不执行跨 rank 发布。

以下 Layout、lookup 和生命周期行为不因本次文件合并而变化。

本次评审清理：公共 `UCMKVCacheSpec` 不再持有 `scheduler_block_size`，握手也不传该字段；它仅在 parser 作为初始化诊断和无存储组默认退路使用。Runtime将模型/config、设备、dtype、chunk、完整ParallelTopology、版本和layerwise模式放入同一个schema，规范化JSON后生成完整SHA-256 fingerprint。删除ParallelTopology.namespace和字符串后缀拼接，初始化INFO日志单独打印配置。公共逻辑 Spec 不持有 TP 字段；Worker 的 GroupStoreLayout 在 PP 模板交换后确定整组 `tp_shards`，接受组内冗余。`group_infos` 统一改名 `group_specs`。

`parse_kv_cache_config` 与原生 Scheduler 一样调用 `dcp_world_size_for_kv_cache_spec(native_group.kv_cache_spec, actual_dcp_size)`，获得组级有效 DCP size，再计算 `block_token_span`。删除固定传2的探测函数；`dcp_sharded` 表示当前 group 实际开启 DCP 分片，DCP1 和 State 为 False。SWA 的有效 DCP size 为1，因此保持本地跨度；删除按进程DCP拒绝整个含SWA模型的检查。

删除 `layout_debug`、`LAYOUT_DEBUG`、`UCM_V2_LAYOUT_DEBUG` 和 stderr 打印。注册/parser 初始化摘要使用标准 `logger.info`；热路径诊断使用 `logger.debug`，不增加 INFO trace；真实IO失败仍保留原有warning。旧 `prefix_replay_tokens`/小chunk取整warning仅是推测性提示，不参与调度；当前本地vLLM/Ascend 0.30源码没有对应字段/取整路径，已删除该分支，不把它当成引擎保证。

只有持久SWA计算 `tail_tokens`；FA、State和非持久group统一为None。SWA的0表示没有需要保留的tail；是否持久化只由is_persistent判断。内部common ID来自固定15字节截断，追加策略tag时不再重复遍历校验长度。

## 布局与初始化

| 组件 | 承载内容 |
|---|---|
| `UCMKVCacheGroupSpec` | 原生 `block_size`、派生的 `block_token_span`、恢复策略；公共 spec 统一派生 `blocks_per_chunk` |
| `KVCacheGroupLayout` | 本 rank 实际 HBM views、segment 几何、ptr 和 block ID 寻址 |
| `UCMShard(pp_rank, tp_shard, dcp_shard)` | 原生 group 的数据身份；不是新的 Worker 轴 |
| `GroupStoreLayout` | 一次构建完成的本 rank chunk IO sizes/offsets、UCMShard/副本域和总 bytes；引用本地 HBM，不保存中间模板 |
| `UCMTransferBuilder` 初始化字典 | groups_by_policy、group_byte_offsets和record_nbytes；按策略编排native groups，不改变各group的TP/DCP规则 |
| `UCMKVCacheLayout` | 注册完成后的 HBM/StoreLayouts和层索引，不计算缓存身份摘要 |

注册入口在 `UCMConnectorWorker.register_kv_caches`，顺序为：

1. `UCMKVCacheLayout(...)` 只构造实际 HBM 和层索引；不创建半成品 StoreLayout。
2. `layout.export_layout_metadata(tp_size)` 调用各非空持久 group 的 `KVCacheGroupLayout.export_metadata(tp_size)`，导出 `GroupLayoutMetadata`，只包含`sizes`、`states_per_block`、`supports_partial_tokens`和实际`tp_shards`。没有digest、地址、层名/shape描述；空本地PP group保留空元数据位置。
3. Worker 中的 `exchange_pp_metadata((pp_rank, metadata, error), topology)`：PP1 本地返回；PP>1 仅一次 `get_pp_group().cpu_group` 的 `all_gather_object`，上述所有字段一起交换。空本地 group 和构建错误也参与，交换后统一失败；PP1 的构建错误原样抛出。不使用包含 DP 的 world group，不新增 TP collective。
4. `layout.build_store_layouts(...)` 对各 group 调用一次 `GroupStoreLayout.build`：跨非空 PP stage 统一组级 TP 规则 → 按有效 `(tp_shard,dcp_shard)` 重复各 stage 的完整 sizes → 拼出完整逻辑 block 并确定本 rank 的区间/副本域 → 按 chunk 展开并选择本 rank 的最终 sizes/offsets → 创建完整 StoreLayout。中间 sizes/states 仅是初始化局部变量，不保存 `local_template`/`block_template`，没有 `build_local`/`bind_shards`/`compile_chunk` 半成品链路。
5. TransferBuilder初始化时按FA、WA、State取得各native group的StoreLayout，累加group_byte_offsets和各策略record_nbytes；UCMKVCacheLayout只保存native group布局，不再建立另一层分组对象。Worker不生成格式digest。

最终Store字段明确区分HBM与外存：`segment_byte_offsets_in_block`是每个本地segment的固定块内偏移，`chunk_segment_bytes`是本rank的IO字节数，`chunk_segment_offsets`是相对当前native group在chunk内起点的外存偏移，三者均为[block, local segment]矩阵。TransferBuilder再添加该恢复策略对象内的group起点。`block_segment_offsets`是一整个分布式block内本rank各segment的基准位置（1-D）；`whole_block_nbytes`是完整分布式block总bytes，`whole_chunk_nbytes`是该native group本chunk/window实际存储总bytes，二者均包含全部有效PP/shards。FA小chunk在运行时追加token_starts对应的HBM偏移；SWA固定tail裁剪在segment_byte_offsets_in_block中。字段改名和错误文案合并不改变文件格式，版本保持6。

HBM注册阶段仅在单个layer_name的LayerView内归并segments：只在同backing且block stride相同的范围间去重重叠/合并相邻，不跨layer_name。A5 DSV4的联合full view不再重复计入payload。归并后的范围按完整block访问，LayerView的`supports_partial_tokens=False`，Group对所有LayerView取all；小chunk和部分SWA tail使用现有Store初始化检查拒绝，State始终是完整快照。layer ID仅用于选层，删除跨名称owners集合。不同backing按首次组件顺序排列，同一backing内按offset排列；不引入运行时IO合并。当前为开发期，不增加旧key迁移或格式版本递增。

空 PP group 保留 StoreLayout/公共 spec 槽位，本地没有 HBM layout 或假 ptr。不同 PP stage 的层数、tensor 数量、单段大小可不同；不以代表层补齐其他 stage。

`UCMShard`、`shard_replicas`、`PPStageMetadata` 与 `GroupStoreLayout` 统一位于 `layout/store_layout.py`；实际 PP collective 位于 `connector_worker.py`，布局模块只计算模板。删除独立的 `storage_layout.py`。

已删除 `MemberTemplate`、`StageDeclaration`、`StoragePiece`、`GlobalStorageLayout` 和 `describe_stage`。不恢复旧的 UCMSharding 类。导出元数据只在注册阶段使用，不成为一串运行期布局组件。

## chunk_size 与切片

配置和公共字段统一为 `chunk_size`，含义是一个 UCM key 覆盖的**全局逻辑 token 数**；不是 bytes 或单 rank token 容量。CLI 为 `--chunk-size`，工具环境变量为 `UCM_MODEL_CHECK_CHUNK_SIZE`。开发期不保留旧配置名/CLI/env 兼容，也不增加迁移 guard。

`parse_kv_cache_config(..., dcp_size=...)` 先确定每个 group 的最终 `block_token_span`，再选定一次 chunk size。删除 Runtime 的 `apply_context_parallel` 二次修改 spec/默认值/校验；`blocks_per_chunk` 随最终 spec 一次派生。

未指定时，默认值为**所有实际参与存储的 group 的 block_token_span 的最小公倍数**。非持久化、零 tail 不存储的 group 不参与；没有存储对象时沿用引擎 scheduler 粒度。State 自身 block span 参与共同边界对齐，每 chunk 仍仅存一个精确边界快照。

选定的 chunk size 必须覆盖每个参与持久化的 WA group 的 `tail_tokens`（解析后的有效保留窗口）。默认值和显式值都在 parser 初始化时检查；过小直接报错，不自动放大。每个完成的 chunk 因此都足以形成完整 WA tail，scheduler 不再跳过早期不足窗口的边界。

显式 C=chunk_size、S=group.block_token_span：

| 条件 | 逻辑约束 |
|---|---|
| 非 DCP FA，C≥S | C % S == 0 |
| 非 DCP FA，C<S | S % C == 0，且 C 对齐每个成员的 tokens_per_state |
| State、非零 tail 的 SWA | C % S == 0 |
| DCP>1 | C % S == 0，不支持 DCP block 内切分 |

逻辑约束只在 parser 检查一次；Worker 注册时仅检查实际 HBM segment 的部分访问能力和物理 entry 对齐。Transfer 生成/IO 时不重复做大小合法性判断。State/SWA使用各自有效DCP1的本地跨度，不因其他group开启DCP而拒绝。不增加引擎已保证的并行维度等检查。

例如 FA 的原生 block sizes 为128/256，DCP1 默认 C256，本地分别引用2/1页；DCP8 默认 C2048，本地仍分别引用2/1页。显式 C64 在非 DCP 的逻辑上可行，需实际 HBM 支持切片；加 State256 后 C64/128 不可用。

Layout 初始化只有一条路径：构造本地 HBM → 提取完整 block 元数据 → 必要的 PP-only 交换 → `build_store_layouts` 一次创建最终布局。删除 `defer_store_layout` 参数和构造函数内的隐式 TP1 编译分支；独立组件测试也显式走相同流程。

## TP、DCP 与本地页数

`block_token_span = block_size * effective_dcp_size`，只对原生 DCP 分片 Attention 放大；State 保持原生跨度。原生 `block_size` 不被改写，本地 bytes 取真实 views。

例如 B128、MLA TP8/DCP8：span1024。chunk1024 每 rank 只提供一个本地 block；chunk2048 提供两个。不同 DCP ranks 的本地页拼成完整分布式 block，不把本地页大小或页数再乘 DCP。interleave=1/128 不改变完整页恢复边界。

每个本地group先将成员原生TP声明解析为实际整数：MLA/`tp_replicated=True`为1，未声明上限的普通Attention/State为实际`tp_size`，有`max_tp_shards`则取其与TP size的较小值。组内取这些实际数量的LCM；PP交换后，对同一native group的所有非空stage再取LCM。`tp_shards`始终是整数，不使用None表示混合情况。例如TP12、声明2和3，整组采用6个shard，每shard有2个副本；组内成员仍随完整group保存，可以冗余，不恢复成员级映射。

`GroupStoreLayout.build` 对每个 PP stage 只调用一次 `shard_replicas`，枚举真实 TP×PCP 的有效 `(tp_shard,dcp_shard)` 组合，并为每个有效 shard 复制**整个 stage 的 sizes 模板**。不同PP不归为副本，DCP不是新增Worker轴，不构造TP×DCP笛卡尔积。MLA TP8/DCP1、2、8仍分别有1、2、8份数据，副本8、4、1。

当前 rank 的每个 native group 只对应一个 UCMShard、一个本地连续模板起点，以及一个副本 index/count。`writer_rows` 按绝对chunk index选择整组所有segments，layerwise再筛当前层；删除成员级global_template、local_columns/identity字典、writer_columns以及TransferBuilder按列组合分批。HBM layer/cache→segment映射只用于按层寻址，不参与分片策略。

同native group复制MLA+独立TP成员会重复存MLA部分，Ascend replicated indexer同DCP主KV也继续接受冗余。所有实际IO仍按真实HBM sizes/stride，不把这些冗余误当作额外本地页。PCP映射/原有支持边界不变，未声明真实模型验收。

FA/MLA与State/SWA混合组不因进程DCP开启而一概拒绝：FA使用有效DCP跨度，State/SWA保留本地跨度；实际DCP分片页不做块内切分，WA仍用tail模板，State每边界只存一个完整快照。引擎自身维度/hybrid 组合限制仍由引擎校验。

## 单文件格式

每个 `common_id + restore_policy.tag` 对应一个完整 UCM 对象，文件后端使用 **一个物理文件** 保存所有 PP/TP/DCP 范围，方便 lookup 和后续 GC 以对象管理。

格式顺序固定为 **原生 group → 时间 block → UCMShard → 本地成员/segment**。采用紧凑有效 payload，bulk 和 layerwise 使用同一套外存 offsets，剔除 HBM 页间 padding；HBM 的 BlockFirstView 仍可描述物理连续性，不决定文件格式。

完整页 FA group：令 F 为拼好所有有效 shards 后的单 block bytes，N=chunk/span，则占 N×F bytes；本地 segment 的位置为 `group_base + block_index*F + global_segment_offset`。WA 的部分首 block 和 State 快照分别在 GroupStoreLayout 编译，不能直接套重复 N 次的规则。

例如 MLA PP2×TP2，两个 stage 本地页大小 A/B：DCP2 的单 block 为 `PP0/dcp0(A) | PP0/dcp1(A) | PP1/dcp0(B) | PP1/dcp1(B)`，span=2B_native；chunk 翻倍时重复整个模板。DCP1 时同 stage TP ranks 是副本，只存 `PP0(A) | PP1(B)`。

`UCMTransferBuilder` 选择最终模板列/行，按计划中的本地 block IDs 解析 ptrs，加上同策略对象内的 group 起点后组装二维 Transfer。不重算段大小/前缀和，不比较本地和全局模板，不维护 piece 对象。

## 握手、key 与 Scheduler

Scheduler的Runtime fingerprint是缓存身份的唯一来源：现有模型/config/dtype/布局模式、chunk、并行拓扑和PP partition之外，`stored_groups`按原生顺序记录每个持久group的`(group_id, block_size)`。scratch非持久group、HBM地址和容量不影响身份。策略分组、shard数量、sizes和offsets依固定算法推导，不再额外hash。

`UCMStorageHandshake`只有rank与fingerprint。Scheduler检查rank集合完整且不重复、fingerprint一致，再使用`RequestHasher(context.fingerprint)`；重复有效握手不重建hasher，不改变请求状态。没有公共spec逐字段回传、分组数量或Worker物理格式hash链路。

PP交换固定TP/PCP坐标、仅跨PP：本地vLLM rank顺序为DP×PP×PCP×TP。DP1、PP2×PCP2×TP2时，四个PP CPU组分别为`[0,4]`、`[1,5]`、`[2,6]`、`[3,7]`。各Worker只在自己的两人组内all-gather两个stage；收到各stage模板后在本地复制有效shards。不存在八人world all-gather。

`UCM_CHUNK_ID_VERSION=6`保持。当前仍处开发阶段，不要求每次重构都递增版本，也不维护旧key兼容。

key固定为**15 bytes common chunk hash + 1 byte restore-policy tag**，不含PP/TP/DCP后缀。固定映射为FA=00、WA=01、State=02，与本模型包含哪些策略无关。Runtime仍保留模型/config/revision/chunk/拓扑等静态身份。

Lookup按策略查询完整、已发布对象；FA连续前缀、WA/State共同恢复边界语义保持。各native group可以有不同tp_shards/DCP规则，Worker各自展开后写到同一策略对象，不要求整个对象有统一shard划分。查询和请求状态均由ConnectorScheduler管理，计划方法只读取本轮传入的状态和block tables。查询先查FA连续前缀，再反查WA/State；存在多种边界策略时反复缩小终点，直到它们在同一边界命中，不能仅取各自最后命中的最小值。保留无FA、HBM已命中部分、最后token重算和存储错误降级。

## State 加载与引擎拷贝

State 恢复统一把同一份外存快照加载到 checkpoint 和 running 两个槽，删除 `load_running_state_directly` 参数及设备/runner 分支。命中边界决定外存 key，`load_end_token + scheduled_tokens` 仅用于选择本轮 running 的 HBM 目标；checkpoint 目标按 `load_end_token` 选择。MRV2 使用这条统一路径，不额外接入 Ascend MRV2 延迟 State copy 钩子。`boundary_state_offloads` 只用于保存，由引擎明确交出 `(group_id, block_id, boundary_tokens)`，不用于决定读取目标。

`build_dispatch_plans` 用必填 `is_load` 区分加载与保存，加载时的 `scheduled_tokens` 用于定位 running 槽。内部 `state_target_tokens` 仅表示 HBM 目标位置。原生 State 快照边界可以比 UCM chunk 更细；当前按共同 chunk 边界生成 key 和 lookup，因此只保存对齐 `chunk_size` 的快照，不把其他边界向前或向后取整。

State copy 由引擎管理，UCM 已删除 prepare/finish 钩子、copy 能力声明、缓冲区及按层 copy 调用。当前 CUDA 与 Ascend 默认 MRV2 Mamba 路径均在 connector load 前执行 precopy；UCM 随后恢复两个槽。bulk 等待实际 load 完成；layerwise 在 start_load_kv 返回前等待全部 State 层的 load，其他缓存仍由 wait_for_layer_load 等待。未知 IO 终态保留任务并报错，不能提前释放源/目标描述符。既有 MRV1 copy 钩子记录仅作历史资料，不能作为当前接线说明。

## 保存与 commit

```python
SaveJob(request_id, attempt_id, ucm_block_id, restore_policy, chunk_index)
SaveResult(attempt_id, rank, success)
```

Scheduler 为每次对象保存分配独立 attempt_id，同时作为后端 staging ID；计划不携带物理 bytes、required pieces 或 writer 列表。Worker 从本地最终 StoreLayout 获得对象总 bytes，并写同一隐藏 staging 文件的独立范围。

每个 Worker 等本地 bulk/layerwise IO 全部终止后，对收到的 **每个对象** 返回一条 rank 终态；空 group、未选中 writer 的 rank 也返回 True。Worker 不调用 commit，不持有跨 step 发布 ledger。

Scheduler的 `PendingSave(job, completed_ranks, request_finished, failed)` 按attempt收集终态rank集合，以sticky failed保留失败；不再重复保存每rank布尔值。`_publish_ready`更名为 `_commit_ready_saves`，request_finished和Worker结果到达均触发结算。expected ranks 来自本 DP 内 TP×PP×PCP：

| 条件 | 动作 |
|---|---|
| rank 结果没收齐 | 继续等，不能删除 staging |
| 全部成功，但请求未结束 | 等请求结束 |
| 全部成功且请求结束 | `commit(True, ...)` 发布 |
| 全部终态且任一失败/作废 | `commit(False, ...)` 废弃本次 attempt |

重复 True 不增加覆盖，False 不被后续 True 覆盖，域外 rank 是协议错误。finished/preempted 清理活跃请求不能丢 pending saves，晚到 ACK 仍可聚合。不同对象独立完成。

唯一后端完成接口仍为 `UCMProxy.commit(success, staging_ids, object_ids, chunk_indices)`；Store 无 chunk_indices。False 仅清本次 attempt，不删除已发布同 key 或其他 attempt；True 不可变发布且幂等。不同请求相同 key 仍使用独立 attempts。

## 后端与失败边界

FileTestStore 共享确定性的隐藏 attempt 文件，各 rank O_CREAT/O_RDWR 写不重叠区间，不 O_TRUNC。所有 rank 终态齐全后，Scheduler 以 os.link 原子 create-if-absent 发布。文件长度不代表完成。带文件 backend 的 MemoryTestStore 下传共享 attempts，仅提升完整发布记录。Factory 继续要求进程间共享的最终 Store。

lookup IO 错误按 miss；load 失败反馈无效 blocks 并抑制保存；dump/commit 的终态存储错误 best effort。未知 IO 完成状态仍 fatal，保留任务/描述符，阻止 metadata 清理或下一步绑定。Scheduler shutdown 不强删未确认终止的 staging。

这版没有新增崩溃 staging 自动 GC/恢复、fsync 耐久性、跨对象事务或生产 Store。State copy 接管已移除；设备、MRV2真实引擎 forward、模型和生产代理仍待验证。

## 验证

本轮组件**352 PASS + 50 subtests**，toolkit**102 PASS**。包括State两槽CPU恢复、延迟IO等待和两槽加载失败上报，以及WA chunk覆盖有效窗口的初始化约束。删除已移除copy接管接口对应的9项旧测试组合，保留4项两槽恢复组合并增加2项bulk/layerwise加载失败检查。测试包含真实8进程Gloo，未执行设备/模型forward。

真实8进程Gloo继续覆盖TP2×PP2×DP2及TP2×PP2×PCP2的PP通信、不等/空组/错误传播。未验真实引擎/设备/模型forward或生产Store。投机解码的恢复边界、确认范围及draft身份仍待后续处理，本轮没有扩大支持声明。实施记录见工作区 `reports/connector-v2-design-review-20261023/SCHEDULER-SIMPLIFICATION.md`。
