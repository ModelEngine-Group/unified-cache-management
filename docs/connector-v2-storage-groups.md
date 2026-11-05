# Connector v2 与 native Store 接口

实施分支：`feature_ucm_v2`，移植基线为 `8e03c9d85fba6e803fae3824b89444c703066113`。
connector 源自 `dev_connector@485916d2e70e7d3640fa9574572b0f06935a4cb8`。
本轮只移植接口，不沿用旧测试结论；按用户要求，不新增、迁移或运行测试，不做端到端验证。

## 文件职责

模块直接位于 `ucm/integration/vllm/v2/`，删除 `layout/` 子包。

| 文件 | 职责 |
| --- | --- |
| `ucm_connector.py` | vLLM 公开回调，分别转交 Scheduler/Worker |
| `connector_scheduler.py` | 请求注册、跨 PP lookup 汇总、计划、Prepare/Complete |
| `connector_worker.py` | KV 注册、传输范围组装、提交、等待、终态回报 |
| `kv_cache_spec.py` | 公共 spec、Worker layer/descriptor 声明、原生 config parser |
| `kv_cache_layout.py` | 已解析声明与实际 tensor 绑定，汇总本 stage 布局 |
| `group_layout.py` / `cache_view.py` | HBM 几何、block 寻址和 layer 内 segment 归并 |
| `store_layout.py` | stage 内 TP/DCP shard、副本与对象内字节布局 |
| `runtime.py` | Runtime、并行坐标、fingerprint、PP 对象 key 与握手 |
| `store_client.py` | 按角色拆分的 native client、可序列化 token、完成事件消费 |

不迁入 Python PipelineProxy/ParallelProxy、Memory/File 开发 Store 或 model-check 工具。
`store_client.py` 不负责路由、容量、promotion/writeback、暂存或发布；这些由 `ucm/v2` 负责。

`StoreSchedulerClient` 只获取 Scheduler 接口，显式提供请求注册、
`lookup_on_prefix/reverse`、`prepare_get/put/prefetch`、`complete_get/put/prefetch`。
`StoreWorkerClient` 只获取 Worker 接口，显式提供 `submit_get/put/prefetch`、
`wait`、`poll_finished` 和排空；调用点不再使用通用 `call/prepare/submit`。
可恢复的 native 异常在明确的方法内统一转换，等待异常仍不能当作 IO 已终止。
两类 client 各自持有 StoreAssembler 保证借用接口存活；native `Create(config)`
目前仍创建同一类组合对象，本轮没有增加 native 按角色创建的工厂。

声明解析与实际视图的边界：`kv_cache_spec.py` 中的 `parse_layer_specs` 解析
`TensorDescriptor`、存储粒度、view order 和层号，不访问实际 tensors；这些 Worker
声明不塞入 Scheduler 共用的 `UCMKVCacheSpec`。`kv_cache_layout.py` 仅做绑定和汇总，
实际 stride/segment/pointer 转换仍在 `cache_view.py` 与 `group_layout.py`。
删除未使用的 `export_layout_metadata`、Layout 构造参数 `use_layerwise` 及旧包重导出。

逻辑计划名为 `UCMDispatchPlan`（可含多个 native group），逐请求下发信息为
`RequestMetadata`。Worker 汇总主 job 使用 `JobResult`，与 native 子任务的
`IoResult` 区分；客户端 `poll_finished` 消费并记录完成事件，native 方法仍是
`get_finished`。原 `attempt_id`/SaveJob 已在主 job 改造时移除，不再引入同义类型。
Scheduler 的 PendingJob.operation 使用 JobOperation.GET/PUT/PREFETCH 枚举。
Scheduler 下发的 connector metadata 由 KVConnectorBase_V1 唯一持有，复用其 has/get/bind/clear 接口。
UCMConnector.bind 只额外调用 Worker.begin_step 重置本轮 IO 状态并登记 PrepareGet 失败，
clear 前调用 check_io_drained 保护未终止的 IO。Worker 回调显式接收 metadata，
不再保存第二份 _connector_metadata。无额外行为的 handle_preemptions 继承基类默认实现。

删除 `UCMTransferBuilder`。`UCMConnectorWorker._build_transfer_batches` 直接接收
单个 `UCMDispatchPlan`，按计划与本地 Layout 生成 `TransferBatch`，不再临时包装完整
`UCMConnectorMetadata`、再按请求拆分或合并描述。`TransferBatch` 明确携带
`ucm_keys`、`device_ptrs`、`byte_sizes`、`store_offsets` 和完整对象的
`store_sizes`；矩阵维度是 `[object, segment]`。多个 group/目标范围按 key
直接追加到 native BlockPayload，一次 bulk Submit 仍覆盖同一 Prepare 的全部范围。

引擎钩子的 `layer_name` 在 Worker 入口转换为 `layer_id`。load/dump 内部统一按
model layer ID 选择该层全部 cache segments，同一个 layer ID 每个 step 只 dump
一次，`wait_for_save` 按 ID 补交未触发钩子的层。删除 layer_name/layer_id 二选一
参数和重复 group 一致性检查，直接使用 Scheduler 按同一顺序生成的 group ID/block 列表。

## PP 与分片

PP 分开存。key 由 common hash、PP rank 与恢复策略确定，保持 16 字节；
Worker 只处理本 stage 对象，offset 和完整对象大小都不包含其他 stage。
前 15 字节直接取 RequestHasher 的链式 MD5 摘要前缀，不再混入 PP rank 做二次 SHA256。
最后 1 字节按 `[预留 2 bit][pp_rank 3 bit][restore_policy 3 bit]` 编码：
高 2 bit 置零，`suffix = (pp_rank << 3) | policy_id`；FA/WA/State 的 policy_id
分别是 0/1/2。配置初始化时限制 PP stage 数量不超过 8，pp_rank 范围为 0～7。
同一 chunk 在不同 PP stage 和策略下共享前 15 字节，仅尾字节不同。
完整 key 在 Scheduler 建计划时生成，存放在 `UCMDispatchPlan.ucm_keys_by_pp_rank`；
Prepare 使用这份计划按 PP 顺序展开的 key 集合。Worker 仅选择自身 PP stage 的现成 key，
不再调用 `make_ucm_chunk_ids` 或重新拼接 PP/policy。
一次 Prepare 包含当前 plan 的恢复策略在所有参与 PP stage 上的 ucm_keys。
Worker 只为本 PP stage 提供实际传输范围，其他 PP stage 的 payload 保留为空。
删除 PP 布局 all-gather。`UCMShard(tp_shard, dcp_shard)` 不再包含 PP；
`ParallelRank(tp_rank, pp_rank, pcp_rank)` 仍保留完整 Worker 身份。
每个 stage 在握手中声明实际存储的策略，空 stage/group 不产生空对象。
Scheduler 对全部参与 stage 查询 FA 共同前缀、WA/State 共同恢复边界。
hasher 在 Scheduler 初始化时按其 fingerprint 创建，完整 key 由 Scheduler 统一生成。
首次 lookup 创建 RequestState 并写入 common_chunk_hashes，后续预取轮询和建计划直接复用。
抢占时清除旧 RequestState，恢复查询时按当前 token 序列重建；请求结束后清理。
握手仅声明各 PP stage 的实际存储策略，每个 TP 坐标由 PCP rank 0 代表上报。
这样 CUDA 的 (pp_rank, tp_rank) 握手 key 保持唯一，不要求每个 PCP 副本都握手。
IO 完成聚合仍等待配置中的全部 TP×PP×PCP rank，包括没有参与握手的 PCP 副本。
Worker 本地 fingerprint 不参与握手比较。
本地 vLLM 5df63a72 的 NIXL 已区分副本与分片，PCP 副本仅由 rank 0 握手，
PCP+DCP 分片则分别暴露传输 rank，见 [PR #56645](https://github.com/vllm-project/vllm/pull/56645)。
上述接口核对基于固定源码和 PR，尚未做多 Worker 运行验证。

部署约束：本轮每个 DP 副本的 TP/PP/PCP/DCP Workers 都在单机，跨机并行暂不考虑。
配置初始化时拒绝 `nnodes_within_dp != 1`。尤其跨机 PP 尚未接通远端 shm 的
Prepare/Complete；远端 Lookup 命中不能解决远端准入，不声明配置远端 Store 即可支持。

## 调用位置

| Connector 回调 | native Store 调用 |
| --- | --- |
| `on_new_request` | `on_new_request(RequestContext)`，只注册一次 |
| `get_num_new_matched_tokens` | `lookup_on_prefix/lookup_on_reverse` |
| `build_connector_meta` | 创建各 plan 时分配主 job_id、调用 Prepare，将 PreparedJob 写入 plan |
| `bind_connector_metadata` | 接收 plan 和对应 token |
| `start_load_kv` | Layout 获取 offsets/ptrs/sizes，记录前置 event 并 `submit_get`；bulk 原地 `wait` |
| `wait_for_layer_load` | `wait` 等对应层；State 在 forward 前统一等完 |
| `save_kv_layer` | 按 layer ID 去重，记录前置 event 并 layerwise `submit_put` |
| `wait_for_save` | bulk 提交、layerwise 补交遗漏层，排空 IO 并记录终态 |
| `get_transfer_results` | `get_finished` 消费仍未被消费的完成事件 |
| `build_connector_worker_meta` | 回报各 rank 的 load/save 终态 |
| `update_connector_output` | 全相关 rank 终态齐全后，`complete_get/complete_put` |
| `request_finished_all_groups` | 记录引擎结束通知，取消未下发的预取，等待活跃窗口收尾 |
| 请求结束且最后窗口已关闭 | `on_request_finished` |

`update_state_after_alloc` 保持 no-op，block tables 来自本步
`SchedulerOutput.kv_connector_block_state`。第一版默认固定为预取阶段、重新查命中、
layerwise Get：首次查询始终向 vLLM 返回 `(None, False)`，本轮不为该请求调度 forward。
预取完成后的查询才返回 `(N, False)`，按 layer ID 提交 Get 并在对应层执行前等待。
`use_layerwise` 未配置时默认为 true，cudagraph 的 piecewise 要求使用同一默认值。
needs_load 在首次 scheduled_tokens>0 且已生成初始 Get 计划后置为 false，避免后续
chunked prefill 重复加载已导入前缀。该标志只控制计划生成，IO 成败由 JobResult 回报。
make_dispatch_plan 创建单个策略或单个 State 目标时立即 Prepare，返回的
UCMDispatchPlan.prepared_job 携带该 plan 独立的 job_id、输入 keys 和 AddressedToken。
RequestMetadata 只汇总 load_plans/dump_plans，删除请求级 load_job/dump_job。
State 的 running/checkpoint 双目标各有独立 Connector Get job 和 Prepare 结果，pin 由 Store 管理；
State dump 按原生 boundary_state_offloads 创建 plan，并在创建时 Prepare 独立的 Put job。

现有 `PreparePrefetch(request_id)` 不携带 keys，需要内部 Lookup 先登记查询范围。
内部 Lookup 返回 None 时下发预取 token，由 rank0 同步执行 `submit_prefetch/wait`，
其余 rank 回报 no-op 成功；全部 rank 终态齐全后 `complete_prefetch` 并重新查询。
内部 Lookup 已返回确定命中/未命中时，预取阶段直接完成，无需生成虚假的 Store job；
本次仍返回 None，下次查询才把命中数交给 vLLM。
native Lookup 的 None 表示需要预取，-1 表示未命中，非负数是命中 key 的索引。
中间查询按原样传递 None，直到 Scheduler 为它创建预取任务。
这里是 Store 下层到当前层的 prefetch，CacheStore 的典型目标是 DRAM/shm，
具体源/目标由 Store 解释 token 决定；随后仍需 Get 到设备，并非异步 HBM load。
即使本步为 0-token，MRV2 的 no-forward 路径仍绑定 connector metadata、调用
`start_load_kv`、回传 Worker metadata；待预取请求独立加入 metadata，不依赖已调度请求集合。
这条调用链已对照本地 vLLM/Ascend MRV2 源码，但未做端到端验证。

MRV2 的正常 forward 和 no-forward 都经 post_forward 统一调用一次 wait_for_save，
再调用 get_transfer_results。Connector 删除 get_transfer_results 中重复执行的
no-forward 收尾和已无用途的 metadata.no_forward 字段，不再保留 _step_io_completed。
同 step 的 MTP/speculative decoding 可反复访问同一层，仍由 _dump_submitted_layer_ids
去重；逐层 Load 等待完成后移除该层任务，重复钩子不再等待已消费的完成事件。

Get/Put 的每个有本地传输范围的 IO 在提交前记录当前计算流 event，显式传入
StoreWorkerClient.submit_get/submit_put 的 prerequisite_handle，再写入
IoContext.prerequisite_handle；Proxy 原有接口透传至 JobDescriptor.prerequisiteHandle。
Get 的 event 排在引擎已提交的 block 清零、CoW 和 Mamba pre-copy 后，保证 Store
写入前先等这些操作结束；Put 的 event 排在源 KV/State 的生产操作后，保证 Store
读取前数据就绪。event 记录失败时先同步设备工具对应的流/设备，再以 0 句柄提交。
Prefetch 只在 Store 层间搬运，不使用设备计算流 event，当前仍同步等待预取完成。

Worker 复用 v1 的 create_device 工具，每个 IO 的 event handle 保存在 TransferTask，
只在已消费明确终态后释放，成功或失败均可释放；Submit 明确拒绝、未启动 IO 时立即释放。
wait、完成轮询和 shutdown 共用这一释放规则；无法确认 IO 终止时保留 event 和 KV 描述。
Get 的前置 event 不能代替后置等待：State 仍在 start_load_kv 返回前等完，其他 KV 在
对应层使用前等待，dump 仍在 wait_for_save 同步排空。checkpoint/running 双目标恢复
保持不变，不启用 Ascend 可选的 after-load 或逐层 Mamba copy 接管。
这里只完成 Connector 到既有 Proxy 字段的接口对接，Store 实际执行 event 等待仍待设备联调。

PreparePrefetch 或已终止的预取 IO 发生可恢复失败时，本请求后续返回 0，走重算，
不重复预取而一直返回 None；该失败状态在请求结束且所有窗口关闭后清除。
取消未下发的预取直接 CompletePrefetch(false)，已下发的预取等所有 rank 终态后再关闭窗口。
无法确认 IO 已终止的异常继续保留任务并报错，不按已完成失败处理。

`RequestMetadata.prepare_get_failed` 仅在 Scheduler 的 PrepareGet 抛出可恢复错误时
设置。Lookup 已向引擎报告命中，不能无声丢弃此次 load；Worker 保留计划中的 block ID，
在绑定 metadata 时标记失败，经 `get_block_ids_with_load_errors` 回报引擎。
任一 Get plan 的 Prepare 失败时，本请求跳过全部 Get 提交和本轮 dump；其余已成功
Prepare 的 Get job 由各 rank 回报失败，该请求全部 Get job 终态齐全后 CompleteGet，释放窗口。
Prepare 失败的 plan 不携带 prepared_job，不登记 pending job，也不调用 Complete。
Put Prepare 失败只跳过对应 plan，其余已成功 Prepare 的 Put plan 执行并参与请求级收尾。
真正 Submit/Wait 的 IO 失败走独立的 `failed_load_request_ids` 汇总，两者都会阻止
该请求的 dump 被发布。本地 vLLM 的 kv_load_failure_policy 默认为 fail，失败后终止请求。
K/V 投影的输入 hidden state 来自前面层，历史 KV 加载失败可污染 attention 输出，
再影响后续层的新 K/V；因此即使字节 dump 成功，也不能发布整份依赖该前缀的记录。
显式配置为 recompute 时，单 KV group 的同步 load failure 由引擎回滚进度并重算。
本地 MRV2 的多 KV group 同步失败恢复尚不支持扁平 invalid_block_ids，发生失败时引擎报错。

Scheduler 保留请求引用，每轮保存范围使用 vLLM 当前的 num_computed_tokens，跟随失败回滚。
load 失败后清除外存命中与 needs_load，保留请求状态以继续为重算部分生成保存计划。
抢占按 num_preemptions 识别新一轮请求状态，清除旧计划和预取标记，恢复后重新查询。
普通内存抢占先在 build_connector_meta 处理；reset_prefix_cache(reset_running_requests=True)
可在调度前抢占，请求的 lookup 先于本轮 build_connector_meta，因此保留 lookup 中的识别。
同一步重新准入时保留新状态。旧 Put 标记失败，但仍等待全部 rank 终态后才关闭窗口。
已提交对象保留，所有窗口关闭且请求真正结束后才调用 on_request_finished 清理 Store 请求元数据。

## 窗口与完成事件

每次成功 Prepare 对应一个 AddressedToken，窗口内允许多次 native Submit。
metadata 只复制 token 的 keys/payload，Worker 原样还原；不传进程内指针。
opaque token 不可独立切片，因此 Submit 保持 Prepare 输入 keys 的顺序，未涉及的 key
使用空 BlockPayload；后端必须将空范围视为 no-op，不能清空或发布该对象。
这份有序 key 集合单独保存在 PreparedJob 中，Worker 用它对齐 payload，不读取
token.keys 做业务匹配；token 字段的复制/还原只保留在 binding 适配边界。

主 job_id 由 Connector Scheduler 产生，随 PreparedJob(job_id, ucm_keys, token)
下发到 Worker；一次主 job 对应一次成功 Prepare，不使用额外的 UUID 保存标识。
Get/Put 的 job_id 在 Prepare 前分配，由 Connector 管理并用于 Worker Submit/完成关联。
同一 request_id 的不同 step、策略和 State 目标使用独立 Connector job。
Scheduler 的 itertools.count 从 1 开始、每次增加 2，主编号依次为 1、3、5 等奇数。

bulk 将本 rank 的全部 group/chunk 描述合并为一次 Submit，直接使用主 job_id。
layerwise 由 Worker 在主 job 下生成 sub_job_id，按主 job 保留每个已提交子任务的编号和结果，
本轮所有传输到达终态后，汇总并回报 JobResult(主 job_id, rank, success)。
各 rank 可以有不同数量的子任务，无本地任务的 rank 直接回报主 job 成功。

当前 native Submit/Wait 只接受一个 size_t 整数：bulk 和 prefetch 使用主 job_id；
layerwise 的 sub_job_id 由每个 StoreWorkerClient 独立按 2、4、6 等偶数分配。
TransferTask 只保存 job_id 和 sub_job_id，子编号由 StoreWorkerClient 独立分配。
有 sub_job_id 时直接使用它提交和等待，否则直接使用 job_id；io_job_id 属性仅选择
这两个已有编号之一，不分配第三个 ID。客户端按返回编号查回任务并汇总到主 job。
奇偶集合互不重叠，单个 Scheduler 可分配 2^63 个主 job，每个 Worker 可分配
2^63-1 个 layerwise IO ID。两路分配均检查 uint64 上限，耗尽报错，不回绕复用。
Worker 完成队列按实例独立，各 rank 可使用相同主 ID。
Store Scheduler 接口保留现有签名，Get/Put 不额外传入 job_id：

```python
prepare_get(request_id, ucm_keys)
prepare_put(request_id, ucm_keys)
complete_get(request_id)
complete_put(request_id, success)
```

逐 plan Prepare 的结构保留，追加式 Prepare 或多窗口的具体语义由 Store/Proxy 后续确定。
当前 native Store 仍拒绝同 request/kind 的重叠 Prepare，追加支持需要 Store 侧落实后才能联调。
Connector 收齐同一 request/kind 下全部 job 的所有 rank 终态后，才调用一次 request 级 Complete，
避免只完成一个 plan 就释放其他任务仍在使用的 pin。Prefetch 仍按现有 request 级协议。

`wait` 与 `get_finished` 消费同一完成队列，native IoResult 仅返回一次。
客户端保存已消费的终态，后续等待不重复向 native 等通知；无法确认终止的任务
保持描述符和 KV 引用并报 fatal 错误。

本轮 dump 同步排空，每个主 job 的全部预期 rank 都上报终态。
同一请求当前全部 Put job、全部 rank 均成功后 CompletePut(request_id, True)，存在任务失败时，
在全部相关 IO 终止后 CompletePut(request_id, False)。各 Connector job 分别记录完成状态。
chunked prefill 后续 step 的新增 Prepare 依赖 Store 后续确定的追加或多窗口能力。
真实跨 step 异步 dump 的源 KV pin 生命周期仍待实现。
请求结束可能早于最后一次 no-forward 保存，最终 metadata 准备且所有窗口关闭后
才调用 on_request_finished。未来异步 dump 的真实 block pin 生命周期不在本轮实现。
引擎在释放 blocks 前调用 request_finished_all_groups，随后才把 ID 放入 finished_req_ids。
build_connector_meta 不再重复调用结束回调，只在处理该清单后将请求标记为可最终清理。
_complete_ready_jobs 仅在 update_connector_output 消费 Worker 结果后调用，收齐同
request/kind 的 job/rank 结果并关闭 Store 窗口。未下发的预取取消时可直接关闭窗口。
_close_finished_requests 等请求已结束且所有窗口已关闭后，注销 Store 请求并清理预取状态。

## 字节描述与配置

native BlockPayload 增加 Python 字段 `offsets` 和 `total_size`：
offsets 与 pointers/sizes 一一对应；total_size 是完整 PP 对象的大小，
包含其他 shard 和本次未提交的层。空 pointers 是 no-op。
PreparePut 的现有签名未携带 sizes，变长对象在 Alloc 时如何获知大小仍需后端落实，
不能使用本层 sum(sizes) 代替。原有 memory_region_indices/group_sizes 保持可选。

配置继续通过 ucm.utils.Config 或 UCM_CONFIG_FILE 读取：

```yaml
chunk_size: 256
use_layerwise: true
store_module: ucm_proxy
store_config:
  type: store
  store: posix
  config: {}  # 替换为实际后端配置
```

store_module 默认是 CMake 扩展模块名 ucm_proxy，可按安装位置指定。
store_config 原样交给 StoreAssembler；不提供旧 proxy 配置兼容或测试 Store 回退。
Assembler 持有至 IO 终止，Scheduler/Worker 接口是借用引用。

## 验证边界

本地 CacheStore/PosixStore 的 Setup/Load/Dump 仍是 Unsupported 占位；本轮不实现后端。
offsets/total_size 需要重新编译 native binding 才可用。
未新增、迁移或运行测试，未做设备、分布式、forward 或端到端验证。
静态语法、引用和格式检查不作为功能验收。
