# Connector v2 与 native Store 接口

实施入口：`feature_ucm_v2@8e03c9d85fba6e803fae3824b89444c703066113`。
connector 源自 `dev_connector@485916d2e70e7d3640fa9574572b0f06935a4cb8`。
本轮只移植接口，不沿用旧测试结论；按用户要求，不新增、迁移或运行测试，不做端到端验证。

## 文件职责

模块直接位于 `ucm/integration/vllm/v2/`，删除 `layout/` 子包。

| 文件 | 职责 |
| --- | --- |
| `ucm_connector.py` | vLLM 公开回调，分别转交 Scheduler/Worker |
| `connector_scheduler.py` | 请求注册、跨 PP lookup 汇总、计划、Prepare/Complete |
| `connector_worker.py` | KV 注册、TransferBuilder、提交、等待、终态回报 |
| `kv_cache_spec.py` | 公共 spec、Worker layer/descriptor 声明、原生 config parser |
| `kv_cache_layout.py` | 已解析声明与实际 tensor 绑定，汇总本 stage 布局 |
| `group_layout.py` / `cache_view.py` | HBM 几何、block 寻址和 layer 内 segment 归并 |
| `store_layout.py` | stage 内 TP/DCP shard、副本与对象内字节布局 |
| `runtime.py` | Runtime、并行坐标、fingerprint、PP 对象 key 与握手 |
| `store_client.py` | native binding 类型、可序列化 token、完成事件消费 |

不迁入 Python PipelineProxy/ParallelProxy、Memory/File 开发 Store 或 model-check 工具。
`store_client.py` 不负责路由、容量、promotion/writeback、暂存或发布；这些由 `ucm/v2` 负责。

声明解析与实际视图的边界：`kv_cache_spec.py` 中的 `parse_layer_specs` 解析
`TensorDescriptor`、存储粒度、view order 和层号，不访问实际 tensors；这些 Worker
声明不塞入 Scheduler 共用的 `UCMKVCacheSpec`。`kv_cache_layout.py` 仅做绑定和汇总，
实际 stride/segment/pointer 转换仍在 `cache_view.py` 与 `group_layout.py`。
删除未使用的 `export_layout_metadata`、Layout 构造参数 `use_layerwise` 及旧包重导出。

逻辑计划名为 `UCMDispatchPlan`（可含多个 native group），逐请求下发信息为
`RequestMetadata`。Worker 汇总主 job 使用 `JobResult`，与 native 子任务的
`IoResult` 区分；客户端 `poll_finished` 消费并记录完成事件，native 方法仍是
`get_finished`。原 `attempt_id`/SaveJob 已在主 job 改造时移除，不再引入同义类型。

## PP 与分片

PP 分开存。key 由 common hash、PP rank 与恢复策略确定，保持 16 字节；
Worker 只处理本 stage 对象，offset 和完整对象大小都不包含其他 stage。
前 15 字节直接取 RequestHasher 的链式 MD5 摘要前缀，不再混入 PP rank 做二次 SHA256。
最后 1 字节按 `[预留 2 bit][pp_rank 3 bit][restore_policy 3 bit]` 编码：
高 2 bit 置零，`suffix = (pp_rank << 3) | policy_id`；FA/WA/State 的 policy_id
分别是 0/1/2。配置初始化时限制 PP stage 数量不超过 8，pp_rank 范围为 0～7。
同一 chunk 在不同 PP stage 和策略下共享前 15 字节，仅尾字节不同。
删除 PP 布局 all-gather。`UCMShard(tp_shard, dcp_shard)` 不再包含 PP；
`ParallelRank(tp_rank, pp_rank, pcp_rank)` 仍保留完整 Worker 身份。
每个 stage 在握手中声明实际存储的策略，空 stage/group 不产生空对象。
Scheduler 对全部参与 stage 查询 FA 共同前缀、WA/State 共同恢复边界。

部署约束：每个 stage 内 TP/PCP/DCP 都在单机，跨机 TP/PCP/DCP 不支持。
PP 可以跨节点，但配置的远端 Store 必须允许 Scheduler 和全部 Worker 访问相同对象，
并兑现 AddressedToken 寻址。本轮不实现跨机 shm 或 Store RPC；远端 Lookup 命中
不等于远端 Worker 已能使用本机 shm token。

## 调用位置

| Connector 回调 | native Store 调用 |
| --- | --- |
| `on_new_request` | `on_new_request(RequestContext)`，只注册一次 |
| `get_num_new_matched_tokens` | `lookup_on_prefix/lookup_on_reverse` |
| `build_connector_meta` | `prepare_get/prepare_put`，分配主 job_id，将其与 AddressedToken 写入 metadata |
| `bind_connector_metadata` | 接收 plan 和对应 token |
| `start_load_kv` | Layout 获取 offsets/ptrs/sizes，`submit_get`；bulk 原地 `wait` |
| `wait_for_layer_load` | `wait` 等对应层；State 在 forward 前统一等完 |
| `save_kv_layer` | layerwise `submit_put` |
| `wait_for_save` | bulk 提交、layerwise 补交遗漏层，排空 IO 并记录终态 |
| `get_transfer_results` | `get_finished` 消费仍未被消费的完成事件 |
| `build_connector_worker_meta` | 回报各 rank 的 load/save 终态 |
| `update_connector_output` | 全相关 rank 终态齐全后，`complete_get/complete_put` |
| 请求结束且最后窗口已关闭 | `on_request_finished` |

`update_state_after_alloc` 保持 no-op，block tables 来自本步
`SchedulerOutput.kv_connector_block_state`。同步 load 返回 `(N, False)`。
Lookup 返回 None 时下发预取 token，由 rank0 同步执行 `submit_prefetch/wait`，
其余 rank 回报 no-op 成功；汇总后 `complete_prefetch` 并重新查询。
这不代表跨 step 异步 HBM load 已实现。

## 窗口与完成事件

每次成功 Prepare 对应一个 AddressedToken，窗口内允许多次 native Submit。
metadata 只复制 token 的 keys/payload，Worker 原样还原；不传进程内指针。
opaque token 不可独立切片，因此 Submit 保持完整 keys 顺序，未涉及的 key
使用空 BlockPayload；后端必须将空范围视为 no-op，不能清空或发布该对象。

主 job_id 由 Connector Scheduler 产生，随 PreparedJob(job_id, AddressedToken)
下发到 Worker；一次主 job 对应一次成功 Prepare，不使用额外的 UUID 保存标识。
同一个 request_id 的不同 step、Get/Put/Prefetch 使用不同主 job。

bulk 将本 rank 的全部 group/chunk 描述合并为一次 Submit，直接使用主 job_id。
layerwise 由 Worker 在主 job 下生成 sub_job_id，各层/缓存的 Submit 独立完成，
全部子任务终态明确后才回报 JobResult(主 job_id, rank, success)。
各 rank 可以有不同数量的子任务，无本地任务的 rank 直接回报主 job 成功。

当前 native Submit/Wait 只接受一个 size_t 整数：Scheduler 按 2^32 的间隔分配
uint64 主 ID，低 32 位留作子编号。bulk 使用主 ID，layerwise 使用
主 ID + sub_job_id；客户端按 native ID 消费结果并关联回主 job。
该编码集中在 store_client.py，Worker 完成队列按实例独立，各 rank 可使用相同主 ID。
native Scheduler 的 Prepare/Complete 签名仍按 request_id，主 job 的关联由
Connector Scheduler 的 pending 表维护；本轮不改 native Scheduler 的参数。

`wait` 与 `get_finished` 消费同一完成队列，native IoResult 仅返回一次。
客户端保存已消费的终态，后续等待不重复向 native 等通知；无法确认终止的任务
保持描述符和 KV 引用并报 fatal 错误。

本轮 dump 同步排空，每个主 job 的全部预期 rank 都上报终态。
所有 rank 都成功后 CompletePut(True)，失败则在全部相关 IO 终止后 CompletePut(False)。
chunked prefill 后续 step 在同一 request_id 下重新 PreparePut；不等请求结束才发布。
请求结束可能早于最后一次 no-forward 保存，最终 metadata 准备且所有窗口关闭后
才调用 on_request_finished。未来异步 dump 的真实 block pin 生命周期不在本轮实现。
Store 未来新增 request→多并行 job 签名后再接入；当前遵循 proxy.py.cc 的现有签名。

## 字节描述与配置

native BlockPayload 增加 Python 字段 `offsets` 和 `total_size`：
offsets 与 pointers/sizes 一一对应；total_size 是完整 PP 对象的大小，
包含其他 shard 和本次未提交的层。空 pointers 是 no-op。
PreparePut 当前只有 keys，变长对象在 Alloc 时如何获知大小仍需后端落实，
不能使用本层 sum(sizes) 代替。原有 memory_region_indices/group_sizes 保持可选。

配置继续通过 ucm.utils.Config 或 UCM_CONFIG_FILE 读取：

```yaml
chunk_size: 256
use_layerwise: false
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
