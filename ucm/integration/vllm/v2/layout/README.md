# v2 KV 元数据、HBM 寻址与外存排布

2026-10-19：合并 dev_connector 的分层（公共元数据、实际物理布局、存储模板分开），命名沿用本分支。当前参考 vLLM 0.30.0。

完整链路见 [v2 详细设计](../../../../../docs/connector-v2-detailed-design.md)，更新说明见 [0.30 适配记录](../../../../../docs/connector-v2-030-adaptation.md)。历史图文位于 [qyh/docs/v2-layout](https://github.com/qyh111/unified-cache-management/tree/docs/v2-layout)，其中旧的 BlockAccess 需按本文更新；主开发目录不维护 HTML/SVG/生成脚本。

公共 Spec 不持有 scheduler_block_size 或 TP策略；前者仅供 parser 日志/无对象退路，Worker的GroupStoreLayout在PP交换后确定整组tp_shards，混合声明按每TP rank完整存储并接受冗余。初始化日志使用标准 logger.info，热路径诊断使用 logger.debug，不再提供 layout_debug 或环境开关。

## 对象与职责

| 对象 | 内容与职责 |
| --- | --- |
| `UCMKVCacheGroupSpec` | 原生 group ID、`block_size` / `block_token_span`、kinds、tail、`is_persistent`、分片方式；不含 tensor 指针或存储模板 |
| `UCMKVCacheSpec` | groups、`chunk_size`、`num_attn_module`，以及按原生 group ID 索引的 `blocks_per_chunk`；scheduler/worker 共用 |
| `UCMLayerSpec` | worker 逐层原生 spec、storage entries、descriptor、轴约定；不含物理指针 |
| `LayerView` / `ComponentView` | 实际 tensor 的 shape/stride 和连续 `MemorySegment`；不分配或复制 KV storage |
| `KVCacheGroupLayout` | HBM 几何、`layer_specs`/层选择、vLLM block id 到 segment 起始地址、token→byte 物理换算；不持有 group 策略或 UCM chunk大小 |
| `GroupStoreLayout` | 持有 group 信息和 group_layout；选择 token 范围，调用 HBM 换算并编译源 byte offsets、sizes、外存 offsets 和 whole_chunk_nbytes |
| `UCMKVCacheLayout` | worker 注册时汇总上述对象与层索引 |
| 外部 `UCMTransferBuilder` | 解释 plan、传递每个 UCM chunk 的起始 token、累加各 group 外存起点、解析全局对象 key/组级 offsets 和二维 Transfer |

Layout 不依赖 scheduler/proxy，不接收请求 metadata/plan，不生成 Transfer。HBM 提供物理 byte 换算和 segment 起始地址，由 Store 选择范围并添加块内 byte offset；原 `BlockAccess` / HBM `compile_access` 已删除。

## UCM chunk 的公共约定

`chunk_size` 是一个 UCM key 的全局逻辑 token 跨度。parser 接收 DCP size，先确定 `block_token_span`，默认一次取实际存储 group spans 的 LCM；scratch/零 tail group 不参与。显式非 DCP FA 小 chunk 必须均匀切分 block 并对齐压缩 entry，State/SWA及 DCP 整页路径要求 chunk 是对应 span 的整数倍。`spec.blocks_per_chunk[group_id]` 在最终 spec 构造时派生，State每chunk一个快照。原生 `block_size` 始终保持本地容量；Runtime 不再二次修改 spec。

| group | 每 UCM chunk 的 vLLM block 数 | 保存范围 |
| --- | --- | --- |
| FA | `ceil(chunk_size / block_token_span)` | UCM chunk 的 token 区间；一个或多个完整 block，配置更小的 UCM chunk时为一个 block 的一部分（FA 子块） |
| SWA | `ceil(tail_tokens / block_token_span)` | 恢复边界前的 tail；首块可以只保存后半部分 |
| State | 1 | 恢复边界上的一个完整快照 |
| 非持久（`is_persistent=False`）/ 零 tail | 0 | 不进入存储路由；非持久 group 也不创建 HBM/Store layout |

整除约束在解析配置时检查：FA 的块大小整除 UCM chunk，或配置了更小的 UCM chunk时 UCM chunk整除 FA 块且是 `tokens_per_state` 的整数倍；非零 SWA 和 State 的块大小整除 UCM chunk。HBM 不支持部分 token 时，Store 在初始化阶段拒绝该范围，按层选择不放宽整组能力限制。State token/state 换算必须整除，因此不允许部分快照。

要存哪些 group 直接取 `KVCacheConfig.prefix_cacheable_group_ids`，其余 group `is_persistent=False`，不存；scratch 重建属于引擎。不支持的 spec（chunked local、cross、encoder-only、UNKNOWN）明确拒绝，Sink Full Attention 为显式允许项。

## HBM 与 Store 的接口

```python
layout = UCMKVCacheLayout(
    spec, kv_caches, kv_cache_config=worker_config,
    num_hidden_layers=num_hidden_layers, use_layerwise=use_layerwise,
)
# rank/topology 来自 Worker runtime；PP1 不执行 collective。
from ucm.integration.vllm.v2.connector_worker import exchange_pp_metadata
pp_metadata = exchange_pp_metadata(
    (rank.pp_rank, layout.export_layout_metadata(topology.tp_size), None), topology,
)
layout.build_store_layouts(topology, rank, pp_metadata)
group_layout = layout.group_layouts[group_id]
store_layout = layout.store_layouts[group_id]
mask = group_layout.segment_mask(layer_ids=[7])

# HBM 只返回 [block, segment] 的起始地址，不解释 UCM chunk 或 token 范围。
bases = group_layout.resolve_ptrs(vllm_block_ids, segment_mask=mask)

# Store 返回 [UCM chunk, IO entry] 的三个矩阵。
offsets, ptrs, sizes = store_layout.resolve_matrices(
    vllm_block_ids, token_starts, segment_mask=mask,
)
```

HBM 几何列包含 `base_ptrs`、`block_strides`、`states_per_block`、`payload_bytes`。每层可以有多个 components/segments，layer ID 不必只返回一个地址。

注册时在单个layer_name的LayerView内统一整理segments。Component只解析原始片段；LayerView按backing和block stride分桶，重叠区间取并集，相邻区间合并。A5 DSV4的KV、scale和覆盖两者的full view因此只存一份内容。不同backing保持首次出现的组件顺序，同一backing内按offset排序，不按不同allocation的裸地址决定文件顺序。不跨layer_name合并，layer ID只作选层索引。合并后LayerView的`supports_partial_tokens=False`，Group对所有LayerView取all；显式小chunk或非整块SWA tail由Store初始化的现有whole-block检查拒绝。尾部padding不随之加入有效payload。

Store 在注册时把每个 UCM chunk 的固定 starts/counts 换成 `[window block, segment]` 的 `segment_byte_offsets_in_block` 与 `chunk_segment_bytes`。`group_layout.tokens_to_bytes(tokens, tokens_per_block, segment_mask)` 直接计算 `tokens * payload_bytes // tokens_per_block`。逻辑 block span 由调用方传入，HBM 不保存 UCM chunk/window 信息。物理部分访问能力与完整 entry 对齐在 Store 初始化时统一校验；FA/SWA不支持部分访问时使用同一条whole-block错误，不保留两套文案。

运行时 HBM 只计算 `base + vllm_block_id * stride`；Store 把它组织为 `[UCM chunk, window block, segment]`，添加固定 byte offsets，最后展平。scheduler 在 plan 的 `token_starts` 里给出每个 UCM chunk 的起始 token，transfer builder 原样传递。Store 对非 SWA group 用 `token_starts % block_token_span` 得到块内偏移，只有 FA 子块时它不为 0；SWA 用固定 tail 模板。

## 存储字节格式与 padding

当前格式顺序为 **原生 group → 时间 block → UCMShard → 本地成员/segment**，bulk/layerwise 统一紧凑有效 payload，不复制页间 padding。HBM 仍可发现 BlockFirstView 来描述地址连续性；文件格式由 Store 模板决定。

注册时调用 `KVCacheGroupLayout.export_metadata(tp_size)`，由HBM layout自己导出 `GroupLayoutMetadata`（sizes、states、部分访问能力、组级TP shard数量LCM）；未声明TP上限的成员先解析为实际TP size，组内和跨PP均取LCM。PP1直接使用本地数据，PP>1由 `connector_worker.py` 只做一次 `exchange_pp_metadata`。随后 `build_store_layouts` 调用 `GroupStoreLayout.build`，先拼各PP stage的有效TP/DCP shards，再按chunk展开，一次创建最终sizes/offsets。StoreLayout不再保存中间模板，也没有半成品替换链路。Worker不计算格式digest；缓存身份统一使用包含持久group block size和格式版本的Scheduler fingerprint。TransferBuilder只选择组级writer行与layer mask。

Store 的全局单 block bytes 乘本 group 的时间 block 数，才是该 group 的 chunk bytes；State 始终只有一个快照，WA 使用 tail 模板。空 PP group 保留 StoreLayout 元数据位置，不构造假 HBM。TransferBuilder初始化时以恢复策略为dict key累计各native group的起点和对象总bytes；不同native group各自保留分片规则，完整契约见 [按恢复策略组织对象](../../../../../docs/connector-v2-storage-groups.md)。

## FA 16384 / 512 数值例子

connector 配置 `chunk_size=512`，FA block 为 16384 tokens。两层均为每 token 4 bytes；block-0 基址分别 1000000 / 2000000，vLLM block stride=65536 bytes。

```python
vllm_block_ids = np.array([2, 2], dtype=np.uint64)
token_starts = np.array([0, 512], dtype=np.uint64)
offsets, ptrs, sizes = store_layout.resolve_matrices(vllm_block_ids, token_starts)
```

```text
ptrs   = [[1131072, 2131072], [1133120, 2133120]]
sizes  = [[2048, 2048],       [2048, 2048]]
offsets= [[0, 2048],          [0, 2048]]
whole_chunk_nbytes = 4096
```

选择第二层时 offset 仍是 2048。vLLM block id 来自 scheduler 的 table，与 UCM chunk 序号无直接相等关系。SWA 的 B=8、tail=12 则编译 starts=[4,0]、counts=[4,8]。对应夹具为 `LayerViewSegmentsTest.test_documented_fa_subblocks_preserve_layer_offsets`。

## ABI、并行和生命周期边界

- view 明确归一化到四维 BHNC，不按 shape 猜轴。
- Ascend 原生独立 K/V 或去 padding views 的组件 stride 可不同于 descriptor 整页 stride；直接使用实际 view。CUDA/CPU 保留原生 descriptor 一致性校验。
- Ascend typed State 先把首块的稠密物理 payload 零拷贝展平，再转 byte page；保留 dense permutation、storage_offset、block stride。不调用 contiguous() 复制 KV。
- compact HNC 整块保留连续段；分离 heads 先生成原始片段，统一在所属LayerView内去重/合并。多 head HNC 不接受部分 token。
- 按实际DCP size调用原生Scheduler的 `dcp_world_size_for_kv_cache_spec`：FA/MLA放大逻辑跨度，State/SWA保持本地跨度。实际DCP分片页要求完整保存；不因进程开启DCP拒绝含SWA的模型。
- ptrs 是每次调用独立数组；sizes/offsets 广播自本次复制的小模板，在途 Transfer 不被下一次 dispatch 覆盖。源 KV 内存生命周期另由 worker 处理。
- vLLM block id 信任引擎的计划，HBM 不再检查是否越界。

### 部分 token 能力的来源与使用

`ComponentView.supports_partial_tokens` 在 `_build_bhnc_view` 按真实 shape/stride 设置：物理 NHC 或 singleton head 的 token-contiguous view 为 True，多 head HNC/分离 heads 为 False。LayerView对全部components取all，发生重叠去重或相邻合并时设为False；GroupLayout再对所有LayerView取all。Store 在初始化编译固定范围时检查它，layer mask 不放宽整组能力。运行时 token→byte 仅按 payload 比例换算。

## 并行分片

布局只描述当前 rank 注册的逻辑 views。PP 的空 group 保留全局 group/plan 位置，但不生成本地物理 segments。CP 整页换算在 `ParallelTopology` 完成；Ascend replicated indexer 根据原生 spec 展开连续 kernel rows。哪些 rank 持有同一片段、各自写哪些 chunk 由 `layout/store_layout.py` 的组级分片/副本映射决定。全局格式使用紧凑成员 payload，完整性由 Scheduler 聚合后发布；Lookup 查询已发布对象，不进入 LayerView 的寻址算法。

TP/PP/DCP 的验证范围、PCP 引擎限制及复现入口见 [并行验证记录](../../../../../docs/connector-v2-parallel-validation-20260926.md)。
