# v2 KV 寻址与 UCM block 排布

完整设计见 [UCM Connector v2 详细设计](../../../../../docs/connector-v2-detailed-design.md)，包含 scheduler、hash/窗口规则、worker 生命周期和验证边界。本文件保留布局接口与数值示例。

## 模块边界

Layout 统一负责三部分，不按文件数量强行拆分：

- `kv_cache.py`：解析 KVCacheConfig 的 group/persistence/token 元数据；`UCMKVCacheLayout` 汇总元数据、HBM 布局和外存模板。
- `__init__.py`、`group.py`、`view.py`：绑定 worker 原生 spec/descriptor 和真实 tensor views，解析 LayerView/ComponentView/MemorySegment，并提供 block IDs 与 token 范围的寻址。
- `store_layout.py`：编译每个 group 的外存 offsets、sizes、store_bytes，以及通用或整块 Block First 模板。

`layout` 不依赖 scheduler 或 proxy，不接收请求 metadata/plan，不生成 Transfer。调用方可以给出 block IDs、token offsets 和 segment mask，使用布局计算 byte ranges；这仍是寻址，不是调度。

外部 `../ucm_kv_cache.py / UCMTransferBuilder` 解释 plan、关联 keys、累计 participating groups 的外存起点、拼接二维 Transfer；connector 持有独立的 `layout` 与 `transfer_builder`。路由 `dispatch_routes(spec)` 在 scheduler 中。

```python
from ucm.integration.vllm.v2.layout import UCMKVCacheLayout, parse_kv_cache_config
from ucm.integration.vllm.v2.ucm_kv_cache import UCMTransferBuilder

layout = UCMKVCacheLayout(
    spec, kv_caches, kv_cache_config=worker_config,
    num_hidden_layers=num_hidden_layers, use_layerwise=use_layerwise,
)
transfer_builder = UCMTransferBuilder(layout)
for transfer in transfer_builder.build_load_transfers(metadata, layer_id=7):
    proxy_adapter.submit("load", transfer)
```

本次只调整职责与导入位置，token/State/transient 策略、segment 寻址、padding、bulk/layerwise 外存模板及 namespace r7 不变。当前开发 proxy 同步执行，State layerwise 流水仍待实现。测试/工具已迁移当前接口引用，旧 spec/view/lifecycle 夹具仍待核对；未运行测试。

建议按 `kv_cache.py -> __init__.py -> view.py -> group.py -> store_layout.py` 阅读；理解 plan 与传输时再读外部 `../ucm_kv_cache.py`。

## 谁负责什么

| 对象 | 输入 | 输出/职责 |
|---|---|---|
| `LayerView` | 一个 layer_name 注册的一个或多个 tensor views | 逻辑层描述，不分配 KV storage |
| `ComponentView` | 一个注册 tensor view 的 shape/stride | 一个或多个连续 `MemorySegment`；components 可共享 storage |
| `KVCacheGroupLayout` | 一个 group 的 LayerViews | 把源地址几何编译成 NumPy segment 列，按 layer_id/name 选择列 |
| `BlockAccess` | 固定 token 范围模板；运行时 block IDs、starts | 源内存 `ptrs/sizes`，形状 `[block, segment]` |
| `GroupStoreLayout` | group 的 FA/WA/State 规则与物理布局 | 在 UCM block 中放置数据，维护目标 offsets，判断能否合并 IO |
| `UCMKVCacheLayout` | 元数据、worker KVCacheConfig 与真实 views | 汇总 group_layouts/store_layouts，建立层名与层号索引 |
| 外部 `UCMTransferBuilder` | Layout、scheduler plan 与层选择 | 关联 keys，组合各 group，构造 proxy Transfer |

`view` 表示逻辑视图；`segment` 表示能连续复制的一段。两者不是一一对应关系。`segment_mask` 针对展开后的列。

`supports_partial_tokens` 是 group 级布尔策略：初始化时对全部 ComponentView 的布局能力取 `all`。只要任意 view 不支持部分 token，整个 group 只允许完整 block，按层选择也不能绕过此限制。MemorySegment 只保留寻址参数，不保存逐 segment 的策略标志；固定范围和动态 offset 共用 group 策略。

`compile_access` 将块内 token 起点/长度转换为 `[window block, segment]` 的字节 offset/size 模板，`BlockAccess` 保存整个 group 的模板，层选择只在运行时通过 segment_mask 进行。运行时 `resolve_ptrs` 计算 `base_ptr + block_id * block_stride + fixed_offset + dynamic_offset`；通过 reshape 将 ptrs 看成 `[重复窗口, 窗口内 block, segment]`，广播添加固定模板，不复制 ptrs。`resolve_ptrs` 只返回指针，供 record 层复用固定 size 模板；删除仅旧扁平入口使用的 `resolve/_resolve` 包装。固定范围和动态范围均须落在实际 block 内；State 每组件只有一个存储单元，其整块要求由范围与整除约束自然保证，不再重复写 State 特判。内部模板按非空一维数组契约传入，不在每次编译重复检查形状。

`BlockFirstView` 留在 group 中，它只描述源内存中带 padding 的连续 span。当前只在 CUDA/CPU 的 BLHNC/BLNHC 下根据原生 descriptor 创建，要求 `0 < layer_stride < block_stride` 且本组 descriptors 的 block stride 一致；信任 0.30 原生 allocator 的连续放置契约，不再逐层检查地址或 offset 链。NPU 直接使用 per-layer segments；GLM 专用 slot descriptor 不符合 stride 条件。store 层仅在 use_layerwise=False、完整块且 BlockFirstView 存在时使用独立的整组 IO 路径；layerwise 继续按通用紧凑模板存取物理 segments。

State group 同样参与该路径：符合条件时，bulk 复制整个 group span，保留各层 page padding，不复制 span 之外的全局 block 空闲容量。use_layerwise=True 时始终使用 payload 前缀和模板，State 整快照与加载等待规则继续由原有策略处理。记录格式使用 r7 namespace，隔离旧版带 page 间隙的 layerwise 文件。

## 两种 offset

- `block_byte_offsets`：源 segment 内，从块起点跳过多少字节；用于计算 ptr。
- `ucm_block_offsets`：目标 UCM block 中放在哪；每个 group 模板内先存 group 相对值，最终加 group 基址。

`GroupStoreLayout.group_block_offsets` 是**一个完整 vLLM block** 的目标排布，完整 Block First bulk 为单元素 `[0]`，layerwise 则为有效 payload 的前缀和；`ucm_block_offsets` 是**一个 UCM key 覆盖的窗口**排布。窗口可能是部分 block、多个 blocks 或整页 State，因此不能混用。`whole_block_bytes` 是完整 block 的目标空间，`store_bytes` 是这个 group 在一个 UCM block 中的总空间。

默认排列：

```text
一个 FA hash key -> 一个 UCM block
  group 0
    window block 0
      layer 0 / component 0 / segments ...
      layer 1 / component 0 / segments ...
    window block 1
      ...
  group 1
    ...
```

完整 Block First 有两条存取路径：use_layerwise=False 时每个原生 block 只有一项，offset 为0、size 为 BlockFirstView.block_size_bytes（含 padding）；use_layerwise=True 时保留通用模板，offset=cumsum(payload_bytes)-payload_bytes、block bytes=sum(payload_bytes)，与是否 Block First 无关。部分块也走通用模板。构造 UCMKVCacheLayout 时显式传入 connector 的 use_layerwise，避免只凭 segment_mask 选择存储格式。

## FA 16384 / 512 完整例子

假设一个 FA group 有两层，都是 Ascend BNHC 的连续 view：

```text
shape = [3 blocks, 16384 tokens, 2 heads, 2 channels]
dtype = uint8
strides = [65536, 4, 2, 1]  # 元素单位，此处等于字节
layer 0 base_ptr = 1000000
layer 1 base_ptr = 2000000
```

每层每 token 4 字节，一个源 block 为 65536 字节。每个 UCM key 保存 512 tokens，每层 2048 字节；整个 UCM block 为 4096 字节。

### 1. 注册时编译

`UCMKVCacheLayout(spec, kv_caches, kv_cache_config=..., num_hidden_layers=..., use_layerwise=...)` 创建两个 group 相关对象：

```python
group_layout = layout.group_layouts[group_id]
store_layout = layout.store_layouts[group_id]
```

record 层确定固定长度 512，并调用：

```python
access = group_layout.compile_access(token_counts=512)
# access.segment_bytes = [[2048, 2048]]
# store_layout.ucm_block_offsets = [[0, 2048]]
# store_layout.store_bytes = 4096
```

sizes 在这里算好。运行时无需 ends；范围长度改变时需要重新选择或编译 access 模板。

### 2. scheduler 给物理 blocks，worker 算块内 starts

假设前两个逻辑 UCM keys 对应的数据都在物理 block 2（物理 block ID 不要求等于逻辑序号）：

```python
block_ids = np.array([2, 2], dtype=np.uint64)
starts = np.array([0, 512], dtype=np.uint64)
ptrs = access.resolve_ptrs(block_ids, token_offsets=starts)
sizes = np.broadcast_to(access.segment_bytes, ptrs.shape)
```

逐列使用同一公式：`base_ptr + block_id * block_stride + start * bytes_per_token`。

```text
                 layer 0     layer 1
key A start 0    1131072     2131072
key B start 512  1133120     2133120
sizes              2048        2048  # 每行相同
```

完整 32 个子块的 starts 为 `0, 512, ..., 15872`。HNC 时一层可能有多个 head segments，公式对每个 segment 计算，不能把它们误当成一个连续 token 区间。

### 3. 关联目标 offsets 与 keys

`GroupStoreLayout.resolve_matrices` 把相同 offset 模板重复到两个 key：

```text
key A: offset 0    <- layer 0 的 2048 字节
       offset 2048 <- layer 1 的 2048 字节
key B: offset 0    <- layer 0 的下一段 2048 字节
       offset 2048 <- layer 1 的下一段 2048 字节
```

外部 `UCMTransferBuilder._iter_plan_segments` 加 group 起点并关联 keys，最终扁平化为：

```text
keys:    [A,       A,       B,       B      ]
offsets: [0,       2048,    0,       2048   ]
ptrs:    [1131072, 2131072, 1133120, 2133120]
sizes:   [2048,    2048,    2048,    2048   ]
```

若只请求 layer_id=1，先选 segment 列再展开地址，得到 `ptrs=[2131072,2133120]`、`offsets=[2048,2048]`。目标 offset 不变成零，因为同一个 UCM block 中 layer 0 的位置仍被保留。

对应夹具为 `LayerViewSegmentsTest.test_documented_fa_subblocks_preserve_layer_offsets`；当前未运行，不能据此声称新接口已验证。实际 connector 通过外部 transfer_builder 的 `build_load_transfers/build_dump_transfers` 下发对应二维矩阵，不需要自己展开 head 或解释 strides。

## 当前边界

- 官方 0.29 四维 view 按 BHNC 语义；Ascend 0.26 按 BNHC 语义。不能靠轴大小猜测。
- State 整页；token/state 换算必须整除。
- 多个连续 kernel rows 支持；多个 kernel rows 且 head 分离仍明确拒绝。
- NumPy 做批量地址计算，不保证多核并行。本轮职责整理不引入地址结果的长期缓存。


## 输出数组的生命周期

物理寻址 `BlockAccess.resolve_ptrs` 返回独立可写的 ptrs；二维 Transfer 则使用独立 ptrs，以及本次调用小模板广播出的只读 sizes/offsets。单组直接持有这次结果，多组再拼接；后续 dispatch 不会覆盖已下发的描述符。这里的独立性针对描述符数组，KV 源内存本身的在途保护仍由既有生命周期机制负责。

## 并行分片

布局只描述当前 rank 注册的逻辑 views。PP 的空 group 保留全局 group/plan 位置，但不生成本地物理 segments。CP 整页换算在 `ParallelLayout` 完成；Ascend replicated indexer 根据原生 spec 展开连续 kernel rows。rank 隔离与全分片 commit 可见性在存储包装层完成，不进入 LayerView 的寻址算法。

TP/PP/DCP 的验证范围、PCP 引擎限制及复现入口见 [并行验证记录](../../../../../docs/connector-v2-parallel-validation-20260926.md)。
