# v2 KV 寻址与 UCM block 排布

完整设计见 [UCM Connector v2 详细设计](../../../../../docs/connector-v2-detailed-design.md)，包含 scheduler、hash/窗口规则、worker 生命周期和验证边界。本文件保留布局接口与数值示例。

## 当前 worker 主路径：二维 Transfer

```python
for transfer in layout.build_load_transfers(metadata, layer_id=7):
    proxy_adapter.submit("load", transfer)
```

`UCMProxyTransfer` 的字段与约定：

```python
keys: tuple[bytes, ...]          # [K]，本批次唯一
ptrs: np.ndarray                # [K, S]
sizes: np.ndarray               # [K, S]
ucm_block_offsets: np.ndarray   # [K, S]，相对完整 UCM block
```

同一 plan 的多个 group 在列方向合并，顺序为 group -> window block -> layer/component/segment。每个 plan 属于单个请求的 hash 链，keys 已保证唯一，直接生成一个矩形 Transfer，不扫描或分区。不同 plan 的 S 可以不同，分别下发，不补零。不同请求共享前缀时可以有相同 key；请求各自的 Transfer 保留各自的目标地址，不做跨请求去重。

固定 sizes/offsets 通过每次调用独立的一维模板 broadcast 成二维只读视图；ptrs 为本次计算结果。多 group ptrs 当前仍按列 concatenate，没有实现直接写入最终矩阵。Proxy 不要求连续数组，不修改数组，异步返回 task 时由 adapter 的现有 wait 契约保持描述符存活；adapter 当前依旧同步等待，不是新增异步执行器。

单 group 直接传递本次生成的矩阵，不再重复拼接 sizes/offsets。layout 不扫描 keys 唯一性，也不维护相关缓存。不缓存 ptrs 或跨批次复用可写缓冲区。

原生 `Adapter.submit` 信任构造层的契约，直接传递 keys/offsets/ptrs/sizes 原对象，不重复校验 key 长度、唯一性、dtype、shape、数值或范围，也不转换数组。Adapter 的 key 校验缓存已删除。操作分派、空批次跳过、异常包装及既有 wait 行为保留；旧扁平兼容入口的校验不属于这条路径。

性能基准以“构造 Transfer → Adapter 转发 → 空后端接口返回”为口径；不把文件 Proxy 的解析、字节搬运或 IO 算入接口下发耗时。此前含 Adapter 校验的历史耗时不代表移除校验后的结果。

文件 Proxy 对二维输入逐 key 行处理，不创建全批次 segment-to-key 字典；实际字节 IO 时才迭代这一行的 segments。dump 按 offset 累积写入私有 `.tmp`，`commit(keys)` 才发布为可见文件。调用方负责保证完整性。

`use_layerwise: true` 接入 worker 逐层回调：start_load 提交各模型层加载，层回调等待该层关联缓存；状态层在 forward 前等待。save 回调只提交当前 layer_name，步末补存未回调的缓存，等待全部任务后 commit。未配置时仍走 bulk。当前文件 Proxy 同步执行；替换为异步 Proxy 时，任务保留 Transfer 数组直至 wait 完成。所有任务在本次 wait_for_save 返回前完成，不跨调度步。

layerwise 不复制 Block First slot padding，因此文件后端使用独立的 `-layerwise` namespace，暂不混用 bulk 文件。model-check 在此模式下按每个 layer_name 的有效 payload 填充和比对，bulk 模式继续比对完整 span。

旧 `build_load_batches/build_dump_batches`、`UCMProxyBatch` 与四参数扁平 adapter 方法暂留作旧工具兼容入口，**worker 已不使用它们**。不要用旧入口测量二维接口收益。下文的旧扁平示例用于解释相同的物理地址及磁盘字节位置；二维形式将 keys 变为 `[A,B]`，后三个数组按两行排列。


建议按 `view.py -> group.py -> ../record_layout.py -> ../ucm_kv_cache.py` 阅读。

## 谁负责什么

| 对象 | 输入 | 输出/职责 |
|---|---|---|
| `LayerView` | 一个 layer_name 注册的一个或多个 tensor views | 逻辑层描述，不分配 KV storage |
| `ComponentView` | 一个注册 tensor view 的 shape/stride | 一个或多个连续 `MemorySegment`；components 可共享 storage |
| `KVCacheGroupLayout` | 一个 group 的 LayerViews | 把源地址几何编译成 NumPy segment 列，按 layer_id/name 选择列 |
| `BlockAccess` | 固定 token 范围模板；运行时 block IDs、starts | 源内存 `ptrs/sizes`，形状 `[block, segment]` |
| `GroupRecordLayout` | group 的 FA/WA/State 规则与物理布局 | 在 UCM block 中放置数据，维护目标 offsets，判断能否合并 IO |
| `UCMKVCacheLayout` | scheduler 给出的 keys/windows 和可选层选择 | 组合 groups，关联 keys，构造 proxy 批次 |

`view` 表示逻辑视图；`segment` 表示能连续复制的一段。两者不是一一对应关系。`segment_mask/segment_count/selected_segments` 都针对展开后的列。

`BlockFirstView` 留在 group 中，它只描述源内存中带 padding 的连续 span。record 层只有在整块、未筛层、源 span 与目标 offsets 一致时，才选择合并 IO。多个声明的 offset 相邻还不够：实际起点和 block stride 也必须匹配。

## 两种 offset

- `block_byte_offsets`：源 segment 内，从块起点跳过多少字节；用于计算 ptr。
- `ucm_block_offsets`：目标 UCM block 中放在哪；每个 group 模板内先存 group 相对值，最终加 group 基址。

`GroupRecordLayout.group_block_offsets` 是**一个完整 vLLM block** 的目标排布；`ucm_block_offsets` 是**一个 UCM key 覆盖的窗口**排布。窗口可能是部分 block、多个 blocks 或整页 State，因此不能混用。`whole_block_bytes` 是完整 block 的目标空间，`record_bytes` 是这个 group 在一个 UCM block 中的总空间。

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

为兼容既有字节格式，完整 Block First 使用声明的 slots 并保留 padding；多 descriptor 时 slots 按 descriptor 排列，可能与逻辑 layer 遍历次序不同。部分块则按选中范围的 segment sizes 紧凑排列。layerwise 只筛列，不重新打包目标位置。

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

`UCMKVCacheLayout(spec, kv_caches)` 创建两个 group 相关对象：

```python
group_layout = layout.group_layouts[group_id]
record_layout = layout.record_layouts[group_id]
```

record 层确定固定长度 512，并调用：

```python
access = group_layout.compile_access(token_counts=512)
# access.segment_bytes = [[2048, 2048]]
# record_layout.ucm_block_offsets = [[0, 2048]]
# record_layout.record_bytes = 4096
```

sizes 在这里算好。运行时无需 ends；范围长度改变时需要重新选择或编译 access 模板。

### 2. scheduler 给物理 blocks，worker 算块内 starts

假设前两个逻辑 UCM keys 对应的数据都在物理 block 2（物理 block ID 不要求等于逻辑序号）：

```python
block_ids = np.array([2, 2], dtype=np.uint64)
starts = np.array([0, 512], dtype=np.uint64)
ptrs, sizes = access.resolve(block_ids, token_offsets=starts)
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

`GroupRecordLayout.resolve` 把相同 offset 模板重复到两个 key：

```text
key A: offset 0    <- layer 0 的 2048 字节
       offset 2048 <- layer 1 的 2048 字节
key B: offset 0    <- layer 0 的下一段 2048 字节
       offset 2048 <- layer 1 的下一段 2048 字节
```

`UCMKVCacheLayout._iter_plan_segments` 加 group 起点并关联 keys，最终扁平化为：

```text
keys:    [A,       A,       B,       B      ]
offsets: [0,       2048,    0,       2048   ]
ptrs:    [1131072, 2131072, 1133120, 2133120]
sizes:   [2048,    2048,    2048,    2048   ]
```

若只请求 layer_id=1，先选 segment 列再展开地址，得到 `ptrs=[2131072,2133120]`、`offsets=[2048,2048]`。目标 offset 不变成零，因为同一个 UCM block 中 layer 0 的位置仍被保留。

这个例子的数值由 `LayerViewSegmentsTest.test_documented_fa_subblocks_preserve_layer_offsets` 验证，覆盖物理寻址到旧扁平批次。实际 connector 通过 `build_load_transfers/build_dump_transfers` 下发对应二维矩阵，不需要自己展开 head 或解释 strides。

## 当前边界

- 官方 0.29 四维 view 按 BHNC 语义；Ascend 0.26 按 BNHC 语义。不能靠轴大小猜测。
- State 整页；token/state 换算必须整除。
- 多个连续 kernel rows 支持；多个 kernel rows 且 head 分离仍明确拒绝。
- NumPy 做批量地址计算，不保证多核并行。本轮职责整理不引入地址结果的长期缓存。


## 输出数组的生命周期

物理寻址 `BlockAccess.resolve` 返回独立可写的 ptrs/sizes；二维 Transfer 则使用独立 ptrs，以及本次调用小模板广播出的只读 sizes/offsets。单组直接持有这次结果，多组再拼接；后续 dispatch 不会覆盖已下发的描述符。这里的独立性针对描述符数组，KV 源内存本身的在途保护仍由既有生命周期机制负责。
