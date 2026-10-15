# v2 chunk 元数据、HBM 寻址与外存排布

2026-10-15：按公共 chunk 契约、实际物理布局和存储模板分清职责。当前参考 vLLM 0.30.0；本批仅静态检查，未运行组件/模型测试或设备 IO。

完整链路见 [v2 详细设计](../../../../../docs/connector-v2-detailed-design.md)，更新说明见 [0.30 适配记录](../../../../../docs/connector-v2-030-adaptation.md)。历史图文位于 [qyh/docs/v2-layout](https://github.com/qyh111/unified-cache-management/tree/docs/v2-layout)，其中旧的 BlockAccess 和命名需按本文更新；主开发目录不维护 HTML/SVG/生成脚本。

## 对象与职责

| 对象 | 内容与职责 |
| --- | --- |
| `UCMKVCacheGroupSpec` | 原生 group ID、`tokens_per_block`、kinds、tail、参与标记；不含 tensor 指针或存储模板 |
| `UCMKVCacheSpec` | groups、`tokens_per_chunk`、scheduler 粒度和按原生 group ID 索引的 `blocks_per_chunk`；scheduler/worker 共用 |
| `UCMLayerSpec` | worker 逐层原生 spec、storage entries、descriptor、轴约定；不含物理指针 |
| `LayerView` / `ComponentView` | 实际 tensor 的 shape/stride 和连续 `MemorySegment`；不分配或复制 KV storage |
| `KVCacheGroupLayout` | HBM 几何、`layer_specs`/层选择、block ID 到 segment 起始地址、token→byte 物理换算；不持有 GroupSpec/chunk 参数 |
| `GroupStoreLayout` | 持有公共 spec 和 group_layout；选择 token 范围，调用 HBM 换算并编译源 byte offsets、sizes、外存 offsets 和 store_bytes |
| `UCMKVCacheLayout` | worker 注册时汇总上述对象与层索引 |
| 外部 `UCMTransferBuilder` | 解释 plan、传递逻辑起点/IDs、累加各 group 外存起点、关联 keys 并生成二维 Transfer |

Layout 不依赖 scheduler/proxy，不接收请求 metadata/plan，不生成 Transfer。HBM 提供物理 byte 换算和 segment 起始地址，由 Store 选择范围并添加块内 byte offset；原 `BlockAccess` / HBM `compile_access` 已删除。

## chunk 的公共约定

启动配置和内部字段统一使用 `tokens_per_chunk`，替代 `ucm_cache_block_size`。开发期只使用当前字段，不保留旧名兼容或迁移校验。

`tokens_per_chunk` 是逻辑记录/恢复边界的 token 粒度，不是固定字节数，也不保证每推进这些 token 就立即执行 IO。`UCMKVCacheSpec.blocks_per_chunk[group_id]` 是一个 key 要引用的物理 block ID 数，在 spec 构造时统一派生；CP 修改有效 token span 后通过同一入口重新派生。

| group | 每 chunk block 数 | 保存范围 |
| --- | --- | --- |
| FA | `ceil(tokens_per_chunk / tokens_per_block)` | chunk 的 token 区间；部分、一个或多个完整 block |
| SWA | `ceil(tail_tokens / tokens_per_block)` | chunk 恢复边界前的 tail；首块可以只保存后半部分 |
| State | 1 | 恢复边界上的一个完整快照，与 chunk 跨几个 State token spans 无关 |
| 原生不参与项 / 零 tail | 0 | 不进入存储路由；不参与项也不创建 HBM/Store layout |

显式 chunk 粒度必须为 scheduler 粒度的正整数倍；FA 的 chunk/block token 数须互相整除；非零 SWA 和 State 的 block token span 须整除 chunk 粒度。HBM 不支持部分 token 时，Store 在初始化阶段拒绝该范围，按层选择不放宽整组能力限制。State token/state 换算必须整除，因此不允许部分快照。State 的 chunk 粒度已保证整除其 block span，所以 Store 直接与 FA 共用 `tokens_per_chunk % tokens_per_block` 的分支。

持久化集合直接取 `KVCacheConfig.prefix_cacheable_group_ids`；scratch 重建属于引擎。未支持的持久 spec（chunked local、cross、encoder-only、UNKNOWN）明确拒绝，Sink Full Attention 为显式允许项。

## HBM 与 Store 的接口

```python
layout = UCMKVCacheLayout(
    spec, kv_caches, kv_cache_config=worker_config,
    num_hidden_layers=num_hidden_layers, use_layerwise=use_layerwise,
)
group_layout = layout.group_layouts[group_id]
store_layout = layout.store_layouts[group_id]
mask = group_layout.segment_mask(layer_ids=[7])

# HBM 只返回 [block, segment] 的起始地址，不解释 chunk 或 token 范围。
bases = group_layout.resolve_ptrs(block_ids, segment_mask=mask)

# Store 返回 [key, IO entry] 的三个矩阵。
offsets, ptrs, sizes = store_layout.resolve_matrices(
    block_ids, token_starts, segment_mask=mask,
)
```

HBM 几何列包含 `base_ptrs`、`block_strides`、`states_per_block`、`payload_bytes`。每层可以有多个 components/segments，layer ID 不必只返回一个地址。

Store 在注册时把每 chunk 的固定 starts/counts 换成 `[window block, segment]` 的 `block_byte_offsets` 与 `segment_bytes`。`group_layout.tokens_to_bytes(tokens, tokens_per_block, segment_mask)` 直接计算 `tokens * payload_bytes // tokens_per_block`，先乘后整除，不经过 stored states。逻辑 block span 由调用方传入，HBM 不保存 chunk/window 信息。物理部分访问能力与完整 entry 对齐在 Store 初始化的固定范围编译中统一校验；states_per_block 仅用于该初始化检查，删除无其他用途的 state_strides。

运行时 HBM 只计算 `base + block_id * stride`；Store 把它组织为 `[key, window block, segment]`，添加固定/动态 byte offsets，最后展平。scheduler 在 plan.token_starts 中提供与 keys 一一对应的 uint64 逻辑 chunk 起点；TransferBuilder 原样传递，Store 对非 SWA 直接用 `token_starts % tokens_per_block` 得到块内偏移，不生成起点序列、不判断 chunk/block 余数。每 key 的偏移广播到该 key 的各个 block；完整 block 情况自然为0。SWA 使用固定 tail 模板。

token offsets/counts 与字节数组使用 uint64。starts/counts 由 build 根据 block span 和余数直接构造，不重复校验块内范围或正长度；仅 partial_tokens 非零时检查整组部分访问能力及完整 entry 对齐，完整块跳过。动态换算不重复检查。scheduler 保证起点为 chunk 大小 U 的整数倍，初始化保证 U 与 block span T 互相整除：U<T 时起点余数最大为 T-U，U>=T 时余数为0，因此不逐批重复检查动态 payload 边界。block IDs 信任引擎/公共计划契约，不在 HBM 重复扫描范围。

## 存储字节格式与 padding

通用记录顺序是 group → window block → layer/component/segment；每段紧凑累计 offsets。选择某层仅筛列，保留它在完整记录中的 offset。

- `use_layerwise=True`：始终使用紧凑 payload 前缀和模板。
- `use_layerwise=False`、完整 block、已有 `BlockFirstView`：每个物理 block 可以复制整个 group span，含组内 page padding。
- partial token：使用通用 segment 模板。

HBM 发现连续 span 并提供 base/stride/长度；Store 决定是否采用它。仅 CUDA/CPU 原生 BLHNC/BLNHC 尝试整组路径；NPU 使用实际 per-component views。State 同样可走完整 padded span；不复制本组 span 之外的 global block 空闲容量。

本批不改变已有非空记录的 payload 顺序、padding 或目标 offsets，namespace 保持 r7，继续隔离 bulk/layerwise。零 tail 不再构造无路由使用的占位存储模板。

## FA 16384 / 512 数值例子

两层均为每 token 4 bytes；block-0 基址分别 1000000 / 2000000，物理 block stride=65536 bytes，B=16384，U=512。

```python
block_ids = np.array([2, 2], dtype=np.uint64)
token_starts = np.array([0, 512], dtype=np.uint64)
offsets, ptrs, sizes = store_layout.resolve_matrices(
    block_ids, token_starts,
)
```

```text
ptrs   = [[1131072, 2131072], [1133120, 2133120]]
sizes  = [[2048, 2048],       [2048, 2048]]
offsets= [[0, 2048],          [0, 2048]]
store_bytes = 4096
```

选择第二层时 offset 仍是 2048。源/目标物理 block ID 来自 scheduler 的 table，与 key 序号无直接相等关系。SWA 的 B=8、tail=12 则编译 starts=[4,0]、counts=[4,8]。

## ABI、并行和生命周期边界

- view 明确归一化到四维 BHNC，不按 shape 猜轴，不恢复二维/三维通用路径。
- Ascend 原生独立 K/V 或去 padding views 的组件 stride 可不同于 descriptor 整页 stride；直接使用实际 view。CUDA/CPU 保留原生 descriptor 一致性校验。
- Ascend typed State 先把首块的稠密物理 payload 零拷贝展平，再转 byte page；保留 dense permutation、storage_offset、block stride。不调用 contiguous() 复制 KV。
- compact HNC 整块保留连续段；分离 heads 按物理地址分段/合并。多 head HNC 不接受部分 token。
- CP 仅支持完整分布式 FA 页：公共 tokens_per_block 是有效全局跨度，本地物理 states/stride 仍来自原生 view。
- PP 空 group 保留公共 spec 和计划位置，不构建本地 layout；分片可见性与 commit 不属于 HBM。
- State 精确保存边界、CUDA A+B / Ascend A→B、等待与错误处理保持。State layerwise 计算流水仍待实现。
- ptrs 是每次调用独立数组；sizes/offsets 广播自本次复制的小模板，在途 Transfer 不被下一次 dispatch 覆盖。源 KV 内存生命周期另由 worker 处理。

组件夹具位于 `test/suites/Unit/test_ucm_connector_v2.py`：按 0.30 接口重整，保留 hash/proxy 检查并覆盖上述范围、独立元素地址 oracle 与 State byte page。当前没有执行，不能据夹具数量宣称通过。

## 开发期契约与检查边界

内部 Transfer 使用匹配的二维 `[K,S]` 数组和已生成的 bytes keys；Adapter/File Proxy 不重复扫描 key 长度、重验矩阵 shape 或兼容旧 flat 输入。block IDs 在 HBM 求址入口统一为 uint64，Builder 不再重复转换。Store 在 build 内直接创建 starts/counts 并换算字节模板，不保留 _compile_ranges 包装入口。

并行正数参数由引擎配置验证；UCM 保留 DCP 为 TP 子组的实现限制（上游允许的 PCP-spanning DCP 尚未实现），以及 CP 整页和不支持 State/SWA 的限制。实际 view 初始化不变量、chunk/窗口范围与物理对齐、proxy 返回命中数量/下标、短读/文件范围、任务排空与失败处理仍保留。可选 tensor 注册与异步 wait 使用 Protocol，不改成吞异常或猜默认值。

### 部分 token 能力的来源与使用

`ComponentView.supports_partial_tokens` 在 `_build_bhnc_view` 按真实 shape/stride 设置：物理 NHC 或 singleton head 的 token-contiguous view 为 True，多 head HNC/分离 heads 为 False。GroupLayout 对全部 components 取 all，得到整组能力。Store 在初始化编译固定范围时检查它，layer mask 不放宽整组能力。运行时 token→byte 仅按 payload 比例换算。

该 bool 只说明物理排布能否按比例切 token 范围，不替代完整 stored-entry 对齐。State 通过公共 spec 的完整快照约束与固定范围初始化检查保证完整存取，不在每次 byte 换算时重新判断。
