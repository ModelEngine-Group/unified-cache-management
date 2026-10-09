# Connector v2：0.30.0 第一批适配

日期：2026-10-06。工作分支 `dev_connector`，基线 `c6324b8a5d1a6ae3dc8447b390a664869bc84698`；以下实现为未提交改动。源码参考 vLLM `ced6857afa0ea7b2e3f0846a62e1394e90f15607`、Ascend main `f6992d8c4657d02ae75b7d4b92bf450f9f3dbeba`。

用户要求当前先改代码，暂不测试。本批没有运行组件测试、模型构造、CUDA/NPU IO 或 forward。仅做 Python AST 语法检查、diff 空白检查与源码核对；不能作为模型支持或通过验收的证明。

## 本批实现

### Scheduler 与 State

入口直接消费 `SchedulerOutput.kv_connector_block_state`，每步用 `get_block_ids(req_id)` 替换请求的 group block tables。没有旧版 new/cached 增量表 fallback。

State 保存只取 `boundary_state_offloads` 给出的 `(group_id, block_id, boundary_tokens)`。同一边界的全部 State group 都有交接、且边界为 UCM unit 的整数倍，才组装该 State 记录；不按 token 位置从 block table 猜保存源块。只保存请求已建立 hash 链覆盖的完整边界，缺少 group 的交接不发布。

保存源和加载目标分开处理。目前 State 加载针对 **V1 model runner**：

| 平台 | 原生顺序 | connector 加载目标 |
| --- | --- | --- |
| CUDA | State precopy → connector load → forward | 同时填导入边界槽 `(restore_tokens - 1) // group.block_size` 与本步运行槽 `(restore_tokens + scheduled_tokens - 1) // group.block_size`；保证边界块可供本地缓存复用，UCM 不重复 dump 已命中前缀 |
| Ascend MRV1 | 准备 State copy → connector load → 执行 copy → forward | copy 源边界槽：`(restore_tokens - 1) // group.block_size` |

二者都从本步精确表取物理 block ID。State + V2 model runner 当前显式拒绝，避免混用不同 load/copy 顺序。这里的 model runner v1/v2 与 UCM connector v1/v2 是两件事。

WA 与 State 同时参与恢复时，lookup 会反复缩小边界，直到所有边界记录在同一 token 位置存在；不能取各链最新命中的最小值后直接恢复。

### 原生布局和 owner

- 0.30 的 CUDA/Ascend spec `block_size` 均按逻辑 token span 使用，删除旧 Ascend `block_size * compress_ratio` 解释。
- 直接读取 native `tokens_per_state`，保留其 Fraction 类型，不再把解析失败转换成 ratio=1。State 不走 attention 压缩分支。
- GLM5.3 主 KV ratio=1，indexer spec 带 `index_kpool`；Qwen-Next 主 KV ratio=1，compressed indexer spec 带 `indexer_compress_ratio`。worker 直接读取逐层原生 `tokens_per_state`；scheduler 只保留 group 恢复语义，不重建物理 owner ratio。
- Ascend combined attention view `[2,B,N,H,C]` 或 `[2,B,H,N,C]` 在入口拆成 K/V，再按 worker 绑定的 view 轴语义统一为 BHNC；仅 permute 元数据，不复制 tensor。普通 Ascend attention 默认 BNHC，`LBHNC/HND` 为 BHNC；MLA/SFA 保持 BNHC，cache-only/HiddenState 保持 BHNC。
- 跳过 MLA tuple 中的空 component（如 `head_size_v=0`）；不能把第二个 component 一概解释成独立 RoPE 数据。
- State combined byte page 区分有效内容、spec page capacity 与 descriptor block stride。Block First 的 stride 可以跨越其他层；State IO 只复制 spec 的 shapes/dtypes 对应内容，不保存页尾 padding。显式组件 tuple 也按实际 view 处理。
- State 禁用含 padding 的 whole-group Block First 快路径；FA 的整块快路径保留。
- PLE/GDN/KDA 使用本 rank 的 State view。PLE 的 `tp_replicated` 不触发再次除 TP；仍逐 rank 存储，未做 replicated 数据去重。单层 State group 不补层数。

### GLM tail 与 Qwen ring

明确标记 transient，保留原生 group ID 和 block table 的位置，但排除其 record layout、layerwise 保存入口和持久化路由：

- GLM 按模型类型和原生 Kpool tail spec 识别 tail，要求完整池恢复边界。
- Qwen 按模型类型和 `CircularBufferSpec` 识别 raw ring，要求完整压缩组的恢复边界；原生模型/spec 构造负责 ratio、capacity、window 的有效性。
- 原生模型成对创建 raw/compressed 或 tail/k cache；UCM 不再拼接名称、重复检查配对 owner。`transient_alignments` 只记录各 scratch group 的恢复对齐 token 数，检查最终 UCM unit，而不仅检查 scheduler block。

这些规则针对原生实现中只保留未完成压缩组原始数据的 scratch/ring。没有根据 `prefix_cacheable=False` 通用排除，也没有将 DSV4.1 ring 套用到 Qwen 规则。DSV4.1 CircularBufferSpec 当前仍拒绝。

### 生命周期与存储身份

- layerwise 声明 `requires_piecewise_for_cudagraph`，直接配置与 `UCM_CONFIG_FILE` 均使用同一 Config 解析。
- bulk 与 layerwise 都排除本步 load 失败请求的保存；bulk 完成后也标记 `_save_complete`，重复等待不重复提交。
- 无 forward 步骤的 metadata 携带 `no_forward`，在 `get_transfer_results` 补做保存等待；有 forward/draft 时遵循引擎原来的 finalization 时序。
- namespace 当前为 `r7`，摘要包含 HF text config、KV/Mamba dtype、resolved KV layout 和 Ascend 普通 attention 的轴模式，继续区分平台、模型 dtype、block size、layerwise 与并行拓扑。整组 Block First 快路径限定为 CUDA/CPU 原生布局，使用独立 namespace，不读取旧 r2/r3/r4/r5/r6 文件。设备地址与 allocation 容量不参与身份。

后端仍是开发用 `SimpleFileUCMProxy`。变长 Transfer 与 wait/commit 接口保留，没有接入生产 Cache|Posix，也没有将 v1 固定列模板引入 v2。

## 尚未完成与下一步

1. 将既有轻量单测的 SPI/spec doubles 更新为 0.30，替换依赖旧 State 位置推断和 0.26 Ascend 语义的用例。本批没有修改或运行这些测试；此前 111 项基线结果不能代表本批。
2. 补精确 CoW/offload 源块、State 加载目标、缺少任一 State group、WA/State 不同命中边界、无 forward 和 deferred draft finalization 的契约验证。
3. 核对 terminal/finished 请求的 State handoff、源块复用、抢占/取消和 checkpoint 数值；缺少有效边界时当前只会漏存，不能用位置推断补造快照。
4. GLM5.3 原生构造与 IO 尚未验证；Qwen-Next 的 QSA backend 仍声明 `supports_kv_connector=False`，Ascend 的 QSA/PLE 原生模型构造也未打通。本批 parser 能表达角色，不等于引擎已允许运行。
5. DSV4.1 的同源 ring 恢复语义、A5 FULL/indexer K/scale alias owner、量化组件与 SWA 有效窗口还需独立适配。禁止重叠范围重复保存/恢复。
6. State + V2 model runner 暂未实现。现有 PP/CP 等能力也没有新验收，不将 TP 或历史结果外推到这些路径。
7. 生产变长 store/proxy、IO stream/event 契约与发布完整性仍需开发；file commit 不能证明 fsync durability 或跨 rank 原子发布。
8. State layerwise 加载与计算流水尚未实现，用户已确认先记录为待实现项，当前继续 review 其余代码；具体范围见下节。

远端仍按用户最后说明视为下电，不尝试连接。恢复后先验证固定 SHA 的无权重原生构造、再做真实 IO/数值/生命周期与权重推理，bulk 和 layerwise 分别记录结果。

## State layerwise 待实现（2026-10-06）

用户确认：本项先记录，不在当前 review 阶段实现。以原生 vLLM 0.30.0 / Ascend MRV1 为依据，旧 UCM patch 不纳入本次适配。

当前支持范围：Attention/MLA/indexer 已有按层提交、等待和保存的接线；State 虽然能按层拆 Transfer，但 `start_load_kv` 返回前仍统一等待所有 State 层，尚无 State 加载与逐层计算的流水。开发用 `SimpleFileUCMProxy` 同步执行 IO，也不能证明真实异步重叠或运行时支持。

实现范围与顺序：

1. 先接 State 逐层加载，保留当前 State 保存的结束阶段处理。`start_load_kv` 提交任务后，不再统一等待所有 State；在每个实际 State owner 使用 cache 前等待其任务。CUDA GDN/KDA 与 Qwen PLE 的入口需分别核对，不能用同一模型层号的 Attention hook 代替全部 State owner 的等待。
2. CUDA 保留原生 forward 前的 precopy，每层将同一外存快照加载到导入边界 A 和运行槽 B，然后才计算该层。A/B 是两个加载目标，不是两份外存记录。
3. Ascend 接入 `prepare_mamba_state_copy` / `finish_mamba_state_copy`，让 MRV1 跳过 forward 前的批量 copy；该层 `wait_for_layer_load` 完成后，调用 `do_mamba_copy_block_for_layer` 执行 A→B，再计算。参考原生 `AscendStoreConnector`；Ascend GDN/Kimi KDA 已有计算前等待入口，PLE 仍须独立核对。没有外存 load 的本地命中层也必须执行所需 copy。
4. 接异步 proxy 和 stream/event 依赖：CUDA precopy 必须先于 IO 对 A/B 的写入；两平台该层 load 必须先于该层使用 State，Ascend 还需先于该层 A→B。当前 Torch 开发 proxy 在调用线程的当前 stream 写入且 load 返回前同步，不额外增加逐次设备同步；独立 IO stream 的后端需要显式建立跨 stream 等待，不能用 load 后同步补救已发生的竞争。
5. 后续再做 State 逐层保存：源块仍来自精确 `boundary_state_offloads`，在对应快照内容就绪后提交，全部必要范围完成后 commit。不根据运行槽 B 或层计算结束的位置猜快照；依赖原生后处理的边界必须等后处理完成。

后续验证范围：CUDA A/B 双目标与再次本地命中、Ascend 每层 load→copy→compute、无外存 IO 的本地命中 copy、PLE 与同层其他 cache 的 owner 区分、图执行下 hook 可达性、跨 stream 竞争与 State 快照就绪/发布时机。当前未实施此项，也未运行验证。

## View 地址解析复查（2026-10-07）

- 2026-10-09 Layout 职责收拢：按用户要求保留现有文件组织，不强制拆成三个模块文件。配置元数据/parser 与纯 UCMKVCacheLayout 移入 layout/kv_cache.py，store_layout.py 移入 layout 目录；原 __init__/group/view 继续绑定和解析 HBM。Layout 不导入 scheduler/proxy，不接收 metadata/plan，不生成 Transfer；block IDs/token offsets/mask 的 byte-range 寻址仍属于 Layout。外部 ucm_kv_cache.py 只保留 UCMTransferBuilder，组合纯 Layout 解释 plan、累计参与 group 的外存偏移和生成 Transfer；dispatch_routes(spec) 移至 scheduler。connector 持有 layout/transfer_builder，工具、测试调用与当前文档已迁移，无旧路径/方法兼容壳。寻址/State/padding/存储格式和 r7 不变，仅静态检查，未运行测试，旧夹具待完整核对。
- 2026-10-09：GroupStoreLayout.record_bytes 改为 store_bytes，表示该 group 在一个 UCM key 下的外存总字节数；构造、group 累计偏移、调试输出与当前文档引用同步，不保留旧字段别名。覆盖历史记录中的旧字段名，字节排布与 namespace r7 不变；仅静态检查，未运行测试。
- 后续简化通用 ucm_block_offsets：access.segment_bytes 已包含首块部分 token 的实际字节长度，因此删除逐 row 的 partial_tokens 特判；按矩阵 row-major 展平做 cumsum，再减当前 size 并 reshape，统一得到所有 blocks/segments 的紧凑偏移。record_bytes 为实际 sizes.sum()；Block First bulk 仍保留 offset0/整组 span 路径，partial_tokens 仍用于编译实际范围与整块 eligibility。字节排布/r7 不变，仅静态检查，未测试。
- 用户纠正 Block First/通用模板关系：此前仅按是否选层切换、layerwise 保留原生 page 间隙的实现不符合要求，已修正。GroupStoreLayout.build 必须接收 use_layerwise；True 时保留通用 cumsum(payload_bytes)-payload_bytes / sum(payload_bytes) 模板，与物理 Block First 无关；False 且 BlockFirstView 存在且访问完整块时，另用 offset0/block_size_bytes 的 bulk 通路。worker register 显式将 self.use_layerwise 传入 UCMKVCacheLayout，再传入各 group store builder；部分块仍使用通用模板。layerwise 记录由原生 slot 排布变为紧凑 payload 排布（可能同时改变顺序、padding 和后续 group offsets），namespace 升 r7 隔离 r6。既有多 descriptor 测试补充 True/False 模板差异断言；只做 AST/compile/diff 静态检查，未执行测试/IO。此项覆盖下方前一版“两条路径/字节位置不变/r6”的说明。
- 用户确认 Block First 保留 layerwise，同时增加更简单的 bulk 路径：完整 group span 在 store 模板中是一项，group_block_offsets=[0]，ucm_block_offsets 为每个原生 block 的记录起点（单列），record_bytes=block_count*span。build 不再按 layer_name/descriptor_position 重建 slots，也不再重复比较源/目标 offsets。resolve_matrices 在未选层时直接复制完整 span；选层时继续使用原有物理 segment 寻址和 payload sizes，目标位置由原生 segment 相对 group 起点的偏移加 block 记录起点得出，字节排布不变。部分块仍走 segment 模板；NPU不进入整组Block First。同步更新3处模板形状断言，layerwise 数值预期保持；仅 AST/compile/diff 静态检查，未运行测试，r6不变。
- 用户确认存储排布命名：record_layout.py 改为 store_layout.py，GroupRecordLayout 改为 GroupStoreLayout，UCMKVCacheLayout.record_layouts/local record_layout 改为 store_layouts/store_layout。源码、单元测试、布局 README、设计与 review 链接同步，不留旧模块别名；历史日志中的旧名由本项覆盖。职责仍是每个 group 在外存记录中的 offsets/大小/IO 合并，record_bytes 等记录格式字段不变，namespace r6 不变；仅静态检查。
- 用户 review DispatchPlan 命名：windows 字段改为 group_block_ids，表示按 dispatch_routes 参与 group 顺序排列的物理 block ID 数组，每个数组按 key/chunk 顺序展开。scheduler 的 State 双目标 replace、worker 失败块收集、layout zip、model-check 签名与单元/NPU probe 引用同步更新，不保留旧字段别名；记录格式/r6 不变，仅静态检查。
- 用户确认 compile_access 总是编译整组：删除 layer_ids 参数、BlockAccess.selected_segments 字段及 resolver 的双 mask 交集合并。所有模型层/组件的字节 offset/size 仅编译一次；运行时 resolve_ptrs/record 使用 segment_mask 选择本次层。5处旧测试的编译期层选择迁移为运行时 mask，测试辅助函数也删除相应分支。group 部分 token 策略、记录格式/r6 不变；仅静态检查，未运行测试。
- 用户要求整体清理未使用函数：扫描 v2 全部 Python 文件，并追踪当前 ucm/toolkit/test 的普通、字符串动态调用与 vLLM 0.30 SPI。删除仅旧兼容链使用的 build_load_batches/build_dump_batches/_build_batches、GroupRecordLayout.resolve、UCMProxyAdapter._batch/load/dump、UCMProxyBatch/record_sizes；_iter_plan_segments 只保留二维分支。BlockAccess.resolve 与 _resolve 合并到唯一 resolve_ptrs，删除失去调用的 segment_count 和未使用 layer_slices；删除 attn_groups/sw_groups 与 Transfer.total_bytes 冗余属性。旧 UCMDispatcher.build_metadata 只被测试使用，删除后测试改走 build_from_scheduler_output。vLLM 的 requires_kv_delivery/requires_piecewise_for_cudagraph/get_transfer_results/request_finished_all_groups 等 SPI，以及 proxy Protocol/getattr 调用的方法保留。
- 本批工具迁移：model-check 的字符串 builder 调用改为 build_*_transfers，再仅在采样时展开 [K,S]。单元测试旧调用改为当前二维接口，保留既有字节断言；删除专门验证已删除 legacy adapter 校验的测试。NPU probe 改用当前 worker 注册/layout、route plan 和二维 submit/commit，源目标及跨 group 使用不同 block IDs，避免共享 pool 覆盖。仅改脚本，未执行；它是布局/文件 IO probe，不证明 State 原生加载/运行生命周期或跨 rank 恢复。
- 清理验证：只做 AST/compile、静态引用及 git diff 检查，未导入 connector、执行测试或设备 IO。旧 spec/SPI/view 夹具仍需完整迁移，不能把本次接口引用迁移当作测试通过；scheduler State 夹具仍需真实 boundary_state_offloads，旧扁平顺序断言也需按二维 key-row 顺序核对。主路径记录格式与 namespace r6 不变。以上覆盖早期文档保留 legacy batch/resolve API 的说明。
- 用户指出 extract_segments 无主路径调用：搜索当前 ucm/toolkit/test/docs 确认仅旧单元测试引用，已删除 group 的兼容包装入口（包含空 blocks 和 State 特判）。旧测试的5处调用改为 compile_access(...).resolve(...)，不增加新测试；State 仍由统一范围/整除约束保证整快照，不保留忽略传入范围的旧包装行为。源码/工具/测试已无该接口引用；旧 spec/布局夹具的其他迁移仍待完成。本次只做 AST/compile/diff 静态检查，未运行测试，记录格式/r6 不变。
- 用户 review compile_access / BlockAccess 后清理：删除 compile_access 内部非空/一维重复校验（调用方契约），删除 State 整块特判（初始化 storage_block_size=1，实际 view stored-state 数为1；固定范围 + tokens_to_segment_bytes 整除已经保证整快照）。BlockAccess 先生成 base + block_id*stride + fixed offset，然后在同一个动态分支完成 offset 换算、范围校验和相加，去掉 dynamic_offsets=None 和第二次分支；保留 block ID、动态 token 起点上下界与最终 payload 范围校验，层选择交集是功能语义而非重复校验。删除 _checked_blocks 冗余 len 判定和整除失败时逐列构造 layer name 的诊断分支，错误按 group 标识。模板非空/一维、block 窗口数量由内部调用方保证，NumPy 广播/reshape 按契约运行；没有新增 fallback。记录格式和 namespace r6 不变；仅 AST/compile/diff 静态检查，未运行测试。
- 用户后续确认部分 token 为 group 整体布局策略：删除 MemorySegment 的 supports_partial_tokens 字段与 group 的逐 segment bool 数组。`_build_bhnc_view` 在解析每个实际 tensor view 时生成 ComponentView 布局能力，group 对全部 components 取 all 得到单个 bool；任何 view 不支持则整个 group 只允许零 offset/完整 block。固定范围与动态 offset 都使用该策略，按层选择不放宽限制；compile_access 删除按 mask 分别编译的分支，统一编译整组后选择列。此次仅收紧允许的访问范围，不改已有记录字节位置，namespace 保持 r6。State 整快照限制保留。仅静态检查，未迁移/执行测试。
- 2026-10-08 用户确认 State 接受 Block First page padding：删除 group 的 `not is_state_snapshot` 排除，CUDA/CPU 满足原生 layout/descriptor 条件的 State group 与 attention 共用 BlockFirstView。现有 GroupRecordLayout 已按 descriptor slot offset 和完整 span 建记录，bulk 在 offsets 一致时合并为每 block 一段 IO；层选择继续复制有效 payload，不复制 padding。State 整快照、A/B 加载和等待时机保持现有规则，不代表实现 State layerwise 流水。padding 可能改变 State record 大小及后续 group offsets，namespace 升 r6 隔离旧 r5 文件。仅 AST/compile 与 diff 静态检查，未运行测试或模型 IO。
- 2026-10-08 group padding 核对：`connector_v2_0917/models/dsv4/config.json` 前43个主层 ratio 为0×2、C4×21、C128×20，另有三个0条目；实际 group 数仍取决于 cache dtype/block size/投机/并行运行配置，未运行构造。CUDA 通用 packed allocator 的全局 block stride 为所有 group 的 `sum(per-layer page_size)` 最大值，各 group 的 descriptors 从 offset0独立打包并覆盖同一 pool，而非 group 与 group 顺序拼成一个物理 block。不同 descriptor 的 layer_stride/page_size 可以不同；global block_stride 相同。当前每个 native group 独立构建 KVCacheGroupLayout / BlockFirstView / GroupRecordLayout，组 span 是本组去重 descriptors 的 sum(layer_count*layer_stride)，含组内 page padding；源 block_id 步长为global block_stride，不存组span末尾到下一global block之间的空闲容量。route 内 group_record_offset 累加的是外存record布局，不能把它理解成HBM中多个group依次相邻。
- 2026-10-08 padding 核对：CUDA DSV4 的 DeepseekV4IndexerBackend 仅支持 BLHNC/BLNHC，DSV4.1 indexer 继承同一限制，两者原生路径使用跨层 Block First。DSV4 `fp8_ds_mla` 主 cache 的 stored row 为 584B、page 按 576B 对齐，FP8 indexer row 为 132B（含 scale）、page 同样按 576B 对齐；以 token block 256 / C4 为例，主 page 有效 37376B→37440B（pad64B），indexer 8448B→8640B（pad192B）。DSV4.1 根据记录格式选择 584B/576B 对齐或 MXFP8 的528B/512B对齐，也可能有 page padding。当前 BlockFirstView span 包含组内 page padding，长度为 sum(layer_count*layer_stride)；全局 block_stride 末尾超出本组 span 的空闲容量不在该 span 内。用户随后确认接受 State page padding，已删除 `not is_state_snapshot` 排除；State 使用相同的逐 group Block First 判定。未运行布局采集或 IO；DSV4.1 ring 恢复策略仍待实现。
- 最新 Block First 策略（用户确认）：NPU 不探测/不创建整组 BlockFirstView。CUDA/CPU 在 `build_group_layouts` 按原生 resolved layout 只启用 BLHNC/BLNHC，State 同样参与整组快路径并允许 page padding；group 仅保留 descriptor 存在、`0 < layer_stride < block_stride`、各 descriptor block stride 一致的判定。按最小 offset 的 descriptor 找实际 base，按 `sum(layer_count*layer_stride)` 算整组 span；信任 0.30 原生 allocator 将本组 descriptors 从 offset 0 连续放在一个 backing 的契约。删除逐层 segment 连续性、payload/layer stride 容量、跨 group descriptor owner 集合、descriptor offset 链与实际基址位移的重复检查，也删除仅服务这些检查的 `segments_by_name`。普通 view/descriptor block stride 一致性初始化校验和动态 IO 越界校验保留；record 的源/目标 offsets 匹配仍是合并 IO 的必要条件。平台/布局快路径选择可能改变 padding 和 record slot 放置；此前升 r5，现因 State padded record 升 r6，不复用旧文件。旧 group layout 夹具需显式提供 `block_first_layout`，只做静态检查，未迁移/执行测试。
- Block First review 补充：GLM5.3 CUDA 本身也走 `_glm5_next_tensor_layout` 专用配置分支（在通用 layout allocator 配置之前），每个 owner 独立 descriptor，`layer_stride=page_size*num_blocks`、`block_stride=page_size`；全部 slots 在一个 backing 的连续区域中，多个 group 的 descriptor 通过相同 offset 共享对应 slot。它不是跨层 Block First，不能只凭 resolved layout enum 判断。Ascend GLM 每个 slot 独立 backing，同 slot 的 owners 列在一个 descriptor 中，`layer_stride=0`、`block_stride=page_size`；同样不是跨层 Block First。注意区分物理共享槽、分配数量、单层 block-first K/V 与整个 group 可合并的 BlockFirstView；CUDA 普通 BLHNC/BLNHC 路径支持后者，不能推广到 CUDA 所有模型。当前源码检查未运行模型采集。
- 最新入口约束：用户要求删除目前没有必要的二维/三维通用解析。移除 `build_tensor_view` 及非四维 else 分支；K/V 拆分、BNHC→BHNC 和 Ascend State 字节页转换后只接受四维，统一调用 `_build_bhnc_view`。新非四维 attention ABI 需在相应入口显式规范化，不猜测 dense-row token 语义。`row_payload_bytes` 仍用于 Ascend State 字节页转换的密集内容校验。已有四维记录格式不变，namespace 保持 r4；只做语法/diff 静态检查，未运行测试。
- 上游 `KVCacheLayout` 明确描述 `[L,B,H,N,C]` 逻辑维度的物理 stride 排列；`KVCacheTensor` 给 layer/block stride、offset 和 allocation 大小。CUDA allocator 返回逻辑 BHNC view。Ascend backend 则可返回 BNHC/BHNC、combined K/V、MLA/scale tuple 或 tiled views，不能把物理 enum 名称直接当作每个 registered view 的逻辑轴顺序。
- worker 绑定逐层 `attention_view_order`，对照当前 Ascend backend/runner 的明确 ABI；view 入口规范化后只接受四维 BHNC，按实际 stride 判断完整 block 是否连续。物理 NHC 或 compact HNC 直接生成一个 MemorySegment；只有 heads 之间存在其他 block/layer 的间隔时才按 head 拆分。
- 用户随后确定访问边界：物理 NHC 和 singleton head 支持部分 token；多 head 的物理 HNC（包括分离 heads 的 LHBNC/BHLNC）只支持完整 block，不以增加每 head 小 IO 来支持部分 token。group 以单个 `supports_partial_tokens` 布尔值约束固定窗口与动态 offset；SWA 非整块 tail、自定义 UCM 子 block 遇到该布局明确拒绝。按 layer_ids 选择不绕过整个 group 的限制。State 整快照规则保持原样。
- compact HNC 的单层整块访问（含 layerwise）现在直接使用一段，不依赖整组 Block First 合并。完整记录仍保留原有 head-major 字节顺序；此次只改变连续字节的分段数量和允许的访问范围，不改变 record offset/padding，namespace 保持 r4。
- 后续补齐多行且 heads 分离的整块寻址：移除 `rows != 1` 的 ragged 拒绝，每个 row/head 按 `base + (row * stride[0] + head * stride[1]) * element_size` 生成段，逻辑 block 步长为 `rows * stride[0] * element_size`。按物理地址排列后合并首尾连续的段，合并时累加 payload 和该段的 stored-state 数；仍只允许整块访问。原先支持的单行记录顺序不变，多行分离布局此前拒绝，namespace 保持 r4。该能力不表示上游 planner/backend 会接受所有 kernel splitting/layout 组合。
- State 与 attention 现在连 segment 解析也共用：CUDA 0.30 注册给 connector 的 State 是 `[B,1,1,content_bytes]` int8 view（MambaBase.bind_kv_cache 仅为计算层拆分 conv/recurrent，不替换注册字典），直接进入 `_build_bhnc_view`。Ascend MRV1 注册 typed components；入口 `_as_byte_page` 用实际 shape/stride 检查 block 内密集内容，`view(torch.uint8)` + `as_strided` 将各组件规范化为 `[B,1,1,component_bytes]`，保留 storage_offset 和字节 block stride，只创建 view 元数据，不复制/分配 KV 内容。随后统一解析 MemorySegment / group / BlockAccess。
- 删除 `_state_tensor_views` 及 `build_tensor_view(state_snapshot=...)` 分支，不再用 spec.shapes/dtypes 重算 payload 或猜测 raw page 有效长度；CUDA 的 C 维、Ascend 组件 shape 自身给出有效内容，尾部 page padding 只体现在 block stride 中。规范化后的 State ComponentView 记录 BHNC 字节 shape/stride；原始注册 tensor 未修改。一块 State 仍是完整快照，由 group 的访问策略禁止部分 token，不涉及本次未实现的 State layerwise 生命周期。正常原生布局的 payload 顺序/record offsets 不变，namespace 保持 r4。
- 空 component 使用 `numel()==0` 排除；删掉对 zero-width RoPE 的过窄说明。该零字节规则同样适用于 State 组件，空组件没有 IO，有效的其他组件仍照常持久化。
- 删除 group 的 num_blocks 正数重复检查、四维路径重复的 rows/stride 正数检查；State descriptor stride 一致性只在 group 初始化检查一次。通用 trailing density 校验合成一个流程，忽略无地址位移的 singleton 轴，避免把 singleton head 的任意 stride 当成空洞。
- descriptor 缺失时仍使用各 view 的 segments；不是停止持久化。当前 CUDA/CPU 整组 fast path 信任原生 allocator 的放置声明，不再重复证明实际连续性；NPU 不进入此快路径，State 按相同 descriptor 条件参与。具体判定及 r6 namespace 见本节开头最新策略。
- 保留初始化时的真实布局约束：声明与实际 block stride 相符、channel/token 连续性、kernel rows 拼接、Ascend 字节页规范化时 trailing 维密集、窗口恰好对应整数存储条目，以及整组快速路径的适用条件。动态 block ID 与动态 offset 越界检查仍需在 IO 时处理；这些检查不能仅靠全局 layout 名字替代。
- 本次未实现 NZ/opaque quantization packed bytes 的通用 token 范围解析，也没有将任意五维张量都解释为 K/V 两份。五维 combined 路径仅对应当前 runner 的 leading K/V 轴；其他 packing 要按实际 registered ABI 单独处理。

仅做 AST 语法/编译和 diff 静态检查，没有导入/执行 connector、组件测试或模型 IO。待验证：Ascend BNHC/BHNC 和 `HND` 配置、MLA/HiddenState 对全局 GQA 设置的独立性、singleton head strides、空 component、相邻 K/V 的多 component 整块路径及 padding、跨 group descriptor、tiled rows 和动态访问边界；compact HNC 整块/layerwise 单段、LHBNC/BHLNC 分离段、多行分离 heads 的 block 步长/合并/源目标比较、多 head HNC 的固定/动态部分访问拒绝及混合布局的层选择；CUDA 原生 State BHNC 与 Ascend BF16/FP32 组件的 zero-copy 字节 view、非零 storage_offset、shared-slot/page padding 和 State 整块拒绝。旧夹具需使用真实 CUDA BHNC State / Ascend typed State ABI，更新私有 helper 名称、compact HNC 段数和部分 token 预期；未迁移/执行。

## 防御性代码清理（2026-10-06）

用户要求核对 `_concrete_specs` 及其他多余检查，已做以下简化，仍未运行测试：

- 删除 `alignment_block_size` 属性别名，唯一的 model-check 引用直接读取 `ucm_cache_block_size`。v2 中原名 `unit` 的局部变量统一改为 `ucm_block_size`，表示每个 UCM 记录/hash key 覆盖的 token 数；不改变大小选择、对齐或窗口计算规则。
- `_concrete_specs` 直接索引原生逐层 spec map，删除缺失键预扫描和 layer_names 的额外 tuple 副本；保留普通 spec/UniformType spec 两种真实结构的分支。
- 原生 descriptor 直接建立索引，删除空 layers、负 stride、重复 owner 等对已生成配置的二次结构验证。runtime view 与声明的 stride 对照、物理连续性和复制范围验证保留。
- 删除仅转发 `tokens_per_state` 的包装函数和初始化已确认的 indexer ratio 的重复断言/None 检查。
- State shapes/dtypes、worker metadata、speculative/parallel 配置和 config dump 的确定字段直接读取；可选 speculative 配置与不同模型的专属字段仍按实际结构处理。
- 内部计划删除重复窗口计数检查，group/window 配对使用 `zip(strict=True)`；后续 NumPy reshape 维持形状契约。固定 access 的范围在 `compile_access` 校验，resolver 仅重新检查动态偏移、block ID 和最终设备复制边界。
- 核对接口顺序时发现 0.30 可在无同步 load 的步骤把 `start_load_kv` 放到 forward 后。每步任务状态重置与前步排空检查已移至 forward 前的 `bind_connector_metadata`，避免误报本步 layerwise dump 未排空或清除本步保存状态。正常 forward 的 get_transfer_results 仍只报告结果，无 forward 分支补存的行为保留。
- 无 forward 补存进一步限定为 metadata 实际带有 dump plans；无保存计划时直接报告结果。依据是原生 scheduler 独立交接 boundary_state_offloads，零 scheduled tokens 不等于没有前一步 State 的保存工作。

## Parser 职责整理（2026-10-06）

用户逐项 review 后，先拆分成员展开、transient 策略、平台物理行数、sliding tail 与 unit 选择；随后进一步拆开两端职责，最新入口见下节。没有执行测试。

- `num_hidden_layers` 和 `indexer_tokens_per_state` 改为必传模型材料；无专用 indexer 的模型传 None。删除层数 None/非正、indexer ratio 等原生模型已验证字段的重复检查。
- PP 空本地 group 是真实原生情况。`_member_specs` 统一读取 group 的声明成员，不再在 State 分支反复用 `("", spec)` 补齐；实际 layer/view 仍仅来自 PP 本地 layer_names。
- `native_id` 统一命名为 `group_id`，保留原生 config/table 位置。Qwen 临时环使用 `is_transient` 和空 persistence kinds，不再把已识别的环标为 UNKNOWN。未知持久 spec 仍拒绝。
- 原生 CircularBufferSpec 同时被 Qwen QSA 与 DSV4.1 compressor 使用；当前 v2 仅接入 Qwen 的完整池恢复策略。DSV4.1 C2 环在无 speculative、完整 128-token 恢复边界时也有可省略的源码依据，旧 v1 专用实现已采用，但 v2 尚未接入对应策略。
- 删除 tail/ring 的重复 capacity/window/scheduler 容量检查，以及原生模型已保证的配对 owner 检查。保留最终 UCM unit 的完整压缩组对齐，这是 UCM 自身必须满足的恢复条件。
- 物理存储大小显式处理 AscendSlidingWindowMLASpec 的行数和 AscendSFAIndexerCacheSpec 的 DCP 复制倍数，删除通用字段猜测；当前实现移至 `layout._storage_block_size`。该倍数描述额外物理行，不是压缩比或 TP 分片。
- 多 group 可指定同一个 `ucm_cache_block_size`，要求正数且为 scheduler 共同粒度的整数倍，并满足各持久 group 的窗口/页和 scratch pool 对齐。State 默认选择 scheduler 共同恢复粒度，不再要求所有持久 group 原生 block span 相等；State mode 仍只接受 align。这是源码实现，尚无运行时验收。
- 本地 native tiering/offloading 的 `block_size` 支持多个参与 group，但要求它们 token block span 一致且 chunk 大小为整数倍；异构 span 使用共同 `blocks_per_chunk`。UCM 使用一个共享 token unit，与 native 的共同块数配置含义不同。
- layerwise 仅为实际提交的 load 创建 task 列表，避免无 forward 且没有 IO 的 metadata 留下空 task 项被下一步误判为未排空。

后续契约测试需要覆盖多 group 自定义 unit、不同 FA/State span 的共同边界、空 PP group、pool 恢复与 DCP 额外行。此前基线测试结果不能证明这些改动。

未知/不支持的 spec、scratch 恢复对齐、State page/stride 等初始化不变量仍保留。IO 错误上报、异步 task 等待、未排空任务与设备地址越界检查也保留；这些决定复制正确性与块生命周期。

## 统一解析入口与 worker 布局绑定（2026-10-06，当前实现）

- `parse_kv_cache_config` 两端共用，但只返回逻辑 `UCMKVCacheSpec` / `UCMKVCacheGroupInfo`：group ID、token span、FA/WA/State 路由、tail、恢复 unit、transient 与 eagle 标记。逻辑 group 不再携带 layers、原生 group spec 或 descriptor；scheduler 不计算逐层物理行数。
- 用户 review 后合并接口，删除 `parse_worker_kv_cache_spec`、`UCMKVCacheWorkerSpec` 和 `UCMKVCacheWorkerGroupInfo`，只保留一个公共策略 parser 和一套 group/spec 类型。worker 的 `register_kv_caches` 直接创建 `UCMKVCacheLayout(self.spec, kv_caches, kv_cache_config=self._kv_cache_config, num_hidden_layers=self._num_hidden_layers)`。
- `UCMKVCacheLayout` 初始化调用 `layout.build_group_layouts`，使用完整 worker config 绑定本地逐层 spec、num_blocks、descriptor 和层号，再解析实际 views。`KVCacheGroupLayout.layers` 保存这些 `UCMLayerSpec`；逻辑 `self.spec` 始终不携带物理层声明，不再制造一份 worker spec 或复制公共策略字段。
- 原局部变量 `rows` 改为 `storage_block_size`，attention 按 native `tokens_per_state` 计算物理条目数，保留 Ascend storage block / DCP 复制规则。State 不做 attention 行数计算，声明一个不可拆分的 checkpoint（值为 1），不再把逻辑 `spec.block_size` 当物理行数；实际快照内容仍按 shapes/dtypes 和真实 view 解析，传输范围与 schema 不变。
- 删除 `_attention_owner_ratios` 和从 HF config 覆盖物理压缩比的路径。worker 的完整逐层 spec 是物理 `tokens_per_state` 的来源，包含主 KV、indexer 和已注册 draft cache；State 单独处理，transient 不编译层布局。
- 原 `attention_tokens_per_state` 参数改名为 `compressor_tokens_per_state`，只用于旧 DSV4 `.compressor.state_cache` 的 `window - ratio` 历史尾部。普通 SWA 使用完整窗口；GLM tail/Qwen ring 走各自 transient 恢复规则。模型配置条目不再统一按主模型层数截断，但这不是完整 draft 配置合并，也不代表 DSV4.1/speculative 已验收。
- worker 的 State 提前等待使用 `group_layout.layers`；PP 空 shard 发布与 Transfer 组装按 `group_layouts` 中实际有本地层的 group 判断。原生 group ID、路由顺序、transient 排除和 PP 空 group 的占位策略保留。CP 先扩展公共 token span，worker 再按本地 native spec 绑定物理声明。
- DSV4 的 `compressor_tokens_per_state` 仍只用于公共策略的旧 compressor tail 判断，本次不迁移到 worker。CUDA State 加载 A+B、Ascend 加载 A 和精确 State dump 的规则均保持；State layerwise 继续作为待实现项。
- v2 包导出与物理 layout/record 类型已同步移除 WorkerSpec/parser。旧 SPI/spec doubles 仍未迁移：依赖 parser 返回 `.layers` / `.layer_to_group` 的夹具应改为查询 `layout.group_layouts[group_id].layers`；构建 layout 必须传完整 worker `kv_cache_config` 和 `num_hidden_layers`。不要给旧签名加 fallback 来构造不存在的物理信息。本批未修改或执行旧测试。

本批仅完成语法和 diff 静态检查。待验证重点：折叠 scheduler spec 与完整 worker spec 得到相同逻辑策略、scheduler 不依赖 placement/逐层压缩比、worker 使用实际 indexer/draft spec、旧 compressor tail、普通 SWA、State、CP 与 PP 空 shard 路由。

## 外存命中时的 Mamba align 分配核对（2026-10-06）

对照本地 vLLM 0.30.0 `ced6857a` 的 `KVCacheManager.allocate_slots`、coordinator 两阶段 computed-block 分配与 `MambaManager`，以及 Ascend main `f6992d8c` 的 manager/worker 补丁。没有运行模型或组件测试。

- 普通同步加载、无 speculative/partial-hit/internal-checkpoint 时，设恢复边界 N、本轮计算 S、State block span B。`add_local_computed_blocks` 使用 Mamba 的 `get_num_skipped_tokens(N)=N-1`，在 table 的 `[0, (N-1)//B)` 位置放 null；`allocate_external_computed_blocks` 只申请一个真实边界 State block。
- `MambaManager.allocate_new_blocks` 再用 null 补到 `(N+S-1)//B`，只在运行槽申请一个真实 block。因此跨多个 B 的 prefill chunk 也通常只有“边界 + 运行”两个真实 State block，而不是每个逻辑槽都申请内存。例 N=256/S=256/B=128：table 为 `[null, A, null, B]`。checkpoint/CoW 等额外块按原生路径处理。
- Ascend 的 `AscendMambaManager.get_num_blocks_to_allocate` 只为同步外存命中的额外边界块补容量计数；实际上述分配仍继承上游。基本的 sparse-table/external-boundary 规则在本地 v0.29.0 中也已存在。
- CUDA preprocess 在 connector load 前执行 A→B；Ascend MRV1 preprocess 只准备 copy 信息，connector load 完成后才执行 A→B。这一区别在普通 prefill 中就存在，无需引入投机解码。
- 此次核对发现旧的 CUDA v2 load 只填 B，留下 A 未加载。但原生 `MambaManager.cache_blocks` 会给保留的 A 注册 hash，使后续请求可以本地命中它。只填运行槽不足以保证本地缓存内容正确。已增加同一 State key 的边界加载计划：CUDA 填 A 和 B；Ascend 只填 A，由原生 copy 填 B。两份 CUDA load 共用同一存储记录，不更改 record schema；完整恢复边界和正的 scheduled span 保证它们落在不同槽位。
- `boundary_state_offloads` 是引擎交出的候选集合，会包含新注册 hash 的导入边界，并不表示 UCM 必须把它全部写回。已在 `_state_dump_plans` 排除 `boundary <= hbm_hit_tokens + external_hit_tokens` 的初始已命中前缀，只保存其后的新边界。不用持续推进的 `token_processed` 作下限，以免漏掉下一步才交出的新 checkpoint；仍要求完整 unit 与全部 State group 同边界。

后续验收必须同时检查外存导入边界块与运行块、再次本地命中、导入边界不重复 dump、新边界延迟交付仍会保存、跨多个 block 的 chunk 和 null gaps；不能只验证本轮 forward 的运行 State。

### 与旧 HLA 的版本关系

- A（外存恢复边界槽）参与本地 hash 注册不是 0.30 新行为。本地 v0.19.0 已先设置 `num_cached_block`，再申请外存边界块，随后 Mamba `cache_blocks` 调用基类缓存新增非 null 块；v0.26.0、v0.29.0 也保留这条路径。这里以启用引擎本地 prefix caching 的普通同步恢复为前提；关闭本地 caching 时不能直接套用本地再次命中 A 的结论。
- 精确 `boundary_state_offloads` 交接由 `6b110badbb`（Save exact Mamba boundary states）引入，本地版本 tag 中 v0.29.0 已包含。这个较新的保存 SPI 与更早存在的本地 hash 注册是两件事。
- 旧 `hla_connector._append_mamba_align_state_block` 的 load 分支选择最后一个非 null 槽（无 speculative 的普通恢复即 B），并不额外填 A。仅检查本轮运行 State 无法验证外存导入边界的本地缓存内容；尚未运行历史环境复现，不把源码推导当作历史运行失败证据。
- 旧 UCM 还提供 `v0210/vllm_ascend/mamba_copy_order_patch.py`，把 Ascend A→B 复制提前到 connector load 之前，以适配 HLA 加载 B。这是旧 HLA 能在 Ascend 使用 B 的另一项前提。当前 v2 的 NPU 加载 A 策略依据原生 MRV1 顺序；若启用 `ENABLE_UCM_PATCH`，`apply_all_patches` 仍会导入这个复制顺序补丁，两条策略会冲突。用户已确认本次依据原生引擎，旧 UCM patch 不纳入适配范围；此处仅保留历史差异说明，不宣称打开旧补丁也能正确恢复。
