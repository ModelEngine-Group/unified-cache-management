# Connector v2 Layout：2026-10-15—16 验证

重点模型为 GLM-5.3-Flash、DeepSeek V4.1-Flash、Qwen-Flash-Next。测试材料均无配套 checkpoint 权重，按用户要求没有执行真实推理。Qwen 使用用户确认的 CPU 模拟 CUDA 布局方式。

## 固定快照

- UCM：`1c806f569603d65e988a6eab460e86a60208d3b2`，已正常推送 `origin/dev_connector`，正式 case 均为干净 checkout。
- NPU vLLM：`ced6857afa0ea7b2e3f0846a62e1394e90f15607`（0.30.0）；vLLM-Ascend：`5f828ad638ee4f9b12db87f9d5481f29076d37f9`，均干净。
- NPU：Ascend910B3，torch2.10.0、torch-npu2.10.0.post4，重点 case 使用设备1。
- CPU：官方 vLLM0.30.0+cpu，torch2.13.0+cpu；KV spec、KV config/grouping、scheduler、block pool 四个文件的 SHA256 与本地0.30源码一致。这是局部源码核对，不代表完整 wheel 构建身份。

## 结果

| 模型 | 平台与条件 | 整组 bulk | 逐层 layerwise |
| --- | --- | --- | --- |
| GLM-5.3-Flash | Ascend NPU | PASS / 112段 | PASS / 112段 |
| DSV4.1-Flash | Ascend NPU | PASS / 124段 | PASS / 124段 |
| Qwen-Flash-Next | CPU 模拟 CUDA；retention=1664 | PASS / 12段 | PASS / 170段 |

数量为逐字节比较的传输段数；bulk 单段可以包含多层和 padding，不能按数量比较覆盖强弱。

74项 v2 组件测试、8项 model-check 工具测试在同一快照通过。组件覆盖独立元素地址 oracle、六种物理轴序、partial token、padding、CP、State byte-page、精确 hand-off、文件往返及异步任务等待。

补充 GQA bulk/layerwise、UCM chunk256/native block128、GQA TP2/PP2 均在0ad1ec0a通过；DSV4 bulk/layerwise在6aaf41b2各比较2912段通过。它们是各自SHA的结果，不能替代完整并行模型矩阵。

## 实际验证内容

使用原生模型构造、spec/group/config、ModelRunner allocator、Scheduler/KVCacheManager 和 block IDs。执行 UCM 文件写入/commit、外部 lookup 和读回；源块仍占用，只通过原生 eviction 移除本地 prefix hashes，目标非 null block IDs 必须与源不同。加载前对目标范围写0xff，再逐字节比较全部加载范围。模式数据只包含0—250，因此可以发现未执行的加载。

当前使用开发用 `SimpleFileUCMProxy`，配置里的 `Cache|Posix` 不能作为生产 Cache|Posix 后端验收。模型验证没有替换 KV spec、group、shape/dtype、allocation、scheduler/block IDs 或 lookup 结果。

## Qwen 条件与限制

材料目录 `qwen38-next` 实际为 `Qwen4ExpForConditionalGeneration` / `qwen4_exp`，不是另一个带权重的旧 Qwen3.8。原生 scheduler block/chunk 为1664 tokens；六个 group 包含 GDN、PLE State、raw-key ring 和 FA/indexer。ring 按原生不可持久化标志排除。

默认 `prefix_cache_retention_interval=0` 时，本次单轮、无 forward 的合成请求没有交出可保存的 State，只有FA写入，整体 lookup 未命中；该运行不计PASS。正式 Layout/IO case 显式设置真实 `CacheConfig.prefix_cache_retention_interval=1664`，由原生 scheduler 生成周期 State offloads，未补造快照或改变 connector 默认配置。两种模式检查了 CUDA State 的导入边界槽和运行槽。

CPU 工具只在 meta 构造期间跳过 QSA FlashAttention 可用性探针，保留原生 owner/backend/spec/cache 构造。QSA backend 仍声明 `supports_kv_connector=False`，因此这里的 Layout/IO PASS 不证明完整服务入口已支持 connector 推理。

## 测试期间修正

核心 Layout 重构在fd43490d；之后没有再次修改生产 Layout。运行中修正了两处组件 fixture，以及工具与0.30接口的接线：删除失效 Ascend manager factory patch、传递 tokens_per_chunk、外部命中/poison验证、storage_block_size=None日志、dummy无权重入口、原生Ascend custom-op注册、Engram真实CPU控制collective，以及EngineCore resolved-layout hand-off。

DSV4.1 的 Engram node-locality 检查使用真实 CPU/Gloo collective；仅在 meta 模型构造时允许其控制 tensor 留在CPU，不替代判断结果。CPU环境的坏 torchcodec 可选音频包在测试期间临时隔离，结束后恢复；缺少的wrapt使用同机实际1.17.2包。

## 证据与复现

远端根目录：`/home/qyh/ucm_v2_review_20261015_fd43490d`；源码为`src`，证据为`results`，按case隔离的文件缓存为`store`。每个正式case有command.json（完整命令/环境/SHA/干净状态）、log、exit及native-config.json，namespace清单为namespaces-focus.json。

本地完整证据在工作区`connector_v2_validation_20261015/`，入口REPORT.md，包含材料/扩展哈希和早期失败定位记录。正式case前缀：

- `focus-cpu-qwen38-next-{bulk,layerwise}`
- `focus-npu-glm53-{bulk,layerwise}`
- `focus-npu-dsv41-flash-{bulk,layerwise}`

源码固定到上述已推送SHA并保持干净后，远端主机执行`python3 /home/qyh/ucm_v2_review_20261015_fd43490d/run_focus_matrix.py`。复现前需启动CPU容器并临时隔离上述不可加载的可选torchcodec，运行后恢复；wrapt1.17.2已保留。复现应使用新的case名称/存储目录保留原证据。

本轮未覆盖真实CUDA硬件、forward数值、State copy后计算与再次命中、抢占/取消/CoW多轮生命周期、生产proxy和性能。State layerwise加载/计算流水仍待实现；逐层IO通过不代表流水已实现。
