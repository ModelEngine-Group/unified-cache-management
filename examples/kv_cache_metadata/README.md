# KV Cache 元数据导出与 Dump 文件检查

本功能在 DeepSeek V4、V4.1 使用的 `UCMFAWAConnector` 初始化 Store 后，导出实际注册的 KV cache 元数据。离线检查工具按 JSON 里的字节范围读取已完成的 Dump 文件，统计 NaN、Inf，并定位到 group、层名、组件和文件偏移。

实现基于主线 `develop` 的 FAWA connector。服务端需要加载本分支的 UCM Python 代码。

## 随服务启动导出

在启动服务的环境中设置输出目录，然后执行原有启动命令。

```bash
export UCM_KV_CACHE_METADATA_DIR=/tmp/kv-meta
```

容器或分布式启动器需要把这个环境变量传入每个 worker。输出路径属于 worker 所在的文件系统，容器部署可以把该目录挂载到宿主机。

每个 worker 为已初始化的 FA、WA Store 分别生成 JSON，文件名形如 `kv-cache-主机名-PID-FA.json`、`kv-cache-主机名-PID-WA.json`。服务日志会输出实际文件路径。启用 bounded replay、仅持久化 FA 的配置可能只生成 FA JSON。

JSON 通过临时文件和原子替换发布。导出失败时服务日志记录 warning，推理服务继续启动。未设置环境变量时，导出关闭。

## JSON 记录的内容

| 字段 | 含义 |
| --- | --- |
| `model`、`model_dtype` | 运行时模型名称和 dtype |
| `cache_dtype`、`indexer_kv_dtype` | 运行时 KV、indexer 配置 |
| `packages` | 本地可查询到的 UCM、vLLM、vLLM-Ascend、PyTorch 包版本 |
| `runtime` | 主机、PID、engine ID、rank、DP rank 及 TP/PCP/DCP 大小 |
| `byte_order` | 生成文件的平台字节序 |
| `store` | FA/WA、pipeline、存储目录、记录大小、payload 大小及尾部 padding |
| `groups[].specs` | 原生逐层 KV spec 的 dtype、block、scale、量化等字段 |
| `groups[].registered_views` | 注册视图的 dtype、shape、字节 stride、storage offset 和观测地址 |
| `segments` | 与 Store `tensor_size_list` 一致的写盘顺序、每段文件偏移和长度 |
| `segments[].regions` | 可以按 dtype 检查的子范围，附层名和组件编号 |
| `opaque_bytes`、`padding_bytes` | 未解码字节数与视图之外的 padding 字节数 |
| `excluded_group_ids` | 本次布局中跳过持久化的原生 group ID |

`stride_bytes`、`file_offset_bytes`、`size_bytes` 等字段均以字节计。`regions[].offset_bytes` 相对于所属 segment，实际文件偏移为两者之和。观测地址用于追溯启动时的布局，离线检查直接使用文件偏移。

V4 可同时包含 BF16 Attention KV、INT8 indexer、FP16 scale、FP32 state。V4.1 可将 MLA、indexer 和 scale 放在同一物理页，传输时只保留一个页 owner。本工具保留同一 group 的原始 typed aliases，从页内定位浮点组件。实际格式以当前服务导出的 JSON 为准。

## 检查 Dump 文件

选择与被检查文件对应的运行、worker 和 Store JSON。FA 文件使用 FA JSON，WA 文件使用 WA JSON。模型、平台、量化参数和并行配置变更后，应使用新服务导出的元数据。

```bash
UCM_REPO=/path/to/unified-cache-management
KV_META_JSON=/tmp/kv-meta/kv-cache-hostname-12345-FA.json
KV_DUMP_FILE=/cache/fawa_fa/data/实际文件名

python "$UCM_REPO/ucm/integration/vllm/kv_cache_metadata.py" \
  --metadata "$KV_META_JSON" \
  "$KV_DUMP_FILE"
```

命令最后可以传入多个文件或目录。目录会由脚本自动递归扫描，覆盖所有层级的子目录，并跳过 `.tmp` 文件。目录符号链接不参与递归。应指定存放同一种 FA 或 WA 原始记录的数据目录，目录中的配置文件等其他普通文件也会被检查。

```bash
python "$UCM_REPO/ucm/integration/vllm/kv_cache_metadata.py" \
  --metadata "$KV_META_JSON" \
  /cache/fawa_fa/data
```

同一次调用可以混合传入文件和多个目录，全部输入使用同一份元数据。

```bash
python "$UCM_REPO/ucm/integration/vllm/kv_cache_metadata.py" \
  --metadata "$KV_META_JSON" \
  "$KV_DUMP_FILE" /cache/fawa_fa/data /cache/another_fa_data
```

逐文件输出仅包含异常结果，每个异常文件输出一行 JSON，包括 NaN、Inf、大小不符、解码覆盖不完整和读取错误。正常文件保持安静。输入包含目录时，最后仍输出一行 `type: summary` 的汇总，包含扫描文件数、跳过的 `.tmp` 数量、各状态计数、NaN/Inf 总数、未解码字节总数和目录访问错误数。只传入正常文件时，标准输出为空，退出码为 0。遇到异常文件后继续检查后续文件，目录访问失败会记录为错误。

检查端只使用 Python 标准库，逐块读取文件。应选择 Dump 已完成、内容稳定的正式文件。单独传入 `.tmp` 文件会返回错误。

当前支持对 `Cache|Posix`、`YuanRong|Posix` 的未压缩原始记录进行离线检查。如果旧配置中的标量 `tensor_size` 覆盖了真实分段尺寸，元数据会标记 `tensor_size_overrides_layout`，检查工具返回错误。

## 结果解读

| `status` | 含义 |
| --- | --- |
| `no_nan_in_typed_regions` | 已描述范围中没有发现 NaN 或 Inf，且没有未解码字节 |
| `nan_found` | 发现 NaN |
| `inf_found` | 发现 Inf，且本次未发现 NaN |
| `invalid_size` | 实际文件大小与元数据不符 |
| `incomplete` | 已描述浮点范围没有发现 NaN/Inf，同时存在尚未解码的字节 |
| `error` | 文件读取、元数据校验、pipeline 或检查期间文件变化等错误 |

每个结果包含 `nan_count`、`inf_count`、`checked_float_bytes`、`integer_bytes`、`opaque_bytes`、`padding_bytes`。`issues` 列出异常组件的 `group_id`、`tail_block`、`name`、`component`、`dtype`，以及首个 NaN/Inf 的文件字节偏移。

多个状态条件同时出现时，NaN 的状态优先于 Inf，Inf 优先于未解码状态。即使状态为 `nan_found` 或 `inf_found`，也应查看 `opaque_bytes` 了解覆盖范围。INT8 等整数范围计入 `integer_bytes`，NaN 检查跳过这些范围。

至少检查了一个文件、所有文件均返回 `no_nan_in_typed_regions` 且目录遍历没有错误时，进程退出码为 0。存在异常、目录访问错误或实际检查文件数为零时，退出码为 1。

FP8、TurboQuant 等打包页可能包含多种编码。工具对尚未建立解码规则的范围记录 `opaque_bytes`，这类文件可以继续根据 JSON 分析物理布局。检查只能确认记录大小、已描述范围内的浮点编码和本次解码覆盖情况。有限 KV 数值是否正确，需要结合 Dump 前的源 KV 或可信回读结果比较。Inf 编码是否符合模型语义，也需要结合具体组件判断。

## 验证记录

在本地 CPU 环境使用真实 PyTorch tensor views 和合成文件验证了以下场景。

- V4 的 BF16、INT8、FP16 混合分段及异常偏移。
- V4.1 共页 owner、indexer scale、storage offset 与 padding。
- K/scale/full aliases、WA 部分尾窗口及多 block 展开。
- 启动导出开关、原子 JSON 写入、命令行及退出码。
- BF16、FP16、FP32、FP64 和五种 FP8 编码在读取分块边界处的 NaN/Inf 检查。
- 未解码字节、文件大小错误及元数据范围校验。

专项测试共 26 项通过，包含文件与多个目录混合输入、多层目录递归、临时文件过滤、目录符号链接、异常后继续扫描、空目录及目录访问错误。可在仓库根目录执行。

```bash
python -m pytest --confcutdir=test/suites/Unit \
  test/suites/Unit/test_kv_cache_metadata.py -q
```

相关回归共 77 项通过，另有 44 个 subtests 通过。1 项依赖完整 vLLM 环境而跳过。既有 `test_fa_only_prefix_can_be_found_and_saved_without_a_window_store` 的三个子用例因 fixture 缺少 `dump_owner_size` 失败，已在基线 `d2a7efa550d72a0067f01b729f2faada3f74cffe` 复现，回归通过计数排除了该用例。

验证范围为 CPU tensor 布局、合成文件读回及启动导出调用。真实 V4/V4.1 服务启动、设备执行和生产 Store 生成的文件仍待运行验证。

## 源码入口

- [元数据导出与离线检查](../../ucm/integration/vllm/kv_cache_metadata.py)
- [FAWA 初始化接入](../../ucm/integration/vllm/hma_connector.py)
- [专项测试](../../test/suites/Unit/test_kv_cache_metadata.py)
