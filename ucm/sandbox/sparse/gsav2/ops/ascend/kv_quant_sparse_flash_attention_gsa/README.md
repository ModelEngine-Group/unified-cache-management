# KvQuantSparseFlashAttentionGsa — GSA 定制改进版算子

## 概述

本算子是 `kv_quant_sparse_flash_attention`（官方 v1）的 GSA 定制改进版，针对 **sparse_indices 从 2048 扩展到 5120（及以上）** 场景的性能劣化进行优化，同时支持 A5 全类型（含 hifloat8）sparse_indices > 2048。

参考官方 v2（`kv_quant_sparse_flash_attention_v2`）的"复用 v1 tiling 和 kernel"策略，在此基础上叠加 GSA 专属优化。

## 性能问题根因

| 场景 | sparse_indices | SFA 单次耗时 | MAC | MTE3 |
|------|---------------|-------------|-----|------|
| 基线 | 2048 有效 | 94.97 us | 6.42 us (9~15%) | 5.26 us (12%) |
| 劣化 | 5120 (全 padding-1) | 429.46 us | 16.14 us | 122.82 us (71%) |

**根因**：生产路径 `V_TEMPLATE` 在 `GetSparseActualSeqLen` 中直接用 `sparseBlockCount × sparseBlockSize` 当有效 KV 长度，不识别 `-1` padding。2048→5120 导致 s2 迭代数 4→10（2.5×），引发流水线背压级联，MTE3 超线性放大 23×。

## 已落地优化

### P1: 有效 topk 识别（根治 padding-1 场景）

**改动文件**：
- `op_host/kv_quant_sparse_flash_attention_tiling.h`: BaseParams 新增 `effectiveSparseCount` 字段
- `op_host/kv_quant_sparse_flash_attention_tiling.cpp`: `ScanEffectiveSparseCount()` host 侧扫描 sparse_indices 统计非 -1 个数
- `op_kernel/kv_quant_sparse_flash_attention_kernel_mla.h`: V_TEMPLATE 路径 `GetSparseActualSeqLen` 用 `effectiveSparseCount` 替代 `sparseBlockCount`

**原理**：host tiling 阶段遍历 sparse_indices tensor，找到每行最后一个非 -1 的位置，取所有行的最小值作为 `effectiveSparseCount`。kernel 侧 V_TEMPLATE 路径用该值计算 `curActualSeqLen`，避免 padding 项引入的无用迭代。

**收益**：5120-padding-1 场景直接回到 2048 性能（~95us）。

**降级机制**：动态 shape 下 host 无法访问 tensor 数据时，`effectiveSparseCount=0`，kernel 退化为 v1 行为（用 `sparseBlockCount`）。

### P2: 反量化直通（省 FP32 中转，缓解 Cube 空等）

**改动文件**：
- `op_kernel/kv_quant_sparse_flash_attention_service_vector_mla.h`: `CopyOutMrgeResult` 反量化循环

**原理**：原路径 `INT8→FP16→FP32→Mul(FP32)→BF16` 有 5 次 PipeBarrier。改为 `scale FP32→K_ROPE_T` 后直接在 K_ROPE_T（FP16/BF16）域做 Mul：
- K_ROPE_T=half: `INT8→half→Mul(half)` — 省 2 次 Cast + 2 次 PipeBarrier
- K_ROPE_T=bfloat16_t: `INT8→half→BF16→Mul(BF16)` — 省 FP32 中转的 2 次 Cast

**收益**：V0 阶段压缩 ~30-40%，直接提升 Cube 利用率（MAC 9%→目标 30%+）。

**精度说明**：MLA 场景 dequant_scale 为 per-tile(128) FP32，值范围有限，FP16/BF16 域 Mul 精度一般足够。若 GSA 验证精度不达标，可回退为 FP32 Mul（保留 P2 的 barrier 优化）。

### P4: preLoadNum 2→3（加深流水掩盖 MTE3 stall）

**改动文件**：
- `op_kernel/kv_quant_sparse_flash_attention_kernel_mla.h`: `PRELOAD_NUM = 3`

**原理**：3 级 Preload 流水让 MM1/MM2/Vec2 重叠更充分，前一轮 MTE3 stall 被后两轮计算掩盖，平滑 MTE3 利用率波动（12%~71% → 更平稳）。

**约束**：workspace 总量 ×1.5（mm1ResGm/vec1ResGm/mm2ResGm/vec2ResGm 需 3 buffer）。`kvMergeGm_` 已有 4 buffer（`MERGE_CACHE_GM_BUF_NUM=4`），够用。

### A5 全类型 sparse_indices > 2048 放宽

**改动文件**：
- `op_host/kv_quant_sparse_flash_attention_tiling.cpp`: 删除 hifloat8 的 `sparseBlockCount != 2048` 检查，改为 `sparseBlockCount <= 8192` 上限检查
- `op_host/kv_quant_sparse_flash_attention_gsa_def.cpp`: A5 配置支持 INT8 + float8_e4m3 + hifloat8

**原理**：v1 对 A5 hifloat8 强制 sparse_size=2048（因 `AntiquantVFImplFp8D448` SIMD 反量化路径处理单行数据，与 sparseBlockCount 无关）。实际 kernel 侧无 2048 硬编码，限制仅在 tiling 检查层。放宽到 8192。

**hifloat8 反量化路径**：`arch35/kv_quant_sparse_flash_attention_service_vector_mla.h` 的 `AntiquantVFImplFp8D448` 按 `nopeDim=512, kvNumPerLoop=128` 处理，与 sparseBlockCount 无关。sparseBlockCount > 2048 时仅增加外层 s2 迭代次数，反量化单行逻辑不变。

### v2 softmax_lse 输出

**改动文件**：
- `op_host/kv_quant_sparse_flash_attention_gsa_def.cpp`: 新增 `softmax_max`/`softmax_sum` 可选输出 + `return_softmax_lse` 属性
- `op_host/kv_quant_sparse_flash_attention_gsa_proto.cpp`: InferShape/InferDataType 含新输出
- `op_kernel/kv_quant_sparse_flash_attention_common.h`: ConstInfo 新增 `returnSoftmaxLse`
- `torch_binding.cpp`/`torch_binding_meta.cpp`: Python 接口返回 `(attention_out, softmax_max, softmax_sum)`

## 未落地优化（P0/P3 TODO）

### P0: splitKV 跨核并行 s2（最大收益，待开发）

**目标**：s1 较小时（decode 场景），将单 (b,s1) 的 s2=5120 切分给多个 Cube 核并行，最后做 online-softmax 跨核规约。

**实现要点**：
1. Host 侧 `SplitBalanced`/`CalcInnerSize`：当 `sparseBlockCount × sparseBlockSize / s2BaseSize > 4` 且 `totalBaseNum < coreNum` 时启用 splitKV
2. Kernel 侧 `ProcessBalance`：每个 (b,s1) 由 `splitKVPart` 个核协作，各算 s2 一段
3. Vec2 末尾增加跨核 softmax 规约：`AtomicMax(softmaxMax)` + `AtomicAdd(softmaxSum)`，再修正部分输出
4. 复用已有 `splitKVParams.accumOutSize`/`logSumExpSize` workspace

**TODO 代码位置**：
- `op_host/kv_quant_sparse_flash_attention_tiling.cpp` `SplitBalanced()` (line ~328)
- `op_kernel/kv_quant_sparse_flash_attention_kernel_mla.h` `ProcessBalance()` (line ~815)
- `op_kernel/kv_quant_sparse_flash_attention_service_vector_mla.h` `DealBmm2ResBaseBlock()` (line ~982)

### P3: Vec2 中间结果不写 GM（s1 小时可行）

**目标**：s1≤16 时，Vec2 中间块在 UB 内 pingpong 累加，末块才 MTE3 写 GM，MTE3 从 10 次降到 1 次。

**约束**：需 `cubeMSize × headDim × 4B × 2 ≤ UB 余量`。s1=1 时 4.6KB 可行，s1=16 时 72KB 需先释放 UB。

## 构建与使用

### 统一构建流程

所有算子（内置 aclnn 轻量算子 + 自定义 AscendC 算子）统一通过 `gsa_ops` whl 调用。

```
ops/                              # 统一入口
├── build.sh                      # 主构建脚本（4 步流程）
├── setup_wheel.py                # NpuExtension 编译 torch_binding → gsa_ops.whl
├── torch_binding.cpp             # 所有算子注册（pybind11 + TORCH_LIBRARY）
├── torch_binding_meta.cpp        # 自定义算子 Meta 实现（aclgraph 支持）
├── aclnn_torch_adapter/          # NPU 适配层（唯一一份）
└── ascend/                       # AscendC kernel 编译基建
    ├── CMakeLists.txt / build_ops.sh / install.sh
    ├── cmake/ / utils/
    └── kv_quant_sparse_flash_attention_gsa/  # 算子本体
```

**构建流程（build.sh 自动执行 4 步）**：
1. `ascend/build_ops.sh` 编译 AscendC kernel → `GSA-custom_ops*.run`
2. `ascend/install.sh` 安装 .run 到 `vendors/gsa`（提供 aclnn 接口 + libcust_opapi.so）
3. `setup_wheel.py` 编译 torch_binding.cpp → `gsa_ops.whl`（EXEC_NPU_CMD 用 dlsym 动态查找，无需编译时 aclnn 头文件）
4. `pip install gsa_ops.whl`

### 编译

```bash
cd /workspace/unified-cache-management-public/ucm/sandbox/sparse/gsav2/ops

# A3 (默认)，编译并安装
bash build.sh a3 install

# A2
bash build.sh a2 install

# A5
bash build.sh a5 install

# 仅编译不安装
bash build.sh a3

# 仅编译安装 whl（假设 .run 已安装）
bash build.sh whl install
```

### Python 调用

```python
import gsa_custom_ops
import torch_npu

# 调用改进版算子
attention_out, softmax_max, softmax_sum = torch.ops._C_gsa.npu_kv_quant_sparse_flash_attention_gsa(
    query, key, value, sparse_indices,
    scale_value, key_quant_mode, value_quant_mode,
    key_dequant_scale=None, value_dequant_scale=None,
    block_table=block_table,
    actual_seq_lengths_query=actual_seq_q,
    actual_seq_lengths_kv=actual_seq_kv,
    sinks=None,
    sparse_block_size=1,
    layout_query="BSND", layout_kv="BSND",
    sparse_mode=3, pre_tokens=INT64_MAX, next_tokens=INT64_MAX,
    attention_mode=0, quant_scale_repo_mode=1,
    tile_size=128, rope_head_dim=64,
    return_softmax_lse=False
)
```

## 算子接口

### 输入

| 参数 | 类型 | 说明 |
|------|------|------|
| query | FP16/BF16 | Q 输入, D=576 (512 nope + 64 rope) |
| key | INT8/float8_e4m3/hifloat8 | K 输入, D=656 (512+64×2+4×4) |
| value | 同 key | V 输入 |
| sparse_indices | INT32 | 稀疏索引, sparse_size ≤ 8192, padding 用 -1 |
| sinks (可选) | FLOAT | 学习型 sinks |

### 输出

| 参数 | 类型 | 说明 |
|------|------|------|
| attention_out | FP16/BF16 | D=512 (减去 rope) |
| softmax_max (可选) | FLOAT | return_softmax_lse=true 时有效 |
| softmax_sum (可选) | FLOAT | return_softmax_lse=true 时有效 |

### 属性

与 v1 一致，新增 `return_softmax_lse` (Bool, 默认 false)。

## 架构支持

| 架构 | SoC | KV 类型 | sparse_size 上限 |
|------|-----|---------|-----------------|
| A2 | ascend910b | INT8 | 8192 |
| A3 | ascend910_93 | INT8 | 8192 |
| A5 | ascend950 | INT8/float8_e4m3/hifloat8 | 8192 |

## 与官方 v1/v2 的差异

| 维度 | v1 | v2 (官方) | GSA (本算子) |
|------|----|----------|-------------| 
| A2/A3 支持 | ✓ | × | ✓ |
| A5 支持 | ✓ (hifloat8限2048) | ✓ (hifloat8限2048) | ✓ (全类型≤8192) |
| sparse_indices > 2048 | × (hifloat8) | × (hifloat8) | ✓ (全类型) |
| P1 有效topk识别 | × | × | ✓ |
| P2 反量化直通 | × | × | ✓ |
| P4 preLoadNum=3 | × (2) | × (2) | ✓ (3) |
| softmax_lse 输出 | × | ✓ | ✓ |
| sinks 输入 | × | ✓ | ✓ |
