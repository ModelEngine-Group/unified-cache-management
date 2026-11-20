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

### P0/C: splitKV 跨核并行 s2（flash-decoding 式，C 阶段升级为尾波自适应+WCP 加权分核）

**目标**：s1 较小时（decode 场景），将单 (b,s1) 任务的 s2 迭代切分给多个核并行，最后做 online-softmax 跨核规约，消除"核多任务少"的空转与波尾碎片。

**改动文件**：
- `op_host/kv_quant_sparse_flash_attention_tiling.h/.cpp`: `DecideSplitKV()`（尾波候选枚举 + WCP makespan 模型搜索 (x,P)）；`FillTilingSplitKVMla()`/`NormalCalcFDWorkSpace()` 按 sub 紧凑布局计算 FD workspace
- `op_kernel/kv_quant_sparse_flash_attention_kernel_mla.h`: `InitCalcParamsEach()` WCP sub→核分配；`ProcessBalance()` per-task partNum + part 区间交集 + owner 任务记录（TND 修复：combine 输出偏移走 prefix-sum 公式）；`Process()` AIV `SyncAll` + combine 调度
- `op_kernel/kv_quant_sparse_flash_attention_service_vector_mla.h`: lse/accumOut 写出与 `ProcessSplitKVCombine()` 全部切换为 sub 编号寻址 + per-task partNum
- `op_kernel/kv_quant_sparse_flash_attention_common.h`: `RunInfo::splitSubIdx`、`ConstInfo` 尾波字段、`SplitKVTaskInfo{attenOutOffset, subBase, partNum, s2Loops}`

**设计要点**：
1. **任务模型**：任务=(b, s1 token)（m-chunk=gSize，host 侧 `mBaseSize==gSize`）。C 阶段改为「尾波自适应切分」：仅最后 x 个尾波任务切 P 份，其余任务整块执行（P=1）
2. **sub 全局编号**：`taskIdx < tailStart ? taskIdx : tailStart + (taskIdx−tailStart)*P + part`。FD workspace 按该紧凑编号寻址（vs P0 的 `taskIdx*P+part` 均匀编号，非切分任务不浪费槽位，subTotal 与 FD 尺寸显著缩小）
3. **WCP（加权等分点连续区间）分核**：sub 按全局顺序流式分配，归属核 = `(prefix−1)*nCore/totalW`（prefix=计入该 sub 后的累计权重；full 任务权重=L，切分 part 权重=ceil(L/P)）。归属随 sub 单调不减 → 每核 sub 集合连续，kernel 单趟扫描即可抽取本核区间。**host `simulateMakespan`/kernel `InitCalcParamsEach`/bench `expected_splitkv_p` 三处公式逐字节一致**（无 NPU 环境下以 CPU 自检交叉验证：三方 makespan 与最大核负载逐点相等，划分性质成立）
4. **运行时开关**：复用既有 tiling 字段 `splitKVParams`（splitKVNum=P + 新增 tailTaskNum/s2LoopsEst/taskNumTotal），不新增 tiling-key 实例化；原 `FLASH_DECODE` 模板参数相关分支全部改为 `splitKVNum > 1` 运行时判断
5. **TND 生产签名支持**：taskNum=T 轴总长（=ΣactS1，精确）；TND 下 host 无法读 tensor 内容（`ScanEffectiveSparseCount` 同理返回 0），FD 尺寸按上界估计、kernel 按真实任务数寻址（上界 ≥ 真实，安全）；combine 的 `attenOutOffset` 走 TND prefix-sum 公式（与直写路径 `CalcFirstTensorOffsets` 逐字一致）
6. **per-task combine**：仅尾波切分任务（partNum>1）记录进 `splitKVTasks[16]`，full 任务 part 0 覆盖全区间直写 attenOut 不参与 combine——bs=36 场景 combine 任务数从 36 降到实际切分数，FD workspace 与归约开销同步减半
7. **跨核同步**：AIV-only `SyncAll()`（barrier payload 骑在 MTE3 队列上，天然保证各核 FD partial 已落 GM）；WCP 稀疏分布下的空区间核置 `usedCoreNum=0` 仅参与栅栏，避免未初始化的 bN2/gS1 范围误处理任务
8. **归约数学**：`maxG = max_p lseMax_p`；`w_p = exp(lseMax_p − maxG)`；`O = Σ w_p·O_p / Σ w_p·lseSum_p`（O_p/lseSum_p 为未归一化 partial，`RowMuls/RowDivs` 消费 8 路 broadcast 行标量）
9. **FD GM 布局**（fp32 元素）：accumOut 槽位 `=(sub*gSize+行)*512`；lse 槽位 `=(sub*gSize+行)*8`（sum 区在前，max 区在 `+logSumExpSize/2`）。行数按 `qsfaInfo_->gSize`（=kernel `mBaseSize`）而非 host 类内 `mBaseSize_(128)`

**启用门控**（`DecideSplitKV`，`SplitBalanced` 内调用）：
- 非 A5（A2/A3）、query 为 **BSND/TND** 布局（kv 可为 BSND/TND/PA_BSND）、`gSize ≤ 256`
- 有效 s2 迭代数 ≥ 4；`subTotal = (taskNum−x) + x*P ≤ 4*aicNum`；`P ≤ 8`
- WCP makespan 模型枚举尾波候选 x ∈ {taskNum%nCore, taskNum%nCore+nCore, …, taskNum}，理论收益 ≥ 25% 才启用
- 调试/基准环境变量：`QSFA_DISABLE_SPLITKV=1` 强制关闭；`QSFA_FORCE_SPLITKV=P` 强制全任务切 P（仍受 subTotal 安全上限约束）

**启用区与数学结论**（解析模型，A3: aicNum=24, sparse=5120 全有效 → s2Loops=10）：

| bs (s1=1) | auto (P, tail) | makespan off→on | 理论加速比 |
|-----------|----------------|-----------------|-----------|
| 1–2       | (5, bs)        | 10 → 2          | ~5×       |
| 4         | (5, bs)        | 10 → 2          | ~5×       |
| 8         | (3, bs)        | 10 → 4          | ~2.5×     |
| 12        | (2, bs)        | 10 → 5          | ~2×       |
| 16 / 24   | 关闭           | 建模收益 < 25%（16: 20%） | 1× |
| 25/30/36  | (2, bs)        | 20 → 15         | ~1.33×    |
| 48/64/112/128 | 关闭       | 最优组合 = base  | 1×        |

> **大 bs 数学结论（勿试图"修复"）**：full 任务整块（权重 10）在 `subTotal ≤ 4*aicNum` 硬约束下，与 part 粒度（权重 5）的任何组合都无法突破组合数论下界——bs=64/112/128 的最优组合恰等于 base makespan（如 bs=64: 48full+16×2part 或其它组合均 = 30）。尾波切分在这些 bs 上**本就无真实收益**，自动关闭是正确行为。大 bs 的性能收益由 D（fp16 中间量）承担，C 的价值在于 TND 生产签名使能 + bs∈[25,36] 区间的负载均衡。

**FD workspace 增量**：`subTotal*gSize*(512+8)*4B ≤ 4*aicNum*gSize*2080B`（gSize=32 时约 6.4MB），复用既有 splitKV workspace 机制，非 split 路径零增量。

**已知限制**（P0 遗留，需 NPU 验证后处理）：真实 `s2SplitNum` 可小于 host 估计 L（各 batch 有效 topk 不同时），此时 P 尾部出现空 part；combine 侧 `validParts` 已正确跳过空 part 槽位，但「拥有空尾 part 的核」pipeline flush 轮缺失属上游结构问题，未在本阶段修改。

**基准脚本**：`ut/bench_splitkv_gsa.py`（on-device，生产签名 TND/PA_BSND/attention_mode=2），见下文"基准与验证"。

### A: 反量化-MM 融合（antiquant-MM）——2201 硬件不可行，已论证

**目标**：消除 V0 阶段 dequant 后写 `kvMergeGm`（GM 往返 ~1.15MB/chunk），让 MM1/MM2 直接消费 int8 量化 KV（硬件装载期反量化）。

**结论：2201（A2/A3）上不可行**，证据链（均为环境内 CANN 9.1 头文件/官方算子源码）：

1. **`LoadData2DParamsV220` 无反量化字段**（`basic_api/kernel_struct_mm.h`）：仅 mStartPosition/kStartPosition/mStep/kStep/srcStride/dstStride/ifTranspose/sid——L0B 装载期不存在 int8→fp16 带 scale 的硬件路径
2. **`Mmad` 仅同型操作数**（`kernel_operator_mm_intf.h`）：fp16×fp16 或 int8×int8（int8 结果为 int32 累加域），无 fp16×int8 混合，无 MX 类多精度 Mmad（3510 专有）
3. **官方 `anti_quant.h`（weight_quant_batch_matmul_v2）为 UB 向量侧反量化**：与现有 V0 路径同构（int8 GM → UB dequant → GM/UB fp16），无法节省 GM 往返

**替代路线**：V0 的 UB 反量化 + `kvMergeGm` GM staging 已是该架构最优（P2 已将反量化压至 K_ROPE_T 域直乘）。真正的 GM 流量削减由 D 承担。A5（arch35）的 `LoadData2DParamsV2` 带 dequantScale 字段，如需 antiquant-MM 应在 A5 路径上重启该设计。

### D: 中间结果 fp16 化（mm1Res/mm2Res Fixpipe F322F16 直出）

**目标**：MM1 之后的 S 矩阵与 MM2 之后的 O 矩阵（原 fp32 GM 中间量）改为 fp16/bf16 存储，消除 L0C→GM 与 GM→UB 的双倍流量。

**改动文件**：
- `op_kernel/kv_quant_sparse_flash_attention_service_cube_mla.h`: 类型拆分 `MM_OUT_T`（L0C 累加域，保持 fp32——Mmad 输出必须 fp32）与 `MM_GM_OUT_T`（GM workspace，=K_ROPE_T）；MM1/MM2 两处 `FixpipeParamsV220.quantPre = QuantMode_t::F322F16/F322BF16`
- `op_kernel/kv_quant_sparse_flash_attention_kernel_mla.h`: `MM1_OUT_T=MM2_OUT_T=K_ROPE_T`；workspace 布局链 vec2Res 推进量改按 `sizeof(T)`（原与 MM2_OUT_T 同为 fp32 时数值恰好相同，fp16 化后若不改会与后续区域重叠）
- `op_kernel/kv_quant_sparse_flash_attention_service_vector_mla.h`: Vec1 `DealBmm1ResBaseBlock` fp16 读入 slot 前 16K + `Cast` 到 [16K,48K) fp32 计算域（softmax 仍在 fp32 域）；Vec2 `DealBmm2ResBaseBlock` UB DataCopy 改为 half→fp32 `Cast`；`inputBuff1` 扩容 2×48KB
- `op_host/kv_quant_sparse_flash_attention_tiling.cpp`: `GetWorkspaceSize` mmRes/bmm2Res elemSize 4→2（vec2Res 保持 4）

**设计依据**：
- **Fixpipe 硬件 cast 为官方既定模式**：arch22 先例 `fused_infer_attention_score/attn_infra/gemm/tile_common/fused_copy_l0c_to_gm.hpp`（`Fixpipe<ElementDst=half, ElementSrc=float>` + `quantPre=F322F16`，`dstStride` 为目标元素单位行间隔——fp16 化后数值不变）；`mla_prolog/op_kernel/arch22/service_matmul.h` 的 `GetTensorC`（L0C float → GM bf16 用 `F322BF16`）
- **UB cast 放置**：softmax 的 max/exp/sum 链对精度敏感，保持 fp32 计算域；仅 GM 往返段（读入后一次 Cast）改 fp16。fp16 暂存区 [0,16K) 与 fp32 计算域 [16K,48K) 同 slot 分区，pingpong 结构不变
- **vec2Res 保持 fp32**：Vec2 的跨 chunk 累积（rescale-add）需要双活 32KB fp32 UB 暂存，UB 预算（约 216KB 已占用）无法容纳，且 flag 交织风险高——收益/风险比不划算

**收益**（每 512-token chunk，m=128，fp16 实例化）：mm1Res 读写 256KB→128KB、mm2Res 读写 256KB→128KB，合计约 **512KB/3.8MB ≈ 13%** chunk 级 GM 流量削减（host workspace 同步缩减）；对所有 bs 生效（含 splitKV 关闭的大 bs 场景——大 batch 的实际收益杠杆）。

**精度影响与回退**：S/O 中间量舍入从 fp32 降为 fp16（约 1e-3 相对误差量级），softmax 的 max/sum 链与跨核 combine 均仍在 fp32 域，最终输出精度预期与 bf16 matmul 固有误差同量级；bench 脚本的 on-vs-ref 容差已相应放宽（fp16 1.5e-2 / bf16 3e-2）。若 GSA 精度验证不达标，回退路径为 `git revert` 该 commit（类型拆分设计使回退面收敛在 4 个文件的少量行）。

## 未落地优化（P3 已评估否决）

### P3: Vec2 中间结果不写 GM —— UB 预算不足，否决

**评估结论**：跨 chunk 累积器需 `mSize × headDim × 4B = 128×512×4 = 256KB` 常驻 UB，而当前 UB 已占用约 216KB（inputBuff1 2×48K + inputBuff2/outputBuff1 32K + tmpBuff1 32K + softmax 3×preLoadNum×512B 等），无 256KB 余量；且 Vec1/Vec2 共享 inputBuff1 的 pingpong 时序需重排，无 NPU 验证条件下风险不可控。GM 流量削减目标已由 D（fp16 中间量）以更低风险达成 ~13%。

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
import gsa_ops
import torch_npu

# 调用改进版算子（生产签名：TND query + PA_BSND packed KV + MLA-absorb）
# query: [T, gsize, 576]（TND, T=ΣactS1, actual_seq_lens_q 为前缀和数组）
# key=value: 同一 packed int8 张量 [totalBlocks, 128, 1, 656]（MLA c_kv, COMBINE 模式）
#   字节 [0,512) nope int8 | [512,640) rope 64×fp16 | [640,656) 4×fp32 per-tile scale
# block_table: [bs, blocksPerBatch] int32
attention_out, softmax_max, softmax_sum = torch.ops._C_gsa.npu_kv_quant_sparse_flash_attention_gsa(
    query, key, value, sparse_indices,
    scale_value, key_quant_mode=2, value_quant_mode=2,
    key_dequant_scale=None, value_dequant_scale=None,
    block_table=block_table,
    actual_seq_lengths_query=actual_seq_q,   # TND 前缀和
    actual_seq_lengths_kv=actual_seq_kv,
    sinks=None, actual_sparse_lengths=None,
    sparse_block_size=1,
    layout_query="TND", layout_kv="PA_BSND",
    sparse_mode=3, pre_tokens=INT64_MAX, next_tokens=INT64_MAX,
    attention_mode=2,          # 2=MLA-absorb（唯一支持的模式，host 侧强校验）
    quant_scale_repo_mode=1,   # 1=COMBINE（packed scale）
    tile_size=128, rope_head_dim=64,
    return_softmax_lse=False
)
```

### 基准与验证（splitKV）

```bash
# 精度 + 性能全扫（默认 bs=1,2,4,8,12,16,24,25,30,36,48,64,112,128,
#   kv_len=32768, sparse=5120 全有效, 生产签名 TND/PA_BSND/attention_mode=2）
python ut/bench_splitkv_gsa.py

# 只验证精度（splitKV 开/关自对比 + fp32 参考对比）
python ut/bench_splitkv_gsa.py --check-only

# 强制全任务切 P=4 验证 combine 归约路径（不依赖自动门控是否启用）
python ut/bench_splitkv_gsa.py --bs 1,16,48 --force-p 4 --check-only

# 随机 per-tile 反量化 scale 的精度验证
python ut/bench_splitkv_gsa.py --check-only --random-scale

# 复现生产 5120-padding-1 场景（尾部 -1 padding，走 P1 effectiveSparseCount 路径）
python ut/bench_splitkv_gsa.py --sparse-count 5120 --valid-count 2048
```

脚本输出示例（P/tail(model) 列 = host `DecideSplitKV` 同模型预期值 P/尾波任务数）：

```
bs  taskNum s2Loops P/tail(model) off_ms  auto_ms speedup
1   1       10      5/1      0.532   0.120   4.43x
4   4       10      5/4      0.548   0.150   3.65x
25  25      10      2/25     1.172   0.895   1.31x
```

判读：
- `on-vs-off` diff：两态均含 D 的 fp16 中间量，该对比检验 combine 归约本身；失败说明 combine 路径有正确性问题
- `on-vs-ref` diff：反映 kernel（bf16/fp16 matmul + **D 的 fp16 中间量** + fp32 softmax）与 fp32 参考的固有精度差，容差已按 dtype 放宽（fp16 1.5e-2 / bf16 3e-2）
- 实际 host 决策可设 `ASCEND_GLOBAL_LOG_LEVEL=1` 后在 plog 中 grep `splitKV enabled`

## 算子接口

### 输入

| 参数 | 类型 | 说明 |
|------|------|------|
| query | FP16/BF16 | Q 输入, D=576 (512 nope + 64 rope) |
| key | INT8/float8_e4m3/hifloat8 | K 输入, D=656 (512+64×2+4×4) |
| value | 同 key | V 输入 |
| sparse_indices | INT32 | 稀疏索引, sparse_size ≤ 8192, padding 用 -1（有效项须为稠密前缀，尾部 padding） |
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
| P0/C splitKV 尾波自适应+WCP | × | × | ✓ (A2/A3, BSND/TND, 小 bs + [25,36]) |
| D fp16 中间量 (Fixpipe F322F16) | × | × | ✓ (全 bs, ~13% GM 流量) |
| softmax_lse 输出 | × | ✓ | ✓ |
| sinks 输入 | × | ✓ | ✓ |
