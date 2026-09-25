# Connector v2 并行适配与验证（2026-09-26）

本次在 layerwise/commit 基线 `338a847a` 上增加同拓扑 TP、PP、CP 分片处理及多进程 model-check。已通过的 CPU/NPU 场景共 20 个；PCP 在现有引擎配置阶段被拒绝，尚无运行时通过证据。

## 实现与边界

- `ParallelLayout` 从原生 parallel_config 读取 TP/PP/PCP/DCP/interleave。worker 数为 `TP × PP × PCP`，DCP 是 TP 子组，不能再次乘进进程数。
- 多 rank 文件路径增加 `tp…-pp…-pcp…-dcp…-i…/rank-N`；手工 `VLLM_PP_LAYER_PARTITION` 也进入 namespace。worker 使用真实 world group rank。同一个逻辑 key 在各 rank 下保存各自字节，不做 TP replicated 数据去重。
- `AllShardLookup` 对每个 key 的所有分片可见性取 AND。dump/load/commit 仍发到本地分片，接口继续使用 keys 和三个 `[K,S]` 数组。某个 rank 未 commit 时整条 key 不命中。替换 Proxy 时也必须提供等价的全分片就绪语义。
- PP 保留引擎全局 group 顺序和 plan.windows 位置；物理寻址只处理本 rank 有层的 group。某种 key 在本 rank 没有任何层时，提交 `[K,0]` 空分片，commit 后才可见。无法从原生 spec 得到空 sliding group 的逐层语义时明确拒绝，不猜测。
- CP 将 group 的逻辑 token block 扩为 `原值 × PCP × DCP`，物理寻址仍针对 rank 本地存储。当前仅支持 FA 完整分布式页；WA/State CP、CP 页内条带切片明确不支持。
- Ascend 的 `sfa_dcp_replicated_indexer_size` 决定本地 indexer 一个逻辑 block 包含多少连续 kernel rows，不能当作普通单行 block。
- 当前支持相同拓扑保存/恢复，不支持跨 TP/PP/CP reshard。只验证单机共享文件根目录；跨节点共享存储、DP/EP 和其他组合未验收。KV dtype 等已有 namespace 边界仍见详细设计。

## model-check 方法

新增 `--tp/--pp/--pcp/--dcp`，均默认 1。多 rank 使用 torchrun 启动真实进程组。各 rank 创建本地 meta model、收集原生 KV specs，经引擎生成 worker KVCacheConfig 和 Scheduler KVCacheConfig，并分配实际缓存 tensor。

各 rank 独立运行原生 Scheduler，再集体断言 request、key、token 范围、block windows 完全一致；这不是一个中央 Scheduler 广播 metadata 的推理执行器。填充值包含 rank，避免 replicated KV 恰好相同而掩盖 rank 文件覆盖。所有 dump/commit 完成后加载到不同 blocks，逐段比较字节，最后要求所有 rank 都输出 PASS。

保留现有 model-check 的限制：没有加载权重或执行 attention forward；填充和比对仍共用 UCM 寻址，不是独立地址 oracle；CPU 模拟 CUDA 布局不能代替真实 CUDA 卡验收。文件 Proxy 同步，本轮没有测试真实异步 IO 与计算重叠。

例如已安装对应引擎和本次 UCM/toolkit 的环境可以运行：

```bash
ucm-toolkit run model-check --model /path/to/model --tp 2 --pp 2 \
  --layerwise --device-id 1,3,4,6 --tokens 1024 --block-size 128 \
  --storage-backends /isolated/model-check-store \
  --connector-module-path ucm.integration.vllm.v2.ucm_connector
```

设备数量至少为 TP×PP×PCP；模型还需对应 dtype、align 和 backend 配置。精确复现本轮请使用归档的 run_cpu.sh/run_npu.sh。

## 环境与源码证据

授权服务器 `110.138.0.3`，容器 `codex_kvcache_ascend_20260905`。源码仅上传隔离目录 `/home/qyh/ucm_parallel_20260926_v1` 至 `v4`，没有覆盖共享仓库。NPU 使用可见设备 1、3、4、6。

- CPU：vLLM `0.29.0+cpu`、torch `2.13.0+cpu`、NumPy `2.3.5`，用于 CUDA 布局模拟。
- NPU：vLLM `0.26.0+empty`、torch `2.10.0+cpu`、torch-npu `2.10.0.post4`、NumPy `1.26.4`。
- 实际 Ascend 检出为 `cf0baa38dfb2aef4faf6baaaa97beb4ac974ae35`，包版本 `0.19.1rc2.dev1373+gcf0baa38d`。不能将本轮标为精确 Ascend 0.26.0rc1 验收，也没有覆盖 Ascend 0.27–0.29。

工作区归档：`connector_v2_0917/runtime/parallel_20260926/`（仓库外）。包括 v1–v4 源码 zip、results.zip、各版本日志/状态、每 rank views 捕获和 SHA256 manifest。最终源码的 57 个打包文件与 v4 manifest 一致。

v3 是主矩阵；v4 增加 DCP replicated indexer 修正、空 PP sliding spec 处理和 namespace 完善后，针对相关路径及 bulk/TP1 做回归。下面明确区分测试快照，不将 v3 结果冒称为 v4 全矩阵重跑。

## 通过结果

除 bulk 行外均启用 layerwise。数字为每个 rank 的 `compared_loaded_tensor_blocks`，按 rank 排序；它是检查器计数，不是吞吐或 UCM key 数量。

| 模型 / 配置 | CPU | NPU |
|---|---|---|
| GLM52 TP2 | v3：1092 / 1092 | v3：1638 / 1638 |
| GLM52 PP2 | v3：546 / 546 | v3：819 / 819 |
| GLM52 TP2 × PP2 | v3：546 / 546 / 546 / 546 | v3：819 / 819 / 819 / 819 |
| GLM52 TP2 DCP2 | v3：468 / 468 | v4：702 / 702 |
| MiniMax27 TP2 | v3：434 / 434 | v3：868 / 868 |
| Kimi PP2 | v3：47 / 46 | v3：94 / 92 |
| DSV4 TP2 | v4：314 / 314 | v4：168 / 168 |
| DSV4 PP2 | v4：154 / 160 | v4：82 / 86 |
| GLM52 TP2 bulk | v4：1092 / 1092 | v4：1638 / 1638 |
| GLM52 TP1 | v4：1092 | v4：1638 |

默认 1024 tokens、block size 128。Kimi 为 16384 tokens、block size 768；CPU DSV4 为 block size 256、KV fp8。NPU GLM 启用 sparse SFA/indexer C8。各模型使用已有的缩减配置进行布局/字节验证。

本地：111 项 connector v2 组件测试、8 项 toolkit 测试通过。新增用例覆盖 rank 文件隔离、全分片就绪、空分片 commit、拓扑 namespace、CP 完整页与拒绝路径、PP 空组映射和 replicated indexer 寻址。

## 未通过项与修正过程

- CPU 最初 DCP 失败来自布局模拟 backend 未声明 decode LSE 能力，按所模拟 backend 补齐 `can_return_lse_for_decode`；这不代表执行了 LSE 内核。
- CPU 多进程需要共享 `VLLM_DIST_IDENT`；launcher 已设置。各 rank 使用相同 token salt，以产生相同逻辑 keys。
- NPU CP 要求原生 interleave 等于 block size；检查器按平台设置。随后 DCP 暴露 replicated indexer 多行问题，修正后 v4 通过。
- CPU PCP 在 ModelRunnerV1 中被拒绝；当前 CPU 环境也不能直接切到依赖 Triton 的 V2 路径。
- NPU PCP 在 `EngineArgs.create_engine_config` 报 `PCP (Prefill Context Parallelism) is not supported by vLLM Ascend`，尚未进入 connector。v4 保留该失败日志。PCP 目前只有拓扑/完整页逻辑及组件测试，必须在原生支持 PCP 的引擎上进一步验证，不能宣称已支持运行时使用。

后续优先补：支持 PCP 的明确引擎版本、真实 CUDA、跨节点全分片可见性，以及多种并行组合与实际推理回归。CP hybrid/子页和跨拓扑 reshard 属于后续功能，不在本轮支持范围。
