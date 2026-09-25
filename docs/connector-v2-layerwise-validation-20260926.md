# Connector v2 layerwise 容器验证（2026-09-26）

本次工作区实现通过 **8/8 model-check**（CPU/NPU 各四种模型配置），本地组件测试 **103/103**。基于 `68c3b2c2` 加当前未提交改动；准确源码以快照和 SHA256 清单为准。本轮没有修改 connector 实现来绕过测试。

## 环境与隔离

- 用户明确授权的主机：`110.138.0.3`；容器：`codex_kvcache_ascend_20260905`。
- 隔离目录：`/home/qyh/ucm_layerwise_20260926`。未改动 CPU/NPU 原有 UCM 工作区。
- CPU：vLLM `0.29.0+cpu`、Torch `2.13.0+cpu`、NumPy `2.3.5`。
- NPU：vLLM `0.26.0+empty`、Torch `2.10.0+cpu` 加 torch-npu `2.10.0.post4`、NumPy `1.26.4`，可见设备 0，TP1。
- Ascend 实际源码：`/vllm-workspace/vllm-ascend`，干净的 detached HEAD `cf0baa38dfb2aef4faf6baaaa97beb4ac974ae35`。安装元数据版本为 `0.19.1rc2.dev1373+gcf0baa38d`；git describe 为 `v0.19.1rc1-1373-gcf0baa38d`。结果中的 `npu26` 指 vLLM 0.26 环境，**不能据此声明精确的 vllm-ascend 0.26.0rc1 发布版已验收**。

## 结果

所有命令返回 0。段数来自日志中的 `compared_loaded_tensor_blocks`，在 v2 下表示参与比较的 payload segment 条目，并非物理 block 数。

| 模型配置 | CPU 比较段数 | NPU 比较段数 | 结果 |
|---|---:|---:|---|
| GLM52 | 1092 | 1638 | 两端通过 |
| MiniMax27 | 434 | 868 | 两端通过 |
| Kimi | 93 | 186 | 两端通过 |
| DSV4 | 314 | 168 | 两端通过 |

四种配置的保存回调去重后数量分别为 156、62、24、167；DSV4 的回调中部分没有本步 dump 数据，实际在途保存任务为 147。保存和加载两个步骤结束后均断言 `_save_complete` 为真，且 load/dump 任务容器为空。

CPU DSV4 此次按逐层 payload 检查，不能与历史 bulk 合并传输的 8 个 span 直接比较数量或性能。

## 执行与证据

使用原有 `ucm-toolkit run model-check`，显式传入 `--layerwise`。沿用真实 ModelRunner KV 分配、Scheduler、源/目标不同 block 和确定性字节填充/比对流程，不加载模型权重做推理。

隔离 connector 以 `candidate_v2` 包加载。探针只记录 views、断言二维 Transfer 和步末任务状态；原有 toolkit 启动器保持原样，将本次 `common.py` 中四个 v2 字节 oracle helper 覆盖到运行进程，用于按 layer_name payload 填充和检查，避免把未传输的 slot padding 纳入 layerwise 比对。`TorchTensorByteAccess` 使用同一快照版本。

10 个源码文件的远端 SHA256 与上传清单一致；另外逐一检查八份日志实际加载的 connector、proxy、common 路径及哈希，全部匹配。

- [运行脚本及参数](../../connector_v2_0917/runtime/layerwise_20260926/remote/run.sh)
- [汇总状态](../../connector_v2_0917/runtime/layerwise_20260926/remote/status.txt)
- [机器可读结果](../../connector_v2_0917/runtime/layerwise_20260926/summary.json)
- [源码哈希清单](../../connector_v2_0917/runtime/layerwise_20260926/manifest.json)
- [完整结果归档](../../connector_v2_0917/runtime/layerwise_20260926/results.zip)
- [源码快照](../../connector_v2_0917/runtime/layerwise_20260926/snapshot.zip)

首次启动遇到 Windows CRLF 和 `--layerwise true` 参数错误，均发生在模型测试前；修正为 LF 和 `--layerwise` 后运行上述八组，初始错误日志保留在归档中。复现应使用 `remote/run.sh`，而不是快照内未经修正的启动脚本。

这些运行产物位于主仓库外的交接目录，不随主仓库提交自动发布。

## 验证边界

- 已验证实际 CPU/NPU 环境下 TP1 layerwise dump、commit、lookup/load 和有效 payload 字节一致性。
- 当前文件 Proxy 同步执行；异步 enqueue/wait、失败不发布及描述符生命周期有本地测试替身覆盖，尚未实测真实异步后端的计算/IO 重叠。
- 本次不是性能基准、多 rank 测试、CUDA 实卡测试或模型推理精度测试。
- 字节 oracle 共用 UCM 寻址逻辑，不能排除共同的地址计算错误；本地另外覆盖六种物理排列、部分块和未选中区域保护。
- layerwise 文件 namespace 与 bulk 分开，未声称两种文件格式互用。
