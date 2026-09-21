# Kimi

vLLM Ascend 已发布的 Moonshot AI Kimi 模型教程。

## 模型

| 模型 | vLLM Ascend latest 指南 |
| --- | --- |
| Kimi-K2-Thinking | [官方指南](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Kimi-K2-Thinking.html) |
| Kimi-K2.5 | [官方指南](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Kimi-K2.5.html) |
| Kimi-K2.6 | [官方指南](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Kimi-K2.6.html) |

## 为该模型启用 UCM

先按上述官方教程配置模型所需的引擎参数，再按 [vLLM](../../quick_start/index.md#vllm)、[vLLM-Ascend](../../quick_start/index.md#vllm-ascend) 或 [SGLang](../../quick_start/index.md#sglang) 集成指南接入 UCM。请在[支持矩阵](../../support-matrix/index.md)中确认模型与所需功能。官方引擎教程说明引擎的使用方式，并不单独证明该模型的 UCM 外部缓存行为已经通过验证。
