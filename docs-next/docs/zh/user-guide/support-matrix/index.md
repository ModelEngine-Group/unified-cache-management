# 特性与模型支持矩阵

本页面提供 UCM（Unified Cache Manager）在不同模型和推理框架间的兼容性概览。
使用此矩阵作为模型选择、部署和特性验证的兼容性参考。

## 图例

| 符号 | 描述 |
|--------|-------------|
| ✅ | 完全支持 |
| ❌ | 不支持 |
| 🟡 | 未测试或未验证 |

## 模型支持与特性兼容性

### 前缀缓存支持

本节展示各个模型在支持的推理框架中的前缀缓存支持情况。
此信息作为评估需要前缀缓存的部署场景中框架兼容性的参考。

| Model | vLLM<br>(main) | vLLM-Ascend<br>(main) | SGLang<br>(main) |
|-------|:-----------:|:------------------:|:------:|
| DeepSeek V3/3.1 | ✅ | ✅ | ✅ |
| DeepSeek R1 | ✅ | ✅ | ✅ |
| DeepSeek V3.2 | ✅ | ✅ | ✅ |
| DeepSeek V4 Pro | ✅ | ✅ | ❌ |
| DeepSeek V4 Flash | ✅ | ✅ | ❌ |
| Qwen2.5 | ✅ | ✅ | ✅ |
| Qwen3 | ✅ | ✅ | ✅ |
| Qwen3-MoE | ✅ | ✅ | ✅ |
| Qwen3-Next | ✅ | ✅ | ❌ |
| Qwen3.5 | ✅ | ✅ | ❌ |
| Qwen3.6 | ✅ | ✅ | ❌ |
| Qwen3.8 | ✅ | ✅ | ❌ |
| Qwen3.8-Flash-Next | 🟡 | 🟡 | ❌ |
| GLM-4.x | ✅ | ✅ | ✅ |
| GLM-5 | ✅ | ✅ | ❌ |
| GLM-5.1 | ✅ | ✅ | ❌ |
| GLM-5.2 | ✅ | ✅ | ❌ |
| GLM-5.3 | 🟡 | 🟡 | ❌ |
| MiniMax-M2.5 | ✅ | ✅ | ✅ |
| MiniMax-M2.7 | ✅ | ✅ | ✅ |
| MiniMax-M3 | 🟡 | 🟡 | ❌ |
| Kimi-K2.5 | ✅ | ✅ | ❌ |
| Kimi-K3 | 🟡 | 🟡 | ❌ |

> **注意**：表格列出了部分代表性模型。
> 详见 [**前缀缓存**](../capabilities/prefix-cache/index.md)。

### 推理增强特性

本节展示推理增强特性（包括稀疏注意力、ReRoPE 和 CacheBlend）在列出的模型和框架版本中的支持情况。

| Model | GsaOnDevice<br>vLLM / vLLM-Ascend 0.11.0 | ReRoPE<br>vLLM 0.11.0 | CacheBlend<br>vLLM 0.9.2 |
|-------|:-------------------------:|:------------------------:|:---------------------:|
| DeepSeek V3/3.1 | ✅ | ✅ | ✅ |
| DeepSeek R1 | ✅ | ✅ | ✅ |
| DeepSeek V3.2 | ✅ | ✅ | ✅ |
| Qwen2.5 | ✅ | ✅ | ✅ |
| Qwen3 | ✅ | ✅ | ✅ |

> **注意**：推理增强特性（稀疏注意力、ReRoPE、CacheBlend）不在本文档范围内。

## 支持的计算平台与设备

本节展示当前支持的计算平台和设备。

| 计算平台 | 厂商 | 设备 |
|:----------------:|:------:|:------:|
| CANN | Ascend | 910C, 910B |
| CUDA | NVIDIA | H100, H20, L40, L20 |
| MUSA | Mthreads | S5000 |
| MACA | MetaX | C500 |

> **注意**：表格仅显示部分平台。

## 注意事项与限制

- 本矩阵作为本页面列出的配置的兼容性参考。
- 实际行为可能因硬件、运行时设置、后端变更和模型变体而异。
- 本支持矩阵持续更新。**有关最新信息，请参考 GitHub issues 和 pull requests。**
