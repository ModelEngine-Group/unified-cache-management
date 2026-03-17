# Feature and Model Support Matrix

This page provides an overview of the supported models and features in UCM (Unified Cache Manager).

## Legend

| Symbol | Description |
|--------|-------------|
| ✅ | Fully supported |
| ❌ | Not supported |
| 🟡 | Not tested or verified |

## Model Support and Feature Compatibility

### Prefix Cache Support

| Model | vLLM（main） | vLLM-Ascend (main) | SGLang |
|-------|:-----------------:|:-----------:|:------:|
| DeepSeek V3.2 | ✅ | ✅ | 🟡 |
| DeepSeek R1 | ✅ | ✅ | ✅ |
| DeepSeek V3/3.1 | ✅ | ✅ | ✅ |
| Qwen3.5 | ❌ | ❌ | ❌ |
| Qwen3 | ✅ | ✅ | ✅ |
| Qwen3-Coder | ✅ | ✅ | ✅ |
| Qwen3-Moe | ✅ | ✅ | ✅ |
| Qwen3-Next | ❌ | ❌ | ❌ |
| Qwen2.5 | ✅ | ✅ | ✅ |
| GLM-5 | ✅ | ❌ | 🟡 |
| GLM-4.x | ✅ | ✅ | 🟡 |
| MiniMax-M2.5 | ✅ | ✅ | ✅ |


### Sparse Attention Support

| Model | GsaOnDevice<br>vLLM 0.11.0 | CacheBlend<br>vLLM v0.9.2 | ReRoPE<br>vLLM 0.11.0 |
|-------|:-------------------------:|:-------------------------:|:---------------------:|
| DeepSeek V3.2 | ✅ | ✅ | ✅ |
| DeepSeek R1 | ✅ | ✅ | ✅ |
| DeepSeek V3/3.1 | ✅ | ✅ | ✅ |
| Qwen3 | ✅ | ✅ | ✅ |
| Qwen2.5 | ✅ | ✅ | ✅ |

> **Note**: vLLM-Ascend currently only supports GsaOnDevice.



## Notes



> **Note**
> This support matrix is continuously updated. For the latest information, please refer to the GitHub issues and pull requests.
