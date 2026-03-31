# Feature and Model Support Matrix

This page provides an overview of UCM (Unified Cache Manager) compatibility across different models and inference frameworks.
Use this matrix as a compatibility reference for model selection, deployment, and feature validation.

## Legend 🧭

| Symbol | Description |
|--------|-------------|
| ✅ | Fully supported |
| ❌ | Not supported |
| 🟡 | Not tested or verified |

## Model Support and Feature Compatibility 🧩

### Prefix Cache Support

This section presents prefix cache support for each model across the supported inference frameworks.
This information serves as a reference for evaluating framework compatibility in deployments that require prefix cache.

| Model | vLLM<br>(v0.17.0) | vLLM-Ascend<br>(v0.17.0rc1) | SGLang<br>(v0.5.9) |
|-------|:-----------:|:------------------:|:------:|
| DeepSeek V3.2 | ✅ | ✅ | ✅ |
| DeepSeek R1 | ✅ | ✅ | ✅ |
| DeepSeek V3/3.1 | ✅ | ✅ | ✅ |
| Qwen3.5 | ❌ | ❌ | ❌ |
| Qwen3 | ✅ | ✅ | ✅ |
| Qwen3-Moe | ✅ | ✅ | ✅ |
| Qwen3-Next | ❌ | ❌ | ❌ |
| Qwen2.5 | ✅ | ✅ | ✅ |
| GLM-5 | ✅ | ✅ | ❌ |
| GLM-4.x | ✅ | ✅ | ✅ |
| MiniMax-M2.5 | ✅ | ✅ | ✅ |
| Kimi-K2.5 | ❌ | ❌ | ❌ |

> **Note**: The table lists a selected set of representative models.
> The framework versions shown in the table indicate the latest adapted versions.
> See [**Prefix Cache**](../prefix-cache/index.md) for more details.

### Inference Enhancement Features

This section presents support information for inference enhancement features, including Sparse Attention, ReRoPE, and CacheBlend, across the listed models and framework versions.

| Model | GsaOnDevice<br>vLLM / vLLM-Ascend 0.11.0 | ReRoPE<br>vLLM 0.11.0 | CacheBlend<br>vLLM 0.9.2 |
|-------|:-------------------------:|:------------------------:|:---------------------:|
| DeepSeek V3.2 | ✅ | ✅ | ✅ |
| DeepSeek R1 | ✅ | ✅ | ✅ |
| DeepSeek V3/3.1 | ✅ | ✅ | ✅ |
| Qwen3 | ✅ | ✅ | ✅ |
| Qwen2.5 | ✅ | ✅ | ✅ |

> **Note**: See [**Sparse Attention**](../sparse-attention/index.md) and [**ReRoPE**](../rerope/rerope.md) for more details.

## Notes and Limitations 📌

- This matrix is provided as a compatibility reference for the configurations listed on this page.
- Actual behavior may vary depending on hardware, runtime settings, backend changes, and model variants.
- This support matrix is continuously updated. **For the latest information, please refer to the GitHub issues and pull requests.**
