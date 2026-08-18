# Feature and Model Support Matrix

This page provides an overview of UCM (Unified Cache Manager) compatibility across different models and inference frameworks.
Use this matrix as a compatibility reference for model selection, deployment, and feature validation.

## Legend

| Symbol | Description |
|--------|-------------|
| Yes | Fully supported |
| No | Not supported |
| Untested | Not tested or verified |

## Model Support and Feature Compatibility

### Prefix Cache Support

This section presents prefix cache support for each model across the supported inference frameworks.
This information serves as a reference for evaluating framework compatibility in deployments that require prefix cache.

| Model | vLLM<br>(main) | vLLM-Ascend<br>(main) | SGLang<br>(main) |
|-------|:-----------:|:------------------:|:------:|
| DeepSeek V3.2 | Yes | Yes | Yes |
| DeepSeek R1 | Yes | Yes | Yes |
| DeepSeek V3/3.1 | Yes | Yes | Yes |
| DeepSeek V4 Pro | Yes | Yes | No |
| DeepSeek V4 Flash | Yes | Yes | No |
| Qwen3.6 | Yes | Yes | No |
| Qwen3.5 | Yes | Yes | No |
| Qwen3 | Yes | Yes | Yes |
| Qwen3-Moe | Yes | Yes | Yes |
| Qwen3-Next | Yes | Yes | No |
| Qwen2.5 | Yes | Yes | Yes |
| GLM-5.2 | Yes | Yes | No |
| GLM-5.1 | Yes | Yes | No |
| GLM-5 | Yes | Yes | No |
| GLM-4.x | Yes | Yes | Yes |
| MiniMax-M2.5 | Yes | Yes | Yes |
| MiniMax-M2.7 | Yes | Yes | Yes |
| MiniMax-M3 | No | No | No |
| Kimi-K2.5 | No | No | No |

> **Note**: The table lists a selected set of representative models.
> See [**Prefix Cache**](../user-guide/capabilities/prefix-cache/index.md) for more details.

### Inference Enhancement Features

This section presents support information for inference enhancement features, including Sparse Attention, ReRoPE, and CacheBlend, across the listed models and framework versions.

| Model | GsaOnDevice<br>vLLM / vLLM-Ascend 0.11.0 | ReRoPE<br>vLLM 0.11.0 | CacheBlend<br>vLLM 0.9.2 |
|-------|:-------------------------:|:------------------------:|:---------------------:|
| DeepSeek V3.2 | Yes | Yes | Yes |
| DeepSeek R1 | Yes | Yes | Yes |
| DeepSeek V3/3.1 | Yes | Yes | Yes |
| Qwen3 | Yes | Yes | Yes |
| Qwen2.5 | Yes | Yes | Yes |

> **Note**: See [**Sparse Attention**](../user-guide/capabilities/sparse-attention/index.md) and [**ReRoPE**](../user-guide/capabilities/rerope.md) for more details.

## Supported Compute Platforms and Devices

This section presents the currently supported compute platforms and devices.

| Compute Platform | Vendor | Device |
|:----------------:|:------:|:------:|
| CANN | Ascend | 910C, 910B |
| CUDA | NVIDIA | H100, H20, L40, L20 |
| MUSA | Mthreads | S5000 |
| MACA | MetaX | C500 |

> **Note**: The table shows only selected platforms.

## Notes and Limitations

- This matrix is provided as a compatibility reference for the configurations listed on this page.
- Actual behavior may vary depending on hardware, runtime settings, backend changes, and model variants.
- This support matrix is continuously updated. **For the latest information, please refer to the GitHub issues and pull requests.**
