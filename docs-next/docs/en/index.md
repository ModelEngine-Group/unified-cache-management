---
hide:
  - navigation
  - toc
---

<div align="center" markdown>

![UCM](/assets/images/UCM-light.png#only-light){: style="height:120px;width:auto"}
![UCM](/assets/images/UCM-dark.png#only-dark){: style="height:120px;width:auto"}

</div>

# Unified Cache Manager

The core principle of **Unified Cache Manager (UCM)** is to persist the LLM KVCache
and replace redundant computations through multiple retrieval mechanisms. UCM not
only supports prefix caching but also offers a variety of training-free sparse
attention retrieval methods, delivering higher performance when handling extremely
long sequence inference tasks.

Additionally, UCM provides a **PD disaggregation** solution based on a
storage-compute separation architecture, which enables more straightforward and
flexible management of heterogeneous computing resources. When integrated with
vLLM, UCM achieves a **3-10x reduction** in inference latency across various
scenarios, including multi-turn dialogue and long-context reasoning tasks.

<div align="center" markdown>

[![GitHub stars](https://img.shields.io/github/stars/ModelEngine-Group/unified-cache-management?style=social)](https://github.com/ModelEngine-Group/unified-cache-management)
[![GitHub forks](https://img.shields.io/github/forks/ModelEngine-Group/unified-cache-management?style=social)](https://github.com/ModelEngine-Group/unified-cache-management)
[![GitHub watch](https://img.shields.io/github/watchers/ModelEngine-Group/unified-cache-management?style=social)](https://github.com/ModelEngine-Group/unified-cache-management)

</div>

<div class="grid cards" markdown>

-   :material-database-clock-outline: **Prefix Cache**

    ---

    Persist KVCache across requests and reuse it to avoid redundant prefill for
    multi-turn dialogue and shared prefixes. Choose from pipeline, NFS, DS3FS,
    Mooncake, and compress stores.

    [:octicons-arrow-right-24: Learn more](user-guide/capabilities/index.md)

-   :material-radar: **Sparse Attention**

    ---

    Training-free sparse retrieval methods (GSA, CacheBlend) that select the most
    relevant KV slices, reducing attention overhead on extremely long sequences.

    [:octicons-arrow-right-24: Learn more](user-guide/capabilities/index.md)

-   :material-split-horizontal: **PD Disaggregation**

    ---

    Storage-compute separation for prefill/decode disaggregation, giving flexible
    control over heterogeneous GPU/NPU clusters with centralized, distributed,
    and large-scale EP topologies.

    [:octicons-arrow-right-24: Learn more](user-guide/capabilities/index.md)

-   :material-chart-bell-curve: **ReRoPE**

    ---

    Corrected rotary position embedding that extends context length without
    retraining, restoring position encoding quality for long-context inference.

    [:octicons-arrow-right-24: Learn more](user-guide/capabilities/index.md)

</div>

## Get Started

<div class="grid cards" markdown>

-   :material-tools: **Installation**

    ---

    Pick your UCM version, engine, device, OS, and install method, and get the
    exact command to deploy.

    [:octicons-arrow-right-24: Installation](user-guide/installation.md)

-   :material-engine: **Engines**

    ---

    Integrate UCM with vLLM, vLLM Ascend, SGLang, and MindIE.

    [:octicons-arrow-right-24: Engines](user-guide/engines/index.md)

-   :material-view-grid-plus: **Compatibility Matrix**

    ---

    Supported models, platforms, and feature coverage at a glance.

    [:octicons-arrow-right-24: Matrix](reference/api-parameters.md)

-   :material-calculator: **KV Cache Calculator**

    ---

    Estimate KV cache memory usage for your model configuration.

    [:octicons-arrow-right-24: Calculator](toolkit/kv-cache-calculator.md)

</div>

## Publications

- [HATA: Trainable and Hardware-Efficient Hash-Aware Top-k Attention for Scalable Large Model Inference](https://arxiv.org/abs/2506.02572)
- [ReTaKe: Reducing Temporal and Knowledge Redundancy for Long Video Understanding](https://arxiv.org/abs/2412.20504)
- [AdaReTaKe: Adaptive Redundancy Reduction to Perceive Longer for Video-language Understanding](https://arxiv.org/abs/2503.12559)
- [Dynamic Early Exit in Reasoning Models](https://arxiv.org/abs/2504.15895)
- [Sparse Attention across Multiple-context KV Cache](https://arxiv.org/abs/2508.11661)

## Community

- [UCM on the ModelEngine Community](https://modelengine-ai.net/#/ucm)
- [GitHub Discussions](https://github.com/ModelEngine-Group/unified-cache-management/discussions)
