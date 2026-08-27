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

**Unified Cache Manager (UCM)** persists LLM KVCache to replace redundant computations.
When integrated with vLLM, UCM achieves a **3-10x reduction** in inference latency across
various scenarios, including multi-turn dialogue and long-context reasoning tasks.

<div align="center" markdown>

[![GitHub stars](https://img.shields.io/github/stars/ModelEngine-Group/unified-cache-management?style=social)](https://github.com/ModelEngine-Group/unified-cache-management)
[![GitHub forks](https://img.shields.io/github/forks/ModelEngine-Group/unified-cache-management?style=social)](https://github.com/ModelEngine-Group/unified-cache-management)
[![GitHub watch](https://img.shields.io/github/watchers/ModelEngine-Group/unified-cache-management?style=social)](https://github.com/ModelEngine-Group/unified-cache-management)

</div>

<div class="grid cards" markdown>

-   :material-database-clock-outline: **Prefix Cache**

    ---

    Persist KVCache across requests and reuse it to avoid redundant prefill for
    multi-turn dialogue and shared prefixes. Supports non-HBM storage media including
    DRAM, SSD, and remote storage with pipeline, NFS, DS3FS, Mooncake, and compress backends.

    [:octicons-arrow-right-24: Learn more](user-guide/capabilities/prefix-cache/index.md)

-   :material-chart-line: **Observability**

    ---

    Export Prometheus metrics through the vLLM connector and visualize with Grafana
    to monitor key performance metrics like KVCache hit rate, latency, and throughput in real-time.

    [:octicons-arrow-right-24: Learn more](user-guide/observability/metrics.md)

-   :material-magnify: **Trace Mode**

    ---

    Lightweight diagnostic and evaluation mode that records request traces without
    performing actual KV cache operations, used to simulate theoretical hit rates
    and validate UCM deployment effectiveness.

    [:octicons-arrow-right-24: Learn more](user-guide/diagnostics/trace-mode.md)

</div>

## Get Started

<div class="grid cards" markdown>

-   :material-tools: **Installation**

    ---

    Pick your UCM version, engine, device, OS, and install method, and get the
    exact command to deploy.

    [:octicons-arrow-right-24: Installation](user-guide/installation.md)

-   :material-rocket-launch: **Quick Start**

    ---

    Get started with UCM in your inference engine quickly. Choose your engine
    (vLLM, vLLM-Ascend, SGLang, MindIE) and follow the integration guide.

    [:octicons-arrow-right-24: Quick Start](user-guide/quick-start/index.md)

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
