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

**Unified Cache Manager（UCM）** 的核心原理是持久化 LLM KVCache，并通过多种检索机制替换冗余计算。
UCM 不仅支持前缀缓存（Prefix Cache），还提供多种免训练的稀疏注意力检索方法，在处理超长序列推理任务时
带来更高的性能。

此外，UCM 提供基于存算分离架构的 **PD 分离**方案，可以更直接、灵活地管理异构计算资源。与 vLLM 集成后，
UCM 在多轮对话、长上下文推理等多种场景下可实现 **3-10 倍延迟降低**。

!!! note "中文站点说明"

    当前中文内容为 AI 自动生成的待评审样例，技术准确性以英文版本为准。后续将通过 AI Robot 在 PR 中
    同步生成并评审完整中文内容。

<div align="center" markdown>

[![GitHub stars](https://img.shields.io/github/stars/ModelEngine-Group/unified-cache-management?style=social)](https://github.com/ModelEngine-Group/unified-cache-management)
[![GitHub forks](https://img.shields.io/github/forks/ModelEngine-Group/unified-cache-management?style=social)](https://github.com/ModelEngine-Group/unified-cache-management)
[![GitHub watch](https://img.shields.io/github/watchers/ModelEngine-Group/unified-cache-management?style=social)](https://github.com/ModelEngine-Group/unified-cache-management)

</div>

<div class="grid cards" markdown>

-   :material-database-clock-outline: **Prefix Cache 前缀缓存**

    ---

    跨请求持久化 KVCache 并复用，避免多轮对话与共享前缀的冗余预填充。可选 pipeline、NFS、DS3FS、
    Mooncake、compress 等存储后端。

    [:octicons-arrow-right-24: 了解更多](user-guide/capabilities/prefix-cache/index.md)

-   :material-radar: **Sparse Attention 稀疏注意力**

    ---

    免训练稀疏检索方法（GSA、CacheBlend），选取最相关的 KV 切片，降低超长序列的注意力开销。

    [:octicons-arrow-right-24: 了解更多](user-guide/capabilities/sparse-attention/index.md)

-   :material-split-horizontal: **PD Disaggregation PD 分离**

    ---

    预填充/解码分离的存算分离架构，灵活管控异构 GPU/NPU 集群，支持集中式、分布式与大规模 EP 拓扑。

    [:octicons-arrow-right-24: 了解更多](user-guide/capabilities/pd-disaggregation/index.md)

-   :material-chart-bell-curve: **ReRoPE**

    ---

    修正旋转位置编码，无需重训练即可扩展上下文长度，恢复长上下文推理的位置编码质量。

    [:octicons-arrow-right-24: 了解更多](user-guide/capabilities/rerope.md)

</div>

## 快速开始

<div class="grid cards" markdown>

-   :material-tools: **安装**

    ---

    选择 UCM 版本、引擎、设备、操作系统与安装方式，获取对应的部署命令。

    [:octicons-arrow-right-24: 安装](user-guide/installation.md)

-   :material-engine: **引擎**

    ---

    将 UCM 与 vLLM、vLLM Ascend、SGLang、MindIE 集成。

    [:octicons-arrow-right-24: 引擎](user-guide/engines/vllm.md)

-   :material-view-grid-plus: **兼容性矩阵**

    ---

    一览支持的模型、平台与特性覆盖。

    [:octicons-arrow-right-24: 矩阵](reference/compatibility.md)

-   :material-calculator: **KV Cache 计算器**

    ---

    根据模型配置估算 KV Cache 显存占用。

    [:octicons-arrow-right-24: 计算器](toolkit/kv-cache-calculator.md)

</div>

## 社区

- [ModelEngine 社区 UCM](https://modelengine-ai.net/#/ucm)
- [GitHub Discussions](https://github.com/ModelEngine-Group/unified-cache-management/discussions)
