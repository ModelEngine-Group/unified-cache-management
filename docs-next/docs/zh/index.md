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

**Unified Cache Manager（UCM）** 通过持久化 LLM KVCache 替换冗余计算，与 vLLM 集成后，在多轮对话、
长上下文推理等多种场景下可实现 **3-10 倍延迟降低**。

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

    跨请求持久化 KVCache 并复用，避免多轮对话与共享前缀的冗余预填充。支持 DRAM、SSD、远程存储等
    非 HBM 存储介质，可选 pipeline、NFS、DS3FS、Mooncake、compress 等存储后端。

    [:octicons-arrow-right-24: 了解更多](user-guide/capabilities/prefix-cache/index.md)

-   :material-chart-line: **观测能力**

    ---

    通过 vLLM connector 导出 Prometheus 指标，支持 Grafana 可视化，实时监控 KVCache 命中率、延迟、
    吞吐量等关键性能指标。

    [:octicons-arrow-right-24: 了解更多](user-guide/observability/metrics.md)

-   :material-magnify: **Trace 模式**

    ---

    轻量级诊断和评估模式，记录请求跟踪信息而不执行实际 KV cache 操作，用于模拟理论命中率和验证 UCM
    部署效果。

    [:octicons-arrow-right-24: 了解更多](user-guide/diagnostics/trace-mode.md)

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
