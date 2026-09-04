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

## 功能特性

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

-   :material-engine: **部署**

    ---

    将 UCM 与 vLLM、vLLM Ascend、SGLang、MindIE 集成。

    [:octicons-arrow-right-24: 部署](user-guide/quick_start/quickstart_vllm.md)

-   :material-view-grid-plus: **兼容性矩阵**

    ---

    一览支持的模型、平台与特性覆盖。

    [:octicons-arrow-right-24: 矩阵](user-guide/support-matrix/index.md)

</div>

## 工具

<div class="grid cards" markdown>

-   :material-check-circle: **环境预检**

    ---

    在 UCM 部署前运行环境预检，验证版本、驱动、内核和带宽。

    [:octicons-arrow-right-24: 预检](toolkit/user/precheck.md)

-   :material-harddisk: **带宽模拟**

    ---

    测试 POSIX AIO 存储 dump/load 性能，用于存储基准测试。

    [:octicons-arrow-right-24: 带宽模拟](toolkit/user/posix-aio.md)

-   :material-chart-box: **指标监控**

    ---

    收集 Prometheus/OpenMetrics 样本到 SQLite 并在终端查询聚合指标。

    [:octicons-arrow-right-24: 指标监控](toolkit/user/metrics-view.md)

-   :material-network: **网卡监控**

    ---

    监控物理网卡实时流量，支持后台采样和阶段统计。

    [:octicons-arrow-right-24: 网卡监控](toolkit/user/nic-monitor.md)

-   :material-test-tube: **开发沙箱**

    ---

    测量主机到设备内存复制带宽和磁盘 AIO 吞吐量，用于性能测试。

    [:octicons-arrow-right-24: 开发沙箱](toolkit/user/dev-sandbox.md)

-   :material-calculator: **KV Cache 计算器**

    ---

    根据模型配置估算 KV Cache 显存占用。

    [:octicons-arrow-right-24: 计算器](toolkit/kv-cache-calculator.md)

</div>

**[关于我们](about.md)** — 了解 UCM 团队与我们的使命。

## 版本兼容性

| 分支 | 状态 | vLLM 版本 | vLLM-Ascend 版本 |
| --- | --- | --- | --- |
| `main` | 维护中 | v0.27.1 | nightly-0.26.0 |
| `develop` | 维护中 | v0.27.1 | nightly-0.26.0 |
