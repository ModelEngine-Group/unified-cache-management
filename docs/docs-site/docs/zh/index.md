---
hide:
  - navigation
  - toc
---

<div align="center" markdown>

![UCM](../assets/images/UCM-light.png#only-light){ style="height:96px;width:auto" }
![UCM](../assets/images/UCM-dark.png#only-dark){ style="height:96px;width:auto" }

</div>

# Unified Cache Manager

**Unified Cache Manager（UCM）是一套面向大模型推理的 KV Cache 管理与复用系统。它将可复用的 KV Cache 持久化到外部存储，并在不同请求和兼容的推理实例之间共享，减少重复的 Prefill 计算。**

在多轮对话、长上下文推理、Coding 和多 Agent 协作等场景中，项目代码、任务说明和对话历史会在连续请求中重复出现。UCM 可以复用这些上下文已经生成的 KV Cache，减少对相同前缀的重复计算，从而降低计算开销和响应延迟。

相比仅保存在推理进程内部的 KV Cache，UCM 利用外部存储扩展缓存容量，并支持 KV Cache **跨请求复用、跨进程持久保存，以及兼容实例之间的共享**，让已有计算结果能够被持续复用。

与 vLLM 集成后，UCM 在多轮对话、长上下文推理等重复前缀较多的场景下可实现 **3–10 倍的延迟降低**。

## 从你的任务开始

<div class="ucm-home-cards" markdown>

[<span class="ucm-home-card__icon" aria-hidden="true">:material-play-circle-outline:</span>
<span class="ucm-home-card__title">快速开始</span>
<span class="ucm-home-card__description">选择安装制品，配置推理引擎，运行第一个接入 UCM 的服务。</span>
<span class="ucm-home-card__arrow" aria-hidden="true">:material-arrow-right:</span>](user-guide/quick_start/index.md){ .ucm-home-card }

[<span class="ucm-home-card__icon" aria-hidden="true">:material-view-grid-outline:</span>
<span class="ucm-home-card__title">支持范围</span>
<span class="ucm-home-card__description">查看引擎、模型、设备与功能的支持情况，确认部署组合。</span>
<span class="ucm-home-card__arrow" aria-hidden="true">:material-arrow-right:</span>](user-guide/support-matrix/index.md){ .ucm-home-card }

[<span class="ucm-home-card__icon" aria-hidden="true">:material-layers-triple-outline:</span>
<span class="ucm-home-card__title">从模型开始</span>
<span class="ucm-home-card__description">找到模型的官方指南与 UCM 示例，按模型完成部署和调用。</span>
<span class="ucm-home-card__arrow" aria-hidden="true">:material-arrow-right:</span>](user-guide/model-tour/index.md){ .ucm-home-card }

[<span class="ucm-home-card__icon" aria-hidden="true">:material-code-braces:</span>
<span class="ucm-home-card__title">理解和扩展 UCM</span>
<span class="ucm-home-card__description">沿一条请求理解架构与缓存机制，开发存储后端或添加指标。</span>
<span class="ucm-home-card__arrow" aria-hidden="true">:material-arrow-right:</span>](developer-guide/index.md){ .ucm-home-card }

</div>

## 运行与评测

[运行与排障](user-guide/observability/index.md)帮助你判断缓存是否生效、后端是否健康，以及时间花在哪个阶段。[工具集](toolkit/index.md)提供部署预检、存储测试和指标查看，容量规划可使用 [KV Cache 计算器](toolkit/kv-cache-calculator.md)。

## 支持范围

UCM 提供 vLLM、vLLM-Ascend 和 SGLang 的集成路径。模型和平台范围见[支持矩阵](user-guide/support-matrix/index.md)；具体可安装的引擎与后端组合以[安装页面](user-guide/quick_start/index.md)中的发布制品为准。

[GitHub 源码](https://github.com/ModelEngine-Group/unified-cache-management) · [参与贡献](developer-guide/contribute.md) · [关于 UCM](about.md)
