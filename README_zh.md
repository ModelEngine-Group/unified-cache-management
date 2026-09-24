<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/ModelEngine-Group/unified-cache-management/main/docs/source/logos/UCM-dark.png">
    <img alt="UCM" src="https://raw.githubusercontent.com/ModelEngine-Group/unified-cache-management/main/docs/source/logos/UCM-light.png" width="55%">
  </picture>
</p>

<h3 align="center">面向 LLM 推理的统一 KV Cache 管理</h3>

<p align="center">
  <a href="https://ucm.readthedocs.io/en/latest/">文档</a> ·
  <a href="https://ucm.readthedocs.io/en/latest/user-guide/quick_start/">快速开始</a> ·
  <a href="https://modelengine-ai.net/#/ucm">官网</a> ·
  <a href="./README.md">English</a>
</p>
<div align="center">

[![DeepWiki](https://img.shields.io/badge/DeepWiki-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAyCAYAAAAnWDnqAAAAAXNSR0IArs4c6QAAA05JREFUaEPtmUtyEzEQhtWTQyQLHNak2AB7ZnyXZMEjXMGeK/AIi+QuHrMnbChYY7MIh8g01fJoopFb0uhhEqqcbWTp06/uv1saEDv4O3n3dV60RfP947Mm9/SQc0ICFQgzfc4CYZoTPAswgSJCCUJUnAAoRHOAUOcATwbmVLWdGoH//PB8mnKqScAhsD0kYP3j/Yt5LPQe2KvcXmGvRHcDnpxfL2zOYJ1mFwrryWTz0advv1Ut4CJgf5uhDuDj5eUcAUoahrdY/56ebRWeraTjMt/00Sh3UDtjgHtQNHwcRGOC98BJEAEymycmYcWwOprTgcB6VZ5JK5TAJ+fXGLBm3FDAmn6oPPjR4rKCAoJCal2eAiQp2x0vxTPB3ALO2CRkwmDy5WohzBDwSEFKRwPbknEggCPB/imwrycgxX2NzoMCHhPkDwqYMr9tRcP5qNrMZHkVnOjRMWwLCcr8ohBVb1OMjxLwGCvjTikrsBOiA6fNyCrm8V1rP93iVPpwaE+gO0SsWmPiXB+jikdf6SizrT5qKasx5j8ABbHpFTx+vFXp9EnYQmLx02h1QTTrl6eDqxLnGjporxl3NL3agEvXdT0WmEost648sQOYAeJS9Q7bfUVoMGnjo4AZdUMQku50McDcMWcBPvr0SzbTAFDfvJqwLzgxwATnCgnp4wDl6Aa+Ax283gghmj+vj7feE2KBBRMW3FzOpLOADl0Isb5587h/U4gGvkt5v60Z1VLG8BhYjbzRwyQZemwAd6cCR5/XFWLYZRIMpX39AR0tjaGGiGzLVyhse5C9RKC6ai42ppWPKiBagOvaYk8lO7DajerabOZP46Lby5wKjw1HCRx7p9sVMOWGzb/vA1hwiWc6jm3MvQDTogQkiqIhJV0nBQBTU+3okKCFDy9WwferkHjtxib7t3xIUQtHxnIwtx4mpg26/HfwVNVDb4oI9RHmx5WGelRVlrtiw43zboCLaxv46AZeB3IlTkwouebTr1y2NjSpHz68WNFjHvupy3q8TFn3Hos2IAk4Ju5dCo8B3wP7VPr/FGaKiG+T+v+TQqIrOqMTL1VdWV1DdmcbO8KXBz6esmYWYKPwDL5b5FA1a0hwapHiom0r/cKaoqr+27/XcrS5UwSMbQAAAABJRU5ErkJggg==)](https://deepwiki.com/ModelEngine-Group/unified-cache-management)

</div>

## UCM 是什么？

**统一缓存管理（Unified Cache Manager，UCM）是位于推理引擎与存储之间的统一 KV Cache 层。** 它负责管理缓存的复用、放置与迁移，请求和推理实例能够复用已有的计算结果。

Agent 工作流、多轮对话和多模态应用常常会重复访问相同的上下文。UCM 让可复用的 KV Cache 超越单个推理进程的边界而可用，并通过外部存储扩展缓存容量，从而帮助减少重复计算、缓解加速器显存压力。

引擎适配器、缓存机制与存储后端可以通过可插拔接口独立演进。

## 架构

![UCM 逻辑架构：推理负载与引擎、统一缓存生命周期管理，以及带可插拔后端的全局 KV 存储。](./docs/source/_static/architecture.svg)
- **应用与推理引擎**产生并消费 KV Cache。复用模式包括多模态上下文、共享前缀，以及拼接复用等机制所使用的缓存分块。
- **统一缓存（Unified Cache）** 管理缓存的使用生命周期：复用什么、保留在哪里、何时迁移。Prefill/Decode（PD）传输是缓存迁移的一种形式。可插拔的检索与加载策略同时支撑稀疏注意力。
- **全局 KV 存储（Global KV Store）** 在文件系统、内存池等后端之上提供统一的存储抽象。各存储实现根据自身能力处理数据的保留、回收与可靠性。

功能的可用性与存储保证取决于所选的引擎、模型、平台与后端；请参阅[支持矩阵](https://ucm.readthedocs.io/en/latest/user-guide/support-matrix/)。

## 快速开始

1. 查看[支持矩阵](https://ucm.readthedocs.io/en/latest/user-guide/support-matrix/)，确认兼容的配置组合。
2. 按照[快速开始](https://ucm.readthedocs.io/en/latest/user-guide/quick_start/)指引，选择安装产物、配置存储，并使用 UCM 运行推理引擎。
3. 使用[运维指南](https://ucm.readthedocs.io/en/latest/user-guide/observability/)验证缓存复用情况并检查运行时行为。

安装命令、镜像标签、引擎版本和配置示例均维护在文档中。如遇部署问题，请参阅[故障排查](https://ucm.readthedocs.io/en/latest/reference/troubleshooting/)。

## 参与贡献 & 社区

欢迎为引擎适配器、存储后端、缓存机制、测试和文档做出贡献。

- 从[贡献指南](https://ucm.readthedocs.io/en/latest/developer-guide/contribute/)开始。
- 通过[扩展 Store](https://ucm.readthedocs.io/en/latest/developer-guide/extending-store/) 添加新的后端。
- 通过 [GitHub Issues](https://github.com/ModelEngine-Group/unified-cache-management/issues) 报告缺陷或提出功能建议。

## 许可证

UCM 基于 MIT 许可证并附加额外条件进行授权。完整条款请阅读 [LICENSE](./LICENSE)。
