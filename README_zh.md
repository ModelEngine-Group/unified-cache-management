<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/source/logos/UCM-dark.png">
    <img alt="UCM" src="docs/source/logos/UCM-light.png" width=50%>
  </picture>
</p>

<p align="center">
| <a href="docs/source/index.md"><b>文档</b></a> | <a href="https://modelengine-ai.net/#/ucm"><b>网站</b></a> | <a href="https://github.com/ModelEngine-Group/unified-cache-management/issues/78"><b>发展路线图</b></a> |
</p>

---

## 概述

统一缓存管理器（Unified Cache Management, UCM）的核心原理是持久化 LLM 的 KVCache，并通过多种检索机制替代冗余计算。UCM 不仅支持前缀缓存（prefix cache），还提供了多种无需训练的稀疏注意力检索方法，在处理极长序列推理任务时达到更高性能。此外，UCM 基于存算分离架构提供了 PD 分离方案，使得异构计算资源的管理更加简单灵活。与 vLLM 集成后，UCM 在多轮对话和长上下文推理等多种场景下可将推理延迟降低 3–10 倍。

![architecture.png](./docs/source/_static/images/architecture.png)


---

## 支持特性
- [前缀匹配]()
- [缓存融合]()
- [模型窗口外推]()
- [预填充卸载]()
- [稀疏注意力]()
- [稀疏注意力卸载]()
- [异构PD分离]()

---

## 快速开始

请参考 [快速开始](./docs/source/getting-started/quick_start.md).

---

## 分支

| **分支**   |     状态   | vLLM 版本 | 
|-----------:|-----------:|-------------:|
|       main | 维护中 |       v0.9.2 | 
|    develop | 维护中 |       v0.9.2 |

---

## 联系我们
如需技术咨询或功能请求，请提交 GitHub [Issues](https://github.com/ModelEngine-Group/unified-cache-management/issues).

## 许可协议

UCM 采用 MIT 许可证（附加额外条件），详情请参阅 [LICENSE](./LICENSE) 文件。
