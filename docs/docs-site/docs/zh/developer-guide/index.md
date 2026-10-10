# 开发者指南

理解 UCM 如何把引擎的 KV Cache 生命周期连接到外部存储，再找到需要修改的接口与实现。先读架构和主请求路径，随后按开发任务进入对应章节。

## 建立系统理解

1. [总体架构](architecture.md)：引擎、Connector 和 Store 的职责，以及数据经过的存储层。
2. [缓存工作原理](capability-principles.md)：块如何匹配，什么条件下可以复用，分层缓存如何影响读取成本。
3. [请求生命周期](request-lifecycle.md)：Scheduler 与 Worker 如何交接元数据，加载和保存何时完成。
4. [PD 集成原理](pd-integration.md)：跨请求复用与当前请求 P/D 交接如何配合。

## 按开发任务继续

| 开发任务 | 指南 |
| --- | --- |
| 准备可修改的运行环境 | [源码构建](build_from_source.md) |
| 选择后端并配置缓存 | [缓存配置](cache-configuration/index.md) |
| 接入存储或新增 Pipeline 阶段 | [扩展 Store](extending-store.md) |
| 增加运行时测量 | [指标开发](add-metrics.md) |
| 接入诊断或测试工具 | [Toolkit 架构与扩展](../toolkit/developer/design.md) |
| 修改原生带宽测试工具 | [开发沙箱](../toolkit/developer/dev-sandbox.md) |
| 提交代码或文档改进 | [参与贡献](contribute.md) |

配置部署而不修改代码时，使用[用户指南](../user-guide/index.md)。本文档描述当前源码中的行为；运行验证需要匹配的引擎版本和设备环境。
