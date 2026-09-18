# 部署

引擎快速开始解决单个服务如何接入 UCM。本节帮助你选择进程与集群的组织方式，以及客户端访问服务的入口。

| 部署任务 | 指南 | 主要准备 |
| --- | --- | --- |
| 使用 Docker 部署模型 | [Docker：模型教程](../model-tour/index.md) | 按模型选择镜像、设备、权重与启动配置 |
| 在 Kubernetes 管理模型实例 | [Helm 部署](kubernetes/deploy.md) | 集群依赖、模型配置、镜像和存储 |
| 使用 pyMotor 服务框架 | [pyMotor](pyMotor.md) | 对应昇腾运行环境和框架配置 |
| 将 Prefill 与 Decode 分开运行 | [PD 分离部署](../capabilities/pd-disaggregation/index.md) | 请求协调、KV 交接路径和兼容的 P/D 实例 |

先用[引擎快速开始](../quick_start/index.md)确认目标模型与缓存路径，再扩展部署。部署完成后按[运行与排障](../observability/index.md)检查请求入口、外部缓存与后端状态。
