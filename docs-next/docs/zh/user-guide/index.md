# 用户指南

本指南帮助你接入并运行 UCM。第一次使用时，先选安装制品，再按引擎快速开始配置一条缓存路径；运行成功后再选择其他后端或扩展部署。

| 要完成的任务 | 从哪里开始 | 完成后应得到什么 |
| --- | --- | --- |
| 确认环境并安装 | [支持范围](support-matrix/index.md)、[安装](quick_start/index.md) | 与引擎、设备和后端匹配的环境 |
| 运行第一个服务 | [快速开始](quick_start/index.md) | 接入 UCM 的推理服务和一次外部缓存复用 |
| 调整存储方式 | [开发者指南：缓存配置](../developer-guide/cache-configuration/index.md) | 满足持久化或共享需求的后端配置 |
| 部署到集群或拆分 P/D | [部署](frameworks/index.md) | 请求入口、引擎实例与 KV 路径明确的服务 |
| 按具体模型启动 | [模型教程](model-tour/index.md) | 模型参数与 UCM 配置组合成的启动命令 |
| 检查效果或定位故障 | [运行与排障](observability/index.md) | 可解释的缓存、传输和健康状态 |

配置项的完整说明放在[配置参考](../reference/config-parameters.md)。需要理解内部调用或修改代码时，转到[开发者指南](../developer-guide/index.md)。
