# 在 pyMotor 中使用 UCM

按 [MindIE-Motor 官方 UCM 接入指南](https://mindie-motor.readthedocs.io/zh-cn/latest/user_guide/features/kv_cache_store/backend/ucm/)准备部署。本页说明 Connector 的组合关系，详细配置以官方指南为准。

## 接入关系

Prefill 用 MultiConnector 组合 Mooncake 传输 Connector 与 UCMConnector：Mooncake 交接当前请求的 P/D KV，UCM 复用跨请求前缀。Decode 使用匹配的 Mooncake Connector。

## 配置要点

- `connectors[0]` 为传输 Connector，`connectors[1]` 为 UCMConnector；不能将其配置为 AscendStoreConnector 的 `backend: ucm`。
- `storage_backends` 对应缓存卷的 `mount_path`；`dshm_size` 为 Cache 主机缓冲预留余量。
- Store 参数见[缓存配置](../../developer-guide/cache-configuration/index.md)和[文件系统流水线](../../developer-guide/cache-configuration/pipeline.md)。

## 部署入口

依次阅读官方的[准备 UCM](https://mindie-motor.readthedocs.io/zh-cn/latest/user_guide/features/kv_cache_store/backend/ucm/#准备-ucm)、[修改配置](https://mindie-motor.readthedocs.io/zh-cn/latest/user_guide/features/kv_cache_store/backend/ucm/#修改-user_configjson)、[部署和删除](https://mindie-motor.readthedocs.io/zh-cn/latest/user_guide/features/kv_cache_store/backend/ucm/#部署和删除)。

## 验证

检查 Prefill 的外部命中、加载完成和错误。`hit hbm` 表示引擎缓存，不能单独说明 UCM 复用；按本站[外部缓存验证](../observability/verify-cache.md)确认结果。
