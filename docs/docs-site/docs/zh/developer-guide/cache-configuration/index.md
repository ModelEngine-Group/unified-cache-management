# 缓存配置

UCM 在引擎的设备 KV Cache 之外保存可复用块。配置时需要确定两件事：引擎如何接入 UCM，以及这些块存到哪里。先按[快速开始](../../user-guide/quick_start/index.md)建立连接，再选择存储后端。

## 选择存储后端 {#storage-backends}

后端决定 KV 块的保存位置、读取路径和外部依赖。先确定缓存是否需要跨进程保留、是否需要跨节点共享，再估算容量和读取成本。增加存储容量能够保留更多前缀，但收益仍取决于负载中的重复前缀和读取延迟。

| 后端 | 适用场景 | 需要准备 |
| --- | --- | --- |
| [文件系统流水线](pipeline.md) | 使用 `Cache|Posix` 组合主机缓存与本地或挂载文件系统 | 可写目录、主机内存和可用的 Pipeline 构建 |
| [NFS Connector](nfs.md) | 沿用 `UcmNfsStore` 的直接文件系统配置 | 本地或 NFS 路径、兼容的等大小张量布局 |
| [3FS](ds3fs.md) | 使用 `Cache|Ds3fs` 访问 3FS | 已运行的 3FS 客户端、匹配的头文件和库 |
| [Mooncake](mooncake.md) | 共享内存池，可组合 Posix 持久化 | 匹配的 Mooncake 服务、客户端和网络配置 |
| [KV 压缩](compress.md) | 通过 `Cache|Compress|Posix` 减少存储载荷 | 支持的编解码器、数据类型及质量评测 |

NFS 挂载可以同时作为 Posix 或直接 NFS Connector 的存储路径。挂载方式和 UCM 后端是两个选择，新部署需要主机缓存、容量回收和健康检查时，从[文件系统流水线](pipeline.md)开始。

## 配置前准备

1. 在[快速开始](../../user-guide/quick_start/index.md)中选择包含目标后端的制品，并确认引擎和设备兼容。
2. 为缓存准备独立的命名空间。共享实例使用一致的模型、tokenizer、KV 布局和执行设置。
3. 估算主机缓冲、设备内存与外部存储容量；共享和非共享主机缓冲的预算方式不同。

## 让配置生效

在 vLLM 路径中，`kv_connector_extra_config.UCM_CONFIG_FILE` 指向 UCM YAML。YAML 根级选项控制 Connector，`ucm_connectors[].ucm_connector_config` 配置所选 Store。文件和存储路径都必须在实际运行引擎的进程或容器中可见。

SGLang 的主机缓存由引擎管理，应采用 [SGLang 快速开始](../../user-guide/quick_start/index.md#sglang)中的存储配置。

只填写所选后端接受的字段，详细归属见[配置参数](../../reference/config-parameters.md)。完成配置后执行[外部缓存验证](../../user-guide/observability/verify-cache.md)，确认复用有效后，再比较读取与重新计算的开销。

缓存块如何匹配、引擎如何等待传输，见[缓存工作原理](../capability-principles.md)。需要把 Prefill 和 Decode 放在不同实例上时，继续阅读 [PD 分离部署](../../user-guide/capabilities/pd-disaggregation/index.md)。
