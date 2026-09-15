# PD 分离部署

Prefill 计算提示词的 KV，Decode 在其基础上继续生成 token。拆分两个阶段时，需要让客户端请求依次到达正确的实例，并让 Decode 取得兼容的 KV。本节按 KV 交接方式选择部署步骤。

## 选择交接方式

| 方式 | KV 如何到达 Decode | 适用的操作入口 |
| --- | --- | --- |
| 共享存储交接 | Prefill 保存块，Decode 从同一存储查找和加载 | [共享存储 PD](centralized.md)，使用仓库示例代理建立最小流程 |
| 传输连接器与 UCM 组合 | 传输连接器搬运当前请求的 KV，UCM 为 Prefill 提供外部前缀复用 | [传输连接器组合](distributed.md)，配合 Kubernetes 的请求路由与模型部署 |

共享存储路径要求块在 Decode 查询前可见。传输路径则需要 producer/consumer 连接可用。两种方式的职责与请求顺序见[PD 集成原理](../../../developer-guide/pd-integration.md)。

## 准备与部署

1. 用[引擎快速开始](../../quick_start/index.md)确认目标模型、引擎和存储组合可以运行。
2. 选择一条交接方式，从一个 Prefill 和一个 Decode 实例开始。核对模型、tokenizer、dtype、并行与 KV 布局。
3. 在 Kubernetes 上按 [Helm 部署](../../frameworks/kubernetes/deploy.md)完成集群准备、镜像选择、资源配置和安装。
4. 使用该部署的客户端入口发送请求；Chart PD 部署的入口是 kthena-router 网关。Chart 资源及协议组合见 [Helm 部署的 PD 说明](../../frameworks/kubernetes/deploy.md#pd-resources)。

## 验证后再扩容

先使用未缓存的提示词验证 P/D 请求交接和生成结果，再使用重复前缀验证 UCM 复用。传输成功和缓存复用分别检查，操作方法见[验证外部缓存](../../observability/verify-cache.md)。

单组 P/D 正常后，再按[多节点并行](large-scale-ep.md)调整副本、worker、DP、TP 或 EP。使用相同负载记录每一步变化。
