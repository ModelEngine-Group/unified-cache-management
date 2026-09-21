# PD 分离部署

PD 分离部署将 Prefill 和 Decode 放在不同的服务实例上运行：Prefill 计算提示词的 KV Cache，Decode 使用这些 KV 生成 token。部署时需要让请求到达正确的实例，并让 Decode 取得兼容的 KV。UCM 提供外部前缀复用；采用共享存储方案时，UCM 还提供 P/D 交接 KV 的存储路径。各组件的职责与请求顺序见 [PD 集成原理](../../../developer-guide/pd-integration.md)。

## 选择交接方式与部署规模

先从共享存储和传输连接器中选择一种 KV 交接方式，再确定部署规模。扩容指南承接传输方案，前提是已有一组正常工作的 P/D 实例。

| 指南 | KV 如何到达 Decode | 部署规模 |
| --- | --- | --- |
| [共享存储 PD 分离部署](centralized.md) | Prefill 通过 UCM 保存块，Decode 从同一存储查找和加载 | 从一个 Prefill、一个 Decode 实例和仓库示例代理开始 |
| [传输连接器与 UCM 组合](distributed.md) | 传输连接器搬运当前请求的 KV，UCM 为 Prefill 提供外部前缀复用 | 通过 Ascend 手工部署或 CUDA/Ascend Helm 部署建立可用的 P/D 实例组 |
| [PD 部署扩容](large-scale-ep.md) | 沿用传输连接器与 UCM 组合的交接方式 | 在可用传输部署上调整副本、worker、DP、TP 或 EP |

共享存储路径要求块在 Decode 查询前可见，传输路径则需要 producer/consumer 连接可用。扩容调整的是实例数量和模型并行方式，仍沿用已选择的交接路径。

## 准备与部署

1. 用[引擎快速开始](../../quick_start/index.md)确认目标模型、引擎和存储组合可以运行。
2. 选择一条交接方式，从一个 Prefill 和一个 Decode 实例开始。核对模型、tokenizer、dtype、并行与 KV 布局。
3. 在 Kubernetes 上按 [Helm 部署](../../frameworks/kubernetes/deploy.md)完成集群准备、镜像选择、资源配置和安装。
4. 使用该部署的客户端入口发送请求；Chart PD 分离部署的入口是 kthena-router 网关。Chart 资源及协议组合见 [Helm 部署的 PD 说明](../../frameworks/kubernetes/deploy.md#pd-resources)。

## 验证后再扩容

先使用未缓存的提示词验证 P/D 请求交接和生成结果，再使用重复前缀验证 UCM 复用。传输成功和缓存复用分别检查，操作方法见[验证外部缓存](../../observability/verify-cache.md)。

采用传输方案的单组 P/D 正常后，继续阅读[PD 部署扩容](large-scale-ep.md)。使用相同负载记录每一步变化。
