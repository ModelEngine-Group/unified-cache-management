# PD 集成原理

PD 分离部署把一次推理请求分为 Prefill 和 Decode 两段。Prefill 生成 Decode 所需的 KV，因此分离部署既需要请求协调，也需要数据交接。UCM 的前缀复用可以减少 Prefill 的重复计算；两阶段之间如何交接，则由所选部署决定。

## 共享存储交接

仓库的 `ucm/pd/toy_proxy_server.py` 提供示例请求协调：先将提示词发送到 Prefill，再交给 Decode。Prefill 经 UCM 保存可复用的块，Decode 经自己的 UCM Connector 查找并加载共享存储中的块。

这里跨实例共享的是存储命名空间。两个实例必须使用兼容的模型、tokenizer、缓存标识和张量布局，而且保存结果在 Decode 查询时已经可见。写入尚未完成或块布局不匹配，都会影响 Decode 的外部命中。示例代理的请求顺序不能替代实际写入完成与读取验证。

操作从[共享存储 PD 分离部署](../user-guide/capabilities/pd-disaggregation/centralized.md)开始。该示例不承担生产路由器的调度、容错或透明请求迁移职责。

## 独立传输连接器与 UCM 组合

当前内置 Kubernetes PD 配置把 UCM 与 Mooncake producer 组合在 Prefill 一侧。Prefill 先复用可以找到的外部前缀，计算其余部分，再由传输连接器将当前请求的 KV 交给 Decode consumer。Decode 在这条组合路径中不通过 UCM 完成交接。

| 参与方 | 负责的信息和动作 |
| --- | --- |
| Router | 选择 P/D 请求端点，按照传输协议协调请求 |
| Prefill 的 UCM Connector | 查询、加载和保存跨请求复用的外部前缀 |
| 传输 producer / consumer | 标识对端并交接当前请求的 KV 数据 |
| 推理引擎 | 执行 Prefill/Decode，管理设备缓存和模型并行 |

冷请求没有 UCM 命中时，仍会执行 Prefill 和 P/D 传输。重复请求可以减少 Prefill 计算，但传输路径仍然存在。因此应分别测量外部复用、P/D 传输和客户端延迟。

## 配置如何到达执行进程

Chart 用 `ModelServing` 描述角色和 worker，用 `ModelServer` 声明路由协议，再用 `ModelRoute` 映射客户端模型名。具体资源与支持组合集中在 [Helm 部署](../user-guide/frameworks/kubernetes/deploy.md#pd-resources)。

`files/resolve-kv-transfer-config.py` 在 Pod 启动时将逻辑实例标识解析成传输配置。它处理连接器初始化所需的标识，不执行 Router 调度，也不等同于逐请求注册。检查 P/D 问题时，先区分初始化标识、请求路由和实际数据传输。

## 并行布局与完成条件

副本数描述独立的服务实例，worker 数描述一个实例内部的执行进程。DP、TP 和 EP 由引擎执行配置控制；增加 Pod 数不会自动生成兼容的缓存布局。改变 P/D 并行度时，需要验证传输连接器支持的布局转换和资源预算。

共享存储路径检查 Prefill 保存、Decode 匹配和加载；传输路径检查 producer/consumer 交接，再单独检查 Prefill 的 UCM 复用。两者都需要生成结果与请求连续性的验证。部署步骤见[PD 分离部署](../user-guide/capabilities/pd-disaggregation/index.md)，缓存操作的完成语义见[请求生命周期](request-lifecycle.md)。
