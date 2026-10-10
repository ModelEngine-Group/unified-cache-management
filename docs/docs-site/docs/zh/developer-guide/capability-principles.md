# 缓存工作原理

相同提示词前缀在相同执行条件下能够复用先前计算的 KV。UCM 将这一过程拆成块匹配、数据加载和新块保存，让引擎根据可用数据减少 Prefill 计算。

## 从 token 前缀到缓存块

在常规 vLLM 路径中，Connector 将 token 序列按缓存粒度生成链式块标识。块标识包含前面的前缀关系；相同文本片段出现在不同上下文中，不会因此得到可互换的 KV。只有完整块参与这条持久化路径，末尾不足一块的部分仍需由引擎处理。

Connector 同时把模型及执行设置引入标识与布局管理，例如模型目录名、并行配置、dtype 和 rank。目录名相同不能代替权重及 tokenizer 版本管理：共享实例需要使用兼容的权重、tokenizer、块大小、KV 布局和执行设置，运维侧应为不兼容配置隔离存储命名空间。

## 查找决定计算预算，加载提供数据

Scheduler 已经知道引擎内存缓存覆盖的 token 数。Connector 查询其后连续存在的外部块，将匹配结果转换为可复用 token 数。`lookup_on_prefix()` 返回最后一个连续命中块的索引，`-1` 表示从首块开始就未命中；这不是命中块数量。

查找时存在的块，在 Worker 加载时可能已经缺失或读取失败。因此 Scheduler 报告命中后，Worker 仍需等待传输并反馈错误。当前实现还可能保留少量 token 重计算，以满足模型执行和 hook 的要求，外部匹配数量与最终跳过的计算量应分别理解。

对于 hybrid 布局，集成可能使用 `lookup_on_reverse()` 查找最右侧存在的状态块。应沿对应模型 Connector 阅读这一分支，不将连续前缀语义套用到所有布局。

## 分层缓存如何搬运 KV

`Cache|Posix` 的加载路径从已就绪的主机缓冲读取，或由 Posix 将文件数据填入主机缓冲，再传输到设备。保存路径先等待设备计算满足同步条件，再将设备数据转入主机缓冲并提交文件写入。

Cache 阶段隐藏缓冲管理，Posix 阶段隐藏文件 I/O；Connector 提供布局和设备地址。按层传输进一步让下一层的加载与当前层计算重叠，前提是每层使用数据前完成等待。具体顺序见[请求生命周期](request-lifecycle.md)。

主机缓冲容量有共享与每个 worker 独立分配两种口径。存储容量回收由后端管理，配置值只是逻辑预算，不会预留文件系统空间。部署设置见[文件系统流水线](cache-configuration/pipeline.md)。

## 什么时候值得读取缓存

外部复用把一部分 Prefill 计算替换为查询、读取、传输和同步。只有减少的计算时间超过这些开销时，端到端时延才可能改善。前缀重复比例、块粒度、存储带宽、主机缓存状态以及服务排队都会影响结果。

先按[外部缓存验证](../user-guide/observability/verify-cache.md)确认数据被复用，再在相同输入和并发下比较 TTFT、吞吐与正确性。[指标参考](../user-guide/observability/metrics-reference.md)解释查询 token、传输 shard 和字节计数的不同口径。

## 源码阅读入口

- `ucm/integration/vllm/ucm_connector.py`：块标识、调度匹配和 Worker 传输。
- `ucm/store/ucmstore_v1.py`：查询与任务接口的定义。
- `ucm/store/pipeline/connector.py`：阶段组合与 Python/原生边界。
- `ucm/store/cache/cc/load_queue.cc`、`ucm/store/posix/`：主机缓冲等待与后端 I/O。
