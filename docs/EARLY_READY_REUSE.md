# Early 已准备结果的消费

本文描述 Early 子类如何复用请求的 block IDs 和多 rank 命中结果。后台
reservation、pin、read lease、READY 汇总与 handoff 继续由已有 Early 组件负责。

## 继承与调用边界

`EarlyDirectConnector` 与 `EarlyLayerWiseConnector` 共用 `_EarlyConnectorMixin`，
分别继承原生 Direct 与 LayerWise Connector。它们调用原生
`get_num_new_matched_tokens()` 完成 token 计算、重算尾部、指标与 `RequestMeta`
写入。原生方法及其所属类不增加扩展函数。

Early 在已有的 `generate_hash()` 与 `EarlyPipelineStore.lookup_on_prefix()`
方法上提供覆盖。Store 是原生工厂已经创建的同一实例；结果消费不创建第二个
Store、RankConsistencyManager、backend、buffer、队列或资源保护协议。

```mermaid
sequenceDiagram
    participant E as Early Connector
    participant N as 原生 get_num_new_matched_tokens
    participant R as 原生 RankConsistencyManager
    participant S as 同一 EarlyPipelineStore
    E->>E: probe 校验请求实例与轻量 identity
    alt PENDING
        E-->>E: 返回 (None, False)
    else READY 或 fallback
        E->>E: 设置调用上下文
        E->>N: super()
        N->>E: generate_hash
        E-->>N: 可信的保存 IDs，或原生 hash
        N->>R: 查询 external prefix
        R->>R: 原生 known-missing 过滤
        R->>S: lookup_on_prefix
        S-->>N: READY 保存命中范围，或原生查询
        N->>N: 原生 token 计算、指标和 metadata
        N-->>E: (external_hit_tokens, False)
        E->>E: finally 恢复调用上下文
    end
```

## 上下文和命中范围

`LookupContext` 只保存本次调用的 Connector、Request、已校验 Job 和 HBM 命中
hash 数。`ContextVar` 让 Connector 与 Store 在原生调用期间看到同一结果，调用
结束或异常时恢复前一值。它不是第二份请求状态表，也不拥有 pin 或 lease。
后台 hash 线程没有这一调用上下文，继续执行原生 hash。

PENDING 在进入原生方法前返回。READY 使用 `job.blocks`，Store 返回的连续命中
数为 `min(max(job.hit - hbm_hit_hashes, 0), len(filtered_prefix))`；沿用原生
Store 的末个命中下标约定，返回该数量减一。这里 `job.hit` 与 `hbm_hit_hashes`
的单位都是 hash block，原生方法继续完成 CP 换算与 token 计算。

原生 known-missing 过滤可进一步缩短命中前缀；即使该记录已过时，也不绕过该
过滤。它仅扫描保存的 block IDs，不重新扫描 tokens、查询底层 Store 或读取
L2。最终结果不会超出多 rank 已保护的范围。READY 还跳过原生 hotness prefetch
入口，避免再次触发存储操作。

FAILED/CANCELLED 使用原生查询；只要 `probe()` 确认请求实例、token view 和
identity 仍可信，hash IDs 可以复用。无 job 或 identity 已变化时使用原生 hash
与查询。请求 ID 相同不足以复用结果。

## Trace

原生 Config 仍给 `enable_record_traces` 赋值。Early 子类属性保存这个值，仅在
本实例的 token 查询上下文内关闭原生内联 trace；EarlyScheduler 沿用每请求
实例的去重集合，使用保存的配置值记录一次。READY 重试不重新构造 hash 文本。
上下文在 `finally` 中恢复，其他 Connector 和后台线程不受这一临时控制影响。

Early 关闭时，已有工厂选择原生 Connector，以上覆盖不参与执行。

## 存储侧继承与组合

`EarlyPipelineStore.__init__()` 通过 `super()` 执行原生初始化。一个内部 builder
注册项调用原生 `Cache|Posix` builder，短生命周期的 `_EarlyStack` 只将 Cache、
Posix 的工厂名称映射到 Early 工厂。配置转换、库路径和 metrics 预加载仍由原生
builder 完成。适配器持有同一个 Pipeline 的引用，不分配第二套存储或线程。
内部注册名仅用于选择 builder，用户配置保持 `Cache|Posix`。

三个 Early 绑定是独立模块函数，使用原生 `PipelineStore::Self()` 获取既有入口，
包含原生 health wrapper。原生 C++ PipelineStore 类不增加方法。绑定同步调用
期间保留 Pipeline 和 buffer 参数的生命周期，释放 GIL 后调用存储；失败使用
原生模块注册的异常类型。释放 reservation 不受 unhealthy 状态阻挡。

`EarlyCacheStore::Setup()` 先调用父类 Setup，再复用同一个私有 ParseConfig，
不维护第二份配置字段和默认值。CacheStore 向这一 Early 子类授予 friend 权限，
原有私有变量和函数仍保持 private；不暴露 BufferManager getter。

`ReservedPrefetch` 通过原生 Handle 的 GetState/FailureStatus 完成可取消等待，
等待策略不进入原生 Handle。需要分配时，TryGet 调用原生私有 Alloc 并提供尝试
次数上限；普通 Get/Prealloc 不提供上限，保留原生等待语义。原生槽位查找、回收、
锁、引用计数和初始化只有一份实现。

`SpaceLayout::StorageBackend()` 保持 private。只有 friend PosixReadLease 调用
这一原生路由，消费者与删除 guard 因而仍使用同一个后端和 block identity。
friend 改动是显式的类级访问授权；它没有将成员改为 public/protected，也不是
原生零修改。删除 guard、实际 unlink 和原生 GC 的周期重试仍沿用既有接入。
