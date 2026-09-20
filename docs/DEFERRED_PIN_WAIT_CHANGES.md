# 相对 c50 的修改清单

本轮代码基于精确 `c50dfab7ce560b20ce1ebdabf9a7a0cf0e3a9539`；
实现与测试代码止于 `1bf37bb`，后续提交只补充交付文档。
完整机器可读 diff 在操作证据目录的 `implementation.diff`。

| 位置/新增能力 | 本轮修改和必要性 | 继续复用 |
|---|---|---|
| `early_capacity.py` 的 Claim / ExecutionWindow | 记录已选中批次的唯一block声明，检查是否存在安全完成顺序；防止部分占用等待环 | 原生节点容量、原生实际分配；不建第二套allocator |
| `early_owner.py` | 协议5；execution_request/grant、lookup_complete、resource_wait；退休排空后释放声明；容量等待不被普通PENDING时限误判 | 同一broker、握手/物理域、单owner轮询、消费者/quiesced/drained协议 |
| `early_control.py` | 自定义BrokerState提取可覆写的pending_expired；非owner判断不变 | 原有broker超时/peer心跳和退出流程 |
| `early_owner_worker.py` | 使用deferred查询；接收带attempt/batch范围的资格许可；等待激活结束才能drained | 原有owner消费、层/chunk、失败隔离和metadata交接 |
| `early_prefetch.py` | WorkerJob/Batch增加查询续点、激活和容量等待状态；HRRN选中后激活；暂停后续查询/批次；共享释放事件唤醒 | HRRN公式不变，既有批次窗口、请求交错、数据线程及完成回收 |
| Early::Batch / ReservedPrefetch::Batch | Activate重新取得引用；Defer回滚未提交的迟到激活；Limit/Cutoff只处理异常范围；Step保存NoSpace续点 | Locate、节点提示锁内核验、ProbeFlight、BlockPins、TryGet、CompleteEarlyRead、LoadPlan/handle复用 |
| ReservedPrefetch::Reserve | deferred模式查询完释放临时L1引用，保存identity/L2连续命中/lease；旧调用默认不变 | 原生LookupOnPrefix与已有reservation/block身份，不新增producer等待 |
| Early::DomainProvider / EarlyCacheStore | 增加ReserveDeferred、ReleaseEpoch/WaitRelease可选能力 | 同一个BufferManager、TransBuffer、backend和所有原生I/O队列 |
| EarlyBufferPins | 转接同一buffer的释放epoch和futex等待 | 原有锁和有限分配、相同shard身份 |
| `buffer_release_event.h` | 共享epoch和futex；比较epoch后等待，解决释放先到竞争 | 复用原生L1映射，不建另一个共享对象/资源协议 |
| `early_bindings.h` / EarlyPipelineStore | 暴露上述Early可选能力、batch生命周期/状态；等待和C++操作释放GIL | 原生PipelineStore对象、health wrapper、Status/Expected、现有CPython模块 |
| 测试 | 6个C++用例、12个Python容量/并发用例、deferred真实binding参数化 | 原有模拟backend/原生队列/共享内存测试设施 |

## 对原生的侵入边界

- `trans_buffer.cc` 是本轮唯一修改原生实现职责的文件：共享header现有padding内加入
  release event；初始化/挂接指向同一event；`TransBuffer::Release` 在最后引用归零且
  释放节点锁后通知。共享header magic由2变3，旧共享映射不得混用。
- `BufferStrategy` 新增event成员/指针及friend声明；**没有改变任何既有私有变量或
  私有方法的访问属性，没有给原生TransBuffer增加方法**。通知放在原生最终引用释放
  点，才能覆盖跨进程、非owner及原生load/dump的实际释放；只在Early取消路径通知不充分。
- `cache_store.cc` 虽是原生文件，本轮10行新增均在已有 EarlyCacheStore 子类中，
  未改 CacheStore 初始化/Setup/ParseConfig/Load/Dump/Wait 方法。
- 其余运行时改动位于既有Early文件或新增Early文件。未改原生load_queue、dump_queue、
  AIO实现、Scheduler、vLLM/vLLM-Ascend源码、LMCache及kv_load_failure_policy。

## 保守边界

这不是全局串行。互不破坏安全完成顺序的批次可并行/交错；共享block按并集计算。
资格判定可能改变当前可供HRRN选择的集合，不能称为调度顺序完全相同。
未知超容量候选先允许无pin查询查完；确认真实超容量才截断，不恢复候选全集预占。
有限分配仍是物理裁决。已经承诺命中后的native失败继续如实上报原失败策略，
不承诺任何配置/外部故障下请求都一定成功。

仅验证通过共享域握手的MLA owner路径启用本轮等待；private/非owner保留c50行为。
非MLA路径没有本轮NPU验证，不借此次GLM测试宣称Qwen通过。
