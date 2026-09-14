# Reserve 索引复用与计时

## 本轮身份

- 起点为 UCM `de79bf24502a1c7a4c2f9e4b0622c5ba7fa4b3cb`。
- 原工作树及 `codex/ucm-prefetch-handle-reuse` 正被另一个 GLM 验证任务使用，保持只读。
- 本轮工作树：`/data/liyuhang/glm51-vllm-ascend-ucm-minimal-review/src/ucm-reserve-batch`。
- 本轮 UCM 分支：`codex/ucm-reserve-batch-reuse`，不推送。
- 原生比较点：`5e1427d6b89f69a2bff518dc59fbf5b6a52bc3c1`。
- 管理仓库为 `codex/ucm-minimal-review`；本轮没有管理仓库提交，也没有覆盖其他任务的未提交改动。
- 新构建容器 `glm51-reserve-batch-build`，label `ucm.task=reserve-batch-20260914`；没有映射 NPU 设备。
- 官方镜像 digest：`quay.io/ascend/vllm-ascend@sha256:d03346e4826118aaade06b4425e35dd7a7270f5a54322691e051e93d34897a0e`。
- 依赖沿用管理仓库 `dependencies.lock.json`，独立 vendor checkout、simu/A3 build、venv；没有升级依赖。

## 修改清单

| 位置 | 本轮新增或修改 | 复用与边界 |
|---|---|---|
| EarlyStats / Phase | 每次 reserve、prefetch、load_handles 的计时和计数，失败也记录 | 使用原生 StopWatch 与日志；无逐 shard 时钟调用或日志 |
| EarlyWorker._reserve | lookup executor 排队时间、请求 key/rank 与 C++ reservation ID 关联 | 线程数、HRRN、队列、PENDING 与 READY 语义不变 |
| EarlyBufferPins.PinNode | 将原有 Early PinReady 的必要保护逻辑收敛到批量查询内部 | 使用同一 native strategy、节点/桶锁、原生 Handle；删除不再用于生产的单 shard 查询包装 |
| EarlyBufferPins.PinBlocks | 按原生 Hash 对本批查询分组，每个 bucket 只锁/遍历一次 | 精确检查每个需要查询的 shard，保留活跃 LOADING 借用、容量限制和 READY 判断 |
| EarlyBufferPins.Touch | 对已保护位置调用原生 MarkAccessed | 保留 CLOCK 热度，省略共享索引查询与节点锁；仍有逐 shard 原子标记 |
| EarlyBufferPins.Locations / Recall / HintBlockCapacity | 已释放保护只保留节点序号；再次使用时原生节点锁内验证身份/状态并重新取得引用 | 不锁 bucket、不信任裸位置；部分失败用原生 Handle RAII 释放，转批量查询 |
| ReservedPrefetch.BlockPins / ReadyHint / readyBlocks_ / PublishReady | 共享完整 READY block 的保护组，按原生容量约束提示元数据 | 弱保护组加有界位置提示；没有闲置 pin 池、跨进程新协议或新的共享身份 |
| ReservedPrefetch.Reserve | 候选段 32→64→128→256；先复用有效保护，再批量查询其余 block | 长短命中、L1-only 前缀、剩余工作量均保持精确；不提前启动该 job 的数据 I/O |
| ReservedPrefetch.LoadPlan / LoadEntry / LoadHandles | 候选集合不变时复用定向索引，每次 load 只解析实际请求的 block | 删除每个 reservation 的 blockOffsets 和候选×任务遍历；节点 Handle 复制的锁仍保留 |
| ReservedPrefetch.Release | 使 load plan 失效，保留弱引用与有界节点提示 | 不延长 pin/lease 生命周期；后台预取仍先 drain 已接受 I/O |
| LoadQueue.DispatchOneTask | 记录实际进入 Get 的回退次数 | 原生 Get、Prealloc、H2D 与完成清理流程不变 |

相对 de79bf2，对原生代码的变化只有：TransBuffer/Handle 增加各一条 friend，
移除原先添加的 PinReady 声明与方法，trans_buffer.cc 包含 Early 辅助实现，以及
LoadQueue.DispatchOneTask 的诊断计数。没有更改 private 变量或函数的访问属性，
没有向原生类新增函数。friend 是访问授权变化，不能称为原生零修改。

辅助实现包含在定义私有 BufferStrategy 的编译单元中，以复用原生 Hash、布局、
节点引用与锁；没有把原生策略抽出重写，没有再创建 manager/layout/队列。
该 helper 不是通过继承替换 buffer：必须继续使用原生已经创建的同一个实例。

原生 `ucm/store/dram/cc/drampool/lru_eviction_policy.h` 的 LRU 操作 DramPool Entry，
并通过 TryMarkEvicting 改变存储项状态，不适用于可丢弃的位置提示。因此这里使用
局部 list/map 记录提示访问次序，容量来自同一 buffer 的实际节点数（沿用 de79
已有的 NodeCount 扩展，该方法本身不是上游提供的接口），
没有新增配置、H/R 上限或 KV 淘汰协议。提示分配失败只清空提示，不使已注册的
reservation 丢失其对外 handle。引用仍按 reservation/load 的原生 RAII 生命周期释放。

本轮没有修改 vLLM/vLLM-Ascend、原生 Posix/SpaceLayout/GC/lease、工厂、配置解析、
producer 行为或共享身份。ReadLease 不证明数据已经完整写盘，历史 AIO/GLM 问题仍未关闭。

## 两份差异的范围

相对本轮起点 `de79bf2`：三个新增 Early C++ helper 文件（`early_stats.h`、
`early_buffer_pins.h`、`early_buffer_pins_impl.h`）；`reserved_prefetch.h` 的批量查询、
保护组/位置提示、LoadPlan；Python `_reserve` 的 6 行计时；三个原生文件中的
PinReady 移除/friend/编译接入与 Get 回退计数；两个 C++ 行为测试文件、两个 CPU
探针文件、设计与本操作文档。完整 patch 和 numstat 随交付证据保存。

相对原生 `5e1427d`，还包括此前已经存在的 Early Python connector/control/scheduler/
worker、pipeline 绑定和 Store 子类、reservation/read lease，以及下列原生文件改动。
下表列举全部仍修改的原生源码文件；这些历史扩展不是本轮新增复用。

| 原生位置 | 保留的改动与必要性 |
|---|---|
| integration/vllm/ucm_connector.py | `UCMConnector.__init__` 选择 Early 薄子类；新增 arrival/shutdown 转发；匹配入口允许 PENDING 的 None 透传，避免进入数字统计；原生 direct/layerwise 的 hash/metadata/load 主体不改 |
| cache/cc/cache_store.cc | CacheStore friend；同编译单元的 EarlyCacheStore/factory 调用父类 Setup/ParseConfig/Load，组合同一 buffer/trans manager；句柄完整交接时向原生 trans manager 提交 |
| cache/cc/trans_buffer.h | 之前的 TryGet 和私有 Alloc 的可选尝试次数；本轮的两条 friend，原变量/函数 private 属性保持 |
| cache/cc/trans_buffer.cc | 之前的 BufferStrategy.NodeCount/CanReserve 及两个策略的容量返回、TryGet、Alloc 有界尝试；防止 Early 占用 load-exclusive 区或在满 buffer 无限等待；本轮只移除 PinReady 并包含 Early helper |
| cache/cc/load_queue.cc | DispatchOneTask 接收任务 Handle，缺失才 Get；已经完整 reservation 保护的 block 不再 Prealloc；本轮增加诊断计数；正式 H2D/完成队列保持原生 |
| cache/cc/trans_task.h | loadHandles 随原生 TransTask 移动，保证排队和传输期间引用存活 |
| detail/type/types.h | TaskDesc.loadReservations 传递本地 reservation 授权，不传裸指针 |
| ucmstore_v1.h | ReadLease、PrefixReservation、PrefetchResult 及可选操作默认错误/no-op；由原生 store/pipeline 持有和转发扩展 |
| pipeline/cc/health_breaker_store.h | 四个可选操作转发；健康失败时拒绝获取，但 release 始终下传以清理资源 |
| pipeline/cpy/pipeline_store.py.cc | friend 授权 Early::LoadBinding 复用私有 MakeTaskDesc，注册独立 Early binding，避免复制原生参数解析 |
| posix/cc/global_config.h | Config 的可选删除 guard 回调，不新增一套配置解析 |
| posix/cc/posix_store.cc | Setup 调用 Configure；protected Configure/GetLayout 供 EarlyPosixStore 使用原有 layout，继承原生 I/O；这些扩展在本轮之前已存在 |
| posix/cc/space_layout.h | PosixReadLease friend；guard/延迟删除状态和 TakeDeferredDeletion，保留原字段 private |
| posix/cc/space_layout.cc | Setup 保存 guard；RemoveFile 在同一锁保护期内执行原生删除，保证跨进程实际 GC owner 与消费者互斥 |
| posix/cc/shard_gc.cc | RunGcCycle 遇保护后回到原生等待间隔，避免忙重试 |

原生初始实现没有 reservation Handle 交接和跨进程消费者读保护。因此仍需要这些
扩展或等价能力；不能以删除保护来缩小 diff，也不能称整条分支原生零修改。本轮
没有给原生类增加方法，没有将 private 改成 protected/public。上表列出的历史
新增方法仍存在，未在本次索引优化中扩大重构范围。

## 计时含义

日志使用原生 unlimited 信息级接口，避免默认每调用点限流导致计数丢失；仍受
原生日志级别与输出设置控制。每个 C++ 操作一条，不输出 hash 全文或逐 shard 日志。

| 字段 | 含义 |
|---|---|
| lookup_start.queue_ms | job 进入 worker 到 lookup executor 开始运行，包括调度等待 |
| lookup_done.elapsed_ms | worker hash/ReservePrefix 整体调用时间，通过 key、rank、reservation 关联 |
| reserve.lease_ms | Acquire/Extend，包括锁操作及其等待 |
| reserve.pin_ms | 保护组/提示解析、批量共享索引扫描、引用保护与本地准备时间 |
| reserve.l2_lookup_ms / l2_calls | 同步原生 LookupOnPrefix 的累计墙钟时间与调用次数 |
| reserve.shards / bucket_locks / reused_blocks | 实际索引查询 shard 数、实际 bucket 加锁次数、直接复用保护的 block 数 |
| reserve.recalled_blocks / hint_checks / hint_entries | 从已释放位置重新获取的完整 READY block 数、实际节点核验次数、发布后的提示条目数 |
| prefetch.pin_ms | 总时间减去提交和完成等待，包含索引定位、任务准备与收尾，不是纯 hash CPU 时间 |
| prefetch.submit_ms / wait_ms | 原生 Load 提交与 Wait/共享 shard READY 等待；Wait 时间包括完成处理 |
| prefetch.first_io_ms | 从进入 PrefetchReserved 到首次尝试提交 I/O，不包含此前 reserve 排队时间 |
| load_handles.total_ms / index_builds / block_lookups | 交接整体时间、计划构建次数、实际 block 解析次数 |
| load_dispatch.native_get_fallback | 完成派发的任务中，实际调用原生 Get 的次数；派发中途失败见原生错误日志 |

不能把 fast_path 的端到端延迟当作 PinReady CPU 时间。必须分别关联 executor 排队、
lease、pin、L2 lookup 和 prefetch 等待。零命中 fast_path 也必须与正长度全 L1 命中分开。

## 复现与证据

证据目录：`/data/liyuhang/glm51-vllm-ascend-ucm-minimal-review/evidence/reserve-batch-20260914`。

- `instrumented-start-build.log` / `instrumented-start-ctest.log`：带计时的旧算法，57 项通过。
- `index-before.log`：旧算法 CPU 索引探针。
- `batch-final-build.log` / `batch-ctest.log`：批量 reserve 步骤，60 项通过。
- `index-batch.log`：只完成批量 reserve、尚未改 LoadHandles 的中间探针。
- `hint-simu-build.log` / `hint-ctest.log`：包含释放后位置核验的实现，66 项 C++ 测试通过。
- `hint-python.log`：42 项 Python 行为测试通过；`hint-a3-python.log` 在 A3 构建扩展下同样 42 项通过，未使用 NPU。
- `worker-control-regression.log`：9 项已有 worker/control 回归通过，含真实 ZMQ 多进程；不包含完整历史 scheduler fixture 套件。
- `hint-a3-build.log` / `.manifest.json`：`0072235` 原生 A3 编译；`hint-a3-library-environment.json` 记录实际加载的新路径和库哈希。
- `delivery-simu-build.log`、`delivery-ctest.log`、`delivery-a3-build.log`：删除未使用单 shard 包装后的最终重建与回归；最终状态以交付 manifest 为准。
- 最终编译代码 `2fc0c3884ffa63f9f24ac2b3bc88fbdd0e7077b6`：66 项 C++ 通过，A3 构建成功，`delivery-a3-python.log` 的 42 项通过；`delivery-a3-library-environment.json` 核对 1188 个源码文件及实际加载路径。
- `delivery-static-checks.json`：Python AST、git diff whitespace、框架无 diff、探针操作次数断言通过。锁定容器没有 black/isort/clang-format，未安装或运行这些格式工具；不把编译成功当成这些工具已经通过。

C++ 测试使用原生共享/私有 buffer、引用计数、线程和 LoadQueue，L2/HBM 在 cache
测试中是替身。`SharedShardHasOneReaderAndTwoLocalHandoffs` 使用两个真实进程，
各自创建/挂接原生共享内存、合并读取、重新核验释放后的提示并模拟传输。
Posix 测试使用真实 Linux OFD 锁、子进程和实际 GC 删除；覆盖多读者、最后读者、
进程退出、部分获取失败和保护解除后的回收。它们不能替代 NPU 模型运行。

### 同负载 CPU 探针

`hint-index-comparison.json` 使用同一个最终探针源文件，分别编译链接独立的
`9c53dbc`（de79 算法加计时）和新版 simu 库。两个版本交替运行 5 次，均设置
`UCM_LOG_LEVEL=error`；初次 reserve 每类 5 个样本，其余各 20 个样本。

| 场景 | 私有 L1 旧→新中位数 ms | 共享 L1 旧→新中位数 ms |
|---|---:|---:|
| 首次冷 L2 全前缀 | 3.301→3.345 | 2.000→1.848 |
| 首次完整 L1 命中，无已有提示 | 4.984→4.645 | 4.557→4.708 |
| 有存活 reservation 的完整 L1 命中 | 4.971→0.112 | 4.685→0.113 |
| 所有旧 reservation 已释放后的完整 L1 命中 | 5.125→2.158 | 4.942→2.111 |
| 16 个候选 reservation 的 LoadHandles | 7.088→2.158 | 7.017→2.146 |

首次扫描没有稳定的大幅改善，私有冷查询与共享首次热查询甚至略慢；不隐去这些
结果。主要收益来自重复请求避开共享 bucket，以及交接计划减少候选遍历。
`hint-index-operation-counts.log` 单独开启 info，验证冷查询仍有 39,936 shard
查询，但 bucket 锁降为 28,170，L2 调用由 16 降为 5；完整受保护的 512 block
复用没有 bucket/node 重新获取；释放后提示命中没有 bucket 查询，但仍执行
39,936 次节点核验；LoadHandles 的 39,936 shard 只解析 512 个实际 block。
同一候选集合的 4 次加载只构建一次 LoadPlan。

探针使用原生私有/共享 buffer、真实节点/桶锁；512 block × 78 shard，每 shard
仅 64 字节，L2 lookup/lease 是测试替身，没有 SSD I/O、HBM 拷贝或服务端排队。
它验证操作次数和局部 CPU 行为，不能复现用户服务器的 72 秒或证明 TTFT 改善。

### 真实模型运行边界

本轮准备了完整 GLM 78 层和 Qwen3-0.6B 的原有 DP2/localDP2/TP8 配置，独立
service venv、任务 L1/IPC/L2 和入口；没有沿用其他任务的 8 层 GLM smoke 作为
完整模型成功证据。第一次 GLM off 启动被设备守卫拒绝，没有创建模型容器。
之后检查中曾短暂空闲，随后明确看到另一个 VLLMWorker_TP 任务占用 4 个设备
（`glm-off-pre-retry-npu.txt`）。没有停该服务或改成更小拓扑。
第二次在完整空闲窗口启动了任务容器 `ucm-reserve-batch-glm-off-01`，源码与库均为
`2fc0c38`，实际 import 已记录。当前仍加载完整 78 层权重，尚无请求结果；加载
耗时不属于 Early reserve 阶段。GLM/Qwen 本轮完整 NPU 回归、冷 L1/L2、全 L1、
多 DP 输出及故障清理仍未验证。后续结果记录在独立 run 目录，不覆盖历史失败证据。

构建入口为本仓库 `test/build_prefetch_reuse.py`。在本轮构建容器中使用绝对路径：

```sh
docker exec --workdir /data/liyuhang/glm51-vllm-ascend-ucm-minimal-review/src/ucm-reserve-batch \
  glm51-reserve-batch-build python3 \
  /data/liyuhang/glm51-vllm-ascend-ucm-minimal-review/src/ucm-reserve-batch/test/build_prefetch_reuse.py \
  --build /data/liyuhang/glm51-vllm-ascend-ucm-minimal-review/build/reserve-batch-a3 \
  --runtime ascend-a3 --install \
  --log /data/liyuhang/glm51-vllm-ascend-ucm-minimal-review/evidence/reserve-batch-20260914/a3-rebuild.log
```

simu 使用独立 `reserve-batch-simu` 目录、`--runtime simu`，不传 install。
运行 ctest 时筛选 `UCCache|EarlyPosix|ReadLease`。Python 使用独立 venv 的 python，
执行 `test/python_with_build.py --build <绝对 build 路径>`，pytest 参数为
`--noconftest -o addopts= -p no:cacheprovider -s -v <test_early_ready_reuse.py 绝对路径>`。
索引探针使用 `test/run_index_probe.py --build <simu build> --output <独立可执行文件>`。
比较旧算法时，应从诊断提交恢复源码并重新构建，不能让旧探针加载新版共享库。

## 尚未解决或验证的范围

- 首次冷前缀仍逐 shard 判断真实状态；批量化减少桶锁与重复遍历，不将完整检查变成 O(block) 查询。提示命中仍需要节点核验与引用锁。
- 完整 READY 保护只在同一 worker 的存活 reservation 之间直接共享；上一请求释放后可通过原生节点锁重新核验位置。没有跨 rank 位置交换协议，各 rank 首次查询仍存在。
- 同一 job 的 reserve 仍在 prefetch 之前完成；PENDING、HRRN 和单数据线程保持，未改为软依赖。
- LoadHandles 仍须复制原生 Handle 引用并拿节点锁；本轮消除候选×任务遍历，不宣称零锁。
- 32/64/128/256 是有界增长策略，没有在用户真实负载上选优；短命中后的推测范围在后续大批中会增大。
- 用户的实验日志在不可访问的服务器，TTFT 根因和改善仍需用新计时复核；历史 GLM 异常不能凭本轮局部测试关闭。
