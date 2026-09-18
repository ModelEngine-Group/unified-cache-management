# Native I/O lifetime 修复与验证

## 身份

- UCM 分支：`codex/ucm-native-io-lifetime-fix`。
- 起点：本地已核验 `codex/ucm-native-allocation-prefetch` 的 `90ce295b5af76c2ea247e87e3d1ecce50e458316`。
- `17ea304`：AIO 取消不强制清零 latch，销毁等待真实完成，完成回收不阻塞。
- `1cd115e`：Early 已取得的读取责任独立推进；就绪后交原生完整任务；共享完成入口与回归测试。
- `c2b9977`：follower READY 移除采用 swap/pop，避免长列表二次复杂度；未知完成有明确保留保护日志。
- `33e2d0b`：协议设计和额外真实 NPU 冷 L1 无 reservation 用例；协议/测试检查点。
- `d9f3f27`：Early 入队侧序列化，遵守原生 SPSC 队列约束；最终构建检查点。
- 原分支仍为 `90ce295`，无回退、无 push。远端核验遇到原有 credential helper 不可用，未据此推断远端有无后续提交。
- 新工作树：`/data/liyuhang/glm51-vllm-ascend-ucm-minimal-review/src/ucm-native-io-lifetime-fix`。
- 管理仓库是另一个仓库，HEAD `97dfba618fd8d680ac91050972aabbc08ded6166`，本轮不提交管理仓库，保留其原有改动。
- vLLM `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`、Ascend `f4a08bddd0cc65a0bd8c3d377b158ae5ca7527db` 无修改。

## 修改范围与必要性

| 位置 | 本轮实际修改 | 必要性/复用 |
|---|---|---|
| `early_native_transfer.h` | 新增 EarlyNativeTask/EarlyNativeTransfer | 继承现有 TaskWrapper；沿用同一 TransManager、backend、handles；唯一的 Early 读取完成推进线程，与 native H2D FIFO 解耦 |
| `early_read_completion.h` | DrainEarlyRead/CompleteEarlyRead | 批次预取与正式回读共用真实完成、状态发布入口；只由提交方回收一次 |
| `cache_store.cc` 的 EarlyCacheStore | 初始化组合对象；Load/Dump 转交；Check/Wait 薄路由 | 原生 CacheStore 类内容不变；有限分配、handle 复用、原任务尺寸保留 |
| `reserved_batch_impl.h` | Complete 复用完成入口，Check 错误保留在途批次 | 未证实完成不能归还 pin/lease，也不能阻塞 Step 执行其他批次 |
| 原生 `io_engine_aio.h` | ForceComplete 不再 Abort；取消/超时日志；新增析构排空 | 原生会强制完成尚在访问 buffer 的任务，外层包装无法修正其完成证据；必须在事件计数源修复 |
| 原生 `aio_impl.cc` | HarvestCompletions 非阻塞 getevents | 避免 eventfd/取消与满批回收竞争时阻塞唯一完成线程 |
| 测试 | 双进程等待环、失败头/在途尾/其他请求与 dump、真实内核 AIO 延迟及部分提交、修正异步 rename 测试前提 | 行为与资源寿命验证 |

未修改任何私有成员可见性，未改变私有函数访问属性。原生 LoadQueue、DumpQueue、
TransBuffer、TransManager、TaskWrapper 和 Scheduler 不变。新增 executor 没有
另一套 L1/manager/共享身份，没有重做 reserve 或复制原生队列方法。

最终 A3/simu 整套源码与动态库来自 `d9f3f27`；最终文档提交不冒充新测试二进制。
与起点及上游 `5e1427d` 的完整 patch/stat/numstat 位于证据目录。

## 可复现入口

证据根目录：`/data/liyuhang/glm51-vllm-ascend-ucm-minimal-review/evidence/native-io-lifetime-20260918`。
其中 build_fixed.py 是 A3 全量构建/安装，build_simu.py 是独立 simu 构建，
verify_build.py 核对源码 SHA256、实际 import、ELF/动态库路径。
构建目录分别为 `build/native-io-lifetime-a3` 与 `build/native-io-lifetime-simu`；
安装只写入新工作树。离线 vendor 源码与 wrapt 1.17.2 wheel 版本不变。

GLM 的 launch.py 输出完整配置后才启动，sequence.py/exercise.py 执行请求并只停止
带本任务 label 的容器。模型和 tokenizer 只读复用，独立 venv、IPC、L2 和日志。
不启动 Qwen，不清理原服务缓存。原生故障恢复必须整体重建共享域，不能混用新旧库。

相关 C++ 测试过滤器：
`UCEarlyNativeTransfer*:UCCacheEarly*:AbnormalExit/*:UCPosixStoreTest.Aio*:UCAioImplTest.*:*ReadLease*:*EarlyPosix*`。
Python 使用 `test/python_with_build.py --build <absolute simu build> --noconftest`，
运行 test_early_batch_binding、test_early_owner_protocol、test_early_batch_pipeline、
test_early_native_scheduler、test_early_ready_reuse、test_early_owner_budget。
`--noconftest` 绕过不相关的上游 GPU benchmark fixtures，不绕过测试自身断言。

## 已执行证据

- A3 全量原生构建及 import/库路径核对成功；独立 simu 全量构建成功。
- 63 项相关 C++ 测试通过，`cpp-d9f3f27.log`。
- 151 项 Python/绑定测试通过，`python-d9f3f27.log`。
- 容量拒绝后回收、并发 load/dump 推进测试重复 100 次全部通过，
  `spsc-stress-100.log`。不是 100 次真实 NPU 压测。
- 双进程测试使用真实共享内存、原生引用与原生 LoadQueue，SSD/H2D 为模拟。
  两节点、充足容量下交叉 ownership 自然完成，没有手工 MarkReady，没有重复 Wait。
- 延迟 AIO 测试使用真实 kernel io_submit/io_getevents，提交/取消 hook 控制竞争；
  取消失败和部分提交失败后仍等真实事件。它不是实盘永久卡死或设备复位测试。
- 既有真正多进程 read lease/GC owner 删除互斥、最后读者释放、异常退出测试通过。
- 首轮 AioTruncatedLoadReturnsNotFound 暴露测试错误地把异步 rename 当成 Wait 保证；
  调整为等文件实际可见，未改产品持久化语义。初始失败日志保留。

## 边界

此修复证明本地重现的 READY 等待环已消除，不能据此认定用户三次 W32 崩溃已同根因定案。
未跑完整 GLM 层数、TP8×DP2、3379/W32 压力；不宣称 TTFT/吞吐收益或全量可用。

容量拒绝仍可能导致请求失败：已承诺命中后遵循实际 `kv_load_failure_policy=fail`，
不暗改 recompute。未知完成保留保护，可能阻止正常关闭，需要共享域级外部恢复；
不是自动恢复。未验证真实 NPU DMA 故障、进程被杀后的硬件排空/共享引用恢复。
原生 AIO 多 shard dump 的提交/rename 时序未重构，read lease 不证明持久化完整。
原生 OFF 队列自身的失败尾项处理仍是原生边界；本轮 Early SSD 回读不再进入该 FIFO。

保守交接的代价是：缺失回读的一个完整 native load task 必须全部 READY 后才进入
H2D，减少该任务内 SSD/H2D 重叠。已有 Early reserve/read 批次流水、不同任务之间
的推进仍保留；全 READY 任务直接提交。没有改变原生 load/dump 任务尺寸和 PENDING。

## 最终 d9f3f27 的真实 A3 冒烟

官方镜像 digest：
`quay.io/ascend/vllm-ascend@sha256:d03346e4826118aaade06b4425e35dd7a7270f5a54322691e051e93d34897a0e`。
实际本地 tag 为 `v0.23.0rc1-a3`；不以历史报告的 tag 代替此次镜像身份。

GLM-5.1-w4a8 只读复用权重/tokenizer，使用 8 层视图，卡 12–15，
DP/local DP=1/1、TP=4、EP 开，BF16/KV BF16、Ascend W4A8、eager，
max-model-len=32768、max-num-seqs=8、batch tokens=4096、HBM utilization=0.60，
GPU KV blocks override=256。APC/chunked prefill 开，seed=20260725。
UCM direct/no-layerwise，共享 L1=3 GiB，自有 L2 上限=1024 GiB，
AIO/direct I/O，Posix 各并发=1，timeout=300000ms，GDR 关闭，owner 开启。
完整有效配置、环境变量、挂载与命令见每臂 `plan.json`；进程 maps/import/库哈希
来自本工作树的本次 A3 构建，不混用旧工作树二进制。

最终 `checked-off` 与 `checked-on` 各完成 20/20 请求，容器正常退出（code=0）。
覆盖冷 L1/热 L2、HBM 命中、驱逐后加载、重复前缀并发与 miss。
首条冷请求 HBM 命中为 0，external hit=27 blocks/3456 tokens（总长 3540）。
四个 TP 进程核验同一物理共享对象/布局，20 个 generation 各唯一 owner，
有 reserve 的 generation 仅其 owner rank 执行；20 次 owner 释放，结束快照 read lease=0。
证据见 `shared-owner-d9f3f27.json`、`smoke-summary.json` 和
`runtime/native-io-lifetime-20260918/checked-{off,on}/`。

**最终 ON/OFF 的生成 token 仅 13/20 逐请求完全相同。输出等价性仍未解决，
不能将 20/20 HTTP/长度校验成功写成模型正确性全部通过。** 中间版本 OFF 重复实验
也有输出波动（下节），但不能据此排除 Early 影响。减层模型不替代完整模型验证。

最终同一套库另做单卡 12 的真实 SSD→L1→NPU 字节测试：

| 路径 | 用例 | 结果 |
|---|---|---|
| native | 冷 L1 热 L2、热 L1 无 owner handles、dump 后重新加载 | 3/3，mismatched bytes=0 |
| Early | 以上三项，以及冷 L1 无 reservation 的正式回读 | 4/4，mismatched bytes=0 |

每项 27 blocks，布局 `[131072,16384,32768] × 8`，真实 AIO/direct I/O、
真实 H2D/D2H；同时检查落盘 dump 字节。证据为
`runtime/native-io-lifetime-20260918/buffer-final-{native,early}/report.json`、
`buffer-final-{native,early}.log`。此测试区别于双进程模拟 H2D 用例，
不覆盖真实设备故障、W32 容量压力或模型输出数值差异。

`native_get_fallback=0`/`load_l2_read_shards=0` 只描述原生 LoadQueue 的这些任务。
新正式回读的 SSD I/O 在 Early 准备阶段执行，不能把上述计数解释为整条路径没有读盘。
owner 的 prefetched/batch 记录另行证明实际 L2 读取。

## 中间版本模型对照与数值限制

`1cd115e` 的 OFF、ON、OFF-repeat 各 20/20 请求完成，三个容器退出码均为 0。
逐请求 16-token 序列：OFF/ON 13/20 完全相同，OFF/OFF-repeat 15/20，
OFF-repeat/ON 14/20。基线自身存在波动，但不能用它解释全部 Early 差异。
这是减层冒烟，不是 GLM 原模型的语义/数值等价认证；完整模型历史差异仍未关闭。
部分模型退出日志含原生 multiprocessing resource_tracker 的 semaphore/shared-memory
清理告警；未据此宣称全部 IPC 异常清理场景均已验证。每次运行使用私有 IPC，
没有清理其他容器或宿主机缓存。正常请求结束快照中 read lease 为 0。

`1cd115e` 字节级 native/early NPU 检查均成功，各项 mismatched bytes=0。
最终 `d9f3f27` 额外覆盖冷 L1、不带 reservation 的正式回读，见上节。
只验证此布局和正常传输，不证明真实 DMA 故障、设备失联或生产 W32 压力安全。

## 本轮复测发现并修复的竞争

`33e2d0b` 的 C++ 全集复测在 CapacityPressureRejectsUnsubmittedDumpAndLoadThenRecovers
超时：4 个并发 ready load 只出现 3 个 native dispatch，丢失的任务一直未完成。
原生 LoadQueue/DumpQueue 入口是 SpscRingQueue；新增 READY 调用方与后台准备线程
形成多生产者。`d9f3f27` 仅在 Early 的非阻塞 native Submit 外加每 store mutex，
不锁磁盘/H2D 等待，不改原生队列。修复后 63 项重新通过，该容量/并发用例重复
100 次全部通过，见 `spsc-stress-100.log`。原失败日志 `cpp-33e2d0b.log` 保留，
不把这次本地竞争当作用户旧版三次 W32 事故的实机根因证据。

## 离线恢复与部署

源码 bundle（包含可达上游和本地提交历史，不含权重/编译产物）：
`/data/liyuhang/glm51-vllm-ascend-ucm-minimal-review/recovery/ucm-native-io-lifetime-fix.bundle`。
验证及恢复示例，在新的空路径中执行：

```bash
git -C /absolute/parent clone -b codex/ucm-native-io-lifetime-fix /absolute/recovery/ucm-native-io-lifetime-fix.bundle /absolute/new-ucm
git -C /absolute/new-ucm bundle verify /absolute/recovery/ucm-native-io-lifetime-fix.bundle
```

构建/测试脚本与精简 runtime 证据归档为同目录下
`ucm-native-io-lifetime-fix-evidence.tar.gz`。其中 `delivery.json` 记录最终文档 HEAD、
测试实现 HEAD、版本差异和 bundle SHA256。不打包 L2、模型权重、凭据或动态库。
依赖源码使用已有 `owner-protocol-vendor-sources.tar.gz` 及锁定 wrapt wheel，
在目标 A3 使用上述官方镜像全量重建，不复用 bundle 外的旧 UCM `.so`。

部署前先停止本任务所属的所有共享域消费者；安装整套匹配库、核对 import/maps，
再重建该共享域。不可单独替换 libcachestore.so，也不可由一个 worker 清理仍被
其他消费者使用的对象。构建与实机脚本中的根路径需要按恢复位置统一调整。
最终源码提交的变更仅文档，测试二进制仍明确标识 `d9f3f27`。
