# Prefetch handle 复用：改动与验证

## 身份与边界

- UCM 派生分支：`codex/ucm-prefetch-handle-reuse`。
- 工作树：`/data/liyuhang/glm51-vllm-ascend-ucm-minimal-review/src/ucm-prefetch-reuse`。
- 起点：`codex/ucm-minimal-subclass-reuse` 的 `2d06330206b5fc3758417577eaa70b7f732ddbb4`，保留不动。
- 指定原生比较点：`5e1427d6b89f69a2bff518dc59fbf5b6a52bc3c1`。
- 本轮功能提交：`2fa09b8`、`638e647`、`d50fce16429563280e552e21b8a0cd804805bcdc`。
- 管理仓库仍为 `codex/ucm-minimal-review` / `3df45f3c818d7724b38e39ad44fc452e1af5cb31`；本轮没有管理仓库提交。
- 不修改原 GLM/Qwen 分支、旧工作区或旧服务，不推送。

官方镜像为
`quay.io/ascend/vllm-ascend@sha256:d03346e4826118aaade06b4425e35dd7a7270f5a54322691e051e93d34897a0e`，
本地 image ID 为 `sha256:611962961cde7e475e75b5391957e8afbcba6b9b0bcd4be9418fe17842de02ca`。
新建 `glm51-prefetch-reuse-build` 容器，仅编译与离线验证；没有给它映射 NPU 设备。
源码、simu build、A3 build、vendor checkout 和 Python venv 分别隔离。

构建容器内实际源码为 vLLM `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`、
vLLM-Ascend `f4a08bddd0cc65a0bd8c3d377b158ae5ca7527db`，两者 tracked diff
均为空；见证据目录的 `build-environment.json`。本轮没有修改框架源码。

## 本轮实际修改

| 位置 | 新增/修改内容 | 原生复用与必要性 |
|---|---|---|
| ReservedPrefetch::Reserve | 32 block 一段的候选检查；连续 miss 截断；已定位在途 Handle；READY 标记与本地 blockOffsets | 继续调用原生 LookupOnPrefix、pin 和存储能力；保留 L1-only 命中，减少短命中后的候选扫描 |
| TransBuffer::PinReady | 已有 Early 方法增加 includeLoading 参数，默认 false | 仅借用活跃 owner 的已有节点；原生成员可见性不变，不在 reserve 抢占数据队列资源 |
| ReservedPrefetch::Prefetch | 复用保存的位置；64 owner shard 一批，最多 2 批在途；局部 submit/complete 及 Transfer 记录；异常先 drain | 原生 backend Load/Wait 和共享 owner 合并；不新增后端线程，不改 Posix 并发配置 |
| ReadLease::Extend | Early 自己定义的 ReadLease 类新增可选扩展，默认不支持 | 此类不在指定原生比较点中；不向 StoreV1 新增方法 |
| PosixReadLease::Acquire/Extend | 保存同一个原生 layout 的引用；复用 Lock 和同一 OFD | 避免候选分段导致 FD 数随段数增长；保留原有 GC 删除 guard |
| ReservedPrefetch::LoadHandles | 按本地 reservation 校验 block/shard 并复制现有 Handle | 不重新查询共享 buffer 索引；节点引用沿用 Handle 的 RAII |
| EarlyCacheStore::Load | 薄覆盖：有有效交接时构造原生 TransTask，否则调用父类 | 复用同一 TransManager、load queue、H2D 与完成流程 |
| Detail::TaskDesc | 新增可选 loadReservations 字段 | 将本 worker 的 reservation IDs 随本次原生 load 参数传入；无跨进程指针 |
| Cache::TransTask | 新增 loadHandles；移动构造保留引用 | 将已获取引用交给原生异步任务，避免排队期间释放 |
| LoadQueue::DispatchOneTask | 优先移动已持有 Handle，缺失时原生 Get；交接项跳过重复 Prealloc | 这是必要的原生执行入口修改；不复制整段 dispatcher/H2D，不新建队列 |
| C++ Early::LoadBinding::Describe / EarlyLoad | 调用同一个 MakeTaskDesc，再通过原有 Self/health wrapper 提交 | 原生 PipelineStore 只增加 friend 声明；不增加类方法，不改变私有解析函数属性 |
| Python _ReservedLoad | 新增 __init__/__getattr__/Load，小型组合适配 | 包装同一个 PipelineStore；原生 load/load_data 转换、Task、Check/Wait、Dump 全部复用 |
| EarlyPipelineStore | 绑定存在性检查、保存同一 native instance、set_load_reservations、释放时撤销绑定 | 没有第二个 Store；Early 操作继续使用原来的实例 |
| EarlyWorker::bind | 从本步 metadata 选择 READY、未取消、需要 load 的本地 job | 沿用原有 metadata、请求 generation 和 handoff 生命周期 |

原生 private 变量与 private 函数没有改为 public/protected。原生类没有新增函数；
有必要的字段、friend 声明，以及 DispatchOneTask 和 TransTask 移动构造的局部修改。
CacheStore 父类方法、PosixStore、SpaceManager、SpaceLayout、GC、AIO 引擎、
原生 Python connector/store 主流程及 vLLM/vLLM-Ascend 文件均无本轮改动。
此前已存在的共同兼容与 Early 差异没有被称为本轮新增复用。

## 相对指定原生点仍修改的原生代码

以下是相对 `5e1427d` 的累计差异，不能全部计作本轮工作。新增 Early 文件中的
完整实现及测试见完整 patch；表中列出原生类型、原生文件上的侵入。

| 原生文件/类型 | 累计差异 | 本轮部分与必要性 |
|---|---|---|
| integration/vllm/ucm_connector.py / UCMConnector | __init__ 选择 Early 子类及检查支持路径；on_new_request、shutdown 转发；get_num_new_matched_tokens 透传 PENDING | 本轮未改；保留当前入口和原生 Scheduler 重试语义 |
| cache_store.cc / CacheStore | friend EarlyCacheStore；同文件新增子类及工厂 | 仅给 Early 子类新增 Load；直接使用同一个私有 transMgr_，不改父类方法 |
| trans_buffer.h/.cc / TransBuffer 与策略类 | TryGet、PinReady；私有 Alloc 的可选尝试上限；BufferStrategy 的 NodeCount/CanReserve 及两种策略的 NodeCount | 本轮只扩展已有 PinReady；必须在原生锁内取得活跃节点引用，不能先查后 pin |
| load_queue.cc / LoadQueue | DispatchOneTask 接收现有 Handle，缺失部分调用 Get；已交接项不再 Prealloc | 本轮新增；这是直接复用正式 H2D 路径所需的消费入口 |
| trans_task.h / TransTask | loadHandles 字段、移动构造 | 本轮新增；任务在原生队列等待期间持有引用 |
| detail/type/types.h / TaskDesc | loadReservations 字段 | 本轮新增；携带本地 reservation 标识，不传进程间裸指针 |
| pipeline/cc/health_breaker_store.h | AcquireReadLease、ReservePrefix、PrefetchReserved、ReleaseReservation 转发 | 本轮未改；保持同一 health wrapper 和释放语义 |
| pipeline/cpy/pipeline_store.py.cc | Early 绑定的 include/注册；friend Early::LoadBinding | 本轮仅 friend；复用私有 MakeTaskDesc，未复制解析或新增父类方法 |
| posix/cc/global_config.h / Config | RemoveGuard 类型及可选回调 | 本轮未改；经原生初始化链向实际删除 owner 传递保护 |
| posix/cc/posix_store.cc / PosixStore | Setup 调用 Configure；受保护 Configure/GetLayout；同文件新增 EarlyPosixStore 及工厂 | 本轮未改；Early 使用父类唯一 manager/layout，不复制 Setup 或 GC |
| posix/cc/space_layout.h/.cc / SpaceLayout | Setup 保存回调、RemoveFile 在 guard 生命周期内 unlink、延期标记/TakeDeferredDeletion、friend PosixReadLease | 本轮未改；检查与删除必须在同一保护区间，不能移到普通进程内表 |
| posix/cc/shard_gc.cc / ShardGarbageCollector | RunGcCycle 读取延期标记，结束本轮重试 | 本轮未改；锁冲突继续使用原生等待周期 |
| ucmstore_v1.h / StoreV1 | 四个可选 Early 虚接口；同文件新增 ReadLease、PrefixReservation、PrefetchResult | 本轮仅 Early 自己定义的 ReadLease 增加 Extend；未再新增 StoreV1 方法 |

原生方法没有完全零修改。新增的正式 load 接入避免复制 dispatcher 和传输线程，
比再创建一套加载路径更小；完整移除这处入口会重新引入共享索引查询，或需要
更大范围的任务/队列替换。本轮没有更改任何原生成员的 private 属性。

## 验证结果与证据范围

证据目录：
`/data/liyuhang/glm51-vllm-ascend-ucm-minimal-review/evidence/prefetch-reuse-20260914`。

| 检查 | 结果 | 范围 |
|---|---|---|
| simu C++ 构建 | 通过 | AArch64；编译包含 -Wall -Werror；独立输出目录，不安装覆盖 A3 |
| 相关 C++ 测试 | 57/57 通过 | native Cache/Posix 回归、私有与共享 buffer、all-L1/部分 L1/短命中、批次顺序与异常 drain、引用交接 |
| 双进程共享 L1 | 通过，包含在上述 57 项中 | 两个真实进程创建/挂接原生 shared L1，一份 L2 读取、各自本地 Handle 与模拟 HBM 拷贝；backend 数据填充是测试替身 |
| 跨进程 GC/OFD | 通过，包含在上述 57 项中 | 真实文件、读锁与实际 GC 删除 owner、最后读者释放和异常路径；不是 NPU 故障注入 |
| 生产 Python 类与新绑定 | 42/42 通过 | direct/layerwise、READY/PENDING/fallback/identity/trace、异步单次提交、MLA/非 MLA identity、原生转换及错误传播；框架和 I/O 对象部分 mock |
| A3 原生构建 | 通过 | AArch64、ascend-a3；本轮全部 UCM 库重新构建并仅安装到本工作树 |
| A3 库加载及绑定回归 | 通过，Python 42/42 | A3 新 build 的真实 C++ 扩展；installed Cache/Posix/Pipeline 的 ELF、依赖和进程 maps 单独核验；仍未执行 NPU 推理 |
| GLM/Qwen 真实 NPU | 未验证 | 本轮没有启动模型，也没有访问用户实验服务器 |
| TTFT/吞吐改善 | 未验证 | 没有复现用户的 3379 请求实验，不声称解决 4.1 倍回退 |

主要日志为 `final-ctest.log`、`final-python.log`、`final-simu-build.log` 和
`a3-build.log`、`a3-python.log`、`a3-library-load.json`；manifest 记录 source/build
路径、commit、源码指纹和库 SHA256。A3 产物对应功能提交 `d50fce1`，其后提交
只增加本设计/验证文档和测试构建入口，生产代码没有变化。
`python_with_build.py` 打印实际加载的源码与新扩展路径，不安装 simu 库。

直接用 ldd/ctypes 检查 installed Cache 库时，初次未设置新 metrics 搜索路径，
得到 libmetrics.so not found；原始结果保存在 `a3-library-load-initial.log`。
在该检查子进程的 LD_LIBRARY_PATH 前加入本工作树 `ucm/shared/metrics` 后，
Cache、Posix、Pipeline 和 metrics 全部从本工作树加载，其他依赖来自容器系统
及 Ascend 运行时。该路径修正没有修改代码或全局环境；进程 maps 与最终依赖
列表保存在 `a3-library-load.json`，不能将 ctypes 加载当作模型启动。

曾补跑管理仓库的旧 `tests/test_early.py`：30 项中 24 项通过、6 项失败。
对起点 2d063302 复跑也得到同样的 6 个失败，日志分别为
`early-lifecycle-control.log` 与 `early-lifecycle-start-baseline.log`。
旧 fixture 通过修改 __class__ 绕过 Early 属性初始化，且仍观察旧 trace 入口；
没有修改生产代码迎合这些 fixture，也没有将这个旧套件报告为全部通过。
其中真实 ZMQ 的两 DP/四 worker 控制测试与 Worker/HRRN 测试通过；当前 Scheduler
行为使用直接调用生产类的 42 项测试验证。

首次 Python 收集缺少锁定依赖 wrapt；在独立 venv 中从已有校验 wheel 安装
wrapt==1.17.2。随后处理测试加载器对 infra/logger 原生构建路径的映射，最终
42 项通过。失败日志保留，没有升级依赖或引用旧 UCM 动态库。

## 构建与恢复

依赖版本沿用管理仓库 dependencies.lock.json。五个 vendor 均在独立目录中按
原 commit clone；wrapt 使用已有 wheel 与 SHA 校验，不复制原编译产物。

| 依赖 | 锁定版本 |
|---|---|
| fmt | 40626af88bd7df9a5fb80be7b25ac85b122d6c21 |
| googletest | b514bdc898e2951020cbdca1304b75f5950d1f59 |
| pybind11 | f5fbe867d2d26e4a0a9177a51f6e568868ad3dc8 |
| spdlog | 6fa36017cfd5731d617e1a934f0e5ea9c4445b13 |
| zlib | 51b7f2abdade71cd9bb0e7a373ef2610ec6f9daf |
| wrapt | 1.17.2；wheel SHA256 5bb1d0dbf99411f3d871deb6faa9aabb9d4e744d67dcaaa05399af89d847a91d |

既有本机复现入口如下。已停止的本任务构建容器可先 docker start；它只挂载
管理目录及只读的 `/usr/local/Ascend/driver`，未挂载 NPU 设备，network=none。
新环境先按上表检出 vendor 到独立目录，将本源码 `ucm/shared/vendor/<名称>`
链接至对应检出，不可链接旧源码或 build。构建脚本不会在线下载依赖。

```sh
docker exec --workdir /data/liyuhang/glm51-vllm-ascend-ucm-minimal-review/src/ucm-prefetch-reuse \
  glm51-prefetch-reuse-build python3 \
  /data/liyuhang/glm51-vllm-ascend-ucm-minimal-review/src/ucm-prefetch-reuse/test/build_prefetch_reuse.py \
  --build /data/liyuhang/glm51-vllm-ascend-ucm-minimal-review/build/prefetch-reuse-a3 \
  --runtime ascend-a3 --install \
  --log /data/liyuhang/glm51-vllm-ascend-ucm-minimal-review/evidence/prefetch-reuse-20260914/a3-rebuild.log
```

simu 使用同一入口，改为独立的 `prefetch-reuse-simu` 目录和 `--runtime simu`，
不传 `--install`。运行 `ctest --test-dir` 指向该 build，筛选
`UCCache|EarlyPosix|ReadLease`。Python 使用独立 venv 执行本仓库
`test/python_with_build.py --build <绝对 build 路径>`，pytest 参数为
`--noconftest -o addopts= -p no:cacheprovider -s -v <本仓库测试文件绝对路径>`。
上游全局 conftest 依赖 CUDA；本测试不使用它的 fixture，不因此安装 CUDA 依赖。

TaskDesc/ReadLease 的 C++ 布局有变化，必须整套重新构建并加载匹配的 UCM 库。
不得仅替换 Python 文件、单个 Cache 库，或把 simu 库用于 A3 服务。

本次 Git bundle、相对起点及指定原生点的完整 patch 放在上述证据目录中；
包含上游历史和本地分支，未推送远端。其他框架与 vendor 的锁定/恢复材料继续使用
管理仓库既有材料。本文件不把管理仓库提交当作 UCM 源码提交。

恢复到一个尚不存在的绝对目录（工作目录 `/data/liyuhang`）：

```sh
git clone --branch codex/ucm-prefetch-handle-reuse \
  /data/liyuhang/glm51-vllm-ascend-ucm-minimal-review/evidence/prefetch-reuse-20260914/ucm-prefetch-handle-reuse.bundle \
  /data/liyuhang/ucm-prefetch-handle-reuse-restored
```

`diff-from-start.patch` 对比 2d063302，`diff-from-native.patch` 对比 5e1427d；
`delivery-manifest.json` 记录最终提交、两份逐文件增删清单、bundle 校验及文件 SHA256。
bundle 不含模型、凭据、动态库、vendor 编译产物或大日志；恢复源码后须重新构建。

## 保留的开销与未验证项

- 真正缺失的 shard 在 reserve 与 prefetch 之间仍须重新检查，不能跨并发间隔信任负查询；本轮消除的是已有在途位置的重复定位和正式 load 的共享索引重查。
- 为准确判断 all-L1 与 remaining_l2_work，接受的前缀仍需逐 shard 检查；没有重建 block 级索引。
- 预取仍覆盖请求候选前缀，未新增 Scheduler/HBM 提示协议；可能预取 HBM 已有部分的问题未在本轮解决。
- PENDING 和所有 rank READY 屏障保持。分批只让预取内部的后续定位与前面 I/O 重叠，不把请求改为软依赖。
- 32/64/2 是当前有界实现常量，没有进行参数寻优，不保证对目标负载的最优批次或更低 TTFT。
- 历史 GLM 冷请求错误、原生 AIO 提交/完成时序问题未关闭；read lease 不证明 KV 已完整写盘。
- 批次异常测试覆盖返回失败和接受任务前抛异常；没有证明违反原生 Wait 完成契约的后端也可安全释放。
- 新代码真实 16 卡 GLM/Qwen 开关对照、冷 L1/L2、真实 HBM 故障、长期 GC 压力和性能实验均未验证。
