# 单 reserve owner：构建、审计与验证入口

本轮在原工作树 `ucm-reserve-pipeline`、源码分支 `codex/ucm-reserve-read-pipeline`
继续实现；审查起点 `a60def22ac9451c3e7af3078e01a5be932fe5909`，不回退、不换树。
原生比较基线 `5e1427d6b89f69a2bff518dc59fbf5b6a52bc3c1`。
管理仓库提交与 UCM 源码提交分别记录，不能互相替代。
协议、引用和容量语义见 [设计文档](EARLY_PREFETCH_DESIGN.md)。

## 本轮审计与落地

| 位置 | 原生/既有能力 | 本轮差异与必要性 |
|---|---|---|
| EarlyCacheStore / EarlyBufferPins | 已创建的 BufferManager/TransBuffer、CanReserve、DataAt/MetaAt | Early 专用只读 DomainProvider 获取实际映射身份、原生容量，不创建 manager/buffer，不改变私有属性 |
| pipeline Early bindings / LoadBinding | 既有 friend、原生 stores/health wrappers、MakeTaskDesc/Self | EarlyDomainInfo、Batch Snapshot；通过相同实例读取健康/对象信息和 I/O 完成计数 |
| LoadQueue::DispatchOneTask | 原生 Get、引用持有、传输与 Wait | 唯一直接变更的原生执行函数：扩展已有汇总日志，给非 owner native Get 记录实际 SSD 读数及引用已取得的时间上界；执行顺序不变 |
| OwnerBroker : BrokerState | 同一 Control、broker/job table、派发/取消和 socket 生命周期 | 必要的新协议：物理域握手、单 owner、轮询、跨 DP 共享额度、generation/attempt、接管/排空和失联停用 |
| OwnerWorker | 组合既有 EarlyWorker、lookup pool/data thread、WorkerJob | owner 才调用既有 reserve/read 流水；非 owner 不 reserve；排空确认、正常关闭握手及故障隔离 |
| Consumer / LoadConsistency | 组合原生同一 RankConsistencyManager、Load/Wait | 以实际请求作用域和原生成功 Wait 确认层/block 完成；不把相同前缀或入队当成接管 |
| EarlyConnector metadata | 继承原生 HBM 扣除、缺失过滤、token 计算、metadata | 附加生成号和冻结的原生消费计划，不重新 hash/lookup，不传远端 reservation ID |
| ReservedPrefetch::Batch | 原 Check/Wait、RAII、drain | Snapshot 分开记录接受/完成字节，失败也可取值；无新 I/O 执行器 |
| analyzer | 既有 PID/reservation/batch 时序关联 | 保留每层/任务消费记录，按 PID/task 关联 native Get/SSD 读，记录 owner 额度，不覆盖成最后一层 |

本轮没有新增原生类方法、没有改原生 private 字段/函数可见性。新方法位于 Early
类或既有 Early friend 辅助中；原生 load_queue.cc 的变动仅为上述日志。
vLLM 与 vLLM-Ascend 源码不改，分别保持
`0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` 和
`f4a08bddd0cc65a0bd8c3d377b158ae5ca7527db`。

相对原生的全部累积差异还包含此前的 Early 工厂选择、TransBuffer friend/容量
原语、TaskDesc/TransTask 的本地 handle、Store/health reservation 路由、Posix
删除 guard/read lease 及原有 pipeline 优化。这些不是本轮新增复用；两份完整
patch 由管理仓库证据目录生成：`ucm-vs-native.patch`、`ucm-vs-a60def2.patch`。

## 构建与测试

继续使用锁定官方 A3 镜像：
`quay.io/ascend/vllm-ascend@sha256:d03346e4826118aaade06b4425e35dd7a7270f5a54322691e051e93d34897a0e`。
CANN 9.0.1、Python 3.12.13、aarch64；依赖不升级。simu 与 ascend-a3 必须使用
不同 build 目录。以下命令在该镜像中执行，所有占位路径均替换为绝对路径，
执行目录显式设为恢复后的 UCM 源码根。

```bash
python /absolute/ucm/test/build_prefetch_reuse.py \
  --build /absolute/build/owner-simu --runtime simu --log /absolute/evidence/simu.log
ctest --test-dir /absolute/build/owner-simu --output-on-failure --timeout 90 \
  -R 'UCCache|EarlyPosix|ReadLease'
python /absolute/ucm/test/python_with_build.py --build /absolute/build/owner-simu \
  --noconftest -o addopts= -o log_cli=false -p no:cacheprovider -q \
  /absolute/ucm/test/test_early_ready_reuse.py \
  /absolute/ucm/test/test_early_batch_pipeline.py \
  /absolute/ucm/test/test_early_batch_binding.py \
  /absolute/ucm/test/test_early_owner_protocol.py \
  /absolute/ucm/test/test_prefetch_timeline.py
python /absolute/ucm/test/build_prefetch_reuse.py \
  --build /absolute/build/owner-a3 --runtime ascend-a3 --install \
  --log /absolute/evidence/a3.log
```

构建脚本记录完整源码指纹、UCM commit、所有 .so 的 SHA256，并拒绝构建中源码
变化。Snapshot/DomainProvider 是 C++ 接口变化，必须重建并一起加载整套 UCM，
不能只替换 Python 或一份 libcachestore.so。simu 测试用普通内存模拟 H2D，
不得把 NumPy host 地址传给真实 A3 传输库。运行依赖通过新 venv 复用锁定镜像；
唯一补装的 wrapt 1.17.2 来自已经锁定的本地 wheel，使用 --no-index --no-deps。

### 证据范围

- 协议测试：轮询、DP 隔离、单 owner、握手拒绝、共享额度、延迟/重复/乱序、
  request ID 复用、取消排空、正常关闭、失败停用；mock backend/完成状态。
- 真实 worker 线程测试：先 reserve 第一批即读、后批继续定位；长请求等待不
  阻塞其他请求、原有 HRRN/窗口、全 L1 不进数据队列；存储为 mock。
- 真正 ZMQ 多进程：两个 DP、四个 worker，只向一个 owner 发 reserve，回收
  握手完成后所有 consumer forget；能力信息/accelerator completion 为 mock。
- C++ 真实共享内存：唯一 native 对象、一个 owner、两个独立 native LoadQueue
  消费进程；其中一个延迟且有淘汰压力，仍读到正确字节。consumer 不 reserve，
  backend Load/Lookup 调用数为零；H2D 和种子 backend 为模拟。
- 原生进程死亡测试：子进程在正常取得引用后 _exit，证明 native reference
  不会自动回收。没有宣称死亡期间持有的 spinlock 可恢复。
- CPython + C++：真实参数转换、Posix PSync 写入和读取、owner handle 复用、
  reservation 释放后 native H2D 引用仍有效；observer 仅在成功 Wait 后确认。
- C++ Posix：实际 GC owner 的跨进程 OFD 锁、多个读者、异常退出、保护解除后
  回收；这一机制复用前轮实现，不在本轮另建共享协议。
- 模型/NPU：以上均不能代替真实 GLM 请求、EP 启动和 TTFT 对照。

开发过程的 C++ 74 项、完整 Python 88 项已通过；后续增加握手/失联等测试后，
协议/流水/时序合计 46 项通过。进一步补齐非 owner READY 的失联失效、停用域的新加入者拒绝、PP>1 拒绝和
未调度取消的定向映射清理。最终构建、全部测试数及 GLM 运行结果放在管理
仓库 `evidence/owner-protocol-20260914/` 和 `docs/OWNER_PROTOCOL_RESULTS_20260914.md`，
以实际日志和 manifest 为准。首次 Python 收集因裸镜像缺 wrapt 失败；新 venv
使用锁定 wheel 后解决。另有四个旧测试绕过 __init__，未初始化新增组合对象，
已补齐 fixture 并通过；没有通过放宽生产代码隐藏该失败。

## GLM 实验与证据规则

仅 GLM，不启动 Qwen。使用同一锁定模型、官方镜像、vLLM/Ascend、DP=2/local
DP=2/TP=8/EP、78 层、上下文/并发/batch/HBM 与存储配置。每次启动前检查设备
和服务，在对话中展示完整有效配置；不得停止 peer 服务腾卡。

必要对照开关：native OFF 为 `enable_early_prefetch: false`；旧 reserve 路径
为本代码 `enable_early_prefetch: true, enable_early_owner: false`；owner 路径
为两个开关 true 且实际能力握手成功。旧路径开关不是旧提交 a60def2 的运行证据，
对照需明确源码/库身份，不能混称。需要严格 a60def2 前后对照时，应由恢复材料
在独立部署环境构建它，不能在当前活动工作树回退或混用旧库。

完整层数/TP/输入长度与稳定积压条件必须保留。HTTP 200、短 smoke 或热 HBM
都不足以证明 owner 或 TTFT 改善。冷 L1 场景只能重启本任务拥有的容器/IPC，
复用本任务自己的已确认 L2；先用日志证明实际 L2→L1、consumer 原生 Get 和
owner drain，再评估阶段/分位数。新队列额度没有上调底层并发，不保证单 owner
下 SSD 并行提交能力与旧多 rank 相同。无 NPU 对照时 TTFT 改善必须写未验证。

历史 GLM OFF 曾有 EP/AICPU 初始化失败、HCCL 端口 bind 失败和启动可用 HBM
不足，根因尚未闭环；本轮一次成功也不能关闭这些问题。read lease 只保护删除，
原生 AIO dump 提交/可见性问题另行记录，不能归因于或宣称被本协议修复。
