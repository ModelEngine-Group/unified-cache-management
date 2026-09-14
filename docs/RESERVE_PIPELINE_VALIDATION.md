# Reserve / read 流水：复现与证据边界

## 身份与比较口径

源码工作树：`/data/liyuhang/glm51-vllm-ascend-ucm-minimal-review/src/ucm-reserve-pipeline`。
分支：`codex/ucm-reserve-read-pipeline`。按用户确认继续使用当前树和 `71c532f`
检查点，没有回退它或后续提交。临时隔离树未构建、未运行，改动迁回后已移除。

管理仓库是其上方的 `glm51-vllm-ascend-ucm-minimal-review`，不是 UCM 源码仓库。
本轮不推送，也不修改原 GLM/Qwen 分支、旧工作区、旧服务或共享缓存。

| 比较点 | 用途 |
|---|---|
| `5e1427d6b89f69a2bff518dc59fbf5b6a52bc3c1` | 锁定原生 UCM |
| `5bfd2b9` | 用户审计版本；已有批量定位、活跃 READY 复用、LoadPlan |
| `2fc0c3884ffa63f9f24ac2b3bc88fbdd0e7077b6` | 实际续作起点，晚于用户给出的提交；增加了释放后位置提示核验、移除闲置单 shard 包装 |
| `9360b5f` | Batch 复用原生 Load/Check/Wait 与排空 |
| `0138a73` | 同 worker 定位合并、保守字节额度 |
| `f74de71` | 已确认批次提前发布；仍按批同步处理，是独立阶段对照快照 |
| `71c532f` | 跨请求有界批次推进及计时检查点 |
| `39b4ba9` | 增加真实 Python/C++/Posix 绑定生命周期测试 |
| `41dd579` | 每批独立 lease，删除无生产调用的 Extend；未交付 handle 的异常清理 |
| `06ad8a0` | 关联计时与异常释放标记；后续文档/解析器提交见 Git 历史 |

完整最终源码身份与每个 `.so` 的哈希以外部构建 manifest 为准。
vLLM 保持 `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`；vLLM-Ascend 保持
`f4a08bddd0cc65a0bd8c3d377b158ae5ca7527db`，两仓 tracked diff 都为零。
共同 UCM bootstrap 注入继续存在；“框架源码零 diff”不表示 Python 运行时没有
原基线的 patch。此次没有更改这些 patch 或升级依赖。

## 自动化证据

证据根目录为管理仓库的 `evidence/reserve-pipeline-20260914/`。
构建全部使用官方 A3 镜像及独立目录，没有从旧树复用 UCM `.so`。

| 证据 | 结果与范围 |
|---|---|
| `pipeline-ctest.log` | 阶段一 72 项 C++ 通过 |
| `pipeline-python-all.log` | 阶段一 50 项 Python 通过 |
| `pipeline-stage1-a3-build.manifest.json` | `f74de71` 整套 A3 原生构建与安装 |
| `interleaved-ctest.log` | 跨请求推进后 72 项 C++ 通过 |
| `interleaved-python-all.log` | 53 项 Python 通过，含真实 worker 线程 + mock backend |
| `batch-binding-python.log` | 真实 CPython/C++ Pipeline、Posix 文件与 PSync、模拟 H2D，1 项通过 |
| `correlation-ctest.log` | lease 简化与计时修正后 72 项 C++ 通过 |
| `correlation-python.log` | 58 项通过，含上述绑定测试和 3 项计时解析测试 |
| `final-process-control.log` | 5 项通过；其中 1 项是两个 DP、四个独立 worker 进程的真实 ZMQ，另外 4 项是 broker 状态测试 |
| `timeline-final.log` | 3 项解析测试通过；区分实际 native Get 次数与未交接 handle 数 |

C++ 包含真实共享内存、多进程节点引用/预取合并，以及实际 Posix GC owner 的
跨进程 OFD 锁互斥、多个读者、最后释放、异常进程退出和 GC 等待/回收。
部分 C++ case 的 backend 为 mock；它们不能证明 SSD 时延、吞吐或 NPU 正确性。
绑定测试用 PSync 完成种子写入再读取，不覆盖 AIO 提交可见性问题。
没有用 Python 类型名或继承关系断言代替真实参数转换、数据传输和资源释放。

可复现构建（所有路径须替换为本地恢复后的绝对路径）：

```bash
python /absolute/ucm/test/build_prefetch_reuse.py \
  --build /absolute/build/simu --runtime simu --log /absolute/evidence/simu.log
ctest --test-dir /absolute/build/simu --output-on-failure --timeout 90 \
  -R 'UCCache|EarlyPosix|ReadLease'
python /absolute/ucm/test/python_with_build.py --build /absolute/build/simu \
  --noconftest -o addopts= -p no:cacheprovider -v \
  /absolute/ucm/test/test_early_ready_reuse.py \
  /absolute/ucm/test/test_early_batch_pipeline.py \
  /absolute/ucm/test/test_early_batch_binding.py \
  /absolute/ucm/test/test_prefetch_timeline.py
python /absolute/ucm/test/build_prefetch_reuse.py \
  --build /absolute/build/ascend-a3 --runtime ascend-a3 --install \
  --log /absolute/evidence/a3.log
```

绑定数据测试只能使用 simu 构建；其 NumPy 地址不是 NPU 地址。A3 编译后不得
把此测试当作真实 NPU 传输调用。所有 C++ 类型/接口变化必须整套重建、一起加载。
构建脚本校验构建前后源码指纹；不能在构建过程中修改源码。

## GLM 启动证据

本轮真实模型实验仅 GLM。此前误创建的 Qwen 容器已按用户要求停止，发送请求数
为零，不记为 Qwen 验证结果；没有继续启动 Qwen。

共同配置：官方镜像
`quay.io/ascend/vllm-ascend@sha256:d03346e4826118aaade06b4425e35dd7a7270f5a54322691e051e93d34897a0e`，
只读 `/data/model/GLM-5.1-w4a8`，78 层；单机 16 卡、一个 serve，DP=2/local DP=2/
TP=8/EP 开；BF16、Ascend W4A8、KV auto→BF16；context=131072、seq=16、
batch tokens=4096、HBM=.95、eager、chunked prefill、prefix caching。
Cache|Posix layerwise、96 GiB MLA shared L1 原生身份、1024 GiB 独立 L2、direct AIO、
所有 Posix 并发为 1。完整参数、环境变量、模型 identity、实际代码/库路径和哈希
保存在每个运行目录 `plan.json`，启动入口再次核验。

| 运行目录（管理仓库下） | 结果 |
|---|---|
| `runtime/reserve-batch-models/glm-off-01` | `2fc0c38` Early-OFF。完整权重加载约 49 分钟后，内存 profile/EP 初始化报 `MoeDistributeDispatchV4`、`RunAicpuKfcResInitV2`、AICPU `507018/0x2a`。未创建 CacheStore，无健康成功、无请求。退出 1，OOMKilled=false |
| `runtime/reserve-pipeline-models/glm-stage1-off-01` | `f74de71` Early-OFF。模型构造期间 HCCL `hcclCommInitRootInfoConfig` 错误 7，`EI0020`：设备网络地址 `192.23.2.191:16666` 已绑定。未完成权重加载，无请求。退出 1，OOMKilled=false |

第二次的直接错误是端口绑定冲突；占用者或设备侧残留来源未确定。不能将它
合并为第一次 AICPU 异常的根因。两次 stdout/CANN 日志均保留，没有重置设备、
停止旧服务、修改 HCCL 参数或清理全局 IPC 来绕过。

这些失败发生于请求及 Early reserve/read 之前，不能作为本轮预取时延证据。
历史 GLM 冷请求内容错误与原生 AIO 过早可见问题仍未关闭。后续成功也不能直接
关闭它们。完整 GLM Early-ON/OFF 的冷 L1、正命中 all-L1、稳定积压、首 token、
跨 DP/rank、异常与回收验收未通过前，均不得宣称 TTFT 已改善。

## 阶段归因方法

使用同一容器运行的日志，不混入其他运行中可能复用的 PID/reservation 数字。
`batch_reserved` 显式记录 PID、key、rank、batch、reservation；C++ stage 含 PID，
二者关联 lease/pin/L2/首提交/末回收时间。数据队列按本批 publication→start 计算。
全 L1 还需正长度 hit、无数据队列、零实际 L2 read；完全 miss 不能当作 L1 命中。

```bash
python /absolute/ucm/test/analyze_prefetch_timeline.py \
  --logs /absolute/run/service.log \
  --clock-proof /absolute/run/clock-proof.json \
  --output /absolute/run/timeline.json
```

缺少 clock proof 时只计算同 worker 内的间隔。proof 须包含 `complete`、
`expected_counts` 和所有参与 `processes` 的 role/boot_id/time_namespace；期望
角色含 client、scheduler、broker、worker，计数必须来自实际完整进程映射。
broker 可在既有进程内运行。不能以“都是 monotonic”代替跨进程时钟域核验。

`first_io_lead_over_reserve_done_ms` 使用 C++ Reserve 结束时间，说明第一批提交
是否早于整个 reserve 结束；不等于 SSD 连续忙碌时间或有效重叠收益。
`pin_ms` 包含分组、分配、等待和锁竞争；`wait_ms` 是 Wait 回收耗时，不是整段
异步 I/O 历时。需要对本任务 PID 采样 profiler 才能进一步归因，当前尚无该证据。

联合 `load_handles` 只记一次，不将其计数重复分摊给每个消费者。未交接 handle
数与 `load_dispatch.native_get_fallback` 的实际执行次数分列；没有 task→请求映射
的记录保留，不能猜测归属。缺失、重复或跨时钟域记录不补成零时延。

真实性能对照须保持 78 层、TP=8、DP=2、输入长度、到达率/并发和相近缓存状态。
可缩短总请求数，但必须覆盖稳定积压，并分别保留 OFF、`5bfd2b9`、阶段一和最终
批次交错的证据。用户服务器的 3379 请求原始记录不可访问，当前没有等价负载
对照结果。短 smoke、CPU 探针和分位数相减都不能替代该对照。

## 保留的限制

- MLA 跨 rank 仍各自获取安全本地引用；没有新建跨进程位置目录。定位合并及
  READY BlockPins 复用仅发生于同 worker。
- PENDING/整请求全 rank READY 保持；窗口是有界推进，不承诺任何指定吞吐或 TTFT。
- 保护额度是 worker 内的保守预算；真实跨进程物理容量仍由原生 buffer 控制。
  压力过高会 NoSpace→原生 fallback，需要在实际负载统计比例。
- 现有 read lease 只防 GC 删除；不保证 dump 完整写入后才可见。没有恢复 producer
  prefix、持久化等待或 revalidation。
- 私有变量可见性未改变；本轮没有新增原生类方法。原来已存在的 TryGet/Alloc
  可选界限、Posix guard、StoreV1 Early 接口和 LoadQueue handoff 仍属于相对原生
  的差异，不能称为原生零修改。
