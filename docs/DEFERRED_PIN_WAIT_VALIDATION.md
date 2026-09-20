# 延后 pin / 容量等待验证入口

起点为 `c50dfab7ce560b20ce1ebdabf9a7a0cf0e3a9539`，独立分支
`codex/ucm-deferred-pin-wait-c50dfab`，原源码工作树及分支保持不动。
本机源码目录：`/data/liyuhang/glm51-vllm-ascend-ucm-minimal-review/src/ucm-deferred-pin-wait`。
管理仓库起点为 `97dfba618fd8d680ac91050972aabbc08ded6166`，不与 UCM 提交混称。

## 构建与隔离

操作脚本和原始证据在管理工作区
`evidence/deferred-pin-wait-20260920/`；`build_simu.py`、`build_a3.py` 使用同一锁定
官方 A3 镜像内的工具链，分别写入 `build/deferred-pin-wait-simu` 和
`build/deferred-pin-wait-a3`。vendor 来自已有锁定源代码归档；没有升级依赖。
A3 安装仅写入新源码树。独立 `runtime/deferred-pin-wait-venv` 通过 pth 指向
该树和原锁定框架。`verify_build.py` 核对 compiled commit、源码 hash、动态库
build ID/import/maps；不得把 simu 库安装到 NPU 运行目录。

构建涉及 shared header magic 3、owner VERSION 5 和 Early::Batch ABI。
必须重建整个 UCM 并重启全部参与 workers，以新的原生 unique_id/共享对象启动；
不在旧共享域挂接新版库。容器使用 private IPC，避免原生 shm 清理触及无关服务。

## 已执行的离线验证

- 77 项相关 C++ 测试通过，包括原生 buffer、Early 批次、独立读取完成、read lease
  和跨进程共享 L1。新增 6 项验证延后 pin、L1-only/淘汰核验、事件等待恢复完整命中、
  部分占用取消后排空，以及跨进程原生 LoadQueue 释放唤醒。
- 164 项 Python 测试通过，包含实际 CPython/C++ binding、Posix 文件读写与模拟 H2D；
  协议测试覆盖 4 槽两个 3 块请求、多 owner/DP 并集计费、短命中、明确超容量、
  重复/迟到授权、线程流水与全 L1 不进数据队列。
- 最终代码 `1bf37bb` 的 70 项定向 Python 回归覆盖释放先于暂停、通知并非额度、迟到激活回滚、资源等待时钟及取消/失联边界；这些包含在上述 164 项中，不重复计数。
- `git diff --check`、新代码 Python 编译检查通过。环境未安装 ruff，未为本任务升级依赖。

C++ 的共享映射、跨进程引用及原生 LoadQueue 是真实实现；SSD/H2D 使用测试 backend /
simu runtime。Python 协议单测的存储是 mock；binding 测试使用真实 Posix 文件。
这些结果不证明 NPU 数值、完整 W32 压力或 TTFT 改善。

复跑命令（在锁定构建容器内，workdir 指向新源码树）：

```bash
python /data/liyuhang/glm51-vllm-ascend-ucm-minimal-review/evidence/deferred-pin-wait-20260920/build_simu.py
/data/liyuhang/glm51-vllm-ascend-ucm-minimal-review/build/deferred-pin-wait-simu/ucm/store/test/ucmstore.test --gtest_filter='*Early*:*ReadLease*:*TransBuffer*'
/data/liyuhang/glm51-vllm-ascend-ucm-minimal-review/runtime/deferred-pin-wait-venv/bin/python /data/liyuhang/glm51-vllm-ascend-ucm-minimal-review/src/ucm-deferred-pin-wait/test/python_with_build.py --build /data/liyuhang/glm51-vllm-ascend-ucm-minimal-review/build/deferred-pin-wait-simu --noconftest /data/liyuhang/glm51-vllm-ascend-ucm-minimal-review/src/ucm-deferred-pin-wait/test/test_early_capacity_wait.py
```

## A3 实际结果与限制

实际模型测试与动态库均为 `1bf37bb5dbd8df3aa5ef25c05d8d3eb54a7dd347`。
之后的文档提交不改变运行代码。官方镜像 digest 为
`d03346e4826118aaade06b4425e35dd7a7270f5a54322691e051e93d34897a0e`；
vLLM=`0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`，
vLLM-Ascend=`f4a08bddd0cc65a0bd8c3d377b158ae5ca7527db`，两者无实际修改。

### 同版本 GLM 减层对照

`verified-off` / `verified-on` 均使用 GLM-5.1-w4a8 的 8 层视图、TP4/DP1/EP、
eager、bfloat16 KV、context32768、seq8、batch4096、HBM0.60、KV blocks256；
UCM direct、shared CPU L1 3GiB、独立 L2 上限1024GiB、AIO/direct、Posix 并发1。
同一个本任务拥有的稳定 OFF 种子复制到两臂，冷 L1；不清理其他服务的缓存。
完整参数、镜像、模型身份、环境及实际 `.so` 路径在各 `plan.json` / 服务启动日志中。

- OFF / ON 各 19/19 HTTP 请求返回有效 4-token 序列，含冷 L1、热缓存、
  12 条 HBM 置换请求、4 条并发共享前缀和完全 miss。两容器退出码均为0。
- ON 冷请求：27 blocks、38,928,384 bytes 实际预取；external3456 tokens，
  本步调度84 tokens。4个实际 rank 均完成消费。
- ON 有4个 `l1_ready`，仅冷请求进入数据批次；全 L1 不提交 SSD 预取。
- 19个 owner 分配均有对应释放；结束快照 read lease 锁为0，无 domain fence。
  该小负载没有触发容量等待；不能用它替代真实 NPU 容量压力。
- 只验证请求执行和传输链；8层模型输出不是完整 GLM 的语义正确性证明。
  OFF/ON 的冷/热/并发/miss token 序列一致，12条置换请求有5条不同；未作数值等价结论。
- 中间版本的 `smoke-off`、`final-off` 仅用于种子或诊断，不混入最终对照。

### NPU 字节检查发现的问题

不能把模型 HTTP 成功写成全部验收通过。`verified-on/npu-byte-probe` 在
冷 L1、没有 reservation 的 Early 正式加载中出现3,264,162个不一致字节，
早于 deferred reserve 调用。失败后独立重建随机输入，核对27个SSD文件的
内容 hash，全部与预期匹配。原始失败报告和CANN日志保留。

后续原生3项、deferred Early4项均通过；交替3次原生/3次Early共21项通过；
新版本连续20次独立冷L1加载加deferred/全L1/dump回读共23项通过。
这些通过记录不能关闭最初的失败。

旧工作树虽然 HEAD=c50，本地旧库实际编译标识为 `d9f3f27`，身份检查拒绝将其
作为基线。因此另建只读诊断检出 `src/ucm-deferred-pin-baseline-c50`，在本机A3
完整重建精确c50（manifest证明tracked diff为空），仅在独立探针进程加载。
该精确基线也在第一次同类冷加载中出现字节不一致。因此该失败签名无需本轮修改
即可出现，但尚不能宣称两次失败的底层根因相同，也不能确定本轮完全不影响概率。

诊断探针切换 `cache_sdma_direct=false` 后，精确c50及新deferred代码各4项通过。
这将调查范围缩小到既有SDMA Direct路径/运行环境，尚未定位具体缺陷。
模型服务配置没有随探针更改；不把这个开关变化混作公平对照或修复。
证据：`failed-probe-seed-audit.json`、`npu-probe-summary.json`、
`npu-c50-rebuilt.log`、`npu-c50-standard.log`、`npu-deferred-standard.log`。

### 完整压力未执行

已生成78层、TP8×DP2/local DP2、EP、seq16、batch4096、HBM0.95、shared L1 96GiB
的同版本配置和W32合成客户端（65536-token输入、61440-token共享前缀、32个独立
前缀族、96条测量请求）。该长度在原始64K–128K范围内，同时保留磁盘余量。
它不是用户服务器上的3379轨迹，也不能恢复其真实依赖分布。

2026-09-20 12:18 UTC 尝试启动时，NPU空闲检查拒绝执行 docker run。
随后核对占用进程属于原有 `va23-offload-wp-a5` 容器（worker从12:14 UTC陆续启动），
未停止或修改其任务。`full-runtime-blocker.json`、`npu-full-blocked.txt` 保留证据。
本任务全部模型容器已停，未启动Qwen。没有完整层数W32、长期容量等待或TTFT收益证据；
c50全量吞吐回归未宣称解决。SDMA字节异常本身也仍是完整正确性验收的阻塞项。

## 操作和恢复

操作目录包含 `launch.py`（启动前源码/库身份及设备检查）、`run_smoke.py`、
`exercise.py`、`pressure.py`、两套构建脚本和原始日志。每个case拒绝覆盖，容器使用
本任务标签/private IPC，清理入口只停止标签匹配的本任务容器。未删除旧服务缓存。

完整压力种子仅在任务拥有的、已停止的源case间复用；完整模型的种子文件使用
独立目录的硬链接以控制磁盘占用，依据Posix对最终文件只读、写入临时文件后rename
的行为；新锁文件独立生成。小模型使用独立文件副本。该完整压力入口尚未执行。

`recovery/deferred-pin-wait-20260920/` 保存UCM新分支Git bundle、操作/配置/小型证据
归档及校验和；不含权重、旧编译库、凭据或大体积日志，不推送远端。
管理仓库保持原HEAD `97dfba618fd8d680ac91050972aabbc08ded6166` 和既有未提交改动。
