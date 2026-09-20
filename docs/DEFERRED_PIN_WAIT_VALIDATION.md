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

- 76 项相关 C++ 测试通过，包括原生 buffer、Early 批次、独立读取完成、read lease
  和跨进程共享 L1。新增 5 项验证延后 pin、L1-only/淘汰核验、事件等待恢复完整命中、
  部分占用取消后排空，以及跨进程原生 LoadQueue 释放唤醒。
- 160 项 Python 测试通过，包含实际 CPython/C++ binding、Posix 文件读写与模拟 H2D；
  协议测试覆盖 4 槽两个 3 块请求、多 owner/DP 并集计费、短命中、明确超容量、
  重复/迟到授权、线程流水与全 L1 不进数据队列。
- 后续定向测试补充释放先于 lookup 暂停发布的竞争；该次 9 项通过。
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

## NPU 证据边界

运行中证据随后补充。只运行 GLM，不启动 Qwen，不停止无关服务。
8 层/TP4 smoke 仅验证入口与实际加载链；完整压力必须保留 78 层、TP8×DP2、W32、
长输入和稳定容量压力。未完成该验证前，不宣称 c50 全量吞吐回归已解决。
