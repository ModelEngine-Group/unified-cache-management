# gsa_prefetch C++ 测试套件

针对 `ucm/sparse/gsa/prefetch` 下 C++ 代码（源码位于 `src/` 与 `ascend/`，
头文件位于 `include/` 与 `ascend/`）的 UT / MST / DT 测试。

整个测试套件使用 `KV_CACHE_PRE_TEST_PRIVATE_AS_PUBLIC` 宏构建，该宏会：

- 用测试桩
  [`ascend/kvcache_trans_ascend_stub.h`](../ascend/kvcache_trans_ascend_stub.h)
  替换 Ascend ACL SDK（不依赖 Ascend NPU / CANN / torch_npu）；同时
- 将 `GSAPrefetchEngineC` 的私有成员暴露为 public，使跨模块（MST）与端到端
  （DT）用例可以直接驱动内部流程。

所有 NPU/Ascend/acl/torch_npu 相关的 H2D、D2D 与 ACL scatter
操作均已打桩（`aclrtMemcpyAsync`、`aclrtMalloc`、`aclnnInplaceScatter`、
`aclCreateTensor` 等），所有 `mStoreV1_`（UC::StoreV1）相关操作均由
[`common/mock_store_v1.h`](common/mock_store_v1.h) 进行 mock。ACL 桩还额外
提供可注入的失败开关（`acl_stub::Stub()`），从而可确定性地覆盖
`kvcache_trans_ascend.cpp` 与预取源码中的异常分支 / 提前返回分支。

## 目录结构

```
tests/
├── CMakeLists.txt              # 测试目标 + 覆盖率目标
├── common/
│   ├── mock_store_v1.h         # UC::StoreV1 mock（记录调用、可注入失败）
│   ├── securec.h               # 空桩（源码未引用任何 securec 符号）
│   └── test_fixture.h          # 共享辅助（model/size info、shm 清理、临时日志）
├── ut/                         # 单元测试 —— 行覆盖率 + 分支覆盖率 ≥ 90% 目标
│   ├── test_hbm_manager.cpp            # HBMManager：LRU、pinned 槽、miss/eviction、不变量
│   ├── test_logger.cpp                 # Logger：日志级别、过滤、文件生命周期
│   ├── test_kvcache_pre_detail.cpp     # KVPackFail/Abort、结构体、average/average2d
│   ├── test_ucm_kv_prefetch.cpp        # UCMKVPrefetch 异步 load/wait/lookup 生命周期
│   ├── test_kvcache_trans_ascend.cpp   # 经桩驱动的 ACL 封装（成功 + 失败分支）
│   ├── test_kvcache_task_manger.cpp    # TaskManager 线程池 + TaskInfo/ThreadInfo/TaskWrapper
│   ├── test_gsa_prefetch_engine.cpp    # GSAPrefetchEngineC 可测试的 public 接口
│   ├── test_kvcache_pre_logic.cpp      # kvcache_pre.cpp 纯逻辑（60 用例）
│   ├── test_kvcache_pre_scatter_logic.cpp    # scatter 纯逻辑（25 用例）
│   ├── test_kvcache_pre_scatter_indexer.cpp  # indexer-load 链（27 用例）
│   ├── test_indexer_map.cpp            # indexer-map 管理（26 用例）
│   ├── test_set_kv_buffer.cpp          # SetKvBuffer 初始化 + mISSPAC8 + 幂等
│   ├── test_prefetch_h2d.cpp           # H2D/scatter 管线 + ProcessPrefetchRequest
│   ├── test_prefetch_scatter_acl.cpp   # ACL scatter + TransferAndScatter + MergeAndCopy
│   ├── test_read_prefetch_results.cpp  # Write/ReadPrefetchResults 往返 + rank != 0
│   ├── test_submit_task.cpp            # SubmitTaskIndxer + worker 线程
│   ├── test_abort_paths.cpp            # KVPackAbort 错误路径（EXPECT_DEATH + __gcov_dump）
│   └── test_acl_runtime_error.cpp      # InitAclRuntime 错误日志分支
├── mst/                        # 模块 / 子系统流程测试（>5 / KLOC）
│   ├── mst_lastchunk_flow.cpp       # engine + store + ACL + token-merge 流水线
│   ├── mst_ucm_merge_flow.cpp       # miss 表 + store + 共享 KV 暂存合并
│   ├── mst_indexer_flow.cpp         # indexer offload 加载 + map 更新
│   └── mst_blockmap_lifecycle.cpp   # SetBlocksMap / DelBlocksMap 并发
└── dt/                         # 端到端测试（含输出正确性校验）
    ├── dt_hbm_lru_correctness.cpp   # LRU 淘汰顺序 / 驻留 / 不变量
    └── dt_prefetch_e2e.cpp          # SetBlocksMap → SetTokensMap → SubmitTask 流水线
```

## 构建与运行

```bash
# 在仓库根目录执行
cmake -S . -B build -DBUILD_UCM_SPARSE=ON -DBUILD_UNIT_TESTS=ON \
                 -DRUNTIME_ENVIRONMENT=ascend
cmake --build build -j --target gsa_prefetch_tests
ctest --test-dir build -R gsa_prefetch_tests --output-on-failure

# 行覆盖率 + 分支覆盖率（lcov/genhtml）
cmake --build build --target gsa_prefetch_coverage
```

测试二进制将四个生产源码（`kvcache_pre.cpp`、`kvcache_pre_scatter.cpp`、
`kvcache_task_manger.cpp`、`kvcache_trans_ascend.cpp`）与所有
`ut/*.cpp`、`mst/*.cpp`、`dt/*.cpp` 一起编译。`pybinds.cpp` 被有意排除：
其 `PYBIND11_MODULE` 属于扩展模块；其辅助函数由驱动绑定方法的 Python DT 层
间接覆盖。

## 一键脚本 `run_tests.sh`

`run_tests.sh` 是一个独立的一键脚本，**不依赖 CMake**，自动完成：
依赖探测 → 编译 gtest 静态库 → 预编译头（PCH，加速 torch/pybind11 重型头）
→ 编译 4 个生产源码（带 `--coverage` 插桩）→ 逐个编译+运行测试 →
`gcov -b` 解析行/分支覆盖率 → 汇总打印报告。

```bash
cd ucm/sparse/gsa/prefetch/tests
./run_tests.sh                 # 全量编译运行 + 覆盖率统计
./run_tests.sh --no-cov        # 跳过覆盖率（更快）
./run_tests.sh --clean         # 清理后重新编译
./run_tests.sh -v              # verbose，打印每个用例详情
```

特性：
- **增量编译**：PCH / 生产 .o / 测试 .o 按时间戳缓存，仅重编变更文件。
- **配置变更检测**：`--no-cov` ⇄ 默认（带覆盖率）切换时，自动强制全量重编
  （因为 `--coverage` 插桩与否影响 .o 内容）。
- **崩溃隔离**：用 `timeout --preserve-status` + 子 shell 重定向运行每个
  测试，单个测试崩溃/卡死不会终止脚本（qemu-aarch64 模拟环境下多线程测试
  可能触发 `signal 11`）。
- **报告输出**：用例总数 / 通过数 / 失败数 / 通过率 + 行覆盖率 / 分支覆盖率
  + 逐生产源码文件明细。

> 注：脚本默认在 qemu-aarch64 模拟环境下能跑通全部 314 个用例（0 失败），
> 仅有 `test_prefetch_h2d` 例外：其引擎构造/析构多次循环会累积 ACL/torch
> 全局状态，导致 qemu 不稳定而崩溃（见测试文件头注释）。
> 此外 `kvcache_pre_scatter.cpp` 的重路径（需 TaskManager 线程 + ACL 设备 +
> 共享内存）在该环境下会崩溃，覆盖率受限（见下文“覆盖率”节）。

## 打桩 / mock 策略

| 生产依赖 | 测试替身 | 失败注入方式 |
|---------|---------|--------------|
| `<acl/acl.h>`、`<aclnnop/…>` | `kvcache_trans_ascend_stub.h` | `acl_stub::Stub().memcpyAsyncFails`、`…mallocFails`、`…createStreamFails` 等 |
| `UC::StoreV1`（`mStoreV1_`） | `common/mock_store_v1.h`（`MockStoreV1`） | `loadFails`、`waitFails`、`lookupFails`、`dumpFails`、`checkFails`。* |
| `securec.h` | `common/securec.h`（空） | 无（源码未引用任何 securec 符号） |
| ACL scatter / H2D | 桩 `aclrtMemcpyAsync` 执行真实 `memcpy`；`aclnnInplaceScatter` 为空操作 | `inplaceScatterFails`、`memcpyAsyncFails` |
| torch_npu | 不需要（仅使用 `at::Tensor` / `torch::Tensor`） | 无 |


## 覆盖率

### 实测结果（`run_tests.sh` 默认带 `--coverage`，gcov 解析）

| 生产源码文件 | 行覆盖率 | 分支覆盖率 |
|---|---|---|
| `kvcache_pre.cpp` | 88.96% (911/1024) | 90.55% (882/974) |
| `kvcache_pre_scatter.cpp` | 89.78% (738/822) | 88.61% (934/1054) |
| `kvcache_task_manger.cpp` | 96.39% (80/83) | 93.22% (110/118) |
| `kvcache_trans_ascend.cpp` | 98.00% (98/100) | 100.00% (105/105) |
| `kvcache_hbm_manger.h` | 96.88% (217/224) | 98.23% (222/226) |
| **合计** | **90.72% (2043/2253)** | **90.95% (2252/2477)** |

> ✅ **行覆盖率 90.72% + 分支覆盖率 90.95%，均超过 90% 目标。**
