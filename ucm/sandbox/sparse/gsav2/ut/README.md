# gsav2 单元测试套件（UT）

针对 `ucm/sandbox/sparse/gsav2/` 下 5 个 Python 模块的单元测试：

| 被测模块 | 测试文件 | 用例数 |
|---|---|---|
| `gsa.py` | `test_gsa.py` | 254 |
| `gsa_shared_memory.py` | `test_gsa_shared_memory.py` | 48 |
| `gsa_log.py` | `test_gsa_log.py` | 48 |
| `gsa_tensor_slot_manager_ds.py` | `test_gsa_slot_manager.py` | 16 |
| `prefetch/prefetch_engine.py` | `test_prefetch_engine.py` | 11 |

合计 **366 个用例**，本机（CPU）实测约 4s 跑完。

## 覆盖率结果（pytest-cov --cov-branch）

| 模块 | 语句 | 行覆盖 | 分支覆盖 |
|---|---|---|---|
| gsa.py | 1431 | **100%** | 99% |
| gsa_shared_memory.py | 270 | **100%** | 100% |
| gsa_log.py | 148 | **100%** | 99% |
| gsa_tensor_slot_manager_ds.py | 66 | **100%** | 98% |
| prefetch_engine.py | 61 | **100%** | 100% |
| **合计** | 1976 | **100%** | **99.2%** |

> 目标：行覆盖 100%、分支覆盖 >90%。剩余少量部分分支均为 `while` 循环退出转移等
> 不可构造路径（如忙等循环的自然退出侧已在超时用例中覆盖）。

## 运行方式

### 本地 CPU（开发调试）

```bash
cd UCM_feacture_gsa_v2/ucm/sandbox/sparse/gsav2/ut
py -3 -X utf8 -m pytest            # Windows（PATH 无 python 时用 py -3）
python3 -X utf8 -m pytest          # Linux
```

依赖：`torch`（CPU 版即可）、`numpy`、`pyyaml`、`pytest`、`pytest-cov`。

- `pytest.ini` 已内置 `--cov-branch --cov-report=term-missing --cov-fail-under=90`。
- 调试单文件：`py -3 -X utf8 -m pytest test_gsa.py -q --no-cov -p no:cacheprovider`

### NPU 服务器（最终验收）

```bash
# 1. 将本目录整体拷贝到 NPU 服务器仓内同一路径
#    UCM_feacture_gsa_v2/ucm/sandbox/sparse/gsav2/ut
# 2. 环境：torch NPU 版 + numpy pyyaml pytest pytest-cov
python3 -X utf8 -m pytest -q
# 3. 检查输出：所有用例 passed，且 term-missing 表中 5 个模块 Miss 均为 0
```

**分别查看行覆盖率 / 分支覆盖率**：pytest 表格里的 `Cover%` 是语句+分支
的**合并**综合值；若需要两个维度分开统计（含达标判定），用一键脚本：

```bash
python3 -X utf8 run_ut.py
# 完整 pytest 输出存档到 ut_result.log（验收留证用），
# 终端额外打印各模块 + 总计的行覆盖率、分支覆盖率明细表
```

**跨环境确定性说明**：`conftest.py` 在任何机器上都向 `sys.modules` 注入
vllm / vllm_ascend / torch_npu / ucm.logger / gsa_prefetch(C++ 扩展) 的 stub，
并覆盖 `torch.ops._C_ascend`。即 NPU 环境上被测的仍是**纯 Python 逻辑**，
不依赖真实 ACL op / C++ 扩展 / NPU 驱动，结果与 CPU 完全一致。
这样做的收益是：验收不依赖集群状态、无设备竞争、可在 CI 上跑。
若需在 NPU 上做真实链路验证，请使用 E2E 套件，不在本 UT 范围内。

## 测试架构

```
ut/                      # 位于 ucm/sandbox/sparse/gsav2/ 下（无 __init__.py，
                         #   保证 conftest 按顶层模块导入，stub 先于 ucm 包链执行）
├── pytest.ini          # 独立 rootdir（confcutdir 隔离外层 E2E conftest）+ cov 配置
├── conftest.py         # stub 注入 / pin_memory 降级 / make_gsa fixtures
├── fakes.py            # FakeVllmConfig / FakeEngineC / FakeUCMPrefetch / ...
├── test_gsa.py         # gsa.py（254 用例，按功能分组 10 余个 Test 类）
├── test_gsa_shared_memory.py  # 真实 multiprocessing.shared_memory
├── test_gsa_log.py
├── test_gsa_slot_manager.py
├── test_prefetch_engine.py
├── FEASIBILITY.md      # 可行性分析报告
└── test_gsa_worker.py  # 旧草稿（--ignore 排除，见下）
```

关键设计：

1. **Mock 边界**：C++ 扩展（`GSAPrefetchEngineC`/`UCMKVPrefetch`）、
   `torch.ops._C_ascend`、vllm/vllm_ascend 全树、`ucm.logger` 用 stub；
   被测 Python 逻辑真实执行。
2. **共享内存**：`gsa_shared_memory.py` 用**真实** `multiprocessing.shared_memory`
   （uuid 唯一命名 + 强制 close/unlink）；GSA 用例中则用进程内 Fake 替身，
   避免跨用例段污染。
3. **平台分支**（gsa.py L30-43）：conftest 保证 import 走 True 侧（fake `torch.npu`），
   reload 用例覆盖 False 侧。
4. **模块常量**（`IS_INDEXER_OFFLOAD`/`PTOPK_PREFETCH_ENABLE`/
   `IS_PERFORMANCE_LOG`/`SEG_PREFILL_THRESHOLD`/`VLLM_ASCEND_VERSION`/
   `UCM_LOOKUP_MAX_RETRY` 等）逐用例 monkeypatch 构造分支矩阵。
5. **微型配置**：FakeVllmConfig（layer=2、max_model_len=2048、max_bs=4、
   block=128、k_dim=16）+ `MAX_SLOTS_BLOCK_NUM/MAX_INDEXER_BLOCK_NUM` patch 为 16，
   常驻内存 < 1GB。
6. **CPU 兼容**：torch ≥ 2.x 在无加速器时对 `pin_memory=True` 直接抛 RuntimeError，
   conftest 将 `torch.zeros/empty` 的 pin_memory 无条件降级为 False
   （UT 中 C++ 端是 mock，不解引用 data_ptr，锁页内存与被测逻辑无关）。
7. **慢路径**：读版本等待用小 timeout；`is_version_same` 超时用假时钟 +
   sleep patch，避免真实 30s 等待。

## 发现的源码问题（测试中以 workaround 覆盖，建议修复）

1. **`indexer_cache_preempt_i_ptr` 属性名不一致**（gsa.py）
   `_init_indexer_preempt`（L2673-2678）设置的是 `indexer_cache_preempt_ptr/_offset`，
   但 `_inject_block_hashes`（L1028-1029）读取 `indexer_cache_preempt_i_ptr/_i_offset`
   —— 后者在任何路径下都不会被赋值。offload + inject_indexer 路径会直接
   `AttributeError`。测试中手动补设 `_i_` 属性以覆盖该分支。
   （`test_gsa.py::TestWarmup::test_warmup_dump_offload_indexer_copy`、
   `TestInjectBlockHashes::test_injection_with_indexer`）

2. **`GSA._inject_store_ptr` 方法不存在**（gsa.py）
   模块级 `inject_store_v1_ptr`（L111）在 `GSA_INSTANCE` 非空时调用
   `GSA_INSTANCE._inject_store_ptr(...)`，但 GSA 类未定义该方法，
   一旦实例存在即触发 `AttributeError`。测试用独立 fake 对象覆盖模块级函数。

3. **`indexer_last_query` 构造后被覆盖为 None**（gsa.py）
   `_init_worker_indexer_state` 中 `self.indexer_last_query = None`（L685）
   执行在 `self.init_gsa_metadata()`（L672，内部创建 TopkIndices）**之后**，
   导致构造完成后 `indexer_last_query/indexer_last_weight` 恒为 None，
   `cal_indexer_topk_blocks` 主路径会 `AttributeError`。
   测试中手动重建这两个 TopkIndices。
   （`test_gsa.py::TestIndexer::_setup_topk`）

4. **`build_sparse_meta` 向 `_load_kv_dram` 传对象而非 req_id**（gsa.py L2391-2392）
   `for req_id in scheduler_output.scheduled_new_reqs: self._load_kv_dram(req_id)`
   —— 迭代得到的是 NewRequestData 对象，而 `_load_kv_dram` 以 req_id 为键查
   `gsa_metadata.gsa_stats`，类型不匹配（真实 vllm 的 dataclass 同样不可哈希，
   会 TypeError/KeyError）。测试用自引用对象（`req_id` 指向自身）兼容两者以覆盖
   该行。修复建议：改为 `self._load_kv_dram(new_req.req_id)`。
   （`test_gsa.py::TestSchedulerSide::test_build_sparse_meta_new_reqs_and_has_prefill`）

## 其他说明

- **`test_gsa_worker.py`**：早期草稿（依赖不存在的 `make_worker` 等 fixtures），
  与 `test_gsa.py` 覆盖重复，已在 `pytest.ini` 中 `--ignore` 排除。如需恢复，
  删除该行即可。
- **PD 语义**：PD 分离（decode 节点，consumer-only）与 PD 混合
  （producer+consumer）两套语义均有用例矩阵覆盖；
  `is_last_chunk`/`is_decode_only`/`prefetch_trigger_output` 在两种模式下取值不同，
  测试断言按模式分别校验。
- **Windows 差异**：`SharedTensorPool.unlink()` 不关闭句柄，Windows 上段销毁由
  resource_tracker 异步完成，相关用例不对此做同步断言（Linux/NPU 上行为一致）。
