# GSA v2 单元测试（UT）可行性分析报告

> 分析对象：`ucm/sandbox/sparse/gsav2/` 下 5 个 Python 文件
> 覆盖率目标：**行覆盖率 100%，分支覆盖率 > 90%**
> 执行环境：**昇腾 NPU 服务器（Linux + torch_npu + vllm + vllm_ascend）**
> 报告日期：2026-10-30

---

## 1. 结论（TL;DR）

**目标可行**，但有 4 个前提条件必须满足：

| # | 前提条件 | 不满足时的后果 |
|---|---------|--------------|
| 1 | Mock 掉 C++ 扩展（`gsa_prefetch`、`torch.ops._C_ascend`）与 vllm/vllm_ascend 重依赖 | UT 无法脱离真实推理进程运行，覆盖率无法度量 |
| 2 | 使用 `importlib.reload` 技巧覆盖 `gsa.py` 平台分支（L30–43） | NPU 机器上 else 分支（L38–43 共 6 行）永不执行 → **行覆盖率上限 ≈ 99.8%，无法达 100%** |
| 3 | 对模块级常量做 monkeypatch 矩阵（`IS_INDEXER_OFFLOAD`、`PTOPK_PREFETCH_ENABLE`、`IS_PERFORMANCE_LOG` 等 10+ 个） | 大量 if 分支只有单侧被覆盖，分支覆盖率跌到 ~70% |
| 4 | 慢超时分支（30s/120s 等待）用假时钟或小 timeout 参数覆盖 | 若干 while/timeout 分支要么测不到，要么单测跑 30+ 分钟 |

预计交付规模：**~160 个用例 / 5 个测试文件 + conftest**，其中 `gsa.py` 占约 60% 用例量。
预计 NPU 环境运行时长 **< 5 分钟**，内存占用 **< 1 GB**（微型 fake 配置）。

---

## 2. 测试对象盘点（AST 实测数据）

| 文件 | 行数 | 类 | 函数/方法 | 分支点估算* | 可测性 |
|------|-----|-----|----------|-----------|--------|
| `gsa.py` | 2809 | 7 | 111 | ~640 | **难**（重依赖 + 大量状态耦合） |
| `gsa_tensor_slot_manager_ds.py` | 117 | 2 | 8 | ~34 | 易（纯 Python） |
| `gsa_shared_memory.py` | 430 | 4 | 22 | ~96 | 中（真实共享内存，Linux 原生支持） |
| `gsa_log.py` | 402 | 2 | 40 | ~48 | 易（纯 Python，文件 IO 用 tmp_path） |
| `prefetch_engine.py` | 132 | 2 | 5 | ~10 | 中（依赖 vllm_ascend + C++ 扩展，须 mock） |
| **合计** | **3890** | 17 | 186 | **~828** | — |

\* 分支点 = if/while/for（各 2）+ 布尔短路子条件（各 2）+ 三元/断言/推导式条件，按 AST 统计。coverage.py 的 arc 口径会略有出入（通常略少），不影响结论。

**关键观察：`gsa.py` 占全部代码量的 72%、分支量的 77%，是覆盖率目标的主战场，工作量约占总工作量 70%。**

---

## 3. 依赖分析

### 3.1 gsa.py / prefetch_engine.py 的外部依赖分类

| 依赖 | 来源 | UT 策略 | 理由 |
|------|-----|---------|------|
| `numpy` / `torch` | pip | **真实** | 基础库，CPU 即可 |
| `vllm.config.VllmConfig` | vllm | **stub**（仅作类型标注 + fake 对象） | 真实构造需要模型权重与完整引擎初始化 |
| `vllm.distributed.*`（get_kv_transfer_group / get_world_group） | vllm | **stub**（可编程 fake） | `_ensure_store_injected` 需要注入各种 connector 形态 |
| `vllm.forward_context.ForwardContext` | vllm | **stub** | 仅作类型 |
| `vllm.utils.torch_utils.make_tensor_with_pad` | vllm | **stub**（或真机可用时用真实） | 仅为 pad 拼接，stub 保证确定性 |
| `vllm.v1.core.sched.output.SchedulerOutput`、`vllm.v1.request.Request/RequestStatus`、`KVCacheBlocks` | vllm | **fake 类** | 用 dataclass/SimpleNamespace 构造可控调度输出 |
| `vllm_ascend.*`（CpuAlloc、AscendDeviceType、get_ascend_device_type、get_sfa_qsfa_packed_head_dim） | vllm_ascend | **stub**（提供 fake CpuAlloc！） | ⚠️ 若只 stub 掉 vllm_ascend 而不提供 fake `CpuAlloc`，`bind_cpus()` 直接 TypeError |
| `torch_npu` | 昇腾 | NPU 环境真实导入；UT 不直接依赖其能力 | `warmup()` 中 `torch.npu.synchronize()` 需 monkeypatch |
| `ucm.logger`（依赖 C 扩展 `ucmlogger`） | ucm | **stub**（`init_logger` → 标准 logging） | 避免依赖 C 日志后端；gsa.py 只用 info/error |
| `ucm.integration.vllm.ucm_connector.RequestHasher` | ucm | **stub**（可编程 block hash 生成器） | 真实实现依赖 vllm |
| `gsa_prefetch.GSAPrefetchEngineC` / `UCMKVPrefetch` | **C++ pybind11 编译产物** | **mock**（记录调用 + 可编程返回值） | UT 不应启动真实预取线程/ACL 上下文 |
| `torch.ops._C_ascend.gsa_gather_copy / gsa_scatter_copy` | **C++ op** | **mock**（monkeypatch `torch.ops._C_ascend`） | 真实 op 需要 ACL stream 上下文 |
| `multiprocessing.shared_memory` | stdlib | **真实** | Linux/NPU 原生支持，且是 `gsa_shared_memory.py` 被测对象本体 |
| NUMA 绑定（`bind_thread_memory_policy`） | ucm.sparse.utils | **monkeypatch**（gsa.py 导入的符号） | 真实实现加载 `libnuma.so.1` 并改线程内存策略，UT 环境不应执行 |

### 3.2 真实保留 vs mock 的边界原则

- **被测逻辑在 Python 层**（调度状态机、slot 映射计算、块表同步、LRU/堆管理、日志路由）→ 全部真实执行；
- **跨进程/跨语言边界**（C++ 引擎、ACL op、NUMA、UCM 存储）→ mock 在边界上，用"调用记录 + 可编程返回值"的 fake 对象。

---

## 4. NPU 环境执行方案

### 4.1 环境要求

| 组件 | 要求 |
|------|------|
| Python | ≥ 3.10（`__pycache__` 显示现网用 3.12） |
| pytest + pytest-cov + coverage | 需在测试环境 pip 安装 |
| torch + torch_npu | 已有（部署机自带） |
| vllm / vllm_ascend | **不强制**——conftest 统一 stub，有无均可运行（保证可移植性与确定性） |

### 4.2 测试套件目录结构（规划）

```
ucm/sandbox/sparse/gsav2/ut/   # 位于被测包内；无 __init__.py（见 §4.3 说明）
├── pytest.ini          # 独立 rootdir：confcutdir 隔离外层 conftest（其依赖 pynvml、common.*）
├── conftest.py         # stub 注入 + fake 对象工厂 + 公共 fixtures
├── fakes.py            # FakeVllmConfig / FakeRequest / FakeSchedulerOutput / FakeConnector ...
├── test_gsa.py                     # gsa.py 主体（~90 用例）
├── test_gsa_slot_manager.py        # gsa_tensor_slot_manager_ds.py（~10 用例）
├── test_gsa_shared_memory.py       # gsa_shared_memory.py（~25 用例）
├── test_gsa_log.py                 # gsa_log.py（~30 用例）
├── test_prefetch_engine.py         # prefetch_engine.py（~7 用例）
└── README.md           # 运行与覆盖率命令
```

> **为何不将测试文件平铺进 gsav2/ 包内**：gsav2 是含 `__init__.py` 的包，
> 平铺后 pytest 会以包模块路径（`ucm.sandbox.sparse.gsav2.conftest`）导入
> conftest，导致 `ucm/__init__.py`（其顶层 `from ucm.integration.vllm.patch
> .logger_patch import patch_logger`）在 stub 安装**之前**执行，纯 CPU 环境
> 直接 ImportError。子目录 `ut/` 不含 `__init__.py`，conftest 按顶层模块
> 导入，stub 注入次序与确定性保持不变。

运行/验收命令（示例）：

```bash
cd ucm/sandbox/sparse/gsav2/ut
pytest --cov=ucm.sandbox.sparse.gsav2.gsa \
       --cov=ucm.sandbox.sparse.gsav2.gsa_tensor_slot_manager_ds \
       --cov=ucm.sandbox.sparse.gsav2.gsa_shared_memory \
       --cov=ucm.sandbox.sparse.gsav2.gsa_log \
       --cov=ucm.sandbox.sparse.gsav2.prefetch.prefetch_engine \
       --cov-branch --cov-report=term-missing --cov-fail-under=90
```

### 4.3 conftest 关键设计

1. **统一 stub 注入**（在 `sys.modules` 中预置，优先于真实包）：
   `vllm.*`、`vllm_ascend.*`、`ucm.logger`、`ucm.integration.vllm.ucm_connector`、`ucm.sandbox.sparse.gsav2.prefetch.gsa_prefetch`。
   NPU 机器上即使装了真实 vllm 也使用 stub → 用例行为跨环境确定。
2. **fake `CpuAlloc`**：提供 `build_cpu_pools/allocate/device_info.running_npu_list/npu_cpu_pool/cpu_node/assign_main`，
   覆盖 `bind_cpus()` 的 3 条路径（空 cpu_pool → None/None；assign_main 充足 → 取尾部 20/24 个；不足 → 取 [1:]）。
3. **微型 FakeVllmConfig**（内存预算的关键）：

   | 参数 | 真实部署（GLM 级） | UT 取值 | 说明 |
   |------|------------------|---------|------|
   | layer_num / topk_layer_num | ~78 | 2–4 | 层数直接乘进所有 buffer |
   | max_model_len | 128K+ | 2048–8192 | 决定 slab/dump buffer 规模 |
   | max_num_seqs (max_bs) | 256 | 4–8 | |
   | block_size | 128 | 16–128 | |
   | num_head / k_dim | 16 / 512 | 1–2 / 16 | |
   | SPA_TOKENS_LEN 等常量 | 5120 等固定值 | monkeypatch 可再调小 | `gsa.MAX_SLOTS_BLOCK_NUM=16`、`gsa.MAX_INDEXER_BLOCK_NUM=16` 等 |

   估算全量测试峰值内存 **< 1 GB**（真实部署参数会到数十 GB，绝不可直接用）。
4. **device 用 `"cpu"` 字符串**：所有 `DSAInputStruct/TopkIndices` 的 "npu" buffer 在 UT 中实际是 CPU 张量（device 字段来自我们可控的 fake config），NPU 机器上同样成立 → 测试不依赖真实 NPU 设备分配。
5. **日志目录隔离**：`GSALogger` 与 prefetch 引擎日志路径是相对路径，用 `monkeypatch.chdir(tmp_path)` 隔离。

### 4.4 模块常量 monkeypatch 矩阵（分支覆盖核心手段）

gsa.py 在 import 时把常量绑定到自身命名空间，逐用例 patch 即可：

| 常量 | 默认值 | 需覆盖的取值 |
|------|-------|-------------|
| `PTOPK_PREFETCH_ENABLE` | True | True / False（所有入口函数的双侧早退） |
| `IS_INDEXER_OFFLOAD` | False | True / False（约 20 处 if） |
| `IS_PERFORMANCE_LOG` | True | True / False（关键流程需双跑） |
| `IS_SHARED_MEMCPY`、`PREFETCH_TYPE`、`STREAM_NUM` 等 | — | 按需 |
| `SEG_PREFILL_THRESHOLD` | 32768 | 配合用例调小（如 64），避免构造超长 prompt |
| `VLLM_ASCEND_VERSION`（allocate_slots 三分支） | "0.18" | "0.18" / "0.13" / "other" |
| `UCM_LOOKUP_MAX_RETRY` | 10000 | patch 为 2，快速触发 lookup 超时分支 |
| `INDEXER_PREFETCH_TOKENS` / `INDEXER_SPARSE_RATE` / `INDEXER_PREFETCH_RATIO` | 32768 等 | 调小以构造 indexer 分支 |
| `VLLM` 侧 flag（`gsa.PTOPK_PREFETCH_ENABLE` 与 `prefetch_engine.PTOPK_PREFETCH_ENABLE` 是两个独立绑定） | — | 分别 patch |

### 4.5 关键测试技术

| 技术 | 解决的问题 |
|------|-----------|
| `importlib.reload` + 临时改 `torch.npu.is_available`/delattr | **100% 行覆盖的唯一硬点**：覆盖 gsa.py L30–43 两侧平台分支（详见 §6-R1） |
| 假时钟（monkeypatch `time.time`/`time.perf_counter`） | `SharedTensorVersion.is_version_same`（硬编码 30s）、`_wait_ranks_version` 超时 |
| 小 `timeout` 构造参数 | `SharedBlockTable(timeout=0.2)` 覆盖 attach 超时 / read 版本等待超时 |
| `pytest.raises` | 2 处 assert（GSASharedCPUPool.allocate、_inject_block_hashes）与 ~6 处 raise |
| 可编程 mock（`side_effect` 脚本化返回序列） | `_load_kv_dram` 的 lookup 重试链、`_ensure_store_injected` 的 5 级重试梯度、prefetch_status 忙等循环 |
| 共享内存唯一命名（uuid 后缀）+ teardown 中 close/unlink | CI 并行/重复运行不互相污染 |
| 状态链用例（execute_begin → get_dsa_input → get_model_input_all → execute_finished） | GSA 主流程的状态耦合只能靠端到端式用例贯穿 |

---

## 5. 逐文件可达性评估

### 5.1 gsa_tensor_slot_manager_ds.py —— ✅ 行 100% / 分支 ~100%，无风险
纯 Python LRU 堆管理。要点：命中/未命中/-1 hash、hash 复用覆盖旧映射、`while...else` 堆耗尽 raise RuntimeError、free 非法 tid 打印、双 free 后再 allocate 的堆回收。约 10 用例。

### 5.2 gsa_log.py —— ✅ 行 100% / 分支 ~100%，无风险
两级日志（Logger / GSALogger）+ 3×4 级别路由。要点：enabled/level 过滤、show_time 双侧、UNKNOWN 级别名、log_type=performance/accuracy/normal/all 四种构造、`is_*_enabled` 属性在 logger 缺失（非 all 模式下访问他类型）时返回 False。约 30 用例（大量是 1 行断言）。

### 5.3 gsa_shared_memory.py —— ✅ 行 100% / 分支 >95%，低风险
4 个类全部可用**真实共享内存**在 Linux 上测。要点：
- creator 正常创建 / FileExistsError→TimeoutError；
- attach 等待循环（先 sleep 后创建，用真实子进程或 `timeout=0.2` 直接触发超时 raise）；
- `write_add_block` / `read_add_block` 的空块跳过、IndexError（超 max_bs）、ValueError（超 max_block_len）、版本等待超时；
- `close` 后 `unlink` 的 re-attach 路径、re-attach FileNotFoundError 静默返回；
- `SharedTensorVersion.update_version` 的 creator 等待全员对齐 / 有 rank 未完成（is_done=0）/ 全完成（is_done=1）三分支 + 等待超时 raise（假时钟或直调 `_wait_ranks_version(version, timeout=0.05)`）；
- `SharedTensorPool.write` 数量不匹配 ValueError、`torch.frombuffer` 视图 reshape 正确性。

风险：`unlink` 后 Windows 与 Linux 的资源跟踪器行为差异 → 最终在 Linux/NPU 验证，无影响。约 25 用例。

### 5.4 prefetch_engine.py —— ✅ 行 100% / 分支 ~95%，低风险
132 行，构造器占 80%。要点：
- `enable_sparse_sfa_c8` False/True × `get_ascend_device_type()` A5/非 A5（stub 可编程返回）4 组合，覆盖 dtype/k_dim/v_dim 与 packed_head_dim 分支；
- `GSAPrefetchEngineC` mock 断言入参列表（12 项配置 + 5 项 dims + log 路径）；
- `update_topk_index` / `del_finish_meta` 的 flag True/False 与 `PTOPK_PREFETCH_ENABLE` 双侧（patch 模块内绑定）。
约 7 用例。

### 5.5 gsa.py —— ⚠️ 行 100% / 分支 >90% 可达，但有 6 个已知硬点，均有对策

**易覆盖部分（~80% 分支）**：`GSABlockTable/GSASharedCPUPool/GSAReqBlockTable/GSAHashRequest` 等小类、`GSAReqStat` 的全部谓词（step/stage/is_gsa/is_indexer_sparse/is_last_chunk×PD混用与PD分离/is_decode_only/get_seq_len/get_spase_len/init_mask/get_indexer_sparse_len/get_indexer_remain_len/add_req_new×preempt分支/update_req_state）、`GSAMetaData.get_model_input/trans_input_tensor`、`compute_parent_block_hash`、`inject_store_v1_ptr`、`print_log`、scheduler 侧接口（`estimate_num_slots_sparsed`×4 分支、`estimate_free_slots_sparse`×3 分支、`allocate_slots`×3 版本分支×preempt 分支、`begin_sch_flag`、`update_scheduler_free_slot` 双分支）。

**中高难度部分与对策**（用例设计清单，节选）：

| 方法 | 分支难点 | 对策 |
|------|---------|------|
| `_find_store_connector` | 普通 connector / `.connector` 二级 / AscendMultiConnector 无 `_connectors` / 遍历命中 `store` / 遍历命中 `.connector.store` / 遍历 miss 共 6 条路径 | fake connector 对象族（attr 可控）逐条覆盖 |
| `_ensure_store_injected` + `_log_store_inject_retry` + `_inject_store_v1` | 5 级重试梯度：已注入直退 / connector=None / 无 store 属性 / store=None / cc_store()==0 / 指针校验失败 raise / 重试上限 raise / 异常路径上限 raise | `get_kv_transfer_group` stub 可编程；`_store_inject_retries` 直接置位触发上限 |
| `_load_kv_dram` | pd_mix 早退 / decode 早退 / 短 prompt 早退 / 首查全中 / 部分未中→重试命中 / 超时 error+return | `ucm_lookup` side_effect 脚本；`UCM_LOOKUP_MAX_RETRY=2`；block_hashes mock 成 16 字节定长 |
| `_calc_decode_slot_mapping` | spec=1 与 spec>1 ×（块内偏移 <spec、==0、else）共 4 组 | 精确构造 `get_seq_len()` 与 `slots` 边界值 |
| `_calc_prefill_slot_mapping` | 命中重算 / dump 范围推导 / slot 不足 error 日志分支 | 控制 `num_computed_tokens`、slots 长度错配 |
| `_collect_*` / `_pre_swap_tokens_all` / `_copy_gsa_buffers_to_npu` | 谓词组合矩阵（is_last_chunk×is_gsa×is_gsa_init×prefetch_token） | 状态注入式用例，`warmup_topk/prefetch_topk/swap_index` 预填 |
| `attention_begin/finished` | PTOPK 早退、layer_id 越界、has_indexer=False、dump_num<=0、li_c8 双路径 | `torch.ops._C_ascend` 整体替换为 SimpleNamespace mock |
| `_sync_block_tables` / `_apply_add_blocks` | rank0/非rank0 × IS_INDEXER_OFFLOAD 4 组合 | SharedBlockTable 用真实 shm 或 mock 均可 |
| `execute_finished` | 无预取分支 / is_prefetch_token 分支 × spec>1 且 rec_result 非 None | `_prepare_prefetch_requests` 的 index 修正逻辑需精确断言 |
| `request_finished_in_worker` / `_release_preempted_gsa_stats` | 状态存在/不存在 × rank0/非0 × slot 非空/空 | 组合用例 |
| `warmup` / `_warmup_dump_blocks` / `_copy_preempt_kv_blocks` / `_inject_block_hashes` | torch.npu.synchronize、preempt slab 拷贝、hash 注入幂等（二次注入早退）、sfa_c8 下 v_refs=[] | monkeypatch synchronize；`_slab_preempt_*` 用微型真张量 |
| `shutdown` | block_table_sm None 早退 / unlink 成功 / unlink 抛异常 | 可编程 mock |
| `init_gsa_metadata` / `init_host_slabs_from_kv` / `_init_*` 系列 | 7 个初始化函数的所有条件组合（role×PTOPK×is_decode_node、c8×A5、spec_config 有无、IS_INDEXER_OFFLOAD） | 构造矩阵用例（fake config 参数化） |

约 90 用例。

---

## 6. 风险清单与对策

| # | 风险 | 等级 | 对策 | 残留影响 |
|---|------|-----|------|---------|
| R1 | **平台分支 L30–43**：NPU 机器上 else 分支 6 行永不执行 → 行覆盖 <100% | 🔴 高 | 用例中 `monkeypatch` `torch.npu.is_available`→False（或临时 delattr `torch.npu`）后 `importlib.reload(gsa模块)`，断言 `torch_npu is None` 等回退值，finally 中 reload 恢复。coverage 按"同一文件多次执行的并集"统计 → 两侧行都计入 | 无。注意 reload 会重置模块级单例（`GSA_INSTANCE`），相关用例需在 reload 之后重新获取模块引用 |
| R2 | **慢超时分支**：`is_version_same` 硬编码 30s、`SharedTensorManager` attach 默认 120s | 🟡 中 | 假时钟 monkeypatch；构造参数传小 timeout；`_wait_ranks_version` 直调小 timeout | 无 |
| R3 | **boolop 短路部分分支**（gsa.py 53 处 and/or） | 🟡 中 | 常量矩阵逐用例 patch（§4.4），入口函数双跑 | 预计仍剩 3–6 个防御性 partial branch（如 `and` 组合中极难同时构造的条件），分支覆盖率预计 95–98%，满足 >90% |
| R4 | **C++ 扩展 mock 保真度**：mock 的 `GSAPrefetchEngineC` 若与真实 C++ 签名漂移，UT 通过但集成失败 | 🟡 中 | conftest 中 mock 签名按 `pybinds.cpp` 逐一核对；标注为 UT 局限，不在 UT 范围内验证 C++ 行为 | 属 UT 固有边界 |
| R5 | **外层 test/conftest.py 依赖 pynvml/common.***，与 UT 目录冲突 | 🟢 低 | UT 目录内置独立 `pytest.ini` 作为 rootdir，confcutdir 隔离，不加载外层 conftest | 无 |
| R6 | **共享内存残留/名称冲突**（重复运行、CI 并行） | 🟢 低 | 名称加 uuid 后缀；fixture teardown 中 close+unlink；失败用例也保证清理 | 无 |
| R7 | **Windows 开发机上 pin_memory=True 行为与 Linux 不同** | 🟢 低 | 最终验收在 NPU/Linux；本地仅做语法级检查 | 无（以 NPU 环境为准） |
| R8 | `estimate_free_slots_sparse` 等调度侧接口需要 `slots_num_manager`（SharedTensorManager 真实 shm） | 🟢 低 | 用真实 SharedTensorManager（每用例独立 shm），或 mock `read_num` | 无 |

**分支覆盖率测算**：gsa.py ~640 分支点，按 R1–R3 对策执行后预计 missed/partial ≤ 16 个 → 分支覆盖率 ≥ 97%；全 5 文件聚合预计 ≥ 97%，**满足 >90% 目标**。

---

## 7. 工作量与交付物估算

| 文件 | 用例数 | 估算工时（人日） |
|------|-------|----------------|
| conftest + fakes（脚手架） | — | 1.5 |
| test_gsa.py | ~90 | 4–5 |
| test_gsa_slot_manager.py | ~10 | 0.5 |
| test_gsa_shared_memory.py | ~25 | 1 |
| test_gsa_log.py | ~30 | 0.5 |
| test_prefetch_engine.py | ~7 | 0.5 |
| NPU 环境联调 + 覆盖率调优到 100%/90%+ | — | 1–2 |
| **合计** | **~160** | **~9–11 人日** |

覆盖率调优阶段（最后 1–2 人日）是达标的必要投入：`--cov-report=term-missing` 驱动补齐遗漏行/分支。

---

## 8. 验收标准（建议）

1. NPU 环境 `pytest --cov-branch` 输出：5 个文件 **stmts miss = 0**（行覆盖 100%）；
2. 各文件分支覆盖率 **> 90%**（预计可达 95%+）；
3. 全量用例运行 **< 5 分钟**，无共享内存残留（`/dev/shm` 无 `gsa_*`、`psm_*` 残留段）；
4. 用例可重复运行（幂等），无环境依赖。

---

## 9. 结论

- **行覆盖率 100%：可达**，唯一硬点是 gsa.py 平台分支（L30–43），用 reload 技巧解决（§6-R1）。
- **分支覆盖率 >90%：可达**（预计 95%+），依赖常量 monkeypatch 矩阵 + 假时钟 + 可编程 mock 三项技术。
- 前提是接受 §3.1 的 mock 边界：C++ 扩展、ACL op、NUMA、vllm 重依赖在 UT 中均被 mock，其正确性由 C++ 侧已有 gtest 套件（`prefetch/tests/ut/`）与集成测试保障——这与本仓库"Python UT + C++ UT 分层"的既有实践一致。

---

## 10. 补充结论：是否必须 NPU（事后评估）

**UT 的开发调试完全不需要 NPU；NPU 仅用于最终验收运行。**

理由：

1. 被 UT 验证的是 **Python 侧控制流**（分支、状态机、共享内存协议、配置解析）。
   真正的 NPU 相关依赖（ACL op `torch.ops._C_ascend`、C++ 扩展
   `gsa_prefetch.GSAPrefetchEngineC/UCMKVPrefetch`、`torch_npu`）在 UT 中
   **本来就全部 mock**（§3.1 的既定边界）——即使坐在 NPU 机器上，被测路径
   也不触碰任何真实 NPU 能力。因此 CPU 与 NPU 上执行的是**同一份代码逻辑**。
2. `multiprocessing.shared_memory` 在 Windows/Linux 语义一致（Linux 原生
   `/dev/shm`），gsa_shared_memory.py 的 48 个用例已直接在 CPU 上验证。
3. conftest 的 stub 注入是**无条件**的（fake `torch.npu`、覆盖
   `torch.ops._C_ascend`），保证任意机器上 import 分支（gsa.py L30–43）走
   True 侧、结果确定；False 侧由 reload 用例覆盖。NPU 环境验收时行为与
   CPU 完全一致，无集群状态依赖、无设备竞争。
4. 唯一的 CPU 兼容处理是 pin_memory 降级（torch ≥ 2.x 在无加速器时对
   `pin_memory=True` 抛 RuntimeError）；UT 中 C++ 端是 mock、不解引用
   data_ptr，锁页内存与被测逻辑无关，NPU 上该降级同样无害。

**实测结果（Windows CPU，torch 2.15.0+cpu）**：366 用例全部通过，
运行 ~4s；行覆盖 100%（5 文件 stmts miss = 0），分支覆盖 99.2%（>90%）。
NPU 验收只需将本目录拷贝到服务器仓内同路径，`python3 -X utf8 -m pytest -q`
复核同样的指标即可（详见 README.md）。
