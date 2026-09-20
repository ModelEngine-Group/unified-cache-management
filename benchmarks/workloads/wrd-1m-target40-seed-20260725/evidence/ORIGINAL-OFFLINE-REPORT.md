# 3379 请求 W17 数据集：F(100 万 token) = 40.6221%

2026-09-20 完成离线生成。枚举 W=1～100 后选择 **W=17**：1332/3279 个有限 WRD 不超过 1,000,000 token，占 **40.6221409%**，距 40% 为 **+0.6221409 个百分点**，满足 38%～42% 目标区间。没有扫描 seed、改动文本、删减请求或调整局部顺序。

| 全量指标 | 原 W32 | 新 W17 |
|---|---:|---:|
| 请求 / trajectory | 3379 / 100 | 3379 / 100 |
| 首次访问 / 有限 WRD | 100 / 3279 | 100 / 3279 |
| F(1,000,000) | 240/3279 = 7.3193% | 1332/3279 = 40.6221% |
| WRD mean | 1,937,369.9787 | 1,040,838.0701 |
| P50 | 2,002,944 | 1,075,200 |
| P90 | 2,620,160 | 1,397,401.6 |
| P95 | 2,743,116.8 | 1,446,656 |
| P99 | 2,934,220.8 | 1,542,681.6 |
| 真正为零的有限 WRD | 4 | 3 |

全部长度单位为 token。分位数采用 `(n-1)*p` 的线性插值，因此不一定是整数。

相邻候选 W16 为 48.9478500%，W18 为 34.7362001%。W17 是所有合法整数中距 40% 最近的结果，也是唯一落入 38%～42% 的候选。搜索未假定单调性：100 个候选之间存在 31 次相邻 W 增大而 CDF 上升，详细记录见 `generated/analysis/comparison.json`。

## 可回放文件与统计产物

交付根目录：

```text
/data/liyuhang/glm51-vllm-ascend-ucm-clean/offline-workloads/wrd-1m-target40-20260920
```

**新数据集目录：**

```text
/data/liyuhang/glm51-vllm-ascend-ucm-clean/offline-workloads/wrd-1m-target40-20260920/generated/closed-loop-w17-64k128k
```

该目录包含 `causal_w17_noprimer.seed-20260725.{order,profiles,intervals}.jsonl`、同 stem 的 `manifest.json`、`BUNDLE-IDENTITY.json` 和 `replay-inputs.json`。后者锁定 **C32、W17、seed=20260725、无 primer、固定 order、等待 depends_on 完成**。没有到达率或跳过依赖的发送策略。

| 文件 | 内容 |
|---|---|
| [search-all-w.csv](generated/analysis/search-all-w.csv) | W1～100 的全量、满额、收尾统计与样本数；包括 mean、P50/P90/P95/P99 和 CDF |
| [w17-per-request-wrd.csv](generated/analysis/w17-per-request-wrd.csv) / [JSON](generated/analysis/w17-per-request-wrd.json) | 3379 行，首次访问 WRD 为 CSV 空值 / JSON null；保留真正的 0 |
| [原 W32 逐请求 WRD](generated/analysis/w32-per-request-wrd.csv) | 同口径原始对照 |
| [comparison.json](generated/analysis/comparison.json) | 最优 W、邻近候选、收尾影响、非单调区间 |
| [validation.json](generated/validation/validation.json) | 独立 WRD、请求集合、依赖、活跃池、原回放器、token 内容的实际校验结果 |
| [prompt-token-hashes.jsonl](generated/validation/prompt-token-hashes.jsonl) | 全部 3379 个含 salt 的实际 token 序列哈希，编码为 uint32 little-endian |
| [independent-process-regeneration.json](generated/validation/independent-process-regeneration.json) | 第二个独立进程生成的关键文件逐字节相同 |
| [SOURCE-LOCK.json](inputs/SOURCE-LOCK.json) | 原始文件绝对路径、独立快照路径、大小和 SHA256 |
| [SHA256SUMS](SHA256SUMS) | 交付文件校验和，不包含工具虚拟环境、缓存和自身 |

图提供 PNG 与 SVG：

![W32 与 W17 的全量 CDF](generated/figures/wrd-cdf-w32-vs-w17.png)

![W17 按请求顺序的 WRD](generated/figures/w17-wrd-by-order.png)

[全部 W 的搜索图](generated/figures/all-w-search.png)。

## 收尾阶段如何影响目标

沿用原 manifest 的阶段划分：**当前请求 `active_sessions_at_emit == W` 为满额阶段，`< W` 为收尾阶段**。收尾不是“第 100 个会话被补入”时立即开始；此后池仍可能保持满额。阶段归属按当前复用请求划分，跨越分界的间隔不截断，仍统计完整中间序列。

| 数据 / 阶段 | 全部请求 | 首次访问 | 有限 WRD | WRD≤100 万 | 阶段 CDF |
|---|---:|---:|---:|---:|---:|
| W32 满额 | 2692 | 98 | 2594 | 39 | 1.5035% |
| W32 收尾 | 687 | 2 | 685 | 201 | 29.3431% |
| W17 满额 | 3053 | 99 | 2954 | 1088 | 36.8314% |
| W17 收尾 | 326 | 1 | 325 | 244 | 75.0769% |

W17 收尾从 **ordinal 3053（第 3054 个请求）** 开始。收尾占有限样本的 9.9116%；全量统计为 `(1088+244)/(2954+325)`，比单独满额阶段高 **3.7907 个百分点**。W32 对应差值为 5.8158 个百分点。没有截掉、延长或修改收尾来达标。

## 输入与生成器核验

原始目录保持只读：

```text
/data/liyuhang/glm51-vllm-ascend-ucm/workload/closed-loop-w32-64k128k
```

原 order SHA256 为 `9cf0b970efa0bc4bb1c755f0ab0cf6d8dfea67eb216c894603064f280eb99392`。order、profiles、intervals、manifest、tier-model 的哈希均与原 BUNDLE-IDENTITY 对上。原 W32 的 3379 行顺序和 3279 行 intervals 已完整重算；order 字节级 SHA256 与原件相同，P50/P90/P99 与给定校验值一致。

现有 `tooling/build_swe_rebench_long_context.py` 的 CLI 属于另一个 10K～128K、均值 40K 的筛选流程，不能直接用其主入口重新选文本。本次仅调用该文件的 **`causal_merge(existing_profiles, BuildConfig(active_sessions=W))`**。函数的原件以相同字节保存到 `vendor/`，SHA256 为 `0fec5d50328db5f539cf090d2eb6a240a5f64b15a531b1e30e57808bed87ac4b`。未使用其中统计未对齐长度的 `_reuse_summary`。

该交错函数先以 `random.Random(20260725)` 打乱原 profiles，初始 W 个会话的 ready time 取 `rng.random()`；后续请求与新补入会话使用 `0.5 + lognormal(log(1.0), 0.6)` 延迟。每个会话只维护下一条待发请求；最后一步发出后，立即按同一打乱列表补入下一会话。堆排序沿用原 ready time 与 tie 序号。新 W 改变随机数调用与事件交错，这是原算法随 W 变化的自然结果。

只有四个 order 字段发生变化：`ordinal`、`virtual_ready_time`、`active_sessions_at_emit`、`unactivated_sessions_remaining`。其余逐请求字段完全一致，包含 request ID、trajectory ID、source row、原 request_index、step_index、正文长度、对齐长度、phase 和 depends_on。profiles 文件逐字节复用。

**salt 仍为 `swe-rebench-w32-noprimer-seed-20260725`。** 其中 `w32` 是原内容命名空间，不能改成 w17，否则请求 token 会变化。实际活跃池由 manifest 和 `--session-pool 17` 指定。

已快照原 compact parquet、tools.json、tokenizer 和 chat template，逐文件哈希不变；未复制模型权重。实际离线调用原回放器 `load_prompt_sources()`，核对了所有原请求的 assistant boundary、正文 token 数和含 salt token 序列，新旧 3379 个请求内容一致。

原 `full_submit_order_sha256 = 29262f...` 实际哈希的是紧凑 JSON ordinal 列表 `[0,1,...,3378]`，无法区分不同 request ID 交错。本数据保留其兼容字段，同时新增 `request_id_sequence_sha256`（request ID 用 LF 连接、末尾无 LF），并以 order 文件 SHA256 核验实际序列。

未生成新的 tier-model。原 W32 tier-model 只保存在 `inputs/baseline/` 作为溯源材料；新数据集没有沿用任何旧 W32 预测值或容量命中统计。

## WRD 计算与验证

每个复用请求找到同 trajectory 的上次请求，排除两个端点；对中间每个 trajectory 取最大 `chunk_aligned_prompt_tokens` 后求和。长度为正文不含 salt 的 `floor(prompt_tokens/256)*256`。首次访问没有有限 WRD，真实相邻复用可为 0。CDF 对 3279 个复用请求等权。

生成器直接扫描每个区间并按 trajectory 取最大值。独立校验器维护每个已出现 trajectory 的开放区间，随事件到达更新其他区间的最大足迹；不复用生成器的 WRD 函数。两种算法对原/新 intervals 的所有字段逐条相等。100 个候选也全部使用第二种算法复核，分位数再与 NumPy 的线性分位数核对。

活跃池校验独立模拟初始打乱、剩余步骤、会话退出和顺序补入，验证每次 emit 的活跃数、未激活数、step_index；还检查请求集合、唯一性、ordinal 连续、同会话顺序和依赖只指向前序请求。

原回放器 `validate_closed_loop_workload()` 已通过。实际 `dispatch_closed_loop()` 以 C32 和本地完成回调处理全部 3379 条，顺序和依赖全部正确；网络连接与 HTTP session 创建在校验进程中被禁止。该 mock 不代表真实并发、延迟或性能。没有启动模型、修改服务配置或发送 HTTP。

## 复现与校验命令

工具环境独立位于本目录 `.venv`，Python 3.12；依赖锁定见 `requirements.lock.txt`，未升级原推理环境。生成本身只用标准库，绘图使用 matplotlib，内容校验使用原 tokenizer 路径、transformers/tokenizers、pyarrow。依赖准备曾下载工具包；数据生成与内容校验不访问网络。

以下命令的工作目录均为交付根目录。生成程序拒绝覆盖已有输出；复现时为 `--output-dir` 选择一个**不存在的绝对路径**。已完成的第二份复现位于 `reproduced/`。

```bash
cd /data/liyuhang/glm51-vllm-ascend-ucm-clean/offline-workloads/wrd-1m-target40-20260920

/data/liyuhang/glm51-vllm-ascend-ucm-clean/offline-workloads/wrd-1m-target40-20260920/.venv/bin/python -B /data/liyuhang/glm51-vllm-ascend-ucm-clean/offline-workloads/wrd-1m-target40-20260920/scripts/generate.py --output-dir /data/liyuhang/glm51-vllm-ascend-ucm-clean/offline-workloads/wrd-1m-target40-20260920/reproduced-next

/data/liyuhang/glm51-vllm-ascend-ucm-clean/offline-workloads/wrd-1m-target40-20260920/.venv/bin/python -B /data/liyuhang/glm51-vllm-ascend-ucm-clean/offline-workloads/wrd-1m-target40-20260920/scripts/validate.py --generated /data/liyuhang/glm51-vllm-ascend-ucm-clean/offline-workloads/wrd-1m-target40-20260920/generated --verify-content

/data/liyuhang/glm51-vllm-ascend-ucm-clean/offline-workloads/wrd-1m-target40-20260920/.venv/bin/python -B /data/liyuhang/glm51-vllm-ascend-ucm-clean/offline-workloads/wrd-1m-target40-20260920/scripts/plot.py --generated /data/liyuhang/glm51-vllm-ascend-ucm-clean/offline-workloads/wrd-1m-target40-20260920/generated

sha256sum --check /data/liyuhang/glm51-vllm-ascend-ucm-clean/offline-workloads/wrd-1m-target40-20260920/SHA256SUMS
```

新 order SHA256：

```text
9e45168e61f540331525edb23738099ca4dee10374c317a33e9b0b3bb012de01
```

manifest 中所有运行输入与输出路径均指向本次交付；原历史路径只存在于溯源快照。跨机器迁移时需更新这些绝对路径，或在同一路径挂载整个交付根目录；不要只拷贝 order 文件。

后续真实回放使用 `vendor/benchmark_swe_rebench_closed_loop.py`，输入参数见 `replay-inputs.json`。`scripts/replay_command.py` **只打印命令，不执行**；现成示例在 `validation/replay-command-example.txt`，其中地址、模型名、run ID 和结果目录仅供未来替换，本次未使用。不要使用原 `31_replay_closed_loop.sh` 原样启动，因为它硬编码 W32 文件和 session pool。真实回放应继续使用已验证的客户端环境，本任务工具环境用于离线验证。

完整使用 C32 不等于保持了相同有效并发：W17 最多有 17 个活跃会话，严格依赖与固定 order 还可能造成队首等待。这个 CDF 只是全局提交顺序下的会话前缀足迹代理，不能解释为每个 DP 的 block/shard 访问距离、实际 L1 命中率或 Early 性能收益。
