# AISBench简介
对标MLPerf等国际先进计算基准，电子标准院依托新一代人工智能标准与应用工业和信息化部重点实验室孵化人工智能系统性能基准AISBench，配套研发了标准符合性检测工具。AISBench是一套应用于人工智能计算产品的性能测试基准，该基准包含测试标准、测试大纲及配套的测试工具。此基准测试对象包含多种人工智能计算产品形态，覆盖人工智能加速卡、服务器、集群、人工智能计算中心等多种计算产品形态的性能测试。

目前AISBench工具支持的测试类目覆盖常规中小模型负载的训\推性能，大模型负载（参数量分布从十亿到千亿）的训\推性能测试；支持通用、专用场景、开放、封闭、训练、推理等多种测试场景、模式及类型；支持运行时间、功耗、吞吐率、资源利用率、能效等多项指标。同时，可由半自动测评到全自动化测评，支持一键运行，插件式开发，支持不中断业务升级，支持多路并发与实时状态管理，支持多框架（Pytorch，MindSpore，Tensorflow）多设备测试；采用CS架构，Tester与Stubs之间采用证书+websocket进行加密通信，可以实现运行训前检查、运行测试、过程监控、结果和性能数据归集汇总、生成测试报告全流程。

我们的大模型测试均基于AISBench，并适配UCM实现了相应的性能测试与精度测试。

## 公共测评参数

| 参数 | 说明 |
|---|---|
| `temperature` | 模型生成随机性的采样参数，值越低输出越确定保守，为保证可复现性取极小值 0.01 |
| `ignore_eos` | 忽略 EOS 提前结束，强制生成到 `max_out_len`，保证输出长度可控；性能测试为 `true`，精度测试为 `false` |
| `CUSTOM_TEST_IMAGE` | GLM-5.3 等模型使用镜像 `<私有registry>/aisbench_benchmark:v3.1_x86_64_py_310-transformersv5`；<br/>其他情况使用 `<私有registry>/aisbench_benchmark:v3.1_x86_64_py_310` |

## 性能测试
### 数据集构建

本轮性能测试的数据集基于 AISBench 的 Prefix Cache 测试场景构造，核心目标是生成总输入长度可控、预埋比例可控、内容可复现的请求集合，以便在 PD 分离 + UCM 部署下观测前缀缓存命中带来的性能收益。

### 构造方法

数据集采用“公共前缀 + 私有前缀 + 私有后缀”三段式结构，并输出三份 JSONL 文件分别服务于预埋与正式测试两个阶段：

| 文件 | 内容构成 | 用途 |
|---|---|---|
| `common.jsonl` | 公共前缀（所有请求共享） | 全局共享前缀预埋 |
| `prefill.jsonl` | 公共前缀 + 私有前缀 | 预埋阶段：将可复用前缀写入 KV Cache |
| `full.jsonl` | 公共前缀 + 私有前缀 + 私有后缀 | 正式测试：前缀部分预期命中，仅后缀需计算 |

其中，**预埋率**定义为可预埋前缀长度占总输入长度的比例。当公共前缀为 0 时，预埋率由私有前缀与私有后缀的拆分比例实现，即正式请求中约约定比例的输入为已预埋、可命中的前缀部分，其余部分为需实时计算的后缀部分。

### 构造参数

数据集以六元组描述一组测试用例，依次为：

**`[input_len, output_len, num_requests, hit_rate, concurrency, seed]`**

各参数含义如下：

| 参数 | 说明 |
|---|---|
| `input_len` | 总输入长度（token 数），决定正式请求的输入规模；造数时按预埋率拆分为可预埋前缀与需计算后缀两部分 |
| `output_len` | 输出长度（token 数），由 AISBench 压测配置侧承接，控制每条请求的生成长度 |
| `num_requests` | 请求数量，即该用例下正式测试的请求条数 |
| `hit_rate` | 预埋率，即输入中可预埋前缀所占比例，决定私有前缀与私有后缀的拆分比例 |
| `concurrency` | 并发数，由 AISBench 压测配置侧承接，控制压测时的并发调度 |
| `seed` | 随机种子，保证数据集内容可复现 |

其中，`input_len`、`num_requests`、`hit_rate`、`seed` 直接作用于数据集构造，决定输入规模、请求条数、前缀/后缀拆分与内容复现；`output_len` 与 `concurrency` 不参与造数，由 AISBench 压测配置承接，分别控制生成长度与并发调度。

### 预埋与正式测试

数据集使用遵循“先预埋、后正式”的两阶段流程：

1. **存储预埋**：以预埋数据集运行 AISBench 性能模式，将各请求的可复用前缀写入 KV Cache；
2. **正式测试**：以完整数据集运行性能测试，前缀部分预期命中缓存，仅私有后缀触发实际计算；
3. **命中观测**：在预埋前后及正式测试前后采集各推理 pod 的缓存命中率，结合时延与吞吐指标，对比验证 UCM 前缀缓存的实际收益。

该构造方式使“已预埋前缀”与“新增后缀”在同一请求内可分离，从而将 KV Cache 命中收益从整体时延中剥离出来单独观测，为 PD 分离架构下 UCM 的性能评估提供可控、可复现的输入基础。

对于每一组构造数据集，测试分为两轮 AISBench 运行，中间夹一次命中率快照，形成「预埋 → 记录基线 → 正式测试 → 记录结果 → 对比」的闭环。

```text
① 存储预埋（prefill.jsonl）
        ↓
② 对接检查 + 保存 hit_rate_before.json
        ↓
③ 正式测试（full.jsonl）
        ↓
④ 保存 hit_rate_after.json
        ↓
⑤ 前后命中率对比（hbm / ext）
```

每一轮AISBench均会输出如下指标。

| 指标 （该表给出每个指标的 **Average / Min / Max / Median / P75 / P90 / P99 / N** 统计）| 含义 |
|---|---|
| `E2EL` | 端到端时延（End-to-End Latency）|
| `TTFT` | 首 token 时延（Time To First Token）|
| `TPOT` | 每输出 token 时延（Time Per Output Token） |
| `ITL` | token 间时延（Inter-Token Latency） |
| `InputTokens` | 单条请求的输入 token 数 |
| `OutputTokens` | 单条请求的输出 token 数 |
| `OutputTokenThroughput` | 输出 token 吞吐|

| 指标 | 含义 |
|---|---|
| `Benchmark Duration` | 整轮测试耗时，从开始到全部请求结束 |
| `Total Requests` | 总请求数 |
| `Failed Requests` | 失败请求数 |
| `Success Requests` | 成功请求数 |
| `Concurrency` | 实际平均并发数 |
| `Max Concurrency` | 配置的最大并发数 |
| `Request Throughput` | 请求吞吐|
| `Total Input Tokens` | 输入 token 总数 |
| `Prefill Token Throughput` | Prefill 阶段 token 吞吐|
| `Total Generated Tokens` | 生成 token 总数 |
| `Input Token Throughput` | 输入 token 吞吐|
| `Output Token Throughput` | 输出 token 吞吐 |
| `Total Token Throughput` | 总 token 吞吐（输入+输出）|

正式测试前后各保存一次快照（`hit_rate_before.json` / `hit_rate_after.json`），终端输出对比表，列出缓存命中情况。

| 指标 | 含义 |
|---|---|
| `endpoint` | 被测服务端点（IP:Port） |
| `hbm_hit_rate` | HBM 缓存命中率 |
| `hbm(hit/query)` | HBM 缓存命中次数 / 查询次数 |
| `ext_hit_rate` | 外部存储（如 UCM 扩展缓存）命中率 |
| `ext(hit/query)` | 外部存储命中次数 / 查询次数 |
| `TOTAL` | 所有 endpoint 的汇总行 |

## 精度测试

### 数据集

精度测试必须**显式选定标准数据集**，不能像性能测试那样随意使用自定义 JSONL。AISBench 为常见数据集提供配置模板，模板通常包含：

- **数据集本身**：如 ARC、GPQA、C-Eval、GSM8K 等；
- **shot 数**：few-shot 示例数量，如 0-shot、4-shot、25-shot；
- **提示形式**：普通提示或思维链（CoT）提示；
- **接口格式**：chat 格式或普通 completion 格式。

例如配置名 `gsm8k_gen_4_shot_cot_chat_prompt` 表示：GSM8K 数据集、生成式模式、4-shot、CoT 提示、chat 格式。

### 评测模式

常见模式为 `gen`（生成式）：模型自由生成答案，再由评测器按指标判分。不同数据集可能适配不同模式，具体以数据集配置为准。

### 评测指标

| 指标 | 含义 |
|---|---|
| `accuracy` | 准确率，答对样本数 / 总样本数，最常用的单数据集指标 |
| `exact_match` | 精确匹配，输出与标准答案完全一致才算对 |
| `naive_average` | 简单平均，对分组内各子集指标取算术平均 |
| `weighted_average` | 加权平均，按子集样本量加权 |

单数据集通常只输出 `accuracy` 或 `exact_match`；多科目数据集（如 C-Eval）会在汇总阶段额外输出分组和总体的平均值指标，用于横向比较。

### 执行流程
```
① 加载模型配置 + 数据集配置 + summarizer
        ↓
② 创建实验目录
        ↓
③ 推理阶段：按数据集批量请求被测服务，收集回答
        ↓
④ 评测阶段：按数据集指标逐条打分
        ↓
⑤ 汇总阶段：生成 summary（txt / csv / md）
```
