# 结合pytest的UC-Eval使用

## 前置准备

### 数据集准备

**文档问答**数据集：

| 数据集       | 链接                                                         |
| ------------ | ------------------------------------------------------------ |
| gsm8k        | [http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/gsm8k.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/gsm8k.zip) |
| LongBench    | [zai-org/LongBench · Datasets at Hugging Face](https://huggingface.co/datasets/zai-org/LongBench) |
| LongBench v2 | [zai-org/LongBench-v2 · Datasets at Hugging Face](https://huggingface.co/datasets/zai-org/LongBench-v2) |

**多轮对话**数据集：

| 数据集                       | Hugging Face 链接                                            |
| ---------------------------- | ------------------------------------------------------------ |
| ShartGPT                     | [anon8231489123/ShareGPT_Vicuna_unfiltered · Datasets at Hugging Face](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) |
| ShartGPT-Chinese-English-90K | [shareAI/ShareGPT-Chinese-English-90k · Datasets at Hugging Face](https://huggingface.co/datasets/shareAI/ShareGPT-Chinese-English-90k) |

多轮对话数据集格式可参照如下两种形式：

- 格式1：
  - 顶层键名（如 `"sharegpt"`）可以自定义，但内部结构必须保持一致
  - `"conversations"` 字段名不可修改
  - 对话必须采用 `"from"` 和 `"value"` 格式

```json
{
  "sharegpt": [
    {
      "conversations": [
        {
          "from": "human",
          "value": "Write a definition of \"photoshop\"."
        },
        {
          "from": "gpt",
          "value": "Photoshop is a software application developed by Adobe that enables users to manipulate digital images by providing a variety of tools and features to alter, enhance, and edit photos. It allows users to adjust the color balance, contrast, and brightness of images, remove backgrounds, add or remove elements from images, and perform numerous other image manipulation tasks. Photoshop is widely used by graphic designers, photographers, and digital artists for creating and enhancing images for a variety of purposes, including print and online media."
        }
    ]
}]}
```

- 格式2：

```json
[
    {
        "id": "dsOTKpn_0",
        "conversations": [
            {
                "from": "human",
                "value": "Why does `dir` command in DOS see the \"<.<\" argument as \"\\*.\\*\"?"
            },
            {
                "from": "human",
                "value": "I said `dir \"<.<\"` , it only has one dot but it is the same as `dir \"\\*.\\*\"`"
            }
        ]
    },
    {
        "id": "60493",
        "conversations": [
            {
                "from": "human",
                "value": "我想用TypeScript编写一个程序，提供辅助函数以生成G代码绘图（Marlin）。我已经在我的3D打印机上添加了笔座，并希望将其用作笔绘图仪。该库应提供类似使用p5.js的体验，但它不是在画布上绘制形状，而是在G代码中产生文本输出。"
            }
        ],
        "lang": "en"
    }
]
```

### stopwords文件

- 用途：在计算f1-score时使用
- **下载地址**：[GitHub - goto456/stopwords: 中文常用停用词表（哈工大停用词表、百度停用词表等）](https://github.com/goto456/stopwords)
- **放置位置**：以 `cn_stopwords.txt` 文件为例，下载后将其放置在 `test/common/uc_eval/utils` 目录下，并重命名为 `stopwords.txt`

## 日志配置

- **默认日志路径**：`test/common/uc_eval/uc_log`
- **日志级别设置**：通过环境变量 `UC_LOG_LEVEL` 控制

```shell
# 设置 DEBUG 级别日志
export UC_LOG_LEVEL=DEBUG

# 设置 INFO 级别日志（默认）
export UC_LOG_LEVEL=INFO
```

## 性能测试

### 基础配置

- **运行文件**：`test/suites/E2E/test_uc_performance.py`
- **配置文件**：`test/config.yaml`
- `config.yaml` 配置说明：

```yaml
models:
  ip_ports: "0.0.0.0:10045"
  tokenizer_path: "/home/models/Qwen3-32B"
  served_model_name: "Qwen3-32B"
  payload: '{"stream": True, "ignore_eos":True, "temperature":0}'
  enable_clear_hbm: false
  max_seq_length: 128000
```

**重要参数说明**：

- `enable_clear_hbm`：是否启用 HBM 清理接口。MindIE 模型无此接口，必须设为 `false`
- `max_seq_length`：模型最大序列长度。输入超过此长度时，会按 `input_ids[:max_seq_length//2] + input_ids[-max_seq_length//2:]` 截断

### 虚拟数据性能测试

- **运行命令**：

```python
python -m pytest --feature=sync_perf_test
```

- **结果保存位置**：所有性能测试数据保存在：`uc_eval/results/reports/{benchmark_mode}/synthetic_latency.xlsx`
- **参数配置说明**：

| 参数                  | 含义                     | 示例值                              |
| :-------------------- | :----------------------- | :---------------------------------- |
| `data_type`           | 数据类型（固定值）       | `"synthetic"`                       |
| `enable_prefix_cache` | 是否进行请求预热         | `true`/`false`                      |
| `parallel_num`        | 请求并发数列表           | `[1, 4, 8]`                         |
| `prompt_tokens`       | 输入长度列表（tokens）   | `[4096, 8192]`                      |
| `output_tokens`       | 输出长度列表（tokens）   | `[1024, 1024]`                      |
| `prefix_cache_num`    | 缓存命中率列表（0-1）    | `[0.8, 0.8]`                        |
| `benchmark_mode`      | 性能统计模式             | `"default-perf"` 或 `"stable-perf"` |
| `kv_hit_type`         | KV缓存命中类型           | `"HBM"` 或 `"DISK"`                 |
| `epoch_num`           | 重复测试次数（取平均值） | `1、5`等                            |
| `test_name`           | 存储在表中的唯一标识符   | `"no gsa and enable prefix cache"`  |

**性能统计模式详解：**

- **default-perf 模式**

  - **说明**：直接统计请求的性能数据

  - **示例**：2并发8K请求 → 发送2条8K请求，统计这2条请求的性能

- **stable-perf 模式**

  - **说明**：统计稳态性能，仅虚拟数据支持

  - **示例**：2并发8K请求 → 发送2×5=10条8K请求，统计并发稳定在2时的性能

**KV缓存命中配置**：

- **HBM命中**

  - 设置config.yaml中`enable_clear_hbm: false`
  - 设置 `kv_hit_type: "HBM"`
  - 设置合适的命中率 `prefix_cache_num`

- **DISK命中**

  - **vLLM模型**：`enable_clear_hbm: true`，发送请求后清理显存，实现DISK命中

  - **MindIE模型**：`enable_clear_hbm: false`，需要两次服务拉起和请求发送

**示例代码：**

```python
sync_perf_cases = [
    pytest.param(
        PerfConfig(
            data_type="synthetic",
            enable_prefix_cache=False,
            parallel_num=[1, 4, 8],
            prompt_tokens=[4000, 8000],
            output_tokens=[1000, 1000],
            benchmark_mode="default-perf",
            kv_hit_type="HBM",
            epoch_num=5,
            test_name="no gsa and no prefix cache"
        ),
    ),
    pytest.param(
        PerfConfig(
            data_type="synthetic",
            enable_prefix_cache=True,
            parallel_num=[1, 4, 8],
            prompt_tokens=[4000, 8000],
            output_tokens=[1000, 1000],
            prefix_cache_num=[0.8, 0.8],
            benchmark_mode="stable-perf",
            kv_hit_type="HBM",
            epoch_num=5,
            test_name="no gsa and enable prefix cache"
        ),
    ),
]


@pytest.mark.feature("sync_perf_test")
@pytest.mark.stage(2)
@pytest.mark.parametrize("perf_config", sync_perf_cases)
@export_vars
def test_sync_perf(
    perf_config: PerfConfig, model_config: ModelConfig
):
    file_save_path = config_instance.get_config("reports").get("base_dir")
    task = SyntheticPerfTask(model_config, perf_config, file_save_path)
    result = task.run()
    return {"_name": perf_config.test_name, "_proj": result}
```

### 多轮对话性能测试

- **运行命令**：

```python
python -m pytest --feature=dialogue_perf_test
```

- **结果保存位置**：所有性能测试数据保存在：`uc_eval/results/reports/default-perf/multi_turn_dialogue_latency.xlsx`
- **参数配置说明**：
  - **dataset_file_path**：多轮对话数据集中multiturndialog.json所在路径，支持绝对路径及相对路径

```python
multiturn_dialogue_perf_cases = [
    pytest.param(
        PerfConfig(
            data_type="multi_turn_dialogue",
            dataset_file_path="datasets/multi_turn_dialogues/multiturndialog.json",
            enable_prefix_cache=False,
            parallel_num=1,
            benchmark_mode="default-perf",
            test_name="shartgpt and no prefix cache"
        ),
    )
]


@pytest.mark.feature("dialogue_perf_test")
@pytest.mark.stage(2)
@pytest.mark.parametrize("perf_config", multiturn_dialogue_perf_cases)
@export_vars
def test_multiturn_dialogue_perf(
    perf_config: PerfConfig, model_config: ModelConfig
):
    file_save_path = config_instance.get_config("reports").get("base_dir")
    task = MultiTurnDialogPerfTask(model_config, perf_config, file_save_path)
    result = task.run()
    return {"_name": perf_config.test_name, "_data": result}
```

- multiturndialog.json格式如下：

```json
{
    "demo": [
        "demo.json"
    ],
    "sharegpt": [
        "demo.json"
    ]
}
```

- 说明：
  - 键名（如 `"demo"`）表示数据集文件夹名称
  - 值列表表示该文件夹下的数据文件名称

### 文档问答性能测试

- **运行命令**：

```python
python -m pytest --feature=qa_perf_test
```

- **结果保存位置**：所有性能测试数据保存在：`uc_eval/results/reports/default-perf/doc_qa_latency.xlsx`
- **参数配置说明**：
  - **dataset_file_path**：文档问答数据集所在路径

```python
doc_qa_perf_cases = [
    pytest.param(
        PerfConfig(
            data_type="doc_qa",
            dataset_file_path="datasets/doc_qa/demo.jsonl",
            enable_prefix_cache=False,
            parallel_num=1,
            benchmark_mode="default-perf",
            test_name="longbench and no prefix cache"
        ),
    )
]

@pytest.mark.feature("qa_perf_test")
@pytest.mark.stage(2)
@pytest.mark.parametrize("perf_config", doc_qa_perf_cases)
@export_vars
def test_doc_qa_perf(
    perf_config: PerfConfig, model_config: ModelConfig
):
    file_save_path = config_instance.get_config("reports").get("base_dir")
    task = DocQaPerfTask(model_config, perf_config, file_save_path)
    result = task.run()
    return {"_name": perf_config.test_name, "_data": result}
```

### 性能测试核心指标解读

| 指标              | 说明                                                         |
| ----------------- | ------------------------------------------------------------ |
| `Total Latench`   | 所有请求处理时间，统计第一条请求开始的时间到最后一条请求结束的时间 |
| `E2E TPS`         | 端到端每秒生成的token数，计算公式：`输出tokens数 / 所有请求耗时`，也即output_tokens * parallel_num / total_latency，稳态测试时计算公式为`位于稳态的请求条数 * output_tokens / total_latency` |
| `Per Request TPS` | 对于每条请求而言，每秒生成的token数，计算公式：`mean(单条请求输出tokens数 / 单条请求处理耗时)` |
| `TTFT`            | Time to First Token，反应prefill阶段耗时                     |
| `TBT`             | Time Between Token，decode阶段相邻两个token输出时间的时间间隔 |
| `TPOT`            | Time Per Output Tokens，所有token生成时间间隔的平均值，计算公式：`decode时间 / 输出tokens数` |

## 精度测试

### 基础配置

- **运行文件**：`test/suites/E2E/test_evaluator.py`
- **配置文件**：`test/config.yaml`
- `config.yaml` 配置说明：

```yaml
models:
  ip_ports: "0.0.0.0:10045"
  tokenizer_path: "/home/models/Qwen3-32B"
  served_model_name: "Qwen3-32B"
  payload: '{"temperature":0, "max_tokens":1024, "ignore_eos":false}'
  enable_clear_hbm: false
  max_seq_length: 128000
```

- 注意：精度测试需要在 `payload` 中配置生成参数（`max_tokens`、`ignore_eos`、`temperature` 等）

### 文档问答性能测试

- **运行命令**：

```python
python -m pytest --feature=qa_eval_test
```

- **结果保存位置**：所有性能测试数据保存在：`uc_eval/results/reports/evaluate/doc_qa_latency.xlsx`，同时，在evaluate目录下会生成一个以日期命名的文件夹，其中包含数据集和模型回复等信息
- **参数配置说明**：

| 参数                  | 含义                   | 示例值                                           |
| :-------------------- | :--------------------- | :----------------------------------------------- |
| `data_type`           | 数据类型（固定值）     | `"doc_qa"`                                       |
| `dataset_file_path`   | 文档问答数据集路径     | `"datasets/doc_qa/demo.jsonl"`                   |
| `enable_prefix_cache` | 是否进行请求预热       | `true`/`false`                                   |
| `parallel_num`        | 请求并发数             | `1`                                              |
| `benchmark_mode`      | 精度统计模式           | `"evaluate"`                                     |
| `metrics`             | 评估指标列表           | `["accuracy", "bootstrap-accuracy", "f1-score"]` |
| `eval_class`          | 答案匹配策略           | `"common.uc_eval.utils.metric:FuzzyMatch"`       |
| `select_data_class`   | 数据筛选条件           | `{"domain": ["Single-Document QA"]}`             |
| `test_name`           | 存储在表中的唯一标识符 | `"no gsa and enable prefix cache"`               |

- 实际运行配置示例：

```python
doc_qa_eval_cases = [
    # longbench v2参考配置
    pytest.param(
        EvalConfig(
            data_type="doc_qa",
            dataset_file_path="datasets/doc_qa/demo_2.json",
            enable_prefix_cache=False,
            parallel_num=1,
            benchmark_mode="evaluate",
            metrics=["accuracy", "bootstrap-accuracy", "f1-score"],
            eval_class="common.uc_eval.utils.metric:MatchPatterns",
            select_data_class={"domain": ["Single-Document QA"]},
            test_name="longbench v2 and no prefix cache"
        ),
    ),
    # longbench参考配置
    pytest.param(
        EvalConfig(
            data_type="doc_qa",
            dataset_file_path="datasets/doc_qa/demo.jsonl",
            enable_prefix_cache=False,
            parallel_num=1,
            benchmark_mode="evaluate",
            metrics=["f1-score"],
            eval_class="common.uc_eval.utils.metric:FuzzyMatch",
            test_name="longbench and no prefix cache"
        ),
    ),
]


@pytest.mark.feature("qa_eval_test")
@pytest.mark.stage(2)
@pytest.mark.parametrize("eval_config", doc_qa_eval_cases)
@export_vars
def test_doc_qa_perf(
    eval_config: EvalConfig, model_config: ModelConfig
):
    file_save_path = config_instance.get_config("reports").get("base_dir")
    task = DocQaEvalTask(model_config, eval_config, file_save_path)
    result = task.run()
    return {"_name": eval_config.test_name, "_data": result}
```

- 不同**匹配策略（eval_class）**区别如下，其路径：test/common/uc_eval/utils/metric.py

| 策略                | 类名                 | 匹配规则                                                     |
| :------------------ | :------------------- | :----------------------------------------------------------- |
| **完全匹配**        | `Match`              | 模型输出必须与参考答案完全一致                               |
| **包含匹配**        | `Includes`           | 模型输出包含参考答案内容即匹配                               |
| **模糊匹配**        | `FuzzyMatch`         | 支持两种模式： 1. `substring`：双向包含匹配 2. `jaccard`：相似度 > 0.8 匹配 |
| **模板匹配**        | `MatchPatterns`      | 根据正则表达式模板提取答案后匹配                             |
| **模板匹配之gsm8k** | `GSM8KMatchPatterns` | 根据正则表达式模板从gsm8k数据集中提取答案进行匹配            |

- **MatchPatterns**方法介绍：
  - **适用场景**：longbench v2等需要从 A/B/C/D 中选择正确答案的数据集
  - **模板文件**：test/common/uc_eval/utils/prompt_config.py

```python
# 文档问答数据集的语言，决定后续的分词方式，以及后续prompt具体使用中文还是英文. 具体使用时首先会读取数据集中是否存在language这个键，如果不存在才使用该配置
# 可选值包含三个: en, zh, None
DEFAULT_LANGUAGE = "None"

# 文档问答提示模板，在使用时会将{}占位符替换为数据集中键值对应的内容，包含英文prompt和中文prompt两种形式
Q&A prompt for document QA – replace the {} placeholders with actual content from the dataset when used.
doc_qa_prompt_zh = [
    """
    阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：
    """
]

doc_qa_prompt_en = [
    """
    Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:
    """
]

# 多项选择题提示模板
COT_KEY = "COT"
multi_answer_prompt = [
    """
    Please read the following text and answer the questions below.\n
    Text: {context}\n
    What is the correct answer to this question: {question}\n
    Choices: \n (A) {choice_A} \n (B) {choice_B} \n (C) {choice_C} \n (D) {choice_D} \n 
    Let's think step by step. Based on the above, what is the single, most likely answer choice?\n
    Format your response as follows: "The correct answer is (insert answer here)'
"""
]

# 答案提取正则表达式模板
match_patterns_longbench_v2 = [
    r"The correct answer is \(([A-D])\)",
    r"The correct answer is ([A-D])",
    r"The \(([A-D])\) is the correct answer",
    r"The ([A-D]) is the correct answer",
]

match_patterns_gsm8k = [
    r"(?i)answer:?\s*(-?[€£¥$]?\d[\d,]*(?:\/\d+|\.\d+)?)(%?)",
    r"(?i)The answer is (-?[€£¥$]?\d[\d,]*(?:\/\d+|\.\d+)?)(%?)",
]
```

- **prompt_config模板使用说明**：
  - `{}` 中的标签必须与数据集中的字段名对应
  - LongBench 和 LongBench v2 的问题字段名不同（分别为 `input` 和 `question`），需在模板中正确使用
  - `multi_answer_prompt` 可以包含多个提示模板（如 COT 推理过程），框架会按顺序发送请求，在使用COT推理时，需要在第一次的prompt中加入第一次prompt的response，COT_KEY表示在multi_answer_prompt中response对应的键，在获取到第一次response后，会将prompt中的COT_KEY替换为实际的response

- 采用MatchPatterns模式时，多项选择题处理流程：
  - 使用 `multi_answer_prompt` 中的模板构造提示
  - 发送请求获取模型回复
  - 使用 `match_patterns` 中的正则表达式提取答案（A/B/C/D）
  - 与数据集的参考答案进行比对，获取精度或者F1-score