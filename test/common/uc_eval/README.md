# 结合pytest的UC-Eval使用

## 前置准备

### 数据集准备

**文档问答**数据集：

| 数据集       | Hugging Face 链接                                            |
| ------------ | ------------------------------------------------------------ |
| AIME2025     | [opencompass/AIME2025 · Datasets at Hugging Face](https://huggingface.co/datasets/opencompass/AIME2025) |
| LongBench    | [zai-org/LongBench · Datasets at Hugging Face](https://huggingface.co/datasets/zai-org/LongBench) |
| LongBench v2 | [zai-org/LongBench-v2 · Datasets at Hugging Face](https://huggingface.co/datasets/zai-org/LongBench-v2) |

**多轮对话**数据集：

| 数据集                       | Hugging Face 链接                                            |
| ---------------------------- | ------------------------------------------------------------ |
| ShartGPT                     | [anon8231489123/ShareGPT_Vicuna_unfiltered · Datasets at Hugging Face](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) |
| ShartGPT-Chinese-English-90K | [shareAI/ShareGPT-Chinese-English-90k · Datasets at Hugging Face](https://huggingface.co/datasets/shareAI/ShareGPT-Chinese-English-90k) |

- 多轮对话数据集格式参照如下：

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

**注意**：

- 顶层键名（如 `"sharegpt"`）可以自定义，但内部结构必须保持一致
- `"conversations"` 字段名不可修改
- 对话必须采用 `"from"` 和 `"value"` 格式

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
| `enable_prefix_cache` | 是否开启前缀缓存         | `true`/`false`                      |
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

  - **优势**：消除性能波动，获得更稳定的性能数据

**KV缓存命中配置**：

- **HBM命中**

  - 设置config.yaml中`enable_clear_hbm: false`
  - 设置 `kv_hit_type: "HBM"`
  - 设置合适的命中率 `prefix_cache_num`

- **DISK命中**

  - **vLLM模型**：`enable_clear_hbm: true`，进行服务重启

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
    "sharrgpt":[
        
    ] 
}
```

- 说明：
  - 键名（如 `"demo"`）表示数据集文件夹名称
  - 值列表包含该文件夹下的数据文件名称
  - 相对路径从 `uc_eval` 目录开始计算

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

## 精度测试

### 基础配置

- **运行文件**：`test/suites/E2E/test_uc_performance.py`
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

- **结果保存位置**：所有性能测试数据保存在：`uc_eval/results/reports/evaluate/doc_qa_latency.xlsx`
- **参数配置说明**：

| 参数                  | 含义                   | 示例值                                           |
| :-------------------- | :--------------------- | :----------------------------------------------- |
| `data_type`           | 数据类型（固定值）     | `"doc_qa"`                                       |
| `dataset_file_path`   | 文档问答数据集路径     | `"datasets/doc_qa/demo.jsonl"`                   |
| `enable_prefix_cache` | 是否开启前缀缓存       | `true`/`false`                                   |
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
            test_name="longbench and no prefix cache"
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
            metrics=["accuracy", "bootstrap-accuracy", "f1-score"],
            eval_class="common.uc_eval.utils.metric:FuzzyMatch",
            test_name="longbench v2 and no prefix cache"
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

| 策略         | 类名            | 匹配规则                                                     |
| :----------- | :-------------- | :----------------------------------------------------------- |
| **完全匹配** | `Match`         | 模型输出必须与参考答案完全一致                               |
| **包含匹配** | `Includes`      | 模型输出包含参考答案内容即匹配                               |
| **模糊匹配** | `FuzzyMatch`    | 支持两种模式： 1. `substring`：双向包含匹配 2. `jaccard`：相似度 > 0.8 匹配 |
| **模板匹配** | `MatchPatterns` | 根据正则表达式模板提取答案后匹配                             |

- **MatchPatterns**方法介绍：
  - **适用场景**：longbench v2等需要从 A/B/C/D 中选择正确答案的数据集
  - **模板文件**：test/common/uc_eval/utils/prompt_config.py

```python
# 非多项选择题提示模板
doc_qa_prompt = ["""
    Please read the following text and answer the questions below.\n
    Text: {context}\n
    Question: {input}
    Instructions: Answer based ONLY on the information in the text above
"""]

# 多项选择题提示模板
multi_answer_prompt = ["""
    Please read the following text and answer the questions below.\n
    Text: {context}\n
    What is the correct answer to this question: {question}\n
    Choices: \n (A) {choice_A} \n (B) {choice_B} \n (C) {choice_C} \n (D) {choice_D} \n 
    Let's think step by step. Based on the above, what is the single, most likely answer choice?\n
    Format your response as follows: "The correct answer is (insert answer here)'
"""]

# 答案提取正则表达式模板
match_patterns = [
    r'The correct answer is \(([A-D])\)',
    r'The correct answer is ([A-D])',
    r'The \(([A-D])\) is the correct answer',
    r'The ([A-D]) is the correct answer'
]
```

- **prompt_config模板使用说明**：
  - `{}` 中的标签必须与数据集中的字段名对应
  - LongBench 和 LongBench v2 的问题字段名不同（分别为 `input` 和 `question`），需在模板中正确使用
  - `multi_answer_prompt` 可以包含多个提示模板（如 CoT 推理过程），框架会按顺序发送请求

- 采用MatchPatterns模式时，多项选择题处理流程：
  - 使用 `multi_answer_prompt` 中的模板构造提示
  - 发送请求获取模型回复
  - 使用 `match_patterns` 中的正则表达式提取答案（A/B/C/D）
  - 与数据集的参考答案进行比对，获取精度