# 结合pytest的UC-Eval使用

## uc-eval性能测试

- 运行文件：test/suites/E2E/test_uc_performance.py
- 配置文件：test/config.yaml
  - enable_clear_hbm：清理HBM的接口，MindIE中无此类接口，必须设置为false
  - max_seq_length：模型的maxSeqLen，在测试时如果输入长度大于该数据，会按照input_ids[:max_seq_length//2] + input_ids[-max_seq_length//2:]对数据进行截取

```yaml
models:
  ip_ports: "0.0.0.0:10045"
  tokenizer_path: "/home/models/Qwen3-32B"
  served_model_name: "Qwen3-32B"
  payload: '{"stream": True, "ignore_eos":True, "temperature":0}'
  enable_clear_hbm: false
  max_seq_length: 128000
```

### 虚拟数据性能测试

- 运行：**python -m pytest --feature=sync_perf_test**
- 运行完后，所有性能测试数据保存在：**uc_eval/results/reports/{benchmark_mode}/synthetic_latency.xlsx**
- 配置如下：

|        参数         |                             含义                             |
| :-----------------: | :----------------------------------------------------------: |
|      data_type      |                   运行的数据形式，无需修改                   |
| enable_prefix_cache |        测试时是否开启prefix-cache，开启后才会存在命中        |
|    parallel_num     |                          请求并发数                          |
|    prompt_tokens    |                    每个并发下输入长度列表                    |
|    output_tokens    |       每个并发下输出长度列表，和prompt_tokens一一对应        |
|  prefix_cache_num   |               命中率，和prompt_tokens一一对应                |
|   benchmark_mode    |       性能统计方式，包含default-perf和stable-perf两种        |
|     kv_hit_type     | 存在命中率情况下kvcache具体命中HBM还是DISK，需要和config.yaml文件中models的enable_clear_hbm一起配置 |
|      epoch_num      | 在性能评估中对epoch_num次数据取平均，以1并发8K输入为例，会分别统计5次1并发8K的性能数据，对其取平均，防止性能波动 |

- 不同benchmark_mode的区别，以2并发下8K的请求输入为例：
  - **default-perf**模式：发送2条8K的请求，并统计2条请求的性能
  - **stable-perf**模式：stable-perf表示稳态性能统计，在stable-perf模式下则会发送**2*5条**8K的请求，统计10条请求中**并发数稳定在2时**的请求性能，仅对synthetic数据存在该模式
- **kv_hit_type**使用：
  - HBM命中的情况下，直接将enable_clear_hbm设置为false，并且设置命中率为非0值即可
  - DISK命中的情况下，针对vllm模型，可以将enable_clear_hbm设置为True，针对MindIE模型，enable_clear_hbm需要设置为False，并进行两次服务拉起和两次请求发送，第二次性能结果即为DISK命中的性能

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
        ),
        id="benchmark-complete-recalculate-default-perf",
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
        ),
        id="benchmark-prefix-cache-stable-perf",
    ),
]

@pytest.mark.feature("sync_perf_test")
@pytest.mark.stage(2)
@pytest.mark.parametrize("perf_config", sync_perf_cases)
@export_vars
def test_sync_perf(
    perf_config: PerfConfig, model_config: ModelConfig, request: pytest.FixtureRequest
):
    file_save_path = config_instance.get_config("reports").get("base_dir")
    task = SyntheticPerfTask(model_config, perf_config, file_save_path)
    result = task.run()
    return {"_name": request.node.callspec.id, "_proj": result}
```

### 多轮对话性能测试

- 运行：**python -m pytest --feature=dialogue_perf_test**
- 运行完后，每轮对话的详细性能及整体性能数据保存在：**uc_eval/results/reports/default-perf/multi_turn_dialogue_latency.xlsx**
- 配置如下：
  - **dataset_file_path**：多轮对话数据集中multiturndialog.json所在路径，可以是绝对路径及相对路径，相对路径是从uc_eval下的路径开始寻找

```python
multiturn_dialogue_perf_cases = [
    pytest.param(
        PerfConfig(
            data_type="multi_turn_dialogue",
            dataset_file_path="uc_eval/datasets/multi_turn_dialogues/multiturndialog.json",
            enable_prefix_cache=False,
            parallel_num=1,
            benchmark_mode="default-perf",
        ),
        id="multiturn-dialogue-complete-recalculate-default-perf",
    )
]

@pytest.mark.feature("dialogue_perf_test")
@pytest.mark.stage(2)
@pytest.mark.parametrize("perf_config", multiturn_dialogue_perf_cases)
@export_vars
def test_multiturn_dialogue_perf(
    perf_config: PerfConfig, model_config: ModelConfig, request: pytest.FixtureRequest
):
    file_save_path = config_instance.get_config("reports").get("base_dir")
    task = MultiTurnDialogPerfTask(model_config, perf_config, file_save_path)
    result = task.run()
    return {"_name": request.node.callspec.id, "_data": result}
```

- multiturndialog.json格式如下：
  - 该文件记录了在多轮对话中将会遍历的多轮对话数据集路径，其中sharegpt_changed表示在multiturndialog.json同级目录下存在名为sharegpt_changed的文件夹，对应的list中为多轮对话数据集的文件名

```json
{
    "sharegpt_changed": [
        "demo.json"
    ],
    "memorybank": [

    ]
}
```

- demo.json格式应参照：

```json
{
  "sharegpt": [
    {
      "conversations": [
        {
          "role": "human",
          "content": "Write a definition of \"photoshop\"."
        },
        {
          "role": "gpt",
          "content": "Photoshop is a software application developed by Adobe that enables users to manipulate digital images by providing a variety of tools and features to alter, enhance, and edit photos. It allows users to adjust the color balance, contrast, and brightness of images, remove backgrounds, add or remove elements from images, and perform numerous other image manipulation tasks. Photoshop is widely used by graphic designers, photographers, and digital artists for creating and enhancing images for a variety of purposes, including print and online media."
        }
    ]
}]}
```

### 文档问答性能测试

- 运行：**python -m pytest --feature=qa_perf_test**
- 运行完后，每篇文档的详细性能及整体性能数据会保存在：**uc_eval/results/reports/default-perf/doc_qa_latency.xlsx**
- 配置如下：
  - dataset_file_path：文档问答数据集所在路径

```python
doc_qa_perf_cases = [
    pytest.param(
        PerfConfig(
            data_type="doc_qa",
            dataset_file_path="uc_eval/datasets/doc_qa/demo.jsonl",
            enable_prefix_cache=False,
            parallel_num=1,
            benchmark_mode="default-perf",
        ),
        id="doc-qa-complete-recalculate-default-perf",
    )
]

@pytest.mark.feature("qa_perf_test")
@pytest.mark.stage(2)
@pytest.mark.parametrize("perf_config", doc_qa_perf_cases)
@export_vars
def test_doc_qa_perf(
    perf_config: PerfConfig, model_config: ModelConfig, request: pytest.FixtureRequest
):
    file_save_path = config_instance.get_config("reports").get("base_dir")
    task = DocQaPerfTask(model_config, perf_config, file_save_path)
    result = task.run()
    return {"_name": request.node.callspec.id, "_data": result}
```

## uc-eval精度测试

- 运行文件：test/suites/E2E/test_evaluator.py
- 配置文件：test/config.yaml，payload配置中需要包含max_tokens等配置

```yaml
models:
  ip_ports: "0.0.0.0:10045"
  tokenizer_path: "/home/models/Qwen3-32B"
  served_model_name: "Qwen3-32B"
  payload: '{"temperature":0, "max_tokens":1024, "ignore_eos":false}'
  enable_clear_hbm: false
  max_seq_length: 128000
```

### 文档问答性能测试

- 运行：**python -m pytest --feature=qa_eval_test**
- 运行完后，每篇文档的详细回复结果及匹配情况会记录在：**uc_eval/results/reports/evaluate/doc_qa_latency.xlsx**
- 配置如下：

| 参数                | 含义                                                         |
| ------------------- | ------------------------------------------------------------ |
| data_type           | 运行的数据形式，无需修改                                     |
| dataset_file_path   | 文档问答数据集所在路径，需要运行多个文件时，在cases中配置多个用例即可 |
| enable_prefix_cache | 测试时是否开启prefix-cache，开启后会先进行一次预热           |
| parallel_num        | 请求并发数                                                   |
| benchmark_mode      | 精度统计方式                                                 |
| metrics             | 性能测试评估指标，目前包含三种，可根据需要自定义配置         |
| eval_class          | 进行精度测试时采用的匹配策略，包含四种，分别是完全匹配Match，包含Includes，模糊匹配FuzzyMatch，以及模板匹配MatchPatterns |
| select_data_class   | 从数据集中挑选数据，以{"domain": ["Single-Document QA"]}为例，会从数据中按照domain挑选Single-Document QA的数据进行测试，为空时不会进行数据筛选 |

- 实际运行时的参考配置：

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
            select_data_class={"domain": ["Single-Document QA"], "difficulty": []}
        ),
        id="doc-qa-complete-recalculate-evaluate",
    )
    
    # longbench v1参考配置
    pytest.param(
        EvalConfig(
            data_type="doc_qa",
            dataset_file_path="datasets/doc_qa/demo.jsonl",
            enable_prefix_cache=False,
            parallel_num=1,
            benchmark_mode="evaluate",
            metrics=["accuracy", "bootstrap-accuracy", "f1-score"],
            eval_class="common.uc_eval.utils.metric:FuzzyMatch",
        ),
        id="doc-qa-complete-recalculate-evaluate",
    )
]


@pytest.mark.feature("qa_eval_test")
@pytest.mark.stage(2)
@pytest.mark.parametrize("eval_config", doc_qa_eval_cases)
@export_vars
def test_doc_qa_perf(
    eval_config: EvalConfig, model_config: ModelConfig, request: pytest.FixtureRequest
):
    file_save_path = config_instance.get_config("reports").get("base_dir")
    task = DocQaEvalTask(model_config, eval_config, file_save_path)
    result = task.run()
    return {"_name": request.node.callspec.id, "_data": result}
```

- 不同eval_class的区别：
  - 路径：test/common/uc_eval/utils/metric.py
  - **common.uc_eval.utils.metric:Match**：用例的answers和模型回复保持完全一致才认为匹配成功 
  - **common.uc_eval.utils.metric:Includes**：模型输出包含了answers中的内容认为匹配成功
  - **common.uc_eval.utils.metric:FuzzyMatch**：默认使用substring模式。对于substring模式，模型输出包含answers或者answers中包含模型输出下认为匹配成功，jaccard模式下answers和模型输出达到一定相似性（默认0.8）认为匹配成功
  - **common.uc_eval.utils.metric:MatchPatterns**：根据提供的模板进行答案提取后，在进行匹配匹配，下面详细介绍

- MatchPatterns方法介绍：
  - **场景**：longbench v2类似数据集，文档问答需要回答具体选择A/B/C/D
  - **模板文件**：test/common/uc_eval/utils/prompt_config.py，其中**doc_qa_prompt**为非多项选择的prompt，**multi_answer_prompt**为多项选择文档的prompt，**match_patterns**为精度计算时从中提取答案，与数据集中answer进行对比的模板
  - doc_qa_prompt和multi_answer_prompt中**{}**包裹的标签需要能和数据集中的标签相对应，比如longbench和longbench v2中问题对应的标签分别为input和question，prompt中Question中也填入这两个标签
  - **multi_answer_prompt**介绍：以longbench v2数据集为例，在测试时其prompt可为其数据集中的0shot对应的内容，或0shot_cot与0shot_cot_ans的组合，为后者时，将两个prompt均存在multi_answer_prompt列表中，进行请求发送时，会先发送0shot_cot对应的请求，然后发送0shot_cot_ans的请求，并以后者的输出去计算精度等性能

```python
# Q&A prompt for document QA – replace the {} placeholders with actual content from the dataset when used.
doc_qa_prompt = ["""
    Please read the following text and answer the questions below.\n
    Text: {context}\n
    Question: {input}
    Instructions: Answer based ONLY on the information in the text above
"""]

multi_answer_prompt = ["""
    Please read the following text and answer the questions below.\n
    Text: {context}\n
    What is the correct answer to this question: {question}\n
    Choices: \n (A) {choice_A} \n (B) {choice_B} \n (C) {choice_C} \n (D) {choice_D} \n 
    Let's think step by step. Based on the above, what is the single, most likely answer choice?\n
    Format your response as follows: "The correct answer is (insert answer here)'
"""]

match_patterns = [
    r'The correct answer is \(([A-D])\)',
    r'The correct answer is ([A-D])',
    r'The \(([A-D])\) is the correct answer',
    r'The ([A-D]) is the correct answer'
]
```

## 日志配置

- 日志默认保存在**test/common/uc_eval/uc_log**路径下
- 可以通过在命令行配置**UC_LOG_LEVEL**去设置日志级别

```shell
# DEBUG日志
export UC_LOG_LEVEL=DEBUG

# INFO级别日志
export UC_LOG_LEVEL=INFO
```

