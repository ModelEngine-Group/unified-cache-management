## Accuracy testing of Sparse method

### Overview
We use two Chinese subsets of  [LongBench](https://huggingface.co/datasets/zai-org/LongBench) to test the accuracy of single-document QA (multifieldqa_zh) and multi-document QA (dureader). The F1 score is adopted to evaluate the accuracy of these sparse methods. For more information about LongBench, please refer to https://github.com/THUDM/LongBench.

We also support testing with [LongBench-v2](https://huggingface.co/datasets/zai-org/LongBench-v2) dataset, which uses Accuracy as the evaluation metric.

### Quick Start

#### Environment Preparation
```shell
# For offline inference
pip install jieba fuzzywuzzy rouge

# For online inference (additional packages)
pip install requests tqdm
```

#### Test Data Preparation
Download the Longbench and Longbench-v2 datasets:

```shell
# Download LongBench and put multifieldqa_zh.jsonl and dureader.jsonl to ./eval/data/longbench/
wget https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip && unzip data.zip

# Download LongBench-v2 and put data.json to ./eval/data/longbench-v2/
wget https://huggingface.co/datasets/zai-org/LongBench-v2/resolve/main/data.json -O ./eval/data/longbench-v2/data.json
```



### Offline Inference Testing

Offline inference tests run the model directly using vLLM with sparse attention support. The KV cache is stored locally and managed by the UCM system.


#### Configure Specific Sparse Method

Settings for different sparse methods are written in a JSON file, for example:
```json
{"ESA": 
    {
    "init_window_sz": 1,
    "local_window_sz": 2,
    "min_blocks": 4,
    "sparse_ratio": 0.2,
    "retrieval_stride": 10
    }
}
```

#### 1. LongBench F1 Score Testing

**Script:** `eval_offline_longbench_F1.sh`

This script tests on LongBench dataset (multifieldqa_zh, dureader) and evaluates using F1 score.

```shell
cd eval

# Run with default settings: Qwen2.5-14B-Instruct batch=20
bash eval_offline_longbench_F1.sh

# Run with custom parameters
bash eval_offline_longbench_F1.sh \
    --model /home/models/QwQ-32B \
    --config ./ucm_sparse_config_esa.json \
    --data ./data/longbench \
    --strip_think 1 \
    --batch 1
```

**Parameters:**
- `--model PATH`: Path to model directory (default: `/home/models/Qwen2.5-14B-Instruct/`)
- `--config PATH`: Path to UCM sparse config JSON file (default: `./ucm_sparse_config_esa.json`)
- `--data PATH`: Path to test data directory (default: `./data/longbench`)
- `--strip_think 0|1`: Whether to extract text after `</think>` from model predictions (default: 0)
- `--batch INT`: Number of requests processed per batch (default: 20)

**Output:** Results are saved in `eval/ucm_sparse_predictions/LongBench/` with F1 scores in `.f1.txt` files.

#### 2. LongBench-v2 Accuracy Testing

**Script:** `eval_offline_longbench_v2_acc.sh`

This script tests on LongBench-v2 dataset and evaluates using Accuracy metric.
NOTE: By default, prompts will be truncated to max_model_len. Setting --no_truncate will disable prompt truncation.
Setting max_context_length will filter the context length for inference based on this value.

```shell
cd eval

# Run with default settings
bash eval_offline_longbench_v2_acc.sh

# Run with custom parameters
bash eval_offline_longbench_v2_acc.sh \
    --model /home/models/Qwen3-32B \
    --storage ./ucm_kv_cache \
    --config ./ucm_sparse_config_esa.json \
    --dataset ./data/longbench-v2/data.json \
    --template ./prompts/0shot.txt \
    --save_dir ./ucm_sparse_predictions/longbench_v2 \
    --domain "Single-Document QA" \
    --max_tokens 16384 \
    --batch_size 20 \
    --cot \
    --max_samples 100
```

**Parameters:**
- `--model PATH`: Path to model directory (default: `/home/models/Qwen3-32B`)
- `--storage PATH`: Path to KV cache storage directory (default: `./ucm_kv_cache`)
- `--config PATH`: Path to UCM sparse config JSON file (default: `./ucm_sparse_config_esa.json`)
- `--dataset PATH`: Path to dataset file (default: `./data/longbench-v2/data.json`)
- `--template PATH`: Path to prompt template file (default: `./prompts/0shot.txt`)
- `--save_dir PATH`: Directory to save results (default: `./ucm_sparse_predictions/longbench_v2`)
- `--domain STR`: Domain to filter, e.g., 'Single-Document QA' (default: `Single-Document QA`)
- `--max_tokens INT`: Maximum tokens to generate (default: 16384)
- `--batch_size INT`: Batch size for inference (default: 20)
- `--max_context_length INT`: Maximum context length (optional)
- `--cot`: Enable Chain-of-Thought evaluation (default: enabled)
- `--max_samples INT`: Maximum number of samples to process (optional)
- `--resume`: Resume from existing output file

**Output:** Results are saved in the specified `--save_dir` directory, and accuracy analysis is automatically performed.

### Online Inference Testing

Online inference tests send requests to an LLM service via HTTP API (OpenAI-compatible). This is useful when the model is running as a separate service.

#### 1. LongBench F1 Score Testing (Online)

**Script:** `eval_online_longbench_F1.sh`

This script tests on LongBench dataset via API and evaluates using F1 score.

```shell
cd eval

# Run with default settings
bash eval_online_longbench_F1.sh

# Run with custom parameters
bash eval_online_longbench_F1.sh \
    --model Qwen2.5-14B-Instruct \
    --llm_url http://127.0.0.1:7800/v1 \
    --data ./data/longbench \
    --save_dir ./ucm_sparse_predictions/longbench_online \
    --strip_think 1 \
    --local_tokenizer /home/models/Qwen2.5-14B-Instruct \
    --max_len 32768 \
    --max_tokens 2048 \
    --timeout 30 \
    --concurrency 5 \
    --resume
```

**Parameters:**
- `--model NAME`: Model name for API (default: `Qwen2.5-14B-Instruct`)
- `--llm_url URL`: LLM service base URL (default: `http://127.0.0.1:7800/v1`)
- `--data PATH`: Path to test data directory (default: `./data/longbench`)
- `--save_dir PATH`: Directory to save results (default: `./ucm_sparse_predictions/longbench_v2`)
- `--strip_think 0|1`: Whether to apply strip_think in eval (default: 0)
- `--local_tokenizer PATH`: Local tokenizer path (optional)
- `--max_len INT`: Maximum input length (default: 32768)
- `--max_tokens INT`: Maximum generation tokens (default: 2048)
- `--timeout INT`: Connection timeout in seconds (default: 30)
- `--concurrency INT`: Number of concurrent requests (default: 1, sequential processing)
- `--resume`: Resume from existing result file

**Output:** Results are saved in the specified `--save_dir` directory with F1 scores in `.f1.txt` files.

#### 2. LongBench-v2 Accuracy Testing (Online)

**Script:** `eval_online_longbench_v2_acc.sh`

This script tests on LongBench-v2 dataset via API and evaluates using Accuracy metric.

```shell
cd eval

# Run with default settings
bash eval_online_longbench_v2_acc.sh

# Run with custom parameters
bash eval_online_longbench_v2_acc.sh \
    --model Qwen3-32B \
    --llm_url http://127.0.0.1:7800/v1 \
    --dataset ./data/longbench-v2/data.json \
    --template ./prompts/0shot.txt \
    --save_dir ./ucm_sparse_predictions/longbench_v2_online \
    --domain "Long Structured Data Understanding" \
    --max_tokens 128 \
    --max_context_length 32768 \
    --cot \
    --max_samples 100 \
    --local_tokenizer /home/models/Qwen3-32B \
    --max_model_len 32768 \
    --timeout 30 \
    --temperature 0.1 \
    --concurrency 5 \
    --no_truncate \
    --resume
```

**Parameters:**
- `--model NAME`: Model name for API (default: `Qwen3-32B`)
- `--llm_url URL`: LLM service base URL (default: `http://127.0.0.1:7800/v1`)
- `--dataset PATH`: Path to dataset file (default: `./data/longbench-v2/data.json`)
- `--template PATH`: Path to prompt template file (default: `./prompts/0shot.txt`)
- `--save_dir PATH`: Directory to save results (default: `./ucm_sparse_predictions/longbench_v2`)
- `--domain STR`: Domain to filter (default: `Long Structured Data Understanding`)
- `--max_tokens INT`: Maximum tokens to generate (default: 16384)
- `--max_context_length INT`: Maximum context length (optional)
- `--cot`: Enable Chain-of-Thought evaluation (default: enabled)
- `--max_samples INT`: Maximum number of samples to process (optional)
- `--resume`: Resume from existing output file
- `--local_tokenizer PATH`: Local tokenizer path (default: `/home/models/Qwen3-32B`)
- `--max_model_len INT`: Model's maximum context length for truncation (default: 32768)
- `--timeout INT`: Connection timeout in seconds (default: 30)
- `--temperature FLOAT`: Sampling temperature (default: 0.1)
- `--concurrency INT`: Number of concurrent requests (default: 1, sequential processing)
- `--no_truncate`: Disable prompt truncation

**Output:** Results are saved in the specified `--save_dir` directory, and accuracy analysis is automatically performed.

### Results

All result files will be saved in the `eval/ucm_sparse_predictions` folder.

Test results of Full Attention (Qwen2.5-14B-Instruct) on LongBench:

| Dataset | F1-Score |
|-------|-----------:|
| multifieldqa_zh | 66.6 |
| dureader | 29.33 |

