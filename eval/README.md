## Accuracy testing of Sparse method

### Overview
We use two Chinese subsets of  [LongBench](https://huggingface.co/datasets/zai-org/LongBench) to test the accuracy of single-document QA (multifieldqa_zh) and multi-document QA (dureader). The F1 score is adopted to evaluate the accuracy of these sparse methods. For more information about LongBench, please refer to https://github.com/THUDM/LongBench.

### Quick Start

#### Environment Preparation
```shell
pip install jieba fuzzywuzzy rouge
```
#### Test Data Preparation
Dowdload the Longbench dataset 

```shell
wget https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip && unzip data.zip

```

#### Configure Specific Sparse Method

Settings for different sparse methods are written in a JSON file, for example:
```python
{"ESA": 
    {
    "init_window_sz": 1,
    "local_window_sz": 2,
    "min_blocks":4,
    "sparse_ratio": 0.2,
    "retrieval_stride": 10
    }
}
```

Run accuracy testing with:
```shell
cd eval

# Run with default settings: Qwen2.5-14B-Instruct batch=20
bash eval_inference_F1.sh

# Run with custom parameters
# --strip_think: extract the text after </think> from model predictions
# --batch:       number of requests processed per batch
bash eval_inference_F1.sh \
    --model /home/models/QwQ-32B \
    --config ./eval/ucm_sparse_config_esa.json \
    --data ./eval/data \
    --strip_think 1 \
    --batch 1

```
The result files will be saved in the eval/ucm_sparse_predictions folder.

### Results
Test results of Full Attention (Qwen2.5-14B-Instruct):

| Dataset | F1-Score |
|-------|-----------:|
| multifieldqa_zh | 66.6 |
| dureader | 29.33 |

