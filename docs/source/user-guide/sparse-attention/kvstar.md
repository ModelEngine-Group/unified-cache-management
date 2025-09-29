# KVStar: 

## 🔍 Overview


## 🚦 Quick Start

### Basic Usage
KVStar can be launched using the following command:
```shell
export MODEL_PATH="/path/to/model" # For example: /home/models/Qwen2.5-14B-Instruct
export DATASET_PATH="/path/to/longbench/multifieldqa_zh.jsonl" # For example: /home/data/Longbench/data/multifieldqa_zh.jsonl
export DATA_DIR="/path/to/data"
python examples/offline_inference_kvstar.py
```
KVStar can be configured by modifying `ucm_sparse_config` in `examples/offline_inference_kvstar.py`.
```python
...
kv_connector_extra_config = {
    "ucm_connector_name": "UcmNfsStore",
    "ucm_connector_config": {
        "storage_backends": "/path/to/data",
        "kv_block_size": 33554432,
    },
    "ucm_sparse_config": {
        "KVStarMultiStep": {
            "init_window_sz": 1,
            "local_window_sz": 2,
            "sparse_ratio": 0.25,
            "retrieval_stride": 8,
            "blk_repre_dim_prune_ratio": 0.25,
            "blk_repre_inner_token_merge": 2,
        }
    },
}
...
```

## 🎯 Key Design





## 🔥 Results
The following results were obtained using `Qwen2.5-14B-Instruct` under the specified hyperparameters:
```python
"ucm_sparse_config": {
    "KVStarMultiStep": {
        "init_window_sz": 1,
        "local_window_sz": 2,
        "sparse_ratio": 0.25,
        "retrieval_stride": 8,
        "blk_repre_dim_prune_ratio": 0.25,  # 块表征维度裁剪
        "blk_repre_inner_token_merge": 2,  # 块内几个token融合成一个表征
    }
}
```

### 🏆 Performance

### 📈 Accuracy
We use [LongBench](https://huggingface.co/datasets/zai-org/LongBench) to evaluate the accuracy of the KVStar algorithm.
| Dataset | F1-Score |
|-------|-----------|
| multifieldqa_zh | 62.63 |
| dureader | 30.96 |