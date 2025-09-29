# KVStar: 

## 🔍 Overview


## 🚦 Quick Start

### Basic Usage
KVStar can be launched using the following command:
```shell
export MODEL_PATH="/path/to/model" # For example: /home/models/Qwen2.5-14B-Instruct
export DATASET_PATH="/path/to/longbench/multifieldqa_zh.jsonl" # For example: /home/data/Longbench/data/multifieldqa_zh.jsonl
python examples/offline_inference_kvstar.py
```
KVStar can be configured by modifying `ucm_sparse_config` in `examples/offline_inference_kvstar.py`.
```python
...
ktc = KVTransferConfig(
        kv_connector=name,
        kv_connector_module_path=module_path,
        kv_role="kv_both",
        kv_connector_extra_config={
            "ucm_connector_name": "UcmNfsStore",
            "ucm_connector_config": {
                "storage_backends": "/tmp/kvstar_nfs",
                "kv_block_size": 33554432,
            },
            "ucm_sparse_config": {
                "KVStarMultiStep": {
                    "init_window_sz": 1,
                    "local_window_sz": 2,
                    "sparse_ratio": 0.25,
                    "retrieval_stride": 8,
                    "blk_repre_dim_prune_ratio": 0.25,  # 块表征维度裁剪
                    "blk_repre_inner_token_merge": 2,  # 块内几个token融合成一个表征
                }
            },
        },
    )
...
```

## 🎯 Key Design



In the second step of each period, the retrieval of the most important KV blocks is initiated. The pseudocode is as follows:
```python
def start_retrieval(self, query, forward_context):
    self.retrieval_task = self.retrieval_worker.submit(
        query, kv_block_representations=kv_block_representations
    )
```
Then, in the last step of the current period, we wait for the retrieval_worker to complete and retrieve the most relevant blocks to load. The pseudocode is:
```python
def wait_retrieval_and_start_load(self):
    topk_blocks = self.retrieval_task.result()
    self.loading_task = self.launch_transfer_task(
        "load", topk_blocks, target_HBM_addresses
    )
```
Finally, at the beginning of the next period, the transfer task is synchronized, and the KV caches in HBM are updated. The pseudocode is:
```python
def wait_transfer_task_done(self):
    ret = self.store_instance.wait(self.loading_task)
```

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
| multifieldqa_zh | |
| dureader |  |