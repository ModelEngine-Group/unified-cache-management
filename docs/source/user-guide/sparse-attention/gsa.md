# GSA: Geometric Sparse Attention for Efficient Inference of Large Models

## 🔍 Overview  

GSA (Geometric Sparse Attention) simultaneously tackles the high computational complexity of long sequences and the concurrency limitations imposed by the HBM capacity wall.


## 🎯 Key Innovations

- Representation-based Sparse Selection

- Efficient KV Transition

- Cross-hardware Support

- Request-level Sparse Strategy

- P+D Multi-stage Sparsity


## 🔥 Key Results

### 🏆 Performance Highlights

### 📈 Accuracy Benchmarks

## 🧠 How It Works

### Core Algorithm

## 🚦 Quick Start  


### Basic Usage
Similr to UCM's `offline_inference_esa.py` examples. We only need to specify `ucm_sparse_method` to be `GSA` as shown below.


```python
...
ktc = KVTransferConfig(
    kv_connector=name,
    kv_connector_module_path=module_path,
    kv_role="kv_both",
    kv_connector_extra_config={
        "ucm_connector_name": "UcmDram",
        "ucm_connector_config": {
            "max_cache_size": 5368709120,
            "kv_block_size": 262144,
        },
        "ucm_sparse_method": "GSA",
    },
)
...
```


## 📊 Supported Models

| Model | Size | Support |
|-------|------|-----------|
| Qwen3-32B | 32B | ✅ |
| QwQ-32B | 32B | ✅ |
| DeepSeek-R1 | 671B | ✅ |