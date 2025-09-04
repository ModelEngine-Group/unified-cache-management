# Sparse Attention

This document provides a usage example and configuration guide for the **sparse attention**, which is increasingly recognized for their ability to mitigate the challenges associated with high memory bandwidth (HBM) usage and to enhance the efficiency of large language models (LLMs).


## Configuration

To use the sparse_attn connector, you need to configure the `ucm_sparse_method` field in your model's launch configuration.

### Example:
```python
kv_connector_extra_config={
    "ucm_connector_name": "UcmDram",
    "ucm_connector_config": {
        "max_cache_size": 5368709120,
        "kv_block_size": 262144,
    },
    "ucm_sparse_method": "ESA" # specify the sparse attention algorithm here
}
```

## Launching Inference

### Offline Inference

To start **offline inference** with the NFS connector，modify the script `examples/offline_inference.py` to include the `ucm_sparse_method` and put a long prompt to see the acceleration effects:

```python
# In examples/offline_inference.py
ktc = KVTransferConfig(
   ...
   kv_connector_extra_config={
    "ucm_connector_name": "UcmDram",
    "ucm_connector_config": {
        "max_cache_size": 5368709120,
        "kv_block_size": 262144,
    },
    "ucm_sparse_method": "ESA" # specify the sparse attention algorithm here
  }
)

prompts = [
    "PUT A LONG PROMPT HERE TO SEE ACCELERATION EFFECTS."
]
```

Then run the script as follows:

```bash
cd examples/
export PYTHONHASHSEED=123456
python offline_inference.py
```

### Online Inference

For **online inference** , vLLM with our connector can also be deployed as a server that implements the OpenAI API protocol. Run the following command to start the vLLM server with the Qwen/Qwen2.5-14B-Instruct model:

```bash
export PYTHONHASHSEED=123456
vllm serve /home/models/Qwen2.5-14B-Instruct \
--max-model-len 20000 \
--tensor-parallel-size 2 \
--gpu_memory_utilization 0.87 \
--trust-remote-code \
--port 7800 \
--kv-transfer-config \
'{
    "kv_connector": "UnifiedCacheConnectorV1",
    "kv_connector_module_path": "unifiedcache.integration.vllm.uc_connector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "ucm_connector_name": "UcmNfsStore",
        "ucm_connector_config": {
            "storage_backends": "/mnt/test",
            "kv_block_size": 33554432
        },
        "ucm_sparse_method": "ESA"
    }
}'
```

If you see log as below:

```bash
INFO:     Started server process [1049932]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

Congratulations, you have successfully started the vLLM server with NFS Connector!

After successfully started the vLLM server，You can interact with the API as following:

```bash
curl http://localhost:7800/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/home/models/Qwen2.5-14B-Instruct",
        "prompt": "PUT A LONG PROMPT HERE TO SEE ACCELERATION EFFECTS.",
        "max_tokens": 100,
        "temperature": 0
    }'
```