# Mooncake Connector

This document provides a usage example and configuration guide for the **Mooncake Connector**. This connector enables offloading of KV cache from GPU HBM to CPU Mooncake, helping reduce memory pressure and support larger models or batch sizes.

## Performance





## Features

The Monncake connector supports the following functionalities:

- `dump`: Offload KV cache blocks from HBM to Mooncake.
- `load`: Load KV cache blocks from Mooncake back to HBM.
- `lookup`: Look up KV blocks stored in Mooncake by block hash.
- `wait`: Ensure that all copy streams between CPU and GPU have completed.

## Configuration

### Start Mooncake Services



1. Follow the [Mooncake official guide](https://github.com/kvcache-ai/Mooncake/blob/v0.3.4/doc/en/build.md) to build Mooncake.

2. Start Mooncake Store Service

    Please change the IP addresses and ports in the following guide according to your env.

```bash
# Unset HTTP proxies
unset http_proxy https_proxy no_proxy HTTP_PROXY HTTPS_PROXY NO_PROXY
# Navigate to the metadata server directory
cd $MOONCAKE_ROOT_DIR/mooncake-transfer-engine/example/http-metadata-server
# Start Metadata Service
go run . --addr=0.0.0.0:23790
# Start Master Service
mooncake_master --port 50001
```
- Replace `$MOONCAKE_ROOT_DIR` with your Mooncake source root path.
- Make sure to unset any HTTP proxies to prevent networking issues.
- Use appropriate port based on your environment.




### Required Parameters

To use the Mooncake connector, you need to configure the `connector_config` dictionary in your model's launch configuration.

- `max_cache_size` *(optional)*:  
  Specifies the maximum allowed Mooncake memory usage (in **byte**) for caching in `kv_connector_extra_config["ucm_connector_config"]`.  
  If not provided, it defaults to **5 GB**.
- `kv_block_size` *(optional)*:  
  Specifies the memory size (in bytes) of a single key or value cache block used in vLLM’s paged attention mechanism, which is calculated as : `block_size * head_size * total_num_kv_heads * element_size`.
- `local_hostname`:   
  The IP address of the current node used to communicate with the metadata server.
- `metadata_server`:  
  The metadata server of the mooncake transfer engine.
- `protocl`  *(optional)*:  
  If not provided, it defaults to **tcp**.
- `device_name`  *(optional)*:  
  The device to be used for data transmission, it is required when “protocol” is set to “rdma”. If multiple NIC devices are used, they can be separated by commas such as “erdma_0,erdma_1”. Please note that there are no spaces between them.
- `master_server_address`:  
  The IP address and the port of the master daemon process of MooncakeStore.

### Example:

```python
# Allocate up to 8GB Mooncake for KV cache
# KV Block size (in byte) is 262144
kv_connector_extra_config={
    "ucm_connector_name": "UcmMooncake", 
    "ucm_connector_config":{
        "max_cache_size": 5368709120, 
        "kv_block_size": 262144,
        "local_hostname": "127.0.0.1",
        "metadata_server": "http://127.0.0.1:23790/metadata",
        "protocol": "tcp",
        "device_name": "",
        "master_server_address": "127.0.0.1:50001"
        }
    }
```

## Launching Inference

### Offline Inference

To start **offline inference** with the Mooncake connector，modify the script `examples/offline_inference.py` to include the `kv_connector_extra_config` for Mooncake connector usage:

```python
# In examples/offline_inference.py
ktc = KVTransferConfig(
    ...
    kv_connector_extra_config={
    "ucm_connector_name": "UcmMooncake", 
    "ucm_connector_config":{
        "max_cache_size": 5368709120, 
        "kv_block_size": 262144,     
        "local_hostname": "127.0.0.1",
        "metadata_server": "http://127.0.0.1:23790/metadata",
        "protocol": "tcp",
        "device_name": "",
        "master_server_address": "127.0.0.1:50001"
        }
    }
)
```

Then run the script as follows:

```bash
cd examples/
python offline_inference.py
```

### Online Inference

For **online inference** , vLLM with our connector can also be deployed as a server that implements the OpenAI API protocol. 

First, specify the python hash seed by:
```bash
export PYTHONHASHSEED=123456
```

Run the following command to start the vLLM server with the Qwen/Qwen2.5-14B-Instruct model:

```bash
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
        "ucm_connector_name": "UcmMooncake", 
        "ucm_connector_config":{
            "max_cache_size": 5368709120, 
            "kv_block_size": 262144,     
            "local_hostname": "127.0.0.1",
            "metadata_server": "http://127.0.0.1:23790/metadata",
            "protocol": "tcp",
            "device_name": "",
            "master_server_address": "127.0.0.1:50001"
            }
        }
    }
}'
```

If you see log as below:

```bash
INFO:     Started server process [321290]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

Congratulations, you have successfully started the vLLM server with Mooncake Connector!

After successfully started the vLLM server，You can interact with the API as following:

```bash
curl http://localhost:7800/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/home/models/Qwen2.5-14B-Instruct",
        "prompt": "Shanghai is a",
        "max_tokens": 7,
        "temperature": 0
    }'
```
