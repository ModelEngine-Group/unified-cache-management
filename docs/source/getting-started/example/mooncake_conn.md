# Mooncake Connector

This document provides a usage example and configuration guide for the **Mooncake Connector**. This connector enables offloading of KV cache from GPU HBM to CPU Mooncake, helping reduce memory pressure and support larger models or batch sizes.

## Performance

| tokens | mooncake-first | mooncake-second  | default                 |
| ------ | ------------------ | ------------------ | ------------------ |
| 2k     | 1.9231491860002279 | 0.8265988459810615 | 0.5419427898712457 |
| 4k     | 3.9460434830747544 | 1.5273493870627135 | 0.991630249004811  |
| 8k     | 7.577957597002387  | 2.7632693520281464 | 2.0716467570047827 |
| 16k    | 16.823639799049126 | 5.515289016952738  | 4.742832682048902  |
| 32k    | 81.98759594326839  | 14.217441103421152 | 12.310140203218907 |

Use mooncake fig && default:
<p align="center">
  <img alt="UCM" src="../../images/mooncake_performance.png" width="40%">
  <span style="display:inline-block; width:10%;"></span>
  <img alt="UCM" src="../../images/mooncake_default_performance.png" width="40%">
</p>

## Features

The Monncake connector supports the following functionalities:

- `dump`: Offload KV cache blocks from HBM to Mooncake.
- `load`: Load KV cache blocks from Mooncake back to HBM.
- `lookup`: Look up KV blocks stored in Mooncake by block hash.
- `wait`: Ensure that all copy streams between CPU and GPU have completed.

## Configuration

### Start Mooncake Services

1. Follow the [Mooncake official guide](https://github.com/kvcache-ai/Mooncake/blob/v0.3.4/doc/en/build.md) to build Mooncake.

> **[Warning]**: Currently, this connector only supports Mooncake v0.3.4, and the updated version is being adapted.

2. Start Mooncake Store Service

    Please change the IP addresses and ports in the following guide according to your env.

```bash
# Unset HTTP proxies
unset http_proxy https_proxy no_proxy HTTP_PROXY HTTPS_PROXY NO_PROXY
# Navigate to the metadata server directory, http server for example.
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

- `local_hostname`:   
  The IP address of the current node used to communicate with the metadata server.
- `metadata_server`:  
  The metadata server of the mooncake transfer engine.
- `master_server_address`:  
  The IP address and the port of the master daemon process of MooncakeStore.
- `protocol`  *(optional)*:  
  If not provided, it defaults to **tcp**.
- `device_name`  *(optional)*:  
  The device to be used for data transmission, it is required when “protocol” is set to “rdma”. If multiple NIC devices are used, they can be separated by commas such as “erdma_0,erdma_1”. Please note that there are no spaces between them.
- `global_segment_size`*(optional)*:  
  The size of each global segment in bytes. `DEFAULT_GLOBAL_SEGMENT_SIZE = 3355443200`  **3.125 GiB**
- `local_buffer_size`*(optional)*:  
  The size of the local buffer in bytes. `DEFAULT_LOCAL_BUFFER_SIZE = 1073741824`   **1.0 GiB**


### Example:

```python
kv_connector_extra_config={
    "ucm_connector_name": "UcmMooncakeStore", 
    "ucm_connector_config":{
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
    "ucm_connector_name": "UcmMooncakeStore", 
    "ucm_connector_config":{    
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
        "ucm_connector_name": "UcmMooncakeStore", 
        "ucm_connector_config":{  
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
