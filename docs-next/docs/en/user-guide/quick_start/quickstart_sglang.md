# Quickstart-SGLang
This document describes how to install unified-cache-management with SGLang on cuda platform.

## Prerequisites
- SGLang >= v0.5.9, device=cuda

## Step 1: UCM Installation

We offer 2 options to install UCM.

If you want to build UCM from source code (e.g. for development or customization), see [Building and Installing UCM from Source](../../developer-guide/build_from_source.md).

### Option 1: Setup from docker

#### Official pre-built image

```bash
docker pull unifiedcachemanager/ucm-sglang:latest
```

Then run your container using following command.
```bash
# Use `--ipc=host` to make sure the shared memory is large enough.
docker run --rm \
    --gpus all \
    --network=host \
    --ipc=host \
    -v <path_to_your_models>:/home/model \
    -v <path_to_your_storage>:/home/storage \
    --name <name_of_your_container> \
    -it unifiedcachemanager/ucm-sglang:latest
```

To build the UCM Docker image from source code, see [Building and Installing UCM from Source](../../developer-guide/build_from_source.md).


### Option 2: Install by pip
1. Prepare SGLang Environment

    It is recommended to use a pre-built SGLang docker image. Follow the guide in [Option 1: Setup from docker](#option-1-setup-from-docker) or pull the official image directly:
    ```bash
    docker pull lmsysorg/sglang:v0.5.9
    ```

2. Install by pip

    Install by pip or find the pre-build wheels on [Pypi](https://pypi.org/project/uc-manager/).
    ```bash
    export PLATFORM=cuda
    pip install uc-manager
    ```
## Step 2: Configuration

### Feature : Prefix Caching

UCM configuration is passed to SGLang via `--hicache-storage-backend-extra-config` in JSON format:

```bash
HICACHE_CONFIG='{
  "backend_name":"unifiedcache",
  "module_path":"ucm.integration.sglang.unifiedcache_store",
  "class_name":"UnifiedCacheStore",
  "interface_v1":1,
  "kv_connector_extra_config":{
    "ucm_connector_name":"UcmPipelineStore",
    "ucm_connector_config":{
      "storage_backends":"/mnt/test"
    }
  }
}'
```

Note: Replace `/mnt/test` with your actual storage directory.

## Step 3: Launching Inference

### Offline Inference

SGLang already provides an offline batch inference example. No UCM-specific code changes are required; just pass the same hierarchical cache flags as the server.

```bash
# Prefix cache config (reuse from Step 2)
HICACHE_CONFIG='{
  "backend_name":"unifiedcache",
  "module_path":"ucm.integration.sglang.unifiedcache_store",
  "class_name":"UnifiedCacheStore",
  "interface_v1":1,
  "kv_connector_extra_config":{
    "ucm_connector_name":"UcmPipelineStore",
    "ucm_connector_config":{
      "storage_backends":"/mnt/test"
    }
  }
}'

python3 /path/to/sglang/examples/runtime/engine/offline_batch_inference.py \
  --model-path Qwen/Qwen2.5-14B-Instruct \
  --tensor-parallel-size 2 \
  --page-size 128 \
  --trust-remote-code \
  --enable-hierarchical-cache \
  --hicache-mem-layout page_first \
  --hicache-write-policy write_through \
  --hicache-storage-backend dynamic \
  --hicache-storage-prefetch-policy wait_complete \
  --hicache-storage-backend-extra-config "$HICACHE_CONFIG"
```

**⚠️ Make sure to replace `Qwen/Qwen2.5-14B-Instruct` with your actual model path or HF repo ID.**

**⚠️ Make sure to replace `/mnt/test` (inside `HICACHE_CONFIG`) with your actual storage directory.**

### OpenAI-Compatible Online API

To start the SGLang server with the Qwen/Qwen2.5-14B-Instruct model, run:

```bash
# Prefix cache config (reuse from Step 2)
HICACHE_CONFIG='{
  "backend_name":"unifiedcache",
  "module_path":"ucm.integration.sglang.unifiedcache_store",
  "class_name":"UnifiedCacheStore",
  "interface_v1":1,
  "kv_connector_extra_config":{
    "ucm_connector_name":"UcmPipelineStore",
    "ucm_connector_config":{
      "storage_backends":"/mnt/test"
    }
  }
}'

python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-14B-Instruct \
  --tensor-parallel-size 2 \
  --page-size 128 \
  --port 7800 \
  --trust-remote-code \
  --enable-hierarchical-cache \
  --hicache-mem-layout page_first \
  --hicache-write-policy write_through \
  --hicache-storage-backend dynamic \
  --hicache-storage-prefetch-policy wait_complete \
  --hicache-storage-backend-extra-config "$HICACHE_CONFIG"
```

**⚠️ Make sure to replace `Qwen/Qwen2.5-14B-Instruct` with your actual model path or HF repo ID.**

**⚠️ Make sure to replace `/mnt/test` (inside `HICACHE_CONFIG`) with your actual storage directory.**

If you see logs like:

```bash
INFO:     Started server process [32890]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

Then you can interact with the API:

```bash
curl http://localhost:7800/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-14B-Instruct",
    "prompt": "Hello!",
    "max_tokens": 64,
    "temperature": 0
  }'
```
