# Quickstart-SGLang
This document describes how to install unified-cache-management with SGLang on cuda platform.

## Prerequisites
- SGLang >= 0.5.5, device=cuda

## Step 1: UCM Installation

We offer 3 options to install UCM.

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

#### Build image from source
Download the pre-built `lmsysorg/sglang:v0.5.5.post3` docker image and build unified-cache-management docker image by commands below:
 ```bash
 # Build docker image using source code, replace <branch_or_tag_name> with the branch or tag name needed
 git clone --depth 1 --branch <branch_or_tag_name> https://github.com/ModelEngine-Group/unified-cache-management.git
 cd unified-cache-management
 docker build -t ucm-sglang:latest -f ./docker/Dockerfile.sglang_gpu ./
 ```


### Option 2: Build from source
1. Prepare SGLang Environment

    For the sake of environment isolation and simplicity, we recommend preparing the SGLang environment by pulling the official, pre-built SGLang Docker image.

    ```bash
    docker pull lmsysorg/sglang:v0.5.5.post3
    ```
    Use the following command to run your own container:
    ```bash
    # Use `--ipc=host` to make sure the shared memory is large enough.
    docker run \
        --gpus all \
        --network=host \
        --ipc=host \
        -v <path_to_your_models>:/home/model \
        -v <path_to_your_storage>:/home/storage \
        --entrypoint /bin/bash \
        --name <name_of_your_container> \
        -it lmsysorg/sglang:v0.5.5.post3
    ```
    Refer to [Using docker](https://docs.sglang.io/get_started/install.html#method-3-using-docker) for more information to run your own SGLang container.

2. Build from source code

    Follow commands below to install unified-cache-management:

    ```bash
    # Replace <branch_or_tag_name> with the branch or tag name needed
    git clone --depth 1 --branch <branch_or_tag_name> https://github.com/ModelEngine-Group/unified-cache-management.git
    cd unified-cache-management
    export PLATFORM=cuda
    pip install -v -e . --no-build-isolation
    ```

3. Apply SGLang Integration Patches (Required)

    To enable Unified Cache Management (UCM) integration with SGLang, you must **manually apply the corresponding SGLang patch**.

    You may directly navigate to the SGLang source directory, which is usually located under `/sgl-workspace`:
    ```bash
    cd <path_to_sglang>
    ```
    Then apply the SGLang patch:

    ```bash
    git apply unified-cache-management/ucm/integration/sglang/patch/0.5.5/sglang-adapt.patch
    ```


### Option 3: Install by pip
1. Prepare SGLang Environment

    It is recommended to use a pre-build SGLang docker image, please follow the guide in Option 2.

2. Install by pip

    Install by pip or find the pre-build wheels on [Pypi](https://pypi.org/project/uc-manager/).
    ```bash
    export PLATFORM=cuda
    pip install uc-manager
    ```
> **Note:** If installing via `pip install`, you need to manually add the `config.yaml` file, similar to `unified-cache-management/examples/ucm_config_example.yaml`, because PyPI packages do not include YAML files.

## Step 2: Configuration

### Feature : Prefix Caching

UCM configuration is passed to SGLang via `--hicache-storage-backend-extra-config` in JSON format:

```bash
HICACHE_CONFIG='{"kv_connector_extra_config":{"ucm_connector_name":"UcmPipelineStore","ucm_connector_config":{"storage_backends":"/mnt/test"}}}'
```

Note: Replace `/mnt/test` with your actual storage directory.

Alternatively, you can use a YAML file to provide the same configuration. A ready example is available at:
`sglang/python/sglang/srt/mem_cache/storage/unifiedcache/unifiedcache_example.yaml`.
When using YAML, set `UNIFIEDCACHE_CONFIG_FILE` to the YAML path and omit
`--hicache-storage-backend-extra-config`.

```bash
export UNIFIEDCACHE_CONFIG_FILE=/path/to/sglang/python/sglang/srt/mem_cache/storage/unifiedcache/unifiedcache_example.yaml
```

## Step 3: Launching Inference

<details open>
<summary><b>Offline Inference</b></summary>

SGLang already provides an offline batch inference example. No UCM-specific code changes are required; just pass the same hierarchical cache flags as the server.

```bash
# Prefix cache config (reuse from Step 2)
HICACHE_CONFIG='{"kv_connector_extra_config":{"ucm_connector_name":"UcmPipelineStore","ucm_connector_config":{"storage_backends":"/mnt/test"}}}'

python3 /path/to/sglang/examples/runtime/engine/offline_batch_inference.py \
  --model-path Qwen/Qwen2.5-14B-Instruct \
  --tensor-parallel-size 2 \
  --page-size 128 \
  --trust-remote-code \
  --enable-hierarchical-cache \
  --hicache-mem-layout page_first \
  --hicache-write-policy write_through \
  --hicache-storage-backend unifiedcache \
  --hicache-storage-prefetch-policy wait_complete \
  --hicache-storage-backend-extra-config "$HICACHE_CONFIG"
```

**⚠️ Make sure to replace `Qwen/Qwen2.5-14B-Instruct` with your actual model path or HF repo ID.**

**⚠️ Make sure to replace `/mnt/test` (inside `HICACHE_CONFIG`) with your actual storage directory.**

</details>

<details open>
<summary><b>OpenAI-Compatible Online API</b></summary>

To start the SGLang server with the Qwen/Qwen2.5-14B-Instruct model, run:

```bash
# Prefix cache config (reuse from Step 2)
HICACHE_CONFIG='{"kv_connector_extra_config":{"ucm_connector_name":"UcmPipelineStore","ucm_connector_config":{"storage_backends":"/mnt/test"}}}'

python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-14B-Instruct \
  --tensor-parallel-size 2 \
  --page-size 128 \
  --port 7800 \
  --trust-remote-code \
  --enable-hierarchical-cache \
  --hicache-mem-layout page_first \
  --hicache-write-policy write_through \
  --hicache-storage-backend unifiedcache \
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

</details>
