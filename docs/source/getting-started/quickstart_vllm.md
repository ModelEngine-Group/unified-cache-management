# Quickstart-vLLM
This document describes how to install unified-cache-management with vllm on cuda platform.

## Prerequisites
- vllm >=0.9.1, device=cuda (Sparse Feature is supported in vllm 0.9.2 and v0.11.0)

## Step 1: UCM Installation

We offer 3 options to install UCM.

### Option 1: Setup from docker

#### Official pre-built image

```bash
docker pull unifiedcachemanager/ucm:latest
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
    -it unifiedcachemanager/ucm:latest

```

#### Build image from source
Download the pre-built `vllm/vllm-openai:v0.9.2` docker image and build unified-cache-management docker image by commands below:
 ```bash
 # Build docker image using source code, replace <branch_or_tag_name> with the branch or tag name needed
 git clone --depth 1 --branch <branch_or_tag_name> https://github.com/ModelEngine-Group/unified-cache-management.git
 cd unified-cache-management
 docker build -t ucm-vllm:latest -f ./docker/Dockerfile ./
 ```


### Option 2: Build from source
1. Prepare vLLM Environment

    For the sake of environment isolation and simplicity, we recommend preparing the vLLM environment by pulling the official, pre-built vLLM Docker image.
    > Note: v0.11.0 is newly supported (replace the tag with v0.11.0 if needed).

    ```bash
    docker pull vllm/vllm-openai:v0.9.2
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
        -it vllm/vllm-openai:v0.9.2
    ```
    Refer to [Set up using docker](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html#set-up-using-docker) for more information to run your own vLLM container.

2. Build from source code

    Follow commands below to install unified-cache-management:

    **Note:** The sparse module was not compiled by default. To enable it, set the environment variable `export ENABLE_SPARSE=TRUE` before you build.

    ```bash
    # Replace <branch_or_tag_name> with the branch or tag name needed
    git clone --depth 1 --branch <branch_or_tag_name> https://github.com/ModelEngine-Group/unified-cache-management.git
    cd unified-cache-management
    export PLATFORM=cuda
    pip install -v -e . --no-build-isolation
    ```

3. Apply vLLM Integration Patches (Required)

    To enable Unified Cache Management (UCM) integration with vLLM, you must **manually apply the corresponding vLLM patch**.

    You may directly navigate to the vLLM source directory:
    ```bash
    cd <path_to_vllm>
    ```
    Apply the patch that matches your development needs:

    #### vLLM 0.11.0 

    Note: v0.11.0 only requires the sparse attention patch.

    ```bash
    git apply unified-cache-management/ucm/integration/vllm/patch/0.11.0/vllm-adapt-sparse.patch
    ```

    #### vLLM 0.9.2 
    - Full UCM integration (recommended):
    ```bash
    git apply unified-cache-management/ucm/integration/vllm/patch/0.9.2/vllm-adapt.patch
    ```

    - Sparse attention only:
    ```bash
    git apply unified-cache-management/ucm/integration/vllm/patch/0.9.2/vllm-adapt-sparse.patch
    ```

    - ReRoPE support only:
    ```bash
    git apply unified-cache-management/ucm/integration/vllm/patch/0.9.2/vllm-adapt-rerope.patch
    ```

    Choose the patch according to your development needs.
    If you are working on **sparse attention** or **ReRoPE** independently, applying only the corresponding patch is sufficient.


### Option 3: Install by pip
1. Prepare vLLM Environment

    It is recommended to use a pre-build vllm docker image, please follow the guide in Option 2.

2. install by pip

    Install by pip or find the pre-build wheels on [Pypi](https://pypi.org/project/uc-manager/).
    ```
    export PLATFORM=cuda
    pip install uc-manager
    ```
> **Note:** If installing via `pip install`, you need to manually add the `config.yaml` file, similar to `unified-cache-management/examples/ucm_config_example.yaml`, because PyPI packages do not include YAML files.

## Step 2: Configuration

### Features Overview

UCM supports two key features: **Prefix Cache** and **Sparse attention**. Each feature supports both **Offline Inference** and **Online API** modes. More details are available via the links
- [Prefix Cache](../user-guide/prefix-cache/index.md)
- [GSA Sparsity](../user-guide/sparse-attention/gsa.md)

For quick start, just follow the guide below to launch your own inference experience.


### Feature 1:  Prefix Caching

You may directly edit the example file at `unified-cache-management/examples/ucm_config_example.yaml`. For more please refer to [Prefix Cache with NFS Store](../user-guide/prefix-cache/nfs_store.md) and [Prefix Cache with Pipeline Store](../user-guide/prefix-cache/pipeline_store.md) document.

⚠️ Make sure to replace `/mnt/test` with your actual storage directory. 

### Feature 2:  Sparsity

The sparse module was not compiled by default. To enable it, set the environment variable `export ENABLE_SPARSE=TRUE` and re-compile the code you built. And uncomment `ucm_sparse_config` code block in `unified-cache-management/examples/ucm_config_example.yaml`. Additionally, if you want to run GSAOnDevice, you also need to set the environment variable `export VLLM_HASH_ATTENTION=1`, and you can fix the hash weight of the HashEncoder by setting the environment variable `export GSA_HASH_ENCODER_SEED=<the seed number>`.

## Step 3: Launching Inference

<details open>
<summary><b>Offline Inference</b></summary>

In the `examples/` directory, you will find the `offline_inference.py` script used for offline inference. Before executing the script, locate line 25 and replace the `UCM_CONFIG_FILE` value with the path to your own configuration file.
```bash
def build_llm_with_uc(module_path: str, name: str, model: str):
    ktc = KVTransferConfig(
        kv_connector=name,
        kv_connector_module_path=module_path,
        kv_role="kv_both",
        kv_connector_extra_config={
            "UCM_CONFIG_FILE": "/workspace/unified-cache-management/examples/ucm_config_example.yaml"
        },
    )
```
Then run following commands:

```bash
cd examples/
# Change the model path to your own model path
python offline_inference.py
```

</details>



<details open>
<summary><b>OpenAI-Compatible Online API</b></summary>

For online inference , vLLM with our connector can also be deployed as a server that implements the OpenAI API protocol.

To start the vLLM server with the Qwen/Qwen2.5-14B-Instruct model, run:

```bash
vllm serve Qwen/Qwen2.5-14B-Instruct \
--max-model-len 20000 \
--tensor-parallel-size 2 \
--gpu_memory_utilization 0.87 \
--block_size 128 \
--trust-remote-code \
--port 7800 \
--enforce-eager \
--no-enable-prefix-caching \
--kv-transfer-config \
'{
    "kv_connector": "UCMConnector",
    "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/workspace/unified-cache-management/examples/ucm_config_example.yaml"}
}'
```
**⚠️ The parameter `--no-enable-prefix-caching` is for SSD performance testing, please remove it for production.**

**⚠️ Make sure to replace `"/workspace/unified-cache-management/examples/ucm_config_example.yaml"` with your actual config file path.**

**⚠️ The log files of UCM module will be put under `log` directory of the path you start vllm service. To use a custom log path, set `export UCM_LOG_PATH=my_log_dir`.**


If you see log as below:

```bash
INFO:     Started server process [32890]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

Congratulations, you have successfully started the vLLM server with UCM!

After successfully started the vLLM server，You can interact with the API as following:

```bash
curl http://localhost:7800/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-14B-Instruct",
    "prompt": "You are a highly specialized assistant whose mission is to faithfully reproduce English literary texts verbatim, without any deviation, paraphrasing, or omission. Your primary responsibility is accuracy: every word, every punctuation mark, and every line must appear exactly as in the original source. Core Principles: Verbatim Reproduction: If the user asks for a passage, you must output the text word-for-word. Do not alter spelling, punctuation, capitalization, or line breaks. Do not paraphrase, summarize, modernize, or \"improve\" the language. Consistency: The same input must always yield the same output. Do not generate alternative versions or interpretations. Clarity of Scope: Your role is not to explain, interpret, or critique. You are not a storyteller or commentator, but a faithful copyist of English literary and cultural texts. Recognizability: Because texts must be reproduced exactly, they will carry their own cultural recognition. You should not add labels, introductions, or explanations before or after the text. Coverage: You must handle passages from classic literature, poetry, speeches, or cultural texts. Regardless of tone—solemn, visionary, poetic, persuasive—you must preserve the original form, structure, and rhythm by reproducing it precisely. Success Criteria: A human reader should be able to compare your output directly with the original and find zero differences. The measure of success is absolute textual fidelity. Your function can be summarized as follows: verbatim reproduction only, no paraphrase, no commentary, no embellishment, no omission. Please reproduce verbatim the opening sentence of the United States Declaration of Independence (1776), starting with \"When in the Course of human events\" and continuing word-for-word without paraphrasing.",
    "max_tokens": 100,
    "temperature": 0
  }'
```
</details>