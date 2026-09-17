# Quickstart-vLLM-Ascend
This document describes how to install unified-cache-management with vllm-ascend on ascend platform.

## Prerequisites
vllm-ascend: >=v0.9.1 (vllm == 0.9.2 to use the Sparse Feature)

**Please refer to the [vLLM-Ascend Installation](https://vllm-ascend.readthedocs.io/en/latest/installation.html#requirements) guide to meet the required dependencies, and prepare the corresponding version of the vllm-ascend environment as needed.**

## Step 1: UCM Installation

We offer 2 options to install UCM.

If you want to build UCM from source code (e.g. for development or customization), see [Building and Installing UCM from Source](../developer-guide/build_from_source.md).

### Option 1: Install by pip
Install by pip or find the pre-build wheels on [Pypi](https://pypi.org/project/uc-manager/).
```
export PLATFORM=ascend
pip install uc-manager
```
> **Note:** If installing via `pip install`, you need to manually add the `config.yaml` file, similar to `unified-cache-management/examples/ucm_config_example.yaml`, because PyPI packages do not include YAML files.

### Enable UCM Integration

UCM integrates with vLLM and vLLM-Ascend automatically at runtime — no manual patching of the source code is required.

Simply enable the patch hook by setting the following environment variable before launching vLLM-Ascend:
```bash
export ENABLE_UCM_PATCH=1
```

UCM detects your vLLM and vLLM-Ascend versions and applies the required patches on the fly.

>**Note:** To enable Sparse Attention (supported on v0.11.0), also set `export ENABLE_SPARSE=1`.

### Option 2: Setup from docker

#### Build image from source
Check the `docker/` directory for available Dockerfile versions (e.g. `v0.20.2`, `v0.18.0`, `v0.17.0`, `v0.11.0`), then build with the desired version:
```bash
# Replace <vllm_ascend_version> with the version you need (e.g. v0.20.2)
docker build -t ucm-vllm:latest -f ./docker/Dockerfile.ucm-vllm-ascend.a2-<vllm_ascend_version> ./
```

For vLLM-Ascend(v0.11.0) with sparse attention support:
```bash
docker build -t ucm-vllm-sparse:latest -f ./docker/Dockerfile.ucm-vllm-ascend.a2-v0.11.0 ./
```

The Dockerfile automatically invokes the build script (`scripts/build_ascend.sh`) to compile the wheel and installs from the built package.

#### Build image from pre-built package

If you have a pre-built tar package (e.g. from CI), extract it and build the image in `package` mode:
```bash
mkdir -p /tmp/ucm-pkg && tar xzf AI-Storage-Kit_*.tar.gz -C /tmp/ucm-pkg
# Replace <vllm_ascend_version> with the version you need (e.g. v0.20.2)
docker build --build-arg INSTALL_MODE=package \
  -t ucm-vllm:latest -f /tmp/ucm-pkg/docker/Dockerfile.ucm-vllm-ascend.a2-<vllm_ascend_version> /tmp/ucm-pkg
```

vllm-ascend provides two variants: **Ubuntu** and **openEuler**.
The Dockerfile uses the **Ubuntu** variant by default.

If you want to use the **openEuler** variant, override the base image with `--build-arg IMAGE_NAME_VERSION`:

```text
quay.io/ascend/vllm-ascend:v0.11.0-openeuler
```
Then run your container using following command. You can add or remove Docker parameters as needed.
```bash
# Update DEVICE according to your device (/dev/davinci[0-7])
export DEVICE=/dev/davinci7
# Update the vllm-ascend image
docker run --rm \
    --network=host \
    --device $DEVICE \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/.cache:/root/.cache \
    -v <path_to_your_models>:/app/model \
    -v <path_to_your_storage>:/app/storage \
    --name <name_of_your_container> \
    -it <image_id> bash
```
## Step 2: Configuration

### Features Overview

UCM supports two key features: **Prefix Cache** and **Sparse attention**. Each feature supports both **Offline Inference** and **Online API** modes. More details are available via the links
- [Prefix Cache](../user-guide/prefix-cache/index.md)
- [GSA Sparsity](../user-guide/sparse-attention/gsa.md)

For quick start, just follow the guide below to launch your own inference experience;

### Feature 1:  Prefix Caching

You may directly edit the example file at `unified-cache-management/examples/ucm_config_example.yaml`. For more please refer to [Prefix Cache with NFS Store](../user-guide/prefix-cache/nfs_store.md) and [Prefix Cache with Pipeline Store](../user-guide/prefix-cache/pipeline_store.md) document.

⚠️ Make sure to replace `/mnt/test` with your actual storage directory. 

### Feature 2:  Sparsity

The sparse module was not compiled by default. To enable it, set the environment variable `export ENABLE_SPARSE=TRUE` and build the package again (see [Building and Installing UCM from Source](../developer-guide/build_from_source.md)). And uncomment `ucm_sparse_config` code block in `unified-cache-management/examples/ucm_config_example.yaml`. Additionally, if you want to run GSA, you also need to set the environment variable `export VLLM_HASH_ATTENTION=1`.

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

For online inference, vLLM with our connector can also be deployed as a server that implements the OpenAI API protocol.

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

### Running with Other Models

To run UCM with other models, first check which models are supported in the [Support Matrix](../user-guide/support-matrix/support_matrix.md). Then refer to the official [vLLM Recipes](https://recipes.vllm.ai/) and [vLLM-Ascend Model Tutorials](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/index.html) for the specific model's serving command and parameters. You only need to add the `--kv-transfer-config` argument as shown in the example above to enable UCM integration.
</details>
