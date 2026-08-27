# Quickstart-vLLM
This document describes how to install unified-cache-management with vllm on cuda platform.

## Prerequisites
- vllm >=0.9.1, device=cuda (Sparse Feature is supported in vllm 0.9.2 and v0.11.0)

## UCM Installation

We offer 2 options to install UCM.

If you want to build UCM from source code (e.g. for development or customization), see [Building and Installing UCM from Source](../../developer-guide/build_from_source.md).

### Option 1: Setup from docker

Run your container using following command.
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

To build the UCM Docker image from source code, see [Building and Installing UCM from Source](../../developer-guide/build_from_source.md).


### Option 2: Install by pip

Find the pre-build wheels on [Pypi](https://pypi.org/project/uc-manager/).
```SH
# It is recommended to use a pre-built vLLM docker image by `docker pull vllm/vllm-openai:<vllm_version>`
export PLATFORM=cuda
pip install uc-manager
```

> **Note:** If installing via `pip install`, you need to manually add the `config.yaml` file, similar to the [ucm_config_example.yaml](https://github.com/ModelEngine-Group/unified-cache-management/blob/develop/examples/ucm_config_example.yaml), because PyPI packages do not include YAML files.

### Enable UCM Integration

UCM integrates with vLLM automatically at runtime — no manual patching of the vLLM source code is required.

Simply enable the patch hook by setting the following environment variable before launching vLLM:
```bash
export ENABLE_UCM_PATCH=1
```

UCM detects your vLLM version and applies the required patches on the fly.

>**Note:** To enable Sparse Attention (vLLM 0.11.0), also set `export ENABLE_SPARSE=1`.


## Launching Inference



For online inference , vLLM with our connector can also be deployed as a server that implements the OpenAI API protocol.

To start the vLLM server with the Qwen/Qwen2.5-14B-Instruct model, run:

```bash
export ENABLE_UCM_PATCH=1
vllm serve Qwen/Qwen2.5-14B-Instruct \
--max-model-len 20000 \
--tensor-parallel-size 2 \
--gpu_memory_utilization 0.87 \
--block_size 128 \
--trust-remote-code \
--port 7800 \
--enforce-eager \
--kv-transfer-config \
'{
    "kv_connector": "UCMConnector",
    "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/workspace/unified-cache-management/examples/ucm_config_example.yaml"}
}'
```
**⚠️ Make sure to replace `Qwen/Qwen2.5-14B-Instruct` with your actual model path or Hugging Face repo ID.**

**⚠️ Make sure to replace `"/workspace/unified-cache-management/examples/ucm_config_example.yaml"` with your actual config file path. For a sample configuration, see the [ucm_config_example.yaml](https://github.com/ModelEngine-Group/unified-cache-management/blob/develop/examples/ucm_config_example.yaml).**


If you see log as below:

```bash
INFO:     Started server process [32890]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

Congratulations, you have successfully started the vLLM server with UCM!

After successfully started the vLLM server，you can interact with the API as following:

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

To run UCM with other models, first check which models are supported in the [Support Matrix](../support-matrix/support_matrix.md). Then refer to the official [vLLM Recipes](https://recipes.vllm.ai/) for the specific model's serving command and parameters. You only need to add the `--kv-transfer-config` argument as shown in the example above to enable UCM integration.

