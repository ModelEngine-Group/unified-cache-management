# Installation
This document describes how to install unified-cache-management.

## Requirements
- OS: Linux
- Python: >= 3.9, < 3.12
- GPU: compute capability 8.0 or higher (e.g., V100, T4, RTX20xx, A100, L4, H100, etc.)
- CUDA 12.8+

You have 2 ways to install for now:
- Setup from code: First, prepare vLLM environment, then install unified-cache-management from source code.
- Setup from docker: use the unified-cache-management docker image directly.

## Setup from code

### Prepare vLLM Environment
For the sake of environment isolation and simplicity, we recommend preparing the vLLM environment by pulling the official, pre-built vLLM Docker image.
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
    -v <path_to_your_models>:/app/model \
    -v <path_to_your_storage>:/app/storage \
    --name <name_of_your_container> \
    -it vllm/vllm-openai:v0.9.2 bash
```
Refer to [Set up using docker](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html#set-up-using-docker) for more information to run your own vLLM container. After installation, please refer to this [issue](https://github.com/vllm-project/vllm/issues/21702) in order to use the uc_connector provided by unified-cache-management.

### Build from source code
Follow commands below to install unified-cache-management:
```bash
git clone --depth 1 --branch develop https://github.com/ModelEngine-Group/unified-cache-management.git
cd unified-cache-management
pip install -v -e .
cd ..
```

## Setup from docker
Download the pre-built docker image provided. Then run your container using following command. You can add or remove Docker parameters as needed.
```bash
# Use `--ipc=host` to make sure the shared memory is large enough.
docker run --rm \
    --gpus all \
    --network=host \
    --ipc=host \
    -v <path_to_your_models>:/app/model \
    -v <path_to_your_storage>:/app/storage \
    --name <name_of_your_container> \
    -it <image_id>
```