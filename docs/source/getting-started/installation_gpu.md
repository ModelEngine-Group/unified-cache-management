# Installation-GPU
This document describes how to install unified-cache-management.

## Requirements
- OS: Linux
- Python: 3.12
- GPU: NVIDIA compute capability 8.0+ (e.g., L20, L40, H20)
- CUDA: CUDA Version 12.8

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
    -v <path_to_your_models>:/home/model \
    -v <path_to_your_storage>:/home/storage \
    --entrypoint /bin/bash \
    --name <name_of_your_container> \
    -it vllm/vllm-openai:v0.9.2
```
Refer to [Set up using docker](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html#set-up-using-docker) for more information to run your own vLLM container.

### Build from source code
Follow commands below to install unified-cache-management:

```bash
# Replace <branch_or_tag_name> with the branch or tag name needed
git clone --depth 1 --branch <branch_or_tag_name> https://github.com/ModelEngine-Group/unified-cache-management.git
cd unified-cache-management
export PLATFORM=cuda
pip install -v -e . --no-build-isolation
```

**Note:** Patches are now applied automatically via dynamic patching when you import the unified-cache-management package. You no longer need to manually apply patches using `git apply`. The patches are automatically applied when you use the `UnifiedCacheConnectorV1` connector.

## Setup from docker
Download the pre-built `vllm/vllm-openai:v0.9.2` docker image and build unified-cache-management docker image by commands below:
 ```bash
 # Build docker image using source code, replace <branch_or_tag_name> with the branch or tag name needed
 git clone --depth 1 --branch <branch_or_tag_name> https://github.com/ModelEngine-Group/unified-cache-management.git
 cd unified-cache-management
 docker build -t ucm-vllm:latest -f ./docker/Dockerfile ./
 ```
Then run your container using following command. You can add or remove Docker parameters as needed.
```bash
# Use `--ipc=host` to make sure the shared memory is large enough.
docker run --rm \
    --gpus all \
    --network=host \
    --ipc=host \
    -v <path_to_your_models>:/home/model \
    -v <path_to_your_storage>:/home/storage \
    --name <name_of_your_container> \
    -it <image_id>
```