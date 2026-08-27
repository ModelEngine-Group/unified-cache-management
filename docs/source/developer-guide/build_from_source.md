# Building and Installing UCM from Source

This guide explains how to build Unified Cache Management (UCM) from source code for development or customization purposes.

If you just want to use UCM, refer to the [Quickstart (vLLM)](../getting-started/quickstart_vllm.md) or [Quickstart (vLLM-Ascend)](../getting-started/quickstart_vllm_ascend.md), which describe installation via Docker images or pip.

## Step 1: Prepare the Framework Environment

Before building UCM from source, prepare the inference framework environment that UCM integrates with.

### vLLM (CUDA Platform)

For the sake of environment isolation and simplicity, we recommend preparing the vLLM environment by pulling the official, pre-built vLLM Docker image.

```bash
docker pull vllm/vllm-openai:<vllm_version>
```

Then run your own container:

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
    -it vllm/vllm-openai:<vllm_version>
```

Refer to [Set up using docker](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html#set-up-using-docker) for more information to run your own vLLM container.

### vLLM-Ascend (Ascend Platform)

Work in a ready vLLM-Ascend environment. Please refer to the [vLLM-Ascend Installation](https://vllm-ascend.readthedocs.io/en/latest/installation.html#requirements) guide to meet the required dependencies and prepare the corresponding version of the vLLM-Ascend environment.

## Step 2: Build UCM from Source

### CUDA Platform (vLLM)

Follow the commands below to install unified-cache-management from source code:

**Note:** The sparse module was not compiled by default. To enable it, set the environment variable `export ENABLE_SPARSE=TRUE` before you build.

```bash
# Replace <branch_or_tag_name> with the branch or tag name needed
git clone --depth 1 --branch <branch_or_tag_name> https://github.com/ModelEngine-Group/unified-cache-management.git
cd unified-cache-management
export PLATFORM=cuda
pip install -v -e . --no-build-isolation
```

### Ascend Platform (vLLM-Ascend)

```bash
# Replace <branch_or_tag_name> with the branch or tag name needed
git clone --depth 1 --branch <branch_or_tag_name> https://github.com/ModelEngine-Group/unified-cache-management.git
cd unified-cache-management
export PLATFORM=ascend
pip install -v -e . --no-build-isolation
```

>**Note:** For the Atlas A3 series, the `PLATFORM` variable should be set to `ascend-a3`.

## Step 3: Enable UCM Integration

UCM integrates with vLLM / vLLM-Ascend automatically at runtime — no manual patching of the source code is required.

Enable the patch hook by setting the following environment variable before launching the inference service:

```bash
export ENABLE_UCM_PATCH=1
```

>**Note:** To enable Sparse Attention (vLLM 0.11.0), also set `export ENABLE_SPARSE=1`.

## Alternative: Build via Docker Build Scripts

When building the Docker images from source, the Dockerfile automatically invokes the corresponding build script to compile the wheel and installs from the built package:

- `scripts/build_cuda.sh` — builds the wheel for the CUDA platform
- `scripts/build_ascend.sh` — builds the wheel for the Ascend platform
- `scripts/build_mindie.sh` — builds the wheel for MindIE
- `scripts/build_sglang.sh` — builds the wheel for SGLang