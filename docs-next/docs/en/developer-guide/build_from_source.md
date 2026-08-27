# Building and Installing UCM from Source

This guide explains how to build Unified Cache Management (UCM) from source code for development or customization purposes.

If you just want to use UCM, refer to the [Quickstart (vLLM)](../user-guide/quick_start/quickstart_vllm.md) or [Quickstart (vLLM-Ascend)](../user-guide/quick_start/quickstart_vllm_ascend.md), which describe installation via Docker images or pip.

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

### SGLang (CUDA Platform)

For the sake of environment isolation and simplicity, we recommend preparing the SGLang environment by pulling the official, pre-built SGLang Docker image.

```bash
docker pull lmsysorg/sglang:v0.5.9
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
    -it lmsysorg/sglang:v0.5.9
```

Refer to [Using docker](https://docs.sglang.io/get_started/install.html#method-3-using-docker) for more information to run your own SGLang container.

### MindIE-LLM (Ascend Platform)

Prepare a MindIE-LLM 2.3.0 environment with the Ascend runtime/toolkit installed. For details, refer to the [MindIE-LLM official documentation](https://gitcode.com/Ascend/MindIE-LLM/blob/dev/docs/zh/user_guide/quick_start/quick_start.md).

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

### SGLang (CUDA Platform)

Follow the commands below to install unified-cache-management:

```bash
# Replace <branch_or_tag_name> with the branch or tag name needed
git clone --depth 1 --branch <branch_or_tag_name> https://github.com/ModelEngine-Group/unified-cache-management.git
cd unified-cache-management
export PLATFORM=cuda
pip install -v -e . --no-build-isolation
```

### MindIE-LLM (Ascend Platform)

Clone the repository and install UCM with MindIE-LLM support enabled:

```bash
git clone --depth 1 https://github.com/ModelEngine-Group/unified-cache-management.git
cd unified-cache-management
export PLATFORM=ascend
export UCM_ENABLE_MINDIE=1
export UCM_CXX11_ABI=1  # Or 0. This must match the target MindIE/PyTorch ABI.
pip install -v -e . --no-build-isolation
cd ..
```

> **Note:** Packages built without `UCM_ENABLE_MINDIE=1` do **not** contain MindIE-LLM integration code.
>
> **ABI requirement:** When `UCM_ENABLE_MINDIE=1`, you must also set `UCM_CXX11_ABI=0` or `1`. The value must match the target MindIE/PyTorch ABI.

## Step 3: Enable UCM Integration

### vLLM / vLLM-Ascend

UCM integrates with vLLM / vLLM-Ascend automatically at runtime — no manual patching of the source code is required.

Enable the patch hook by setting the following environment variable before launching the inference service:

```bash
export ENABLE_UCM_PATCH=1
```

>**Note:** To enable Sparse Attention (vLLM 0.11.0), also set `export ENABLE_SPARSE=1`.

### MindIE-LLM

No runtime environment variables are needed for MindIE-LLM. Once UCM is installed with MindIE support (`UCM_ENABLE_MINDIE=1`), the patch is applied automatically when `mindie_llm` is first imported.

## Alternative: Build Docker Images from Source

If you prefer to build the UCM Docker images from source code, use the Dockerfile provided under the `docker/` directory for the engine version you need. Each Dockerfile automatically invokes the corresponding build script to compile the wheel, installs from the built package, and (where applicable) applies the integration patch.

### vLLM (CUDA Platform)

Check the `docker/` directory for available Dockerfile versions (e.g. `v0.20.2`, `v0.18.0`, `v0.17.0`, `v0.11.0`), then build with the desired version:

```bash
git clone --depth 1 --branch <branch_or_tag_name> https://github.com/ModelEngine-Group/unified-cache-management.git
cd unified-cache-management
# Replace <vllm_version> with the version you need (e.g. v0.20.2)
docker build -t ucm-vllm:latest -f ./docker/Dockerfile.ucm-vllm-cuda-<vllm_version> ./
```

For vLLM (v0.11.0) with sparse attention support:

```bash
docker build -t ucm-vllm-sparse:latest -f ./docker/Dockerfile.ucm-vllm-cuda-v0.11.0 ./
```

### vLLM-Ascend (Ascend Platform)

Check the `docker/` directory for available Dockerfile versions (e.g. `v0.20.2`, `v0.18.0`, `v0.17.0`, `v0.11.0`), then build with the desired version:

```bash
git clone --depth 1 --branch <branch_or_tag_name> https://github.com/ModelEngine-Group/unified-cache-management.git
cd unified-cache-management
# Replace <vllm_ascend_version> with the version you need (e.g. v0.20.2)
docker build -t ucm-vllm:latest -f ./docker/Dockerfile.ucm-vllm-ascend.a2-<vllm_ascend_version> ./
```

For vLLM-Ascend (v0.11.0) with sparse attention support:

```bash
docker build -t ucm-vllm-sparse:latest -f ./docker/Dockerfile.ucm-vllm-ascend.a2-v0.11.0 ./
```

### SGLang (CUDA Platform)

Download the pre-built `lmsysorg/sglang:v0.5.9` docker image and build the unified-cache-management docker image by the commands below:

```bash
# Replace <branch_or_tag_name> with the branch or tag name needed
git clone --depth 1 --branch <branch_or_tag_name> https://github.com/ModelEngine-Group/unified-cache-management.git
cd unified-cache-management
docker build -t ucm-sglang:latest -f ./docker/Dockerfile.ucm-sglang-cuda-v0.5.5 ./
```

### MindIE-LLM (Ascend Platform)

Use the provided MindIE-LLM Dockerfile (Ascend base, MindIE-LLM 2.3.0):

```bash
git clone --depth 1 https://github.com/ModelEngine-Group/unified-cache-management.git
cd unified-cache-management
docker build -t ucm-mindie:latest -f ./docker/Dockerfile.ucm-mindie-ascend.a2-v2 ./
```

This Dockerfile:

* uses `swr.cn-south-1.myhuaweicloud.com/ascendhub/mindie:2.3.0-800I-A2-py311-openeuler24.03-lts`
* sets `UCM_ENABLE_MINDIE=1`
* sets `UCM_CXX11_ABI` (default `1`, override with `--build-arg UCM_CXX11_ABI=0|1` to match your target environment)
* installs UCM with MindIE-LLM support
* applies the patch during image build

### Docker Build Scripts

The Dockerfiles invoke the following build scripts to compile the wheel:

- `scripts/build_cuda.sh` — builds the wheel for the CUDA platform
- `scripts/build_ascend.sh` — builds the wheel for the Ascend platform
- `scripts/build_mindie.sh` — builds the wheel for MindIE
- `scripts/build_sglang.sh` — builds the wheel for SGLang