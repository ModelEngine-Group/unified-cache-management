# Installation-NPU
This document describes how to install unified-cache-management when using Ascend NPU manually.

## Requirements
- OS: Linux
- Python: >= 3.9, < 3.12
- A hardware with Ascend NPU. It’s usually the Atlas 800 A2 series.

The current version of unified-cache-management based on vLLM-Ascend v0.9.2rc1, refer to [vLLM-Ascend Installation Requirements](https://vllm-ascend.readthedocs.io/en/latest/installation.html#requirements) to meet the requirements.

You have 2 ways to install for now:
- Setup from code: First, prepare vLLM-Ascend environment, then install unified-cache-management from source code.
- Setup from docker: use the unified-cache-management docker image directly.

## Setup from code

### Prepare vLLM-Ascend Environment
For the sake of environment isolation and simplicity, we recommend preparing the vLLM-Ascend environment by pulling the official, pre-built vLLM-Ascend Docker image.
```bash
docker pull quay.io/ascend/vllm-ascend:v0.9.2rc1
```
Use the following command to run your own container:
```bash
# Update DEVICE according to your device (/dev/davinci[0-7])
export DEVICE=/dev/davinci7
# Update the vllm-ascend image
export IMAGE=quay.io/ascend/vllm-ascend:v0.9.2rc1
docker run --rm \
    --name vllm-ascend-env \
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
    -it $IMAGE bash
```
Codes of vLLM and vLLM Ascend are placed in /vllm-workspace, you can refer to [vLLM-Ascend Installation](https://vllm-ascend.readthedocs.io/en/latest/installation.html) for more information.

**Note:** For vLLM and vLLM Ascend patches, they are now applied automatically via dynamic patching when you import the unified-cache-management package.

Refer to these issues [vllm-issue](https://github.com/vllm-project/vllm/issues/21702) and [vllm-ascend-issue](https://github.com/vllm-project/vllm-ascend/issues/2057) to see details of patches' changes.

### Build from source code
Follow commands below to install unified-cache-management:
```bash
# Replace <branch_or_tag_name> with the branch or tag name needed
git clone --depth 1 --branch <branch_or_tag_name> https://github.com/ModelEngine-Group/unified-cache-management.git
cd unified-cache-management
export PLATFORM=ascend
pip install -v -e . --no-build-isolation
cd ..
```

## Setup from docker
Download the pre-built `vllm-ascend` docker image or build unified-cache-management docker image by commands below:
 ```bash
 # Build docker image using source code, replace <branch_or_tag_name> with the branch or tag name needed
 git clone --depth 1 --branch <branch_or_tag_name> https://github.com/ModelEngine-Group/unified-cache-management.git
 cd unified-cache-management
 docker build -t ucm-vllm:latest -f ./docker/Dockerfile-NPU ./
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