# 从源码构建 UCM

开发、定制，或[安装](../user-guide/quick_start/index.md)中没有所需引擎与后端组合时，可以从源码构建。请在目标推理环境中构建，确保编译器、设备工具链、Python、PyTorch 和推理引擎彼此兼容。

## 用 Docker 准备构建环境 {#docker-build-environment}

已有匹配引擎环境时直接进入下一节。使用容器构建时，先在 Linux 宿主机选择包含所需引擎与 CUDA 工具链的官方镜像。将 `<vllm-version>` 替换为目标集成所需版本；使用 SGLang 时将镜像替换为 `lmsysorg/sglang:v0.5.9`。将 `/srv/models` 替换为宿主机上的模型目录。

```bash
mkdir -p ucm-build/{source,cache,config}

docker run --rm -it --name ucm-build-env \
  --gpus all --network=host --ipc=host \
  -v "$PWD/ucm-build/source:/workspace" \
  -v /srv/models:/models:ro \
  -v "$PWD/ucm-build/cache:/mnt/ucm-cache" \
  -v "$PWD/ucm-build/config:/etc/ucm" \
  --workdir /workspace \
  --entrypoint /bin/bash 'vllm/vllm-openai:<vllm-version>' -i
```

随后在容器内执行下面的源码检出、依赖安装和编译命令；源码写入宿主机挂载目录，容器退出后仍保留。Ascend 使用匹配的 vLLM-Ascend 镜像，其设备和驱动挂载方式见[Ascend Docker 启动](../user-guide/quick_start/index.md#vllm-ascend-docker)。仅有推理运行库的镜像未必包含编译器，需要准备目标设备的开发工具链。

## 准备源码

显式选择 UCM 分支或标签，并记录最终提交：

```bash
export UCM_REF=your-branch-or-tag
git clone --branch "$UCM_REF" https://github.com/ModelEngine-Group/unified-cache-management.git
cd unified-cache-management
git rev-parse HEAD
python --version
python -m pip install "setuptools>=64" wheel "cmake>=3.18"
```

当前源码要求 Python 3.10 或更新版本。构建 UCM 前先安装引擎及其设备依赖。下面的 `--no-build-isolation` 会使用当前环境，不会自动准备兼容的引擎或工具链。

## vLLM（CUDA） { #vllm-cuda-platform }

为所用模型和 CUDA 设备准备好 vLLM 环境，然后构建：

```bash
export PLATFORM=cuda
python -m pip install -v -e . --no-build-isolation
export ENABLE_UCM_PATCH=1
```

继续按 [vLLM 快速开始](../user-guide/quick_start/index.md#vllm)配置、启动服务，并验证外部缓存。

## vLLM-Ascend { #vllm-ascend-ascend-platform }

准备相互匹配的 vLLM-Ascend、PyTorch、CANN 和驱动环境，并显式选择目标平台：

```bash
export PLATFORM=ascend
# For Atlas A3, use: export PLATFORM=ascend-a3
python -m pip install -v -e . --no-build-isolation
export ENABLE_UCM_PATCH=1
```

继续阅读 [vLLM-Ascend 快速开始](../user-guide/quick_start/index.md#vllm-ascend)。

## SGLang（CUDA） { #sglang-cuda-platform }

当前 HiCache 适配示例使用 SGLang 0.5.9 及其零拷贝 V1 存储接口。在该引擎环境中构建 UCM：

```bash
export PLATFORM=cuda
python -m pip install -v -e . --no-build-isolation
```

继续阅读 [SGLang 快速开始](../user-guide/quick_start/index.md#sglang)。历史文件 `Dockerfile.ucm-sglang-cuda-v0.5.5` 对应另一引擎版本和补丁路径，成功构建它不代表验证了 0.5.9 的 HiCache 接入方式。

## 从当前源码构建镜像

列出所选修订版本中实际存在的 Dockerfile，检查与目标引擎和加速器对应的文件：

```bash
ls docker/Dockerfile.*
```

文件的 `FROM`、镜像参数和构建脚本决定引擎环境。选择其中一个明确版本的文件，在仓库根目录构建：

```bash
export UCM_DOCKERFILE=docker/your-selected-Dockerfile
docker build -t ucm-local:dev -f "$UCM_DOCKERFILE" .
```

### 按引擎构建镜像

下面恢复原按引擎划分的 Docker 构建命令。版本选择必须对应上方列出的实际文件名和目标运行环境，在仓库根目录执行。

**CUDA / vLLM**

```bash
export VLLM_VERSION="<vllm_version>"
docker build -t ucm-vllm:latest -f "./docker/Dockerfile.ucm-vllm-cuda-${VLLM_VERSION}" .
```

**Ascend A2 / vLLM-Ascend**

```bash
export VLLM_ASCEND_VERSION="<vllm_ascend_version>"
docker build -t ucm-vllm:latest -f "./docker/Dockerfile.ucm-vllm-ascend.a2-${VLLM_ASCEND_VERSION}" .
```

A3 使用同一文件列表中匹配的 `ascend.a3` Dockerfile，不能将 A2 镜像当作 A3 构建。

**SGLang**

```bash
docker build -t ucm-sglang:latest -f ./docker/Dockerfile.ucm-sglang-cuda-v0.5.5 .
```

这里恢复仓库中 0.5.5 Dockerfile 的构建路径，不把它描述为上方 0.5.9 HiCache 环境的构建方式。

仓库 Dockerfile 和本地构建用于开发。只有[安装](../user-guide/quick_start/index.md)中列出的制品才有对应的发布记录。本地构建成功或成功导入 UCM，仍需在目标硬件上完成服务启动和外部缓存验证。
