# Build UCM from source

Use a source build for development or an engine/backend combination that is
not published in [Installation](../user-guide/quick_start/index.md). Build inside
the target inference environment: the compiler, device toolkit, Python,
PyTorch, and engine must be compatible with one another.

## Prepare a build environment with Docker {#docker-build-environment}

If a matching engine environment is already available, continue below. Otherwise select an official image containing the engine and CUDA toolchain on the Linux host. Replace `<vllm-version>` with the version required by your integration, or use `lmsysorg/sglang:v0.5.9` for SGLang. Replace `/srv/models` with your host model directory.

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

Run checkout, dependency installation and compilation below inside the container. The mounted source survives container exit. For Ascend, use a matching vLLM-Ascend image and the [Ascend device/driver mounts](../user-guide/quick_start/index.md#vllm-ascend-docker). A runtime-only image may lack compilers; prepare the target device development toolchain.

## Prepare a checkout

Select a UCM branch or tag explicitly and record the resulting commit:

```bash
export UCM_REF=your-branch-or-tag
git clone --branch "$UCM_REF" https://github.com/ModelEngine-Group/unified-cache-management.git
cd unified-cache-management
git rev-parse HEAD
python --version
python -m pip install "setuptools>=64" wheel "cmake>=3.18"
```

Python 3.10 or newer is required by this source tree. Install the engine and
its device dependencies before building UCM. `--no-build-isolation` below
uses that environment; it does not install a compatible engine/toolkit for you.

## vLLM on CUDA { #vllm-cuda-platform }

Prepare a vLLM environment for the model and CUDA device you use, then build:

```bash
export PLATFORM=cuda
python -m pip install -v -e . --no-build-isolation
export ENABLE_UCM_PATCH=1
```

Continue with the [vLLM quickstart](../user-guide/quick_start/index.md#vllm)
for configuration, service startup, and external-cache verification.

## vLLM-Ascend { #vllm-ascend-ascend-platform }

Prepare a matching vLLM-Ascend, PyTorch, CANN, and driver environment. Choose
the platform for the actual target rather than relying on autodetection:

```bash
export PLATFORM=ascend
# For Atlas A3, use: export PLATFORM=ascend-a3
python -m pip install -v -e . --no-build-isolation
export ENABLE_UCM_PATCH=1
```

Continue with the [vLLM-Ascend quickstart](../user-guide/quick_start/index.md#vllm-ascend).

## SGLang on CUDA { #sglang-cuda-platform }

The current HiCache adapter recipe targets SGLang 0.5.9 and its zero-copy V1
storage interface. Build UCM in that engine environment:

```bash
export PLATFORM=cuda
python -m pip install -v -e . --no-build-isolation
```

Continue with the [SGLang quickstart](../user-guide/quick_start/index.md#sglang).
The historical `Dockerfile.ucm-sglang-cuda-v0.5.5` targets a different engine
and patch path; building it does not validate the 0.5.9 HiCache recipe.

## Build an image from the checkout

List the actual Dockerfiles in the selected revision and inspect the one for
your engine and accelerator:

```bash
ls docker/Dockerfile.*
```

Its `FROM`/image arguments and build script determine the engine environment.
Select a versioned file from that list, then build from the repository root:

```bash
export UCM_DOCKERFILE=docker/your-selected-Dockerfile
docker build -t ucm-local:dev -f "$UCM_DOCKERFILE" .
```

### Build by engine

The original per-engine Docker build commands are shown below. Set the engine version to one of the actual filenames listed above; the version is selected by your target environment. Run from the repository root.

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

For A3, select the matching `ascend.a3` Dockerfile from the same list; an A2 image is not an A3 build.

**SGLang**

```bash
docker build -t ucm-sglang:latest -f ./docker/Dockerfile.ucm-sglang-cuda-v0.5.5 .
```

This restores the repository's 0.5.5 Dockerfile route. It does not build the 0.5.9 HiCache environment described above.

Repository Dockerfiles and local builds are development inputs. Only artifacts
listed by [Installation](../user-guide/quick_start/index.md) have the corresponding
release publication record. Building or importing UCM locally does not replace
a service startup and external-cache check on the target hardware.
