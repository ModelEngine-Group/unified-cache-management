---
hide:
  - toc
---

# 快速开始

先选择引擎、版本和运行环境，下方会显示对应的完整接入指南。标准引擎镜像和 UCM 镜像两种方式使用同一套环境选择。Release 表示安装制品的版本。

<div id="ucm-install-app" class="ucm-install" data-locale="zh" data-quickstart-selector>
  <p class="ucm-install__status" data-install-status aria-live="polite">正在加载当前发布清单……</p>
  <div class="ucm-selector" data-install-selector></div>
</div>

<noscript>请启用 JavaScript，以选择环境并显示相应的制品命令。下方的 SGLang 指南仍可直接阅读。</noscript>

<section data-quickstart-guide="vllm" markdown="1">

## vLLM (CUDA) {#vllm}

### 当前环境与准备要求 {#vllm-environment}

<ul data-environment-summary hidden>
<li>引擎: <span data-env-value="engine_version"></span></li>
<li>CUDA / CANN: <span data-env-value="runtime"></span></li>
<li>操作系统: <span data-env-value="os"></span></li>
<li>CPU 架构: <span data-env-value="architecture"></span></li>
<li>Python（镜像环境）: <span data-env-value="python_version"></span></li>
</ul>

准备两张可用设备，与下方 `--tensor-parallel-size 2` 对应，宿主机需安装匹配的驱动。将 `Qwen/Qwen2.5-14B-Instruct` 模型文件放入宿主机的 `<path_to_your_models>` 目录，并准备可写的 `<path_to_your_storage>` 缓存目录。两种镜像都将它们挂载到 `/workspace/model`、`/workspace/storage`，工作目录统一为 `/workspace`。

### 选择镜像方式

=== "vLLM 标准镜像"

    #### 启动标准 vLLM 镜像 {#vllm-standard}

    在宿主机执行。这里使用与上方环境一致的标准引擎镜像，容器内尚未安装 UCM。

    <div data-requires="standard" hidden markdown="1">

    <div data-command-template markdown="1">

    ```bash
    docker pull --platform {{ docker_platform }} "{{ standard_image }}"
    docker run --rm -it \
        --platform {{ docker_platform }} \
        --entrypoint /bin/bash \
        --workdir /workspace \
        --gpus all \
        --network=host \
        --ipc=host \
        -v "<path_to_your_models>:/workspace/model" \
        -v "<path_to_your_storage>:/workspace/storage" \
        --name ucm-quickstart \
        "{{ standard_image }}"
    ```

    </div>

    #### 在容器内安装 UCM {#vllm-wheel}

    上一步进入容器后，执行以下命令安装对应的 UCM 后端<span data-requires="toolkit" hidden>及 Toolkit</span>。工具用法见 [Toolkit 指南](../../toolkit/index.md)。

    <div data-command-template markdown="1">

    ```bash
    {{ pip_install }}
    python{{ python_version }} -c "import ucm; print(ucm.__file__)"
    ```

    </div>

    </div>

    <div data-artifact-missing="standard" markdown="1">

    当前环境没有可确定的标准镜像与 UCM Wheel 组合。请使用[源码构建](../../developer-guide/build_from_source.md)。

    </div>

=== "UCM 镜像"

    #### 启动 UCM 镜像 {#vllm-docker}

    在宿主机执行。这个镜像基于同一标准引擎镜像构建，并已安装 UCM；进入容器后直接继续下方配置。

    <div data-requires="image" hidden markdown="1">

    <div data-command-template markdown="1">

    ```bash
    docker pull --platform {{ docker_platform }} "{{ image }}"
    docker run --rm -it \
        --platform {{ docker_platform }} \
        --entrypoint /bin/bash \
        --workdir /workspace \
        --gpus all \
        --network=host \
        --ipc=host \
        -v "<path_to_your_models>:/workspace/model" \
        -v "<path_to_your_storage>:/workspace/storage" \
        --name ucm-quickstart \
        "{{ image }}"
    ```

    </div>

    </div>

    <div data-artifact-missing="image" markdown="1">

    发布数据中没有可用于当前环境的 UCM 镜像信息。请使用[源码构建](../../developer-guide/build_from_source.md)。

    </div>

两种方式进入容器后，都在容器内完成下面相同的配置和启动步骤；验证请求在另一个宿主机终端执行。

需要调整缓存后端时，查阅[缓存配置](../../developer-guide/cache-configuration/index.md)；字段说明见[配置参数](../../reference/config-parameters.md)。

<span id="vllm-ucm"></span>

### 配置缓存 {#vllm-_1}

创建可写的 `/workspace/storage` 目录；使用容器时将其挂载到持久存储。把以下配置保存为引擎可读取的 `/workspace/ucm.yaml`：

```bash
mkdir -p /workspace/storage
cat > /workspace/ucm.yaml <<'YAML'
ucm_connectors:
  - ucm_connector_name: UcmPipelineStore
    ucm_connector_config:
      store_pipeline: "Cache|Posix"
      storage_backends: /workspace/storage
      posix_capacity_gb: 10240
      posix_io_engine: "psync"
      posix_data_trans_concurrency: 128
enable_metrics: true
YAML
```

配置沿用仓库示例的 Cache 与 Posix 设置，只将存储路径对应到容器挂载目录。未显式设置主机缓存容量，实际默认值见[配置参数](../../reference/config-parameters.md)。`enable_metrics: true` 用于下方的指标检查。

### 启动服务 {#vllm-_2}

示例加载 `/workspace/model` 中挂载的 Qwen2.5-14B-Instruct 模型。请求中的 `model` 与服务使用的模型路径一致；更换模型时，请同时检查并行度、上下文长度和模型特定参数。

```bash
export MODEL_ID=/workspace/model
export ENABLE_UCM_PATCH=1
vllm serve "$MODEL_ID" \
  --tensor-parallel-size 2 \
  --max-model-len 20000 \
  --gpu-memory-utilization 0.87 \
  --trust-remote-code \
  --block-size 128 \
  --port 7800 \
  --enforce-eager \
  --kv-transfer-config '{
    "kv_connector": "UCMConnector",
    "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/workspace/ucm.yaml"}
  }'
```

运行时 patch hook 和 connector 都需要启用。检查启动日志是否包含 `create UcmPipelineStore with config:` 以及预期的存储路径。

### 验证服务与外部缓存 {#vllm-verify-the-service-and-external-cache}

HTTP 服务就绪说明服务已启动；还需要单独确认 UCM 缓存命中。

```bash
curl --fail http://127.0.0.1:7800/health
curl --fail http://127.0.0.1:7800/v1/models
```

在另一宿主机终端生成可重复使用的提示词，使其长度超过数个 128-token 块，然后发送请求：

```bash
python3 - <<'PYREQUEST'
import json
from pathlib import Path
Path('/tmp/ucm-request.json').write_text(json.dumps({
    "model": "/workspace/model",
    "prompt": "Explain how a shared external cache reuses previous computation. " * 128,
    "max_tokens": 100,
    "temperature": 0,
}))
PYREQUEST
curl --fail http://127.0.0.1:7800/v1/completions \
  -H 'Content-Type: application/json' --data-binary @/tmp/ucm-request.json
curl --fail http://127.0.0.1:7800/metrics | grep '^ucm:'
```

等待保存完成后，保留测试缓存并以相同配置重启，重放同一请求。确认外部命中、加载完成和输出正确性，具体信号及判断方法见[验证外部缓存](../observability/verify-cache.md)。

没有命中或加载失败时，按[故障排查](../../reference/troubleshooting.md)检查配置、存储和内存预算。

### 使用其他模型 {#vllm-_3}

在[模型教程](../model-tour/index.md)中找到引擎官方配置，保留模型特定的引擎参数，再加入上面的 UCM connector 配置。

</section>

<section data-quickstart-guide="vllm-ascend" markdown="1">

## vLLM-Ascend (NPU) {#vllm-ascend}

### 当前环境与准备要求 {#vllm-ascend-environment}

<ul data-environment-summary hidden>
<li>引擎: <span data-env-value="engine_version"></span></li>
<li>CUDA / CANN: <span data-env-value="runtime"></span></li>
<li>设备: <span data-env-value="variant"></span></li>
<li>操作系统: <span data-env-value="os"></span></li>
<li>CPU 架构: <span data-env-value="architecture"></span></li>
<li>Python（镜像环境）: <span data-env-value="python_version"></span></li>
</ul>

准备两张可用设备，与下方 `--tensor-parallel-size 2` 对应，宿主机需安装匹配的驱动。将 `Qwen/Qwen2.5-14B-Instruct` 模型文件放入宿主机的 `<path_to_your_models>` 目录，并准备可写的 `<path_to_your_storage>` 缓存目录。两种镜像都将它们挂载到 `/workspace/model`、`/workspace/storage`，工作目录统一为 `/workspace`。

### 选择镜像方式

=== "vLLM-Ascend 标准镜像"

    #### 启动标准 vLLM-Ascend 镜像 {#vllm-ascend-standard}

    在宿主机执行。这里使用与上方环境一致的标准引擎镜像，容器内尚未安装 UCM。

    <div data-requires="standard" hidden markdown="1">

    <div data-command-template markdown="1">

    ```bash
    docker pull --platform {{ docker_platform }} "{{ standard_image }}"
    docker run --rm -it \
        --platform {{ docker_platform }} \
        --entrypoint /bin/bash \
        --workdir /workspace \
        --device /dev/davinci0 \
        --device /dev/davinci1 \
        --device /dev/davinci_manager \
        --device /dev/devmm_svm \
        --device /dev/hisi_hdc \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
        -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        --network=host \
        --ipc=host \
        -v "<path_to_your_models>:/workspace/model" \
        -v "<path_to_your_storage>:/workspace/storage" \
        --name ucm-quickstart \
        "{{ standard_image }}"
    ```

    </div>

    #### 在容器内安装 UCM {#vllm-ascend-wheel}

    上一步进入容器后，执行以下命令安装对应的 UCM 后端<span data-requires="toolkit" hidden>及 Toolkit</span>。工具用法见 [Toolkit 指南](../../toolkit/index.md)。

    <div data-command-template markdown="1">

    ```bash
    {{ pip_install }}
    python{{ python_version }} -c "import ucm; print(ucm.__file__)"
    ```

    </div>

    </div>

    <div data-artifact-missing="standard" markdown="1">

    当前环境没有可确定的标准镜像与 UCM Wheel 组合。请使用[源码构建](../../developer-guide/build_from_source.md)。

    </div>

=== "UCM 镜像"

    #### 启动 UCM 镜像 {#vllm-ascend-docker}

    在宿主机执行。这个镜像基于同一标准引擎镜像构建，并已安装 UCM；进入容器后直接继续下方配置。

    <div data-requires="image" hidden markdown="1">

    <div data-command-template markdown="1">

    ```bash
    docker pull --platform {{ docker_platform }} "{{ image }}"
    docker run --rm -it \
        --platform {{ docker_platform }} \
        --entrypoint /bin/bash \
        --workdir /workspace \
        --device /dev/davinci0 \
        --device /dev/davinci1 \
        --device /dev/davinci_manager \
        --device /dev/devmm_svm \
        --device /dev/hisi_hdc \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
        -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        --network=host \
        --ipc=host \
        -v "<path_to_your_models>:/workspace/model" \
        -v "<path_to_your_storage>:/workspace/storage" \
        --name ucm-quickstart \
        "{{ image }}"
    ```

    </div>

    </div>

    <div data-artifact-missing="image" markdown="1">

    发布数据中没有可用于当前环境的 UCM 镜像信息。请使用[源码构建](../../developer-guide/build_from_source.md)。

    </div>

两种方式进入容器后，都在容器内完成下面相同的配置和启动步骤；验证请求在另一个宿主机终端执行。

需要调整缓存后端时，查阅[缓存配置](../../developer-guide/cache-configuration/index.md)；字段说明见[配置参数](../../reference/config-parameters.md)。

<span id="vllm-ascend-ucm"></span>
<span id="vllm-ascend-_1"></span>

### 配置缓存 {#vllm-ascend-_2}

创建可写的 `/workspace/storage` 目录；使用容器时将其挂载到持久存储。把以下配置保存为引擎可读取的 `/workspace/ucm.yaml`：

```bash
mkdir -p /workspace/storage
cat > /workspace/ucm.yaml <<'YAML'
ucm_connectors:
  - ucm_connector_name: UcmPipelineStore
    ucm_connector_config:
      store_pipeline: "Cache|Posix"
      storage_backends: /workspace/storage
      posix_capacity_gb: 10240
      posix_io_engine: "psync"
      posix_data_trans_concurrency: 128
enable_metrics: true
YAML
```

配置沿用仓库示例的 Cache 与 Posix 设置，只将存储路径对应到容器挂载目录。未显式设置主机缓存容量，实际默认值见[配置参数](../../reference/config-parameters.md)。`enable_metrics: true` 用于下方的指标检查。

### 启动服务 {#vllm-ascend-_3}

示例加载 `/workspace/model` 中挂载的 Qwen2.5-14B-Instruct 模型。请求中的 `model` 与服务使用的模型路径一致；更换模型时，请同时检查并行度、上下文长度和模型特定参数。

```bash
export MODEL_ID=/workspace/model
export ENABLE_UCM_PATCH=1
vllm serve "$MODEL_ID" \
  --tensor-parallel-size 2 \
  --max-model-len 20000 \
  --gpu-memory-utilization 0.87 \
  --trust-remote-code \
  --block-size 128 \
  --port 7800 \
  --enforce-eager \
  --kv-transfer-config '{
    "kv_connector": "UCMConnector",
    "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/workspace/ucm.yaml"}
  }'
```

运行时 patch hook 和 connector 都需要启用。检查启动日志是否包含 `create UcmPipelineStore with config:` 以及预期的存储路径。

### 验证服务与外部缓存 {#vllm-ascend-verify-the-service-and-external-cache}

HTTP 服务就绪说明服务已启动；还需要单独确认 UCM 缓存命中。

```bash
curl --fail http://127.0.0.1:7800/health
curl --fail http://127.0.0.1:7800/v1/models
```

在另一宿主机终端生成可重复使用的提示词，使其长度超过数个 128-token 块，然后发送请求：

```bash
python3 - <<'PYREQUEST'
import json
from pathlib import Path
Path('/tmp/ucm-request.json').write_text(json.dumps({
    "model": "/workspace/model",
    "prompt": "Explain how a shared external cache reuses previous computation. " * 128,
    "max_tokens": 100,
    "temperature": 0,
}))
PYREQUEST
curl --fail http://127.0.0.1:7800/v1/completions \
  -H 'Content-Type: application/json' --data-binary @/tmp/ucm-request.json
curl --fail http://127.0.0.1:7800/metrics | grep '^ucm:'
```

等待保存完成后，保留测试缓存并以相同配置重启，重放同一请求。确认外部命中、加载完成和输出正确性，具体信号及判断方法见[验证外部缓存](../observability/verify-cache.md)。

没有命中或加载失败时，按[故障排查](../../reference/troubleshooting.md)检查配置、存储和内存预算。

### 使用其他模型 {#vllm-ascend-_4}

在[模型教程](../model-tour/index.md)中找到引擎官方配置，保留模型特定的引擎参数，再加入上面的 UCM connector 配置。

</section>

<section data-quickstart-guide="sglang" markdown="1">

## SGLang {#sglang}

<ul data-environment-summary hidden>
<li>引擎：<span data-env-value="engine_version"></span></li>
<li>CUDA / CANN：<span data-env-value="runtime"></span></li>
<li>操作系统：<span data-env-value="os"></span></li>
<li>CPU 架构：<span data-env-value="architecture"></span></li>
<li>镜像 Python：<span data-env-value="python_version"></span></li>
</ul>

### 环境与前置条件 {#sglang-environment}

为 `--tensor-parallel-size 2` 准备两个可用设备和匹配的宿主机驱动。将
`Qwen/Qwen2.5-14B-Instruct` 模型文件放在宿主机的 `<path_to_your_models>`，并准备可写的
`<path_to_your_storage>` 缓存目录。两个镜像都会将它们分别挂载到 `/workspace/model` 和
`/workspace/storage`，并将 `/workspace` 作为工作目录。

### 选择镜像 {#sglang-image}

=== "标准 SGLang 镜像"

    #### 启动标准镜像 {#sglang-sglang}

    请使用上方 Quickstart 选择器根据发布清单选择 SGLang 镜像和 UCM 安装包。

    <div data-requires="standard" hidden markdown="1">
    <div data-runtime="cuda" hidden markdown="1">
    <div data-command-template markdown="1">

    ```bash
    docker pull --platform {{ docker_platform }} "{{ standard_image }}"
    docker run --rm -it \
        --platform {{ docker_platform }} \
        --entrypoint /bin/bash \
        --workdir /workspace \
        --gpus all \
        --ipc=host \
        -v "<path_to_your_models>:/workspace/model" \
        -v "<path_to_your_storage>:/workspace/storage" \
        --name ucm-quickstart "{{ standard_image }}"
    ```

    </div>

    <div data-runtime="cann" hidden markdown="1">
    <div data-command-template markdown="1">
    ```bash
    docker pull --platform {{ docker_platform }} "{{ standard_image }}"
    docker run --rm -it \
        --platform {{ docker_platform }} \
        --entrypoint /bin/bash \
        --workdir /workspace \
        --device /dev/davinci0 \
        --device /dev/davinci1 \
        --device /dev/davinci_manager \
        --device /dev/devmm_svm \
        --device /dev/hisi_hdc \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
        -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        --network=host \
        --ipc=host \
        -v "<path_to_your_models>:/workspace/model" \
        -v "<path_to_your_storage>:/workspace/storage" \
        --name ucm-quickstart "{{ standard_image }}"
    ```
    </div>
    </div>
    </div>

    #### 在容器内安装 UCM {#sglang-wheel}

    进入容器后安装匹配的 UCM backend：

    <div data-command-template markdown="1">
    ```bash
    {{ pip_install }}
    python{{ python_version }} -c "import ucm; print(ucm.__file__)"
    ```
    </div>

    </div>

    <div data-artifact-missing="standard" markdown="1">
    当前环境没有匹配的标准镜像和 UCM Wheel，请使用[源码构建](../../developer-guide/build_from_source.md)。
    </div>

    </div>

=== "UCM 镜像"

    #### 启动 UCM 镜像 {#sglang-ucm-image}

    该镜像已经包含 UCM，进入容器后直接执行下面的配置步骤。

    <div data-requires="image" hidden markdown="1">
    <div data-command-template markdown="1">

    ```bash
    docker pull --platform {{ docker_platform }} "{{ image }}"
    docker run --rm -it \
        --platform {{ docker_platform }} \
        --entrypoint /bin/bash \
        --workdir /workspace \
        --gpus all \
        --ipc=host \
        -v "<path_to_your_models>:/workspace/model" \
        -v "<path_to_your_storage>:/workspace/storage" \
        --name ucm-quickstart "{{ image }}"
    ```

    </div>
    </div>

    <div data-runtime="cann" hidden markdown="1">
    <div data-command-template markdown="1">
    ```bash
    docker pull --platform {{ docker_platform }} "{{ image }}"
    docker run --rm -it \
        --platform {{ docker_platform }} \
        --entrypoint /bin/bash \
        --workdir /workspace \
        --device /dev/davinci0 --device /dev/davinci1 --device /dev/davinci_manager \
        --device /dev/devmm_svm --device /dev/hisi_hdc \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
        -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        --network=host \
        --ipc=host \
        -v "<path_to_your_models>:/workspace/model" \
        -v "<path_to_your_storage>:/workspace/storage" \
        --name ucm-quickstart "{{ image }}"
    ```
    </div>
    </div>

    </div>

### 配置 HiCache {#sglang-hicache}

创建可写的持久目录 `/workspace/storage`。适配器要求 `page_first` 主机内存布局和 `interface_v1`，请勿改成旧版需要拷贝的存储 API。

```bash
export MODEL_ID=/workspace/model
HICACHE_CONFIG='{
  "backend_name": "unifiedcache",
  "module_path": "ucm.integration.sglang.unifiedcache_store",
  "class_name": "UnifiedCacheStore",
  "interface_v1": 1,
  "kv_connector_extra_config": {
    "ucm_connector_name": "UcmPipelineStore",
    "ucm_connector_config": {
      "storage_backends": "/workspace/storage"
    }
  }
}'
```

### 启动服务 {#sglang-start-the-server}

示例使用挂载到 `/workspace/model` 的 Qwen2.5-14B-Instruct 模型。请求使用与服务相同的
模型路径。更换模型时，还需要检查并行度、上下文长度和模型专用参数。

```bash
python3 -m sglang.launch_server \
  --model-path "$MODEL_ID" \
  --tensor-parallel-size 2 \
  --page-size 128 \
  --trust-remote-code \
  --port 7800 \
  --enable-hierarchical-cache \
  --hicache-mem-layout page_first \
  --hicache-write-policy write_through \
  --hicache-storage-backend dynamic \
  --hicache-storage-prefetch-policy wait_complete \
  --hicache-storage-backend-extra-config "$HICACHE_CONFIG"
```

按可用 GPU 调整模型与并行度。适配器负责提供块布局参数并选择 `Posix`；这套输入接口与包含 `ucm_connectors` 的 vLLM YAML 配置不同。

### 验证服务与外部缓存 {#sglang-verify-the-service-and-external-cache}

健康检查和模型接口只能确认 SGLang 服务已经启动，不能证明已经发生了外部缓存读取。

```bash
curl --fail http://127.0.0.1:7800/health
curl --fail http://127.0.0.1:7800/v1/models
```

在另一个宿主机终端中发送可重复的请求，提示词长度应超过多个 128-token block：

```bash
python3 - <<'PYREQUEST'
import json
import os
from pathlib import Path
Path('/tmp/ucm-request.json').write_text(json.dumps({
    "model": os.environ["MODEL_ID"],
    "prompt": "Explain how a shared external cache reuses previous computation. " * 128,
    "max_tokens": 64,
    "temperature": 0,
}))
PYREQUEST
curl --fail http://127.0.0.1:7800/v1/completions \
  -H 'Content-Type: application/json' --data-binary @/tmp/ucm-request.json
```

等待写入完成后正常停止服务，保留缓存目录，再使用相同的模型、page size 和并行度重启。重放请求并确认 HiCache 报告从存储预取。没有外部读取记录时，参见[故障排查](../../reference/troubleshooting.md)。

</section>
