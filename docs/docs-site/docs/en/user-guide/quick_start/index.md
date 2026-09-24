---
hide:
  - toc
---

# Quickstart

Select the engine, version and environment below to display its complete integration guide. The standard engine image and the UCM image use the same environment selection. Release identifies the installation artifact version.

<div id="ucm-install-app" class="ucm-install" data-locale="en" data-quickstart-selector>
  <p class="ucm-install__status" data-install-status aria-live="polite">Loading the current release manifest...</p>
  <div class="ucm-selector" data-install-selector></div>
</div>

<noscript>Enable JavaScript to select an environment and display its artifact commands. The SGLang guide below remains readable.</noscript>

<section data-quickstart-guide="vllm" markdown="1">

## vLLM (CUDA) {#vllm}

### Environment and prerequisites {#vllm-environment}

<ul data-environment-summary hidden>
<li>Engine: <span data-env-value="engine_version"></span></li>
<li>CUDA / CANN: <span data-env-value="runtime"></span></li>
<li>OS: <span data-env-value="os"></span></li>
<li>CPU Architecture: <span data-env-value="architecture"></span></li>
<li>Python (image environment): <span data-env-value="python_version"></span></li>
</ul>

Prepare two available devices for `--tensor-parallel-size 2` and a matching host driver. Place the `Qwen/Qwen2.5-14B-Instruct` model files in `<path_to_your_models>` on the host and prepare a writable `<path_to_your_storage>` cache directory. Both images mount these at `/workspace/model` and `/workspace/storage`, with `/workspace` as the working directory.

### Choose an image

=== "Standard vLLM image"

    #### Start the standard vLLM image {#vllm-standard}

    Run these commands on the host. The standard engine image matches the environment selected above and does not yet include UCM.

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

    #### Install UCM inside the container {#vllm-wheel}

    After entering the container, install the matching UCM backend<span data-requires="toolkit" hidden> and Toolkit</span> with the following commands. See the [Toolkit guide](../../toolkit/index.md) for tool usage.

    <div data-command-template markdown="1">

    ```bash
    {{ pip_install }}
    python{{ python_version }} -c "import ucm; print(ucm.__file__)"
    ```

    </div>

    </div>

    <div data-artifact-missing="standard" markdown="1">

    No matching standard image and UCM Wheel can be determined for this environment. Use a [source build](../../developer-guide/build_from_source.md).

    </div>

=== "UCM image"

    #### Start the UCM image {#vllm-docker}

    Run these commands on the host. This image builds on the same standard engine image and already includes UCM. After entering the container, continue directly with the configuration below.

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

    The release data does not identify a UCM image for this environment. Use a [source build](../../developer-guide/build_from_source.md).

    </div>

After entering either container, complete the same configuration and startup steps below inside that container. Send verification requests from another host terminal.

For storage changes, see [Cache Configuration](../../developer-guide/cache-configuration/index.md); field definitions are in [Configuration Parameters](../../reference/config-parameters.md).

<span id="vllm-ucm-installation"></span>

### Configure the cache {#vllm-configure-the-cache}

Create a writable `/workspace/storage` directory, mounted persistently if running in
a container. Save the following as `/workspace/ucm.yaml`, readable by the engine:

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

The Cache and Posix settings follow the repository example, with the storage path matched to the container mount. Host cache capacity is not overridden; see the effective defaults in [Configuration parameters](../../reference/config-parameters.md). `enable_metrics: true` supports the metrics check below.

### Start the server {#vllm-start-the-server}

The example loads the Qwen2.5-14B-Instruct model mounted at `/workspace/model`. The request uses the same model path as the server. When changing models, also review parallelism, context length and model-specific parameters.

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

The runtime patch hook and connector are both required. Confirm the startup log
contains `create UcmPipelineStore with config:` and the expected storage path.


### Verify the service and external cache {#vllm-verify-the-service-and-external-cache}

A ready HTTP server confirms service startup; it does not prove a UCM cache hit.

```bash
curl --fail http://127.0.0.1:7800/health
curl --fail http://127.0.0.1:7800/v1/models
```

In another host terminal, generate a repeatable prompt longer than several 128-token blocks, then send it:

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


Wait for saves to complete, retain the test cache, restart with the same settings and replay the request. Check external hits, completed loads and correct output using [external-cache verification](../observability/verify-cache.md).

For missing hits or failed loads, check configuration, storage and memory budgets through [troubleshooting](../../reference/troubleshooting.md).

### Use other models {#vllm-use-other-models}

Keep model-specific engine settings from [Model Tour](../model-tour/index.md), then add the UCM Connector configuration above.

</section>

<section data-quickstart-guide="vllm-ascend" markdown="1">

## vLLM-Ascend (NPU) {#vllm-ascend}

### Environment and prerequisites {#vllm-ascend-environment}

<ul data-environment-summary hidden>
<li>Engine: <span data-env-value="engine_version"></span></li>
<li>CUDA / CANN: <span data-env-value="runtime"></span></li>
<li>Device: <span data-env-value="variant"></span></li>
<li>OS: <span data-env-value="os"></span></li>
<li>CPU Architecture: <span data-env-value="architecture"></span></li>
<li>Python (image environment): <span data-env-value="python_version"></span></li>
</ul>

Prepare two available devices for `--tensor-parallel-size 2` and a matching host driver. Place the `Qwen/Qwen2.5-14B-Instruct` model files in `<path_to_your_models>` on the host and prepare a writable `<path_to_your_storage>` cache directory. Both images mount these at `/workspace/model` and `/workspace/storage`, with `/workspace` as the working directory.

### Choose an image

=== "Standard vLLM-Ascend image"

    #### Start the standard vLLM-Ascend image {#vllm-ascend-standard}

    Run these commands on the host. The standard engine image matches the environment selected above and does not yet include UCM.

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

    #### Install UCM inside the container {#vllm-ascend-wheel}

    After entering the container, install the matching UCM backend<span data-requires="toolkit" hidden> and Toolkit</span> with the following commands. See the [Toolkit guide](../../toolkit/index.md) for tool usage.

    <div data-command-template markdown="1">

    ```bash
    {{ pip_install }}
    python{{ python_version }} -c "import ucm; print(ucm.__file__)"
    ```

    </div>

    </div>

    <div data-artifact-missing="standard" markdown="1">

    No matching standard image and UCM Wheel can be determined for this environment. Use a [source build](../../developer-guide/build_from_source.md).

    </div>

=== "UCM image"

    #### Start the UCM image {#vllm-ascend-docker}

    Run these commands on the host. This image builds on the same standard engine image and already includes UCM. After entering the container, continue directly with the configuration below.

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

    The release data does not identify a UCM image for this environment. Use a [source build](../../developer-guide/build_from_source.md).

    </div>

After entering either container, complete the same configuration and startup steps below inside that container. Send verification requests from another host terminal.

For storage changes, see [Cache Configuration](../../developer-guide/cache-configuration/index.md); field definitions are in [Configuration Parameters](../../reference/config-parameters.md).

<span id="vllm-ascend-ucm-installation"></span>
<span id="vllm-ascend-pre-built-image"></span>

### Configure the cache {#vllm-ascend-configure-the-cache}

Create a writable `/workspace/storage` directory, mounted persistently if running in
a container. Save the following as `/workspace/ucm.yaml`, readable by the engine:

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

The Cache and Posix settings follow the repository example, with the storage path matched to the container mount. Host cache capacity is not overridden; see the effective defaults in [Configuration parameters](../../reference/config-parameters.md). `enable_metrics: true` supports the metrics check below.

### Start the server {#vllm-ascend-start-the-server}

The example loads the Qwen2.5-14B-Instruct model mounted at `/workspace/model`. The request uses the same model path as the server. When changing models, also review parallelism, context length and model-specific parameters.

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

The runtime patch hook and connector are both required. Confirm the startup log
contains `create UcmPipelineStore with config:` and the expected storage path.


### Verify the service and external cache {#vllm-ascend-verify-the-service-and-external-cache}

A ready HTTP server confirms service startup; it does not prove a UCM cache hit.

```bash
curl --fail http://127.0.0.1:7800/health
curl --fail http://127.0.0.1:7800/v1/models
```

In another host terminal, generate a repeatable prompt longer than several 128-token blocks, then send it:

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


Wait for saves to complete, retain the test cache, restart with the same settings and replay the request. Check external hits, completed loads and correct output using [external-cache verification](../observability/verify-cache.md).

For missing hits or failed loads, check configuration, storage and memory budgets through [troubleshooting](../../reference/troubleshooting.md).

### Use other models {#vllm-ascend-use-other-models}

Keep model-specific engine settings from [Model Tour](../model-tour/index.md), then add the UCM Connector configuration above.

</section>

<section data-quickstart-guide="sglang" markdown="1">

## SGLang {#sglang}

### Environment and prerequisites {#sglang-environment}

<ul data-environment-summary hidden>
<li>Engine: <span data-env-value="engine_version"></span></li>
<li>CUDA / CANN: <span data-env-value="runtime"></span></li>
<li>OS: <span data-env-value="os"></span></li>
<li>CPU Architecture: <span data-env-value="architecture"></span></li>
<li>Python (image environment): <span data-env-value="python_version"></span></li>
</ul>

Prepare two available devices for `--tensor-parallel-size 2` and a matching host driver. Place the `Qwen/Qwen2.5-14B-Instruct` model files in `<path_to_your_models>` on the host and prepare a writable `<path_to_your_storage>` cache directory. Both images mount these at `/workspace/model` and `/workspace/storage`, with `/workspace` as the working directory.

### Choose an image {#sglang-image}

=== "Standard SGLang image"

    #### Start the standard image {#sglang-sglang-image}

    Select the SGLang image and UCM package from the release manifest in the Quickstart selector above.

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
        --name ucm-quickstart \
        "{{ standard_image }}"
    ```

    </div>
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

    #### Install UCM inside the container {#sglang-wheel}

    After entering the container, install the matching UCM backend:

    <div data-command-template markdown="1">

    ```bash
    {{ pip_install }}
    python{{ python_version }} -c "import ucm; print(ucm.__file__)"
    ```

    </div>

    <div data-artifact-missing="standard" markdown="1">
    No matching standard image and UCM Wheel can be determined for this environment. Use a [source build](../../developer-guide/build_from_source.md).
    </div>

    </div>

=== "UCM image"

    #### Start the UCM image {#sglang-ucm-image}

    This image already contains UCM. After entering the container, continue directly with the configuration below.

    <div data-requires="image" hidden markdown="1">
    <div data-runtime="cuda" hidden markdown="1">
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
        --name ucm-quickstart "{{ image }}"
    ```
    </div>
    </div>

### Configure HiCache {#sglang-configure-hicache}

Create a writable, persistent `/workspace/storage` directory. The adapter requires
`page_first` host layout and `interface_v1`; do not change them to the older
copy-based storage API.

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

### Start the server {#sglang-start-the-server}

The example loads the Qwen2.5-14B-Instruct model mounted at `/workspace/model`.
The request uses the same model path as the server. When changing models, also
review parallelism, context length and model-specific parameters.

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

Adjust the model and parallelism to the available GPUs. The adapter supplies
block geometry and selects `Posix`; a vLLM YAML file with `ucm_connectors` is not
the same configuration interface.

### Verify the service and external cache {#sglang-verify-the-service-and-external-cache}

The health and model endpoints confirm that the SGLang service is ready; they do
not prove that an external cache read occurred.

```bash
curl --fail http://127.0.0.1:7800/health
curl --fail http://127.0.0.1:7800/v1/models
```

In another host terminal, send a repeatable request with a prompt longer than
several 128-token blocks:

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

Wait for the write to complete, stop the server, retain the cache directory,
and restart with the same model, page size and parallelism. Replay the request
and confirm HiCache reports a storage-prefetched load. See
[Troubleshooting](../../reference/troubleshooting.md) if no external read is reported.


</section>
