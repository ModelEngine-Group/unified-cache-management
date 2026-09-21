# MiniMax 系列模型

=== "A2"

    本教程使用预构建 Docker 镜像，在 Atlas 800 A2 上通过 vLLM-Ascend 部署 [MiniMax-M2.7-w8a8-QuaRot](https://www.modelscope.ai/models/vllm-ascend/MiniMax-M2.7-w8a8-QuaRot) 并接入 UCM。

    ## 1. 启动 Docker 容器

    将 `MODEL_PATH` 设为本地模型目录，从[快速开始](../../quick_start/index.md#vllm-ascend)复制支持 MiniMax-M2.7 且匹配 Ascend 硬件的 **UCM 镜像**地址到 `IMAGE`，在**宿主机**执行；其中 `sysctl` 命令需要 root 权限，会修改宿主机内核参数：

    ```bash
    export MODEL_PATH=/data/weights/MiniMax-M2.7-w8a8-QuaRot
    export IMAGE='<full Ascend UCM image reference from Quickstart>'
    export CONTAINER_NAME=minimax27-ucm

    sysctl -w vm.swappiness=0
    sysctl -w kernel.numa_balancing=0
    sysctl -w kernel.sched_migration_cost_ns=50000

    mkdir -p /data/ucm/cache /data/ucm/log
    docker pull "$IMAGE"

    docker run --rm -it \
        --name "$CONTAINER_NAME" \
        --network host \
        --ipc=host \
        --device /dev/davinci0 \
        --device /dev/davinci1 \
        --device /dev/davinci2 \
        --device /dev/davinci3 \
        --device /dev/davinci4 \
        --device /dev/davinci5 \
        --device /dev/davinci6 \
        --device /dev/davinci7 \
        --device /dev/davinci_manager \
        --device /dev/devmm_svm \
        --device /dev/hisi_hdc \
        -v /usr/local/dcmi:/usr/local/dcmi:ro \
        -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool:ro \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
        -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
        -v /etc/hccn.conf:/etc/hccn.conf:ro \
        -v "$MODEL_PATH:/data/weights/MiniMax-M2.7-w8a8-QuaRot:ro" \
        -v /data/ucm:/data/ucm \
        -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        -e ENABLE_UCM_PATCH=1 \
        -e UCM_LOG_PATH=/data/ucm/log \
        --workdir /workspace \
        --entrypoint /bin/bash \
        "$IMAGE"
    ```

    ## 2. 配置 UCM

    在**容器内**执行，创建 `/workspace/ucm_config_example.yaml`：

    ```bash
    cat > /workspace/ucm_config_example.yaml <<'EOF'
    ucm_connectors:
      - ucm_connector_name: "UcmPipelineStore"
        ucm_connector_config:
          store_pipeline: "Cache|Posix"
          storage_backends: "/data/ucm/cache"
          io_direct: false
          cache_buffer_capacity_gb: 64
          posix_capacity_gb: 0
          use_gdr: false

    enable_event_sync: true
    use_layerwise: true
    enable_record_traces: false
    use_lite: false
    persist_token_threshold: 0
    EOF
    ```

    - `Cache|Posix` 通过主机内存传输 KV 数据，并将其持久化到 `/data/ucm/cache`。
    - `cache_buffer_capacity_gb: 64` 设置主机缓存缓冲区预算。
    - `posix_capacity_gb: 0` 关闭基于容量的 Posix 垃圾回收，不会关闭文件系统写入。

    参数定义见[配置参数](../../../reference/config-parameters.md)。

    ## 3. 启动在线服务

    服务参数参考 [vLLM-Ascend 官方指南](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/MiniMax-M2.html)。

    在**容器内**执行：

    ```bash
    export MODEL_PATH=/data/weights/MiniMax-M2.7-w8a8-QuaRot

    export HCCL_BUFFSIZE=512
    export HCCL_OP_EXPANSION_MODE=AIV
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    export ENABLE_UCM_PATCH=1
    export UCM_LOG_PATH=/data/ucm/log
    export UC_LOGGER_LEVEL=info

    vllm serve "$MODEL_PATH" \
        --served-model-name MiniMax-M2.7 \
        --host 0.0.0.0 \
        --port 7800 \
        --trust-remote-code \
        --tensor-parallel-size 8 \
        --quantization ascend \
        --enable-expert-parallel \
        --max-num-seqs 32 \
        --seed 1024 \
        --max-num-batched-tokens 32768 \
        --gpu-memory-utilization 0.85 \
        --enable-prefix-caching \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --additional-config '{"enable_cpu_binding":true}' \
        --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":16}' \
        --kv-transfer-config '{
            "kv_connector": "UCMConnector",
            "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "UCM_CONFIG_FILE": "/workspace/ucm_config_example.yaml"
            }
        }'
    ```

    `--kv-transfer-config` 用于将 vLLM 接入 UCM，并指定上一步创建的配置文件：

    | 字段 | 作用 |
    | --- | --- |
    | `kv_connector` | 选择 `UCMConnector`。 |
    | `kv_connector_module_path` | 指定 Connector 模块：`ucm.integration.vllm.ucm_connector`。 |
    | `kv_role` | `kv_both` 同时启用缓存加载和保存。 |
    | `kv_connector_extra_config.UCM_CONFIG_FILE` | 指向 `/workspace/ucm_config_example.yaml`。 |

    ## 4. 调用 API

    服务就绪后，打开**第二个宿主机终端**：

    ```bash
    curl --fail http://127.0.0.1:7800/health
    curl --fail http://127.0.0.1:7800/v1/models

    curl --fail-with-body http://127.0.0.1:7800/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{
            "model": "MiniMax-M2.7",
            "messages": [
                {
                    "role": "user",
                    "content": "Explain the role of KV cache in large language model inference."
                }
            ],
            "max_completion_tokens": 1024,
            "temperature": 0
        }'
    ```

    预期返回 HTTP 200，模型列表包含 `MiniMax-M2.7`，Chat Completions 响应包含 `choices` 字段。

=== "H100"

    本教程使用预构建 Docker 镜像，在 NVIDIA H100 上通过 vLLM 部署 `MiniMax-M2.7` 并接入 UCM。

    ## 1. 启动 Docker 容器

    将 `MODEL_PATH` 设为本地模型目录，从[快速开始](../../quick_start/index.md#vllm)复制支持该模型和 H100 的 CUDA **UCM 镜像**地址到 `IMAGE`，在**宿主机**执行：

    ```bash
    export MODEL_PATH=/data/weights/MiniMax-M2.7
    export IMAGE='<full CUDA UCM image reference from Quickstart>'
    export CONTAINER_NAME=minimax27-ucm-h100

    mkdir -p /data/ucm/cache /data/ucm/log
    docker pull "$IMAGE"

    docker run --rm -it \
        --name "$CONTAINER_NAME" \
        --gpus '"device=0,1,2,3,4,5,6,7"' \
        --ipc=host \
        --network=host \
        -v "$MODEL_PATH:/data/weights/MiniMax-M2.7:ro" \
        -v /data/ucm:/data/ucm \
        --workdir /workspace \
        --entrypoint /bin/bash \
        "$IMAGE"
    ```

    ## 2. 配置 UCM

    在**容器内**执行，创建 `/workspace/ucm_config_example.yaml`：

    ```bash
    mkdir -p /data/ucm/cache /data/ucm/log

    cat > /workspace/ucm_config_example.yaml <<'EOF'
    ucm_connectors:
      - ucm_connector_name: "UcmPipelineStore"
        ucm_connector_config:
          store_pipeline: "Cache|Posix"
          storage_backends: "/data/ucm/cache"
          io_direct: false
          cache_buffer_capacity_gb: 64
          posix_capacity_gb: 0
          use_gdr: false

    enable_event_sync: true
    use_layerwise: true
    enable_record_traces: false
    use_lite: false
    persist_token_threshold: 0
    EOF
    ```

    - `Cache|Posix` 通过主机内存传输 KV 数据，并将其持久化到 `/data/ucm/cache`。
    - `cache_buffer_capacity_gb: 64` 设置主机缓存缓冲区预算。
    - `posix_capacity_gb: 0` 关闭基于容量的 Posix 垃圾回收，不会关闭文件系统写入。

    参数定义见[配置参数](../../../reference/config-parameters.md)。

    ## 3. 启动在线服务

    在**容器内**执行：

    ```bash
    export MODEL_PATH=/data/weights/MiniMax-M2.7
    export ENABLE_UCM_PATCH=1
    export UCM_LOG_PATH=/data/ucm/log

    vllm serve "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port 7800 \
        --data-parallel-size 1 \
        --tensor-parallel-size 4 \
        --served-model-name minimax-m2.7 \
        --trust-remote-code \
        --max-model-len 32768 \
        --max-num-seqs 8 \
        --max-num-batched-tokens 8192 \
        --gpu-memory-utilization 0.90 \
        --enable-prefix-caching \
        --block-size 128 \
        --enable-auto-tool-choice \
        --tool-call-parser minimax_m2 \
        --reasoning-parser minimax_m2 \
        --enforce-eager \
        --kv-transfer-config '{
            "kv_connector": "UCMConnector",
            "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "UCM_CONFIG_FILE": "/workspace/ucm_config_example.yaml"
            }
        }'
    ```

    `--kv-transfer-config` 用于将 vLLM 接入 UCM，并指定上一步创建的配置文件：

    | 字段 | 作用 |
    | --- | --- |
    | `kv_connector` | 选择 `UCMConnector`。 |
    | `kv_connector_module_path` | 指定 Connector 模块：`ucm.integration.vllm.ucm_connector`。 |
    | `kv_role` | `kv_both` 同时启用缓存加载和保存。 |
    | `kv_connector_extra_config.UCM_CONFIG_FILE` | 指向 `/workspace/ucm_config_example.yaml`。 |

    ## 4. 调用 API

    服务就绪后，打开**第二个宿主机终端**：

    ```bash
    curl --fail http://127.0.0.1:7800/health
    curl --fail http://127.0.0.1:7800/v1/models

    curl --fail-with-body http://127.0.0.1:7800/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{
            "model": "minimax-m2.7",
            "messages": [
                {
                    "role": "user",
                    "content": "Explain the role of KV cache in large language model inference."
                }
            ],
            "max_completion_tokens": 1024,
            "temperature": 0
        }'
    ```

    预期返回 HTTP 200，模型列表包含 `minimax-m2.7`，Chat Completions 响应包含 `choices` 字段。
