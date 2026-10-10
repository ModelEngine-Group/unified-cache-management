# Qwen 系列模型

=== "A2"

    本教程使用预构建 Docker 镜像，在 Atlas 800 A2 上通过 vLLM-Ascend 部署 [Qwen3.8-27B-w8a8](https://www.modelscope.cn/models/Eco-Tech/Qwen3.8-27B-w8a8) 并接入 UCM。

    ## 1. 启动 Docker 容器

    将 `MODEL_PATH` 设为本地模型目录，从[快速开始](../../quick_start/index.md#vllm-ascend)复制 A2 兼容的 **UCM 镜像**地址到 `IMAGE`，在**宿主机**执行：

    ```bash
    export MODEL_PATH=/data/weights/Qwen3.8-27B-w8a8
    export IMAGE='<full A2 UCM image reference from Quickstart>'
    export CONTAINER_NAME=qwen38-ucm

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
        -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
        -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
        -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
        -v "$MODEL_PATH:/data/weights/Qwen3.8-27B-w8a8:ro" \
        -v /data/ucm:/data/ucm \
        -e ASCEND_RT_VISIBLE_DEVICES=0,1 \
        -e ENABLE_UCM_PATCH=1 \
        -e UCM_LOG_PATH=/data/ucm/log \
        --workdir /workspace \
        --entrypoint /bin/bash \
        "$IMAGE"
    ```

    ## 2. 配置 UCM

    在**容器内**执行，创建 `/workspace/ucm_config_example.yaml`：

    ```bash
    cat > /workspace/ucm_config_example.yaml <<'YAML'
    ucm_connectors:
      - ucm_connector_name: UcmPipelineStore
        ucm_connector_config:
          store_pipeline: "Cache|Posix"
          store_health:
            enabled: true
          storage_backends: /data/ucm/cache
          posix_io_engine: psync
          io_direct: false
          cache_buffer_capacity_gb: 64
          share_buffer_enable: false
          use_gdr: false

    enable_event_sync: true
    use_layerwise: true
    enable_record_traces: false
    use_lite: false
    persist_token_threshold: 0
    enable_metrics: true
    YAML
    ```

    该配置启用逐层传输、存储健康检查和指标。

    - `persist_token_threshold: 0` 不按请求长度排除短请求，但可复用数据仍需满足缓存块和混合状态对齐要求。
    - 设置 `share_buffer_enable: false` 后，每个 TP worker 独立拥有 64 GiB 主机缓冲区。TP2 需预留 128 GiB，另加运行时所需内存。如果初始化根据实际 KV 布局报告更大的最小缓冲区容量，应增大容量并配备相应 RAM。
    - Qwen3.8 使用混合注意力布局。UCM 恢复前缀时需要同时恢复注意力 KV 和对应的循环状态。请保持混合 KV Cache 管理器开启，并在下方启动命令中设置 `--mamba-cache-mode align`。

    参数定义和默认值见[配置参数](../../../reference/config-parameters.md)。

    ## 3. 启动在线服务

    服务参数参考 [vLLM-Ascend 官方指南](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3.8-27B.html)。

    在**容器内**执行：

    ```bash
    export MODEL_PATH=/data/weights/Qwen3.8-27B-w8a8
    export ASCEND_RT_VISIBLE_DEVICES=0,1
    export HCCL_BUFFSIZE=512
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export ENABLE_UCM_PATCH=1
    export UCM_LOG_PATH=/data/ucm/log

    vllm serve "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port 7800 \
        --data-parallel-size 1 \
        --tensor-parallel-size 2 \
        --quantization ascend \
        --served-model-name qwen3.8 \
        --max-num-seqs 32 \
        --max-model-len 131072 \
        --max-num-batched-tokens 16384 \
        --trust-remote-code \
        --enable-prefix-caching \
        --block-size 128 \
        --mamba-cache-mode align \
        --gpu-memory-utilization 0.85 \
        --speculative-config '{"method":"mtp","num_speculative_tokens":3,"enforce_eager":true}' \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --additional-config '{"enable_cpu_binding":true}' \
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
            "model": "qwen3.8",
            "messages": [
                {
                    "role": "user",
                    "content": "Explain the role of KV cache in large language model inference."
                }
            ],
            "max_completion_tokens": 512,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": false}
        }'
    ```

    预期返回 HTTP 200，模型列表包含 `qwen3.8`，生成文本位于 `choices[0].message.content`。

=== "H100"

    本教程使用预构建 Docker 镜像，在 NVIDIA H100 上通过 vLLM 部署 `Qwen3.8-27B` 并接入 UCM。

    ## 1. 启动 Docker 容器

    将 `MODEL_PATH` 设为本地模型目录，从[快速开始](../../quick_start/index.md#vllm)复制支持该模型和 H100 的 CUDA **UCM 镜像**地址到 `IMAGE`，在**宿主机**执行：

    ```bash
    export MODEL_PATH=/data/weights/Qwen3.8-27B
    export IMAGE='<full CUDA UCM image reference from Quickstart>'
    export CONTAINER_NAME=qwen38-ucm-h100

    mkdir -p /data/ucm/cache /data/ucm/log
    docker pull "$IMAGE"

    docker run --rm -it \
        --name "$CONTAINER_NAME" \
        --gpus '"device=0,1,2,3,4,5,6,7"' \
        --ipc=host \
        --network=host \
        -v "$MODEL_PATH:/data/weights/Qwen3.8-27B:ro" \
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
    export MODEL_PATH=/data/weights/Qwen3.8-27B
    export ENABLE_UCM_PATCH=1
    export UCM_LOG_PATH=/data/ucm/log

    vllm serve "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port 7800 \
        --data-parallel-size 1 \
        --tensor-parallel-size 2 \
        --dtype bfloat16 \
        --served-model-name qwen3.8 \
        --max-num-seqs 32 \
        --max-model-len 131072 \
        --max-num-batched-tokens 16384 \
        --trust-remote-code \
        --enable-prefix-caching \
        --block-size 128 \
        --mamba-cache-mode align \
        --gpu-memory-utilization 0.85 \
        --language-model-only \
        --reasoning-parser qwen3 \
        --speculative-config '{"method":"mtp","num_speculative_tokens":3,"enforce_eager":true}' \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
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
            "model": "qwen3.8",
            "messages": [
                {
                    "role": "user",
                    "content": "Explain the role of KV cache in large language model inference."
                }
            ],
            "max_completion_tokens": 512,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": false}
        }'
    ```

    预期返回 HTTP 200，模型列表包含 `qwen3.8`，生成文本位于 `choices[0].message.content`。
