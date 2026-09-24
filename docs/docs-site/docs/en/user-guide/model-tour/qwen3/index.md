# Qwen Model Family

=== "A2"

    Deploy [Qwen3.8-27B-w8a8](https://www.modelscope.cn/models/Eco-Tech/Qwen3.8-27B-w8a8) with UCM on vLLM-Ascend using a prebuilt Docker image on Atlas 800 A2.

    ## 1. Start the Docker container

    Set `MODEL_PATH` to the local model directory and copy an A2-compatible **UCM image** reference from [Quickstart](../../quick_start/index.md#vllm-ascend) into `IMAGE`. Run on the **host**:

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

    ## 2. Configure UCM

    Run **inside the container** to create `/workspace/ucm_config_example.yaml`:

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

    This configuration enables layerwise transfers, storage health checks and metrics.

    - `persist_token_threshold: 0` does not exclude short requests, although reusable data still needs to satisfy cache-block and hybrid-state alignment.
    - With `share_buffer_enable: false`, each TP worker owns a 64 GiB host buffer. Budget 128 GiB for TP2 plus the rest of the runtime. If initialization reports a larger minimum buffer capacity for the actual KV layout, increase the capacity and provision the corresponding host RAM.
    - Qwen3.8 uses a hybrid attention layout. UCM needs both attention KV and the matching recurrent state to resume a prefix. Keep the hybrid KV cache manager enabled and set `--mamba-cache-mode align` in the serving command below.

    For parameter definitions and defaults, see [Configuration Parameters](../../../reference/config-parameters.md).

    ## 3. Launch the service

    For serving parameters, refer to the [official vLLM-Ascend guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3.8-27B.html).

    Run **inside the container**:

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

    `--kv-transfer-config` connects vLLM to UCM and specifies the configuration file created above:

    | Field | Purpose |
    | --- | --- |
    | `kv_connector` | Selects `UCMConnector`. |
    | `kv_connector_module_path` | Specifies the connector module: `ucm.integration.vllm.ucm_connector`. |
    | `kv_role` | `kv_both` enables both cache loading and saving. |
    | `kv_connector_extra_config.UCM_CONFIG_FILE` | Points to `/workspace/ucm_config_example.yaml`. |

    ## 4. Call the API

    Once the service is ready, open a **second host terminal**:

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

    Expect HTTP 200, `qwen3.8` in the model list, and generated text under `choices[0].message.content`.

=== "H100"

    Deploy `Qwen3.8-27B` with UCM on vLLM using a prebuilt Docker image on NVIDIA H100.

    ## 1. Start the Docker container

    Set `MODEL_PATH` to the local model directory and copy a CUDA **UCM image** reference supporting this model and H100 from [Quickstart](../../quick_start/index.md#vllm) into `IMAGE`. Run on the **host**:

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

    ## 2. Configure UCM

    Run **inside the container** to create `/workspace/ucm_config_example.yaml`:

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

    - `Cache|Posix` transfers KV data through host memory and persists it under `/data/ucm/cache`.
    - `cache_buffer_capacity_gb: 64` sets the host cache buffer budget.
    - `posix_capacity_gb: 0` disables capacity-based Posix garbage collection; it does not disable filesystem writes.

    For parameter definitions, see [Configuration Parameters](../../../reference/config-parameters.md).

    ## 3. Launch the service

    Run **inside the container**:

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

    `--kv-transfer-config` connects vLLM to UCM and specifies the configuration file created above:

    | Field | Purpose |
    | --- | --- |
    | `kv_connector` | Selects `UCMConnector`. |
    | `kv_connector_module_path` | Specifies the connector module: `ucm.integration.vllm.ucm_connector`. |
    | `kv_role` | `kv_both` enables both cache loading and saving. |
    | `kv_connector_extra_config.UCM_CONFIG_FILE` | Points to `/workspace/ucm_config_example.yaml`. |

    ## 4. Call the API

    Once the service is ready, open a **second host terminal**:

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

    Expect HTTP 200, `qwen3.8` in the model list, and generated text under `choices[0].message.content`.
