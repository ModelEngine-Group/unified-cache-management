# GLM Model Family

=== "A2"

    Deploy [GLM-4.7-W8A8-floatmtp](https://www.modelscope.cn/models/Eco-Tech/GLM-4.7-W8A8-floatmtp) with UCM on vLLM-Ascend using a prebuilt Docker image on Atlas 800 A2.

    ## 1. Start the Docker container

    Set `MODEL_PATH` to the local model directory and copy a **UCM image** reference supporting GLM-4.7 and your Ascend hardware from [Quickstart](../../quick_start/index.md#vllm-ascend) into `IMAGE`. Run on the **host**:

    ```bash
    export MODEL_PATH=/data/weights/GLM-4.7-W8A8-floatmtp
    export IMAGE='<full Ascend UCM image reference from Quickstart>'
    export CONTAINER_NAME=glm47-ucm

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
        -v "$MODEL_PATH:/data/weights/GLM-4.7-W8A8-floatmtp:ro" \
        -v /data/ucm:/data/ucm \
        -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        -e ENABLE_UCM_PATCH=1 \
        -e UCM_LOG_PATH=/data/ucm/log \
        --workdir /workspace \
        --entrypoint /bin/bash \
        "$IMAGE"
    ```

    ## 2. Configure UCM

    Run **inside the container** to create `/workspace/ucm_config_example.yaml`:

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

    - `Cache|Posix` transfers KV data through host memory and persists it under `/data/ucm/cache`.
    - `cache_buffer_capacity_gb: 64` sets the host cache buffer budget.
    - `posix_capacity_gb: 0` disables capacity-based Posix garbage collection; it does not disable filesystem writes.

    For parameter definitions, see [Configuration Parameters](../../../reference/config-parameters.md).

    ## 3. Launch the service

    For serving parameters, refer to the [official vLLM-Ascend guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/GLM4.x.html).

    Run **inside the container**:

    ```bash
    export MODEL_PATH=/data/weights/GLM-4.7-W8A8-floatmtp
    export ENABLE_UCM_PATCH=1
    export VLLM_CPU_AFFINITY=1
    export HCCL_BUFFSIZE=512

    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=1
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export HCCL_OP_EXPANSION_MODE=AIV

    export VLLM_ASCEND_BALANCE_SCHEDULING=1
    export VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE=1

    vllm serve "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port 7800 \
        --data-parallel-size 1 \
        --tensor-parallel-size 8 \
        --enable-expert-parallel \
        --seed 1024 \
        --served-model-name glm \
        --max-model-len 133000 \
        --max-num-batched-tokens 8192 \
        --max-num-seqs 16 \
        --async-scheduling \
        --quantization ascend \
        --trust-remote-code \
        --gpu-memory-utilization 0.9 \
        --speculative-config '{"num_speculative_tokens":3,"method":"mtp"}' \
        --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,32,64,128,256,512],"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --additional-config '{"enable_cpu_binding":false,"enable_shared_expert_dp":true,"ascend_fusion_config":{"fusion_ops_gmmswigluquant":false}}' \
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
            "model": "glm",
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

    Expect HTTP 200, `glm` in the model list, and a chat completion response containing the `choices` field.

=== "H100"

    Deploy `GLM-4.7-FP8` with UCM on vLLM using a prebuilt Docker image on NVIDIA H100.

    ## 1. Start the Docker container

    Set `MODEL_PATH` to the local model directory and copy a CUDA **UCM image** reference supporting this model and H100 from [Quickstart](../../quick_start/index.md#vllm) into `IMAGE`. Run on the **host**:

    ```bash
    export MODEL_PATH=/data/weights/GLM-4.7-FP8
    export IMAGE='<full CUDA UCM image reference from Quickstart>'
    export CONTAINER_NAME=glm47-ucm-h100

    mkdir -p /data/ucm/cache /data/ucm/log
    docker pull "$IMAGE"

    docker run --rm -it \
        --name "$CONTAINER_NAME" \
        --gpus '"device=0,1,2,3,4,5,6,7"' \
        --ipc=host \
        --network=host \
        -v "$MODEL_PATH:/data/weights/GLM-4.7-FP8:ro" \
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
    export MODEL_PATH=/data/weights/GLM-4.7-FP8
    export ENABLE_UCM_PATCH=1
    export UCM_LOG_PATH=/data/ucm/log

    vllm serve "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port 7800 \
        --data-parallel-size 1 \
        --tensor-parallel-size 8 \
        --served-model-name glm47 \
        --max-model-len 32768 \
        --max-num-seqs 8 \
        --max-num-batched-tokens 8192 \
        --trust-remote-code \
        --gpu-memory-utilization 0.90 \
        --enable-prefix-caching \
        --block-size 128 \
        --enable-auto-tool-choice \
        --tool-call-parser glm47 \
        --reasoning-parser glm45 \
        --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
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
            "model": "glm47",
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

    Expect HTTP 200, `glm47` in the model list, and a chat completion response containing the `choices` field.
