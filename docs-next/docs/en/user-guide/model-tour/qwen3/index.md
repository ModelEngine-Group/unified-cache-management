# Qwen

[Official vLLM Ascend tutorials](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/index.html) for the Qwen family. To use UCM, jump to the [Qwen3.8-27B deployment and API example](#qwen38-27b-ucm) below.

## vLLM Ascend model guides

| Model | vLLM Ascend latest guide |
| --- | --- |
| Qwen3-Dense(0.6B/1.7B/4B/8B/14B/32B) | [Official guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3-Dense.html) |
| Qwen-VL-Dense(8B/32B) | [Official guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen-VL-Dense.html) |
| Qwen3-30B-A3B | [Official guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3-30B-A3B.html) |
| Qwen3-235B-A22B | [Official guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3-235B-A22B.html) |
| Qwen3-VL-30B-A3B-Instruct | [Official guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3-VL-30B-A3B-Instruct.html) |
| Qwen3-VL-235B-A22B-Instruct | [Official guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3-VL-235B-A22B-Instruct.html) |
| Qwen3-Coder-30B-A3B | [Official guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3-Coder-30B-A3B.html) |
| Qwen3-Embedding | [Official guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3-Embedding.html) |
| Qwen3-VL-Embedding | [Official guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3-VL-Embedding.html) |
| Qwen3-Reranker | [Official guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3-Reranker.html) |
| Qwen3-VL-Reranker | [Official guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3-VL-Reranker.html) |
| Qwen3-Next | [Official guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3-Next.html) |
| Qwen3-Omni-30B-A3B-Thinking | [Official guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3-Omni-30B-A3B-Thinking.html) |
| Qwen3.5-27B & Qwen3.6-27B | [Official guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3.5-27B-Qwen3.6-27B.html) |
| Qwen3.5-Dense (2B/4B/9B) | [Official guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3.5-Dense.html) |
| Qwen3.5-397B-A17B | [Official guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3.5-397B-A17B.html) |
| Qwen3.6-35B-A3B | [Official guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3.6-35B-A3B.html) |
| Qwen3.8-2.4T-A95B | [Official guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3.8-2.4T-A95B.html) |
| Qwen3.8-27B | [Official guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3.8-27B.html) |
| Qwen3-ASR-1.7B | [Official guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3-ASR-1.7B.html) |
| Qwen2.5-Math-RM-72B | [Official guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen2.5-Math-RM-72B.html) |

## Qwen3.8-27B UCM example { #qwen38-27b-ucm }

Run [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) in a single Docker
container with UCM Prefix Cache. This example serves text with BF16 weights,
an 8,192-token context and up to eight concurrent sequences.
These example commands have not been validated on accelerator hardware.

### Deploy

Download the complete model into a local directory first. Use a Linux host with
Docker and either NVIDIA Container Toolkit or the Ascend driver required by the
selected image. The tabs below use two 80 GB CUDA GPUs or one Atlas 800 A2 node
with eight 64 GB NPUs; these are example allocations, not minimum requirements.

Open [Installation](../../quick_start/index.md), select **Image**, your platform and
host architecture, and copy the full image reference into `UCM_IMAGE`. Choose a
published vLLM **0.28.0** runtime for CUDA or vLLM-Ascend **0.25.1rc0** for
the A2 example. The image already includes UCM; use the complete reference from the
selector. Set `UCM_MODEL_DIR` to the downloaded model directory, then run:

```bash
export UCM_MODEL_DIR=/srv/models/Qwen3.8-27B
export UCM_WORK_DIR="$PWD/ucm-qwen38"
export UCM_IMAGE='<copy the image reference from Installation>'
mkdir -p "$UCM_WORK_DIR/cache"
docker pull "$UCM_IMAGE"
```

Each tab writes its UCM configuration before launching Docker. `Cache` transfers
KV between the device and host memory; `Posix` persists it in the mounted
directory. Budget **160 GiB** of host cache memory for CUDA TP2 or **256 GiB**
for Ascend TP8, plus engine memory. If a different engine layout reports a
larger minimum buffer capacity during initialization, increase the configured
capacity accordingly. Both commands use host shared memory through `--ipc=host`.

Qwen3.8-27B mixes full attention and Gated DeltaNet. Keep the hybrid KV cache
manager enabled and use `--mamba-cache-mode align` so UCM can restore attention
KV and the corresponding recurrent state. `use_layerwise: false` selects bulk
loading for this initial deployment.

=== "CUDA · vLLM"

    Select the CUDA image before running this tab. Devices 0 and 1 are exposed
    to the container; the command uses TP2.

    ```bash
    cat > "$UCM_WORK_DIR/ucm.yaml" <<'YAML'
    use_layerwise: false
    ucm_connectors:
      - ucm_connector_name: UcmPipelineStore
        ucm_connector_config:
          store_pipeline: "Cache|Posix"
          storage_backends: /mnt/ucm-cache
          cache_buffer_capacity_gb: 80
          posix_io_engine: psync
          io_direct: false
          timeout_ms: 30000
    YAML
    ```

    ```bash
    docker run -d --name ucm-qwen38 \
      --gpus '"device=0,1"' --ipc=host \
      -p 8000:8000 \
      -e ENABLE_UCM_PATCH=1 \
      -v "$UCM_MODEL_DIR:/models/Qwen3.8-27B:ro" \
      -v "$UCM_WORK_DIR/ucm.yaml:/etc/ucm/ucm.yaml:ro" \
      -v "$UCM_WORK_DIR/cache:/mnt/ucm-cache" \
      --entrypoint vllm "$UCM_IMAGE" serve /models/Qwen3.8-27B \
      --served-model-name qwen38-27b \
      --host 0.0.0.0 --port 8000 \
      --tensor-parallel-size 2 \
      --dtype bfloat16 --max-model-len 8192 --max-num-seqs 8 \
      --block-size 128 --mamba-cache-mode align \
      --enable-prefix-caching --enforce-eager \
      --language-model-only --reasoning-parser qwen3 \
      --kv-transfer-config '{"kv_connector":"UCMConnector","kv_connector_module_path":"ucm.integration.vllm.ucm_connector","kv_role":"kv_both","kv_connector_extra_config":{"UCM_CONFIG_FILE":"/etc/ucm/ucm.yaml"}}'
    ```

=== "Ascend · vLLM-Ascend"

    Select an A2-compatible image before running this tab. The command exposes
    devices 0–7 and the host driver to the container and uses TP8. Its driver
    paths match the Atlas 800 A2 example; A3 hosts require their corresponding
    image and device configuration.

    ```bash
    cat > "$UCM_WORK_DIR/ucm.yaml" <<'YAML'
    use_layerwise: false
    ucm_connectors:
      - ucm_connector_name: UcmPipelineStore
        ucm_connector_config:
          store_pipeline: "Cache|Posix"
          storage_backends: /mnt/ucm-cache
          cache_buffer_capacity_gb: 32
          posix_io_engine: psync
          io_direct: false
          timeout_ms: 30000
    YAML
    ```

    ```bash
    docker run -d --name ucm-qwen38 \
      --network host --ipc=host \
      --device /dev/davinci0 --device /dev/davinci1 \
      --device /dev/davinci2 --device /dev/davinci3 \
      --device /dev/davinci4 --device /dev/davinci5 \
      --device /dev/davinci6 --device /dev/davinci7 \
      --device /dev/davinci_manager \
      --device /dev/devmm_svm --device /dev/hisi_hdc \
      -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
      -v /usr/local/dcmi:/usr/local/dcmi:ro \
      -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
      -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
      -e ENABLE_UCM_PATCH=1 \
      -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
      -e HCCL_BUFFSIZE=200 \
      -v "$UCM_MODEL_DIR:/models/Qwen3.8-27B:ro" \
      -v "$UCM_WORK_DIR/ucm.yaml:/etc/ucm/ucm.yaml:ro" \
      -v "$UCM_WORK_DIR/cache:/mnt/ucm-cache" \
      --entrypoint vllm "$UCM_IMAGE" serve /models/Qwen3.8-27B \
      --served-model-name qwen38-27b \
      --host 0.0.0.0 --port 8000 \
      --tensor-parallel-size 8 \
      --dtype bfloat16 --max-model-len 8192 --max-num-seqs 8 \
      --block-size 128 --mamba-cache-mode align \
      --enable-prefix-caching --enforce-eager \
      --language-model-only --reasoning-parser qwen3 \
      --kv-transfer-config '{"kv_connector":"UCMConnector","kv_connector_module_path":"ucm.integration.vllm.ucm_connector","kv_role":"kv_both","kv_connector_extra_config":{"UCM_CONFIG_FILE":"/etc/ucm/ucm.yaml"}}'
    ```

Inspect startup with `docker logs -f ucm-qwen38`. After the server is ready,
leave the log view with Ctrl-C and use the same host terminal for the call below.

### Call

Use `curl` and `jq` on the host. `/health` should return HTTP 200; the request
then prints `choices[0].message.content` from the JSON response.

```bash
curl --fail http://127.0.0.1:8000/health

curl --fail-with-body --silent --show-error \
  http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen38-27b",
    "messages": [{"role": "user", "content": "Explain what a KV cache stores in two sentences."}],
    "max_tokens": 128,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": false}
  }' | jq -r '.choices[0].message.content'
```

Official references: [model and weights](https://huggingface.co/Qwen/Qwen3.8-27B),
[vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B),
[vLLM-Ascend recipe](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3.8-27B.html).
