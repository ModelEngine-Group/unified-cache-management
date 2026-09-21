# DeepSeek

[Official vLLM Ascend tutorials](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/index.html) for the DeepSeek family. To use UCM, jump to the [DeepSeek-V4-Flash deployment and API example](#deepseek-v4-flash-ucm) below.

## vLLM Ascend model guides

| Model | vLLM Ascend latest guide |
| --- | --- |
| DeepSeek-V3 & 3.1 | [Official guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-V3.1.html) |
| DeepSeek-V3.2 | [Official guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-V3.2.html) |
| DeepSeek-V4-Flash | [Official guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-V4-Flash.html) |
| DeepSeek-V4-Flash-Vision-Exp (Experimental) | [Official guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-V4-Flash-Vision.html) |
| DeepSeek-V4.1-Flash | [Official guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-V4.1-Flash.html) |
| DeepSeek-V4-Pro | [Official guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-V4-Pro.html) |
| DeepSeek-R1 | [Official guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-R1.html) |
| DeepSeek-OCR-2 | [Official guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeekOCR2.html) |

## DeepSeek-V4-Flash UCM example { #deepseek-v4-flash-ucm }

Run DeepSeek-V4-Flash in one Docker container with UCM Prefix Cache. Choose the
hardware tab below; CUDA and Ascend use different weight formats. Validate model output and external reuse on the target accelerator hardware.

### Deploy

Use Linux with Docker and a working host accelerator driver. For CUDA, install the
NVIDIA Container Toolkit; for Ascend, ensure the host driver and device files match
the selected runtime. Download the complete model, including its tokenizer and
configuration, using the link in your platform tab.

Open [Installation](../../quick_start/index.md), select **Image**, and copy the published
UCM runtime coordinate into `UCM_IMAGE`. The commands below target vLLM **0.28.0**
on CUDA and vLLM-Ascend **0.25.1rc0 / A2** on Ascend. Select the matching architecture
and backend; if that combination is absent, use a release that publishes it.

Prepare the UCM configuration on the host:

```bash
export UCM_WORKDIR="$PWD/ucm-deepseek-v4"
mkdir -p "$UCM_WORKDIR/cache/cuda" "$UCM_WORKDIR/cache/ascend"
cat > "$UCM_WORKDIR/ucm.yaml" <<'YAML'
use_layerwise: false
load_tokens_threshold: 2048
ucm_connectors:
  - ucm_connector_name: UcmPipelineStore
    ucm_connector_config:
      store_pipeline: "Cache|Posix"
      storage_backends: /mnt/ucm-cache
      cache_buffer_capacity_gb: 128
      io_direct: false
YAML
```

Reserve 128 GiB of host memory for UCM, split into two 64 GiB Stores, plus memory
for the engine and model loading. The containers allow 160 GiB of shared memory.
Cache data stays in the mounted host directory. Run one container at a time.

=== "CUDA / vLLM"

    Use one node with **8 B200 or B300 GPUs**, following the official
    [single-node TP recipe](https://github.com/vllm-project/recipes/blob/main/models/deepseek-ai/DeepSeek-V4-Flash.yaml).
    Download [deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
    and set `MODEL_DIR` to its absolute local directory. This is the original
    FP4-expert/FP8 checkpoint; these commands do not cover Hopper or the separate
    `0731`/DSpark checkpoints.

    ```bash
    export UCM_IMAGE='<CUDA UCM image copied from Installation>'
    export MODEL_DIR=/absolute/path/DeepSeek-V4-Flash

    docker run -d --name ucm-deepseek-cuda \
      --gpus all --shm-size 160g \
      -p 127.0.0.1:8000:8000 \
      -e ENABLE_UCM_PATCH=1 \
      -v "$MODEL_DIR:/models/DeepSeek-V4-Flash:ro" \
      -v "$UCM_WORKDIR/ucm.yaml:/etc/ucm/ucm.yaml:ro" \
      -v "$UCM_WORKDIR/cache/cuda:/mnt/ucm-cache" \
      --entrypoint vllm "$UCM_IMAGE" \
      serve /models/DeepSeek-V4-Flash \
      --served-model-name deepseek-v4-flash \
      --host 0.0.0.0 --port 8000 \
      --tensor-parallel-size 8 \
      --data-parallel-size 1 \
      --pipeline-parallel-size 1 \
      --distributed-executor-backend mp \
      --tokenizer-mode deepseek_v4 \
      --trust-remote-code \
      --kv-cache-dtype fp8 \
      --block-size 256 \
      --max-model-len 8192 \
      --max-num-seqs 1 \
      --gpu-memory-utilization 0.9 \
      --enforce-eager \
      --kv-transfer-config '{
        "kv_connector": "UCMConnector",
        "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/etc/ucm/ucm.yaml"}
      }'

    docker logs -f ucm-deepseek-cuda
    ```

=== "Ascend A2 / vLLM-Ascend"

    Use one **Atlas 800 A2 node with 8 × 64 GB NPUs** and the quantized
    [DeepSeek-V4-Flash-w8a8-mtp weights](https://www.modelscope.cn/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp).
    Set `MODEL_DIR` to their absolute local directory. The checkpoint includes an
    MTP head; this example does not enable speculative decoding. The command uses
    the standard device and driver paths from the
    [Ascend deployment guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-V4-Flash.html).

    ```bash
    export UCM_IMAGE='<Ascend A2 UCM image copied from Installation>'
    export MODEL_DIR=/absolute/path/DeepSeek-V4-Flash-w8a8-mtp

    docker run -d --name ucm-deepseek-ascend \
      --network host --shm-size 160g --privileged \
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
      -v /etc/hccn.conf:/etc/hccn.conf:ro \
      -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
      -e ENABLE_UCM_PATCH=1 \
      -e OMP_PROC_BIND=false -e OMP_NUM_THREADS=10 \
      -e PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
      -e HCCL_BUFFSIZE=1024 -e HCCL_OP_EXPANSION_MODE=AIV \
      -e TASK_QUEUE_ENABLE=1 \
      -v "$MODEL_DIR:/models/DeepSeek-V4-Flash-w8a8-mtp:ro" \
      -v "$UCM_WORKDIR/ucm.yaml:/etc/ucm/ucm.yaml:ro" \
      -v "$UCM_WORKDIR/cache/ascend:/mnt/ucm-cache" \
      --entrypoint vllm "$UCM_IMAGE" \
      serve /models/DeepSeek-V4-Flash-w8a8-mtp \
      --served-model-name deepseek-v4-flash \
      --host 127.0.0.1 --port 8000 \
      --tensor-parallel-size 8 \
      --data-parallel-size 1 \
      --pipeline-parallel-size 1 \
      --enable-expert-parallel \
      --distributed-executor-backend mp \
      --tokenizer-mode deepseek_v4 \
      --trust-remote-code \
      --quantization ascend \
      --block-size 128 \
      --max-model-len 8192 \
      --max-num-seqs 1 \
      --gpu-memory-utilization 0.9 \
      --enforce-eager \
      --kv-transfer-config '{
        "kv_connector": "UCMConnector",
        "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/etc/ucm/ucm.yaml"}
      }'

    docker logs -f ucm-deepseek-ascend
    ```

Wait for the server to finish loading. The UCM startup log should contain
`Init UCM FAWA connector` and the `FAWA FA` / `FAWA WA` Store configurations.
Press Ctrl+C to leave `docker logs`; the detached container keeps running.

### Call

From the same host, check readiness and send a completion request:

```bash
curl --fail http://127.0.0.1:8000/health

curl --fail http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4-flash",
    "prompt": "A KV cache speeds up language model inference by",
    "max_tokens": 128,
    "temperature": 0
  }'
```

`/health` returns HTTP 200 when ready. The completion response contains generated
text in `choices[0].text` and token usage in `usage`. This short request verifies
the API call; it does not demonstrate an external-cache hit.

For other hardware layouts, consult the [vLLM recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash)
or the [Ascend model guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-V4-Flash.html).
