# Qwen3-32B

Qwen3-32B (GQA cache layout).

## Docker

```bash
# CUDA
docker pull ghcr.io/modelengine-group/ucm-vllm:0.5.0
docker run --gpus all --rm --net=host --shm-size=16g \
  -v /path/to/models:/models \
  -v /path/to/ucm_config_example.yaml:/app/ucm_config_example.yaml \
  ghcr.io/modelengine-group/ucm-vllm:0.5.0

# NPU (Ascend 910C)
docker pull quay.io/ascend/vllm-ascend:v0.18.0rc1-a3
docker run --rm --net=host --shm-size=500g --privileged=true \
  --device=/dev/davinci_manager --device=/dev/hisi_hdc --device=/dev/devmm_svm \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /path/to/models:/models \
  -v /path/to/ucm_config_example.yaml:/app/ucm_config_example.yaml \
  quay.io/ascend/vllm-ascend:v0.18.0rc1-a3
```

## Serve

```bash
# CUDA
vllm serve Qwen/Qwen3-32B \
  --max-model-len 131072 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.87 \
  --block-size 128 \
  --trust-remote-code \
  --enable-expert-parallel \
  --port 7800 \
  --kv-transfer-config \
  '{"kv_connector":"UCMConnector","kv_connector_module_path":"ucm.integration.vllm.ucm_connector","kv_role":"kv_both","kv_connector_extra_config":{"UCM_CONFIG_FILE":"/app/ucm_config_example.yaml"}}'

# NPU (Ascend)
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ENABLE_UCM_PATCH=1
vllm serve /models/Qwen3-32B \
  --max-model-len 131072 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.9 \
  --quantization ascend \
  --enable-expert-parallel \
  --trust-remote-code \
  --port 7800 \
  --kv-transfer-config \
  '{"kv_connector":"UCMConnector","kv_connector_module_path":"ucm.integration.vllm.ucm_connector","kv_role":"kv_both","kv_connector_extra_config":{"UCM_CONFIG_FILE":"/app/ucm_config_example.yaml"}}'
```

## UCM Configuration

`ucm_config_example.yaml` (GQA, ~32 GB/card):

```yaml
ucm_connectors:
  - ucm_connector_name: "UcmPipelineStore"
    ucm_connector_config:
      store_pipeline: "Cache|Posix"
      storage_backends: "/mnt/ucm-storage"
      io_direct: true
      posix_io_engine: "aio"
      cache_buffer_capacity_gb: 32
      posix_capacity_gb: 1024
enable_event_sync: true
use_layerwise: true
```

## Verify

```bash
curl http://127.0.0.1:7800/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-32B","messages":[{"role":"user","content":"hi"}],"max_tokens":32}'
```
