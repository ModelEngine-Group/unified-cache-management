Use four GPUs for the standard `GLM-4.5-Air` checkpoint. The exact count may
change with quantization and available device memory.

```bash
export ENABLE_UCM_PATCH=1
export MODEL=zai-org/GLM-4.5-Air

vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name glm \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --max-model-len 32768 \
  --block-size 128 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --reasoning-parser glm45 \
  --enable-auto-tool-choice \
  --tool-call-parser glm45 \
  --kv-transfer-config \
  '{
    "kv_connector":"UCMConnector",
    "kv_connector_module_path":"ucm.integration.vllm.ucm_connector",
    "kv_role":"kv_both",
    "kv_connector_extra_config":{
      "UCM_CONFIG_FILE":"/etc/ucm/model-tour.yaml"
    }
  }'
```

`--block-size 128` must remain identical across processes that share the UCM
store. For a quantized GLM checkpoint, add the quantization option required by
that checkpoint rather than assuming that the BF16 command applies unchanged.
