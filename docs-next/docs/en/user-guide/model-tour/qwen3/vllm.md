`Qwen3-8B` can start with tensor parallel size 1 when it fits on one GPU. Set
`--tensor-parallel-size` to the number of GPUs used by larger variants.

```bash
export ENABLE_UCM_PATCH=1
export MODEL=Qwen/Qwen3-8B

vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name qwen3 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --block-size 128 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --reasoning-parser qwen3 \
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

For Qwen3 MoE, VL, Next, or quantized checkpoints, begin with the corresponding
upstream model recipe and preserve the UCM `--block-size` and
`--kv-transfer-config` arguments shown here.
