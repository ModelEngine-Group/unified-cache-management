The upstream Qwen3 Dense guide covers BF16 checkpoints from 0.6B through 32B
on both Atlas A2 and A3. This 8B example starts on one logical NPU; increase TP
when the selected checkpoint or concurrency target requires it.

```bash
export ENABLE_UCM_PATCH=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
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

Add `--quantization ascend` only for a checkpoint packaged for Ascend
quantization. This recipe covers A2/A3; it does not claim A5 validation.
