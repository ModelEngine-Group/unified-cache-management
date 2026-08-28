The full FP8 checkpoint normally starts on eight high-memory Hopper-class GPUs.
This command favors a simple TP8 correctness run before DP/EP performance
tuning.

```bash
export ENABLE_UCM_PATCH=1
export MODEL=deepseek-ai/DeepSeek-R1

vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name deepseek-r1 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --max-model-len 32768 \
  --max-num-batched-tokens 8192 \
  --block-size 128 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --reasoning-parser deepseek_r1 \
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

If eight devices cannot hold the checkpoint, use the model's official
multi-node recipe and add the same UCM connector to every engine process. All
processes must mount the same storage directory and use the same block size.
Distilled R1 checkpoints can be used for a smaller smoke test, but their cache
layout follows their base architecture and does not validate full DeepSeek-R1
MLA behavior.
