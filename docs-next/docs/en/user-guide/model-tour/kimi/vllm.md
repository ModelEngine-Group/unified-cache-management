!!! warning

    This is a UCM qualification command, not a release-certified production
    profile. Record the exact UCM, vLLM, CUDA, and checkpoint revisions with
    test results.

The upstream low-latency Kimi-K2-Thinking recipe uses eight H200/H20 GPUs.

```bash
export ENABLE_UCM_PATCH=1
export MODEL=moonshotai/Kimi-K2-Thinking

vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name kimi-k2-thinking \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --max-model-len 8192 \
  --max-num-batched-tokens 8192 \
  --block-size 128 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --reasoning-parser kimi_k2 \
  --enable-auto-tool-choice \
  --tool-call-parser kimi_k2 \
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

Do not add `--no-enable-prefix-caching`: that disables the prefix-cache path
the UCM qualification is intended to exercise.
