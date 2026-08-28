!!! warning

    This is a UCM qualification command, not a release-certified production
    profile. The upstream model launch is supported, but Kimi-K2 UCM cache
    correctness must still be established for the selected release.

The upstream guide validates Kimi-K2-Thinking on one Atlas 800 A3 server with
16 logical devices and requires TP16.

```bash
export ENABLE_UCM_PATCH=1
export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1
export OMP_PROC_BIND=false
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

vllm serve moonshotai/Kimi-K2-Thinking \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name kimi-k2-thinking \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --max-model-len 8192 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 12 \
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

The official upstream example disables prefix caching; this UCM variant
intentionally omits that flag. A2 and A5 are not claimed by this recipe.
