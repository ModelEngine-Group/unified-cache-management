The upstream GLM-4.x guide validates `Eco-Tech/GLM-4.7-W8A8-floatmtp` on one
Atlas 800 A3 (16 logical devices) or one Atlas 800 A2 (8 devices). The following
keeps its model-specific settings and adds the UCM connector.

```bash
export ENABLE_UCM_PATCH=1
export HCCL_BUFFSIZE=512
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_OP_EXPANSION_MODE=AIV
export VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE=1

vllm serve Eco-Tech/GLM-4.7-W8A8-floatmtp \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name glm \
  --data-parallel-size 2 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --max-model-len 32768 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 16 \
  --block-size 128 \
  --quantization ascend \
  --trust-remote-code \
  --reasoning-parser glm45 \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --gpu-memory-utilization 0.90 \
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

This first-run recipe omits MTP speculation and graph-tuning flags so cache
correctness can be isolated. Add them only after the repeated-prefix check
passes. Use the A2/A3 image and CANN combination selected in
[Installation](../../installation.md); A5 is not covered by this recipe.
