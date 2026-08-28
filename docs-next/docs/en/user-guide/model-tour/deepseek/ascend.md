The official `DeepSeek-R1-W8A8` recipe requires one Atlas 800 A3 server (16
logical devices) or two Atlas 800 A2 servers (8 devices each). The command below
is the single-node A3 path with UCM added.

```bash
export ENABLE_UCM_PATCH=1
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

vllm serve vllm-ascend/DeepSeek-R1-W8A8 \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name deepseek-r1 \
  --data-parallel-size 4 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --quantization ascend \
  --seed 1024 \
  --max-num-seqs 16 \
  --max-model-len 16384 \
  --max-num-batched-tokens 4096 \
  --block-size 128 \
  --trust-remote-code \
  --gpu-memory-utilization 0.92 \
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

For two-node A2 deployment, first reproduce the upstream multi-node launch and
HCCL communication check, then append the same `--block-size` and
`--kv-transfer-config` arguments on both nodes. This recipe deliberately omits
MTP speculation and balance-scheduling tuning until UCM reuse is verified.
