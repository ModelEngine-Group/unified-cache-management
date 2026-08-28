The `HICACHE_CONFIG` variable is defined in **Before you start**.

```bash
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name qwen3 \
  --tensor-parallel-size 1 \
  --context-length 32768 \
  --page-size 128 \
  --trust-remote-code \
  --reasoning-parser qwen3 \
  --enable-hierarchical-cache \
  --hicache-mem-layout page_first \
  --hicache-write-policy write_through \
  --hicache-storage-backend dynamic \
  --hicache-storage-prefetch-policy wait_complete \
  --hicache-storage-backend-extra-config "$HICACHE_CONFIG"
```

The model's native context is sufficient for the 32K example. Enable YaRN only
when a workload genuinely needs a longer context and validate cache hits again,
because changing RoPE configuration changes model execution semantics.
