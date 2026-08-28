The `HICACHE_CONFIG` variable is defined in **Before you start**. TP8 is a
starting point for a quantized checkpoint on one high-memory GPU server; use
SGLang's official multi-node topology when the weights do not fit.

```bash
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-R1 \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name deepseek-r1 \
  --tensor-parallel-size 8 \
  --context-length 32768 \
  --page-size 128 \
  --trust-remote-code \
  --reasoning-parser deepseek-r1 \
  --enable-hierarchical-cache \
  --hicache-mem-layout page_first \
  --hicache-write-policy write_through \
  --hicache-storage-backend dynamic \
  --hicache-storage-prefetch-policy wait_complete \
  --hicache-storage-backend-extra-config "$HICACHE_CONFIG"
```

Keep SGLang's RadixAttention cache enabled; UCM HiCache extends the hierarchy
below device memory. Disabling the radix cache changes the normal prefix-cache
path and is not part of this recipe.
