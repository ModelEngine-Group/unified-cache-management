The `HICACHE_CONFIG` variable is defined in **Before you start**. Use the same
four-GPU starting point as the vLLM recipe.

```bash
python3 -m sglang.launch_server \
  --model-path zai-org/GLM-4.5-Air \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name glm \
  --tensor-parallel-size 4 \
  --context-length 32768 \
  --page-size 128 \
  --trust-remote-code \
  --reasoning-parser glm45 \
  --tool-call-parser glm45 \
  --enable-hierarchical-cache \
  --hicache-mem-layout page_first \
  --hicache-write-policy write_through \
  --hicache-storage-backend dynamic \
  --hicache-storage-prefetch-policy wait_complete \
  --hicache-storage-backend-extra-config "$HICACHE_CONFIG"
```

The command covers GLM-4.x. Do not substitute GLM-5/5.1/5.2 on SGLang until
that combination appears as supported in the UCM release matrix.
