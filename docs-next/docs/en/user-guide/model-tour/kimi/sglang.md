!!! warning

    This is a UCM qualification command, not a release-certified production
    profile. Run cache-correctness and accuracy checks before performance work.

The `HICACHE_CONFIG` variable is defined in **Before you start**. Moonshot's
reference TP deployment uses 16 GPUs across two nodes; the command below is a
single-server TP8 starting point for hardware on which the checkpoint fits.

```bash
python3 -m sglang.launch_server \
  --model-path moonshotai/Kimi-K2-Thinking \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name kimi-k2-thinking \
  --tensor-parallel-size 8 \
  --context-length 8192 \
  --page-size 128 \
  --trust-remote-code \
  --reasoning-parser kimi_k2 \
  --tool-call-parser kimi_k2 \
  --enable-hierarchical-cache \
  --hicache-mem-layout page_first \
  --hicache-write-policy write_through \
  --hicache-storage-backend dynamic \
  --hicache-storage-prefetch-policy wait_complete \
  --hicache-storage-backend-extra-config "$HICACHE_CONFIG"
```

For two-node TP16, add SGLang's `--dist-init-addr`, `--nnodes 2`, and
node-specific `--node-rank` arguments on both nodes. Mount `/mnt/ucm` at the
same path on both nodes and keep the HiCache settings identical.
