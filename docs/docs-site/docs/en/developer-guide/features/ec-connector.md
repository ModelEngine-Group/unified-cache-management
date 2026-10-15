# Multimodal Encoder Cache Reuse

In multimodal request processing, the vision encoder (ViT) recomputes every image and video for every request. With UCM's EC Connector enabled, encoder outputs are stored in UCM's tiered storage: a shared-memory hot tier plus a Posix cold tier on local or mounted disks. Within the same model instance, subsequent requests carrying identical media content hit the cache directly and skip encoder recomputation, substantially reducing TTFT for repeated-media workloads.

Typical benefit scenarios: the same set of images/videos referenced repeatedly across requests, such as multi-turn image Q&A, repeated asset review, or batch image tagging. The larger the images and the more repeats, the bigger the gain.

This page covers how to start the service, configure the YAML, and verify the effect. For internals (chunking, key encoding, failure semantics), see the [UCM EC Connector design](../design-docs/ucm-ec-connector.md).

## Prerequisites

- A vLLM version with the EC Connector scheduling interfaces and UCM installed (providing the `ucm.integration.vllm.ucm_ec_connector` module);
- A multimodal model (example: `Qwen3.8-27B-w8a8`);
- A device environment that supports EC. The commands below use Ascend NPU with a vllm-ascend container as the reference; the structure is identical on CUDA — just replace the device-specific environment variables.

## Start the service (Qwen3.8-27B-w8a8 example)

```bash
# Enable cache-reset endpoints for verification and EC load/save logs.
export VLLM_SERVER_DEV_MODE=1
export UCM_LOG_LEVEL=debug

vllm serve /models/Qwen3.8-27B-w8a8/ \
  --host 0.0.0.0 --port 8000 \
  --data-parallel-size 1 --tensor-parallel-size 2 \
  --quantization ascend \
  --served-model-name qwen3.8 \
  --max-num-seqs 32 --max-model-len 131072 --max-num-batched-tokens 16384 \
  --trust-remote-code \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.85 \
  --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --additional-config '{"enable_cpu_binding":true}' \
  --no-disable-hybrid-kv-cache-manager \
  --ec-transfer-config '{"ec_connector":"UCMECConnector","ec_connector_module_path":"ucm.integration.vllm.ucm_ec_connector","ec_role":"ec_both","ec_connector_extra_config":{"UCM_CONFIG_FILE":"/workspace/unified-cache-management/examples/ucm_ec_config.yaml"}}'
```

Key parameters:

| Parameter | Description |
| --- | --- |
| `--ec-transfer-config` | The EC feature entry point. `ec_role: ec_both` makes the same instance both write and read the cache; `UCM_CONFIG_FILE` points to the UCM YAML — the example uses the repository's `examples/ucm_ec_config.yaml` directly |
| `--speculative-config` / `--compilation-config` / `--additional-config` | qwen3_5_mtp speculative decoding, full decode CUDA Graph, and CPU binding — engine-side performance settings, unrelated to EC |
| `--no-disable-hybrid-kv-cache-manager` / `--enable-prefix-caching` | KV cache manager and prefix caching settings required by hybrid models (linear attention + full attention) |
| `--quantization ascend` | w8a8 quantized models on vllm-ascend must use `ascend`, not `w8a8_dynamic` |

Successful startup logs (`Initialized UCM EC connector` appears once for SCHEDULER and once for WORKER):

```text
Initialized UCM EC connector: role=SCHEDULER, producer=True, consumer=True, width=..., dtype=..., rows_per_chunk=..., ...
Initialized UCM EC connector: role=WORKER, producer=True, consumer=True, ...
Creating UCM EC store UcmPipelineStore: unique_id=..., device_id=..., chunk_bytes=..., ...
```

## Configure the UCM YAML

The example command points `UCM_CONFIG_FILE` directly at the repository's `examples/ucm_ec_config.yaml`. The YAML has a single top-level section `ucm_ec_connector`:

```yaml
ucm_ec_connector:
  ucm_connector_name: "UcmPipelineStore"
  ucm_connector_config:
    store_pipeline: "Cache|Posix"
    storage_backends: "/mnt/test"         # Cold-tier directory; adjust to a writable path, colon-separate for several
    posix_capacity_gb: 10240
    posix_io_engine: "psync"
    posix_data_trans_concurrency: 128
    cache_buffer_capacity_gb: 32
  encoder_cache_config:
    chunk_size: 128                                    # Rows per chunk, must be a positive integer
```

| Field | Required | Description |
| --- | --- | --- |
| `ucm_connector_name` | Yes | Producer mode currently requires `UcmPipelineStore` |
| `store_pipeline` | No | Commonly `Cache\|Posix`: shared-memory hot tier + disk cold tier |
| `storage_backends` | No | Cold-tier directories; configure several by colon-separating them |
| `cache_buffer_capacity_gb` | No | Hot-tier capacity, default 128; startup fails if `/dev/shm` is insufficient |
| `chunk_size` | Yes | Rows per chunk. Too small inflates block counts and lookups; too large wastes tail padding; the example uses 128 |

Width (`encoder_cache_hidden_dim`), namespace (`cache_namespace`), and the cross-instance sharing flag (`store_unique_id`) are optional and inferred by default; the full field reference is in the [design document](../design-docs/ucm-ec-connector.md).

## Verify the effect

### 1. Quick check: confirm hits from logs

Send the same image request twice (any OpenAI-compatible client works; `/v1/chat/completions` with an `image_url`). After the first request finishes, run the cache-reset commands below before sending the second request to exclude KV prefix caching and local encoder caching. Watch the logs:

- 1st request (miss then save): request finalization logs `EC request cache hits: req=..., mm_items=1, hits_total=0`, together with `EC dump item ...`;
- 2nd request (EC hit): logs `EC request cache hits: req=..., mm_items=1, hits_total=1`, plus per-item `EC load item ...` entries under `UCM_LOG_LEVEL=debug`.

A stable cold-tier file count is supporting evidence; use actual `EC load item ...` logs to confirm UCM reuse.

### 2. Benchmark: run the same seed twice

Run the same command twice, clearing vLLM's local caches between rounds: round 1 populates UCM, and round 2 verifies external EC reuse and compares TTFT. An all-miss baseline requires that the test media is not already stored in UCM.

```bash
vllm bench serve \
  --backend openai-chat --host 127.0.0.1 --port 8000 \
  --endpoint /v1/chat/completions \
  --model qwen3.8 --tokenizer /models/Qwen3.8-27B-w8a8 \
  --dataset-name random-mm \
  --num-prompts 20 --seed 1000 \
  --random-input-len 8 --random-output-len 1 --random-range-ratio 0 \
  --random-mm-base-items-per-request 2 --random-mm-num-mm-items-range-ratio 0.5 \
  --random-mm-limit-mm-per-prompt '{"image":4,"video":0}' \
  --random-mm-bucket-config "{(5120,5120,1):1.0}" \
  --request-rate inf --max-concurrency 10 --num-warmups 0 \
  --ignore-eos
```

After round 1 finishes and no other requests are in flight, run these commands. Start round 2 with the same benchmark command only after every reset succeeds:

```bash
set -euo pipefail

# Clear local KV prefix caching; HTTP 200 can still return success=false.
curl -fsS -X POST http://127.0.0.1:8000/reset_prefix_cache \
  | python -c 'import json, sys; sys.exit(0 if json.load(sys.stdin).get("success") is True else 1)'

# Clear multimodal preprocessing caches.
curl -fsS -X POST http://127.0.0.1:8000/reset_mm_cache

# Clear the Scheduler and Worker's local encoder caches.
curl -fsS -X POST http://127.0.0.1:8000/reset_encoder_cache
```

Set `VLLM_SERVER_DEV_MODE=1` before starting the service to expose these endpoints. `reset_mm_cache` does not clear encoder outputs, so `reset_encoder_cache` is also required. If a reset fails, ensure no requests are in flight and retry before starting round 2.

With a fixed `--seed`, both rounds contain identical images. Outputs successfully saved in round 1 remain in UCM, while resetting vLLM's local caches makes round 2 load those outputs through UCM. Do not restart the service or clear UCM's hot tier or cold-tier directory between rounds.

| Argument | Purpose |
| --- | --- |
| `--dataset-name random-mm` | Synthetic multimodal dataset; generates random images per bucket size |
| `--num-prompts 20 --max-concurrency 10 --request-rate inf` | 20 requests, 10 concurrent, no pacing |
| `--random-input-len 8 --random-output-len 1` | Minimal text and one output token, so TTFT is dominated by image encoding |
| `--random-mm-base-items-per-request 2 --random-mm-num-mm-items-range-ratio 0.5` | Images per request vary around 2, sampled in the 1–3 range |
| `--random-mm-limit-mm-per-prompt '{"image":4,"video":0}'` | Images only, up to 4 per request |
| `--random-mm-bucket-config "{(5120,5120,1):1.0}"` | A single image size 5120×5120 (the trailing 1 means one frame, i.e., an image) |
| `--seed 1000` | Fixed seed so both rounds produce identical datasets |

### 3. Reading the results

| Round | Expected TTFT | Meaning |
| --- | --- | --- |
| Round 1 (seed=1000) | High | All EC misses: encoder recomputation + cache dump |
| Round 2 (same seed, local caches reset) | Expected to decrease | Load outputs successfully saved in round 1 from UCM and skip encoder recomputation; confirm with actual load logs |

Hit evidence:

- Round 1 request finalization logs `EC request cache hits: req=..., mm_items=N, hits_total=0`, with `EC dump item ...` entries under `UCM_LOG_LEVEL=debug`;
- Round 2 should log `EC load item ...`, proving actual UCM loading. Without repeated identifiers or local sharing within this round, expect `hits_total=N`, where N counts distinct media identifiers in the request;
- A stable cold-tier file count is supporting evidence and cannot establish UCM hits by itself.

## Advanced: running alongside the KV Connector

UCM supports enabling a KV connector (e.g., HLA) and the EC connector on the same instance. They use **two independent YAML files**, specified separately via `--kv-transfer-config` and `--ec-transfer-config`. Note the format difference: KV's top level is a `ucm_connectors:` list, while EC's top level is the `ucm_ec_connector:` section.

```bash
vllm serve /home/models/Qwen3.8-27B-w8a8 \
  --served-model-name Qwen3.8-27B \
  --tensor-parallel-size 2 \
  --quantization ascend \
  --max-model-len 16384 --max-num-seqs 32 \
  --host 0.0.0.0 --port 8002 \
  --trust-remote-code --gpu-memory-utilization 0.9 \
  --no-disable-hybrid-kv-cache-manager \
  --enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"UCMConnector","kv_connector_module_path":"ucm.integration.vllm.ucm_connector","kv_role":"kv_both","kv_connector_extra_config":{"UCM_CONFIG_FILE":"/home/yhq/dual-connector-test/ucm_kv_config.yaml"}}' \
  --ec-transfer-config '{"ec_connector":"UCMECConnector","ec_connector_module_path":"ucm.integration.vllm.ucm_ec_connector","ec_role":"ec_both","ec_connector_extra_config":{"UCM_CONFIG_FILE":"/home/yhq/dual-connector-test/ucm_ec_config.yaml"}}'
```

Notes:

- The KV side's `cache_buffer_capacity_gb` is validated against the model's KV pool; the hybrid 27B model measured requires ≥160 (a smaller value fails directly with `too small buffer`);
- Keep the EC cold-tier directory separate from the KV one (e.g., `store/kv` and `store/ec`);
- Success markers: `UCMHybridLinearAttentionConnector initialized` ×3 plus `Initialized UCM EC connector` (one for SCHEDULER, one for WORKER).

## Notes

- `/dev/shm` capacity must be ≥ `cache_buffer_capacity_gb`; it is validated automatically at startup, and startup fails when insufficient;
- Stop the service with `pkill -TERM` (or simply remove the container). On NPU environments SIGKILL can leak driver-level device memory that only a reboot releases;
- In this version a load failure is fail-closed (an engine-level error) with no automatic recomputation; in production, watch the availability of the Store backend. The design trade-offs are in [Section 7 of the design document](../design-docs/ucm-ec-connector.md).
