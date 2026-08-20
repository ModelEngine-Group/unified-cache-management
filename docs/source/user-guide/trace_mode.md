# UCM Trace Mode User Guide

This document describes how to use **UCM Trace Mode**, a lightweight diagnostic and evaluation mode that records per-request traces during inference **without** performing any actual KV cache dump/load operations. Trace Mode lets you collect real request traffic data and simulate the theoretical KV cache hit rate UCM could deliver before committing to a full UCM storage rollout.

It is recommended to first collect hit ratio statistics with Trace Mode and confirm with relevant project members whether to adopt UCM.

## 1. Overview

Trace Mode is enabled by setting two options to `true` in the UCM configuration file:

| Option | Default | Description |
| :----- | :------ | :---------- |
| `enable_record_traces` | `false` | Logs per-request traces (timestamp, input_length, output_length, hash_ids). Each hash_id takes 32 bytes. |
| `use_lite` | `false` | Switches to the **UCM Lite Connector**, which works with a Fake Store that skips all actual KV dump/load operations. |

When both are enabled, `UCMConnector` internally instantiates a `UCMLiteConnector` instead of the regular storage-backed connectors. The Lite connector:

- Computes the same block hash IDs that a real UCM deployment would use.
- Logs a trace record for every request on its first lookup.
- Returns `0` external hit tokens (there is no real store to look up), so inference correctness is unaffected and no KV data is persisted.
- Implements all KV transfer hooks (`start_load_kv`, `save_kv_layer`, `wait_for_save`, etc.) as no-ops.


### Logged Trace Format

Each request produces two log lines on first lookup:

```text
[UC][I] timestamp: 1234567.890123, request_id: req-42, input_length: 8192, output_length: 128, ucm_block_ids: ['a1b2...', 'c3d4...', ...]
[UC][I] request_id: req-42, hash_time_ms: 0.512, print_time_ms: 0.034
```

| Field | Description |
| :---- | :---------- |
| `timestamp` | `time.perf_counter()` value at lookup time, used to preserve request ordering during analysis. |
| `request_id` | vLLM request identifier (present in Lite connector traces). |
| `input_length` | Number of input tokens in the request (`request.num_tokens`). |
| `output_length` | Maximum output tokens for the request (`request.max_tokens`). |
| `ucm_block_ids` | List of hex-encoded block hash IDs. Each block corresponds to `block_size` tokens; each hash is 32 bytes. |
| `hash_time_ms` | Time spent computing the block hash IDs (Lite connector only). |
| `print_time_ms` | Time spent formatting/logging the trace line (Lite connector only). |

## 2. Configuration

You can start from the sample file at `unified-cache-management/examples/ucm_config_example.yaml`

Trace Mode is enabled by setting two options to `true`: 
```yaml
enable_record_traces: true
use_lite: true
```


### Log Configuration(Optional)

Trace line can be large because each hash_id is 32 bytes and a long request can contain many block IDs. Tune the following environment variables before launching the service:

| Environment Variable | Default | Description |
| :------------------- | :------ | :---------- |
| `UCM_LOG_PATH` | `log` | Directory for per-process log files (e.g. `ucm-<pid>.log`). |
| `UCM_LOG_MAX_FILES` | `10` | Maximum number of rotated log files kept per process. |
| `UCM_LOG_MAX_SIZE` | `5` | Maximum size in **MiB** per log file before rotation. Increase this significantly when recording traces for long requests. |
| `UCM_LOG_LEVEL` | `info` | Log level. Traces are emitted at `INFO` level, so keep this at `info` (or lower for extra debug output). |

Example for a trace-collection run:

```bash
export UCM_LOG_PATH=/workspace/ucm-trace-logs
export UCM_LOG_MAX_SIZE=256      # 256 MiB per file
export UCM_LOG_MAX_FILES=50     # keep up to 50 rotated files per process
export UCM_LOG_LEVEL=info
```

## 3. Launching the Inference Service

Trace Mode is deployed as an OpenAI-compatible vLLM server. Start it the same way as a normal UCM deployment — the only difference is the UCM config file contents.

Take the Qwen/Qwen2.5-14B-Instruct model as an example:

```bash
vllm serve Qwen/Qwen2.5-14B-Instruct \
  --max-model-len 32000 \
  --tensor-parallel-size 2 \
  --gpu_memory_utilization 0.87 \
  --block_size 128 \
  --trust-remote-code \
  --port 7800 \
  --enforce-eager \
  --no-enable-prefix-caching \
  --kv-transfer-config \
  '{
      "kv_connector": "UCMConnector",
      "kv_role": "kv_both",
      "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
      "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/workspace/unified-cache-management/examples/ucm_config_example.yaml"}
  }'
```

**⚠️ Replace the `UCM_CONFIG_FILE` path with the actual path to your trace-mode config file on your machine.**

When the service starts, the following log confirms that the Lite connector is active:

```text
[UC][I] Init UCMLiteConnector.
```

You can now send production-equivalent traffic to the server. Every request will produce a trace record in the UCM log directory. No KV cache is dumped or loaded.

## 4. Trace Analysis

After collecting traces, run `benchmarks/auto_trace_analysis.py` to simulate the theoretical KV cache hit rate. The script parses trace lines **plus** the `available kv cache memory` (or `current kv cache memory`) and `tensor_parallel_size` values that vLLM/UCM emit at startup, then simulates an LRU multi-tier cache (HBM → DRAM → FS) to estimate the hit rate UCM would achieve.

```bash
python benchmarks/auto_trace_analysis.py \
  --service-url <ip:port of vllm service> \
  --log-dir <path to log folder> \
  --block-kv-cache-size <bytes_per_block> \
  --is-mla <true|false> \
  --dram-pool-size-gb <dram_gb> \
  --fs-pool-size-gb <fs_gb> 
```

Required arguments:

| Argument | Description |
| :------- | :---------- |
| `--service-url`  | vLLM `/metrics` endpoint (Prometheus). When set, the tool fetches the service's actual prefix-cache hit rate for comparison. |
| `--log-dir` | Directory containing the UCM log files (scanned recursively for `*.log`, `*.log.*`, `*.log.gz`). It must include vLLM's startup logs (captured to `<UCM_LOG_PATH>/vllm-<pid>.log` by default) so the available KV cache memory and tensor-parallel size can be parsed. |
| `--block-kv-cache-size`  | Size in bytes of a single KV cache block. Use the [KV Cache Size Calculator](../getting-started/kv_cache_calculator.md) to determine this value for your model. |
| `--is-mla` | Whether the model uses Multi-Latent Attention (`true`/`false`). When `true`, the HBM capacity is taken per-rank; otherwise it is multiplied by the tensor-parallel size. |
| `--dram-pool-size-gb` | Simulated DRAM (host memory) pool size in GiB. |
| `--fs-pool-size-gb` | Simulated filesystem (SSD/NFS) pool size in GiB. |

### Configuration Reference for Common Models

`--block-kv-cache-size` is computed from the model architecture. Formulas (per block, **not** divided by tensor-parallel — it is the full block size across all layers/heads):

```text
GQA:          2 × num_hidden_layers × block_size × num_kv_heads × head_dim × dtype_bytes
MLA:          num_hidden_layers × block_size × (kv_lora_rank + qk_rope_head_dim) × dtype_bytes
DSA:          num_hidden_layers × block_size × (kv_lora_rank + qk_rope_head_dim + index_head_dim) × dtype_bytes
Hybrid (V4):  bytesPerToken × block_size
```

`--is-mla` reflects whether the KV is **shared across ranks** (compressed MLA/DSA/V4 KV is rank-shared → `true`; GQA shards KV by head → `false`). It corresponds to UCM's `is_deepseek_mla` / `share_buffer_enable` (`ucm_connector.py:1063`, `ucm_connector.py:1251`): when `true`, the single-rank available KV memory parsed from logs already represents the full cluster budget; otherwise it is multiplied by the tensor-parallel size.

Values below assume **bfloat16** (`dtype_bytes=2`) and **`block_size=128`** (UCM default):

| Model | Attention | `--is-mla` | `--block-kv-cache-size` |
| :--- | :--- | :--- | :--- |
| GLM-4.7 | GQA | `false` | `48234496` |
| GLM-4.7-Flash | MLA | `true` | `6930432` |
| GLM-5 / GLM-5.1 / GLM-5.2 | DSA | `true` | `14057472` |
| MiniMax-M2.7 | GQA | `false` | `32505856` |
| MiniMax-M3 | GQA | `false` | `15728640` |

**DeepSeek V4 (hybrid)** uses `bytesPerToken × block_size`:

| Model | Deployment | `block_size` | bytesPerToken | `--is-mla` | `--block-kv-cache-size` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| DeepSeek-V4-Pro | vllm-ascend | 32 | 27175 | `true` | `869600` |
| DeepSeek-V4-Pro | vllm-ascend | 128 | 27175 | `true` | `3478400` |
| DeepSeek-V4-Pro | vllm | 256 | 28415.4375 | `true` | `7274352` |
| DeepSeek-V4-Flash | vllm-ascend | 32 | 19162.5 | `true` | `613200` |
| DeepSeek-V4-Flash | vllm-ascend | 128 | 19162.5 | `true` | `2452800` |
| DeepSeek-V4-Flash | vllm | 256 | 20058.25 | `true` | `5134912` |

Notes:

- For fp16 keep the same value; for fp8 halve it; for fp32 double it.
- `block_size` must equal the vLLM `--block_size` used when collecting traces, otherwise the capacity derived here will not line up with the recorded hashes.
- Architecture params are sourced from `docs/source/_static/model-configs.js`; DeepSeek V4 `bytesPerToken` from `calculator.js` (`DEEPSEEK_V4_CONFIGS`). Always cross-check with the [KV Cache Size Calculator](../getting-started/kv_cache_calculator.md) for your exact model config.

## 5. Report

`print_summary` prints the following to stdout (the same values are written to the `analysis` section of the `--output` JSON). Example:

```text
Trace cache hit rate analysis
  Total request count: 1200
  Total request token count: 19660800
  Average tokens per request: 16384.00
  Total HBM available KV cache size: 12.50 GiB
  TP size: 2
  DRAM pool size: 64.00 GiB
  FS pool size: 1024.00 GiB
  Theoretical max KV cache hit rate: 78.340000%
  HBM theoretical hit rate: 21.560000%
  HBM + DRAM pool theoretical hit rate: 45.210000%
  HBM + DRAM pool + FS pool theoretical hit rate: 72.890000%
  Request lifetime sample count: 980
  Average request lifetime: 142.350000 s
  P90 request lifetime: 318.700000 s
  P95 request lifetime: 405.120000 s
```

**Workload & capacity echo** — confirms what was parsed from the logs and derived from your arguments:

| Metric | Meaning |
| :--- | :--- |
| `Total request count` | Number of trace records parsed (= request count). |
| `Total request token count` | Sum of `input_length` across all requests. |
| `Average tokens per request` | Mean input tokens per request. |
| `Total HBM available KV cache size` | HBM KV budget in GiB. Parsed from vLLM's `Current/Available KV cache memory`; for non-MLA it is already multiplied by `TP size`, for MLA it is the shared per-rank value. |
| `TP size` | Tensor-parallel size resolved from the logs. |
| `DRAM pool size` / `FS pool size` | The `--dram-pool-size-gb` / `--fs-pool-size-gb` you passed in. |

**Hit-rate scenarios** — four LRU simulations with different tier capacities; each rate is `hit_tokens / total_tokens`:

| Metric | Tier capacity used | What it tells you |
| :--- | :--- | :--- |
| `Theoretical max KV cache hit rate` | every tier = `unique_block_count` (effectively unlimited) | Upper bound — the most UCM could ever reach for this traffic. |
| `HBM theoretical hit rate` | HBM only (DRAM=FS=0) | What you get with no external pool — pure on-device KV. |
| `HBM + DRAM pool theoretical hit rate` | HBM + DRAM (FS=0) | Marginal gain from adding a host-memory pool. |
| `HBM + DRAM pool + FS pool theoretical hit rate` | all three tiers | Closest to a real UCM deployment's expected hit rate. |

Read them as a monotonic ladder: `HBM` ≤ `HBM+DRAM` ≤ `HBM+DRAM+FS` ≤ `Theoretical max`. Key points:

- **`Theoretical max`** is the ceiling for this traffic. If it is already low, the workload has little prefix reuse and UCM will not deliver much uplift regardless of pool size — in that case adopting UCM may not be worthwhile.
- **`HBM`** is the no-external-pool baseline. Note the simulated value assumes a single in-flight request; under real **concurrency** multiple requests compete for the HBM KV budget and evict each other, so the actual on-device hit rate will be **lower** than this value.
- **`HBM + DRAM pool`** is what a DRAM pool of the specified size can reach. Compared against the live `Service actual KV cache hit rate` (only shown when `--service-url` is configured, fetched from vLLM's `/metrics`), the difference is the uplift the DRAM pool brings.
- **`HBM + DRAM pool + FS pool`** vs **`HBM + DRAM pool`**: the delta is the additional hit rate contributed by the filesystem (SSD/NFS) tier.
- If the three-tier value is already near the theoretical max, enlarging DRAM/FS yields little; if there is a big gap, bigger pools still help.

**Request lifetime** — how long a request's blocks stay reusable (time from first appearance to the last hit on any of its blocks):

| Metric | Meaning |
| :--- | :--- |
| `Request lifetime sample count` | Number of request groups that had at least one block reused. |
| `Average request lifetime` | Mean reuse lifetime. |
| `P90 request lifetime` | 90% of reused blocks are hit again within this window. |
| `P95 request lifetime` | 95% of reused blocks are hit again within this window. |

**`Average` / `P90` / `P95 request lifetime`** is the request's actual alive time — the span from when the request first appears (chat start) to the last time any of its blocks was reused. It measures how long a conversation's KV stays useful. Use it to size retention: if `P95 = 405 s`, blocks must stay in cache ~7 min to capture 95% of reuse — compare against your DRAM/FS capacity and eviction to judge whether the pool is large enough and retention is long enough.

