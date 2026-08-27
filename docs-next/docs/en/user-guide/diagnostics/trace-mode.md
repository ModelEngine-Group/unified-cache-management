# UCM Trace Mode

UCM Trace Mode is a lightweight diagnostic and evaluation mode that records per-request traces during inference **without**
performing any actual KV cache dump/load operations. Trace Mode lets you collect real request traffic data and simulate
the theoretical KV cache hit rate UCM could deliver before committing to a full UCM storage rollout.

It is recommended to first collect hit ratio statistics with Trace Mode and confirm with relevant project members whether
to adopt UCM.

## Overview

Trace Mode is enabled by setting two options to `true` in the UCM configuration file:

| Option | Default | Description |
| :----- | :------ | :---------- |
| `enable_record_traces` | `false` | Logs per-request traces (timestamp, input_length, output_length, hash_ids). Each hash_id takes 32 bytes. |
| `use_lite` | `false` | Switches to the **UCM Lite Connector**, which works with a Fake Store that skips all actual KV dump/load operations. |

When both are enabled, `UCMConnector` internally instantiates a `UCMLiteConnector` instead of the regular storage-backed
connectors. The Lite connector:

- Computes the same block hash IDs that a real UCM deployment would use.
- Logs a trace record for every request on its first lookup.
- Returns `0` external hit tokens (there is no real store to look up), so inference correctness is unaffected and no KV
  data is persisted.
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

## Configuration

You can start from the sample file at `unified-cache-management/examples/ucm_config_example.yaml`

Trace Mode is enabled by setting two options to `true`:

```yaml
enable_record_traces: true
use_lite: true
```

### Log Configuration (Optional)

Trace line can be large because each hash_id is 32 bytes and a long request can contain many block IDs. Tune the following
environment variables before launching the service:

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

## Launching the Inference Service

Trace Mode is deployed as an OpenAI-compatible vLLM server. Start it the same way as a normal UCM deployment — the only
difference is the UCM config file contents.

Take the Qwen/Qwen2.5-14B-Instruct model as an example:

```bash
export ENABLE_UCM_PATCH=1
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

**Replace the `UCM_CONFIG_FILE` path with the actual path to your trace-mode config file on your machine.**

When the service starts, the following log confirms that the Lite connector is active:

```text
[UC][I] Init UCMLiteConnector.
```

You can now send production-equivalent traffic to the server. Every request will produce a trace record in the UCM log
directory. No KV cache is dumped or loaded.

## Trace Analysis

After collecting traces, run `benchmarks/auto_trace_analysis.py` to simulate the theoretical KV cache hit rate. The script
parses trace lines **plus** the `available kv cache memory` (or `current kv cache memory`) and `tensor_parallel_size`
values that vLLM/UCM emit at startup, then simulates an LRU multi-tier cache (HBM → DRAM → FS) to estimate the hit rate
UCM would achieve.

```bash
python benchmarks/auto_trace_analysis.py \
  --service-url <ip:port of vllm service> \
  --log-dir <path to log folder> \
  --block-kv-cache-size <bytes_per_block> \
  --is-mla <true|false> \
  --dram-pool-size-gb <dram_gb> \
  --fs-pool-size-gb <fs_gb>
```

### Required Arguments

| Argument | Description |
| :------- | :---------- |
| `--service-url` | vLLM `/metrics` endpoint (Prometheus). When set, the tool fetches the service's actual prefix-cache hit rate for comparison. |
| `--log-dir` | Directory containing the UCM log files (scanned recursively for `*.log`, `*.log.*`, `*.log.gz`). It must include vLLM's startup logs so the available KV cache memory and tensor-parallel size can be parsed. |
| `--block-kv-cache-size` | Size in bytes of a single KV cache block. Use the KV Cache Size Calculator to determine this value for your model. |
| `--is-mla` | Whether the model uses Multi-Latent Attention (`true`/`false`). |
| `--dram-pool-size-gb` | Simulated DRAM (host memory) pool size in GiB. |
| `--fs-pool-size-gb` | Simulated filesystem (SSD/NFS) pool size in GiB. |

## Analysis Report

The analysis produces a comprehensive report including:

- **Total request count and token count**: Summary of the workload
- **Hit rate scenarios**: Theoretical max, HBM only, HBM+DRAM, and HBM+DRAM+FS hit rates
- **Request lifetime**: How long a request's blocks stay reusable (average, P90, P95)

These metrics help you evaluate whether UCM is suitable for your workload and how much capacity to allocate for each
storage tier.

For complete documentation, see the detailed Trace Mode guide in the original UCM documentation.