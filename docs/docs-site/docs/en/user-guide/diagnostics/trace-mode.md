# UCM Trace Mode

UCM Trace Mode is a lightweight diagnostic and evaluation mode that records per-request traces during inference **without**
performing any actual KV cache dump/load operations. Trace Mode lets you collect real request traffic data and simulate
the theoretical KV cache hit rate UCM could deliver before committing to a full UCM storage rollout.

It is recommended to first collect hit ratio statistics with Trace Mode and confirm with relevant project members whether
to adopt UCM.

## Overview

Enable Trace Mode with `use_lite: true`. UCM chooses the Lite connector from the model's KV cache layout: `UCMLiteConnector` for standard models, `UCMHLALiteConnector` for hybrid linear-attention models, and `UCMFAWALiteConnector` for FAWA models.

| Option | Default | Description |
| :----- | :------ | :---------- |
| `use_lite` | `false` | Switches to the **UCM Lite Connector**, which works with a Fake Store that skips all actual KV dump/load operations. |

This mode records request and block metadata without real KV I/O or avoided inference compute. Analysis estimates reuse opportunities; measure actual storage performance separately.

### Logged Trace Format

At startup the Lite connector writes a `UCMTraceMeta:` line with the model type, block dimensions and cache topology. Keep this line and the vLLM startup logs together with the request traces; the analyzer uses them to select the simulation engine and determine cache capacity.

```text
UCMTraceMeta: type=standard, is_mla=False, vllm_hash_block_size=128, trace_hash_block_size=512, hbm_block_data_size=<bytes>
UCMTrace: timestamp: 1234567.890123, request_id: req-42, input_length: 8192, output_length: 128, block_hashes: ['a1b2...', 'c3d4...', ...]
```

The topology fields differ for `standard`, `mamba` and `fawa` models. The request line records:

| Field | Description |
| :---- | :---------- |
| `timestamp` | Lookup time from `time.perf_counter()`, used to preserve request order. |
| `request_id` | The vLLM request identifier. |
| `input_length` | Input token count (`request.num_tokens`). |
| `output_length` | Maximum output token count (`request.max_tokens`). |
| `block_hashes` | Hex-encoded block hashes at the granularity reported by `trace_hash_block_size`. |

The standard Lite connector records requests once they contain at least one complete trace block. Timing diagnostics may appear on a separate log line.

## Configuration

You can start from the sample file at `unified-cache-management/examples/ucm_config_example.yaml`

Enable Trace Mode with:

```yaml
use_lite: true
```

### Log Configuration (Optional)

A long request can produce a large list of block hashes. Tune the following environment variables before launching the service:

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

For a standard model, the following startup log confirms that the Lite connector is active:

```text
[UC][I] Init UCMLiteConnector.
```

You can now send production-equivalent traffic to the server. Every request will produce a trace record in the UCM log
directory. No KV cache is dumped or loaded.

## Trace Analysis

After collecting traces, run `benchmarks/auto_trace_analysis.py` to estimate KV cache hit rates. The script reads `UCMTraceMeta:` and the vLLM startup information to select the standard, mamba or fawa simulation engine and model HBM → DRAM → FS caching. Block sizes and model topology come from the logs.

```bash
python benchmarks/auto_trace_analysis.py \
  --log-dir /workspace/ucm-trace-logs \
  --dram-pool-size-gb <dram_gb> \
  --fs-pool-size-gb <fs_gb>
```

### Required Arguments

| Argument | Description |
| :------- | :---------- |
| `--log-dir` | Directory scanned recursively for `*.log`, `*.log.*` and `*.log.gz`. Include the `UCMTraceMeta:` line and vLLM startup logs. |
| `--dram-pool-size-gb` | Simulated DRAM pool capacity in GiB. |
| `--fs-pool-size-gb` | Simulated filesystem pool capacity in GiB. |

### Optional Arguments

| Argument | Description |
| :------- | :---------- |
| `--service-url` | vLLM service address. Fetches `/metrics` to compare against the service's actual prefix-cache hit rate. |
| `--num-nodes` | Physical node count; defaults to `1`. |
| `--unified-memory-pool` | Simulate a shared DRAM pool across nodes instead of separate per-node pools. |
| `--output` | Write the analysis summary to a JSON file. |
| `--trace-output` | Write the parsed trace records to a JSON Lines file. |

The current analyzer no longer accepts `--block-kv-cache-size` or `--is-mla`. Logs without `UCMTraceMeta:` must be collected again with the matching Lite connector.

## Analysis Report

The analysis produces a comprehensive report including:

- **Total request count and token count**: Summary of the workload
- **Hit rate scenarios**: Theoretical max, HBM only, HBM+DRAM, and HBM+DRAM+FS hit rates
- **Request lifetime**: How long a request's blocks stay reusable (average, P90, P95)

These metrics help you evaluate whether UCM is suitable for your workload and how much capacity to allocate for each
storage tier.
