# Profiling

UCM provides a file-based operation logging feature that records detailed operation data (load and dump operations) to log files. This feature is useful for offline analysis, debugging, and auditing.


## Quick Start Guide

### 1）: Configure Record Config

Configure the `record_config` parameters in `/vllm-workspace/unified-cache-management/examples/ucm_config_example.yaml`. Ensure the following parameters are set:

```yaml
record_config:
  enable: false
  log_path: "/workspace/ucm_ops.log"
  flush_size: 10
  flush_interval: 5.0
```

### 2）: Start UCM Service

Start the UCM service using the following command:

```bash
export MODEL_PATH=/home/models/Qwen2.5-14B-Instruct
vllm serve ${MODEL_PATH} \
--served-model-name vllm_cpu_offload \
--max-model-len 20000 \
--tensor-parallel-size 2 \
--gpu_memory_utilization 0.87 \
--trust-remote-code \
--port 7800 \
--kv-transfer-config \
'{
    "kv_connector": "UnifiedCacheConnectorV1",
    "kv_connector_module_path": "ucm.integration.vllm.uc_connector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "UCM_CONFIG_FILE": "/workspace/unified-cache-management/examples/ucm_config_example.yaml"
    }
}'
```

You can view the operation logs in the log file specified by `log_path: "/workspace/ucm_ops.log"`.

### 3）: Replay Data from Logs

Pass the `log_path` (e.g., `/workspace/ucm_ops.log`) to the `test_trace.py` script in `ucm/profiling/` directory, then run the script to replay data from the logs:

```bash
python3 ucm/profiling/test_trace.py
```

The script will read the log data from the specified `log_path` and replay the operations recorded in the log file.

## Configuration Parameters

The operation logging feature is configured in the `record_config` section of the UCM configuration file (`ucm_config_example.yaml`). The following parameters control how operation logs are recorded:

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `enable` | `false` | Enable/disable operation logging. 
| `log_path` | `"/workspace/ucm_ops.log"` | Full path to the log file where operation data will be written. 
| `flush_size` | `10` | Number of log entries to buffer in memory before writing to disk. 
| `flush_interval` | `5.0` | Time interval (in seconds) to force flush buffered logs to disk. 



---

