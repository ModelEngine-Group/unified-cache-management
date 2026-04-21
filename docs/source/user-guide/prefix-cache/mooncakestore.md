# Mooncake Store

This document describes how to use `UcmMooncakeStoreV1` as the storage backend for UCM Prefix Cache in Ascend environments.

## Overview

`UcmMooncakeStoreV1` is a Mooncake-based storage backend provided by UCM for Prefix Cache scenarios. It is designed for Ascend platforms and integrated into the vLLM inference workflow through `UCMConnector`. It is responsible for prefix cache lookup, loading, and dumping, so that Prefix Cache is no longer limited to local memory within a single process or a single instance.

By integrating Mooncake, UCM extends its original local caching capability with both DRAM pooling and remote storage support. As a result, in Prefix Cache scenarios, a tiered cache hierarchy can be formed:

- Local DRAM on the serving node acts as the high-speed near-end cache.
- The DRAM pool provided by Mooncake serves as a shareable intermediate cache layer.
- Remote storage connected through UCM serves as a larger-capacity persistence layer.

This three-tier design provides a better balance among capacity, shareability, and access cost, allowing Prefix Cache to be reused across a broader scope and improving overall cache-hit benefits in long-prefix scenarios.

This document focuses on the capability boundaries, configuration, and basic usage flow of `UcmMooncakeStoreV1` in vLLM.

## Features

The current `UcmMooncakeStoreV1` implementation supports:

- `lookup` / `lookup_on_prefix`: probing prefix hits by block hash
- `load_data`: loading KV blocks from Mooncake into model KV buffers
- `dump_data`: dumping KV blocks from model KV buffers into Mooncake
- `wait` / `check`: handling asynchronous task completion
- Register NPU buffers for RDMA transfer

## Prerequisites

`UcmMooncakeStoreV1` is intended for Ascend-based deployments and requires:

- Linux
- Ascend/NPU runtime with `torch.npu` available
- vLLM + vLLM-Ascend + UCM integration environment
- Mooncake runtime environment

For deployment, it is recommended to use the pre-built vLLM-Ascend Docker image directly. The `vllm-ascend 0.17.0 image` already includes the Mooncake runtime dependencies required by this guide.

If Mooncake needs to be installed manually, refer to the [official Ascend Store / KV Pool guide](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/kv_pool.html) and follow its Mooncake installation instructions.

## Configuration for Prefix Caching

Edit or copy:

`unified-cache-management/examples/ucm_config_example.yaml`

### Minimal Configuration Example

```yaml
ucm_connectors:
  - ucm_connector_name: "UcmMooncakeStoreV1"
    ucm_connector_config:
      protocol: "ascend"
      local_hostname: "127.0.0.1"
      metadata_server: "P2PHANDSHAKE"
      master_server_address: "127.0.0.1:50088"
      device_name: ""
      global_segment_size: "5GB"
      local_buffer_size: "5GB"
      executor_workers: 4
```

### Required Parameters

- `ucm_connector_name`
  - Must be set to `UcmMooncakeStoreV1`.
- `protocol`
  - Must be set to `ascend`.
- `metadata_server`
  - Specifies the Mooncake metadata discovery mode or endpoint. In the common Ascend deployment path, use `P2PHANDSHAKE`.
- `master_server_address`
  - Specifies the address of the Mooncake master service, for example `127.0.0.1:50088`.

### Common Optional Parameters

- `local_hostname` (default: `127.0.0.1`)
  - Local host address passed into Mooncake setup.
- `device_name` (default: empty)
  - Optional device identifier passed to Mooncake.
- `global_segment_size` (default: `5GB`)
  - Size of the global Mooncake segment. This represents the registered memory size per card.
- `local_buffer_size` (default: `5GB`)
  - Size of the local buffer used by the connector.
- `executor_workers` (default: `4`)
  - Number of worker threads used for asynchronous load and dump execution.

## Run Mooncake Master

Before launching vLLM, start the Mooncake master service:

```bash
mooncake_master \
  --port 50088 \
  --eviction_high_watermark_ratio 0.9 \
  --eviction_ratio 0.1 \
  --default_kv_lease_ttl 11000
```

Parameter description:

- `eviction_high_watermark_ratio`
  - Controls the watermark at which eviction is triggered.
- `eviction_ratio`
  - Controls the fraction of objects to evict once eviction starts.
- `default_kv_lease_ttl`
  - Controls the default KV lease TTL. It should be configured larger than both `ASCEND_CONNECT_TIMEOUT` and `ASCEND_TRANSFER_TIMEOUT`.

## Launching Inference

Use `vllm serve` with `UCMConnector`, and pass the Mooncake-backed UCM configuration file through `UCM_CONFIG_FILE`.

### Recommended Launch Command

```bash
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH
export PYTHONHASHSEED=0
export HCCL_INTRA_ROCE_ENABLE=1
export HCCL_RDMA_TIMEOUT=17
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000

vllm serve <your-model> \
  --host 0.0.0.0 \
  --port 8100 \
  --trust-remote-code \
  --enforce-eager \
  --no-enable-prefix-caching \
  --tensor-parallel-size 1 \
  --data-parallel-size 1 \
  --max-model-len 32768 \
  --block-size 128 \
  --max-num-batched-tokens 16384 \
  --kv-transfer-config \
  '{
    "kv_connector": "UCMConnector",
    "kv_role": "kv_both",
    "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
    "kv_connector_extra_config": {
      "UCM_CONFIG_FILE": "/path/to/unified-cache-management/examples/ucm_config_example.yaml"
    }
  }'
```