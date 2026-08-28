## P2P hand-off (UCM + Mooncake)

Distributed PD uses two cache paths for two different purposes:

- UCM persists reusable prefixes on Prefill instances.
- Mooncake transfers the current request's KV cache directly from Prefill to
  Decode.

This is not a faster form of centralized storage hand-off; it is a different
connector topology. The Prefill side uses vLLM `MultiConnector` to run
`MooncakeConnectorV1` and `UCMConnector` together. Decode uses only
`MooncakeConnectorV1` as a consumer.

```mermaid
flowchart LR
  C[Client] --> R[PD router]
  R --> P[Prefill pool]
  P <--> U[UCM prefix store]
  P -->|request KV via P2P| M[Mooncake]
  M --> D[Decode pool]
  R --> D
  D --> C
```

!!! important "Current scope"

    This guide follows the documented vLLM Ascend + Mooncake architecture. It
    is a multi-node qualification recipe, not a generic CUDA/Ascend contract.
    Starting with vLLM Ascend 0.11.0, Mooncake is included in the official
    image; verify the exact image and release compatibility before deployment.

### P2P prerequisites

- Complete [Installation](../installation.md) on every Prefill and Decode node.
- Verify HCCL communication and IP/interface selection between the nodes.
- Reserve the Mooncake master port, one unique transfer port per local engine,
  and one API port per engine.
- Make the model and `mooncake.json` available at identical paths on all nodes.
- Mount the UCM storage backend on every Prefill node. Decode nodes do not need
  this mount unless another feature uses it.

The runnable example below uses one Prefill node and one Decode node, each with
four NPUs:

| Role | Address | Devices | API port | Mooncake port |
| --- | --- | --- | --- | --- |
| Mooncake master / Prefill | `192.168.10.1` | `0,1,2,3` | `9000` | `20001` |
| Decode | `192.168.10.2` | `0,1,2,3` | `9000` | `20001` |
| Router | `192.168.10.1` | — | `7850` | — |

The same Mooncake port is safe here because it is bound on different hosts. If
multiple engines run on one host, give each engine a unique transfer port.

### 1. Start the Mooncake master

Run on `192.168.10.1`:

```bash
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

mooncake_master \
  --port 50088 \
  --eviction_high_watermark_ratio 0.9 \
  --eviction_ratio 0.1 \
  --default_kv_lease_ttl 11000
```

Create `/etc/ucm/mooncake.json` on both nodes:

```json
{
  "metadata_server": "P2PHANDSHAKE",
  "protocol": "ascend",
  "device_name": "",
  "master_server_address": "192.168.10.1:50088",
  "global_segment_size": "1GB"
}
```

### 2. Configure UCM for Prefill

Create `/etc/ucm/ucm-prefill.yaml` on every Prefill node:

```yaml
ucm_connectors:
  - ucm_connector_name: "UcmPipelineStore"
    ucm_connector_config:
      store_pipeline: "Cache|Posix"
      storage_backends: "/mnt/ucm-kv"
      cache_buffer_capacity_gb: 64
      io_direct: false
      store_health:
        enabled: true

enable_event_sync: true
enable_metrics: true
use_layerwise: true
```

For multiple Prefill nodes, `/mnt/ucm-kv` must point to the same NFS/3FS or
distributed storage backend. Its role is prefix reuse in the Prefill pool, not
the live P→D transfer.

### 3. Start Prefill

Run on `192.168.10.1`:

```bash
export ENABLE_UCM_PATCH=1
export PYTHONHASHSEED=0
export MOONCAKE_CONFIG_PATH=/etc/ucm/mooncake.json
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1

vllm serve /models/QwQ-32B \
  --host 0.0.0.0 \
  --port 9000 \
  --served-model-name ucm-model \
  --tensor-parallel-size 4 \
  --max-model-len 17000 \
  --max-num-batched-tokens 8000 \
  --max-num-seqs 4 \
  --block-size 128 \
  --gpu-memory-utilization 0.92 \
  --trust-remote-code \
  --enforce-eager \
  --kv-transfer-config \
  '{
    "kv_connector":"MultiConnector",
    "kv_role":"kv_producer",
    "kv_connector_extra_config":{
      "connectors":[
        {
          "kv_connector":"MooncakeConnectorV1",
          "kv_role":"kv_producer",
          "kv_port":20001,
          "kv_connector_extra_config":{
            "prefill":{"dp_size":1,"tp_size":4},
            "decode":{"dp_size":1,"tp_size":4}
          }
        },
        {
          "kv_connector":"UCMConnector",
          "kv_role":"kv_both",
          "kv_connector_module_path":"ucm.integration.vllm.ucm_connector",
          "kv_connector_extra_config":{
            "UCM_CONFIG_FILE":"/etc/ucm/ucm-prefill.yaml"
          }
        }
      ]
    }
  }'
```

The outer producer role describes the P2P hand-off. The nested UCM connector
remains `kv_both` because it performs prefix lookup and persistence within the
Prefill engine.

### 4. Start Decode

Run on `192.168.10.2`:

```bash
export PYTHONHASHSEED=0
export MOONCAKE_CONFIG_PATH=/etc/ucm/mooncake.json
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1

vllm serve /models/QwQ-32B \
  --host 0.0.0.0 \
  --port 9000 \
  --served-model-name ucm-model \
  --tensor-parallel-size 4 \
  --max-model-len 17000 \
  --max-num-batched-tokens 8000 \
  --max-num-seqs 4 \
  --block-size 128 \
  --gpu-memory-utilization 0.92 \
  --trust-remote-code \
  --kv-transfer-config \
  '{
    "kv_connector":"MooncakeConnectorV1",
    "kv_role":"kv_consumer",
    "kv_port":20001,
    "kv_connector_extra_config":{
      "prefill":{"dp_size":1,"tp_size":4},
      "decode":{"dp_size":1,"tp_size":4}
    }
  }'
```

Do not add `UCMConnector` to Decode merely to mirror Prefill. In this
architecture, Decode consumes the live request cache from Mooncake.

### 5. Start the router

Run on a host that can reach both API endpoints:

```bash
python /vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
  --host 0.0.0.0 \
  --port 7850 \
  --prefiller-hosts 192.168.10.1 \
  --prefiller-ports 9000 \
  --decoder-hosts 192.168.10.2 \
  --decoder-ports 9000
```

For XpYd, list every engine endpoint. The host count and port count must match
inside each pool, and every local engine requires a unique Mooncake port.

### 6. Verify and benchmark

First query each engine's `/health` endpoint directly. Then send traffic to the
router:

```bash
curl http://192.168.10.1:7850/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"ucm-model",
    "prompt":"Explain how a direct KV transfer differs from shared storage.",
    "max_tokens":64,
    "temperature":0
  }'
```

Benchmark the routed endpoint:

```bash
vllm bench serve \
  --backend vllm \
  --model ucm-model \
  --host 192.168.10.1 \
  --port 7850 \
  --dataset-name random \
  --seed 123456 \
  --num-prompts 10 \
  --random-input-len 8000 \
  --random-output-len 1000 \
  --request-rate 1 \
  --ignore-eos
```

Verify three signals independently:

1. Prefill reports UCM prefix hits on repeated prefixes.
2. Mooncake reports successful transfer for each P→D hand-off.
3. Decode begins from transferred KV rather than recomputing the full prompt.

### Scale-out notes

- Dense models can scale as independent DP engines behind the router.
- MoE deployments commonly combine external DP and EP. Each DP rank needs its
  own API/transfer ports and correct global rank configuration.
- Do not copy a DP1TP4 connector topology into DP4TP4 or asymmetric P/D
  parallelism without updating the nested `prefill` and `decode` dimensions.
- For DP/EP shell templates and an eight-node example, use the
  [large-scale EP guide](https://ucm.readthedocs.io/en/latest/user-guide/pd-disaggregation/large_scale_ep.html).

### P2P common failures

| Symptom | Check |
| --- | --- |
| Prefill starts but `MultiConnector` fails | Confirm the vLLM/vLLM Ascend release includes `MultiConnector`, Mooncake, and the UCM registration patch. |
| Mooncake cannot register a segment | Check master reachability, `MOONCAKE_CONFIG_PATH`, transfer-port uniqueness, and HCCL/NIC selection. |
| Decode recomputes the prompt | Verify the router preserves request identity and that P/D DP/TP dimensions match the connector JSON. |
| Repeated requests never hit UCM | Check the shared Prefill backend, deterministic hash inputs, model/cache compatibility, and Prefill logs. |
| P2P works at 1P1D but fails after scale-out | Check per-engine API/Mooncake port allocation and global DP rank/address configuration. |
