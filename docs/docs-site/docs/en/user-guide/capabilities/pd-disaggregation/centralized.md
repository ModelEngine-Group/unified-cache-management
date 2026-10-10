# Shared-store PD

This guide covers the shared-storage handoff path in [PD Disaggregation](index.md), starting with one Prefill and one Decode instance.

In a shared-store setup, prefill and decode run UCM against the same external
cache. The request first visits prefill to compute and save prompt blocks, then
visits decode with the original prompt. Decode discovers reusable blocks by
lookup; it does not receive a peer address from UCM.

The repository's `ucm/pd/toy_proxy_server.py` demonstrates this sequencing.
The bundled Kubernetes PD profiles use a different path, described in
[Transport with UCM](distributed.md).

## Before starting

Use the [vLLM quickstart](../../quick_start/index.md#vllm) to validate each
engine independently. Select the engine image or package from
[Installation](../../quick_start/index.md). This guide provides shared-store configuration, both engine commands, proxy startup and request verification.

Both instances must have:

- The same weights, tokenizer, model-directory name, dtype, block size, and
  compatible KV layout. Start with identical tensor-parallel settings.
- Access to the same stored objects. Equal path strings on two local disks do
  not provide shared storage.
- Sufficient store permissions, capacity, and read-after-write visibility for
  decode to observe completed prefill writes.
- Distinct engine HTTP ports and explicit device allocation. A running process
  on the wrong device is not a second independent instance.

## 1p1d

This is the original Qwen2.5-7B-Instruct example: TP1, a 20,000-token context and a 32 GiB host cache. Start Prefill and Decode on their respective hosts, using device 0 on each. On one host, allocate different devices instead. Install the matching engine and UCM first; the commands do not establish compatibility with newer engines.

Save this YAML as `/etc/ucm/pd.yaml` in both environments. `/mnt/ucm-shared` must refer to the same shared filesystem on both hosts.

```yaml
ucm_connectors:
  - ucm_connector_name: "UcmPipelineStore"
    ucm_connector_config:
      store_pipeline: "Cache|Posix"
      storage_backends: "/mnt/ucm-shared"
      cache_buffer_capacity_gb: 32
enable_event_sync: true
use_layerwise: false
```

### Start Prefill

```bash
export CUDA_VISIBLE_DEVICES=0
export ENABLE_UCM_PATCH=1
vllm serve /home/models/Qwen2.5-7B-Instruct \
--max-model-len 20000 \
--tensor-parallel-size 1 \
--gpu_memory_utilization 0.87 \
--trust-remote-code \
--port 7800 \
--block-size 128 \
--kv-transfer-config \
'{
    "kv_connector": "UCMConnector",
    "kv_role": "kv_both",
    "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
    "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/etc/ucm/pd.yaml"}
}'
```

### Start Decode

```bash
export CUDA_VISIBLE_DEVICES=0
export ENABLE_UCM_PATCH=1
vllm serve /home/models/Qwen2.5-7B-Instruct \
--max-model-len 20000 \
--tensor-parallel-size 1 \
--gpu_memory_utilization 0.87 \
--trust-remote-code \
--port 7801 \
--block-size 128 \
--kv-transfer-config \
'{
    "kv_connector": "UCMConnector",
    "kv_role": "kv_both",
    "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
    "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/etc/ucm/pd.yaml"}
}'
```

### Start the proxy and call the service

Run from the UCM checkout root. Replace the quoted host placeholders with reachable engine IPs; keep the proxy running and send the request from another terminal.

```bash
python3 ucm/pd/toy_proxy_server.py \
+  --pd-disaggregation --host localhost --port 7802 \
+  --prefiller-hosts "<prefill-node-ip>" --prefiller-ports 7800 \
+  --decoder-hosts "<decode-node-ip>" --decoder-ports 7801
```

```bash
curl http://localhost:7802/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/home/models/Qwen2.5-7B-Instruct",
        "prompt": "What date is today?",
        "max_tokens": 20,
        "temperature": 0
    }'
```

### Verify the handoff

1. Use a dedicated cache directory and a prompt long enough to span several
   configured blocks and meet any persistence threshold.
2. Send one deterministic request through the proxy and inspect both engine
   logs. Confirm that prefill computes and submits cache writes.
3. Check decode's UCM hit tokens and successful loads. Compare its output with
   an independently validated full-compute request using the same settings.
4. Repeat with changed prompt content to confirm that unrelated prefixes do
   not appear as full external hits.
5. Restart both engines while retaining the cache, then repeat the original
   request to separate external reuse from engine-memory hits.

A prefill HTTP response is the proxy's sequencing point; the proxy does not
query a storage durability receipt. UCM dump work has its own completion
lifecycle. If decode starts before blocks are visible, the request may still
succeed by computing a missing prefix. Measure write completion and decode
hits before claiming that this path transfers all prompt KV through storage.
`enable_event_sync` coordinates device events; it is not a cross-server
commit protocol.


## 1p1d with Different Platforms

The original heterogeneous example uses Ascend Prefill and CUDA Decode with `--dtype bfloat16` on both. Reuse the same UCM YAML and model directory name. Matching dtype alone does not establish compatible KV layout: verify external hits and output on this exact engine/device pair.

### Ascend Prefill

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
export ENABLE_UCM_PATCH=1
vllm serve /home/models/Qwen2.5-7B-Instruct \
--max-model-len 20000 \
--tensor-parallel-size 1 \
--gpu_memory_utilization 0.87 \
--trust-remote-code \
--port 7800 \
--block-size 128 \
--dtype bfloat16 \
--kv-transfer-config \
'{
    "kv_connector": "UCMConnector",
    "kv_role": "kv_both",
    "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
    "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/etc/ucm/pd.yaml"}
}'
```

### CUDA Decode

```bash
export CUDA_VISIBLE_DEVICES=0
export ENABLE_UCM_PATCH=1
vllm serve /home/models/Qwen2.5-7B-Instruct \
--max-model-len 20000 \
--tensor-parallel-size 1 \
--gpu_memory_utilization 0.87 \
--trust-remote-code \
--port 7801 \
--block-size 128 \
--dtype bfloat16 \
--kv-transfer-config \
'{
    "kv_connector": "UCMConnector",
    "kv_role": "kv_both",
    "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
    "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/etc/ucm/pd.yaml"}
}'
```

### Proxy and request

```bash
python3 ucm/pd/toy_proxy_server.py \
+  --pd-disaggregation --host localhost --port 7802 \
+  --prefiller-hosts "<prefill-node-ip>" --prefiller-ports 7800 \
+  --decoder-hosts "<decode-node-ip>" --decoder-ports 7801
```

```bash
curl http://localhost:7802/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/home/models/Qwen2.5-7B-Instruct",
        "prompt": "What date is today?",
        "max_tokens": 20,
        "temperature": 0
    }'
```

## XpYd

The original two-Prefill/two-Decode example allocates GPUs 0–3 on one host. Run each block in a separate terminal, using the same model and shared YAML. For multiple hosts, adjust devices locally and supply each host separately to the proxy.

### Prefill 0

```bash
export CUDA_VISIBLE_DEVICES=0
export ENABLE_UCM_PATCH=1
vllm serve /home/models/Qwen2.5-7B-Instruct \
--max-model-len 20000 \
--tensor-parallel-size 1 \
--gpu_memory_utilization 0.87 \
--trust-remote-code \
--port 7800 \
--block-size 128 \
--kv-transfer-config \
'{
    "kv_connector": "UCMConnector",
    "kv_role": "kv_both",
    "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
    "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/etc/ucm/pd.yaml"}
}'
```

### Prefill 1

```bash
export CUDA_VISIBLE_DEVICES=1
export ENABLE_UCM_PATCH=1
vllm serve /home/models/Qwen2.5-7B-Instruct \
--max-model-len 20000 \
--tensor-parallel-size 1 \
--gpu_memory_utilization 0.87 \
--trust-remote-code \
--port 7801 \
--block-size 128 \
--kv-transfer-config \
'{
    "kv_connector": "UCMConnector",
    "kv_role": "kv_both",
    "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
    "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/etc/ucm/pd.yaml"}
}'
```

### Decode 0

```bash
export PYTHONHASHSEED=123456
export CUDA_VISIBLE_DEVICES=2
export ENABLE_UCM_PATCH=1
vllm serve /home/models/Qwen2.5-7B-Instruct \
--max-model-len 20000 \
--tensor-parallel-size 1 \
--gpu_memory_utilization 0.87 \
--trust-remote-code \
--enforce-eager \
--port 7802 \
--block-size 128 \
--kv-transfer-config \
'{
    "kv_connector": "UCMConnector",
    "kv_role": "kv_both",
    "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
    "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/etc/ucm/pd.yaml"}
}'
```

### Decode 1

```bash
export PYTHONHASHSEED=123456
export CUDA_VISIBLE_DEVICES=3
export ENABLE_UCM_PATCH=1
vllm serve /home/models/Qwen2.5-7B-Instruct \
--max-model-len 20000 \
--tensor-parallel-size 1 \
--gpu_memory_utilization 0.87 \
--trust-remote-code \
--enforce-eager \
--port 7803 \
--block-size 128 \
--kv-transfer-config \
'{
    "kv_connector": "UCMConnector",
    "kv_role": "kv_both",
    "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
    "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/etc/ucm/pd.yaml"}
}'
```

### Proxy and request

```bash
python3 ucm/pd/toy_proxy_server.py \
+  --pd-disaggregation --host localhost --port 7805 \
+  --prefiller-hosts "<prefill-node-ip>" "<prefill-node-ip>" --prefiller-ports 7800 7801 \
+  --decoder-hosts "<decode-node-ip>" "<decode-node-ip>" --decoder-ports 7802 7803
```

```bash
curl http://localhost:7805/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/home/models/Qwen2.5-7B-Instruct",
        "prompt": "What date is today?",
        "max_tokens": 20,
        "temperature": 0
    }'
```

The host and port lists must match in length. The proxy uses round-robin selection, without health-aware scheduling or in-flight recovery. Every selected Decode must be able to read every Prefill instance's cache.

## Diagnose missing reuse

- **No writes:** check persistence thresholds, the selected connector, and
  store errors on prefill.
- **Writes but no decode hits:** check shared mount identity, write visibility,
  model/cache compatibility, and decode's effective UCM configuration.
- **Hits but failed loads:** inspect UCM load errors and backend health before
  comparing latency.
- **Fast response without external hits:** check engine-memory hits and the
  amount of recomputation; latency alone cannot identify the KV path.

Use [Metrics](../../observability/metrics.md) to collect each phase separately.
