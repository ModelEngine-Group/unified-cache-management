# PD Disaggregated Deployment

PD disaggregation separates Prefill and Decode into independent serving pools.
After choosing this topology, select how the request KV cache moves from P to
D:

| Hand-off | Connector layout | Recommended starting point |
| --- | --- | --- |
| [Centralized](#centralized-hand-off-ucm-shared-storage) | UCM on both Prefill and Decode | First deployment, heterogeneous P/D, or storage-backed decoupling |
| [P2P](#p2p-hand-off-ucm-mooncake) | UCM on Prefill plus Mooncake producer/consumer | Ascend scale-out after storage hand-off becomes a measured bottleneck |

Single-node and multi-node describe deployment scale; centralized and P2P
describe the KV hand-off. They are independent concepts.

## Centralized hand-off (UCM shared storage)

PD disaggregation assigns Prefill and Decode to different vLLM instances. With
UCM's centralized path, the Prefill instance writes reusable KV blocks to UCM
and the Decode instance loads those blocks before generating the response.

The bundled proxy implements the request sequence used in these examples:

1. select a Prefill instance and send the request with `max_tokens: 1`;
2. wait for Prefill to finish and commit its reusable KV blocks;
3. select a Decode instance and forward the original request as a stream.

Because the transfer happens through UCM storage, both sides use
`UCMConnector` with `kv_role: kv_both`. The logical Prefill/Decode roles are
assigned by the router, not by different UCM connector roles.

Choose this architecture when operational simplicity, heterogeneous P/D pools,
or storage-backed decoupling matters more than minimizing the hand-off latency.
For direct P2P transfer, continue to [P2P hand-off](#p2p-hand-off-ucm-mooncake)
on this page; its connector roles are intentionally different.

## Before you start

- Install a compatible UCM, vLLM, and vLLM Ascend release. See
  [Installation](../installation.md).
- Reserve separate accelerators for Prefill and Decode.
- Keep the model, tokenizer, dtype, block size, cache layout, UCM release, and
  `PYTHONHASHSEED` compatible on both sides.
- Ensure the proxy can reach every vLLM service port.
- For multi-node deployment, mount or configure one storage backend that every
  Prefill and Decode process can access.
- For a heterogeneous P/D pair, set the same explicit `--dtype` on both sides
  and validate cache correctness before benchmarking.

!!! important

    The commands below demonstrate **centralized, storage-backed PD**. Native
    Mooncake connectors use producer/consumer roles, `kv_port`, and direct P2P
    transfer. Do not combine those role settings with this UCM workflow unless
    you are intentionally deploying the separately documented `MultiConnector`
    P2P architecture.

## Prepare the UCM configuration

Create `/etc/ucm/ucm-pd-disaggregated.yaml` on every participating node:

```yaml
ucm_connectors:
  - ucm_connector_name: "UcmPipelineStore"
    ucm_connector_config:
      store_pipeline: "Cache|Posix"
      storage_backends: "/mnt/ucm-kv"
      cache_buffer_capacity_gb: 32
      io_direct: false
      store_health:
        enabled: true

enable_event_sync: true
enable_metrics: true
use_layerwise: false
```

For a single node, `/mnt/ucm-kv` may be local storage. Across nodes, the path
must be backed by the same NFS/3FS export or replaced with another shared UCM
pipeline. Merely creating the same directory name on two local disks does not
share KV cache.

`use_layerwise: false` keeps this first deployment easy to validate by
committing complete cache blocks before Decode starts. Enable layer-wise or
direct P2P transfer only with a topology and release that explicitly supports
it. `cache_buffer_capacity_gb` applies to each process; a 1P1D deployment using
the example value can reserve up to 64 GB of host DRAM for the two CacheStore
instances.

| Parameter | Deployment effect |
| --- | --- |
| `store_pipeline` | `Cache|Posix` places host DRAM in front of the POSIX backend. |
| `storage_backends` | Must identify storage visible to both P and D; identical local path strings on two hosts are not shared storage. |
| `cache_buffer_capacity_gb` | Reserved per serving process, not once per node or cluster. |
| `enable_event_sync` | Synchronizes accelerator work before cache data is consumed by the store. |
| `use_layerwise` | `false` is the conservative first-run profile used here; change only after release-specific validation. |

## Start order

1. Create and cross-check the storage backend.
2. Start every Prefill and Decode service.
3. Query each backend's `/health` endpoint directly.
4. Start the router and inspect `/healthcheck`.
5. Send a functional request before running a benchmark.

## Deploy

=== "Single node"

    The Prefill and Decode services run on different accelerators in the same
    host and share a local UCM backend.

    ```mermaid
    flowchart LR
      C[Client] --> R[PD router]
      R -->|1. prefill| P[NPU 0: Prefill]
      P -->|KV blocks| S[Local UCM backend]
      R -->|2. decode| D[NPU 1: Decode]
      S -->|KV blocks| D
      D -->|stream| C
    ```

    Create the storage path:

    ```bash
    mkdir -p /mnt/ucm-kv
    ```

    The commands select Ascend devices. On CUDA, replace
    `ASCEND_RT_VISIBLE_DEVICES` with `CUDA_VISIBLE_DEVICES`.

    Start the Prefill service:

    ```bash
    export PYTHONHASHSEED=123456
    export ASCEND_RT_VISIBLE_DEVICES=0

    vllm serve /models/Qwen2.5-7B-Instruct \
      --served-model-name ucm-model \
      --host 0.0.0.0 \
      --port 8100 \
      --tensor-parallel-size 1 \
      --block-size 128 \
      --enable-prefix-caching \
      --trust-remote-code \
      --kv-transfer-config \
      '{
        "kv_connector": "UCMConnector",
        "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
          "UCM_CONFIG_FILE": "/etc/ucm/ucm-pd-disaggregated.yaml"
        }
      }'
    ```

    In another shell, start Decode on a different accelerator and port:

    ```bash
    export PYTHONHASHSEED=123456
    export ASCEND_RT_VISIBLE_DEVICES=1

    vllm serve /models/Qwen2.5-7B-Instruct \
      --served-model-name ucm-model \
      --host 0.0.0.0 \
      --port 8200 \
      --tensor-parallel-size 1 \
      --block-size 128 \
      --enable-prefix-caching \
      --trust-remote-code \
      --kv-transfer-config \
      '{
        "kv_connector": "UCMConnector",
        "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
          "UCM_CONFIG_FILE": "/etc/ucm/ucm-pd-disaggregated.yaml"
        }
      }'
    ```

    Start the validation proxy:

    ```bash
    python -m ucm.pd.toy_proxy_server \
      --pd-disaggregation \
      --host 0.0.0.0 \
      --port 8000 \
      --prefiller-hosts 127.0.0.1 \
      --prefiller-ports 8100 \
      --decoder-hosts 127.0.0.1 \
      --decoder-ports 8200
    ```

=== "Multi-node"

    The smallest multi-node topology is 1P1D: one Prefill node, one Decode
    node, and a shared UCM backend. The router may run on either node or on a
    separate gateway.

    ```mermaid
    flowchart LR
      C[Client] --> R[PD router]
      R --> P[Node A: Prefill]
      R --> D[Node B: Decode]
      P --> S[Shared UCM backend]
      S --> D
      D --> R
    ```

    This example uses:

    | Role | Address | Port |
    | --- | --- | --- |
    | Prefill | `192.168.10.1` | `8100` |
    | Decode | `192.168.10.2` | `8200` |
    | Router | `192.168.10.1` | `8000` |

    Before starting either service, verify cross-node storage visibility:

    ```bash
    # Run on the Prefill node.
    touch /mnt/ucm-kv/ucm-storage-check

    # Run on the Decode node; this must print the same file.
    ls -l /mnt/ucm-kv/ucm-storage-check
    ```

    Start Prefill on node A using the Prefill command from the single-node tab.
    Keep port `8100`, set the local accelerator list, and use the shared
    `/mnt/ucm-kv` backend.

    Start Decode on node B using the Decode command from the single-node tab.
    Keep port `8200`, set the local accelerator list, and use the same shared
    backend. Export the same `PYTHONHASHSEED` on both nodes.

    Start the router on node A or a gateway:

    ```bash
    python -m ucm.pd.toy_proxy_server \
      --pd-disaggregation \
      --host 0.0.0.0 \
      --port 8000 \
      --prefiller-hosts 192.168.10.1 \
      --prefiller-ports 8100 \
      --decoder-hosts 192.168.10.2 \
      --decoder-ports 8200
    ```

    To scale to xPyD, pass corresponding host and port lists. The number of
    hosts must match the number of ports in each pool:

    ```bash
    python -m ucm.pd.toy_proxy_server \
      --pd-disaggregation \
      --host 0.0.0.0 \
      --port 8000 \
      --prefiller-hosts 192.168.10.1 192.168.10.2 \
      --prefiller-ports 8100 8100 \
      --decoder-hosts 192.168.10.3 192.168.10.4 \
      --decoder-ports 8200 8200
    ```

    ### Heterogeneous 1P1D

    Centralized storage hand-off can place Prefill and Decode on different
    compute platforms. For an Ascend Prefill node and CUDA Decode node:

    - export `ASCEND_RT_VISIBLE_DEVICES=0` on Prefill;
    - export `CUDA_VISIBLE_DEVICES=0` on Decode;
    - build/install the matching UCM platform package on each node;
    - add `--dtype bfloat16` to both vLLM commands;
    - keep model revision, tokenizer, block size, served name, and shared UCM
      configuration aligned.

    !!! warning

        Matching dtype is necessary but not sufficient. Qualify the exact
        model, vLLM, vLLM Ascend, UCM, quantization, and cache-layout
        combination before treating cross-platform cache data as compatible.

    The validation proxy selects each pool in round-robin order. Production
    routers should add health-aware selection, queueing, retries, admission
    control, and high availability.

## Verify the deployment

Check the router topology:

```bash
curl http://127.0.0.1:8000/healthcheck
```

A 1P1D deployment returns a response similar to:

```json
{
  "status": "ok",
  "mode": "pd-disaggregation",
  "prefill_instances": 1,
  "decode_instances": 1
}
```

Send inference traffic only after both backends are healthy:

```bash
curl http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ucm-model",
    "prompt": "Explain the difference between Prefill and Decode.",
    "max_tokens": 64,
    "temperature": 0,
    "stream": true
  }'
```

Expected evidence in the logs:

1. the proxy reports one Prefill selection followed by one Decode selection;
2. Prefill commits KV blocks to the configured UCM backend;
3. Decode reports external cache hits for the shared prompt blocks;
4. only Decode streams the final response to the client.

## Benchmark

Run the benchmark against the router, not an individual P or D endpoint:

```bash
vllm bench serve \
  --backend vllm \
  --model ucm-model \
  --host 127.0.0.1 \
  --port 8000 \
  --dataset-name random \
  --seed 123456 \
  --num-prompts 10 \
  --random-input-len 4096 \
  --random-output-len 128 \
  --request-rate 1 \
  --ignore-eos
```

Record at least TTFT, TPOT/ITL, end-to-end latency, throughput, cache hit ratio,
and storage bandwidth. Compare against a colocated baseline with the same model,
request set, parallel configuration, and accelerator count.

## Common failures

| Symptom | Check |
| --- | --- |
| Decode recomputes the entire prompt | Verify shared storage visibility, identical cache settings, the same hash seed, and successful Prefill commits. |
| Cache lookup succeeds but load fails | Check model revision, dtype, block size, TP/cache layout, and UCM release compatibility. |
| Proxy reports an upstream 4xx/5xx | Call the selected Prefill and Decode `/health` endpoints directly and inspect their logs. |
| Multi-node deployment works locally but not remotely | Open the vLLM and router ports and verify routing/DNS from the proxy container or host. |
| Storage fills up | Configure backend capacity/GC, inspect UCM store-health metrics, and avoid using an unbounded test directory in production. |
| P/D latency is worse than colocated | Measure storage bandwidth and queueing separately; PD disaggregation adds a KV hand-off and is not automatically faster at low load. |

--8<-- "docs/en/user-guide/deploy/_pd-distributed.inc"

## Related documentation

- [Deploy mode overview](index.md)
- [PD colocated deployment](pd-colocated.md)
- [Official UCM centralized PD guide](https://ucm.readthedocs.io/en/latest/user-guide/pd-disaggregation/centralized_pd.html)
- [Official UCM distributed PD guide](https://ucm.readthedocs.io/en/latest/user-guide/pd-disaggregation/distributed_pd.html)
- [vLLM Ascend single-node PD reference](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/features/pd_disaggregation_mooncake_single_node.html)
- [vLLM Ascend multi-node PD reference](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/features/pd_disaggregation_mooncake_multi_node.html)
