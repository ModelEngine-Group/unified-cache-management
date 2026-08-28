# PD Colocated (Mixed) Deployment

In PD colocated mode, each vLLM instance performs both Prefill and Decode. UCM
adds persistent external KV cache without changing that scheduling model. This
mode is the smallest operational step beyond a normal vLLM deployment.

Use the tabs below for a single-node service or a multi-node worker pool.

In the UCM reference proxy this mode is called `pd-mixed`: the router forwards
the complete request to one worker and does not perform a separate Prefill
request. UCM adds reusable external cache, but it does not split the serving
scheduler into P and D roles.

## Before you start

- Install a compatible UCM, vLLM, and vLLM Ascend release. See
  [Installation](../installation.md).
- Replace all example model and configuration paths.
- On Ascend, select devices with `ASCEND_RT_VISIBLE_DEVICES`. On CUDA, use
  `CUDA_VISIBLE_DEVICES` instead.
- Keep `PYTHONHASHSEED` and the cache compatibility settings identical across
  workers that share a backend.

## Prepare the UCM configuration

Create `/etc/ucm/ucm-pd-colocated.yaml`:

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
use_layerwise: true
```

`Cache|Posix` uses local DRAM as the fast tier and a POSIX path as the durable
tier. For one node, `/mnt/ucm-kv` may be a local SSD directory. For multiple
nodes, it must resolve to shared storage, such as an NFS/3FS mount. A local path
on each node creates isolated caches even when the path strings are identical.

`cache_buffer_capacity_gb` applies to each serving process, so include every
worker on the host when calculating total DRAM use. Set `io_direct` only when
the filesystem and alignment requirements for direct I/O are met.

## Start order

1. Create the storage backend and configuration on every worker.
2. Start the full P+D worker processes and query each `/health` endpoint.
3. For a worker pool, start the router and inspect `/healthcheck`.
4. Send a functional request, then repeat a deterministic workload to verify
   external cache reuse.

## Deploy

=== "Single node"

    The client talks directly to one mixed worker. Prefill and Decode share the
    worker's accelerator allocation and scheduler.

    ```mermaid
    flowchart LR
      C[Client] --> W[PD-mixed worker]
      W <--> U[UCM DRAM and local storage]
    ```

    Create the storage directory, then start the service:

    ```bash
    mkdir -p /mnt/ucm-kv

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
          "UCM_CONFIG_FILE": "/etc/ucm/ucm-pd-colocated.yaml"
        }
      }'
    ```

    On CUDA, replace `ASCEND_RT_VISIBLE_DEVICES=0` with
    `CUDA_VISIBLE_DEVICES=0`.

    Send a request directly to the worker:

    ```bash
    curl http://127.0.0.1:8100/v1/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "ucm-model",
        "prompt": "Explain why KV cache reuse reduces TTFT.",
        "max_tokens": 64,
        "temperature": 0
      }'
    ```

=== "Multi-node"

    Every worker still performs both phases. The router distributes complete
    requests, while UCM makes compatible KV blocks available to all workers.

    ```mermaid
    flowchart LR
      C[Client] --> R[Router]
      R --> W1[Node A: P and D]
      R --> W2[Node B: P and D]
      W1 <--> S[Shared UCM backend]
      W2 <--> S
    ```

    Mount the same shared backend on every worker before starting vLLM. The
    following example assumes both nodes can access `/mnt/ucm-kv`.

    On node A (`192.168.10.1`):

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
          "UCM_CONFIG_FILE": "/etc/ucm/ucm-pd-colocated.yaml"
        }
      }'
    ```

    Run the same command on node B (`192.168.10.2`). Change only the device
    selection when the local topology requires it; keep the model, block size,
    hash seed, and UCM configuration compatible.

    Before starting the router, confirm that a file created through
    `/mnt/ucm-kv` on node A is immediately visible through the same mount on
    node B. Otherwise the workers have independent caches.

    For a functional test, start the repository's round-robin proxy on a node
    that can reach both workers:

    ```bash
    python -m ucm.pd.toy_proxy_server \
      --host 0.0.0.0 \
      --port 8000 \
      --worker-hosts 192.168.10.1 192.168.10.2 \
      --worker-ports 8100 8100
    ```

    The proxy runs in mixed mode when `--pd-disaggregation` is absent. Confirm
    the discovered workers:

    ```bash
    curl http://127.0.0.1:8000/healthcheck
    ```

    Then send inference traffic to port `8000`, not directly to a worker.

    !!! warning "The bundled proxy is a validation tool"

        `ucm.pd.toy_proxy_server` provides round-robin routing and has no
        production-grade admission control, health-aware balancing, retries, or
        high availability. Replace it with the deployment framework's router in
        production.

## Verify cache reuse

Send the same deterministic prompt twice. The first request populates UCM; the
second should report external cache hits in the UCM logs and normally has a lower
TTFT. To generate repeatable traffic, run the following command twice against
the service or router port:

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

For a single-node service without a router, change `--port` to `8100`.

## Common failures

| Symptom | Check |
| --- | --- |
| Node B never reports external hits | Confirm both nodes mount the same storage export and can see files created by the other node. |
| The service fails while allocating the host cache | Reduce `cache_buffer_capacity_gb` and check available DRAM and shared memory. |
| Repeated prompts generate different cache keys | Align the model/tokenizer revision, cache dtype, block size, UCM version, and `PYTHONHASHSEED`. |
| Direct I/O fails at startup or during writes | Set `io_direct: false`, then verify filesystem and alignment support before re-enabling it. |
| Router returns an upstream error | Query each worker's `/health` endpoint directly and verify inter-node firewall rules. |

## Related documentation

- [Deploy mode overview](index.md)
- [PD disaggregated deployment](pd-disaggregated.md)
- [PD disaggregated P2P hand-off](pd-disaggregated.md#p2p-hand-off-ucm-mooncake)
- [UCM configuration example](https://github.com/ModelEngine-Group/unified-cache-management/blob/feature/docs-next/examples/ucm_config_example.yaml)
- [vLLM Ascend PD colocated reference](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/features/pd_colocated_mooncake_multi_instance.html)
