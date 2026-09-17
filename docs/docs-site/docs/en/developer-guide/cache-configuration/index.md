# Cache configuration

UCM stores reusable blocks outside the engine's device KV Cache. Choose how the engine connects to UCM and where those blocks are stored. Establish the integration with an [engine quickstart](../../user-guide/quick_start/index.md), then select a backend.

## Choose a storage backend {#storage-backends}

A backend determines where KV blocks live, how they are read and which services are required. Decide whether cache must survive processes or be shared across nodes, then estimate capacity and read cost. More capacity retains more prefixes; the benefit still depends on repeated prefixes and read latency.

| Backend | Use case | Prerequisites |
| --- | --- | --- |
| [Filesystem pipeline](pipeline.md) | `Cache|Posix` combines host cache with local or mounted storage | Writable paths, host memory and a compatible Pipeline build |
| [NFS connector](nfs.md) | Keep direct filesystem access through `UcmNfsStore` | Local/NFS paths and compatible equal-sized tensors |
| [3FS](ds3fs.md) | Access 3FS through `Cache|Ds3fs` | Running 3FS client with matching headers and libraries |
| [Mooncake](mooncake.md) | Shared memory pool, optionally persisted through Posix | Compatible Mooncake services, client and network settings |
| [KV compression](compress.md) | Reduce stored payload with `Cache|Compress|Posix` | Supported codec and dtype, with quality evaluation |

An NFS mount can serve either Posix or the direct NFS connector. The mount and the UCM backend are separate choices. For a new deployment requiring host caching, capacity reclamation and health checks, start with the [filesystem pipeline](pipeline.md).

## Before configuring

1. Select artifacts containing the backend in [Quickstart](../../user-guide/quick_start/index.md), with compatible engine and device dependencies.
2. Prepare a dedicated cache namespace. Sharing instances need compatible model, tokenizer, KV layout and execution settings.
3. Budget host buffers, device memory and external storage. Shared and per-worker host buffers have different allocation scopes.

## Apply the configuration

For vLLM, `kv_connector_extra_config.UCM_CONFIG_FILE` points to the UCM YAML. Root options configure the Connector; `ucm_connectors[].ucm_connector_config` configures the selected Store. The file and storage paths must be visible to the engine process or container.

SGLang owns its host cache; use the storage configuration from the [SGLang quickstart](../../user-guide/quick_start/index.md#sglang).

Use only fields accepted by that backend; see the [configuration reference](../../reference/config-parameters.md). Then [verify external reuse](../../user-guide/observability/verify-cache.md) before comparing read and recomputation cost.

For block matching and transfer completion, see [how caching works](../capability-principles.md). For separate Prefill and Decode instances, continue with [PD Disaggregation](../../user-guide/capabilities/pd-disaggregation/index.md).
