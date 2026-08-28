Complete the [UCM installation](../../installation.md) for the selected engine
before starting a model. The commands below assume that UCM and the engine are
installed in the same environment and that the model weights are either
downloadable from Hugging Face or available at the path assigned to `MODEL`.

Create a writable cache directory and save the following as
`/etc/ucm/model-tour.yaml`:

```yaml
ucm_connectors:
  - ucm_connector_name: UcmPipelineStore
    ucm_connector_config:
      store_pipeline: Cache|Posix
      storage_backends: /mnt/ucm
      posix_capacity_gb: 1024
      io_direct: false
      store_health:
        enabled: true

enable_event_sync: true
enable_metrics: true
use_layerwise: true
persist_token_threshold: 0
load_tokens_threshold: 0
```

Create `/mnt/ucm` on local NVMe or replace it with a mounted shared directory.
Every serving process that should reuse the same KV cache must see the same
path. Start with `io_direct: false`; enable direct I/O only after the filesystem
and block alignment have been verified.

For an SGLang command, export the equivalent HiCache configuration in the same
shell:

```bash
export HICACHE_CONFIG='{
  "backend_name":"unifiedcache",
  "module_path":"ucm.integration.sglang.unifiedcache_store",
  "class_name":"UnifiedCacheStore",
  "interface_v1":1,
  "kv_connector_extra_config":{
    "ucm_connector_name":"UcmPipelineStore",
    "ucm_connector_config":{
      "store_pipeline":"Cache|Posix",
      "storage_backends":"/mnt/ucm",
      "posix_capacity_gb":1024,
      "io_direct":false
    }
  }
}'
```

