# Use UCM with pyMotor

Follow the [official MindIE-Motor UCM guide](https://mindie-motor.readthedocs.io/zh-cn/latest/user_guide/features/kv_cache_store/backend/ucm/) for deployment. This page explains the Connector composition.

## Integration

Prefill combines Mooncake transport and UCMConnector through MultiConnector. Mooncake transfers the current request's P/D KV; UCM reuses prefixes across requests. Decode uses matching Mooncake transport.

## Configuration

- Transport occupies `connectors[0]`, UCMConnector `connectors[1]`. Do not configure AscendStoreConnector with `backend: ucm`.
- Match `storage_backends` to the volume's `mount_path`; allow Cache buffer headroom in `dshm_size`.
- See [Cache Configuration](../../developer-guide/cache-configuration/index.md) and [Filesystem Pipeline](../../developer-guide/cache-configuration/pipeline.md) for Store settings.

## Deployment

Follow the official [UCM preparation](https://mindie-motor.readthedocs.io/zh-cn/latest/user_guide/features/kv_cache_store/backend/ucm/#准备-ucm), [configuration](https://mindie-motor.readthedocs.io/zh-cn/latest/user_guide/features/kv_cache_store/backend/ucm/#修改-user_configjson) and [deployment](https://mindie-motor.readthedocs.io/zh-cn/latest/user_guide/features/kv_cache_store/backend/ucm/#部署和删除) sections.

## Verification

Check Prefill external hits, completed loads and errors. HBM hits alone do not establish UCM reuse; follow [external-cache verification](../observability/verify-cache.md).
