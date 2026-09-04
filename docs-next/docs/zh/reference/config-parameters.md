# 原生UCM方案参数

---

## 顶层参数

> 直接写在 YAML 根级别的全局参数。

| 配置项 | 是否必填 | 取值类型 | 取值范围 | 配置说明 |
|---|---|---|---|---|
| `use_layerwise` | 选填 | bool | 默认 `true` | 是否启用分层（逐层）加载和保存模式。推荐设置为 `true`，DeepSeek v4 系列模型推荐设置为 `false`。 |
| `enable_event_sync` | 选填 | bool | 默认 `true` | 性能优化开关，推荐开启。 |
| `persist_token_threshold` | 选填 | int | `0` | 当请求长度小于 `persist_token_threshold` 时，UCM 软件不对该请求进行处理。 |
| `timeout_ms` | 选填 | int | >0，默认 `30000` | 显存和 DRAM/SHM 之间的拷贝以及读写盘的超时时间（单位：ms）。 |
| `wa_dump_block_wise` | 选填 | bool | `true` | 仅在 FAWA connector 中使用。`true`: 每个 block 的 WA cache 都会被 dump（高频 dump）；`false`: 只 dump 每个 chunk prefill 最后 block 的 WA cache（低频 dump）。 |
| `load_tokens_threshold` | 选填 | int | `0` | 设置触发 KV cache 加载的最小 token 阈值，仅在 DeepSeek V4 系列模型生效。当外部命中 tokens 数 > `load_tokens_threshold` 时触发 KV Cache 加载。 |
| `enable_record_traces` | 选填 | bool | `false` | 用来记录请求信息（时间戳，输入长度，输出长度等信息）。 |
| `enable_metrics` | 选填 | bool | 默认 `true` | 是否开启 metrics 收集。 |
| `use_lite` | 选填 | bool | `false` | 是否启用 UCM Lite 功能。不对 KV Cache 数据进行保存和加载，仅对元数据进行保存和查询。仅可用于评估 KV Cache 命中率情况，无加速效果。 |
| `metrics_config_path` | 选填 | string | 自行配置 | 指定监控指标配置文件路径，启用后可通过 toolkit 进行 UCM 在线监控。参考配置文件：`examples/metrics/metrics_configs.yaml`。 |

---

## ucm_connector_config（存储后端配置）

> 写在 `ucm_connectors[0].ucm_connector_config` 下的参数。

| 配置项 | 是否必填 | 取值类型 | 取值范围 | 配置说明 |
|---|---|---|---|---|
| `store_pipeline` | 选填 | string | 见下方可选值 | 管线名称，决定 Cache 与 Store 的组合方式。默认使用 `Cache\|Posix`。 |
| `storage_backends` | **必填** | string | 自行配置，多个挂载点用冒号隔开 | 填写本地目录或者挂载点，如果有多个挂载点需要用冒号隔开。 |
| `cache_buffer_capacity_gb` | 选填 | int | 见配置说明 | 对于 GQA，默认值是每张卡会占 32GB DRAM 内存。对于 MLA，默认值是单机会占 128GB 的 shm 空间。目前建议全部使用默认值。 |
| `cache_io_aggregation` | 选填 | bool | 默认 `false`，仅在 `PLATFORM=ascend` 且模型为 V4 时自动开启 | 启用 IO 聚合 h2d 传输，仅在 A2 设备生效。 |
| `share_buffer_enable` | 选填 | bool | MLA 默认启用，GQA 默认不启用 | 是否启用共享内存。MLA 如果不用 shm 或 GQA 用 shm 都会导致性能下降。 |

### store_pipeline 可选值

| 值 | 说明 |
|---|---|
| `Cache\|Posix` | 正常使用场景 |
| `Cache\|Empty` | MLA 纯 Cache 测试 |
| `Cache\|Fake` | MLA/GQA 纯 Cache 测试（有精度问题） |
| `Empty` | Store 全部接口空实现，引擎侧永远不命中 |
| `Fake` | 不实际存取，仅存元数据 lookup，测试极限性能 |
| `Mooncake` | 对接 Mooncake 内存池，仅支持 vllm-ascend |
| `Mooncake\|Posix` | 对接 Mooncake 内存池并支持落盘 |
| `YuanRong` | 对接 YuanRong 内存池 |
| `YuanRong\|Posix` | 对接 YuanRong 内存池并支持落盘 |
