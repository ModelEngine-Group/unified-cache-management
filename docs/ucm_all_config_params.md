# UCM 所有可配置参数汇总

---

## 目录

1. [顶层 YAML 参数](#1-顶层-yaml-参数)
2. [ucm_connector_config 子参数](#2-ucm_connector_config-子参数)
3. [store_health 健康熔断配置](#3-store_health-健康熔断配置)
4. [各 Store C++ 配置结构体](#4-各-store-c-配置结构体)
5. [已注册管线](#5-已注册管线)
6. [稀疏注意力配置](#6-稀疏注意力配置)
7. [Metrics 配置](#7-metrics-配置)
8. [环境变量](#8-环境变量)
9. [MindIE ucm_config.json](#9-mindie-ucm_configjson)

---

## 1. 顶层 YAML 参数

> 消费于 `ucm/integration/vllm/ucm_connector.py` 和 `hma_connector.py`

### ✏️ 需要手动填写

| 参数名 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `ucm_connectors` | list[dict] | `[]` | 存储连接器配置列表，每项含 `ucm_connector_name`、`ucm_connector_module_path`、`ucm_connector_config` |
| `enable_event_sync` | bool | `True` | dump 前用 GPU event 同步计算完成；**NFS 场景需设为 `false`** |
| `use_layerwise` | bool | `False` | 逐层 load/save 流水；**PP 场景必须为 `true`** |
| `use_lite` | bool | `False` | 使用 Lite Connector（仅统计命中率，不做真实 I/O） |
| `use_inference_duration_monitor` | bool | `False` | 启用无 I/O 推理时长监控连接器 |
| `inference_duration_monitor_fake_hit_ratio` | float | `0.0` | 监控连接器模拟命中率（0.0 ~ 1.0） |
| `hit_ratio` | float | 未设 | 设置后选 Mock Connector 模拟命中率（测试用） |
| `persist_token_threshold` | int | `0` | 请求 token 数低于此值时跳过 KV 持久化 |
| `load_tokens_threshold` | int | `0` | 外部命中 token 数 ≤ 此值时跳过加载（直接重算更快） |
| `wa_dump_block_wise` | bool | `True` | WA 缓存是否逐 block dump；`false` 时仅 dump chunk prefill 最后一块 |
| `use_consistency_manager` | bool | `not is_mla` | 跨 rank 一致性管理器，MLA 模型默认关闭，非 MLA 默认开启 |
| `hybrid_linear_attention_layerwise` | bool | `True` | HLA 模型是否启用 layerwise |
| `enable_metrics` | bool | `True` | UCM 指标采集总开关 |
| `metrics_config` | dict | 内联 | 内联 metrics 配置字典（优先于 path） |
| `metrics_config_path` | str | `""` | metrics YAML 配置文件路径 |
| `ucm_sparse_config` | dict | 未设 | 稀疏注意力子配置，键为方法名（见第 6 节） |

### 🤖 自动/运行时填充（无需手动填）

| 参数名 | 类型 | 说明 |
|---|---|---|
| `enable_record_traces` | bool | 调试用，框架按需开启，默认 `false` |

---

## 2. ucm_connector_config 子参数

> 处理于 `ucm/integration/vllm/ucm_connector.py` 第 1236–1313 行

### ✏️ 需要手动填写

| 参数名 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `store_pipeline` | str | **必填** | 管线名，如 `Cache\|Posix`、`Mooncake\|Posix`（见第 5 节） |
| `storage_backends` | str | **必填** | 存储路径，冒号分隔，如 `/mnt/nfs:/mnt/ssd` |
| `io_direct` | bool | `false` | 启用 Direct I/O (O_DIRECT)，SSD 场景建议开启 |
| `posix_capacity_gb` | int | `0` | Posix 容量上限（GB），>0 自动开启 GC |
| `posix_io_engine` | str | `"psync"` | Posix IO 引擎：`"aio"` 或 `"psync"` |
| `use_gdr` | bool | `false` | 启用 GPUDirect RDMA（需 `ENABLE_GDR=1` 构建） |
| `cache_buffer_capacity_gb` | int | `128` | CacheStore 共享内存缓冲容量（GB） |
| `share_buffer_capacity_gb` | int | `64` | MooncakeStore 共享缓冲容量（GB） |
| `share_buffer_enable` | bool | MLA 默认开 | 共享缓冲模式 |
| `global_segment_size_gb` | int | `30` | Mooncake 全局段大小（GB） |
| `local_hostname` | str | `""` | Mooncake 本地主机名（多机场景必填） |
| `master_server_address` | str | `"127.0.0.1:50088"` | Mooncake master 地址（多机场景须改为实际地址） |
| `metadata_server` | str | `"P2PHANDSHAKE"` | Mooncake metadata 服务类型 |
| `protocol` | str | `"ascend"` | Mooncake 协议 |
| `replica_num` | uint32 | `1` | Mooncake 副本数 |
| `cpu_affinity_cores` | list[int] | `None` | Store 线程 CPU 亲和核，需按机器 NUMA 拓扑填写 |
| `posix_gc_enable` | bool | 自动 | 仅 DP0 scheduler 进程运行 GC，多数情况无需手动干预 |

### 🤖 自动/运行时填充（无需手动填）

| 参数名 | 类型 | 说明 |
|---|---|---|
| `unique_id` | str | 框架自动生成，唯一标识存储实例 |
| `device_id` | int | Worker 侧自动注入，对应当前设备 |
| `tensor_size_list` | list[int] | 由模型结构运行时计算，每个 tensor 字节大小 |
| `shard_size` | int | 运行时计算，对齐到 4096 |
| `block_size` | int | 运行时计算，KV block 大小 |
| `local_rank_size` | int | MLA 取 tp_size，非 MLA 为 1，自动设置 |
| `gpu_kv_buffer_addrs` | list[int] | 运行时从 vLLM 获取，GPU KV 缓冲地址列表 |
| `gpu_kv_buffer_sizes` | list[int] | 运行时从 vLLM 获取，GPU KV 缓冲大小列表 |

---

## 3. store_health 健康熔断配置

> 结构定义于 `ucm/store/pipeline/cc/store_health_config.h` 第 33–54 行

### ✏️ 需要手动填写

| 参数名 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `enabled` | bool | `false` | 启用健康熔断器，生产环境建议开启 |
| `health_check_interval_s` | int | `10` | 健康探测间隔（秒） |
| `health_check_timeout_s` | int | `3` | 单次探测超时（须 < interval） |
| `health_window_size` | size_t | `8` | 健康结果滑动窗口大小 |
| `failure_threshold` | size_t | `2` | 熔断阈值（须 ≤ window_size） |

> 该配置块所有字段均为手动填写，默认值即关闭状态，需按需配置。

**YAML 示例：**

```yaml
store_health:
  enabled: true
  health_check_interval_s: 10
  health_check_timeout_s: 3
  health_window_size: 8
  failure_threshold: 2
```

---

## 4. 各 Store C++ 配置结构体

> 均通过 pybind11 从 YAML 字典注入。大部分字段由框架自动从顶层配置映射，用户通过 `ucm_connector_config` 的高层键间接配置。

### 4.1 CacheStore::Config

> `ucm/store/cache/cc/global_config.h` 第 43–66 行

#### ✏️ 需要手动填写（通过 ucm_connector_config 映射）

| 字段 | 类型 | 默认值 | 对应 YAML 键 | 说明 |
|---|---|---|---|---|
| `bufferCapacity` | size_t | `256 GB` | `cache_buffer_capacity_gb` | 缓冲容量，通过顶层键配置 |
| `ioDirect` | bool | `false` | `io_direct` | Direct I/O |
| `shareBufferEnable` | bool | `true` | `share_buffer_enable` | 共享缓冲模式 |
| `useGdr` | bool | `false` | `use_gdr` | GPUDirect RDMA |
| `cpuAffinityCores` | vector\<ssize_t\> | `{}` | `cpu_affinity_cores` | CPU 亲和核 |
| `cacheSdmaDirect` | bool | `false` | 编译宏控制 | Ascend SDMA 直接传输 |
| `sdmaDirectLaunchGranularity` | string | `"shard"` | 可手动覆盖 | SDMA 粒度：`"shard"` 或 `"task"` |

#### 🤖 自动/运行时填充（无需手动填）

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `uniqueId` | string | `""` | 框架自动生成 |
| `deviceId` | int32_t | `-1` | Worker 侧自动注入 |
| `tensorSizes` | vector\<size_t\> | `{}` | 运行时从模型计算 |
| `shardSize` | size_t | `0` | 运行时计算 |
| `blockSize` | size_t | `0` | 运行时计算 |
| `loadExclusiveBufferNumber` | size_t | `1024` | 内部默认值，无需修改 |
| `waitingQueueDepth` | size_t | `8192` | 内部默认值 |
| `runningQueueDepth` | size_t | `524288` | 内部默认值 |
| `timeoutMs` | size_t | `30000` | 内部默认值 |
| `streamNumber` | size_t | `4` | 内部默认值 |
| `cacheLoadBackendOnly` | bool | `false` | 内部控制逻辑 |
| `gpuKvBufferAddrs` | vector\<uintptr_t\> | `{}` | 运行时从 vLLM 注入 |
| `gpuKvBufferSizes` | vector\<size_t\> | `{}` | 运行时从 vLLM 注入 |
| `localRankSize` | size_t | `8` | 运行时从 tp_size 计算 |

---

### 4.2 PosixStore::Config

> `ucm/store/posix/cc/global_config.h` 第 32–55 行

#### ✏️ 需要手动填写（通过 ucm_connector_config 映射）

| 字段 | 类型 | 默认值 | 对应 YAML 键 | 说明 |
|---|---|---|---|---|
| `storageBackends` | vector\<string\> | `{}` | `storage_backends` | 存储路径列表 |
| `ioEngine` | string | `"psync"` | `posix_io_engine` | IO 引擎 |
| `ioDirect` | bool | `false` | `io_direct` | Direct I/O |
| `cpuAffinityCores` | vector\<ssize_t\> | `{}` | `cpu_affinity_cores` | CPU 亲和核 |
| `posixCapacityGb` | size_t | `0` | `posix_capacity_gb` | 容量（GB），>0 启用 GC |
| `posixGcEnable` | bool | `true` | `posix_gc_enable` | 启用 GC |
| `posixGcRecyclePercent` | double | `0.1` | 可手动覆盖 | GC 回收比例 |
| `posixGcTriggerThresholdRatio` | double | `0.7` | 可手动覆盖 | GC 触发阈值比例 |

#### 🤖 自动/运行时填充（无需手动填）

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `deviceId` | int32_t | `-1` | Worker 侧自动注入 |
| `tensorSize` | size_t | `0` | 运行时计算 |
| `shardSize` | size_t | `0` | 运行时计算 |
| `blockSize` | size_t | `0` | 运行时计算 |
| `dataTransConcurrency` | size_t | `128` | 内部默认值 |
| `lookupConcurrency` | size_t | `16` | 内部默认值 |
| `openConcurrency` | size_t | `32` | 内部默认值 |
| `commitConcurrency` | size_t | `4` | 内部默认值 |
| `timeoutMs` | size_t | `30000` | 内部默认值 |
| `dataDirShardBytes` | size_t | `3` | 内部默认值 |
| `posixGcConcurrency` | size_t | `16` | 内部默认值 |
| `posixGcCheckIntervalSec` | size_t | `30` | 内部默认值 |
| `posixGcMaxRecycleCountPerShard` | size_t | `50000` | 内部默认值 |
| `posixGcShardSampleRatio` | double | `0.1` | 内部默认值 |

---

### 4.3 MooncakeStore::Config

> `ucm/store/mooncakestore/cc/global_config.h` 第 37–71 行

#### ✏️ 需要手动填写（通过 ucm_connector_config 映射）

| 字段 | 类型 | 默认值 | 对应 YAML 键 | 说明 |
|---|---|---|---|---|
| `localHostname` | string | `""` | `local_hostname` | 本地主机名，多机必填 |
| `metadataServer` | string | `"P2PHANDSHAKE"` | `metadata_server` | metadata 服务 |
| `masterServerAddress` | string | `"127.0.0.1:50088"` | `master_server_address` | master 地址，多机须改 |
| `protocol` | string | `"ascend"` | `protocol` | 协议 |
| `globalSegmentSize` | uint64 | `30 GB` | `global_segment_size_gb` | 全局段大小 |
| `replicaNum` | uint32 | `1` | `replica_num` | 副本数 |
| `shareBufferCapacity` | uint64 | `64 GB` | `share_buffer_capacity_gb` | 共享缓冲容量 |
| `ioDirect` | bool | `false` | `io_direct` | Direct I/O |
| `cpuAffinityCores` | vector\<ssize_t\> | `{}` | `cpu_affinity_cores` | CPU 亲和核 |

#### 🤖 自动/运行时填充（无需手动填）

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `deviceId` | int32_t | `-1` | Worker 侧自动注入 |
| `uniqueId` | string | `""` | 框架自动生成 |
| `tensorSizeList` | vector\<uint64\> | `{}` | 运行时计算 |
| `localBufferSize` | uint64 | `1 GB` | 内部默认值 |
| `withSoftPin` | bool | `false` | 内部控制 |
| `loadQueueDepth` | uint32 | `524288` | 内部默认值 |
| `dumpQueueDepth` | uint32 | `8192` | 内部默认值 |
| `hostBufPoolSize` | uint32 | `1024` | 内部默认值 |
| `timeoutMs` | size_t | `0` | 内部默认值 |
| `streamNumber` | size_t | `4` | 内部默认值 |
| `shareBufferNumber` | size_t | `0` | 内部计算 |
| `localRankSize` | size_t | `1` | 运行时计算 |
| `gpuKvBufferAddrs` | vector\<uintptr_t\> | `{}` | 运行时从 vLLM 注入 |
| `gpuKvBufferSizes` | vector\<size_t\> | `{}` | 运行时从 vLLM 注入 |
| `deviceName` | string | `""` | 运行时检测 |

---

### 4.4 Ds3fsStore::Config

> `ucm/store/ds3fs/cc/global_config.h` 第 32–44 行

#### ✏️ 需要手动填写

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `storageBackends` | vector\<string\> | `{}` | 存储路径（通过 `storage_backends` 配置） |
| `ioDirect` | bool | `true` | Direct I/O，默认 true，与 PosixStore 不同 |
| `numaId` | int32_t | `-1` | NUMA 节点 ID，多 NUMA 场景可手动指定 |
| `streamNumber` | size_t | `32` | 并发流数量，可按需调整 |
| `iorEntries` | size_t | `1` | IOR 条目数 |
| `iorDepth` | int32_t | `1` | IOR 深度 |

#### 🤖 自动/运行时填充（无需手动填）

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `deviceId` | int32_t | `-1` | Worker 侧自动注入 |
| `tensorSize` | size_t | `0` | 运行时计算 |
| `shardSize` | size_t | `0` | 运行时计算 |
| `blockSize` | size_t | `0` | 运行时计算 |
| `timeoutMs` | size_t | `30000` | 内部默认值 |

---

### 4.5 Compressor::Config

> `ucm/store/compress/cc/global_config.h` 第 34–47 行

#### ✏️ 需要手动填写

| 字段 | 类型 | 默认值 | 对应 YAML 键 | 说明 |
|---|---|---|---|---|
| `compressRatio` | int32_t | `32` | `compress_ratio` | 压缩比 |
| `dataType` | int32_t | `100` | `data_type` | 数据类型 |
| `decompressThreadNum` | size_t | `6` | `decompress_thread_num` | 解压线程数 |
| `streamNumber` | size_t | `8` | 可手动覆盖 | 流数量 |

#### 🤖 自动/运行时填充（无需手动填）

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `uniqueId` | string | `""` | 框架自动生成 |
| `deviceId` | int32_t | `-1` | Worker 侧自动注入 |
| `tensorSize` | size_t | `0` | 运行时计算 |
| `shardSize` | size_t | `0` | 运行时计算 |
| `blockSize` | size_t | `0` | 运行时计算 |
| `layerSize` | size_t | `0` | 运行时计算 |
| `timeoutMs` | size_t | `30000` | 内部默认值 |

---

### 4.6 FakeStore::Config

> `ucm/store/fake/cc/global_config.h` 第 32–36 行（测试用）

#### ✏️ 需要手动填写

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `bufferNumber` | size_t | `1048576` | 缓冲数，测试时可调整 |
| `shareBufferEnable` | bool | `true` | 共享缓冲 |

#### 🤖 自动/运行时填充（无需手动填）

| 字段 | 类型 | 说明 |
|---|---|---|
| `uniqueId` | string | 框架自动生成 |

---

### 4.7 PcStore 配置键映射

> `ucm/store/pcstore/pcstore_connector_v1.py` 第 55–69 行

#### ✏️ 需要手动填写

| YAML 键 | C++ 属性 | 说明 |
|---|---|---|
| `io_direct` | `transferIoDirect` | Direct I/O |
| `stream_number` | `transferStreamNumber` | 流数量 |
| `buffer_number` | `transferBufferNumber` | 缓冲数 |
| `timeout_ms` | `transferTimeoutMs` | 超时 |
| `use_scatter_gather` | `transferScatterGatherEnable` | Scatter-Gather |
| `use_gdr` | `transferUseGdr` | GPUDirect RDMA |
| `shard_data_dir` | `shardDataDir` | 分片数据目录 |

#### 🤖 自动/运行时填充（无需手动填）

| YAML 键 | C++ 属性 | 说明 |
|---|---|---|
| `unique_id` | `uniqueId` | 框架自动生成 |
| `local_rank_size` | `transferLocalRankSize` | 运行时计算 |
| `device_id` | `transferDeviceId` | Worker 侧自动注入 |
| `tensor_size` | `transferIoSize` | 由 `tensor_size_list` 运行时派生 |
| `gpu_kv_buffer_addrs` | `gpuKvBufferAddrs` | 运行时从 vLLM 注入 |
| `gpu_kv_buffer_sizes` | `gpuKvBufferSizes` | 运行时从 vLLM 注入 |

---

## 5. 已注册管线

> `ucm/store/pipeline/connector.py` 第 306–317 行

> ✏️ `store_pipeline` 字段为**必填**，从下表选择一项填入。

| 管线名 | 说明 |
|---|---|
| `Cache\|Posix` | Cache + Posix（**生产推荐**，NFS/SSD 场景） |
| `Cache\|Ds3fs` | Cache + Ds3fs（华为 DS3 存储） |
| `Cache\|Compress\|Posix` | Cache + 压缩 + Posix |
| `Cache\|Empty` | Cache + Empty（调试用） |
| `Cache\|Fake` | Cache + Fake（测试用） |
| `Mooncake` | 仅 Mooncake（分布式内存场景） |
| `Mooncake\|Posix` | Mooncake + Posix |
| `Posix` | 仅 Posix |
| `Fake` | 仅 Fake（测试用） |
| `Empty` | 仅 Empty |

---

## 6. 稀疏注意力配置

> 通过顶层 `ucm_sparse_config` 下的键配置（`ucm/sparse/factory.py`）

### 6.1 GSAOnDevice

> `ucm/sparse/gsa_on_device/gsa_on_device_config.py` 第 36–97 行

#### ✏️ 需要手动填写

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model_name` | str | `"DummyModel"` | 模型名，须与实际模型对应 |
| `is_mla` | bool | `false` | 是否 MLA 架构（DeepSeek 等） |
| `hash_weight_type` | str | `"random"` | 哈希权重类型：`"random"` 或 `"fixed"` |
| `num_hidden_layers` | int | `36` | 隐藏层数，须与模型一致 |
| `gpu_seq_len_threshold` | int | `2048` | GPU 触发 GSA 的最小序列长度 |
| `gpu_concurrency_threshold` | int | `4` | GPU 触发 GSA 的最小并发 |
| `npu_seq_len_threshold` | int | `2048` | NPU 触发 GSA 的最小序列长度 |
| `npu_concurrency_threshold` | int | `4` | NPU 触发 GSA 的最小并发 |
| `chunk_size` | int | `128` | 分块大小（须被 128 整除） |
| `chunk_repre_method` | str | `"max"` | 块表示方法：`"max"` / `"min"` / `"sum"` |
| `head_dim` | int | `128` | 头维度，须与模型一致 |
| `hash_bits` | int | `128` | 哈希位数 |
| `top_k_ratio_per_layer` | list[float] | `[0.3]*36` | 每层 top-k 比例，长度须等于 `num_hidden_layers` |
| `must_select_blocks` | list[int] | `[0,-2,-1]` | 必选块（非负从头，负从尾） |
| `kv_lora_rank` | int? | `None` | **MLA 必填**：KV-LoRA 秩 |
| `qk_rope_head_dim` | int? | `None` | **MLA 必填**：QK-RoPE 头维度 |

#### 🤖 自动/运行时填充（无需手动填）

| 字段 | 类型 | 说明 |
|---|---|---|
| `top_k_index_reuse` | list[int] | 内部复用索引，默认 `[-1]*36` |
| `hash_weight` | list[list[float]]? | 仅 `hash_weight_type="fixed"` 时手动提供，否则自动生成 |
| `hash_bits_kv_lora` | int? | MLA 内部计算 |
| `hash_bits_qk_rope` | int? | MLA 内部计算 |
| `hash_weight_kv_lora` | list[list[float]]? | MLA 内部计算 |
| `hash_weight_qk_rope` | list[list[float]]? | MLA 内部计算 |
| `vllm_hash_attention_topk` | int? | 框架内部设置 |
| `vllm_hash_attention_reduction_head_num` | int? | 框架内部设置 |
| `vllm_hash_attention_rollback_layers` | list[int] | 框架内部设置，默认 `[]` |
| `vllm_hash_attention_skip_layers` | list[int] | 框架内部设置，默认 `[]` |

---

### 6.2 ESA

> `ucm/sparse/esa/esa.py` — **所有字段均需手动填写**

| 字段 | 类型 | 说明 |
|---|---|---|
| `sparse_ratio` | float | 稀疏比例（决定 top-k = sparse_range × sparse_ratio） |
| `init_window_sz` | int | 初始窗口大小（块数） |
| `local_window_sz` | int | 本地窗口大小（块数） |
| `retrieval_stride` | int | 检索步幅 |
| `min_blocks` | int | 触发稀疏的最小块数 |

---

### 6.3 KVStarMultiStep

> `ucm/sparse/kvstar/multistep.py` — **所有字段均需手动填写**

| 字段 | 类型 | 说明 |
|---|---|---|
| `init_window_sz` | int | 初始窗口大小 |
| `local_window_sz` | int | 本地窗口大小 |
| `retrieval_stride` | int | 检索步幅 |
| `sparse_ratio` | float | 稀疏比例 |
| `blk_repre_dim_prune_ratio` | float | 块表示维度裁剪比例（< 0.98 时启用） |
| `blk_repre_inner_token_merge` | int | 块表示内部 token 合并数 |

---

### 6.4 Blend

> `ucm/sparse/blend/blend.py` — **所有字段均需手动填写**

| 字段 | 类型 | 说明 |
|---|---|---|
| `compute_meta` | dict | 每层 blend 计算元数据，含每层 `ratio`（float） |

---

### 6.5 ReRope

> ReRope 通过**环境变量**配置，见第 8.2 节（`VLLM_USE_REROPE`、`REROPE_WINDOW`、`TRAINING_LENGTH`）。

---

## 7. Metrics 配置

> `ucm/metrics_config.py`、`ucm/observability.py`

### ✏️ 需要手动填写

| 键 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `enable_metrics` | bool | `true` | 总开关，不需要监控可关闭 |
| `log_interval` | int | `5` | 指标日志间隔（秒） |
| `vllm_connector_prefix` | str | `"ucm:"` | vLLM 连接器指标前缀 |
| `consumers` | dict | `{"vllm_connector": true}` | 消费者开关（键 `multiproc` / `vllm_connector`） |
| `multiproc_dir` | str | `"/vllm-workspace"` | Prometheus 多进程目录，需与实际路径一致 |
| `metrics_config_path` | str | `""` | metrics YAML 路径（与内联二选一） |

### 🤖 自动/运行时填充（无需手动填）

> 单个 `MetricDefinition` 条目由框架内置定义，用户无需手动声明各指标项。

---

## 8. 环境变量

### 8.1 UCM 日志

> `ucm/logger.py` 第 28–41 行

#### ✏️ 需要手动填写

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `UCM_LOG_LEVEL` | `"info"` | 日志级别：debug / info / warning / error / critical / off |
| `UCM_LOG_PATH` | `"log"` | 日志目录，生产建议设为持久化路径 |
| `UCM_LOG_MAX_FILES` | `10` | 每进程最大轮转文件数 |
| `UCM_LOG_MAX_SIZE` | `5` | 单文件最大大小（MiB） |
| `UCM_LOG_TO_FILE` | `true` | 是否启用文件日志 |
| `UCM_LOG_RATE_LIMIT_ENABLE` | `true` | 启用日志限流 |
| `UCM_LOG_RATE_LIMIT_WINDOW_MS` | `10000` | 限流时间窗口（ms） |
| `UCM_LOG_RATE_LIMIT_MAX_LOGS` | `3` | 窗口内最大日志数 |

#### 🤖 自动/运行时填充（无需手动填）

| 环境变量 | 说明 |
|---|---|
| `UCM_CAPTURE_VLLM_LOG` | 默认 `true`，自动将 vLLM 日志写入文件，通常无需修改 |
| `UC_LOGGER_LEVEL` | 旧版日志级别，`UCM_LOG_LEVEL` 未设时框架自动回退，新部署直接用 `UCM_LOG_LEVEL` |

---

### 8.2 UCM 功能开关

#### ✏️ 需要手动填写

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `ENABLE_UCM_PATCH` | `""` | **必须设为 `"1"`**，否则 UCM 不生效 |
| `UCM_CONFIG_FILE` | 未设 | YAML 配置文件路径（与 `kv_connector_extra_config` 二选一） |
| `ENABLE_SPARSE` | `"0"` | 设为 `1` 启用稀疏注意力 patch |
| `VLLM_HASH_ATTENTION` | `"0"` | 设为 `"1"` 启用 hash attention 路径 |
| `VLLM_CPU_AFFINITY` | 未设 | 设为 `"1"` 启用 CPU 亲和性绑定（非 NPU 场景） |
| `VLLM_USE_REROPE` | `"0"` | 设为 `"1"` 启用 ReRope |
| `REROPE_WINDOW` | `32768` | ReRope 窗口大小 |
| `TRAINING_LENGTH` | `32768` | 训练长度 |
| `VLLM_USE_V1` | `"1"` | 是否使用 vLLM V1（默认已开启） |

---

### 8.3 MindIE 集成

> `ucm/integration/mindie/unifiedcache_mempool.py`

#### ✏️ 需要手动填写

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `MINDIE_UC_TIME_STAT` | `"0"` | 设为 `"1"` 启用 MindIE UC 时间统计 |
| `NUM_ACCELERATOR` | 未设 | 加速器数量，MindIE 场景必填 |
| `BYPASS_UC` | `0` | 非 0 时绕过 UC（调试用） |

---

### 8.4 构建相关

> 仅在编译时设置，运行时无需关注

| 环境变量 | 说明 |
|---|---|
| `UCM_ENABLE_MINDIE` | 非 `"0"` / `"false"` / `""` 时构建 MindIE 集成 |
| `UCM_CXX11_ABI` | MindIE 构建时 CXX11 ABI 标志（`0` 或 `1`） |
| `ENABLE_GDR` | 设为 `1` 时编译 GPUDirect RDMA 支持 |

---

### 8.5 系统级环境变量

> 非 UCM 专有，配合 UCM 使用的 vLLM / Ascend / HCCL / OpenMP 变量，**均需手动按部署环境设置**

#### OpenMP / PyTorch NPU

| 环境变量 | 常用值 | 说明 |
|---|---|---|
| `OMP_PROC_BIND` | `false` | OpenMP 线程不绑核 |
| `OMP_NUM_THREADS` | `10`（单机）/ `100`（多机） | OpenMP 线程数 |
| `PYTORCH_NPU_ALLOC_CONF` | `expandable_segments:True` | NPU 显存可扩展段分配 |
| `PYTHONHASHSEED` | `0` | Python 哈希种子（保证可复现） |

#### HCCL 通信（Ascend）

| 环境变量 | 常用值 | 说明 |
|---|---|---|
| `HCCL_BUFFSIZE` | `200`（多机）/ `1024`（大模型） | 通信缓冲大小（MB） |
| `HCCL_OP_EXPANSION_MODE` | `"AIV"` | 算子展开模式 |
| `HCCL_RDMA_TIMEOUT` | `17` | RDMA 超时 |
| `HCCL_INTRA_PCIE_ENABLE` | `1` | PCIe 通信 |
| `HCCL_INTRA_ROCE_ENABLE` | `0` | RoCE 通信 |

#### Ascend ACL

| 环境变量 | 常用值 | 说明 |
|---|---|---|
| `ASCEND_RT_VISIBLE_DEVICES` | `0,1,2,3,4,5,6,7` | NPU 设备选择 |
| `ACL_OP_INIT_MODE` | `1` | 算子初始化模式 |
| `ASCEND_CONNECT_TIMEOUT` | `10000` | 连接超时 |
| `ASCEND_TRANSFER_TIMEOUT` | `10000` | 传输超时 |

#### vLLM Ascend 专用

| 环境变量 | 常用值 | 说明 |
|---|---|---|
| `VLLM_ASCEND_ENABLE_FLASHCOMM1` | `1` | 启用 FlashComm 优化 |
| `VLLM_ASCEND_APPLY_DSV4_PATCH` | `1` | 应用 DeepSeek V4 专用 patch |
| `VLLM_ASCEND_ENABLE_MLAPO` | `1` | MLAPO 优化 |
| `TASK_QUEUE_ENABLE` | `1` | 启用任务队列 |
| `VLLM_ALLREDUCE_USE_SYMM_MEM` | `0` | allreduce 对称内存 |
| `VLLM_USE_DEEP_GEMM` | `0` | DeepGEMM |
| `VLLM_LOGGING_LEVEL` | `INFO` | vLLM 日志级别 |

#### Ray 分布式

| 环境变量 | 常用值 | 说明 |
|---|---|---|
| `RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES` | `1` | Ray 不覆盖 NPU 设备设置 |
| `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES` | `1` | Ray 不覆盖 CUDA 设备设置 |

#### 设备选择

| 环境变量 | 常用值 | 说明 |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3,4,5,6,7` | GPU 设备选择 |
| `ASCEND_RT_VISIBLE_DEVICES` | `0,1,2,3,4,5,6,7` | NPU 设备选择 |

---

## 9. MindIE ucm_config.json

> `ucm/integration/mindie/ucm_config.json`

### ✏️ 需要手动填写

| 键 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `storage_backends` | list[str] | `["/path/to/kvcache"]` | 存储路径，**必填** |
| `mindie_config_path` | str | — | MindIE 服务配置路径，**必填** |
| `kvcs_ucm_over_tcp_ip_list` | str | `"127.0.0.1"` | UCM over TCP IP 列表，多机须改 |
| `kvcs_tls_enable` | bool | `false` | TLS 启用 |
| `kvcs_instance_name` | str | `"default_instance"` | KVCS 实例名 |
| `kvcs_failure_rate_threshold` | int | `10` | 失败率阈值 |
| `kvcs_consecutive_fail_limit` | int | `5` | 连续失败上限 |

### 🤖 自动/运行时填充（无需手动填，保持默认即可）

| 键 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `backend` | str | `"unifiedcache"` | 后端类型，固定值 |
| `block_elem_size` | int | `2` | 块元素大小，框架内部使用 |
| `kvcs_store_id` | int | `0` | KVCS store ID，单实例场景默认 0 |
| `kvcs_block_size` | int | `128` | KVCS 块大小，内部默认 |
| `kvcs_sliding_window_size` | int | `100` | 滑动窗口，内部默认 |


