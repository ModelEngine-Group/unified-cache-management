# DramPool

[English](../../../en/developer-guide/cache-configuration/drampool.md) | [中文](drampool.md)

DramPool 将主机 DRAM 组织为可供多个推理实例共享的 KV 缓存池，推理实例通过 DramStore 接入 DramPool。目前仅支持基于 HIXL 的数据传输。

## 快速开始

### 当前环境与准备要求

- 设备：昇腾 A3 及以上。
- CANN：9.1.0 及以上。
- 驱动与通信库：与设备及 CANN 版本兼容的昇腾驱动和 HIXL 库。
- 操作系统：Linux。
- 网络：DramPool 与推理节点之间需具备 RDMA 网络。
- 主机内存：可用容量需满足 KV 缓存池及运行时缓冲区的内存需求。

### 编译

在 UCM 仓库根目录加载 CANN 环境，并按 A3 平台编译 UCM。此命令会同时编译并安装 DramStore 和 DramPool：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PLATFORM=ascend-a3
pip install -v -e . --no-build-isolation
```

使用上述 editable 安装方式时，产物位于：

- `ucm/store/dram/libdramstore.so`：DramStore 客户端库，选择 `store_pipeline: "Dram"` 时由 `UcmPipelineStore` 加载。
- `bin/drampool`：独立运行的 DramPool 服务端程序。

### 配置

#### DramPool 配置

DramPool 的主要配置项如下。

| 配置项 | 说明 |
| --- | --- |
| `transport.device_ids` | 用于数据传输的设备 ID 列表。 |
| `transport.hixl.listen_port` | HIXL 通信监听端口。 |
| `transport.hixl.enable_hixl_cs` | 是否启用 HIXL CS。 |
| `health.port` | 健康检查服务端口，设置为 `0` 时关闭。 |
| `logger.dir` | 日志输出目录。 |

完整配置示例参见 [drampool.yaml](https://github.com/ModelEngine-Group/unified-cache-management/blob/develop/examples/drampool.yaml)。

#### DramStore 配置

DramStore 的主要配置项如下。

| 配置项 | 说明 |
| --- | --- |
| `ucm_connector_name` | UCM connector 名称，DramStore 使用 `UcmPipelineStore`。 |
| `store_pipeline` | 存储管线，`Dram` 表示使用 DramStore。 |
| `local_addr` | 必填，DramStore 本地服务 IP 和基准端口，格式为 `IP:port`。实际端口根据物理设备 ID 和 scheduler/worker 角色偏移，各实例应预留互不重叠的端口范围。 |
| `peer_addrs` | 必填，非空的 DramPool 服务地址列表，与各 DramPool 实例的 `--addr` 一致。 |
| `hixl_listen_port` | 本地 HIXL 基准端口，默认 `36666`。实际端口根据物理设备 ID 和 scheduler/worker 角色偏移，各实例应预留互不重叠的端口范围。 |
| `enable_hixl_cs` | 是否启用 HIXL CS，默认 `false`。 |
| `router_type` | 路由策略，支持 `ring_hash`、`maglev`、`contiguous_block_affinity` 和 `batch_topk_affinity`，默认 `ring_hash`。 |
| `enable_event_sync` | 是否启用计算流事件同步，默认 `true`。启用后，远程访问 KV 内存前等待计算流事件。 |
| `use_layerwise` | 是否启用逐层 KV 缓存传输，默认 `true`。 |

完整配置示例参见 [ucm_config_dramstore.yaml](https://github.com/ModelEngine-Group/unified-cache-management/blob/develop/examples/ucm_config_dramstore.yaml)。

### 启动

#### 启动 DramPool

在 UCM 仓库根目录执行以下命令启动 DramPool：

```bash
./bin/drampool \
  --addr 192.168.1.10:9000 \
  --pool-size-gb 64 \
  --kvcache-block-sizes 262144 \
  --config /absolute/path/to/drampool.yaml
```

启动参数说明如下。

| 参数 | 说明 |
| --- | --- |
| `--addr` | DramPool 控制服务地址，需对所有 DramStore 客户端可达。 |
| `--pool-size-gb` | KV 缓存池容量，单位为 GiB。主机内存需为操作系统和运行时缓冲区预留容量。 |
| `--kvcache-block-sizes` | 支持的 KV 缓存块大小，单位为字节，需与 DramStore 传输的张量大小一致。多个值以空格分隔。该参数与 vLLM 中以 token 为单位的 `block_size` 不同。 |
| `--kvcache-block-proportions` | 各块大小对应的容量权重，默认等分。权重数量需与块大小数量一致。 |
| `--ttl-minutes` | 缓存块的绝对有效期，单位为分钟，默认 `120`。 |
| `--config` | 运行配置文件路径，默认 `examples/drampool.yaml`，相对于当前工作目录解析。 |

服务启动完成后，日志输出以下信息：

```text
DramPool service ready
```

启用健康检查（`health.port` 非零）后，可通过以下命令检查服务状态，IP 与 `--addr` 一致，端口与 `health.port` 一致：

```bash
curl --fail http://192.168.1.10:9100/health
```

停止服务时，向 DramPool 进程发送 `SIGINT` 或 `SIGTERM` 信号。

#### 启动推理服务

DramPool 就绪后，执行以下命令启动单卡推理服务：

```bash
vllm serve <model-path> \
  --tensor-parallel-size 1 \
  --block-size 128 \
  --kv-transfer-config '{"kv_connector":"UCMConnector","kv_role":"kv_both","kv_connector_module_path":"ucm.integration.vllm.ucm_connector","kv_connector_extra_config":{"UCM_CONFIG_FILE":"/etc/ucm/dramstore.yaml"}}'
```

模型及推理参数说明参见 [vLLM 启动参数文档](https://docs.vllm.ai/en/latest/cli/serve/)。`--kv-transfer-config` 中的 UCM 接入配置如下。

| 配置项 | 说明 |
| --- | --- |
| `kv_connector` | UCM connector 名称，使用 `UCMConnector`。 |
| `kv_role` | KV 缓存传输角色，`kv_both` 表示支持缓存 load 和 dump。 |
| `kv_connector_module_path` | UCM connector 模块路径，使用 `ucm.integration.vllm.ucm_connector`。 |
| `kv_connector_extra_config.UCM_CONFIG_FILE` | DramStore 配置文件路径。 |

## Metrics

DramStore 指标和 DramPool 指标通过 vLLM 的 `/metrics` 接口导出。

### 启用指标

UCM 指标默认启用，也可在 DramStore 配置文件中显式配置：

```yaml
enable_metrics: true
```

#### DramPool 配置

在 `drampool.yaml` 中配置：

```yaml
metrics:
  enabled: true
  output_dir: ./logs
  interval_ms: 10000
```

| 配置项 | 说明 |
| --- | --- |
| `metrics.enabled` | 是否启用指标快照导出。 |
| `metrics.output_dir` | 快照输出目录，为空时使用 `logger.dir`。上述配置对应的快照文件为 `./logs/drampool_metrics.json`。 |
| `metrics.interval_ms` | 快照导出周期，单位为毫秒，启用时需大于 `0`。 |

#### DramStore 配置

在 `ucm_config_dramstore.yaml` 中配置：

```yaml
drampool_resource_log_path: "./logs/drampool_metrics.json"
drampool_resource_metrics_enable: true
drampool_resource_metrics_interval_sec: 15
```

| 配置项 | 说明 |
| --- | --- |
| `drampool_resource_log_path` | DramPool 指标快照文件路径，相对路径基于进程工作目录解析。跨主机或容器部署时，需共享或挂载快照目录。 |
| `drampool_resource_metrics_enable` | 是否启用快照读取，配置非空路径后默认启用。 |
| `drampool_resource_metrics_interval_sec` | 快照读取周期，单位为秒，默认 `15`。 |

### 访问指标

启动推理服务并发送推理请求后，执行以下命令查看指标：

```bash
curl http://<vllm-ip>:<vllm-port>/metrics \
  | grep -E '^ucm:(dramstore_|drampool_)'
```

`dramstore_` 和 `drampool_` 分别为 DramStore 指标和导入的 DramPool 指标，均通过 vLLM 的 `/metrics` 接口导出，更新依赖 vLLM 收集 connector 统计，无推理请求时可能不刷新；DramPool 不提供独立的 HTTP `/metrics` 接口，`health.port` 仅用于健康检查。

### 指标说明

#### DramStore 指标

| 指标 | 类型 | 含义与观测内容 |
| --- | --- | --- |
| `dramstore_lookup_duration_ms` | Histogram | 缓存查找耗时，单位为毫秒，观察查找延迟及波动。 |
| `dramstore_dump_duration_ms` | Histogram | KV 缓存 dump 耗时，单位为毫秒，观察 dump 延迟。 |
| `dramstore_load_duration_ms` | Histogram | KV 缓存 load 耗时，单位为毫秒，观察 load 延迟。 |
| `dramstore_lookup_tasks_failed_total` | Counter | 缓存查找任务失败数，观察查找异常的发生频率。 |
| `dramstore_dump_tasks_failed_total` | Counter | 缓存 dump 任务失败数，观察 dump 异常的发生频率。 |
| `dramstore_load_tasks_failed_total` | Counter | 缓存 load 任务失败数，观察 load 异常的发生频率。 |
| `dramstore_dump_task_timeouts_total` | Counter | 缓存 dump 任务超时数，观察 dump 超时的发生频率。 |
| `dramstore_load_task_timeouts_total` | Counter | 缓存 load 任务超时数，观察 load 超时的发生频率。 |
| `dramstore_task_queue_size` | Gauge | 等待处理的任务数，观察任务积压情况。 |
| `dramstore_connect_failures_total` | Counter | DramPool 连接失败次数，观察服务连接异常。 |

#### DramPool 指标

| 指标 | 类型 | 含义与观测内容 |
| --- | --- | --- |
| `drampool_buffer_pool_usage_ratio` | Gauge | 缓存池使用比例，按缓存块大小区分，观察各规格缓存池的容量压力。 |
| `drampool_metadata_entry_count` | Gauge | 缓存条目总数，观察缓存规模变化。 |
| `drampool_dump_nospace_failures_total` | Counter | 因缓存空间不足而 dump 失败的条目数，观察缓存容量不足的情况。 |
| `drampool_lookup_miss_entries_total` | Counter | 缓存查找未命中的条目数，观察缓存未命中情况。 |
| `drampool_transfer_failures_total` | Counter | 缓存 dump 和 load 过程中的数据传输失败次数，观察传输异常。 |
| `drampool_response_failures_total` | Counter | 向 DramStore 返回响应失败的次数，观察服务响应异常。 |
| `drampool_queue_request_size` | Gauge | 等待处理的请求数，观察请求积压情况。 |
| `drampool_lookup_batch_total_duration_ms` | Histogram | 服务端每批缓存查找请求的处理耗时，单位为毫秒，观察查找延迟。 |
| `drampool_dump_batch_total_duration_ms` | Histogram | 服务端每批缓存 dump 请求的处理耗时，单位为毫秒，观察 dump 延迟。 |
| `drampool_load_batch_total_duration_ms` | Histogram | 服务端每批缓存 load 请求的处理耗时，单位为毫秒，观察 load 延迟。 |

### Prometheus 和 Grafana 集成

Prometheus 通过 vLLM 的 `/metrics` 接口采集 DramStore 和 DramPool 指标。部署步骤参见[接入指标监控](https://ucm.readthedocs.io/zh-cn/latest/user-guide/observability/metrics/)。

在 Grafana 中导入 [Dram 仪表板](https://github.com/ModelEngine-Group/unified-cache-management/blob/develop/examples/metrics/grafana_dram.json)，并选择对应的 Prometheus 数据源。

## 性能

### 测试环境

- 测试模型：DeepSeek-V3.1-Terminus-w8a8。
- 硬件环境：单机昇腾 A3。
- 测试工具：AISBench。

### 测试部署参数

#### 基线推理服务

基线推理服务部署参数如下。

```bash
vllm serve <model-path> \
  --port <port> \
  --tensor-parallel-size 4 \
  --data-parallel-size 4 \
  --quantization ascend \
  --seed 1024 \
  --enable-expert-parallel \
  --max-num-seqs 16 \
  --max-model-len 130000 \
  --max-num-batched-tokens 2048 \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.95 \
  --speculative-config '{"num_speculative_tokens":3, "method":"mtp"}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
```

#### DramPool 测试推理服务

推理参数与基线一致，仅增加通过 `--kv-transfer-config` 加载 DramStore 配置文件的参数。

```bash
vllm serve <model-path> \
  --port <port> \
  --tensor-parallel-size 4 \
  --data-parallel-size 4 \
  --quantization ascend \
  --seed 1024 \
  --enable-expert-parallel \
  --max-num-seqs 16 \
  --max-model-len 130000 \
  --max-num-batched-tokens 2048 \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.95 \
  --speculative-config '{"num_speculative_tokens":3, "method":"mtp"}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --kv-transfer-config '{"kv_connector":"UCMConnector","kv_role":"kv_both","kv_connector_module_path":"ucm.integration.vllm.ucm_connector","kv_connector_extra_config":{"UCM_CONFIG_FILE":"<dramstore-config-path>"}}'
```

#### DramPool 启动参数

DramPool 性能测试启动参数如下。

```bash
./bin/drampool \
  --addr <ip:port> \
  --pool-size-gb 64 \
  --kvcache-block-sizes 131072 16384 \
  --config <drampool-config-path>
```

#### DramStore 测试配置

DramStore 性能测试配置如下。

```yaml
ucm_connectors:
  - ucm_connector_name: "UcmPipelineStore"
    ucm_connector_config:
      store_pipeline: "Dram"
      local_addr: "127.0.0.1:4502"
      peer_addrs: ["127.0.0.1:4501"]
      hixl_listen_port: 36666
      enable_hixl_cs: false
      manager_max_threads: 4
      max_io_entries: 10000
      lookup_timeout_ms: 1000
      dump_timeout_ms: 3000
      load_timeout_ms: 3000
      reconnect_interval_ms: 3000
      router_type: "ring_hash"
      transport_worker_count: 1
      node_runner_count: 1
enable_event_sync: true
use_layerwise: true
```

#### DramPool 测试配置

DramPool 性能测试配置如下。

```yaml
transport:
  device_ids: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
  manager_max_threads: 4
  hixl:
    listen_port: 26666
    enable_hixl_cs: false
queue:
  request_depth: 65536
  completion_depth: 65536
request_receiver:
  idle_wait_us: 100
poller:
  pending_depth: 64
flag_buffer:
  capacity_mb: 64
  slot_size_bytes: 128
gc:
  enabled: true
  interval_ms: 1000
metadata:
  periodic_eviction_policy: TTL
  deep_eviction_policy: POSITION
  lease_time_ms: 5000
  default_evict_ratio: 0.0
  evict_period_ms: 31536000000
operation:
  timeout_ms: 5000
logger:
  dir: ./logs
  max_files: 10
  max_size_mb: 5
```

### 测试说明

以下为 UCM 在前缀缓存场景下使用 DramPool 的性能测试结果。输入长度为 4,000、8,000、16,000 和 32,000 tokens，最大并发数为 1、4、16 和 32，平均输出长度为 1,000 tokens。下表展示基线、hit - 50 和 hit - 90 场景下的 TTFT（首 token 延迟）及加速比。TTFT 加速比按 **（基线 TTFT / 对应场景 TTFT − 1）× 100%** 计算，结果保留一位小数。

| **输入长度** | **请求数** | **最大并发数** | **基线 TTFT** | **hit - 50 TTFT** | **TTFT 加速 (%)** | **hit - 90 TTFT** | **TTFT 加速 (%)** |
| -----------: | ---------: | -------------: | ------------: | ----------------: | ----------------: | ----------------: | ----------------: |
|            4 000 |            4 |              1 |             893.8 |             702.5 |           **+27.2%** |             639.2 |           **+39.8%** |
|            4 000 |           16 |              4 |            1092.5 |            1020.0 |            **+7.1%** |             922.1 |           **+18.5%** |
|            4 000 |           64 |             16 |            1789.4 |            1731.8 |            **+3.3%** |            1697.7 |            **+5.4%** |
|            4 000 |          128 |             32 |            2491.3 |            2076.5 |           **+20.0%** |            2151.3 |           **+15.8%** |
|            8 000 |            4 |              1 |            1716.2 |            1111.7 |           **+54.4%** |             671.7 |          **+155.5%** |
|            8 000 |           16 |              4 |            2105.9 |            1707.8 |           **+23.3%** |             986.0 |          **+113.6%** |
|            8 000 |           64 |             16 |            3919.6 |            2774.0 |           **+41.3%** |            1768.6 |          **+121.6%** |
|            8 000 |          128 |             32 |            5939.1 |            3243.9 |           **+83.1%** |            2383.2 |          **+149.2%** |
|           16 000 |            4 |              1 |            3949.0 |            2321.5 |           **+70.1%** |             801.9 |          **+392.5%** |
|           16 000 |           16 |              4 |            5422.5 |            3318.2 |           **+63.4%** |            1242.9 |          **+336.3%** |
|           16 000 |           64 |             16 |            8579.4 |            4977.5 |           **+72.4%** |            2264.1 |          **+278.9%** |
|           16 000 |          128 |             32 |           65085.5 |           29686.6 |          **+119.2%** |           18672.3 |          **+248.6%** |
|           32 000 |            4 |              1 |            9781.7 |            6070.6 |           **+61.1%** |            2055.6 |          **+375.9%** |
|           32 000 |           16 |              4 |           12574.2 |            8200.6 |           **+53.3%** |            2686.5 |          **+368.1%** |
|           32 000 |           64 |             16 |           46070.9 |           37792.1 |           **+21.9%** |           22434.8 |          **+105.4%** |
|           32 000 |          128 |             32 |          178751.1 |          149170.5 |           **+19.8%** |           96544.9 |           **+85.1%** |
