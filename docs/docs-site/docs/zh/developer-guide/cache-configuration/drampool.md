# DramPool

[English](../../../en/developer-guide/cache-configuration/drampool.md) | [中文](../../../zh/developer-guide/cache-configuration/drampool.md)

DramPool 将主机 DRAM 组成 KV 缓存池，供多个推理实例复用。DramStore 是 UCM 的客户端，通过 HIXL 读写池中的 KV 块。缓存块设有过期时间，重启 DramPool 会清空内存缓存。

## 环境要求

- 昇腾 A3 及以上硬件，**CANN 9.1.0 及以上**，以及兼容的驱动和 HIXL 库。
- Linux 环境，DramPool 与推理节点之间具备 RDMA 网络，主机可用内存足以容纳配置的缓存池和运行时缓冲区。
- 已安装 UCM 和兼容的 vLLM-Ascend。推理侧集成步骤参见 [Quickstart-vLLM-Ascend](../../user-guide/quick_start/index.md#vllm-ascend)。

## 启动 DramPool

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

两者均包含在 UCM 的昇腾构建中，无需分别执行编译命令。推理节点随 UCM 部署客户端库，池端节点运行服务端程序。

### 配置与启动

复制 [examples/drampool.yaml](https://github.com/ModelEngine-Group/unified-cache-management/blob/develop/examples/drampool.yaml) 作为运行配置。将 `transport.device_ids` 设置为可用设备 ID，服务地址通过 `--addr` 指定，池端无需配置客户端地址列表。

例如，DramPool 对应的配置为：

```yaml
transport:
  device_ids: [0]
  manager_max_threads: 4
  hixl:
    listen_port: 26666
    enable_hixl_cs: false
health:
  port: 9100
```

将上述设置合并到复制的配置文件。editable 编译会将程序安装到 `bin/`，从仓库根目录运行。

启动池端前先确定支持的张量大小。DramPool 须支持 DramStore 的 `tensor_size_list` 中每种不同的字节大小，而非这些大小的总和。标准 GQA 且 K/V 张量分开存储时，每个大小为“每块 token 数 × 每个 worker 的 KV head 数 × head 维度 × 每个元素字节数”。Qwen3-0.6B 在 TP=1、block size=128、BF16 下为 `128 × 8 × 128 × 2 = 262144` 字节，对应下方命令。其它模型和布局应使用实际张量大小，可在 worker 的 `create UcmPipelineStore with config` 日志中核对。

```bash
./bin/drampool \
  --addr 192.168.1.10:9000 \
  --pool-size-gb 64 \
  --kvcache-block-sizes 262144 \
  --config /absolute/path/to/drampool.yaml
```

按实际部署替换 IP、容量和块大小，并确保 DramPool 与 DramStore 服务地址及 HIXL 通信端口之间的网络连通。

| 参数 | 说明 |
| --- | --- |
| `--addr` | DramPool 控制地址，所有 DramStore 客户端均需能够访问。 |
| `--pool-size-gb` | 当前进程贡献的主机内存容量，单位为 GiB；需为操作系统和运行时缓冲区预留内存。 |
| `--kvcache-block-sizes` | 支持的 KV 块大小，单位为**字节**，须与 DramStore 发送的块匹配；与 vLLM 按 token 计的 `block_size` 含义不同。多个值以空格分隔。 |
| `--kvcache-block-proportions` | 可选，各块大小对应的容量权重；默认等分，每种块大小需对应一个权重。 |
| `--ttl-minutes` | 可选，缓存块的绝对有效期，默认 `120` 分钟。 |
| `--config` | 运行配置路径；默认 `examples/drampool.yaml`，相对于当前工作目录。 |

日志出现 `DramPool service ready` 表示服务已就绪。启用 `health.port` 后可检查：

```bash
curl --fail http://192.168.1.10:9100/health
```

向 DramPool 发送 `SIGINT` 或 `SIGTERM` 可正常退出。

### 配置 DramStore 并启动推理

复制 [examples/ucm_config_dramstore.yaml](https://github.com/ModelEngine-Group/unified-cache-management/blob/develop/examples/ucm_config_dramstore.yaml) 作为推理侧配置，例如 `/etc/ucm/dramstore.yaml`，按实际部署设置本地服务地址和 DramPool 地址：

```yaml
ucm_connectors:
  - ucm_connector_name: "UcmPipelineStore"
    ucm_connector_config:
      store_pipeline: "Dram"
      local_addr: "192.168.1.20:9001"
      peer_addrs: ["192.168.1.10:9000"]
      hixl_listen_port: 36666
      enable_hixl_cs: false
      router_type: "ring_hash"

enable_event_sync: true
use_layerwise: true
```

| 参数 | 说明 |
| --- | --- |
| `local_addr` | DramStore 本地服务 IP 和基准端口；使用推理节点 IP，同一节点的每个推理实例应使用不同的基准端口。 |
| `peer_addrs` | DramPool 服务地址，与各池端的 `--addr` 一致；可列出多个池端。 |
| `hixl_listen_port` | 本地 HIXL 基准端口，默认 `36666`。实际端口根据物理设备 ID 和 scheduler/worker 角色偏移，各实例应预留互不重叠的端口范围。 |
| `enable_hixl_cs` | 是否启用 HIXL CS，默认 `false`。 |
| `router_type` | 路由策略，默认 `ring_hash`。 |

`device_id`、`tensor_size_list` 和 KV 缓冲区地址由 connector 在运行时提供，无需手工填写。保持 `enable_event_sync: true`，使 dump 在远程访问 KV 内存前等待计算流事件。

DramPool 就绪且完成 vLLM-Ascend 集成后，可启动单卡推理实例：

```bash
vllm serve <model-path> \
  --tensor-parallel-size 1 \
  --block-size 128 \
  --kv-transfer-config '{"kv_connector":"UCMConnector","kv_role":"kv_both","kv_connector_module_path":"ucm.integration.vllm.ucm_connector","kv_connector_extra_config":{"UCM_CONFIG_FILE":"/etc/ucm/dramstore.yaml"}}'
```

按实际部署替换模型和配置路径，并调整推理参数。

两端部署在同一主机且 DramPool 仅使用一张 NPU 时，应为池端和推理进程选择不同的物理设备。例如，池端设置 `device_ids: [0]`，推理侧设置 `ASCEND_RT_VISIBLE_DEVICES=1`。

验证外部缓存复用时，先发送覆盖多个完整块的 prompt，确认 DramStore 的 DUMP 任务成功。保持 DramPool 运行，以相同配置重启推理进程并重复该 prompt，确认外部命中及 LOAD 任务成功，详见[验证外部缓存](../../user-guide/observability/verify-cache.md)。仅收到正常推理响应不足以证明缓存复用成功。

### Metrics

DramStore 操作指标和 DramPool 指标通过 vLLM 的 `/metrics` 暴露。按以下配置开启 DramPool 快照导出，并在 DramStore 侧读取快照。

**DramPool 端**

按实际部署修改已复制的 `examples/drampool.yaml` 中的 `metrics` 配置：

```yaml
metrics:
  enabled: true
  output_dir: /var/log/ucm/drampool
  interval_ms: 10000
```

服务每 10 秒原子替换 `output_dir` 下的 `drampool_metrics.json`。设置 `metrics.enabled: false` 可关闭快照采集和导出。示例 YAML 已开启导出，省略 `metrics.enabled` 时默认为 `false`。`output_dir` 为空时使用 `logger.dir`，开启时 `interval_ms` 必须大于 0。

**DramStore 端**

将以下配置合并到推理侧 `Dram` 管线的 `ucm_connector_config` 中：

```yaml
ucm_connectors:
  - ucm_connector_name: "UcmPipelineStore"
    ucm_connector_config:
      store_pipeline: "Dram"
      # 保留其它 DramStore 连接配置。
      drampool_resource_metrics_enable: true
      drampool_resource_log_path: /var/log/ucm/drampool/drampool_metrics.json
      drampool_resource_metrics_interval_sec: 15
```

这三个参数控制客户端读取 DramPool 快照，不控制 DramStore 自身的操作指标。配置非空路径后默认启用读取，设置 `drampool_resource_metrics_enable: false` 可关闭；读取周期默认 15 秒，仅在 scheduler 进程（`device_id < 0`）运行。快照须对该进程可见且可读；池端与推理端位于不同容器或主机时，需将输出目录挂载到读取端，路径以推理进程的文件系统为准。

发送推理请求后，通过 vLLM 检查 DramStore 和导入的 DramPool 指标：

```bash
curl http://<vllm-ip>:<vllm-port>/metrics | grep -E '^ucm:(dramstore_|drampool_)'
```

DramPool 导出文件快照，不提供 HTTP `/metrics`，`health.port` 仅提供 `/health`。可使用 [Dram 大盘](https://github.com/ModelEngine-Group/unified-cache-management/blob/develop/examples/metrics/grafana_dram.json) 查看两端指标。vLLM 在推理过程中收集 connector 统计时，指标才会更新；Prometheus 和 Grafana 配置参见 [UCM Metrics Observability](../../user-guide/observability/metrics.md)。

## 性能

### 测试说明

以下为 DramPool 在前缀缓存复用场景下的 TTFT（首 token 延迟）测试结果。输入长度分别为 4,000、8,000、16,000 和 32,000 tokens，最大并发数分别为 1、4、16 和 32，平均输出长度为 1,000 tokens。表格对比了“基线”、“hit - 50”和“hit - 90”三个场景的性能。

TTFT 加速百分比按 **（基线 TTFT / 对应场景 TTFT − 1）× 100%** 计算，保留一位小数。该值表示 TTFT 倒数的提升，不是延迟下降百分比。

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

在这些测试中，`hit - 50` 的 TTFT 加速为 **3.3%–119.2%**，`hit - 90` 为 **5.4%–392.5%**。输入长度为 4,000、并发数为 32 时，`hit - 90` 的 TTFT 高于 `hit - 50`。
