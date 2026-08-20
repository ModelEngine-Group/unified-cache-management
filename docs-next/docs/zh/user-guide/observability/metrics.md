# UCM 观测能力

UCM 通过 vLLM connector 导出指标，并重用 vLLM 的 Prometheus `/metrics` 端点，不需要单独的导出器、导出模式或服务端口。

建议使用 Prometheus 抓取 vLLM 指标，使用 Grafana 可视化收集的数据。

使用**至少 5 秒**的抓取和仪表板刷新间隔来处理 UCM 指标。本指南中的 Prometheus 和 Metrics-view 示例都使用 5 秒。更短的间隔通常不会使 UCM 指标更新更快。

实际刷新频率还取决于 vLLM。UCM 首先在内部累积指标。新数据仅在 vLLM 处理请求并调用 connector 的 `get_kv_connector_stats()` 方法后才同步到 vLLM 暴露的 Prometheus 指标。**当没有推理请求时，vLLM 不会调用此方法，UCM 指标不会更新。**

## 启用或禁用指标

### 使用内置配置

UCM 指标**默认启用**。当省略 `metrics_config_path` 时，UCM 使用完整的内置指标集。也可以显式启用指标：

```yaml
enable_metrics: true
```

要禁用所有 UCM 指标：

```yaml
enable_metrics: false
```

### 使用自定义配置

要限制导出的指标集或自定义 Histogram 桶，设置以下顶层 UCM 选项：

```yaml
enable_metrics: true
metrics_config_path: "/workspace/unified-cache-management/examples/metrics/metrics_configs.yaml"
```

一旦设置了 `metrics_config_path`，该文件就成为了指标启用列表。只有该文件中定义的指标才会被注册。

指标文件必须存在且可被 vLLM 进程读取。否则，UCM 指标将不会暴露。

## 访问指标

使用 UCM connector 启动 vLLM 并发送至少一个推理请求。然后验证 UCM 指标是否可用：

```bash
curl http://<vllm-ip>:<vllm-port>/metrics | grep '^ucm:'
```

大多数 UCM 指标只有在相应代码路径运行后才会出现。当没有外部存储命中时，可能只存在一小部分指标。

### UCM 指标标签

通过 vLLM connector 导出的每个指标都带有这些标签：

| 标签 | 含义 | 示例 |
| --- | --- | --- |
| `model_name` | vLLM 提供的模型名称，取自 vLLM 模型配置 | `Qwen3-32B` |
| `engine` | 产生指标的 vLLM 引擎；区分同一服务中的 DP 实例 | `engine-0` |
| `worker_rank` | 产生指标的 UCM 进程，对应 TP 实例；worker 使用其分布式 rank，scheduler 使用 `scheduler` | `0`, `1`, `scheduler` |

例如：

```text
ucm:cache_load_bytes_total{model_name="Qwen3-32B",engine="engine-0",worker_rank="0"} 1.048576e+08
```

## Prometheus 和 Grafana 集成

Prometheus 和 Grafana 是观察 UCM 的推荐组合。Prometheus 定期抓取 vLLM 指标端点，存储历史时间序列数据，并提供查询接口。Grafana 查询 Prometheus 并将指标显示为仪表板。

### 配置 Prometheus

如果 Prometheus 已经在抓取 vLLM 的 `/metrics` 端点，则不需要为 UCM 添加额外的抓取任务，因为 vLLM 和 UCM 指标都通过同一端点暴露。

创建 `prometheus.yml` 并配置 Prometheus 抓取 vLLM 服务：

```yaml
global:
  scrape_interval: 5s
  evaluation_interval: 30s

scrape_configs:
  - job_name: vllm
    metrics_path: /metrics
    static_configs:
      - targets:
          - "<vllm-ip>:8000"
```

### 安装 Grafana

创建持久卷并启动 Grafana：

```bash
docker volume create grafana-data

docker run -d \
  --name grafana \
  --restart unless-stopped \
  -p 3000:3000 \
  -v grafana-data:/var/lib/grafana \
  grafana/grafana
```

打开 `http://<grafana-ip>:3000`。首次登录时，用户名和密码都使用 `admin`，然后在提示时更改密码。

### 添加 Prometheus 数据源

在 Grafana 中，进入**连接** → **添加新连接**，搜索 **Prometheus**，并配置：

- Prometheus 服务器 URL：`http://prometheus:9090`
- 身份验证：对于未认证的本地部署选择**无身份验证**
- 选择**保存并测试**并验证 Grafana 可以查询 Prometheus

### 导入 UCM 仪表板

进入**仪表板** → **新建** → **导入**，上传所需的仪表板 JSON 文件，选择 Prometheus 数据源，然后点击**导入**。

UCM 提供这些仪表板：

| 文件 | 用途 |
| --- | --- |
| `examples/metrics/grafana_vllm.json` | vLLM 请求延迟、令牌吞吐量、调度器状态和缓存状态 |
| `examples/metrics/grafana_ucm_overview.json` | vLLM/UCM 概览、输入和输出令牌计数、Store 健康状态和探测趋势 |
| `examples/metrics/grafana_connector.json` | Connector 查找/加载/保存请求计数、块计数、持续时间、吞吐量和错误 |

有关完整的指标文档，请参阅原始 UCM 文档中的详细指标指南。