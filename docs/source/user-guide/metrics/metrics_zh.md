# UCM Metrics 可观测性

UCM 默认通过 vLLM connector 上报指标，并复用 vLLM 的 Prometheus `/metrics` 端点对外暴露，不需要单独启动 exporter、选择额外导出模式或增加服务端口。

推荐使用 Prometheus 抓取 vLLM metrics，使用 Grafana 来可视化展示抓取到的数据。

UCM metrics 建议使用 **不短于 5 秒**的抓取和展示刷新周期，本文中的 Prometheus 和 Metrics-view 示例也都配置为 5 秒。设置更短的抓取周期通常不会让 UCM 指标更快更新。

UCM 指标的实际刷新还取决于 vLLM 的调用。UCM 会先在内部累计指标，vLLM 处理请求并调用 connector 的 `get_kv_connector_stats()` 接口后，新增数据才会同步到 vLLM 暴露的 Prometheus 指标中。**如果完全没有推理请求，vLLM 不会调用该接口，UCM 指标将不会更新**。

## Metrics workflow

下面以 DP=2、TP=1 为例。每个 DP 包含 1 个 worker Store 和 1 个 scheduler Store；两个 DP 分别处理各自的请求或 batch。

```{mermaid}
sequenceDiagram
    autonumber
    participant C as 客户端
    participant A as vLLM API Server
    participant S0 as DP 0 Scheduler
    participant W0 as DP 0 Worker
    participant S1 as DP 1 Scheduler
    participant W1 as DP 1 Worker
    participant M as vLLM /metrics Endpoint<br/>(位于 API Server 内)
    participant P as Prometheus

    C->>A: 发送推理请求
    par 请求或 batch 分配给 DP 0
        A->>S0: 调度 batch A
        S0->>W0: 执行模型及 UCM Lookup/Load/Save
        W0->>W0: 在 UCM 内部累计指标
        W0->>W0: vLLM 调用 get_kv_connector_stats() 获取 UCM 累计的指标
        W0-->>S0: 返回 worker 指标 (worker_rank=0)
        S0->>S0: 返回 scheduler 指标 (worker_rank=scheduler)
        S0-->>A: 上报 connector stats (engine=engine-0)
    and 请求或 batch 分配给 DP 1
        A->>S1: 调度 batch B
        S1->>W1: 执行模型及 UCM Lookup/Load/Save
        W1->>W1: 在 UCM 内部累计指标
        W1->>W1: vLLM 调用 get_kv_connector_stats() 获取 UCM 累计的指标
        W1-->>S1: 返回 worker 指标 (worker_rank=1)
        S1->>S1: 返回 scheduler 指标 (worker_rank=scheduler)
        S1-->>A: 上报 connector stats (engine=engine-1)
    end
    A->>M: 按指标类型更新 ucm:* series
    P->>M: GET /metrics
    M-->>P: 返回 vllm:* 和 ucm:* metrics
```

UCM 在发生 Lookup、Load、Save、健康探测等操作的进程中累计 Counter、Gauge 和 Histogram。vLLM 处理请求时，分别从 worker 和 scheduler 的 connector 获取 UCM 累计的指标；这些指标随各 DP 的 engine stats 返回，由 vLLM Prometheus metrics 按指标类型写入对应 series，并添加 `model_name`、`engine` 和 `worker_rank` 标签。

vLLM `/metrics` endpoint 和 Prometheus registry 位于 API Server 进程内。Prometheus 直接抓取该 endpoint；API Server 的 HTTP 路由只负责返回 registry 中已经同步的数据，不会调用 `get_kv_connector_stats()`。如果没有推理请求，UCM 仍可能在内部产生新数据，但这些数据要等到下一次 vLLM 请求触发同步后才会出现在 `/metrics` 中。

## 1. 启用与关闭 Metrics

### 1.1 使用内置默认配置

UCM metrics **默认启用**，不配置 `metrics_config_path` 时使用内置完整指标集。也可以显式配置：

```yaml
enable_metrics: true
```

关闭所有 UCM metrics：

```yaml
enable_metrics: false
```

### 1.2 使用自定义配置

如需限制指标集合或修改 Histogram bucket，可以在 UCM 配置顶层指定：

```yaml
enable_metrics: true
metrics_config_path: "/workspace/unified-cache-management/examples/metrics/metrics_configs.yaml"
```

一旦设置 `metrics_config_path`，该文件就是指标 enable-list；只有文件中定义的指标会被注册。

注意：如果配置了 metrics 文件，必须保证文件存在且 vLLM 进程可以读取，否则 UCM 指标不会显示。

## 2. 获取 Metrics

启动带 UCM connector 的 vLLM 并产生至少一次推理请求后，可以使用以下命令检查 UCM metrics 是否正常：

```bash
curl http://<vllm-ip>:<vllm-port>/metrics | grep '^ucm:'
```

注意：UCM 大部分指标需要产生对应路径的调用后才会出现。没有外存命中时，只有少数指标存在。

### 2.1 UCM 指标标签（tag）

UCM 默认通过 vLLM connector 导出的每条指标都带有以下标签：

| 标签 | 含义 | 示例 |
| --- | --- | --- |
| `model_name` | vLLM 对外提供服务的模型名，来自 vLLM model config | `Qwen3-32B` |
| `engine` | 产生指标的 vLLM engine 标识，用于区分同一服务中的不同 DP 实例 | `engine-0` |
| `worker_rank` | 产生原始指标的 UCM 进程，对应于一个 TP 实例；worker 使用 distributed rank，scheduler 使用字符串 `scheduler` | `0`、`1`、`scheduler` |

例如，vLLM `/metrics` 中的一条 UCM 指标可能为：

```text
ucm:cache_load_bytes_total{model_name="Qwen3-32B",engine="engine-0",worker_rank="0"} 1.048576e+08
```

Prometheus 抓取该端点后还会增加以下 target 标签：

| 标签 | 来源 | 含义 |
| --- | --- | --- |
| `job` | `prometheus.yml` 中的 `job_name` | 抓取任务名；本文示例中为 `vllm` |
| `instance` | Prometheus scrape target | 被抓取的 vLLM 地址和端口，例如 `10.0.0.8:8000` |

`job` 和 `instance` 由 Prometheus 添加，所以直接执行 `curl /metrics` 时通常看不到它们。Histogram 的 `_bucket` series 还会带有 `le` 标签，表示该 bucket 的上界；`le` 是 Prometheus Histogram 标签，不是 UCM 业务标签。

## 3. Prometheus 与 Grafana

要方便查看 UCM 各种指标，推荐使用 Prometheus + Grafana 的组合，Prometheus是时序数据库，会定时请求 vLLM metrics 地址，保存历史metrics数据，并提供查询能力。Grafana是展示面板，通过查询Prometheus数据库将指标数据转化为图表。

### 3.1 安装并配置 Prometheus

如果已有 Prometheus 抓取 vLLM 的 `/metrics`，则无需为 UCM 增加新的 scrape job，因为两类指标由同一个端点暴露。

如果没有安装 Prometheus，可以参考如下说明安装配置并抓取 vLLM metrics。

下面以 Docker 安装为例。其他安装方式参见 [Prometheus 官方安装文档](https://prometheus.io/docs/prometheus/latest/installation/)。

新建 `prometheus.yml`，配置 Prometheus 抓取 vLLM 服务的 `/metrics` 端点：

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

`<vllm-ip>:8000` 需要替换为 Prometheus 容器能够访问的 vLLM 地址和端口。不能填写容器自身的 `127.0.0.1:8000`；vLLM 位于宿主机时，可以填写宿主机实际 IP，Docker Desktop 也可以使用 `host.docker.internal:8000`。

创建监控容器使用的网络和 Prometheus 数据卷：

```bash
docker network create ucm-monitoring
docker volume create prometheus-data
```

在 `prometheus.yml` 所在目录启动 Prometheus：

```bash
docker run -d \
  --name prometheus \
  --restart unless-stopped \
  --network ucm-monitoring \
  -p 9090:9090 \
  -v "$(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml:ro" \
  -v prometheus-data:/prometheus \
  prom/prometheus
```

浏览器访问 `http://<prometheus-ip>:9090/targets`，确认 `vllm` target 状态为 **UP**。然后在 Prometheus 查询页面搜索 `vllm:` 和 `ucm:`，应能分别看到 vLLM 与 UCM 指标。

### 3.2 安装 Grafana

下面使用 Grafana 官方 Docker 镜像。其他安装方式参见 [Grafana 官方安装文档](https://grafana.com/docs/grafana/latest/setup-grafana/installation/)。

创建数据卷并启动 Grafana：

```bash
docker volume create grafana-data

docker run -d \
  --name grafana \
  --restart unless-stopped \
  --network ucm-monitoring \
  -p 3000:3000 \
  -v grafana-data:/var/lib/grafana \
  grafana/grafana
```

浏览器访问 `http://<grafana-ip>:3000`。首次登录使用用户名和密码 `admin`/`admin`，然后按提示修改密码。

### 3.3 添加 Prometheus Data Source

在 Grafana 中依次进入 **Connections** → **Add new connection**，搜索并选择 **Prometheus**，然后配置：

- Prometheus server URL：`http://prometheus:9090`；
- Authentication：本地无认证部署选择 No authentication；
- 点击 **Save & test**，确认 Grafana 可以查询 Prometheus。

这里可以直接使用主机名 `prometheus`，因为前面的两个容器都加入了 `ucm-monitoring` 网络。如果 Grafana 和 Prometheus 不是按上述方式部署，应填写 Grafana 服务能够访问的实际 Prometheus URL。Grafana 已内置 Prometheus data source，不需要额外安装插件。

### 3.4 导入 UCM Dashboard

进入 **Dashboards** → **New** → **Import**，上传所需的 dashboard JSON，选择前一步创建的 Prometheus data source，然后点击 **Import**。

UCM 提供以下 dashboard：

| 文件 | 用途 |
| --- | --- |
| `examples/metrics/grafana_vllm.json` | vLLM 请求延迟、token 吞吐、scheduler 和 cache 状态 |
| `examples/metrics/grafana_ucm_overview.json` | vLLM/UCM 总览、输入输出token数、Store 健康状态和探测趋势 |
| `examples/metrics/grafana_connector.json` | Connector Lookup/Load/Save 请求数、块数、耗时、速度和错误 |
| `examples/metrics/grafana_pipeline_store.json` | Cache、Posix、Mooncake 队列等待、传输、后端提交、带宽与瓶颈分析 |
| `examples/metrics/grafana_layerwise.json` | `use_layerwise=true` 时的逐层加载等待、提交和异步保存诊断 |

Dashboard 的 `job` 默认选择 All。UCM dashboard 还提供 Aggregated/Per Worker 视图和 `worker_rank` 过滤器。

聚合时遵循以下规则：

- Counter 先对每个 series 在相同时间窗口求 `rate()` 或 `increase()`，再跨 worker 求和；
- 比例使用聚合后的分子除以聚合后的分母，不能先计算各 worker 比例再做算术平均；
- Gauge 根据语义按 worker 展示，或使用 `min`/`max`；
- Histogram 跨 worker 聚合 `_bucket` 后再计算百分位。

## 4. Metrics-view

如果环境不方便安装 Prometheus/Grafana，或者没有图形化界面，可以使用 UCM toolkit中的命令行查看工具 Metrics-view ，将指标汇总输出到命令行中。

Metrics-view 可以直接检查一次 `/metrics` 快照，也可以在本机后台持续采集到 SQLite，再查询指定时间窗口，不依赖 Prometheus/Grafana。

安装 toolkit：

```bash
cd unified-cache-management
pip install -e toolkit
```

查看内置配置：

```bash
ucm-toolkit run metrics-view list-configs
```

### 4.1 检查当前快照

`check` 抓取一次当前 `/metrics`，展示从服务启动到现在的累计聚合结果：

```bash
ucm-toolkit run metrics-view check \
  --url http://127.0.0.1:8000/metrics \
  --config metrics_lite
```

GQA/MHA 模型使用默认参数。MLA 模型需要传入服务实际 TP，例如 TP=8：

```bash
ucm-toolkit run metrics-view check \
  --url http://127.0.0.1:8000/metrics \
  --config metrics_lite \
  --config-param tp_size=8
```

`check` 中的 GB/s 是累计数据量除以服务累计时间，不适合观察瞬时带宽；带宽分析建议使用后台采集。

### 4.2 后台采集与查询

启动后台采集时无需指定查询配置，可以重复传入多个 URL：

```bash
ucm-toolkit run metrics-view start \
  --url http://prefill:8000/metrics \
  --url http://decode:8000/metrics \
  --interval 5s
```

每个样本会增加 `url=<完整 metrics URL>` 标签，便于区分 PD 分离部署中的不同实例。单个 URL 抓取失败不会阻断其他 URL。

查看状态或停止采集：

```bash
ucm-toolkit run metrics-view status
ucm-toolkit run metrics-view stop
```

查询最近 10 分钟并按 1 分钟聚合；MLA 模型继续传入实际 TP：

```bash
ucm-toolkit run metrics-view query \
  --window 10m \
  --aggr-by 1m \
  --config metrics_lite \
  --config-param tp_size=8
```

使用 Prometheus 标签过滤：

```bash
ucm-toolkit run metrics-view query \
  --window 10m \
  --tag url=http://prefill:8000/metrics \
  --tag model_name=qwen
```

默认数据库、PID 和日志分别是 `/tmp/ucm_metrics.db`、`/tmp/ucm_metrics.pid` 和 `/tmp/terminal_metrics.log`。清空数据库：

```bash
ucm-toolkit run metrics-view clean
```

## 5. 常见问题

### 5.1 `/metrics` 中没有 `ucm:` 指标

依次检查：

1. `enable_metrics` 是否被设置为 false；
2. 自定义 `metrics_config_path` 是否存在、可读且包含目标指标；
3. vLLM 是否已产生经过对应 UCM 路径的请求；
4. `curl http://<vllm-ip>:<vllm-port>/metrics` 是否能够访问服务端点。

## 相关文档

- [UCM 健康指标](health_metrics_zh.md)：健康探测、熔断状态和聚合方法。
- [UCM metrics列表](metrics_list_zh.md)：完整指标清单。

```{toctree}
:maxdepth: 1
:hidden:

health_metrics_zh
metrics_list_zh
```
