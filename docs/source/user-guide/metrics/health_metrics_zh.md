# UCM 健康指标

UCM Pipeline Store 可以为各层 Store 启用健康探测和熔断。健康指标用于回答两个不同的问题：

- 一段时间内健康探测的成功和失败次数；
- 当前 Store 是否已被熔断器禁止承接新请求。

这两类信息分别由 Counter 和 Gauge 表示，不能使用相同的方式聚合。本文介绍 Posix Store 和 Mooncake Store 的探测方式、指标语义及推荐的 PromQL 聚合方法。

## 1. 探测内容

每个 UCM connector 实例都会为支持健康检查的 Store 独立探测并维护熔断状态。worker Store 使用数值 `worker_rank`，scheduler Store 使用字符串 `worker_rank="scheduler"`。scheduler 虽然不是 distributed rank，但它持有自己的 Store，因此应计入 Store 数量。

当前只有以下远端 Store 实现了健康检查和熔断机制，其他 Store 不受影响。

### 1.1 Posix Store

Posix 健康探测会在每个健康检查路径上执行一次完整的小文件 I/O：创建文件、写入 4 KiB 测试数据、必要时执行同步、读取并校验数据，最后删除文件。任一路径的打开、读写、同步、校验或清理失败，都会使本次探测失败。

因此该探测验证的是实际 I/O 路径，不只是目录是否存在。对于 NFS 等远端文件系统，它也会反映挂载、网络和远端存储异常。

### 1.2 Mooncake Store

Mooncake 健康探测会使用独立的临时 key 执行一次小数据的 Put、Get、内容校验和 Remove。客户端不可用、任一步骤失败或读取内容不一致，都会使本次探测失败。

## 2. 健康指标

默认配置包含以下六个健康指标：

| 指标 | 类型 | 含义 | 更新时机 |
| --- | --- | --- | --- |
| `ucm:posix_healthy_count_total` | Counter | Posix 健康探测成功次数 | Posix 探测成功后加 1 |
| `ucm:posix_unhealthy_count_total` | Counter | Posix 健康探测失败或超时次数 | Posix 探测失败后加 1 |
| `ucm:posix_store_health` | Gauge | Posix 熔断器有效状态：1 可用，0 已熔断 | 启动时及每次 Posix 探测后更新 |
| `ucm:mooncake_healthy_count_total` | Counter | Mooncake 健康探测成功次数 | Mooncake 探测成功后加 1 |
| `ucm:mooncake_unhealthy_count_total` | Counter | Mooncake 健康探测失败或超时次数 | Mooncake 探测失败后加 1 |
| `ucm:mooncake_store_health` | Gauge | Mooncake 熔断器有效状态：1 可用，0 已熔断 | 启动时及每次 Mooncake 探测后更新 |

判断当前是否熔断应使用 Gauge；分析一段时间内的探测质量应同时使用成功和失败 Counter。当前没有单独的熔断或恢复事件 Counter。

指标名称区分 Store 类型，标签区分 vLLM 实例和 UCM 进程。默认 connector 指标带有 `model_name`、`engine` 和 `worker_rank` 标签；Prometheus 抓取后还会添加 `job` 和 `instance`。完整标签说明见 [UCM Metrics 可观测性](metrics_zh.md)。

### 2.1 connector 模式下的指标同步延迟

健康线程会按配置在 UCM 内部持续探测，但 connector metrics 只有在 vLLM 调用 `get_kv_connector_stats()` 时才同步到 `/metrics`。完全没有推理请求时，Prometheus 中的健康指标不会更新，即使后台探测结果已经变化。

## 3. 推荐聚合方法

以下示例省略了部分筛选标签。生产环境应至少按 `job`、`instance`、`model_name` 和 `engine` 限定监控范围，并根据需要筛选 `worker_rank`。

- 只查看 scheduler：`worker_rank="scheduler"`；
- 只查看 worker：`worker_rank!="scheduler"`；
- 不筛选 `worker_rank`：同时包含 scheduler 和所有 worker。

### 3.1 查看每个 Store 的当前状态

```promql
ucm:posix_store_health{
  job="vllm",
  instance="10.0.0.8:8000",
  model_name="Qwen3-32B"
}
```

返回值为 1 表示该 `worker_rank` 对应的 Store 当前可用，0 表示已熔断。这是定位单个异常 worker Store 或 scheduler Store 最直接的查询。

### 3.2 统计健康和熔断 Store 数量

由于 Gauge 只有 0 和 1，可以利用 `sum` 统计健康 Store 数量，利用 `count - sum` 统计熔断 Store 数量：

```promql
# 健康 Store 数量
sum by (job, instance, model_name, engine) (
  ucm:posix_store_health
)
```

```promql
# 熔断 Store 数量
clamp_min(
  count by (job, instance, model_name, engine) (
    ucm:posix_store_health
  )
  -
  sum by (job, instance, model_name, engine) (
    ucm:posix_store_health
  ),
  0
)
```

该方法也是 UCM Overview dashboard 中健康状态数量面板的基本口径。Posix 和 Mooncake 应分别计算，避免不同 Store 类型的状态混在一起。

#### 如何理解 Store 数量

| 场景 | 指标可见 Store count | 说明 |
| --- | ---: | --- |
| DP1、TP1 | 2 | 1 个 worker Store，加 1 个 scheduler Store |
| DP1、多 TP | `TP + 1` | TP 个 worker Store，加 1 个 scheduler Store |
| 多 DP | `DP × (TP + 1)` | 每个 DP 都创建自己的 worker Store 和 scheduler Store |

对于deepseek v4，实际每个worker有两套 Store ，但在metrics里面这两套的状态会聚合，最终显示一个健康或不健康数量。

### 3.3 计算健康 Store 比例

```promql
sum by (job, instance, model_name, engine) (
  ucm:posix_store_health
)
/
clamp_min(
  count by (job, instance, model_name, engine) (
    ucm:posix_store_health
  ),
  1
)
```

分子是健康 Store 数，分母是已上报 Store 总数。由于 `posix_store_health` 只有 0 和 1，这个公式与对 Gauge 求 `avg()` 等价，但“健康数 ÷ 总数”的含义更直观。例如 8 个 Store 中有 2 个熔断，结果为 0.75。

未过滤 `worker_rank` 时，scheduler Store 也会计入分子和分母。DeepSeek V4/HMA/FAWA 路径中的两套 Store 会聚合为一个 Gauge，因此这里计算的是可见健康状态的比例，不是底层 FA、WA Store 的精确健康比例。

### 3.4 计算探测失败比例

应先聚合成功和失败探测次数，再计算比例；不要先计算每个 Store 的失败率后做算术平均。

```promql
(
  sum by (job, instance, model_name, engine) (
    rate(ucm:posix_unhealthy_count_total[5m])
  )
  or
  0 * sum by (job, instance, model_name, engine) (
    rate({__name__=~"ucm:posix_(healthy|unhealthy)_count_total"}[5m])
  )
)
/
clamp_min(
  sum by (job, instance, model_name, engine) (
    rate({__name__=~"ucm:posix_(healthy|unhealthy)_count_total"}[5m])
  ),
  1e-12
)
```

这种写法按探测次数加权。`or 0 * ...` 会在尚未产生失败 series 时补零，避免健康状态下查询结果显示 No data。Mooncake 只需将指标前缀替换为 `mooncake`。

查看时间窗口内的失败探测总数时使用 `increase()`：

```promql
(
  sum by (job, instance, model_name, engine) (
    increase(ucm:posix_unhealthy_count_total[15m])
  )
  or
  0 * sum by (job, instance, model_name, engine) (
    increase({__name__=~"ucm:posix_(healthy|unhealthy)_count_total"}[15m])
  )
)
```

`rate()` 和 `increase()` 能处理进程重启造成的 Counter reset。不要对 Counter 原始值做差，也不要把 Counter 原始值当作当前健康状态。

默认探测间隔为 10 秒，但 connector 同步依赖请求。低流量服务应使用较长窗口，例如 5～15 分钟，以降低延迟同步和少量样本造成的波动。

## 4. 多实例聚合注意事项

| 监控目标 | 推荐方法 | 不推荐方法 |
| --- | --- | --- |
| 单个 Store 当前是否熔断 | 保留 `worker_rank` 查看 Gauge | 对 Gauge 求和后当作布尔值 |
| 实例内是否有任一 Store 熔断 | 按实例对 Gauge 求 `min` | 求 `avg` 后只判断是否大于 0 |
| 实例内健康/熔断 Store 数量 | `sum` 和 `count - sum` | 将 Gauge 状态跨时间累加 |
| 一段时间内的探测失败比例 | 聚合 Counter 分子和分母后相除 | 先算各 Store 比例再做算术平均 |
| 独立物理后端故障次数 | 结合后端标识、日志或外部监控去重 | 直接把所有 Store 的失败 Counter 求和 |

如果需要按集群、节点或存储故障域聚合，建议在 Prometheus target 配置中增加稳定标签，例如：

```yaml
static_configs:
  - targets:
      - "10.0.0.8:8000"
    labels:
      cluster: "production-a"
      node: "inference-01"
      storage_domain: "posix-cluster-a"
```

之后把这些标签加入 `by (...)`。只有确认多个 series 对应同一故障域后，才能合理解释聚合结果；UCM 当前不会从存储路径或 endpoint 自动生成这些标签。

## 5. 告警建议

### 5.1 任一 Posix 熔断器持续异常

```promql
min by (job, instance, model_name, engine) (
  ucm:posix_store_health
) == 0
```

建议配置适当的 `for`，例如 30 秒，避免观察端的短暂抖动直接触发通知。熔断器自身已经通过滑动窗口过滤单次失败，因此告警等待时间不需要替代熔断逻辑。

### 5.2 探测失败比例持续升高

可以使用第 3.4 节的失败比例，并同时要求窗口内存在足够的失败样本。例如在 15 分钟窗口内失败比例超过 20%，且失败次数不少于 3 次时告警。阈值应结合 Store 的探测间隔、Store 数量和业务容忍度调整。

## 6. 排查建议

发现 Gauge 为 0 或失败 Counter 增长时，按以下顺序排查：

1. 按 `instance`、`engine` 和 `worker_rank` 定位异常进程；
2. 查看 UCM 日志中的 `Store health check` 和 `transitioned to UNHEALTHY/HEALTHY`；
3. Posix 检查挂载状态、目录权限、剩余空间、读写和删除能力；
4. Mooncake 检查 client、metadata/master 服务、网络以及 Put/Get/Remove 路径；
5. 确认服务有请求触发 connector metrics 同步，并核对 Prometheus target 为 UP；
6. 后端恢复后观察连续成功探测以及 Gauge 是否回到 1。

Grafana 中可导入 `examples/metrics/grafana_ucm_overview.json` 查看健康/熔断 Store 数量及 Posix、Mooncake 探测趋势。
