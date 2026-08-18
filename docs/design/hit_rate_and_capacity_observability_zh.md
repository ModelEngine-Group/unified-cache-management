# UCM 分层命中率与水位观测设计

## 1. 目标

当前设计为以下两类 Store pipeline 提供统一的可观测性：

- `YuanRong|Posix`：展示 HBM、YuanRong DRAM、YuanRong SSD、Posix Store 命中率。
- `Cache|Posix`：展示 HBM、Cache、Posix Store 命中率。
- 展示 YuanRong DRAM、YuanRong SSD、Posix Store 的使用量、容量和水位。

结果可以在 Grafana dashboard 和 UCM `metrics_view` 中查看。

## 2. 统计原则

最终口径为：

```text
某层命中率 = 该层命中 token 数 / 总请求 token 数
```

## 3. 命中率计算

### 3.1 HBM 和外存总命中率

```text
H_hbm = rate(vllm:prefix_cache_hits_total)
        / rate(vllm:prefix_cache_queries_total)

H_external_conditional = rate(vllm:external_prefix_cache_hits_total)
                         / rate(vllm:external_prefix_cache_queries_total)

H_external = H_external_conditional * (1 - H_hbm)
```

其中 `H_external` 是外存相对于全部请求 token 的命中率。

### 3.2 YuanRong|Posix

先根据最终加载成功的 shard 来源拆分 YuanRong 和 Posix：

```text
Y = rate(yuanrong_load_success_shards_total)

P = rate(yuanrong_lookup_miss_posix_load_success_shards_total)
  + rate(yuanrong_load_fallback_posix_load_success_shards_total)

H_yuanrong = H_external * Y / (Y + P)
H_posix    = H_external * P / (Y + P)
```

再根据 YuanRong 实际 Get 命中来源拆分 DRAM 和本地 SSD：

```text
D = rate(yuanrong_local_dram_load_hits_total)
  + rate(yuanrong_remote_load_hits_total)

S = rate(yuanrong_local_ssd_load_hits_total)

H_yuanrong_dram = H_yuanrong * D / (D + S)
H_yuanrong_ssd  = H_yuanrong * S / (D + S)
```

当前部署中远端 SSD 加载不会成功，因此 `remote_hit_num` 全部按远端 DRAM 处理。`l2_hit_num` 不参与命中率拆分，仅作为异常诊断指标。

### 3.3 Cache|Posix

```text
C = rate(cache_load_success_shards_total)
P = rate(cache_posix_load_success_shards_total)

H_cache = H_external * C / (C + P)
H_posix = H_external * P / (C + P)
```

`cache_load_success_shards_total` 表示由已有 Cache buffer 成功加载的 shard；`cache_posix_load_success_shards_total` 表示 Cache 未就绪、经 Posix 填充后成功加载的 shard。

### 3.4 总命中率

Grafana 的 `KV Cache Hit Rate Breakdown` 面板包含 `Total` 曲线：

```text
H_total = H_hbm + H_external
```

它等价于当前 pipeline 中各层命中率之和：

```text
YuanRong|Posix: H_hbm + H_yuanrong_dram + H_yuanrong_ssd + H_posix
Cache|Posix:    H_hbm + H_cache + H_posix
```

加载失败是极少数情况，当前公式仍将外存命中率按成功 shard 比例分配。`yuanrong_load_failed_shards_total` 和 `cache_load_failed_shards_total` 单独用于判断估算结果是否可靠。

## 4. 容量与水位

| 层级 | 使用量 | 容量 | 水位 |
| --- | --- | --- | --- |
| YuanRong DRAM | `yuanrong_dram_used_bytes` | `yuanrong_dram_capacity_bytes` | `yuanrong_dram_usage_ratio` |
| YuanRong SSD | `yuanrong_ssd_used_bytes` | `yuanrong_ssd_capacity_bytes` | `yuanrong_ssd_usage_ratio` |
| Posix Store | `posix_store_used_bytes` | `posix_store_capacity_bytes` | `posix_store_usage_ratio` |

YuanRong 指标来自 `kv_resource.log`：

```text
DRAM used     = metrics.shared_memory.physical_memory_usage
DRAM capacity = metrics.shared_memory.total_limit
SSD used      = metrics.spill_hard_disk.physical_space_usage
SSD capacity  = metrics.spill_hard_disk.total_limit
```

Posix 使用量由 GC 按目录 shard 采样文件数量后估算：

```text
used_bytes     = estimated_file_count * block_size
capacity_bytes = posix_capacity_gb * GiB
usage_ratio    = used_bytes / capacity_bytes
```

Posix 水位是 Store 的逻辑占用估算值，不是整个文件系统的物理使用率。多个 DP/worker 可能上报同一个共享目录的值，因此 Grafana 使用 `avg` 聚合 Posix Gauge，避免重复相加。YuanRong 节点级 Gauge 由单机 leader 上报，使用 `max` 聚合。

## 5. YuanRong 日志采集

UCM scheduler 启动后台 reporter，默认每 15 秒读取一次 `kv_resource.log` 的最新完整 `resource_snapshot`：

```text
kv_resource.log -> UCM scheduler reporter -> vLLM /metrics
                -> Prometheus -> Grafana / metrics_view
```

同一台机器上的多个 UCM instance 在 reporter 线程启动时使用 `/dev/shm` 中的 `flock` 竞争一次 leader。抢锁成功的线程负责周期采集；抢锁失败的线程立即退出，不再重试，因此不会为同一个 YuanRong 进程重复上报节点级数据，也不会保留空闲检查线程。当前不支持运行期间切换 leader。

日志中的 `mem_hit_num`、`remote_hit_num`、`disk_hit_num` 是累计值。reporter 保存上一份快照，只把相邻快照差值写入 UCM Counter；容量和水位直接作为 Gauge 上报。

示例配置：

```yaml
yuanrong_resource_metrics_enable: true # 可选，默认true
yuanrong_resource_log_path: /var/log/yuanrong/kv_resource.log # 必填，指向yuanrong启动目录
yuanrong_resource_metrics_interval_sec: 15 # 可选，默认15s
```

未配置日志路径时不启动采集；读取或解析失败不会影响推理服务，只会增加 `yuanrong_resource_log_read_errors_total`。

## 6. 展示位置

- Grafana `KV Cache Hit Rate Breakdown`：各层命中率及 `Total`。
- Grafana `Store Capacity Watermark`：各层水位。
- Grafana Store used/capacity 面板：实际使用量和容量。
- UCM `metrics_view`：使用相同 PromQL 公式展示分层命中率与容量指标。

## 7. 使用 metrics lite 查看指标

先在 UCM 仓库根目录安装 toolkit：

```bash
python -m pip install -e toolkit
ucm-toolkit run metrics-view list-configs
```

### 7.1 查看当前快照

`check` 直接拉取 vLLM 的 `/metrics` 接口并使用内置 `metrics_lite` 配置计算结果，不需要部署 Prometheus：

```bash
ucm-toolkit run metrics-view check \
  --url http://127.0.0.1:8000/metrics \
  --config metrics_lite
```

将地址和端口替换为实际的 vLLM 服务地址。当前分层命中率按 shard 比例计算，不需要传入 TP、DP 参数。

重点关注以下输出：

| 输出字段 | 含义 |
| --- | --- |
| `hbm_hit_rate` | HBM 命中率 |
| `yuanrong_dram_hit_rate` | YuanRong DRAM 命中率 |
| `yuanrong_ssd_hit_rate` | YuanRong SSD 命中率 |
| `cache_hit_rate` | Cache 命中率 |
| `posix_hit_rate` | Posix Store 命中率 |
| `yuanrong_dram_used_bytes` / `yuanrong_dram_capacity_bytes` | YuanRong DRAM 使用量和容量 |
| `yuanrong_dram_usage_ratio` | YuanRong DRAM 水位 |
| `yuanrong_ssd_used_bytes` / `yuanrong_ssd_capacity_bytes` | YuanRong SSD 使用量和容量 |
| `yuanrong_ssd_usage_ratio` | YuanRong SSD 水位 |
| `posix_store_used_bytes` / `posix_store_capacity_bytes` | Posix 使用量和容量 |
| `posix_store_usage_ratio` | Posix 水位 |

命中率和水位的单位都是 `ratio`，例如 `0.85` 表示 `85%`。当前 pipeline 不包含的层可能不显示或显示为无数据。

### 7.2 查看时间窗口

需要观察一段时间内的变化时，可以先启动后台采集：

```bash
ucm-toolkit run metrics-view start \
  --url http://127.0.0.1:8000/metrics \
  --interval 5s
```

查询最近 10 分钟，并按 1 分钟聚合：

```bash
ucm-toolkit run metrics-view query \
  --window 10m \
  --aggr-by 1m \
  --config metrics_lite
```

查看采集状态或停止采集：

```bash
ucm-toolkit run metrics-view status
ucm-toolkit run metrics-view stop
```

PD 分离或多实例部署可以重复传入 `--url`。查询时可使用 `--tag url=<完整metrics地址>` 或 `--tag model_name=<模型名>` 筛选目标实例。
