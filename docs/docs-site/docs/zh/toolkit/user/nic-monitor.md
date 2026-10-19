# nic-monitor

`nic-monitor` 用于观察 Linux 物理网卡的收发速率和利用率。跨节点加载 KV 或访问网络存储时，它可以帮助确认流量经过哪些网卡、负载是否集中在某一张卡，以及网卡是否接近链路带宽上限。

适合在以下情况下使用：

- P/D 传输或远程缓存读取较慢时，检查网络负载与流量分布。
- 压测期间持续记录网卡流量，回看高负载时段是否与请求延迟上升重合。
- 多网卡部署中，比较各卡的收发速率，发现负载不均的现象。

前台模式实时显示统计，后台模式保存 CSV 和阶段报告，采样结果还可生成图表。统计覆盖整张物理网卡上的流量，将其归因到 UCM 请求时需要结合服务指标。

脚本通过 `/sys/class/net` 找物理网卡，通过 `ethtool` 优先读取厂商统计计数器，失败时回退到 `/proc/net/dev`。

← 返回 [UCM Toolkit 文档](../index.md)

## 依赖

- Linux、`bash`、`ethtool`，并且需要 root 或 sudo 权限读取网卡统计。
- NIC CSV 离线绘图：`pandas`、`matplotlib`。

因为需要访问 `ethtool` 统计，通常需要 root 或 sudo：

```bash
sudo ucm-toolkit run nic-monitor fg
```

## 前台模式

前台模式实时刷新终端，不落盘：

```bash
sudo ucm-toolkit run nic-monitor fg
sudo ucm-toolkit run nic-monitor fg 5
```

参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `fg [interval_sec]` | `2` | 前台刷新间隔，单位秒。 |

按 `Ctrl+C` 停止。

## 后台模式

后台模式会创建 `.log`、`.csv`、`.pid` 三类文件，并把采样数据持续写入 CSV：

```bash
sudo ucm-toolkit run nic-monitor bg
sudo ucm-toolkit run nic-monitor bg 24 5
sudo ucm-toolkit run nic-monitor bg 24 5 --log-dir /mnt/test/net_log
sudo ucm-toolkit run nic-monitor bg 24 5 --log-dir /mnt/test/net_log --stat-cycle-seconds 600
```

参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `bg [duration_hours] [interval_sec]` | `12 10` | 后台运行时长和采样间隔。 |
| `--log-dir PATH` | 当前工作目录下的 `net_log` | 后台日志、CSV、PID 输出目录。 |
| `--stat-cycle-seconds SECONDS` | `3600` | 阶段统计周期，单位秒。 |

输出文件名格式：

```text
Eth_Perf_Monitor_YYYYmmdd_HHMMSS.log
Eth_Perf_Monitor_YYYYmmdd_HHMMSS.csv
Eth_Perf_Monitor_YYYYmmdd_HHMMSS.pid
```

后台启动时会检查同一 `--log-dir` 下是否已有存活的 `.pid` 进程，避免重复启动。

## NIC 结果可视化

`toolkit/src/nic_monitor` 还提供两个离线可视化入口。它们当前没有注册为 `ucm-toolkit run` 顶层工具。

### Python 绘图

安装依赖：

```bash
python -m pip install pandas matplotlib
```

生成 PNG 图表：

```bash
python toolkit/src/nic_monitor/visualize_traffic.py net_log/Eth_Perf_Monitor_*.csv -o net_log/charts
python toolkit/src/nic_monitor/visualize_traffic.py net_log/Eth_Perf_Monitor_*.csv -o net_log/charts -i eth0 eth1
```

参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `csv` | 当前目录下 `*.csv` | 输入 CSV，可传多个。 |
| `-o`, `--output` | `.` | 图片输出目录。 |
| `-i`, `--interfaces` | 全部网卡 | 只绘制指定网卡，可传多个名称。 |

每个 CSV 会生成一个同名子目录，包含流量时序、利用率时序、总流量堆叠、统计摘要等 PNG。

### 浏览器页面

可以直接打开：

```text
toolkit/src/nic_monitor/index.html
```

页面支持上传或拖拽 `nic-monitor bg` 生成的 CSV，并在浏览器中查看交互式图表。

## 常见问题

### 权限失败

用 root 或 sudo 运行：

```bash
sudo ucm-toolkit run nic-monitor fg
sudo ucm-toolkit run nic-monitor bg 12 10
```

同时确认系统安装了 `ethtool`。
