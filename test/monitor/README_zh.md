# Quick Monitor

基于 VictoriaMetrics 的单容器监控方案，替代 Prometheus + Grafana，内存占用 < 100MB，内置 Web UI 支持 CSV 导出。

## 前置条件

- Docker ≥ 20.10，且 vLLM 已开启 `--enable-metrics`
- 监控主机可访问 vLLM 的 `/metrics` 端点（默认 8000 端口）

## 快速开始

**1. 配置采集目标**

```bash
cp scrape.yml.template scrape.yml
# 编辑 scrape.yml，将 192.168.1.100:8000 改为实际 vLLM IP，model 标签改为实际模型名
```

**2. 启动**

```bash
docker compose up -d
```

**3. 验证**

```bash
curl "http://localhost:8428/api/v1/query?query=up"
# 返回 1 表示采集正常，0 表示连不上 vLLM（检查 IP 和 --enable-metrics）
```

## 离线部署

```bash
# 有网环境提前准备
docker pull victoriametrics/victoria-metrics:v1.99.0
docker save -o vm.tar victoriametrics/victoria-metrics:v1.99.0

# 离线主机加载后启动
docker load -i vm.tar
docker compose up -d
```

## 查看数据

**VMUI 图形界面**：`http://<host>:8428/vmui`
- 输入框粘贴 PromQL → Execute 生成图表
- 点击 **Download CSV** 导出原始数据（Excel 分析）

## 数据使用
提供了`export_metrics.py`, 可导出所有数据为 CSV 文件并进行图表绘制。
```bash
# 导出前2小时，步长5s的所有数据并输出到metrics_output文件夹
python export_metrics.py \
    --vm-url http://141.111.33.118:8428 \
    --duration 2h \
    --step 5 \
    --output ./metrics_output
```

## 数据持久化

数据保存在 `./data` 目录，可直接打包带走：

```bash
# 备份
docker compose down
tar czf metrics-backup.tar.gz data/

# 恢复
tar xzf metrics-backup.tar.gz
docker compose up -d
```

## 服务停止
docker compose down

## 故障排查

**容器不断重启**：`docker logs vm-vllm`，检查 `scrape.yml` 中 IP 是否填成 `localhost`（应填实际网卡 IP）。
**无数据（up=0）**：`curl http://<vllm-ip>:8000/metrics` 确认 vLLM 指标已开启。