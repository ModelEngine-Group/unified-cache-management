```markdown
# Quick Monitor

A single-container monitoring solution based on VictoriaMetrics, serving as a lightweight alternative to Prometheus + Grafana. Uses less than 100MB memory and includes a built-in Web UI with CSV export support.

## Prerequisites

- Docker ≥ 20.10 with `docker compose` support
- vLLM service started with `--enable-metrics` (or env `VLLM_ENABLE_METRICS=1`)
- Monitoring host can reach the vLLM `/metrics` endpoint (default port 8000)

## Quick Start

**1. Configure Scrape Targets**

```bash
cp scrape.yml.template scrape.yml
# Edit scrape.yml: replace 192.168.1.100:8000 with actual vLLM IP:port, update model label
```

**2. Start Services**

```bash
docker compose up -d
```

**3. Verify Collection**

```bash
curl "http://localhost:8428/api/v1/query?query=up"
# Returns 1 = connected, 0 = check IP and --enable-metrics flag
```

## Offline Deployment

For air-gapped environments without internet access:

```bash
# On a machine with internet
docker pull victoriametrics/victoria-metrics:v1.99.0
docker save -o vm.tar victoriametrics/victoria-metrics:v1.99.0

# Transfer vm.tar to target host, then load and start
docker load -i vm.tar
docker compose up -d
```

## Accessing Data

**VMUI Web Interface**: `http://<host>:8428/vmui`
- Paste PromQL queries into the input box → Click **Execute** to generate charts
- Click **Download CSV** to export raw data for Excel analysis

## Data Usage
The `export_metrics.py` script is provided, which can export all data as CSV files and generate charts.
```bash
# Export 2 hours with 5s granularity to specific folder
python export_metrics.py \
    --vm-url http://141.111.33.118:8428 \
    --duration 2h \
    --step 5 \
    --output ./metrics_output
```

## Data Persistence

Metrics are stored in `./data` directory and can be archived for offline analysis:

```bash
# Backup
docker compose down
tar czf metrics-backup.tar.gz data/

# Restore
tar xzf metrics-backup.tar.gz
docker compose up -d
```

## Stop Services

```bash
docker compose down
```

## Troubleshooting

**Container keeps restarting**: Check logs with `docker logs vm-vllm`. Ensure `scrape.yml` uses actual NIC IP (not `localhost`).

**No data (up=0)**: Verify vLLM metrics endpoint: `curl http://<vllm-ip>:8000/metrics`. Ensure `--enable-metrics` is enabled.
```