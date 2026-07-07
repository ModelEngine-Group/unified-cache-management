from __future__ import annotations

import signal
import sys
import time
import urllib.request
from pathlib import Path

from .config import load_config, metric_names_for_scrape, parse_duration_seconds
from .parser import parse_prometheus_text
from .storage import MetricsStore


def scrape_url(url: str, timeout: float = 5.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def collect_loop(
    url: str,
    db_path: str | Path,
    interval_seconds: float,
    config_path: str | Path | None = None,
    retention_seconds: float | None = None,
    once: bool = False,
    timeout_seconds: float = 5.0,
) -> None:
    stop = {"value": False}

    def _stop(_signum, _frame):
        stop["value"] = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    config = load_config(config_path) if config_path else {}
    include_names = metric_names_for_scrape(config) if config else None
    store = MetricsStore(db_path)
    try:
        while not stop["value"]:
            started = time.time()
            try:
                text = scrape_url(url, timeout_seconds)
                ts_ms = int(time.time() * 1000)
                store.write_samples(parse_prometheus_text(text), ts_ms, include_names)
                if retention_seconds:
                    store.prune_before(
                        ts_ms - int(parse_duration_seconds(retention_seconds) * 1000)
                    )
            except Exception as exc:
                print(f"scrape failed: {exc}", file=sys.stderr, flush=True)
                if once:
                    raise
            if once:
                return
            elapsed = time.time() - started
            time.sleep(max(0.0, interval_seconds - elapsed))
    finally:
        store.close()
