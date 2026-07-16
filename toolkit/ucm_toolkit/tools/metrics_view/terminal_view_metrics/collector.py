from __future__ import annotations

import signal
import sys
import time
import urllib.request
from pathlib import Path

from .config import load_config, metric_names_for_scrape, parse_duration_seconds
from .parser import MetricSample, parse_prometheus_text
from .storage import MetricsStore


def scrape_url(url: str, timeout: float = 5.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def add_url_label(samples: list[MetricSample], url: str) -> list[MetricSample]:
    return [
        MetricSample(
            name=sample.name,
            labels={**sample.labels, "url": url},
            value=sample.value,
            timestamp_ms=sample.timestamp_ms,
        )
        for sample in samples
    ]


def scrape_urls(
    urls: list[str], timeout_seconds: float
) -> tuple[list[MetricSample], list[tuple[str, Exception]]]:
    samples: list[MetricSample] = []
    failures: list[tuple[str, Exception]] = []
    for url in urls:
        try:
            samples.extend(
                add_url_label(
                    parse_prometheus_text(scrape_url(url, timeout_seconds)), url
                )
            )
        except Exception as exc:
            failures.append((url, exc))
    return samples, failures


def collect_cycle(
    urls: list[str],
    store: MetricsStore,
    scrape_ts_ms: int,
    timeout_seconds: float,
    include_names: set[str] | None = None,
) -> list[tuple[str, Exception]]:
    samples, failures = scrape_urls(urls, timeout_seconds)
    store.write_samples(samples, scrape_ts_ms, include_names)
    return failures


def collect_loop(
    urls: list[str],
    db_path: str | Path,
    interval_seconds: float,
    config_path: str | Path | None = None,
    retention_seconds: float | None = None,
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
                ts_ms = int(time.time() * 1000)
                failures = collect_cycle(
                    urls, store, ts_ms, timeout_seconds, include_names
                )
                for url, exc in failures:
                    print(
                        f"scrape failed for {url}: {exc}", file=sys.stderr, flush=True
                    )
                if retention_seconds:
                    store.prune_before(
                        ts_ms - int(parse_duration_seconds(retention_seconds) * 1000)
                    )
            except Exception as exc:
                print(f"collection failed: {exc}", file=sys.stderr, flush=True)
            elapsed = time.time() - started
            time.sleep(max(0.0, interval_seconds - elapsed))
    finally:
        store.close()
