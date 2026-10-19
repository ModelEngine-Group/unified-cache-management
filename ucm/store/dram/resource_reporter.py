#
# MIT License
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ucm.logger import init_logger
from ucm.shared.metrics import ucmmetrics
from ucm.shared.metrics.resource_reporter import (
    MAX_RECORD_BYTES,
    FileResourceMetricsReporter,
    counter_deltas,
)

logger = init_logger(__name__)
UINT64_MAX = (1 << 64) - 1
_REPORTER: "DramPoolResourceReporter | None" = None


def _number(value: Any) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Expected a numeric metric")
    if value < 0 or not math.isfinite(value):
        raise ValueError("Expected a finite nonnegative metric")
    return value


def _count(value: Any) -> int:
    if type(value) is not int or not 0 <= value <= UINT64_MAX:
        raise ValueError("Expected a uint64 count")
    return value


@dataclass(frozen=True)
class HistogramSnapshot:
    upper_bounds: tuple[float, ...]
    bucket_counts: tuple[int, ...]
    sum: float

    @property
    def count(self) -> int:
        return sum(self.bucket_counts)


@dataclass(frozen=True)
class DramPoolResourceSnapshot:
    timestamp: float
    counters: dict[str, int | float]
    gauges: dict[str, int | float]
    histograms: dict[str, HistogramSnapshot]


def parse_drampool_resource_snapshot(line: str) -> DramPoolResourceSnapshot:
    record = json.loads(line)
    if record.get("event") != "drampool_metrics_snapshot":
        raise ValueError("Not a DramPool metrics snapshot")
    timestamp = record["timestamp"]
    if isinstance(timestamp, str):
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("Snapshot timestamp must include a timezone")
        timestamp = parsed.timestamp()
    counters, gauges, histograms = {}, {}, {}
    for section, metric_type, destination in (
        ("counters", "counter", counters),
        ("gauges", "gauge", gauges),
        ("histograms", "histogram", histograms),
    ):
        values = record[section]
        if not isinstance(values, dict):
            raise ValueError(f"Expected {section} mapping")
        for name, value in values.items():
            if not name.startswith("drampool_") or name.startswith(
                "drampool_resource_"
            ):
                continue  # Only source metrics; reporter health is recorded locally.
            if metric_type != "histogram":
                destination[name] = _number(value)
                continue
            bounds = tuple(float(_number(v)) for v in value["upper_bounds"])
            if any(a >= b for a, b in zip(bounds, bounds[1:])):
                raise ValueError(f"Histogram boundaries must be increasing: {name}")
            counts = tuple(_count(v) for v in value["bucket_counts"])
            count = _count(value["count"])
            total = float(_number(value["sum"]))
            if len(counts) != len(bounds) + 1 or sum(counts) != count:
                raise ValueError(f"Histogram count/shape mismatch for {name}")
            if count == 0 and total != 0:
                raise ValueError(f"Nonzero sum for empty histogram {name}")
            destination[name] = HistogramSnapshot(bounds, counts, total)
    return DramPoolResourceSnapshot(
        float(_number(timestamp)),
        counters,
        gauges,
        histograms,
    )


def snapshot_deltas(
    current: DramPoolResourceSnapshot, previous: DramPoolResourceSnapshot | None
):
    """Compute metric deltas, treating decreases as source resets."""
    if previous is None:
        previous = current
    counters = counter_deltas(current.counters, previous.counters)
    histograms = {}
    for name, value in current.histograms.items():
        old = previous.histograms.get(name)
        if old is None:
            counts, total = list(value.bucket_counts), value.sum
        else:
            if old.upper_bounds != value.upper_bounds:
                raise ValueError(f"Histogram schema changed: {name}")
            counts = [a - b for a, b in zip(value.bucket_counts, old.bucket_counts)]
            total = value.sum - old.sum
            if (
                any(v < 0 for v in counts)
                or total < 0
                or (all(v == 0 for v in counts) and total != 0)
            ):
                # Like a counter decrease, infer a reset. Reset the whole
                # distribution rather than mixing reset and differenced buckets.
                counts, total = list(value.bucket_counts), value.sum
        histograms[name] = (counts, total)
    return counters, current.gauges, histograms


def _snapshot_record(snapshot: DramPoolResourceSnapshot) -> dict:
    return {
        "event": "drampool_metrics_snapshot",
        "timestamp": snapshot.timestamp,
        "counters": snapshot.counters,
        "gauges": snapshot.gauges,
        "histograms": {
            name: {
                "upper_bounds": h.upper_bounds,
                "bucket_counts": h.bucket_counts,
                "count": h.count,
                "sum": h.sum,
            }
            for name, h in snapshot.histograms.items()
        },
    }


class DramPoolResourceReporter(FileResourceMetricsReporter):
    def __init__(
        self,
        log_path: str,
        interval_sec: float = 15.0,
        shared_memory_dir: str = "/dev/shm",
    ):
        super().__init__(
            log_path=log_path,
            reporter_name="DramPool",
            identity=str(Path(log_path).resolve()),
            interval_sec=interval_sec,
            shared_memory_dir=shared_memory_dir,
        )

    def _read_state(self):
        try:
            state = self._read_previous_state()
            if state is None:
                return None
            return parse_drampool_resource_snapshot(json.dumps(state["snapshot"]))
        except Exception as error:
            logger.warning(f"Ignoring invalid DramPool reporter state: {error}")
            return None

    def _write_state(self, snapshot):
        self._write_previous_state({"snapshot": _snapshot_record(snapshot)})

    def _report_snapshot(self, snapshot, previous):
        counters, gauges, histograms = snapshot_deltas(snapshot, previous)
        try:
            ucmmetrics.merge_histogram_stats(histograms)
            ucmmetrics.update_stats(counters | gauges)
        except Exception as error:
            logger.warning(f"Failed to import DramPool resource metrics: {error}")
            return
        try:
            self._write_state(snapshot)
        except OSError as error:
            logger.warning(f"Failed to write DramPool reporter state: {error}")

    def _collect_once(self):
        snapshot = parse_drampool_resource_snapshot(self._read_latest_complete_line())
        previous = self._read_state()
        self._report_snapshot(snapshot, previous)


def start_drampool_resource_reporter(config: dict) -> DramPoolResourceReporter | None:
    global _REPORTER

    path = str(config.get("drampool_resource_log_path", ""))
    enabled = bool(config.get("drampool_resource_metrics_enable", bool(path)))
    if not enabled or not path or int(config.get("device_id", -1)) >= 0:
        return None
    if _REPORTER is None:
        _REPORTER = DramPoolResourceReporter(
            path,
            interval_sec=float(
                config.get("drampool_resource_metrics_interval_sec", 15)
            ),
        )
        _REPORTER.start()
    return _REPORTER
