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

import atexit
import hashlib
import json
import math
import os
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ucm.logger import init_logger
from ucm.shared.metrics import ucmmetrics

logger = init_logger(__name__)
UINT64_MAX = (1 << 64) - 1
MAX_RECORD_BYTES = 1024 * 1024
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
    unit: str

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
            # This importer currently accepts duration histograms in microseconds.
            if value["unit"] != "us" or not name.endswith("_us"):
                raise ValueError(f"Unsupported histogram unit for {name}")
            if len(counts) != len(bounds) + 1 or sum(counts) != count:
                raise ValueError(f"Histogram count/shape mismatch for {name}")
            if count == 0 and total != 0:
                raise ValueError(f"Nonzero sum for empty histogram {name}")
            destination[name] = HistogramSnapshot(bounds, counts, total, "us")
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
    counters, histograms = {}, {}
    for name, value in current.counters.items():
        old = previous.counters.get(name, 0) if previous is not None else None
        counters[name] = 0 if old is None else value - old if value >= old else value
    for name, value in current.histograms.items():
        old = previous.histograms.get(name) if previous is not None else None
        if previous is None:
            counts, total = [0] * len(value.bucket_counts), 0.0
        elif old is None:
            counts, total = list(value.bucket_counts), value.sum
        else:
            if old.upper_bounds != value.upper_bounds or old.unit != value.unit:
                raise ValueError(f"Histogram schema changed: {name}")
            counts = [a - b for a, b in zip(value.bucket_counts, old.bucket_counts)]
            total = value.sum - old.sum
            if any(v < 0 for v in counts) or total < 0:
                # Like a counter decrease, infer a reset. Reset the whole
                # distribution rather than mixing reset and differenced buckets.
                counts, total = list(value.bucket_counts), value.sum
            if sum(counts) == 0 and total != 0:
                raise ValueError(f"Histogram sum changed without samples: {name}")
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
                "unit": h.unit,
            }
            for name, h in snapshot.histograms.items()
        },
    }


class DramPoolResourceReporter:
    def __init__(
        self,
        log_path: str,
        interval_sec: float = 15.0,
        shared_memory_dir: str = "/dev/shm",
    ):
        self.log_path = Path(log_path)
        self.interval_sec = max(float(interval_sec), 1.0)
        self.shared_dir = Path(shared_memory_dir)
        if not self.shared_dir.is_dir():
            self.shared_dir = Path(tempfile.gettempdir())
        self._lock_file = None
        self._lock_path: Path | None = None
        self._state_path: Path | None = None
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="drampool-resource-reporter", daemon=True
        )
        atexit.register(self.stop)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=min(self.interval_sec + 1.0, 5.0))
        if not self._thread.is_alive():
            self._release_leadership()

    def _release_leadership(self):
        if self._lock_file is None:
            return
        try:
            import fcntl

            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        self._lock_file.close()
        self._lock_file = None

    def _read_latest_complete_line(self):
        with self.log_path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            end = stream.tell()
            # Two maximum-sized records allow skipping a partial final record.
            start = max(0, end - 2 * MAX_RECORD_BYTES)
            stream.seek(start)
            lines = stream.read(end - start).split(b"\n")
        lines.pop()  # EOF fragment, including an empty fragment after a newline.
        if start:
            lines = lines[1:]  # May begin in the middle of a record.
        for line in reversed(lines):
            if not line.strip():
                continue
            if len(line) > MAX_RECORD_BYTES:
                raise ValueError("Snapshot exceeds 1 MiB")
            return line.decode("utf-8")
        raise ValueError("DramPool resource log has no complete record")

    def _try_become_leader(self):
        try:
            import fcntl
        except ImportError:
            logger.warning(
                "DramPool resource reporter requires fcntl for host election"
            )
            self._stop_event.set()
            return False

        identity = hashlib.sha256(str(self.log_path.resolve()).encode()).hexdigest()[
            :24
        ]
        self._lock_path = self.shared_dir / f"ucm_drampool_metrics_{identity}.lock"
        self._state_path = self.shared_dir / f"ucm_drampool_metrics_{identity}.json"
        lock_file = self._lock_path.open("a+")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock_file.close()
            return False
        except Exception:
            lock_file.close()
            raise
        self._lock_file = lock_file
        logger.info(f"Became DramPool resource metrics reporter for {self.log_path}")
        return True

    def _read_state(self):
        try:
            with self._state_path.open("rb") as stream:
                data = stream.read(MAX_RECORD_BYTES + 1)
            if len(data) > MAX_RECORD_BYTES:
                raise ValueError("Oversized reporter state")
            state = json.loads(data)
            return parse_drampool_resource_snapshot(json.dumps(state["snapshot"]))
        except FileNotFoundError:
            return None
        except Exception as error:
            logger.warning(f"Ignoring invalid DramPool reporter state: {error}")
            return None

    def _write_state(self, snapshot):
        temporary = self._state_path.with_suffix(f".{os.getpid()}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(
                    {"snapshot": _snapshot_record(snapshot)},
                    stream,
                    allow_nan=False,
                    separators=(",", ":"),
                )
            os.replace(temporary, self._state_path)
        finally:
            temporary.unlink(missing_ok=True)

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

    def _run(self):
        try:
            if self._stop_event.is_set():
                return
            try:
                if not self._try_become_leader():
                    return
            except Exception as error:
                logger.warning(f"Failed to elect DramPool resource reporter: {error}")
                return
            while not self._stop_event.is_set():
                try:
                    self._collect_once()
                except Exception as error:
                    logger.warning(
                        f"Failed to collect DramPool resource metrics: {error}"
                    )
                self._stop_event.wait(self.interval_sec)
        finally:
            self._release_leadership()


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
