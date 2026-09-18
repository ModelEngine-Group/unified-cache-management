#
# MIT License
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#

import atexit
import hashlib
import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ucm.logger import init_logger
from ucm.shared.metrics import ucmmetrics

logger = init_logger(__name__)

_REPORTERS: list["YuanRongResourceReporter"] = []

_COUNTER_METRICS = {
    "mem_hit_num": "yuanrong_local_dram_load_hits_total",
    "remote_hit_num": "yuanrong_remote_load_hits_total",
    "disk_hit_num": "yuanrong_local_ssd_load_hits_total",
    "l2_hit_num": "yuanrong_l2_load_hits_total",
}


@dataclass(frozen=True)
class YuanRongResourceSnapshot:
    counters: dict[str, float]
    gauges: dict[str, float]
    timestamp: float


def parse_yuanrong_resource_snapshot(line: str) -> YuanRongResourceSnapshot:
    record = json.loads(line)
    if record.get("event") != "resource_snapshot":
        raise ValueError("not a resource_snapshot record")
    if record.get("version") != "v0":
        raise ValueError(
            f"unsupported resource snapshot version: {record.get('version')}"
        )

    metrics = record["metrics"]
    hit_metrics = metrics["oc_hit_num"]
    shared_memory = metrics["shared_memory"]
    spill_disk = metrics["spill_hard_disk"]

    counters = {
        metric_name: _nonnegative_number(hit_metrics[field_name], field_name)
        for field_name, metric_name in _COUNTER_METRICS.items()
    }
    dram_used = _nonnegative_number(
        shared_memory["physical_memory_usage"], "physical_memory_usage"
    )
    dram_capacity = _nonnegative_number(
        shared_memory["total_limit"], "shared_memory.total_limit"
    )
    ssd_used = _nonnegative_number(
        spill_disk["physical_space_usage"], "physical_space_usage"
    )
    ssd_capacity = _nonnegative_number(
        spill_disk["total_limit"], "spill_hard_disk.total_limit"
    )
    gauges = {
        "yuanrong_dram_used_bytes": dram_used,
        "yuanrong_dram_capacity_bytes": dram_capacity,
        "yuanrong_dram_usage_ratio": _ratio(dram_used, dram_capacity),
        "yuanrong_ssd_used_bytes": ssd_used,
        "yuanrong_ssd_capacity_bytes": ssd_capacity,
        "yuanrong_ssd_usage_ratio": _ratio(ssd_used, ssd_capacity),
        "yuanrong_resource_log_last_update_timestamp_seconds": _parse_timestamp(
            record["time"]
        ),
        "yuanrong_resource_log_reporter_leader": 1.0,
    }
    timestamp = gauges["yuanrong_resource_log_last_update_timestamp_seconds"]
    return YuanRongResourceSnapshot(counters, gauges, timestamp)


def counter_deltas(
    current: dict[str, float], previous: dict[str, float] | None
) -> dict[str, float]:
    if previous is None:
        return {name: 0.0 for name in current}
    return {
        name: (
            value - previous.get(name, 0.0)
            if value >= previous.get(name, 0.0)
            else value
        )
        for name, value in current.items()
    }


def _nonnegative_number(value: Any, name: str) -> float:
    number = float(value)
    if number < 0:
        raise ValueError(f"negative YuanRong metric {name}: {number}")
    return number


def _ratio(used: float, capacity: float) -> float:
    return used / capacity if capacity > 0 else 0.0


def _parse_timestamp(value: Any) -> float:
    if not isinstance(value, str) or not value:
        raise ValueError("missing YuanRong resource snapshot time")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    return datetime.fromisoformat(normalized).timestamp()


class YuanRongResourceReporter:
    def __init__(
        self,
        log_path: str,
        endpoint: str,
        interval_sec: float = 15.0,
        shared_memory_dir: str = "/dev/shm",
    ):
        self.log_path = Path(log_path)
        self.interval_sec = max(float(interval_sec), 1.0)
        shared_dir = Path(shared_memory_dir)
        if not shared_dir.is_dir():
            shared_dir = Path(tempfile.gettempdir())
        identity = hashlib.sha256(
            f"{endpoint}|{self.log_path.resolve()}".encode()
        ).hexdigest()[:24]
        self.lock_path = shared_dir / f"ucm_yuanrong_metrics_{identity}.lock"
        self.state_path = shared_dir / f"ucm_yuanrong_metrics_{identity}.json"
        self._stop_event = threading.Event()
        self._lock_file = None
        self._thread = threading.Thread(
            target=self._run,
            name="yuanrong-resource-reporter",
            daemon=True,
        )
        atexit.register(self.stop)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=min(self.interval_sec + 1.0, 5.0))
        self._release_leadership()

    def _run(self) -> None:
        if self._stop_event.is_set():
            return
        try:
            if not self._try_become_leader():
                return
        except Exception as error:
            logger.warning(f"Failed to elect YuanRong resource reporter: {error}")
            ucmmetrics.update_stats({"yuanrong_resource_log_read_errors_total": 1.0})
            return

        while not self._stop_event.is_set():
            try:
                self._collect_once()
            except Exception as error:
                logger.warning(f"Failed to collect YuanRong resource metrics: {error}")
                ucmmetrics.update_stats(
                    {"yuanrong_resource_log_read_errors_total": 1.0}
                )
            self._stop_event.wait(self.interval_sec)

    def _try_become_leader(self) -> bool:
        if self._lock_file is not None:
            return True
        try:
            import fcntl
        except ImportError:
            logger.warning(
                "YuanRong resource reporter requires fcntl for host election"
            )
            self._stop_event.set()
            return False

        lock_file = open(self.lock_path, "a+")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock_file.close()
            return False
        self._lock_file = lock_file
        logger.info(f"Became YuanRong resource metrics reporter for {self.log_path}")
        return True

    def _release_leadership(self) -> None:
        if self._lock_file is None:
            return
        try:
            import fcntl

            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
        except (ImportError, OSError):
            pass
        self._lock_file.close()
        self._lock_file = None

    def _collect_once(self) -> None:
        line = self._read_latest_complete_line()
        snapshot = parse_yuanrong_resource_snapshot(line)
        previous = self._read_previous_counters()
        updates = snapshot.gauges | counter_deltas(snapshot.counters, previous)
        ucmmetrics.update_stats(updates)
        self._write_previous_counters(snapshot.counters)

    def _read_latest_complete_line(self) -> str:
        with open(self.log_path, "rb") as log_file:
            log_file.seek(0, os.SEEK_END)
            end = log_file.tell()
            if end == 0:
                raise ValueError("YuanRong resource log is empty")
            position = end
            data = b""
            while position > 0:
                chunk_size = min(position, 64 * 1024)
                position -= chunk_size
                log_file.seek(position)
                data = log_file.read(chunk_size) + data
                ends_with_newline = data.endswith((b"\n", b"\r"))
                complete = data.splitlines()
                if not ends_with_newline and complete:
                    complete.pop()
                if complete:
                    for candidate in reversed(complete):
                        if candidate.strip():
                            return candidate.decode("utf-8")
            raise ValueError("YuanRong resource log has no complete JSON record")

    def _read_previous_counters(self) -> dict[str, float] | None:
        try:
            with open(self.state_path, "r", encoding="utf-8") as state_file:
                state = json.load(state_file)
            return {
                name: float(value) for name, value in state.get("counters", {}).items()
            }
        except FileNotFoundError:
            return None
        except (OSError, ValueError, TypeError) as error:
            logger.warning(f"Ignoring invalid YuanRong reporter state: {error}")
            return None

    def _write_previous_counters(self, counters: dict[str, float]) -> None:
        temporary_path = self.state_path.with_suffix(f".{os.getpid()}.tmp")
        with open(temporary_path, "w", encoding="utf-8") as state_file:
            json.dump({"version": 1, "counters": counters}, state_file)
        os.replace(temporary_path, self.state_path)


def start_yuanrong_resource_reporter(
    config: dict[str, object],
) -> YuanRongResourceReporter | None:
    log_path = str(config.get("yuanrong_resource_log_path", ""))
    enabled = bool(config.get("yuanrong_resource_metrics_enable", bool(log_path)))
    if not enabled or not log_path or int(config.get("device_id", -1)) >= 0:
        return None

    endpoint = f"{config.get('yuanrong_host', '')}:{config.get('yuanrong_port', '')}"
    reporter = YuanRongResourceReporter(
        log_path=log_path,
        endpoint=endpoint,
        interval_sec=float(config.get("yuanrong_resource_metrics_interval_sec", 15)),
    )
    _REPORTERS.append(reporter)
    reporter.start()
    return reporter
