#
# MIT License
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ucm.logger import init_logger
from ucm.shared.metrics import ucmmetrics
from ucm.shared.metrics.resource_reporter import (
    FileResourceMetricsReporter,
    counter_deltas,
)

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


class YuanRongResourceReporter(FileResourceMetricsReporter):
    def __init__(
        self,
        log_path: str,
        endpoint: str,
        interval_sec: float = 15.0,
        shared_memory_dir: str = "/dev/shm",
    ):
        super().__init__(
            log_path=log_path,
            reporter_name="YuanRong",
            identity=f"{endpoint}|{Path(log_path).resolve()}",
            interval_sec=interval_sec,
            shared_memory_dir=shared_memory_dir,
        )

    def _read_previous_counters(self) -> dict[str, float] | None:
        try:
            state = self._read_previous_state()
            if state is None:
                return None
            return {
                name: float(value) for name, value in state.get("counters", {}).items()
            }
        except (OSError, ValueError, TypeError) as error:
            logger.warning(f"Ignoring invalid YuanRong reporter state: {error}")
            return None

    def _collect_once(self) -> None:
        line = self._read_latest_complete_line()
        snapshot = parse_yuanrong_resource_snapshot(line)
        previous = self._read_previous_counters()
        updates = snapshot.gauges | counter_deltas(snapshot.counters, previous)
        ucmmetrics.update_stats(updates)
        self._write_previous_state({"version": 1, "counters": snapshot.counters})


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
