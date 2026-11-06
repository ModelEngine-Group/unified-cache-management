"""Per-worker storage-call observations; export uses the existing UCM logger."""

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


def create_metrics_logger(*, enabled, model_name, ranks, launch_config):
    """Register once per worker, before Store construction, using SGLang's dir."""
    if not enabled:
        return None
    from ucm.metrics_config import (
        consumer_enabled,
        get_metric_definitions,
        load_metrics_config,
        metrics_enabled,
    )

    if not metrics_enabled(launch_config):
        return None
    config_path = launch_config.get("metrics_config_path")
    if config_path:
        config = load_metrics_config(config_path)
    else:
        from ucm.integration.sglang.default_metrics_config import (
            get_default_sglang_metrics_config,
        )

        config = get_default_sglang_metrics_config()
    if not get_metric_definitions(config):
        raise ValueError(f"No UCM metrics defined in {config_path}")
    if not consumer_enabled(config, "multiproc") or consumer_enabled(
        config, "vllm_connector"
    ):
        raise ValueError(
            "SGLang metrics config must enable multiproc and disable vllm_connector"
        )
    directory = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    if not directory or not Path(directory).is_dir():
        raise RuntimeError(
            "SGLang must set PROMETHEUS_MULTIPROC_DIR before UCM metrics"
        )
    from prometheus_client import values

    if not getattr(values.ValueClass, "_multiprocess", False):
        raise RuntimeError(
            "PROMETHEUS_MULTIPROC_DIR must be set before importing prometheus_client"
        )
    from ucm.observability import PrometheusStatsLogger, _metric_mappings

    if _metric_mappings:
        raise RuntimeError(
            "UCM metrics already registered in this worker; only one metrics-enabled "
            "connector is supported. Restart the worker before recreating it."
        )
    worker_id = "_".join(
        f"{name}{ranks.get(name + '_rank', 0)}"
        for name in ("dp", "tp", "pp", "attn_cp")
    )
    return PrometheusStatsLogger(model_name or "", worker_id, config=config)


def shutdown_metrics_logger(metrics_logger):
    """Stop polling, then export the last deltas after all Store work has drained."""
    if metrics_logger is None:
        return
    metrics_logger.shutdown()
    try:
        dispatcher = metrics_logger.metrics_dispatcher
        dispatcher.drain_to_consumers()
        metrics_logger.update_stats(*dispatcher.get_stats_and_clear("multiproc"))
    except Exception:
        logger.warning("Failed to flush SGLang UCM metrics", exc_info=True)


def record_metrics(values):
    """Business threads only update accumulators; export remains on the logger."""
    try:
        from ucm.shared.metrics import ucmmetrics

        ucmmetrics.update_stats(values)
    except Exception:
        logger.warning("Failed to record SGLang UCM metrics", exc_info=True)


@dataclass
class AsyncIoObservation:
    """One actual shard submission, retained until Wait confirms final status."""

    is_set: bool
    pages: int
    byte_count: int
    started: float
    submitted: bool = False
    failed: bool = False
    check_failed: bool = False
    finished: bool = False

    @classmethod
    def begin(cls, enabled, is_set, pages, shard_size):
        if not enabled or pages == 0:
            return None
        return cls(is_set, pages, pages * shard_size, time.perf_counter())

    def fail_check(self):
        self.failed = True
        if not self.check_failed:
            self.check_failed = True
            operation = "dump" if self.is_set else "load"
            record_metrics({f"sglang_linker_{operation}_check_errors_total": 1})

    def finish(self, failed=False):
        if self.finished:
            return
        self.finished = True
        elapsed = max(time.perf_counter() - self.started, 0.0)
        prefix = "sglang_connector_dump" if self.is_set else "sglang_connector_load"
        values = {
            f"{prefix}_requests_total": 1,
            f"{prefix}_duration_ms": elapsed * 1000,
        }
        if failed or self.failed:
            phase = "wait" if self.submitted else "submit"
            values[f"{prefix}_{phase}_errors_total"] = 1
        else:
            values[f"{prefix}_bytes_total"] = self.byte_count
            values[f"{prefix}_pages"] = self.pages
            if elapsed > 0:
                values[f"{prefix}_bandwidth_gbps"] = self.byte_count / elapsed / 1e9
        record_metrics(values)


@dataclass
class LinkerOperation:
    """One load batch or owner offload node, from enqueue to worker completion."""

    kind: str
    queued_at: float
    finished: bool = False

    @classmethod
    def begin(cls, enabled, kind):
        if not enabled:
            return None
        observation = cls(kind, time.perf_counter())
        record_metrics({f"sglang_linker_{kind}_operations_total": 1})
        return observation

    def start(self):
        record_metrics(
            {
                f"sglang_linker_{self.kind}_started_total": 1,
                f"sglang_linker_{self.kind}_queue_duration_ms": max(
                    time.perf_counter() - self.queued_at, 0.0
                )
                * 1000,
            }
        )

    def finish(self, success):
        if self.finished:
            return
        self.finished = True
        values = {
            f"sglang_linker_{self.kind}_completed_total": 1,
            f"sglang_linker_{self.kind}_duration_ms": max(
                time.perf_counter() - self.queued_at, 0.0
            )
            * 1000,
        }
        if not success:
            values[f"sglang_linker_{self.kind}_errors_total"] = 1
        record_metrics(values)


@dataclass
class IoObservation:
    submitted: bool = False
    failed: bool = False


@contextmanager
def measure_io(enabled: bool, is_set: bool, pages: int, byte_count: Callable[[], int]):
    """Observe one physical v1 call or v2 pool call, never an inference batch.

    Duration covers submission through wait, including failed attempts. Page and
    bandwidth histograms and byte counters contain successful transfers only.
    Callers mark caught wait failures so existing return/exception behavior is
    preserved. No native metrics import, clock read, or byte calculation occurs
    when disabled. No metric collection/draining happens on the business thread.
    """
    observation = IoObservation()
    if not enabled or pages == 0:
        yield observation
        return

    started = time.perf_counter()
    try:
        yield observation
    except BaseException:
        observation.failed = True
        raise
    finally:
        elapsed = max(time.perf_counter() - started, 0.0)
        prefix = "sglang_connector_dump" if is_set else "sglang_connector_load"
        values = {
            f"{prefix}_requests_total": 1,
            f"{prefix}_duration_ms": elapsed * 1000,
        }
        if observation.failed:
            phase = "wait" if observation.submitted else "submit"
            values[f"{prefix}_{phase}_errors_total"] = 1
        # Observability must not change the result of a storage operation.
        try:
            if not observation.failed:
                size = int(byte_count())
                if size < 0:
                    raise ValueError("negative storage transfer size")
                values[f"{prefix}_bytes_total"] = size
                values[f"{prefix}_pages"] = pages
                if elapsed > 0:
                    values[f"{prefix}_bandwidth_gbps"] = size / elapsed / 1e9
            from ucm.shared.metrics import ucmmetrics

            ucmmetrics.update_stats(values)
        except Exception:
            logger.warning(
                "Failed to record SGLang connector I/O metrics", exc_info=True
            )
