"""SGLang synchronous call timers and worker phase metrics.

Interface durations measure only Python call lifetime. Worker load-layer and
offload-I/O durations end at Wait/cleanup, including ordered-reaping delay.
Legacy connector I/O duration/bandwidth definitions remain available for HiCache;
the direct adapter emits physical counts/bytes/pages, not those timing samples.
"""

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)
_active_metrics_logger = None


def counter_metrics_enabled():
    """Counter patches sample only while this worker's UCM metrics logger is active."""
    return _active_metrics_logger is not None


def create_metrics_logger(*, enabled, model_name, ranks, launch_config):
    """Register once per worker, before Store construction, using SGLang's dir."""
    global _active_metrics_logger
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
    metrics_logger = PrometheusStatsLogger(model_name or "", worker_id, config=config)
    _active_metrics_logger = metrics_logger
    return metrics_logger


def shutdown_metrics_logger(metrics_logger):
    """Stop polling, then export the last deltas after all Store work has drained."""
    global _active_metrics_logger
    if _active_metrics_logger is metrics_logger:
        _active_metrics_logger = None
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


def _record_interface_duration(func):
    """Measure synchronous call time only, including exceptional returns."""
    owner, method = func.__qualname__.split(".")[-2:]
    scope = "adapter" if owner == "UcmDirectStoreAdapter" else "linker"
    name = f"sglang_{scope}_interface_{method.lstrip('_')}_duration_ms"

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        enabled = getattr(
            self, "_metrics_enabled", getattr(self, "metrics_enabled", False)
        )
        if not enabled:
            return func(self, *args, **kwargs)
        started = time.perf_counter()
        try:
            return func(self, *args, **kwargs)
        finally:
            record_metrics({name: max(0.0, time.perf_counter() - started) * 1000})

    return wrapper


@contextmanager
def measure_phase(enabled, name, timings=None):
    """Keep local timestamps and emit one phase sample even on failure."""
    if not enabled:
        yield
        return
    started = time.perf_counter()
    try:
        yield
    finally:
        ended = time.perf_counter()
        if timings is not None:
            timings[name] = {"started": started, "ended": ended}
        record_metrics(
            {f"sglang_linker_{name}_duration_ms": max(0.0, ended - started) * 1000}
        )


def start_timing(timings, name, enabled):
    if enabled:
        timings[name] = time.perf_counter()


def finish_timing(timings, name):
    started = timings.pop(name, None)
    if started is not None:
        ended = time.perf_counter()
        record_metrics({f"sglang_linker_{name}": max(0.0, ended - started) * 1000})
