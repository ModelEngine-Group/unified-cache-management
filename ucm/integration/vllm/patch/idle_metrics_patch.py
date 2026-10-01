"""Collect connector metrics through utility RPCs while an engine is idle."""

from __future__ import annotations

import asyncio
import math
import os
import sys
import time
from collections.abc import Iterator
from functools import wraps
from types import ModuleType
from typing import TYPE_CHECKING, Any

from ucm.integration.vllm.patch.utils import when_imported
from ucm.logger import init_logger

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
    from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
        KVConnectorProm,
        KVConnectorPromMetrics,
        KVConnectorStats,
    )
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.engine.core import EngineCoreProc
    from vllm.v1.engine.core_client import AsyncMPClient
    from vllm.v1.worker.worker_base import WorkerBase

logger = init_logger(__name__)
IdleMetrics = tuple[int, dict[str, Any]]


def _read_connector_stats(
    connector: KVConnectorBase_V1 | None,
) -> KVConnectorStats | None:
    if connector is None:
        return None
    try:
        return connector.get_kv_connector_stats()
    except Exception:
        logger.warning("Failed to collect idle KV connector metrics", exc_info=True)
        return None


def _patch_worker(module: ModuleType) -> None:
    def ucm_get_kv_connector_stats(self: WorkerBase) -> KVConnectorStats | None:
        from vllm.distributed.kv_transfer import (
            get_kv_transfer_group,
            has_kv_transfer_group,
        )

        if not has_kv_transfer_group():
            return None
        return _read_connector_stats(get_kv_transfer_group())

    # Both the vLLM GPU Worker and vllm-ascend NPUWorker inherit WorkerBase.
    module.WorkerBase.ucm_get_kv_connector_stats = ucm_get_kv_connector_stats


def _patch_engine_core(module: ModuleType) -> None:
    def ucm_collect_idle_metrics(
        self: EngineCoreProc, interval_seconds: float
    ) -> IdleMetrics | None:
        import msgspec

        if not math.isfinite(interval_seconds) or interval_seconds <= 0:
            return None
        if self.has_work():
            return None
        now = time.monotonic()
        if now - self._ucm_last_idle_metrics < interval_seconds:
            return None
        connector = self.scheduler.get_kv_connector()
        if connector is None:
            return None
        self._ucm_last_idle_metrics = now
        # Stay on the engine's serialized RPC path; a timeout can leave worker
        # replies queued for the next executor operation.
        snapshots = self.collective_rpc("ucm_get_kv_connector_stats")
        snapshots.append(_read_connector_stats(connector))
        combined = None
        for stats in snapshots:
            if stats is None or stats.is_empty():
                continue
            combined = stats if combined is None else combined.aggregate(stats)
        if combined is None:
            return None
        # Newer MultiConnector stats flatten child payloads through to_dict().
        to_dict = getattr(combined, "to_dict", None)
        data = to_dict() if callable(to_dict) else combined.data
        return self.engine_index, msgspec.to_builtins(data)

    if "ucm_collect_idle_metrics" not in module.EngineCoreProc.__dict__:
        module.EngineCoreProc._ucm_last_idle_metrics = float("-inf")
        module.EngineCoreProc.ucm_collect_idle_metrics = ucm_collect_idle_metrics


def _patch_engine_client(module: ModuleType) -> None:
    async def ucm_collect_idle_metrics_async(
        self: AsyncMPClient, interval_seconds: float
    ) -> list[IdleMetrics | None | BaseException]:
        # DP's public utility wrapper returns only the first engine's result.
        return await asyncio.gather(
            *(
                self._call_utility_async(
                    "ucm_collect_idle_metrics", interval_seconds, engine=engine
                )
                for engine in self.core_engines
            ),
            return_exceptions=True,
        )

    module.AsyncMPClient.ucm_collect_idle_metrics_async = ucm_collect_idle_metrics_async


def _patch_async_llm(module: ModuleType) -> None:
    if os.getenv("ENABLE_UCM_IDLE_METRICS", "1").strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return
    if getattr(module.AsyncLLM.do_log_stats, "_ucm_idle_metrics_patched", False):
        return
    original = module.AsyncLLM.do_log_stats

    @wraps(original)
    async def do_log_stats(self: AsyncLLM) -> None:
        await original(self)
        if self.logger_manager is None or self._ucm_idle_metrics_running:
            return

        from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
            MultiKVConnectorPromMetrics,
        )
        from vllm.v1.metrics.loggers import PrometheusStatLogger

        from ucm.integration.vllm.metrics import UCMPromMetrics

        def collection_intervals(
            prom_metrics: KVConnectorPromMetrics | None,
        ) -> Iterator[float]:
            if isinstance(prom_metrics, UCMPromMetrics):
                yield prom_metrics.idle_collection_interval
            elif isinstance(prom_metrics, MultiKVConnectorPromMetrics):
                for child in prom_metrics._prom_metrics.values():
                    yield from collection_intervals(child)

        sinks: list[KVConnectorProm] = []
        intervals: list[float] = []
        for stat_logger in self.logger_manager.stat_loggers:
            if isinstance(stat_logger, PrometheusStatLogger):
                sink = stat_logger.kv_connector_prom
                sink_intervals = list(collection_intervals(sink.prom_metrics))
                if sink_intervals:
                    sinks.append(sink)
                    intervals.extend(sink_intervals)
        if not sinks:
            return
        self._ucm_idle_metrics_running = True
        try:
            results = await self.engine_core.ucm_collect_idle_metrics_async(
                min(intervals)
            )
            for result in results:
                if isinstance(result, BaseException):
                    logger.warning("Idle UCM metrics utility failed: %s", result)
                    continue
                if result is not None:
                    engine_index, data = result
                    for sink in sinks:
                        sink.observe(data, engine_index)
        except Exception:
            logger.warning("Failed to publish idle UCM metrics", exc_info=True)
        finally:
            self._ucm_idle_metrics_running = False

    do_log_stats._ucm_idle_metrics_patched = True
    module.AsyncLLM._ucm_idle_metrics_running = False
    module.AsyncLLM.do_log_stats = do_log_stats


for _module_name, _patch in (
    ("vllm.v1.worker.worker_base", _patch_worker),
    ("vllm.v1.engine.core", _patch_engine_core),
    ("vllm.v1.engine.core_client", _patch_engine_client),
    ("vllm.v1.engine.async_llm", _patch_async_llm),
):
    # Other UCM hooks may already have marked an imported module as patched.
    if _module_name in sys.modules:
        _patch(sys.modules[_module_name])
    else:
        when_imported(_module_name)(_patch)
