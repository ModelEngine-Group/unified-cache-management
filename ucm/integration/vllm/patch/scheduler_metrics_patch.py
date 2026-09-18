from functools import wraps
from typing import Any

from ucm.integration.vllm.patch.utils import when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


def patch_recompute_scheduler_cls(scheduler_cls: type) -> None:
    original_make_stats = scheduler_cls.make_stats
    if getattr(original_make_stats, "_ucm_collects_scheduler_kv_stats", False):
        return

    @wraps(original_make_stats)
    def make_stats(
        self: Any,
        spec_decoding_stats: Any = None,
        kv_connector_stats: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if kv_connector_stats is None:
            kv_connector_stats = _scheduler_kv_connector_stats(self)
        return original_make_stats(
            self,
            spec_decoding_stats,
            kv_connector_stats,
            *args,
            **kwargs,
        )

    make_stats._ucm_collects_scheduler_kv_stats = True
    scheduler_cls.make_stats = make_stats


def _scheduler_kv_connector_stats(scheduler: Any) -> Any:
    connector = getattr(scheduler, "connector", None)
    if connector is None:
        return None
    get_stats = getattr(connector, "get_kv_connector_stats", None)
    if not callable(get_stats):
        return None
    stats = get_stats()
    if stats is None:
        return None
    is_empty = getattr(stats, "is_empty", None)
    if callable(is_empty) and is_empty():
        return None
    return stats


@when_imported("vllm_ascend.core.recompute_scheduler")
def patch_recompute_scheduler(mod: Any) -> None:
    logger.debug(f"Patched {mod} called")
    patch_recompute_scheduler_cls(mod.RecomputeScheduler)
