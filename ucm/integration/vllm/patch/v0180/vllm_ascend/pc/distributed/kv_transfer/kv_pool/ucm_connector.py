from collections.abc import Iterable
from typing import Any, Optional

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    SupportsHMA,
)

from ucm.integration.vllm.patch.utils import patch_or_inject
from ucm.integration.vllm.ucm_connector import UCMConnector


def __init__(
    self: Any,
    vllm_config: Any,
    role: Any,
    kv_cache_config: Any = None,
) -> None:
    KVConnectorBase_V1.__init__(
        self,
        vllm_config=vllm_config,
        role=role,
        kv_cache_config=kv_cache_config,
    )
    assert vllm_config.kv_transfer_config is not None

    self._ucm_engine = UCMConnector(vllm_config, role, kv_cache_config)


def _get_ucm_delegate_for(self: Any, name: str) -> Any:
    """Return the UCM object that owns a reserved connector interface."""
    ucm_engine = self._ucm_engine
    if name not in type(ucm_engine).__dict__:
        inner_connector = getattr(ucm_engine, "connector", None)
        if inner_connector is not None:
            return inner_connector
    return ucm_engine


def _call_ucm_reserved_hook(
    self: Any,
    name: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    hook = getattr(self._get_ucm_delegate_for(name), name, None)
    if callable(hook):
        return hook(*args, **kwargs)
    return None


def shutdown(self: Any) -> None:
    self._call_ucm_reserved_hook("shutdown")


def set_host_xfer_buffer_ops(self: Any, copy_operation: Any) -> None:
    self._call_ucm_reserved_hook("set_host_xfer_buffer_ops", copy_operation)


def handle_preemptions(self: Any, kv_connector_metadata: Any) -> None:
    self._call_ucm_reserved_hook("handle_preemptions", kv_connector_metadata)


def request_finished_all_groups(
    self: Any,
    request: Any,
    block_ids: tuple[list[int], ...],
) -> tuple[bool, dict[str, Any] | None]:
    engine = self._ucm_engine
    if hasattr(type(engine), "request_finished_all_groups"):
        return engine.request_finished_all_groups(request, block_ids)
    if block_ids:
        return engine.request_finished(request, block_ids[0])
    return engine.request_finished(request, [])


def build_connector_worker_meta(self: Any) -> Any:
    return self._call_ucm_reserved_hook("build_connector_worker_meta")


def take_events(self: Any) -> Iterable[Any]:
    events = self._call_ucm_reserved_hook("take_events")
    return () if events is None else events


def get_kv_connector_stats(self: Any) -> Optional[Any]:
    return self._call_ucm_reserved_hook("get_kv_connector_stats")


def get_kv_connector_kv_cache_events(self: Any) -> Optional[Any]:
    return self._call_ucm_reserved_hook("get_kv_connector_kv_cache_events")


def get_handshake_metadata(self: Any) -> Any:
    return self._call_ucm_reserved_hook("get_handshake_metadata")


def set_xfer_handshake_metadata(self: Any, metadata: dict[int, Any]) -> None:
    self._call_ucm_reserved_hook("set_xfer_handshake_metadata", metadata)


def get_finished_count(self: Any) -> int | None:
    return self._call_ucm_reserved_hook("get_finished_count")


def reset_cache(self: Any) -> bool | None:
    return self._call_ucm_reserved_hook("reset_cache")


def build_kv_connector_stats(cls: type, data: dict[str, Any] | None = None) -> Any:
    return UCMConnector.build_kv_connector_stats(data)


def build_prom_metrics(
    cls: type,
    vllm_config: Any,
    metric_types: dict[type[Any], type[Any]],
    labelnames: list[str],
    per_engine_labelvalues: dict[int, list[object]],
) -> Any:
    return UCMConnector.build_prom_metrics(
        vllm_config,
        metric_types,
        labelnames,
        per_engine_labelvalues,
    )


def patch_ucm_connector_v1(mod: Any) -> None:
    target_cls = mod.UCMConnectorV1

    SupportsHMA.register(target_cls)

    patch_or_inject(target_cls, "__init__", __init__)
    patch_or_inject(target_cls, "_get_ucm_delegate_for", _get_ucm_delegate_for)
    patch_or_inject(target_cls, "_call_ucm_reserved_hook", _call_ucm_reserved_hook)
    patch_or_inject(target_cls, "shutdown", shutdown)
    patch_or_inject(target_cls, "set_host_xfer_buffer_ops", set_host_xfer_buffer_ops)
    patch_or_inject(target_cls, "handle_preemptions", handle_preemptions)
    patch_or_inject(
        target_cls,
        "request_finished_all_groups",
        request_finished_all_groups,
    )
    patch_or_inject(
        target_cls,
        "build_connector_worker_meta",
        build_connector_worker_meta,
    )
    patch_or_inject(target_cls, "take_events", take_events)
    patch_or_inject(target_cls, "get_kv_connector_stats", get_kv_connector_stats)
    patch_or_inject(
        target_cls,
        "get_kv_connector_kv_cache_events",
        get_kv_connector_kv_cache_events,
    )
    patch_or_inject(target_cls, "get_handshake_metadata", get_handshake_metadata)
    patch_or_inject(
        target_cls,
        "set_xfer_handshake_metadata",
        set_xfer_handshake_metadata,
    )
    patch_or_inject(target_cls, "get_finished_count", get_finished_count)
    patch_or_inject(target_cls, "reset_cache", reset_cache)
    patch_or_inject(
        target_cls,
        "build_kv_connector_stats",
        classmethod(build_kv_connector_stats),
    )
    patch_or_inject(
        target_cls,
        "build_prom_metrics",
        classmethod(build_prom_metrics),
    )
