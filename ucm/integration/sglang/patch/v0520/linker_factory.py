"""Import and construct a dynamic linker in the cache-owning worker."""

from __future__ import annotations

import importlib
import inspect
from copy import deepcopy

from .linker_config import UnifiedCacheLinkerConfig


def create_dynamic_linker(server_args, params, *, components, forward_stream=None):
    config = UnifiedCacheLinkerConfig.from_dict(
        server_args.__dict__.get("_ucm_linker_config")
    )
    module = importlib.import_module(config.linker_module_path)
    linker_cls = getattr(module, config.linker, None)
    from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
        UnifiedCacheLinker,
    )

    if not inspect.isclass(linker_cls) or not issubclass(
        linker_cls, UnifiedCacheLinker
    ):
        raise ValueError(
            f"{config.linker_module_path}.{config.linker} must be a "
            "UnifiedCacheLinker subclass."
        )
    if inspect.isabstract(linker_cls):
        raise ValueError(
            f"{config.linker_module_path}.{config.linker} must implement "
            f"all UnifiedCacheLinker methods: {sorted(linker_cls.__abstractmethods__)}"
        )
    return linker_cls(
        server_args,
        params,
        components=components,
        extra_config=deepcopy(config.linker_extra_config),
        forward_stream=forward_stream,
    )
