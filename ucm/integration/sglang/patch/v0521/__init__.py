"""Shared runtime patch for SGLang 0.5.21 and 0.5.22."""

from __future__ import annotations

import importlib.metadata
import inspect

from ucm.integration.sglang.patch.apply_patch import patch_package_for
from ucm.integration.sglang.patch.v0520 import server_args_patch
from ucm.logger import init_logger

from . import registry_patch

logger = init_logger(__name__)


def apply() -> None:
    version = importlib.metadata.version("sglang")
    if patch_package_for(version) != "v0521":
        raise RuntimeError(
            "The shared v0521 patch requires SGLang 0.5.21 or 0.5.22; "
            f"got {version}."
        )
    # Fail early if a supported build changes the extension points used here.
    from sglang.srt import server_args
    from sglang.srt.mem_cache import registry
    from sglang.srt.mem_cache.unified_cache import unified_cache_linker

    if not callable(getattr(server_args, "add_cli_args_from_dataclass", None)):
        raise RuntimeError("SGLang ServerArgs CLI registration API has changed.")
    if not callable(getattr(server_args.ServerArgs, "from_cli_args", None)):
        raise RuntimeError("SGLang ServerArgs.from_cli_args API is missing.")
    factory = getattr(registry, "create_unified_radix_cache", None)
    if not callable(factory):
        raise RuntimeError("SGLang unified radix cache factory API is missing.")
    parameters = inspect.signature(factory).parameters
    if "ctx" not in parameters or "cache_class" not in parameters:
        raise RuntimeError(
            "SGLang unified radix cache factory signature changed: "
            f"{inspect.signature(factory)}"
        )
    if not hasattr(unified_cache_linker, "UnifiedCacheLinker"):
        raise RuntimeError(
            "SGLang unified cache linker API is missing UnifiedCacheLinker."
        )
    # 0.5.21 lacks this class; 0.5.22 already provides it.
    if not hasattr(unified_cache_linker, "LayerWiseLoadCounter"):
        from ucm.integration.sglang.patch.v0520.layerwise_load_counter import (
            LayerWiseLoadCounter,
        )

        unified_cache_linker.LayerWiseLoadCounter = LayerWiseLoadCounter
    server_args_patch.apply()
    registry_patch.apply()
    logger.info("Applied UCM SGLang dynamic linker patch for %s.", version)
