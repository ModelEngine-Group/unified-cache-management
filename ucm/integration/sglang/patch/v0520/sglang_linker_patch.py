"""Apply the versioned UCM external-linker additions to SGLang modules."""

from __future__ import annotations

import importlib

from ucm.integration.sglang.patch.apply_patch import patch_package_for
from ucm.logger import init_logger

logger = init_logger(__name__)
EXPECTED_VERSION = "0.5.20"


def apply() -> None:
    import importlib.metadata

    version = importlib.metadata.version("sglang")
    if patch_package_for(version) != "v0520":
        raise RuntimeError(
            f"SGLang version mismatch: expected {EXPECTED_VERSION}, got {version}."
        )
    from . import server_args_patch

    server_args_patch.apply(allow_legacy_ucm=True)
    # The 0.5.20 linker module predates LayerWiseLoadCounter. Backport the
    # public API used by later compatible SGLang releases before UCM linkers
    # are imported.
    linker_module = importlib.import_module(
        "sglang.srt.mem_cache.unified_cache.unified_cache_linker"
    )
    if not hasattr(linker_module, "LayerWiseLoadCounter"):
        from .layerwise_load_counter import LayerWiseLoadCounter

        linker_module.LayerWiseLoadCounter = LayerWiseLoadCounter
    registry_patch = importlib.import_module(
        "ucm.integration.sglang.patch.v0520.registry_patch"
    )
    registry_patch.apply()
    logger.info("Applied UCM SGLang linker patch for %s.", EXPECTED_VERSION)
