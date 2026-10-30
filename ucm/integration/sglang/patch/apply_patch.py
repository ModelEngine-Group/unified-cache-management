"""Select and apply the UCM patch matching the installed SGLang version."""

from __future__ import annotations

import importlib
import importlib.metadata
import os

from ucm.logger import init_logger

logger = init_logger(__name__)


PATCH_PACKAGE_BY_VERSION = {
    "0.5.20": "v0520",
    "0.5.21": "v0521",
    "0.5.22": "v0521",
}


def patch_package_for(version: str) -> str | None:
    """Select a release's patch, normalizing rc/post/build suffixes like vLLM."""
    release = version.strip().split("+", 1)[0]
    release = release.split(".post", 1)[0].split("rc", 1)[0]
    return PATCH_PACKAGE_BY_VERSION.get(release)


def _has_native_external_linker() -> bool:
    """Return whether this SGLang install includes PR #40595's linker path."""
    try:
        from sglang.srt.arg_groups.arg_utils import record_fields
        from sglang.srt.arg_groups.fields.memory import Memory
        from sglang.srt.mem_cache import registry
        from sglang.srt.mem_cache.unified_cache.linker_config import (
            UnifiedCacheLinkerConfig,
        )
        from sglang.srt.mem_cache.unified_cache.linker_factory import (
            create_unified_cache_linker,
        )
    except (ImportError, AttributeError):
        return False
    memory_fields = {field.name for field in record_fields(Memory)}
    return (
        "unified_cache_external_linker_config" in memory_fields
        and callable(getattr(registry, "create_unified_radix_cache", None))
        and callable(UnifiedCacheLinkerConfig.from_dict)
        and callable(create_unified_cache_linker)
    )


def apply_all_patches() -> None:
    if os.environ.get("ENABLE_UCM_PATCH", "").lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return
    try:
        version = importlib.metadata.version("sglang")
    except importlib.metadata.PackageNotFoundError:
        logger.warning(
            "Cannot detect the installed SGLang version; skipping UCM patch."
        )
        return
    expected = os.environ.get("UCM_SGLANG_VERSION", version)
    if expected != version:
        raise RuntimeError(
            f"SGLang version mismatch: expected {expected}, installed {version}."
        )
    if _has_native_external_linker():
        logger.info(
            "SGLang %s has native out-of-tree linker support; skipping UCM patch.",
            version,
        )
        return
    package_suffix = patch_package_for(version)
    if package_suffix is None:
        logger.warning(
            "No UCM compatibility patch or native external linker support for "
            "SGLang %s; skipping UCM patch. Existing SGLang backends remain available.",
            version,
        )
        return
    package = f"ucm.integration.sglang.patch.{package_suffix}"
    try:
        module = importlib.import_module(package)
        module.apply()
    except Exception as error:
        raise RuntimeError(
            f"Failed to apply UCM SGLang patch for {expected}."
        ) from error
    logger.info("UCM SGLang patch applied for version %s.", expected)
