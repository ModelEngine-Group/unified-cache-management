"""Consumer wait timing installed by the opt-in SGLang patch entry point."""

import importlib
from functools import wraps
from importlib.metadata import PackageNotFoundError, version

from ucm.integration.sglang.metrics import (
    counter_metrics_enabled,
    finish_timing,
    start_timing,
)
from ucm.logger import init_logger

logger = init_logger(__name__)


def install_counter_metrics():
    """Time the native counter's wait_until without replacing the counter class."""
    try:
        sglang_version = version("sglang")
    except PackageNotFoundError:
        logger.warning("Cannot detect SGLang version; skipping native counter metrics.")
        return
    # Older releases use the already-decorated compatibility counter.
    if not _supports_native_counter_metrics(sglang_version):
        return
    try:
        module = importlib.import_module(
            "sglang.srt.mem_cache.unified_cache.unified_cache_linker"
        )
        counter = module.LayerWiseLoadCounter
        original = counter.wait_until
    except (ImportError, AttributeError):
        logger.warning("SGLang native counter unavailable; counter metrics skipped.")
        return
    if getattr(original, "_ucm_wait_metrics", False):
        return
    counter.wait_until = record_layer_wait(original)


def _supports_native_counter_metrics(sglang_version):
    # Match compatibility patch selection's rc/post/build normalization.
    release = sglang_version.strip().split("+", 1)[0]
    release = release.split(".post", 1)[0].split("rc", 1)[0]
    parts = release.split(".")
    if len(parts) != 3 or not all(part.isdecimal() for part in parts):
        return False
    return tuple(map(int, parts)) >= (0, 5, 22)


def record_layer_wait(func):
    @wraps(func)
    def wait_until(self, threshold):
        if not counter_metrics_enabled():
            return func(self, threshold)
        futures = self.futures.get(self.consumer_index)
        # Preserve inactive/invalid-call behavior without recording a load wait.
        if futures is None or not 0 <= threshold < len(futures):
            return func(self, threshold)
        timings = {}
        name = f"load_layer_wait_duration_ms_{threshold}"
        start_timing(timings, name, True)
        try:
            return func(self, threshold)
        finally:
            finish_timing(timings, name)

    wait_until._ucm_wait_metrics = True
    return wait_until
