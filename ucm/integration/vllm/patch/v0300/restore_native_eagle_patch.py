"""Restore native vLLM EAGLE last-block drop / Mamba one-block backoff for UCM.

vllm-ascend suppresses both the EAGLE last-block drop and the Mamba one-block
backoff for a pure ``kv_producer`` (``_skips_eagle_block_drop``). The exemption
assumes a PD prefill node that never consumes external KV: every content-hash
match is a verified local prompt block and ``num_external_computed_tokens`` is
always zero. A UCM connector violates both assumptions in every deployment -
it dumps *and* loads (PD-mixed or DP-disaggregated MultiConnector), and resumed
requests come back with generated tokens in their lookup prefix.

Suppressing the drop also removes the one-block backoff, which keeps the
exported mamba state page one page below the settled frontier. With async
scheduling that page is exported while it is still inside the in-flight window;
measured with UCM + MTP + preemption on Qwen3.8-27B, the suppressed backoff
drops GPQA accuracy to 73.2 and decays MTP acceptance from ~55% to 0%, while
restoring the backoff keeps it at 86-88 (with or without the drop).

The override only applies to processes that have a kv connector, so the
standalone (HBM-only) path keeps vllm-ascend's current behavior. It assumes any
process holding a ``kv_transfer_config`` while ``ENABLE_UCM_PATCH`` is on runs a
UCM connector: the exemption is intentional for a native PD prefill producer
that never loads, which such a process is not. This module is imported from
``apply_patch()``, i.e. it is enabled by ``ENABLE_UCM_PATCH`` together with every
other UCM patch and adds no environment variable of its own.
"""

import functools
import sys

from ucm.integration.vllm.patch.utils import when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)

_COORD_MODULE = "vllm_ascend.patch.platform.patch_kv_cache_coordinator"
_SPLIT_MODULE = "vllm_ascend.patch.platform.patch_mamba_block_aligned_split"
_MODULES = (_COORD_MODULE, _SPLIT_MODULE)


def _make_verdict(original):
    """Build a verdict that reports "do not suppress" for connector runs."""

    def _verdict(kv_transfer_config=None):
        if kv_transfer_config is None:
            return original(kv_transfer_config)
        return False

    _verdict._ucm_native = True
    _verdict._ucm_original = original
    return _verdict


def _install_verdict(mod) -> None:
    name = getattr(mod, "__name__", str(mod))
    original = getattr(mod, "_skips_eagle_block_drop", None)
    if original is None:
        logger.warning(
            f"UCM native eagle: {name} has no _skips_eagle_block_drop; "
            "nothing to override"
        )
        return
    if getattr(original, "_ucm_native", False):
        return
    mod._skips_eagle_block_drop = _make_verdict(original)
    logger.info(
        f"UCM native eagle: {name}._skips_eagle_block_drop overridden; "
        "EAGLE drop + mamba backoff follow native vLLM when a connector is present"
    )


def _install_coordinator_probe(mod) -> None:
    """Log the effective verdict once per coordinator, for field verification."""
    cls = getattr(mod, "AscendHybridKVCacheCoordinator", None)
    if cls is None or getattr(cls, "_ucm_native_eagle_probe", False):
        return
    original_init = cls.__init__

    @functools.wraps(original_init)
    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        logger.info(
            "UCM native eagle: coordinator.skips_eagle_block_drop="
            f"{getattr(self, 'skips_eagle_block_drop', None)} "
            "(connector runs expect False; standalone keeps the vllm-ascend default)"
        )

    cls.__init__ = __init__
    cls._ucm_native_eagle_probe = True


def _install(mod) -> None:
    _install_verdict(mod)
    if getattr(mod, "__name__", "") == _COORD_MODULE:
        _install_coordinator_probe(mod)


def _hook(mod) -> None:
    _install(mod)


# Two module-level bindings exist: the definition site in
# ``patch_kv_cache_coordinator`` and the ``from ... import`` copy in
# ``patch_mamba_block_aligned_split``. Both must be replaced, and either may be
# imported before or after this module, hence the hooks plus a direct pass for
# anything that is already imported.
for _module_name in _MODULES:
    when_imported(_module_name)(_hook)

for _module_name in _MODULES:
    _imported_module = sys.modules.get(_module_name)
    if _imported_module is not None:
        _install(_imported_module)
