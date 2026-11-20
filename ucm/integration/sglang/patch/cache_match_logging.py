"""Log request-level matches after SGLang synchronizes external-hit boundaries.

Installed by the opt-in patch entry point for both native and backported
linker support. The linker itself does not install or depend on this patch.
"""

import importlib
from functools import wraps

from ucm.logger import init_logger

logger = init_logger(__name__)


def install_cache_match_logging() -> None:
    try:
        module = importlib.import_module(
            "sglang.srt.mem_cache.unified_cache.unified_cache_linker"
        )
        wrapper = module.UnifiedCacheLinkerWrapper
        original = wrapper.match
    except (ImportError, AttributeError):
        logger.warning("SGLang match wrapper unavailable; cache-match logging skipped.")
        return
    if getattr(original, "_ucm_cache_match_logging", False):
        return
    linker_type = importlib.import_module(
        "ucm.integration.sglang.ucm_direct_linker"
    ).UcmDirectLinker
    get_parallel = importlib.import_module("sglang.srt.runtime_context").get_parallel

    @wraps(original)
    def match(self, key, req, result):
        matched = original(self, key, req, result)
        linker = self.cache_linker
        if isinstance(linker, linker_type):
            try:
                parallel = get_parallel()
                if (
                    linker._tp_rank != 0
                    or parallel.attn_cp_rank != 0
                    or parallel.pp_rank != 0
                ):
                    return matched
                if getattr(req, "_ucm_cache_match_logged", False):
                    return matched
                # match returns its input unchanged when no external prefix is
                # accepted. Existing host-cache hits must not count as UCM hits.
                external_tokens = 0 if matched is result else matched.host_hit_length
                message = (
                    "[dp_rank=%d attn_dp_rank=%d tp_rank=%d] [CacheMatch] "
                    "request_id=%s total_tokens=%d "
                    "hit_hbm_tokens=%d hit_external_tokens=%d"
                )
                values = (
                    linker._dp_rank,
                    linker._attn_dp_rank,
                    linker._tp_rank,
                    req.rid,
                    len(req.origin_input_ids),
                    result.device_prefix_len,
                    external_tokens,
                )
                # info_once deduplicates full arguments, so changed hit counts
                # would log again after external KV has been loaded into HBM.
                # Keep the first-match marker on the request instead.
                req._ucm_cache_match_logged = True
                logger.info(message, *values)
            except (AttributeError, TypeError):
                # Logging must not invalidate a successful match on API drift.
                logger.warning_once(
                    "SGLang request/match fields unavailable; cache-match log skipped."
                )
        return matched

    match._ucm_cache_match_logging = True
    wrapper.match = match
