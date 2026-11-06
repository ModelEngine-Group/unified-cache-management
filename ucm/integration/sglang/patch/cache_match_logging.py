"""Log request-level matches after SGLang synchronizes external-hit boundaries.

Installed by UcmDirectLinker for both native and backported linker support;
this does not require enabling the version-specific compatibility patches.
"""

import importlib
import logging
from functools import wraps

from ucm.logger import init_logger

logger = init_logger(__name__)


def install_cache_match_logging(linker_type):
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

    @wraps(original)
    def match(self, key, req, result):
        matched = original(self, key, req, result)
        linker = self.cache_linker
        if (
            isinstance(linker, linker_type)
            and linker._cache_match_log_rank
            and logger.isEnabledFor(logging.INFO)
        ):
            try:
                repeated = getattr(req, "_ucm_cache_match_logged", False)
                if repeated and not logger.isEnabledFor(logging.DEBUG):
                    return matched
                # match returns its input unchanged when no external prefix is
                # accepted. Existing host-cache hits must not count as UCM hits.
                external_tokens = 0 if matched is result else matched.host_hit_length
                message = (
                    "request_id=%s total_tokens=%d "
                    "hit_hbm_tokens=%d hit_external_tokens=%d "
                    "dp_rank=%d attn_dp_rank=%d tp_rank=%d"
                )
                values = (
                    req.rid,
                    len(req.origin_input_ids),
                    result.device_prefix_len,
                    external_tokens,
                    linker._dp_rank,
                    linker._attn_dp_rank,
                    linker._tp_rank,
                )
                if repeated:
                    logger.debug(
                        "[CacheRematch] " + message + " sequence_tokens=%d",
                        *values,
                        len(req.origin_input_ids) + len(req.output_ids),
                    )
                else:
                    # Keep state on the request, not in an unbounded rid cache.
                    # A later match may see externally loaded KV in HBM; it
                    # must not replace the initial input-cache attribution.
                    req._ucm_cache_match_logged = True
                    logger.info("[CacheMatch] " + message, *values)
            except (AttributeError, TypeError):
                # Logging must not invalidate a successful match on API drift.
                logger.warning_once(
                    "SGLang request/match fields unavailable; cache-match log skipped."
                )
        return matched

    match._ucm_cache_match_logging = True
    wrapper.match = match
