from ucm.integration.vllm.patch.utils import patch_or_inject, when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


@when_imported("vllm.v1.metrics.stats")
def patch_stats(mod):
    logger.debug(f"Patched {mod} called")

    from ucm.integration.vllm.patch.v0180.vllm.pc.metrics import stats

    patch_or_inject(
        mod.PromptTokenStats,
        "update_from_output",
        stats.update_from_output,
    )
