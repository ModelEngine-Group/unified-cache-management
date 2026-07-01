# UCM patch for vllm-ascend 0.19.1:
# Remove bind_memory call from bind_threads to avoid excessive MLA model initialization latency

from ucm.integration.vllm.patch.utils import patch_or_inject, when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


@when_imported("vllm_ascend.cpu_binding")
def patch_cpu_binding(mod):
    logger.debug(f"Patched {mod} called")

    from ucm.integration.vllm.patch.v0191.vllm_ascend.cpu_binding import (
        bind_threads,
    )

    if not hasattr(mod.CpuAlloc, "bind_threads"):
        logger.warning("Skip CPU binding patch: CpuAlloc.bind_threads is missing")
        return

    patch_or_inject(
        mod.CpuAlloc,
        "bind_threads",
        bind_threads,
    )
    logger.info(
        "UCM CPU binding patch applied: CpuAlloc.bind_threads (removed bind_memory)"
    )
