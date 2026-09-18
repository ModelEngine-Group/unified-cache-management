"""Disable CpuAlloc.bind_memory for all vLLM-Ascend versions.

Hooks into vllm_ascend.cpu_binding module import and replaces
CpuAlloc.bind_memory with a no-op. Must be registered before
cpu_binding_patch so bind_memory is patched before bind_threads.

Version-agnostic: no version detection required.
"""

from ucm.integration.vllm.patch.utils import when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


@when_imported("vllm_ascend.cpu_binding")
def patch_bind_memory(mod):
    logger.debug(f"Patched {mod} called")

    if not hasattr(mod.CpuAlloc, "bind_memory"):
        logger.warning("Skip bind_memory patch: CpuAlloc.bind_memory is missing")
        return

    if getattr(mod.CpuAlloc, "_ucm_bind_memory_patched", False):
        return

    mod.CpuAlloc.bind_memory = lambda self, *args, **kwargs: None
    setattr(mod.CpuAlloc, "_ucm_bind_memory_patched", True)
    logger.info("UCM bind_memory patch applied: CpuAlloc.bind_memory is now a no-op")
