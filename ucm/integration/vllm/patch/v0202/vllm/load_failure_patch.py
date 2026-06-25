from ucm.integration.vllm.patch.utils import patch_or_inject, when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


# UCM double-free guard patch start
@when_imported("vllm.v1.core.block_pool")
def patch_core_block_pool(mod):
    logger.debug(f"Patched {mod} called")

    from ucm.integration.vllm.patch.v0202.vllm.v1.core import block_pool

    patch_or_inject(
        mod.BlockPool,
        "free_blocks",
        block_pool.BlockPool.free_blocks,
    )
    logger.info("UCM double-free guard patch applied: BlockPool.free_blocks")
# UCM double-free guard patch end


@when_imported("vllm.v1.core.sched.scheduler")
def patch_core_sched_scheduler(mod):
    logger.debug(f"Patched {mod} called")

    from ucm.integration.vllm.patch.v0202.vllm.v1.core.sched import scheduler

    patch_or_inject(
        mod.Scheduler,
        "_update_requests_with_invalid_blocks",
        scheduler.Scheduler._update_requests_with_invalid_blocks,
    )
