from ucm.integration.vllm.patch.utils import patch_or_inject, when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


@when_imported("vllm.v1.core.sched.scheduler")
def patch_core_sched_scheduler(mod):
    logger.debug(f"Patched {mod} called")

    from ucm.integration.vllm.patch.v0202.vllm.v1.core.sched import scheduler

    patch_or_inject(
        mod.Scheduler,
        "_update_requests_with_invalid_blocks",
        scheduler.Scheduler._update_requests_with_invalid_blocks,
    )

