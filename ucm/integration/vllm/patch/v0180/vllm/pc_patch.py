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


@when_imported("vllm.config.vllm")
def patch_vllm_config(mod):
    logger.debug(f"Patched {mod} called")

    from ucm.integration.vllm.patch.v0180.vllm.pc.config import vllm

    vllm.patch_vllm_config_cls(mod.VllmConfig)


@when_imported("vllm.model_executor.models.config")
def patch_model_config(mod):
    logger.debug(f"Patched {mod} called")

    from ucm.integration.vllm.patch.v0180.vllm.pc.model_executor.models import (
        config,
    )

    config.patch_mamba_model_config_cls(mod.MambaModelConfig)


@when_imported("vllm.v1.core.sched.scheduler")
def patch_core_sched_scheduler(mod):
    logger.debug(f"Patched {mod} called")

    from ucm.integration.vllm.patch.v0180.vllm.pc.v1.core.sched import scheduler

    patch_or_inject(
        mod.Scheduler,
        "_mamba_block_aligned_split",
        scheduler.Scheduler._mamba_block_aligned_split,
    )


@when_imported("vllm.v1.worker.gpu_model_runner")
def patch_worker_gpu_model_runner(mod):
    logger.debug(f"Patched {mod} called")

    from ucm.integration.vllm.patch.v0180.vllm.pc.v1.worker import gpu_model_runner

    patch_or_inject(
        mod.GPUModelRunner,
        "may_reinitialize_input_batch",
        gpu_model_runner.GPUModelRunner.may_reinitialize_input_batch,
    )


@when_imported("vllm.v1.worker.mamba_utils")
def patch_worker_mamba_utils(mod):
    logger.debug(f"Patched {mod} called")

    from ucm.integration.vllm.patch.v0180.vllm.pc.v1.worker import mamba_utils

    mamba_utils.patch_preprocess_mamba(mod)
