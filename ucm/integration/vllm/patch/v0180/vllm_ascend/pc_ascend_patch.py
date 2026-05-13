import inspect

from ucm.integration.vllm.patch.utils import (
    patch_or_inject,
    when_imported,
)
from ucm.logger import init_logger

logger = init_logger(__name__)


def _npu_sample_tokens_has_spec_kv_finalize(npu_cls: type) -> bool:
    try:
        src = inspect.getsource(npu_cls.sample_tokens)
    except (OSError, TypeError, AttributeError):
        return False
    return "vLLM v0.18 defers KV connector" in src and "finalize_kv_connector" in src


@when_imported("vllm_ascend.attention.sfa_v1")
def patch_sfa_v1(mod):
    logger.debug(f"Patched {mod} called")

    from ucm.integration.vllm.patch.v0180.vllm_ascend.pc.attention import sfa_v1

    patch_or_inject(mod.AscendSFAImpl, "forward", sfa_v1.AscendSFAImpl.forward)


@when_imported("vllm_ascend.worker.model_runner_v1")
def patch_worker_npu_model_runner(mod):
    logger.debug(f"Patched {mod} called")

    if _npu_sample_tokens_has_spec_kv_finalize(mod.NPUModelRunner):
        return

    from ucm.integration.vllm.patch.v0180.vllm_ascend.pc.v1.worker import (
        npu_model_runner,
    )

    patch_or_inject(
        mod.NPUModelRunner,
        "sample_tokens",
        npu_model_runner.NPUModelRunner.sample_tokens,
    )


@when_imported("vllm_ascend.patch.platform.patch_mamba_config")
def patch_platform_mamba_config(mod):
    logger.debug(f"Patched {mod} called")

    from ucm.integration.vllm.patch.v0180.vllm_ascend.pc.platform import (
        patch_mamba_config,
    )

    patch_mamba_config.patch_hybrid_attention_mamba_model_config()
