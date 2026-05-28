import inspect

import ucm.integration.vllm.patch.v0180.vllm_ascend.ucm_connector_patch  # noqa: F401
from ucm.integration.vllm.patch.utils import (
    patch_or_inject,
    when_imported,
)
from ucm.logger import init_logger

logger = init_logger(__name__)

_SFA_V1_PATCH_BY_VERSION = {
    "0.18.0rc1": "sfa_v1_rc1",
    "0.18.0": "sfa_v1_0180",
}


def _npu_sample_tokens_has_spec_kv_finalize(npu_cls: type) -> bool:
    try:
        src = inspect.getsource(npu_cls.sample_tokens)
    except (OSError, TypeError, AttributeError):
        return False
    return "vLLM v0.18 defers KV connector" in src and "finalize_kv_connector" in src


@when_imported("vllm_ascend.attention.sfa_v1")
def patch_sfa_v1(mod):
    from ucm.integration.vllm.patch.apply_patch import get_vllm_ascend_version_full

    full_ver = get_vllm_ascend_version_full()
    patch_name = _SFA_V1_PATCH_BY_VERSION.get(full_ver)
    if patch_name is None:
        logger.warning(
            f"Skip sfa_v1 patch: unsupported vllm-ascend version {full_ver!r}, "
            f"expected one of {list(_SFA_V1_PATCH_BY_VERSION)}"
        )
        return

    logger.info(f"Patched {mod} with {patch_name} for vllm-ascend {full_ver}")
    if patch_name == "sfa_v1_rc1":
        from ucm.integration.vllm.patch.v0180.vllm_ascend.pc.attention import sfa_v1_rc1

        forward = sfa_v1_rc1.AscendSFAImpl.forward
        update_graph_params = sfa_v1_rc1.AscendSFAImpl.update_graph_params
    else:
        from ucm.integration.vllm.patch.v0180.vllm_ascend.pc.attention import (
            sfa_v1_0180,
        )

        forward = sfa_v1_0180.AscendSFAImpl.forward
        update_graph_params = sfa_v1_0180.AscendSFAImpl.update_graph_params

    patch_or_inject(mod.AscendSFAImpl, "forward", forward)
    patch_or_inject(mod.AscendSFAImpl, "update_graph_params", update_graph_params)


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
