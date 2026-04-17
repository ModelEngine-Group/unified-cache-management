from ucm.integration.vllm.patch.utils import patch_or_inject, when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


@when_imported("vllm_ascend.attention.sfa_v1")
def patch_sfa_v1(mod):
    logger.debug(f"Patched {mod} called")

    from ucm.integration.vllm.patch.v0180.vllm_ascend.pc.attention import sfa_v1

    patch_or_inject(mod.AscendSFAImpl, "forward", sfa_v1.AscendSFAImpl.forward)
