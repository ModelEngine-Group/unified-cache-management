from ucm.integration.vllm.patch.utils import when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


@when_imported("vllm_ascend.distributed.kv_transfer.kv_pool.ucm_connector")
def patch_kv_pool_ucm_connector(mod):
    logger.debug(f"Patched {mod} called")

    from ucm.integration.vllm.patch.v0180.vllm_ascend.pc.distributed.kv_transfer.kv_pool import (
        ucm_connector,
    )

    ucm_connector.patch_ucm_connector_v1(mod)
