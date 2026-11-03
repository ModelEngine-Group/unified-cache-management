"""Restore synchronous KV-load ordering for Ascend balance scheduling."""

from functools import wraps

from ucm.integration.vllm.patch.utils import when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


def _has_ucm_sync_load(metadata):
    # Keep imports lazy: registration happens during platform initialization.
    from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
        MultiKVConnectorMetadata,
    )

    from ucm.integration.vllm.ucm_connector import UCMConnectorMetadata

    if isinstance(metadata, UCMConnectorMetadata):
        return any(
            bool(request.load_block_ids[0]) and not request.load_async
            for request in metadata.request_meta.values()
        )
    if isinstance(metadata, MultiKVConnectorMetadata):
        return any(_has_ucm_sync_load(child) for child in metadata.metadata)
    return False

@when_imported("vllm_ascend.patch.platform.patch_balance_schedule")
def patch_balance_sync_kv_load(mod):
    cls = mod.BalanceScheduler
    original = cls.schedule
    if getattr(original, "_ucm_sync_kv_load_patched", False):
        return

    @wraps(original)
    def schedule(self, *args, **kwargs):
        output = original(self, *args, **kwargs)
        if getattr(output, "has_sync_kv_loads", False):
            return output

        # Inspect child plans too: MultiConnector/AscendMultiConnector wrap
        # UCM metadata, while explicit request-async loads remain deferred.
        if _has_ucm_sync_load(output.kv_connector_metadata):
            output.has_sync_kv_loads = True
        return output

    schedule._ucm_sync_kv_load_patched = True
    cls.schedule = schedule
    logger.info("UCM Ascend balance scheduler sync KV-load patch applied")
