"""Restore synchronous KV-load ordering for Ascend balance scheduling."""

from functools import wraps

from ucm.integration.vllm.patch.utils import when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


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

        # Lazy import avoids pulling connector/worker dependencies into patch
        # registration while vLLM and the Ascend platform are being imported.
        from ucm.integration.vllm.ucm_connector import UCMConnectorMetadata

        metadata = output.kv_connector_metadata
        if not isinstance(metadata, UCMConnectorMetadata):
            return output

        # A copied Ascend schedule() can omit has_sync_kv_loads, leaving the
        # default False. The worker would then start these loads after forward.
        # Derive the flag from the actual plan and preserve request-async loads.
        if any(
            bool(request.load_block_ids[0]) and not request.load_async
            for request in metadata.request_meta.values()
        ):
            output.has_sync_kv_loads = True
        return output

    schedule._ucm_sync_kv_load_patched = True
    cls.schedule = schedule
    logger.info("UCM Ascend balance scheduler sync KV-load patch applied")
