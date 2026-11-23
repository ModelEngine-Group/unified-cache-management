"""Forward get_transfer_results on the vllm-ascend UCM adapter.

The adapter delegates ``get_finished`` to the UCM engine but has no
``get_transfer_results`` override, so the upstream wrapper repacks the
tuple with an always-empty ``failed_recving`` — a failed async load is
then promoted as a success (half-loaded blocks get cached; multi-group
engines cannot report failures through flat block ids).

Imported unconditionally: every guard is inside, and the patch stays
inert unless all hold — the adapter module is imported, the engine
defines ``KVConnectorTransferResults`` (vLLM >= 0.29.1), and the
adapter does not already implement the hook (an upstream fix is never
shadowed). The forwarding override routes through the reserved-hook
mechanism to the UCM facade or inner connector, reproducing base-class
behavior where the inner connector does not override it.
"""

from typing import Any

from ucm.integration.vllm.patch.utils import when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


def patch_ucm_connector_transfer_results(mod: Any) -> None:
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorTransferResults,
        )
    except ImportError:
        # Legacy engines have no failed_recving channel; nothing to forward.
        return

    cls = getattr(mod, "UCMConnectorV1", None)
    if cls is None:
        return
    if "get_transfer_results" in cls.__dict__:
        # Upstream (or a newer UCM) already owns this hook; never shadow it.
        return

    def get_transfer_results(self: Any, finished_req_ids: set[str]):
        return self._call_ucm_reserved_hook("get_transfer_results", finished_req_ids)

    cls.get_transfer_results = get_transfer_results
    logger.debug(
        "Patched vllm-ascend UCMConnectorV1.get_transfer_results to "
        "forward failed_recving from the UCM engine"
    )


when_imported("vllm_ascend.distributed.kv_transfer.kv_pool.ucm_connector.connector")(
    patch_ucm_connector_transfer_results
)
