"""Register the concrete vllm-ascend UCM class name used by metrics."""

from typing import Any

from ucm.integration.vllm.patch.utils import when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)

UCM_CONNECTOR_MODULE = (
    "vllm_ascend.distributed.kv_transfer.kv_pool.ucm_connector"
)


def _register_ucm_connector_class_alias(factory: type) -> None:
    if "UCMConnector" not in factory._registry:
        return
    if "UCMConnectorV1" in factory._registry:
        return

    # UCM PATCH: MultiConnector serializes stats with the concrete class name
    # UCMConnectorV1, while vllm-ascend registers only UCMConnector.
    factory.register_connector(
        "UCMConnectorV1", UCM_CONNECTOR_MODULE, "UCMConnectorV1"
    )


def patch_ucm_connector_registration(mod: Any) -> None:
    if getattr(mod, "_ucm_connector_metrics_alias_patch", False):
        return

    original = mod.register_connector

    def register_connector() -> None:
        original()
        _register_ucm_connector_class_alias(mod.KVConnectorFactory)

    mod.register_connector = register_connector
    mod._ucm_connector_metrics_alias_patch = True

    # Handle the case where vllm-ascend registration ran before this hook.
    _register_ucm_connector_class_alias(mod.KVConnectorFactory)
    logger.debug("Patched vllm-ascend UCMConnectorV1 metrics alias registration")


when_imported("vllm_ascend.distributed.kv_transfer")(
    patch_ucm_connector_registration
)
