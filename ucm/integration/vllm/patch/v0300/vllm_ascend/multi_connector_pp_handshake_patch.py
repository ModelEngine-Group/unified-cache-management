"""Keep PP handshake metadata scoped to each Ascend KV connector."""

from dataclasses import dataclass
from functools import wraps

from ucm.integration.vllm.patch.utils import when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class _MultiConnectorHandshakeMetadata:
    by_connector: tuple[object | None, ...]


def _uses_ucm_pp_handshake(connector) -> bool:
    config = connector._vllm_config
    return (
        config.parallel_config.pipeline_parallel_size > 1
        and config.kv_transfer_config.has_connector("UCMConnector")
    )


@when_imported("vllm_ascend.distributed.kv_transfer.ascend_multi_connector")
def patch_multi_connector_pp_handshake(mod):
    cls = mod.AscendMultiConnector
    original_get = cls.get_handshake_metadata
    original_set = cls.set_xfer_handshake_metadata_pp_aware
    if getattr(original_get, "_ucm_pp_handshake_patched", False):
        return

    @wraps(original_get)
    def get_handshake_metadata(self):
        if not _uses_ucm_pp_handshake(self):
            return original_get(self)

        by_connector = tuple(
            connector.get_handshake_metadata() for connector in self._connectors
        )
        if all(child is None for child in by_connector):
            return None
        return _MultiConnectorHandshakeMetadata(by_connector)

    @wraps(original_set)
    def set_xfer_handshake_metadata_pp_aware(self, metadata):
        if not _uses_ucm_pp_handshake(self):
            return original_set(self, metadata)

        by_connector = [{} for _ in self._connectors]
        for ranks, worker_metadata in metadata.items():
            if not isinstance(worker_metadata, _MultiConnectorHandshakeMetadata):
                raise ValueError(
                    f"Expected per-connector PP handshake metadata from worker {ranks}. "
                    "Verify that all PP workers use the same UCM patch."
                )
            if len(worker_metadata.by_connector) != len(by_connector):
                raise ValueError(
                    f"PP handshake connector count differs for worker {ranks}"
                )
            for connector_metadata, child in zip(
                by_connector, worker_metadata.by_connector
            ):
                if child is not None:
                    connector_metadata[ranks] = child

        for connector, connector_metadata in zip(self._connectors, by_connector):
            if connector_metadata:
                connector.set_xfer_handshake_metadata_pp_aware(connector_metadata)

    get_handshake_metadata._ucm_pp_handshake_patched = True
    cls.get_handshake_metadata = get_handshake_metadata
    cls.set_xfer_handshake_metadata_pp_aware = set_xfer_handshake_metadata_pp_aware
    logger.info("UCM Ascend MultiConnector PP handshake patch applied")
