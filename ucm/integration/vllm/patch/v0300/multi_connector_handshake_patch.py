"""Backport vLLM #67174 to MultiConnector builds that lack the fix."""

from dataclasses import dataclass

from ucm.integration.vllm.patch.utils import when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


@when_imported("vllm.distributed.kv_transfer.kv_connector.v1.multi_connector")
def patch_multi_connector_handshake(mod):
    cls = mod.MultiConnector
    if hasattr(mod, "MultiKVConnectorHandshakeMetadata") and hasattr(
        cls, "_sub_connector_handshake_metadata"
    ):
        return

    @dataclass
    class MultiKVConnectorHandshakeMetadata(mod.KVConnectorHandshakeMetadata):
        """One worker's handshake metadata, in sub-connector order."""

        metadata: tuple[object | None, ...]

    # Serialize the wrapper under the same name as the upstream implementation.
    MultiKVConnectorHandshakeMetadata.__module__ = mod.__name__
    MultiKVConnectorHandshakeMetadata.__qualname__ = "MultiKVConnectorHandshakeMetadata"
    mod.MultiKVConnectorHandshakeMetadata = MultiKVConnectorHandshakeMetadata

    def get_handshake_metadata(self):
        metadata = tuple(
            connector.get_handshake_metadata() for connector in self._connectors
        )
        if all(entry is None for entry in metadata):
            return None
        return MultiKVConnectorHandshakeMetadata(metadata=metadata)

    def _sub_connector_handshake_metadata(self, metadata, index):
        selected = {}
        for rank, worker_metadata in metadata.items():
            if not isinstance(worker_metadata, MultiKVConnectorHandshakeMetadata):
                # Preserve the broadcast used by subclasses that report a single
                # sub-connector's metadata without the wrapper.
                return metadata
            entry = worker_metadata.metadata[index]
            if entry is not None:
                selected[rank] = entry
        return selected

    def set_xfer_handshake_metadata(self, metadata):
        for index, connector in enumerate(self._connectors):
            connector.set_xfer_handshake_metadata(
                self._sub_connector_handshake_metadata(metadata, index)
            )

    def set_xfer_handshake_metadata_pp_aware(self, metadata):
        for index, connector in enumerate(self._connectors):
            connector.set_xfer_handshake_metadata_pp_aware(
                self._sub_connector_handshake_metadata(metadata, index)
            )

    cls.get_handshake_metadata = get_handshake_metadata
    cls._sub_connector_handshake_metadata = _sub_connector_handshake_metadata
    cls.set_xfer_handshake_metadata = set_xfer_handshake_metadata
    cls.set_xfer_handshake_metadata_pp_aware = set_xfer_handshake_metadata_pp_aware
    logger.info("UCM vLLM MultiConnector handshake patch applied")
