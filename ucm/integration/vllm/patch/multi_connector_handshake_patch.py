"""Route Worker handshakes to the corresponding Scheduler-side child.

Remove once upstream MultiConnector sends and routes every child's handshake.
"""

from dataclasses import dataclass
from functools import wraps

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
)


@dataclass
class MultiConnectorHandshake(KVConnectorHandshakeMetadata):
    by_connector: dict[int, KVConnectorHandshakeMetadata]


def _get_handshake_metadata(self) -> MultiConnectorHandshake | None:
    by_connector = {}
    for index, connector in enumerate(self._connectors):
        if (metadata := connector.get_handshake_metadata()) is not None:
            by_connector[index] = metadata
    return MultiConnectorHandshake(by_connector) if by_connector else None


def _route_handshakes(connectors, metadata, setter):
    # Keep the engine's rank key intact, including Ascend's optional PCP rank.
    by_connector = [{} for _ in connectors]
    for rank, handshake in metadata.items():
        for index, child_handshake in handshake.by_connector.items():
            by_connector[index][rank] = child_handshake

    for connector, child_metadata in zip(connectors, by_connector):
        if child_metadata:
            getattr(connector, setter)(child_metadata)


def _set_xfer_handshake_metadata(self, metadata):
    _route_handshakes(self._connectors, metadata, "set_xfer_handshake_metadata")


def _set_xfer_handshake_metadata_pp_aware(self, metadata):
    _route_handshakes(
        self._connectors, metadata, "set_xfer_handshake_metadata_pp_aware"
    )


def patch_handshake(multi_connector):
    multi_connector.get_handshake_metadata = _get_handshake_metadata
    multi_connector.set_xfer_handshake_metadata = _set_xfer_handshake_metadata
    # AscendMultiConnector inherits these methods. Older vLLM releases only
    # use the legacy setter.
    if hasattr(multi_connector, "set_xfer_handshake_metadata_pp_aware"):
        multi_connector.set_xfer_handshake_metadata_pp_aware = (
            _set_xfer_handshake_metadata_pp_aware
        )


def patch_handshake_collection(executor):
    original = executor.get_kv_connector_handshake_metadata

    @wraps(original)
    def get_kv_connector_handshake_metadata(self):
        worker_reports = original(self)
        if not worker_reports:
            return worker_reports

        # Some CUDA PCP replicas use the same engine rank key. EngineCore would
        # replace the whole report with the last one, dropping children that
        # only report from one replica. Merge those reports by child first.
        reports_by_rank = {}
        merged = False
        for report in worker_reports:
            for rank, handshake in (report or {}).items():
                previous = reports_by_rank.get(rank)
                if isinstance(previous, MultiConnectorHandshake) and isinstance(
                    handshake, MultiConnectorHandshake
                ):
                    handshake = MultiConnectorHandshake(
                        previous.by_connector | handshake.by_connector
                    )
                    merged = True
                reports_by_rank[rank] = handshake
        return [reports_by_rank] if merged else worker_reports

    executor.get_kv_connector_handshake_metadata = get_kv_connector_handshake_metadata
