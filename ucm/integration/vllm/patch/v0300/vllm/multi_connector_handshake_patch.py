# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Let vLLM's MultiConnector hand each sub-connector its own handshake.

vLLM's MultiConnector sends only the first non-None handshake of its
sub-connectors, and gives that same dict to every sub-connector. With UCM
and Mooncake or NIXL together, one of them loses its handshake. This patch
sends all of them and gives each sub-connector only its own entries, keyed
by (pp_rank, tp_rank) as before. A sub-connector sees exactly what it would
see if it were the only connector. vllm-ascend's AscendMultiConnector does
not override these methods, so it gets the patch too.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
)

from ucm.integration.vllm.patch.utils import when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class MultiConnectorHandshakeMetadata(KVConnectorHandshakeMetadata):
    # One entry per sub-connector, in MultiConnector's order.
    per_connector: tuple[KVConnectorHandshakeMetadata | None, ...]


def _get_handshake_metadata(self) -> KVConnectorHandshakeMetadata | None:
    per_connector = tuple(
        connector.get_handshake_metadata() for connector in self._connectors
    )
    if all(metadata is None for metadata in per_connector):
        return None
    else:
        return MultiConnectorHandshakeMetadata(per_connector)


def _split_by_connector(self, metadata: Mapping[Any, Any]) -> list[dict[Any, Any]]:
    """Each sub-connector's own entries, keyed like the input."""
    num_connectors = len(self._connectors)
    if all(
        isinstance(item, MultiConnectorHandshakeMetadata)
        and len(item.per_connector) == num_connectors
        for item in metadata.values()
    ):
        return [
            {
                key: item.per_connector[index]
                for key, item in metadata.items()
                if item.per_connector[index] is not None
            }
            for index in range(num_connectors)
        ]
    else:
        # A worker without this patch sent one sub-connector's metadata;
        # keep vLLM's behavior and give everyone the same dict.
        return [dict(metadata) for _ in range(num_connectors)]


def _set_xfer_handshake_metadata(self, metadata) -> None:
    for connector, own in zip(self._connectors, _split_by_connector(self, metadata)):
        connector.set_xfer_handshake_metadata(own)


def _set_xfer_handshake_metadata_pp_aware(self, metadata) -> None:
    for connector, own in zip(self._connectors, _split_by_connector(self, metadata)):
        connector.set_xfer_handshake_metadata_pp_aware(own)


@when_imported("vllm.distributed.kv_transfer.kv_connector.v1.multi_connector")
def patch_multi_connector_handshake(mod):
    multi_connector = mod.MultiConnector
    if getattr(multi_connector, "_ucm_handshake_patched", False):
        return
    else:
        multi_connector.get_handshake_metadata = _get_handshake_metadata
        multi_connector.set_xfer_handshake_metadata = _set_xfer_handshake_metadata
        multi_connector.set_xfer_handshake_metadata_pp_aware = (
            _set_xfer_handshake_metadata_pp_aware
        )
        multi_connector._ucm_handshake_patched = True
        logger.info("Patched MultiConnector to hand out per-connector handshakes")
