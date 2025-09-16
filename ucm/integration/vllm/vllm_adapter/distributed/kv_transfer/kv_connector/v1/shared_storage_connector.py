from dataclasses import dataclass, field

import vllm.distributed.kv_transfer.kv_connector.v1.shared_storage_connector as shared_storage_conn
from vllm.distributed.kv_transfer.kv_connector.v1.shared_storage_connector import (
    ReqMeta,
)


@dataclass
class SharedStorageConnectorMetadata(
    shared_storage_conn.SharedStorageConnectorMetadata
):

    requests: list[ReqMeta] = field(default_factory=list)

    def __init__(self):
        pass


shared_storage_conn.SharedStorageConnectorMetadata = SharedStorageConnectorMetadata
