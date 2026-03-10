from dataclasses import dataclass, field
from typing import Optional


@dataclass
class KVConnectorOutput:
    finished_sending: Optional[set[str]] = None
    finished_recving: Optional[set[str]] = None
    kv_connector_stats: Optional[object] = None
    invalid_block_ids: set[int] = field(default_factory=set)

    def is_empty(self):
        return (
            not self.finished_sending
            and not self.finished_recving
            and not self.kv_connector_stats
            and not self.invalid_block_ids
        )
