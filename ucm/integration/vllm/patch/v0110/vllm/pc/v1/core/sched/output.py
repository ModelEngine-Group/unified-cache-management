from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from vllm._bc_linter import bc_linter_include

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata

from vllm.v1.core.sched.output import CachedRequestData, NewRequestData


@bc_linter_include
@dataclass
class SchedulerOutput:

    scheduled_new_reqs: list[NewRequestData]
    scheduled_cached_reqs: CachedRequestData

    num_scheduled_tokens: dict[str, int]
    total_num_scheduled_tokens: int
    scheduled_spec_decode_tokens: dict[str, list[int]]
    scheduled_encoder_inputs: dict[str, list[int]]
    num_common_prefix_blocks: list[int]

    finished_req_ids: set[str]
    free_encoder_mm_hashes: list[str]

    structured_output_request_ids: dict[str, int]
    grammar_bitmask: Optional[npt.NDArray[np.int32]]

    kv_connector_metadata: Optional[KVConnectorMetadata] = None

    num_external_computed_tokens_per_req: dict[str, int] = field(default_factory=dict)
