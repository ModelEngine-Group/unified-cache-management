from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from vllm.v1.core.sched.output import CachedRequestData, NewRequestData

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata

import vllm.v1.core.sched.output as vllm_v1_scheduler_output


@dataclass
class SchedulerOutput(vllm_v1_scheduler_output.SchedulerOutput):
    # modified slots by sparse algorithm
    req_sparsed_slots: dict[str, int] = None


vllm_v1_scheduler_output.SchedulerOutput = SchedulerOutput
