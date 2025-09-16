from dataclasses import dataclass
from typing import NamedTuple, Optional

import torch
import vllm.v1.outputs as vllm_v1_outputs
from vllm.v1.outputs import LogprobsLists, LogprobsTensors


@dataclass
class ModelRunnerOutput(vllm_v1_outputs.ModelRunnerOutput):

    finished_dumping: Optional[dict[str, list[str]]] = None
    # IDs of externally computed KV blocks that failed to load.
    # Requests referencing these blocks should be rescheduled to recompute them.
    invalid_block_ids: Optional[set[int]] = None


vllm_v1_outputs.ModelRunnerOutput = ModelRunnerOutput
