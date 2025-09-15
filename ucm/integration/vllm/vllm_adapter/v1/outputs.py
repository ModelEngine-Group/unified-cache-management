from dataclasses import dataclass
from typing import NamedTuple, Optional

import torch
from vllm.v1.outputs import LogprobsLists, LogprobsTensors


class ModelRunnerOutput:

    # [num_reqs]
    req_ids: list[str]
    # req_id -> index
    req_id_to_index: dict[str, int]

    # num_reqs x num_generated_tokens
    # num_generated_tokens is the number of tokens
    # generated in the current step. It can be different for
    # each request due to speculative/jump decoding.
    sampled_token_ids: list[list[int]]

    # num_reqs x num_spec_tokens
    spec_token_ids: Optional[list[list[int]]]

    # [num_reqs, max_num_logprobs + 1]
    # [num_reqs, max_num_logprobs + 1]
    # [num_reqs]
    logprobs: Optional[LogprobsLists]

    # req_id -> (token_ids, logprobs, ranks)
    # [prompt_len, num_prompt_logprobs]
    # [prompt_len, num_prompt_logprobs]
    # [prompt_len]
    prompt_logprobs_dict: dict[str, Optional[LogprobsTensors]]

    # [num_reqs, hidden_size]
    pooler_output: list[Optional[torch.Tensor]]

    # [req_ids]
    finished_sending: Optional[set[str]] = None
    finished_recving: Optional[set[str]] = None
    finished_dumping: Optional[dict[str, list[str]]] = None
    
    # IDs of externally computed KV blocks that failed to load.
    # Requests referencing these blocks should be rescheduled to recompute them.
    invalid_block_ids: Optional[set[int]] = None

    # req_id -> num_nans_in_logits
    num_nans_in_logits: Optional[dict[str, int]] = None
