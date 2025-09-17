from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import vllm.v1.worker.gpu_input_batch as vllm_v1_gpu_input_batch
from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import MultiModalKwargs, PlaceholderRange
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.sample.logits_processor import BatchUpdateBuilder, init_builtin_logitsprocs
from vllm.v1.spec_decode.utils import is_spec_decode_unsupported
from vllm.v1.worker.block_table import MultiGroupBlockTable


@dataclass
class CachedRequestState:

    req_id: str
    prompt_token_ids: list[int]
    mm_inputs: list[MultiModalKwargs]
    mm_positions: list[PlaceholderRange]
    sampling_params: Optional[SamplingParams]
    pooling_params: Optional[PoolingParams]
    generator: Optional[torch.Generator]

    block_ids: tuple[list[int], ...]
    num_computed_tokens: int
    output_token_ids: list[int]

    mrope_positions: Optional[torch.Tensor] = None
    mrope_position_delta: Optional[int] = None

    lora_request: Optional[LoRARequest] = None

    def __post_init__(self):
        self.num_prompt_tokens = len(self.prompt_token_ids)
        # 'last_generator_offset' and 'last_gelen_last_output_token_ids' are
        # used to allow safe rollback in case a sampled token turns out to be
        # invalid (e.g., due to KV load errors).
        self.last_generator_offset = 0 if self.generator else None
        self.len_last_output_token_ids = len(self.output_token_ids)

    @property
    def num_tokens(self) -> int:
        return self.num_prompt_tokens + len(self.output_token_ids)

    def get_token_id(self, idx: int) -> int:
        if idx < self.num_prompt_tokens:
            return self.prompt_token_ids[idx]
        else:
            return self.output_token_ids[idx - self.num_prompt_tokens]


class InputBatch(vllm_v1_gpu_input_batch.InputBatch):

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        device: torch.device,
        pin_memory: bool,
        vocab_size: int,
        block_sizes: list[int],  # The block_size of each kv cache group
        is_spec_decode: bool = False,
        logits_processing_needs_token_ids: bool = False,
    ):
        super().__init__(
            max_num_reqs,
            max_model_len,
            max_num_batched_tokens,
            device,
            pin_memory,
            vocab_size,
            block_sizes,
            is_spec_decode,
            logits_processing_needs_token_ids,
        )
        self.generators_last_offset: dict[int, int] = {}

    def add_request(
        self,
        request: "CachedRequestState",
    ) -> int:
        req_index = self._register_add_request(request)

        req_id = request.req_id
        if req_index == len(self._req_ids):
            self._req_ids.append(req_id)
            self.req_output_token_ids.append(request.output_token_ids)
        else:
            self._req_ids[req_index] = req_id
            self.req_output_token_ids[req_index] = request.output_token_ids

        self.req_id_to_index[req_id] = req_index

        # Copy the prompt token ids and output token ids.
        num_prompt_tokens = len(request.prompt_token_ids)
        self.num_prompt_tokens[req_index] = num_prompt_tokens
        self.token_ids_cpu[req_index, :num_prompt_tokens] = request.prompt_token_ids
        start_idx = num_prompt_tokens
        end_idx = start_idx + len(request.output_token_ids)
        self.token_ids_cpu[req_index, start_idx:end_idx] = request.output_token_ids
        # Number of token ids in token_ids_cpu.
        # NOTE(woosuk): This may include spec decode tokens.
        self.num_tokens[req_index] = request.num_tokens
        # Number of tokens without spec decode tokens.
        self.num_tokens_no_spec[req_index] = request.num_tokens

        self.num_computed_tokens_cpu[req_index] = request.num_computed_tokens
        self.block_table.add_row(request.block_ids, req_index)

        if sampling_params := request.sampling_params:
            if self.is_spec_decode and is_spec_decode_unsupported(sampling_params):
                self.spec_decode_unsupported_reqs.add(req_id)
            if sampling_params.sampling_type == SamplingType.GREEDY:
                # Avoid later division by zero.
                self.temperature_cpu[req_index] = -1.0
                self.greedy_reqs.add(req_id)
            else:
                self.temperature_cpu[req_index] = sampling_params.temperature
                self.random_reqs.add(req_id)

            self.top_p_cpu[req_index] = sampling_params.top_p
            if sampling_params.top_p < 1:
                self.top_p_reqs.add(req_id)
            top_k = sampling_params.top_k
            if 0 < top_k < self.vocab_size:
                self.top_k_reqs.add(req_id)
            else:
                top_k = self.vocab_size
            self.top_k_cpu[req_index] = top_k
            self.frequency_penalties_cpu[req_index] = sampling_params.frequency_penalty
            if sampling_params.frequency_penalty != 0.0:
                self.frequency_penalties_reqs.add(req_id)
            self.presence_penalties_cpu[req_index] = sampling_params.presence_penalty
            if sampling_params.presence_penalty != 0.0:
                self.presence_penalties_reqs.add(req_id)
            self.repetition_penalties_cpu[req_index] = (
                sampling_params.repetition_penalty
            )
            if sampling_params.repetition_penalty != 1.0:
                self.repetition_penalties_reqs.add(req_id)

            # NOTE(woosuk): self.generators should not include the requests that
            # do not have their own generator.
            if request.generator is not None:
                self.generators[req_index] = request.generator
                assert request.last_generator_offset is not None
                self.generators_last_offset[req_index] = request.last_generator_offset

            if sampling_params.logprobs is not None:
                self.num_logprobs[req_id] = sampling_params.logprobs
            if sampling_params.prompt_logprobs is not None:
                self.num_prompt_logprobs[req_id] = sampling_params.prompt_logprobs

            if sampling_params.allowed_token_ids:
                self.has_allowed_token_ids.add(req_id)
                if self.allowed_token_ids_mask_cpu_tensor is None:
                    # Lazy allocation for this tensor, which can be large.
                    # False means we don't fill with -inf.
                    self.allowed_token_ids_mask = torch.zeros(
                        self.max_num_reqs,
                        self.vocab_size,
                        dtype=torch.bool,
                        device=self.device,
                    )
                    self.allowed_token_ids_mask_cpu_tensor = torch.zeros(
                        self.max_num_reqs,
                        self.vocab_size,
                        dtype=torch.bool,
                        device="cpu",
                    )
                self.allowed_token_ids_mask_cpu_tensor[req_index] = True
                # False means we don't fill with -inf.
                self.allowed_token_ids_mask_cpu_tensor[req_index][
                    sampling_params.allowed_token_ids
                ] = False

            if sampling_params.bad_words_token_ids:
                self.bad_words_token_ids[req_index] = (
                    sampling_params.bad_words_token_ids
                )
        else:
            assert request.pooling_params is not None
            self.pooling_params[req_id] = request.pooling_params

        # Add request lora ID
        if request.lora_request:
            lora_id = request.lora_request.lora_int_id
            if lora_id not in self.lora_id_to_request_ids:
                self.lora_id_to_request_ids[lora_id] = set()

            self.request_lora_mapping[req_index] = lora_id
            self.lora_id_to_request_ids[lora_id].add(request.req_id)
            self.lora_id_to_lora_request[lora_id] = request.lora_request
        else:
            # No LoRA
            self.request_lora_mapping[req_index] = 0

        return req_index


# vllm_v1_gpu_input_batch.CachedRequestState = CachedRequestState
# vllm_v1_gpu_input_batch.InputBatch = InputBatch
