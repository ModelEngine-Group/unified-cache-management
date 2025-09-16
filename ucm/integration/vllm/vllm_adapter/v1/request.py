from typing import TYPE_CHECKING, Optional

from vllm.multimodal.inputs import MultiModalKwargs, PlaceholderRange
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.v1.structured_output.request import StructuredOutputRequest

if TYPE_CHECKING:
    from vllm.lora.request import LoRARequest

import vllm.v1.request as vllm_v1_request


class Request(vllm_v1_request.Request):

    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        multi_modal_inputs: Optional[list[MultiModalKwargs]],
        multi_modal_hashes: Optional[list[str]],
        multi_modal_placeholders: Optional[list[PlaceholderRange]],
        sampling_params: Optional[SamplingParams],
        pooling_params: Optional[PoolingParams],
        eos_token_id: Optional[int],
        client_index: int = 0,
        arrival_time: Optional[float] = None,
        lora_request: Optional["LoRARequest"] = None,
        structured_output_request: Optional["StructuredOutputRequest"] = None,
        cache_salt: Optional[str] = None,
        priority: int = 0,
    ) -> None:
        super().__init__(
            request_id,
            prompt_token_ids,
            multi_modal_inputs,
            multi_modal_hashes,
            multi_modal_placeholders,
            sampling_params,
            pooling_params,
            eos_token_id,
            client_index,
            arrival_time,
            lora_request,
            structured_output_request,
            cache_salt,
            priority,
        )
        self.succeed_dumped_blocks: list[str] = []


vllm_v1_request.Request = Request
