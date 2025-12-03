#
# MIT License
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
from __future__ import annotations

import os

from ucm.logger import init_logger

logger = init_logger(__name__)

ENABLE_SPARSE = os.getenv("ENABLE_SPARSE")


def _enable_sparse() -> bool:
    return ENABLE_SPARSE is not None and ENABLE_SPARSE.lower() == "true"


def _apply_sparse_adapt() -> None:
    """Apply sparse adapt patches."""
    try:
        if _enable_sparse():
            _patch_block_table()
            _patch_kv_cache_manager()
            _patch_shared_storage_connector()
            _patch_attention_layer()
            _patch_mla_common()
            _patch_gpu_model_runner()
            _patch_gpu_worker()
            _patch_scheduler_output()
            _patch_scheduler()
            _patch_llama_model()
            _patch_qwen_model()
            logger.info("UCM sparse adapt patches applied successfully")
    except Exception as e:
        logger.error(f"Could not apply sparse adapt patches: {e}")
        raise e


# ==================== vllm/v1/core/sched/output.py  ====================
def _patch_scheduler_output() -> None:
    """Patch scheduler output to add UCM sparse support."""
    try:
        from dataclasses import dataclass
        from typing import TYPE_CHECKING, Optional

        if TYPE_CHECKING:
            import numpy as np
            import numpy.typing as npt
            from vllm.distributed.kv_transfer.kv_connector.v1.base import (
                KVConnectorMetadata,
            )
        from vllm.v1.core.sched import output
        from vllm.v1.core.sched.output import CachedRequestData, NewRequestData

        @dataclass
        class SchedulerOutput:

            # list of the requests that are scheduled for the first time.
            # We cache the request's data in each worker process, so that we don't
            # need to re-send it every scheduling step.
            scheduled_new_reqs: list[NewRequestData]
            # list of the requests that have been scheduled before.
            # Since the request's data is already cached in the worker processes,
            # we only send the diff to minimize the communication cost.
            scheduled_cached_reqs: CachedRequestData

            # req_id -> num_scheduled_tokens
            # Number of tokens scheduled for each request.
            num_scheduled_tokens: dict[str, int]
            # Total number of tokens scheduled for all requests.
            # Equal to sum(num_scheduled_tokens.values())
            total_num_scheduled_tokens: int
            # req_id -> spec_token_ids
            # If a request does not have any spec decode tokens, it will not be
            # included in the dictionary.
            scheduled_spec_decode_tokens: dict[str, list[int]]
            # req_id -> encoder input indices that need processing.
            # E.g., if a request has [0, 1], it could mean the vision encoder needs
            # to process that the request's 0-th and 1-th images in the current step.
            scheduled_encoder_inputs: dict[str, list[int]]
            # Number of common prefix blocks for all requests in each KV cache group.
            # This can be used for cascade attention.
            num_common_prefix_blocks: list[int]

            # Request IDs that are finished in between the previous and the current
            # steps. This is used to notify the workers about the finished requests
            # so that they can free the cached states for those requests.
            finished_req_ids: set[str]
            # list of (req_id, encoder_input_index) tuples.
            # Used to free the encoder cache.
            free_encoder_input_ids: list[tuple[str, int]]

            # Dict of request ids to their index within the batch
            # for filling the next token bitmask
            structured_output_request_ids: dict[str, int]
            # the bitmask for the whole batch
            grammar_bitmask: Optional[npt.NDArray[np.int32]]

            # KV Cache Connector metadata.
            kv_connector_metadata: Optional[KVConnectorMetadata] = None

            # modified slots by sparse algorithm
            req_sparsed_slots: dict[str, int] = None

        # Set module and qualname to make the class pickleable
        # This ensures pickle can find the class when serializing
        SchedulerOutput.__module__ = output.__name__
        SchedulerOutput.__qualname__ = "SchedulerOutput"

        output.SchedulerOutput = SchedulerOutput

    except ImportError:
        logger.warning("Could not patch scheduler output - module not found")


# ==================== vllm/attention/layer.py  ====================
def _patch_attention_layer() -> None:
    """Patch attention layer & unified_attention_with_output C++ op."""
    try:
        from typing import Optional

        import torch
        from vllm.attention.layer import (
            maybe_save_kv_layer_to_connector,
            wait_for_kv_layer_from_connector,
        )
        from vllm.forward_context import ForwardContext, get_forward_context

        from ucm.sparse.state import get_ucm_sparse, has_ucm_sparse

        def attn_forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            # For some alternate attention backends like MLA the attention output
            # shape does not match the query shape, so we optionally let the model
            # definition specify the output tensor shape.
            output_shape: Optional[torch.Size] = None,
        ) -> torch.Tensor:
            """
            The KV cache is stored inside this class and is accessed via
            `self.kv_cache`.

            Attention metadata (`attn_metadata`) is set using a context manager in
            the model runner's `execute_model` method. It is accessed via forward
            context using
            `vllm.forward_context.get_forward_context().attn_metadata`.
            """
            if self.calculate_kv_scales:
                attn_metadata = get_forward_context().attn_metadata
                if attn_metadata.enable_kv_scales_calculation:
                    self.calc_kv_scales(query, key, value)
            if self.use_output:
                output_shape = output_shape if output_shape is not None else query.shape
                output = torch.zeros(
                    output_shape, dtype=query.dtype, device=query.device
                )
                hidden_size = output_shape[-1]
                # We skip reshaping query, key and value tensors for the MLA
                # backend since these tensors have different semantics and are
                # processed differently.
                if not self.use_mla:
                    # Reshape the query, key, and value tensors.
                    # NOTE(woosuk): We do this outside the custom op to minimize the
                    # CPU overheads from the non-CUDA-graph regions.
                    query = query.view(-1, self.num_heads, self.head_size)
                    output = output.view(-1, self.num_heads, self.head_size)
                    if key is not None:
                        key = key.view(-1, self.num_kv_heads, self.head_size)
                    if value is not None:
                        value = value.view(-1, self.num_kv_heads, self.head_size)
                if self.use_direct_call:
                    forward_context: ForwardContext = get_forward_context()
                    attn_metadata = forward_context.attn_metadata
                    if isinstance(attn_metadata, dict):
                        attn_metadata = attn_metadata[self.layer_name]
                    self_kv_cache = self.kv_cache[forward_context.virtual_engine]
                    self.impl.forward(
                        self,
                        query,
                        key,
                        value,
                        self_kv_cache,
                        attn_metadata,
                        output=output,
                    )
                else:
                    torch.ops.vllm.unified_attention_with_output(
                        query, key, value, output, self.layer_name
                    )
                return output.view(-1, hidden_size)
            else:
                if self.use_direct_call:
                    forward_context = get_forward_context()
                    attn_metadata = forward_context.attn_metadata
                    if isinstance(attn_metadata, dict):
                        attn_metadata = attn_metadata[self.layer_name]
                    self_kv_cache = self.kv_cache[forward_context.virtual_engine]
                    return self.impl.forward(
                        self, query, key, value, self_kv_cache, attn_metadata
                    )
                else:
                    return torch.ops.vllm.unified_attention(
                        query, key, value, self.layer_name
                    )

        def maybe_execute_sparse_attention_begin(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            layer_name: str,
            forward_context: ForwardContext,
            output: Optional[torch.Tensor] = None,
            phase: Optional[str] = None,
        ):
            if not has_ucm_sparse():
                return query, key, value, output

            ucm_sparse = get_ucm_sparse()

            attn_metadata = forward_context.attn_metadata
            if attn_metadata is None:
                return query, key, value, output

            return ucm_sparse.attention_begin(
                query, key, value, layer_name, forward_context, output, phase
            )

        def maybe_execute_sparse_attention_finished(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attn_output: torch.Tensor,
            layer_name: str,
            forward_context: ForwardContext,
            phase: Optional[str] = None,
        ):
            if not has_ucm_sparse():
                return

            ucm_sparse = get_ucm_sparse()

            attn_metadata = forward_context.attn_metadata
            if attn_metadata is None:
                return

            ucm_sparse.attention_finished(
                query, key, value, attn_output, layer_name, forward_context, phase
            )

        vllm_ops = torch.ops.vllm
        orig_unified_attention_with_output = vllm_ops.unified_attention_with_output
        orig_unified_attention = vllm_ops.unified_attention

        def _wrap_op_overload(orig, impl):
            class _Wrapper:
                def __init__(self, orig):
                    self._orig = orig

                def __call__(self, *args, **kwargs):
                    return impl(*args, **kwargs)

                def __getattr__(self, name):
                    return getattr(self._orig, name)

            return _Wrapper(orig)

        def unified_attention_impl(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            layer_name: str,
        ) -> torch.Tensor:
            wait_for_kv_layer_from_connector(layer_name)

            forward_context: ForwardContext = get_forward_context()
            attn_metadata = forward_context.attn_metadata
            if isinstance(attn_metadata, dict):
                attn_metadata = attn_metadata[layer_name]
            self = forward_context.no_compile_layers[layer_name]
            kv_cache = self.kv_cache[forward_context.virtual_engine]
            query, key, value, _ = maybe_execute_sparse_attention_begin(
                query, key, value, layer_name, forward_context
            )
            output = self.impl.forward(self, query, key, value, kv_cache, attn_metadata)
            maybe_execute_sparse_attention_finished(
                query, key, value, output, layer_name, forward_context
            )
            maybe_save_kv_layer_to_connector(layer_name, kv_cache)
            return output

        def unified_attention_with_output_impl(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            output: torch.Tensor,
            layer_name: str,
            output_scale: Optional[torch.Tensor] = None,
        ) -> None:
            wait_for_kv_layer_from_connector(layer_name)
            forward_context: ForwardContext = get_forward_context()
            attn_metadata = forward_context.attn_metadata
            if isinstance(attn_metadata, dict):
                attn_metadata = attn_metadata[layer_name]
            self = forward_context.no_compile_layers[layer_name]
            kv_cache = self.kv_cache[forward_context.virtual_engine]
            if not self.use_mla:
                query, key, value, output = maybe_execute_sparse_attention_begin(
                    query, key, value, layer_name, forward_context, output
                )
            self.impl.forward(
                self,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output=output,
                output_scale=output_scale,
            )
            if not self.use_mla:
                maybe_execute_sparse_attention_finished(
                    query, key, value, output, layer_name, forward_context
                )

            maybe_save_kv_layer_to_connector(layer_name, kv_cache)

        vllm_ops.unified_attention_with_output = _wrap_op_overload(
            orig_unified_attention_with_output, unified_attention_with_output_impl
        )
        vllm_ops.unified_attention = _wrap_op_overload(
            orig_unified_attention, unified_attention_impl
        )
        from vllm.attention import layer

        layer.maybe_execute_sparse_attention_begin = (
            maybe_execute_sparse_attention_begin
        )
        layer.maybe_execute_sparse_attention_finished = (
            maybe_execute_sparse_attention_finished
        )
        layer.Attention.forward = attn_forward
        layer.unified_attention = unified_attention_impl
        layer.unified_attention_with_output = unified_attention_with_output_impl

    except ImportError:
        logger.warning(
            "Could not patch unified attention with output - module not found"
        )


# ==================== v1/shared_storage_connector.py  ====================
def _patch_shared_storage_connector() -> None:
    """Patch kv connector utils to add UCM sparse support."""
    try:
        from dataclasses import dataclass, field

        from vllm.distributed.kv_transfer.kv_connector.v1 import (
            shared_storage_connector,
        )
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorMetadata,
        )
        from vllm.distributed.kv_transfer.kv_connector.v1.shared_storage_connector import (
            ReqMeta,
        )

        @dataclass
        class SharedStorageConnectorMetadata(KVConnectorMetadata):
            requests: list[ReqMeta] = field(default_factory=list)

            def add_request(
                self,
                token_ids: list[int],
                block_ids: list[int],
                block_size: int,
                is_store: bool,
            ) -> None:
                self.requests.append(
                    ReqMeta.make_meta(token_ids, block_ids, block_size, is_store)
                )

        shared_storage_connector.SharedStorageConnectorMetadata = (
            SharedStorageConnectorMetadata
        )
    except ImportError:
        logger.warning("Could not patch shared storage connector - module not found")


# ==================== vllm/v1/attention/backends/mla/common.py  ====================
def _patch_mla_common() -> None:
    """Patch mla common to add UCM sparse support."""
    try:
        from typing import Optional, TypeVar

        import torch
        from vllm import _custom_ops as ops
        from vllm.attention.backends.abstract import AttentionLayer
        from vllm.attention.layer import (
            maybe_execute_sparse_attention_begin,
            maybe_execute_sparse_attention_finished,
        )
        from vllm.forward_context import ForwardContext, get_forward_context
        from vllm.v1.attention.backends.mla.common import (
            MLACommonImpl,
            MLACommonMetadata,
        )

        M = TypeVar("M", bound=MLACommonMetadata)

        def forward(
            self,
            layer: AttentionLayer,
            q: torch.Tensor,
            k_c_normed: torch.Tensor,  # key in unified attn
            k_pe: torch.Tensor,  # value in unified attn
            kv_cache: torch.Tensor,
            attn_metadata: M,
            output: Optional[torch.Tensor] = None,
            output_scale: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            forward_context: ForwardContext = get_forward_context()
            assert output is not None, "Output tensor must be provided."

            if output_scale is not None:
                raise NotImplementedError(
                    "fused output quantization is not yet supported"
                    " for MLACommonImpl"
                )

            if attn_metadata is None:
                # The zero fill is required when used with DP + EP
                # to ensure all ranks within a DP group compute the
                # same expert outputs.
                return output.fill_(0)

            num_actual_toks = attn_metadata.num_actual_tokens

            # Inputs and outputs may be padded for CUDA graphs
            output_padded = output
            output = output[:num_actual_toks, ...]
            q = q[:num_actual_toks, ...]
            k_c_normed = k_c_normed[:num_actual_toks, ...]
            k_pe = k_pe[:num_actual_toks, ...]

            assert (
                attn_metadata.num_decodes is not None
                and attn_metadata.num_prefills is not None
                and attn_metadata.num_decode_tokens is not None
            )

            has_decode = attn_metadata.num_decodes > 0
            has_prefill = attn_metadata.num_prefills > 0
            num_decode_tokens = attn_metadata.num_decode_tokens

            decode_q = q[:num_decode_tokens]

            prefill_q = q[num_decode_tokens:]
            prefill_k_pe = k_pe[num_decode_tokens:]
            prefill_k_c_normed = k_c_normed[num_decode_tokens:]

            # write the latent and rope to kv cache
            if kv_cache.numel() > 0:
                ops.concat_and_cache_mla(
                    k_c_normed,
                    k_pe.squeeze(1),
                    kv_cache,
                    attn_metadata.slot_mapping.flatten(),
                    kv_cache_dtype=self.kv_cache_dtype,
                    scale=layer._k_scale,
                )

            if has_prefill:
                prefill_q, prefill_k_c_normed, prefill_k_pe, _ = (
                    maybe_execute_sparse_attention_begin(
                        prefill_q,
                        prefill_k_c_normed,
                        prefill_k_pe,
                        layer.layer_name,
                        forward_context,
                        phase="prefill",
                    )
                )
                output[num_decode_tokens:] = self._forward_prefill(
                    prefill_q, prefill_k_c_normed, prefill_k_pe, kv_cache, attn_metadata
                )
                maybe_execute_sparse_attention_finished(
                    prefill_q,
                    prefill_k_c_normed,
                    prefill_k_pe,
                    output[num_decode_tokens:],
                    layer.layer_name,
                    forward_context,
                    "prefill",
                )
            if has_decode:
                assert attn_metadata.decode is not None
                decode_q_nope, decode_q_pe = decode_q.split(
                    [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
                )
                # Convert from (B, N, P) to (N, B, P)
                decode_q_nope = decode_q_nope.transpose(0, 1)
                # Multiply (N, B, P) x (N, P, L) -> (N, B, L)
                decode_ql_nope = torch.bmm(decode_q_nope, self.W_UK_T)
                # Convert from (N, B, L) to (B, N, L)
                decode_ql_nope = decode_ql_nope.transpose(0, 1)
                _, decode_ql_nope, decode_q_pe, _ = (
                    maybe_execute_sparse_attention_begin(
                        torch.cat([decode_ql_nope, decode_q_pe], dim=-1),
                        decode_ql_nope,
                        decode_q_pe,
                        layer.layer_name,
                        forward_context,
                        phase="decode",
                    )
                )
                output[:num_decode_tokens] = self._forward_decode(
                    decode_ql_nope, decode_q_pe, kv_cache, attn_metadata
                )
                maybe_execute_sparse_attention_finished(
                    torch.cat([decode_ql_nope, decode_q_pe], dim=-1),
                    decode_ql_nope,
                    decode_q_pe,
                    output[:num_decode_tokens],
                    layer.layer_name,
                    forward_context,
                    "decode",
                )
            return output_padded

        MLACommonImpl.forward = forward
    except ImportError:
        logger.warning("Could not patch mla common - module not found")


# ==================== v1/core/kv_cache_manager.py  ====================
def _patch_kv_cache_manager() -> None:
    """Patch kv cache manager to add UCM sparse support."""
    try:
        from typing import Optional, Union

        from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
        from vllm.v1.request import Request

        from ucm.sparse.base import INVALID_SLOT
        from ucm.sparse.state import get_ucm_sparse

        original_allocate_slots = KVCacheManager.allocate_slots

        def patched_allocate_slots(
            self,
            request: Request,
            num_new_tokens: int,
            num_new_computed_tokens: int = 0,
            new_computed_blocks: Optional[KVCacheBlocks] = None,
            num_draft_tokens: int = 0,
            num_lookahead_tokens: int = 0,
            delay_cache_blocks: bool = False,
            num_slots_sparsed: Union[None, int] = None,
        ) -> Optional[KVCacheBlocks]:
            if num_new_tokens == 0:
                raise ValueError("num_new_tokens must be greater than 0")
            # Only route to UCM sparse path when caller explicitly provided
            # a valid sparsified slot count.
            if (num_slots_sparsed is not None) and (num_slots_sparsed != INVALID_SLOT):
                return get_ucm_sparse().allocate_slots(self, request, num_slots_sparsed)
            return original_allocate_slots(
                self,
                request,
                num_new_tokens,
                num_new_computed_tokens,
                new_computed_blocks,
                num_draft_tokens,
                num_lookahead_tokens,
                delay_cache_blocks,
            )

        KVCacheManager.allocate_slots = patched_allocate_slots
    except ImportError:
        logger.warning("Could not patch kv cache manager - module not found")


# ==================== vllm/v1/core/sched/scheduler.py ====================
def _patch_scheduler() -> None:
    """Patch Scheduler to add num_output_tokens field."""
    try:
        import itertools
        import time
        from collections import defaultdict
        from collections.abc import Iterable
        from typing import Optional

        from vllm.distributed.kv_events import KVEventBatch
        from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
            MultiConnector,
        )
        from vllm.v1.core.sched.output import (
            CachedRequestData,
            NewRequestData,
            SchedulerOutput,
        )
        from vllm.v1.core.sched.request_queue import (
            SchedulingPolicy,
            create_request_queue,
        )
        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm.v1.core.sched.utils import check_stop
        from vllm.v1.engine import (
            EngineCoreEventType,
            EngineCoreOutput,
            EngineCoreOutputs,
        )
        from vllm.v1.outputs import ModelRunnerOutput
        from vllm.v1.request import Request, RequestStatus
        from vllm.v1.spec_decode.metrics import SpecDecodingStats

        from ucm.sparse.base import INVALID_SLOT, UcmSparseRole
        from ucm.sparse.state import ensure_ucm_sparse_initialized, get_ucm_sparse

        def init_ucm_sparse(self):
            self.ucm_sparse = None
            if self.vllm_config.kv_transfer_config is not None:
                if (
                    "ucm_sparse_config"
                    in self.vllm_config.kv_transfer_config.kv_connector_extra_config
                ):
                    ensure_ucm_sparse_initialized(
                        self.vllm_config, role=UcmSparseRole.SCHEDULER
                    )
                    self.ucm_sparse = get_ucm_sparse()
                    logger.info(
                        "UCM Sparse initialized successfully: {}".format(
                            self.ucm_sparse
                        )
                    )

        def patched_schedule(self) -> SchedulerOutput:
            # NOTE(woosuk) on the scheduling algorithm:
            # There's no "decoding phase" nor "prefill phase" in the scheduler.
            # Each request just has the num_computed_tokens and
            # num_tokens_with_spec. num_tokens_with_spec =
            # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
            # At each step, the scheduler tries to assign tokens to the requests
            # so that each request's num_computed_tokens can catch up its
            # num_tokens_with_spec. This is general enough to cover
            # chunked prefills, prefix caching, speculative decoding,
            # and the "jump decoding" optimization in the future.

            scheduled_new_reqs: list[Request] = []
            scheduled_resumed_reqs: list[Request] = []
            scheduled_running_reqs: list[Request] = []
            preempted_reqs: list[Request] = []

            # NOTE: structured_output_request_ids maps
            # a request's (request that uses structured output)
            # request_id to the running request index.
            # This will helps us determine to slice the grammar bitmask
            # and only applies valid mask for requests that
            # uses structured decoding.
            structured_output_request_ids: dict[str, int] = {}

            req_to_new_block_ids: dict[str, tuple[list[int], ...]] = {}
            num_scheduled_tokens: dict[str, int] = {}
            token_budget = self.max_num_scheduled_tokens
            # Encoder-related.
            scheduled_encoder_inputs: dict[str, list[int]] = {}
            encoder_budget = self.max_num_encoder_input_tokens
            # Spec decode-related.
            scheduled_spec_decode_tokens: dict[str, list[int]] = {}

            # For logging.
            scheduled_timestamp = time.monotonic()

            # First, schedule the RUNNING requests.
            req_index = 0
            req_sparsed_slots: dict[str, int] = {}
            if not hasattr(self, "ucm_sparse"):
                init_ucm_sparse(self)
            while req_index < len(self.running) and token_budget > 0:
                request = self.running[req_index]
                num_slots_sparsed = INVALID_SLOT
                if self.ucm_sparse:
                    num_slots_sparsed = self.ucm_sparse.estimate_num_slots_sparsed(
                        request
                    )
                req_sparsed_slots.update({request.request_id: num_slots_sparsed})

                num_new_tokens = (
                    request.num_tokens_with_spec - request.num_computed_tokens
                )
                if (
                    0
                    < self.scheduler_config.long_prefill_token_threshold
                    < num_new_tokens
                ):
                    num_new_tokens = self.scheduler_config.long_prefill_token_threshold
                num_new_tokens = min(num_new_tokens, token_budget)

                # Make sure the input position does not exceed the max model len.
                # This is necessary when using spec decoding.
                num_new_tokens = min(
                    num_new_tokens, self.max_model_len - 1 - request.num_computed_tokens
                )

                # Schedule encoder inputs.
                encoder_inputs_to_schedule = None
                new_encoder_budget = encoder_budget
                if request.has_encoder_inputs:
                    (encoder_inputs_to_schedule, num_new_tokens, new_encoder_budget) = (
                        self._try_schedule_encoder_inputs(
                            request,
                            request.num_computed_tokens,
                            num_new_tokens,
                            encoder_budget,
                        )
                    )

                if num_new_tokens == 0:
                    # The request cannot be scheduled because one of the following
                    # reasons:
                    # 1. No new tokens to schedule. This may happen when PP>1 and
                    #    we have already scheduled all prompt tokens but they are
                    #    not finished yet.
                    # 2. The encoder budget is exhausted.
                    # 3. The encoder cache is exhausted.
                    # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                    # we do not strictly follow the FCFS scheduling policy and
                    # allow the lower-priority requests to be scheduled.
                    req_index += 1
                    continue

                num_draft_tokens = max(
                    num_new_tokens + request.num_computed_tokens - request.num_tokens, 0
                )

                while True:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens,
                        num_draft_tokens=num_draft_tokens,
                        num_lookahead_tokens=self.num_lookahead_tokens,
                        num_slots_sparsed=num_slots_sparsed,
                    )
                    if new_blocks is None:
                        # The request cannot be scheduled.
                        # Preempt the lowest-priority request.
                        if self.policy == SchedulingPolicy.PRIORITY:
                            preempted_req = max(
                                self.running,
                                key=lambda r: (r.priority, r.arrival_time),
                            )
                            self.running.remove(preempted_req)
                        else:
                            preempted_req = self.running.pop()

                        self.kv_cache_manager.free(preempted_req)
                        preempted_req.status = RequestStatus.PREEMPTED
                        preempted_req.num_computed_tokens = 0
                        if self.log_stats:
                            preempted_req.record_event(
                                EngineCoreEventType.PREEMPTED, scheduled_timestamp
                            )

                        self.waiting.prepend_request(preempted_req)
                        preempted_reqs.append(preempted_req)
                        if preempted_req == request:
                            # No more request to preempt.
                            can_schedule = False
                            break
                    else:
                        # The request can be scheduled.
                        can_schedule = True
                        break
                if not can_schedule:
                    break
                assert new_blocks is not None

                # Schedule the request.
                scheduled_running_reqs.append(request)
                if request.use_structured_output:
                    # PERF: in case of chunked prefill,
                    # request might not include any new tokens.
                    # Therefore, we might introduce some additional
                    # cycle to fill in the bitmask, which could be a big no-op.
                    structured_output_request_ids[request.request_id] = req_index
                req_to_new_block_ids[request.request_id] = new_blocks.get_block_ids()
                num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens
                req_index += 1

                # Speculative decode related.
                if request.spec_token_ids:
                    num_scheduled_spec_tokens = (
                        num_new_tokens
                        + request.num_computed_tokens
                        - request.num_tokens
                    )
                    if num_scheduled_spec_tokens > 0:
                        # Trim spec_token_ids list to num_scheduled_spec_tokens.
                        del request.spec_token_ids[num_scheduled_spec_tokens:]
                        scheduled_spec_decode_tokens[request.request_id] = (
                            request.spec_token_ids
                        )

                # Encoder-related.
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request.request_id] = (
                        encoder_inputs_to_schedule
                    )
                    # Allocate the encoder cache.
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                    encoder_budget = new_encoder_budget

            # Record the LoRAs in scheduled_running_reqs
            scheduled_loras: set[int] = set()
            if self.lora_config:
                scheduled_loras = set(
                    req.lora_request.lora_int_id
                    for req in scheduled_running_reqs
                    if req.lora_request and req.lora_request.lora_int_id > 0
                )
                assert len(scheduled_loras) <= self.lora_config.max_loras

            # Use a temporary RequestQueue to collect requests that need to be
            # skipped and put back at the head of the waiting queue later
            skipped_waiting_requests = create_request_queue(self.policy)

            # Next, schedule the WAITING requests.
            if not preempted_reqs:
                while self.waiting and token_budget > 0:
                    if len(self.running) == self.max_num_running_reqs:
                        break

                    request = self.waiting.peek_request()
                    num_slots_sparsed = INVALID_SLOT
                    if self.ucm_sparse:
                        num_slots_sparsed = self.ucm_sparse.estimate_num_slots_sparsed(
                            request
                        )
                    req_sparsed_slots.update({request.request_id: num_slots_sparsed})

                    # KVTransfer: skip request if still waiting for remote kvs.
                    if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                        is_ready = self._update_waiting_for_remote_kv(request)
                        if is_ready:
                            request.status = RequestStatus.WAITING
                        else:
                            logger.debug(
                                "%s is still in WAITING_FOR_REMOTE_KVS state.",
                                request.request_id,
                            )
                            self.waiting.pop_request()
                            skipped_waiting_requests.prepend_request(request)
                            continue

                    # Skip request if the structured output request is still waiting
                    # for FSM compilation.
                    if request.status == RequestStatus.WAITING_FOR_FSM:
                        structured_output_req = request.structured_output_request
                        if structured_output_req and structured_output_req.grammar:
                            request.status = RequestStatus.WAITING
                        else:
                            self.waiting.pop_request()
                            skipped_waiting_requests.prepend_request(request)
                            continue

                    # Check that adding the request still respects the max_loras
                    # constraint.
                    if (
                        self.lora_config
                        and request.lora_request
                        and (
                            len(scheduled_loras) == self.lora_config.max_loras
                            and request.lora_request.lora_int_id not in scheduled_loras
                        )
                    ):
                        # Scheduling would exceed max_loras, skip.
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                    num_external_computed_tokens = 0
                    load_kv_async = False

                    # Get already-cached tokens.
                    if request.num_computed_tokens == 0:
                        # Get locally-cached tokens.
                        new_computed_blocks, num_new_local_computed_tokens = (
                            self.kv_cache_manager.get_computed_blocks(request)
                        )

                        # Get externally-cached tokens if using a KVConnector.
                        if self.connector is not None:
                            num_external_computed_tokens, load_kv_async = (
                                self.connector.get_num_new_matched_tokens(
                                    request, num_new_local_computed_tokens
                                )
                            )

                        # Total computed tokens (local + external).
                        num_computed_tokens = (
                            num_new_local_computed_tokens + num_external_computed_tokens
                        )
                    # KVTransfer: WAITING reqs have num_computed_tokens > 0
                    # after async KV recvs are completed.
                    else:
                        new_computed_blocks = (
                            self.kv_cache_manager.create_empty_block_list()
                        )
                        num_new_local_computed_tokens = 0
                        num_computed_tokens = request.num_computed_tokens

                    encoder_inputs_to_schedule = None
                    new_encoder_budget = encoder_budget

                    # KVTransfer: loading remote KV, do not allocate for new work.
                    if load_kv_async:
                        assert num_external_computed_tokens > 0
                        num_new_tokens = 0
                    # Number of tokens to be scheduled.
                    else:
                        # We use `request.num_tokens` instead of
                        # `request.num_prompt_tokens` to consider the resumed
                        # requests, which have output tokens.
                        num_new_tokens = request.num_tokens - num_computed_tokens
                        if (
                            0
                            < self.scheduler_config.long_prefill_token_threshold
                            < num_new_tokens
                        ):
                            num_new_tokens = (
                                self.scheduler_config.long_prefill_token_threshold
                            )

                        # chunked prefill has to be enabled explicitly to allow
                        # pooling requests to be chunked
                        if (
                            not self.scheduler_config.chunked_prefill_enabled
                            and num_new_tokens > token_budget
                        ):
                            self.waiting.pop_request()
                            skipped_waiting_requests.prepend_request(request)
                            continue

                        num_new_tokens = min(num_new_tokens, token_budget)
                        assert num_new_tokens > 0

                        # Schedule encoder inputs.
                        if request.has_encoder_inputs:
                            (
                                encoder_inputs_to_schedule,
                                num_new_tokens,
                                new_encoder_budget,
                            ) = self._try_schedule_encoder_inputs(
                                request,
                                num_computed_tokens,
                                num_new_tokens,
                                encoder_budget,
                            )
                            if num_new_tokens == 0:
                                # The request cannot be scheduled.
                                break

                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens + num_external_computed_tokens,
                        num_new_local_computed_tokens,
                        new_computed_blocks,
                        num_lookahead_tokens=self.num_lookahead_tokens,
                        delay_cache_blocks=load_kv_async,
                        num_slots_sparsed=num_slots_sparsed,
                    )
                    if new_blocks is None:
                        # The request cannot be scheduled.
                        break

                    # KVTransfer: the connector uses this info to determine
                    # if a load is needed. Note that
                    # This information is used to determine if a load is
                    # needed for this request.
                    if self.connector is not None:
                        self.connector.update_state_after_alloc(
                            request,
                            new_computed_blocks + new_blocks,
                            num_external_computed_tokens,
                        )

                    # Request was already popped from self.waiting
                    # unless it was re-added above due to new_blocks being None.
                    request = self.waiting.pop_request()
                    if load_kv_async:
                        # If loading async, allocate memory and put request
                        # into the WAITING_FOR_REMOTE_KV state.
                        skipped_waiting_requests.prepend_request(request)
                        request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                        continue

                    if request.use_structured_output:
                        structured_output_request_ids[request.request_id] = req_index
                    req_index += 1
                    self.running.append(request)
                    if self.log_stats:
                        request.record_event(
                            EngineCoreEventType.SCHEDULED, scheduled_timestamp
                        )
                    if request.status == RequestStatus.WAITING:
                        scheduled_new_reqs.append(request)
                    elif request.status == RequestStatus.PREEMPTED:
                        scheduled_resumed_reqs.append(request)
                    else:
                        raise RuntimeError(f"Invalid request status: {request.status}")

                    if self.lora_config and request.lora_request:
                        scheduled_loras.add(request.lora_request.lora_int_id)
                    req_to_new_block_ids[request.request_id] = (
                        self.kv_cache_manager.get_block_ids(request.request_id)
                    )
                    num_scheduled_tokens[request.request_id] = num_new_tokens
                    token_budget -= num_new_tokens
                    request.status = RequestStatus.RUNNING
                    request.num_computed_tokens = num_computed_tokens
                    # Count the number of prefix cached tokens.
                    if request.num_cached_tokens < 0:
                        request.num_cached_tokens = num_computed_tokens
                    # Encoder-related.
                    if encoder_inputs_to_schedule:
                        scheduled_encoder_inputs[request.request_id] = (
                            encoder_inputs_to_schedule
                        )
                        # Allocate the encoder cache.
                        for i in encoder_inputs_to_schedule:
                            self.encoder_cache_manager.allocate(request, i)
                        encoder_budget = new_encoder_budget

            # Put back any skipped requests at the head of the waiting queue
            if skipped_waiting_requests:
                self.waiting.prepend_requests(skipped_waiting_requests)

            # Check if the scheduling constraints are satisfied.
            total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
            assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
            assert token_budget >= 0
            assert len(self.running) <= self.max_num_running_reqs
            # Since some requests in the RUNNING queue may not be scheduled in
            # this step, the total number of scheduled requests can be smaller than
            # len(self.running).
            assert len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(
                scheduled_running_reqs
            ) <= len(self.running)

            # Get the longest common prefix among all requests in the running queue.
            # This can be potentially used for cascade attention.
            num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
            if self.running:
                any_request = self.running[0]
                num_common_prefix_blocks = (
                    self.kv_cache_manager.get_num_common_prefix_blocks(
                        any_request, len(self.running)
                    )
                )

            grammar_bitmask = self.structured_output_manager.grammar_bitmask(
                self.requests,
                structured_output_request_ids,
                scheduled_spec_decode_tokens,
            )
            # Construct the scheduler output.
            new_reqs_data = [
                NewRequestData.from_request(req, req_to_new_block_ids[req.request_id])
                for req in scheduled_new_reqs
            ]
            cached_reqs_data = self._make_cached_request_data(
                scheduled_running_reqs,
                scheduled_resumed_reqs,
                num_scheduled_tokens,
                scheduled_spec_decode_tokens,
                req_to_new_block_ids,
            )
            scheduler_output = SchedulerOutput(
                scheduled_new_reqs=new_reqs_data,
                scheduled_cached_reqs=cached_reqs_data,
                num_scheduled_tokens=num_scheduled_tokens,
                total_num_scheduled_tokens=total_num_scheduled_tokens,
                scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
                scheduled_encoder_inputs=scheduled_encoder_inputs,
                num_common_prefix_blocks=num_common_prefix_blocks,
                req_sparsed_slots=req_sparsed_slots,
                # finished_req_ids is an existing state in the scheduler,
                # instead of being newly scheduled in this step.
                # It contains the request IDs that are finished in between
                # the previous and the current steps.
                finished_req_ids=self.finished_req_ids,
                free_encoder_input_ids=self.encoder_cache_manager.get_freed_ids(),
                structured_output_request_ids=structured_output_request_ids,
                grammar_bitmask=grammar_bitmask,
            )

            # NOTE(Kuntai): this function is designed for multiple purposes:
            # 1. Plan the KV cache store
            # 2. Wrap up all the KV cache load / save ops into an opaque object
            # 3. Clear the internal states of the connector
            if self.connector is not None:
                meta = self.connector.build_connector_meta(scheduler_output)
                scheduler_output.kv_connector_metadata = meta

            events = self.kv_cache_manager.take_events()
            if events:
                batch = KVEventBatch(ts=time.time(), events=events)
                self.kv_event_publisher.publish(batch)

            self._update_after_schedule(scheduler_output)
            return scheduler_output

        Scheduler.schedule = patched_schedule

        def patched_add_request(self, request: Request) -> None:
            if not hasattr(self, "ucm_sparse"):
                init_ucm_sparse(self)
            self.waiting.add_request(request)
            self.requests[request.request_id] = request
            if self.ucm_sparse:
                self.ucm_sparse.request_begin(
                    request.request_id, request.prompt_token_ids
                )
            if self.log_stats:
                request.record_event(EngineCoreEventType.QUEUED)

        Scheduler.add_request = patched_add_request

        original_free_request = Scheduler._free_request

        def patched_free_request(self, request: Request):
            assert request.is_finished()
            if not hasattr(self, "ucm_sparse"):
                init_ucm_sparse(self)
            if self.ucm_sparse:
                self.ucm_sparse.request_finished_in_scheduler(request.request_id)
            original_free_request(self, request)

        Scheduler._free_request = patched_free_request
    except ImportError:
        logger.warning("Could not patch Scheduler - module not found")


# ==================== vllm/v1/worker/block_table.py  ====================
def _patch_block_table() -> None:
    """Patch block table to add UCM sparse support."""
    try:
        from vllm.v1.worker.block_table import BlockTable, MultiGroupBlockTable

        def reset_row(
            self,
            row_idx: int,
        ) -> None:
            self.num_blocks_per_row[row_idx] = 0
            self.block_table[row_idx].fill_(0)
            self.block_table_cpu[row_idx].fill_(0)
            self.block_table_np[row_idx].fill(0)

        BlockTable.reset_row = reset_row

        def reset_row(self, row_idx: int) -> None:
            for i, block_table in enumerate(self.block_tables):
                block_table.reset_row(row_idx)

        MultiGroupBlockTable.reset_row = reset_row
    except ImportError:
        logger.warning("Could not patch multigroup block table - module not found")


# ==================== vllm/v1/worker/gpu_model_runner.py  ====================
def _patch_gpu_model_runner() -> None:
    """Patch gpu model runner to add UCM sparse support."""
    try:
        import copy
        from typing import TYPE_CHECKING, Any, Optional

        import numpy as np
        import torch
        import vllm.envs as envs
        from vllm.distributed.kv_transfer import (
            get_kv_transfer_group,
            has_kv_transfer_group,
        )
        from vllm.distributed.parallel_state import get_pp_group
        from vllm.forward_context import set_forward_context
        from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
        from vllm.sampling_params import SamplingType
        from vllm.sequence import IntermediateTensors
        from vllm.utils import round_up
        from vllm.v1.attention.backends.utils import CommonAttentionMetadata
        from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
        from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
        from vllm.v1.worker.block_table import BlockTable
        from vllm.v1.worker.gpu_input_batch import CachedRequestState

        from ucm.sparse.base import INVALID_SLOT
        from ucm.sparse.state import get_ucm_sparse, has_ucm_sparse

        if TYPE_CHECKING:
            from vllm.v1.core.sched.output import SchedulerOutput

        from vllm.v1.worker.gpu_model_runner import GPUModelRunner

        @staticmethod
        def maybe_wait_for_kv_save() -> Optional[dict[str, list[str]]]:
            if has_kv_transfer_group():
                return get_kv_transfer_group().wait_for_save()
            return None

        GPUModelRunner.maybe_wait_for_kv_save = maybe_wait_for_kv_save

        def maybe_execute_ucm_sparse_begin(
            self,
            scheduler_output: "SchedulerOutput",
            attn_metadata: CommonAttentionMetadata,
        ):
            if not has_ucm_sparse():
                return

            if has_kv_transfer_group():
                uc_connector = get_kv_transfer_group()
                uc_setup_model = getattr(uc_connector, "setup_model", None)
                if callable(uc_setup_model):
                    uc_setup_model(self.model)

            ucm_sparse = get_ucm_sparse()
            ucm_sparse.build_sparse_meta(
                scheduler_output, self.requests, self.input_batch, attn_metadata
            )
            ucm_sparse.execute_begin(scheduler_output)

        def maybe_execute_ucm_sparse_finished(self, logits_indices):
            if not has_ucm_sparse():
                return logits_indices
            ucm_sparse = get_ucm_sparse()
            return ucm_sparse.execute_finished(logits_indices)

        def ucm_sparse_request_finished_in_worker(self, request_id: str | int):
            if not has_ucm_sparse():
                return
            ucm_sparse = get_ucm_sparse()
            ucm_sparse.request_finished_in_worker(request_id)

        GPUModelRunner.maybe_execute_ucm_sparse_begin = maybe_execute_ucm_sparse_begin
        GPUModelRunner.maybe_execute_ucm_sparse_finished = (
            maybe_execute_ucm_sparse_finished
        )
        GPUModelRunner.ucm_sparse_request_finished_in_worker = (
            ucm_sparse_request_finished_in_worker
        )

        def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
            """Update the cached states and the persistent batch with the scheduler
            output.

            The updated states are used by the `_prepare_inputs` function to create
            the input GPU tensors for the model.

            The SamplingMetadata is updated and copied to the GPU if there is a
            new/resumed/paused/finished request in the batch.
            """
            # Remove finished requests from the cached states.
            for req_id in scheduler_output.finished_req_ids:
                self.ucm_sparse_request_finished_in_worker(req_id)
                self.requests.pop(req_id, None)
                self.encoder_cache.pop(req_id, None)
            # Remove the finished requests from the persistent batch.
            # NOTE(woosuk): There could be an edge case where finished_req_ids and
            # scheduled_req_ids overlap. This happens when a request is aborted and
            # then resubmitted with the same ID. In this case, we treat them as two
            # distinct requests - clearing the cached states for the first request
            # and handling the second as a new request.
            for req_id in scheduler_output.finished_req_ids:
                self.input_batch.remove_request(req_id)

            # Free the cached encoder outputs.
            for req_id, input_id in scheduler_output.free_encoder_input_ids:
                encoder_outputs = self.encoder_cache.get(req_id)
                if encoder_outputs is not None:
                    encoder_outputs.pop(input_id, None)
                    if not encoder_outputs:
                        self.encoder_cache.pop(req_id, None)

            # Remove the unscheduled requests from the persistent batch.
            # NOTE(woosuk): The unscheduled requests are either preempted requests
            # or running requests that are not scheduled in this step. We remove
            # them from the persistent batch but keep their cached states since
            # they will be scheduled again sometime in the future.
            scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
            cached_req_ids = self.input_batch.req_id_to_index.keys()
            unscheduled_req_ids = cached_req_ids - scheduled_req_ids
            # NOTE(woosuk): The persistent batch optimization assumes that
            # consecutive batches contain mostly the same requests. If batches
            # have low request overlap (e.g., alternating between two distinct
            # sets of requests), this optimization becomes very inefficient.
            for req_id in unscheduled_req_ids:
                self.input_batch.remove_request(req_id)

            req_ids_to_add: list[str] = []
            # Add new requests to the cached states.
            for new_req_data in scheduler_output.scheduled_new_reqs:
                req_id = new_req_data.req_id
                sampling_params = new_req_data.sampling_params
                pooling_params = new_req_data.pooling_params
                if (
                    sampling_params
                    and sampling_params.sampling_type == SamplingType.RANDOM_SEED
                ):
                    generator = torch.Generator(device=self.device)
                    generator.manual_seed(sampling_params.seed)
                else:
                    generator = None

                self.requests[req_id] = CachedRequestState(
                    req_id=req_id,
                    prompt_token_ids=new_req_data.prompt_token_ids,
                    mm_inputs=new_req_data.mm_inputs,
                    mm_positions=new_req_data.mm_positions,
                    sampling_params=sampling_params,
                    pooling_params=pooling_params,
                    generator=generator,
                    block_ids=new_req_data.block_ids,
                    num_computed_tokens=new_req_data.num_computed_tokens,
                    output_token_ids=[],
                    lora_request=new_req_data.lora_request,
                )

                # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
                if self.uses_mrope:
                    image_grid_thw = []
                    video_grid_thw = []
                    second_per_grid_ts = []
                    audio_feature_lengths = []
                    use_audio_in_video = False
                    for mm_input in self.requests[req_id].mm_inputs:
                        if mm_input.get("image_grid_thw") is not None:
                            image_grid_thw.extend(mm_input["image_grid_thw"].tolist())
                        if mm_input.get("video_grid_thw") is not None:
                            video_grid_thw.extend(mm_input["video_grid_thw"].tolist())
                        if mm_input.get("second_per_grid_ts") is not None:
                            second_per_grid_ts.extend(mm_input["second_per_grid_ts"])
                        if mm_input.get("audio_feature_lengths") is not None:
                            audio_feature_lengths.extend(
                                mm_input["audio_feature_lengths"]
                            )
                        if mm_input.get("use_audio_in_video") is True:
                            use_audio_in_video = True

                    hf_config = self.model_config.hf_config

                    (
                        self.requests[req_id].mrope_positions,
                        self.requests[req_id].mrope_position_delta,
                    ) = MRotaryEmbedding.get_input_positions_tensor(
                        self.requests[req_id].prompt_token_ids,
                        hf_config=hf_config,
                        image_grid_thw=image_grid_thw,
                        video_grid_thw=video_grid_thw,
                        second_per_grid_ts=second_per_grid_ts,
                        audio_feature_lengths=audio_feature_lengths,
                        use_audio_in_video=use_audio_in_video,
                    )

                req_ids_to_add.append(req_id)

            # Update the states of the running/resumed requests.
            is_last_rank = get_pp_group().is_last_rank
            req_data = scheduler_output.scheduled_cached_reqs
            req_sparsed_slots = scheduler_output.req_sparsed_slots
            for i, req_id in enumerate(req_data.req_ids):
                req_state = self.requests[req_id]
                num_computed_tokens = req_data.num_computed_tokens[i]
                new_block_ids = req_data.new_block_ids[i]
                resumed_from_preemption = req_data.resumed_from_preemption[i]
                is_sparsed_request = req_sparsed_slots[req_id] != INVALID_SLOT

                # Update the cached states.
                req_state.num_computed_tokens = num_computed_tokens

                if not is_last_rank:
                    # When using PP, the scheduler sends the sampled tokens back,
                    # because there's no direct communication between the first-
                    # stage worker and the last-stage worker.
                    new_token_ids = req_data.new_token_ids[i]
                    # Add the sampled token(s) from the previous step (if any).
                    # This doesn't include "unverified" tokens like spec tokens.
                    num_new_tokens = (
                        num_computed_tokens + len(new_token_ids) - req_state.num_tokens
                    )
                    if num_new_tokens == 1:
                        # Avoid slicing list in most common case.
                        req_state.output_token_ids.append(new_token_ids[-1])
                    elif num_new_tokens > 0:
                        req_state.output_token_ids.extend(
                            new_token_ids[-num_new_tokens:]
                        )

                # Update the block IDs.
                if resumed_from_preemption or is_sparsed_request:
                    # The request is resumed from preemption.
                    # Replace the existing block IDs with the new ones.
                    req_state.block_ids = new_block_ids
                else:
                    # Append the new blocks to the existing block IDs.
                    for block_ids, new_ids in zip(req_state.block_ids, new_block_ids):
                        block_ids.extend(new_ids)

                req_index = self.input_batch.req_id_to_index.get(req_id)
                if req_index is None:
                    # The request is not in the persistent batch.
                    # The request was either preempted and resumed later, or was not
                    # scheduled in the previous step and needs to be added again.
                    req_ids_to_add.append(req_id)
                    continue

                # Update the persistent batch.
                self.input_batch.num_computed_tokens_cpu[req_index] = (
                    num_computed_tokens
                )
                if is_sparsed_request:
                    self.input_batch.block_table.reset_row(req_index)
                self.input_batch.block_table.append_row(new_block_ids, req_index)

                # For the last rank, we don't need to update the token_ids_cpu
                # because the sampled tokens are already cached.
                if not is_last_rank:
                    # Add new_token_ids to token_ids_cpu.
                    start_token_index = num_computed_tokens
                    end_token_index = num_computed_tokens + len(new_token_ids)
                    self.input_batch.token_ids_cpu[
                        req_index, start_token_index:end_token_index
                    ] = new_token_ids
                    self.input_batch.num_tokens_no_spec[req_index] = end_token_index
                    # Add spec_token_ids to token_ids_cpu.
                    spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(
                        req_id, ()
                    )
                    if spec_token_ids:
                        start_index = end_token_index
                        end_token_index += len(spec_token_ids)
                        self.input_batch.token_ids_cpu[
                            req_index, start_index:end_token_index
                        ] = spec_token_ids
                    # NOTE(woosuk): `num_tokens` here may include spec tokens.
                    self.input_batch.num_tokens[req_index] = end_token_index

            # Add the new or resumed requests to the persistent batch.
            # The smaller empty indices are filled first.
            for req_id in req_ids_to_add:
                req_state = self.requests[req_id]
                self.input_batch.add_request(req_state)

            # Condense the batched states if there are gaps left by removed requests
            self.input_batch.condense()
            # Allow attention backend to reorder the batch, potentially
            self._may_reorder_batch(scheduler_output)
            # Refresh batch metadata with any pending updates.
            self.input_batch.refresh_metadata()

        GPUModelRunner._update_states = _update_states

        def _prepare_inputs(
            self,
            scheduler_output: "SchedulerOutput",
        ) -> tuple[
            dict[str, Any], bool, torch.Tensor, Optional[SpecDecodeMetadata], np.ndarray
        ]:
            """
            :return: tuple[
                attn_metadata: layer-to-attention_metadata mapping,
                attention_cuda_graphs: whether attention can run in cudagraph
                logits_indices, spec_decode_metadata
            ]
            """
            total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
            assert total_num_scheduled_tokens > 0
            num_reqs = self.input_batch.num_reqs
            assert num_reqs > 0

            # OPTIMIZATION: Start copying the block table first.
            # This way, we can overlap the copy with the following CPU operations.
            self.input_batch.block_table.commit(num_reqs)

            # Get the number of scheduled tokens for each request.
            req_ids = self.input_batch.req_ids
            tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
            num_scheduled_tokens = np.array(tokens, dtype=np.int32)
            max_num_scheduled_tokens = max(tokens)

            # Get request indices.
            # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
            req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)

            # cu_num_tokens: [2, 5, 3] -> [2, 7, 10]
            # arange: [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
            cu_num_tokens, arange = self._get_cumsum_and_arange(num_scheduled_tokens)

            # Get positions.
            positions_np = self.positions_np[:total_num_scheduled_tokens]
            np.add(
                self.input_batch.num_computed_tokens_cpu[req_indices],
                arange,
                out=positions_np,
            )

            # Calculate M-RoPE positions.
            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            if self.uses_mrope:
                self._calc_mrope_positions(scheduler_output)

            self.seq_lens_np[:num_reqs] = (
                self.input_batch.num_computed_tokens_cpu[:num_reqs]
                + num_scheduled_tokens
            )

            # TODO: improve performance, no `positions_np.copy()`
            sparsed_positions = positions_np.copy()
            req_sparsed_slots = scheduler_output.req_sparsed_slots
            for req_id in self.input_batch.req_id_to_index:
                is_sparsed_request = req_sparsed_slots[req_id] != INVALID_SLOT
                req_index = self.input_batch.req_id_to_index[req_id]
                offset = (
                    0 if req_index == 0 else cu_num_tokens[req_index - 1]
                )  # TODO: support MTP
                if is_sparsed_request:
                    sparsed_positions[offset] = req_sparsed_slots[req_id] - 1

            # Get token indices.
            # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
            # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
            # where M is the max_model_len.
            token_indices = (
                positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]
            )

            # NOTE(woosuk): We use torch.index_select instead of np.take here
            # because torch.index_select is much faster than np.take for large
            # tensors.
            torch.index_select(
                self.input_batch.token_ids_cpu_tensor.flatten(),
                0,
                torch.from_numpy(token_indices),
                out=self.input_ids_cpu[:total_num_scheduled_tokens],
            )

            # Calculate the slot mapping for each KV cache group.
            for kv_cache_group_id, kv_cache_group_spec in enumerate(
                self.kv_cache_config.kv_cache_groups
            ):
                block_size = kv_cache_group_spec.kv_cache_spec.block_size
                block_table: BlockTable = self.input_batch.block_table[
                    kv_cache_group_id
                ]
                # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
                # -> [0, 0, K, K, K + 1, K + 1, K + 2, 2 * K, 2 * K, 2 * K + 1]
                # where K is the max_num_blocks_per_req and the block size is 2.
                # NOTE(woosuk): We can't simply use `token_indices // block_size`
                # here because M (max_model_len) is not necessarily divisible by
                # block_size.
                block_table_indices = (
                    req_indices * block_table.max_num_blocks_per_req
                    + sparsed_positions // block_size
                )
                block_table_cpu = block_table.get_cpu_tensor()
                block_numbers = block_table_cpu.flatten()[block_table_indices].numpy()
                block_offsets = sparsed_positions % block_size
                np.add(
                    block_numbers * block_size,
                    block_offsets,
                    out=block_table.slot_mapping_np[:total_num_scheduled_tokens],
                )

            # Prepare the attention metadata.
            self.query_start_loc_np[0] = 0
            self.query_start_loc_np[1 : num_reqs + 1] = cu_num_tokens

            for req_id in self.input_batch.req_id_to_index:
                req_index = self.input_batch.req_id_to_index[req_id]
                is_sparsed_request = (
                    scheduler_output.req_sparsed_slots[req_id] != INVALID_SLOT
                )
                if is_sparsed_request:
                    self.seq_lens_np[req_index] = scheduler_output.req_sparsed_slots[
                        req_id
                    ]

            # Copy the tensors to the GPU.
            self.input_ids[:total_num_scheduled_tokens].copy_(
                self.input_ids_cpu[:total_num_scheduled_tokens], non_blocking=True
            )
            if self.uses_mrope:
                # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
                self.mrope_positions[:, :total_num_scheduled_tokens].copy_(
                    self.mrope_positions_cpu[:, :total_num_scheduled_tokens],
                    non_blocking=True,
                )
            else:
                # Common case (1D positions)
                self.positions_cpu[:total_num_scheduled_tokens] = torch.from_numpy(
                    positions_np[:total_num_scheduled_tokens]
                )
                self.positions[:total_num_scheduled_tokens].copy_(
                    self.positions_cpu[:total_num_scheduled_tokens], non_blocking=True
                )

            self.query_start_loc[: num_reqs + 1].copy_(
                self.query_start_loc_cpu[: num_reqs + 1], non_blocking=True
            )
            self.seq_lens[:num_reqs].copy_(
                self.seq_lens_cpu[:num_reqs], non_blocking=True
            )

            # Fill unused with -1. Needed for reshape_and_cache
            self.seq_lens[num_reqs:].fill_(0)
            # Note: pad query_start_loc to be non-decreasing, as kernels
            # like FlashAttention requires that
            self.query_start_loc[num_reqs + 1 :].fill_(
                self.query_start_loc_cpu[num_reqs].item()
            )

            query_start_loc = self.query_start_loc[: num_reqs + 1]
            seq_lens = self.seq_lens[:num_reqs]

            common_attn_metadata = CommonAttentionMetadata(
                query_start_loc=query_start_loc,
                seq_lens=seq_lens,
                num_reqs=num_reqs,
                num_actual_tokens=total_num_scheduled_tokens,
                max_query_len=max_num_scheduled_tokens,
            )

            attn_metadata: dict[str, Any] = {}
            # Prepare the attention metadata for each KV cache group and make layers
            # in the same group share the same metadata.
            for kv_cache_group_id, kv_cache_group_spec in enumerate(
                self.kv_cache_config.kv_cache_groups
            ):

                # Prepare for cascade attention if enabled & beneficial.
                common_prefix_len = 0
                builder = self.attn_metadata_builders[kv_cache_group_id]
                if self.cascade_attn_enabled:
                    common_prefix_len = self._compute_cascade_attn_prefix_len(
                        num_scheduled_tokens,
                        scheduler_output.num_common_prefix_blocks[kv_cache_group_id],
                        kv_cache_group_spec.kv_cache_spec,
                        builder,
                    )

                attn_metadata_i = builder.build(
                    common_prefix_len=common_prefix_len,
                    common_attn_metadata=common_attn_metadata,
                )

                for layer_name in kv_cache_group_spec.layer_names:
                    attn_metadata[layer_name] = attn_metadata_i

            attention_cuda_graphs = all(
                b.can_run_in_cudagraph(common_attn_metadata)
                for b in self.attn_metadata_builders
            )

            use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
            if not use_spec_decode:
                # NOTE(woosuk): Due to chunked prefills, the batch may contain
                # partial requests. While we should not sample any token
                # from these partial requests, we do so for simplicity.
                # We will ignore the sampled tokens from the partial requests.
                # TODO: Support prompt logprobs.
                logits_indices = query_start_loc[1:] - 1
                spec_decode_metadata = None
            else:
                # Get the number of draft tokens for each request.
                # Iterate over the dictionary rather than all requests since not all
                # requests have draft tokens.
                num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
                for (
                    req_id,
                    draft_token_ids,
                ) in scheduler_output.scheduled_spec_decode_tokens.items():
                    req_idx = self.input_batch.req_id_to_index[req_id]
                    num_draft_tokens[req_idx] = len(draft_token_ids)

                spec_decode_metadata = self._calc_spec_decode_metadata(
                    num_draft_tokens, cu_num_tokens
                )
                logits_indices = spec_decode_metadata.logits_indices

            # Hot-Swap lora model
            if self.lora_config:
                self.set_active_loras(self.input_batch, num_scheduled_tokens)

            return (
                attn_metadata,
                attention_cuda_graphs,
                logits_indices,
                spec_decode_metadata,
                num_scheduled_tokens,
            )

        GPUModelRunner._prepare_inputs = _prepare_inputs

        @torch.inference_mode()
        def execute_model(
            self,
            scheduler_output: "SchedulerOutput",
            intermediate_tensors: Optional[IntermediateTensors] = None,
        ) -> Union[ModelRunnerOutput, IntermediateTensors]:
            self._update_states(scheduler_output)
            if not scheduler_output.total_num_scheduled_tokens:
                if not has_kv_transfer_group():
                    # Return empty ModelRunnerOutput if there's no work to do.
                    return EMPTY_MODEL_RUNNER_OUTPUT

                return self.kv_connector_no_forward(scheduler_output)

            # Prepare the decoder inputs.
            (
                attn_metadata,
                attention_cuda_graphs,
                logits_indices,
                spec_decode_metadata,
                num_scheduled_tokens_np,
            ) = self._prepare_inputs(scheduler_output)
            num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
            if (
                self.use_cuda_graph
                and num_scheduled_tokens <= self.cudagraph_batch_sizes[-1]
            ):
                # Use piecewise CUDA graphs.
                # Add padding to the batch size.
                num_input_tokens = self.vllm_config.pad_for_cudagraph(
                    num_scheduled_tokens
                )
            else:
                # Eager mode.
                # Pad tokens to multiple of tensor_parallel_size when
                # enabled collective fusion for SP
                tp_size = self.vllm_config.parallel_config.tensor_parallel_size
                if (
                    self.compilation_config.pass_config.enable_sequence_parallelism
                    and tp_size > 1
                ):
                    num_input_tokens = round_up(num_scheduled_tokens, tp_size)
                else:
                    num_input_tokens = num_scheduled_tokens

            # Padding for DP
            num_pad, num_tokens_across_dp = self.get_dp_padding(num_input_tokens)
            num_input_tokens += num_pad

            # _prepare_inputs may reorder the batch, so we must gather multi
            # modal outputs after that to ensure the correct order
            if self.is_multimodal_model:
                # Run the multimodal encoder if any.
                self._execute_mm_encoder(scheduler_output)
                mm_embeds = self._gather_mm_embeddings(scheduler_output)
            else:
                mm_embeds = []

            if self.is_multimodal_model and get_pp_group().is_first_rank:
                # NOTE(woosuk): To unify token ids and soft tokens (vision
                # embeddings), we always use embeddings (rather than token ids)
                # as input to the multimodal model, even when the input is text.
                input_ids = self.input_ids[:num_scheduled_tokens]
                if mm_embeds:
                    inputs_embeds = self.model.get_input_embeddings(
                        input_ids, mm_embeds
                    )
                else:
                    inputs_embeds = self.model.get_input_embeddings(input_ids)
                # TODO(woosuk): Avoid the copy. Optimize.
                self.inputs_embeds[:num_scheduled_tokens].copy_(inputs_embeds)
                inputs_embeds = self.inputs_embeds[:num_input_tokens]
                input_ids = None
            else:
                # For text-only models, we use token ids as input.
                # While it is possible to use embeddings as input just like the
                # multimodal models, it is not desirable for performance since
                # then the embedding layer is not included in the CUDA graph.
                input_ids = self.input_ids[:num_input_tokens]
                inputs_embeds = None
            if self.uses_mrope:
                positions = self.mrope_positions[:, :num_input_tokens]
            else:
                positions = self.positions[:num_input_tokens]

            if get_pp_group().is_first_rank:
                intermediate_tensors = None
            else:
                intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                    num_input_tokens, intermediate_tensors, True
                )

            # Some attention backends only support CUDA Graphs in pure decode.
            # If attention doesn't support CUDA Graphs for this batch, but we
            # compiled with full CUDA graphs, we have to skip them entirely.
            skip_cuda_graphs = self.full_cuda_graph and not attention_cuda_graphs

            # Run the model.
            # Use persistent buffers for CUDA graphs.
            with set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                skip_cuda_graphs=skip_cuda_graphs,
            ):
                self.maybe_setup_kv_connector(scheduler_output)
                self.maybe_execute_ucm_sparse_begin(scheduler_output, attn_metadata)

                model_output = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                )

                self.maybe_wait_for_kv_save()
                logits_indices = self.maybe_execute_ucm_sparse_finished(logits_indices)

                finished_sending, finished_recving = self.get_finished_kv_transfers(
                    scheduler_output
                )

            if self.use_aux_hidden_state_outputs:
                hidden_states, aux_hidden_states = model_output
            else:
                hidden_states = model_output
                aux_hidden_states = None

            # Broadcast PP output for external_launcher (torchrun)
            # to make sure we are synced across pp ranks
            # TODO: Support overlapping mirco-batches
            # https://github.com/vllm-project/vllm/issues/18019
            broadcast_pp_output = (
                self.parallel_config.distributed_executor_backend == "external_launcher"
                and len(get_pp_group().ranks) > 0
            )
            if not get_pp_group().is_last_rank:
                # For mid-pipeline stages, return the hidden states.
                if not broadcast_pp_output:
                    return hidden_states
                assert isinstance(hidden_states, IntermediateTensors)
                get_pp_group().send_tensor_dict(
                    hidden_states.tensors, all_gather_group=get_tp_group()
                )
                logits = None
            else:
                if self.input_batch.pooling_params:
                    return self._pool(
                        hidden_states,
                        num_scheduled_tokens,
                        num_scheduled_tokens_np,
                        finished_sending,
                        finished_recving,
                    )

                sample_hidden_states = hidden_states[logits_indices]
                logits = self.model.compute_logits(sample_hidden_states, None)
            if broadcast_pp_output:
                model_output_broadcast_data = (
                    {
                        "logits": logits.contiguous(),
                    }
                    if logits is not None
                    else {}
                )
                model_output_broadcast_data = get_pp_group().broadcast_tensor_dict(
                    model_output_broadcast_data, src=len(get_pp_group().ranks) - 1
                )
                assert model_output_broadcast_data is not None
                logits = model_output_broadcast_data["logits"]

            # Apply structured output bitmasks if present
            if scheduler_output.grammar_bitmask is not None:
                self.apply_grammar_bitmask(scheduler_output, logits)

            # Sample the next token and get logprobs if needed.
            sampling_metadata = self.input_batch.sampling_metadata
            if spec_decode_metadata is None:
                sampler_output = self.sampler(
                    logits=logits,
                    sampling_metadata=sampling_metadata,
                )
            else:
                # When indexing with a tensor (bonus_logits_indices), PyTorch
                # creates a new tensor with separate storage from the original
                # logits tensor. This means any in-place operations on bonus_logits
                # won't affect the original logits tensor.
                assert logits is not None
                bonus_logits = logits[spec_decode_metadata.bonus_logits_indices]
                sampler_output = self.sampler(
                    logits=bonus_logits,
                    sampling_metadata=sampling_metadata,
                )
                bonus_token_ids = sampler_output.sampled_token_ids

                # Just like `bonus_logits`, `target_logits` is a new tensor with
                # separate storage from the original `logits` tensor. Therefore,
                # it is safe to update `target_logits` in place.
                target_logits = logits[spec_decode_metadata.target_logits_indices]
                output_token_ids = self.rejection_sampler(
                    spec_decode_metadata,
                    None,  # draft_probs
                    target_logits,
                    bonus_token_ids,
                    sampling_metadata,
                )
                sampler_output.sampled_token_ids = output_token_ids

            num_nans_in_logits = {}
            if envs.VLLM_COMPUTE_NANS_IN_LOGITS:
                num_nans_in_logits = self._get_nans_in_logits(logits)

            # TODO(woosuk): The following loop can be slow since it iterates over
            # the requests one by one. Optimize.
            discard_sampled_tokens_req_indices = []
            for i, req_id in enumerate(self.input_batch.req_ids):
                req_state = self.requests[req_id]
                seq_len = (
                    req_state.num_computed_tokens
                    + scheduler_output.num_scheduled_tokens[req_id]
                )
                if seq_len < req_state.num_tokens:
                    # Ignore the sampled token for partial prefills.
                    # Rewind the generator state as if the token was not sampled.
                    # This relies on cuda-specific torch-internal impl details
                    generator = self.input_batch.generators.get(i)
                    if generator is not None:
                        generator.set_offset(generator.get_offset() - 4)
                    # Record the index of the request that should not be sampled,
                    # so that we could clear the sampled tokens before returning.
                    discard_sampled_tokens_req_indices.append(i)

            # NOTE: GPU -> CPU Sync happens here.
            # Move as many CPU operations as possible before this sync point.
            logprobs_tensors = sampler_output.logprobs_tensors
            logprobs_lists = (
                logprobs_tensors.tolists() if logprobs_tensors is not None else None
            )

            # Compute prompt logprobs if needed.
            prompt_logprobs_dict = self._get_prompt_logprobs_dict(
                hidden_states[:num_scheduled_tokens],
                scheduler_output,
            )

            # Get the valid generated tokens.
            sampled_token_ids = sampler_output.sampled_token_ids
            max_gen_len = sampled_token_ids.shape[-1]
            if max_gen_len == 1:
                # No spec decode tokens.
                valid_sampled_token_ids = sampled_token_ids.tolist()
            else:
                # Includes spec decode tokens.
                valid_sampled_token_ids = self.rejection_sampler.parse_output(
                    sampled_token_ids,
                    self.input_batch.vocab_size,
                )
            # Mask out the sampled tokens that should not be sampled.
            for i in discard_sampled_tokens_req_indices:
                valid_sampled_token_ids[i].clear()

            # Cache the sampled tokens in the model runner, so that the scheduler
            # doesn't need to send them back.
            # NOTE(woosuk): As an exception, when using PP, the scheduler sends
            # the sampled tokens back, because there's no direct communication
            # between the first-stage worker and the last-stage worker.
            for req_idx, sampled_ids in enumerate(valid_sampled_token_ids):
                if not sampled_ids:
                    continue

                start_idx = self.input_batch.num_tokens_no_spec[req_idx]
                end_idx = start_idx + len(sampled_ids)
                assert end_idx <= self.max_model_len, (
                    "Sampled token IDs exceed the max model length. "
                    f"Total number of tokens: {end_idx} > max_model_len: "
                    f"{self.max_model_len}"
                )

                self.input_batch.token_ids_cpu[req_idx, start_idx:end_idx] = sampled_ids
                self.input_batch.num_tokens_no_spec[req_idx] = end_idx
                self.input_batch.num_tokens[req_idx] = end_idx
                req_id = self.input_batch.req_ids[req_idx]
                req_state = self.requests[req_id]
                req_state.output_token_ids.extend(sampled_ids)

            if not self.speculative_config:
                # Speculative decoding is not enabled.
                spec_token_ids = None
            else:
                spec_token_ids = self.propose_draft_token_ids(
                    scheduler_output,
                    valid_sampled_token_ids,
                    sampling_metadata,
                    hidden_states,
                    sample_hidden_states,
                    aux_hidden_states,
                    spec_decode_metadata,
                    attn_metadata,
                )

            # Clear KVConnector state after all KVs are generated.
            if has_kv_transfer_group():
                get_kv_transfer_group().clear_connector_metadata()

            self.eplb_step()

            return ModelRunnerOutput(
                req_ids=self.input_batch.req_ids,
                req_id_to_index=self.input_batch.req_id_to_index,
                sampled_token_ids=valid_sampled_token_ids,
                spec_token_ids=spec_token_ids,
                logprobs=logprobs_lists,
                prompt_logprobs_dict=prompt_logprobs_dict,
                pooler_output=[],
                finished_sending=finished_sending,
                finished_recving=finished_recving,
                num_nans_in_logits=num_nans_in_logits,
            )

        GPUModelRunner.execute_model = execute_model

    except ImportError:
        logger.warning("Could not patch prepare inputs - module not found")


# ==================== vllm/v1/worker/gpu_worker.py  ====================
def _patch_gpu_worker() -> None:
    """Patch gpu worker to add UCM sparse support."""
    try:
        from typing import Optional

        from vllm.config import VllmConfig
        from vllm.v1.worker import gpu_worker

        from ucm.sparse.state import ensure_ucm_sparse_initialized

        original_init_worker_distributed_environment = (
            gpu_worker.init_worker_distributed_environment
        )

        def patched_init_worker_distributed_environment(
            vllm_config: VllmConfig,
            rank: int,
            distributed_init_method: Optional[str] = None,
            local_rank: int = -1,
            backend: str = "nccl",
        ) -> None:
            original_init_worker_distributed_environment(
                vllm_config, rank, distributed_init_method, local_rank, backend
            )
            ensure_ucm_sparse_initialized(vllm_config)

        gpu_worker.init_worker_distributed_environment = (
            patched_init_worker_distributed_environment
        )
    except ImportError:
        logger.warning("Could not patch gpu worker - module not found")


# ==================== vllm/model_executor/models/llama.py  ====================
def _patch_llama_model() -> None:
    """Patch gpu worker to add UCM sparse support."""
    try:
        from typing import Optional, Union

        import torch
        from vllm.config import VllmConfig
        from vllm.distributed import get_pp_group
        from vllm.model_executor.models.llama import LlamaDecoderLayer, LlamaModel
        from vllm.sequence import IntermediateTensors

        from ucm.sparse.state import (
            get_ucm_sparse,
            has_ucm_sparse,
            maybe_execute_sparse_ffn_begin,
            maybe_execute_sparse_ffn_finished,
            maybe_execute_sparse_layer_begin,
            maybe_execute_sparse_layer_finished,
        )

        def llamaDecoderLayer_forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            residual: Optional[torch.Tensor],
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # Self Attention
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)
            hidden_states = self.self_attn(
                positions=positions, hidden_states=hidden_states
            )
            ######################
            ### UCM PATCH START###
            hidden_states, residual = maybe_execute_sparse_ffn_begin(
                hidden_states, residual
            )
            ### UCM PATCH END  ###
            ######################
            # Fully Connected
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )
            hidden_states = self.mlp(hidden_states)
            ######################
            ### UCM PATCH START###
            hidden_states, residual = maybe_execute_sparse_ffn_finished(
                hidden_states, residual
            )
            ### UCM PATCH END  ###
            ######################
            return hidden_states, residual

        LlamaDecoderLayer.forward = llamaDecoderLayer_forward

        def llamaModel_forward(
            self,
            input_ids: Optional[torch.Tensor],
            positions: torch.Tensor,
            intermediate_tensors: Optional[IntermediateTensors],
            inputs_embeds: Optional[torch.Tensor] = None,
        ) -> Union[
            torch.Tensor, IntermediateTensors, tuple[torch.Tensor, list[torch.Tensor]]
        ]:
            if get_pp_group().is_first_rank:
                if inputs_embeds is not None:
                    hidden_states = inputs_embeds
                else:
                    hidden_states = self.get_input_embeddings(input_ids)
                residual = None
            else:
                assert intermediate_tensors is not None
                hidden_states = intermediate_tensors["hidden_states"]
                residual = intermediate_tensors["residual"]

            aux_hidden_states = []
            for idx, layer in enumerate(self.layers[self.start_layer : self.end_layer]):
                ######################
                ### UCM PATCH START###
                positions, hidden_states, residual = maybe_execute_sparse_layer_begin(
                    positions, hidden_states, residual
                )
                ### UCM PATCH END  ###
                ######################
                if idx in self.aux_hidden_state_layers:
                    aux_hidden_states.append(hidden_states + residual)
                hidden_states, residual = layer(positions, hidden_states, residual)
                ######################
                ### UCM PATCH START###
                positions, hidden_states, residual = (
                    maybe_execute_sparse_layer_finished(
                        positions, hidden_states, residual
                    )
                )
                ### UCM PATCH END  ###
                ######################

            if not get_pp_group().is_last_rank:
                return IntermediateTensors(
                    {"hidden_states": hidden_states, "residual": residual}
                )

            hidden_states, _ = self.norm(hidden_states, residual)

            if len(aux_hidden_states) > 0:
                return hidden_states, aux_hidden_states
            return hidden_states

        LlamaModel.forward = llamaModel_forward

    except ImportError:
        logger.warning("Could not patch llama modelr - module not found")


# ==================== vllm/model_executor/models/qwen2.py  ====================
def _patch_qwen_model() -> None:
    """Patch gpu worker to add UCM sparse support."""
    try:
        from typing import Optional, Union

        import torch
        from vllm.config import VllmConfig
        from vllm.distributed import get_pp_group
        from vllm.model_executor.models.qwen2 import Qwen2DecoderLayer, Qwen2Model
        from vllm.sequence import IntermediateTensors

        from ucm.sparse.state import (
            get_ucm_sparse,
            has_ucm_sparse,
            maybe_execute_sparse_ffn_begin,
            maybe_execute_sparse_ffn_finished,
            maybe_execute_sparse_layer_begin,
            maybe_execute_sparse_layer_finished,
        )

        def qwen2DecoderLayer_forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            residual: Optional[torch.Tensor],
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # Self Attention
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
            )
            ######################
            ### UCM PATCH START###
            residual, hidden_states = maybe_execute_sparse_ffn_begin(
                residual, hidden_states
            )
            ### UCM PATCH END  ###
            ######################
            # Fully Connected
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )
            hidden_states = self.mlp(hidden_states)
            ######################
            ### UCM PATCH START###
            residual, hidden_states = maybe_execute_sparse_ffn_finished(
                residual, hidden_states
            )
            ### UCM PATCH END  ###
            ######################
            return hidden_states, residual

        Qwen2DecoderLayer.forward = qwen2DecoderLayer_forward

        def qwen2Model_forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            intermediate_tensors: Optional[IntermediateTensors] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
        ) -> Union[torch.Tensor, IntermediateTensors]:
            if get_pp_group().is_first_rank:
                if inputs_embeds is not None:
                    hidden_states = inputs_embeds
                else:
                    hidden_states = self.get_input_embeddings(input_ids)
                residual = None
            else:
                assert intermediate_tensors is not None
                hidden_states = intermediate_tensors["hidden_states"]
                residual = intermediate_tensors["residual"]
            for layer in self.layers[self.start_layer : self.end_layer]:
                ######################
                ### UCM PATCH START###
                positions, hidden_states, residual = maybe_execute_sparse_layer_begin(
                    positions,
                    hidden_states,
                    residual,
                )
                ### UCM PATCH END  ###
                ######################
                hidden_states, residual = layer(
                    positions,
                    hidden_states,
                    residual,
                )
                ######################
                ### UCM PATCH START###
                positions, hidden_states, residual = (
                    maybe_execute_sparse_layer_finished(
                        positions, hidden_states, residual
                    )
                )
                ### UCM PATCH END  ###
                ######################
            if not get_pp_group().is_last_rank:
                return IntermediateTensors(
                    {"hidden_states": hidden_states, "residual": residual}
                )
            hidden_states, _ = self.norm(hidden_states, residual)
            return hidden_states

        Qwen2Model.forward = qwen2Model_forward

    except ImportError:
        logger.warning("Could not patch llama modelr - module not found")
