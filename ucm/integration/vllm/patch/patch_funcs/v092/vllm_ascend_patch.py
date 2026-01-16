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


def _apply_ascend_patch() -> None:
    """Apply patch for vLLM-Ascend."""
    try:
        from vllm_ascend.patch import platform, worker

        if _enable_sparse():
            _patch_attention_v1()
            _patch_mla_v1()
            _patch_model_runner_v1()
            _patch_worker_v1()
            logger.info("UCM sparse adapt patches applied successfully")

    except Exception as e:
        logger.error(f"Could not apply sparse adapt patches: {e}")
        raise e


# ========================= vllm_ascend/attention/attention_v1.py =========================
def _patch_attention_v1() -> None:
    """Patch attention_v1.py for vLLM-Ascend."""
    try:
        from typing import List, Optional

        import torch
        from vllm.forward_context import ForwardContext, get_forward_context
        from vllm_ascend.attention import attention_v1

        from ucm.sparse.state import get_ucm_sparse, has_ucm_sparse

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

        attention_v1.maybe_execute_sparse_attention_begin = (
            maybe_execute_sparse_attention_begin
        )

        def maybe_execute_sparse_attention_finished(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attn_output: torch.Tensor,
            layer_name: str,
            forward_context: ForwardContext,
        ):
            if not has_ucm_sparse():
                return
            ucm_sparse = get_ucm_sparse()
            attn_metadata = forward_context.attn_metadata
            if attn_metadata is None:
                return
            ucm_sparse.attention_finished(
                query, key, value, attn_output, layer_name, forward_context
            )

        attention_v1.maybe_execute_sparse_attention_finished = (
            maybe_execute_sparse_attention_finished
        )

        vllm_ops = torch.ops.vllm
        orig_unified_ascend_attention_with_output = (
            vllm_ops.unified_ascend_attention_with_output
        )

        def _wrap_op_overload(orig, impl):
            class _Wrapper:
                def __init__(self, orig):
                    self._orig = orig

                def __call__(self, *args, **kwargs):
                    return impl(*args, **kwargs)

                def __getattr__(self, name):
                    return getattr(self._orig, name)

            return _Wrapper(orig)

        def unified_ascend_attention_with_output_impl(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            output: torch.Tensor,
            layer_name: str,
        ) -> None:

            forward_context: ForwardContext = get_forward_context()
            attn_metadata = forward_context.attn_metadata
            self = forward_context.no_compile_layers[layer_name]
            kv_cache = self.kv_cache[forward_context.virtual_engine]
            if not self.use_mla:
                query, key, value, _ = maybe_execute_sparse_attention_begin(
                    query, key, value, layer_name, forward_context
                )
            self.impl.forward(
                self,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                trace_flag=False,
            )
            if not self.use_mla:
                maybe_execute_sparse_attention_finished(
                    query, key, value, output, layer_name, forward_context
                )
            return

        vllm_ops.unified_ascend_attention_with_output = _wrap_op_overload(
            orig_unified_ascend_attention_with_output,
            unified_ascend_attention_with_output_impl,
        )

        attention_v1.unified_ascend_attention_with_output = (
            unified_ascend_attention_with_output_impl
        )
    except ImportError as e:
        logger.error(f"Failed to patch attention_v1.py: {e}")
        raise


# ========================= vllm_ascend/attention/mla_v1.py =========================
def _patch_mla_v1() -> None:
    """Patch mla_v1.py for vLLM-Ascend."""
    try:
        from typing import Optional

        import torch
        import torch_npu
        from vllm.attention.backends.abstract import AttentionLayer
        from vllm.attention.layer import (
            maybe_execute_sparse_attention_begin,
            maybe_execute_sparse_attention_finished,
        )
        from vllm.forward_context import ForwardContext, get_forward_context
        from vllm_ascend.attention.attention_v1 import (
            AscendAttentionState,
        )
        from vllm_ascend.attention.mla_v1 import AscendMLAImpl
        from vllm_ascend.multistream.context import get_multistream_comm_context
        from vllm_ascend.utils import npu_stream_switch, npu_wait_tensor

        def forward(
            self,
            layer: AttentionLayer,
            hidden_states_or_q_c: torch.Tensor,  # query in unified attn
            hidden_states_or_kv_c_normed: torch.Tensor,  # key in unified attn
            k_pe: torch.Tensor,  # value in unified attn
            kv_cache: torch.Tensor,
            attn_metadata: M,
            output: Optional[torch.Tensor] = None,
            enable_multistream_mla: bool = False,
            ckq: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            forward_context: ForwardContext = get_forward_context()
            assert output is not None, "Output tensor must be provided."
            if attn_metadata is None:
                # Profiling run.
                return output
            self.running_in_graph = (
                self.torchair_graph_enabled
                and attn_metadata.attn_state
                in [AscendAttentionState.DecodeOnly, AscendAttentionState.SpecDecoding]
            )
            num_actual_toks = attn_metadata.num_actual_tokens
            if k_pe is None and not self.running_in_graph:
                if not self.torchair_graph_enabled:
                    kv_c, k_pe = self.kv_a_proj_with_mqa(hidden_states_or_kv_c_normed)[
                        0
                    ].split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
                    kv_c_normed = self.kv_a_layernorm(kv_c.contiguous())
            else:
                kv_c_normed = hidden_states_or_kv_c_normed
            assert (
                attn_metadata.num_decodes is not None
                and attn_metadata.num_prefills is not None
                and attn_metadata.num_decode_tokens is not None
            )
            has_decode = attn_metadata.num_decodes > 0
            has_prefill = attn_metadata.num_prefills > 0
            num_decode_tokens = attn_metadata.num_decode_tokens
            if not self.running_in_graph:
                # Inputs and outputs may be padded for CUDA graphs
                output_padded = output
                output = output[:num_actual_toks, ...]
                if not self.torchair_graph_enabled:
                    kv_c_normed = kv_c_normed[:num_actual_toks, ...]
                    prefill_k_c_normed = kv_c_normed[num_decode_tokens:]
            if not self.running_in_graph:
                hidden_states_or_q_c = hidden_states_or_q_c[:num_actual_toks, ...]
                prefill_hs_or_q_c = hidden_states_or_q_c[num_decode_tokens:]
                if not self.torchair_graph_enabled:
                    decode_hs_or_q_c = hidden_states_or_q_c[:num_decode_tokens]
                    k_pe = k_pe[:num_actual_toks, ...]
                    k_pe = k_pe.unsqueeze(1)
                    decode_k_pe = k_pe[:num_decode_tokens]
                    prefill_k_pe = k_pe[num_decode_tokens:]
            else:
                decode_hs_or_q_c = hidden_states_or_q_c
            if has_decode:
                decode_k_nope = None
                assert attn_metadata.decode is not None
                if self.running_in_graph:
                    seq_len = (
                        self.rotary_emb.max_position_embeddings
                        * self.rotary_emb.scaling_factor
                    )
                    cos = self.rotary_emb.cos_cached[:seq_len].to(
                        dtype=decode_hs_or_q_c.dtype
                    )
                    sin = self.rotary_emb.sin_cached[:seq_len].to(
                        dtype=decode_hs_or_q_c.dtype
                    )
                    cos = cos[attn_metadata.decode.input_positions]
                    sin = sin[attn_metadata.decode.input_positions]
                    cos = cos[:, None, None, :]
                    sin = sin[:, None, None, :]
                    with npu_stream_switch(
                        "mla_secondary", 0, enabled=enable_multistream_mla
                    ):
                        npu_wait_tensor(
                            hidden_states_or_kv_c_normed,
                            ckq,
                            enabled=enable_multistream_mla,
                        )
                        decode_k_pe, decode_k_nope, decode_kv = self.exec_kv(
                            hidden_states_or_kv_c_normed,
                            cos,
                            sin,
                            kv_cache,
                            attn_metadata.slot_mapping,
                        )
                    # Without explicitly controlling the order, IndexByTensor operations
                    # would be placed after `matmul W_KV_T` hindering the overlapping of
                    # KvRmsNormRopeCache and SingleRope.
                    npu_wait_tensor(
                        decode_hs_or_q_c, cos, enabled=enable_multistream_mla
                    )
                    npu_wait_tensor(
                        decode_hs_or_q_c, sin, enabled=enable_multistream_mla
                    )
                    npu_wait_tensor(
                        decode_hs_or_q_c, decode_kv, enabled=enable_multistream_mla
                    )

                decode_ql_nope, decode_q_pe = self._q_proj_and_k_up_proj(
                    decode_hs_or_q_c
                )
                if self.running_in_graph:
                    with npu_stream_switch(
                        "mla_secondary", 0, enabled=enable_multistream_mla
                    ):
                        npu_wait_tensor(
                            decode_q_pe, decode_k_pe, enabled=enable_multistream_mla
                        )
                        decode_q_pe = self.rope_single(decode_q_pe, cos, sin)
                else:
                    decode_q_pe[...], decode_k_pe[...] = self.rotary_emb(
                        attn_metadata.decode.input_positions,
                        decode_q_pe.contiguous(),
                        decode_k_pe,
                        max_seq_len=attn_metadata.decode.max_seq_lens,
                    )
            if has_prefill:
                assert attn_metadata.prefill is not None
                prefill_q = self.q_proj(prefill_hs_or_q_c)[0].view(
                    -1, self.num_heads, self.qk_head_dim
                )
                prefill_q_pe = prefill_q[..., self.qk_nope_head_dim :]
                prefill_q_nope = prefill_q[..., : self.qk_nope_head_dim]
                if self.torchair_graph_enabled:
                    num_tokens = prefill_hs_or_q_c.shape[0]
                    seq_len = (
                        self.rotary_emb.max_position_embeddings
                        * self.rotary_emb.scaling_factor
                    )
                    cos = self.rotary_emb.cos_cached[:seq_len].to(
                        dtype=prefill_q_pe.dtype
                    )
                    sin = self.rotary_emb.sin_cached[:seq_len].to(
                        dtype=prefill_q_pe.dtype
                    )
                    cos = cos[attn_metadata.prefill.input_positions]
                    sin = sin[attn_metadata.prefill.input_positions]
                    cos = cos[:, None, None, :]
                    sin = sin[:, None, None, :]

                    prefill_q_pe = self.rope_single(prefill_q_pe, cos, sin)
                    prefill_k_pe, prefill_k_nope = self.exec_kv_prefill(
                        hidden_states_or_kv_c_normed,
                        cos,
                        sin,
                        kv_cache,
                        attn_metadata.slot_mapping,
                    )

                    kv_c_normed = prefill_k_nope[:num_actual_toks, ...]
                    prefill_k_c_normed = prefill_k_nope[num_decode_tokens:]
                    prefill_k_pe = prefill_k_pe.view(num_tokens, self.num_kv_heads, -1)
                    prefill_q = torch.cat([prefill_q_nope, prefill_q_pe], dim=-1)
                else:
                    prefill_q_pe[...], prefill_k_pe[...] = self.rotary_emb(
                        attn_metadata.prefill.input_positions,
                        prefill_q_pe.contiguous(),
                        prefill_k_pe,
                        max_seq_len=attn_metadata.prefill.max_seq_lens,
                    )
            if self.torchair_graph_enabled:
                if (
                    len(kv_cache) > 0
                    and kv_cache[0].numel() > 0
                    and attn_metadata.attn_state == AscendAttentionState.PrefillNoCache
                ):
                    slots = attn_metadata.slot_mapping
                    # NOTE: Separate the kv cache in advance to avoid OOM or other issues
                    torch_npu._npu_reshape_and_cache(
                        key=kv_c_normed.view(num_tokens, self.num_kv_heads, -1),
                        value=prefill_k_pe,
                        key_cache=kv_cache[0],
                        value_cache=kv_cache[1],
                        slot_indices=slots,
                    )
            elif kv_cache.numel() > 0:
                key = torch.cat(
                    [kv_c_normed.view([num_actual_toks, self.num_kv_heads, -1]), k_pe],
                    dim=2,
                )
                torch_npu._npu_reshape_and_cache_siso(
                    key=key,
                    key_cache=kv_cache,
                    slot_indices=attn_metadata.slot_mapping.flatten(),
                )
            if has_prefill:
                # FIX: aicore move should be also placed on the comm stream in dbo,
                # otherwise it may affect the accuracy
                # TODO: use an elegant way to overlap
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
                output_prefill = self._forward_prefill(
                    prefill_q, prefill_k_c_normed, prefill_k_pe, kv_cache, attn_metadata
                )
                current_ms_metadata = get_multistream_comm_context()
                if current_ms_metadata is not None:
                    with torch.npu.stream(current_ms_metadata.comm_stream):
                        output[num_decode_tokens:] = output_prefill
                        current_ms_metadata.after_comm_event.record()
                else:
                    output[num_decode_tokens:] = output_prefill
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
                if self.running_in_graph:
                    return self._forward_decode(
                        decode_ql_nope,
                        decode_q_pe,
                        decode_k_nope,
                        decode_k_pe,
                        kv_cache,
                        attn_metadata,
                        enable_multistream_mla,
                    )
                else:
                    output_decode = self._forward_decode(
                        decode_ql_nope,
                        decode_q_pe,
                        decode_k_nope,
                        decode_k_pe,
                        kv_cache,
                        attn_metadata,
                    )
                current_ms_metadata = get_multistream_comm_context()
                if current_ms_metadata is not None:
                    with torch.npu.stream(current_ms_metadata.comm_stream):
                        output[:num_decode_tokens] = output_decode
                        current_ms_metadata.after_comm_event.record()
                else:
                    output[:num_decode_tokens] = output_decode
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

        AscendMLAImpl.forward = forward
    except ImportError as e:
        logger.error(f"Failed to patch mla_v1.py: {e}")
        raise


# ========================= vllm_ascend/worker/model_runner_v1.py =========================
def _patch_model_runner_v1() -> None:
    """Patch model_runner_v1.py for vLLM-Ascend."""
    try:
        from typing import TYPE_CHECKING, List, Optional, Union

        import numpy as np
        import torch
        from vllm.distributed.parallel_state import get_pp_group, get_tp_group
        from vllm.forward_context import set_forward_context
        from vllm.logger import logger
        from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
        from vllm.sampling_params import SamplingType
        from vllm.sequence import IntermediateTensors
        from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
        from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
        from vllm_ascend.ascend_config import get_ascend_config
        from vllm_ascend.attention.attention_v1 import (
            AscendAttentionState,
            AscendMetadata,
        )
        from vllm_ascend.attention.attention_v1_torchair import AscendTorchairMetadata
        from vllm_ascend.attention.mla_v1 import (
            AscendMLAMetadata,
            CommonAttentionMetadata,
        )
        from vllm_ascend.utils import (
            ACL_FORMAT_FRACTAL_ND,
            ACL_FORMAT_FRACTAL_NZ,
            ProfileExecuteDuration,
            maybe_converting_weight_acl_format,
        )
        from vllm_ascend.worker.npu_input_batch import CachedRequestState

        from ucm.sparse.base import INVALID_SLOT

        if TYPE_CHECKING:
            from vllm.v1.core.sched.output import SchedulerOutput
        from vllm.distributed.kv_transfer import (
            get_kv_transfer_group,
            has_kv_transfer_group,
        )
        from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorBase_V1
        from vllm.forward_context import get_forward_context, set_forward_context
        from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

        from ucm.sparse.base import INVALID_SLOT
        from ucm.sparse.state import get_ucm_sparse, has_ucm_sparse

        def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
            """Update the cached states and the persistent batch with the scheduler
            output.

            The SamplingMetadata is updated and copied to the NPU if there is a
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
            removed_req_indices: List[int] = []
            for req_id in scheduler_output.finished_req_ids:
                req_index = self.input_batch.remove_request(req_id)
                if req_index is not None:
                    removed_req_indices.append(req_index)

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
                req_index = self.input_batch.remove_request(req_id)
                assert req_index is not None
                removed_req_indices.append(req_index)

            req_ids_to_add: List[str] = []
            # Add new requests to the cached states.
            for new_req_data in scheduler_output.scheduled_new_reqs:
                req_id = new_req_data.req_id
                sampling_params = new_req_data.sampling_params
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
                    pooling_params=new_req_data.pooling_params,
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
            req_data = scheduler_output.scheduled_cached_reqs
            req_sparsed_slots = scheduler_output.req_sparsed_slots
            is_last_rank = get_pp_group().is_last_rank
            for i, req_id in enumerate(req_data.req_ids):
                req_state = self.requests[req_id]
                num_computed_tokens = req_data.num_computed_tokens[i]
                new_block_ids = req_data.new_block_ids[i]
                resumed_from_preemption = req_data.resumed_from_preemption[i]
                is_sparsed_request = req_sparsed_slots[req_id] != INVALID_SLOT

                req_state.num_computed_tokens = num_computed_tokens
                if not is_last_rank:
                    new_token_ids = req_data.new_token_ids[i]
                    # Add the sampled token(s) from the previous step (if any).
                    # This doesn't include "unverified" tokens like spec decode tokens.
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
                    for block_ids, new_ids in zip(  # type: ignore[call-overload]
                        req_state.block_ids, new_block_ids
                    ):
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
                    # NOTE(woosuk): `num_tokens` here may include spec decode tokens.
                    self.input_batch.num_tokens[req_index] = end_token_index

            # Check if the batch has changed. If not, we can skip copying the
            # sampling metadata from CPU to GPU.
            batch_changed = len(removed_req_indices) > 0 or len(req_ids_to_add) > 0

            # Add the new or resumed requests to the persistent batch.
            # The smaller empty indices are filled first.
            removed_req_indices.sort(reverse=True)
            for req_id in req_ids_to_add:
                req_state = self.requests[req_id]
                if removed_req_indices:
                    # Fill the empty index.
                    req_index = removed_req_indices.pop()
                else:
                    # Append to the end.
                    req_index = None
                self.input_batch.add_request(req_state, req_index)

            # Condense the batched states if there are empty indices.
            if removed_req_indices:
                self.input_batch.condense(removed_req_indices)

            if batch_changed:
                self.input_batch.refresh_sampling_metadata()

        NPUModelRunner._update_states = _update_states

        def _process_reqs(
            self,
            scheduler_output: "SchedulerOutput",
            intermediate_tensors: Optional[IntermediateTensors] = None,
        ) -> tuple[
            Union[AscendMetadata, AscendMLAMetadata, AscendTorchairMetadata],
            torch.Tensor,
            SpecDecodeMetadata,
            torch.Tensor,
            int,
            torch.Tensor,
            torch.Tensor,
            np.ndarray,
            Optional[dict[str, list[str]]],
        ]:
            # Check input valid
            total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
            assert total_num_scheduled_tokens > 0
            num_reqs = self.input_batch.num_reqs
            assert num_reqs > 0
            if (
                self.use_aclgraph
                and total_num_scheduled_tokens <= self.aclgraph_batch_sizes[-1]
            ):
                # Add padding to the batch size.
                num_input_tokens = self.vllm_config.pad_for_cudagraph(
                    total_num_scheduled_tokens
                )
            else:
                # Eager mode.
                num_input_tokens = total_num_scheduled_tokens

            modified_batch = self.attn_metadata_builder.reorder_batch(
                self.input_batch, scheduler_output
            )
            if modified_batch:
                self.input_batch.refresh_sampling_metadata()

            # OPTIMIZATION: Start copying the block table first.
            # This way, we can overlap the copy with the following CPU operations.
            self.input_batch.block_table.commit(num_reqs)

            # Get the number of scheduled tokens for each request.
            # TODO: The Python loop can be slow. Optimize.
            num_scheduled_tokens = np.empty(num_reqs, dtype=np.int32)
            num_valid_tokens = np.empty(num_reqs, dtype=np.int32)
            max_num_scheduled_tokens = 0
            for i, req_id in enumerate(self.input_batch.req_ids):
                num_tokens = scheduler_output.num_scheduled_tokens[req_id]
                num_scheduled_tokens[i] = num_tokens
                num_valid_tokens[i] = num_tokens - len(
                    scheduler_output.scheduled_spec_decode_tokens.get(req_id, [])
                )
                max_num_scheduled_tokens = max(max_num_scheduled_tokens, num_tokens)

            # Hot-Swap lora model
            if self.lora_config:
                self.set_active_loras(self.input_batch, num_scheduled_tokens)

            # Prepare positions
            req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)
            cu_num_tokens = np.cumsum(num_scheduled_tokens)
            cumsums_offsets = np.repeat(
                cu_num_tokens - num_scheduled_tokens, num_scheduled_tokens
            )
            logits_indices = cu_num_tokens - 1
            logits_indices = torch.from_numpy(logits_indices).to(
                self.device, non_blocking=True
            )
            arange = self.arange_np[:total_num_scheduled_tokens] - cumsums_offsets

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

            if self.uses_mrope:
                # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
                self.mrope_positions[:, :total_num_scheduled_tokens].copy_(
                    self.mrope_positions_cpu[:, :total_num_scheduled_tokens],
                    non_blocking=True,
                )

            self.positions[total_num_scheduled_tokens:num_input_tokens].zero_()
            self.positions[:total_num_scheduled_tokens].copy_(
                self.positions_cpu[:total_num_scheduled_tokens], non_blocking=True
            )
            positions = self.positions[:num_input_tokens]
            self.query_lens = torch.from_numpy(num_scheduled_tokens)

            self.seq_lens_np[:num_reqs] = (
                self.input_batch.num_computed_tokens_cpu[:num_reqs]
                + num_scheduled_tokens
            )
            seq_lens = self.seq_lens_cpu[:num_reqs]

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

            block_table_indices = (
                req_indices * self.max_num_blocks_per_req
                + sparsed_positions // self.block_size
            )

            block_table_cpu = self.input_batch.block_table[0].get_cpu_tensor()
            block_numbers = block_table_cpu.flatten()[block_table_indices].numpy()
            block_offsets = sparsed_positions % self.block_size
            np.add(
                block_numbers * self.block_size,
                block_offsets,
                out=self.slot_mapping_np[:total_num_scheduled_tokens],
            )

            ascend_config = get_ascend_config()
            use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
            if np.array_equal(self.seq_lens_np[:num_reqs], num_scheduled_tokens):
                attn_state = AscendAttentionState.PrefillNoCache
            # We assume it is the decode stage, where prefill occurs but only one token is not hit in cache.
            elif np.all(num_scheduled_tokens == 1):
                attn_state = AscendAttentionState.DecodeOnly
            # Speculative decoding.
            elif np.all(num_valid_tokens == 1):
                if self.use_eagle:
                    attn_state = AscendAttentionState.ChunkedPrefill
                else:
                    attn_state = AscendAttentionState.SpecDecoding
            # splitfuse
            elif (
                not ascend_config.ascend_scheduler_config.enabled
                or self.chunked_prefill_enabled
            ):
                attn_state = AscendAttentionState.ChunkedPrefill
            else:
                attn_state = AscendAttentionState.PrefillCacheHit

            for req_id in self.input_batch.req_id_to_index:
                is_sparsed_request = req_sparsed_slots[req_id] != INVALID_SLOT
                req_index = self.input_batch.req_id_to_index[req_id]
                if is_sparsed_request:
                    seq_lens[req_index] = req_sparsed_slots[req_id]

            self.attn_mask = self._make_attention_mask(
                seq_lens=seq_lens,
                query_lens=num_scheduled_tokens,
                position=torch.tensor(sparsed_positions).npu(),
                attn_state=attn_state,
            )
            self.attn_state = attn_state  # type: ignore

            extra_builder_kwargs = {}

            self.query_start_loc_np[0] = 0
            self.query_start_loc_np[1 : num_reqs + 1] = cu_num_tokens
            self.query_start_loc[: num_reqs + 1].copy_(
                self.query_start_loc_cpu[: num_reqs + 1], non_blocking=True
            )
            self.seq_lens[:num_reqs].copy_(
                self.seq_lens_cpu[:num_reqs], non_blocking=True
            )

            # Fill unused with -1. Needed for reshape_and_cache
            self.seq_lens[num_reqs:].fill_(0)
            self.query_start_loc[num_reqs + 1 :].fill_(-1)

            with_prefill = attn_state not in [
                AscendAttentionState.DecodeOnly,
                AscendAttentionState.SpecDecoding,
            ]

            if self.dp_size > 1:
                max_num_tokens, with_prefill = self._get_forward_metadata_across_dp(
                    total_num_scheduled_tokens, with_prefill
                )
                extra_builder_kwargs["max_num_tokens_across_dp"] = max_num_tokens
                extra_builder_kwargs["with_prefill_across_dp"] = with_prefill

            # Add graph_pad_size here
            if self.torchair_graph_enabled and not with_prefill:
                if self.dp_size > 1:
                    padded_batch_size = self.select_torchair_padded_batch_size(
                        max_num_tokens
                    )
                else:
                    padded_batch_size = self.select_torchair_padded_batch_size(
                        total_num_scheduled_tokens
                    )
                graph_pad_size = padded_batch_size - total_num_scheduled_tokens

                extra_builder_kwargs["graph_pad_size"] = graph_pad_size

            if self.vllm_config.model_config.use_mla:
                query_start_loc = self.query_start_loc[: num_reqs + 1]
                seq_lens = self.seq_lens[:num_reqs]
                common_attn_metadata = CommonAttentionMetadata(
                    query_start_loc=query_start_loc, seq_lens=seq_lens
                )
                attn_metadata = self.attn_metadata_builder.build(  # type: ignore
                    num_reqs=num_reqs,
                    num_actual_tokens=total_num_scheduled_tokens,
                    max_query_len=max_num_scheduled_tokens,
                    common_attn_metadata=common_attn_metadata,
                    common_prefix_len=None,
                    **extra_builder_kwargs,
                )
            else:
                attn_metadata = self.attn_metadata_builder.build(  # type: ignore
                    num_reqs=num_reqs,
                    num_actual_tokens=total_num_scheduled_tokens,
                    max_query_len=max_num_scheduled_tokens,
                    common_prefix_len=None,
                    **extra_builder_kwargs,
                )
            attn_metadata.num_input_tokens = num_input_tokens

            # Prepare input_ids
            token_indices = (
                positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]
            )
            torch.index_select(
                self.input_batch.token_ids_cpu_tensor.flatten(),
                0,
                torch.from_numpy(token_indices),
                out=self.input_ids_cpu[:total_num_scheduled_tokens],
            )
            # Copy the tensors to the NPU.
            self.input_ids[:total_num_scheduled_tokens].copy_(
                self.input_ids_cpu[:total_num_scheduled_tokens], non_blocking=True
            )

            # _prepare_inputs may reorder the batch, so we must gather multi
            # modal outputs after that to ensure the correct order
            if self.is_multimodal_model:
                # Run the multimodal encoder if any.
                self._execute_mm_encoder(scheduler_output)
                mm_embeds = self._gather_mm_embeddings(scheduler_output)
            else:
                mm_embeds = []

            if self.is_multimodal_model:
                # NOTE(woosuk): To unify token ids and soft tokens (vision
                # embeddings), we always use embeddings (rather than token ids)
                # as input to the multimodal model, even when the input is text.
                input_ids = self.input_ids[:total_num_scheduled_tokens]
                if mm_embeds:
                    inputs_embeds = self.model.get_input_embeddings(
                        input_ids, mm_embeds
                    )
                else:
                    inputs_embeds = self.model.get_input_embeddings(input_ids)
                # TODO(woosuk): Avoid the copy. Optimize.
                self.inputs_embeds[:total_num_scheduled_tokens].copy_(inputs_embeds)
                inputs_embeds = self.inputs_embeds[:num_input_tokens]
                input_ids = None
            else:
                # For text-only models, we use token ids as input.
                # While it is possible to use embeddings as input just like the
                # multimodal models, it is not desirable for performance since
                # then the embedding layer is not included in the ACL graph.
                input_ids = self.input_ids[:num_input_tokens]
                inputs_embeds = None
            if self.uses_mrope:
                positions = self.mrope_positions[:, :num_input_tokens]

            if self.torchair_graph_enabled and not with_prefill:
                input_ids = self.input_ids[:padded_batch_size]
                positions = self.positions[:padded_batch_size]

            # Run forward pass
            with set_forward_context(
                attn_metadata, self.vllm_config, num_tokens=num_input_tokens
            ):
                with ProfileExecuteDuration().capture_async("forward"):
                    model_kwargs = {}
                    if self.torchair_graph_enabled:
                        model_kwargs["kv_caches"] = self.kv_caches
                        model_kwargs["attn_metadata"] = attn_metadata
                    if self.torchair_graph_enabled and not with_prefill:
                        maybe_converting_weight_acl_format(
                            self.model, ACL_FORMAT_FRACTAL_NZ
                        )

                        compiled_model = self._get_torchair_lazy_compiled_model(
                            padded_batch_size
                        )
                        hidden_states = compiled_model(
                            input_ids=input_ids,
                            positions=positions,
                            intermediate_tensors=intermediate_tensors,
                            inputs_embeds=inputs_embeds,
                            **model_kwargs,
                        )
                    else:
                        assert self.model is not None
                        maybe_converting_weight_acl_format(
                            self.model, ACL_FORMAT_FRACTAL_ND
                        )
                        self.maybe_setup_kv_connector(scheduler_output)
                        self.maybe_execute_ucm_sparse_begin(
                            scheduler_output, attn_metadata
                        )

                        hidden_states = self.model(
                            input_ids=input_ids,
                            positions=positions,
                            intermediate_tensors=intermediate_tensors,
                            inputs_embeds=inputs_embeds,
                            **model_kwargs,
                        )
                        self.maybe_wait_for_kv_save()
                        self.maybe_execute_ucm_sparse_finished()

            use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
            if not use_spec_decode:
                # NOTE(woosuk): Due to chunked prefills, the batch may contain
                # partial requests. While we should not sample any token
                # from these partial requests, we do so for simplicity.
                # We will ignore the sampled tokens from the partial requests.
                # TODO: Support prompt logprobs.
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

            aux_hidden_states = None
            if self.use_aux_hidden_state_outputs:
                hidden_states, aux_hidden_states = hidden_states

            return (
                attn_metadata,
                hidden_states,
                spec_decode_metadata,
                positions,
                total_num_scheduled_tokens,
                logits_indices,
                aux_hidden_states,
                num_scheduled_tokens,
            )

        NPUModelRunner._process_reqs = _process_reqs

        @torch.inference_mode()
        def execute_model(
            self,
            scheduler_output: "SchedulerOutput",
            intermediate_tensors: Optional[IntermediateTensors] = None,
        ) -> Union[ModelRunnerOutput, torch.Tensor]:
            with ProfileExecuteDuration().capture_async("prepare input and forward"):
                self._update_states(scheduler_output)
                if not scheduler_output.total_num_scheduled_tokens:
                    # Return empty ModelRunnerOuptut if there's no work to do.
                    return EMPTY_MODEL_RUNNER_OUTPUT
                (
                    attn_metadata,
                    hidden_states,
                    spec_decode_metadata,
                    positions,
                    num_scheduled_tokens,
                    logits_indices,
                    aux_hidden_states,
                    num_scheduled_tokens_np,
                ) = self._process_reqs(scheduler_output, intermediate_tensors)

            with ProfileExecuteDuration().capture_async("post process"):
                # Broadcast PP output for external_launcher (torchrun)
                # to make sure we are synced across pp ranks
                # TODO: Support overlapping mirco-batches
                # https://github.com/vllm-project/vllm/issues/18019
                broadcast_pp_output = (
                    self.parallel_config.distributed_executor_backend
                    == "external_launcher"
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
                            hidden_states, num_scheduled_tokens, num_scheduled_tokens_np
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
                    logits = self.apply_grammar_bitmask(scheduler_output, logits)

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

                discard_sampled_tokens_req_indices: list[int] = []
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
                        # Ignore the sampled token.
                        # Rewind the generator state as if the token was not sampled.
                        generator = self.input_batch.generators.get(i)
                        if generator is not None:
                            generator.set_offset(generator.get_offset() - 4)
                        discard_sampled_tokens_req_indices.append(i)

                # NOTE: NPU -> CPU Sync happens here.
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

                for i in discard_sampled_tokens_req_indices:
                    valid_sampled_token_ids[i].clear()
                # Cache the sampled tokens in the model runner, so that the schedulerAdd commentMore actions
                # doesn't need to send them back.
                # NOTE(woosuk): As an exception, when using PP, the scheduler sends
                # the sampled tokens back, because there's no direct communication
                # between the first-stage worker and the last-stage worker.
                for req_idx, sampled_ids in enumerate(valid_sampled_token_ids):
                    if not sampled_ids:
                        continue

                    start_idx = self.input_batch.num_tokens_no_spec[req_idx]
                    end_idx = start_idx + len(sampled_ids)
                    assert end_idx <= self.model_config.max_model_len, (
                        "Sampled token IDs exceed the max model length. "
                        f"Total number of tokens: {end_idx} > max_model_len: "
                        f"{self.model_config.max_model_len}"
                    )

                    self.input_batch.token_ids_cpu[req_idx, start_idx:end_idx] = (
                        sampled_ids
                    )
                    self.input_batch.num_tokens_no_spec[req_idx] = end_idx
                    self.input_batch.num_tokens[req_idx] = end_idx
                    req_id = self.input_batch.req_ids[req_idx]
                    req_state = self.requests[req_id]
                    req_state.output_token_ids.extend(sampled_ids)

                spec_token_ids = self._get_spec_token_ids(
                    valid_sampled_token_ids,
                    sampling_metadata,
                    scheduler_output,
                    spec_decode_metadata,
                    positions,
                    num_scheduled_tokens,
                    hidden_states,
                    attn_metadata,
                    aux_hidden_states,
                )

                model_runner_output = ModelRunnerOutput(
                    req_ids=self.input_batch.req_ids,
                    req_id_to_index=self.input_batch.req_id_to_index,
                    sampled_token_ids=valid_sampled_token_ids,
                    spec_token_ids=spec_token_ids,
                    logprobs=logprobs_lists,
                    prompt_logprobs_dict=prompt_logprobs_dict,
                    pooler_output=[],
                )

            durations = ProfileExecuteDuration().pop_captured_sync()
            if durations:
                dr_str = [
                    f"[{tag}]:{duration:.2f}ms" for tag, duration in durations.items()
                ]
                captured_name = (
                    "Decode"
                    if self.attn_state == AscendAttentionState.DecodeOnly
                    else "Prefill"
                )
                logger.info(
                    "Profile execute duration [%s]:%s", captured_name, " ".join(dr_str)
                )

            return model_runner_output

        NPUModelRunner.execute_model = execute_model

        @staticmethod
        def maybe_setup_kv_connector(scheduler_output: "SchedulerOutput"):
            # Update KVConnector with the KVConnector metadata forward().
            if has_kv_transfer_group():
                kv_connector = get_kv_transfer_group()
                assert isinstance(kv_connector, KVConnectorBase_V1)
                assert scheduler_output.kv_connector_metadata is not None
                kv_connector.bind_connector_metadata(
                    scheduler_output.kv_connector_metadata
                )
                # Background KV cache transfers happen here.
                # These transfers are designed to be async and the requests
                # involved may be disjoint from the running requests.
                # Do this here to save a collective_rpc.
                kv_connector.start_load_kv(get_forward_context())

        NPUModelRunner.maybe_setup_kv_connector = maybe_setup_kv_connector

        @staticmethod
        def maybe_wait_for_kv_save():
            if has_kv_transfer_group():
                get_kv_transfer_group().wait_for_save()

        NPUModelRunner.maybe_wait_for_kv_save = maybe_wait_for_kv_save

        def maybe_execute_ucm_sparse_begin(
            self,
            scheduler_output: "SchedulerOutput",
            attn_metadata: CommonAttentionMetadata,
        ):
            if not has_ucm_sparse():
                return
            ucm_sparse = get_ucm_sparse()
            ucm_sparse.build_sparse_meta(
                scheduler_output, self.requests, self.input_batch, attn_metadata
            )
            ucm_sparse.execute_begin(scheduler_output)

        def maybe_execute_ucm_sparse_finished(self):
            if not has_ucm_sparse():
                return
            ucm_sparse = get_ucm_sparse()
            ucm_sparse.execute_finished()

        def ucm_sparse_request_finished_in_worker(self, request_id: str | int):
            if not has_ucm_sparse():
                return
            ucm_sparse = get_ucm_sparse()
            ucm_sparse.request_finished_in_worker(request_id)

        NPUModelRunner.maybe_execute_ucm_sparse_begin = maybe_execute_ucm_sparse_begin
        NPUModelRunner.maybe_execute_ucm_sparse_finished = (
            maybe_execute_ucm_sparse_finished
        )
        NPUModelRunner.ucm_sparse_request_finished_in_worker = (
            ucm_sparse_request_finished_in_worker
        )
    except ImportError as e:
        logger.error(f"Failed to patch model_runner_v1.py: {e}")
        raise


# ========================= vllm_ascend/worker/worker_v1.py =========================
def _patch_worker_v1() -> None:
    """Patch worker_v1.py for vLLM-Ascend."""
    try:
        import copy
        from typing import Optional

        from vllm.distributed.parallel_state import get_pp_group, get_tp_group
        from vllm.logger import logger
        from vllm.sequence import IntermediateTensors
        from vllm.v1.core.sched.output import SchedulerOutput
        from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
        from vllm_ascend.worker.worker_v1 import NPUWorker

        from ucm.sparse.state import ensure_ucm_sparse_initialized

        def execute_model(
            self,
            scheduler_output: "SchedulerOutput",
        ) -> Optional[ModelRunnerOutput]:
            intermediate_tensors = None
            if not get_pp_group().is_first_rank:
                intermediate_tensors = IntermediateTensors(
                    get_pp_group().recv_tensor_dict(all_gather_group=get_tp_group())
                )

            output = self.model_runner.execute_model(
                scheduler_output, intermediate_tensors
            )
            parallel_config = self.vllm_config.parallel_config
            if (
                parallel_config.distributed_executor_backend != "external_launcher"
                and not get_pp_group().is_last_rank
            ):
                assert isinstance(output, IntermediateTensors)
                get_pp_group().send_tensor_dict(
                    output.tensors, all_gather_group=get_tp_group()
                )

                kv_connector_output = output.kv_connector_output
                finished_sending = kv_connector_output.finished_sending
                finished_recving = kv_connector_output.finished_recving

                if not finished_sending and not finished_recving:
                    return EMPTY_MODEL_RUNNER_OUTPUT

                new_output = copy.copy(EMPTY_MODEL_RUNNER_OUTPUT)
                new_output.kv_connector_output = kv_connector_output
                return new_output

            assert isinstance(output, ModelRunnerOutput)
            return output

        NPUWorker.execute_model = execute_model

        original_init_worker_distributed_environment = (
            NPUWorker._init_worker_distributed_environment
        )

        def patched_init_worker_distributed_environment(self) -> None:
            original_init_worker_distributed_environment(self)
            ensure_ucm_sparse_initialized(self.vllm_config)

        NPUWorker._init_worker_distributed_environment = (
            patched_init_worker_distributed_environment
        )
    except ImportError as e:
        logger.error(f"Failed to patch worker_v1.py: {e}")
        raise
