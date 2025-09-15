# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import gc
import time
import weakref
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
import torch.distributed
import torch.nn as nn
import vllm.envs as envs
from tqdm import tqdm
from ucm.integration.vllm.ucm_sparse.base import (
    INVALID_SLOT,
    UcmSparseMetadata,
)
from ucm.integration.vllm.ucm_sparse.state import (
    get_ucm_sparse,
    has_ucm_sparse,
)
from vllm.attention import AttentionType, get_attn_backend
from vllm.attention.backends.abstract import AttentionBackend
from vllm.attention.layer import Attention
from vllm.compilation.counter import compilation_counter
from vllm.config import CompilationLevel, VllmConfig, get_layers_from_vllm_config
from vllm.distributed.eplb.eplb_state import EplbState
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorBase_V1
from vllm.distributed.parallel_state import (
    get_pp_group,
    get_tp_group,
    graph_capture,
    is_global_first_rank,
    prepare_communication_buffer_for_model,
)
from vllm.forward_context import DPMetadata, get_forward_context, set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.model_executor.model_loader import TensorizerLoader, get_model_loader
from vllm.model_executor.models.interfaces import has_step_pooler, is_mixture_of_experts
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalKwargs, PlaceholderRange
from vllm.multimodal.utils import group_mm_inputs_by_modality
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingType
from vllm.sequence import IntermediateTensors
from vllm.utils import (
    STR_DTYPE_TO_TORCH_DTYPE,
    DeviceMemoryProfiler,
    GiB_bytes,
    LazyLoader,
    async_tensor_h2d,
    cdiv,
    check_use_alibi,
    get_dtype_size,
    is_pin_memory_available,
    round_up,
)
from vllm.v1.attention.backends.mamba_attn import Mamba2AttentionBackend
from vllm.v1.attention.backends.utils import (
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.core.encoder_cache_manager import compute_encoder_budget
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
    SlidingWindowSpec,
)
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    LogprobsTensors,
    ModelRunnerOutput,
)
from vllm.v1.pool.metadata import PoolingMetadata
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.medusa import MedusaProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.ngram_proposer import NgramProposer
from vllm.v1.utils import bind_kv_cache
from vllm.v1.worker.block_table import BlockTable
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin

from ..sample.logits_processor import LogitsProcessorManager
from .utils import (
    gather_mm_placeholders,
    initialize_kv_cache_for_kv_sharing,
    sanity_check_mm_encoder_outputs,
    scatter_mm_placeholders,
)

if TYPE_CHECKING:
    import xgrammar as xgr
    import xgrammar.kernels.apply_token_bitmask_inplace_torch_compile as xgr_torch_compile  # noqa: E501
    from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
    from vllm.v1.core.sched.output import SchedulerOutput
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")
    xgr_torch_compile = LazyLoader(
        "xgr_torch_compile",
        globals(),
        "xgrammar.kernels.apply_token_bitmask_inplace_torch_compile",
    )

logger = init_logger(__name__)

from ucm.integration.vllm.vllm_adapter.v1.outputs import ModelRunnerOutput


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
    if self.use_cuda_graph and num_scheduled_tokens <= self.cudagraph_batch_sizes[-1]:
        # Use piecewise CUDA graphs.
        # Add padding to the batch size.
        num_input_tokens = self.vllm_config.pad_for_cudagraph(num_scheduled_tokens)
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
            inputs_embeds = self.model.get_input_embeddings(input_ids, mm_embeds)
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
        self.maybe_execute_ucm_sparse_begin(scheduler_output)

        model_output = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        finished_dumping = self.maybe_wait_for_kv_save()
        self.maybe_execute_ucm_sparse_finished()
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
        finished_dumping=finished_dumping,
    )


@staticmethod
def maybe_wait_for_kv_save() -> Optional[dict[str, list[str]]]:
    if has_kv_transfer_group():
        return get_kv_transfer_group().wait_for_save()


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
                    audio_feature_lengths.extend(mm_input["audio_feature_lengths"])
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
        if num_computed_tokens <= req_state.num_computed_tokens:
            # The request was rescheduled after a KV load failure. Clear
            # the last sampled tokens and rewind the generator state
            len_output_token_ids = len(req_state.output_token_ids)
            del req_state.output_token_ids[req_state.len_last_output_token_ids :]
            if req_state.generator:
                req_state.generator.set_offset(req_state.last_generator_offset)
            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is not None:
                len_last_sampled = (
                    len_output_token_ids - req_state.len_last_output_token_ids
                )
                end_idx = (
                    self.input_batch.num_tokens_no_spec[req_index] - len_last_sampled
                )
                self.input_batch.num_tokens[req_index] = end_idx
                self.input_batch.num_tokens_no_spec[req_index] = end_idx

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
                req_state.output_token_ids.extend(new_token_ids[-num_new_tokens:])

        req_state.len_last_output_token_ids = len(req_state.output_token_ids)
        if req_state.generator:
            req_state.last_generator_offset = req_state.generator.get_offset()

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

        if req_state.generator:
            assert req_state.last_generator_offset is not None
            self.input_batch.generators_last_offset[req_index] = (
                req_state.last_generator_offset
            )

        # Update the persistent batch.
        self.input_batch.num_computed_tokens_cpu[req_index] = num_computed_tokens
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
            self.input_batch.num_tokens[req_index] = end_token_index

        # Add spec_token_ids to token_ids_cpu.
        spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(req_id, ())
        if spec_token_ids:
            num_spec_tokens = len(spec_token_ids)
            start_index = self.input_batch.num_tokens_no_spec[req_index]
            end_token_index = start_index + num_spec_tokens
            self.input_batch.token_ids_cpu[req_index, start_index:end_token_index] = (
                spec_token_ids
            )
            # NOTE(woosuk): `num_tokens` here may include spec tokens.
            self.input_batch.num_tokens[req_index] += num_spec_tokens

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
        self.input_batch.num_computed_tokens_cpu[req_indices], arange, out=positions_np
    )

    # Calculate M-RoPE positions.
    # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
    if self.uses_mrope:
        self._calc_mrope_positions(scheduler_output)

        self.seq_lens_np[:num_reqs] = (
            self.input_batch.num_computed_tokens_cpu[:num_reqs] + num_scheduled_tokens
        )

        # TODO: improve performance, no `positions_np.copy()`
        sparsed_positions = positions_np.copy()
        req_sparsed_slots = scheduler_output.req_sparsed_slots
        for req_id in self.input_batch.req_id_to_index:
            is_sparsed_request = req_sparsed_slots[req_id] != INVALID_SLOT
            req_index = self.input_batch.req_id_to_index[req_id]
            if is_sparsed_request:
                sparsed_positions[req_index] -= (
                    self.seq_lens_cpu[:num_reqs][req_index] - req_sparsed_slots[req_id]
                )

    # Get token indices.
    # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
    # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
    # where M is the max_model_len.
    token_indices = positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]

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
        block_table: BlockTable = self.input_batch.block_table[kv_cache_group_id]
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
        is_sparsed_request = scheduler_output.req_sparsed_slots[req_id] != INVALID_SLOT
        if is_sparsed_request:
            self.seq_lens_np[req_index] = scheduler_output.req_sparsed_slots[req_id]

    # Copy the tensors to the GPU.
    self.input_ids[:total_num_scheduled_tokens].copy_(
        self.input_ids_cpu[:total_num_scheduled_tokens], non_blocking=True
    )
    if self.uses_mrope:
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        self.mrope_positions[:, :total_num_scheduled_tokens].copy_(
            self.mrope_positions_cpu[:, :total_num_scheduled_tokens], non_blocking=True
        )
    else:
        # Common case (1D positions)
        self.positions_cpu[:total_num_scheduled_tokens] = torch.from_numpy(
            sparsed_positions[:total_num_scheduled_tokens]
        )
        self.positions[:total_num_scheduled_tokens].copy_(
            self.positions_cpu[:total_num_scheduled_tokens], non_blocking=True
        )

    self.query_start_loc[: num_reqs + 1].copy_(
        self.query_start_loc_cpu[: num_reqs + 1], non_blocking=True
    )
    self.seq_lens[:num_reqs].copy_(self.seq_lens_cpu[:num_reqs], non_blocking=True)

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


def maybe_execute_ucm_sparse_begin(self, scheduler_output: "SchedulerOutput"):
    if not has_ucm_sparse():
        return
    ucm_sparse = get_ucm_sparse()
    ucm_sparse.build_sparse_meta(scheduler_output, self.requests, self.input_batch)
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
