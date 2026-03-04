import os
import math
from typing import (TYPE_CHECKING, Any, Dict, Optional,
                    Union, cast)

import numpy as np
import torch
import torch.nn as nn
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group)
from vllm.distributed.parallel_state import (get_pp_group,
                                             get_tp_group)
from vllm.forward_context import BatchDescriptor
from vllm.logger import logger
from vllm.model_executor.models.interfaces_base import VllmModelForPooling
from vllm.sampling_params import SamplingType
from vllm.sequence import IntermediateTensors
from vllm.utils import (LazyLoader, get_dtype_size)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.kv_cache_interface import (EncoderOnlyAttentionSpec,
                                        FullAttentionSpec, KVCacheConfig,
                                        MambaSpec)
from vllm.v1.outputs import (EMPTY_MODEL_RUNNER_OUTPUT, AsyncModelRunnerOutput,
                             ModelRunnerOutput)
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorOutput
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin
from vllm.v1.worker.utils import bind_kv_cache

from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.spec_decode.interface import SpecDcodeType
from vllm_ascend.utils import (ProfileExecuteDuration,
                               enable_sp, 
                               lmhead_tp_enable)
from vllm_ascend.worker.npu_input_batch import CachedRequestState
from vllm_ascend.worker.model_runner_v1 import AsyncNPUModelRunnerOutput

from ucm.sparse.state import get_ucm_sparse, has_ucm_sparse
from ucm.sparse.base import INVALID_SLOT

if TYPE_CHECKING:
    import xgrammar as xgr  # type: ignore[import-untyped]
    from vllm.v1.core.sched.output import SchedulerOutput
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")


class NPUModelRunner(LoRAModelRunnerMixin):
    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        # Remove finished requests from the cached states.
        self.ucm_sparse_update_states(scheduler_output)
        for req_id in scheduler_output.finished_req_ids:
            self.ucm_sparse_request_finished_in_worker(req_id)
            self.requests.pop(req_id, None)

        # Remove the finished requests from the persistent batch.
        # NOTE(woosuk): There could be an edge case where finished_req_ids and
        # scheduled_req_ids overlap. This happens when a request is aborted and
        # then resubmitted with the same ID. In this case, we treat them as two
        # distinct requests - clearing the cached states for the first request
        # and handling the second as a new request.
        for req_id in scheduler_output.finished_req_ids:
            self.input_batch.remove_request(req_id)
        for mm_hash in scheduler_output.free_encoder_mm_hashes:
            self.encoder_cache.pop(mm_hash, None)
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

            if sampling_params and \
                sampling_params.sampling_type == SamplingType.RANDOM_SEED:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None

            if pooling_params:
                assert (task := pooling_params.task) is not None, (
                    "You did not set `task` in the API")
                model = cast(VllmModelForPooling, self.get_model())
                to_update = model.pooler.get_pooling_updates(task)
                to_update.apply(pooling_params)

            backward_kwargs = {}
            backward_kwargs["mm_features"] = new_req_data.mm_features

            self.requests[req_id] = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                sampling_params=sampling_params,
                pooling_params=pooling_params,
                generator=generator,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
                **backward_kwargs,
            )

            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            if self.uses_mrope:
                self._init_mrope_positions(self.requests[req_id])

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
                num_new_tokens = (num_computed_tokens + len(new_token_ids) -
                                  req_state.num_tokens)
                if num_new_tokens == 1:
                    # Avoid slicing list in most common case.
                    req_state.output_token_ids.append(new_token_ids[-1])
                elif num_new_tokens > 0:
                    req_state.output_token_ids.extend(
                        new_token_ids[-num_new_tokens:])

            # Update the block IDs.
            if resumed_from_preemption or is_sparsed_request:
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                req_state.block_ids = new_block_ids
            else:
                if new_block_ids is not None:
                    # Append the new blocks to the existing block IDs.
                    for block_ids, new_ids in zip(req_state.block_ids,
                                                  new_block_ids):
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
                num_computed_tokens)

            if is_sparsed_request:
                self.input_batch.block_table.reset_row(req_index)

            if new_block_ids is not None:
                self.input_batch.block_table.append_row(
                    new_block_ids, req_index)

            # For the last rank, we don't need to update the token_ids_cpu
            # because the sampled tokens are already cached.
            if not is_last_rank:
                # Add new_token_ids to token_ids_cpu.
                start_token_index = num_computed_tokens
                end_token_index = num_computed_tokens + len(new_token_ids)
                self.input_batch.token_ids_cpu[
                    req_index,
                    start_token_index:end_token_index] = new_token_ids
                self.input_batch.num_tokens_no_spec[
                    req_index] = end_token_index
                self.input_batch.num_tokens[req_index] = end_token_index

            # Add spec_token_ids to token_ids_cpu.
            spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id, ()))
            if spec_token_ids:
                num_spec_tokens = len(spec_token_ids)
                start_index = self.input_batch.num_tokens_no_spec[req_index]
                end_token_index = start_index + num_spec_tokens
                self.input_batch.token_ids_cpu[
                    req_index, start_index:end_token_index] = spec_token_ids
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
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> tuple[dict[str, Any], torch.Tensor, np.ndarray, int, torch.Tensor,
               int, torch.Tensor, SpecDecodeMetadata, Optional[torch.Tensor],
               Optional[torch.Tensor], Optional[torch.Tensor], int]:
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        self.input_batch.block_table.commit_block_table(num_reqs)

        # Get the number of scheduled tokens for each request.
        req_ids = self.input_batch.req_ids
        tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
        num_scheduled_tokens = np.array(tokens, dtype=np.int32)
        max_num_scheduled_tokens = num_scheduled_tokens.max()
        num_valid_tokens = np.array([
            num_tokens -
            len(scheduler_output.scheduled_spec_decode_tokens.get(i, []))
            for num_tokens, i in zip(tokens, req_ids)
        ],
                                    dtype=np.int32)

        if (self.use_aclgraph and total_num_scheduled_tokens
                <= self.aclgraph_batch_sizes[-1]):
            # Add padding to the batch size.
            num_input_tokens = self.vllm_config.pad_for_cudagraph(
                total_num_scheduled_tokens)
        elif self.use_aclgraph and enable_sp(self.vllm_config):
            # When using aclgraph, if total_num_scheduled_tokens exceeds the maximum graph size,
            # the model will fall back to running its FX graph in eager mode.
            # In this case, when sequence parallelism is enabled, we need to pad tokens to align
            # with tp_size because pad_size cannot be captured by the FX graph
            tp_size = self.vllm_config.parallel_config.tensor_parallel_size
            num_input_tokens = math.ceil(
                total_num_scheduled_tokens / tp_size) * tp_size
        else:
            # Eager mode.
            num_input_tokens = total_num_scheduled_tokens

        # Get the attention state.
        attn_state = self._build_attn_state(num_reqs, num_scheduled_tokens,
                                            num_valid_tokens)
        self.attn_state = attn_state  # type: ignore

        # Determine if it's a splitfuse batch
        with_prefill = attn_state not in [
            AscendAttentionState.DecodeOnly, AscendAttentionState.SpecDecoding
        ]

        self.query_lens = torch.from_numpy(num_scheduled_tokens)
        enable_dbo = self._check_dbo_is_valid(self.query_lens.tolist(),
                                              attn_state,
                                              total_num_scheduled_tokens)

        # Get info across DP ranks.
        # NOTE: maybe_padded_num_tokens is only used when using TorchAir with DP,
        # Otherwise, it's just max_tokens_across_dp_cpu
        (maybe_padded_num_tokens, num_tokens_across_dp, with_prefill,
         enable_dbo) = self._sync_metadata_across_dp(num_input_tokens,
                                                     with_prefill, enable_dbo)

        # TODO: Now that num_input_tokens is basically identical with maybe_padded_num_tokens
        # We should consider removing maybe_padded_num_tokens later
        num_input_tokens = maybe_padded_num_tokens

        # Hot-Swap lora model
        if self.lora_config:
            self.set_active_loras(self.input_batch, num_scheduled_tokens)

        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        req_indices = np.repeat(self.arange_np[:num_reqs],
                                num_scheduled_tokens)

        # cu_num_tokens: [2, 5, 3] -> [2, 7, 10]
        # arange: [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        cu_num_tokens, arange = self._get_cumsum_and_arange(
            num_scheduled_tokens)

        positions_np = self.positions_np[:total_num_scheduled_tokens]
        np.add(self.input_batch.num_computed_tokens_cpu[req_indices],
               arange,
               out=positions_np)

        # Calculate M-RoPE positions.
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            self._calc_mrope_positions(scheduler_output)

            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            self.mrope_positions[:, :total_num_scheduled_tokens].copy_(
                self.mrope_positions_cpu[:, :total_num_scheduled_tokens],
                non_blocking=True)

        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        token_indices = (positions_np +
                         req_indices * self.input_batch.token_ids_cpu.shape[1])

        # Prepare input_ids.
        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        torch.index_select(self.input_batch.token_ids_cpu_tensor.flatten(),
                           0,
                           torch.from_numpy(token_indices),
                           out=self.input_ids_cpu[:total_num_scheduled_tokens])

        sparsed_positions = positions_np.copy()
        req_sparsed_slots = scheduler_output.req_sparsed_slots
        for req_id in self.input_batch.req_id_to_index:
            is_sparsed_request = req_sparsed_slots[req_id] != INVALID_SLOT
            req_index = self.input_batch.req_id_to_index[req_id]
            offset = 0 if req_index == 0 else cu_num_tokens[req_index - 1] # TODO: support MTP
            if is_sparsed_request:
                sparsed_positions[offset] = req_sparsed_slots[req_id] - 1

        # Prepare some information for building Attention-Metadata
        # Compute and commit slot mapping
        self.input_batch.block_table.compute_slot_mapping(
            req_indices, sparsed_positions)
        self.input_batch.block_table.commit_slot_mapping(
            total_num_scheduled_tokens)

        self.query_start_loc_np[0] = 0
        self.query_start_loc_np[1:num_reqs + 1] = cu_num_tokens
        self.query_start_loc[:num_reqs + 1].copy_(
            self.query_start_loc_cpu[:num_reqs + 1], non_blocking=True)

        self.seq_lens_np[:num_reqs] = (
            self.input_batch.num_computed_tokens_cpu[:num_reqs] +
            num_scheduled_tokens)
        self.seq_lens[:num_reqs].copy_(self.seq_lens_cpu[:num_reqs],
                                       non_blocking=True)

        # Fill unused with -1. Needed for reshape_and_cache
        self.query_start_loc[num_reqs + 1:].fill_(-1)
        self.seq_lens[num_reqs:].fill_(0)

        self.query_lens = torch.from_numpy(num_scheduled_tokens)

        # Copy the tensors to the NPU.
        self._prepare_input_ids(total_num_scheduled_tokens, cu_num_tokens)
        self.positions_cpu[total_num_scheduled_tokens:num_input_tokens].zero_()
        self.positions[:num_input_tokens].copy_(
            self.positions_cpu[:num_input_tokens], non_blocking=True)

        # Make Attention metadata
        positions_cpu = self.positions_cpu[:num_input_tokens]
        positions = self.positions[:num_input_tokens]
        seq_lens_cpu = self.seq_lens_cpu[:num_reqs]

        for req_id in self.input_batch.req_id_to_index:
            is_sparsed_request = scheduler_output.req_sparsed_slots[req_id] != INVALID_SLOT
            req_index = self.input_batch.req_id_to_index[req_id]
            if is_sparsed_request:
                seq_lens_cpu[req_index] = req_sparsed_slots[req_id]

        attn_state = self._build_attn_state(num_reqs, num_scheduled_tokens,
                                            num_valid_tokens)
        self.attn_mask = self._make_attention_mask(seq_lens=seq_lens_cpu,
                                                   position=torch.tensor(sparsed_positions),
                                                   attn_state=attn_state)
        self.attn_state = attn_state  # type: ignore

        self.with_prefill = with_prefill
        self.num_tokens_across_dp = num_tokens_across_dp
        self._update_graph_pad_size(with_prefill, maybe_padded_num_tokens)
        attn_metadata: dict[str, Any] = {}

        # _prepare_inputs may reorder the batch, so we must gather
        # multi-modal outputs after that to ensure the correct order
        if self.is_multimodal_model:
            # Run the multimodal encoder if any.
            self._execute_mm_encoder(scheduler_output)
            mm_embeds, is_mm_embed = self._gather_mm_embeddings(
                scheduler_output)
            # NOTE(woosuk): To unify token ids and soft tokens (vision
            # embeddings), we always use embeddings (rather than token ids)
            # as input to the multimodal model, even when the input is text.
            input_ids = self.input_ids[:total_num_scheduled_tokens]
            model_type = self.vllm_config.model_config.hf_config.model_type
            if model_type == "qwen2_5_vl" or model_type == "qwen3_vl_moe":
                inputs_embeds = self.model.get_input_embeddings(
                    input_ids,
                    multimodal_embeddings=mm_embeds,
                    is_multimodal=is_mm_embed,
                )
            else:
                if mm_embeds:
                    inputs_embeds = self.model.get_input_embeddings(
                        input_ids, mm_embeds)
                else:
                    inputs_embeds = self.model.get_input_embeddings(input_ids)
            # TODO(woosuk): Avoid the copy. Optimize.
            self.inputs_embeds[:total_num_scheduled_tokens].copy_(
                inputs_embeds)
            inputs_embeds = self.inputs_embeds[:num_input_tokens]
            input_ids = None
        else:
            # For text-only models, we use token ids as input.
            # While it is possible to use embeddings as input just like the
            # multimodal models, it is not desirable for performance since
            # then the embedding layer is not included in the ACL graph.
            input_ids = self.input_ids[:num_input_tokens]
            inputs_embeds = None
        positions = self.positions[:num_input_tokens]
        input_ids, positions = self._update_input_ids_and_positions(
            input_ids, positions, num_input_tokens, with_prefill,
            maybe_padded_num_tokens)

        if get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            assert intermediate_tensors is not None
            assert self.intermediate_tensors is not None
            for k, v in intermediate_tensors.items():
                self.intermediate_tensors[k][:num_input_tokens].copy_(
                    v[:num_input_tokens], non_blocking=True)
            intermediate_tensors = IntermediateTensors({
                k: v[:num_input_tokens]
                for k, v in self.intermediate_tensors.items()
            })

        use_spec_decode = len(
            scheduler_output.scheduled_spec_decode_tokens) > 0
        if not use_spec_decode:
            # NOTE(woosuk): Due to chunked prefills, the batch may contain
            # partial requests. While we should not sample any token
            # from these partial requests, we do so for simplicity.
            # We will ignore the sampled tokens from the partial requests.
            # TODO: Support prompt logprobs.
            spec_decode_metadata = None
            logits_indices = torch.from_numpy(cu_num_tokens - 1).to(
                self.device, non_blocking=True)
        else:
            # Get the number of draft tokens for each request.
            # Iterate over the dictionary rather than all requests since not all
            # requests have draft tokens.
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
            for req_id, draft_token_ids in (
                    scheduler_output.scheduled_spec_decode_tokens.items()):
                req_idx = self.input_batch.req_id_to_index[req_id]
                num_draft_tokens[req_idx] = len(draft_token_ids)

            spec_decode_metadata = self._calc_spec_decode_metadata(
                num_draft_tokens, cu_num_tokens)
            logits_indices = spec_decode_metadata.logits_indices
            self.num_draft_tokens.np[:num_reqs] = num_draft_tokens
            self.num_draft_tokens.np[num_reqs:].fill(0)
            self.num_draft_tokens.copy_to_gpu()

        # Used in the below loop.
        # query_start_loc_cpu = self.query_start_loc.cpu[:num_reqs + 1]
        num_computed_tokens_cpu = (
            self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs])
        spec_decode_common_attn_metadata = None
        if use_spec_decode and self.need_accepted_tokens:
            self.num_accepted_tokens.np[:num_reqs] = (
                self.input_batch.num_accepted_tokens_cpu[:num_reqs])
            self.num_accepted_tokens.np[num_reqs:].fill(1)
            self.num_accepted_tokens.copy_to_gpu()

        # Prepare the attention metadata for each KV cache group and make layers
        # in the same group share the same metadata.
        for kv_cache_group_id, kv_cache_group_spec in enumerate(
                self.kv_cache_config.kv_cache_groups):
            if isinstance(kv_cache_group_spec.kv_cache_spec,
                          EncoderOnlyAttentionSpec):
                # Encoder-only layers do not have KV cache, so we need to
                # create a dummy block table and slot mapping for them.
                blk_table_tensor = torch.zeros(
                    (num_reqs, 1),
                    dtype=torch.int32,
                    device=self.device,
                )
                slot_mapping = torch.zeros(
                    (total_num_scheduled_tokens, ),
                    dtype=torch.int64,
                    device=self.device,
                )
            else:
                blk_table = self.input_batch.block_table[kv_cache_group_id]
                blk_table_tensor = blk_table.get_device_tensor()
                slot_mapping = blk_table.slot_mapping_cpu[:
                                                          total_num_scheduled_tokens]
                self.slot_mapping[:total_num_scheduled_tokens].copy_(
                    slot_mapping[:total_num_scheduled_tokens],
                    non_blocking=True,
                )
                self.slot_mapping[total_num_scheduled_tokens:].fill_(0)

            # Make AscendCommonAttentionMetadata
            common_attn_metadata = AscendCommonAttentionMetadata(
                query_start_loc=self.query_start_loc[:num_reqs + 1],
                query_start_loc_cpu=self.query_start_loc_cpu[:num_reqs + 1],
                seq_lens_cpu=self.seq_lens_cpu,
                seq_lens=self.seq_lens_cpu[:num_reqs],
                num_reqs=num_reqs,
                num_actual_tokens=total_num_scheduled_tokens,
                num_input_tokens=num_input_tokens,
                actual_seq_lengths_q=self.actual_seq_lengths_q,
                # TODO: change this to the right block table for linear attn
                block_table_tensor=blk_table_tensor[:num_reqs],
                slot_mapping=self.slot_mapping,
                num_computed_tokens_cpu=num_computed_tokens_cpu,
                positions=self.positions,
                attn_mask=self.attn_mask,
                spec_attn_mask=self.spec_attn_mask,
                attn_state=self.attn_state,
                enable_dbo_across_dp=enable_dbo,
                is_only_prefill=bool(np.all(num_valid_tokens != 1)),
                max_query_len=max_num_scheduled_tokens,
                graph_pad_size=self.graph_pad_size,
                decode_token_per_req=self.decode_token_per_req,
                cos=self.cos,
                sin=self.sin,
            )

            if self.speculative_config and \
                spec_decode_common_attn_metadata is None:
                spec_decode_common_attn_metadata = common_attn_metadata

            for attn_group in self.attn_groups[kv_cache_group_id]:
                common_prefix_len = 0
                extra_attn_metadata_args = {}
                builder = attn_group.get_metadata_builder()
                if isinstance(builder, GDNAttentionMetadataBuilder
                              ) or self.model_config.runner_type == "pooling":
                    if use_spec_decode:
                        extra_attn_metadata_args = dict(
                            num_accepted_tokens=self.num_accepted_tokens.
                            gpu[:num_reqs],
                            num_decode_draft_tokens_cpu=self.num_draft_tokens.
                            gpu[:num_reqs],
                        )
                    attn_metadata_i = builder.build(
                        common_prefix_len=common_prefix_len,
                        common_attn_metadata=common_attn_metadata,
                        **extra_attn_metadata_args)
                else:
                    attn_metadata_i = builder.build(
                        common_prefix_len=common_prefix_len,
                        common_attn_metadata=common_attn_metadata,
                        model=self.get_model(),
                        **extra_attn_metadata_args)

                for layer_name in attn_group.layer_names:
                    attn_metadata[layer_name] = attn_metadata_i

        if lmhead_tp_enable():
            max_num_reqs_across_dp = maybe_padded_num_tokens if not with_prefill else self.max_num_reqs
            logits_indices = nn.functional.pad(
                logits_indices,
                (0, max_num_reqs_across_dp - logits_indices.shape[0]))

        return (attn_metadata, positions, num_scheduled_tokens,
                num_input_tokens, num_tokens_across_dp,
                maybe_padded_num_tokens, logits_indices, spec_decode_metadata,
                input_ids, inputs_embeds, intermediate_tensors,
                max_num_scheduled_tokens)
    
    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, AsyncModelRunnerOutput, IntermediateTensors]:
        with ProfileExecuteDuration().capture_async("prepare input"):
            self._update_states(scheduler_output)
            if not scheduler_output.total_num_scheduled_tokens:
                if not has_kv_transfer_group():
                    logger.debug(
                        "skip this step for we receive the data from remote disaggregate prefill node"
                    )
                    # Return empty ModelRunnerOuptut if there's no work to do.
                    return EMPTY_MODEL_RUNNER_OUTPUT
                return self.kv_connector_no_forward(scheduler_output)

            if self.dynamic_eplb:
                self.eplb_updator.forward_before()

            (attn_metadata, positions, num_scheduled_tokens_np,
             num_input_tokens, num_tokens_across_dp, maybe_padded_num_tokens,
             logits_indices, spec_decode_metadata, input_ids, inputs_embeds,
             intermediate_tensors,
             max_query_len) = (self._prepare_inputs(scheduler_output,
                                                    intermediate_tensors))

            if self.dynamic_eplb:
                self.eplb_updator.take_update_info_from_eplb_process()

        moe_comm_type = self._select_moe_comm_method(num_input_tokens,
                                                     self.with_prefill)

        uniform_decode = (max_query_len == self.uniform_decode_query_len) and (
            scheduler_output.total_num_scheduled_tokens
            == self.input_batch.num_reqs * max_query_len)
        batch_descriptor = BatchDescriptor(num_tokens=num_input_tokens,
                                           uniform_decode=uniform_decode)
        aclgraph_runtime_mode, batch_descriptor = \
            self.aclgraph_dispatcher.dispatch(batch_descriptor)

        # Run forward pass
        with ProfileExecuteDuration().capture_async("forward"):
            with set_ascend_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_input_tokens,
                    num_tokens_across_dp=num_tokens_across_dp,
                    with_prefill=self.with_prefill,
                    reserved_mc2_mask=self.reserved_mc2_mask,
                    moe_comm_type=moe_comm_type,
                    aclgraph_runtime_mode=aclgraph_runtime_mode,
                    batch_descriptor=batch_descriptor,
                    num_actual_tokens=scheduler_output.
                    total_num_scheduled_tokens,
                    prefetch_stream=self.prefetch_stream,
                    model_instance=self.model,
                    weight_prefetch_method=self.weight_prefetch_method):
                self.maybe_setup_kv_connector(scheduler_output)
                self.maybe_execute_ucm_sparse_begin(scheduler_output, attn_metadata)


                hidden_states = self._generate_process_reqs_hidden_states(
                    attn_metadata, self.with_prefill, maybe_padded_num_tokens,
                    input_ids, positions, intermediate_tensors, inputs_embeds)

            self.maybe_wait_for_kv_save()
            logits_indices = self.maybe_execute_ucm_sparse_finished(logits_indices)
            finished_sending, finished_recving = self.get_finished_kv_transfer(
                scheduler_output)

            aux_hidden_states = None
            if self.drafter and self.drafter.name == SpecDcodeType.EAGLE3:
                hidden_states, aux_hidden_states = hidden_states

        kv_connector_output = KVConnectorOutput(
            finished_sending=finished_sending,
            finished_recving=finished_recving)
        finished_sending = None
        finished_recving = None
        with ProfileExecuteDuration().capture_async("post process"):
            # Broadcast PP output for external_launcher (torchrun)
            # to make sure we are synced across pp ranks
            # TODO: Support overlapping mirco-batches
            # https://github.com/vllm-project/vllm/issues/18019
            broadcast_pp_output = \
                self.parallel_config.distributed_executor_backend \
                == "external_launcher" and len(get_pp_group().ranks) > 0
            if not get_pp_group().is_last_rank:
                # For mid-pipeline stages, return the hidden states.
                if not broadcast_pp_output:
                    hidden_states.kv_connector_output = kv_connector_output
                    return hidden_states
                assert isinstance(hidden_states, IntermediateTensors)
                get_pp_group().send_tensor_dict(
                    hidden_states.tensors, all_gather_group=get_tp_group())
                logits = None
            else:
                if self.input_batch.pooling_params:
                    return self._pool(
                        hidden_states,
                        scheduler_output.total_num_scheduled_tokens,
                        num_scheduled_tokens_np, finished_sending,
                        finished_recving, kv_connector_output)
                sample_hidden_states = hidden_states[logits_indices]
                logits = self.model.compute_logits(sample_hidden_states)
            if broadcast_pp_output:
                model_output_broadcast_data = {
                    "logits": logits.contiguous(),
                } if logits is not None else {}
                model_output_broadcast_data = get_pp_group(
                ).broadcast_tensor_dict(model_output_broadcast_data,
                                        src=len(get_pp_group().ranks) - 1)
                assert model_output_broadcast_data is not None
                logits = model_output_broadcast_data["logits"]

            # Apply structured output bitmasks if present
            if scheduler_output.grammar_bitmask is not None:
                logits = self.apply_grammar_bitmask(scheduler_output, logits)

            # Sample the next token and get logprobs if needed.
            sampling_metadata = self.input_batch.sampling_metadata
            if spec_decode_metadata is None:
                if lmhead_tp_enable() and logits is not None:
                    logits = logits[:self.input_batch.num_reqs]
                sampler_output = self.sampler(
                    logits=logits,
                    sampling_metadata=sampling_metadata,
                )
            else:
                if lmhead_tp_enable() and logits is not None:
                    logits = logits[:len(spec_decode_metadata.logits_indices)]
                # When indexing with a tensor (bonus_logits_indices), PyTorch
                # creates a new tensor with separate storage from the original
                # logits tensor. This means any in-place operations on bonus_logits
                # won't affect the original logits tensor.
                assert logits is not None
                bonus_logits = logits[
                    spec_decode_metadata.bonus_logits_indices]
                sampler_output = self.sampler(
                    logits=bonus_logits,
                    sampling_metadata=sampling_metadata,
                )
                bonus_token_ids = sampler_output.sampled_token_ids

                # Just like `bonus_logits`, `target_logits` is a new tensor with
                # separate storage from the original `logits` tensor. Therefore,
                # it is safe to update `target_logits` in place.
                target_logits = logits[
                    spec_decode_metadata.target_logits_indices]
                output_token_ids = self.rejection_sampler(
                    spec_decode_metadata,
                    None,  # draft_probs
                    target_logits,
                    bonus_token_ids,
                    sampling_metadata,
                )
                sampler_output.sampled_token_ids = output_token_ids
                if self.need_accepted_tokens:
                    self._update_states_after_model_execute(output_token_ids)

            discard_sampled_tokens_req_indices: list[int] = []
            # TODO(woosuk): The following loop can be slow since it iterates over
            # the requests one by one. Optimize.
            discard_sampled_tokens_req_indices = []
            for i, req_id in enumerate(self.input_batch.req_ids):
                req_state = self.requests[req_id]
                seq_len = (req_state.num_computed_tokens +
                           scheduler_output.num_scheduled_tokens[req_id])
                if seq_len < req_state.num_tokens:
                    # Ignore the sampled token.
                    # Rewind the generator state as if the token was not sampled.
                    generator = self.input_batch.generators.get(i)
                    if generator is not None:
                        generator.set_offset(generator.get_offset() - 4)
                    discard_sampled_tokens_req_indices.append(i)

            # Copy some objects so they don't get modified after returning.
            # This is important when using async scheduling.
            req_ids_output_copy = self.input_batch.req_ids.copy()
            req_id_to_index_output_copy = \
                self.input_batch.req_id_to_index.copy()

            # NOTE: NPU -> CPU Sync happens here.
            # Move as many CPU operations as possible before this sync point.
            logprobs_tensors = sampler_output.logprobs_tensors
            logprobs_lists = logprobs_tensors.tolists() \
                if logprobs_tensors is not None else None

            # Compute prompt logprobs if needed.
            prompt_logprobs_dict = self._get_prompt_logprobs_dict(
                hidden_states[:scheduler_output.total_num_scheduled_tokens],
                scheduler_output,
            )

            num_sampled_tokens = sampler_output.sampled_token_ids.shape[0]
            sampled_token_ids = sampler_output.sampled_token_ids
            if not self.use_async_scheduling:
                # Get the valid generated tokens.
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
            else:
                valid_sampled_token_ids = []
                invalid_req_indices = list(discard_sampled_tokens_req_indices)
                invalid_req_indices_set = set(invalid_req_indices)
                assert sampled_token_ids.shape[-1] == 1

                # Cache the sampled tokens on the NPU and avoid CPU sync.
                # These will be copied into input_ids in the next step
                # when preparing inputs.
                self.input_batch.prev_sampled_token_ids = \
                    sampled_token_ids
                self.input_batch.prev_sampled_token_ids_invalid_indices = \
                    invalid_req_indices_set
                self.input_batch.prev_req_id_to_index = {
                    req_id: i
                    for i, req_id in enumerate(self.input_batch.req_ids)
                    if i not in invalid_req_indices_set
                }
            # Cache the sampled tokens in the model runner, so that the scheduler
            # doesn't need to send them back.
            # NOTE(woosuk): As an exception, when using PP, the scheduler sends
            # the sampled tokens back, because there's no direct communication
            # between the first-stage worker and the last-stage worker.
            for req_idx in range(num_sampled_tokens):
                if self.use_async_scheduling:
                    sampled_ids = [-1] * 1 if \
                        req_idx not in invalid_req_indices_set else None
                else:
                    sampled_ids = valid_sampled_token_ids[req_idx]
                if not sampled_ids:
                    continue

                start_idx = self.input_batch.num_tokens_no_spec[req_idx]
                end_idx = start_idx + len(sampled_ids)
                assert end_idx <= self.model_config.max_model_len, (
                    "Sampled token IDs exceed the max model length. "
                    f"Total number of tokens: {end_idx} > max_model_len: "
                    f"{self.model_config.max_model_len}")

                self.input_batch.token_ids_cpu[req_idx,
                                               start_idx:end_idx] = sampled_ids
                self.input_batch.num_tokens_no_spec[req_idx] = end_idx
                self.input_batch.num_tokens[req_idx] = end_idx
                req_id = self.input_batch.req_ids[req_idx]
                req_state = self.requests[req_id]
                req_state.output_token_ids.extend(sampled_ids)

            if self.speculative_config:
                self._draft_token_ids = self.propose_draft_token_ids(
                    valid_sampled_token_ids,
                    sampling_metadata,
                    scheduler_output,
                    spec_decode_metadata,
                    positions,
                    scheduler_output.total_num_scheduled_tokens,
                    hidden_states,
                    attn_metadata,
                    aux_hidden_states,
                )

            if has_kv_transfer_group():
                get_kv_transfer_group().clear_connector_metadata()

        extra_args = ({"kv_connector_output": kv_connector_output})

        model_runner_output = ModelRunnerOutput(
            req_ids=req_ids_output_copy,
            req_id_to_index=req_id_to_index_output_copy,
            sampled_token_ids=valid_sampled_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=[],
            **extra_args,
        )

        durations = ProfileExecuteDuration().pop_captured_sync()
        if durations:
            dr_str = [
                f"[{tag}]:{duration:.2f}ms"
                for tag, duration in durations.items()
            ]
            captured_name = "Decode" if self.attn_state == AscendAttentionState.DecodeOnly else "Prefill"
            logger.info("Profile execute duration [%s]:%s", captured_name,
                        " ".join(dr_str))
        if self.dynamic_eplb:
            self.eplb_updator.forward_end()
        if not self.use_async_scheduling:
            return model_runner_output

        return AsyncNPUModelRunnerOutput(
            model_runner_output=model_runner_output,
            sampled_token_ids=sampled_token_ids,
            invalid_req_indices=invalid_req_indices,
            async_output_copy_stream=self.async_output_copy_stream,
        )
    
    def maybe_execute_ucm_sparse_begin(self, scheduler_output: "SchedulerOutput", attn_metadata: AscendCommonAttentionMetadata):
        if not has_ucm_sparse():
            return
        if has_kv_transfer_group():
            uc_connector = get_kv_transfer_group()
            uc_setup_model = getattr(uc_connector, "setup_model", None)
            if callable(uc_setup_model):
                uc_setup_model(self.model)
        ucm_sparse = get_ucm_sparse()
        ucm_sparse.build_sparse_meta(scheduler_output, self.requests, self.input_batch, attn_metadata)
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

    def ucm_sparse_update_states(self, scheduler_output: "SchedulerOutput"):
            if not has_ucm_sparse():
                return
            ucm_sparse = get_ucm_sparse()
            ucm_sparse.update_states(scheduler_output)

    def initialize_kv_cache_tensors_deepseek_mla(
            self, kv_cache_config: KVCacheConfig) -> dict[str, torch.Tensor]:
        kv_cache_sizes = {}
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            assert len(kv_cache_tensor.shared_by) == 1, (
                "KV cache tensor shared by multiple layers is not supported in "
                "NPU.")
            kv_cache_sizes[kv_cache_tensor.shared_by[0]] = kv_cache_tensor.size

        kv_caches: Dict[str, torch.Tensor] = {}
        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            attn_backend = group.backend
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                tensor_size = kv_cache_sizes[layer_name]
                num_blocks = tensor_size // kv_cache_spec.page_size_bytes
                if self.vllm_config.additional_config.get(
                        "kv_cache_dtype", None) == 'int8':
                    kv_cache_shape = attn_backend.get_bsh_kv_cache_shape(
                        num_blocks, kv_cache_spec.block_size,
                        kv_cache_spec.num_kv_heads, kv_cache_spec.head_size)
                elif hasattr(attn_backend, "get_supported_block_size"
                             ) and not self.model_config.is_deepseek_mla:
                    block_size = attn_backend.get_supported_block_size()[0]
                    block_size_chunk = kv_cache_spec.block_size // block_size
                    kv_cache_shape = attn_backend.get_kv_cache_shape(
                        num_blocks * block_size_chunk, block_size,
                        kv_cache_spec.num_kv_heads, kv_cache_spec.head_size)
                else:
                    kv_cache_shape = self.attn_backend.get_kv_cache_shape(
                        num_blocks, kv_cache_spec.block_size,
                        kv_cache_spec.num_kv_heads, kv_cache_spec.head_size)
                dtype = kv_cache_spec.dtype

                alignment = 2 * 1024 * 1024
                num_blocks, block_size, num_kv_heads, head_size = kv_cache_shape
                rope_dim = self.model_config.hf_text_config.qk_rope_head_dim
                nope_dim = head_size - rope_dim
                nope_cache_shape = (num_blocks, block_size, num_kv_heads,
                                    nope_dim)
                rope_cache_shape = (num_blocks, block_size, num_kv_heads,
                                    rope_dim)
                if self.vllm_config.kv_transfer_config is None:
                    # For no disaggregate pd scenario, allocate kv cache in normal way
                    rope_cache = torch.zeros(rope_cache_shape,
                                             dtype=dtype,
                                             device=self.device)
                    nope_cache = torch.zeros(nope_cache_shape,
                                             dtype=dtype,
                                             device=self.device)
                    rope_cache = self._convert_torch_format(rope_cache)
                    nope_cache = self._convert_torch_format(nope_cache)
                else:

                    # In order to transfer kv cache through the reigster_memory api from llmdatadist, the memory
                    # address should be aligned by 2M. In most case, torch_npu can allocate 2M aligned memory, but
                    # we found there are also some exceptions during test, so we manual align those memory here, this part
                    # of code may consume 2M * 2 * elem_size memory every layer.
                    nope_allocate_shape = num_blocks * block_size * num_kv_heads * nope_dim
                    nope_allocate_shape_alignment = nope_allocate_shape + alignment
                    rope_allocate_shape = num_blocks * block_size * num_kv_heads * rope_dim
                    rope_allocate_shape_alignment = rope_allocate_shape + alignment

                    nope_cache = torch.zeros(nope_allocate_shape_alignment,
                                             dtype=dtype,
                                             device=self.device)
                    rope_cache = torch.zeros(rope_allocate_shape_alignment,
                                             dtype=dtype,
                                             device=self.device)
                    nope_cache = self._align_memory(
                        nope_cache,
                        alignment)[:nope_allocate_shape].view(nope_cache_shape)
                    rope_cache = self._align_memory(
                        rope_cache,
                        alignment)[:rope_allocate_shape].view(rope_cache_shape)
                kv_caches[layer_name] = (nope_cache, rope_cache)

        if has_ucm_sparse() and os.getenv("VLLM_HASH_ATTENTION", "0") == "1":
            ucm_sparse = get_ucm_sparse()
            ucm_sparse.initialize_kv_hash_cache_tensors_npu(kv_caches, self.device)

        bind_kv_cache(kv_caches,
                      self.compilation_config.static_forward_context,
                      self.kv_caches)

        return kv_caches

    def initialize_kv_cache_tensors(
            self, kv_cache_config: KVCacheConfig) -> dict[str, torch.Tensor]:
        """
        Initialize the memory buffer for KV cache.

        Args:
            kv_cache_config: The KV cache config
        Returns:
            Dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        # init kv cache tensors
        kv_cache_raw_tensors: dict[str, Union[torch.Tensor,
                                              Optional[torch.Tensor]]] = {}
        # llmdatadist need the addr of cache tensor be aligned with 2M
        alignment = 2 * 1024 * 1024
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            # TODO: REFACTOR ME to sharing hybrid cache
            for idx in range(len(kv_cache_tensor.shared_by)):
                layer_name = kv_cache_tensor.shared_by[idx]
                if "linear_attn" in layer_name:
                    # for mamba linear attention
                    for layer_name_inner in kv_cache_tensor.shared_by:
                        if ("attn" in layer_name_inner and "linear_attn" not in layer_name_inner) or \
                            layer_name_inner in kv_cache_raw_tensors.keys():
                            continue
                        if self.vllm_config.kv_transfer_config is None:
                            tensor = torch.zeros(kv_cache_tensor.size,
                                                 dtype=torch.int8,
                                                 device=self.device)
                        else:
                            cache_size_aligned = kv_cache_tensor.size + alignment
                            tensor = torch.zeros(cache_size_aligned,
                                                 dtype=torch.int8,
                                                 device=self.device)
                            tensor = self._align_memory(
                                tensor, alignment)[:kv_cache_tensor.size]
                        kv_cache_raw_tensors[layer_name_inner] = tensor
                elif "attn" in layer_name:
                    # for other attentions, e.g., self_attn, sliding window attn
                    if self.vllm_config.kv_transfer_config is None:
                        k_tensor = torch.zeros(kv_cache_tensor.size // 2,
                                               dtype=torch.int8,
                                               device=self.device)
                        v_tensor = torch.zeros(kv_cache_tensor.size // 2,
                                               dtype=torch.int8,
                                               device=self.device)
                    else:
                        cache_size = kv_cache_tensor.size // 2
                        cache_size_aligned = kv_cache_tensor.size // 2 + alignment
                        k_tensor = torch.zeros(cache_size_aligned,
                                               dtype=torch.int8,
                                               device=self.device)
                        v_tensor = torch.zeros(cache_size_aligned,
                                               dtype=torch.int8,
                                               device=self.device)
                        k_tensor = self._align_memory(k_tensor,
                                                      alignment)[:cache_size]
                        v_tensor = self._align_memory(v_tensor,
                                                      alignment)[:cache_size]
                    kv_cache_raw_tensors[layer_name] = (k_tensor, v_tensor)

        layer_names = set()
        for group in kv_cache_config.kv_cache_groups:
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                layer_names.add(layer_name)
        assert layer_names == set(kv_cache_raw_tensors.keys(
        )), "Some layers are not correctly initialized"

        kv_caches: Dict[str, torch.Tensor] = {}
        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            attn_backend = group.backend
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue

                # TODO: remove this after the OOM issue is located and fixed, otherwise, some model may
                # encounter OOM issue
                if isinstance(kv_cache_spec, FullAttentionSpec):
                    raw_k_tensor, raw_v_tensor = kv_cache_raw_tensors[  # type: ignore
                        layer_name]
                    assert raw_k_tensor is not None
                    assert raw_v_tensor is not None
                    assert (raw_k_tensor.numel() + raw_v_tensor.numel()
                            ) % kv_cache_spec.page_size_bytes == 0
                    num_blocks = (raw_k_tensor.numel() + raw_v_tensor.numel()
                                  ) // kv_cache_spec.page_size_bytes

                    # `num_blocks` is the number of blocks the model runner can use.
                    # `kv_cache_config.num_blocks` is the number of blocks that
                    # KVCacheManager may allocate.
                    # Since different GPUs may have different number of layers and
                    # different memory capacities, `num_blocks` can be different on
                    # different GPUs, and `kv_cache_config.num_blocks` is set to
                    # the min of all `num_blocks`. Verify it here.
                    assert num_blocks >= kv_cache_config.num_blocks

                    if self.vllm_config.additional_config.get(
                            "kv_cache_dtype", None) == 'int8':
                        kv_cache_shape = attn_backend.get_bsh_kv_cache_shape(
                            num_blocks, kv_cache_spec.block_size,
                            kv_cache_spec.num_kv_heads,
                            kv_cache_spec.head_size)
                    elif hasattr(attn_backend, "get_supported_block_size"
                                 ) and self.use_hybrid_blocks:
                        block_size = attn_backend.get_supported_block_size()[0]

                        block_size_chunk = kv_cache_spec.block_size // block_size
                        kv_cache_shape = attn_backend.get_kv_cache_shape(
                            num_blocks * block_size_chunk, block_size,
                            kv_cache_spec.num_kv_heads,
                            kv_cache_spec.head_size)
                    else:
                        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
                            num_blocks, kv_cache_spec.block_size,
                            kv_cache_spec.num_kv_heads,
                            kv_cache_spec.head_size)
                    dtype = kv_cache_spec.dtype
                    k_cache = raw_k_tensor.view(dtype).view(kv_cache_shape[1:])
                    k_cache = self._convert_torch_format(k_cache)
                    v_cache = raw_v_tensor.view(dtype).view(kv_cache_shape[1:])
                    v_cache = self._convert_torch_format(v_cache)
                    kv_caches[layer_name] = (k_cache, v_cache)
                elif isinstance(kv_cache_spec, MambaSpec):
                    raw_tensor = kv_cache_raw_tensors[layer_name]
                    assert raw_tensor is not None
                    assert raw_tensor.numel(
                    ) % kv_cache_spec.page_size_bytes == 0
                    num_blocks = raw_tensor.numel(
                    ) // kv_cache_spec.page_size_bytes

                    # `num_blocks` is the number of blocks the model runner can use.
                    # `kv_cache_config.num_blocks` is the number of blocks that
                    # KVCacheManager may allocate.
                    # Since different GPUs may have different number of layers and
                    # different memory capacities, `num_blocks` can be different on
                    # different GPUs, and `kv_cache_config.num_blocks` is set to
                    # the min of all `num_blocks`. Verify it here.
                    assert num_blocks >= kv_cache_config.num_blocks

                    state_tensors = []
                    storage_offset_bytes = 0
                    for (shape, dtype) in zip(kv_cache_spec.shapes,
                                              kv_cache_spec.dtypes):
                        dtype_size = get_dtype_size(dtype)
                        num_element_per_page = (
                            kv_cache_spec.page_size_bytes // dtype_size)
                        target_shape = (num_blocks, *shape)
                        stride = torch.empty(target_shape).stride()
                        target_stride = (num_element_per_page, *stride[1:])
                        assert storage_offset_bytes % dtype_size == 0
                        tensor = torch.as_strided(
                            raw_tensor.view(dtype),
                            size=target_shape,
                            stride=target_stride,
                            storage_offset=storage_offset_bytes // dtype_size,
                        )
                        state_tensors.append(tensor)
                        storage_offset_bytes += stride[0] * dtype_size
                    kv_caches[layer_name] = state_tensors
                else:
                    raise ValueError("Unknown KV cache spec type.")

        if has_ucm_sparse() and os.getenv("VLLM_HASH_ATTENTION", "0") == "1":
            ucm_sparse = get_ucm_sparse()
            ucm_sparse.initialize_kv_hash_cache_tensors_npu(kv_caches, self.device)

        bind_kv_cache(kv_caches,
                      self.compilation_config.static_forward_context,
                      self.kv_caches)

        return kv_caches