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

from ucm.logger import init_logger

logger = init_logger(__name__)


def _apply_ascend_patch() -> None:
    """Apply patches for vLLM-Ascend 0.9.1."""
    _patch_attention_v1()
    _patch_mla_v1()
    _patch_model_runner_v1()
    _patch_worker_v1()


# ========================= vllm_ascend/attention/attention_v1.py =========================
def _patch_attention_v1() -> None:
    """Patch attention_v1.py for vLLM-Ascend0.9.1."""
    try:
        from typing import List

        import torch
        from vllm.distributed.kv_transfer import (
            get_kv_transfer_group,
            has_kv_transfer_group,
            is_v1_kv_transfer_group,
        )
        from vllm.forward_context import ForwardContext, get_forward_context
        from vllm_ascend.attention import attention_v1

        def wait_for_kv_layer_from_connector(layer_name: str):
            if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
                return

            connector = get_kv_transfer_group()

            forward_context: ForwardContext = get_forward_context()
            attn_metadata = forward_context.attn_metadata
            if attn_metadata is None:
                return
            connector.wait_for_layer_load(layer_name)

        def maybe_save_kv_layer_to_connector(
            layer_name: str,
            kv_cache_layer: List[torch.Tensor],
        ):
            if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
                return

            connector = get_kv_transfer_group()

            forward_context: ForwardContext = get_forward_context()
            attn_metadata = forward_context.attn_metadata
            if attn_metadata is None:
                return
            connector.save_kv_layer(layer_name, kv_cache_layer, attn_metadata)

        vllm_ops = torch.ops.vllm
        orig_unified_ascend_attention_with_output = (
            vllm_ops.unified_ascend_attention_with_output
        )

        class _UnifiedAscendWrapper:
            def __init__(self, orig_op):
                self._orig_op = orig_op

            def __call__(
                self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                output: torch.Tensor,
                layer_name: str,
            ) -> None:
                return unified_ascend_attention_with_output_impl(
                    query, key, value, output, layer_name
                )

            def __getattr__(self, name):
                return getattr(self._orig_op, name)

        vllm_ops.unified_ascend_attention_with_output = _UnifiedAscendWrapper(
            orig_unified_ascend_attention_with_output
        )

        def unified_ascend_attention_with_output_impl(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            output: torch.Tensor,
            layer_name: str,
        ) -> None:

            wait_for_kv_layer_from_connector(layer_name)

            forward_context: ForwardContext = get_forward_context()
            attn_metadata = forward_context.attn_metadata
            self = forward_context.no_compile_layers[layer_name]
            kv_cache = self.kv_cache[forward_context.virtual_engine]

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

            maybe_save_kv_layer_to_connector(layer_name, kv_cache)

            return

        attention_v1.wait_for_kv_layer_from_connector = wait_for_kv_layer_from_connector
        attention_v1.maybe_save_kv_layer_to_connector = maybe_save_kv_layer_to_connector
        attention_v1.unified_ascend_attention_with_output = (
            unified_ascend_attention_with_output_impl
        )

    except Exception as e:
        logger.error(f"Failed to patch attention_v1.py: {e}", exc_info=True)
        raise


# ========================= vllm_ascend/worker/model_runner_v1.py =========================
def _patch_model_runner_v1() -> None:
    """Patch model_runner_v1.py for vLLM-Ascend0.9.1."""
    try:
        from typing import TYPE_CHECKING, Optional, Union

        import numpy as np
        import torch

        # from vllm.logger import logger
        from vllm.sequence import IntermediateTensors
        from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
        from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
        from vllm_ascend.ascend_config import get_ascend_config
        from vllm_ascend.attention.attention_v1 import (
            AscendAttentionState,
        )
        from vllm_ascend.utils import (
            ProfileExecuteDuration,
        )

        if TYPE_CHECKING:
            from vllm.v1.core.sched.output import SchedulerOutput
        import torch.nn as nn
        import vllm_ascend.envs as envs_ascend
        from vllm.distributed.kv_transfer import (
            get_kv_transfer_group,
            has_kv_transfer_group,
        )
        from vllm_ascend.ascend_forward_context import set_ascend_forward_context
        from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
        from vllm_ascend.distributed.utils import is_lmhead_tp
        from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

        def _process_reqs(
            self,
            scheduler_output: "SchedulerOutput",
            intermediate_tensors: Optional[IntermediateTensors] = None,
        ) -> tuple[
            SpecDecodeMetadata,
            torch.Tensor,
            SpecDecodeMetadata,
            torch.Tensor,
            int,
            torch.Tensor,
            Optional[set[str]],
            Optional[set[str]],
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
            sample_indices = cu_num_tokens - 1
            sample_indices = torch.from_numpy(sample_indices).to(
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

            self.positions_cpu[total_num_scheduled_tokens:num_input_tokens].zero_()
            self.positions[:num_input_tokens].copy_(
                self.positions_cpu[:num_input_tokens], non_blocking=True
            )
            positions_cpu = self.positions_cpu[:num_input_tokens]
            positions = self.positions[:num_input_tokens]
            self.query_lens = torch.from_numpy(num_scheduled_tokens)

            self.seq_lens_np[:num_reqs] = (
                self.input_batch.num_computed_tokens_cpu[:num_reqs]
                + num_scheduled_tokens
            )
            seq_lens_cpu = self.seq_lens_cpu[:num_reqs]

            block_table_indices = (
                req_indices * self.max_num_blocks_per_req
                + positions_np // self.block_size
            )

            block_table_cpu = self.input_batch.block_table[0].get_cpu_tensor()
            block_numbers = block_table_cpu.flatten()[block_table_indices].numpy()
            block_offsets = positions_np % self.block_size
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
                if (
                    self.speculative_config
                    and self.speculative_config.method == "deepseek_mtp"
                ):
                    # support deepseek mtp spec decode in disaggregated-prefill scenario
                    attn_state = AscendAttentionState.SpecDecoding
            # Speculative decoding.
            elif np.all(num_valid_tokens == 1):
                attn_state = AscendAttentionState.SpecDecoding
            # splitfuse
            elif (
                not ascend_config.ascend_scheduler_config.enabled
                or self.chunked_prefill_enabled
            ):
                attn_state = AscendAttentionState.ChunkedPrefill
            else:
                attn_state = AscendAttentionState.PrefillCacheHit

            # NOTE: when use ring_mla, attn_mask don't need to generate here.
            if not self.vllm_config.model_config.use_mla:
                attn_mask = self._make_attention_mask(
                    seq_lens=seq_lens_cpu, position=positions_cpu, attn_state=attn_state
                )
                self.attn_mask = attn_mask
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
            self.slot_mapping[:total_num_scheduled_tokens].copy_(
                self.slot_mapping_cpu[:total_num_scheduled_tokens], non_blocking=True
            )

            # Fill unused with -1. Needed for reshape_and_cache
            self.slot_mapping[total_num_scheduled_tokens:].fill_(-1)
            self.seq_lens[num_reqs:].fill_(0)
            self.query_start_loc[num_reqs + 1 :].fill_(-1)

            # Use host tensor, other wise error: tensor.hostData is null
            self.seq_lens_list = self.seq_lens_np.tolist()[:num_input_tokens]
            with_prefill = attn_state not in [
                AscendAttentionState.DecodeOnly,
                AscendAttentionState.SpecDecoding,
            ]

            is_only_prefill = bool(np.all(num_valid_tokens != 1))

            enable_dbo = self._check_dbo_is_valid(
                self.query_lens.tolist(), attn_state, total_num_scheduled_tokens
            )

            maybe_padded_num_tokens = total_num_scheduled_tokens
            if self.torchair_graph_enabled and not with_prefill:
                maybe_padded_num_tokens = self.select_torchair_padded_batch_size(
                    total_num_scheduled_tokens
                )
            (
                padded_num_tokens_across_dp,
                num_tokens_across_dp,
                with_prefill,
                enable_dbo,
            ) = self._get_forward_metadata_across_dp(
                maybe_padded_num_tokens,
                total_num_scheduled_tokens,
                with_prefill,
                enable_dbo,
            )

            common_attn_metadata = AscendCommonAttentionMetadata(
                query_start_loc=self.query_start_loc[: num_reqs + 1],
                query_start_loc_cpu=self.query_start_loc_cpu[: num_reqs + 1],
                seq_lens=self.seq_lens[:num_reqs],
                seq_lens_cpu=self.seq_lens_cpu[:num_reqs],
                num_reqs=num_reqs,
                num_actual_tokens=total_num_scheduled_tokens,
                max_query_len=max_num_scheduled_tokens,
                actual_seq_lengths_q=self.actual_seq_lengths_q,
                block_table_tensor=self.input_batch.block_table[0].get_device_tensor(),
                slot_mapping_cpu=self.slot_mapping_cpu[:total_num_scheduled_tokens],
                positions=self.positions[:num_input_tokens],
                attn_mask=self.attn_mask,
                spec_attn_mask=self.spec_attn_mask,
                attn_state=self.attn_state,  # type: ignore
                decode_token_per_req=self.decode_token_per_req,
                max_num_blocks_per_req=self.max_num_blocks_per_req,
                enable_dbo_across_dp=enable_dbo,
                is_only_prefill=is_only_prefill,
            )

            # TODO(zzzzwwjj): this code need to refactor afterwards.
            self.with_prefill = with_prefill
            # Add num_token_pad_size and num_reqs_pad_size here for torchair graph mode
            if self.torchair_graph_enabled and not with_prefill:
                num_token_pad_size = (
                    padded_num_tokens_across_dp - total_num_scheduled_tokens
                )
                num_reqs_pad_size = (
                    padded_num_tokens_across_dp // self.decode_token_per_req - num_reqs
                )
                assert num_token_pad_size >= 0 and num_reqs_pad_size >= 0

                extra_builder_kwargs["num_token_pad_size"] = num_token_pad_size
                extra_builder_kwargs["num_reqs_pad_size"] = num_reqs_pad_size
                self.num_reqs_pad_size = num_reqs_pad_size
                self.num_token_pad_size = num_token_pad_size
            self.extra_builder_kwargs = extra_builder_kwargs
            self.num_tokens_across_dp = num_tokens_across_dp

            attn_metadata = self.attn_metadata_builder.build(  # type: ignore
                common_attn_metadata=common_attn_metadata,
                **extra_builder_kwargs,
            )
            attn_metadata.num_input_tokens = padded_num_tokens_across_dp

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
                # then the embedding layer is not included in the ACL Graph.
                input_ids = self.input_ids[:num_input_tokens]
                inputs_embeds = None
            if self.uses_mrope:
                positions = self.mrope_positions[:, :num_input_tokens]

            if self.torchair_graph_enabled and not with_prefill:
                input_ids = self.input_ids[:padded_num_tokens_across_dp]
                positions = self.positions[:padded_num_tokens_across_dp]

            # Run forward pass
            finished_dumping = None
            # TODO(zzzzwwjj): check param `num_tokens_across_dp` later.
            with set_ascend_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=padded_num_tokens_across_dp,
                num_tokens_across_dp=num_tokens_across_dp,
                with_prefill=with_prefill,
                num_actual_tokens=total_num_scheduled_tokens,
            ):
                with ProfileExecuteDuration().capture_async("forward"):
                    self.maybe_setup_kv_connector(scheduler_output)
                    model_kwargs = {}
                    if self.torchair_graph_enabled:
                        model_kwargs["kv_caches"] = self.kv_caches
                        model_kwargs["attn_metadata"] = attn_metadata
                    if envs_ascend.VLLM_ASCEND_ENABLE_DBO:
                        if with_prefill:
                            model_kwargs["graph_enable"] = False  # type: ignore
                        else:
                            model_kwargs["graph_enable"] = True  # type: ignore
                    if self.torchair_graph_enabled and not with_prefill:
                        compiled_model = self._get_torchair_lazy_compiled_model(
                            padded_num_tokens_across_dp
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
                        hidden_states = self.model(
                            input_ids=input_ids,
                            positions=positions,
                            intermediate_tensors=intermediate_tensors,
                            inputs_embeds=inputs_embeds,
                            **model_kwargs,
                        )

            finished_dumping = self.maybe_wait_for_kv_save()
            finished_sending, finished_recving = self.get_finished_kv_transfer(
                scheduler_output
            )
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
                sample_indices = spec_decode_metadata.logits_indices

            if is_lmhead_tp():
                if not with_prefill:
                    padded_num_indices = padded_num_tokens_across_dp
                else:
                    padded_num_indices = self.max_num_reqs
                sample_indices = nn.functional.pad(
                    sample_indices, (0, padded_num_indices - sample_indices.shape[0])
                )

            return (
                attn_metadata,
                hidden_states,
                spec_decode_metadata,
                positions,
                total_num_scheduled_tokens,
                sample_indices,
                finished_sending,
                finished_recving,
                finished_dumping,
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
                    if not has_kv_transfer_group():
                        logger.debug(
                            "skip this step for we receive the data from remote disaggregate prefill node"
                        )
                        # Return empty ModelRunnerOuptut if there's no work to do.
                        return EMPTY_MODEL_RUNNER_OUTPUT
                    return self.kv_connector_no_forward(scheduler_output)

                if self.dynamic_eplb:
                    self.eplb_updator.forward_before()

                (
                    attn_metadata,
                    hidden_states,
                    spec_decode_metadata,
                    positions,
                    num_scheduled_tokens,
                    sample_indices,
                    finished_sending,
                    finished_recving,
                    finished_dumping,
                ) = self._process_reqs(scheduler_output, intermediate_tensors)

                if self.dynamic_eplb:
                    self.eplb_updator.take_update_info_from_eplb_process()

            with ProfileExecuteDuration().capture_async("post process"):
                logits = self.model.compute_logits(hidden_states[sample_indices], None)

                # Apply structured output bitmasks if present
                if scheduler_output.grammar_bitmask is not None:
                    logits = self.apply_grammar_bitmask(scheduler_output, logits)

                # Sample the next token and get logprobs if needed.
                sampling_metadata = self.input_batch.sampling_metadata
                if spec_decode_metadata is None:
                    if is_lmhead_tp():
                        logits = logits[: self.input_batch.num_reqs]

                    sampler_output = self.sampler(
                        logits=logits,
                        sampling_metadata=sampling_metadata,
                    )
                else:
                    if is_lmhead_tp():
                        logits = logits[: len(spec_decode_metadata.logits_indices)]

                    # When indexing with a tensor (bonus_logits_indices), PyTorch
                    # creates a new tensor with separate storage from the original
                    # logits tensor. This means any in-place operations on bonus_logits
                    # won't affect the original logits tensor.
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

                spec_token_ids = self._get_spec_token_ids(
                    valid_sampled_token_ids,
                    sampling_metadata,
                    scheduler_output,
                    spec_decode_metadata,
                    positions,
                    num_scheduled_tokens,
                    hidden_states,
                    attn_metadata,
                )
                if has_kv_transfer_group():
                    get_kv_transfer_group().clear_connector_metadata()

                model_runner_output = ModelRunnerOutput(
                    req_ids=self.input_batch.req_ids,
                    req_id_to_index=self.input_batch.req_id_to_index,
                    sampled_token_ids=valid_sampled_token_ids,
                    spec_token_ids=spec_token_ids,
                    logprobs=logprobs_lists,
                    prompt_logprobs_dict={},
                    finished_sending=finished_sending,
                    finished_recving=finished_recving,
                    finished_dumping=finished_dumping,
                    invalid_block_ids=set(),
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

            if self.dynamic_eplb:
                self.eplb_updator.forward_end()

            return model_runner_output

        NPUModelRunner.execute_model = execute_model

        @staticmethod
        def maybe_wait_for_kv_save() -> None:
            if has_kv_transfer_group():
                return get_kv_transfer_group().wait_for_save()

        NPUModelRunner.maybe_wait_for_kv_save = maybe_wait_for_kv_save

    except ImportError as e:
        logger.error(f"Failed to patch model_runner_v1.py: {e}", exc_info=True)
        raise


# ========================= vllm_ascend/attention/mla_v1.py =========================
def _patch_mla_v1() -> None:
    try:
        from dataclasses import dataclass
        from typing import TYPE_CHECKING, Any, Optional, Tuple, Type, TypeVar

        import numpy as np
        import torch
        import torch_npu
        from vllm.attention.backends.abstract import (
            AttentionBackend,
            AttentionLayer,
            AttentionMetadata,
            MLAAttentionImpl,
        )
        from vllm.attention.backends.utils import PAD_SLOT_ID
        from vllm.config import VllmConfig, get_current_vllm_config
        from vllm.model_executor.layers.linear import (
            LinearBase,
            UnquantizedLinearMethod,
        )
        from vllm.utils import cdiv, round_down
        from vllm_ascend import envs
        from vllm_ascend.ascend_config import get_ascend_config
        from vllm_ascend.attention.attention_v1 import (
            AscendAttentionState,
            maybe_save_kv_layer_to_connector,
            wait_for_kv_layer_from_connector,
        )
        from vllm_ascend.attention.utils import (
            AscendCommonAttentionMetadata,
            split_decodes_and_prefills,
        )
        from vllm_ascend.multistream.base import MSAttentionMetadataSplitConfig
        from vllm_ascend.multistream.context import get_multistream_comm_context
        from vllm_ascend.multistream.ms_split import model_input_split_v1_mla_attn
        from vllm_ascend.utils import npu_prefetch, npu_stream_switch, npu_wait_tensor

        if TYPE_CHECKING:
            from vllm.v1.core.sched.output import SchedulerOutput
            from vllm.v1.worker.gpu_input_batch import InputBatch

        from vllm.forward_context import ForwardContext, get_forward_context
        from vllm_ascend.attention.mla_v1 import AscendMLAImpl

        def forward(
            self,
            layer: AttentionLayer,
            hidden_states_or_q_c: torch.Tensor,  # query in unified attn
            hidden_states_or_kv_c_normed: torch.Tensor,  # key in unified attn
            k_pe: torch.Tensor,  # value in unified attn
            kv_cache: Tuple[torch.Tensor],
            attn_metadata: M,
            output: Optional[torch.Tensor] = None,
            enable_multistream_mla=False,
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
                decode_hs_or_q_c = hidden_states_or_q_c[:num_decode_tokens]
                prefill_hs = hidden_states_or_kv_c_normed[num_decode_tokens:]
                # if not self.torchair_graph_enabled:
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
                    cos = attn_metadata.decode.cos
                    sin = attn_metadata.decode.sin
                    with npu_stream_switch(
                        "mla_secondary", 0, enabled=enable_multistream_mla
                    ):
                        decode_k_pe, decode_k_nope = self.exec_kv(
                            hidden_states_or_kv_c_normed,
                            cos,
                            sin,
                            kv_cache,
                            attn_metadata.slot_mapping,
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
                    cos = attn_metadata.prefill.cos
                    sin = attn_metadata.prefill.sin

                    prefill_q_pe = self.rope_single(prefill_q_pe, cos, sin)
                    prefill_k_pe, prefill_k_nope = self.exec_kv_prefill(
                        prefill_hs,
                        cos,
                        sin,
                        kv_cache,
                        attn_metadata.slot_mapping[num_decode_tokens:],
                    )

                    kv_c_normed_prefill = prefill_k_nope[:num_actual_toks, ...]
                    prefill_k_c_normed = prefill_k_nope
                    prefill_k_pe = prefill_k_pe.view(num_tokens, self.num_kv_heads, -1)
                    prefill_q = torch.cat([prefill_q_nope, prefill_q_pe], dim=-1)
                else:
                    prefill_q_pe[...], prefill_k_pe[...] = self.rotary_emb(
                        attn_metadata.prefill.input_positions,
                        prefill_q_pe.contiguous(),
                        prefill_k_pe,
                    )

            assert (
                len(kv_cache) > 1
            ), "the number of kv cache should be greater than 1, namely (nope_cache and rope_cache)"
            if self.torchair_graph_enabled:
                if (
                    kv_cache[0].numel() > 0
                    and attn_metadata.attn_state == AscendAttentionState.PrefillNoCache
                ):
                    slots = attn_metadata.slot_mapping
                    # NOTE: Separate the kv cache in advance to avoid OOM or other issues
                    torch_npu._npu_reshape_and_cache(
                        key=kv_c_normed_prefill.view(num_tokens, self.num_kv_heads, -1),
                        value=prefill_k_pe,
                        key_cache=kv_cache[0],
                        value_cache=kv_cache[1],
                        slot_indices=slots,
                    )

                if (
                    kv_cache[0].numel() > 0
                    and attn_metadata.attn_state == AscendAttentionState.ChunkedPrefill
                    and has_decode
                ):
                    slots = attn_metadata.slot_mapping[:num_decode_tokens]
                    k_c_normed_decode = kv_c_normed[:num_decode_tokens]
                    torch_npu._npu_reshape_and_cache(
                        key=k_c_normed_decode.view(
                            num_decode_tokens, self.num_kv_heads, -1
                        ),
                        value=decode_k_pe,
                        key_cache=kv_cache[0],
                        value_cache=kv_cache[1],
                        slot_indices=slots,
                    )
            else:
                kv_c_normed = kv_c_normed.view([num_actual_toks, self.num_kv_heads, -1])
                torch_npu._npu_reshape_and_cache(
                    key=kv_c_normed,
                    value=k_pe,
                    key_cache=kv_cache[0],
                    value_cache=kv_cache[1],
                    slot_indices=attn_metadata.slot_mapping,
                )
            if not self.running_in_graph:
                o_proj_input_shape = (num_actual_toks, self.num_heads * self.v_head_dim)
                o_proj_input = torch.empty(
                    o_proj_input_shape,
                    dtype=hidden_states_or_q_c.dtype,
                    device=hidden_states_or_q_c.device,
                )
            if has_prefill:
                # FIX: aicore move should be also placed on the comm stream in dbo,
                # otherwise it may affect the accuracy
                # TODO: use an elegant way to overlap
                wait_for_kv_layer_from_connector(layer.layer_name)
                output_prefill = self._forward_prefill(
                    prefill_q, prefill_k_c_normed, prefill_k_pe, kv_cache, attn_metadata
                )
                current_ms_metadata = get_multistream_comm_context()
                if current_ms_metadata is not None:
                    current_ms_metadata.before_comm_event.record()
                    with torch.npu.stream(current_ms_metadata.comm_stream):
                        current_ms_metadata.before_comm_event.wait()
                        o_proj_input[num_decode_tokens:] = output_prefill
                else:
                    o_proj_input[num_decode_tokens:] = output_prefill
                maybe_save_kv_layer_to_connector(layer.layer_name, kv_cache)

            if has_decode:
                wait_for_kv_layer_from_connector(layer.layer_name)
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
                        o_proj_input[:num_decode_tokens] = output_decode
                else:
                    o_proj_input[:num_decode_tokens] = output_decode
                maybe_save_kv_layer_to_connector(layer.layer_name, kv_cache)

            current_ms_metadata = get_multistream_comm_context()
            MAX_O_PROJ_PREFETCH_SIZE = 16 * 1024 * 1024
            if current_ms_metadata is None:
                npu_prefetch(
                    self.o_proj.weight,
                    o_proj_input,
                    max_size=MAX_O_PROJ_PREFETCH_SIZE,
                    enabled=enable_multistream_mla,
                )
                output[...] = self.o_proj(o_proj_input)[0]
            else:
                with torch.npu.stream(current_ms_metadata.comm_stream):
                    npu_prefetch(
                        self.o_proj.weight,
                        o_proj_input,
                        max_size=MAX_O_PROJ_PREFETCH_SIZE,
                        enabled=enable_multistream_mla,
                    )
                    output[...] = self.o_proj(o_proj_input)[0]
                    current_ms_metadata.after_comm_event.record()
            del o_proj_input
            return output_padded

        AscendMLAImpl.forward = forward

    except ImportError as e:
        logger.error(f"Failed to patch mla_v1.py: {e}", exc_info=True)
        raise


# ========================= vllm_ascend/worker/worker_v1.py =========================
def _patch_worker_v1() -> None:
    try:
        import copy
        from typing import Optional

        from vllm.distributed.kv_transfer import (
            ensure_kv_transfer_initialized,
            has_kv_transfer_group,
        )
        from vllm.distributed.parallel_state import get_pp_group, get_tp_group

        # from vllm.logger import logger
        from vllm.sequence import IntermediateTensors
        from vllm.v1.core.sched.output import SchedulerOutput
        from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
        from vllm_ascend.worker.worker_v1 import NPUWorker

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
                if not has_kv_transfer_group():
                    return None

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

    except ImportError as e:
        logger.error(f"Failed to patch worker_v1.py: {e}", exc_info=True)
        raise
