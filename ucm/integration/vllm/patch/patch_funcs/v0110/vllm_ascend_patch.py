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


"""Monkey patches for vLLM Ascend 0.11.0 (UCM)."""

from __future__ import annotations


from ucm.logger import init_logger

logger = init_logger(__name__)


def _apply_ascend_patches() -> None:
    """Apply all patches for vLLM 0.11.0 on Ascend."""
    _patch_ascend_attention_layer()
    _patch_ascend_npu_model_runner()
    _patch_ascend_npu_worker()
    logger.info("vLLM Ascend 0.11.0 patches applied successfully")



def _patch_ascend_attention_layer() -> None:
    """Patch ascend attention layer"""
    try:
        # vLLM-Ascend uses its own attention implementation; patch its module
        # rather than `vllm.attention.layer`.
        from typing import List

        import torch
        from vllm_ascend.attention import utils as va_utils_mod

        if getattr(va_utils_mod, "__ucm_load_failure_ascend_patched__", False):
            return

        # TODO(ucm): Implement load-failure recovery hooks for vLLM-Ascend 0.11.0.
        # For now, keep as a no-op placeholder so that Ascend 0.11.0 can enable the
        # patch pipeline without breaking imports.
        def patched_wait_for_kv_layer_from_connector(layer_name: str):
            from vllm.distributed.kv_transfer import (
                get_kv_transfer_group,
                has_kv_transfer_group,
                is_v1_kv_transfer_group,
            )
            from vllm.forward_context import ForwardContext, get_forward_context

            if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
                return

            connector = get_kv_transfer_group()
            if (
                hasattr(connector, "has_connector_metadata")
                and not connector.has_connector_metadata()
            ):
                return

            forward_context: ForwardContext = get_forward_context()
            attn_metadata = forward_context.attn_metadata
            if attn_metadata is None:
                return
            # TODO: assert ascendMetadata
            connector.wait_for_layer_load(layer_name)

        def patched_maybe_save_kv_layer_to_connector(
            layer_name: str,
            kv_cache_layer: List[torch.Tensor],
        ):
            from vllm.distributed.kv_transfer import (
                get_kv_transfer_group,
                has_kv_transfer_group,
                is_v1_kv_transfer_group,
            )
            from vllm.forward_context import ForwardContext, get_forward_context

            if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
                return

            connector = get_kv_transfer_group()
            if (
                hasattr(connector, "has_connector_metadata")
                and not connector.has_connector_metadata()
            ):
                return

            forward_context: ForwardContext = get_forward_context()
            attn_metadata = forward_context.attn_metadata
            if attn_metadata is None:
                return
            # TODO: assert ascendMetadata
            connector.save_kv_layer(layer_name, kv_cache_layer, attn_metadata)

        va_utils_mod.wait_for_kv_layer_from_connector = (
            patched_wait_for_kv_layer_from_connector
        )
        va_utils_mod.maybe_save_kv_layer_to_connector = (
            patched_maybe_save_kv_layer_to_connector
        )
        va_utils_mod.__ucm_load_failure_ascend_patched__ = True  # type: ignore[attr-defined]
    except ImportError as e:
        logger.warning(
            "Could not patch ascend attention layer (vllm_ascend missing): %s", e
        )
    except Exception as e:
        logger.warning("Could not patch ascend attention layer: %s", e)

def _patch_ascend_npu_model_runner() -> None:
    """Patch _update_states method in NPUModelRunner."""
    try:
        from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

        if getattr(NPUModelRunner, "__ucm_load_failure_patched__", False):
            return
        import torch

        @torch.inference_mode()
        def patched_execute_model(
            self,
            scheduler_output: "SchedulerOutput",
            intermediate_tensors: Optional[IntermediateTensors] = None,
        ) -> Union[ModelRunnerOutput, AsyncModelRunnerOutput, IntermediateTensors]:
            from vllm.distributed.kv_transfer import (
                get_kv_transfer_group,
                has_kv_transfer_group,
            )
            from vllm.distributed.parallel_state import get_pp_group, get_tp_group
            from vllm.forward_context import BatchDescriptor
            from vllm.sequence import IntermediateTensors
            from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
            from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorOutput
            from vllm_ascend.ascend_forward_context import set_ascend_forward_context
            from vllm_ascend.attention.attention_v1 import AscendAttentionState
            from vllm_ascend.spec_decode.interface import SpecDcodeType
            from vllm_ascend.utils import ProfileExecuteDuration, lmhead_tp_enable
            from vllm_ascend.worker.model_runner_v1 import AsyncNPUModelRunnerOutput

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

                (
                    attn_metadata,
                    positions,
                    num_scheduled_tokens_np,
                    num_input_tokens,
                    num_tokens_across_dp,
                    maybe_padded_num_tokens,
                    logits_indices,
                    spec_decode_metadata,
                    input_ids,
                    inputs_embeds,
                    intermediate_tensors,
                    max_query_len,
                ) = self._prepare_inputs(scheduler_output, intermediate_tensors)

                if self.dynamic_eplb:
                    self.eplb_updator.take_update_info_from_eplb_process()

            moe_comm_type = self._select_moe_comm_method(
                num_input_tokens, self.with_prefill
            )

            uniform_decode = (max_query_len == self.uniform_decode_query_len) and (
                scheduler_output.total_num_scheduled_tokens
                == self.input_batch.num_reqs * max_query_len
            )
            batch_descriptor = BatchDescriptor(
                num_tokens=num_input_tokens, uniform_decode=uniform_decode
            )
            aclgraph_runtime_mode, batch_descriptor = self.aclgraph_dispatcher.dispatch(
                batch_descriptor
            )

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
                    num_actual_tokens=scheduler_output.total_num_scheduled_tokens,
                    prefetch_stream=self.prefetch_stream,
                    model_instance=self.model,
                    weight_prefetch_method=self.weight_prefetch_method,
                ):
                    self.maybe_setup_kv_connector(scheduler_output)

                    hidden_states = self._generate_process_reqs_hidden_states(
                        attn_metadata,
                        self.with_prefill,
                        maybe_padded_num_tokens,
                        input_ids,
                        positions,
                        intermediate_tensors,
                        inputs_embeds,
                    )

                self.maybe_wait_for_kv_save()
                finished_sending, finished_recving = self.get_finished_kv_transfer(
                    scheduler_output
                )
                invalid_block_ids = None
                if has_kv_transfer_group():
                    invalid_block_ids = (
                        get_kv_transfer_group().get_block_ids_with_load_errors()
                    )

                aux_hidden_states = None
                if self.drafter and self.drafter.name == SpecDcodeType.EAGLE3:
                    hidden_states, aux_hidden_states = hidden_states

            kv_connector_output = KVConnectorOutput(
                finished_sending=finished_sending,
                finished_recving=finished_recving,
                invalid_block_ids=invalid_block_ids,
            )
            finished_sending = None
            finished_recving = None
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
                        hidden_states.kv_connector_output = kv_connector_output
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
                            scheduler_output.total_num_scheduled_tokens,
                            num_scheduled_tokens_np,
                            finished_sending,
                            finished_recving,
                            kv_connector_output,
                        )
                    sample_hidden_states = hidden_states[logits_indices]
                    logits = self.model.compute_logits(sample_hidden_states)
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
                    if lmhead_tp_enable() and logits is not None:
                        logits = logits[: self.input_batch.num_reqs]
                    sampler_output = self.sampler(
                        logits=logits,
                        sampling_metadata=sampling_metadata,
                    )
                else:
                    if lmhead_tp_enable() and logits is not None:
                        logits = logits[: len(spec_decode_metadata.logits_indices)]
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
                    if self.need_accepted_tokens:
                        self._update_states_after_model_execute(output_token_ids)

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

                # Copy some objects so they don't get modified after returning.
                # This is important when using async scheduling.
                req_ids_output_copy = self.input_batch.req_ids.copy()
                req_id_to_index_output_copy = self.input_batch.req_id_to_index.copy()

                # NOTE: NPU -> CPU Sync happens here.
                # Move as many CPU operations as possible before this sync point.
                logprobs_tensors = sampler_output.logprobs_tensors
                logprobs_lists = (
                    logprobs_tensors.tolists() if logprobs_tensors is not None else None
                )

                # Compute prompt logprobs if needed.
                prompt_logprobs_dict = self._get_prompt_logprobs_dict(
                    hidden_states[: scheduler_output.total_num_scheduled_tokens],
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
                    self.input_batch.prev_sampled_token_ids = sampled_token_ids
                    self.input_batch.prev_sampled_token_ids_invalid_indices = (
                        invalid_req_indices_set
                    )
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
                        sampled_ids = (
                            [-1] * 1 if req_idx not in invalid_req_indices_set else None
                        )
                    else:
                        sampled_ids = valid_sampled_token_ids[req_idx]
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

            extra_args = {"kv_connector_output": kv_connector_output}

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
            if not self.use_async_scheduling:
                return model_runner_output

            return AsyncNPUModelRunnerOutput(
                model_runner_output=model_runner_output,
                sampled_token_ids=sampled_token_ids,
                invalid_req_indices=invalid_req_indices,
                async_output_copy_stream=self.async_output_copy_stream,
            )

        NPUModelRunner.execute_model = patched_execute_model

        def patched_update_states(self, scheduler_output: "SchedulerOutput") -> None:
            # Remove finished requests from the cached states.
            from typing import cast

            import torch
            from vllm.distributed.parallel_state import get_pp_group
            from vllm.sampling_params import SamplingType
            from vllm.v1.worker.gpu_input_batch import CachedRequestState
            from vllm.v1.worker.gpu_model_runner import VllmModelForPooling

            for req_id in scheduler_output.finished_req_ids:
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

                if (
                    sampling_params
                    and sampling_params.sampling_type == SamplingType.RANDOM_SEED
                ):
                    generator = torch.Generator(device=self.device)
                    generator.manual_seed(sampling_params.seed)
                else:
                    generator = None

                if pooling_params:
                    assert (
                        task := pooling_params.task
                    ) is not None, "You did not set `task` in the API"
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
            for i, req_id in enumerate(req_data.req_ids):
                req_state = self.requests[req_id]
                num_computed_tokens = req_data.num_computed_tokens[i]
                new_block_ids = req_data.new_block_ids[i]
                resumed_from_preemption = req_data.resumed_from_preemption[i]
                num_output_tokens = req_data.num_output_tokens[i]
                req_index = self.input_batch.req_id_to_index.get(req_id)

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
                elif num_output_tokens < len(req_state.output_token_ids):
                    # Some output tokens were discarded due to a sync-KV-load
                    # failure. Align the cached state.
                    del req_state.output_token_ids[num_output_tokens:]
                    if req_index is not None:
                        end_idx = (
                            self.input_batch.num_prompt_tokens[req_index]
                            + num_output_tokens
                        )
                        self.input_batch.num_tokens[req_index] = end_idx
                        self.input_batch.num_tokens_no_spec[req_index] = end_idx

                # Update the block IDs.
                if not resumed_from_preemption:
                    if new_block_ids is not None:
                        # Append the new blocks to the existing block IDs.
                        for block_ids, new_ids in zip(
                            req_state.block_ids, new_block_ids
                        ):
                            block_ids.extend(new_ids)
                else:
                    assert new_block_ids is not None
                    # The request is resumed from preemption.
                    # Replace the existing block IDs with the new ones.
                    req_state.block_ids = new_block_ids

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
                if new_block_ids is not None:
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
                spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(
                    req_id, ()
                )
                if spec_token_ids:
                    num_spec_tokens = len(spec_token_ids)
                    start_index = self.input_batch.num_tokens_no_spec[req_index]
                    end_token_index = start_index + num_spec_tokens
                    self.input_batch.token_ids_cpu[
                        req_index, start_index:end_token_index
                    ] = spec_token_ids
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

        NPUModelRunner._update_states = patched_update_states
        NPUModelRunner.__ucm_load_failure_patched__ = True  # type: ignore[attr-defined]
    except Exception as e:
        logger.warning("Could not patch NPUModelRunner._update_states: %s", e)


def _patch_ascend_npu_worker() -> None:
    """Patch Worker to use kv_connector_output.is_empty()."""
    try:
        from typing import Optional, Union

        from vllm_ascend.worker.worker_v1 import NPUWorker

        if getattr(NPUWorker, "__ucm_load_failure_patched__", False):
            return

        def patched_execute_model(
            self,
            scheduler_output: "SchedulerOutput",
        ) -> Optional[Union[ModelRunnerOutput, AsyncModelRunnerOutput]]:
            # enable msMonitor to monitor the performance of vllm-ascend
            import copy

            import vllm_ascend.envs as envs_ascend
            from torch_npu.profiler import dynamic_profile as dp
            from vllm.distributed.parallel_state import (
                get_pp_group,
                get_tp_group,
            )
            from vllm.sequence import IntermediateTensors
            from vllm.v1.outputs import (
                EMPTY_MODEL_RUNNER_OUTPUT,
                AsyncModelRunnerOutput,
                ModelRunnerOutput,
            )

            if envs_ascend.MSMONITOR_USE_DAEMON:
                dp.step()

            intermediate_tensors = None
            forward_pass = scheduler_output.total_num_scheduled_tokens > 0
            if forward_pass and not get_pp_group().is_first_rank:
                intermediate_tensors = IntermediateTensors(
                    get_pp_group().recv_tensor_dict(all_gather_group=get_tp_group())
                )

            output = self.model_runner.execute_model(
                scheduler_output, intermediate_tensors
            )
            if isinstance(output, (ModelRunnerOutput, AsyncModelRunnerOutput)):
                return output

            assert isinstance(output, IntermediateTensors)
            parallel_config = self.vllm_config.parallel_config
            assert (
                parallel_config.distributed_executor_backend != ("external_launcher")
                and not get_pp_group().is_last_rank
            )

            get_pp_group().send_tensor_dict(
                output.tensors, all_gather_group=get_tp_group()
            )

            kv_connector_output = output.kv_connector_output
            if not kv_connector_output:
                return None

            # In case of PP with kv transfer, we need to pass through the
            # kv_connector_output
            if kv_connector_output.is_empty():
                return EMPTY_MODEL_RUNNER_OUTPUT

            output = copy.copy(EMPTY_MODEL_RUNNER_OUTPUT)
            output.kv_connector_output = kv_connector_output
            return output

        NPUWorker.execute_model = patched_execute_model
        NPUWorker.__ucm_load_failure_patched__ = True  # type: ignore[attr-defined]
    except Exception as e:
        logger.warning("Could not patch NPUWorker: %s", e)