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

"""Load-failure recovery monkey patches for vLLM 0.11.0 (UCM)."""

from __future__ import annotations

import os

from ucm.logger import init_logger

logger = init_logger(__name__)


def _is_ascend() -> bool:
    # Keep consistent with ucm.integration.vllm.patch.apply_patch._patch_ascend()
    return os.getenv("PLATFORM") == "ascend"


def _apply_load_failure_patches() -> None:
    """Apply all load-failure recovery patches for vLLM 0.11.0."""
    _patch_kv_connector_base_v1()
    _patch_kv_connector_output()
    _patch_cached_request_data()
    _patch_kv_output_aggregator()
    _patch_block_pool()
    _patch_single_type_kv_cache_manager()
    _patch_scheduler()
    _patch_gpu_model_runner()
    _patch_gpu_worker()
    _patch_kv_connector_model_runner_mixin()
    _patch_attention_layer()
    if _is_ascend():
        _patch_ascend_attention_layer()
        _patch_ascend_npu_model_runner()
        _patch_ascend_npu_worker()
    logger.info("UCM load-failure patches applied successfully for vLLM 0.11.0")


def _patch_kv_connector_base_v1() -> None:
    """Add get_block_ids_with_load_errors to KVConnectorBase_V1."""
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1 import base as base_mod

        KVConnectorBase_V1 = base_mod.KVConnectorBase_V1
        if hasattr(KVConnectorBase_V1, "get_block_ids_with_load_errors"):
            return

        def get_block_ids_with_load_errors(self) -> set[int]:
            """Get the set of block IDs that failed to load."""
            return set()

        KVConnectorBase_V1.get_block_ids_with_load_errors = (
            get_block_ids_with_load_errors
        )

        # add has_connector_metadata to KVConnectorBase_V1
        def has_connector_metadata(self) -> bool:
            """Check whether the connector metadata is currently set.

            Returns:
                bools: True if connector metadata exists, False otherwise.
            """
            return self._connector_metadata is not None

        KVConnectorBase_V1.has_connector_metadata = has_connector_metadata

    except ImportError as e:
        logger.warning("Could not patch KVConnectorBase_V1: %s", e)


def _patch_kv_connector_output() -> None:
    """Add invalid_block_ids to KVConnectorOutput and update is_empty()."""
    try:
        import functools

        from vllm.v1 import outputs as outputs_mod

        KVConnectorOutput = outputs_mod.KVConnectorOutput
        if getattr(KVConnectorOutput, "__ucm_load_failure_patched__", False):
            return

        # 1. Wrap original __init__ to add invalid_block_ids parameter
        original_init = KVConnectorOutput.__init__

        @functools.wraps(original_init)
        def patched_init(self, *args, invalid_block_ids=None, **kwargs):
            original_init(self, *args, **kwargs)
            self.invalid_block_ids = (
                invalid_block_ids if invalid_block_ids is not None else set()
            )

        KVConnectorOutput.__init__ = patched_init

        # 2. Replace is_empty method to include invalid_block_ids check
        def patched_is_empty(self):
            return (
                not self.finished_sending
                and not self.finished_recving
                and not self.kv_connector_stats
                and not getattr(self, "invalid_block_ids", None)
            )

        KVConnectorOutput.is_empty = patched_is_empty

        # 3. Update __annotations__ for type hints
        if hasattr(KVConnectorOutput, "__annotations__"):
            KVConnectorOutput.__annotations__["invalid_block_ids"] = set[int]

        KVConnectorOutput.__ucm_load_failure_patched__ = True  # type: ignore[attr-defined]
    except Exception as e:
        logger.warning("Could not patch KVConnectorOutput: %s", e)


def _patch_cached_request_data() -> None:
    """Add num_output_tokens to CachedRequestData (keep all original fields)."""
    try:
        from dataclasses import dataclass, field
        from typing import Optional

        from vllm.v1.core.sched import output as output_mod

        CachedRequestData = output_mod.CachedRequestData
        if getattr(CachedRequestData, "__ucm_load_failure_patched__", False):
            return

        @dataclass
        class CachedRequestDataPatched:
            req_ids: list[str]
            resumed_from_preemption: list[bool]
            new_token_ids: list[list[int]]
            new_block_ids: list[Optional[tuple[list[int], ...]]]
            num_computed_tokens: list[int]
            # Default so vLLM's _make_cached_request_data (5 args) can still construct; we overwrite in patched_schedule.
            num_output_tokens: list[int] = field(default_factory=list)

            @property
            def num_reqs(self) -> int:
                return len(self.req_ids)

            @classmethod
            def make_empty(cls) -> "CachedRequestDataPatched":
                return cls(
                    req_ids=[],
                    resumed_from_preemption=[],
                    new_token_ids=[],
                    new_block_ids=[],
                    num_computed_tokens=[],
                    num_output_tokens=[],
                )

        CachedRequestDataPatched.__qualname__ = "CachedRequestData"
        CachedRequestDataPatched.__module__ = output_mod.__name__
        output_mod.CachedRequestData = CachedRequestDataPatched
        CachedRequestDataPatched.__ucm_load_failure_patched__ = True  # type: ignore[attr-defined]
    except Exception as e:
        logger.warning("Could not patch CachedRequestData: %s", e)


def _patch_kv_output_aggregator() -> None:
    """Patch KVOutputAggregator.aggregate to aggregate invalid_block_ids."""
    try:
        from vllm.distributed.kv_transfer.kv_connector import utils as utils_mod

        KVOutputAggregator = utils_mod.KVOutputAggregator
        if getattr(KVOutputAggregator, "__ucm_load_failure_patched__", False):
            return

        original_aggregate = KVOutputAggregator.aggregate

        def patched_aggregate(self, outputs, output_rank):
            result = original_aggregate(self, outputs, output_rank)
            invalid_block_ids: set[int] = set()
            for model_runner_output in outputs:
                out = model_runner_output.kv_connector_output
                if out is not None:
                    invalid_block_ids |= getattr(out, "invalid_block_ids", set())
            # invalid_block_ids belongs on KVConnectorOutput (result.kv_connector_output), not on ModelRunnerOutput
            if result.kv_connector_output is not None:
                setattr(
                    result.kv_connector_output, "invalid_block_ids", invalid_block_ids
                )
            return result

        KVOutputAggregator.aggregate = patched_aggregate
        KVOutputAggregator.__ucm_load_failure_patched__ = True  # type: ignore[attr-defined]
    except Exception as e:
        logger.warning("Could not patch KVOutputAggregator: %s", e)


def _patch_block_pool() -> None:
    """Change BlockPool.cache_full_blocks condition from == to >=."""
    try:
        from vllm.v1.core import block_pool as block_pool_mod

        BlockPool = block_pool_mod.BlockPool
        if getattr(BlockPool.cache_full_blocks, "__ucm_load_failure_patched__", False):
            return

        original_cache_full_blocks = BlockPool.cache_full_blocks

        def patched_cache_full_blocks(
            self,
            request,
            blocks,
            num_cached_blocks,
            num_full_blocks,
            block_size,
            kv_cache_group_id,
        ):
            if num_cached_blocks >= num_full_blocks:
                return
            return original_cache_full_blocks(
                self,
                request,
                blocks,
                num_cached_blocks,
                num_full_blocks,
                block_size,
                kv_cache_group_id,
            )

        patched_cache_full_blocks.__ucm_load_failure_patched__ = True
        BlockPool.cache_full_blocks = patched_cache_full_blocks
    except Exception as e:
        logger.warning("Could not patch BlockPool: %s", e)


def _patch_single_type_kv_cache_manager() -> None:
    """Add early return when num_cached_blocks >= num_full_blocks in cache_blocks."""
    try:
        from vllm.v1.core import single_type_kv_cache_manager as stkvm_mod

        SingleTypeKVCacheManager = stkvm_mod.SingleTypeKVCacheManager
        if getattr(
            SingleTypeKVCacheManager.cache_blocks, "__ucm_load_failure_patched__", False
        ):
            return

        original_cache_blocks = SingleTypeKVCacheManager.cache_blocks

        def patched_cache_blocks(self, request, num_tokens):
            num_cached_blocks = self.num_cached_block[request.request_id]
            num_full_blocks = num_tokens // self.block_size
            if num_cached_blocks >= num_full_blocks:
                return
            return original_cache_blocks(self, request, num_tokens)

        patched_cache_blocks.__ucm_load_failure_patched__ = True
        SingleTypeKVCacheManager.cache_blocks = patched_cache_blocks
    except Exception as e:
        logger.warning("Could not patch SingleTypeKVCacheManager: %s", e)


def _patch_scheduler() -> None:
    """Patch Scheduler for load-failure recovery."""
    try:
        from collections.abc import Iterable

        from vllm.v1.core.sched.output import CachedRequestData
        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm.v1.request import Request, RequestStatus

        if getattr(Scheduler, "__ucm_load_failure_patched__", False):
            return

        orig_init = Scheduler.__init__

        def patched_init(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            if not hasattr(self, "failed_recving_kv_req_ids"):
                self.failed_recving_kv_req_ids: set[str] = set()

        Scheduler.__init__ = patched_init

        def _ensure_failed_recving_kv_req_ids(self: Scheduler) -> set[str]:
            """Ensure `failed_recving_kv_req_ids` exists even for pre-existing instances.

            NOTE: Monkeypatching `Scheduler.__init__` only affects instances created
            after the patch is applied. Engine cores may already have created the
            Scheduler, so we must lazily initialize the attribute on first use.
            """
            s = getattr(self, "failed_recving_kv_req_ids", None)
            if s is None:
                s = set()
                setattr(self, "failed_recving_kv_req_ids", s)
            return s

        def _update_requests_with_invalid_blocks(
            self: Scheduler,
            requests: Iterable[Request],
            invalid_block_ids: set[int],
        ) -> tuple[set[str], int]:
            """
            Identify and update requests affected by invalid KV cache blocks.

            This method scans the given requests, detects those with invalid blocks
            and adjusts their `num_computed_tokens` to the longest valid prefix.
            For observability, it also accumulates the total number of tokens that
            will need to be recomputed across all affected requests.

            Args:
                requests: The set of requests to scan for invalid blocks.
                invalid_block_ids: IDs of invalid blocks.

            Returns:
                tuple:
                    - affected_req_ids (set[str]): IDs of requests impacted by
                    invalid blocks.
                    - total_affected_tokens (int): Total number of tokens that must
                    be recomputed across all affected requests (for observability).
            """
            affected_req_ids: set[str] = set()
            total_affected_tokens = 0
            failed_recving_kv_req_ids = _ensure_failed_recving_kv_req_ids(self)
            # If a block is invalid and shared by multiple requests in the batch,
            # these requests must be rescheduled, but only the first will recompute
            # it. This set tracks blocks already marked for recomputation.
            marked_invalid_block_ids: set[int] = set()
            for request in requests:
                is_affected = False
                marked_invalid_block = False
                req_id = request.request_id
                # TODO (davidb): add support for hybrid memory allocator
                (req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)
                # We iterate only over blocks that may contain externally computed
                # tokens
                if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                    # Async loading. If num_computed_tokens is set it implies we
                    # already processed some block failures for it in a prior step
                    req_num_computed_tokens = (
                        request.num_computed_tokens
                        if req_id in failed_recving_kv_req_ids
                        else len(req_block_ids) * self.block_size
                    )
                else:
                    # Sync loading. num_computed_tokens includes new tokens
                    req_num_computed_tokens = request.num_cached_tokens

                req_num_computed_blocks = (
                    req_num_computed_tokens + self.block_size - 1
                ) // self.block_size
                for idx, block_id in zip(range(req_num_computed_blocks), req_block_ids):
                    if block_id not in invalid_block_ids:
                        continue

                    is_affected = True

                    if block_id in marked_invalid_block_ids:
                        # This invalid block is shared with a previous request
                        # and was already marked for recomputation.
                        # This means this request can still consider this block
                        # as computed when rescheduled.
                        # Currently this only applies to sync loading; Async
                        # loading does not yet support block sharing
                        continue

                    marked_invalid_block_ids.add(block_id)

                    if marked_invalid_block:
                        # This request has already marked an invalid block for
                        # recomputation and updated its num_computed_tokens.
                        continue

                    marked_invalid_block = True
                    # Truncate the computed tokens at the first failed block
                    request.num_computed_tokens = idx * self.block_size
                    total_affected_tokens += (
                        req_num_computed_tokens - request.num_computed_tokens
                    )

                if is_affected:
                    if not marked_invalid_block:
                        # All invalid blocks of this request are shared with
                        # previous requests and will be recomputed by them.
                        # Revert to considering only cached tokens as computed.
                        # Currently this only applies to sync loading; Async
                        # loading does not yet support block sharing
                        total_affected_tokens += (
                            request.num_computed_tokens - request.num_cached_tokens
                        )
                        request.num_computed_tokens = request.num_cached_tokens

                    affected_req_ids.add(request.request_id)

            return affected_req_ids, total_affected_tokens

        def _handle_invalid_blocks(
            self: Scheduler, invalid_block_ids: set[int]
        ) -> set[str]:
            failed_recving_kv_req_ids = _ensure_failed_recving_kv_req_ids(self)
            total_requests_to_reschedule = 0
            total_tokens_to_reschedule = 0

            # --- Handle async KV loads (WAITING_FOR_REMOTE_KVS) ---
            async_load_reqs = (
                req
                for req in self.waiting
                if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
            )
            async_affected_req_ids, num_tokens_to_reschedule = (
                self._update_requests_with_invalid_blocks(
                    async_load_reqs, invalid_block_ids
                )
            )

            total_requests_to_reschedule += len(async_affected_req_ids)
            total_tokens_to_reschedule += num_tokens_to_reschedule

            # Mark requests with async KV load failures; they will be rescheduled
            # once loading completes.
            failed_recving_kv_req_ids |= async_affected_req_ids

            # --- Handle sync KV loads (running requests) ---
            sync_affected_req_ids, num_tokens_to_reschedule = (
                self._update_requests_with_invalid_blocks(
                    self.running, invalid_block_ids
                )
            )

            total_requests_to_reschedule += len(sync_affected_req_ids)
            total_tokens_to_reschedule += num_tokens_to_reschedule

            if total_requests_to_reschedule:
                logger.warning(
                    "Recovered from KV load failure: "
                    "%d request(s) rescheduled (%d tokens affected).",
                    total_requests_to_reschedule,
                    total_tokens_to_reschedule,
                )

            # Return the IDs of affected running requests to skip in
            # update_from_output.
            return sync_affected_req_ids

        Scheduler._update_requests_with_invalid_blocks = (
            _update_requests_with_invalid_blocks
        )
        Scheduler._handle_invalid_blocks = _handle_invalid_blocks

        def patched_make_cached_request_data(
            self: Scheduler,
            running_reqs: list[Request],
            resumed_reqs: list[Request],
            num_scheduled_tokens: dict[str, int],
            spec_decode_tokens: dict[str, list[int]],
            req_to_new_blocks: dict[str, KVCacheBlocks],
        ) -> CachedRequestData:
            req_ids: list[str] = []
            new_token_ids: list[list[int]] = []
            new_block_ids: list[Optional[tuple[list[int], ...]]] = []
            num_computed_tokens: list[int] = []
            num_output_tokens: list[int] = []

            import itertools

            use_connector = self.connector is not None
            for req in itertools.chain(running_reqs, resumed_reqs):
                req_id = req.request_id
                req_ids.append(req_id)
                num_tokens = num_scheduled_tokens[req_id] - len(
                    spec_decode_tokens.get(req_id, ())
                )
                if self.use_pp:
                    # When using PP, the scheduler sends the sampled tokens back,
                    # because there's no direct communication between the first-
                    # stage worker and the last-stage worker. Otherwise, we don't
                    # need to send the sampled tokens back because the model runner
                    # will cache them.
                    token_ids = req.all_token_ids[
                        req.num_computed_tokens : req.num_computed_tokens + num_tokens
                    ]
                    new_token_ids.append(token_ids)
                elif use_connector:
                    # When using a KVConnector, we add a placeholder to avoid index
                    # out of bounds errors. TODO: Remove this once the KVConnector
                    # is updated to handle token IDs properly.
                    new_token_ids.append([])
                new_block_ids.append(
                    req_to_new_blocks[req_id].get_block_ids(allow_none=True)
                )
                num_computed_tokens.append(req.num_computed_tokens)
                num_output_tokens.append(len(req.output_token_ids))
            # Because resumed_reqs is usually empty, it is more efficient to do
            # in-place appending so that we don't need to allocate a new list.
            resumed_from_preemption = [False] * len(running_reqs)
            resumed_from_preemption += [True] * len(resumed_reqs)

            return CachedRequestData(
                req_ids=req_ids,
                resumed_from_preemption=resumed_from_preemption,
                new_token_ids=new_token_ids,
                new_block_ids=new_block_ids,
                num_computed_tokens=num_computed_tokens,
                num_output_tokens=num_output_tokens,
            )

        Scheduler._make_cached_request_data = patched_make_cached_request_data
        from collections import defaultdict
        from typing import Optional

        from vllm.v1.core.sched.output import SchedulerOutput
        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm.v1.core.sched.utils import check_stop, remove_all
        from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
        from vllm.v1.outputs import ModelRunnerOutput
        from vllm.v1.request import Request

        def patched_update_from_output(
            self,
            scheduler_output: SchedulerOutput,
            model_runner_output: ModelRunnerOutput,
        ) -> dict[int, EngineCoreOutputs]:

            sampled_token_ids = model_runner_output.sampled_token_ids
            logprobs = model_runner_output.logprobs
            prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens
            pooler_outputs = model_runner_output.pooler_output
            num_nans_in_logits = model_runner_output.num_nans_in_logits
            kv_connector_output = model_runner_output.kv_connector_output

            outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
            from vllm.v1.spec_decode.metrics import SpecDecodingStats

            spec_decoding_stats: Optional[SpecDecodingStats] = None
            kv_connector_stats = (
                kv_connector_output.kv_connector_stats if kv_connector_output else None
            )

            failed_kv_load_req_ids = None
            if kv_connector_output and kv_connector_output.invalid_block_ids:
                # These blocks contain externally computed tokens that failed to
                # load. Identify affected requests and adjust their computed token
                # count to trigger recomputation of the invalid blocks.
                failed_kv_load_req_ids = self._handle_invalid_blocks(
                    kv_connector_output.invalid_block_ids
                )
            # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
            # the below loop can be a performance bottleneck. We should do our best
            # to avoid expensive operations inside the loop.
            stopped_running_reqs: set[Request] = set()
            stopped_preempted_reqs: set[Request] = set()
            for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
                assert num_tokens_scheduled > 0
                if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:
                    # Skip requests that were recovered from KV load failure
                    continue
                request = self.requests.get(req_id)
                if request is None:
                    # The request is already finished. This can happen if the
                    # request is aborted while the model is executing it (e.g.,
                    # in pipeline parallelism).
                    continue

                req_index = model_runner_output.req_id_to_index[req_id]
                generated_token_ids = (
                    sampled_token_ids[req_index] if sampled_token_ids else []
                )

                scheduled_spec_token_ids = (
                    scheduler_output.scheduled_spec_decode_tokens.get(req_id)
                )
                if scheduled_spec_token_ids:
                    num_draft_tokens = len(scheduled_spec_token_ids)
                    num_accepted = len(generated_token_ids) - 1
                    num_rejected = num_draft_tokens - num_accepted
                    # num_computed_tokens represents the number of tokens
                    # processed in the current step, considering scheduled
                    # tokens and rejections. If some tokens are rejected,
                    # num_computed_tokens is decreased by the number of rejected
                    # tokens.
                    if request.num_computed_tokens > 0:
                        request.num_computed_tokens -= num_rejected
                    if request.num_output_tokens > 0:
                        request.num_output_tokens -= num_rejected
                    spec_decoding_stats = self.make_spec_decoding_stats(
                        spec_decoding_stats,
                        num_draft_tokens=num_draft_tokens,
                        num_accepted_tokens=num_accepted,
                    )

                stopped = False
                new_logprobs = None
                new_token_ids = generated_token_ids
                kv_transfer_params = None
                status_before_stop = request.status

                # Check for stop and update request status.
                if new_token_ids:
                    new_token_ids, stopped = self._update_request_with_output(
                        request, new_token_ids
                    )

                # Stop checking for pooler models.
                pooler_output = None
                if pooler_outputs:
                    pooler_output = pooler_outputs[req_index]
                    stopped = check_stop(request, self.max_model_len, pooler_output)

                if stopped:
                    kv_transfer_params = self._free_request(request)
                    if status_before_stop == RequestStatus.RUNNING:
                        stopped_running_reqs.add(request)
                    else:
                        stopped_preempted_reqs.add(request)

                # Extract sample logprobs if needed.
                if (
                    request.sampling_params is not None
                    and request.sampling_params.logprobs is not None
                    and logprobs
                ):
                    # NOTE: once we support N tokens per step (spec decode),
                    # the outer lists can be of length > 1.
                    new_logprobs = logprobs.slice(req_index, req_index + 1)

                if new_token_ids and self.structured_output_manager.should_advance(
                    request
                ):
                    # NOTE: structured_output_request
                    # should not be None if use_structured_output, we have
                    # checked above, so safe to ignore type warning
                    request.structured_output_request.grammar.accept_tokens(  # type: ignore[union-attr]
                        req_id, new_token_ids
                    )

                if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                    request.num_nans_in_logits = num_nans_in_logits[req_id]

                # Get prompt logprobs for this request.
                prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
                if new_token_ids or pooler_output is not None or kv_transfer_params:

                    # Add EngineCoreOutput for this Request.
                    outputs[request.client_index].append(
                        EngineCoreOutput(
                            request_id=req_id,
                            new_token_ids=new_token_ids,
                            finish_reason=request.get_finished_reason(),
                            new_logprobs=new_logprobs,
                            new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                            pooling_output=pooler_output,
                            stop_reason=request.stop_reason,
                            events=request.take_events(),
                            kv_transfer_params=kv_transfer_params,
                            trace_headers=request.trace_headers,
                            num_cached_tokens=request.num_cached_tokens,
                        )
                    )
                else:
                    # Invariant: EngineCore returns no partial prefill outputs.
                    assert not prompt_logprobs_tensors

            # Remove the stopped requests from the running and waiting queues.
            if stopped_running_reqs:
                self.running = remove_all(self.running, stopped_running_reqs)
            if stopped_preempted_reqs:
                # This is a rare case and unlikely to impact performance.
                self.waiting.remove_requests(stopped_preempted_reqs)

            # KV Connector: update state for finished KV Transfers.
            if model_runner_output.kv_connector_output:
                self._update_from_kv_xfer_finished(
                    model_runner_output.kv_connector_output
                )

            # Create EngineCoreOutputs for all clients that have requests with
            # outputs in this step.
            engine_core_outputs = {
                client_index: EngineCoreOutputs(outputs=outs)
                for client_index, outs in outputs.items()
            }

            finished_req_ids = self.finished_req_ids_dict
            if finished_req_ids:
                # Include ids of requests that finished since last outputs
                # were sent.
                for client_index, finished_set in finished_req_ids.items():
                    # Set finished request set in EngineCoreOutputs for this client.
                    if (eco := engine_core_outputs.get(client_index)) is not None:
                        eco.finished_requests = finished_set
                    else:
                        engine_core_outputs[client_index] = EngineCoreOutputs(
                            finished_requests=finished_set
                        )
                finished_req_ids.clear()

            if (
                stats := self.make_stats(spec_decoding_stats, kv_connector_stats)
            ) is not None:
                # Return stats to only one of the front-ends.
                if (eco := next(iter(engine_core_outputs.values()), None)) is None:
                    # We must return the stats even if there are no request
                    # outputs this step.
                    engine_core_outputs[0] = eco = EngineCoreOutputs()
                eco.scheduler_stats = stats

            return engine_core_outputs

        Scheduler.update_from_output = patched_update_from_output

        def patched_update_waiting_for_remote_kv(self, request: Request) -> bool:
            """
            KV Connector: check if the request_id is finished_recving.

            The finished_recving_kv_req_ids list is populated
            on the previous steps()'s update_from_output based
            on the worker side connector.

            When the kv transfer is ready, we cache the blocks
            and the request state will be moved back to WAITING from
            WAITING_FOR_REMOTE_KV.
            """
            assert self.connector is not None
            failed_recving_kv_req_ids = _ensure_failed_recving_kv_req_ids(self)
            if request.request_id not in self.finished_recving_kv_req_ids:
                return False

            if request.request_id in failed_recving_kv_req_ids:
                # Request had KV load failures; num_computed_tokens was already
                # updated in _update_requests_with_invalid_blocks
                if request.num_computed_tokens:
                    # Cache any valid computed tokens.
                    self.kv_cache_manager.cache_blocks(
                        request, request.num_computed_tokens
                    )
                else:
                    # No valid computed tokens, release allocated blocks.
                    # There may be a local cache hit on retry.
                    self.kv_cache_manager.free(request)

                failed_recving_kv_req_ids.discard(request.request_id)
            else:
                # Now that the blocks are ready, actually cache them.
                (block_ids,) = self.kv_cache_manager.get_block_ids(request.request_id)
                num_computed_tokens = len(block_ids) * self.block_size
                # Handle the case where num request tokens less than one block.
                num_computed_tokens = min(num_computed_tokens, request.num_tokens)
                if num_computed_tokens == request.num_tokens:
                    num_computed_tokens -= 1
                # This will cache the blocks iff caching is enabled.
                self.kv_cache_manager.cache_blocks(request, num_computed_tokens)

                # Update the request state for scheduling.
                request.num_computed_tokens = num_computed_tokens

            # Return that we are ready.
            self.finished_recving_kv_req_ids.remove(request.request_id)
            return True

        Scheduler._update_waiting_for_remote_kv = patched_update_waiting_for_remote_kv

        Scheduler.__ucm_load_failure_patched__ = True  # type: ignore[attr-defined]

    except Exception as e:
        logger.warning("Could not patch Scheduler: %s", e)


def _patch_gpu_model_runner() -> None:
    """Patch _update_states method in GPUModelRunner."""
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner

        if getattr(GPUModelRunner, "__ucm_load_failure_patched__", False):
            return

        def patched_update_states(self, scheduler_output: "SchedulerOutput") -> None:
            """Update the cached states and the persistent batch with the scheduler
            output.

            The updated states are used by the `_prepare_inputs` function to create
            the input GPU tensors for the model.

            The SamplingMetadata is updated and copied to the GPU if there is a
            new/resumed/paused/finished request in the batch.
            """
            from typing import cast

            import torch
            from vllm.distributed.parallel_state import get_pp_group
            from vllm.sampling_params import SamplingType
            from vllm.v1.worker.gpu_input_batch import CachedRequestState
            from vllm.v1.worker.gpu_model_runner import VllmModelForPooling

            # Remove finished requests from the cached states.
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

            # Free the cached encoder outputs.
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

            reqs_to_add: list[CachedRequestState] = []
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

                if self.is_pooling_model:
                    assert pooling_params is not None
                    task = pooling_params.task
                    assert task is not None, "You did not set `task` in the API"

                    model = cast(VllmModelForPooling, self.get_model())
                    to_update = model.pooler.get_pooling_updates(task)
                    to_update.apply(pooling_params)

                req_state = CachedRequestState(
                    req_id=req_id,
                    prompt_token_ids=new_req_data.prompt_token_ids,
                    prompt_embeds=new_req_data.prompt_embeds,
                    mm_features=new_req_data.mm_features,
                    sampling_params=sampling_params,
                    pooling_params=pooling_params,
                    generator=generator,
                    block_ids=new_req_data.block_ids,
                    num_computed_tokens=new_req_data.num_computed_tokens,
                    output_token_ids=[],
                    lora_request=new_req_data.lora_request,
                )
                self.requests[req_id] = req_state

                # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
                if self.uses_mrope:
                    self._init_mrope_positions(req_state)

                reqs_to_add.append(req_state)

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
                    reqs_to_add.append(req_state)
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
            for request in reqs_to_add:
                self.input_batch.add_request(request)

            # Condense the batched states if there are gaps left by removed requests
            self.input_batch.condense()
            # Allow attention backend to reorder the batch, potentially
            self._may_reorder_batch(scheduler_output)
            # Refresh batch metadata with any pending updates.
            self.input_batch.refresh_metadata()

        GPUModelRunner._update_states = patched_update_states
        GPUModelRunner.__ucm_load_failure_patched__ = True  # type: ignore[attr-defined]
    except Exception as e:
        logger.warning("Could not patch GPUModelRunner: %s", e)


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


def _patch_gpu_worker() -> None:
    """Patch Worker to use kv_connector_output.is_empty()."""
    try:
        import copy
        from typing import Optional, Union

        import torch
        from vllm.v1.worker import gpu_worker as gpu_worker_mod

        Worker = gpu_worker_mod.Worker
        if getattr(Worker, "__ucm_load_failure_patched__", False):
            return

        @torch.inference_mode()
        def patched_execute_model(
            self,
            scheduler_output: "SchedulerOutput",
        ) -> Optional[Union["ModelRunnerOutput", "AsyncModelRunnerOutput"]]:
            """Execute model with kv_connector_output.is_empty() check."""
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
            from vllm.v1.worker.utils import is_residual_scattered_for_sp

            intermediate_tensors = None
            forward_pass = scheduler_output.total_num_scheduled_tokens > 0
            num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
            num_input_tokens = self.model_runner._get_num_input_tokens(
                num_scheduled_tokens
            )
            all_gather_tensors = {
                "residual": not is_residual_scattered_for_sp(
                    self.vllm_config, num_input_tokens
                )
            }
            if forward_pass and not get_pp_group().is_first_rank:
                intermediate_tensors = IntermediateTensors(
                    get_pp_group().recv_tensor_dict(
                        all_gather_group=get_tp_group(),
                        all_gather_tensors=all_gather_tensors,
                    )
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
                output.tensors,
                all_gather_group=get_tp_group(),
                all_gather_tensors=all_gather_tensors,
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

        Worker.execute_model = patched_execute_model
        Worker.__ucm_load_failure_patched__ = True  # type: ignore[attr-defined]
    except Exception as e:
        logger.warning("Could not patch Worker: %s", e)


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


def _patch_kv_connector_model_runner_mixin() -> None:
    """Set output.invalid_block_ids in KVConnectorModelRunnerMixin."""
    try:
        from contextlib import contextmanager
        from typing import Generator

        from vllm.v1.worker import kv_connector_model_runner_mixin as mixin_mod

        KVConnectorModelRunnerMixin = mixin_mod.KVConnectorModelRunnerMixin
        if getattr(KVConnectorModelRunnerMixin, "__ucm_load_failure_patched__", False):
            return

        @staticmethod
        @contextmanager
        def patched_get_kv_connector_output(
            scheduler_output: "SchedulerOutput",
            wait_for_save: bool = True,
        ) -> Generator["KVConnectorOutput", None, None]:
            """Get KVConnectorOutput with invalid_block_ids populated."""
            # Keep imports local to reduce import-time coupling across vLLM versions.
            from vllm.distributed.kv_transfer import get_kv_transfer_group
            from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
            from vllm.forward_context import get_forward_context
            from vllm.v1.outputs import KVConnectorOutput

            output = KVConnectorOutput()

            # Update KVConnector with the KVConnector metadata forward().
            kv_connector = get_kv_transfer_group()
            assert isinstance(kv_connector, KVConnectorBase)
            assert scheduler_output.kv_connector_metadata is not None
            kv_connector.bind_connector_metadata(scheduler_output.kv_connector_metadata)

            # Background KV cache transfers happen here.
            # These transfers are designed to be async and the requests
            # involved may be disjoint from the running requests.
            # Do this here to save a collective_rpc.
            kv_connector.start_load_kv(get_forward_context())
            try:
                yield output
            finally:
                if wait_for_save:
                    kv_connector.wait_for_save()

                output.finished_sending, output.finished_recving = (
                    kv_connector.get_finished(scheduler_output.finished_req_ids)
                )
                output.invalid_block_ids = kv_connector.get_block_ids_with_load_errors()

                output.kv_connector_stats = (
                    KVConnectorModelRunnerMixin.get_kv_connector_stats()
                )
                kv_connector.clear_connector_metadata()

        # Directly replace the whole method (same style as other patches).
        KVConnectorModelRunnerMixin._get_kv_connector_output = (
            patched_get_kv_connector_output
        )
        KVConnectorModelRunnerMixin.__ucm_load_failure_patched__ = True  # type: ignore[attr-defined]
    except Exception as e:
        logger.warning("Could not patch KVConnectorModelRunnerMixin: %s", e)


def _patch_attention_layer() -> None:
    """Patch attention layer"""
    try:
        # IMPORTANT: Rebinding names imported with
        # `from vllm.attention.layer import foo` only updates local variables in
        # this patch module, NOT the original `vllm.attention.layer` module.
        # We must patch the module attributes directly.
        import vllm.attention.layer as layer_mod

        if getattr(layer_mod, "__ucm_load_failure_patched__", False):
            return

        def patched_wait_for_kv_layer_from_connector(layer_name: str) -> None:
            # Keep imports local to reduce import-time coupling across vLLM versions.
            from vllm.distributed.kv_transfer import (
                get_kv_transfer_group,
                has_kv_transfer_group,
                is_v1_kv_transfer_group,
            )
            from vllm.forward_context import get_forward_context

            if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
                return

            connector = get_kv_transfer_group()
            # Guard for older connectors to avoid AttributeError.
            if (
                hasattr(connector, "has_connector_metadata")
                and not connector.has_connector_metadata()
            ):
                return

            attn_metadata = get_forward_context().attn_metadata
            if attn_metadata is None:
                return
            assert isinstance(attn_metadata, dict)
            connector.wait_for_layer_load(layer_name)

        def patched_maybe_save_kv_layer_to_connector(
            layer_name: str,
            kv_cache_layer: "list[object]",
        ) -> None:
            from vllm.distributed.kv_transfer import (
                get_kv_transfer_group,
                has_kv_transfer_group,
                is_v1_kv_transfer_group,
            )
            from vllm.forward_context import get_forward_context

            if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
                return

            connector = get_kv_transfer_group()
            if (
                hasattr(connector, "has_connector_metadata")
                and not connector.has_connector_metadata()
            ):
                return

            attn_metadata = get_forward_context().attn_metadata
            if attn_metadata is None:
                return
            assert isinstance(attn_metadata, dict)
            connector.save_kv_layer(
                layer_name, kv_cache_layer, attn_metadata[layer_name]
            )

        layer_mod.wait_for_kv_layer_from_connector = (
            patched_wait_for_kv_layer_from_connector
        )
        layer_mod.maybe_save_kv_layer_to_connector = (
            patched_maybe_save_kv_layer_to_connector
        )
        layer_mod.__ucm_load_failure_patched__ = True  # type: ignore[attr-defined]
    except Exception as e:
        logger.warning("Could not patch attention layer: %s", e)


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
