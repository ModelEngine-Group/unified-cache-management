from __future__ import annotations

import itertools
from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Optional, Union

from vllm.config import VllmConfig
from vllm.distributed.kv_events import EventPublisherFactory, KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.encoder_cache_manager import (
    EncoderCacheManager,
    compute_encoder_budget,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.core.sched.utils import check_stop, remove_all
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import DraftTokenIds, KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputManager

logger = init_logger(__name__)


class Scheduler(SchedulerInterface):
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.kv_cache_config = kv_cache_config
        self.kv_events_config = vllm_config.kv_events_config
        self.parallel_config = vllm_config.parallel_config
        self.log_stats = log_stats
        self.structured_output_manager = structured_output_manager
        self.is_encoder_decoder = vllm_config.model_config.is_encoder_decoder

        # include_finished_set controls whether a separate set of finished
        # request ids should be included in the EngineCoreOutputs returned
        # by update_from_outputs(). This is currently used in the multi-engine
        # case to track request lifetimes efficiently.
        self.finished_req_ids_dict: Optional[dict[int, set[str]]] = (
            defaultdict(set) if include_finished_set else None
        )

        # Scheduling constraints.
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = self.scheduler_config.max_num_batched_tokens
        self.max_model_len = self.scheduler_config.max_model_len
        self.enable_kv_cache_events = (
            self.kv_events_config is not None
            and self.kv_events_config.enable_kv_cache_events
        )

        # Create KVConnector for the Scheduler. Note that each Worker
        # will have a corresponding KVConnector with Role=WORKER.
        # KV Connector pushes/pull of remote KVs for P/D and offloading.
        self.connector = None
        if self.vllm_config.kv_transfer_config is not None:
            assert len(self.kv_cache_config.kv_cache_groups) == 1, (
                "Multiple KV cache groups are not currently supported "
                "with KV connectors"
            )
            assert not self.is_encoder_decoder, (
                "Encoder-decoder models are not currently supported "
                "with KV connectors"
            )
            self.connector = KVConnectorFactory.create_connector(
                config=self.vllm_config, role=KVConnectorRole.SCHEDULER
            )

        self.kv_event_publisher = EventPublisherFactory.create(
            self.kv_events_config,
            self.parallel_config.data_parallel_rank,
        )

        num_gpu_blocks = self.cache_config.num_gpu_blocks
        assert num_gpu_blocks is not None and num_gpu_blocks > 0

        self.block_size = self.cache_config.block_size

        self.dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        # Note(hc): The scheduler’s block_size must be multiplied
        # by dcp_world_size, since block hashes are computed on the
        # original full token sequence at a granularity of
        # original_block_size × dcp_world_size.
        if self.dcp_world_size > 1:
            self.block_size *= self.dcp_world_size

        # req_id -> Request
        self.requests: dict[str, Request] = {}
        # Scheduling policy
        if self.scheduler_config.policy == "priority":
            self.policy = SchedulingPolicy.PRIORITY
        elif self.scheduler_config.policy == "fcfs":
            self.policy = SchedulingPolicy.FCFS
        else:
            raise ValueError(
                f"Unknown scheduling policy: {self.scheduler_config.policy}"
            )
        # Priority queues for requests.
        self.waiting = create_request_queue(self.policy)
        self.running: list[Request] = []

        # The request IDs that are finished in between the previous and the
        # current steps. This is used to notify the workers about the finished
        # requests so that they can free the cached states for those requests.
        # This is flushed at the end of each scheduling step.
        self.finished_req_ids: set[str] = set()

        # KV Connector: requests in process of async KV loading or recving
        self.finished_recving_kv_req_ids: set[str] = set()

        self.failed_recving_kv_req_ids: set[str] = set()

        # Encoder-related.
        # Calculate encoder cache size if applicable
        # NOTE: For now we use the same budget for both compute and space.
        # This can be changed when we make encoder cache for embedding caching
        # across requests.
        encoder_compute_budget, encoder_cache_size = compute_encoder_budget(
            model_config=vllm_config.model_config,
            scheduler_config=vllm_config.scheduler_config,
            mm_registry=mm_registry,
        )

        # NOTE(woosuk): Here, "encoder" includes the vision encoder (and
        # projector if needed) for MM models as well as encoder-decoder
        # transformers.
        self.max_num_encoder_input_tokens = encoder_compute_budget
        # NOTE: For the models without encoder (e.g., text-only models),
        # the encoder cache will not be initialized because cache size is 0
        # for these models.
        self.encoder_cache_manager = EncoderCacheManager(cache_size=encoder_cache_size)

        speculative_config = vllm_config.speculative_config
        self.use_eagle = False
        self.num_spec_tokens = self.num_lookahead_tokens = 0
        if speculative_config:
            self.num_spec_tokens = speculative_config.num_speculative_tokens
            if speculative_config.use_eagle():
                self.use_eagle = True
                self.num_lookahead_tokens = self.num_spec_tokens

        # Create the KV cache manager.
        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            enable_caching=self.cache_config.enable_prefix_caching,
            use_eagle=self.use_eagle,
            log_stats=self.log_stats,
            enable_kv_cache_events=self.enable_kv_cache_events,
            dcp_world_size=self.dcp_world_size,
        )
        self.use_pp = self.parallel_config.pipeline_parallel_size > 1

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
        failed_recving_kv_req_ids = self.failed_recving_kv_req_ids
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

            prev_num_computed_tokens = request.num_computed_tokens
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

    def _handle_invalid_blocks(self, invalid_block_ids: set[int]) -> set[str]:
        failed_recving_kv_req_ids = self.failed_recving_kv_req_ids
        total_requests_to_reschedule = 0
        total_tokens_to_reschedule = 0

        # logger.warning(
        #    f"[kv-load] scheduler handle invalid_block_ids={len(invalid_block_ids)} "
        #    f"sample={list(invalid_block_ids)[:10]}"
        # )

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
        if async_affected_req_ids:
            logger.warning(
                f"[kv-load] async affected reqs (WAITING_FOR_REMOTE_KVS): "
                f"count={len(async_affected_req_ids)} sample={list(async_affected_req_ids)[:10]}"
            )

        # --- Handle sync KV loads (running requests) ---
        sync_affected_req_ids, num_tokens_to_reschedule = (
            self._update_requests_with_invalid_blocks(self.running, invalid_block_ids)
        )
        if sync_affected_req_ids and self.connector is not None:
            # Keep UCM request metadata consistent with rescheduled requests.
            # self.connector may be either the real connector implementation
            # or a wrapper that stores the implementation in `.connector`.
            connector_impl = getattr(self.connector, "connector", self.connector)
            requests_meta = getattr(connector_impl, "requests_meta", None)
            if isinstance(requests_meta, dict):
                for req_id in sync_affected_req_ids:
                    requests_meta.pop(req_id, None)
                    logger.warning(f"Removed request metadata for request {req_id}")

        total_requests_to_reschedule += len(sync_affected_req_ids)
        total_tokens_to_reschedule += num_tokens_to_reschedule

        if total_requests_to_reschedule:
            logger.warning(
                f"Recovered from KV load failure: {total_requests_to_reschedule} request(s) "
                f"rescheduled ({total_tokens_to_reschedule} tokens affected)."
            )
            # if sync_affected_req_ids:
            #    logger.warning(
            #        f"[kv-load] sync affected running reqs (skip output update this step): "
            #        f"count={len(sync_affected_req_ids)} sample={list(sync_affected_req_ids)[:10]}"
            #    )

        # Return the IDs of affected running requests to skip in
        # update_from_output.
        return sync_affected_req_ids

    def _make_cached_request_data(
        self,
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

    def update_from_output(
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

        spec_decoding_stats: Optional[SpecDecodingStats] = None
        kv_connector_stats = (
            kv_connector_output.kv_connector_stats if kv_connector_output else None
        )

        failed_kv_load_req_ids = None
        if kv_connector_output and kv_connector_output.invalid_block_ids:
            # These blocks contain externally computed tokens that failed to
            # load. Identify affected requests and adjust their computed token
            # count to trigger recomputation of the invalid blocks.
            # logger.warning(
            #    f"[kv-load] scheduler received invalid_block_ids={len(kv_connector_output.invalid_block_ids)} "
            #    f"sample={list(kv_connector_output.invalid_block_ids)[:10]}"
            # )
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

            if new_token_ids and self.structured_output_manager.should_advance(request):
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
            self._update_from_kv_xfer_finished(model_runner_output.kv_connector_output)

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

    def _update_waiting_for_remote_kv(self, request) -> bool:
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
        failed_recving_kv_req_ids = self.failed_recving_kv_req_ids
        if request.request_id not in self.finished_recving_kv_req_ids:
            return False

        if request.request_id in failed_recving_kv_req_ids:
            # Request had KV load failures; num_computed_tokens was already
            # updated in _update_requests_with_invalid_blocks
            if request.num_computed_tokens:
                # Cache any valid computed tokens.
                # logger.warning(
                #    f"[kv-load] async recv finished (had failures): cache valid prefix. "
                #    f"req_id={request.request_id} num_computed_tokens={request.num_computed_tokens}"
                # )
                self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
            else:
                # No valid computed tokens, release allocated blocks.
                # There may be a local cache hit on retry.
                # logger.warning(
                #    f"[kv-load] async recv finished (had failures): free blocks (no valid prefix). "
                #    f"req_id={request.request_id}"
                # )
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
