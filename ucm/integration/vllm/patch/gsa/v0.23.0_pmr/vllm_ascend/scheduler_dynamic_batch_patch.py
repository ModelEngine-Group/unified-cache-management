"""GSA SchedulerDynamicBatch.schedule patch for vLLM-Ascend 0.23.0."""

from __future__ import annotations

from types import FunctionType

from ucm.integration.vllm.patch.utils import patch_or_inject, when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


def schedule(self) -> SchedulerOutput:
    from ucm.sparse.base import INVALID_SLOT

    # NOTE: This scheduling algorithm is developed based on the "super.schedule()"
    # with the implementations of the dynamic batch and some modifications:
    # 1. Token budget can be dynamically refined according to the self.running
    # through the BudgetRefiner;
    # 2. This scheduling algorithm follows decode-first chunked prefills and FCFS
    # strategy, which is slightly different to the "super.schedule()"
    # 3. Similar to the "super.schedule()", at each step, the scheduler tries to
    # assign tokens to the requests so that each request's num_computed_tokens can
    # catch up its num_tokens_with_spec.
    # 4. So far, the dynamic batch only supports 910B3 NPU. Further work will include
    # more devices and finer optimization strategy.

    scheduled_new_reqs: list[Request] = []
    scheduled_resumed_reqs: list[Request] = []
    scheduled_running_reqs: list[Request] = []
    preempted_reqs: list[Request] = []

    req_to_new_blocks: dict[str, KVCacheBlocks] = {}
    num_scheduled_tokens: dict[str, int] = {}
    token_budget = self.max_num_scheduled_tokens
    refined_budget = self.budget_refiner.refine_budget(self.running, token_budget)
    if refined_budget != token_budget:
        logger.debug("Refined token_budget: %s -> %s", token_budget, refined_budget)
    token_budget = refined_budget

    # NOTE: We move the prefill requests to the end of the self.running
    # list and keep the relative order unchanged. This rearrangement makes this
    # scheduling algorithm a strict decode-first chunked prefills.
    d_lst = [
        req for req in self.running if req.num_computed_tokens >= req.num_prompt_tokens
    ]
    p_lst = [
        req for req in self.running if req.num_computed_tokens < req.num_prompt_tokens
    ]
    self.running = d_lst + p_lst

    # Encoder-related.
    scheduled_encoder_inputs: dict[str, list[int]] = {}
    encoder_compute_budget = self.max_num_encoder_input_tokens
    # Spec decode-related.
    scheduled_spec_decode_tokens: dict[str, list[int]] = {}

    # For logging.
    scheduled_timestamp = time.monotonic()

    # First, schedule the RUNNING requests.
    req_index = 0
    req_sparsed_slots: dict[str, int] = {}
    while req_index < len(self.running) and token_budget > 0:
        request = self.running[req_index]

        num_slots_sparsed = INVALID_SLOT
        if self.ucm_sparse:
            num_slots_sparsed = self.ucm_sparse.estimate_num_slots_sparsed(
                request, self.num_lookahead_tokens
            )
        req_sparsed_slots.update({request.request_id: num_slots_sparsed})

        num_new_tokens = (
            request.num_tokens_with_spec
            + request.num_output_placeholders
            - request.num_computed_tokens
        )
        if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
            num_new_tokens = self.scheduler_config.long_prefill_token_threshold
        num_new_tokens = min(num_new_tokens, token_budget)

        # Make sure the input position does not exceed the max model len.
        # This is necessary when using spec decoding.
        num_new_tokens = min(
            num_new_tokens, self.max_model_len - 1 - request.num_computed_tokens
        )

        # Schedule encoder inputs.
        encoder_inputs_to_schedule = None
        new_encoder_compute_budget = encoder_compute_budget
        if request.has_encoder_inputs:
            (encoder_inputs_to_schedule, num_new_tokens, new_encoder_compute_budget) = (
                self._try_schedule_encoder_inputs(
                    request,
                    request.num_computed_tokens,
                    num_new_tokens,
                    encoder_compute_budget,
                )
            )

        if num_new_tokens == 0:
            # The request cannot be scheduled because one of the following
            # reasons:
            # 1. No new tokens to schedule. This may happen when
            #    (1) PP>1 and we have already scheduled all prompt tokens
            #    but they are not finished yet.
            #    (2) Async scheduling and the request has reached to either
            #    its max_total_tokens or max_model_len.
            # 2. The encoder budget is exhausted.
            # 3. The encoder cache is exhausted.
            # NOTE(woosuk): Here, by doing `break` instead of `continue` as
            # in v1 scheduler, we strictly follow the FCFS scheduling policy.
            req_index += 1
            break

        while True:
            new_blocks = self.kv_cache_manager.allocate_slots(
                request,
                num_new_tokens,
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
                    if preempted_req in scheduled_running_reqs:
                        scheduled_running_reqs.remove(preempted_req)
                else:
                    preempted_req = self.running.pop()

                self.kv_cache_manager.free(preempted_req)
                self.encoder_cache_manager.free(preempted_req)
                preempted_req.status = RequestStatus.PREEMPTED
                preempted_req.num_computed_tokens = 0
                if self.log_stats:
                    preempted_req.record_event(
                        EngineCoreEventType.PREEMPTED, scheduled_timestamp
                    )

                self.waiting.prepend_request(preempted_req)
                preempted_reqs.append(preempted_req)
                logger.info(
                    "Preempted request %s (priority=%s). running_count=%s, token_budget=%s",
                    preempted_req.request_id,
                    getattr(preempted_req, "priority", "N/A"),
                    len(self.running),
                    token_budget,
                )
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
        req_to_new_blocks[request.request_id] = new_blocks
        num_scheduled_tokens[request.request_id] = num_new_tokens
        token_budget -= num_new_tokens
        req_index += 1

        # Speculative decode related.
        if request.spec_token_ids:
            num_scheduled_spec_tokens = (
                num_new_tokens + request.num_computed_tokens - request.num_tokens
            )
            if num_scheduled_spec_tokens > 0:
                # Trim spec_token_ids list to num_scheduled_spec_tokens.
                del request.spec_token_ids[num_scheduled_spec_tokens:]
                scheduled_spec_decode_tokens[request.request_id] = (
                    request.spec_token_ids
                )

        # Encoder-related.
        if encoder_inputs_to_schedule:
            scheduled_encoder_inputs[request.request_id] = encoder_inputs_to_schedule
            # Allocate the encoder cache.
            for i in encoder_inputs_to_schedule:
                self.encoder_cache_manager.allocate(request, i)
            encoder_compute_budget = new_encoder_compute_budget

    # Record the LoRAs in scheduled_running_reqs
    scheduled_loras: set[int] = set()
    if self.lora_config:
        scheduled_loras = set(
            req.lora_request.lora_int_id
            for req in scheduled_running_reqs
            if req.lora_request and req.lora_request.lora_int_id > 0
        )
        assert len(scheduled_loras) <= self.lora_config.max_loras

    # Next, schedule the WAITING requests.
    if not preempted_reqs:
        step_skipped_waiting = create_request_queue(self.policy)

        while (self.waiting or self.skipped_waiting) and token_budget > 0:
            if len(self.running) == self.max_num_running_reqs:
                break

            request_queue = self._select_waiting_queue_for_scheduling()
            if request_queue is None:
                break

            request = request_queue.peek_request()

            ########## zhangzhao add ##########
            num_slots_sparsed = INVALID_SLOT  ####  suit for skipped
            if self.ucm_sparse:
                num_slots_sparsed = self.ucm_sparse.estimate_num_slots_sparsed(
                    request, self.num_lookahead_tokens
                )
            req_sparsed_slots.update({request.request_id: num_slots_sparsed})
            ###################################

            # try to promote blocked statuses while traversing skipped queue.
            if self._is_blocked_waiting_status(
                request.status
            ) and not self._try_promote_blocked_waiting_request(request):
                if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                    logger.debug(
                        "%s is still in WAITING_FOR_REMOTE_KVS state.",
                        request.request_id,
                    )
                request_queue.pop_request()
                step_skipped_waiting.prepend_request(request)
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
                request_queue.pop_request()
                step_skipped_waiting.prepend_request(request)
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

                    if num_external_computed_tokens is None:
                        # The request cannot be scheduled because
                        # the KVConnector couldn't determine
                        # the number of matched tokens.
                        request_queue.pop_request()
                        step_skipped_waiting.prepend_request(request)
                        continue

                # Total computed tokens (local + external).
                num_computed_tokens = (
                    num_new_local_computed_tokens + num_external_computed_tokens
                )
            # KVTransfer: WAITING reqs have num_computed_tokens > 0
            # after async KV recvs are completed.
            else:
                new_computed_blocks = self.kv_cache_manager.create_empty_block_list()
                num_new_local_computed_tokens = 0
                num_computed_tokens = request.num_computed_tokens

            encoder_inputs_to_schedule = None
            new_encoder_compute_budget = encoder_compute_budget

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
                    num_new_tokens = self.scheduler_config.long_prefill_token_threshold

                # chunked prefill has to be enabled explicitly to allow
                # pooling requests to be chunked
                if (
                    not self.scheduler_config.enable_chunked_prefill
                    and num_new_tokens > token_budget
                ):
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue

                num_new_tokens = min(num_new_tokens, token_budget)
                assert num_new_tokens > 0

                # Schedule encoder inputs.
                if request.has_encoder_inputs:
                    (
                        encoder_inputs_to_schedule,
                        num_new_tokens,
                        new_encoder_compute_budget,
                        _,
                    ) = self._try_schedule_encoder_inputs(
                        request,
                        num_computed_tokens,
                        num_new_tokens,
                        encoder_compute_budget,
                    )
                    if num_new_tokens == 0:
                        # The request cannot be scheduled.
                        break

            # Handles an edge case when P/D Disaggregation
            # is used with Spec Decoding where an
            # extra block gets allocated which
            # creates a mismatch between the number
            # of local and remote blocks.
            effective_lookahead_tokens = (
                0 if request.num_computed_tokens == 0 else self.num_lookahead_tokens
            )

            # Determine if we need to allocate cross-attention blocks.
            if self.is_encoder_decoder and request.has_encoder_inputs:
                # TODO(russellb): For Whisper, we know that the input is
                # always padded to the maximum length. If we support other
                # encoder-decoder models, this will need to be updated if we
                # want to only allocate what is needed.
                num_encoder_tokens = self.scheduler_config.max_num_encoder_input_tokens
            else:
                num_encoder_tokens = 0

            new_blocks = self.kv_cache_manager.allocate_slots(
                request,
                num_new_tokens + num_external_computed_tokens,
                num_new_local_computed_tokens,
                new_computed_blocks,
                num_lookahead_tokens=effective_lookahead_tokens,
                delay_cache_blocks=load_kv_async,
                num_encoder_tokens=num_encoder_tokens,
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

            request = request_queue.pop_request()
            if load_kv_async:
                # If loading async, allocate memory and put request
                # into the WAITING_FOR_REMOTE_KV state.
                request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                step_skipped_waiting.prepend_request(request)
                request.num_computed_tokens = num_computed_tokens
                continue

            req_index += 1
            self.running.append(request)
            if self.log_stats:
                request.record_event(EngineCoreEventType.SCHEDULED, scheduled_timestamp)
            if request.status == RequestStatus.WAITING:
                scheduled_new_reqs.append(request)
            elif request.status == RequestStatus.PREEMPTED:
                scheduled_resumed_reqs.append(request)
            else:
                raise RuntimeError(f"Invalid request status: {request.status}")

            if self.lora_config and request.lora_request:
                scheduled_loras.add(request.lora_request.lora_int_id)
            req_to_new_blocks[request.request_id] = self.kv_cache_manager.get_blocks(
                request.request_id
            )
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            request.status = RequestStatus.RUNNING
            request.num_computed_tokens = num_computed_tokens
            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request.request_id] = (
                    encoder_inputs_to_schedule
                )
                # Allocate the encoder cache.
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                encoder_compute_budget = new_encoder_compute_budget

        # re-queue requests skipped in this pass ahead of older skipped items.
        if step_skipped_waiting:
            self.skipped_waiting.prepend_requests(step_skipped_waiting)

    # Check if the scheduling constraints are satisfied.
    total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
    if total_num_scheduled_tokens > self.max_num_scheduled_tokens:
        logger.error(
            "Scheduling constraint violated: total_num_scheduled_tokens=%s > max=%s",
            total_num_scheduled_tokens,
            self.max_num_scheduled_tokens,
        )
    assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
    assert token_budget >= 0
    if len(self.running) > self.max_num_running_reqs:
        logger.error(
            "Scheduling constraint violated: running=%s > max_running=%s",
            len(self.running),
            self.max_num_running_reqs,
        )
    assert len(self.running) <= self.max_num_running_reqs
    # Since some requests in the RUNNING queue may not be scheduled in
    # this step, the total number of scheduled requests can be smaller than
    # len(self.running).
    scheduled_count = (
        len(scheduled_new_reqs)
        + len(scheduled_resumed_reqs)
        + len(scheduled_running_reqs)
    )
    if scheduled_count > len(self.running):
        logger.error(
            "Scheduling constraint violated: scheduled=%s > running=%s",
            scheduled_count,
            len(self.running),
        )
    assert scheduled_count <= len(self.running)

    # Get the longest common prefix among all requests in the running queue.
    # This can be potentially used for cascade attention.
    num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
    if self.running:
        any_request = self.running[0]
        num_common_prefix_blocks = self.kv_cache_manager.get_num_common_prefix_blocks(
            any_request.request_id
        )
    # Construct the scheduler output.
    new_reqs_data = [
        NewRequestData.from_request(
            req, req_to_new_blocks[req.request_id].get_block_ids()
        )
        for req in scheduled_new_reqs
    ]
    cached_reqs_data = self._make_cached_request_data(
        scheduled_running_reqs,
        scheduled_resumed_reqs,
        num_scheduled_tokens,
        scheduled_spec_decode_tokens,
        req_to_new_blocks,
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
        free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
    )

    # NOTE(Kuntai): this function is designed for multiple purposes:
    # 1. Plan the KV cache store
    # 2. Wrap up all the KV cache load / save ops into an opaque object
    # 3. Clear the internal states of the connector
    if self.connector is not None:
        meta = self.connector.build_connector_meta(scheduler_output)
        scheduler_output.kv_connector_metadata = meta

    # collect KV cache events from KV cache manager
    events = self.kv_cache_manager.take_events()

    # collect KV cache events from connector
    if self.connector is not None:
        connector_events = self.connector.take_events()
        if connector_events:
            if events is None:
                events = list(connector_events)
            else:
                events.extend(connector_events)

    # publish collected KV cache events
    if events:
        batch = KVEventBatch(ts=time.time(), events=events)
        self.kv_event_publisher.publish(batch)

    self._update_after_schedule(scheduler_output)
    return scheduler_output


@when_imported("vllm_ascend.core.scheduler_dynamic_batch")
def _patch(mod):
    if hasattr(mod.SchedulerDynamicBatch, "schedule"):
        _patched = FunctionType(
            schedule.__code__,
            vars(mod),
            name=schedule.__name__,
            argdefs=schedule.__defaults__,
            closure=schedule.__closure__,
        )
        _patched.__annotations__ = schedule.__annotations__
        _patched.__kwdefaults__ = schedule.__kwdefaults__
        patch_or_inject(mod.SchedulerDynamicBatch, "schedule", _patched)
        logger.info("UCM GSA SchedulerDynamicBatch.schedule patch applied")
