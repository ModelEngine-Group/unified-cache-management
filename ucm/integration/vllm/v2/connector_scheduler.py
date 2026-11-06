"""Scheduler request state, lookup, dispatch planning and save publication."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from itertools import count
from typing import TYPE_CHECKING

import numpy as np
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
)

from ucm.integration.vllm.request_hasher import RequestHasher
from ucm.logger import init_logger

from .kv_cache_spec import RestorePolicy, UCMKVCacheGroupSpec, groups_by_restore_policy
from .runtime import (
    ParallelRank,
    UCMStorageHandshake,
    make_common_ucm_chunk_ids,
    make_ucm_chunk_ids,
)
from .store_client import (
    JOB_ID_STRIDE,
    MAX_NATIVE_JOB_ID,
    AddressedToken,
    PreparedJob,
    StoreIOError,
    StoreSchedulerClient,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request

    from .runtime import UCMRuntimeContext

logger = init_logger(__name__)


@dataclass(frozen=True)
class UCMDispatchPlan:
    """Consecutive UCM chunks of one request for one restore policy.

    Scheduler resolves final UCM keys for every participating PP stage.
    Workers select their stage and resolve local byte ranges without hashing.
    Each plan owns its Prepare result and Connector job identity.
    """

    restore_policy: RestorePolicy
    ucm_keys_by_pp_rank: dict[int, tuple[bytes, ...]]
    # First token of each UCM chunk (uint64), one per UCM key in each PP stage.
    token_starts: np.ndarray
    # Block-major IDs for the explicitly named native groups below.
    vllm_block_ids_by_group: tuple[np.ndarray, ...]
    group_ids: tuple[int, ...]
    # None only when this plan's Prepare failed.
    prepared_job: PreparedJob | None
    # State loads share the running target's keys, Prepare and job with this
    # checkpoint target. Empty for other loads and all dumps.
    state_checkpoint_block_ids_by_group: tuple[np.ndarray, ...] = ()


@dataclass(frozen=True)
class RequestMetadata:
    request_id: str
    load_plans: tuple[UCMDispatchPlan, ...] = ()
    dump_plans: tuple[UCMDispatchPlan, ...] = ()
    prefetch_job: PreparedJob | None = None
    prepare_get_failed: bool = False


@dataclass
class UCMConnectorMetadata(KVConnectorMetadata):
    requests: dict[str, RequestMetadata] = field(default_factory=dict)
    preempted_req_ids: set[str] = field(default_factory=set)
    finished_req_ids: set[str] = field(default_factory=set)


class JobOperation(Enum):
    """The Store operation completed by a group of jobs for one request."""

    GET = "Get"
    PUT = "Put"
    PREFETCH = "Prefetch"


@dataclass
class PendingJob:
    """One Prepare: all ranks report its parent job before Complete."""

    request_id: str
    operation: JobOperation
    prepared_job: PreparedJob
    completed_ranks: set[ParallelRank] = field(default_factory=set)
    failed: bool = False
    dispatched: bool = False


@dataclass(frozen=True)
class UCMLookupResult:
    """None defers vLLM admission until the prefetch phase is resolved."""

    external_hit_tokens: int | None
    common_chunk_hashes: tuple[bytes, ...]


@dataclass
class RequestState:
    """Mutable cross-step state, owned exclusively by ConnectorScheduler."""

    request: Request
    num_preemptions: int
    hbm_hit_tokens: int = 0
    external_hit_tokens: int = 0
    num_tokens: int = 0
    common_chunk_hashes: tuple[bytes, ...] = ()
    # Cleared after the first scheduled step builds its initial Get plans.
    # JobResult records actual IO completion and success independently.
    needs_load: bool = False


class UCMConnectorScheduler:
    def __init__(self, context: UCMRuntimeContext) -> None:
        self.context = context
        self.spec = context.spec
        self.parallel_topology = context.parallel_topology
        self._store_scheduler = StoreSchedulerClient(context.launch_config)
        self.groups_by_policy = groups_by_restore_policy(self.spec)
        self._recompute_tokens = 1
        self._chunk_hasher = RequestHasher(
            context.fingerprint
        ).make_request_block_hasher(self.spec.chunk_size)
        self.load_threshold_tokens = max(
            int(context.launch_config.get("load_tokens_threshold", 0)), 0
        )
        self._request_states: dict[str, RequestState] = {}
        self._expected_worker_ranks = {
            ParallelRank(tp_rank, pp_rank, pcp_rank)
            for pp_rank in range(self.parallel_topology.pp_size)
            for pcp_rank in range(self.parallel_topology.pcp_size)
            for tp_rank in range(self.parallel_topology.tp_size)
        }
        # Odd main IDs leave even native IDs available for Worker-local subjobs.
        self._job_ids = count(1, JOB_ID_STRIDE)
        self._pending_jobs: dict[int, PendingJob] = {}
        self._prefetch_job_ids_by_request: dict[str, int] = {}
        self._prefetch_failed_request_ids: set[str] = set()
        self._registered_request_ids: set[str] = set()
        # Keys select the policies each PP stage stores; values are the complete
        # stage-local object sizes that Store admission will need.
        self._store_sizes_by_stage: dict[int, dict[RestorePolicy, int]] = {}
        self._finished_request_ids: set[str] = set()
        self._retired_request_ids: set[str] = set()

    def set_xfer_handshake_metadata(
        self, metadata: Mapping[int, UCMStorageHandshake]
    ) -> None:
        # Handshake representatives declare policies; IO completion still needs
        # every configured rank, including PCP replicas that do not handshake.
        self._store_sizes_by_stage = {
            worker.rank.pp_rank: worker.store_sizes_by_policy
            for worker in metadata.values()
        }

    def on_new_request(self, request: Request) -> None:
        request_id = str(request.request_id)
        self._store_scheduler.on_new_request(request_id)
        self._registered_request_ids.add(request_id)

    def lookup(self, request: Request, num_computed_tokens: int) -> UCMLookupResult:
        request_id = str(request.request_id)
        request_state = self._request_states.get(request_id)
        # reset_prefix_cache can preempt before schedule(), so waiting-request
        # lookup may run before build_connector_meta sees the preempted IDs.
        if (
            request_state is not None
            and request_state.num_preemptions != request.num_preemptions
        ):
            self._reset_preempted_request(request_id)
            request_state = None
        if request_state is None:
            request_state = RequestState(
                request=request,
                num_preemptions=request.num_preemptions,
                num_tokens=len(request.all_token_ids),
                common_chunk_hashes=make_common_ucm_chunk_ids(
                    self._chunk_hasher(request)
                ),
            )
            self._request_states[request_id] = request_state
        common_chunk_hashes = request_state.common_chunk_hashes
        if request_id in self._prefetch_job_ids_by_request:
            return UCMLookupResult(None, common_chunk_hashes)
        hit_tokens = (
            0
            if request_id in self._prefetch_failed_request_ids
            else self._lookup_hit_tokens(
                request_id,
                common_chunk_hashes,
                len(request.all_token_ids),
                num_computed_tokens,
            )
        )
        if hit_tokens is None:
            job_id = self._allocate_job_id()
            try:
                prefetch_token = self._store_scheduler.prepare_prefetch(request_id)
            except StoreIOError:
                # No Get hit was promised. Recompute instead of repeatedly
                # retrying the failed prefetch window.
                logger.warning(
                    "UCM PreparePrefetch failed; recomputing this request",
                    exc_info=True,
                )
                self._prefetch_failed_request_ids.add(request_id)
                hit_tokens = 0
            else:
                prefetch_job = self._track_prepared_job(
                    job_id, JobOperation.PREFETCH, request_id, prefetch_token
                )
                self._prefetch_job_ids_by_request[request_id] = prefetch_job.job_id
                return UCMLookupResult(None, common_chunk_hashes)
        external_hit_tokens = (
            hit_tokens if hit_tokens > self.load_threshold_tokens else 0
        )
        request_state.hbm_hit_tokens = num_computed_tokens
        request_state.external_hit_tokens = external_hit_tokens
        request_state.needs_load = external_hit_tokens > 0
        return UCMLookupResult(external_hit_tokens, common_chunk_hashes)

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        lookup_result = self.lookup(request, num_computed_tokens)
        # IO may use asynchronous native jobs, but this connector drains loads
        # within the step. It does not claim cross-step asynchronous loading.
        return lookup_result.external_hit_tokens, False

    def _lookup_across_pp_stages(
        self,
        request_id: str,
        common_chunk_hashes: Sequence[bytes],
        restore_policy: RestorePolicy,
        *,
        reverse: bool = False,
    ) -> int | None:
        """Return the shared hit index, or None when a stage needs prefetch.

        Store uses -1 for a miss. None is the native prefetchable signal and
        must reach lookup() before hit-index arithmetic can continue.
        """
        hit_indices = []
        for pp_rank, store_sizes_by_policy in self._store_sizes_by_stage.items():
            if restore_policy not in store_sizes_by_policy:
                continue
            ucm_keys = make_ucm_chunk_ids(common_chunk_hashes, restore_policy, pp_rank)
            if reverse:
                hit_index = self._store_scheduler.lookup_on_reverse(
                    request_id, ucm_keys
                )
            else:
                hit_index = self._store_scheduler.lookup_on_prefix(request_id, ucm_keys)
            if hit_index is None:
                return None
            hit_indices.append(hit_index)
        return min(hit_indices, default=len(common_chunk_hashes) - 1)

    def _lookup_hit_tokens(
        self,
        request_id: str,
        common_chunk_hashes: Sequence[bytes],
        num_tokens: int,
        hbm_hit_tokens: int,
    ) -> int | None:
        """Return external hit tokens, or None if Store needs prefetch."""
        chunk_size = self.spec.chunk_size
        first_chunk_index = hbm_hit_tokens // chunk_size
        end_chunk_index = max(num_tokens - self._recompute_tokens, 0) // chunk_size
        if not self.groups_by_policy or end_chunk_index <= first_chunk_index:
            return 0
        try:
            if RestorePolicy.FA in self.groups_by_policy:
                hit_index = self._lookup_across_pp_stages(
                    request_id,
                    common_chunk_hashes[first_chunk_index:end_chunk_index],
                    RestorePolicy.FA,
                )
                if hit_index is None:
                    return None
                end_chunk_index = first_chunk_index + hit_index + 1
            boundary_policies = tuple(
                restore_policy
                for restore_policy in self.groups_by_policy
                if restore_policy is not RestorePolicy.FA
            )
            while end_chunk_index > first_chunk_index:
                previous_end_chunk_index = end_chunk_index
                for restore_policy in boundary_policies:
                    hit_index = self._lookup_across_pp_stages(
                        request_id,
                        common_chunk_hashes[first_chunk_index:end_chunk_index],
                        restore_policy,
                        reverse=True,
                    )
                    if hit_index is None:
                        return None
                    end_chunk_index = first_chunk_index + hit_index + 1
                    if end_chunk_index == first_chunk_index:
                        return 0
                if end_chunk_index == previous_end_chunk_index:
                    break
            return max(end_chunk_index * chunk_size - hbm_hit_tokens, 0)
        except StoreIOError:
            logger.warning("UCM lookup failed; recomputing this request", exc_info=True)
            return 0

    def make_ucm_keys_by_pp_rank(
        self, common_chunk_hashes: Sequence[bytes], restore_policy: RestorePolicy
    ) -> dict[int, tuple[bytes, ...]]:
        return {
            pp_rank: make_ucm_chunk_ids(common_chunk_hashes, restore_policy, pp_rank)
            for pp_rank, store_sizes_by_policy in self._store_sizes_by_stage.items()
            if restore_policy in store_sizes_by_policy
        }

    def _allocate_job_id(self) -> int:
        job_id = next(self._job_ids)
        if job_id > MAX_NATIVE_JOB_ID:
            raise OverflowError("UCM job IDs exhausted the uint64 range")
        return job_id

    def _track_prepared_job(
        self,
        job_id: int,
        operation: JobOperation,
        request_id: str,
        token: AddressedToken,
        ucm_keys: tuple[bytes, ...] = (),
    ) -> PreparedJob:
        prepared_job = PreparedJob(job_id=job_id, ucm_keys=ucm_keys, token=token)
        self._pending_jobs[prepared_job.job_id] = PendingJob(
            request_id, operation, prepared_job
        )
        return prepared_job

    def _prepare_plan_job(
        self,
        request_id: str,
        ucm_keys_by_pp_rank: dict[int, tuple[bytes, ...]],
        *,
        is_load: bool,
    ) -> PreparedJob | None:
        ucm_keys = tuple(
            ucm_key
            for stage_ucm_keys in ucm_keys_by_pp_rank.values()
            for ucm_key in stage_ucm_keys
        )
        job_id = self._allocate_job_id()
        operation = JobOperation.GET if is_load else JobOperation.PUT
        try:
            if operation is JobOperation.GET:
                token = self._store_scheduler.prepare_get(request_id, ucm_keys)
            else:
                token = self._store_scheduler.prepare_put(request_id, ucm_keys)
        except StoreIOError:
            logger.warning(
                "UCM Prepare%s failed for request %s, job %d",
                operation.value,
                request_id,
                job_id,
                exc_info=True,
            )
            return None
        return self._track_prepared_job(job_id, operation, request_id, token, ucm_keys)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> UCMConnectorMetadata:
        block_state = scheduler_output.kv_connector_block_state
        assert block_state is not None
        metadata = UCMConnectorMetadata(
            preempted_req_ids=set(scheduler_output.preempted_req_ids or ()),
            finished_req_ids=set(scheduler_output.finished_req_ids),
        )
        for request_id in metadata.preempted_req_ids:
            request_state = self._request_states.get(request_id)
            # Preserve a state already rebuilt by lookup after a cache reset.
            if (
                request_state is not None
                and request_state.num_preemptions
                != request_state.request.num_preemptions
            ):
                self._reset_preempted_request(request_id)
        for request_id in block_state.req_ids:
            request_state = self._request_states.get(request_id)
            if request_state is None:
                continue
            block_tables = block_state.get_block_ids(request_id)
            scheduled_tokens = scheduler_output.num_scheduled_tokens.get(request_id, 0)
            metadata.requests[request_id] = self._build_request_plan(
                request_id,
                request_state,
                block_tables,
                scheduled_tokens,
                block_state.boundary_state_offloads.get(request_id, ()),
            )
            if scheduled_tokens > 0:
                request_state.needs_load = False
        for request_id, job_id in self._prefetch_job_ids_by_request.items():
            pending_job = self._pending_jobs[job_id]
            if (
                request_id not in self._finished_request_ids
                and not pending_job.dispatched
            ):
                metadata.requests[request_id] = RequestMetadata(
                    request_id, prefetch_job=pending_job.prepared_job
                )
                pending_job.dispatched = True
        # The finish callback may precede a final no-forward dump plan. Only
        # retire the store request once that plan has been prepared above.
        self._retired_request_ids.update(metadata.finished_req_ids)
        self._close_finished_requests()
        for request_id in metadata.finished_req_ids:
            self._request_states.pop(request_id, None)
        return metadata

    def _reset_preempted_request(self, request_id: str) -> None:
        self._request_states.pop(request_id, None)
        self._prefetch_failed_request_ids.discard(request_id)
        for pending_job in self._pending_jobs.values():
            if (
                pending_job.request_id == request_id
                and pending_job.operation is JobOperation.PUT
            ):
                pending_job.failed = True

    def request_finished(self, request_id: str) -> None:
        request_id = str(request_id)
        self._finished_request_ids.add(request_id)
        job_id = self._prefetch_job_ids_by_request.get(request_id)
        if job_id is not None and not self._pending_jobs[job_id].dispatched:
            self._store_scheduler.complete_prefetch(request_id, False)
            del self._pending_jobs[job_id]
            del self._prefetch_job_ids_by_request[request_id]
        self._close_finished_requests()

    def _complete_ready_jobs(self) -> None:
        """Close a request/kind window once all its jobs and ranks are terminal."""
        # Complete is request-scoped. Drain every job appended to the same
        # request/kind before releasing its pins or committing its writes.
        jobs_by_window: dict[tuple[str, JobOperation], list[tuple[int, PendingJob]]] = (
            defaultdict(list)
        )
        for job_id, pending_job in self._pending_jobs.items():
            jobs_by_window[(pending_job.request_id, pending_job.operation)].append(
                (job_id, pending_job)
            )
        for (request_id, operation), window_jobs in jobs_by_window.items():
            if any(
                pending_job.completed_ranks != self._expected_worker_ranks
                for _, pending_job in window_jobs
            ):
                continue
            success = not any(pending_job.failed for _, pending_job in window_jobs)
            if operation is JobOperation.GET:
                self._store_scheduler.complete_get(request_id)
            elif operation is JobOperation.PUT:
                self._store_scheduler.complete_put(request_id, success)
            else:
                self._store_scheduler.complete_prefetch(request_id, success)
                if not success:
                    self._prefetch_failed_request_ids.add(request_id)
                del self._prefetch_job_ids_by_request[request_id]
            for job_id, _ in window_jobs:
                del self._pending_jobs[job_id]

    def _close_finished_requests(self) -> None:
        """Unregister ended requests after final metadata and window completion."""
        active_request_ids = {
            pending_job.request_id for pending_job in self._pending_jobs.values()
        }
        for request_id in tuple(self._retired_request_ids):
            if request_id in active_request_ids:
                continue
            if request_id in self._registered_request_ids:
                self._store_scheduler.on_request_finished(request_id)
                self._registered_request_ids.remove(request_id)
            self._finished_request_ids.discard(request_id)
            self._prefetch_failed_request_ids.discard(request_id)
            self._retired_request_ids.remove(request_id)

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        metadata = connector_output.kv_connector_worker_meta
        if metadata is None:
            return
        # A load failure invalidates this request's saves even if another rank
        # has already reported a locally successful Put.
        # Later layers derive new K/V from attention outputs using the loaded KV.
        for request_id in metadata.failed_load_request_ids:
            request_state = self._request_states.get(request_id)
            if request_state is not None:
                # vLLM handles invalid blocks before delivering connector output.
                # Keep the request so subsequent recomputation can still dump KV.
                request_state.hbm_hit_tokens = request_state.request.num_computed_tokens
                request_state.external_hit_tokens = 0
                request_state.needs_load = False
            for pending_job in self._pending_jobs.values():
                if (
                    pending_job.operation is JobOperation.PUT
                    and pending_job.request_id == request_id
                ):
                    pending_job.failed = True
        for job_result in metadata.job_results:
            pending_job = self._pending_jobs.get(job_result.job_id)
            if pending_job is None:
                continue
            pending_job.completed_ranks.add(job_result.rank)
            pending_job.failed |= not job_result.success
        self._complete_ready_jobs()
        self._close_finished_requests()

    def has_pending_push_work(self) -> bool:
        return bool(self._prefetch_job_ids_by_request)

    def shutdown(self) -> None:
        # An unacknowledged parent can still have Worker subjobs in flight.
        # Leave such reservations to backend recovery, never release them early.
        for pending_job in self._pending_jobs.values():
            pending_job.failed = True
        self._close_finished_requests()

    def _state_dump_plans(
        self, request_state: RequestState, offloads: Sequence[tuple[int, int, int]]
    ) -> tuple[UCMDispatchPlan, ...]:
        """Build one dump plan per complete State checkpoint boundary."""
        state_groups = self.groups_by_policy.get(RestorePolicy.STATE, ())
        if not state_groups:
            return ()
        group_ids = tuple(group.group_id for group in state_groups)
        chunk_size = self.spec.chunk_size
        matched_prefix_tokens = (
            request_state.hbm_hit_tokens + request_state.external_hit_tokens
        )
        blocks_by_boundary: dict[int, dict[int, int]] = defaultdict(dict)
        for group_id, block_id, boundary_token in offloads:
            # Exclude imported snapshots using the initial hit boundary:
            # engine hand-offs can arrive after the request has advanced.
            if (
                boundary_token <= matched_prefix_tokens
                or boundary_token > request_state.num_tokens
            ):
                continue
            # Native State checkpoints may be finer than the UCM key stride.
            if boundary_token % chunk_size:
                continue
            blocks_by_boundary[boundary_token][group_id] = block_id

        plans = []
        for boundary_token, blocks_by_group in sorted(blocks_by_boundary.items()):
            # One State object contains every participating State group.
            if any(group_id not in blocks_by_group for group_id in group_ids):
                continue
            chunk_index = boundary_token // chunk_size - 1
            ucm_keys_by_pp_rank = self.make_ucm_keys_by_pp_rank(
                (request_state.common_chunk_hashes[chunk_index],), RestorePolicy.STATE
            )
            plans.append(
                UCMDispatchPlan(
                    restore_policy=RestorePolicy.STATE,
                    ucm_keys_by_pp_rank=ucm_keys_by_pp_rank,
                    token_starts=np.asarray(
                        [chunk_index * chunk_size], dtype=np.uint64
                    ),
                    vllm_block_ids_by_group=tuple(
                        np.asarray([blocks_by_group[group_id]], dtype=np.uint64)
                        for group_id in group_ids
                    ),
                    group_ids=group_ids,
                    prepared_job=self._prepare_plan_job(
                        str(request_state.request.request_id),
                        ucm_keys_by_pp_rank,
                        is_load=False,
                    ),
                )
            )
        return tuple(plans)

    def _build_request_plan(
        self,
        request_id: str,
        request_state: RequestState,
        block_tables: Sequence[Sequence[int]],
        scheduled_tokens: int,
        state_offloads: Sequence[tuple[int, int, int]] = (),
    ) -> RequestMetadata:
        """Build this step's load and save plans."""
        # With the recompute failure policy, vLLM rolls back this valid prefix.
        # num_tokens bounds the request range hashed at the last lookup.
        step_start_token = min(
            request_state.request.num_computed_tokens, request_state.num_tokens
        )
        step_end_token = min(
            step_start_token + scheduled_tokens,
            request_state.num_tokens,
        )
        should_load = request_state.needs_load and scheduled_tokens > 0
        load_end_token = (
            request_state.hbm_hit_tokens + request_state.external_hit_tokens
        )
        load_start_token = (
            request_state.hbm_hit_tokens if should_load else load_end_token
        )
        dump_start_token = step_start_token
        dump_end_token = step_end_token
        load_plans = self.build_dispatch_plans(
            request_state,
            block_tables,
            load_start_token,
            load_end_token,
            is_load=True,
            scheduled_tokens=scheduled_tokens,
        )
        prepare_get_failed = any(plan.prepared_job is None for plan in load_plans)
        if prepare_get_failed:
            # Worker reports the promised-but-unavailable KV to vLLM and
            # acknowledges any other Get jobs already prepared for this request.
            return RequestMetadata(
                request_id, load_plans=load_plans, prepare_get_failed=True
            )
        dump_plans = self.build_dispatch_plans(
            request_state, block_tables, dump_start_token, dump_end_token, is_load=False
        )
        return RequestMetadata(
            request_id,
            load_plans,
            dump_plans + self._state_dump_plans(request_state, state_offloads),
        )

    def build_dispatch_plans(
        self,
        request_state: RequestState,
        block_tables: Sequence[Sequence[int]],
        token_start: int,
        token_end: int,
        *,
        is_load: bool,
        scheduled_tokens: int = 0,
    ) -> tuple[UCMDispatchPlan, ...]:
        """FA/WA plans and both State load destinations.

        FA gets every complete UCM chunk the range touches; WA gets the newest
        complete boundary. State loads use that restore boundary's UCM chunk
        and restore it into both the running block and imported checkpoint.
        is_load selects the operation; scheduled_tokens selects the running
        destination on loads. State dumps are built from the engine's exact
        boundary_state_offloads.
        """
        plans: list[UCMDispatchPlan] = []
        if token_end <= token_start:
            return ()
        chunk_size = self.spec.chunk_size
        # Type-invariant UCM chunk bounds: the first chunk the range touches
        # and the boundary it ends on.  No complete UCM chunk in range means
        # no plan for any restore policy: FA has nothing to store or load, and the
        # WA/State boundary was not reached.
        first_chunk_index = token_start // chunk_size
        end_chunk_index = token_end // chunk_size
        if end_chunk_index <= first_chunk_index:
            return ()
        for restore_policy, groups in self.groups_by_policy.items():
            if restore_policy is RestorePolicy.STATE and not is_load:
                continue
            if restore_policy in (RestorePolicy.WA, RestorePolicy.STATE):
                # WA keeps the newest completed boundary; a State load
                # ends exactly at the selected restore boundary.
                start_chunk_index = end_chunk_index - 1
            else:
                # FA chunks are stored once and never re-stored, so a plan
                # covers every complete UCM chunk the range touches -- not
                # just the newest boundary.
                start_chunk_index = first_chunk_index
            plans.append(
                self.make_dispatch_plan(
                    restore_policy,
                    groups,
                    request_state,
                    block_tables,
                    start_chunk_index,
                    end_chunk_index,
                    # State uses this step's running slot and adds the imported
                    # checkpoint target to the same prepared plan.
                    state_target_tokens=token_end + scheduled_tokens,
                    is_load=is_load,
                )
            )
        return tuple(plans)

    def make_dispatch_plan(
        self,
        restore_policy: RestorePolicy,
        groups: Sequence[UCMKVCacheGroupSpec],
        request_state: RequestState,
        block_tables: Sequence[Sequence[int]],
        start_chunk_index: int,
        end_chunk_index: int,
        state_target_tokens: int | None,
        *,
        is_load: bool,
    ) -> UCMDispatchPlan:
        """Build and Prepare one policy plan with its own job and token."""
        chunk_size = self.spec.chunk_size
        ucm_keys_by_pp_rank = self.make_ucm_keys_by_pp_rank(
            request_state.common_chunk_hashes[start_chunk_index:end_chunk_index],
            restore_policy,
        )
        state_checkpoint_block_ids_by_group = ()
        if restore_policy is RestorePolicy.STATE and is_load:
            # The engine's pre-copy already ran. Load this source snapshot into
            # both the running slot and its imported checkpoint, with one pin.
            state_checkpoint_block_ids_by_group = tuple(
                self._select_group_block_ids(
                    restore_policy,
                    group,
                    block_tables[group.group_id],
                    start_chunk_index,
                    end_chunk_index,
                    state_target_tokens=end_chunk_index * chunk_size,
                )
                for group in groups
            )
        return UCMDispatchPlan(
            restore_policy,
            ucm_keys_by_pp_rank,
            np.arange(start_chunk_index, end_chunk_index, dtype=np.uint64)
            * np.uint64(chunk_size),
            tuple(
                self._select_group_block_ids(
                    restore_policy,
                    group,
                    block_tables[group.group_id],
                    start_chunk_index,
                    end_chunk_index,
                    state_target_tokens=state_target_tokens,
                )
                for group in groups
            ),
            tuple(group.group_id for group in groups),
            prepared_job=self._prepare_plan_job(
                str(request_state.request.request_id),
                ucm_keys_by_pp_rank,
                is_load=is_load,
            ),
            state_checkpoint_block_ids_by_group=state_checkpoint_block_ids_by_group,
        )

    def _select_group_block_ids(
        self,
        restore_policy: RestorePolicy,
        group: UCMKVCacheGroupSpec,
        vllm_block_table: Sequence[int],
        start_chunk_index: int,
        end_chunk_index: int,
        *,
        state_target_tokens: int | None = None,
    ) -> np.ndarray:
        """One group's window vLLM block ids for the plan's UCM chunks.

        Every window is ``blocks_per_chunk`` vLLM blocks ending at the block
        containing the window's last token -- HMA's boundary indices: FA UCM
        chunk k's last token is (k + 1) * chunk_size - 1; a WA UCM chunk
        anchors its window at the restore boundary. State uses a platform
        load slot. Parse-time chunk/tail and divisibility checks keep the
        block-table windows valid. Block-major per UCM chunk.
        """

        chunk_size = self.spec.chunk_size
        blocks_per_chunk = self.spec.blocks_per_chunk[group.group_id]
        if restore_policy is RestorePolicy.STATE:
            # The caller selects one of the two State load destinations.
            table_index = (state_target_tokens - 1) // group.block_token_span
            vllm_block_id = vllm_block_table[table_index]
            assert vllm_block_id != 0
            return np.asarray([vllm_block_id], dtype=np.uint64)
        if restore_policy is RestorePolicy.FA:
            boundary_tokens = (
                np.arange(start_chunk_index + 1, end_chunk_index + 1, dtype=np.uint64)
                * chunk_size
                - 1
            )
        else:
            # WA: one UCM chunk at the plan's newest boundary.
            boundary_tokens = np.asarray(
                [end_chunk_index * chunk_size - 1], dtype=np.uint64
            )
        boundary_table_indices = boundary_tokens // group.block_token_span
        # Convert only the table range the windows touch -- a WA group's
        # table spans the whole request while its window holds a couple
        # of blocks. The parse-time chunk/tail and divisibility checks keep
        # first_table_index non-negative.
        first_table_index = int(boundary_table_indices.min()) - (blocks_per_chunk - 1)
        last_table_index = int(boundary_table_indices.max())
        vllm_block_ids = np.asarray(
            vllm_block_table[first_table_index : last_table_index + 1], dtype=np.uint64
        )
        window_end_offsets = boundary_table_indices - first_table_index
        if blocks_per_chunk == 1:
            return vllm_block_ids[window_end_offsets]
        steps_back = np.arange(blocks_per_chunk, dtype=np.uint64)[::-1]
        return vllm_block_ids[
            (window_end_offsets[:, None] - steps_back[None, :]).reshape(-1)
        ]
