"""Scheduler request state, lookup, dispatch planning and save publication."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
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

    Scheduler resolves the UCM keys shared by all PP stages.
    Workers resolve their local byte ranges without hashing.
    Each plan owns its Prepare result and Connector job identity.
    """

    restore_policy: RestorePolicy
    ucm_keys: tuple[bytes, ...]
    # First token of each UCM chunk (uint64), one per UCM key.
    token_starts: np.ndarray
    # Block-major IDs for the explicitly named native groups below.
    vllm_block_ids_by_group: tuple[np.ndarray, ...]
    group_ids: tuple[int, ...]
    prepared_job: PreparedJob | None = None
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
    offload_jobs: tuple[PreparedJob, ...] = ()


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


@dataclass
class RequestState:
    """Mutable cross-step state, owned exclusively by ConnectorScheduler."""

    request: Request
    # The preemption count of the Store lookup snapshot, once initialized.
    num_preemptions: int | None = None
    finished: bool = False
    prefetch_job_id: int | None = None
    hbm_hit_tokens: int = 0
    external_hit_tokens: int = 0
    num_tokens: int = 0
    common_chunk_hashes: tuple[bytes, ...] = ()
    # Cleared after the first scheduled step builds its initial Get plans.
    # JobResult records actual IO completion and success independently.
    needs_load: bool = False


class UCMConnectorScheduler:
    def __init__(
        self, context: UCMRuntimeContext, *, recompute_tokens: int = 1
    ) -> None:
        self.context = context
        self.spec = context.spec
        self.parallel_topology = context.parallel_topology
        # CacheStore joins the worker-led domain after receiving its slot size.
        self._store_scheduler: StoreSchedulerClient | None = None
        self.groups_by_policy = groups_by_restore_policy(self.spec)
        self._recompute_tokens = recompute_tokens
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
        self._offload_jobs: dict[int, PreparedJob] = {}
        self._needs_offload_drain = False

    def set_xfer_handshake_metadata(
        self, metadata: Mapping[int | tuple[int, ...], UCMStorageHandshake]
    ) -> None:
        if self._store_scheduler is None:
            block_size = max(worker.cache_block_size for worker in metadata.values())
            shared_spec = next(iter(metadata.values())).spec
            if shared_spec is not None:
                self.spec = shared_spec
                self.groups_by_policy = groups_by_restore_policy(self.spec)
                self._chunk_hasher = RequestHasher(
                    self.context.fingerprint
                ).make_request_block_hasher(self.spec.chunk_size)
            self._store_scheduler = StoreSchedulerClient(
                self.context.launch_config,
                self.context.store_runtime(block_size),
            )

    def on_new_request(self, request: Request) -> None:
        if self._store_scheduler is None:
            raise RuntimeError(
                "UCM v2 CacheStore has not received the worker handshake"
            )
        request_id = str(request.request_id)
        self._store_scheduler.on_new_request(request_id)
        self._request_states[request_id] = RequestState(request)

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        """Return external hit tokens or None for prefetch. KV loads are synchronous."""
        request_id = str(request.request_id)
        request_state = self._request_states[request_id]
        if request_state.prefetch_job_id is not None:
            return None, False
        # Rebuild the local snapshot on resume. Store registration and open
        # windows outlive preemption and complete through the worker receipts.
        if request_state.num_preemptions != request.num_preemptions:
            request_state.external_hit_tokens = 0
            request_state.needs_load = False
            request_state.num_tokens = len(request.all_token_ids)
            request_state.common_chunk_hashes = make_common_ucm_chunk_ids(
                self._chunk_hasher(request)
            )
            request_state.num_preemptions = request.num_preemptions
        request_state.hbm_hit_tokens = num_computed_tokens
        if request.skip_reading_prefix_cache:
            return 0, False
        hit_tokens, prefetch_keys = self._lookup_hit_tokens(
            request_id,
            request_state.common_chunk_hashes,
            request_state.num_tokens,
            num_computed_tokens,
        )
        if prefetch_keys:
            job_id = self._allocate_job_id()
            try:
                prefetch_token = self._store_scheduler.prepare_prefetch(
                    request_id, prefetch_keys
                )
            except StoreIOError:
                logger.warning(
                    "UCM PreparePrefetch failed; using the current hit snapshot",
                    exc_info=True,
                )
            else:
                self._track_prepared_job(
                    job_id, JobOperation.PREFETCH, request_id, prefetch_token
                )
                request_state.prefetch_job_id = job_id
                return None, False
        external_hit_tokens = (
            hit_tokens if hit_tokens > self.load_threshold_tokens else 0
        )
        return external_hit_tokens, False

    def update_state_after_alloc(
        self, request: Request, num_external_tokens: int
    ) -> None:
        request_state = self._request_states[str(request.request_id)]
        if request_state.num_preemptions is not None:
            request_state.external_hit_tokens = num_external_tokens
            request_state.needs_load = num_external_tokens > 0

    def _lookup_key_groups(
        self,
        common_chunk_hashes: Sequence[bytes],
    ) -> dict[RestorePolicy, tuple[bytes, ...]]:
        return {
            policy: make_ucm_chunk_ids(common_chunk_hashes, policy)
            for policy in self.groups_by_policy
        }

    def _lookup_hit_tokens(
        self,
        request_id: str,
        common_chunk_hashes: Sequence[bytes],
        num_tokens: int,
        hbm_hit_tokens: int,
    ) -> tuple[int, tuple[bytes, ...]]:
        """Find the current shared restore boundary and candidates for prefetch."""
        chunk_size = self.spec.chunk_size
        first_chunk_index = hbm_hit_tokens // chunk_size
        max_end = max(num_tokens - self._recompute_tokens, 0) // chunk_size
        if not self.groups_by_policy or max_end <= first_chunk_index:
            return 0, ()
        key_groups = self._lookup_key_groups(
            common_chunk_hashes[first_chunk_index:max_end]
        )
        fa_keys = key_groups.get(RestorePolicy.FA, ())
        boundary_keys = tuple(
            keys
            for policy, keys in key_groups.items()
            if policy is not RestorePolicy.FA
        )
        end = max_end - first_chunk_index
        prefetch_keys = []
        try:
            if fa_keys:
                index, prefetchable = self._store_scheduler.lookup_on_prefix(
                    request_id, fa_keys
                )
                end = index + 1
                if prefetchable:
                    prefetch_keys.extend(fa_keys[end:])
            while end and boundary_keys:
                previous_end = end
                for keys in boundary_keys:
                    index, prefetchable = self._store_scheduler.lookup_on_reverse(
                        request_id, keys[:end]
                    )
                    if prefetchable:
                        prefetch_keys.extend(keys[index + 1 : end])
                    end = index + 1
                    if end == 0:
                        break
                # If a later policy moved the boundary, earlier policies
                # must also hit at the new position.
                if end == previous_end or len(boundary_keys) == 1:
                    break
        except StoreIOError:
            logger.warning("UCM lookup failed; recomputing this request", exc_info=True)
            return 0, ()
        hit_tokens = (first_chunk_index + end) * chunk_size - hbm_hit_tokens
        return max(hit_tokens, 0), tuple(prefetch_keys)

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
        plan: UCMDispatchPlan,
        operation: JobOperation,
    ) -> PreparedJob | None:
        ucm_keys = plan.ucm_keys
        if not ucm_keys:
            return None
        job_id = self._allocate_job_id()
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
        metadata = UCMConnectorMetadata()
        if self._needs_offload_drain:
            jobs = []
            for token in self._store_scheduler.prepare_offload():
                job = PreparedJob(self._allocate_job_id(), token.keys, token)
                self._offload_jobs[job.job_id] = job
                jobs.append(job)
            metadata.offload_jobs = tuple(jobs)
            self._needs_offload_drain = False
        for request_id in block_state.req_ids:
            request_state = self._request_states.get(request_id)
            if (
                request_state is None
                or request_state.num_preemptions
                != request_state.request.num_preemptions
                or request_state.finished
            ):
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
        for pending_job in self._pending_jobs.values():
            request_id = pending_job.request_id
            if (
                pending_job.operation is JobOperation.PREFETCH
                and not pending_job.dispatched
            ):
                metadata.requests[request_id] = replace(
                    metadata.requests.get(request_id, RequestMetadata(request_id)),
                    prefetch_job=pending_job.prepared_job,
                )
                pending_job.dispatched = True
        return metadata

    def request_finished(self, request_id: str) -> None:
        request_id = str(request_id)
        request_state = self._request_states.get(request_id)
        if request_state is None:
            return
        job_id = request_state.prefetch_job_id
        if job_id is not None and not self._pending_jobs[job_id].dispatched:
            self._store_scheduler.complete_prefetch(request_id, False)
            del self._pending_jobs[job_id]
            request_state.prefetch_job_id = None
        request_state.finished = True
        self._close_finished_request(request_id, request_state)

    def _complete_ready_jobs(self) -> None:
        """Close each prepared window after its own ranks reach terminal IO."""
        for job_id, pending_job in tuple(self._pending_jobs.items()):
            if pending_job.completed_ranks != self._expected_worker_ranks:
                continue
            request_id = pending_job.request_id
            operation = pending_job.operation
            prepared_job = pending_job.prepared_job
            success = not pending_job.failed
            if operation is JobOperation.GET:
                self._store_scheduler.complete_get(request_id, prepared_job.token)
            elif operation is JobOperation.PUT:
                self._store_scheduler.complete_put(
                    request_id, prepared_job.token, success
                )
                self._needs_offload_drain = True
            else:
                self._store_scheduler.complete_prefetch(request_id, success)
                self._needs_offload_drain = True
                self._request_states[request_id].prefetch_job_id = None
            del self._pending_jobs[job_id]
            self._close_finished_request(request_id, self._request_states[request_id])

    def _close_finished_request(self, request_id: str, state: RequestState) -> None:
        if state.finished and not any(
            job.request_id == request_id for job in self._pending_jobs.values()
        ):
            self._store_scheduler.on_request_finished(request_id)
            del self._request_states[request_id]

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        metadata = connector_output.kv_connector_worker_meta
        if metadata is None:
            return
        for request_id in metadata.failed_load_request_ids:
            request_state = self._request_states.get(request_id)
            if request_state is not None:
                # vLLM handles invalid blocks before delivering connector output.
                # Keep the request so subsequent recomputation can still dump KV.
                request_state.hbm_hit_tokens = request_state.request.num_computed_tokens
                request_state.external_hit_tokens = 0
                request_state.needs_load = False
        for job_result in metadata.job_results:
            offload_job = self._offload_jobs.get(job_result.job_id)
            if offload_job is not None:
                # Only the worker owning Store-only IO reports this job. The
                # receipt releases source pins on both success and failure.
                self._store_scheduler.complete_offload(offload_job.token)
                if not job_result.success:
                    logger.warning("UCM offload job %d failed", job_result.job_id)
                del self._offload_jobs[job_result.job_id]
                continue
            pending_job = self._pending_jobs.get(job_result.job_id)
            if pending_job is None:
                continue
            pending_job.completed_ranks.add(job_result.rank)
            pending_job.failed |= not job_result.success
            # KV derived from a failed load must not be published. Scope this
            # to the saves reported in this step, including other ranks' saves.
            if (
                pending_job.operation is JobOperation.PUT
                and pending_job.request_id in metadata.failed_load_request_ids
            ):
                pending_job.failed = True
        self._complete_ready_jobs()

    def has_pending_push_work(self) -> bool:
        return bool(
            self._needs_offload_drain
            or self._offload_jobs
            or any(
                job.operation is JobOperation.PREFETCH
                for job in self._pending_jobs.values()
            )
        )

    def shutdown(self) -> None:
        # An unacknowledged parent can still have Worker subjobs in flight.
        # Leave such reservations to backend recovery, never release them early.
        for request_id, state in tuple(self._request_states.items()):
            self._close_finished_request(request_id, state)

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
            ucm_keys = make_ucm_chunk_ids(
                (request_state.common_chunk_hashes[chunk_index],), RestorePolicy.STATE
            )
            plans.append(
                UCMDispatchPlan(
                    restore_policy=RestorePolicy.STATE,
                    ucm_keys=ucm_keys,
                    token_starts=np.asarray(
                        [chunk_index * chunk_size], dtype=np.uint64
                    ),
                    vllm_block_ids_by_group=tuple(
                        np.asarray([blocks_by_group[group_id]], dtype=np.uint64)
                        for group_id in group_ids
                    ),
                    group_ids=group_ids,
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
        # A bounded prefix replay rewinds the scheduler's compute cursor into
        # already cached KV. Forward rebuilds scratch windows there without
        # producing new persistent KV, so never dump the replayed prefix again.
        dump_start_token = max(step_start_token, load_end_token)
        dump_end_token = step_end_token
        load_plans = self.build_dispatch_plans(
            request_state,
            block_tables,
            load_start_token,
            load_end_token,
            is_load=True,
            scheduled_tokens=scheduled_tokens,
        )
        load_plans = tuple(
            replace(
                plan,
                prepared_job=self._prepare_plan_job(request_id, plan, JobOperation.GET),
            )
            for plan in load_plans
        )
        if any(plan.prepared_job is None for plan in load_plans):
            # Report the missing KV and acknowledge any Gets already prepared.
            return RequestMetadata(
                request_id, load_plans=load_plans, prepare_get_failed=True
            )
        dump_plans = self.build_dispatch_plans(
            request_state, block_tables, dump_start_token, dump_end_token, is_load=False
        ) + self._state_dump_plans(request_state, state_offloads)
        dump_plans = tuple(
            replace(
                plan,
                prepared_job=self._prepare_plan_job(request_id, plan, JobOperation.PUT),
            )
            for plan in dump_plans
        )
        return RequestMetadata(request_id, load_plans, dump_plans)

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
                # FA plans cover every complete UCM chunk the range touches.
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
        """Build the device ranges for one restore policy."""
        chunk_size = self.spec.chunk_size
        ucm_keys = make_ucm_chunk_ids(
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
            ucm_keys,
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
