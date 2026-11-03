"""Scheduler request state, lookup, dispatch planning and save publication."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
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
from .store_client import JOB_ID_STRIDE, PreparedJob, StoreClient, StoreIOError

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request

    from .runtime import UCMRuntimeContext

logger = init_logger(__name__)


@dataclass(frozen=True)
class UCMDispatchPlan:
    """Consecutive UCM chunks of one request for one restore policy.

    Plans carry common chunk hashes. PP rank and restore policy identify
    independent objects; workers resolve their own stage-local byte ranges.
    """

    restore_policy: RestorePolicy
    common_ucm_chunk_ids: tuple[bytes, ...]
    # First token of each UCM chunk (uint64), one per common UCM chunk id.
    token_starts: np.ndarray
    # Block-major IDs for the explicitly named native groups below.
    vllm_block_ids_by_group: tuple[np.ndarray, ...]
    group_ids: tuple[int, ...]


@dataclass(frozen=True)
class RequestMetadata:
    request_id: str
    load_plans: tuple[UCMDispatchPlan, ...] = ()
    dump_plans: tuple[UCMDispatchPlan, ...] = ()
    load_job: PreparedJob | None = None
    dump_job: PreparedJob | None = None
    prefetch_job: PreparedJob | None = None
    load_error: bool = False


@dataclass
class UCMConnectorMetadata(KVConnectorMetadata):
    requests: dict[str, RequestMetadata] = field(default_factory=dict)
    preempted_req_ids: set[str] = field(default_factory=set)
    finished_req_ids: set[str] = field(default_factory=set)
    no_forward: bool = False


@dataclass
class PendingJob:
    """One Prepare: all ranks report its parent job before Complete."""

    request_id: str
    operation: str
    job: PreparedJob
    completed_ranks: set[ParallelRank] = field(default_factory=set)
    failed: bool = False
    submitted: bool = False


@dataclass(frozen=True)
class UCMLookupResult:
    external_hit_tokens: int | None
    common_ucm_chunk_ids: tuple[bytes, ...]


@dataclass
class RequestState:
    """Mutable cross-step state, owned exclusively by ConnectorScheduler."""

    hbm_hit_tokens: int = 0
    external_hit_tokens: int = 0
    num_token_ids: int = 0
    token_processed: int = 0
    common_ucm_chunk_ids: tuple[bytes, ...] = ()
    # Cleared when the initial external-load plans have been built.
    need_load: bool = False


class UCMConnectorScheduler:
    def __init__(self, context: UCMRuntimeContext) -> None:
        self.context = context
        self.spec = context.spec
        self.parallel_topology = context.parallel_topology
        self._store = StoreClient(context.launch_config)
        self.groups_by_policy = groups_by_restore_policy(self.spec)
        self._recompute_tokens = 1
        self._chunk_hasher: Callable[[Request], list[bytes]] | None = None
        self.load_threshold_tokens = max(
            int(context.launch_config.get("load_tokens_threshold", 0)), 0
        )
        self.requests: dict[str, RequestState] = {}
        self._expected_ranks = {
            ParallelRank(tp, pp, pcp)
            for pp in range(self.parallel_topology.pp_size)
            for pcp in range(self.parallel_topology.pcp_size)
            for tp in range(self.parallel_topology.tp_size)
        }
        self._job_ids = count(JOB_ID_STRIDE, JOB_ID_STRIDE)
        self._pending_jobs: dict[int, PendingJob] = {}
        self._prefetch_jobs: dict[str, int] = {}
        self._live_requests: set[str] = set()
        self._stage_policies: dict[int, tuple[RestorePolicy, ...]] = {}
        self._ended_requests: set[str] = set()
        self._retired_requests: set[str] = set()

    def set_xfer_handshake_metadata(
        self, metadata: Mapping[int, UCMStorageHandshake]
    ) -> None:
        ranks = [worker.rank for worker in metadata.values()]
        if len(ranks) != len(set(ranks)) or set(ranks) != self._expected_ranks:
            raise ValueError(
                "Worker handshake must cover every expected rank exactly once"
            )
        if any(
            worker.fingerprint != self.context.fingerprint
            for worker in metadata.values()
        ):
            raise ValueError("Workers disagree on UCM fingerprint")
        if self._chunk_hasher is not None:
            return
        for pp_rank in range(self.parallel_topology.pp_size):
            stage = [
                worker for worker in metadata.values() if worker.rank.pp_rank == pp_rank
            ]
            policies = stage[0].stored_policies
            if any(worker.stored_policies != policies for worker in stage):
                raise ValueError("Workers disagree on PP stage storage policies")
            self._stage_policies[pp_rank] = policies
        hasher = RequestHasher(self.context.fingerprint)
        self._chunk_hasher = hasher.make_request_block_hasher(self.spec.chunk_size)

    def _ready(self) -> None:
        if self._chunk_hasher is None:
            raise RuntimeError(
                "UCM requires a complete worker storage handshake before scheduling"
            )

    def on_new_request(self, request: Request) -> None:
        request_id = str(request.request_id)
        native_context = self._store.module.RequestContext()
        native_context.request_id = request_id
        self._store.call(self._store.scheduler.on_new_request, native_context)
        self._live_requests.add(request_id)

    def lookup(self, request: Request, num_computed_tokens: int) -> UCMLookupResult:
        self._ready()
        request_id = str(request.request_id)
        common_ids = make_common_ucm_chunk_ids(self._chunk_hasher(request))
        if request_id in self._prefetch_jobs:
            return UCMLookupResult(None, common_ids)
        hit_tokens = self._lookup_hit_tokens(
            request_id, common_ids, len(request.all_token_ids), num_computed_tokens
        )
        if hit_tokens is None:
            job = self._prepare_job("prefetch", request_id)
            self._prefetch_jobs[request_id] = job.job_id
            return UCMLookupResult(None, common_ids)
        external_hit_tokens = (
            hit_tokens if hit_tokens > self.load_threshold_tokens else 0
        )
        self.requests[request_id] = RequestState(
            hbm_hit_tokens=num_computed_tokens,
            external_hit_tokens=external_hit_tokens,
            num_token_ids=len(request.all_token_ids),
            token_processed=num_computed_tokens + external_hit_tokens,
            common_ucm_chunk_ids=common_ids,
            need_load=external_hit_tokens > 0,
        )
        return UCMLookupResult(external_hit_tokens, common_ids)

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        result = self.lookup(request, num_computed_tokens)
        # IO may use asynchronous native jobs, but this connector drains loads
        # within the step. It does not claim cross-step asynchronous loading.
        return result.external_hit_tokens, False

    def _scan(self, request_id, common_ids, policy, operation):
        """A global hit requires the independent objects of every PP stage."""
        hits = []
        for pp_rank, policies in self._stage_policies.items():
            if policy not in policies:
                continue
            keys = make_ucm_chunk_ids(common_ids, policy, pp_rank)
            value = self._store.call(
                getattr(self._store.scheduler, f"lookup_on_{operation}"),
                request_id,
                b"".join(keys),
            )
            if value is None:
                return None
            hits.append(int(value))
        return min(hits, default=len(common_ids) - 1)

    def _lookup_hit_tokens(
        self,
        request_id: str,
        common_ids: Sequence[bytes],
        num_tokens: int,
        hbm_hit_tokens: int,
    ) -> int | None:
        chunk_size = self.spec.chunk_size
        first_chunk = hbm_hit_tokens // chunk_size
        end_chunk = max(num_tokens - self._recompute_tokens, 0) // chunk_size
        if not self.groups_by_policy or end_chunk <= first_chunk:
            return 0
        try:
            if RestorePolicy.FA in self.groups_by_policy:
                hit = self._scan(
                    request_id,
                    common_ids[first_chunk:end_chunk],
                    RestorePolicy.FA,
                    "prefix",
                )
                if hit is None:
                    return None
                end_chunk = first_chunk + hit + 1
            boundary_policies = tuple(
                policy
                for policy in self.groups_by_policy
                if policy is not RestorePolicy.FA
            )
            while end_chunk > first_chunk:
                previous_end = end_chunk
                for policy in boundary_policies:
                    hit = self._scan(
                        request_id, common_ids[first_chunk:end_chunk], policy, "reverse"
                    )
                    if hit is None:
                        return None
                    end_chunk = first_chunk + hit + 1
                    if end_chunk == first_chunk:
                        return 0
                if end_chunk == previous_end:
                    break
            return max(end_chunk * chunk_size - hbm_hit_tokens, 0)
        except StoreIOError:
            logger.warning("UCM lookup failed; recomputing this request", exc_info=True)
            return 0

    def _plan_keys(self, plans) -> tuple[bytes, ...]:
        return tuple(
            dict.fromkeys(
                key
                for plan in plans
                for pp_rank, policies in self._stage_policies.items()
                if plan.restore_policy in policies
                for key in make_ucm_chunk_ids(
                    plan.common_ucm_chunk_ids, plan.restore_policy, pp_rank
                )
            )
        )

    def _prepare_job(
        self, operation: str, request_id: str, keys: tuple[bytes, ...] = ()
    ) -> PreparedJob:
        token = self._store.prepare(operation, request_id, keys)
        job = PreparedJob(next(self._job_ids), token)
        self._pending_jobs[job.job_id] = PendingJob(request_id, operation, job)
        return job

    def _prepare_request(self, request: RequestMetadata) -> RequestMetadata:
        request_id = request.request_id
        if request.load_plans:
            try:
                job = self._prepare_job(
                    "get", request_id, self._plan_keys(request.load_plans)
                )
            except StoreIOError:
                logger.warning("UCM PrepareGet failed", exc_info=True)
                return replace(request, load_error=True, dump_plans=())
            request = replace(request, load_job=job)
        if request.dump_plans:
            try:
                job = self._prepare_job(
                    "put", request_id, self._plan_keys(request.dump_plans)
                )
            except StoreIOError:
                logger.warning("UCM PreparePut failed; skipping save", exc_info=True)
                return replace(request, dump_plans=())
            request = replace(request, dump_job=job)
        return request

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> UCMConnectorMetadata:
        self._ready()
        block_state = scheduler_output.kv_connector_block_state
        assert block_state is not None
        metadata = UCMConnectorMetadata(
            preempted_req_ids=set(scheduler_output.preempted_req_ids or ()),
            finished_req_ids=set(scheduler_output.finished_req_ids),
            no_forward=scheduler_output.total_num_scheduled_tokens == 0,
        )
        updates = []
        for request_id in block_state.req_ids:
            request_state = self.requests.get(request_id)
            if request_state is None:
                continue
            block_tables = block_state.get_block_ids(request_id)
            scheduled_tokens = scheduler_output.num_scheduled_tokens.get(request_id, 0)
            request_plan = self._build_request_plan(
                request_id,
                request_state,
                block_tables,
                scheduled_tokens,
                block_state.boundary_state_offloads.get(request_id, ()),
            )
            metadata.requests[request_id] = request_plan
            updates.append((request_state, scheduled_tokens))
        # Open windows only after every request's plan is valid.
        for request_id, plan in tuple(metadata.requests.items()):
            metadata.requests[request_id] = self._prepare_request(plan)
        for request_id, job_id in self._prefetch_jobs.items():
            pending = self._pending_jobs[job_id]
            if request_id not in self._ended_requests and not pending.submitted:
                metadata.requests[request_id] = RequestMetadata(
                    request_id, prefetch_job=pending.job
                )
                pending.submitted = True
        # Commit progress only after every request's plan has been built.
        for request_state, scheduled_tokens in updates:
            request_state.token_processed = min(
                request_state.token_processed + scheduled_tokens,
                request_state.num_token_ids,
            )
            if scheduled_tokens > 0:
                request_state.need_load = False
        for request_id in metadata.finished_req_ids:
            self.request_finished(request_id)
        # The finish callback may precede a final no-forward dump plan. Only
        # retire the store request once that plan has been prepared above.
        self._retired_requests.update(metadata.finished_req_ids)
        for request_id in metadata.preempted_req_ids:
            for pending in self._pending_jobs.values():
                if pending.operation == "put" and pending.request_id == request_id:
                    pending.failed = True
        self._complete_ready_jobs()
        self._close_finished_requests()
        for request_id in metadata.preempted_req_ids | metadata.finished_req_ids:
            self.requests.pop(request_id, None)
        return metadata

    def request_finished(self, request_id: str) -> None:
        request_id = str(request_id)
        self._ended_requests.add(request_id)
        job_id = self._prefetch_jobs.get(request_id)
        if job_id is not None and not self._pending_jobs[job_id].submitted:
            self._store.call(self._store.scheduler.complete_prefetch, request_id, False)
            del self._pending_jobs[job_id]
            del self._prefetch_jobs[request_id]
        self._complete_ready_jobs()
        self._close_finished_requests()

    def _complete_ready_jobs(self) -> None:
        for job_id, pending in tuple(self._pending_jobs.items()):
            if pending.completed_ranks != self._expected_ranks:
                continue
            complete = getattr(self._store.scheduler, f"complete_{pending.operation}")
            if pending.operation == "get":
                self._store.call(complete, pending.request_id)
            else:
                self._store.call(complete, pending.request_id, not pending.failed)
            if pending.operation == "prefetch":
                del self._prefetch_jobs[pending.request_id]
            del self._pending_jobs[job_id]

    def _close_finished_requests(self) -> None:
        active = {pending.request_id for pending in self._pending_jobs.values()}
        for request_id in tuple(self._retired_requests):
            if request_id in active:
                continue
            if request_id in self._live_requests:
                self._store.call(self._store.scheduler.on_request_finished, request_id)
                self._live_requests.remove(request_id)
            self._ended_requests.discard(request_id)
            self._retired_requests.remove(request_id)

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        metadata = connector_output.kv_connector_worker_meta
        if metadata is None:
            return
        # A load failure invalidates this request's saves even if another rank
        # has already reported a locally successful Put.
        for request_id in metadata.load_failed_reqs:
            self.requests.pop(request_id, None)
            for pending in self._pending_jobs.values():
                if pending.operation == "put" and pending.request_id == request_id:
                    pending.failed = True
        for result in metadata.job_results:
            pending = self._pending_jobs.get(result.job_id)
            if pending is None:
                continue
            pending.completed_ranks.add(result.rank)
            pending.failed |= not result.success
        self._complete_ready_jobs()
        self._close_finished_requests()

    def has_pending_push_work(self) -> bool:
        return bool(self._prefetch_jobs)

    def shutdown(self) -> None:
        # An unacknowledged parent can still have Worker subjobs in flight.
        # Leave such reservations to backend recovery, never release them early.
        for pending in self._pending_jobs.values():
            pending.failed = True
        self._complete_ready_jobs()
        self._close_finished_requests()
        self._store.close()

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
            # engine hand-offs can arrive after token_processed advances.
            if (
                boundary_token <= matched_prefix_tokens
                or boundary_token > request_state.num_token_ids
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
            plans.append(
                UCMDispatchPlan(
                    restore_policy=RestorePolicy.STATE,
                    common_ucm_chunk_ids=(
                        request_state.common_ucm_chunk_ids[chunk_index],
                    ),
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
        """Calculate plans without advancing the request state."""
        step_end_token = min(
            request_state.token_processed + scheduled_tokens,
            request_state.num_token_ids,
        )
        should_load = request_state.need_load and scheduled_tokens > 0
        load_end_token = (
            request_state.hbm_hit_tokens + request_state.external_hit_tokens
        )
        load_start_token = (
            request_state.hbm_hit_tokens if should_load else load_end_token
        )
        dump_start_token = request_state.token_processed
        dump_end_token = step_end_token
        load_plans = self.build_dispatch_plans(
            request_state,
            block_tables,
            load_start_token,
            load_end_token,
            is_load=True,
            scheduled_tokens=scheduled_tokens,
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
                start_chunk = end_chunk_index - 1
                end_chunk = end_chunk_index
            else:
                # FA chunks are stored once and never re-stored, so a plan
                # covers every complete UCM chunk the range touches -- not
                # just the newest boundary.
                start_chunk = first_chunk_index
                end_chunk = end_chunk_index
            plans.append(
                self.make_dispatch_plan(
                    restore_policy,
                    groups,
                    request_state,
                    block_tables,
                    start_chunk,
                    end_chunk,
                    # State loads fill the running slot for this step first.
                    # The snapshot itself still belongs to token_end.
                    state_target_tokens=token_end + scheduled_tokens,
                )
            )
            if restore_policy is RestorePolicy.STATE:
                # Precopy has already run, so restoring only the imported
                # boundary cannot initialize the running state. Restoring
                # only the running block leaves a local cached boundary unfilled.
                # Complete restore boundaries and a positive scheduled span
                # place these in distinct slots for every State group.
                plans.append(
                    self.make_dispatch_plan(
                        restore_policy,
                        groups,
                        request_state,
                        block_tables,
                        start_chunk,
                        end_chunk,
                        token_end,
                    )
                )
        return tuple(plans)

    def make_dispatch_plan(
        self,
        restore_policy: RestorePolicy,
        groups: Sequence[UCMKVCacheGroupSpec],
        request_state: RequestState,
        block_tables: Sequence[Sequence[int]],
        start_chunk: int,
        end_chunk: int,
        state_target_tokens: int | None,
    ) -> UCMDispatchPlan:
        """Plan for UCM chunks [start_chunk, end_chunk) of one restore policy."""
        chunk_size = self.spec.chunk_size
        return UCMDispatchPlan(
            restore_policy,
            tuple(request_state.common_ucm_chunk_ids[start_chunk:end_chunk]),
            np.arange(start_chunk, end_chunk, dtype=np.uint64) * np.uint64(chunk_size),
            tuple(
                self._group_blocks(
                    restore_policy,
                    group,
                    block_tables[group.group_id],
                    start_chunk,
                    end_chunk,
                    state_target_tokens=state_target_tokens,
                )
                for group in groups
            ),
            tuple(group.group_id for group in groups),
        )

    def _group_blocks(
        self,
        restore_policy: RestorePolicy,
        group: UCMKVCacheGroupSpec,
        vllm_block_table: Sequence[int],
        start_chunk: int,
        end_chunk: int,
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
                np.arange(start_chunk + 1, end_chunk + 1, dtype=np.uint64) * chunk_size
                - 1
            )
        else:
            # WA: one UCM chunk at the plan's newest boundary.
            boundary_tokens = np.asarray([end_chunk * chunk_size - 1], dtype=np.uint64)
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
