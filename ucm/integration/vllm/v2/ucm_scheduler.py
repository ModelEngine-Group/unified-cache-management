"""Unified scheduler-side lookup, request state, and dispatch planning."""

from __future__ import annotations

import enum
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata

from ucm.integration.vllm.request_hasher import RequestHasher

from .layout import UCMKVCacheGroupInfo, UCMKVCacheSpec
from .ucm_proxy import UCMProxyAdapter

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request


class AttentionType(enum.Enum):
    """What one UCM record stores; the value names it in dump metadata."""

    FA = "FA"  # full-attention groups
    WA = "WA"  # sliding-window groups that re-store a window tail
    STATE = "State"  # mamba state snapshot groups


def groups_by_attention_type(
    spec: UCMKVCacheSpec,
) -> tuple[tuple[AttentionType, tuple["UCMKVCacheGroupInfo", ...]], ...]:
    """Attention types in dispatch order, each with the groups it stores.

    FA holds the full-attention groups; WA the sliding groups that
    re-store a window tail (tail 0 groups store nothing); State the
    mamba snapshot groups. Absent attention types are omitted, and
    ``ucm_block_ids_by_attention_type`` / dispatch plans index attention
    types in this order.
    """

    result: list[tuple[AttentionType, tuple["UCMKVCacheGroupInfo", ...]]] = []
    if spec.fa_groups:
        result.append((AttentionType.FA, spec.fa_groups))
    wa_stored = tuple(group for group in spec.wa_groups if (group.tail_tokens or 0) > 0)
    if wa_stored:
        result.append((AttentionType.WA, wa_stored))
    if spec.state_groups:
        result.append((AttentionType.STATE, spec.state_groups))
    return tuple(result)


@dataclass(frozen=True)
class UCMLookupResult:
    external_hit_tokens: int
    ucm_block_ids_by_attention_type: tuple[tuple[bytes, ...], ...]


@dataclass
class RequestState:
    hbm_hit_tokens: int = 0
    external_hit_tokens: int = 0
    num_token_ids: int = 0
    token_processed: int = 0
    ucm_block_ids_by_attention_type: tuple[tuple[bytes, ...], ...] = ()
    vllm_block_tables: tuple[list[int], ...] = ()
    load_pending: bool = False


@dataclass(frozen=True)
class UCMGroupDispatchPlan:
    """Consecutive UCM blocks of one request for one attention type.

    ucm_block_ids are unique within a plan.
    """

    attention_type: AttentionType
    ucm_block_ids: tuple[bytes, ...]
    token_start: int
    token_end: int
    # vLLM block IDs per participating group, in groups_by_attention_type()
    # order, block-major (tail_blocks blocks per UCM block id) -- the static
    # window shape lives on the spec's groups (``tail_blocks``), so
    # only the ids travel.
    vllm_block_ids_by_group: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class RequestDispatchMeta:
    request_id: str
    load_plans: tuple[UCMGroupDispatchPlan, ...] = ()
    dump_plans: tuple[UCMGroupDispatchPlan, ...] = ()


@dataclass
class UCMConnectorMetadata(KVConnectorMetadata):
    requests: dict[str, RequestDispatchMeta] = field(default_factory=dict)
    preempted_req_ids: set[str] = field(default_factory=set)
    finished_req_ids: set[str] = field(default_factory=set)
    no_forward: bool = False


_ATTENTION_TYPE_BITS: Mapping[AttentionType, int] = {
    AttentionType.FA: 0,
    AttentionType.WA: 1,
    AttentionType.STATE: 2,
}


def _ucm_block_id_tag(
    attention_type: AttentionType, tp_rank: int = 0, pp_rank: int = 0
) -> bytes:
    """The 2-byte suffix of a UCM block id, big-endian bit layout:

    type(2) group(4) tp_rank(4) pp_rank(4) reserved(2).  All groups of a
    attention type share one record, so no group bits are set today; the layout
    keeps them for a future per-group split.  Ranks default to the
    logical rank-0 namespace the scheduler hashes in.
    """

    value = (
        (_ATTENTION_TYPE_BITS[attention_type] << 14)
        | ((tp_rank & 0b1111) << 6)
        | ((pp_rank & 0b1111) << 2)
    )
    return value.to_bytes(2, "big")


class UCMDispatcher:
    """Scheduler-side lookup, request state, and dispatch planning.

    ``lookup`` probes the external cache and records the per-request
    snapshot; ``build_from_scheduler_output`` then maintains block tables
    from vLLM's SchedulerOutput and produces pointer-free plans.
    """

    def __init__(
        self,
        kv_cache_spec: UCMKVCacheSpec,
        proxy: UCMProxyAdapter,
        request_hasher: RequestHasher,
        base_seed: bytes,
        *,
        load_threshold_tokens: int = 0,
        recompute_tokens: int = 1,
        tp_rank: int = 0,
        pp_rank: int = 0,
        need_load_to_running_state: bool = True,
    ) -> None:
        self.spec = kv_cache_spec
        # Whether State loads must also fill the running block. True when the
        # model runner copies Mamba state from the boundary block into the
        # running block before connector loads (CUDA V1, CUDA/Ascend V2), so
        # that copy saw no restored data. False when it copies after loads
        # (Ascend V1), filling the running block from the restored boundary.
        self.need_load_to_running_state = need_load_to_running_state
        self.proxy = proxy
        # Folds vLLM's block-hash extra keys (multimodal inputs, LoRA, cache
        # salt, prompt embeds) into the hash, so equal token IDs with
        # different KV-affecting inputs get different UCM block ids.
        self._block_hasher = request_hasher.make_request_block_hasher(
            kv_cache_spec.ucm_cache_block_size, base_seed
        )
        self.load_threshold_tokens = max(int(load_threshold_tokens), 0)
        self.recompute_tokens = max(int(recompute_tokens), 0)
        self.requests: dict[str, RequestState] = {}
        # Attention type -> participating groups, built once: FA, then the
        # tail-storing WA groups, then State.
        self._groups_by_attention_type = groups_by_attention_type(kv_cache_spec)
        self._attention_type_tags = tuple(
            (attention_type, _ucm_block_id_tag(attention_type, tp_rank, pp_rank))
            for attention_type, _ in self._groups_by_attention_type
        )

    def _ucm_block_ids_by_attention_type(
        self, request: "Request"
    ) -> tuple[tuple[bytes, ...], ...]:
        """UCM block ids per attention type: shared block hashes plus a type tag.

        Hashes cover ucm_cache_block_size token blocks from the base seed; a
        UCM block id is a hash's first 14 bytes plus the attention type's 2-byte
        tag, so FA, WA and State ids at one boundary differ only in the tag.
        """

        hashes = self._block_hasher(request)
        return tuple(
            tuple(block_hash[:14] + tag for block_hash in hashes)
            for _attention_type, tag in self._attention_type_tags
        )

    def lookup(self, request: "Request", num_computed_tokens: int) -> UCMLookupResult:
        """Probe the external cache and record the request snapshot."""
        result = self._lookup(request, num_computed_tokens)
        if result.external_hit_tokens <= self.load_threshold_tokens:
            result = UCMLookupResult(0, result.ucm_block_ids_by_attention_type)
        self.requests[str(request.request_id)] = RequestState(
            hbm_hit_tokens=num_computed_tokens,
            external_hit_tokens=result.external_hit_tokens,
            num_token_ids=len(request.all_token_ids),
            token_processed=num_computed_tokens + result.external_hit_tokens,
            ucm_block_ids_by_attention_type=result.ucm_block_ids_by_attention_type,
            vllm_block_tables=tuple([] for _ in self.spec.groups),
            load_pending=result.external_hit_tokens > 0,
        )
        return result

    def _lookup(self, request: "Request", hbm: int) -> UCMLookupResult:
        """FA prefix restore; WA/State additionally need their boundary.

        Every attention type has a UCM block id at each ucm_cache_block_size
        boundary.  FA ids must hit as a contiguous prefix; WA (window tail)
        and State (mamba snapshot) ids are boundary records -- restoring
        requires a complete FA prefix up to some boundary plus that
        boundary's WA and State records, all at the same restore boundary.
        """

        ucm_block_size = self.spec.ucm_cache_block_size
        ucm_block_ids_by_attention_type = self._ucm_block_ids_by_attention_type(request)
        # Every attention type scans up to the last complete block below the
        # recompute margin; the margin's block is left to recompute (v1
        # semantics -- a full hit still recomputes recompute_tokens).
        num_tokens = len(request.all_token_ids)
        last = max(num_tokens - self.recompute_tokens, 0) // ucm_block_size
        first = hbm // ucm_block_size
        fa_end = hbm
        for (attention_type, _tag), ucm_block_ids in zip(
            self._attention_type_tags, ucm_block_ids_by_attention_type
        ):
            if attention_type is AttentionType.FA:
                hits = self.proxy.lookup_on_prefix(
                    ucm_block_ids[first:last], np.arange(first, last, dtype=np.int64)
                )
                if hits >= 0:
                    fa_end = max((first + hits + 1) * ucm_block_size, hbm)
                break
        boundary_ucm_block_ids = [
            ucm_block_ids
            for (attention_type, _tag), ucm_block_ids in zip(
                self._attention_type_tags, ucm_block_ids_by_attention_type
            )
            if attention_type is not AttentionType.FA
        ]
        restore_end = fa_end
        while boundary_ucm_block_ids and restore_end > hbm:
            end = restore_end // ucm_block_size
            hits = [
                self.proxy.lookup_on_reverse(
                    ucm_block_ids[first:end], np.arange(first, end, dtype=np.int64)
                )
                for ucm_block_ids in boundary_ucm_block_ids
            ]
            if min(hits) < 0:
                restore_end = hbm
                break
            candidate = (first + min(hits) + 1) * ucm_block_size
            if candidate == restore_end:
                break
            restore_end = candidate
        return UCMLookupResult(
            max(restore_end - hbm, 0), ucm_block_ids_by_attention_type
        )

    def update_blocks(
        self,
        request_id: str,
        vllm_block_tables: Sequence[Sequence[int]],
        *,
        append: bool,
    ) -> None:
        state = self.requests[request_id]
        if append:
            for destination, source in zip(state.vllm_block_tables, vllm_block_tables):
                destination.extend(int(value) for value in source)
        else:
            state.vllm_block_tables = tuple(
                [int(value) for value in source] for source in vllm_block_tables
            )

    def build_from_scheduler_output(
        self, scheduler_output: "SchedulerOutput"
    ) -> UCMConnectorMetadata:
        """Consume 0.30's scheduler-local tables and exact State hand-offs."""
        block_state = scheduler_output.kv_connector_block_state
        assert block_state is not None
        metadata = UCMConnectorMetadata(
            preempted_req_ids=set(scheduler_output.preempted_req_ids or ()),
            finished_req_ids=set(scheduler_output.finished_req_ids),
            no_forward=scheduler_output.total_num_scheduled_tokens == 0,
        )
        for request_id in block_state.req_ids:
            state = self.requests.get(request_id)
            if state is None:
                continue
            vllm_block_tables = block_state.get_block_ids(request_id)
            self.update_blocks(request_id, vllm_block_tables, append=False)
            scheduled = scheduler_output.num_scheduled_tokens.get(request_id, 0)
            request_meta = self._request_meta(request_id, state, scheduled)
            snapshots = self._state_dump_plans(
                state, block_state.boundary_state_offloads.get(request_id, ())
            )
            metadata.requests[request_id] = RequestDispatchMeta(
                request_id, request_meta.load_plans, request_meta.dump_plans + snapshots
            )
        for request_id in metadata.preempted_req_ids | metadata.finished_req_ids:
            self.requests.pop(request_id, None)
        return metadata

    def _state_dump_plans(
        self, state: RequestState, offloads: Sequence[tuple[int, int, int]]
    ) -> tuple[UCMGroupDispatchPlan, ...]:
        """Publish complete checkpoints beyond the initially matched prefix."""
        if not self.spec.state_groups:
            return ()
        type_index = next(
            i
            for i, (attention_type, _) in enumerate(self._groups_by_attention_type)
            if attention_type is AttentionType.STATE
        )
        groups = self._groups_by_attention_type[type_index][1]
        ucm_block_size = self.spec.ucm_cache_block_size
        matched_prefix = state.hbm_hit_tokens + state.external_hit_tokens
        boundaries: dict[int, dict[int, int]] = {}
        for group_id, vllm_block_id, boundary in offloads:
            # Engine offers include newly hashed imported State blocks. Their
            # prefix is already cached; do not dump it back to the store.
            # Use the initial hit boundary, not token_processed: a checkpoint
            # computed after that hit can be offered on a later step.
            if (
                matched_prefix < boundary <= state.num_token_ids
                and boundary % ucm_block_size == 0
            ):
                boundaries.setdefault(boundary, {})[group_id] = vllm_block_id
        plans = []
        for boundary, vllm_block_ids in sorted(boundaries.items()):
            if not all(group.group_id in vllm_block_ids for group in groups):
                continue
            ucm_block_id = state.ucm_block_ids_by_attention_type[type_index][
                boundary // ucm_block_size - 1
            ]
            plans.append(
                UCMGroupDispatchPlan(
                    AttentionType.STATE,
                    (ucm_block_id,),
                    boundary - ucm_block_size,
                    boundary,
                    tuple(
                        np.asarray([vllm_block_ids[group.group_id]], dtype=np.uint64)
                        for group in groups
                    ),
                )
            )
        return tuple(plans)

    def _request_meta(
        self, request_id: str, state: RequestState, scheduled_tokens: int
    ) -> RequestDispatchMeta:
        step_end = min(state.token_processed + scheduled_tokens, state.num_token_ids)
        should_load = state.load_pending and scheduled_tokens > 0
        load_end = state.hbm_hit_tokens + state.external_hit_tokens
        load_start = state.hbm_hit_tokens if should_load else load_end
        dump_start = state.token_processed
        dump_end = step_end
        # Load the running block when the runner's precopy already ran, plus
        # (below) the imported boundary block, which the engine registers as a
        # reusable local checkpoint. Otherwise load only the boundary block the
        # runner copies from after loads. Imported boundaries are not dumped.
        state_target_tokens = (
            load_end + scheduled_tokens if self.need_load_to_running_state else load_end
        )
        load = self._plans(
            state,
            load_start,
            load_end,
            state_target_tokens=state_target_tokens,
        )
        if should_load:
            state.load_pending = False
        dump = self._plans(state, dump_start, dump_end)
        state.token_processed = step_end
        return RequestDispatchMeta(request_id, load, dump)

    def _plans(
        self,
        state: RequestState,
        token_start: int,
        token_end: int,
        *,
        state_target_tokens: int | None = None,
    ) -> tuple[UCMGroupDispatchPlan, ...]:
        """FA/WA plans and runner-specific State load destinations.

        FA gets every complete UCM block the range touches; WA gets the newest
        complete boundary. State loads use that restore boundary's UCM block
        and its load destinations: the running block and imported checkpoint
        when need_load_to_running_state, else only the checkpoint the runner
        copies from. State dumps are built separately
        from the engine's exact boundary_state_offloads.
        """
        plans: list[UCMGroupDispatchPlan] = []
        if token_end <= token_start:
            return ()
        ucm_block_size = self.spec.ucm_cache_block_size
        # Type-invariant UCM block bounds: the first block the range touches
        # and the boundary it ends on.  No complete UCM block in range means
        # no plan for any attention type: FA has nothing to store or load, and the
        # WA/State boundary was not reached.
        first_block_index = token_start // ucm_block_size
        end_block_index = token_end // ucm_block_size
        if end_block_index <= first_block_index:
            return ()
        for type_index, (attention_type, groups) in enumerate(
            self._groups_by_attention_type
        ):
            if attention_type is AttentionType.STATE and state_target_tokens is None:
                continue
            ucm_block_ids = state.ucm_block_ids_by_attention_type[type_index]
            if attention_type in (AttentionType.WA, AttentionType.STATE):
                # WA keeps the newest completed boundary; a State load
                # ends exactly at the selected restore boundary.
                start = end_block_index - 1
                end = end_block_index
            else:
                # FA blocks are stored once and never re-stored, so a plan
                # covers every complete UCM block the range touches -- not
                # just the newest boundary.
                start = first_block_index
                end = end_block_index
            if attention_type is AttentionType.WA:
                # A tail window is only worth storing once it is
                # complete: an early boundary shorter than its groups'
                # largest tail would clamp the window head to token 0 --
                # the request's own prefix, which a restore recomputes
                # anyway.  Skipping keeps every dumped UCM block's record
                # complete, so reverse lookup never selects a partial
                # one (the load branch above cannot select one either).
                if end * ucm_block_size < max(
                    group.tail_tokens or 0 for group in groups
                ):
                    continue
            plan = UCMGroupDispatchPlan(
                attention_type,
                tuple(ucm_block_ids[start:end]),
                start * ucm_block_size,
                end * ucm_block_size,
                tuple(
                    self._group_blocks(
                        attention_type,
                        group,
                        state,
                        start,
                        end,
                        state_target_tokens=state_target_tokens,
                    )
                    for group in groups
                ),
            )
            plans.append(plan)
            if (
                attention_type is AttentionType.STATE
                and self.need_load_to_running_state
            ):
                # Precopy has already run, so restoring only the imported
                # boundary cannot initialize the running state. Restoring
                # only the running block leaves a local cached boundary unfilled.
                # Complete restore boundaries and a positive scheduled span
                # place these in distinct slots for every State group.
                plans.append(
                    replace(
                        plan,
                        vllm_block_ids_by_group=tuple(
                            self._group_blocks(
                                attention_type,
                                group,
                                state,
                                start,
                                end,
                                state_target_tokens=token_end,
                            )
                            for group in groups
                        ),
                    )
                )
        return tuple(plans)

    def _group_blocks(
        self,
        attention_type: AttentionType,
        group: UCMKVCacheGroupInfo,
        state: RequestState,
        start: int,
        end: int,
        *,
        state_target_tokens: int | None = None,
    ) -> np.ndarray:
        """One group's window vLLM block ids for the plan's UCM blocks.

        Every window is ``tail_blocks`` vLLM blocks ending at the block
        containing the window's last token -- HMA's boundary indices: FA UCM
        block k's last token is (k + 1) * ucm_block_size - 1; a WA UCM block
        anchors its window at the restore boundary. State uses a platform
        load slot. The _plans clamp gate keeps the WA gather non-negative and
        the parse-time divisibility checks keep an FA UCM block inside one
        shared vLLM block.  Block-major per UCM block.
        """

        ucm_block_size = self.spec.ucm_cache_block_size
        table = state.vllm_block_tables[group.group_id]
        tail_blocks = group.tail_blocks
        if attention_type is AttentionType.STATE:
            # The caller picks the running or boundary slot from the
            # runner's load/copy ordering.
            index = (state_target_tokens - 1) // group.token_block_size
            vllm_block_id = table[index]
            assert vllm_block_id != 0
            return np.asarray([vllm_block_id], dtype=np.uint64)
        if attention_type is AttentionType.FA:
            boundary_tokens = (
                np.arange(start + 1, end + 1, dtype=np.uint64) * ucm_block_size - 1
            )
        else:
            # WA: one UCM block at the plan's newest boundary.
            boundary_tokens = np.asarray([end * ucm_block_size - 1], dtype=np.uint64)
        boundary_block_idx = boundary_tokens // group.token_block_size
        # Convert only the table range the windows touch -- a WA group's
        # table spans the whole request while its window holds a couple
        # of blocks.  The clamp gate and the FA divisibility checks keep
        # first_idx non-negative.
        first_idx = int(boundary_block_idx.min()) - (tail_blocks - 1)
        last_idx = int(boundary_block_idx.max())
        vllm_block_ids = np.asarray(table[first_idx : last_idx + 1], dtype=np.uint64)
        within = boundary_block_idx - first_idx
        if tail_blocks == 1:
            return vllm_block_ids[within]
        steps_back = np.arange(tail_blocks, dtype=np.uint64)[::-1]
        return vllm_block_ids[(within[:, None] - steps_back[None, :]).reshape(-1)]
