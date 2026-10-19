"""Scheduler-side request state and dispatch planning over common UCM block ids."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata

from ucm.integration.vllm.request_hasher import RequestHasher

from .layout import UCMKVCacheGroupInfo, UCMKVCacheSpec
from .ucm_block_id import AttentionType, make_common_ucm_block_ids
from .ucm_sharding import groups_by_attention_type

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request

    from .ucm_lookup import UCMLookup


def make_request_hasher(vllm_config, rank_id: int) -> RequestHasher:
    """Build the request hasher with the v2 namespace metadata.

    The metadata is the same as before RequestHasher took a prebuilt string,
    so block hashes do not change. v2 adds its own layout and parallel
    namespace on top of this seed.
    """
    speculative_config = getattr(vllm_config, "speculative_config", None)
    spec_info = ""
    if speculative_config is not None:
        spec_method = getattr(speculative_config, "method", "") or ""
        spec_tokens = getattr(speculative_config, "num_speculative_tokens", 0)
        spec_info = f":{spec_method}:{spec_tokens}"
    additional_config = getattr(vllm_config, "additional_config", None) or {}
    sparse_sfa_c8 = bool(additional_config.get("enable_sparse_sfa_c8", False))
    sparse_li_c8 = bool(additional_config.get("enable_sparse_li_c8", False))
    sparse_c8_info = f":sfa_c8={int(sparse_sfa_c8)}:li_c8={int(sparse_li_c8)}"
    model_name = vllm_config.model_config.model.rstrip("/").split("/")[-1]
    meta = (
        f"{model_name}:"
        f"{vllm_config.parallel_config.tensor_parallel_size}:"
        f"{vllm_config.model_config.dtype}:{rank_id}{spec_info}{sparse_c8_info}"
    )
    return RequestHasher(meta)


@dataclass(frozen=True)
class UCMLookupResult:
    external_hit_tokens: int
    common_ucm_block_ids: tuple[bytes, ...]


@dataclass
class RequestState:
    hbm_hit_tokens: int = 0
    external_hit_tokens: int = 0
    num_token_ids: int = 0
    token_processed: int = 0
    # One common UCM block id per full UCM block, shared by all attention
    # types.
    common_ucm_block_ids: tuple[bytes, ...] = ()
    vllm_block_tables: tuple[list[int], ...] = ()
    load_pending: bool = False


@dataclass(frozen=True)
class UCMGroupDispatchPlan:
    """Consecutive UCM blocks of one request for one attention type.

    Plans carry common UCM block ids. Each worker appends the tag of its own
    record classes and shards to get its local UCM block ids.
    """

    attention_type: AttentionType
    common_ucm_block_ids: tuple[bytes, ...]
    token_start: int
    token_end: int
    # vLLM block IDs per participating group, in groups_by_attention_type()
    # order, block-major (tail_blocks blocks per UCM block) -- the static
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


class UCMDispatcher:
    """Scheduler-side request state and dispatch planning.

    lookup hashes a request, asks UCMLookup how many tokens can be loaded and
    saves the request state. build_from_scheduler_output updates block tables
    from vLLM's SchedulerOutput and builds the plans. The dispatcher does not
    access the Store.
    """

    def __init__(
        self,
        kv_cache_spec: UCMKVCacheSpec,
        lookup: "UCMLookup",
        request_hasher: RequestHasher,
        base_seed: bytes,
        *,
        load_threshold_tokens: int = 0,
        need_load_to_running_state: bool = True,
    ) -> None:
        self.spec = kv_cache_spec
        # Whether State loads must also fill the running block. True when the
        # model runner copies Mamba state from the boundary block into the
        # running block before connector loads (CUDA V1, CUDA/Ascend V2), so
        # that copy saw no restored data. False when it copies after loads
        # (Ascend V1), filling the running block from the restored boundary.
        self.need_load_to_running_state = need_load_to_running_state
        self._lookup = lookup
        # Folds vLLM's block-hash extra keys (multimodal inputs, LoRA, cache
        # salt, prompt embeds) into the hash, so equal token IDs with
        # different KV-affecting inputs get different UCM block ids.
        self._block_hasher = request_hasher.make_request_block_hasher(
            kv_cache_spec.ucm_cache_block_size, base_seed
        )
        self.load_threshold_tokens = max(int(load_threshold_tokens), 0)
        self.requests: dict[str, RequestState] = {}
        # Attention type -> participating groups, built once: FA, then the
        # tail-storing WA groups, then State.
        self._groups_by_attention_type = groups_by_attention_type(kv_cache_spec)

    def lookup(self, request: "Request", num_computed_tokens: int) -> UCMLookupResult:
        """Probe the external cache and record the request snapshot."""
        common_ucm_block_ids = make_common_ucm_block_ids(self._block_hasher(request))
        hit_tokens = self._lookup.hit_tokens(
            common_ucm_block_ids, len(request.all_token_ids), num_computed_tokens
        )
        if hit_tokens > self.load_threshold_tokens:
            external_hit_tokens = hit_tokens
        else:
            external_hit_tokens = 0
        self.requests[str(request.request_id)] = RequestState(
            hbm_hit_tokens=num_computed_tokens,
            external_hit_tokens=external_hit_tokens,
            num_token_ids=len(request.all_token_ids),
            token_processed=num_computed_tokens + external_hit_tokens,
            common_ucm_block_ids=common_ucm_block_ids,
            vllm_block_tables=tuple([] for _ in self.spec.groups),
            load_pending=external_hit_tokens > 0,
        )
        return UCMLookupResult(external_hit_tokens, common_ucm_block_ids)

    def update_blocks(
        self,
        request_id: str,
        vllm_block_tables: Sequence[Sequence[int]],
        *,
        append: bool,
    ) -> None:
        state = self.requests[request_id]
        if append:
            for current_table, new_table in zip(
                state.vllm_block_tables, vllm_block_tables
            ):
                current_table.extend(int(vllm_block_id) for vllm_block_id in new_table)
        else:
            state.vllm_block_tables = tuple(
                [int(vllm_block_id) for vllm_block_id in new_table]
                for new_table in vllm_block_tables
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
            scheduled_tokens = scheduler_output.num_scheduled_tokens.get(request_id, 0)
            request_meta = self._request_meta(request_id, state, scheduled_tokens)
            state_dump_plans = self._state_dump_plans(
                state, block_state.boundary_state_offloads.get(request_id, ())
            )
            metadata.requests[request_id] = RequestDispatchMeta(
                request_id,
                request_meta.load_plans,
                request_meta.dump_plans + state_dump_plans,
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
        state_groups = dict(self._groups_by_attention_type)[AttentionType.STATE]
        ucm_block_size = self.spec.ucm_cache_block_size
        matched_prefix_tokens = state.hbm_hit_tokens + state.external_hit_tokens
        vllm_block_ids_by_boundary: dict[int, dict[int, int]] = {}
        for group_id, vllm_block_id, boundary_token in offloads:
            # Engine offers include newly hashed imported State blocks. Their
            # prefix is already cached; do not dump it back to the store.
            # Use the initial hit boundary, not token_processed: a checkpoint
            # computed after that hit can be offered on a later step.
            if (
                matched_prefix_tokens < boundary_token <= state.num_token_ids
                and boundary_token % ucm_block_size == 0
            ):
                vllm_block_ids_by_boundary.setdefault(boundary_token, {})[
                    group_id
                ] = vllm_block_id
        plans = []
        for boundary_token, vllm_block_id_by_group in sorted(
            vllm_block_ids_by_boundary.items()
        ):
            if not all(
                group.group_id in vllm_block_id_by_group for group in state_groups
            ):
                continue
            plans.append(
                UCMGroupDispatchPlan(
                    AttentionType.STATE,
                    (state.common_ucm_block_ids[boundary_token // ucm_block_size - 1],),
                    boundary_token - ucm_block_size,
                    boundary_token,
                    tuple(
                        np.asarray(
                            [vllm_block_id_by_group[group.group_id]], dtype=np.uint64
                        )
                        for group in state_groups
                    ),
                )
            )
        return tuple(plans)

    def _request_meta(
        self, request_id: str, state: RequestState, scheduled_tokens: int
    ) -> RequestDispatchMeta:
        step_end_token = min(
            state.token_processed + scheduled_tokens, state.num_token_ids
        )
        should_load = state.load_pending and scheduled_tokens > 0
        load_end_token = state.hbm_hit_tokens + state.external_hit_tokens
        load_start_token = state.hbm_hit_tokens if should_load else load_end_token
        dump_start_token = state.token_processed
        dump_end_token = step_end_token
        # Load the running block when the runner's precopy already ran, plus
        # (below) the imported boundary block, which the engine registers as a
        # reusable local checkpoint. Otherwise load only the boundary block the
        # runner copies from after loads. Imported boundaries are not dumped.
        state_target_tokens = (
            load_end_token + scheduled_tokens
            if self.need_load_to_running_state
            else load_end_token
        )
        load_plans = self._plans(
            state,
            load_start_token,
            load_end_token,
            state_target_tokens=state_target_tokens,
        )
        if should_load:
            state.load_pending = False
        dump_plans = self._plans(state, dump_start_token, dump_end_token)
        state.token_processed = step_end_token
        return RequestDispatchMeta(request_id, load_plans, dump_plans)

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
        for attention_type, groups in self._groups_by_attention_type:
            if attention_type is AttentionType.STATE and state_target_tokens is None:
                continue
            if attention_type in (AttentionType.WA, AttentionType.STATE):
                # WA keeps the newest completed boundary; a State load
                # ends exactly at the selected restore boundary.
                start_block = end_block_index - 1
                end_block = end_block_index
            else:
                # FA blocks are stored once and never re-stored, so a plan
                # covers every complete UCM block the range touches -- not
                # just the newest boundary.
                start_block = first_block_index
                end_block = end_block_index
            if attention_type is AttentionType.WA:
                # A tail window is only worth storing once it is
                # complete: an early boundary shorter than its groups'
                # largest tail would clamp the window head to token 0 --
                # the request's own prefix, which a restore recomputes
                # anyway.  Skipping keeps every dumped UCM block's record
                # complete, so reverse lookup never selects a partial
                # one (the load branch above cannot select one either).
                if end_block * ucm_block_size < max(
                    group.tail_tokens or 0 for group in groups
                ):
                    continue
            plans.append(
                self._plan(
                    attention_type,
                    groups,
                    state,
                    start_block,
                    end_block,
                    state_target_tokens,
                )
            )
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
                    self._plan(
                        attention_type, groups, state, start_block, end_block, token_end
                    )
                )
        return tuple(plans)

    def _plan(
        self,
        attention_type: AttentionType,
        groups: Sequence[UCMKVCacheGroupInfo],
        state: RequestState,
        start_block: int,
        end_block: int,
        state_target_tokens: int | None,
    ) -> UCMGroupDispatchPlan:
        """Plan for UCM blocks [start_block, end_block) of one attention type."""
        ucm_block_size = self.spec.ucm_cache_block_size
        return UCMGroupDispatchPlan(
            attention_type,
            tuple(state.common_ucm_block_ids[start_block:end_block]),
            start_block * ucm_block_size,
            end_block * ucm_block_size,
            tuple(
                self._group_blocks(
                    attention_type,
                    group,
                    state,
                    start_block,
                    end_block,
                    state_target_tokens=state_target_tokens,
                )
                for group in groups
            ),
        )

    def _group_blocks(
        self,
        attention_type: AttentionType,
        group: UCMKVCacheGroupInfo,
        state: RequestState,
        start_block: int,
        end_block: int,
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
        vllm_block_table = state.vllm_block_tables[group.group_id]
        tail_blocks = group.tail_blocks
        if attention_type is AttentionType.STATE:
            # The caller picks the running or boundary slot from the
            # runner's load/copy ordering.
            table_index = (state_target_tokens - 1) // group.token_block_size
            vllm_block_id = vllm_block_table[table_index]
            assert vllm_block_id != 0
            return np.asarray([vllm_block_id], dtype=np.uint64)
        if attention_type is AttentionType.FA:
            boundary_tokens = (
                np.arange(start_block + 1, end_block + 1, dtype=np.uint64)
                * ucm_block_size
                - 1
            )
        else:
            # WA: one UCM block at the plan's newest boundary.
            boundary_tokens = np.asarray(
                [end_block * ucm_block_size - 1], dtype=np.uint64
            )
        boundary_table_indices = boundary_tokens // group.token_block_size
        # Convert only the table range the windows touch -- a WA group's
        # table spans the whole request while its window holds a couple
        # of blocks.  The clamp gate and the FA divisibility checks keep
        # first_table_index non-negative.
        first_table_index = int(boundary_table_indices.min()) - (tail_blocks - 1)
        last_table_index = int(boundary_table_indices.max())
        vllm_block_ids = np.asarray(
            vllm_block_table[first_table_index : last_table_index + 1], dtype=np.uint64
        )
        window_end_offsets = boundary_table_indices - first_table_index
        if tail_blocks == 1:
            return vllm_block_ids[window_end_offsets]
        steps_back = np.arange(tail_blocks, dtype=np.uint64)[::-1]
        return vllm_block_ids[
            (window_end_offsets[:, None] - steps_back[None, :]).reshape(-1)
        ]
