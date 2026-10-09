"""Unified scheduler-side lookup, request state, and dispatch planning."""

from __future__ import annotations

import hashlib
import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal

import numpy as np

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata

from .layout import UCMKVCacheGroupInfo, UCMKVCacheSpec
from .ucm_proxy import UCMProxyAdapter

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request


def dispatch_routes(
    spec: UCMKVCacheSpec,
) -> tuple[
    tuple[Literal["FA", "WA", "State"], tuple["UCMKVCacheGroupInfo", ...]], ...
]:
    """The routing table every dump/load works over: key kind -> groups.

    FA holds the full-attention groups; WA the sliding groups that
    re-store a window tail (tail 0 groups store nothing); State the
    mamba snapshot groups. Empty kinds are absent, and
    ``group_ucm_block_ids`` / dispatch plans index these in order.
    """

    routes: list[
        tuple[Literal["FA", "WA", "State"], tuple["UCMKVCacheGroupInfo", ...]]
    ] = []
    if spec.fa_groups:
        routes.append(("FA", spec.fa_groups))
    wa_stored = tuple(
        group for group in spec.wa_groups if (group.tail_tokens or 0) > 0
    )
    if wa_stored:
        routes.append(("WA", wa_stored))
    if spec.state_groups:
        routes.append(("State", spec.state_groups))
    return tuple(routes)


class RequestHasher:
    """MD5 hasher compatible with the existing connector namespace format."""

    def __init__(self, vllm_config: "VllmConfig", rank_id: int | None) -> None:
        speculative = vllm_config.speculative_config
        spec_info = ""
        if speculative is not None:
            method = speculative.method or ""
            tokens = speculative.num_speculative_tokens
            spec_info = f":{method}:{tokens}"
        additional = vllm_config.additional_config
        sparse = (
            f":sfa_c8={int(bool(additional.get('enable_sparse_sfa_c8', False)))}"
            f":li_c8={int(bool(additional.get('enable_sparse_li_c8', False)))}"
        )
        model_config = vllm_config.model_config
        model_name = model_config.model.rstrip("/").split("/")[-1]
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        meta = (
            f"{model_name}:{tp_size}:{model_config.dtype}:"
            f"{rank_id}{spec_info}{sparse}"
        )
        self.meta_bytes = meta.encode("utf-8")

    def __call__(self, value: object) -> bytes:
        payload = (
            value
            if isinstance(value, bytes)
            else pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        )
        return hashlib.md5(self.meta_bytes + payload).digest()


@dataclass(frozen=True)
class UCMLookupResult:
    external_hit_tokens: int
    group_ucm_block_ids: tuple[tuple[bytes, ...], ...]


@dataclass
class RequestState:
    hbm_hit_tokens: int = 0
    external_hit_tokens: int = 0
    num_token_ids: int = 0
    token_processed: int = 0
    group_ucm_block_ids: tuple[tuple[bytes, ...], ...] = ()
    group_vllm_block_ids: tuple[list[int], ...] = ()
    load_pending: bool = False


@dataclass(frozen=True)
class UCMGroupDispatchPlan:
    """One request's hash-chain slice; keys are unique within this plan."""

    hash_group: Literal["FA", "WA", "State"]
    keys: tuple[bytes, ...]
    token_start: int
    token_end: int
    # Physical block IDs per participating group, in dispatch_routes()
    # order, block-major (tail_blocks blocks per key) -- the static
    # window shape lives on the spec's groups (``tail_blocks``), so
    # only the ids travel.
    group_block_ids: tuple[np.ndarray, ...]


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


def _token_ids(request: "Request") -> tuple[int, ...]:
    # int() normalizes numpy scalars so the hash chain pickles stably.
    return tuple(int(value) for value in request.all_token_ids)


_KEY_TYPE_BITS: Mapping[str, int] = {"FA": 0, "WA": 1, "State": 2}


def _key_tag(chain: str, tp_rank: int = 0, pp_rank: int = 0) -> bytes:
    """The 2-byte suffix of a UCM key, big-endian bit layout:

    type(2) group(4) tp_rank(4) pp_rank(4) reserved(2).  Every group of a
    chain shares one record, so no group bits are set today; the layout
    keeps them for a future per-group split.  Ranks default to the
    logical rank-0 key namespace the scheduler hashes in.
    """

    value = (
        (_KEY_TYPE_BITS[chain] << 14)
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
    ) -> None:
        self.spec = kv_cache_spec
        self.proxy = proxy
        self.hasher = request_hasher
        self.base_seed = base_seed
        self.load_threshold_tokens = max(int(load_threshold_tokens), 0)
        self.recompute_tokens = max(int(recompute_tokens), 0)
        self.requests: dict[str, RequestState] = {}
        # The per-kind routing table (key kind -> participating groups),
        # built once: FA, then the tail-storing WA groups, then State.
        self._routes = dispatch_routes(kv_cache_spec)
        self._chain_tags = tuple(
            (label, _key_tag(label, tp_rank, pp_rank)) for label, _ in self._routes
        )

    def _chain(
        self, token_ids: Sequence[int], block_size: int, parent: bytes
    ) -> tuple[bytes, ...]:
        result: list[bytes] = []
        for start in range(0, len(token_ids), block_size):
            block = tuple(token_ids[start : start + block_size])
            if len(block) != block_size:
                break
            parent = self.hasher((parent, block))
            result.append(parent)
        return tuple(result)

    def _chain_keys(
        self, token_ids: Sequence[int]
    ) -> tuple[tuple[bytes, ...], ...]:
        """Per-chain keys: one shared hash chain, each chain's tag appended.

        The chain runs over ucm_cache_block_size token blocks from the
        base seed; a key is the chain value's first 14 bytes plus the
        chain's 2-byte tag, so the FA and boundary keys at one boundary
        differ only in their tag.
        """

        values = self._chain(
            token_ids, self.spec.ucm_cache_block_size, self.base_seed
        )
        return tuple(
            tuple(value[:14] + tag for value in values)
            for _label, tag in self._chain_tags
        )

    def lookup(self, request: "Request", num_computed_tokens: int) -> UCMLookupResult:
        """Probe the external cache and record the request snapshot."""
        token_ids = _token_ids(request)
        result = self._lookup(token_ids, num_computed_tokens)
        if result.external_hit_tokens <= self.load_threshold_tokens:
            result = UCMLookupResult(0, result.group_ucm_block_ids)
        self.requests[str(request.request_id)] = RequestState(
            hbm_hit_tokens=num_computed_tokens,
            external_hit_tokens=result.external_hit_tokens,
            num_token_ids=len(token_ids),
            token_processed=num_computed_tokens + result.external_hit_tokens,
            group_ucm_block_ids=result.group_ucm_block_ids,
            group_vllm_block_ids=tuple([] for _ in self.spec.groups),
            load_pending=result.external_hit_tokens > 0,
        )
        return result

    def _lookup(
        self, token_ids: Sequence[int], hbm: int
    ) -> UCMLookupResult:
        """FA prefix restore; WA/State additionally need their boundary.

        All chains share one hash chain at ucm_cache_block_size, so every
        chain has a key at the same boundary.  The FA chain is a prefix
        requirement; the WA (window tail) and State (mamba snapshot)
        chains are boundary records -- restoring requires a complete FA
        prefix up to some boundary and that boundary's tail or snapshot.
        Boundary chains must all exist at the same restore boundary.
        """

        ucm_block_size = self.spec.ucm_cache_block_size
        group_ucm_block_ids = self._chain_keys(token_ids)
        # Every chain scans up to the last complete block below the
        # recompute margin; the margin's block is left to recompute (v1
        # semantics -- a full hit still recomputes recompute_tokens).
        last = max(len(token_ids) - self.recompute_tokens, 0) // ucm_block_size
        first = hbm // ucm_block_size
        fa_end = hbm
        for (label, _tag), keys in zip(self._chain_tags, group_ucm_block_ids):
            if label == "FA":
                hits = self.proxy.lookup_on_prefix(keys[first:last])
                if hits >= 0:
                    fa_end = max((first + hits + 1) * ucm_block_size, hbm)
                break
        boundary_keys = [
            keys
            for (label, _tag), keys in zip(self._chain_tags, group_ucm_block_ids)
            if label != "FA"
        ]
        restore_end = fa_end
        while boundary_keys and restore_end > hbm:
            end = restore_end // ucm_block_size
            hits = [
                self.proxy.lookup_on_reverse(keys[first:end])
                for keys in boundary_keys
            ]
            if min(hits) < 0:
                restore_end = hbm
                break
            candidate = (first + min(hits) + 1) * ucm_block_size
            if candidate == restore_end:
                break
            restore_end = candidate
        return UCMLookupResult(max(restore_end - hbm, 0), group_ucm_block_ids)


    def update_blocks(
        self,
        request_id: str,
        group_block_ids: Sequence[Sequence[int]],
        *,
        append: bool,
    ) -> None:
        state = self.requests[request_id]
        if append:
            for destination, source in zip(state.group_vllm_block_ids, group_block_ids):
                destination.extend(int(value) for value in source)
        else:
            state.group_vllm_block_ids = tuple(
                [int(value) for value in source] for source in group_block_ids
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
            blocks = block_state.get_block_ids(request_id)
            self.update_blocks(request_id, blocks, append=False)
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
        route_index = next(
            i for i, (kind, _) in enumerate(self._routes) if kind == "State"
        )
        groups = self._routes[route_index][1]
        ucm_block_size = self.spec.ucm_cache_block_size
        matched_prefix = state.hbm_hit_tokens + state.external_hit_tokens
        boundaries: dict[int, dict[int, int]] = {}
        for group_id, block_id, boundary in offloads:
            # Engine offers include newly hashed imported State blocks. Their
            # prefix is already cached; do not dump it back to the store.
            # Use the initial hit boundary, not token_processed: a checkpoint
            # computed after that hit can be offered on a later step.
            if (
                matched_prefix < boundary <= state.num_token_ids
                and boundary % ucm_block_size == 0
            ):
                boundaries.setdefault(boundary, {})[group_id] = block_id
        plans = []
        for boundary, blocks in sorted(boundaries.items()):
            if not all(group.group_id in blocks for group in groups):
                continue
            key = state.group_ucm_block_ids[route_index][boundary // ucm_block_size - 1]
            plans.append(
                UCMGroupDispatchPlan(
                    "State",
                    (key,),
                    boundary - ucm_block_size,
                    boundary,
                    tuple(
                        np.asarray([blocks[group.group_id]], dtype=np.uint64)
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
        # CUDA precopy runs before connector load. Ascend MRV1 stages it
        # before load and executes it afterwards, reading the boundary slot.
        # CUDA additionally restores the imported boundary block below: the
        # engine registers that block as a reusable local checkpoint. Imported
        # boundaries are excluded from State dump plans.
        state_target_tokens = (
            load_end
            if self.spec.device_type == "npu"
            else load_end + scheduled_tokens
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
        """FA/WA plans and platform-specific State load destinations.

        FA gets every complete key the range touches; WA gets the newest
        complete boundary. State loads use that restore boundary's key
        and its load destinations. CUDA restores both the imported checkpoint
        and the running block; Ascend copies from the restored checkpoint.
        State dumps are built separately
        from the engine's exact boundary_state_offloads.
        """
        plans: list[UCMGroupDispatchPlan] = []
        if token_end <= token_start:
            return ()
        ucm_block_size = self.spec.ucm_cache_block_size
        # Route-invariant key bounds: the first key the range touches and
        # the boundary it ends on.  No complete key in range means no
        # plan for any kind: FA has nothing to store or load, and the
        # WA/State boundary was not reached.
        first_key = token_start // ucm_block_size
        last_key = token_end // ucm_block_size
        if last_key <= first_key:
            return ()
        for route_index, (hash_group, groups) in enumerate(self._routes):
            if hash_group == "State" and state_target_tokens is None:
                continue
            keys_available = state.group_ucm_block_ids[route_index]
            if hash_group in ("WA", "State"):
                # WA keeps the newest completed boundary; a State load
                # ends exactly at the selected restore boundary.
                start = last_key - 1
                end = last_key
            else:
                # FA keys are stored once and never re-stored, so a plan
                # covers every complete key the range touches -- not just
                # the newest boundary.
                start = first_key
                end = last_key
            if hash_group == "WA":
                # A tail window is only worth storing once it is
                # complete: an early boundary shorter than the chain's
                # largest tail would clamp the window head to token 0 --
                # the request's own prefix, which a restore recomputes
                # anyway.  Skipping keeps every dumped key's record
                # complete, so reverse lookup never selects a partial
                # one (the load branch above cannot select one either).
                if end * ucm_block_size < max(
                    group.tail_tokens or 0 for group in groups
                ):
                    continue
            plan = UCMGroupDispatchPlan(
                hash_group,
                tuple(keys_available[start:end]),
                start * ucm_block_size,
                end * ucm_block_size,
                tuple(
                    self._group_blocks(
                        hash_group,
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
            if hash_group == "State" and self.spec.device_type != "npu":
                # Precopy has already run, so restoring only the imported
                # boundary cannot initialize CUDA's running state. Restoring
                # only the running block leaves a local cached boundary unfilled.
                # Complete restore boundaries and a positive scheduled span
                # place these in distinct slots for every State group.
                plans.append(
                    replace(
                        plan,
                        group_block_ids=tuple(
                            self._group_blocks(
                                hash_group, group, state, start, end,
                                state_target_tokens=token_end,
                            )
                            for group in groups
                        ),
                    )
                )
        return tuple(plans)

    def _group_blocks(
        self,
        hash_group: Literal["FA", "WA", "State"],
        group: UCMKVCacheGroupInfo,
        state: RequestState,
        start: int,
        end: int,
        *,
        state_target_tokens: int | None = None,
    ) -> np.ndarray:
        """One group's window block ids for the plan's keys, vectorized.

        Every window is ``tail_blocks`` vllm blocks ending at the block
        containing the window's last token -- HMA's boundary indices: a
        FA key's last token is (k + 1) * ucm_block_size - 1; a WA key anchors its
        window at the restore boundary. State uses a platform load slot.
        The _plans clamp gate keeps the WA gather non-negative and the
        parse-time divisibility checks keep an FA key inside one shared
        block.  Block-major per key.
        """

        ucm_block_size = self.spec.ucm_cache_block_size
        table = state.group_vllm_block_ids[group.group_id]
        tail_blocks = group.tail_blocks
        if hash_group == "State":
            # The caller resolves CUDA's current running slot versus
            # Ascend's precopy source slot from the load/copy ordering.
            index = (state_target_tokens - 1) // group.token_block_size
            block_id = table[index]
            assert block_id != 0
            return np.asarray([block_id], dtype=np.uint64)
        if hash_group == "FA":
            boundary_tokens = (
                np.arange(start + 1, end + 1, dtype=np.uint64)
                * ucm_block_size
                - 1
            )
        else:
            # WA: one boundary key at the plan's newest boundary.
            boundary_tokens = np.asarray(
                [end * ucm_block_size - 1], dtype=np.uint64
            )
        boundary_block_idx = boundary_tokens // group.token_block_size
        # Convert only the table range the windows touch -- a WA group's
        # table spans the whole request while its window holds a couple
        # of blocks.  The clamp gate and the FA divisibility checks keep
        # first_idx non-negative.
        first_idx = int(boundary_block_idx.min()) - (tail_blocks - 1)
        last_idx = int(boundary_block_idx.max())
        table_ids = np.asarray(
            table[first_idx : last_idx + 1], dtype=np.uint64
        )
        within = boundary_block_idx - first_idx
        if tail_blocks == 1:
            return table_ids[within]
        steps_back = np.arange(tail_blocks, dtype=np.uint64)[::-1]
        return table_ids[(within[:, None] - steps_back[None, :]).reshape(-1)]
