"""Unified scheduler-side lookup, request state, and dispatch planning."""

from __future__ import annotations

import hashlib
import math
import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata

from .ucm_kv_cache import UCMKVCacheGroupInfo, UCMKVCacheSpec
from .ucm_proxy import UCMProxyAdapter

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request


class RequestHasher:
    """MD5 hasher compatible with the existing connector namespace format."""

    def __init__(self, vllm_config: "VllmConfig", rank_id: int | None) -> None:
        speculative = getattr(vllm_config, "speculative_config", None)
        spec_info = ""
        if speculative is not None:
            method = getattr(speculative, "method", "") or ""
            tokens = getattr(speculative, "num_speculative_tokens", 0)
            spec_info = f":{method}:{tokens}"
        additional = getattr(vllm_config, "additional_config", None) or {}
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
    restore_end_tokens: int
    group_ucm_block_ids: tuple[tuple[bytes, ...], ...]


@dataclass
class RequestState:
    hbm_hit_tokens: int = 0
    external_hit_tokens: int = 0
    restore_end_tokens: int = 0
    num_token_ids: int = 0
    token_processed: int = 0
    group_ucm_block_ids: tuple[tuple[bytes, ...], ...] = ()
    group_vllm_block_ids: tuple[list[int], ...] = ()
    load_pending: bool = False


@dataclass(frozen=True)
class UCMGroupBlockIds:
    group_id: int
    start_block_index: int
    block_ids: tuple[int, ...]


@dataclass(frozen=True)
class UCMGroupDispatchPlan:
    hash_group: int | Literal["FA", "WA"]
    keys: tuple[bytes, ...]
    key_start_index: int
    token_start: int
    token_end: int
    vllm_blocks: tuple[UCMGroupBlockIds, ...]


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


def _token_ids(request: "Request") -> tuple[int, ...]:
    values = getattr(request, "all_token_ids", None)
    if values is None:
        values = getattr(request, "prompt_token_ids", None)
    if values is None:
        raise ValueError("Request does not expose all_token_ids")
    return tuple(int(value) for value in values)


class UCMLookupCoordinator:
    def __init__(
        self,
        kv_cache_spec: UCMKVCacheSpec,
        proxy: UCMProxyAdapter,
        request_hasher: RequestHasher,
        base_seed: bytes,
        *,
        load_threshold_tokens: int = 0,
        recompute_tokens: int = 1,
    ) -> None:
        self.spec = kv_cache_spec
        self.proxy = proxy
        self.hasher = request_hasher
        self.base_seed = base_seed
        self.load_threshold_tokens = max(int(load_threshold_tokens), 0)
        self.recompute_tokens = max(int(recompute_tokens), 0)

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

    def _group_seed(self, group_id: int) -> bytes:
        return self.hasher((b"UCM_GROUP_SEED", self.base_seed, group_id))

    def _attention_keys(
        self, token_ids: Sequence[int], group: UCMKVCacheGroupInfo
    ) -> tuple[bytes, ...]:
        if len(self.spec.groups) == 1:
            base = self._chain(
                token_ids, self.spec.scheduler_block_size, self.base_seed
            )
            multiple = self.spec.chunk_size // self.spec.scheduler_block_size
            return tuple(
                base[index] for index in range(multiple - 1, len(base), multiple)
            )
        return self._chain(
            token_ids, group.hash_block_size, self._group_seed(group.group_id)
        )

    def _attention_unit(self, group: UCMKVCacheGroupInfo) -> int:
        return (
            self.spec.chunk_size
            if len(self.spec.groups) == 1
            else group.hash_block_size
        )

    def _prefix_end(
        self,
        keys: Sequence[bytes],
        key_tokens: int,
        hbm_tokens: int,
        candidate_end: int,
    ) -> int:
        if candidate_end <= hbm_tokens:
            return hbm_tokens
        first = hbm_tokens // key_tokens
        last = candidate_end // key_tokens
        candidates = tuple(keys[first:last])
        if not candidates:
            return hbm_tokens
        hits = self.proxy.lookup(candidates)
        count = 0
        for hit in hits:
            if not hit:
                break
            count += 1
        # A miss in the key that straddles an HBM-only prefix means "no new
        # external progress"; it must never reduce the already computed HBM
        # boundary to the previous key boundary.
        return min(max((first + count) * key_tokens, hbm_tokens), candidate_end)

    def lookup(self, request: "Request", num_computed_tokens: int) -> UCMLookupResult:
        if num_computed_tokens < 0:
            raise ValueError("num_computed_tokens must not be negative")
        token_ids = _token_ids(request)
        result = (
            self._lookup_dsv4(token_ids, num_computed_tokens)
            if self.spec.is_dsv4
            else self._lookup_grouped(token_ids, num_computed_tokens)
        )
        if result.external_hit_tokens <= self.load_threshold_tokens:
            return UCMLookupResult(0, num_computed_tokens, result.group_ucm_block_ids)
        return result

    def _cacheable_end(self, length: int, unit: int) -> int:
        return max(length - self.recompute_tokens, 0) // unit * unit

    def _lookup_grouped(self, token_ids: Sequence[int], hbm: int) -> UCMLookupResult:
        all_keys: list[tuple[bytes, ...]] = [() for _ in self.spec.groups]
        state_groups = self.spec.state_groups
        attention_end = (
            self._cacheable_end(len(token_ids), self.spec.alignment_block_size)
            if state_groups
            else len(token_ids)
        )
        for group in self.spec.attn_groups:
            unit = self._attention_unit(group)
            keys = self._attention_keys(token_ids, group)
            all_keys[group.group_id] = keys
            attention_end = min(
                attention_end,
                self._prefix_end(
                    keys,
                    unit,
                    hbm,
                    attention_end // unit * unit,
                ),
            )
        attention_end = max(
            hbm,
            attention_end
            // self.spec.alignment_block_size
            * self.spec.alignment_block_size,
        )
        restore_end = attention_end
        state_keys: dict[int, tuple[bytes, ...]] = {}
        all_boundaries = tuple(
            range(
                self.spec.alignment_block_size,
                len(token_ids)
                // self.spec.alignment_block_size
                * self.spec.alignment_block_size
                + 1,
                self.spec.alignment_block_size,
            )
        )
        for group in state_groups:
            primary = self.spec.attn_groups[0]
            primary_keys = all_keys[primary.group_id]
            keys = tuple(
                self.hasher(
                    (
                        self._group_seed(group.group_id),
                        b"UCM_MAMBA_ALIGN_STATE",
                        boundary,
                        primary_keys[boundary // primary.hash_block_size - 1],
                    )
                )
                for boundary in all_boundaries
            )
            state_keys[group.group_id] = keys
            all_keys[group.group_id] = keys
        if state_groups and restore_end > hbm:
            candidate_boundaries = tuple(
                range(
                    max(
                        math.ceil((hbm + 1) / self.spec.alignment_block_size)
                        * self.spec.alignment_block_size,
                        self.spec.alignment_block_size,
                    ),
                    restore_end + 1,
                    self.spec.alignment_block_size,
                )
            )
            lookup_start = candidate_boundaries[0] // self.spec.alignment_block_size - 1
            for boundary_index in range(len(candidate_boundaries) - 1, -1, -1):
                boundary_keys = tuple(
                    state_keys[group.group_id][lookup_start + boundary_index]
                    for group in state_groups
                )
                if all(self.proxy.lookup(boundary_keys)):
                    restore_end = candidate_boundaries[boundary_index]
                    break
            else:
                restore_end = hbm
        visible_end = min(restore_end, max(len(token_ids) - self.recompute_tokens, 0))
        return UCMLookupResult(max(visible_end - hbm, 0), restore_end, tuple(all_keys))

    def _lookup_dsv4(self, token_ids: Sequence[int], hbm: int) -> UCMLookupResult:
        unit = self.spec.chunk_size
        fa_keys = self._chain(token_ids, unit, self.hasher(b"FA_Block"))
        wa_keys = self._chain(token_ids, unit, self.hasher(b"WA_Block"))
        # WA is a boundary snapshot for sliding-window/state data. Leave one
        # complete canonical block for vLLM to allocate and recompute; unlike
        # pure FA, reporting T-1 while restoring the snapshot at T is invalid.
        candidate = self._cacheable_end(len(token_ids), unit)
        fa_end = self._prefix_end(fa_keys, unit, hbm, candidate)
        restore_end = hbm
        first = max(math.ceil((hbm + 1) / unit), 1)
        last = fa_end // unit
        if last >= first:
            indexes = tuple(range(first - 1, last))
            hits = self.proxy.lookup(tuple(wa_keys[index] for index in indexes))
            for index, hit in reversed(tuple(zip(indexes, hits))):
                if hit:
                    restore_end = (index + 1) * unit
                    break
        return UCMLookupResult(
            max(restore_end - hbm, 0), restore_end, (fa_keys, wa_keys)
        )


class UCMDispatcher:
    """Own scheduler request snapshots and produce pointer-free plans."""

    def __init__(self, spec: UCMKVCacheSpec) -> None:
        self.spec = spec
        self.requests: dict[str, RequestState] = {}

    def record_lookup(
        self, request: "Request", hbm_hit_tokens: int, result: UCMLookupResult
    ) -> RequestState:
        request_id = str(request.request_id)
        state = RequestState(
            hbm_hit_tokens=hbm_hit_tokens,
            external_hit_tokens=result.external_hit_tokens,
            restore_end_tokens=result.restore_end_tokens,
            num_token_ids=len(_token_ids(request)),
            token_processed=hbm_hit_tokens + result.external_hit_tokens,
            group_ucm_block_ids=result.group_ucm_block_ids,
            group_vllm_block_ids=tuple([] for _ in self.spec.groups),
            load_pending=result.restore_end_tokens > hbm_hit_tokens,
        )
        self.requests[request_id] = state
        return state

    def update_blocks(
        self,
        request_id: str,
        group_block_ids: Sequence[Sequence[int]],
        *,
        append: bool,
    ) -> None:
        state = self.requests[request_id]
        if len(group_block_ids) != len(self.spec.groups):
            raise ValueError("group block table count does not match KV cache groups")
        if append:
            for destination, source in zip(state.group_vllm_block_ids, group_block_ids):
                destination.extend(int(value) for value in source)
        else:
            state.group_vllm_block_ids = tuple(
                [int(value) for value in source] for source in group_block_ids
            )

    def finish(self, request_id: str) -> None:
        self.requests.pop(request_id, None)

    def preempt(self, request_id: str) -> None:
        self.requests.pop(request_id, None)

    def build_metadata(
        self,
        scheduled_tokens: Mapping[str, int],
        *,
        preempted_req_ids: Sequence[str] = (),
        finished_req_ids: Sequence[str] = (),
    ) -> UCMConnectorMetadata:
        metadata = UCMConnectorMetadata(
            preempted_req_ids=set(preempted_req_ids),
            finished_req_ids=set(finished_req_ids),
        )
        for request_id, num_scheduled in scheduled_tokens.items():
            state = self.requests.get(str(request_id))
            if state is None:
                continue
            metadata.requests[str(request_id)] = self._request_meta(
                str(request_id), state, int(num_scheduled)
            )
        for request_id in (*preempted_req_ids, *finished_req_ids):
            self.requests.pop(str(request_id), None)
        return metadata

    def build_from_scheduler_output(
        self, scheduler_output: "SchedulerOutput"
    ) -> UCMConnectorMetadata:
        """Consume the vLLM 0.26 SchedulerOutput shape.

        New and resumed block tables replace the snapshot; ordinary cached
        allocations append only their newly allocated blocks.
        """

        for request in scheduler_output.scheduled_new_reqs:
            request_id = str(request.req_id)
            if request_id in self.requests:
                self.update_blocks(
                    request_id,
                    request.block_ids,
                    append=False,
                )

        cached = scheduler_output.scheduled_cached_reqs
        for index, request_id_value in enumerate(cached.req_ids):
            request_id = str(request_id_value)
            if request_id not in self.requests:
                continue
            incoming = cached.new_block_ids[index]
            resumed = request_id in cached.resumed_req_ids
            if incoming is not None:
                self.update_blocks(request_id, incoming, append=not resumed)
            elif resumed:
                self.update_blocks(
                    request_id,
                    tuple([] for _ in self.spec.groups),
                    append=False,
                )

        return self.build_metadata(
            scheduler_output.num_scheduled_tokens,
            preempted_req_ids=tuple(scheduler_output.preempted_req_ids or ()),
            finished_req_ids=tuple(scheduler_output.finished_req_ids),
        )

    def _request_meta(
        self, request_id: str, state: RequestState, scheduled_tokens: int
    ) -> RequestDispatchMeta:
        step_end = min(state.token_processed + scheduled_tokens, state.num_token_ids)
        should_load = state.load_pending and scheduled_tokens > 0
        load_start = state.hbm_hit_tokens if should_load else state.restore_end_tokens
        load_end = state.restore_end_tokens
        dump_start = state.token_processed
        dump_end = step_end
        load = self._plans(state, load_start, load_end, is_dump=False)
        if should_load:
            state.load_pending = False
        dump = self._plans(state, dump_start, dump_end, is_dump=True)
        state.token_processed = step_end
        return RequestDispatchMeta(request_id, load, dump)

    def _plans(
        self, state: RequestState, token_start: int, token_end: int, *, is_dump: bool
    ) -> tuple[UCMGroupDispatchPlan, ...]:
        plans: list[UCMGroupDispatchPlan] = []
        if token_end <= token_start:
            return ()
        hash_groups: tuple[int | Literal["FA", "WA"], ...]
        units: tuple[int, ...]
        if self.spec.is_dsv4:
            hash_groups = ("FA", "WA")
            units = (self.spec.chunk_size, self.spec.chunk_size)
        else:
            hash_groups = tuple(range(len(self.spec.groups)))
            units = tuple(
                (
                    self.spec.alignment_block_size
                    if group.is_state_snapshot
                    else (
                        self.spec.chunk_size
                        if len(self.spec.groups) == 1
                        else group.hash_block_size
                    )
                )
                for group in self.spec.groups
            )
        for hash_index, (hash_group, unit) in enumerate(zip(hash_groups, units)):
            keys_available = state.group_ucm_block_ids[hash_index]
            is_dsv4_wa = self.spec.is_dsv4 and hash_group == "WA"
            is_group_state = (
                isinstance(hash_group, int)
                and self.spec.groups[hash_group].is_state_snapshot
            )
            if is_group_state or is_dsv4_wa:
                if token_end % unit:
                    continue
                start = max(token_end // unit - 1, 0)
                end = start + 1
            else:
                start = token_start // unit
                end = token_end // unit
            if end <= start:
                continue
            selected_keys = tuple(keys_available[start:end])
            if not selected_keys:
                continue
            physical_groups = self._physical_groups(hash_group)
            selections: list[UCMGroupBlockIds] = []
            for group in physical_groups:
                if is_dsv4_wa:
                    if not group.tail_tokens:
                        continue
                    boundary = end * unit
                    physical_token_start = max(boundary - group.tail_tokens, 0)
                    block_start = physical_token_start // group.token_block_size
                    block_end = math.ceil(boundary / group.token_block_size)
                else:
                    block_start = start * unit // group.token_block_size
                if is_group_state:
                    block_start = max((token_end - 1) // group.token_block_size, 0)
                    block_end = block_start + 1
                elif not is_dsv4_wa:
                    block_end = math.ceil(end * unit / group.token_block_size)
                table = state.group_vllm_block_ids[group.group_id]
                selections.append(
                    UCMGroupBlockIds(
                        group.group_id,
                        block_start,
                        tuple(table[block_start:block_end]),
                    )
                )
            plans.append(
                UCMGroupDispatchPlan(
                    hash_group,
                    selected_keys,
                    start,
                    start * unit,
                    end * unit,
                    tuple(selections),
                )
            )
        return tuple(plans)

    def _physical_groups(
        self, hash_group: int | Literal["FA", "WA"]
    ) -> tuple[UCMKVCacheGroupInfo, ...]:
        if isinstance(hash_group, int):
            return (self.spec.groups[hash_group],)
        if hash_group == "FA":
            return self.spec.fa_groups
        return self.spec.wa_groups
