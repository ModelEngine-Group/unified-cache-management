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
    hash_group: Literal["FA", "WA", "State"]
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


_KEY_TYPE_BITS: Mapping[str, int] = {"FA": 0, "WA": 1, "State": 2}


def _key_tag(chain: str, tp_rank: int = 0, pp_rank: int = 0) -> bytes:
    """The 2-byte suffix of a UCM key, big-endian bit layout:

    type(2) group(2) tp_rank(5) pp_rank(4) reserved(3).  Every group of a
    chain shares one record, so no group bits are set today; the layout
    keeps them for a future per-group split.  Ranks default to the
    logical rank-0 key namespace the scheduler hashes in.
    """

    value = (
        (_KEY_TYPE_BITS[chain] << 14)
        | ((tp_rank & 0b11111) << 7)
        | ((pp_rank & 0b1111) << 3)
    )
    return value.to_bytes(2, "big")


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
        tp_rank: int = 0,
        pp_rank: int = 0,
    ) -> None:
        self.spec = kv_cache_spec
        self.proxy = proxy
        self.hasher = request_hasher
        self.base_seed = base_seed
        self.load_threshold_tokens = max(int(load_threshold_tokens), 0)
        self.recompute_tokens = max(int(recompute_tokens), 0)
        self.chains = kv_cache_spec.dispatch_chains()
        self._chain_tags = tuple(
            (label, _key_tag(label, tp_rank, pp_rank)) for label, _ in self.chains
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

    def _prefix_end(
        self,
        keys: Sequence[bytes],
        key_tokens: int,
        hbm_tokens: int,
        candidate_end: int,
    ) -> int:
        """The FA prefix end, probed through the proxy's prefix scan.

        The proxy scans the keys after the HBM boundary and stops at the
        first miss.  A miss in the key that straddles an HBM-only prefix
        means "no new external progress"; it must never reduce the
        already computed HBM boundary to the previous key boundary.
        """

        if candidate_end <= hbm_tokens:
            return hbm_tokens
        first = hbm_tokens // key_tokens
        last = candidate_end // key_tokens
        hits = self.proxy.lookup_on_prefix(keys[first:last])
        end = (first + hits + 1) * key_tokens if hits >= 0 else hbm_tokens
        return min(max(end, hbm_tokens), candidate_end)

    def lookup(self, request: "Request", num_computed_tokens: int) -> UCMLookupResult:
        if num_computed_tokens < 0:
            raise ValueError("num_computed_tokens must not be negative")
        token_ids = _token_ids(request)
        result = self._lookup(token_ids, num_computed_tokens)
        if result.external_hit_tokens <= self.load_threshold_tokens:
            return UCMLookupResult(0, num_computed_tokens, result.group_ucm_block_ids)
        return result

    def _cacheable_end(self, length: int, unit: int) -> int:
        return max(length - self.recompute_tokens, 0) // unit * unit

    def _lookup(
        self, token_ids: Sequence[int], hbm: int
    ) -> UCMLookupResult:
        """FA prefix restore; WA/State additionally need their boundary.

        All chains share one hash chain at ucm_cache_block_size, so every
        chain has a key at the same boundaries.  The FA chain is a prefix
        requirement; the WA (window tail) and State (mamba snapshot)
        chains are boundary records -- restoring requires a complete FA
        prefix up to some boundary and that boundary's tail or snapshot.
        Each boundary chain is reverse-scanned to its latest hit; the
        restore boundary is the earliest of those (the latest boundary
        where every chain hits), never past the FA prefix.
        """

        unit = self.spec.ucm_cache_block_size
        keys_per_chain = self._chain_keys(token_ids)
        # A pure-FA restore may load the block the recompute margin sits
        # in (its KV just gets overwritten); a boundary restore must stop
        # one complete block earlier, leaving the margin's block to
        # recompute against the restored boundary record.
        has_boundary = any(label != "FA" for label, _tag in self._chain_tags)
        candidate = (
            self._cacheable_end(len(token_ids), unit)
            if has_boundary
            else len(token_ids)
        )
        fa_end = hbm
        boundary_keys: list[tuple[bytes, ...]] = []
        for (label, _tag), keys in zip(self._chain_tags, keys_per_chain):
            if label == "FA":
                fa_end = self._prefix_end(keys, unit, hbm, candidate)
            else:
                boundary_keys.append(keys)
        restore_end = fa_end
        if boundary_keys:
            restore_end = hbm
            first = max(math.ceil((hbm + 1) / unit), 1)
            last = fa_end // unit
            if last >= first:
                latest: int | None = None
                for keys in boundary_keys:
                    hit = self.proxy.lookup_on_reverse(keys[first - 1 : last])
                    if hit < 0:
                        latest = None
                        break
                    boundary_index = first - 1 + hit
                    latest = (
                        boundary_index
                        if latest is None
                        else min(latest, boundary_index)
                    )
                if latest is not None:
                    restore_end = (latest + 1) * unit
        visible_end = min(restore_end, max(len(token_ids) - self.recompute_tokens, 0))
        return UCMLookupResult(
            max(visible_end - hbm, 0), restore_end, keys_per_chain
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
        unit = self.spec.ucm_cache_block_size
        chains = self.spec.dispatch_chains()
        for chain_index, (hash_group, physical_groups) in enumerate(chains):
            keys_available = state.group_ucm_block_ids[chain_index]
            if hash_group in ("WA", "State"):
                # Boundary chains record the last complete cache block in
                # the range; earlier boundaries were already recorded when
                # they completed.
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
            selections: list[UCMGroupBlockIds] = []
            for group in physical_groups:
                table = state.group_vllm_block_ids[group.group_id]
                if hash_group == "WA":
                    boundary = end * unit
                    tail_tokens = group.tail_tokens
                    assert tail_tokens is not None  # WA chains store tails
                    window_start = max(boundary - tail_tokens, 0)
                    block_start = window_start // group.token_block_size
                    block_end = math.ceil(boundary / group.token_block_size)
                elif hash_group == "State":
                    # A state snapshot lives in the last complete block of
                    # the boundary.
                    block_start = max(
                        (token_end - 1) // group.token_block_size, 0
                    )
                    block_end = block_start + 1
                else:
                    block_start = start * unit // group.token_block_size
                    block_end = math.ceil(end * unit / group.token_block_size)
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
