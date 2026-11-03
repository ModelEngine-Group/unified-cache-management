import copy
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, List, Optional

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.platforms import current_platform
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

from ucm.integration.vllm.device import create_device
from ucm.integration.vllm.request_hasher import RequestHasher
from ucm.integration.vllm.ucm_connector import (
    KVCacheLayout,
    KVCacheSegment,
    PendingDumpTask,
    RequestDispatchMeta,
    RequestMeta,
    UCMConnectorMetadata,
    UCMDirectConnector,
    UCMLiteConnector,
    _record_counter,
    _scheduler_read_block_size,
    _short_list,
    _use_ucm_connector_cpu_affinity,
)
from ucm.logger import init_logger
from ucm.shared.metrics import ucmmetrics
from ucm.sparse.state import has_ucm_sparse
from ucm.store.cache.capabilities import uses_partitioned_cache
from ucm.store.factory_v1 import UcmConnectorFactoryV1
from ucm.store.ucmstore_v1 import Task, UcmKVStoreBaseV1

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


def _kv_cache_tensor_layers(raw_tensor) -> list[str]:
    """Read layer names from both new and legacy vLLM tensor descriptors."""
    names = getattr(raw_tensor, "layers", None)
    if names is None:
        names = getattr(raw_tensor, "shared_by", None)
    return names if names is not None else []


@dataclass
class HLARequestMeta(RequestMeta):
    """RequestMeta extended with per-group block tracking for hybrid models."""

    group_ucm_block_ids: list[list[bytes]] = field(default_factory=list)
    group_vllm_block_ids: list[list[int]] = field(default_factory=list)


@dataclass
class HLARequestDispatchMeta(RequestDispatchMeta):
    """Extends RequestDispatchMeta with full-attn block count for MLA rank scoping."""

    load_full_attn_count: int = 0
    dump_full_attn_count: int = 0
    load_group_ids: list[int] = field(default_factory=list)
    dump_group_ids: list[int] = field(default_factory=list)


def layer_name_to_kv_cache_spec(
    kv_cache_config: "KVCacheConfig",
) -> dict[str, list[KVCacheSpec]]:
    """Map each model layer name to its concrete KVCacheSpec.

    Handles merged group specs and UniformTypeKVCacheSpecs (per-layer
    ``kv_cache_specs`` entries).
    """
    out: dict[str, list[KVCacheSpec]] = defaultdict(list)
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            by_name = spec.kv_cache_specs
            for name in group.layer_names:
                out[name].append(by_name[name])
        else:
            for name in group.layer_names:
                out[name].append(spec)
    return out


def block_size_from_kv_cache_spec(spec: KVCacheSpec) -> int:
    """Token block size used for KV scheduling / hashing for one group spec."""
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return next(iter(spec.kv_cache_specs.values())).block_size
    return spec.block_size


def is_mamba_align_kv_cache_spec(spec: KVCacheSpec) -> bool:
    if isinstance(spec, UniformTypeKVCacheSpecs):
        sample = next(iter(spec.kv_cache_specs.values()))
        return is_mamba_align_kv_cache_spec(sample)
    return isinstance(spec, MambaSpec) and spec.mamba_cache_mode == "align"


def participates_in_prefix_caching(spec: KVCacheSpec) -> bool:
    """Read HLA prefix-cache eligibility, excluding transient GLM indexer state."""
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return all(
            participates_in_prefix_caching(inner)
            for inner in spec.kv_cache_specs.values()
        )
    # Ascend KpoolTail uses the new flag; it is request-local ring state,
    # not a historical prefix, even when KV transfer is enabled.
    return bool(getattr(spec, "prefix_cacheable", True)) and bool(
        getattr(spec, "participates_in_prefix_caching", True)
    )


def extend_non_null(
    dst_ucm_block_ids: list[bytes],
    dst_vllm_block_ids: list[int],
    dst_group_ids: list[int],
    src_ucm_block_ids: list[bytes],
    src_vllm_block_ids: list[int],
    group_id: int,
) -> None:
    # Skip vLLM null blocks (block_id=0) used as mamba-align placeholders.
    for ucm_block_id, vllm_block_id in zip(src_ucm_block_ids, src_vllm_block_ids):
        if vllm_block_id == 0:
            continue
        dst_ucm_block_ids.append(ucm_block_id)
        dst_vllm_block_ids.append(vllm_block_id)
        dst_group_ids.append(group_id)


def _normalize_tensor_size_list(tensor_size_list: Any) -> list[int]:
    if isinstance(tensor_size_list, np.ndarray):
        return [int(v) for v in tensor_size_list.reshape(-1).tolist()]
    if isinstance(tensor_size_list, (list, tuple)):
        return [int(v) for v in tensor_size_list]
    return [int(tensor_size_list)]


@dataclass
class GroupInfo:
    """Per-group metadata used by :class:`KVCacheGroupManager`."""

    group_id: int
    block_size: int
    layer_names: tuple[str, ...]
    # Independent hash chain seed per group (see ``KVCacheGroupManager``).
    seed: bytes
    is_mamba_align: bool = False
    block_hasher: Optional[Callable[["Request"], list[bytes]]] = None
    participates_in_prefix_caching: bool = True

    @property
    def is_full_attention(self) -> bool:
        return self.participates_in_prefix_caching and not self.is_mamba_align


class KVCacheGroupManager:
    """Group-aware hashing and two-stage lookup for hybrid (HLA) connectors."""

    def __init__(
        self,
        kv_cache_config: "KVCacheConfig",
        request_hasher: "RequestHasher",
        base_seed: bytes,
    ) -> None:
        self.request_hasher = request_hasher
        self.groups_by_id: list[GroupInfo] = []
        self.full_attn_groups: list[GroupInfo] = []
        self.state_groups: list[GroupInfo] = []
        self.non_prefix_groups: list[GroupInfo] = []

        for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
            spec = group.kv_cache_spec
            block_size = block_size_from_kv_cache_spec(spec)
            is_mamba_align = is_mamba_align_kv_cache_spec(spec)
            seed = request_hasher((b"UCM_GROUP_SEED", base_seed, group_id))
            info = GroupInfo(
                group_id=group_id,
                block_size=block_size,
                layer_names=tuple(group.layer_names),
                seed=seed,
                is_mamba_align=is_mamba_align,
                participates_in_prefix_caching=participates_in_prefix_caching(spec),
            )
            if not is_mamba_align:
                info.block_hasher = request_hasher.make_request_block_hasher(
                    block_size, seed
                )
            self.groups_by_id.append(info)
            if info.is_full_attention:
                self.full_attn_groups.append(info)
            elif info.is_mamba_align:
                self.state_groups.append(info)
            else:
                self.non_prefix_groups.append(info)

        assert len(self.full_attn_groups) >= 1, (
            "UCMHybridLinearAttentionConnector expects at least one full-attention group in "
            "kv_cache_config.kv_cache_groups."
        )

        # Non-prefix groups (for example GLM-5.3-Flash KpoolTail) have independent
        # lifetimes and must not constrain prefix-cache resume boundaries.
        cached_groups = self.full_attn_groups + self.state_groups
        all_block_sizes = [g.block_size for g in cached_groups]
        self.lcm_block_size: int = math.lcm(*all_block_sizes)

        for g in cached_groups:
            assert self.lcm_block_size % g.block_size == 0, (
                f"group {g.group_id} block_size={g.block_size} does not "
                f"divide LCM={self.lcm_block_size}"
            )
        for sg in self.state_groups:
            assert sg.is_mamba_align, (
                f"state group {sg.group_id} is not mamba-align; "
                f"UCMHybridLinearAttentionConnector only supports mamba-align "
                f"state groups."
            )

        logger.info(
            "KVCacheGroupManager initialized: "
            f"lcm_block_size={self.lcm_block_size}, "
            f"full_attn_groups="
            f"{[(g.group_id, g.block_size) for g in self.full_attn_groups]}, "
            f"state_groups="
            f"{[(g.group_id, g.block_size, g.is_mamba_align) for g in self.state_groups]}, "
            f"non_prefix_groups="
            f"{[(g.group_id, g.block_size) for g in self.non_prefix_groups]}"
        )

    @property
    def num_groups(self) -> int:
        return len(self.groups_by_id)

    def compute_block_hashes(self, group: GroupInfo, request: "Request") -> list[bytes]:
        """Hash a request at one group's block boundaries and chain seed."""
        if not group.participates_in_prefix_caching:
            return []
        if group.is_mamba_align:
            # mamba-align pads block table with null blocks; no per-block hash.
            return [b""] * (len(request.all_token_ids) // group.block_size)

        assert group.block_hasher is not None
        return group.block_hasher(request)

    def compute_all_group_block_ids(self, request: "Request") -> list[list[bytes]]:
        """Compute full block hashes for every group, indexed by group_id."""
        return [self.compute_block_hashes(g, request) for g in self.groups_by_id]

    def compute_mamba_align_state_hash(
        self,
        group: GroupInfo,
        seq_len: int,
        group_block_ids: list[list[bytes]],
    ) -> Optional[bytes]:
        """Derive the mamba-align state hash at ``seq_len`` from the prefix hash."""
        if seq_len <= 0 or seq_len % self.lcm_block_size != 0:
            return None
        primary = self.full_attn_groups[0]
        prefix_idx = seq_len // primary.block_size - 1
        if prefix_idx < 0:
            return None
        try:
            prefix_hash = group_block_ids[primary.group_id][prefix_idx]
        except IndexError:
            logger.error(
                "mamba-align state hash missing primary prefix hash: "
                f"group_id={group.group_id}, seq_len={seq_len}, "
                f"primary_group_id={primary.group_id}, "
                f"prefix_idx={prefix_idx}, "
                f"num_primary_hashes="
                f"{len(group_block_ids[primary.group_id])}"
            )
            return None
        if not prefix_hash:
            return None
        return self.request_hasher(
            (group.seed, b"UCM_MAMBA_ALIGN_STATE", seq_len, prefix_hash)
        )

    def lookup_external_hit_tokens(
        self,
        num_computed_tokens: int,
        group_block_ids: list[list[bytes]],
        lookup_on_prefix: Callable[[list[bytes]], int],
        lookup_on_reverse: Callable[[list[bytes]], int],
    ) -> tuple[int, int, list[bytes]]:
        """Two-stage HLA lookup using precomputed per-group hashes.

        Stage 1: min of full-attn groups' ``lookup_on_prefix`` hits, rounded
        down to ``lcm_block_size``. Stage 2: ``lookup_on_reverse`` over the
        mamba state hashes at candidate LCM boundaries; any state group with
        no hit downgrades the external hit to zero. Returns
        (external_hit_tokens, external_hit_lcm_blocks, mamba_prefetch_hashes
        for GC heat update).
        """
        assert len(group_block_ids) == self.num_groups, (
            f"group_block_ids length {len(group_block_ids)} does not match "
            f"num_groups {self.num_groups}"
        )
        assert num_computed_tokens % self.lcm_block_size == 0, (
            f"num_computed_tokens={num_computed_tokens} is not aligned to "
            f"lcm_block_size={self.lcm_block_size}"
        )

        # Stage 1: each full-attn group contributes a candidate hit count.
        candidates: list[int] = []
        for fa in self.full_attn_groups:
            fa_block_ids = group_block_ids[fa.group_id]
            fa_hbm_blocks = num_computed_tokens // fa.block_size
            fa_external = fa_block_ids[fa_hbm_blocks:]
            if not fa_external:
                candidates.append(0)
                continue
            try:
                fa_hit_blocks = lookup_on_prefix(fa_external) + 1
            except Exception as e:
                logger.error(
                    f"full-attn group {fa.group_id} lookup error. "
                    f"{type(e).__name__}: {e}"
                )
                _record_counter("connector_lookup_errors_total")
                candidates.append(0)
                continue
            candidates.append(max(fa_hit_blocks, 0) * fa.block_size)

        # Resume boundary must be a multiple of lcm_block_size so every
        # group's tail/dispatch slicing lands on a real block boundary.
        min_external_hit_tokens = min(candidates)
        external_hit_tokens = (
            min_external_hit_tokens // self.lcm_block_size
        ) * self.lcm_block_size
        if external_hit_tokens <= 0:
            return 0, 0, []

        # Stage 2: reverse scan for mamba state at LCM boundaries — find the
        # rightmost position where ALL state groups' states are present.
        total_hit_tokens = num_computed_tokens + external_hit_tokens

        if not self.state_groups:
            return (
                external_hit_tokens,
                external_hit_tokens // self.lcm_block_size,
                [],
            )

        positions = list(
            range(
                num_computed_tokens + self.lcm_block_size,
                total_hit_tokens + self.lcm_block_size,
                self.lcm_block_size,
            )
        )

        best_pos = total_hit_tokens
        for sg in self.state_groups:
            # Truncate to positions <= best_pos so earlier state groups
            # can shrink the search window for subsequent ones.
            sg_positions = [p for p in positions if p <= best_pos]
            sg_hashes: list[bytes] = []
            for pos in sg_positions:
                state_hash = self.compute_mamba_align_state_hash(
                    sg, pos, group_block_ids
                )
                sg_hashes.append(state_hash if state_hash is not None else b"")
            try:
                idx = lookup_on_reverse(sg_hashes)
            except Exception as e:
                logger.error(
                    f"mamba-align state reverse lookup error for "
                    f"group={sg.group_id}. {type(e).__name__}: {e}"
                )
                _record_counter("connector_lookup_errors_total")
                return 0, 0, []
            if idx < 0:
                # This state group has no state at any candidate position.
                return 0, 0, []
            sg_pos = sg_positions[idx]
            if sg_pos < best_pos:
                best_pos = sg_pos

        external_hit_tokens = best_pos - num_computed_tokens
        if external_hit_tokens <= 0:
            return 0, 0, []

        # Collect mamba state hashes for GC heat update.
        mamba_prefetch_hashes: list[bytes] = []
        for pos in range(
            self.lcm_block_size,
            best_pos + self.lcm_block_size,
            self.lcm_block_size,
        ):
            for sg in self.state_groups:
                state_hash = self.compute_mamba_align_state_hash(
                    sg, pos, group_block_ids
                )
                if state_hash is not None:
                    mamba_prefetch_hashes.append(state_hash)

        return (
            external_hit_tokens,
            external_hit_tokens // self.lcm_block_size,
            mamba_prefetch_hashes,
        )


class HybridLinearAttentionLayout(KVCacheLayout):
    """Per-layer physical layout for hybrid full/linear-attention models.

    Derived purely from ``KVCacheConfig`` descriptors and the live
    ``kv_caches`` tensors. Store columns: one per page type — ``[P]`` for
    group-packed descriptors, ``[main_page, small_page]`` for GLM's
    per-slot descriptors.
    """

    def __init__(
        self,
        kvcaches,
        ucm_config: dict,
        vllm_config: "VllmConfig",
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        super().__init__(kvcaches, ucm_config, vllm_config, kv_cache_config)

    def _build_layout(self, kvcaches) -> None:
        specs = layer_name_to_kv_cache_spec(self.kv_cache_config)
        self._map_layers_to_slots()
        self._build_layer_anchors(kvcaches, specs)
        self._finalize_layer_arrays()

    def _map_layers_to_slots(self) -> None:
        """Assign every cached layer its pair index and page type.

        Group-packed descriptors (``layer_stride > 0``): pair = region index
        within the family, ``(offset - family_base) // layer_stride +
        position``. Per-slot descriptors (``layer_stride == 0``, GLM-Next
        Ascend): pair = the descriptor's ordinal within its type. A
        dump/load for pair p covers every page type the layer's GROUP owns
        (null address for the rest — the column size stays, the store
        skips that copy).
        """
        self.layer_name_to_slot: dict[str, int] = {}
        self.layer_alloc_blocks: dict[str, int] = {}
        self.layer_name_to_group: dict[str, int] = {}
        self.layer_type: dict[str, int] = {}
        self.page_types: list[int] = []
        type_ids: dict[int, int] = {}
        type_counters: dict[int, int] = {}

        # Group numbering must match KVCacheGroupManager (plain enumerate over
        # kv_cache_groups, non-prefix groups included).
        for gid, group in enumerate(self.kv_cache_config.kv_cache_groups):
            for name in group.layer_names:
                self.layer_name_to_group[name] = gid

        num_slots = 0
        descriptors = []
        for raw in self.kv_cache_config.kv_cache_tensors:
            names = _kv_cache_tensor_layers(raw)
            if not names:
                continue
            descriptors.append(
                (
                    names,
                    int(getattr(raw, "layer_stride", 0) or 0),
                    int(getattr(raw, "offset", 0) or 0),
                    int(getattr(raw, "block_stride", 0) or 0),
                    int(getattr(raw, "size", 0) or 0),
                )
            )

        # Family bases for group-packed descriptors: each page-type family
        # lays its regions consecutively from its own base offset.
        family_base: dict[int, int] = {}
        for _, layer_stride, offset, block_stride, _ in descriptors:
            if layer_stride > 0 and block_stride > 0:
                family_base[block_stride] = min(
                    family_base.get(block_stride, offset), offset
                )

        for names, layer_stride, offset, block_stride, size in descriptors:
            type_id = type_ids.get(block_stride)
            if type_id is None:
                type_id = len(self.page_types)
                type_ids[block_stride] = type_id
                self.page_types.append(block_stride)
                type_counters[type_id] = 0
            if layer_stride == 0:
                # Per-slot descriptor: the descriptor IS one (type, pair).
                pair = type_counters[type_id]
                type_counters[type_id] = pair + 1
                alloc_blocks = self.num_blocks
                if block_stride > 0 and size > 0 and size % block_stride == 0:
                    alloc_blocks = size // block_stride
                for name in names:
                    self.layer_name_to_slot[name] = pair
                    self.layer_type[name] = type_id
                    self.layer_alloc_blocks[name] = alloc_blocks
            else:
                base = family_base.get(block_stride, offset)
                rel = offset - base
                if rel % layer_stride != 0:
                    raise ValueError(
                        "KV tensor offset is not aligned to its family's "
                        "region grid: offset="
                        f"{offset}, family_base={base}, "
                        f"layer_stride={layer_stride}, "
                        f"layers={_short_list(names)}"
                    )
                base_slot = rel // layer_stride
                for pos, name in enumerate(names):
                    self.layer_name_to_slot[name] = base_slot + pos
                    self.layer_type[name] = type_id
                    self.layer_alloc_blocks[name] = self.num_blocks
                type_counters[type_id] = max(
                    type_counters[type_id], base_slot + len(names)
                )
            num_slots = max(num_slots, type_counters[type_id])

        self.num_slots = num_slots
        if num_slots <= 0:
            raise ValueError("Hybrid per-layer layout found no KV cache tensor slots.")
        if not self.page_types or any(page <= 0 for page in self.page_types):
            raise ValueError(f"Invalid KV cache page sizes: {self.page_types}")

        missing = [
            name
            for name in self.layer_name_to_group
            if name not in self.layer_name_to_slot
        ]
        if missing:
            raise ValueError(
                "Cached layers missing from KV cache tensors: "
                f"{_short_list(missing)}"
            )

    def _layer_components(
        self,
        layer_name: str,
        kv_layer,
        allocation_blocks: int,
    ) -> list[KVCacheSegment]:
        """Extract one layer's live components with byte-level geometry.

        ``allocation_blocks`` is the number of scheduler blocks the layer's
        backing holds (``num_blocks`` for group-packed descriptors, the
        descriptor's own block count for per-slot allocations which may pad).
        A component's dim-0 must be a multiple of it; each scheduler block
        then covers ``chunks = dim0 // allocation_blocks`` kernel rows.
        """
        if isinstance(kv_layer, torch.Tensor):
            tensors: tuple = (kv_layer,)
        elif isinstance(kv_layer, (tuple, list)):
            tensors = tuple(kv_layer)
        else:
            raise TypeError(
                f"Unsupported KV cache entry for layer={layer_name}: "
                f"type={type(kv_layer)}."
            )

        segments: list[KVCacheSegment] = []
        for tensor in tensors:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"Unsupported KV cache component: layer={layer_name}, "
                    f"type={type(tensor)}."
                )
            if tensor.numel() == 0:
                # Empty views (e.g. an all-rank-0 rope projection) carry no
                # state; they consume no store column.
                continue
            if tensor.dim() < 1 or tensor.shape[0] <= 0:
                raise ValueError(f"Empty KV cache block dim: layer={layer_name}.")
            chunks, remainder = divmod(int(tensor.shape[0]), allocation_blocks)
            if remainder or chunks < 1:
                raise ValueError(
                    "KV component rows do not divide the allocation blocks: "
                    f"layer={layer_name}, rows={tensor.shape[0]}, "
                    f"allocation_blocks={allocation_blocks}"
                )
            inner = math.prod(int(dim) for dim in tensor.shape[1:])
            # The row (dims 1..n-1) must form one dense contiguous run in any
            # dim order (vLLM's default LBNHC stores N before H).
            packing = sorted(
                (int(tensor.stride(dim)), int(tensor.shape[dim]))
                for dim in range(1, tensor.dim())
                if int(tensor.shape[dim]) > 1
            )
            expected_stride = 1
            for stride_dim, shape_dim in packing:
                if stride_dim != expected_stride:
                    raise ValueError(
                        f"KV block component is not a dense page: "
                        f"layer={layer_name}"
                    )
                expected_stride *= shape_dim
            # stride(0) may exceed `inner` legitimately: vllm-ascend
            # interleaves partner K/V kernel blocks between this component's
            # rows, and padded hybrid pages insert inter-row padding. The
            # whole-page copy covers the raw span; only overlaps are invalid.
            element_size = tensor.element_size()
            content_size = chunks * inner * element_size
            block_stride = chunks * int(tensor.stride(0)) * element_size
            if block_stride < content_size:
                raise ValueError(f"Overlapping KV blocks: layer={layer_name}.")
            segments.append(
                KVCacheSegment(
                    ptr=int(tensor.data_ptr()),
                    copy_size=block_stride,
                    block_stride=block_stride,
                    buffer_size=self.num_blocks * block_stride,
                )
            )
        if not segments:
            raise ValueError(f"Empty KV cache entry for layer={layer_name}.")
        return segments

    def _build_layer_anchors(self, kvcaches, specs: dict) -> None:
        """Per-layer component anchors for every prefix-cacheable layer."""
        self.layer_anchors: dict[str, list[KVCacheSegment]] = {}
        for layer_name, kv_layer in kvcaches.items():
            gid = self.layer_name_to_group.get(layer_name)
            if gid is None:
                logger.warning(
                    "KV cache layer not in any cache group, skipping: %s",
                    layer_name,
                )
                continue
            group = self.kv_cache_config.kv_cache_groups[gid]
            if not participates_in_prefix_caching(group.kv_cache_spec):
                # Transient request-local state (e.g. GLM KPoolTail): never
                # dumped or loaded, so it gets no columns and no anchors.
                continue
            if layer_name not in specs:
                logger.warning(
                    "KV cache layer without spec, skipping: %s",
                    layer_name,
                )
                continue
            allocation_blocks = self.layer_alloc_blocks.get(layer_name, self.num_blocks)
            self.layer_anchors[layer_name] = self._layer_components(
                layer_name, kv_layer, allocation_blocks
            )
        if not self.layer_anchors:
            raise ValueError("Hybrid per-layer layout found no cacheable layers.")

    def _finalize_layer_arrays(self) -> None:
        """Store columns (one per page type) plus the flat arrays.

        ``tensor_size_list`` = the page types (Qwen/Kimi: ``[P]``; GLM:
        ``[main_page, small_page]``), ``shard_size`` = their sum,
        ``block_size = shard_size x num_pairs``. Every component must stride
        at its type's page size. Page bases are per (type, pair): the minimum
        component pointer of the layers sharing that pair — the raw page copy
        is interpretation-free because a block is owned by one group at a
        time.
        """
        self.pair_bases: dict[tuple[int, int], int] = {}
        self.group_owns_type: dict[tuple[int, int], bool] = {}
        for name, anchors in self.layer_anchors.items():
            type_id = self.layer_type[name]
            page = self.page_types[type_id]
            for segment in anchors:
                if segment.block_stride != page:
                    raise ValueError(
                        "KV component block stride does not match its page "
                        f"size: layer={name}, "
                        f"stride={segment.block_stride}, page={page}"
                    )
            key = (type_id, self.layer_name_to_slot[name])
            base = min(segment.ptr for segment in anchors)
            self.pair_bases[key] = min(self.pair_bases.get(key, base), base)
            self.group_owns_type[(self.layer_name_to_group[name], type_id)] = True

        tensor_size_list = list(self.page_types)
        self.layer_column_bytes = {
            name: sum(
                page
                for type_id, page in enumerate(self.page_types)
                if self.group_owns_type.get((self.layer_name_to_group[name], type_id))
            )
            for name in self.layer_anchors
        }

        # Direct (whole-block) transfer support: which pairs each group covers
        # (it has a cacheable layer there) and the actual DMA bytes one block
        # of that group transfers (covered pairs × owned page types).
        self.group_pairs: dict[int, set[int]] = defaultdict(set)
        for name in self.layer_anchors:
            self.group_pairs[self.layer_name_to_group[name]].add(
                self.layer_name_to_slot[name]
            )
        self.direct_group_bytes: dict[int, int] = {
            gid: sum(
                self.page_types[type_id]
                for pair in pairs
                for type_id in range(len(self.page_types))
                if self.group_owns_type.get((gid, type_id))
                and (type_id, pair) in self.pair_bases
            )
            for gid, pairs in self.group_pairs.items()
        }

        # ``tensor_size_list`` / ``shard_size`` are base-class properties
        # derived from ``tensor_size_lists``; feed them one row so they stay
        # correct for anything that reads them.
        self.tensor_size_lists = np.asarray([tensor_size_list], dtype=np.uint64)
        self.width = len(tensor_size_list)
        self.slot_shard_size = int(sum(tensor_size_list))

        # Buffer registration: one span per (type, pair) — every generated
        # address lies within [base, base + num_blocks x page).
        pairs = sorted(self.pair_bases)
        self.base_ptrs = np.asarray(
            [self.pair_bases[key] for key in pairs], dtype=np.uint64
        )
        self.buffer_sizes = np.asarray(
            [self.page_types[key[0]] * self.num_blocks for key in pairs],
            dtype=np.uint64,
        )
        self.block_stride_lists = np.asarray(
            [self.page_types[key[0]] for key in pairs], dtype=np.uint64
        )

    def extract_layer_addrs(
        self,
        layer_name: str,
        vllm_block_ids: list[int],
    ) -> np.ndarray:
        """Address matrix [num_blocks, num_page_types] for one layer.

        Column t is the layer's pair's page of type t (``base + block ×
        page``) when the layer's group owns that type, 0 otherwise (the
        column size stays; the store skips the null address). E.g. GLM:
        an attention layer covers [main, small], a mamba layer [main, 0].
        """
        anchors = self.layer_anchors.get(layer_name)
        if anchors is None:
            raise ValueError(f"Layer has no per-layer anchors: {layer_name}")
        pair = self.layer_name_to_slot[layer_name]
        gid = self.layer_name_to_group[layer_name]
        block_ids = np.asarray(vllm_block_ids, dtype=np.uint64)
        addrs = np.zeros((block_ids.shape[0], self.width), dtype=np.uint64)
        for type_id, page in enumerate(self.page_types):
            if not self.group_owns_type.get((gid, type_id)):
                continue
            base = self.pair_bases.get((type_id, pair))
            if base is None:
                continue
            addrs[:, type_id] = base + block_ids * page
        return np.ascontiguousarray(addrs)

    def direct_store_config(self) -> tuple[list[int], int, int]:
        """Store config for whole-block (direct) transfer: one shard per block.

        ``tensor_size_list`` tiles the page-type columns per pair, so
        ``shard_size`` is the full per-block cost and ``block_size ==
        shard_size`` (the commit fires when the single task lands).
        """
        tensor_size_list = list(self.page_types) * self.num_slots
        shard_size = sum(self.page_types) * self.num_slots
        return tensor_size_list, shard_size, shard_size

    def extract_block_addrs(
        self,
        vllm_block_ids: list[int],
        group_ids: list[int],
    ) -> np.ndarray:
        """Address matrix [num_blocks, num_slots * width] for whole-block transfer.

        One row per block, one column per (pair, page type): the raw page
        address when the block's owning group covers that (pair, type), else
        0 — the column size stays and the store skips the null address (e.g.
        GLM mamba blocks cover only main pages).
        """
        if len(vllm_block_ids) != len(group_ids):
            raise ValueError(
                "Direct block/group id lengths differ: "
                f"blocks={len(vllm_block_ids)}, groups={len(group_ids)}"
            )
        block_ids = np.asarray(vllm_block_ids, dtype=np.uint64)
        group_ids_np = np.asarray(group_ids, dtype=np.int64)
        addrs = np.zeros(
            (block_ids.shape[0], self.num_slots * self.width), dtype=np.uint64
        )
        for gid in np.unique(group_ids_np):
            pairs = self.group_pairs.get(int(gid))
            if not pairs:
                continue
            selected = np.flatnonzero(group_ids_np == gid)
            for pair in pairs:
                for type_id, page in enumerate(self.page_types):
                    if not self.group_owns_type.get((int(gid), type_id)):
                        continue
                    base = self.pair_bases.get((type_id, pair))
                    if base is None:
                        continue
                    addrs[selected, pair * self.width + type_id] = (
                        base + block_ids[selected] * page
                    )
        return np.ascontiguousarray(addrs)


class UCMHybridLinearAttentionConnector(UCMDirectConnector, SupportsHMA):
    """Direct (whole-block) connector for hybrid multi-group KV cache layouts.

    The scheduler side (group-aware hashing, two-stage lookup, per-group
    dispatch meta) is shared between direct and layerwise modes.  The worker
    side here is DIRECT: one whole-block task per request at
    ``start_load_kv`` (waited inline), ONE whole-batch dump task in
    ``wait_for_save`` — a single shard per block, no deferred-commit logic.
    The layerwise subclass overrides ``register_kv_caches`` and the whole
    transfer pipeline.
    """

    @classmethod
    def supports_kv_cache_layout(cls, kv_cache_config) -> bool:
        if kv_cache_config is None:
            return False

        if (
            current_platform.device_type != "npu"
            and not current_platform.is_cuda_alike()
        ):
            return False

        layer_to_specs = layer_name_to_kv_cache_spec(kv_cache_config)
        # Attention and Mamba layers may use separate tensor allocations.
        has_full_attention = False
        has_mamba_align = False
        for raw_tensor in kv_cache_config.kv_cache_tensors:
            shared_by = _kv_cache_tensor_layers(raw_tensor)
            shared_specs = [
                spec
                for layer_name in shared_by
                for spec in layer_to_specs.get(layer_name, [])
            ]
            has_full_attention |= any(
                isinstance(spec, FullAttentionSpec) for spec in shared_specs
            )
            has_mamba_align |= any(
                isinstance(spec, MambaSpec) and spec.mamba_cache_mode == "align"
                for spec in shared_specs
            )
            if has_full_attention and has_mamba_align:
                return True

        return False

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(
            vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config
        )
        self._skip_null_vllm_blocks = True
        # group manager only lives on the scheduler side, where ``self._seed``
        # and ``self.request_hasher`` are populated by the parent ctor.
        self.group_manager: Optional[KVCacheGroupManager] = None
        if role == KVConnectorRole.SCHEDULER:
            self.group_manager = KVCacheGroupManager(
                kv_cache_config=kv_cache_config,
                request_hasher=self.request_hasher,
                base_seed=self._seed,
            )
            lcm_block_size = self.group_manager.lcm_block_size
            self.block_size = lcm_block_size
            self.hash_block_size = lcm_block_size
            self._bind_request_block_hasher()

        # ---- worker-side transfer state --------------------------------------
        # Direct uses: request_data / _failure_req_ids / _dump_plan /
        # _load_bytes_total / is_save. The remaining fields are owned by the
        # LAYERWISE pipeline but live here so the shared per-step reset can
        # clear them uniformly; direct leaves them empty.
        self.request_data: list[
            tuple[str, list[bytes], list[bytes], list[int], list[int]]
        ] = []
        self.load_tasks: dict[tuple[int, int], dict[str, Task]] = defaultdict(dict)
        self._load_bytes_by_pair: dict[tuple[int, int], int] = defaultdict(int)
        self._failure_req_ids: set[str] = set()
        self._submitted_positions: set[int] = set()
        self._waited_layers: set[str] = set()
        self._dump_plan: Optional[
            tuple[
                list[bytes],
                list[int],
                list[int],
                set[str],
                dict[str, set[bytes]],
            ]
        ] = None
        self._dump_tasks: list[PendingDumpTask] = []
        self._dumped_layers: set[str] = set()
        # One submission per (group, pair): e.g. GLM's MLA layer and indexer
        # layer share a repeating-unit pair and would dump identical bytes.
        self._dumped_group_slots: set[tuple[int, int]] = set()
        self._submitted_group_slots: set[tuple[int, int]] = set()
        self._waited_group_slots: set[tuple[int, int]] = set()
        self.layer_order: list[str] = []
        self.layer_pos: dict[str, int] = {}
        self.layer_to_group: dict[str, int] = {}
        self.layer_to_slot: dict[str, int] = {}
        self.need_load = False
        self.is_save = False
        # Direct vs layerwise store geometry differs (whole-block shard vs
        # per-pair shard); the layerwise subclass flips this to True.
        self.use_layerwise = False
        self._load_bytes_total = 0
        self._load_bytes_recorded = False
        self._save_bytes_total = 0

        logger.info(f"{type(self).__name__} initialized")

    def get_block_size(self) -> int:
        if self.group_manager is not None:
            return self.group_manager.lcm_block_size
        return self.block_size

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        assert self.group_manager is not None, (
            "get_num_new_matched_tokens must be called on the scheduler-side "
            "connector, where the group manager is initialized."
        )

        lcm_block_size = self.group_manager.lcm_block_size
        assert num_computed_tokens % lcm_block_size == 0, (
            f"num_computed_tokens={num_computed_tokens} is not aligned to "
            f"lcm_block_size={lcm_block_size}"
        )
        hbm_hit_block_num = num_computed_tokens // lcm_block_size

        if self.persist_token_threshold > request.num_tokens:
            logger.info_once(
                f"Skip persistence: req {request.request_id}, "
                f"input tokens ({request.num_tokens}) < threshold "
                f"({self.persist_token_threshold})."
            )
            return 0, False

        try:
            group_ucm_block_ids = self.group_manager.compute_all_group_block_ids(
                request
            )
        except Exception as e:
            logger.error(
                f"request {request.request_id} hash error. " f"{type(e).__name__}: {e}"
            )
            return 0, False
        primary_full_attn = self.group_manager.full_attn_groups[0]
        primary_block_ids = group_ucm_block_ids[primary_full_attn.group_id]

        # Pre-lookup reduction: leave at least recompute_tokens for vLLM to
        # recompute (avoids dispatching the batch as uniform decode into
        # FULL cudagraph; for hybrid also avoids the last blocks whose
        # mamba state may not be valid in HBM at dump time).
        recompute_tokens = self._get_full_hit_recompute_tokens()
        max_hit_lcm_blocks = max(
            0, (request.num_tokens - recompute_tokens) // lcm_block_size
        )
        total_lcm_blocks = request.num_tokens // lcm_block_size

        if max_hit_lcm_blocks < total_lcm_blocks:
            lookup_block_ids = []
            for gid, group in enumerate(self.group_manager.groups_by_id):
                ids = group_ucm_block_ids[gid]
                group_max = max_hit_lcm_blocks * (lcm_block_size // group.block_size)
                lookup_block_ids.append(ids[:group_max])
        else:
            lookup_block_ids = group_ucm_block_ids

        external_hit_tokens, external_hit_lcm_blocks, mamba_prefetch_hashes = (
            self.group_manager.lookup_external_hit_tokens(
                num_computed_tokens,
                lookup_block_ids,
                lambda block_ids: self._rank_consistency.lookup_on_prefix(
                    self.store, block_ids
                ),
                lambda block_ids: self._rank_consistency.lookup_on_reverse(
                    self.store, block_ids
                ),
            )
        )

        if (
            self.enable_record_traces
            and request.request_id not in self.requests_meta
            and len(primary_block_ids) > 0
        ):
            hex_block_ids = [b.hex() for b in primary_block_ids]
            logger.info_once(
                f"timestamp: {time.perf_counter()}, "
                f"input_length: {request.num_tokens}, "
                f"output_length: {request.max_tokens}, "
                f"ucm_block_ids: {hex_block_ids}"
            )

        total_hit_block_num = hbm_hit_block_num + external_hit_lcm_blocks

        # GC heat update for all hit blocks across ranks.
        total_hit_tokens = total_hit_block_num * lcm_block_size
        hbm_hit_full_attn = num_computed_tokens // primary_full_attn.block_size
        total_hit_full_attn = total_hit_tokens // primary_full_attn.block_size
        all_hit_full_attn = primary_block_ids[0:total_hit_full_attn]
        hbm_full_attn = primary_block_ids[0:hbm_hit_full_attn]
        # MLA full-attn is TP-replicated (shared hash), no per-rank entries to prefetch.
        # Only mamba blocks have per-rank entries needing heat update.
        per_rank_hashes = mamba_prefetch_hashes
        if not self.is_mla:
            per_rank_hashes = all_hit_full_attn + mamba_prefetch_hashes
        # Use the base connector's best-effort hotness update. It refreshes
        # rank 0 as well as other ranks and isolates prefetch failures.
        self._prefetch_direct_hit_key_hotness(hbm_full_attn, per_rank_hashes)

        if len(primary_block_ids) > 0:
            ucmmetrics.update_stats(
                {
                    "interval_lookup_hit_rates": external_hit_lcm_blocks
                    * lcm_block_size
                    / (len(primary_block_ids) * primary_full_attn.block_size)
                },
            )

        # No post-lookup workaround needed: pre-lookup truncation already
        # ensures the mamba state at total_hit_tokens is verified.
        num_total_hit_tokens = total_hit_block_num * lcm_block_size

        logger.info_once(
            f"request_id: {request.request_id}, "
            f"total_lcm_blocks: {request.num_tokens // lcm_block_size}, "
            f"hit hbm: {hbm_hit_block_num}, "
            f"hit external: {total_hit_block_num - hbm_hit_block_num}, "
            f"total_tokens: {len(request.all_token_ids)}"
        )

        self.requests_meta[request.request_id] = HLARequestMeta(
            ucm_block_ids=primary_block_ids,
            hbm_hit_block_num=hbm_hit_block_num,
            total_hit_block_num=total_hit_block_num,
            num_token_ids=len(request.all_token_ids),
            token_processed=num_total_hit_tokens,
            group_ucm_block_ids=group_ucm_block_ids,
            group_vllm_block_ids=[[] for _ in range(self.group_manager.num_groups)],
        )

        return external_hit_tokens, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        req_meta = self.requests_meta.get(request.request_id)
        if req_meta is None:
            return
        assert isinstance(req_meta, HLARequestMeta)
        block_ids = blocks.get_block_ids()
        if self.group_manager is not None:
            assert len(block_ids) == self.group_manager.num_groups, (
                f"allocated block group count {len(block_ids)} does not match "
                f"HLA group count {self.group_manager.num_groups}"
            )
        req_meta.group_vllm_block_ids = [list(group) for group in block_ids]

    def _append_mamba_align_state_block(
        self,
        dst_ucm_block_ids: list[bytes],
        dst_vllm_block_ids: list[int],
        dst_group_ids: list[int],
        req_meta: "HLARequestMeta",
        request_id: str,
        gid: int,
        seq_len: int,
        reason: str,
        num_scheduled_tokens: int = 0,
    ) -> None:
        group = self.group_manager.groups_by_id[gid]
        state_idx = max((seq_len - 1) // group.block_size, 0)
        state_indices = [state_idx]
        if reason == "load":
            # External-hit restoration: restore the checkpoint slot for
            # later prefix reuse and seed the worker-selected running slot
            # for the upcoming computation (do not pick speculative or
            # over-allocated table-tail slots).
            running_state_idx = max(
                (seq_len + num_scheduled_tokens - 1) // group.block_size, 0
            )
            # Match the worker's running position; table-tail blocks may be
            # speculative scratch slots or left over from over-allocation.
            state_indices.append(running_state_idx)

        block_ids = req_meta.group_vllm_block_ids[gid]
        vllm_block_ids: list[int] = []
        for vllm_state_idx in state_indices:
            try:
                vllm_block_id = block_ids[vllm_state_idx]
            except IndexError:
                logger.error(
                    "HLA mamba-align state vLLM block missing: "
                    f"request_id={request_id}, group_id={gid}, reason={reason}, "
                    f"seq_len={seq_len}, state_idx={state_idx}, "
                    f"vllm_state_idx={vllm_state_idx}, "
                    f"num_vllm_blocks={len(block_ids)}"
                )
                return
            if vllm_block_id != 0 and vllm_block_id not in vllm_block_ids:
                vllm_block_ids.append(vllm_block_id)
        if not vllm_block_ids:
            return

        ucm_block_id = self.group_manager.compute_mamba_align_state_hash(
            group, seq_len, req_meta.group_ucm_block_ids
        )
        if ucm_block_id is None:
            logger.error(
                "HLA mamba-align state hash missing: "
                f"request_id={request_id}, group_id={gid}, reason={reason}, "
                f"seq_len={seq_len}, state_idx={state_idx}"
            )
            return
        for vllm_block_id in vllm_block_ids:
            dst_ucm_block_ids.append(ucm_block_id)
            dst_vllm_block_ids.append(vllm_block_id)
            dst_group_ids.append(gid)

    def _generate_hla_dispatch_meta(
        self,
        req_meta: "HLARequestMeta",
        new_tokens: int,
        new_vllm_block_ids_per_group: tuple[list[int], ...],
        need_load: bool = True,
        request_id: str = "",
        incoming_block_ids_are_full: bool = False,
    ) -> HLARequestDispatchMeta:
        """Build a flat (ucm, vllm) block id pair list across all groups."""
        assert self.group_manager is not None
        groups_by_id = self.group_manager.groups_by_id
        num_groups = self.group_manager.num_groups
        lcm_block_size = self.group_manager.lcm_block_size

        assert len(new_vllm_block_ids_per_group) == num_groups, (
            f"new_vllm_block_ids_per_group length "
            f"{len(new_vllm_block_ids_per_group)} does not match "
            f"num_groups {num_groups}"
        )
        for gid in range(num_groups):
            incoming_vllm_block_ids = list(new_vllm_block_ids_per_group[gid])
            existing_vllm_block_ids = req_meta.group_vllm_block_ids[gid]
            if incoming_block_ids_are_full:
                req_meta.group_vllm_block_ids[gid] = incoming_vllm_block_ids
            elif not existing_vllm_block_ids:
                req_meta.group_vllm_block_ids[gid] = incoming_vllm_block_ids
            elif incoming_vllm_block_ids:
                suffix_len = len(incoming_vllm_block_ids)
                if existing_vllm_block_ids[-suffix_len:] != incoming_vllm_block_ids:
                    existing_vllm_block_ids.extend(incoming_vllm_block_ids)

        load_ucm_block_ids: list[bytes] = []
        load_vllm_block_ids: list[int] = []
        load_group_ids: list[int] = []
        dump_ucm_block_ids: list[bytes] = []
        dump_vllm_block_ids: list[int] = []
        dump_group_ids: list[int] = []

        external_hit_lcm_blocks = (
            req_meta.total_hit_block_num - req_meta.hbm_hit_block_num
        )
        hbm_hit_tokens = req_meta.hbm_hit_block_num * lcm_block_size
        total_hit_tokens = req_meta.total_hit_block_num * lcm_block_size

        if need_load and external_hit_lcm_blocks > 0:
            # Pass 1: full-attention blocks first (for MLA rank-0-only dump)
            for gid, group in enumerate(groups_by_id):
                if not group.is_full_attention:
                    continue
                load_tok_start = hbm_hit_tokens
                load_tok_end = total_hit_tokens
                start_blk = load_tok_start // group.block_size
                end_blk = load_tok_end // group.block_size
                if start_blk >= end_blk:
                    continue
                extend_non_null(
                    load_ucm_block_ids,
                    load_vllm_block_ids,
                    load_group_ids,
                    req_meta.group_ucm_block_ids[gid][start_blk:end_blk],
                    req_meta.group_vllm_block_ids[gid][start_blk:end_blk],
                    gid,
                )
            load_full_attn_count = len(load_ucm_block_ids) if self.is_mla else 0
            # Pass 2: mamba state blocks
            for gid, group in enumerate(groups_by_id):
                if not group.is_mamba_align:
                    continue
                self._append_mamba_align_state_block(
                    load_ucm_block_ids,
                    load_vllm_block_ids,
                    load_group_ids,
                    req_meta,
                    request_id,
                    gid,
                    total_hit_tokens,
                    "load",
                    num_scheduled_tokens=new_tokens,
                )
        else:
            load_full_attn_count = 0

        if req_meta.token_processed < req_meta.num_token_ids:
            dump_tok_start = req_meta.token_processed
            dump_tok_end = min(
                req_meta.token_processed + new_tokens, req_meta.num_token_ids
            )
            first_lcm_b = (dump_tok_start // lcm_block_size + 1) * lcm_block_size
            last_lcm_b = (dump_tok_end // lcm_block_size) * lcm_block_size

            # Pass 1: full-attention blocks first
            for gid, group in enumerate(groups_by_id):
                if not group.is_full_attention:
                    continue
                start_blk = dump_tok_start // group.block_size
                end_blk = dump_tok_end // group.block_size
                if start_blk >= end_blk:
                    continue
                extend_non_null(
                    dump_ucm_block_ids,
                    dump_vllm_block_ids,
                    dump_group_ids,
                    req_meta.group_ucm_block_ids[gid][start_blk:end_blk],
                    req_meta.group_vllm_block_ids[gid][start_blk:end_blk],
                    gid,
                )
            dump_full_attn_count = len(dump_ucm_block_ids) if self.is_mla else 0
            # Pass 2: mamba state blocks
            for gid, group in enumerate(groups_by_id):
                if not group.is_mamba_align:
                    continue
                if dump_tok_end != last_lcm_b or last_lcm_b < first_lcm_b:
                    continue
                self._append_mamba_align_state_block(
                    dump_ucm_block_ids,
                    dump_vllm_block_ids,
                    dump_group_ids,
                    req_meta,
                    request_id,
                    gid,
                    last_lcm_b,
                    "dump",
                )
        else:
            dump_full_attn_count = 0

        req_meta.token_processed += new_tokens

        return HLARequestDispatchMeta(
            load_block_ids=(load_ucm_block_ids, load_vllm_block_ids),
            dump_block_ids=(dump_ucm_block_ids, dump_vllm_block_ids),
            load_full_attn_count=load_full_attn_count,
            dump_full_attn_count=dump_full_attn_count,
            load_group_ids=load_group_ids,
            dump_group_ids=dump_group_ids,
        )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        assert self.group_manager is not None
        num_groups = self.group_manager.num_groups
        empty_per_group: tuple[list[int], ...] = tuple([] for _ in range(num_groups))

        requests_dispatch_meta: dict[str, HLARequestDispatchMeta] = {}

        for request in scheduler_output.scheduled_new_reqs:
            request_id = request.req_id
            req_meta = self.requests_meta.get(request_id)
            if req_meta is None:
                continue
            assert isinstance(req_meta, HLARequestMeta)
            requests_dispatch_meta[request_id] = self._generate_hla_dispatch_meta(
                req_meta,
                scheduler_output.num_scheduled_tokens[request_id],
                request.block_ids,
                request_id=request_id,
                incoming_block_ids_are_full=True,
            )

        # Same three situations as the parent: chunked prefill (dump only),
        # resumed (load + dump), decode (no-op).
        scheduled_cached_reqs = scheduler_output.scheduled_cached_reqs
        if not isinstance(scheduled_cached_reqs, list):
            for i, request_id in enumerate(scheduled_cached_reqs.req_ids):
                req_meta = self.requests_meta.get(request_id)
                if req_meta is None:
                    continue
                assert isinstance(req_meta, HLARequestMeta)
                raw_new_block_ids = scheduled_cached_reqs.new_block_ids[i]
                new_block_ids = (
                    empty_per_group if raw_new_block_ids is None else raw_new_block_ids
                )
                if hasattr(scheduled_cached_reqs, "resumed_from_preemption"):
                    resumed_from_preemption = (
                        scheduled_cached_reqs.resumed_from_preemption[i]
                    )
                else:
                    resumed_from_preemption = (
                        request_id in scheduled_cached_reqs.resumed_req_ids
                    )
                requests_dispatch_meta[request_id] = self._generate_hla_dispatch_meta(
                    req_meta,
                    scheduler_output.num_scheduled_tokens[request_id],
                    new_block_ids,
                    resumed_from_preemption,
                    request_id=request_id,
                    incoming_block_ids_are_full=resumed_from_preemption,
                )
        else:
            for request in scheduled_cached_reqs:
                request_id = request.req_id
                req_meta = self.requests_meta.get(request_id)
                if req_meta is None:
                    continue
                assert isinstance(req_meta, HLARequestMeta)
                requests_dispatch_meta[request_id] = self._generate_hla_dispatch_meta(
                    req_meta,
                    scheduler_output.num_scheduled_tokens[request_id],
                    request.new_block_ids,
                    request.resumed_from_preemption,
                    request_id=request_id,
                    incoming_block_ids_are_full=request.resumed_from_preemption,
                )

        for request_id in scheduler_output.finished_req_ids:
            self.requests_meta.pop(request_id, None)

        return UCMConnectorMetadata(
            requests_dispatch_meta,
            scheduler_output.preempted_req_ids or set(),
        )

    def _scope_blocks(self, ucm_ids, vllm_ids, group_ids, full_attn_count, is_dump):
        """Rank-scope block IDs for dump or load.

        Returns this rank's unscoped (context) hashes, scoped store keys,
        matching vLLM block IDs, and the matching KV-cache group IDs.

        MLA latent is TP-replicated under the shared rank-0 hash, so for
        dumps each rank writes a proportional contiguous slice of the MLA
        blocks; KDA state is per-rank sharded (rank0 shared / non-rank0
        per-rank hashing). Loads are unchanged: every rank reads all of its
        MLA blocks plus its own KDA blocks.
        """
        is_rank0 = self.tp_rank % self.tp_size == 0
        if not self.is_mla:
            scoped = ucm_ids if is_rank0 else [self.request_hasher(b) for b in ucm_ids]
            return ucm_ids, scoped, vllm_ids, group_ids

        n = int(full_attn_count) if full_attn_count else 0
        mla_ucm, kda_ucm = ucm_ids[:n], ucm_ids[n:]
        mla_vllm, kda_vllm = vllm_ids[:n], vllm_ids[n:]
        mla_groups, kda_groups = group_ids[:n], group_ids[n:]
        # MLA: shared hash for all ranks; KDA: rank0 shared, non-rank0 per-rank hash
        if is_rank0:
            kda_scoped = kda_ucm
        else:
            kda_scoped = [self.request_hasher(b) for b in kda_ucm]
        if is_dump and n and self.tp_size > 1:
            # Spread the TP-replicated MLA dump across ranks; slices are
            # disjoint and their union is the whole MLA block list.
            rank = self.tp_rank % self.tp_size
            start = n * rank // self.tp_size
            end = n * (rank + 1) // self.tp_size
            mla_ucm = mla_ucm[start:end]
            mla_vllm = mla_vllm[start:end]
            mla_groups = mla_groups[start:end]
        return (
            mla_ucm + kda_ucm,
            mla_ucm + kda_scoped,
            mla_vllm + kda_vllm,
            mla_groups + kda_groups,
        )

    def _create_store(
        self,
        kv_cache_layout: Optional[KVCacheLayout],
        cpu_affinity_cores: Optional[list[int]] = None,
        tensor_size_list_override: Optional[list[int]] = None,
        shard_size_override: Optional[int] = None,
        block_size_override: Optional[int] = None,
        unique_id_suffix: str = "",
    ) -> UcmKVStoreBaseV1:
        if len(self.connector_configs) != 1:
            raise RuntimeError(
                f"Expected exactly one connector config, "
                f"but got {len(self.connector_configs)}: "
                f"{self.connector_configs}"
            )

        name = self.connector_configs[0]["ucm_connector_name"]
        module_path = self.connector_configs[0].get("ucm_connector_module_path", None)
        config = copy.deepcopy(self.connector_configs[0]["ucm_connector_config"])
        config.setdefault("share_buffer_enable", self.is_mla)
        if uses_partitioned_cache():
            config["local_rank_size"] = self._get_world_size()
            self._configure_numa_placement(config)
        self._set_default_shm_buffer_capacity(config)
        if "storage_backends" in config:
            backends = [path for path in config["storage_backends"].split(":")]
            config["storage_backends"] = backends
        config["unique_id"] = f"{self.unique_id}{unique_id_suffix}"
        if self._role == KVConnectorRole.WORKER:
            config["device_id"] = self.device_id
            tensor_size_list = _normalize_tensor_size_list(
                tensor_size_list_override
                if tensor_size_list_override is not None
                else kv_cache_layout.tensor_size_list
            )
            config["tensor_size_list"] = tensor_size_list * self.blocks_per_chunk
            shard_size = (
                shard_size_override
                if shard_size_override is not None
                else kv_cache_layout.shard_size
            )
            block_size = (
                block_size_override
                if block_size_override is not None
                else kv_cache_layout.block_size
            )
            config["shard_size"] = shard_size * self.blocks_per_chunk
            config["block_size"] = block_size * self.blocks_per_chunk
            self._publish_block_size(config["block_size"])
            if not uses_partitioned_cache():
                config["local_rank_size"] = self.tp_size if self.is_mla else 1
            buffer_addrs = kv_cache_layout.base_ptrs.reshape(-1).tolist()
            buffer_sizes = kv_cache_layout.buffer_sizes.reshape(-1).tolist()
            gpu_kv_buffer_set = set()
            gpu_kv_buffer_addrs = []
            gpu_kv_buffer_sizes = []
            for addr, size in zip(buffer_addrs, buffer_sizes):
                # Layerwise padding is store metadata only. Never register a
                # ghost (nullptr, zero-sized) slot as a real device buffer.
                if int(addr) == 0 or int(size) == 0:
                    continue
                key = (int(addr), int(size))
                if key in gpu_kv_buffer_set:
                    continue
                gpu_kv_buffer_set.add(key)
                gpu_kv_buffer_addrs.append(key[0])
                gpu_kv_buffer_sizes.append(key[1])
            config["gpu_kv_buffer_addrs"] = gpu_kv_buffer_addrs
            config["gpu_kv_buffer_sizes"] = gpu_kv_buffer_sizes
            if cpu_affinity_cores:
                config["cpu_affinity_cores"] = list(cpu_affinity_cores)
        elif self._gc_owner:
            bs = _scheduler_read_block_size()
            if bs is None:
                config_base = self.block_size * self.element_size * self.head_size
                bs = (
                    config_base
                    * self.num_layers
                    * (1 if self.is_mla else self.num_head * 2)
                    * self.blocks_per_chunk
                )
                logger.warning(f"Falling back to manual block_size estimate: {bs}")
            config["block_size"] = bs
        config["posix_gc_enable"] = self._gc_owner
        logger.info(f"create {name} with config: {config}")
        return UcmConnectorFactoryV1.create_connector(name, config, module_path)

    def _register_common(
        self, kv_caches: dict[str, torch.Tensor]
    ) -> tuple[HybridLinearAttentionLayout, Optional[list[int]], Optional[list[int]]]:
        """Shared registration: layout, layer maps, device, affinity cores.

        Returns ``(layout, worker_cores, store_cores)``; the store itself is
        mode-specific and created by each subclass's ``register_kv_caches``.
        """
        # Block-first: whole-page copy overruns the last block. Not supported yet.
        layout_name = str(
            getattr(self._kv_cache_config, "kv_cache_layout", "") or ""
        ).split(".")[-1]
        if layout_name.startswith("B"):
            raise ValueError(
                f"Block-first KV cache layout '{layout_name}' is not supported"
            )
        if has_ucm_sparse() and os.getenv("VLLM_HASH_ATTENTION") == "1":
            for layer_name, value in kv_caches.items():
                kv_cache, _ = value
                self.kv_caches[layer_name] = kv_cache
        else:
            self.kv_caches = kv_caches

        self.kv_cache_layout = HybridLinearAttentionLayout(
            self.kv_caches, self.launch_config, self._vllm_config, self._kv_cache_config
        )
        layout = self.kv_cache_layout
        assert isinstance(layout, HybridLinearAttentionLayout)
        self.layer_name_to_id = layout.layer_name_to_id
        self.layer_ids = sorted(set(self.layer_name_to_id.values()))
        self.first_layer_id = self.layer_ids[0]

        self.layer_to_group = layout.layer_name_to_group
        self.layer_to_slot = layout.layer_name_to_slot
        # Model execution order: ascending layer index over cacheable layers.
        self.layer_order = sorted(
            layout.layer_anchors.keys(),
            key=lambda name: (layout.layer_name_to_id[name], name),
        )
        self.layer_pos = {name: pos for pos, name in enumerate(self.layer_order)}
        if not self.layer_order:
            raise RuntimeError("Hybrid per-layer layout has no cache layers.")

        self.device = create_device()
        enable_affinity = _use_ucm_connector_cpu_affinity()
        worker_cores, store_cores = (
            self.device.split_cores(self.device_id) if enable_affinity else (None, None)
        )
        return layout, worker_cores, store_cores

    def _create_store_bound(
        self,
        layout: HybridLinearAttentionLayout,
        tensor_size_list: list[int],
        shard_size: int,
        block_size: int,
        worker_cores: Optional[list[int]],
        store_cores: Optional[list[int]],
    ) -> None:
        """Create the store with the mode's geometry and bind worker cores."""
        self.store = self._create_store(
            kv_cache_layout=layout,
            cpu_affinity_cores=store_cores,
            tensor_size_list_override=tensor_size_list,
            shard_size_override=shard_size,
            block_size_override=block_size,
        )
        if worker_cores:
            try:
                os.sched_setaffinity(0, worker_cores)
                logger.info(f"[VLLM CPU Affinity] Worker bound to cores {worker_cores}")
            except Exception as e:
                logger.warning(f"Failed to bind worker: {e}")

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Direct registration: whole-block store — one shard per block.

        ``shard_size`` is the full per-block cost and ``block_size ==
        shard_size``: a single task per block covers every pair.
        """
        layout, worker_cores, store_cores = self._register_common(kv_caches)
        tensor_size_list, shard_size, block_size = layout.direct_store_config()
        self.block_data_size = shard_size
        self._create_store_bound(
            layout,
            tensor_size_list,
            shard_size,
            block_size,
            worker_cores,
            store_cores,
        )

    def _mark_load_failed(
        self,
        metadata: "UCMConnectorMetadata",
        request_id: str,
    ) -> None:
        request_meta = metadata.request_meta.get(request_id)
        if request_meta is not None:
            self._invalid_block_ids.update(request_meta.load_block_ids[1])
        self._failure_req_ids.add(request_id)
        self._connector_worker_meta.mark_failed(request_id)

    def _reset_transfer_state(self) -> None:
        """Per-step reset of the worker-side load/dump transfer state."""
        self.load_tasks.clear()
        self._load_bytes_by_pair.clear()
        self.request_data.clear()
        self._failure_req_ids.clear()
        self._submitted_positions.clear()
        self._waited_layers.clear()
        self.need_load = False
        self._load_bytes_total = 0
        self._load_bytes_recorded = False
        # Dump state must survive a post-forward start_load_kv: with no sync
        # loads in the step, vLLM defers start_load_kv to AFTER the forward,
        # i.e. after the save hooks already submitted dump tasks. Clearing
        # here would orphan them (finish_dump raises "still pending");
        # wait_for_save owns clearing once they complete.
        if (
            self._dump_tasks
            or self._dumped_layers
            or self._dumped_group_slots
            or self._dump_plan is not None
            or self.is_save
        ):
            logger.debug(
                "start_load_kv arrived after save hooks (post-forward "
                "load order); keeping %d in-flight dump task(s)",
                len(self._dump_tasks),
            )
            return
        self._dump_tasks.clear()
        self._dump_plan = None
        self._dumped_layers.clear()
        self._dumped_group_slots.clear()
        self._submitted_group_slots.clear()
        self._waited_group_slots.clear()
        self.is_save = False
        self._save_bytes_total = 0

    def _prepare_load_plan(self, metadata: "UCMConnectorMetadata") -> None:
        """Rank-scope every request's load blocks into ``request_data``."""
        for request_id, request in metadata.request_meta.items():
            if len(request.load_block_ids[0]) == 0:
                continue
            n = getattr(request, "load_full_attn_count", 0)
            _, scoped_ucm, scoped_vllm, scoped_groups = self._scope_blocks(
                request.load_block_ids[0],
                request.load_block_ids[1],
                getattr(request, "load_group_ids", []),
                n,
                is_dump=False,
            )
            if not scoped_ucm:
                continue
            self.need_load = True
            self.request_data.append(
                (
                    request_id,
                    request.load_block_ids[0],
                    scoped_ucm,
                    scoped_vllm,
                    scoped_groups,
                )
            )

    def _ensure_dump_plan(self):
        """Lazily collect rank-scoped dump blocks for the current batch."""
        if self._dump_plan is not None:
            return self._dump_plan
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, UCMConnectorMetadata)
        scoped_ucm_ids: list[bytes] = []
        scoped_vllm_ids: list[int] = []
        scoped_group_ids: list[int] = []
        dump_request_ids: set[str] = set()
        block_ids_by_request: dict[str, set[bytes]] = {}
        for request_id, request in metadata.request_meta.items():
            if len(request.dump_block_ids[0]) == 0:
                continue
            dump_request_ids.add(request_id)
            n = getattr(request, "dump_full_attn_count", 0)
            context_ucm, s_ucm, s_vllm, s_groups = self._scope_blocks(
                request.dump_block_ids[0],
                request.dump_block_ids[1],
                getattr(request, "dump_group_ids", []),
                n,
                is_dump=True,
            )
            if not s_ucm:
                continue
            block_ids_by_request[request_id] = set(context_ucm)
            scoped_ucm_ids.extend(s_ucm)
            scoped_vllm_ids.extend(s_vllm)
            scoped_group_ids.extend(s_groups)
        self._dump_plan = (
            scoped_ucm_ids,
            scoped_vllm_ids,
            scoped_group_ids,
            dump_request_ids,
            block_ids_by_request,
        )
        return self._dump_plan

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        """Direct load: one whole-block task per request, then wait for all."""
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, UCMConnectorMetadata)
        self._reset_transfer_state()
        self._prepare_load_plan(metadata)

        if not self.need_load or not self.layer_order:
            return

        # Ensure do_mamba_copy_block (preprocess_mamba, compute stream) has
        # completed before load DMA (store stream) can overwrite the state
        # it copies.
        self.device.synchronize()
        layout = self.kv_cache_layout
        assert isinstance(layout, HybridLinearAttentionLayout)
        load_start_time = time.perf_counter() * 1000
        num_loaded_block = 0
        num_loaded_request = 0
        request_tasks: list[tuple[str, Task, int]] = []
        for (
            request_id,
            unscoped_ucm,
            scoped_ucm,
            scoped_vllm,
            scoped_groups,
        ) in self.request_data:
            if request_id in self._failure_req_ids:
                continue
            num_loaded_block += len(scoped_ucm)
            num_loaded_request += 1
            try:
                ptrs = layout.extract_block_addrs(scoped_vllm, scoped_groups)
                task = self._rank_consistency.submit_load(
                    self.store,
                    {request_id: unscoped_ucm},
                    scoped_ucm,
                    [0] * len(scoped_ucm),
                    ptrs,
                )
                request_tasks.append((request_id, task, len(scoped_ucm)))
                self._load_bytes_total += sum(
                    layout.direct_group_bytes.get(gid, 0) for gid in scoped_groups
                )
            except Exception as e:
                logger.error(
                    f"request {request_id} submit direct load task error. "
                    f"{type(e).__name__}: {e}"
                )
                self._mark_load_failed(metadata, request_id)
                num_loaded_block -= len(scoped_ucm)
                num_loaded_request -= 1

        for request_id, task, n_blocks in request_tasks:
            try:
                self._rank_consistency.wait_load(task)
            except Exception as e:
                logger.error(
                    f"request {request_id} wait direct load task error. "
                    f"{type(e).__name__}: {e}"
                )
                self._mark_load_failed(metadata, request_id)
                num_loaded_block -= n_blocks

        load_end_time = time.perf_counter() * 1000
        load_duration_ms = load_end_time - load_start_time
        load_speed = self._load_bytes_total / max(load_duration_ms, 1) / 1024 / 1024
        ucmmetrics.update_stats(
            {
                "load_requests_num": num_loaded_request,
                "load_blocks_num": num_loaded_block,
                "load_duration": load_duration_ms,
                "load_speed": load_speed,
                "load_bytes_total": self._load_bytes_total,
            }
        )
        self._load_bytes_total = 0

    def wait_for_layer_load(self, layer_name: str) -> None:
        """No-op in direct mode: loads were fully waited in start_load_kv."""

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        """No-op in direct mode: dumps happen in wait_for_save."""

    def wait_for_save(self) -> None:
        """Direct dump: ONE whole-block task for the whole batch, then finish.

        Every block file has a single shard, so the commit fires when this
        task's writes land — no deferred commit layers, no per-layer catch-up.
        """
        if not self._connector_metadata:
            self._dump_plan = None
            return
        try:
            plan = self._ensure_dump_plan()
            (
                scoped_ucm_ids,
                scoped_vllm_ids,
                scoped_group_ids,
                dump_request_ids,
                block_ids_by_request,
            ) = plan
            if not scoped_ucm_ids:
                return

            layout = self.kv_cache_layout
            assert isinstance(layout, HybridLinearAttentionLayout)
            save_bytes = sum(
                layout.direct_group_bytes.get(gid, 0) for gid in scoped_group_ids
            )
            event_handle = 0
            try:
                ptrs = layout.extract_block_addrs(scoped_vllm_ids, scoped_group_ids)
                event_handle = self._get_dump_event_handle()
                save_start_time = time.perf_counter() * 1000
                task = self._rank_consistency.submit_dump(
                    self.store,
                    block_ids_by_request,
                    scoped_ucm_ids,
                    [0] * len(scoped_ucm_ids),
                    ptrs,
                    event_handle,
                )
            except Exception as e:
                logger.error(f"dump kv cache failed. {type(e).__name__}: {e}")
                if self.enable_event_sync and event_handle and self.device is not None:
                    self.device.destroy_event_handle(event_handle)
                self._rank_consistency.finish_dump(set(block_ids_by_request))
                return

            try:
                self._rank_consistency.wait_dump(task)
                save_end_time = time.perf_counter() * 1000
            except Exception as e:
                logger.error_limit(
                    f"wait for dump kv cache failed. {type(e).__name__}: {e}"
                )
                self._rank_consistency.finish_dump(set(block_ids_by_request))
                return
            finally:
                if self.enable_event_sync and event_handle and self.device is not None:
                    self.device.destroy_event_handle(event_handle)

            self._rank_consistency.finish_dump(set(block_ids_by_request))
            ucmmetrics.update_stats(
                {
                    "save_duration": save_end_time - save_start_time,
                    "save_bytes_total": save_bytes,
                }
            )
        finally:
            self._dump_tasks.clear()
            self._dump_plan = None
            self._dumped_layers.clear()
            self._dumped_group_slots.clear()
            self._submitted_group_slots.clear()
            self._waited_group_slots.clear()
            self.is_save = False

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, object] | None]:
        return False, None


class UCMHybridLinearAttentionLayerWiseConnector(UCMHybridLinearAttentionConnector):
    """Layerwise connector: per-pair store, per-layer dump/load, depth-2 load.

    One shard per (block, pair); the posix commit fires on the max-pair
    shard, so max-pair layers defer their dump to the commit task in
    ``wait_for_save``. Loads pipeline with depth 2: ``start_load_kv``
    submits the first two layers; each ``wait_for_layer_load`` hook waits
    its pair and submits two positions ahead. Dumps fire per layer
    right after the layer's forward (attention layers: vLLM's decorator;
    GDN/KDA layers: ``mamba_layer_hooks_patch``).
    """

    # These families' linear layers have no custom-op boundary, so their
    # module-forward hooks are pruned under compiled execution.
    _MODULE_TIER_ONLY_ARCHITECTURES = (
        "KimiK3ForConditionalGeneration",
        "Glm5NextForConditionalGeneration",
    )

    @classmethod
    def layerwise_engine_config_error(cls, vllm_config) -> Optional[str]:
        """Why layerwise cannot run under this engine config, or None."""
        hf_config = getattr(
            getattr(vllm_config, "model_config", None), "hf_config", None
        )
        architectures = list(getattr(hf_config, "architectures", None) or [])
        matched = [
            arch
            for arch in architectures
            if arch in cls._MODULE_TIER_ONLY_ARCHITECTURES
        ]
        if not matched:
            return None
        cc = getattr(vllm_config, "compilation_config", None)
        mode = str(getattr(getattr(cc, "mode", None), "name", None) or "NONE")
        cg_mode = str(
            getattr(getattr(cc, "cudagraph_mode", None), "name", None) or "NONE"
        )
        if mode == "NONE" and cg_mode != "FULL":
            return None
        return (
            f"Layerwise for {matched[0]} requires mode=NONE and "
            f"cudagraph_mode != FULL (got mode={mode}, "
            f"cudagraph_mode={cg_mode}). Use --enforce-eager or "
            '\'{"mode": "NONE", "cudagraph_mode": "FULL_DECODE_ONLY"}\'.'
        )

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        self.launch_config = copy.deepcopy(self.launch_config)
        self.launch_config["use_layerwise"] = True
        self.use_layerwise = True
        # Deferred-commit plan (filled by register_kv_caches).
        self._max_slot = 0
        self._deferred_dump_layers: set[str] = set()
        self._group_commit_layer: dict[int, str] = {}

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Layerwise registration: per-pair store + deferred commit plan.

        One shard per (block, pair); the posix layer commits a block file
        when its highest shard dump lands — hence the deferred commit plan.
        """
        layout, worker_cores, store_cores = self._register_common(kv_caches)
        self.block_data_size = layout.slot_shard_size
        self._create_store_bound(
            layout,
            list(layout.tensor_size_list),
            layout.slot_shard_size,
            layout.slot_shard_size * layout.num_slots,
            worker_cores,
            store_cores,
        )
        # Max-pair layers defer their dump to the commit task (see class doc).
        self._max_slot = layout.num_slots - 1
        self._deferred_dump_layers = {
            name
            for name in self.layer_order
            if layout.layer_name_to_slot[name] == self._max_slot
        }
        self._group_commit_layer = {}
        for name in self._deferred_dump_layers:
            gid = self.layer_to_group[name]
            existing = self._group_commit_layer.get(gid)
            if (
                existing is None
                or layout.layer_name_to_id[name] > layout.layer_name_to_id[existing]
            ):
                self._group_commit_layer[gid] = name

    def _submit_layer_load(self, pos: int) -> None:
        """Submit load tasks for the layer at execution position ``pos``."""
        if pos < 0 or pos >= len(self.layer_order) or pos in self._submitted_positions:
            return
        layer_name = self.layer_order[pos]
        gid = self.layer_to_group[layer_name]
        slot = self.layer_to_slot[layer_name]
        pair_key = (gid, slot)
        if pair_key in self._submitted_group_slots:
            # A same-(group, pair) layer already submitted these blocks
            # (e.g. GLM's MLA + indexer pair) — nothing left to load.
            self._submitted_positions.add(pos)
            return
        layout = self.kv_cache_layout
        assert isinstance(layout, HybridLinearAttentionLayout)
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, UCMConnectorMetadata)
        for (
            request_id,
            unscoped_ucm,
            scoped_ucm,
            scoped_vllm,
            scoped_groups,
        ) in self.request_data:
            if request_id in self._failure_req_ids:
                continue
            selected = [
                i for i, group_id in enumerate(scoped_groups) if group_id == gid
            ]
            if not selected:
                continue
            sel_vllm = [scoped_vllm[i] for i in selected]
            sel_scoped = [scoped_ucm[i] for i in selected]
            sel_unscoped = [unscoped_ucm[i] for i in selected]
            try:
                ptrs = layout.extract_layer_addrs(layer_name, sel_vllm)
                task = self._rank_consistency.submit_load(
                    self.store,
                    {request_id: sel_unscoped},
                    sel_scoped,
                    [slot] * len(sel_scoped),
                    ptrs,
                )
                self.load_tasks[pair_key][request_id] = task
                self._load_bytes_by_pair[pair_key] += (
                    len(sel_scoped) * layout.layer_column_bytes[layer_name]
                )
            except Exception as e:
                logger.error(
                    f"request {request_id} submit layer load task for "
                    f"{layer_name} failed. {type(e).__name__}: {e}"
                )
                self._mark_load_failed(metadata, request_id)
        self._submitted_positions.add(pos)
        self._submitted_group_slots.add(pair_key)

    def _wait_layer_load_pair(self, pair_key: tuple[int, int]) -> None:
        """Wait and drop one (group, pair)'s per-request load tasks.

        Tasks are keyed by pair, not by layer position: several layers can
        share a pair (GLM's indexer + MLA) and only one of them may ever
        fire a wait hook, so the wait must find the task regardless of
        which position submitted it.
        """
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, UCMConnectorMetadata)
        for request_id, task in self.load_tasks.pop(pair_key, {}).items():
            try:
                self._rank_consistency.wait_load(task)
            except Exception as e:
                logger.error(
                    f"request {request_id} wait layer load failed for "
                    f"group={pair_key[0]} pair={pair_key[1]}. "
                    f"{type(e).__name__}: {e}"
                )
                self._mark_load_failed(metadata, request_id)
        self._load_bytes_total += self._load_bytes_by_pair.pop(pair_key, 0)

    def _record_load_bytes(self) -> None:
        if self._load_bytes_recorded:
            return
        if self._load_bytes_total > 0:
            ucmmetrics.update_stats({"load_bytes_total": self._load_bytes_total})
        self._load_bytes_recorded = True

    def _submit_layer_dump(self, layer_name: str) -> None:
        """Dump one layer's group-owned blocks into its pair's shard.

        Multiple layers of one group can share a pair (GLM's MLA layer and
        indexer layer of the same repeating unit) and would dump identical
        bytes; only the first submission per (group, pair) goes through.
        """
        self._dumped_layers.add(layer_name)
        pair_key = (
            self.layer_to_group[layer_name],
            self.layer_to_slot[layer_name],
        )
        if pair_key in self._dumped_group_slots:
            return
        plan = self._ensure_dump_plan()
        (
            scoped_ucm_ids,
            scoped_vllm_ids,
            scoped_group_ids,
            dump_request_ids,
            block_ids_by_request,
        ) = plan
        if not scoped_ucm_ids:
            self._dumped_group_slots.add(pair_key)
            return
        gid = self.layer_to_group[layer_name]
        selected = [i for i, group_id in enumerate(scoped_group_ids) if group_id == gid]
        if not selected:
            self._dumped_group_slots.add(pair_key)
            return
        layout = self.kv_cache_layout
        assert isinstance(layout, HybridLinearAttentionLayout)
        sel_vllm = [scoped_vllm_ids[i] for i in selected]
        sel_ucm = [scoped_ucm_ids[i] for i in selected]
        slot = self.layer_to_slot[layer_name]
        try:
            ptrs = layout.extract_layer_addrs(layer_name, sel_vllm)
            event_handle = self._get_dump_event_handle()
            task = self._rank_consistency.submit_dump(
                self.store,
                block_ids_by_request,
                sel_ucm,
                [slot] * len(sel_ucm),
                ptrs,
                event_handle,
            )
            self._dump_tasks.append(
                PendingDumpTask(
                    task=task,
                    request_ids=set(dump_request_ids),
                    event_handle=event_handle,
                )
            )
            self._save_bytes_total += (
                len(sel_ucm) * layout.layer_column_bytes[layer_name]
            )
            self.is_save = True
            self._dumped_group_slots.add(pair_key)
        except Exception as e:
            logger.error(
                f"submit layer dump task for {layer_name} failed. "
                f"{type(e).__name__}: {e}"
            )

    def _submit_commit_dump(self) -> None:
        """One ordered write of the max-pair shard for every dumped block.

        Blocks whose group owns a max-pair layer get that layer's real
        component addresses; blocks of groups without a max-pair layer get
        null rows — nothing was ever written to that shard for them, and the
        null write only triggers the block-file commit.
        """
        plan = self._dump_plan
        if plan is None:
            return
        (
            scoped_ucm_ids,
            scoped_vllm_ids,
            scoped_group_ids,
            dump_request_ids,
            block_ids_by_request,
        ) = plan
        if not scoped_ucm_ids:
            return
        layout = self.kv_cache_layout
        assert isinstance(layout, HybridLinearAttentionLayout)
        addrs = np.zeros((len(scoped_vllm_ids), layout.width), dtype=np.uint64)
        save_bytes = 0
        group_ids_np = np.asarray(scoped_group_ids, dtype=np.int64)
        for gid in np.unique(group_ids_np):
            commit_layer = self._group_commit_layer.get(int(gid))
            if commit_layer is None:
                continue
            selected = np.flatnonzero(group_ids_np == gid).tolist()
            sel_vllm = [scoped_vllm_ids[i] for i in selected]
            addrs[selected] = layout.extract_layer_addrs(commit_layer, sel_vllm)
            save_bytes += len(selected) * layout.layer_column_bytes[commit_layer]
        try:
            event_handle = self._get_dump_event_handle()
            task = self._rank_consistency.submit_dump(
                self.store,
                block_ids_by_request,
                scoped_ucm_ids,
                [self._max_slot] * len(scoped_ucm_ids),
                np.ascontiguousarray(addrs),
                event_handle,
            )
            self._dump_tasks.append(
                PendingDumpTask(
                    task=task,
                    request_ids=set(dump_request_ids),
                    event_handle=event_handle,
                )
            )
            self._save_bytes_total += save_bytes
            self.is_save = True
        except Exception as e:
            logger.error(f"submit commit dump task failed. {type(e).__name__}: {e}")

    def _wait_dump_tasks(self) -> None:
        """Wait and consume every pending dump task (each task waits once)."""
        while self._dump_tasks:
            pending_dump_task = self._dump_tasks.pop(0)
            try:
                self._rank_consistency.wait_dump(pending_dump_task.task)
            except Exception as e:
                logger.error_limit(
                    f"wait for dump kv cache failed. {type(e).__name__}: {e}"
                )

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, UCMConnectorMetadata)
        self._reset_transfer_state()
        self._prepare_load_plan(metadata)

        if not self.need_load or not self.layer_order:
            return

        # Ensure do_mamba_copy_block (preprocess_mamba, compute stream)
        # has completed before load DMA (store stream) can overwrite the
        # state it copies.
        self.device.synchronize()
        # Depth-2 pipeline: submit the first two layers only; each layer's
        # wait_for_layer_load hook then waits its own layer and submits the
        # layer two positions ahead.
        self._submit_layer_load(0)
        self._submit_layer_load(1)

    def wait_for_layer_load(self, layer_name: str) -> None:
        if not self._connector_metadata or not self.need_load:
            return
        # Entry dedup on the layer's own name: repeated calls (e.g. the GLM
        # entry hook firing again inside the MLA op's own
        # @maybe_transfer_kv_layer) short-circuit here.
        if layer_name not in self.layer_pos or layer_name in self._waited_layers:
            return
        self._waited_layers.add(layer_name)
        pos = self.layer_pos[layer_name]
        # Wait by (group, pair), not by position: a pair's load task may sit
        # at a partner layer's position (GLM's indexer sorts before its MLA
        # and submits the pair), and only one of the two may fire a hook.
        pair_key = (
            self.layer_to_group[layer_name],
            self.layer_to_slot[layer_name],
        )
        if pair_key not in self._waited_group_slots:
            self._waited_group_slots.add(pair_key)
            self._wait_layer_load_pair(pair_key)
            if pos == len(self.layer_order) - 1:
                self._record_load_bytes()
        # Keep at most two layers in flight: the next position was normally
        # submitted two waits ago; resubmitting is a no-op and covers a gap
        # where an earlier layer's wait hook never fired.
        self._submit_layer_load(pos + 1)
        self._submit_layer_load(pos + 2)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        if not self._connector_metadata:
            return
        if layer_name in self._dumped_layers:
            return
        if layer_name in self._deferred_dump_layers:
            # Deferred to the commit task in wait_for_save (see class doc).
            self._dumped_layers.add(layer_name)
            self._dumped_group_slots.add(
                (self.layer_to_group[layer_name], self.layer_to_slot[layer_name])
            )
            return
        if layer_name not in self.layer_pos:
            return
        self._submit_layer_dump(layer_name)

    def wait_for_save(self) -> None:
        """Layerwise dump: catch up undumped layers, then the max-pair commit."""
        if self._connector_metadata:
            self._ensure_dump_plan()
            # Dump every layer not yet dumped (hooks that never fired,
            # e.g. full-graph decode steps).
            for layer_name in self.layer_order:
                if layer_name in self._dumped_layers:
                    continue
                if layer_name in self._deferred_dump_layers:
                    self._dumped_layers.add(layer_name)
                    continue
                self._submit_layer_dump(layer_name)

        if self._dump_plan is not None:
            self._submit_commit_dump()
        self._wait_dump_tasks()

        self._record_load_bytes()
        if not self.is_save:
            self._dump_tasks.clear()
            self._dump_plan = None
            self._dumped_layers.clear()
            self._dumped_group_slots.clear()
            self._submitted_group_slots.clear()
            self._waited_group_slots.clear()
            return

        dump_request_ids = self._dump_plan[3] if self._dump_plan is not None else set()
        self._rank_consistency.finish_dump(dump_request_ids)
        if self._save_bytes_total > 0:
            ucmmetrics.update_stats({"save_bytes_total": self._save_bytes_total})
            self._save_bytes_total = 0
        # Clear the per-step dedup sets here (mirroring direct mode): with
        # the post-forward load order, start_load_kv no longer clears them
        # between steps.
        self._dump_tasks.clear()
        self._dump_plan = None
        self._dumped_layers.clear()
        self._dumped_group_slots.clear()
        self._submitted_group_slots.clear()
        self._waited_group_slots.clear()
        self.is_save = False
        if self.enable_event_sync:
            self.device.destroy_event_handles()


class UCMHLALiteConnector(UCMLiteConnector, SupportsHMA):
    """UCM Lite connector for full-attention + linear-attention hybrids.

    A thin subclass of :class:`UCMLiteConnector` that only adds a
    :class:`KVCacheGroupManager`, so the startup ``KVCacheGroupManager
    initialized:`` line records the HLA topology for the offline hit-rate
    simulator. Logs only the prefix chain per request (the simulator
    derives mamba state keys from it).
    """

    @staticmethod
    def is_glm53_flash_architecture(vllm_config: "VllmConfig") -> bool:
        architectures = (
            getattr(vllm_config.model_config.hf_config, "architectures", None) or []
        )
        return "Glm5NextForConditionalGeneration" in architectures

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ) -> None:
        if kv_cache_config is None:
            raise RuntimeError("UCMHLALiteConnector requires kv_cache_config.")
        super().__init__(vllm_config, role, kv_cache_config)
        self.group_manager = KVCacheGroupManager(
            kv_cache_config=kv_cache_config,
            request_hasher=self.request_hasher,
            base_seed=self._seed,
        )
        self.block_size = self.group_manager.lcm_block_size
        self.hash_block_size = self.group_manager.lcm_block_size
        self.trace_hash_block_size = self.hash_block_size
        is_mla = getattr(vllm_config.model_config, "is_deepseek_mla", False)
        hbm_data_size = self._compute_hla_block_data_size(
            vllm_config, kv_cache_config, is_mla
        )
        logger.info(
            f"UCMTraceMeta: type=mamba, "
            f"is_mla={is_mla}, "
            f"vllm_hash_block_size={self.hash_block_size}, "
            f"trace_hash_block_size={self.trace_hash_block_size}, "
            f"hbm_block_data_size={hbm_data_size}, "
            f"lcm_block_size={self.group_manager.lcm_block_size}, "
            f"mamba_groups={len(self.group_manager.state_groups)}"
        )
        logger.info("Init UCMHLALiteConnector.")

    @staticmethod
    def _compute_hla_block_data_size(
        vllm_config: "VllmConfig", kv_cache_config: "KVCacheConfig", is_mla: bool
    ) -> int:
        """Per-rank HBM bytes per vLLM block id, for the trace hit-rate
        simulator's HBM BlockPool sizing.

        Prefers the exact figure from the planned ``kv_cache_tensors``
        (per-slot descriptors sum; group-packed descriptors share one
        backing). Falls back to a model-config derivation: page = block_size
        × attn_per_token, block_size aligned so the page also covers
        mamba_total (alignment 16 for GQA hybrids, 128 for MLA hybrids).
        """
        tensors = list(getattr(kv_cache_config, "kv_cache_tensors", None) or [])
        num_blocks = int(getattr(kv_cache_config, "num_blocks", 0) or 0)
        if tensors and num_blocks > 0:
            sizes = [int(getattr(t, "size", 0) or 0) for t in tensors]
            if all(int(getattr(t, "layer_stride", 0) or 0) == 0 for t in tensors):
                total = sum(sizes)  # one descriptor per physical slot
            else:
                # Group-packed descriptors alias one shared backing.
                total = max(sizes)
            if total > 0 and total % num_blocks == 0:
                return total // num_blocks

        mc = vllm_config.model_config
        hf = mc.hf_text_config
        tp = vllm_config.parallel_config.tensor_parallel_size

        num_full = getattr(hf, "num_full_attn_layers", None)
        num_linear = getattr(hf, "num_linear_layers", None)
        if num_full is None or num_linear is None:
            layer_types = getattr(hf, "layer_types", None)
            if layer_types and isinstance(layer_types, list):
                num_full = sum(1 for t in layer_types if t == "full_attention")
                num_linear = sum(1 for t in layer_types if t != "full_attention")
            else:
                return 0

        has_mtp = (
            vllm_config.speculative_config is not None
            and vllm_config.speculative_config.num_speculative_tokens > 0
        )
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens if has_mtp else 0
        )
        num_full_eff = num_full + (1 if has_mtp else 0)
        num_tensors = min(num_full_eff, num_linear)
        if num_tensors <= 0:
            return 0

        dt_map = {"bfloat16": 2, "float16": 2, "float32": 4, "int8": 1}
        model_dt = dt_map.get(str(getattr(mc, "dtype", "bfloat16")), 2)
        ssm_dt = dt_map.get(
            str(getattr(hf, "mamba_ssm_dtype", getattr(mc, "dtype", "float32"))), 4
        )

        lin_kh = getattr(hf, "linear_num_key_heads", 0)
        lin_vh = getattr(hf, "linear_num_value_heads", 0)
        lin_kd = getattr(hf, "linear_key_head_dim", 0)
        lin_vd = getattr(hf, "linear_value_head_dim", 0)
        conv_k = getattr(hf, "linear_conv_kernel_dim", 4)

        v_heads_per_rank = lin_vh / tp if tp else lin_vh
        ssm_size = int(v_heads_per_rank * lin_vd * lin_kd * ssm_dt)
        conv_dim = lin_kd * lin_kh * 2 + lin_vd * lin_vh
        conv_dim_per_rank = conv_dim / tp if tp else conv_dim
        conv_size = int((conv_k - 1 + num_spec) * conv_dim_per_rank * model_dt)
        mamba_total = conv_size + ssm_size

        if is_mla:
            kv_lora = getattr(hf, "kv_lora_rank", 0)
            qk_rope = getattr(hf, "qk_rope_head_dim", 0)
            attn_per_tok = (kv_lora + qk_rope) * 1 * model_dt
        else:
            head_dim = getattr(hf, "head_dim", 0) or (
                getattr(hf, "hidden_size", 0)
                // max(getattr(hf, "num_attention_heads", 1), 1)
            )
            kv_heads = getattr(
                hf, "num_key_value_heads", getattr(hf, "num_attention_heads", 0)
            )
            # GQA KV heads are divided by TP, replicated (min 1 per rank)
            # when there are fewer heads than ranks — mirrors vLLM's
            # ModelConfig.get_num_kv_heads.
            kv_heads_per_rank = max(1, kv_heads // tp) if tp else kv_heads
            attn_per_tok = 2 * head_dim * int(kv_heads_per_rank) * model_dt

        if not attn_per_tok:
            return 0

        bs = 0
        for g in kv_cache_config.kv_cache_groups:
            bs = max(bs, block_size_from_kv_cache_spec(g.kv_cache_spec))

        if not bs:
            # Derive block_size so the attention page also covers the mamba
            # state (vLLM 0.30 pads conv+ssm into the attention page). The
            # alignment kernel follows the attention type, not the platform.
            kernel = 128 if is_mla else 16
            ratio = -(-mamba_total // (kernel * attn_per_tok)) if attn_per_tok else 1
            bs = kernel * ratio

        if UCMHLALiteConnector.is_glm53_flash_architecture(vllm_config):
            # GLM-5.3-Flash: one block id spans two physical page families.
            # Main slots hold the MLA latent / KDA state page; small slots
            # (one per MLA layer) hold the indexer compressed K — entries =
            # bs / index_kpool, 132 B per entry on CUDA, 256 B on Ascend.
            # Main slot count follows the mamba grouping: CUDA round-robins
            # KDA into 4 groups over 11 per-layer slots; Ascend run-strides
            # into 3 groups (12/11/11) over 12 per-slot backings.
            is_ascend = current_platform.device_type == "npu"
            index_head_dim = int(getattr(hf, "index_head_dim", 0) or 0)
            index_kpool = int(getattr(hf, "index_kpool", 0) or 0)
            main_page = bs * attn_per_tok
            if index_head_dim and index_kpool and bs % index_kpool == 0:
                entries = bs // index_kpool
                entry_bytes = index_head_dim * 2 if is_ascend else index_head_dim + 4
                small_page = entries * entry_bytes
            else:
                small_page = 0
            main_slots = -(-num_linear // 3) if is_ascend else num_full_eff
            return main_slots * main_page + num_full_eff * small_page

        page_size = bs * attn_per_tok

        return num_tensors * page_size

    def get_block_size(self) -> int:
        return self.group_manager.lcm_block_size

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, object] | None]:
        self.requests_meta.pop(request.request_id, None)
        return False, None
