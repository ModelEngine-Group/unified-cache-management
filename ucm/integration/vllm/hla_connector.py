import copy
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

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
    UniformTypeKVCacheSpecs,
)

from ucm.integration.vllm.device import create_device
from ucm.integration.vllm.ucm_connector import (
    KVCacheLayout,
    PendingDumpTask,
    RequestDispatchMeta,
    RequestHasher,
    RequestMeta,
    UCMConnectorMetadata,
    UCMDirectConnector,
    _drop_null_vllm_blocks,
    _record_counter,
    _short_list,
)
from ucm.logger import init_logger
from ucm.shared.metrics import ucmmetrics
from ucm.sparse.state import has_ucm_sparse
from ucm.store.ucmstore_v1 import Task, UcmKVStoreBaseV1

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class HLARequestMeta(RequestMeta):
    """RequestMeta extended with per-group block tracking for hybrid models.

    The inherited fields (``ucm_block_ids``, ``hbm_hit_block_num``,
    ``total_hit_block_num``, ``num_token_ids``, ``vllm_block_ids``,
    ``token_processed``) keep their original semantics and mirror the
    full-attention group exactly, so dispatch/load/save paths inherited from
    :class:`UCMDirectConnector` keep working.

    The two new fields are 2D lists indexed by the original
    ``kv_cache_config.kv_cache_groups`` order (i.e. ``[group_id]``):
    - ``group_ucm_block_ids[gid]``: full block hashes obtained by hashing
      ``request.all_token_ids`` with group ``gid``'s own block size and
      chain seed. ``group_ucm_block_ids[full_attn_group_id]`` equals the
      inherited ``ucm_block_ids``.
    - ``group_vllm_block_ids[gid]``: per-group VLLM physical block ids; this
      is initialized as an empty list per group here, then filled from the
      scheduler allocation snapshot by :meth:`UCMHybridLinearAttentionConnector.update_state_after_alloc`
      and maintained by :meth:`UCMHybridLinearAttentionConnector._generate_hla_dispatch_meta`.
      HLA dispatch later slices these per-group tables to build the flattened
      load/dump pairs consumed by the inherited I/O path.
    """

    group_ucm_block_ids: list[list[bytes]] = field(default_factory=list)
    group_vllm_block_ids: list[list[int]] = field(default_factory=list)


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
    block_size = 0
    if isinstance(spec, UniformTypeKVCacheSpecs):
        block_size = next(iter(spec.kv_cache_specs.values())).block_size
    else:
        block_size = spec.block_size

    return block_size


def is_mamba_align_kv_cache_spec(spec: KVCacheSpec) -> bool:
    if isinstance(spec, UniformTypeKVCacheSpecs):
        sample = next(iter(spec.kv_cache_specs.values()))
        return is_mamba_align_kv_cache_spec(sample)
    return isinstance(spec, MambaSpec) and spec.mamba_cache_mode == "align"


def extend_non_null(
    dst_ucm_block_ids: list[bytes],
    dst_vllm_block_ids: list[int],
    src_ucm_block_ids: list[bytes],
    src_vllm_block_ids: list[int],
) -> None:
    # Mamba align mode pads req block tables with vLLM's null block
    # (block_id=0). These are metadata placeholders, not physical pages
    # that should be loaded from or dumped to the store.
    for ucm_block_id, vllm_block_id in zip(src_ucm_block_ids, src_vllm_block_ids):
        if vllm_block_id == 0:
            continue
        dst_ucm_block_ids.append(ucm_block_id)
        dst_vllm_block_ids.append(vllm_block_id)


@dataclass
class GroupInfo:
    """Per-group metadata used by :class:`KVCacheGroupManager`."""

    group_id: int
    block_size: int
    layer_names: tuple[str, ...]
    # Independent hash chain seed per group (see ``KVCacheGroupManager``).
    seed: bytes
    is_mamba_align: bool = False

    @property
    def is_full_attention(self) -> bool:
        return not self.is_mamba_align


class KVCacheGroupManager:
    """Group-aware hashing and lookup for hybrid (HLA) connectors.

    Splits ``kv_cache_config.kv_cache_groups`` into full-attention groups
    (one or more) and mamba-align state groups, derives a per-group hash
    chain seed, and exposes a two-stage lookup that:

    1. For every full-attention group, hashes ``request.all_token_ids`` with
       that group's block size and runs ``store.lookup_on_prefix`` on the
       blocks beyond its own ``hbm_hit_block_num``. The candidate hits (in
       tokens) are min'd across full-attn groups and rounded down to
       ``lcm_block_size``.
    2. For each mamba-align state group, derives a single state hash from
       the primary full-attention prefix and verifies it exists in the
       store. If any state group fails this check, the whole external hit
       is downgraded to zero.
    """

    def __init__(
        self,
        kv_cache_config: "KVCacheConfig",
        request_hasher: "RequestHasher",
        base_seed: bytes,
    ) -> None:
        self.request_hasher = request_hasher
        # Indexed by original group_id; positions match
        # ``kv_cache_config.kv_cache_groups``.
        self.groups_by_id: list[GroupInfo] = []
        # All full-attention groups (non-mamba-align). Order follows group_id.
        self.full_attn_groups: list[GroupInfo] = []
        self.state_groups: list[GroupInfo] = []

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
            )
            self.groups_by_id.append(info)
            if info.is_full_attention:
                self.full_attn_groups.append(info)
            else:
                self.state_groups.append(info)

        assert len(self.full_attn_groups) >= 1, (
            "UCMHybridLinearAttentionConnector expects at least one full-attention group in "
            "kv_cache_config.kv_cache_groups."
        )

        # Resume points must be aligned to the LCM of every group's
        # block_size so that per-group block accounting (including each
        # full-attn group's lookup result and every state group's tail slice)
        # lands on a clean block boundary.
        all_block_sizes = [g.block_size for g in self.groups_by_id]
        self.lcm_block_size: int = math.lcm(*all_block_sizes)

        for g in self.groups_by_id:
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
            # Mamba-align state occupies exactly one block per checkpoint.
            # ``block_size <= lcm_block_size`` is already guaranteed by the
            # LCM construction above.

        logger.info(
            "KVCacheGroupManager initialized: "
            f"lcm_block_size={self.lcm_block_size}, "
            f"full_attn_groups="
            f"{[(g.group_id, g.block_size) for g in self.full_attn_groups]}, "
            f"state_groups="
            f"{[(g.group_id, g.block_size, g.is_mamba_align) for g in self.state_groups]}"
        )

    @property
    def num_groups(self) -> int:
        return len(self.groups_by_id)

    def compute_block_hashes(
        self, group: GroupInfo, token_ids: list[int]
    ) -> list[bytes]:
        """Hash ``token_ids`` into per-block ids using ``group``'s chain seed."""
        if group.is_mamba_align:
            # In mamba-align mode vLLM pads the per-request block table with
            # block_id=0 and only keeps the current state block as a real
            # physical page. Hashing every logical token block here would
            # create keys for pages that can never be loaded or dumped.
            return [b""] * (len(token_ids) // group.block_size)

        ret: list[bytes] = []
        parent = group.seed
        block_size = group.block_size
        for start in range(0, len(token_ids), block_size):
            end = start + block_size
            block_token_ids = token_ids[start:end]
            if len(block_token_ids) < block_size:
                break
            hash_value = self.request_hasher((parent, tuple(block_token_ids)))
            parent = hash_value
            ret.append(hash_value)
        return ret

    def compute_all_group_block_ids(self, token_ids: list[int]) -> list[list[bytes]]:
        """Compute full block hashes for every group, indexed by group_id.

        ``ret[gid]`` covers all aligned blocks of ``token_ids`` using group
        ``gid``'s ``block_size`` and chain seed. The trailing partial block
        (if any) is dropped, matching :meth:`compute_block_hashes`.
        """
        return [self.compute_block_hashes(g, token_ids) for g in self.groups_by_id]

    def compute_mamba_align_state_hash(
        self,
        group: GroupInfo,
        seq_len: int,
        group_block_ids: list[list[bytes]],
    ) -> Optional[bytes]:
        """Derive the hash for the real mamba-align state page at ``seq_len``.

        The mamba state represents the whole prefix up to ``seq_len`` instead
        of a normal KV block. We derive its key from the primary full-attention
        prefix hash, so the state key still changes with every prefix token but
        we do not need to materialize hashes for mamba's leading null blocks.
        """
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
        store: "UcmKVStoreBaseV1",
        group_block_ids: list[list[bytes]],
    ) -> tuple[int, int, list[bytes]]:
        """Two-stage HLA lookup using precomputed per-group hashes.

        ``group_block_ids`` must have one entry per group, indexed by the
        original ``group_id`` (see :meth:`compute_all_group_block_ids`).

        Stage 1 — every full-attention group runs ``lookup_on_prefix``
        beyond its own ``hbm_hit_block_num``; the candidate hits are taken
        as a min and rounded down to ``lcm_block_size`` so the final
        external hit is consistent across all full-attn groups and aligns
        to the kv-cache page granularity expected by the scheduler.

        Stage 2 — mamba-align state groups are checked via a sequential
        backward scan: starting from the Stage 1 candidate position, each
        LCM boundary is checked one at a time going backwards toward
        ``num_computed_tokens``. The scan stops at the first position where
        ALL state groups have their state present. If no position has all
        states present, the external hit is downgraded to zero.

        Returns:
            Tuple of
            - ``external_hit_tokens``: tokens hit beyond ``num_computed_tokens``,
              aligned to ``lcm_block_size``. ``0`` if any check fails.
            - ``external_hit_lcm_blocks``: ``external_hit_tokens //
              lcm_block_size`` (also ``0`` on downgrade).
            - ``mamba_prefetch_hashes``: rank-0 mamba state hashes from
              ``num_computed_tokens + lcm_block_size`` to ``best_pos``,
              for GC heat update (rank-0 un-checked positions + other ranks).
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
                fa_hit_blocks = store.lookup_on_prefix(fa_external) + 1
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

        # Stage 2: sequential backward scan for mamba-align state.
        # Check one LCM boundary at a time from total_hit_tokens downwards.
        # Stop at the first position where ALL state groups are present.
        total_hit_tokens = num_computed_tokens + external_hit_tokens

        if not self.state_groups:
            return (
                external_hit_tokens,
                external_hit_tokens // self.lcm_block_size,
                [],
            )

        best_pos = num_computed_tokens
        for pos in range(total_hit_tokens, num_computed_tokens, -self.lcm_block_size):
            pos_hashes: list[bytes] = []
            for sg in self.state_groups:
                state_hash = self.compute_mamba_align_state_hash(
                    sg, pos, group_block_ids
                )
                pos_hashes.append(state_hash if state_hash is not None else b"")
            try:
                results = store.lookup(pos_hashes)
            except Exception as e:
                logger.error(
                    f"mamba-align state lookup error at pos={pos}. "
                    f"{type(e).__name__}: {e}"
                )
                _record_counter("connector_lookup_errors_total")
                return 0, 0, []
            if all(results):
                best_pos = pos
                break

        external_hit_tokens = best_pos - num_computed_tokens
        if external_hit_tokens <= 0:
            return 0, 0, []

        # Collect mamba state hashes for all LCM boundaries from the first
        # boundary (lcm_block_size) to best_pos (inclusive) for GC heat
        # update.  This covers both HBM positions (below num_computed_tokens,
        # not checked by the sequential scan) and external positions.
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
    """Physical layout for hybrid full-attention + linear-attention pages.

    vLLM may back full-attention and linear-attention layers with one shared
    raw int8 tensor. The physical layout is backend dependent:

    - Ascend stores the shared page in component-major order:
        [conv_block_or_padding, k_or_ssm_block, v_block_or_padding]
      across all physical blocks.
    - CUDA stores one contiguous page per physical block. The same bytes are
      viewed as either attention [K, V] or mamba [conv, ssm, padding].

    The store receives one unified tensor_size_list, so we expose the three
    physical slices for Ascend, while CUDA is exposed as one contiguous page
    with a full-page stride.
    """

    def __init__(
        self,
        kvcaches,
        ucm_config: dict,
        vllm_config: "VllmConfig",
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(kvcaches, ucm_config, vllm_config, kv_cache_config)

    @staticmethod
    def _dtype_size(dtype: torch.dtype) -> int:
        return torch.empty((), dtype=dtype).element_size()

    @staticmethod
    def _mamba_component_sizes(spec: MambaSpec) -> list[int]:
        return [
            math.prod(shape) * HybridLinearAttentionLayout._dtype_size(dtype)
            for shape, dtype in zip(spec.shapes, spec.dtypes)
        ]

    @staticmethod
    def _attention_component_sizes(spec: KVCacheSpec) -> tuple[int, int]:
        assert isinstance(spec, FullAttentionSpec)
        k_size = (
            spec.block_size
            * spec.num_kv_heads
            * spec.head_size
            * HybridLinearAttentionLayout._dtype_size(spec.dtype)
        )
        head_size_v = getattr(spec, "head_size_v", spec.head_size)
        v_size = (
            spec.block_size
            * spec.num_kv_heads
            * head_size_v
            * HybridLinearAttentionLayout._dtype_size(spec.dtype)
        )
        return k_size, v_size

    def _finalize_layout_arrays(
        self,
        base_ptrs: list[list[int]],
        buffer_size_rows: list[list[int]],
        tensor_size_lists: list[list[int]],
        block_stride_lists: list[list[int]],
    ) -> None:
        # MTP can add attention-only raw tensors next to hybrid attention+Mamba
        # tensors.  Those rows naturally have a different number of physical
        # slices, so keep the UCM schema flattened instead of forcing a
        # rectangular layer-by-slice matrix.
        self.row_slices: list[slice] = []
        self.row_tensor_size_lists: list[list[int]] = [
            [int(size) for size in row] for row in tensor_size_lists
        ]
        self.row_shard_sizes: list[int] = [
            sum(row) for row in self.row_tensor_size_lists
        ]

        offset = 0
        for row in tensor_size_lists:
            next_offset = offset + len(row)
            self.row_slices.append(slice(offset, next_offset))
            offset = next_offset

        self.base_ptrs = np.asarray(
            [ptr for row in base_ptrs for ptr in row], dtype=np.uint64
        )
        self.buffer_sizes = np.asarray(
            [size for row in buffer_size_rows for size in row], dtype=np.uint64
        )
        self.tensor_size_lists = np.asarray(
            [size for row in tensor_size_lists for size in row], dtype=np.uint64
        )
        self.block_stride_lists = np.asarray(
            [stride for row in block_stride_lists for stride in row], dtype=np.uint64
        )

        all_block_ids = np.arange(self.num_blocks, dtype=np.uint64)
        self.row_addr_lookup: dict[int, np.ndarray] = {}
        for row_id, row_slice in enumerate(self.row_slices):
            stride = np.ascontiguousarray(self.block_stride_lists[row_slice])
            base = np.ascontiguousarray(self.base_ptrs[row_slice])
            self.row_addr_lookup[row_id] = np.ascontiguousarray(
                all_block_ids[:, None] * stride[None, :] + base[None, :]
            )

    def extract_block_addrs(
        self, vllm_block_ids: List[int], layer_first: bool = False
    ) -> np.ndarray:
        if layer_first:
            raise ValueError("layer_first is not supported for flattened hybrid layout")
        vllm_block_ids_np = np.asarray(vllm_block_ids, dtype=np.uint64)
        return (
            vllm_block_ids_np[:, None] * self.block_stride_lists[None, :]
            + self.base_ptrs[None, :]
        )

    def extract_block_addrs_for_row(
        self, vllm_block_ids: List[int], row_id: int
    ) -> np.ndarray:
        if row_id < 0 or row_id >= len(self.row_slices):
            raise ValueError(
                f"Invalid hybrid row_id={row_id}; row_count={len(self.row_slices)}"
            )
        lookup = self.row_addr_lookup.get(row_id)
        if lookup is not None:
            return lookup[np.asarray(vllm_block_ids, dtype=np.uint64)]
        row_slice = self.row_slices[row_id]
        vllm_block_ids_np = np.asarray(vllm_block_ids, dtype=np.uint64)
        return (
            vllm_block_ids_np[:, None] * self.block_stride_lists[row_slice][None, :]
            + self.base_ptrs[row_slice][None, :]
        )

    def _collect_shared_tensor_info(
        self,
        raw_tensor,
        kvcaches,
    ) -> tuple[list[KVCacheSpec], list[int]]:
        shared_specs: list[KVCacheSpec] = []
        shared_ptrs: list[int] = []
        layer_to_specs = layer_name_to_kv_cache_spec(self.kv_cache_config)
        for layer_name in raw_tensor.shared_by:
            kv_layer = kvcaches.get(layer_name)
            if kv_layer is None:
                continue
            shared_specs.extend(layer_to_specs[layer_name])
            if isinstance(kv_layer, torch.Tensor):
                shared_ptrs.append(kv_layer.data_ptr())
            elif isinstance(kv_layer, (tuple, list)):
                for tensor in kv_layer:
                    if isinstance(tensor, torch.Tensor):
                        shared_ptrs.append(tensor.data_ptr())
            else:
                logger.warning(f"unsupported kv_layer type: {type(kv_layer)}")
        return shared_specs, shared_ptrs

    def _append_contiguous_page_layout(
        self,
        raw_tensor,
        shared_ptrs: list[int],
        base_ptrs: list[list[int]],
        buffer_size_rows: list[list[int]],
        tensor_size_lists: list[list[int]],
        block_stride_lists: list[list[int]],
    ) -> None:
        if raw_tensor.size % self.num_blocks != 0:
            raise ValueError(
                "Invalid hybrid linear-attention raw tensor size: "
                f"raw_size={raw_tensor.size}, num_blocks={self.num_blocks}"
            )
        page_size = raw_tensor.size // self.num_blocks
        base = min(shared_ptrs)
        base_ptrs.append([base])
        buffer_size_rows.append([raw_tensor.size])
        tensor_size_lists.append([page_size])
        block_stride_lists.append([page_size])

    def _append_ascend_component_major_layout(
        self,
        raw_tensor,
        shared_ptrs: list[int],
        mamba_specs: list[MambaSpec],
        attn_specs: list[FullAttentionSpec],
        base_ptrs: list[list[int]],
        buffer_size_rows: list[list[int]],
        tensor_size_lists: list[list[int]],
        block_stride_lists: list[list[int]],
    ) -> None:
        mamba_sizes = self._mamba_component_sizes(mamba_specs[0])
        if len(mamba_sizes) < 2:
            logger.warning(
                f"unexpected mamba component sizes {mamba_sizes}; "
                "falling back to contiguous page layout"
            )
            self._append_contiguous_page_layout(
                raw_tensor,
                shared_ptrs,
                base_ptrs,
                buffer_size_rows,
                tensor_size_lists,
                block_stride_lists,
            )
            return

        conv_size = mamba_sizes[0]
        ssm_size = mamba_sizes[1]
        k_size, v_size = self._attention_component_sizes(attn_specs[0])
        middle_size = max(k_size, ssm_size)
        page_size = raw_tensor.size // self.num_blocks
        tail_size = page_size - conv_size - middle_size
        if tail_size <= 0:
            raise ValueError(
                "Invalid Ascend hybrid linear-attention page layout: "
                f"page_size={page_size}, conv_size={conv_size}, "
                f"middle_size={middle_size}, tail_size={tail_size}"
            )
        if tail_size < v_size:
            raise ValueError(
                "Ascend hybrid linear-attention tail cannot hold attention V: "
                f"tail_size={tail_size}, v_size={v_size}"
            )

        base = min(shared_ptrs)
        offsets = [
            0,
            conv_size * self.num_blocks,
            (conv_size + middle_size) * self.num_blocks,
        ]
        sizes = [conv_size, middle_size, tail_size]
        base_ptrs.append([base + offset for offset in offsets])
        buffer_size_rows.append([size * self.num_blocks for size in sizes])
        tensor_size_lists.append(sizes)
        block_stride_lists.append(sizes)

    def _append_ascend_attn_only_layout(
        self,
        raw_tensor,
        shared_ptrs: list[int],
        attn_specs: list[FullAttentionSpec],
        base_ptrs: list[list[int]],
        buffer_size_rows: list[list[int]],
        tensor_size_lists: list[list[int]],
        block_stride_lists: list[list[int]],
    ) -> None:
        """Component-major layout for attention-only tensors on Ascend.

        Ascend forces attention-only tensors (e.g. MTP draft layers in a
        hybrid model) to use the same ``[conv_padding, K, V]``
        component-major format as genuine hybrid tensors, even though
        they carry no mamba state.  This method reconstructs the
        three-slice layout from the attention component sizes and the
        page size, producing a *tensor_size_list* identical to hybrid
        rows so that both block-level and layerwise stores see a uniform
        shard schema.

        Slice equivalence with ``_append_ascend_component_major_layout``:
          conv_padding == conv_size
            (page_size - k_size - v_size == mamba_page_size_padded
             - attn_page_size == conv_block_page_size)
          k_size     == middle_size  (== max(k_size, ssm_size),
            because Ascend block_size alignment guarantees
            k_size >= ssm_size)
          v_size     == tail_size    (arithmetic identity)

        Base-pointer correction:
          Unlike hybrid rows where ``min(shared_ptrs)`` happens to be the
          raw buffer base (mamba layers contribute a ``conv_state`` view at
          offset 0), attention-only tensors have no mamba layer.  Their
          smallest shared pointer is the K-cache view's ``data_ptr()``,
          which vllm-ascend slices *after* the conv-padding region
          (``raw_k_tensor[conv_block_padding_size:]``).  Using this as
          ``base`` would shift every computed address by
          ``conv_padding_size * num_blocks`` bytes, causing the V
          component of high block ids to read beyond the raw buffer and
          crash the NPU with error 507001.  We subtract the padding
          offset so ``base`` lands on the true raw buffer start.
        """
        k_size, v_size = self._attention_component_sizes(attn_specs[0])
        page_size = raw_tensor.size // self.num_blocks
        conv_padding_size = page_size - k_size - v_size
        if conv_padding_size <= 0:
            self._append_contiguous_page_layout(
                raw_tensor,
                shared_ptrs,
                base_ptrs,
                buffer_size_rows,
                tensor_size_lists,
                block_stride_lists,
            )
            return

        # min(shared_ptrs) is the K-cache view's data_ptr, which vllm-ascend
        # slices past the conv-padding region.  Subtract the padding span
        # to recover the raw buffer base so offsets [0, conv*K, (conv+K)*K]
        # land on the correct [conv_padding, K, V] regions.
        base = min(shared_ptrs) - conv_padding_size * self.num_blocks
        sizes = [conv_padding_size, k_size, v_size]
        offsets = [
            0,
            conv_padding_size * self.num_blocks,
            (conv_padding_size + k_size) * self.num_blocks,
        ]
        base_ptrs.append([base + offset for offset in offsets])
        buffer_size_rows.append([size * self.num_blocks for size in sizes])
        tensor_size_lists.append(sizes)
        block_stride_lists.append(sizes)

    def _build_layout(self, kvcaches):
        base_ptrs = []
        buffer_size_rows = []
        tensor_size_lists = []
        block_stride_lists = []
        self.layer_name_to_row: dict[str, int] = {}

        is_npu = current_platform.device_type == "npu"

        for raw_tensor in self.kv_cache_config.kv_cache_tensors:
            if not raw_tensor.shared_by:
                continue

            shared_specs, shared_ptrs = self._collect_shared_tensor_info(
                raw_tensor, kvcaches
            )

            if not shared_ptrs:
                logger.warning(
                    f"no kv cache tensor found for shared layers {raw_tensor.shared_by}"
                )
                continue

            row_id = len(base_ptrs)
            mamba_specs = [s for s in shared_specs if isinstance(s, MambaSpec)]
            attn_specs = [s for s in shared_specs if isinstance(s, FullAttentionSpec)]

            # CUDA: all tensors use per-block contiguous page layout.
            # Ascend: tensors are component-major ([conv, K/V-or-ssm, V-or-pad]).
            #   - hybrid (mamba+attn)   → _append_ascend_component_major_layout
            #   - attn-only (e.g. MTP)  → _append_ascend_attn_only_layout
            #     (Ascend forces the same [conv_padding, K, V] format even
            #     without a mamba spec, so the tensor_size_list matches
            #     hybrid rows and layerwise shard checks pass.)
            #   - others (mamba-only, etc.) → contiguous page fallback
            if is_npu and mamba_specs and attn_specs:
                self._append_ascend_component_major_layout(
                    raw_tensor,
                    shared_ptrs,
                    mamba_specs,
                    attn_specs,
                    base_ptrs,
                    buffer_size_rows,
                    tensor_size_lists,
                    block_stride_lists,
                )
            elif is_npu and attn_specs:
                self._append_ascend_attn_only_layout(
                    raw_tensor,
                    shared_ptrs,
                    attn_specs,
                    base_ptrs,
                    buffer_size_rows,
                    tensor_size_lists,
                    block_stride_lists,
                )
            else:
                self._append_contiguous_page_layout(
                    raw_tensor,
                    shared_ptrs,
                    base_ptrs,
                    buffer_size_rows,
                    tensor_size_lists,
                    block_stride_lists,
                )

            for layer_name in raw_tensor.shared_by:
                self.layer_name_to_row[layer_name] = row_id

        self._finalize_layout_arrays(
            base_ptrs,
            buffer_size_rows,
            tensor_size_lists,
            block_stride_lists,
        )


class UCMHybridLinearAttentionConnector(UCMDirectConnector, SupportsHMA):
    """UCM connector for hybrid multi-group KV cache layouts.

    Merges the former UCMHMAConnector logic (group-aware hashing, two-stage
    lookup, per-group dispatch) with the HybridLinearAttentionLayout
    specialization for shared KV tensor pages.
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
        for raw_tensor in kv_cache_config.kv_cache_tensors:
            shared_specs = [
                spec
                for layer_name in raw_tensor.shared_by
                for spec in layer_to_specs.get(layer_name, [])
            ]
            if any(
                isinstance(spec, FullAttentionSpec) for spec in shared_specs
            ) and any(
                isinstance(spec, MambaSpec) and spec.mamba_cache_mode == "align"
                for spec in shared_specs
            ):
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
            # Override the inherited ``block_size`` (which comes from
            # ``cache_config.block_size``) so prefix accounting in this class
            # is consistent with every group's block boundaries — vLLM's
            # hybrid scheduler aligns ``num_computed_tokens`` to the LCM of
            # all groups' block_size, and so do we.
            self.block_size = lcm_block_size
            self.hash_block_size = lcm_block_size

        logger.info(
            f"UCMHybridLinearAttentionConnector initialized with use_layerwise={self.use_layerwise}"
        )

    def _create_kv_cache_layout(
        self, kv_caches: dict[str, torch.Tensor]
    ) -> KVCacheLayout:
        return HybridLinearAttentionLayout(
            kv_caches,
            self.launch_config,
            self._vllm_config,
            self._kv_cache_config,
        )

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        self.kv_caches = kv_caches
        self.kv_cache_layout = self._create_kv_cache_layout(self.kv_caches)
        self.store = self._create_store(self.kv_cache_layout)
        self.block_data_size = self.kv_cache_layout.block_size
        self.device = create_device()

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
        # ``hbm_hit_block_num`` and ``total_hit_block_num`` are tracked in
        # LCM-block units in HLA mode; per-group block ids/counts are derived
        # from these via each group's own block_size when needed.
        hbm_hit_block_num = num_computed_tokens // lcm_block_size

        # Skip persistence if token count is below the threshold.
        if self.persist_token_threshold > request.num_tokens:
            logger.info_once(
                f"Skip persistence: req {request.request_id}, "
                f"input tokens ({request.num_tokens}) < threshold "
                f"({self.persist_token_threshold})."
            )
            return 0, False

        # Hash once per group so dump path can later reuse the same block ids.
        group_ucm_block_ids = self.group_manager.compute_all_group_block_ids(
            request.all_token_ids
        )
        # Legacy ``ucm_block_ids`` mirrors the first full-attn group (by
        # group_id order) for callers that still consume the flat list.
        primary_full_attn = self.group_manager.full_attn_groups[0]
        primary_block_ids = group_ucm_block_ids[primary_full_attn.group_id]

        external_hit_tokens, external_hit_lcm_blocks, mamba_prefetch_hashes = (
            self.group_manager.lookup_external_hit_tokens(
                num_computed_tokens, self.store, group_ucm_block_ids
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

        # GC heat update: a request with any cache hit is a hot request —
        # update heat for ALL hit blocks (including HBM-hit prefix) to
        # prevent GC eviction on any rank.
        total_hit_tokens = total_hit_block_num * lcm_block_size
        hbm_hit_full_attn = num_computed_tokens // primary_full_attn.block_size
        total_hit_full_attn = total_hit_tokens // primary_full_attn.block_size
        all_hit_full_attn = primary_block_ids[0:total_hit_full_attn]
        # Rank 0: HBM full-attn blocks were not checked by lookup_on_prefix;
        # mamba positions below best_pos were not checked by the sequential
        # scan.  Prefetch updates heat for existing entries (no-op if absent).
        hbm_full_attn = primary_block_ids[0:hbm_hit_full_attn]
        if hbm_full_attn:
            self.store.prefetch(hbm_full_attn)
        if mamba_prefetch_hashes:
            self.store.prefetch(mamba_prefetch_hashes)
        # Other ranks: all hit full-attn blocks + mamba hashes, re-hashed.
        self._prefetch_other_rank_hashes(all_hit_full_attn + mamba_prefetch_hashes)

        logger.info_once(
            f"request_id: {request.request_id}, "
            f"total_lcm_blocks: {request.num_tokens // lcm_block_size}, "
            f"hit hbm: {hbm_hit_block_num}, "
            f"hit external: {external_hit_lcm_blocks}, "
            f"total_tokens: {len(request.all_token_ids)}"
        )
        if len(primary_block_ids) > 0:
            ucmmetrics.update_stats(
                {
                    "interval_lookup_hit_rates": external_hit_lcm_blocks
                    * lcm_block_size
                    / (len(primary_block_ids) * primary_full_attn.block_size)
                },
            )

        # When all the tokens are cached in ssd or hbm, we need to recompute
        # the last token. This branch will be removed once vLLM scheduler
        # provides a better solution in the future.
        num_total_hit_tokens = total_hit_block_num * lcm_block_size
        if num_total_hit_tokens == request.num_tokens and external_hit_tokens > 0:
            external_hit_tokens -= 1

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

    def _generate_hla_dispatch_meta(
        self,
        req_meta: "HLARequestMeta",
        new_tokens: int,
        new_vllm_block_ids_per_group: tuple[list[int], ...],
        need_load: bool = True,
        request_id: str = "",
        incoming_block_ids_are_full: bool = False,
    ) -> RequestDispatchMeta:
        """Build a flat (ucm, vllm) block id pair list across all groups.

        The output ``RequestDispatchMeta`` keeps the same shape as the
        non-HLA path (``tuple[list[bytes], list[int]]``) so that
        ``start_load_kv`` / ``wait_for_save`` and the underlying store APIs
        do not need to know about groups. Per-group slices are concatenated
        in ascending ``group_id`` order, with ``ucm_block_ids[k]`` and
        ``vllm_block_ids[k]`` always referring to the same block.

        Layout per group within ``[token_processed, token_processed + new_tokens)``:
        - **load** (only when ``external_hit_blocks > 0`` and ``need_load``):
          - full-attn group: load blocks covering tokens
            ``[hbm_hit_tokens, total_hit_tokens)`` via ``extend_non_null``.
          - mamba-align state group: load the single state block at
            ``total_hit_tokens`` via ``append_mamba_align_state_block``.
            The state represents the entire prefix and is reloaded each
            resume because older blocks are evicted by the scheduler.
        - **dump** of ``[token_processed, token_processed + new_tokens)``:
          - full-attn group: every newly-completed full block (the
            ``lookup_on_prefix`` chain needs every prefix block to be
            present).
          - mamba-align state group: the state block at each LCM boundary
            reached in this range, via ``append_mamba_align_state_block``.
            Lookup always resumes at LCM boundaries and stage-2 only
            inspects state hashes at those points, so blocks between
            boundaries would be dead weight in the store.
        """
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
                # update_state_after_alloc() usually gives us the full block
                # table before build_connector_meta(). If that happened, the
                # scheduler's "new" block ids are already the suffix of the
                # full table and must not be appended again. If the connector is
                # used with an older scheduler path that did not call
                # update_state_after_alloc(), append as a fallback.
                suffix_len = len(incoming_vllm_block_ids)
                if existing_vllm_block_ids[-suffix_len:] != incoming_vllm_block_ids:
                    existing_vllm_block_ids.extend(incoming_vllm_block_ids)

        load_ucm_block_ids: list[bytes] = []
        load_vllm_block_ids: list[int] = []
        dump_ucm_block_ids: list[bytes] = []
        dump_vllm_block_ids: list[int] = []

        def append_mamba_align_state_block(
            dst_ucm_block_ids: list[bytes],
            dst_vllm_block_ids: list[int],
            gid: int,
            seq_len: int,
            reason: str,
        ) -> None:
            group = groups_by_id[gid]
            state_idx = max((seq_len - 1) // group.block_size, 0)
            vllm_state_idx = state_idx
            if reason == "load":
                # For resumed mamba-align requests, vLLM keeps the cached
                # prefix state at ``state_idx`` and allocates a fresh running
                # state block at the tail of the block table. UCM must read
                # the prefix hash but write into that current running block.
                block_ids = req_meta.group_vllm_block_ids[gid]
                for i in range(len(block_ids) - 1, -1, -1):
                    if block_ids[i] != 0:
                        vllm_state_idx = i
                        break

            try:
                vllm_block_id = req_meta.group_vllm_block_ids[gid][vllm_state_idx]
            except IndexError:
                logger.error(
                    "HLA mamba-align state vLLM block missing: "
                    f"request_id={request_id}, group_id={gid}, reason={reason}, "
                    f"seq_len={seq_len}, state_idx={state_idx}, "
                    f"vllm_state_idx={vllm_state_idx}, "
                    f"num_vllm_blocks={len(req_meta.group_vllm_block_ids[gid])}"
                )
                return
            if vllm_block_id == 0:
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
            dst_ucm_block_ids.append(ucm_block_id)
            dst_vllm_block_ids.append(vllm_block_id)

        external_hit_lcm_blocks = (
            req_meta.total_hit_block_num - req_meta.hbm_hit_block_num
        )
        hbm_hit_tokens = req_meta.hbm_hit_block_num * lcm_block_size
        total_hit_tokens = req_meta.total_hit_block_num * lcm_block_size

        if need_load and external_hit_lcm_blocks > 0:
            for gid, group in enumerate(groups_by_id):
                if group.is_mamba_align:
                    append_mamba_align_state_block(
                        load_ucm_block_ids,
                        load_vllm_block_ids,
                        gid,
                        total_hit_tokens,
                        "load",
                    )
                    continue
                # Full-attention group: load the external hit prefix.
                load_tok_start = hbm_hit_tokens
                load_tok_end = total_hit_tokens
                start_blk = load_tok_start // group.block_size
                end_blk = load_tok_end // group.block_size
                if start_blk >= end_blk:
                    continue
                extend_non_null(
                    load_ucm_block_ids,
                    load_vllm_block_ids,
                    req_meta.group_ucm_block_ids[gid][start_blk:end_blk],
                    req_meta.group_vllm_block_ids[gid][start_blk:end_blk],
                )

        if req_meta.token_processed < req_meta.num_token_ids:
            dump_tok_start = req_meta.token_processed
            dump_tok_end = min(
                req_meta.token_processed + new_tokens, req_meta.num_token_ids
            )
            # LCM boundaries B with ``dump_tok_start < B <= dump_tok_end``.
            # State groups only need the tail at these boundaries because lookup
            # always resumes at LCM boundaries (see
            # ``lookup_external_hit_tokens`` stage 2).
            first_lcm_b = (dump_tok_start // lcm_block_size + 1) * lcm_block_size
            last_lcm_b = (dump_tok_end // lcm_block_size) * lcm_block_size

            for gid, group in enumerate(groups_by_id):
                if group.is_full_attention:
                    # Dump every newly completed block: ``lookup_on_prefix``
                    # walks the full prefix chain so any gap would truncate
                    # future hits.
                    start_blk = dump_tok_start // group.block_size
                    end_blk = dump_tok_end // group.block_size
                    if start_blk >= end_blk:
                        continue
                    extend_non_null(
                        dump_ucm_block_ids,
                        dump_vllm_block_ids,
                        req_meta.group_ucm_block_ids[gid][start_blk:end_blk],
                        req_meta.group_vllm_block_ids[gid][start_blk:end_blk],
                    )
                else:
                    # Mamba-align state: dump only the state block at each LCM
                    # boundary reached in this range. Consecutive boundaries'
                    # tails do not overlap and we can extend the lists without
                    # dedup.
                    if first_lcm_b > last_lcm_b:
                        continue
                    b = first_lcm_b
                    while b <= last_lcm_b:
                        append_mamba_align_state_block(
                            dump_ucm_block_ids,
                            dump_vllm_block_ids,
                            gid,
                            b,
                            "dump",
                        )
                        b += lcm_block_size
            req_meta.token_processed += new_tokens

        return RequestDispatchMeta(
            (load_ucm_block_ids, load_vllm_block_ids),
            (dump_ucm_block_ids, dump_vllm_block_ids),
        )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        assert self.group_manager is not None
        num_groups = self.group_manager.num_groups
        empty_per_group: tuple[list[int], ...] = tuple([] for _ in range(num_groups))

        requests_dispatch_meta: dict[str, RequestDispatchMeta] = {}

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

    def wait_for_save(self) -> None:
        if self.is_mla and self.tp_rank % self.tp_size != 0:
            return

        metadata = self._get_connector_metadata()
        assert isinstance(metadata, UCMConnectorMetadata)

        total_ucm_block_ids: list[bytes] = []
        total_vllm_block_ids: list[int] = []
        num_saved_block = 0
        num_saved_request = 0
        for request_id, request in metadata.request_meta.items():
            if len(request.dump_block_ids[0]) == 0:
                continue

            ucm_block_ids, vllm_block_ids = request.dump_block_ids
            if self._skip_null_vllm_blocks:
                ucm_block_ids, vllm_block_ids = _drop_null_vllm_blocks(
                    ucm_block_ids,
                    vllm_block_ids,
                    f"UCM dump request {request_id}",
                )
                if len(ucm_block_ids) == 0:
                    continue
            num_saved_block += len(ucm_block_ids)
            num_saved_request += 1
            ucm_block_ids = self._rank_scoped_ucm_block_ids(ucm_block_ids)
            total_ucm_block_ids.extend(ucm_block_ids)
            total_vllm_block_ids.extend(vllm_block_ids)

        if not total_ucm_block_ids:
            return

        event_handle = 0
        try:
            total_ptrs = self.kv_cache_layout.extract_block_addrs(total_vllm_block_ids)
            total_ptrs = total_ptrs.reshape(total_ptrs.shape[0], -1)
            shard_indexs = [0] * len(total_ucm_block_ids)
            event_handle = self._get_dump_event_handle()
            save_start_time = time.perf_counter() * 1000
            task = self.store.dump_data(
                total_ucm_block_ids, shard_indexs, total_ptrs, event_handle
            )
        except Exception as e:
            logger.error(f"dump kv cache failed. {type(e).__name__}: {e}")
            if self.enable_event_sync and event_handle and self.device is not None:
                self.device.destroy_event_handle(event_handle)
            return

        try:
            self.store.wait(task)
            save_end_time = time.perf_counter() * 1000
        except Exception as e:
            logger.error(f"wait for dump kv cache failed. {type(e).__name__}: {e}")
            return
        finally:
            if self.enable_event_sync and event_handle and self.device is not None:
                self.device.destroy_event_handle(event_handle)

        save_bytes = num_saved_block * self.block_data_size
        save_speed = save_bytes / max(save_end_time - save_start_time, 1) / 1024 / 1024
        ucmmetrics.update_stats(
            {
                "save_requests_num": num_saved_request,
                "save_blocks_num": num_saved_block,
                "save_duration": save_end_time - save_start_time,
                "save_speed": save_speed,
                "save_bytes_total": save_bytes,
            }
        )

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, object] | None]:
        return False, None

    def _rank_scoped_ucm_block_ids(self, ucm_block_ids: list[bytes]) -> list[bytes]:
        if self.tp_rank % self.tp_size == 0 or self.is_mla:
            return ucm_block_ids
        return [self.request_hasher(b) for b in ucm_block_ids]


class UCMHybridLinearAttentionLayerWiseConnector(UCMHybridLinearAttentionConnector):
    """Layerwise connector for full-attention + linear-attention hybrid layouts."""

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
        self.load_tasks: dict[int, dict[str, Task]] = defaultdict(dict)
        self.dump_tasks: dict[int, list[PendingDumpTask]] = defaultdict(list)
        self.request_data: list[tuple[str, list[bytes], list[int]]] = []
        self._failure_req_ids: set[str] = set()
        self._submitted_load_rows: set[int] = set()
        self._dump_transfer_data: tuple[list[bytes], list[int], set[str]] | None = None
        prefetch_rows_config = self.launch_config.get(
            "hybrid_layerwise_prefetch_rows", 2
        )
        try:
            self._load_prefetch_rows = max(1, int(prefetch_rows_config))
        except (TypeError, ValueError):
            logger.warning(
                "Invalid hybrid_layerwise_prefetch_rows=%r; fallback to 2.",
                prefetch_rows_config,
            )
            self._load_prefetch_rows = 2
        self.is_save = False
        self.need_load = False
        self._layerwise_batch_start: Optional[float] = None
        self._layerwise_prev_wait_end: Optional[float] = None
        # MTP draft layers can be revisited several times in one speculative
        # decode batch (1 draft prefill + num_speculative_steps-1 multi-step
        # decode calls).  Each visit triggers save_kv_layer for the MTP row;
        # deferring the dump and keeping only the last snapshot avoids
        # duplicate I/O and ensures the most complete KV cache is persisted.
        self._deferred_mtp_row_dumps: dict[
            int, tuple[list[bytes], list[int], set[str]]
        ] = {}
        self._is_mtp = False
        self._init_mtp_layerwise_dump_state()
        logger.info(
            "Init UCMHybridLinearAttentionLayerWiseConnector "
            f"with prefetch_rows={self._load_prefetch_rows}."
        )

    def _init_mtp_layerwise_dump_state(self) -> None:
        """Detect whether MTP is enabled."""
        speculative_config = getattr(self._vllm_config, "speculative_config", None)
        if speculative_config is None:
            return

        mtp_method = getattr(speculative_config, "method", None)
        self._is_mtp = mtp_method == "mtp" or (
            isinstance(mtp_method, str) and mtp_method.endswith("_mtp")
        )

    def _is_mtp_layer(self, layer_name: str) -> bool:
        """Check whether a layer belongs to the MTP draft model.

        Uses the layer name (contains 'mtp') instead of layer_id, because
        extract_layer_index('mtp.layers.0...') returns 0 (the index within
        the MTP model), not the global layer index (e.g. 64).
        """
        return self._is_mtp and "mtp" in layer_name

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        if has_ucm_sparse() and os.getenv("VLLM_HASH_ATTENTION") == "1":
            for layer_name, value in kv_caches.items():
                kv_cache, _ = value
                self.kv_caches[layer_name] = kv_cache
        else:
            self.kv_caches = kv_caches

        self.kv_cache_layout = self._create_kv_cache_layout(self.kv_caches)
        self.block_data_size = int(self.kv_cache_layout.tensor_size_lists.sum())
        self.layer_name_to_id = self.kv_cache_layout.layer_name_to_id
        self.layer_ids = sorted(set(self.layer_name_to_id.values()))
        self.first_layer_id = self.layer_ids[0]
        self.layer_name_to_row = getattr(self.kv_cache_layout, "layer_name_to_row", {})
        self.row_ids = sorted(set(self.layer_name_to_row.values()))
        row_tensor_size_lists = getattr(
            self.kv_cache_layout, "row_tensor_size_lists", []
        )
        if not self.row_ids:
            raise RuntimeError("Hybrid layerwise layout has no cache rows.")
        if max(self.row_ids) >= len(row_tensor_size_lists):
            raise RuntimeError(
                "Hybrid layerwise row mapping is inconsistent with layout rows: "
                f"row_ids={_short_list(self.row_ids)}, "
                f"row_tensor_size_lists={len(row_tensor_size_lists)}"
            )

        first_row_id = self.row_ids[0]
        row_tensor_size_list = list(row_tensor_size_lists[first_row_id])
        row_shard_size = sum(row_tensor_size_list)
        for row_id in self.row_ids:
            tensor_size_list = list(row_tensor_size_lists[row_id])
            if tensor_size_list != row_tensor_size_list:
                raise RuntimeError(
                    "Hybrid layerwise rows must share the same tensor layout for "
                    "one row-sharded store: "
                    f"row_id={row_id}, tensor_size_list={tensor_size_list}, "
                    f"expected={row_tensor_size_list}"
                )

        self.device = create_device()

        self.store = self._create_store(
            self.kv_cache_layout,
            tensor_size_list_override=row_tensor_size_list,
            shard_size_override=row_shard_size,
            block_size_override=row_shard_size * (max(self.row_ids) + 1),
            compact_cache_buffer_capacity=True,
        )

        row_to_layers: dict[int, list[str]] = defaultdict(list)
        for layer_name, row_id in self.layer_name_to_row.items():
            row_to_layers[row_id].append(layer_name)
        self.row_save_layer = {
            row_id: max(
                layer_names,
                key=lambda name: self.layer_name_to_id.get(name, self.first_layer_id),
            )
            for row_id, layer_names in row_to_layers.items()
        }
        logger.info(
            "Hybrid layerwise layout: "
            f"rows={len(self.row_ids)}, row_ids={_short_list(self.row_ids)}, "
            f"row_shard_size={row_shard_size}, "
            f"row_tensor_size_list={row_tensor_size_list}, "
            f"row_save_layers={len(self.row_save_layer)}"
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

    def _submit_request_load_tasks_for_row(
        self,
        row_id: int,
        metadata: "UCMConnectorMetadata",
    ) -> None:
        for request_id, ucm_block_ids, vllm_block_ids in self.request_data:
            if request_id in self._failure_req_ids:
                continue
            try:
                row_ptrs = self.kv_cache_layout.extract_block_addrs_for_row(
                    vllm_block_ids, row_id
                )
                shard_indexs = [row_id] * len(ucm_block_ids)
                task = self.store.load_data(ucm_block_ids, shard_indexs, row_ptrs)
                self.load_tasks[row_id][request_id] = task
            except Exception as e:
                logger.error(
                    f"request {request_id} submit load task for row {row_id} "
                    f"error. {type(e).__name__}: {e}"
                )
                self._mark_load_failed(metadata, request_id)
        self._submitted_load_rows.add(row_id)

    def _submit_request_load_tasks_for_row_once(
        self,
        row_id: int,
        metadata: "UCMConnectorMetadata",
    ) -> None:
        if row_id in self._submitted_load_rows:
            return
        self._submit_request_load_tasks_for_row(row_id, metadata)

    def _wait_row_load(self, row_id: int, metadata: "UCMConnectorMetadata") -> int:
        """Pop and wait for a row's per-request load tasks, marking failures."""
        row_tasks = self.load_tasks.pop(row_id, {})
        for request_id, task in row_tasks.items():
            try:
                self.store.wait(task)
            except Exception as e:
                logger.error(
                    f"request {request_id} wait row {row_id} "
                    f"load failed. {type(e).__name__}: {e}"
                )
                self._mark_load_failed(metadata, request_id)
        return len(row_tasks)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        self._layerwise_batch_start = time.perf_counter()
        self._layerwise_prev_wait_end = None
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, UCMConnectorMetadata)
        self.load_tasks.clear()
        self.request_data.clear()
        self._failure_req_ids.clear()
        self._submitted_load_rows.clear()
        self._dump_transfer_data = None
        self._deferred_mtp_row_dumps.clear()
        self.need_load = False

        for request_id, request in metadata.request_meta.items():
            if len(request.load_block_ids[0]) == 0:
                continue

            ucm_block_ids, vllm_block_ids = request.load_block_ids
            if self._skip_null_vllm_blocks:
                ucm_block_ids, vllm_block_ids = _drop_null_vllm_blocks(
                    ucm_block_ids,
                    vllm_block_ids,
                    f"UCM hybrid layerwise load request {request_id}",
                )
                if len(ucm_block_ids) == 0:
                    continue
            self.need_load = True
            ucm_block_ids = self._rank_scoped_ucm_block_ids(ucm_block_ids)
            self.request_data.append((request_id, ucm_block_ids, vllm_block_ids))

        if self.need_load and self.row_ids:
            # Submit row 0 + prefetch rows, then synchronously wait for
            # row 0.  vLLM only calls wait_for_layer_load at the
            # full_attn layer (last layer of each row), so row 0 must
            # be loaded here before the first linear_attn layer begins.
            num_submit = min(self._load_prefetch_rows + 1, len(self.row_ids))
            for idx in range(num_submit):
                self._submit_request_load_tasks_for_row_once(idx, metadata)
            self._wait_row_load(0, metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        if not self._connector_metadata or not self.need_load:
            return
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, UCMConnectorMetadata)
        row_id = self.layer_name_to_row.get(layer_name)
        if row_id is None:
            return

        # wait_for_layer_load is only called at the full_attn layer
        # (last layer of the current row).  Wait for the NEXT row so
        # its first layers (linear_attn) have KV cache loaded before
        # their forward begins.
        next_row_id = row_id + 1
        if next_row_id >= len(self.row_ids):
            return

        self._submit_request_load_tasks_for_row_once(next_row_id, metadata)

        wait_start = time.perf_counter()
        n_tasks = self._wait_row_load(next_row_id, metadata)
        wait_end = time.perf_counter()

        # Prefetch rows ahead of the one we just waited for.
        prefetch_start = next_row_id + 1
        prefetch_end = min(prefetch_start + self._load_prefetch_rows, len(self.row_ids))
        for idx in range(prefetch_start, prefetch_end):
            self._submit_request_load_tasks_for_row_once(idx, metadata)

        blocking_ms = (wait_end - wait_start) * 1000
        stats: dict[str, float] = {
            "layerwise_wait_blocking_ms": blocking_ms,
            "layerwise_wait_tasks_count": float(n_tasks),
        }
        if self._layerwise_prev_wait_end is not None:
            stats["layerwise_inter_wait_interval_ms"] = (
                wait_start - self._layerwise_prev_wait_end
            ) * 1000
        if prefetch_start < prefetch_end:
            submit_end = time.perf_counter()
            stats["layerwise_next_layer_submit_ms"] = (submit_end - wait_end) * 1000
        ucmmetrics.update_stats(stats)
        self._layerwise_prev_wait_end = wait_end

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        if not self._connector_metadata:
            return
        if self.is_mla and self.tp_rank % self.tp_size != 0:
            return

        row_id = self.layer_name_to_row.get(layer_name)
        if row_id is None:
            return
        if self.row_save_layer.get(row_id) != layer_name:
            return

        metadata = self._get_connector_metadata()
        assert isinstance(metadata, UCMConnectorMetadata)
        if self._dump_transfer_data is None:
            self._dump_transfer_data = self._build_dump_transfer_data(metadata, row_id)
        total_ucm_block_ids, total_vllm_block_ids, dump_request_ids = (
            self._dump_transfer_data
        )

        if not total_ucm_block_ids:
            return

        self.is_save = True

        if self._is_mtp_layer(layer_name):
            # Defer: MTP layers are revisited N times per speculative decode
            # step (1 draft prefill + num_speculative_steps-1 multi-step
            # decode).  Each visit overwrites this entry so only the last
            # (most complete) snapshot is flushed in wait_for_save.
            self._deferred_mtp_row_dumps[row_id] = (
                list(total_ucm_block_ids),
                list(total_vllm_block_ids),
                set(dump_request_ids),
            )
            return

        row_ptrs = self.kv_cache_layout.extract_block_addrs_for_row(
            total_vllm_block_ids, row_id
        )
        shard_indexs = [row_id] * len(total_ucm_block_ids)
        try:
            row_ptrs = np.ascontiguousarray(row_ptrs)
            event_handle = self._get_dump_event_handle()
            task = self.store.dump_data(
                total_ucm_block_ids, shard_indexs, row_ptrs, event_handle
            )
            self.dump_tasks[row_id].append(
                PendingDumpTask(
                    task=task,
                    request_ids=set(dump_request_ids),
                    event_handle=event_handle,
                )
            )
        except Exception as e:
            logger.error(
                f"submit hybrid layerwise row {row_id} dump task failed. "
                f"{type(e).__name__}: {e}"
            )

    def _build_dump_transfer_data(
        self,
        metadata: "UCMConnectorMetadata",
        row_id: int,
    ) -> tuple[list[bytes], list[int], set[str]]:
        total_ucm_block_ids: list[bytes] = []
        total_vllm_block_ids: list[int] = []
        dump_request_ids: set[str] = set()
        for request_id, request in metadata.request_meta.items():
            if len(request.dump_block_ids[0]) == 0:
                continue

            dump_request_ids.add(request_id)
            ucm_block_ids, vllm_block_ids = request.dump_block_ids
            if self._skip_null_vllm_blocks:
                ucm_block_ids, vllm_block_ids = _drop_null_vllm_blocks(
                    ucm_block_ids,
                    vllm_block_ids,
                    f"UCM hybrid layerwise dump row {row_id}",
                )
                if len(ucm_block_ids) == 0:
                    continue
            ucm_block_ids = self._rank_scoped_ucm_block_ids(ucm_block_ids)
            total_ucm_block_ids.extend(ucm_block_ids)
            total_vllm_block_ids.extend(vllm_block_ids)
        return total_ucm_block_ids, total_vllm_block_ids, dump_request_ids

    def _flush_deferred_mtp_dumps(self) -> None:
        """Submit the last saved snapshot for each deferred MTP row."""
        if not self._deferred_mtp_row_dumps:
            return

        deferred = self._deferred_mtp_row_dumps
        self._deferred_mtp_row_dumps = {}
        for row_id, (ucm_ids, vllm_ids, req_ids) in deferred.items():
            try:
                row_ptrs = self.kv_cache_layout.extract_block_addrs_for_row(
                    vllm_ids, row_id
                )
                row_ptrs = np.ascontiguousarray(row_ptrs)
                shard_indexs = [row_id] * len(ucm_ids)
                event_handle = self._get_dump_event_handle()
                task = self.store.dump_data(
                    ucm_ids, shard_indexs, row_ptrs, event_handle
                )
                self.dump_tasks[row_id].append(
                    PendingDumpTask(
                        task=task,
                        request_ids=set(req_ids),
                        event_handle=event_handle,
                    )
                )
            except Exception as e:
                logger.error(
                    f"submit deferred MTP row {row_id} dump task failed. "
                    f"{type(e).__name__}: {e}"
                )

    def wait_for_save(self) -> None:
        if not self.is_save:
            total_end = time.perf_counter()
            if self._layerwise_batch_start is not None:
                batch_total_ms = (total_end - self._layerwise_batch_start) * 1000
                ucmmetrics.update_stats({"layerwise_batch_total_ms": batch_total_ms})
                self._layerwise_batch_start = None
            return

        self._flush_deferred_mtp_dumps()

        total_start = time.perf_counter()
        for row_id in self.row_ids:
            for pending_dump_task in self.dump_tasks.pop(row_id, []):
                try:
                    self.store.wait(pending_dump_task.task)
                except Exception as e:
                    logger.error(
                        f"wait for dump kv cache failed. " f"{type(e).__name__}: {e}"
                    )
        total_end = time.perf_counter()
        stats: dict[str, float] = {
            "layerwise_save_tail_total_ms": (total_end - total_start) * 1000,
        }
        if self._layerwise_batch_start is not None:
            stats["layerwise_batch_total_ms"] = (
                total_end - self._layerwise_batch_start
            ) * 1000
            self._layerwise_batch_start = None
        ucmmetrics.update_stats(stats)

        self.dump_tasks.clear()
        self._dump_transfer_data = None
        self.is_save = False
        if self.enable_event_sync:
            self.device.destroy_event_handles()
