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
from ucm.integration.vllm.ucm_connector import (
    KVCacheLayout,
    PendingDumpTask,
    RequestDispatchMeta,
    RequestHasher,
    RequestMeta,
    UCMConnectorMetadata,
    UCMDirectConnector,
    _record_counter,
    _scheduler_read_block_size,
    _short_list,
    _use_ucm_connector_cpu_affinity,
)
from ucm.logger import init_logger
from ucm.shared.metrics import ucmmetrics
from ucm.sparse.state import has_ucm_sparse
from ucm.store.factory_v1 import UcmConnectorFactoryV1
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
    """RequestMeta extended with per-group block tracking for hybrid models."""

    group_ucm_block_ids: list[list[bytes]] = field(default_factory=list)
    group_vllm_block_ids: list[list[int]] = field(default_factory=list)


@dataclass
class HLARequestDispatchMeta(RequestDispatchMeta):
    """Extends RequestDispatchMeta with full-attn block count for MLA rank scoping."""

    load_full_attn_count: int = 0
    dump_full_attn_count: int = 0


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
    # Skip vLLM null blocks (block_id=0) used as mamba-align placeholders.
    for ucm_block_id, vllm_block_id in zip(src_ucm_block_ids, src_vllm_block_ids):
        if vllm_block_id == 0:
            continue
        dst_ucm_block_ids.append(ucm_block_id)
        dst_vllm_block_ids.append(vllm_block_id)


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

    @property
    def is_full_attention(self) -> bool:
        return not self.is_mamba_align


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

        # Resume points must align to the LCM of all group block_sizes.
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
            # mamba-align pads block table with null blocks; no per-block hash.
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
        """Compute full block hashes for every group, indexed by group_id."""
        return [self.compute_block_hashes(g, token_ids) for g in self.groups_by_id]

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
        lookup_all: Callable[[list[bytes]], list[bool]],
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

        # Stage 2: backward scan for mamba state at LCM boundaries.
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
                results = lookup_all(pos_hashes)
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

    def _attention_component_sizes(self, spec: KVCacheSpec) -> tuple[int, int]:
        assert isinstance(spec, FullAttentionSpec)
        if isinstance(spec, MLAAttentionSpec):
            # MLA: head_size = kv_lora_rank + qk_rope_head_dim
            hf = self.vllm_config.model_config.hf_text_config
            k_dim = getattr(hf, "kv_lora_rank", spec.head_size)
            v_dim = getattr(hf, "qk_rope_head_dim", spec.head_size)
        else:
            k_dim = spec.head_size
            v_dim = getattr(spec, "head_size_v", spec.head_size)
        k_size = (
            spec.block_size * spec.num_kv_heads * k_dim * self._dtype_size(spec.dtype)
        )
        v_size = (
            spec.block_size * spec.num_kv_heads * v_dim * self._dtype_size(spec.dtype)
        )
        return k_size, v_size

    def _finalize_layout_arrays(
        self,
        base_ptrs: list[list[int]],
        buffer_size_rows: list[list[int]],
        tensor_size_lists: list[list[int]],
        block_stride_lists: list[list[int]],
    ) -> None:
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
        """Component-major [conv_padding, K, V] layout for Ascend attn-only tensors."""
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

        # K-cache view starts past conv_padding; subtract to get raw base.
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

            # Ascend: hybrid → component_major, attn-only → attn_only, else contiguous.
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
            self.block_size = lcm_block_size
            self.hash_block_size = lcm_block_size

        logger.info(f"{type(self).__name__} initialized")

    def get_block_size(self) -> int:
        if self.group_manager is not None:
            return self.group_manager.lcm_block_size
        return self.block_size

    def _create_kv_cache_layout(
        self, kv_caches: dict[str, torch.Tensor]
    ) -> KVCacheLayout:
        return HybridLinearAttentionLayout(
            kv_caches,
            self.launch_config,
            self._vllm_config,
            self._kv_cache_config,
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
        self._set_default_shm_buffer_capacity(config)
        if "storage_backends" in config:
            backends = [path for path in config["storage_backends"].split(":")]
            config["storage_backends"] = backends
        config["unique_id"] = f"{self.unique_id}{unique_id_suffix}"
        if self._role == KVConnectorRole.WORKER:
            config["device_id"] = self.local_rank
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

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        self.kv_caches = kv_caches
        self.kv_cache_layout = self._create_kv_cache_layout(self.kv_caches)
        self.block_data_size = self.kv_cache_layout.block_size
        self.device = create_device()

        enable_affinity = _use_ucm_connector_cpu_affinity()
        worker_cores, store_cores = (
            self.device.split_cores(self.local_rank)
            if enable_affinity
            else (None, None)
        )

        self.store = self._create_store(
            kv_cache_layout=self.kv_cache_layout,
            cpu_affinity_cores=store_cores,
        )

        if worker_cores:
            try:
                os.sched_setaffinity(0, worker_cores)
                logger.info(f"[VLLM CPU Affinity] Worker bound to cores {worker_cores}")
            except Exception as e:
                logger.warning(f"Failed to bind worker: {e}")

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

        group_ucm_block_ids = self.group_manager.compute_all_group_block_ids(
            request.all_token_ids
        )
        primary_full_attn = self.group_manager.full_attn_groups[0]
        primary_block_ids = group_ucm_block_ids[primary_full_attn.group_id]

        # Pre-lookup reduction: leave at least recompute_tokens for vLLM to
        # recompute, so the batch isn't dispatched as uniform decode into FULL
        # cudagraph. For hybrid this also avoids looking up the last block(s)
        # whose mamba state may not be valid in HBM at dump time.
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
                lambda block_ids: self._rank_consistency.lookup_all(
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
        if hbm_full_attn:
            self.store.prefetch(hbm_full_attn)
        if mamba_prefetch_hashes:
            self.store.prefetch(mamba_prefetch_hashes)
        # MLA full-attn is TP-replicated (shared hash), no per-rank entries to prefetch.
        # Only mamba blocks have per-rank entries needing heat update.
        per_rank_hashes = mamba_prefetch_hashes
        if not self.is_mla:
            per_rank_hashes = all_hit_full_attn + mamba_prefetch_hashes
        self._prefetch_other_rank_hashes(per_rank_hashes)

        if len(primary_block_ids) > 0:
            ucmmetrics.update_stats(
                {
                    "interval_lookup_hit_rates": external_hit_lcm_blocks
                    * lcm_block_size
                    / (len(primary_block_ids) * primary_full_attn.block_size)
                },
            )

        # No post-lookup workaround: pre-lookup truncation already ensures
        # total_hit_tokens == external_hit_tokens, and the mamba state at
        # total_hit_tokens is a position the store actually verified.
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
        req_meta: "HLARequestMeta",
        request_id: str,
        gid: int,
        seq_len: int,
        reason: str,
    ) -> None:
        group = self.group_manager.groups_by_id[gid]
        state_idx = max((seq_len - 1) // group.block_size, 0)
        vllm_state_idx = state_idx
        if reason == "load":
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
        dump_ucm_block_ids: list[bytes] = []
        dump_vllm_block_ids: list[int] = []

        external_hit_lcm_blocks = (
            req_meta.total_hit_block_num - req_meta.hbm_hit_block_num
        )
        hbm_hit_tokens = req_meta.hbm_hit_block_num * lcm_block_size
        total_hit_tokens = req_meta.total_hit_block_num * lcm_block_size

        if need_load and external_hit_lcm_blocks > 0:
            # Pass 1: full-attention blocks first (for MLA rank-0-only dump)
            for gid, group in enumerate(groups_by_id):
                if group.is_mamba_align:
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
                    req_meta.group_ucm_block_ids[gid][start_blk:end_blk],
                    req_meta.group_vllm_block_ids[gid][start_blk:end_blk],
                )
            load_full_attn_count = len(load_ucm_block_ids) if self.is_mla else 0
            # Pass 2: mamba state blocks
            for gid, group in enumerate(groups_by_id):
                if not group.is_mamba_align:
                    continue
                self._append_mamba_align_state_block(
                    load_ucm_block_ids,
                    load_vllm_block_ids,
                    req_meta,
                    request_id,
                    gid,
                    total_hit_tokens,
                    "load",
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
                if group.is_mamba_align:
                    continue
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
            (load_ucm_block_ids, load_vllm_block_ids),
            (dump_ucm_block_ids, dump_vllm_block_ids),
            load_full_attn_count,
            dump_full_attn_count,
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

    def _mla_split_scope(self, ucm_ids, vllm_ids, full_attn_count, is_dump):
        """Split into MLA/KDA and apply rank scoping for MLA hybrid.

        Returns ``(rank0_ucm, scoped_ucm, scoped_vllm)`` where *rank0_ucm*
        holds the rank-0 hashes of the blocks that will actually be stored
        (for the rank-consistency tracker), *scoped_ucm* holds the per-rank
        store keys, and *scoped_vllm* holds the matching vLLM block IDs.
        """
        n = full_attn_count
        mla_ucm, kda_ucm = ucm_ids[:n], ucm_ids[n:]
        mla_vllm, kda_vllm = vllm_ids[:n], vllm_ids[n:]
        is_rank0 = self.tp_rank % self.tp_size == 0
        # MLA: shared hash for all ranks; KDA: rank0 shared, non-rank0 per-rank hash
        if is_rank0:
            kda_scoped = kda_ucm
        else:
            kda_scoped = [self.request_hasher(b) for b in kda_ucm]
        if is_dump and not is_rank0:
            return kda_ucm, kda_scoped, kda_vllm
        return mla_ucm + kda_ucm, mla_ucm + kda_scoped, mla_vllm + kda_vllm

    def _scope_blocks(self, ucm_ids, vllm_ids, full_attn_count, is_dump):
        """Rank-scope block IDs for dump or load.

        Returns ``(rank0_ucm, scoped_ucm, scoped_vllm)`` where *rank0_ucm*
        is the rank-0 hash (for tracker clear/mark), *scoped_ucm* is the
        per-rank store key, and *scoped_vllm* is the matching vLLM block IDs.
        """
        n = int(full_attn_count) if full_attn_count else 0
        if self.is_mla:
            return self._mla_split_scope(ucm_ids, vllm_ids, n, is_dump)
        if self.tp_rank % self.tp_size == 0:
            return ucm_ids, ucm_ids, vllm_ids
        scoped = [self.request_hasher(b) for b in ucm_ids]
        return ucm_ids, scoped, vllm_ids

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        """Bulk load override: MLA blocks shared hash, KDA blocks per-rank hash."""
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, UCMConnectorMetadata)
        request_to_task: dict[str, Task] = {}
        is_load = False
        num_loaded_block = 0
        num_loaded_request = 0
        load_start_time = time.perf_counter() * 1000
        request_to_load_blocks: dict[str, int] = {}
        for request_id, request in metadata.request_meta.items():
            if len(request.load_block_ids[0]) == 0:
                continue
            is_load = True
            num_loaded_block += len(request.load_block_ids[0])
            num_loaded_request += 1
            n = getattr(request, "load_full_attn_count", 0)
            _, scoped_ucm, scoped_vllm = self._scope_blocks(
                request.load_block_ids[0], request.load_block_ids[1], n, is_dump=False
            )
            if not scoped_ucm:
                num_loaded_block -= len(request.load_block_ids[0])
                num_loaded_request -= 1
                continue
            num_loaded_block -= len(request.load_block_ids[0]) - len(scoped_ucm)
            try:
                ptrs = self.kv_cache_layout.extract_block_addrs(scoped_vllm)
                ptrs = ptrs.reshape(ptrs.shape[0], -1)
                shard_indexs = [0] * len(scoped_ucm)
                task = self._rank_consistency.submit_load(
                    self.store,
                    {request_id: request.load_block_ids[0]},
                    scoped_ucm,
                    shard_indexs,
                    ptrs,
                )
                request_to_task[request_id] = task
                request_to_load_blocks[request_id] = len(scoped_ucm)
            except Exception as e:
                logger.error(
                    f"request {request_id} submit load task error. "
                    f"{type(e).__name__}: {e}"
                )
                self._record_load_error(
                    "connector_load_submit_errors_total",
                    metadata.request_meta[request_id].load_block_ids[1]
                    + metadata.request_meta[request_id].dump_block_ids[1],
                )
                self._connector_worker_meta.mark_failed(request_id)
                num_loaded_block -= len(scoped_ucm)

        for request_id, task in request_to_task.items():
            try:
                self._rank_consistency.wait_load(task)
            except Exception as e:
                logger.error(
                    f"request {request_id} wait load task error. "
                    f"{type(e).__name__}: {e}"
                )
                self._record_load_error(
                    "connector_load_wait_errors_total",
                    metadata.request_meta[request_id].load_block_ids[1]
                    + metadata.request_meta[request_id].dump_block_ids[1],
                )
                self._connector_worker_meta.mark_failed(request_id)
                num_loaded_block -= request_to_load_blocks.get(request_id, 0)

        if is_load:
            load_end_time = time.perf_counter() * 1000
            load_duration_ms = load_end_time - load_start_time
            load_bytes = num_loaded_block * self.block_data_size
            load_speed = load_bytes / max(load_duration_ms, 1) / 1024 / 1024
            ucmmetrics.update_stats(
                {
                    "load_requests_num": num_loaded_request,
                    "load_blocks_num": num_loaded_block,
                    "load_duration": load_duration_ms,
                    "load_speed": load_speed,
                    "load_bytes_total": load_bytes,
                }
            )

    def wait_for_save(self) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, UCMConnectorMetadata)

        total_ucm_block_ids: list[bytes] = []
        total_vllm_block_ids: list[int] = []
        block_ids_by_request: dict[str, set[bytes]] = {}
        num_saved_block = 0
        num_saved_request = 0
        for request_id, request in metadata.request_meta.items():
            if len(request.dump_block_ids[0]) == 0:
                continue
            n = getattr(request, "dump_full_attn_count", 0)
            rank0_ucm, scoped_ucm, scoped_vllm = self._scope_blocks(
                request.dump_block_ids[0], request.dump_block_ids[1], n, is_dump=True
            )
            if not scoped_ucm:
                continue
            block_ids_by_request[request_id] = set(rank0_ucm)
            num_saved_block += len(scoped_ucm)
            num_saved_request += 1
            total_ucm_block_ids.extend(scoped_ucm)
            total_vllm_block_ids.extend(scoped_vllm)

        if not total_ucm_block_ids:
            return

        event_handle = 0
        try:
            total_ptrs = self.kv_cache_layout.extract_block_addrs(total_vllm_block_ids)
            total_ptrs = total_ptrs.reshape(total_ptrs.shape[0], -1)
            shard_indexs = [0] * len(total_ucm_block_ids)
            event_handle = self._get_dump_event_handle()
            save_start_time = time.perf_counter() * 1000
            task = self._rank_consistency.submit_dump(
                self.store,
                block_ids_by_request,
                total_ucm_block_ids,
                shard_indexs,
                total_ptrs,
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
        self.request_data: list[tuple[str, list[bytes], list[bytes], list[int]]] = []
        self._failure_req_ids: set[str] = set()
        self._submitted_load_rows: set[int] = set()
        self._dump_transfer_data: (
            tuple[list[bytes], list[int], set[str], dict[str, set[bytes]]] | None
        ) = None
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
        # MTP save_kv_layer is called N times per decode step; defer & keep last.
        self._deferred_mtp_row_dumps: dict[
            int, tuple[list[bytes], list[int], set[str]]
        ] = {}
        self._is_mtp = False
        self._init_mtp_layerwise_dump_state()

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
        """Check whether a layer belongs to the MTP draft model."""
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

        enable_affinity = _use_ucm_connector_cpu_affinity()
        worker_cores, store_cores = (
            self.device.split_cores(self.local_rank)
            if enable_affinity
            else (None, None)
        )

        self.store = self._create_store(
            kv_cache_layout=self.kv_cache_layout,
            cpu_affinity_cores=store_cores,
            tensor_size_list_override=row_tensor_size_list,
            shard_size_override=row_shard_size,
            block_size_override=row_shard_size * (max(self.row_ids) + 1),
        )

        if worker_cores:
            try:
                os.sched_setaffinity(0, worker_cores)
                logger.info(f"[VLLM CPU Affinity] Worker bound to cores {worker_cores}")
            except Exception as e:
                logger.warning(f"Failed to bind worker: {e}")

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
        for (
            request_id,
            ucm_block_ids,
            store_block_ids,
            vllm_block_ids,
        ) in self.request_data:
            if request_id in self._failure_req_ids:
                continue
            try:
                row_ptrs = self.kv_cache_layout.extract_block_addrs_for_row(
                    vllm_block_ids, row_id
                )
                shard_indexs = [row_id] * len(store_block_ids)
                task = self._rank_consistency.submit_load(
                    self.store,
                    {request_id: ucm_block_ids},
                    store_block_ids,
                    shard_indexs,
                    row_ptrs,
                )
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
                self._rank_consistency.wait_load(task)
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
            n = getattr(request, "load_full_attn_count", 0)
            _, scoped_ucm, scoped_vllm = self._scope_blocks(
                request.load_block_ids[0], request.load_block_ids[1], n, is_dump=False
            )
            if not scoped_ucm:
                continue
            self.need_load = True
            self.request_data.append(
                (request_id, request.load_block_ids[0], scoped_ucm, scoped_vllm)
            )

        if self.need_load and self.row_ids:
            # vLLM only calls wait_for_layer_load at full_attn (last layer of
            # each row), so row 0 must be loaded here before linear_attn begins.
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

        # Wait for NEXT row so its linear_attn layers have KV loaded.
        next_row_id = row_id + 1
        if next_row_id >= len(self.row_ids):
            return

        self._submit_request_load_tasks_for_row_once(next_row_id, metadata)

        wait_start = time.perf_counter()
        n_tasks = self._wait_row_load(next_row_id, metadata)
        wait_end = time.perf_counter()

        # Prefetch rows ahead.
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

        row_id = self.layer_name_to_row.get(layer_name)
        if row_id is None:
            return
        if self.row_save_layer.get(row_id) != layer_name:
            return

        metadata = self._get_connector_metadata()
        assert isinstance(metadata, UCMConnectorMetadata)
        if self._dump_transfer_data is None:
            self._dump_transfer_data = self._build_dump_transfer_data(metadata, row_id)
        (
            total_ucm_block_ids,
            total_vllm_block_ids,
            dump_request_ids,
            block_ids_by_request,
        ) = self._dump_transfer_data

        if not total_ucm_block_ids:
            return

        self.is_save = True

        if self._is_mtp_layer(layer_name):
            # Defer: last snapshot wins (MTP revisited N times per decode step).
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
            task = self._rank_consistency.submit_dump(
                self.store,
                block_ids_by_request,
                total_ucm_block_ids,
                shard_indexs,
                row_ptrs,
                event_handle,
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
    ) -> tuple[list[bytes], list[int], set[str], dict[str, set[bytes]]]:
        total_ucm_block_ids: list[bytes] = []
        total_vllm_block_ids: list[int] = []
        dump_request_ids: set[str] = set()
        block_ids_by_request: dict[str, set[bytes]] = {}
        for request_id, request in metadata.request_meta.items():
            if len(request.dump_block_ids[0]) == 0:
                continue
            dump_request_ids.add(request_id)
            n = getattr(request, "dump_full_attn_count", 0)
            rank0_ucm, scoped_ucm, scoped_vllm = self._scope_blocks(
                request.dump_block_ids[0], request.dump_block_ids[1], n, is_dump=True
            )
            if not scoped_ucm:
                continue
            block_ids_by_request[request_id] = set(rank0_ucm)
            total_ucm_block_ids.extend(scoped_ucm)
            total_vllm_block_ids.extend(scoped_vllm)
        return (
            total_ucm_block_ids,
            total_vllm_block_ids,
            dump_request_ids,
            block_ids_by_request,
        )

    def _flush_deferred_mtp_dumps(self) -> None:
        """Submit the last saved snapshot for each deferred MTP row."""
        if not self._deferred_mtp_row_dumps:
            return

        deferred = self._deferred_mtp_row_dumps
        self._deferred_mtp_row_dumps = {}
        block_ids_by_request = (
            self._dump_transfer_data[3] if self._dump_transfer_data is not None else {}
        )
        for row_id, (ucm_ids, vllm_ids, req_ids) in deferred.items():
            try:
                row_ptrs = self.kv_cache_layout.extract_block_addrs_for_row(
                    vllm_ids, row_id
                )
                row_ptrs = np.ascontiguousarray(row_ptrs)
                shard_indexs = [row_id] * len(ucm_ids)
                event_handle = self._get_dump_event_handle()
                task = self._rank_consistency.submit_dump(
                    self.store,
                    block_ids_by_request,
                    ucm_ids,
                    shard_indexs,
                    row_ptrs,
                    event_handle,
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

        dump_request_ids = (
            self._dump_transfer_data[2]
            if self._dump_transfer_data is not None
            else set()
        )
        self._flush_deferred_mtp_dumps()

        total_start = time.perf_counter()
        for row_id in self.row_ids:
            for pending_dump_task in self.dump_tasks.pop(row_id, []):
                try:
                    self._rank_consistency.wait_dump(pending_dump_task.task)
                except Exception as e:
                    logger.error_limit(
                        f"wait for dump kv cache failed. " f"{type(e).__name__}: {e}"
                    )
        self._rank_consistency.finish_dump(dump_request_ids)
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
