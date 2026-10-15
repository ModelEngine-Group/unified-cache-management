import copy
import math
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Optional, Sequence, Tuple

import numpy as np
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.core.sched.output import SchedulerOutput

try:
    from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
except ImportError:  # Older vLLM versions do not expose this helper.
    resolve_kv_cache_block_sizes = None

from ucm.integration.vllm.device import create_device
from ucm.integration.vllm.request_hasher import RequestHasher
from ucm.integration.vllm.ucm_connector import (
    UCMDirectConnector,
    UCMLiteConnector,
    _check_shm_capacity,
    _get_store_gc_block_size,
    _scheduler_read_block_size,
    _use_ucm_connector_cpu_affinity,
    build_kv_hash_meta,
)
from ucm.logger import init_logger
from ucm.shared.metrics import ucmmetrics
from ucm.sparse.utils import round_up
from ucm.store.factory_v1 import UcmConnectorFactoryV1
from ucm.store.ucmstore_v1 import Task, UcmKVStoreBaseV1

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


def _build_v41_hash_meta(
    vllm_config: "VllmConfig", rank_id: int, kv_cache_config: "KVCacheConfig"
) -> str:
    """Extend the common KV namespace with the resolved V4.1 transfer layout."""
    hf_config = getattr(vllm_config.model_config, "hf_config", None)
    compress_ratios = getattr(hf_config, "compress_ratios", None)
    cache_config = getattr(vllm_config, "cache_config", None)
    if cache_config is None:
        raise ValueError("DeepSeek V4.1 cache namespace needs cache_config.")
    ratios = ",".join(str(int(ratio)) for ratio in compress_ratios or ())
    layout_info = (
        ":kv_layout=v41_packed_v3"
        f":cache_dtype={cache_config.cache_dtype}"
        f":block_size={cache_config.block_size}"
        f":compress_ratios={ratios}"
    )
    # The Ascend adapter transfers one aliased physical page per
    # owner, not the CUDA value/scale segments. Never reuse files
    # written by the old Ascend C4 adapter or a GPU deployment.
    if any(
        type(inner).__name__.startswith("Ascend")
        and getattr(inner, "model_version", None) == "deepseek_v41"
        for group in kv_cache_config.kv_cache_groups
        for inner in (
            getattr(group.kv_cache_spec, "kv_cache_specs", None)
            or {"": group.kv_cache_spec}
        ).values()
    ):
        layout_info += ":ascend_v41_page_transfer=v1"
    tensors = getattr(kv_cache_config, "kv_cache_tensors", ())
    has_placement = bool(tensors) and hasattr(tensors[0], "layer_stride")
    groups = []
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        nested = getattr(spec, "kv_cache_specs", None)
        layers = []
        for name in sorted(group.layer_names):
            if has_placement:
                layers.append(name)
                continue
            inner = nested[name] if nested else spec
            layers.append(
                (
                    name,
                    getattr(inner, "tokens_per_state", 1),
                    getattr(inner, "state_content_size_bytes", None),
                    str(getattr(inner, "dtype", None)),
                    str(getattr(inner, "kv_quant_mode", None)),
                )
            )
        groups.append(
            (
                spec.block_size,
                getattr(spec, "prefix_cacheable", True),
                getattr(spec, "prefix_replay_tokens", 0),
                getattr(group, "enable_kv_transfer", True),
                tuple(layers),
            )
        )
    layout_info += f":transfer_layout=v4:{tuple(groups)!r}"
    if has_placement:
        placement = tuple(
            (tuple(t.layers), t.layer_stride, t.block_stride, t.offset) for t in tensors
        )
        layout_info += f":placement={placement!r}"
    return build_kv_hash_meta(vllm_config, rank_id) + layout_info


@dataclass(frozen=True)
class KVCacheGroupMeta:
    """Logical storage shape for one vLLM KV-cache group."""

    group_id: int
    token_block_size: int
    tail_blocks: int
    tail_tokens: int


class KVCacheGroupLayout:
    """Flat pointer layout for one vLLM KV cache group.

    The cache views belonging to one KV group are not necessarily contiguous by
    layer id, so this layout flattens all registered tensors in a deterministic
    order and records enough stride metadata to address arbitrary block rows.
    """

    def __init__(
        self,
        kvcaches: dict[str, torch.Tensor],
        *,
        is_ascend_layout: bool = False,
        expected_block_size: Optional[int] = None,
        standardized_layout: bool = False,
        expected_tokens_per_state: int = 1,
        expected_tokens_per_state_by_layer: Optional[dict[str, int]] = None,
        physical_layout_by_layer: Optional[dict[str, tuple[int, int]]] = None,
        whole_page_sizes_by_layer: Optional[dict[str, int]] = None,
    ) -> None:
        self.kvcaches = dict(sorted(kvcaches.items(), key=self._sort_key))
        self.is_ascend_layout = is_ascend_layout
        self.expected_block_size = expected_block_size
        self.standardized_layout = standardized_layout
        if standardized_layout and (
            expected_block_size is None or expected_block_size <= 0
        ):
            raise ValueError(
                "Standardized KV cache layout requires a positive block size."
            )
        if standardized_layout and expected_tokens_per_state <= 0:
            raise ValueError(
                "Standardized KV cache layout requires positive "
                f"tokens_per_state, got {expected_tokens_per_state}."
            )
        if standardized_layout and expected_block_size % expected_tokens_per_state:
            raise ValueError(
                "Standardized KV cache block size must be divisible by "
                f"tokens_per_state: block_size={expected_block_size}, "
                f"tokens_per_state={expected_tokens_per_state}."
            )
        self.expected_tokens_per_state = expected_tokens_per_state
        self.expected_tokens_per_state_by_layer = dict(
            expected_tokens_per_state_by_layer or {}
        )
        self.physical_layout_by_layer = dict(physical_layout_by_layer or {})
        self.whole_page_sizes_by_layer = dict(whole_page_sizes_by_layer or {})
        if standardized_layout and expected_tokens_per_state_by_layer is not None:
            cache_layers = set(self.kvcaches)
            mapped_layers = set(self.expected_tokens_per_state_by_layer)
            if mapped_layers != cache_layers:
                raise ValueError(
                    "Standardized KV cache layer/tokens_per_state mismatch: "
                    f"missing={sorted(cache_layers - mapped_layers)!r}, "
                    f"extra={sorted(mapped_layers - cache_layers)!r}."
                )
        self.base_ptrs: np.ndarray
        self.block_strides: np.ndarray
        self.tensor_token_strides: np.ndarray
        self.tensor_sizes_per_token: np.ndarray
        self.tensor_block_sizes: np.ndarray
        self._build_layout()

    @staticmethod
    def _sort_key(item: tuple[str, torch.Tensor]) -> tuple[int, str]:
        name, _ = item
        return (extract_layer_index(name), name)

    def _is_combined_kv_4d(
        self,
        shape: Sequence[int],
        layer_name: str,
    ) -> bool:
        """Classify a 4-D cache without confusing block_size=2 with K/V."""

        if self.is_ascend_layout:
            if (
                self.expected_block_size is not None
                and shape[1] != self.expected_block_size
            ):
                raise ValueError(
                    f"Ascend KV cache tensor block size mismatch for {layer_name}: "
                    f"shape={tuple(shape)}, expected={self.expected_block_size}."
                )
            return False

        is_combined_kv = shape[1] == 2
        token_dim = 2 if is_combined_kv else 1
        if (
            self.expected_block_size is not None
            and shape[token_dim] != self.expected_block_size
        ):
            raise ValueError(
                f"GPU KV cache tensor block size mismatch for {layer_name}: "
                f"shape={tuple(shape)}, expected={self.expected_block_size}."
            )
        return is_combined_kv

    def _build_layout(self) -> None:
        """Flatten registered KV tensors into store-compatible pointer rows."""

        ptrs: list[int] = []
        strides: list[int] = []
        tensor_token_strides: list[int] = []
        tensor_sizes_per_token: list[int] = []
        tensor_block_sizes: list[int] = []
        view_meta: list[tuple[str, tuple[int, ...], tuple[int, ...], str, int]] = []

        def handle_tensor(
            t: torch.Tensor,
            size_dims: Sequence[int],
            layer_name: str,
        ) -> None:
            ptrs.append(t[0].data_ptr())
            strides.append(t.stride(0) * t.element_size())
            tensor_size = math.prod([t.shape[i] for i in size_dims]) * t.element_size()
            token_dim = 1
            tensor_block_size = int(t.shape[token_dim])
            tensor_token_strides.append(t.stride(token_dim) * t.element_size())
            tensor_sizes_per_token.append(tensor_size // tensor_block_size)
            tensor_block_sizes.append(tensor_block_size)
            view_meta.append(
                (
                    layer_name,
                    tuple(t.shape),
                    tuple(t.stride()),
                    str(t.dtype),
                    tensor_block_size,
                )
            )

        def handle_kv_layer_tensor(tensor: torch.Tensor, layer_name: str) -> None:
            if self.is_ascend_layout and self.whole_page_sizes_by_layer:
                # Ascend V4.1 aliases MLA, indexer and state/window groups on
                # page-strided allocations. Only the page-origin owner is
                # registered here; its page includes the indexer's scale plane.
                page_bytes = self.whole_page_sizes_by_layer[layer_name]
                if (
                    tensor.dim() != 4
                    or tensor.stride(0) * tensor.element_size() != page_bytes
                ):
                    raise ValueError(
                        f"Ascend V4.1 page stride mismatch for {layer_name}: "
                        f"shape={tuple(tensor.shape)}, expected={page_bytes}."
                    )
                ptrs.append(tensor[0].data_ptr())
                strides.append(page_bytes)
                tensor_token_strides.append(0)
                tensor_sizes_per_token.append(page_bytes)
                tensor_block_sizes.append(1)
                view_meta.append(
                    (
                        layer_name,
                        tuple(tensor.shape),
                        tuple(tensor.stride()),
                        str(tensor.dtype),
                        1,
                    )
                )
                return
            if tensor.dim() == 5:
                # [2, num_blocks, block_size, num_head, head_dim]
                handle_tensor(tensor[0], (-3, -2, -1), layer_name)
                handle_tensor(tensor[1], (-3, -2, -1), layer_name)
            elif tensor.dim() == 4:
                if self.standardized_layout:
                    # Current vLLM unified views are [B, H, N, C]. N is the
                    # number of physical states, not the logical token block.
                    if (
                        tensor.shape[1] != 1
                        or tensor.stride(3) != 1
                        or tensor.stride(2) != tensor.shape[3]
                    ):
                        raise ValueError(
                            "Standardized V4.1 KV layout requires a contiguous "
                            "single-head [B, 1, N, C] view for transfer: "
                            f"{layer_name} shape={tuple(tensor.shape)}, "
                            f"strides={tuple(tensor.stride())}."
                        )
                    physical_states = int(tensor.shape[2])
                    if self.expected_tokens_per_state_by_layer:
                        tokens_per_state = self.expected_tokens_per_state_by_layer[
                            layer_name
                        ]
                    else:
                        tokens_per_state = self.expected_tokens_per_state
                    if not isinstance(tokens_per_state, int) or tokens_per_state <= 0:
                        raise ValueError(
                            f"Invalid tokens_per_state for {layer_name}: "
                            f"{tokens_per_state!r}."
                        )
                    if self.expected_block_size % tokens_per_state:
                        raise ValueError(
                            f"Block size {self.expected_block_size} is not divisible "
                            f"by tokens_per_state={tokens_per_state} for {layer_name}."
                        )
                    expected_states = self.expected_block_size // tokens_per_state
                    if physical_states != expected_states:
                        raise ValueError(
                            f"Standardized KV cache state count mismatch for "
                            f"{layer_name}: N={physical_states}, expected "
                            f"{expected_states} from block_size="
                            f"{self.expected_block_size} and "
                            f"tokens_per_state={tokens_per_state}."
                        )
                    page_stride = tensor.stride(0) * tensor.element_size()
                    physical_layout = self.physical_layout_by_layer.get(layer_name)
                    whole_page_size = self.whole_page_sizes_by_layer.get(layer_name)
                    if whole_page_size is not None:
                        if whole_page_size % physical_states:
                            raise ValueError("Whole KV page size must divide by N.")
                        descriptors = ((0, 0, whole_page_size // physical_states),)
                    elif physical_layout is None:
                        descriptors = (
                            (
                                0,
                                tensor.stride(2) * tensor.element_size(),
                                tensor.shape[3] * tensor.element_size(),
                            ),
                        )
                    else:
                        value_bytes, scale_bytes = physical_layout
                        if value_bytes <= 0 or scale_bytes <= 0:
                            raise ValueError(
                                f"Invalid physical KV layout for {layer_name}: "
                                f"{physical_layout!r}."
                            )
                        device = getattr(tensor, "device", None)
                        if (
                            getattr(device, "type", None) == "cuda"
                            and "indexer.k_cache" in layer_name
                        ):
                            # ROCm's block_size > 1 indexer cache is tiled in the
                            # value region.  A partial hash segment cannot be
                            # represented by contiguous descriptors safely.
                            if getattr(getattr(torch, "version", None), "hip", None):
                                raise NotImplementedError(
                                    "UCM does not support partial transfers of "
                                    "the ROCm tiled indexer KV layout."
                                )
                        if value_bytes + scale_bytes != (
                            tensor.shape[3] * tensor.element_size()
                        ):
                            raise ValueError(
                                f"Physical KV layout for {layer_name} does not "
                                f"match C={tensor.shape[3]}: {physical_layout!r}."
                            )
                        descriptors = (
                            (0, value_bytes, value_bytes),
                            (physical_states * value_bytes, scale_bytes, scale_bytes),
                        )
                    page_payload = sum(
                        bytes_per_state * physical_states
                        for _, _, bytes_per_state in descriptors
                    )
                    if page_stride < page_payload:
                        raise ValueError(
                            f"Standardized KV page stride is smaller than its "
                            f"payload for {layer_name}: stride={page_stride}, "
                            f"payload={page_payload}."
                        )
                    for base_offset, token_stride, bytes_per_state in descriptors:
                        ptrs.append(tensor[0].data_ptr() + base_offset)
                        strides.append(page_stride)
                        tensor_token_strides.append(token_stride)
                        tensor_sizes_per_token.append(bytes_per_state)
                        tensor_block_sizes.append(physical_states)
                        view_meta.append(
                            (
                                layer_name,
                                tuple(tensor.shape),
                                tuple(tensor.stride()),
                                str(tensor.dtype),
                                physical_states,
                            )
                        )
                    return
                if self._is_combined_kv_4d(tensor.shape, layer_name):
                    # GPU kernels may register [num_blocks, 2, block_size, ...];
                    # split the K/V axis before reading the token dimension.
                    handle_tensor(tensor[:, 0], (-2, -1), layer_name)
                    handle_tensor(tensor[:, 1], (-2, -1), layer_name)
                else:
                    # Ascend registers split KV/state tensors as
                    # [num_blocks, block_size, num_head, head_dim].
                    handle_tensor(tensor, (-3, -2, -1), layer_name)
            elif tensor.dim() == 3:
                # [num_blocks, block_size, head_dim]. Some DeepSeek V4 caches
                # use block_size=2 here and share a group with larger pages.
                handle_tensor(tensor, (-2, -1), layer_name)
            else:
                raise ValueError(
                    f"Unsupported KV cache tensor shape for "
                    f"{layer_name}: {tensor.shape}"
                )

        for layer_name, kv_layer in self.kvcaches.items():
            if isinstance(kv_layer, torch.Tensor):
                handle_kv_layer_tensor(kv_layer, layer_name)
            elif isinstance(kv_layer, Tuple):
                for tensor in kv_layer:
                    handle_kv_layer_tensor(tensor, layer_name)
            else:
                raise TypeError(
                    f"Unsupported KV cache type for " f"{layer_name}: {type(kv_layer)}"
                )

        if not ptrs:
            raise ValueError("KV cache group layout is empty.")

        self.base_ptrs = np.asarray(ptrs, dtype=np.uint64)
        self.block_strides = np.asarray(strides, dtype=np.uint64)
        self.tensor_token_strides = np.asarray(tensor_token_strides, dtype=np.uint64)
        self.tensor_sizes_per_token = np.asarray(
            tensor_sizes_per_token, dtype=np.uint64
        )
        self.tensor_block_sizes = np.asarray(tensor_block_sizes, dtype=np.uint64)
        self.view_meta = [
            {
                "name": name,
                "shape": shape,
                "stride": stride,
                "dtype": dtype,
                "tensor_block_size": tensor_block_size,
            }
            for name, shape, stride, dtype, tensor_block_size in view_meta
        ]
        logger.info(
            f"KV cache group layout: views={len(self.kvcaches)}, "
            f"ptrs={len(ptrs)}, "
            f"tensor_block_sizes={sorted(set(tensor_block_sizes))}"
        )

    def extract_addrs_with_offsets(
        self,
        block_ids: np.ndarray,
        group_token_block_size: int,
        offsets: np.ndarray,
    ) -> np.ndarray:
        """Return per-view addresses for logical blocks with token offsets."""

        if self.whole_page_sizes_by_layer and np.any(offsets):
            raise ValueError("Whole-page KV transfers cannot start inside a page.")
        physical_token_offsets = (
            offsets[:, None]
            * self.tensor_block_sizes[None, :]
            // group_token_block_size
        )

        return (
            block_ids[:, None] * self.block_strides[None, :]
            + physical_token_offsets * self.tensor_token_strides[None, :]
            + self.base_ptrs[None, :]
        ).astype(np.uint64, copy=False)

    def extract_addrs(
        self,
        block_ids: np.ndarray,
    ) -> np.ndarray:
        """Return per-view base addresses for complete tensor blocks."""

        return (
            block_ids[:, None] * self.block_strides[None, :] + self.base_ptrs[None, :]
        ).astype(np.uint64, copy=False)

    def segment_tensor_size_list(
        self,
        logical_tokens: int,
        group_token_block_size: int,
    ) -> list[int]:
        """Return byte sizes for one logical segment across all tensor views."""

        if self.whole_page_sizes_by_layer and logical_tokens != group_token_block_size:
            raise ValueError("Whole-page KV transfers require a complete group block.")
        tensor_tokens = (
            self.tensor_block_sizes * logical_tokens // group_token_block_size
        )
        return (self.tensor_sizes_per_token * tensor_tokens).tolist()

    @property
    def tensor_block_size(self) -> int:
        if len(set(self.tensor_block_sizes.tolist())) != 1:
            raise ValueError(
                "KV cache group layout has mixed view tensor block sizes: "
                f"{self.tensor_block_sizes.tolist()}"
            )
        return int(self.tensor_block_sizes[0])


def _group_tokens_per_state_by_layer(group_spec) -> dict[str, int]:
    spec = group_spec.kv_cache_spec
    nested = getattr(spec, "kv_cache_specs", None)
    if nested:
        expected = set(group_spec.layer_names)
        actual = set(nested)
        if actual != expected:
            raise ValueError(
                "Uniform KV cache group layer/spec mismatch: "
                f"missing={sorted(expected - actual)!r}, "
                f"extra={sorted(actual - expected)!r}."
            )
        source = nested
    else:
        source = {name: spec for name in group_spec.layer_names}
    result = {}
    for name, inner in source.items():
        value = getattr(inner, "tokens_per_state", 1)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"Standardized KV cache requires positive integer "
                f"tokens_per_state for {name}, got {value!r}."
            )
        result[name] = value
    return result


def _v41_whole_page_sizes(kv_cache_config) -> dict[str, int]:
    """Read padded layer-page sizes without relying on scheduler's merged spec."""
    layout = getattr(kv_cache_config, "kv_cache_layout", None)
    tensors = getattr(kv_cache_config, "kv_cache_tensors", ())
    if layout is None or not tensors:
        return {}
    if layout not in ("BLHNC", "BLNHC"):
        raise ValueError(
            "UCM V4.1 whole-page transfer requires BLHNC or BLNHC KV layout."
        )
    sizes = {name: tensor.layer_stride for tensor in tensors for name in tensor.layers}
    if any(size <= 0 for size in sizes.values()):
        raise ValueError("UCM V4.1 requires positive layer-page sizes.")
    return sizes


def _ascend_v41_whole_page_sizes(kv_cache_config) -> dict[str, int]:
    """Select one page-origin view per group/allocation for Ascend 0.30."""
    sizes = {}
    tensors = getattr(kv_cache_config, "kv_cache_tensors", ())
    if not tensors:
        raise ValueError("Ascend V4.1 requires physical KV page descriptors.")
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        nested = getattr(spec, "kv_cache_specs", None)
        specs = list(nested.values()) if nested else [spec]
        if not all(getattr(s, "prefix_cacheable", True) for s in specs):
            if not all(type(s).__name__ == "CircularBufferSpec" for s in specs):
                raise ValueError(
                    "Ascend V4.1 UCM requires persisted SWA windows; "
                    "bounded replay and other transient groups are not supported."
                )
            continue
        if not getattr(group, "enable_kv_transfer", True):
            raise ValueError("Ascend V4.1 cannot omit a persistent KV group.")
        remaining = set(group.layer_names)
        for tensor in tensors:
            members = remaining.intersection(tensor.layers)
            if not members:
                continue
            if (
                tensor.layer_stride != 0
                or tensor.offset != 0
                or tensor.block_stride <= 0
            ):
                raise ValueError("Unsupported Ascend V4.1 physical page placement.")
            if getattr(specs[0], "sliding_window", None) is not None:
                if len(members) != 1:
                    raise ValueError(
                        "Ascend V4.1 SWA groups must use distinct page slots."
                    )
                owner = next(iter(members))
            else:
                owners = [name for name in members if name.endswith(".long_kv_cache")]
                if len(owners) != 1:
                    raise ValueError("Ascend V4.1 FA page needs one long KV owner.")
                owner = owners[0]
                indexer = owner.removesuffix(".long_kv_cache") + ".indexer.k_cache"
                if members != {owner, indexer}:
                    raise ValueError("Unsupported Ascend V4.1 FA page contents.")
            sizes[owner] = tensor.block_stride
            remaining.difference_update(members)
        if remaining:
            raise ValueError(
                f"Missing Ascend V4.1 page descriptors: {sorted(remaining)}"
            )
    return sizes


def _group_physical_layout_by_layer(
    group_spec,
) -> dict[str, tuple[int, int]]:
    """Return ``(value_bytes, scale_bytes)`` for packed V4.1 cache pages."""
    spec = group_spec.kv_cache_spec
    nested = getattr(spec, "kv_cache_specs", None)
    source = nested if nested else {name: spec for name in group_spec.layer_names}
    layouts: dict[str, tuple[int, int]] = {}
    for name in group_spec.layer_names:
        inner = source[name]
        total = getattr(inner, "state_content_size_bytes", None)
        cache_dtype = getattr(inner, "cache_dtype_str", None)
        if "indexer.k_cache" in name:
            if total not in (68, 132):
                raise ValueError(
                    f"Unsupported V4.1 indexer cache width for {name}: {total!r}."
                )
            layouts[name] = (int(total) - 4, 4)
        elif cache_dtype in ("fp8_ds_mla", "nvfp4_ds_mla") or total == 584:
            packed_records = {584: (576, 8), 528: (512, 16), 288: (256, 32)}
            if total not in packed_records:
                raise ValueError(
                    f"Unsupported V4.1 main KV cache width for {name}: {total!r}."
                )
            layouts[name] = packed_records[total]
    return layouts


@dataclass
class FAWARequestMeta:
    """Scheduler-side state accumulated for one request."""

    ucm_block_ids: list[bytes] = field(default_factory=list)
    hbm_hit_block_num: int = 0
    total_hit_block_num: int = 0
    num_token_ids: int = 0
    vllm_block_ids: tuple[list[int], ...] = field(default_factory=tuple)
    token_processed: int = 0


@dataclass
class FAWARequestDispatchMeta:
    """Per-step load and dump plan sent from scheduler to workers."""

    load_keys: list[bytes] = field(default_factory=list)
    load_hash_start: int = 0
    load_hash_end: int = 0
    load_vllm_block_ids: tuple[list[int], ...] = field(default_factory=tuple)
    dump_keys: list[bytes] = field(default_factory=list)
    dump_hash_start: int = 0
    dump_hash_end: int = 0
    dump_vllm_block_ids: tuple[list[int], ...] = field(default_factory=tuple)


@dataclass
class UCMFAWAConnectorMetadata(KVConnectorMetadata):
    """Connector metadata carrying FAWA dispatch plans for this step."""

    request_meta: dict[str, FAWARequestDispatchMeta] = field(default_factory=dict)
    preempted_req_ids: set[str] = field(default_factory=set)


@dataclass
class FAWALoadTask:
    """Outstanding FAWA load task plus scheduler-visible failure anchors."""

    request_id: str
    label: str
    store: UcmKVStoreBaseV1
    task: Task
    key_count: int


@dataclass
class FAWADumpTask:
    """Outstanding FAWA dump task submitted to one backing store."""

    label: str
    store: UcmKVStoreBaseV1
    task: Task
    key_count: int
    event_handle: int


@dataclass(frozen=True)
class BlockGeometry:
    logical: int
    physical: int


@lru_cache(maxsize=None)
def _ascend_block_size_is_logical() -> bool:
    """Detect once whether Ascend specs report logical block sizes.

    vLLM-Ascend introduced get_storage_block_size together with the runner
    switch to logical block sizes (block_size = physical * compress_ratio).
    """
    try:
        from vllm_ascend.core.kv_cache_interface import (  # noqa: F401
            get_storage_block_size,
        )
    except ImportError:
        is_logical = False
    else:
        is_logical = True
    logger.info(
        "Ascend KV cache block_size convention: "
        f"{'logical' if is_logical else 'physical'} "
        f"(block_size_is_logical={is_logical})."
    )
    return is_logical


class UCMFAWAConnector(UCMDirectConnector, SupportsHMA):
    """UCM connector for mixed full-attention and window KV cache groups.

    Full-attention groups are stored once per reusable prefix block and are
    loaded for every external prefix hit. WA groups store the tail blocks
    needed at each prefix boundary, and only the final matched boundary is
    loaded.
    """

    DEFAULT_HASH_BLOCK_SIZE = 256
    ASCEND_SUPPORTED_VLLM_BLOCK_SIZES = frozenset({32, 64, 128})
    ASCEND_C4_COMPRESS_RATIO = 4

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        self._defer_scheduler_store = True
        super().__init__(vllm_config, role, kv_cache_config)
        self.hash_block_size = self.DEFAULT_HASH_BLOCK_SIZE
        self.group_layouts: dict[int, KVCacheGroupLayout] = {}
        if self._kv_cache_config is None:
            raise RuntimeError("FAWA connector requires kv_cache_config.")

        self.is_ascend_layout = False
        self.ascend_base_block_size: Optional[int] = None
        self.fa_group_ids, self.window_group_ids = [], []
        self.resume_token_alignment = 1
        self.group_metas: dict[int, KVCacheGroupMeta] = {}
        self.file_size = {}

        # The maximum token block size across all groups, used for aligning the number of computed tokens in the scheduler.
        self.max_token_block_size = 0
        self._init_group_metas()
        if self.is_v41_layout:
            rank_id = (
                0 if role == KVConnectorRole.SCHEDULER else self.tp_rank % self.tp_size
            )
            self.request_hasher = RequestHasher(
                _build_v41_hash_meta(vllm_config, rank_id, self._kv_cache_config)
            )
            if role == KVConnectorRole.SCHEDULER:
                self._seed = self.request_hasher.seed
        self._bind_request_block_hasher()
        self.fa_store: Optional[UcmKVStoreBaseV1] = None
        self.wa_store: Optional[UcmKVStoreBaseV1] = None
        self.requests_meta: dict[str, FAWARequestMeta] = {}
        self.tp_dump_tasks: dict[tuple, list[FAWADumpTask]] = {}
        self.wa_dump_block_wise = self.launch_config.get("wa_dump_block_wise", True)

        # If the number of external hit blocks is small, it's possible that the load overhead is larger than the compute of a few blocks.
        # In that case, we can skip loading and directly compute the missed blocks, which can be faster.
        # This threshold can be tuned based on the performance characteristics of the system.
        self.load_tokens_threshold = self.launch_config.get(
            "load_tokens_threshold", 2048
        )

        if role == KVConnectorRole.SCHEDULER:
            self.fa_store = self._create_fa_store(None)
            self.wa_store = self._create_wa_store(None)

        group_meta_summary = tuple(
            {
                "group_id": meta.group_id,
                "token_block_size": meta.token_block_size,
                "tail_blocks": meta.tail_blocks,
                "tail_tokens": meta.tail_tokens,
            }
            for _, meta in sorted(self.group_metas.items())
        )
        logger.info(
            f"FAWA KV group config: fa_groups={self.fa_group_ids}, "
            f"window_groups={self.window_group_ids}, "
            f"hash_block_size={self.hash_block_size}, "
            f"transient_group_ids={sorted(self.transient_group_ids)}, "
            f"is_v41_layout={self.is_v41_layout}, "
            f"is_ascend_v41_layout={self.is_ascend_v41_layout}, "
            f"ascend_base_block_size={self.ascend_base_block_size}, "
            f"is_ascend_layout={self.is_ascend_layout}, "
            f"group_metas={group_meta_summary}"
        )
        logger.info("Init UCM FAWA connector.")

    def get_block_size(self) -> int:
        return self.hash_block_size

    @classmethod
    def ascend_block_geometry(
        cls, spec, *, block_size_is_logical: bool | None = None
    ) -> BlockGeometry:
        """Follow Ascend's block convention, not the inherited storage property.

        Old v1 runners allocate block_size rows even if vLLM exposes an inherited
        storage_block_size = block_size // compress_ratio property. The newer
        logical-block implementation exposes get_storage_block_size in Ascend.
        Detection assumes that helper and the runner's logical-block convention
        are introduced together; partial cherry-picks must explicitly provide
        block_size_is_logical to match their runner.
        """
        if block_size_is_logical is None:
            block_size_is_logical = _ascend_block_size_is_logical()

        nested = getattr(spec, "kv_cache_specs", None)
        if nested:
            layers_by_geometry: dict[BlockGeometry, list[str]] = {}
            for layer_name, member in nested.items():
                geometry = cls.ascend_block_geometry(
                    member, block_size_is_logical=block_size_is_logical
                )
                layers_by_geometry.setdefault(geometry, []).append(layer_name)
            if len(layers_by_geometry) != 1:
                details = {
                    f"(logical={g.logical}, physical={g.physical})": (
                        f"{len(names)} layers, e.g. {names[0]}"
                    )
                    for g, names in layers_by_geometry.items()
                }
                raise ValueError(
                    "All layers in one Ascend KV cache group must share one "
                    "block geometry (block_size_is_logical="
                    f"{block_size_is_logical}); got {details}."
                )
            return next(iter(layers_by_geometry))

        size = int(spec.block_size)
        ratio = int(getattr(spec, "compress_ratio", 1))
        if min(size, ratio) <= 0:
            raise ValueError("Ascend block size and compress ratio must be positive.")
        if block_size_is_logical:
            if size % ratio:
                raise ValueError(
                    "Logical block size must be divisible by compress ratio."
                )
            return BlockGeometry(size, size // ratio)
        return BlockGeometry(size * ratio, size)

    @classmethod
    def can_handle_kv_cache_config(
        cls, kv_cache_config: Optional["KVCacheConfig"]
    ) -> bool:
        """Return whether this connector supports the given hybrid KV layout."""

        if kv_cache_config is None:
            return False

        kv_cache_groups = kv_cache_config.kv_cache_groups
        spec_names = {
            type(spec).__name__
            for group in kv_cache_groups
            for spec in cls._specs_for_group(group)
        }
        # GPU FAWA is currently selected for DeepSeek-V4 style MLA+SWA layouts.
        DS_V4_REQUIRED_SPECS = frozenset({"SlidingWindowMLASpec"})
        gpu_support = DS_V4_REQUIRED_SPECS.issubset(spec_names)
        if gpu_support:
            return True
        return cls.can_handle_ascend_kv_cache_config(kv_cache_config)

    @classmethod
    def can_handle_ascend_kv_cache_config(
        cls, kv_cache_config: Optional["KVCacheConfig"]
    ) -> bool:
        """Return whether the KV config matches the supported Ascend FAWA layout."""

        if kv_cache_config is None:
            return False
        kv_cache_groups = kv_cache_config.kv_cache_groups
        spec_names = {
            type(spec).__name__
            for group in kv_cache_groups
            for spec in cls._specs_for_group(group)
        }
        ASCEND_REQUIRED_SPECS = frozenset({"AscendSlidingWindowMLASpec"})
        npu_support = ASCEND_REQUIRED_SPECS.issubset(spec_names)
        return npu_support

    @classmethod
    def _get_ascend_base_block_size(cls, kv_cache_config: "KVCacheConfig") -> int:
        """Read the user-scale block size from the Ascend C4 FA group.

        The scheduler mutates ``vllm_config.cache_config.block_size`` to the
        smallest hybrid-group block size, while workers retain the configured
        value. The C4 full-attention group is present in both roles and keeps
        the original 32/64/128 block size, so it is the stable source here.
        """

        c4_block_sizes = set()
        for group in kv_cache_config.kv_cache_groups:
            kv_cache_spec = group.kv_cache_spec
            for spec in cls._specs_for_group(group):
                if (
                    getattr(spec, "sliding_window", None) is None
                    and getattr(spec, "compress_ratio", 1)
                    == cls.ASCEND_C4_COMPRESS_RATIO
                ):
                    c4_block_sizes.add(
                        cls.ascend_block_geometry(kv_cache_spec).physical
                    )

        if len(c4_block_sizes) != 1:
            raise ValueError(
                "Expected exactly one Ascend C4 full-attention block size, "
                f"got {sorted(c4_block_sizes)}."
            )
        block_size = c4_block_sizes.pop()

        if block_size not in cls.ASCEND_SUPPORTED_VLLM_BLOCK_SIZES:
            supported = sorted(cls.ASCEND_SUPPORTED_VLLM_BLOCK_SIZES)
            raise ValueError(
                f"Unsupported DeepSeek V4 Ascend block size {block_size}; "
                f"expected one of {supported}."
            )
        return block_size

    def _init_group_metas(self) -> None:
        """Classify FA/WA groups and compute their logical segment sizes."""

        groups = self._kv_cache_config.kv_cache_groups
        self.is_ascend_v41_layout = self.can_handle_ascend_kv_cache_config(
            self._kv_cache_config
        ) and any(
            getattr(spec, "model_version", None) == "deepseek_v41"
            for group in groups
            for spec in self._specs_for_group(group)
        )
        if self.can_handle_ascend_kv_cache_config(self._kv_cache_config):
            self.is_ascend_layout = True
            if not self.is_ascend_v41_layout:
                self.ascend_base_block_size = self._get_ascend_base_block_size(
                    self._kv_cache_config
                )
                self.hash_block_size = (
                    self.ascend_base_block_size * self.ASCEND_C4_COMPRESS_RATIO
                )

        self.transient_group_ids = {
            group_id
            for group_id, group in enumerate(groups)
            if not getattr(group, "enable_kv_transfer", True)
            or not getattr(group.kv_cache_spec, "prefix_cacheable", True)
        }
        self.prefix_replay_tokens = max(
            (
                getattr(group.kv_cache_spec, "prefix_replay_tokens", 0)
                for group in groups
            ),
            default=0,
        )
        hf_config = self._vllm_config.model_config.hf_config
        self.is_v41_layout = bool(
            self.is_ascend_v41_layout
            or getattr(hf_config, "kv_source_layer_ids", None)
            or getattr(hf_config, "index_source_layer_ids", None)
        )
        self.fa_group_ids, self.window_group_ids = [], []
        layer_compress_ratios = getattr(
            self._vllm_config.model_config.hf_config,
            "compress_ratios",
            None,
        )
        if layer_compress_ratios is None:
            raise ValueError("current only support DSV4")
        compressed_ratios = {int(ratio) for ratio in layer_compress_ratios}
        self.is_v41_layout |= compressed_ratios.issubset({0, 1, 2}) and bool(
            compressed_ratios & {1, 2}
        )
        if self.is_ascend_v41_layout:
            parallel = getattr(self._vllm_config, "parallel_config", None)
            if any(
                getattr(parallel, name, 1) != 1
                for name in (
                    "pipeline_parallel_size",
                    "prefill_context_parallel_size",
                    "decode_context_parallel_size",
                )
            ):
                raise ValueError("Ascend V4.1 UCM currently requires PP=PCP=DCP=1.")
            self.whole_page_sizes = _ascend_v41_whole_page_sizes(self._kv_cache_config)
            # Ascend already expresses block_size in original tokens. Use a
            # complete logical page, even when the engine hashes finer chunks.
            block_sizes = {
                group.kv_cache_spec.block_size
                for group in groups
                if any(name.endswith(".long_kv_cache") for name in group.layer_names)
            }
            if len(block_sizes) != 1 or next(iter(block_sizes)) <= 0:
                raise ValueError(
                    "Ascend V4.1 needs one positive logical FA block size."
                )
            self.hash_block_size = self.ascend_base_block_size = block_sizes.pop()
            if self.prefix_replay_tokens:
                raise ValueError("Ascend V4.1 UCM does not support bounded replay.")
        else:
            self.whole_page_sizes = (
                _v41_whole_page_sizes(self._kv_cache_config)
                if self.is_v41_layout
                else {}
            )
        if self.is_v41_layout and not self.is_ascend_v41_layout:
            if resolve_kv_cache_block_sizes is None:
                raise RuntimeError(
                    "DeepSeek V4.1 requires vLLM's "
                    "resolve_kv_cache_block_sizes helper."
                )
            try:
                resolved = resolve_kv_cache_block_sizes(
                    self._kv_cache_config, self._vllm_config
                )
                if not (
                    isinstance(resolved, tuple)
                    and len(resolved) == 2
                    and all(isinstance(size, int) and size > 0 for size in resolved)
                ):
                    raise TypeError(
                        "expected (scheduler_block_size, hash_block_size), "
                        f"got {resolved!r}"
                    )
                # Whole-page transfer uses scheduler alignment, not finer hashes.
                self.hash_block_size = int(
                    resolved[0] if self.whole_page_sizes else resolved[1]
                )
            except Exception as exc:
                raise RuntimeError(
                    "Unable to resolve DeepSeek V4.1 scheduler hash block size "
                    "with resolve_kv_cache_block_sizes."
                ) from exc
        for group_id, group in enumerate(groups):
            kv_cache_spec = group.kv_cache_spec
            # Use the representative spec when vLLM wraps multiple layer specs.
            nested_specs = getattr(kv_cache_spec, "kv_cache_specs", None)
            spec = next(iter(nested_specs.values())) if nested_specs else kv_cache_spec
            if group_id in self.transient_group_ids:
                # Keep a metadata entry so scheduler/worker tuples retain the
                # vLLM group index, but never put the transient ring in a store.
                self.group_metas[group_id] = KVCacheGroupMeta(
                    group_id, kv_cache_spec.block_size, 0, 0
                )
                continue
            window_size = getattr(spec, "sliding_window", None)
            token_block_size = kv_cache_spec.block_size
            if self.is_ascend_layout and not self.is_ascend_v41_layout:
                # Ascend compressed groups expose a logical block span scaled by
                # the compression ratio.
                geometry = self.ascend_block_geometry(kv_cache_spec)
                token_block_size = geometry.logical
                logger.info(
                    f"FAWA Ascend group {group_id} block geometry: "
                    f"spec.block_size={kv_cache_spec.block_size}, "
                    f"logical={geometry.logical}, physical={geometry.physical}"
                )

            if window_size is None:
                if self.whole_page_sizes and token_block_size != self.hash_block_size:
                    raise ValueError(
                        "UCM V4.1 whole-page transfer requires FA group blocks "
                        "to match the scheduler alignment."
                    )
                # FA groups store one canonical hash block per row.
                tail_tokens = self.hash_block_size
                self.fa_group_ids.append(group_id)
            else:
                tensor_name = group.layer_names[0]
                if self.is_ascend_v41_layout or tensor_name.split(".")[-1] in [
                    "swa_cache"
                ]:
                    # SWA caches keep the full sliding-window tail.
                    tail_tokens = window_size
                else:
                    # Compressor state caches keep only the uncompressed tail.
                    layer_index = extract_layer_index(tensor_name)
                    tail_tokens = window_size - layer_compress_ratios[layer_index]

                self.window_group_ids.append(group_id)

            if self.is_ascend_v41_layout and window_size is not None:
                tail_tokens = round_up(tail_tokens, token_block_size)
            tail_blocks = max(tail_tokens // token_block_size, 1)
            if self.is_ascend_v41_layout and self.hash_block_size % token_block_size:
                raise ValueError(
                    "Ascend V4.1 hits must align to every persistent group."
                )
            self.max_token_block_size = max(self.max_token_block_size, token_block_size)
            self.group_metas[group_id] = KVCacheGroupMeta(
                group_id=group_id,
                token_block_size=token_block_size,
                tail_blocks=tail_blocks,
                tail_tokens=tail_tokens,
            )
        logger.info_once(
            f"max token_block_size of all groups: {self.max_token_block_size}"
        )
        if (
            self.max_token_block_size
            and self.max_token_block_size % self.hash_block_size != 0
        ):
            raise ValueError(
                f"Maximum token block size {self.max_token_block_size} must be "
                f"divisible by hash block size {self.hash_block_size}."
            )
        if self.is_v41_layout:
            # The transient compressor state is deliberately not persisted.
            # A request resumed from an external FAWA hit must therefore start
            # at a complete compression group so the first new token does not
            # consume a missing predecessor from that ring.
            compression_granularities = (
                [r for r in compressed_ratios if r > 0]
                if self.is_ascend_v41_layout
                else []
            )
            for group in groups:
                for spec in self._specs_for_group(group):
                    granularity = getattr(spec, "tokens_per_state", None)
                    if granularity is None:
                        granularity = getattr(spec, "compress_ratio", 1)
                    if isinstance(granularity, int) and granularity > 0:
                        compression_granularities.append(granularity)
            self.resume_token_alignment = math.lcm(*compression_granularities)
            if self.hash_block_size % self.resume_token_alignment:
                raise ValueError(
                    "DeepSeek V4.1 hash block size must be divisible by the "
                    "KV compression group size: "
                    f"hash_block_size={self.hash_block_size}, "
                    f"compression_group_size={self.resume_token_alignment}."
                )
        # V4.1 layouts derive payloads from the actual KV specs.  V4.0 keeps
        # the established model-specific sizing for compatibility.
        if self.is_v41_layout:
            self.file_size = self._derive_v41_file_sizes(groups)
            logger.info(
                "DeepSeek V4.1 UCM groups: excluded=%s, windows=%s, replay_tokens=%s",
                sorted(self.transient_group_ids),
                self.window_group_ids,
                self.prefix_replay_tokens,
            )
            return

        # get file size for block gc
        if len(layer_compress_ratios) < 61:
            # for dsv4 flash
            num_c4a_layers = 21
            num_c128a_layers = 20
            # TODO only support for dp tp
            num_total_layers = 43
        else:
            # for dsv4 pro
            num_c4a_layers = 30
            num_c128a_layers = 31
            num_total_layers = 61

        if (
            self._vllm_config.speculative_config is not None
            and self._vllm_config.speculative_config.num_speculative_tokens > 0
        ):
            num_total_layers += 1

        # TODO we should get file size in worker thread
        if self.is_ascend_layout:
            if self.ascend_base_block_size is None:
                raise RuntimeError("Ascend base block size was not initialized.")
            # One C4 row consumes a complete physical block. One C128 row
            # consumes block_size / 32 physical tokens, so both contributions
            # scale linearly with the configured vLLM block size.
            c4a_bytes_per_block_token = 1024 + 128 + 2
            c128a_bytes_per_block_token = 32
            self.file_size["FA"] = self.ascend_base_block_size * (
                c4a_bytes_per_block_token * num_c4a_layers
                + c128a_bytes_per_block_token * num_c128a_layers
            )
            self.file_size["WA"] = (
                131072 * num_total_layers + (32768 + 8192) * num_c4a_layers
            )
        else:
            self.file_size["FA"] = (
                37376 + 8448
            ) * num_c4a_layers + 1168 * num_c128a_layers
            self.file_size["WA"] = (37376 * 2) * num_total_layers + (
                8192 + 32768
            ) * num_c4a_layers
        self.file_size["FA"] = round_up(self.file_size["FA"], 4096)
        self.file_size["WA"] = round_up(self.file_size["WA"], 4096)

    @staticmethod
    def _specs_for_group(group) -> list[object]:
        spec = group.kv_cache_spec
        nested = getattr(spec, "kv_cache_specs", None)
        if nested:
            missing = [name for name in group.layer_names if name not in nested]
            if missing:
                raise ValueError(
                    f"KV cache group is missing specs for layers {missing!r}."
                )
            return [nested[name] for name in group.layer_names]
        return [spec] * len(group.layer_names)

    def _derive_v41_file_sizes(self, groups) -> dict[str, int]:
        sizes = {"FA": 0, "WA": 0}
        for group_id, group in enumerate(groups):
            if group_id in self.transient_group_ids:
                continue
            if group_id in self.fa_group_ids:
                label = "FA"
            elif group_id in self.window_group_ids:
                label = "WA"
            else:
                raise ValueError(
                    f"V4.1 persistent group {group_id} is neither FA nor WA."
                )
            payload = 0
            meta = self.group_metas[group_id]
            segment_tokens = meta.tail_tokens // meta.tail_blocks
            if segment_tokens <= 0 or meta.tail_blocks <= 0:
                raise ValueError(
                    f"V4.1 group {group_id} has invalid tail metadata: {meta}"
                )
            if getattr(self, "whole_page_sizes", None):
                if segment_tokens != meta.token_block_size:
                    raise ValueError("UCM V4.1 cannot store a partial padded page.")
                payload = sum(
                    self.whole_page_sizes[name]
                    for name in group.layer_names
                    if not getattr(self, "is_ascend_v41_layout", False)
                    or name in self.whole_page_sizes
                )
                sizes[label] += payload * meta.tail_blocks
                continue
            for spec in self._specs_for_group(group):
                page_size = next(
                    (
                        getattr(spec, name, None)
                        for name in (
                            "unpadded_page_size_bytes",
                            "real_page_size_bytes",
                            "page_size_bytes",
                        )
                        if getattr(spec, name, None) is not None
                    ),
                    None,
                )
                if page_size is None:
                    raise ValueError(
                        f"V4.1 group {group_id} layer spec {type(spec).__name__} "
                        "has no page-size byte attribute."
                    )
                spec_block_size = getattr(spec, "block_size", None)
                if not isinstance(spec_block_size, int) or spec_block_size <= 0:
                    raise ValueError(
                        f"V4.1 group {group_id} has invalid spec block size "
                        f"{spec_block_size!r}."
                    )
                numerator = int(page_size) * segment_tokens
                if numerator % spec_block_size:
                    raise ValueError(
                        f"V4.1 group {group_id} page payload {page_size} does "
                        f"not scale evenly for segment_tokens={segment_tokens} "
                        f"and spec.block_size={spec_block_size}."
                    )
                payload += numerator // spec_block_size
            sizes[label] += payload * meta.tail_blocks
        return {label: round_up(size, 4096) for label, size in sizes.items()}

    def _create_fa_store(
        self,
        group_layouts: Optional[dict[int, KVCacheGroupLayout]],
        cpu_affinity_cores: Optional[list[int]] = None,
    ) -> UcmKVStoreBaseV1:
        """Create the backing store used for full-attention rows."""

        tensor_size_list = None
        if self._role == KVConnectorRole.WORKER:
            if group_layouts is None:
                raise RuntimeError("Worker FA store needs layouts.")
            tensor_size_list = self._store_tensor_size_list(
                group_layouts,
                self.fa_group_ids,
            )
        return self._create_store(
            label="FA",
            store_suffix="fa",
            tensor_size_list=tensor_size_list,
            cpu_affinity_cores=cpu_affinity_cores,
        )

    def _create_wa_store(
        self,
        group_layouts: Optional[dict[int, KVCacheGroupLayout]],
        cpu_affinity_cores: Optional[list[int]] = None,
    ) -> Optional[UcmKVStoreBaseV1]:
        """Create the backing store used for window-tail rows."""

        if not self.window_group_ids:
            return None
        tensor_size_list = None
        if self._role == KVConnectorRole.WORKER:
            if group_layouts is None:
                raise RuntimeError("Worker WA store needs layouts.")
            tensor_size_list = self._store_tensor_size_list(
                group_layouts,
                self.window_group_ids,
            )
        return self._create_store(
            label="WA",
            store_suffix="wa",
            tensor_size_list=tensor_size_list,
            cpu_affinity_cores=cpu_affinity_cores,
        )

    def _base_store_config(
        self,
        store_suffix: str,
    ) -> tuple[str, Optional[str], dict[str, object]]:
        """Build a namespaced UCM store config for either FA or WA data."""

        if len(self.connector_configs) != 1:
            raise RuntimeError(
                f"Expected exactly one connector config, "
                f"but got {len(self.connector_configs)}: "
                f"{self.connector_configs}"
            )

        name = self.connector_configs[0]["ucm_connector_name"]
        module_path = self.connector_configs[0].get("ucm_connector_module_path", None)
        config = copy.deepcopy(self.connector_configs[0]["ucm_connector_config"])
        config.setdefault("store_pipeline", "Cache|Empty")
        config["tensor_layout"] = "hma"
        # MLA ranks share one logical store buffer; non-MLA stores are per rank.
        config.setdefault("share_buffer_enable", self.is_mla)
        if isinstance(config.get("storage_backends"), str):
            config["storage_backends"] = [
                path for path in config["storage_backends"].split(":")
            ]
        config["unique_id"] = f"{self.unique_id}_fawa_{store_suffix}"
        self._namespace_storage_backends(config, store_suffix)
        dp_rank = self._vllm_config.parallel_config.data_parallel_rank
        config["posix_gc_enable"] = (
            self._role != KVConnectorRole.WORKER and dp_rank == 0
        )
        if config.get("posix_capacity_gb", None) is not None:
            config["posix_capacity_gb"] = int(config["posix_capacity_gb"]) // 2
        return name, module_path, config

    def _set_default_shm_buffer_capacity(self, config: dict[str, object]) -> None:
        if not bool(config.get("share_buffer_enable", False)):
            return

        # HMA creates two shared-buffer stores, FA and WA, so split the
        # shared-buffer capacity evenly between them, whether user-set or the
        # 128GB direct-connector default.
        if config.get("cache_buffer_capacity_gb") is None:
            config["cache_buffer_capacity_gb"] = 128
        capacity = int(config["cache_buffer_capacity_gb"])
        config["cache_buffer_capacity_gb"] = max(capacity // 2, 1)
        logger.info(
            f"Set FAWA cache_buffer_capacity_gb to "
            f"{config['cache_buffer_capacity_gb']}GB by splitting the "
            f"{capacity}GB shared-buffer capacity across FA/WA stores."
        )
        # The shared buffer is allocated via shm_open in /dev/shm; fail early
        # (before store creation) if the tmpfs cannot hold the FA+WA total.
        _check_shm_capacity(capacity)

    @staticmethod
    def _namespace_storage_backends(
        config: dict[str, object],
        store_suffix: str,
    ) -> None:
        """Place FA and WA store files in separate backend subdirectories."""

        backends = config.get("storage_backends")
        if not isinstance(backends, list):
            return
        namespaced_backends: list[str] = []
        for backend in backends:
            if not os.path.isdir(str(backend)):
                raise FileNotFoundError(
                    f"Storage backend path does not exist: {backend}"
                )
            backend_path = os.path.join(str(backend), f"fawa_{store_suffix}")
            os.makedirs(backend_path, exist_ok=True)
            namespaced_backends.append(backend_path)
        config["storage_backends"] = namespaced_backends

    def _create_store(
        self,
        label: str,
        store_suffix: str,
        tensor_size_list: Optional[list[int]],
        cpu_affinity_cores: Optional[list[int]] = None,
    ) -> UcmKVStoreBaseV1:
        """Instantiate one UCM store with worker tensor layout metadata."""

        name, module_path, config = self._base_store_config(store_suffix)
        self._set_default_shm_buffer_capacity(config)
        if label == "FA":
            config.setdefault("cache_io_aggregation", True)
        else:
            config["cache_io_aggregation"] = False
        if self._role == KVConnectorRole.WORKER:
            if tensor_size_list is None:
                raise RuntimeError(f"Worker FAWA {label} store needs tensor sizes.")
            config["device_id"] = self.device_id
            if self.is_v41_layout:
                # CacheStore prefers a scalar tensor_size over tensor_size_list.
                # Packed V4.1 pages need their per-segment sizes preserved.
                config.pop("tensor_size", None)
            config["tensor_size_list"] = tensor_size_list
            # io_direct requires shard and block sizes to be 4KB aligned.
            aligned_size = 4096
            padded_size = round_up(sum(tensor_size_list), aligned_size)
            config["shard_size"] = padded_size
            config["block_size"] = padded_size
            if self.file_size[label] != padded_size and self.is_v41_layout:
                raise ValueError(
                    f"V4.1 FAWA {label} payload mismatch: scheduler derived "
                    f"{self.file_size[label]} bytes, worker layout padded to "
                    f"{padded_size} bytes from tensor_size_list={tensor_size_list}."
                )
            if self.file_size[label] != padded_size and not self.is_v41_layout:
                logger.info_once(
                    f"GC file size of {label} does not match real file size. "
                    f"Worker: {padded_size}, Scheduler: {self.file_size[label]}"
                )
            gc_size = _get_store_gc_block_size(
                str(config.get("store_pipeline", "")),
                tensor_size_list,
                padded_size,
                padded_size,
            )
            # Count load/save metrics with the same per-record size as GC.
            self.file_size[label] = gc_size
            self._publish_block_size(gc_size, store_suffix=store_suffix)
            # MLA stores aggregate TP shards under one logical rank group.
            config["local_rank_size"] = self.tp_size if self.is_mla else 1
            if cpu_affinity_cores:
                config["cpu_affinity_cores"] = list(cpu_affinity_cores)
        else:
            if self._gc_owner:
                # EngineCore finishes worker KV cache registration (which
                # publishes this size) synchronously before it creates the
                # scheduler. The file is keyed by the worker's ppid, which
                # equals the EngineCore pid only for the multiproc executor;
                # Ray and external_launcher keep the layer-count estimate.
                size = _scheduler_read_block_size(
                    store_suffix=store_suffix,
                )
                if size is not None:
                    self.file_size[label] = size
                else:
                    logger.warning(
                        f"FAWA {label} worker-published record size not found "
                        "(expected with Ray or external_launcher); falling "
                        f"back to estimated block_size {self.file_size[label]}."
                    )
            config["block_size"] = self.file_size[label]
        logger.info(
            f"create FAWA {label} {name} with config: "
            f"{self._summarize_store_config(config)}"
        )
        return UcmConnectorFactoryV1.create_connector(name, config, module_path)

    @staticmethod
    def _summarize_store_config(config: dict[str, object]) -> dict[str, object]:
        """Return a log-friendly store config without dumping large size lists."""

        summary = dict(config)
        # CacheStore initialization logs the runtime IO aggregation configuration.
        summary.pop("cache_io_aggregation", None)
        tensor_size_list = summary.pop("tensor_size_list", None)
        if tensor_size_list is not None:
            tensor_sizes = [int(size) for size in tensor_size_list]
            summary["tensor_count"] = len(tensor_sizes)
            summary["tensor_bytes"] = sum(tensor_sizes)
        return summary

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        self.kv_caches = kv_caches
        self.device = create_device()

        enable_affinity = _use_ucm_connector_cpu_affinity()
        worker_cores, store_cores = (
            self.device.split_cores(self.device_id) if enable_affinity else (None, None)
        )

        standardized_layers = {
            layer
            for raw_tensor in self._kv_cache_config.kv_cache_tensors
            for layer in (getattr(raw_tensor, "layers", None) or ())
        }
        for group_id, group_spec in enumerate(self._kv_cache_config.kv_cache_groups):
            if group_id in self.transient_group_ids:
                continue
            group_caches: dict[str, torch.Tensor] = {}
            for layer_name in group_spec.layer_names:
                if (
                    self.is_ascend_v41_layout
                    and layer_name not in self.whole_page_sizes
                ):
                    continue
                if isinstance(kv_caches[layer_name], torch.Tensor):
                    group_caches[layer_name] = kv_caches[layer_name]
                else:
                    group_caches[layer_name] = tuple(kv_caches[layer_name])
            standardized_layout = not self.is_ascend_v41_layout and any(
                layer_name in standardized_layers
                for layer_name in group_spec.layer_names
            )
            layout = KVCacheGroupLayout(
                group_caches,
                is_ascend_layout=self.is_ascend_layout,
                expected_block_size=(
                    self.ascend_block_geometry(group_spec.kv_cache_spec).physical
                    if self.is_ascend_layout and not self.is_ascend_v41_layout
                    else group_spec.kv_cache_spec.block_size
                ),
                standardized_layout=standardized_layout,
                expected_tokens_per_state=1,
                expected_tokens_per_state_by_layer=(
                    _group_tokens_per_state_by_layer(group_spec)
                    if standardized_layout
                    else None
                ),
                physical_layout_by_layer=(
                    _group_physical_layout_by_layer(group_spec)
                    if standardized_layout
                    else None
                ),
                whole_page_sizes_by_layer=(
                    {name: self.whole_page_sizes[name] for name in group_caches}
                    if self.whole_page_sizes
                    else None
                ),
            )
            self.group_layouts[group_id] = layout

        self.store = self._create_fa_store(self.group_layouts, store_cores)
        self.fa_store = self.store
        self.wa_store = self._create_wa_store(self.group_layouts, store_cores)

        if worker_cores:
            try:
                os.sched_setaffinity(0, worker_cores)
                logger.info(f"[VLLM CPU Affinity] Worker bound to cores {worker_cores}")
            except Exception as e:
                logger.warning(f"Failed to bind worker: {e}")

    def _store_tensor_size_list(
        self,
        group_layouts: dict[int, KVCacheGroupLayout],
        group_ids: tuple[int, ...],
    ) -> list[int]:
        """Build the per-tensor byte-size vector expected by UCM stores."""

        tensor_size_list: list[int] = []
        for group_id in group_ids:
            layout = group_layouts.get(group_id)
            if layout is None:
                continue
            meta = self.group_metas[group_id]

            if not meta.tail_tokens:
                continue

            segment_tokens = meta.tail_tokens // meta.tail_blocks

            for _ in range(meta.tail_blocks):
                segment_sizes = layout.segment_tensor_size_list(
                    segment_tokens,
                    meta.token_block_size,
                )
                tensor_size_list.extend(segment_sizes)
        if not tensor_size_list:
            group_label = (
                "FA"
                if group_ids == self.fa_group_ids
                else "WA" if group_ids == self.window_group_ids else str(group_ids)
            )
            raise RuntimeError(f"Worker FAWA {group_label} layout is empty.")
        return tensor_size_list

    def _lookup_external_hit_blocks(self, external_keys: list[bytes]) -> int:
        """Find the longest reusable prefix present in both FA and WA stores."""

        if self.fa_store is None:
            raise RuntimeError("FA store is not initialized.")
        if self.window_group_ids and self.wa_store is None:
            raise RuntimeError("WA store is not initialized.")
        fa_hit_blocks = (
            self._rank_consistency.lookup_on_prefix(self.fa_store, external_keys) + 1
        )
        if fa_hit_blocks <= 0:
            return 0
        if not self.window_group_ids:
            return fa_hit_blocks

        # WA rows represent window boundary state, so they are not required to
        # form a prefix. Search only inside the FA-contiguous hit range and use
        # the latest boundary that exists.
        wa_keys = external_keys[:fa_hit_blocks]
        reverse_idx = self._rank_consistency.lookup_on_reverse(self.wa_store, wa_keys)
        if reverse_idx < 0:
            return 0
        return reverse_idx + 1

    def _prefetch_hit_key_hotness(
        self,
        fa_hbm_hit_keys: list[bytes],
        all_hit_keys: list[bytes],
    ) -> None:
        """Best-effort GC hotness update for FAWA's two backing stores.

        External FA keys are already touched by ``lookup_on_prefix``, so the
        FA store only needs the local-HBM prefix that lookup did not inspect.
        WA uses ``lookup_on_reverse``, which touches only the selected boundary;
        update every reusable boundary key so sparse WA snapshots age together
        with the corresponding FA prefix. Missing sparse WA files are harmless:
        store ``prefetch`` is a hint and Posix simply ignores a failed ``utime``.
        """

        if self.fa_store is None:
            raise RuntimeError("FA store is not initialized.")
        if self.window_group_ids and self.wa_store is None:
            raise RuntimeError("WA store is not initialized.")

        updates = (
            ("FA", self.fa_store, fa_hbm_hit_keys),
            ("WA", self.wa_store, all_hit_keys),
        )
        for label, store, keys in updates:
            if store is None or not keys:
                continue
            try:
                store.prefetch(keys)
            except Exception as e:
                # Prefetch is only a GC hotness hint. A failure must not turn a
                # valid cache hit into a scheduler-side miss.
                logger.warning(
                    f"FAWA {label} hotness update failed. " f"{type(e).__name__}: {e}"
                )

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        wa_hbm_hit_block_num = num_computed_tokens // self.hash_block_size
        wa_computed_tokens = wa_hbm_hit_block_num * self.hash_block_size

        if request.num_tokens <= self.persist_token_threshold:
            return 0, False
        skip_external_load = request.num_tokens <= (
            wa_computed_tokens + self.load_tokens_threshold
        )
        if skip_external_load and wa_hbm_hit_block_num <= 0:
            return 0, False

        assert self.request_block_hasher is not None
        try:
            canonical_hashes = self.request_block_hasher(request)
        except Exception as e:
            logger.error(
                f"request {request.request_id} hash error. " f"{type(e).__name__}: {e}"
            )
            return 0, False
        fa_hbm_hit_keys = canonical_hashes[:wa_hbm_hit_block_num]

        # Even when no external load is worthwhile, the local-HBM prefix is a
        # cache hit and should remain hot in both FAWA backing stores.
        if skip_external_load:
            self._prefetch_hit_key_hotness(fa_hbm_hit_keys, fa_hbm_hit_keys)
            return 0, False

        # A full prompt hit must leave its final complete hash block for the
        # model to compute.  The corresponding WA row is a snapshot *after*
        # that block; loading it and resuming inside the block would require
        # the transient compressor ring state that is intentionally not
        # persisted.  For a partial final block, every canonical hash ends
        # strictly before EOS and remains eligible for reuse.
        eligible_block_count = (
            (request.num_tokens - 1) // self.hash_block_size
            if self.is_v41_layout
            else len(canonical_hashes)
        )
        external_keys = canonical_hashes[wa_hbm_hit_block_num:eligible_block_count]
        if not external_keys:
            self._prefetch_hit_key_hotness(fa_hbm_hit_keys, fa_hbm_hit_keys)
            return 0, False

        external_hit_blocks = 0
        if external_keys:
            try:
                external_hit_blocks = self._lookup_external_hit_blocks(external_keys)
            except Exception as e:
                logger.error(
                    f"request {request.request_id} FAWA lookup error. "
                    f"{type(e).__name__}: {e}"
                )
                self._record_counter("connector_lookup_errors_total")

        total_hit_block_num = wa_hbm_hit_block_num + external_hit_blocks
        if (
            getattr(self, "prefix_replay_tokens", 0)
            and total_hit_block_num * self.hash_block_size <= self.prefix_replay_tokens
        ):
            # The scheduler discards hits that would be replayed in full.
            wa_hbm_hit_block_num = external_hit_blocks = total_hit_block_num = 0
            wa_computed_tokens = 0
        self._prefetch_hit_key_hotness(
            fa_hbm_hit_keys,
            canonical_hashes[:total_hit_block_num],
        )
        num_total_hit_tokens = (
            external_hit_blocks * self.hash_block_size + wa_computed_tokens
        )
        external_hit_tokens = max(0, num_total_hit_tokens - num_computed_tokens)
        if not self.is_v41_layout and num_total_hit_tokens == request.num_tokens:
            external_hit_tokens = max(0, external_hit_tokens - 1)

        if external_hit_blocks * self.hash_block_size <= self.load_tokens_threshold:
            external_hit_tokens = 0
            num_total_hit_tokens = num_computed_tokens
            # let wa_hbm_hit_block_num equal to total_hit_block_num,so no need to load external blocks
            wa_hbm_hit_block_num += external_hit_blocks

        # TODO :for HMA, vllm should offer all kv group's prefix block hits，so that more FA blocks can be reused
        self.requests_meta[request.request_id] = FAWARequestMeta(
            ucm_block_ids=canonical_hashes,
            hbm_hit_block_num=wa_hbm_hit_block_num,
            total_hit_block_num=total_hit_block_num,
            num_token_ids=request.num_tokens,
            token_processed=num_total_hit_tokens,
        )
        logger.info_once(
            f"FAWA request_id: {request.request_id}, "
            f"total tokens: {request.num_tokens}, "
            f"hit hbm tokens: {num_computed_tokens}, "
            f"hit external tokens: {external_hit_tokens}, "
            f"load blocks: {total_hit_block_num - wa_hbm_hit_block_num}, "
            f"dump blocks: {len(canonical_hashes) - total_hit_block_num}, "
        )
        return external_hit_tokens, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        pass

    def _slice_group_block_ids(
        self,
        group_id: int,
        group_block_ids: list[int],
        window_boundary_token_idx: np.ndarray,
        fetch_wa_block_wise: bool,
    ) -> list[int]:
        """Select the physical group blocks needed for FA or WA store rows."""

        is_window_group = group_id in self.window_group_ids
        if group_id in self.transient_group_ids:
            return []
        group_meta = self.group_metas[group_id]
        if is_window_group:
            if not group_meta.tail_tokens:
                return []
            if fetch_wa_block_wise:
                # Block-wise WA stores one tail row for each canonical boundary.
                boundary_block_indices = (
                    window_boundary_token_idx // group_meta.token_block_size
                )
                offsets = np.arange(group_meta.tail_blocks - 1, -1, -1, dtype=np.int64)
                boundary_block_indices = (
                    boundary_block_indices[:, None] - offsets[None, :]
                )
                if boundary_block_indices.max(initial=-1) >= len(group_block_ids):
                    raise RuntimeError(
                        "FAWA window load exceeds the available request blocks; "
                        "refusing an invalid WA slice."
                    )
                # Pad short snapshots with identical copies of the first block;
                # loading repeats the same bytes into the same destination.
                boundary_block_indices = np.maximum(boundary_block_indices, 0)
                return np.array(group_block_ids)[
                    boundary_block_indices.flatten()
                ].tolist()
            else:
                # Chunk-wise WA stores only the tail for the final boundary.
                boundary_block_idx = (
                    window_boundary_token_idx[-1] // group_meta.token_block_size
                ) + 1
                if boundary_block_idx > len(group_block_ids):
                    raise RuntimeError(
                        "FAWA window load exceeds the available request blocks; "
                        "refusing an invalid WA slice."
                    )
                indices = np.arange(
                    boundary_block_idx - group_meta.tail_blocks,
                    boundary_block_idx,
                    dtype=np.int64,
                )
                indices = np.maximum(indices, 0)
                return np.array(group_block_ids)[indices].tolist()
        # FA rows map each canonical hash block to its containing group block.
        return np.array(group_block_ids)[
            window_boundary_token_idx // group_meta.token_block_size
        ].tolist()

    def _generate_dispatch_meta(
        self,
        req_meta: FAWARequestMeta,
        new_tokens: int,
        new_vllm_block_ids: tuple[list[int], ...],
        need_load: bool = True,
        computed_tokens: Optional[int] = None,
    ) -> FAWARequestDispatchMeta:
        """Build one request's worker-side load and dump plan.

        Canonical hash blocks are split into:

        - `[0, hbm_hit_block_num)`: already resident in local HBM.
        - `[hbm_hit_block_num, total_hit_block_num)`: external hit to load.
        - `[token_processed, token_processed + new_tokens)`: newly computed
          tokens whose complete canonical blocks should be dumped.

        `new_vllm_block_ids` is appended to the accumulated per-group block
        rows before slicing FA/WA rows for this step.
        """

        if not req_meta.vllm_block_ids:
            req_meta.vllm_block_ids = tuple([] for _ in self.group_metas)
        if len(new_vllm_block_ids) != len(req_meta.vllm_block_ids):
            raise RuntimeError(
                f"FAWA dispatch metadata expected {len(req_meta.vllm_block_ids)} "
                f"KV cache groups, got {len(new_vllm_block_ids)}."
            )
        for group_id, block_ids in enumerate(new_vllm_block_ids):
            req_meta.vllm_block_ids[group_id].extend(block_ids)

        all_group_block_ids = req_meta.vllm_block_ids
        load_block_keys: list[bytes] = []
        load_start, load_end = 0, 0
        load_vllm_block_ids: list[list[int]] = []
        if need_load and req_meta.total_hit_block_num > req_meta.hbm_hit_block_num:
            load_start = req_meta.hbm_hit_block_num
            load_end = req_meta.total_hit_block_num
            load_block_keys = req_meta.ucm_block_ids[load_start:load_end]
            window_boundary_token_idx = (
                np.arange(load_start + 1, load_end + 1) * self.hash_block_size - 1
            )
            for group_id, group_block_ids in enumerate(all_group_block_ids):
                load_vllm_block_ids.append(
                    self._slice_group_block_ids(
                        group_id,
                        group_block_ids,
                        window_boundary_token_idx,
                        fetch_wa_block_wise=False,  # always fetch the full WA tail on load to simplify logic
                    )
                )

        computed_start_token = (
            req_meta.token_processed if computed_tokens is None else computed_tokens
        )
        computed_end_token = min(
            req_meta.num_token_ids,
            computed_start_token + new_tokens,
        )
        dump_start = max(
            req_meta.total_hit_block_num,
            computed_start_token // self.hash_block_size,
        )
        dump_end = computed_end_token // self.hash_block_size
        dump_block_keys: list[bytes] = []
        dump_vllm_block_ids: list[list[int]] = []
        if dump_end > dump_start:
            dump_block_keys = req_meta.ucm_block_ids[dump_start:dump_end]
            window_boundary_token_idx = (
                np.arange(dump_start + 1, dump_end + 1) * self.hash_block_size - 1
            )
            for group_id, group_block_ids in enumerate(all_group_block_ids):
                dump_vllm_block_ids.append(
                    self._slice_group_block_ids(
                        group_id,
                        group_block_ids,
                        window_boundary_token_idx,
                        fetch_wa_block_wise=self.wa_dump_block_wise,
                    )
                )
        req_meta.token_processed = computed_end_token

        return FAWARequestDispatchMeta(
            load_keys=load_block_keys,
            load_hash_start=load_start,
            load_hash_end=load_end,
            load_vllm_block_ids=tuple(load_vllm_block_ids),
            dump_keys=dump_block_keys,
            dump_hash_start=dump_start,
            dump_hash_end=dump_end,
            dump_vllm_block_ids=tuple(dump_vllm_block_ids),
        )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> UCMFAWAConnectorMetadata:
        requests_dispatch_meta: dict[str, FAWARequestDispatchMeta] = {}
        # New requests may need both external-prefix load and new-block dump.
        for request in scheduler_output.scheduled_new_reqs:
            request_id, vllm_block_ids = request.req_id, request.block_ids
            req_meta = self.requests_meta.get(request_id)
            if req_meta:
                requests_dispatch_meta[request_id] = self._generate_dispatch_meta(
                    req_meta,
                    scheduler_output.num_scheduled_tokens[request_id],
                    tuple(vllm_block_ids),
                    computed_tokens=(
                        request.num_computed_tokens
                        if self.prefix_replay_tokens
                        else None
                    ),
                )

        scheduled_cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, request_id in enumerate(scheduled_cached_reqs.req_ids):
            req_meta = self.requests_meta.get(request_id)
            if req_meta:
                new_block_ids = scheduled_cached_reqs.new_block_ids[i]
                if new_block_ids is None:
                    new_block_ids = tuple([] for _ in self.group_metas)
                else:
                    new_block_ids = tuple(new_block_ids)
                if hasattr(scheduled_cached_reqs, "resumed_from_preemption"):
                    resumed_from_preemption = (
                        scheduled_cached_reqs.resumed_from_preemption[i]
                    )
                else:
                    resumed_from_preemption = (
                        request_id in scheduled_cached_reqs.resumed_req_ids
                    )
                if resumed_from_preemption:
                    req_meta.vllm_block_ids = tuple([] for _ in self.group_metas)
                requests_dispatch_meta[request_id] = self._generate_dispatch_meta(
                    req_meta,
                    scheduler_output.num_scheduled_tokens[request_id],
                    new_block_ids,
                    need_load=resumed_from_preemption,
                    computed_tokens=(
                        scheduled_cached_reqs.num_computed_tokens[i]
                        if self.prefix_replay_tokens
                        else None
                    ),
                )

        for request_id in scheduler_output.finished_req_ids:
            self.requests_meta.pop(request_id, None)

        preempted_req_ids = set(scheduler_output.preempted_req_ids or ())
        return UCMFAWAConnectorMetadata(requests_dispatch_meta, preempted_req_ids)

    def _submit_load_task(
        self,
        request_id: str,
        label: str,
        store: UcmKVStoreBaseV1,
        keys: list[bytes],
        ptrs: np.ndarray,
    ) -> FAWALoadTask:
        """Submit one store load and retain block ids for failure reporting."""

        shard_indices = [0] * len(keys)
        task = self._rank_consistency.submit_load(
            store,
            {request_id: keys},
            keys,
            shard_indices,
            ptrs,
        )
        return FAWALoadTask(
            request_id=request_id,
            label=label,
            store=store,
            task=task,
            key_count=len(keys),
        )

    def _handle_load_err(self, request_id: str):
        affected_block_ids = self._get_request_all_block_ids(request_id)
        self._record_load_error(
            "connector_load_wait_errors_total",
            affected_block_ids,
        )
        self._connector_worker_meta.mark_failed(request_id)

    def _wait_load_task(
        self,
        load_task: FAWALoadTask,
    ) -> bool:
        """Wait a load task and mark its anchor blocks invalid on failure."""

        try:
            self._rank_consistency.wait_load(load_task.task)
        except Exception as e:
            logger.error(
                f"request {load_task.request_id} wait FAWA load "
                f"task label={load_task.label} error. {type(e).__name__}: {e}"
            )
            self._handle_load_err(load_task.request_id)
            return False
        return True

    def get_block_ids_with_load_errors(self) -> set[int]:
        res = self._invalid_block_ids
        self._invalid_block_ids = set()
        return res

    def _submit_dump_task(
        self,
        label: str,
        store: UcmKVStoreBaseV1,
        keys: list[bytes],
        ptrs: np.ndarray,
        event_handle: int,
        block_ids_by_request: dict[str, set[bytes]],
    ) -> FAWADumpTask:
        """Submit one store dump for FA or WA rows."""

        shard_indices = [0] * len(keys)
        task = self._rank_consistency.submit_dump(
            store,
            block_ids_by_request,
            keys,
            shard_indices,
            ptrs,
            event_handle,
        )
        return FAWADumpTask(
            label=label,
            store=store,
            task=task,
            key_count=len(keys),
            event_handle=event_handle,
        )

    def _extract_fa_ptr(self, store_keys, hash_start, hash_end, candidate_vllm_ids):
        """Build store pointer rows for full-attention cache segments."""

        all_ptrs = []
        for group_id in self.fa_group_ids:
            layout = self.group_layouts.get(group_id)
            if layout is None:
                continue
            meta = self.group_metas[group_id]
            block_ids = np.asarray(candidate_vllm_ids[group_id], dtype=np.uint64)
            # GPU layouts usually use one tensor block per hash block. Ascend
            # layouts may pack several canonical hash blocks in one tensor
            # block, so the row starts at a token offset inside the block.
            if self.hash_block_size == meta.token_block_size:
                group_ptrs = layout.extract_addrs(block_ids)
            else:
                token_start = np.arange(hash_start, hash_end) * self.hash_block_size
                token_offsets = token_start % meta.token_block_size
                group_ptrs = layout.extract_addrs_with_offsets(
                    block_ids, meta.token_block_size, token_offsets
                )
            all_ptrs.append(group_ptrs)

        return np.concatenate(all_ptrs, axis=1)

    def _extract_wa_ptr(self, store_keys, vllm_ids):
        """Build store pointer rows for window-attention tail segments."""

        all_ptrs = []
        for group_id in self.window_group_ids:
            layout = self.group_layouts.get(group_id)
            if layout is None:
                continue
            meta = self.group_metas[group_id]
            if not meta.tail_tokens:
                continue

            block_ids = np.asarray(vllm_ids[group_id], dtype=np.uint64)
            if meta.tail_blocks == 1 and meta.token_block_size > meta.tail_tokens:
                # A short tail stored inside a larger group block starts near
                # the end of the physical tensor block.
                token_offsets = np.ones_like(block_ids) * (
                    meta.token_block_size - meta.tail_tokens
                )
                group_ptrs = layout.extract_addrs_with_offsets(
                    block_ids, meta.token_block_size, token_offsets
                )
            else:
                token_offsets = np.zeros_like(block_ids)
                group_ptrs = layout.extract_addrs(block_ids)
                # Multi-block WA tails are flattened into one store row per
                # canonical boundary key.
                group_ptrs = group_ptrs.reshape(len(store_keys), -1)

            all_ptrs.append(group_ptrs)

        return np.concatenate(all_ptrs, axis=1)

    def _get_request_all_block_ids(self, request_id: str) -> set[int]:
        """Get all VLLM block ids referenced by one request's load and dump plan."""
        metadata = self._get_connector_metadata()
        request = metadata.request_meta.get(request_id, None)
        if request is None:
            return set()
        all_group_vllm_block_ids = {
            block_id
            for block_ids in request.load_vllm_block_ids
            for block_id in block_ids
        }
        all_group_vllm_block_ids |= {
            block_id
            for block_ids in request.dump_vllm_block_ids
            for block_id in block_ids
        }
        return all_group_vllm_block_ids

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, UCMFAWAConnectorMetadata):
            raise RuntimeError(f"Unexpected FAWA metadata type: {type(metadata)}")

        if self.is_v41_layout and any(
            request.load_keys for request in metadata.request_meta.values()
        ):
            # Finish pending KV page writes before the store starts its H2D copies.
            self.device.synchronize()
            logger.info_once("V4.1 compute stream synchronized before external KV load")

        tasks: list[FAWALoadTask] = []
        for request_id, request in metadata.request_meta.items():
            if not request.load_keys:
                continue

            try:
                if self.fa_store is None:
                    raise RuntimeError("FA store is not initialized.")
                if self.window_group_ids and self.wa_store is None:
                    raise RuntimeError("WA store is not initialized.")

                # FA groups are loaded for every external-hit canonical block.
                fa_ptrs = self._extract_fa_ptr(
                    request.load_keys,
                    request.load_hash_start,
                    request.load_hash_end,
                    request.load_vllm_block_ids,
                )
                fa_task = self._submit_load_task(
                    request_id,
                    "FA",
                    self.fa_store,
                    request.load_keys,
                    fa_ptrs,
                )
                tasks.append(fa_task)

                if not self.window_group_ids:
                    continue
                # WA groups only need the final matched boundary.
                window_keys = request.load_keys[-1:]
                window_ptrs = self._extract_wa_ptr(
                    window_keys,
                    request.load_vllm_block_ids,
                )
                wa_task = self._submit_load_task(
                    request_id,
                    "WA",
                    self.wa_store,
                    window_keys,
                    window_ptrs,
                )
                tasks.append(wa_task)
            except Exception as e:
                logger.error(
                    f"request {request_id} submit FAWA load task "
                    f"error. {type(e).__name__}: {e}"
                )
                self._handle_load_err(request_id)

        self._wait_all_load_task(tasks)

    def _wait_all_load_task(self, tasks: list[FAWALoadTask]):
        load_bytes = 0
        for load_task in tasks:
            if self._wait_load_task(load_task):
                load_bytes += load_task.key_count * self.file_size[load_task.label]
        if tasks:
            ucmmetrics.update_stats({"load_bytes_total": load_bytes})

    def wait_for_save(self) -> None:
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, UCMFAWAConnectorMetadata):
            raise RuntimeError(f"Unexpected FAWA metadata type: {type(metadata)}")

        if self.fa_store is None:
            raise RuntimeError("FA store is not initialized.")
        if self.window_group_ids and self.wa_store is None:
            raise RuntimeError("WA store is not initialized.")

        self._poll_completed_dump_tasks()

        fa_dump_keys: list[bytes] = []
        wa_dump_keys: list[bytes] = []
        fa_ptr_rows: list[np.ndarray] = []
        wa_ptr_rows: list[np.ndarray] = []
        dump_request_ids: tuple[str] = ()
        fa_dump_blocks_by_request: dict[str, set[bytes]] = {}
        wa_dump_blocks_by_request: dict[str, set[bytes]] = {}
        save_bytes = 0
        if self.tp_size > 1:
            # Split FA rows by canonical block index. Block-wise WA follows the same
            # TP key slice; chunk-wise WA assigns one final boundary per request.
            wa_dump_ring_idx = 0
            for request_id, request in metadata.request_meta.items():
                if not request.dump_keys:
                    continue
                dump_request_ids += (request_id,)
                num_keys = len(request.dump_keys)
                tp_block_start = num_keys * self.tp_rank // self.tp_size
                tp_block_end = num_keys * (self.tp_rank + 1) // self.tp_size
                tp_dump_keys = request.dump_keys[tp_block_start:tp_block_end]
                if tp_dump_keys:
                    fa_dump_blocks_by_request[request_id] = set(tp_dump_keys)
                    fa_dump_vllm_block_ids = tuple(
                        (
                            group_block_ids[tp_block_start:tp_block_end]
                            if group_id in self.fa_group_ids
                            else group_block_ids
                        )
                        for group_id, group_block_ids in enumerate(
                            request.dump_vllm_block_ids
                        )
                    )

                    fa_dump_keys.extend(tp_dump_keys)
                    fa_ptr_rows.append(
                        self._extract_fa_ptr(
                            tp_dump_keys,
                            request.dump_hash_start + tp_block_start,
                            request.dump_hash_start + tp_block_end,
                            fa_dump_vllm_block_ids,
                        )
                    )
                if not self.window_group_ids:
                    continue
                if self.wa_dump_block_wise:
                    if tp_dump_keys:
                        wa_dump_blocks_by_request[request_id] = set(tp_dump_keys)
                        wa_dump_vllm_block_ids = tuple(
                            (
                                group_block_ids[
                                    tp_block_start
                                    * self.group_metas[
                                        group_id
                                    ].tail_blocks : tp_block_end
                                    * self.group_metas[group_id].tail_blocks
                                ]
                                if group_id in self.window_group_ids
                                else group_block_ids
                            )
                            for group_id, group_block_ids in enumerate(
                                request.dump_vllm_block_ids
                            )
                        )
                        wa_dump_keys.extend(tp_dump_keys)
                        wa_ptr_rows.append(
                            self._extract_wa_ptr(
                                tp_dump_keys,
                                wa_dump_vllm_block_ids,
                            )
                        )
                elif wa_dump_ring_idx % self.tp_size == self.tp_rank:
                    request_wa_dump_keys = request.dump_keys[-1:]
                    wa_dump_blocks_by_request[request_id] = set(request_wa_dump_keys)
                    wa_dump_keys.extend(request_wa_dump_keys)
                    wa_ptr_rows.append(
                        self._extract_wa_ptr(
                            request_wa_dump_keys,
                            request.dump_vllm_block_ids,
                        )
                    )
                wa_dump_ring_idx += 1
        else:
            for request_id, request in metadata.request_meta.items():
                if not request.dump_keys:
                    continue
                dump_request_ids += (request_id,)
                fa_dump_blocks_by_request[request_id] = set(request.dump_keys)
                fa_dump_keys.extend(request.dump_keys)
                fa_ptr_rows.append(
                    self._extract_fa_ptr(
                        request.dump_keys,
                        request.dump_hash_start,
                        request.dump_hash_end,
                        request.dump_vllm_block_ids,
                    )
                )
                if not self.window_group_ids:
                    continue
                if self.wa_dump_block_wise:
                    wa_dump_blocks_by_request[request_id] = set(request.dump_keys)
                    wa_dump_keys.extend(request.dump_keys)
                    wa_ptr_rows.append(
                        self._extract_wa_ptr(
                            request.dump_keys,
                            request.dump_vllm_block_ids,
                        )
                    )
                else:
                    request_wa_dump_keys = request.dump_keys[-1:]
                    wa_dump_blocks_by_request[request_id] = set(request_wa_dump_keys)
                    wa_dump_keys.extend(request_wa_dump_keys)
                    wa_ptr_rows.append(
                        self._extract_wa_ptr(
                            request_wa_dump_keys,
                            request.dump_vllm_block_ids,
                        )
                    )

        if fa_dump_keys:
            event_handle = self._get_dump_event_handle()
            fa_ptrs = np.vstack(fa_ptr_rows)
            try:
                fa_dump_task = self._submit_dump_task(
                    "FA",
                    self.fa_store,
                    fa_dump_keys,
                    fa_ptrs,
                    event_handle,
                    fa_dump_blocks_by_request,
                )
                self.tp_dump_tasks.setdefault(dump_request_ids, []).append(fa_dump_task)
                save_bytes += fa_dump_task.key_count * self.file_size["FA"]
            except Exception as e:
                self.device.destroy_event_handle(event_handle)
                logger.error(f"dump FAWA kv cache failed. {type(e).__name__}: {e}")
                self._record_counter("connector_dump_submit_errors_total")
        if wa_dump_keys:
            event_handle = self._get_dump_event_handle()
            window_ptrs = np.vstack(wa_ptr_rows)
            try:
                wa_dump_task = self._submit_dump_task(
                    "WA",
                    self.wa_store,
                    wa_dump_keys,
                    window_ptrs,
                    event_handle,
                    wa_dump_blocks_by_request,
                )
                self.tp_dump_tasks.setdefault(dump_request_ids, []).append(wa_dump_task)
                save_bytes += wa_dump_task.key_count * self.file_size["WA"]
            except Exception as e:
                self.device.destroy_event_handle(event_handle)
                logger.error(f"dump FAWA WA kv cache failed. {type(e).__name__}: {e}")
                self._record_counter("connector_dump_submit_errors_total")
        if fa_dump_keys or wa_dump_keys:
            ucmmetrics.update_stats({"save_bytes_total": save_bytes})

    def _poll_completed_dump_tasks(self) -> None:
        """Reap completed FAWA dump tasks without waiting for in-flight tasks."""

        for request_ids, dump_tasks in list(self.tp_dump_tasks.items()):
            in_flight_tasks = []
            for dump_task in dump_tasks:
                task_finished = False

                try:
                    task_finished = dump_task.store.check(dump_task.task)
                except Exception as e:
                    logger.error(
                        "Check FAWA dump task failed; external cache may miss. "
                        f"label={dump_task.label}, keys={dump_task.key_count}, "
                        f"{type(e).__name__}: {e}"
                    )
                    in_flight_tasks.append(dump_task)
                    continue

                if task_finished:
                    try:
                        self._rank_consistency.wait_dump(dump_task.task)
                    except Exception as e:
                        logger.error(
                            "Best-effort FAWA dump task failed; external cache may miss. "
                            f"label={dump_task.label}, keys={dump_task.key_count}, "
                            f"{type(e).__name__}: {e}"
                        )
                    finally:
                        self.device.destroy_event_handle(dump_task.event_handle)
                else:
                    in_flight_tasks.append(dump_task)

            if in_flight_tasks:
                self.tp_dump_tasks[request_ids] = in_flight_tasks
            else:
                self.tp_dump_tasks.pop(request_ids, None)

    def _drain_best_effort_dump_tasks(
        self, request_ids: Optional[set[str]] = None
    ) -> None:
        """Best-effort wait for FAWA dump tasks.

        Dump failures only mean the external cache may miss later. They must not
        block vLLM from releasing HBM blocks, so failed tasks are logged and then
        removed from tracking. If ``request_ids`` is None, drain every pending
        task; otherwise drain only task batches containing one of those requests.
        """
        if request_ids is not None and not request_ids:
            return

        finished_chunk_req_ids = []
        for chunk_request_ids, dump_tasks in self.tp_dump_tasks.items():
            if request_ids is None or request_ids.intersection(chunk_request_ids):
                finished_chunk_req_ids.append(chunk_request_ids)
                for dump_task in dump_tasks:
                    try:
                        self._rank_consistency.wait_dump(dump_task.task)
                    except Exception as e:
                        logger.error(
                            "Best-effort FAWA dump task failed; external cache may miss. "
                            f"label={dump_task.label}, keys={dump_task.key_count}, "
                            f"{type(e).__name__}: {e}"
                        )
                        self._record_counter("connector_dump_wait_errors_total")
                    finally:
                        self.device.destroy_event_handle(dump_task.event_handle)

        for request_ids in finished_chunk_req_ids:
            self.tp_dump_tasks.pop(request_ids, None)

    def handle_preemptions(self, kv_connector_metadata: UCMFAWAConnectorMetadata):
        # Worker side method
        # This hook runs before the next load/forward can overwrite cache data.
        # Drain all previous-step dumps, not only explicitly preempted requests:
        # a running request's sliding-window blocks may also be recycled by the
        # next allocate_slots() call.
        self._drain_best_effort_dump_tasks()

    def get_finished(
        self,
        finished_req_ids: set[str],
    ) -> tuple[set[str] | None, set[str] | None]:
        # Worker side method
        self._drain_best_effort_dump_tasks(finished_req_ids)
        self._rank_consistency.finish_dump(finished_req_ids)
        return finished_req_ids, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, object] | None]:
        # Scheduler side method
        return True, None


class UCMFAWALiteConnector(UCMLiteConnector):
    """UCM Lite connector for DS V4 (FAWA) models.

    Logs the FAWA topology (UCMTraceMeta) and per-request block_hashes
    (UCMTrace) for offline hit-rate simulation. No actual I/O.
    """

    @staticmethod
    def _approximate_gcd(sizes: list[int], lb: int) -> int:
        best_d, best_pad = lb, float("inf")
        hi = max(sizes) if sizes else lb
        for d in range(lb, hi + 1):
            pad = sum(-(-s // d) * d - s for s in sizes)
            if pad < best_pad:
                best_pad, best_d = pad, d
        return best_d

    @classmethod
    def _compute_fawa_sizes(
        cls,
        vllm_config: "VllmConfig",
        kv_cache_config: "KVCacheConfig",
        is_ascend: bool,
        bs: int,
    ) -> tuple[int, int, int]:
        """Compute (hbm_block_data_size, fa_file_size, wa_file_size).

        Mirrors UCMFAWAConnector._init_group_metas file_size logic and
        the calculator's deriveDSv4Params hbm formula.
        """
        hf = vllm_config.model_config.hf_text_config
        ratios = getattr(hf, "compress_ratios", None)
        if ratios is None:
            return 0, 0, 0
        decoder = ratios[:-1] if ratios and ratios[-1] == 0 else ratios
        num_c4a = sum(1 for r in decoder if r == 4)
        num_c128a = sum(1 for r in decoder if r == 128)
        num_total = len(decoder)
        has_mtp = (
            vllm_config.speculative_config is not None
            and vllm_config.speculative_config.num_speculative_tokens > 0
        )
        num_total_layers = num_total + (1 if has_mtp else 0)

        # num_layer_tuples (for both Ascend hbm and CUDA swa sub-group split)
        buckets = [num_c4a, num_c128a, num_total, num_c4a, num_c128a]
        lb = min(num_c4a, num_c128a) or 1
        num_tuples = cls._approximate_gcd(buckets, lb)

        # file_size (same as _init_group_metas)
        c4a_per_blk_tok = 1024 + 128 + 2  # mla + indexer + scale
        c128a_per_blk_tok = 32
        if is_ascend:
            fa_raw = bs * (c4a_per_blk_tok * num_c4a + c128a_per_blk_tok * num_c128a)
            wa_raw = 131072 * num_total_layers + (32768 + 8192) * num_c4a
        else:
            fa_raw = (37376 + 8448) * num_c4a + 1168 * num_c128a
            wa_raw = (37376 * 2) * num_total_layers + (8192 + 32768) * num_c4a
        fa_size = -(-fa_raw // 4096) * 4096  # round_up
        wa_size = -(-wa_raw // 4096) * 4096

        # hbm_block_data_size
        if is_ascend:
            tuple_bytes = bs * c4a_per_blk_tok  # bs × 1154
            hbm = num_tuples * tuple_bytes + (bs * 1024 if has_mtp else 0)
        else:
            # fp8_ds_mla layout (default): 584B/token MLA/SWA, 132B/token indexer, align=576
            mla_per_tok = 584
            idx_per_tok = 132
            align = 576
            g0 = (
                num_c4a * round_up(bs // 4 * mla_per_tok, align)
                + num_c128a * round_up(bs // 128 * mla_per_tok, align)
                + num_c4a * round_up(bs // 4 * idx_per_tok, align)
            )
            swa_subgroups = -(-num_total // num_tuples) if num_tuples else 1
            swa_sub1 = -(-num_total // swa_subgroups) if swa_subgroups else num_total
            g1 = swa_sub1 * round_up(64 * mla_per_tok, align)
            g3 = num_c4a * round_up(32768, align) + num_c4a * round_up(8192, align)
            g4 = num_c128a * round_up(32768, align)
            hbm = max(g0, g1, g3, g4)

        return hbm, fa_size, wa_size

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ) -> None:
        if kv_cache_config is None:
            raise RuntimeError("UCMFAWALiteConnector requires kv_cache_config.")
        super().__init__(vllm_config, role, kv_cache_config)

        groups = kv_cache_config.kv_cache_groups
        is_ascend = UCMFAWAConnector.can_handle_ascend_kv_cache_config(kv_cache_config)

        if is_ascend:
            bs = UCMFAWAConnector._get_ascend_base_block_size(kv_cache_config)
            ucm_hash_bs = bs * UCMFAWAConnector.ASCEND_C4_COMPRESS_RATIO
        else:
            bs = vllm_config.cache_config.block_size
            ucm_hash_bs = UCMFAWAConnector.DEFAULT_HASH_BLOCK_SIZE

        group_specs = []
        block_sizes_for_gcd = []
        effective_block_sizes = []
        for gid, group in enumerate(groups):
            spec = group.kv_cache_spec
            nested = getattr(spec, "kv_cache_specs", None)
            inner = next(iter(nested.values())) if nested else spec
            window = getattr(inner, "sliding_window", None)
            cr = getattr(inner, "compress_ratio", 1) or 1
            cr = max(cr, 1)
            bs_g = spec.block_size
            block_sizes_for_gcd.append(bs_g)
            # Ascend block_size may be physical (<=0.26) or logical (>=0.27).
            # DS V4 window groups (SWA and compressor state caches) never set
            # compress_ratio, so it defaults to 1 and logical == block_size,
            # matching the previous "window is None" special case.
            lbs = (
                UCMFAWAConnector.ascend_block_geometry(spec).logical
                if is_ascend
                else bs_g
            )
            effective_block_sizes.append(lbs)
            mtype = (
                "compress"
                if (is_ascend and window is None)
                else ("full_attention" if window is None else "sliding_window")
            )
            group_specs.append((f"G{gid}", lbs, mtype, window, cr))

        from functools import reduce
        from math import gcd, lcm

        vllm_hash_bs = reduce(gcd, block_sizes_for_gcd) if block_sizes_for_gcd else 1
        alignment = reduce(lcm, effective_block_sizes) if effective_block_sizes else 1

        hbm_data_size, fa_file, wa_file = self._compute_fawa_sizes(
            vllm_config, kv_cache_config, is_ascend, bs
        )

        self.hash_block_size = vllm_hash_bs
        self.ucm_hash_block_size = ucm_hash_bs
        self.trace_hash_block_size = max(512, vllm_hash_bs)

        logger.info(
            f"UCMTraceMeta: type=fawa, "
            f"is_mla=true, "
            f"vllm_hash_block_size={vllm_hash_bs}, "
            f"trace_hash_block_size={self.trace_hash_block_size}, "
            f"ucm_hash_block_size={ucm_hash_bs}, "
            f"hbm_block_data_size={hbm_data_size}, "
            f"fa_file_size={fa_file}, "
            f"wa_file_size={wa_file}, "
            f"alignment_tokens={alignment}, "
            f"groups={group_specs}"
        )
        logger.info("Init UCMFAWALiteConnector.")
