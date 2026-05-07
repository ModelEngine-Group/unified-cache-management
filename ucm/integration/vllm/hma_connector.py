import copy
import math
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence, Tuple

import numpy as np
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.core.sched.output import SchedulerOutput

from ucm.integration.vllm.device import create_device
from ucm.integration.vllm.ucm_connector import UCMDirectConnector
from ucm.logger import init_logger
from ucm.store.factory_v1 import UcmConnectorFactoryV1
from ucm.store.ucmstore_v1 import Task, UcmKVStoreBaseV1

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


class KVCacheGroupLayout:
    """Flat pointer layout for one vLLM KV cache group.

    The cache views belonging to one KV group are not necessarily contiguous by
    layer id, so this layout flattens all registered tensors in a deterministic
    order.
    """

    def __init__(self, kvcaches: dict[str, torch.Tensor]) -> None:
        self.kvcaches = dict(sorted(kvcaches.items(), key=self._sort_key))
        self.base_ptrs: np.ndarray
        self.block_strides: np.ndarray
        self.tensor_size_lists: np.ndarray
        self._build_layout()

    @staticmethod
    def _sort_key(item: tuple[str, torch.Tensor]) -> tuple[int, str]:
        name, _ = item
        return (extract_layer_index(name), name)

    def _build_layout(self) -> None:
        ptrs: list[int] = []
        strides: list[int] = []
        tensor_sizes: list[int] = []

        def handle_tensor(t: torch.Tensor, size_dims: Sequence[int]) -> None:
            ptrs.append(t[0].data_ptr())
            strides.append(t.stride(0) * t.element_size())
            tensor_size = math.prod([t.shape[i] for i in size_dims]) * t.element_size()
            tensor_sizes.append(tensor_size)

        for layer_name, kv_layer in self.kvcaches.items():
            if isinstance(kv_layer, torch.Tensor):
                if kv_layer.dim() == 5:
                    # [2, num_blocks, block_size, num_head, head_dim]
                    handle_tensor(kv_layer[0], (-3, -2, -1))
                    handle_tensor(kv_layer[1], (-3, -2, -1))
                elif kv_layer.dim() == 3:
                    # [num_blocks, block_size, head_dim]
                    handle_tensor(kv_layer, (-2, -1))
                else:
                    raise ValueError(
                        f"Unsupported KV cache tensor shape for "
                        f"{layer_name}: {kv_layer.shape}"
                    )
            elif isinstance(kv_layer, Tuple):
                for tensor in kv_layer:
                    if tensor.dim() == 4:
                        handle_tensor(tensor, (-3, -2, -1))
                    elif tensor.dim() == 3:
                        handle_tensor(tensor, (-2, -1))
                    else:
                        raise ValueError(
                            f"Unsupported tuple KV cache tensor shape for "
                            f"{layer_name}: {tensor.shape}"
                        )
            else:
                raise TypeError(
                    f"Unsupported KV cache type for " f"{layer_name}: {type(kv_layer)}"
                )

        if not ptrs:
            raise ValueError("KV cache group layout is empty.")

        self.base_ptrs = np.asarray(ptrs, dtype=np.uint64)
        self.block_strides = np.asarray(strides, dtype=np.uint64)
        self.tensor_size_lists = np.asarray(tensor_sizes, dtype=np.uint64)
        logger.info(
            f"KV cache group layout: views={len(self.kvcaches)}, "
            f"ptrs={len(ptrs)}, block_size={self.block_size}"
        )

    def extract_block_addrs(self, vllm_block_ids: list[int]) -> np.ndarray:
        vllm_block_ids_np = np.array(vllm_block_ids, np.uint64)
        return (
            vllm_block_ids_np[:, None] * self.block_strides[None, :]
            + self.base_ptrs[None, :]
        )

    def extract_block_tensor_views(
        self, vllm_block_ids: list[int]
    ) -> list[torch.Tensor]:
        tensors: list[torch.Tensor] = []

        def add_views(tensor: torch.Tensor, block_id: int) -> None:
            tensors.append(tensor[block_id])

        for block_id in vllm_block_ids:
            for layer_name, kv_layer in self.kvcaches.items():
                if isinstance(kv_layer, torch.Tensor):
                    if kv_layer.dim() == 5:
                        add_views(kv_layer[0], block_id)
                        add_views(kv_layer[1], block_id)
                    elif kv_layer.dim() == 3:
                        add_views(kv_layer, block_id)
                    else:
                        raise ValueError(
                            f"Unsupported KV cache tensor shape for "
                            f"{layer_name}: {kv_layer.shape}"
                        )
                elif isinstance(kv_layer, Tuple):
                    for tensor in kv_layer:
                        add_views(tensor, block_id)
                else:
                    raise TypeError(
                        f"Unsupported KV cache type for "
                        f"{layer_name}: {type(kv_layer)}"
                    )
        return tensors

    @property
    def tensor_size_list(self) -> list[int]:
        return self.tensor_size_lists.tolist()

    @property
    def shard_size(self) -> int:
        return int(self.tensor_size_lists.sum())

    @property
    def block_size(self) -> int:
        return self.shard_size


KVCacheGroupRow = tuple[list[int], ...]
KVCacheGroupRows = list[KVCacheGroupRow]


@dataclass
class FAWARequestMeta:
    ucm_block_ids: list[bytes] = field(default_factory=list)
    hbm_hit_block_num: int = 0
    total_hit_block_num: int = 0
    num_token_ids: int = 0
    token_processed: int = 0
    group_block_ids: dict[int, KVCacheGroupRow] = field(default_factory=dict)


@dataclass
class FAWARequestDispatchMeta:
    load_block_ids: tuple[list[bytes], KVCacheGroupRows]
    dump_block_ids: tuple[list[bytes], KVCacheGroupRows]


@dataclass
class UCMFAWAConnectorMetadata(KVConnectorMetadata):
    request_meta: dict[str, FAWARequestDispatchMeta] = field(default_factory=dict)


@dataclass
class FAWALoadTask:
    request_id: str
    label: str
    store: UcmKVStoreBaseV1
    task: Task


@dataclass
class FAWADumpTask:
    store: UcmKVStoreBaseV1
    task: Task


class UCMFAWAConnector(UCMDirectConnector):
    """UCM connector for mixed full-attention and window KV cache groups.

    Full-attention groups are stored once per reusable prefix block and are
    loaded for every external prefix hit. WA groups store the tail blocks
    needed at each prefix boundary, and only the final matched boundary is
    loaded.
    """

    DEFAULT_HASH_BLOCK_SIZE = 256

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        self._defer_scheduler_store = True
        super().__init__(vllm_config, role, kv_cache_config)
        self.hash_block_size = self.DEFAULT_HASH_BLOCK_SIZE
        self.block_size = self.DEFAULT_HASH_BLOCK_SIZE
        self.group_layouts: dict[int, KVCacheGroupLayout] = {}
        self._window_scratch_views: dict[tuple[int, int], list[torch.Tensor]] = {}
        self.fa_group_ids, self.window_group_ids = self._partition_kv_cache_groups()
        self.hash_block_size = self._get_hash_block_size()
        self.block_size = self.hash_block_size
        self.group_block_sizes = self._get_group_block_sizes()
        self.group_tail_blocks = self._get_group_tail_blocks()
        self.fa_store: Optional[UcmKVStoreBaseV1] = None
        self.wa_store: Optional[UcmKVStoreBaseV1] = None
        self.requests_meta: dict[str, FAWARequestMeta] = {}
        if role == KVConnectorRole.SCHEDULER:
            self.store = self._create_fa_store(None)
            self.fa_store = self.store
            self.wa_store = self._create_wa_store(None)
        logger.info(
            f"FAWA KV group config: fa_groups={self.fa_group_ids}, "
            f"window_groups={self.window_group_ids}, "
            f"block_sizes={self.group_block_sizes}, "
            f"tail_blocks={self.group_tail_blocks}"
        )
        logger.info("Init UCM FAWA connector.")

    @classmethod
    def can_handle_kv_cache_config(
        cls, kv_cache_config: Optional["KVCacheConfig"]
    ) -> bool:
        if kv_cache_config is None:
            return False
        fa_groups, window_groups = cls._partition_group_specs(
            kv_cache_config.kv_cache_groups
        )
        return bool(fa_groups and window_groups)

    def _create_fa_store(
        self,
        group_layouts: Optional[dict[int, KVCacheGroupLayout]],
        cpu_affinity_cores: Optional[list[int]] = None,
    ) -> UcmKVStoreBaseV1:
        tensor_size_list = None
        if self._role == KVConnectorRole.WORKER:
            if group_layouts is None:
                raise RuntimeError("Worker FA store needs layouts.")
            tensor_size_list = self._fa_tensor_size_list(group_layouts)
        return self._create_store(
            "FA",
            "fa",
            tensor_size_list,
            cpu_affinity_cores,
        )

    def _create_wa_store(
        self,
        group_layouts: Optional[dict[int, KVCacheGroupLayout]],
        cpu_affinity_cores: Optional[list[int]] = None,
    ) -> UcmKVStoreBaseV1:
        tensor_size_list = None
        if self._role == KVConnectorRole.WORKER:
            if group_layouts is None:
                raise RuntimeError("Worker WA store needs layouts.")
            tensor_size_list = self._window_tensor_size_list(group_layouts)
        return self._create_store(
            "WA",
            "wa",
            tensor_size_list,
            cpu_affinity_cores,
        )

    def _base_store_config(
        self,
        store_suffix: str,
    ) -> tuple[str, Optional[str], dict[str, object]]:
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
        config.setdefault("share_buffer_enable", True)
        if isinstance(config.get("storage_backends"), str):
            config["storage_backends"] = [
                path for path in config["storage_backends"].split(":")
            ]
        config["unique_id"] = f"{self.engine_id}_fawa_{store_suffix}"
        dp_rank = self._vllm_config.parallel_config.data_parallel_rank
        config["posix_gc_enable"] = (
            self._role != KVConnectorRole.WORKER and dp_rank == 0
        )
        return name, module_path, config

    def _create_store(
        self,
        label: str,
        store_suffix: str,
        tensor_size_list: Optional[list[int]],
        cpu_affinity_cores: Optional[list[int]] = None,
    ) -> UcmKVStoreBaseV1:
        name, module_path, config = self._base_store_config(store_suffix)
        if self._role == KVConnectorRole.WORKER:
            if tensor_size_list is None:
                raise RuntimeError(f"Worker FAWA {label} store needs tensor sizes.")
            config["device_id"] = self.local_rank
            config["tensor_size_list"] = tensor_size_list
            config["shard_size"] = int(sum(tensor_size_list))
            config["block_size"] = int(sum(tensor_size_list))
            config["local_rank_size"] = 1
            if cpu_affinity_cores:
                config["cpu_affinity_cores"] = list(cpu_affinity_cores)
        logger.info(
            f"create FAWA {label} {name} with config: "
            f"{self._summarize_store_config(config)}"
        )
        return UcmConnectorFactoryV1.create_connector(name, config, module_path)

    @staticmethod
    def _summarize_store_config(config: dict[str, object]) -> dict[str, object]:
        summary = dict(config)
        tensor_size_list = summary.pop("tensor_size_list", None)
        if tensor_size_list is not None:
            tensor_sizes = [int(size) for size in tensor_size_list]
            summary["tensor_count"] = len(tensor_sizes)
            summary["tensor_bytes"] = sum(tensor_sizes)
        return summary

    @staticmethod
    def _group_specs(group_spec) -> tuple[object, ...]:
        nested_specs = getattr(group_spec.kv_cache_spec, "kv_cache_specs", None)
        return (
            tuple(nested_specs.values())
            if nested_specs
            else (group_spec.kv_cache_spec,)
        )

    @staticmethod
    def _group_spec_items(group_spec) -> tuple[tuple[str, object], ...]:
        nested_specs = getattr(group_spec.kv_cache_spec, "kv_cache_specs", None)
        if nested_specs:
            return tuple(nested_specs.items())
        return tuple(
            (layer_name, group_spec.kv_cache_spec)
            for layer_name in group_spec.layer_names
        )

    @staticmethod
    def _spec_has_window(spec: object) -> bool:
        return (
            getattr(spec, "sliding_window", None) is not None
            or getattr(spec, "attention_chunk_size", None) is not None
        )

    @classmethod
    def _group_has_window(cls, group_spec) -> bool:
        return any(cls._spec_has_window(spec) for spec in cls._group_specs(group_spec))

    @classmethod
    def _partition_group_specs(
        cls, group_specs
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        fa_group_ids: list[int] = []
        window_group_ids: list[int] = []
        for group_id, group_spec in enumerate(group_specs):
            if cls._group_has_window(group_spec):
                window_group_ids.append(group_id)
            else:
                fa_group_ids.append(group_id)
        return tuple(fa_group_ids), tuple(window_group_ids)

    def _partition_kv_cache_groups(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if self._kv_cache_config is None:
            raise RuntimeError("FAWA connector requires kv_cache_config.")
        fa_group_ids, window_group_ids = self._partition_group_specs(
            self._kv_cache_config.kv_cache_groups
        )
        if not fa_group_ids:
            raise RuntimeError("FAWA connector found no full-attention groups.")
        if not window_group_ids:
            raise RuntimeError("FAWA connector found no window groups.")
        return fa_group_ids, window_group_ids

    def _get_hash_block_size(self) -> int:
        assert self._kv_cache_config is not None
        fa_block_sizes = {
            int(
                self._kv_cache_config.kv_cache_groups[group_id].kv_cache_spec.block_size
            )
            for group_id in self.fa_group_ids
        }
        if len(fa_block_sizes) != 1:
            raise RuntimeError(
                "FAWA connector requires one FA token block size, got "
                f"{sorted(fa_block_sizes)}."
            )
        return fa_block_sizes.pop()

    def _get_group_block_sizes(self) -> tuple[int, ...]:
        if self._kv_cache_config is None:
            raise RuntimeError("FAWA connector requires kv_cache_config.")

        raw_group_block_sizes = tuple(
            int(group.kv_cache_spec.storage_block_size)
            for group in self._kv_cache_config.kv_cache_groups
        )
        if not raw_group_block_sizes:
            raise RuntimeError("FAWA connector found no KV cache groups.")
        group_block_sizes = list(raw_group_block_sizes)
        for group_id in self.fa_group_ids:
            group_block_sizes[group_id] = self.hash_block_size
        for group_id, group_block_size in enumerate(group_block_sizes):
            if group_block_size <= 0:
                raise RuntimeError(
                    f"FAWA group {group_id} block size must be positive, "
                    f"got {group_block_size}."
                )
            if self.hash_block_size % group_block_size != 0:
                raise RuntimeError(
                    f"FAWA group {group_id} block size {group_block_size} "
                    f"must divide {self.hash_block_size}."
                )
        return tuple(group_block_sizes)

    @staticmethod
    def _group_window_tokens(group_spec) -> Optional[int]:
        window_sizes = {
            int(window_size)
            for spec in UCMFAWAConnector._group_specs(group_spec)
            if (
                window_size := (
                    getattr(spec, "sliding_window", None)
                    or getattr(spec, "attention_chunk_size", None)
                )
            )
            is not None
        }
        if not window_sizes:
            return None
        if len(window_sizes) != 1:
            raise RuntimeError(
                "FAWA KV cache group has mixed window sizes: " f"{sorted(window_sizes)}"
            )
        return window_sizes.pop()

    def _get_group_tail_blocks(self) -> tuple[Optional[int], ...]:
        tail_blocks: list[Optional[int]] = [None] * len(self.group_block_sizes)
        assert self._kv_cache_config is not None
        for group_id in self.window_group_ids:
            group_spec = self._kv_cache_config.kv_cache_groups[group_id]
            group_block_size = self.group_block_sizes[group_id]
            window_tokens = self._group_window_tokens(group_spec)
            if window_tokens is None:
                tail_blocks[group_id] = self.hash_block_size // group_block_size
                continue
            if self._is_compressor_state_group(group_id):
                tail_blocks[group_id] = self._compressor_state_tail_blocks(
                    group_id,
                    window_tokens,
                    group_block_size,
                )
                continue
            tail_blocks[group_id] = max(1, math.ceil(window_tokens / group_block_size))
        return tuple(tail_blocks)

    @staticmethod
    def _is_compressor_state_name(layer_name: str) -> bool:
        return ".compressor.state_cache" in layer_name

    @staticmethod
    def _compressor_state_prefix(layer_name: str) -> str:
        suffix = ".compressor.state_cache"
        if layer_name.endswith(suffix):
            return layer_name[: -len(suffix)]
        return layer_name.split(suffix, 1)[0]

    def _is_compressor_state_group(self, group_id: int) -> bool:
        assert self._kv_cache_config is not None
        group_spec = self._kv_cache_config.kv_cache_groups[group_id]
        layer_names = tuple(group_spec.layer_names)
        return bool(layer_names) and all(
            self._is_compressor_state_name(name) for name in layer_names
        )

    def _group_compress_ratio(self, group_id: int) -> Optional[int]:
        assert self._kv_cache_config is not None
        group_spec = self._kv_cache_config.kv_cache_groups[group_id]
        ratios = {
            int(ratio)
            for spec in self._group_specs(group_spec)
            if (ratio := getattr(spec, "compress_ratio", 1)) and int(ratio) > 1
        }
        if len(ratios) > 1:
            raise RuntimeError(
                f"FAWA KV cache group {group_id} has mixed compress ratios: "
                f"{sorted(ratios)}"
            )
        if ratios:
            return ratios.pop()

        if not self._is_compressor_state_group(group_id):
            return None

        config_ratios = getattr(
            self._vllm_config.model_config.hf_config,
            "compress_ratios",
            None,
        )
        if config_ratios:
            for layer_name in group_spec.layer_names:
                layer_index = extract_layer_index(layer_name)
                if layer_index < len(config_ratios):
                    ratio = int(config_ratios[layer_index])
                    if ratio > 1:
                        ratios.add(ratio)
            if len(ratios) > 1:
                raise RuntimeError(
                    f"FAWA compressor state group {group_id} maps to mixed "
                    f"model config compress ratios: {sorted(ratios)}"
                )
            if ratios:
                return ratios.pop()

        prefixes = tuple(
            self._compressor_state_prefix(layer_name)
            for layer_name in group_spec.layer_names
        )
        for other_group in self._kv_cache_config.kv_cache_groups:
            for layer_name, spec in self._group_spec_items(other_group):
                ratio = getattr(spec, "compress_ratio", 1)
                if not ratio or int(ratio) <= 1:
                    continue
                if any(
                    layer_name == prefix or layer_name.startswith(prefix + ".")
                    for prefix in prefixes
                ):
                    ratios.add(int(ratio))

        if len(ratios) > 1:
            raise RuntimeError(
                f"FAWA compressor state group {group_id} maps to mixed "
                f"compress ratios: {sorted(ratios)}"
            )
        return ratios.pop() if ratios else None

    def _compressor_state_tail_blocks(
        self,
        group_id: int,
        window_tokens: int,
        group_block_size: int,
    ) -> int:
        compress_ratio = self._group_compress_ratio(group_id)
        if compress_ratio is None:
            return max(1, math.ceil(window_tokens / group_block_size))
        if compress_ratio <= 0:
            raise RuntimeError(
                f"FAWA group {group_id} compress ratio must be positive, "
                f"got {compress_ratio}."
            )
        if window_tokens <= compress_ratio:
            return 0
        return math.ceil((window_tokens - compress_ratio) / group_block_size)

    def _split_kv_caches_by_vllm_groups(
        self, kv_caches: dict[str, torch.Tensor]
    ) -> dict[int, dict[str, torch.Tensor]]:
        assert self._kv_cache_config is not None
        groups: dict[int, dict[str, torch.Tensor]] = {}
        used_names: set[str] = set()
        for group_id, group_spec in enumerate(self._kv_cache_config.kv_cache_groups):
            group_caches = {
                name: kv_caches[name]
                for name in group_spec.layer_names
                if name in kv_caches
            }
            if group_caches:
                groups[group_id] = group_caches
                used_names.update(group_caches)

        missing_names = set(kv_caches) - used_names
        if missing_names:
            raise RuntimeError(
                "KV cache config did not include registered caches: "
                f"{sorted(missing_names)}"
            )

        return groups

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        self.kv_caches = kv_caches
        self.device = create_device()

        enable_affinity = os.getenv("VLLM_CPU_AFFINITY") == "1"
        worker_cores, store_cores = (
            self.device.split_cores(self.local_rank)
            if enable_affinity
            else (None, None)
        )

        grouped = self._split_kv_caches_by_vllm_groups(kv_caches)
        for group_id, group_caches in grouped.items():
            if not group_caches:
                logger.warning(f"KV cache group {group_id} is empty.")
                continue
            layout = KVCacheGroupLayout(group_caches)
            self.group_layouts[group_id] = layout

        self.store = self._create_fa_store(self.group_layouts, store_cores)
        self.fa_store = self.store
        self.wa_store = self._create_wa_store(
            self.group_layouts,
            store_cores,
        )

        if worker_cores:
            try:
                os.sched_setaffinity(0, worker_cores)
                logger.info(f"[VLLM CPU Affinity] Worker bound to cores {worker_cores}")
            except Exception as e:
                logger.warning(f"Failed to bind worker: {e}")

    def _block_key(self, canonical_hash: bytes) -> bytes:
        return self.request_hasher((b"fawa", canonical_hash))

    def _store_tensor_size_list(
        self,
        group_layouts: dict[int, KVCacheGroupLayout],
        group_ids: tuple[int, ...],
    ) -> list[int]:
        tensor_size_list: list[int] = []
        for group_id in group_ids:
            layout = group_layouts.get(group_id)
            if layout is None:
                continue
            repeat = (
                1 if group_id in self.fa_group_ids else self.group_tail_blocks[group_id]
            )
            assert repeat is not None
            tensor_size_list.extend(layout.tensor_size_list * repeat)
        return tensor_size_list

    def _fa_tensor_size_list(
        self, group_layouts: dict[int, KVCacheGroupLayout]
    ) -> list[int]:
        tensor_size_list = self._store_tensor_size_list(
            group_layouts, self.fa_group_ids
        )
        if not tensor_size_list:
            raise RuntimeError("Worker FA layout is empty.")
        return tensor_size_list

    def _window_tensor_size_list(
        self, group_layouts: dict[int, KVCacheGroupLayout]
    ) -> list[int]:
        tensor_size_list = self._store_tensor_size_list(
            group_layouts,
            self.window_group_ids,
        )
        if not tensor_size_list:
            raise RuntimeError("Worker WA layout is empty.")
        return tensor_size_list

    def _select_rows(
        self,
        group_rows: KVCacheGroupRows,
        group_ids: tuple[int, ...],
    ) -> KVCacheGroupRows:
        return [
            tuple(list(group_row[group_id]) for group_id in group_ids)
            for group_row in group_rows
        ]

    def _fa_rows(self, group_rows: KVCacheGroupRows) -> KVCacheGroupRows:
        return self._select_rows(group_rows, self.fa_group_ids)

    def _window_rows(self, group_rows: KVCacheGroupRows) -> KVCacheGroupRows:
        return self._select_rows(group_rows, self.window_group_ids)

    def _required_group_block_indices(
        self,
        group_id: int,
        canonical_block_idx: int,
    ) -> list[int]:
        if group_id in self.fa_group_ids:
            return [canonical_block_idx]

        group_block_size = self.group_block_sizes[group_id]
        total_hit_tokens = (canonical_block_idx + 1) * self.hash_block_size
        total_group_blocks = total_hit_tokens // group_block_size
        tail_blocks = self.group_tail_blocks[group_id]
        assert tail_blocks is not None
        start = max(0, total_group_blocks - tail_blocks)
        return list(range(start, total_group_blocks))

    def _group_indices(self, canonical_block_idx: int) -> list[list[int]]:
        return [
            self._required_group_block_indices(group_id, canonical_block_idx)
            for group_id in range(len(self.group_block_sizes))
        ]

    def _scratch_block_tensor_views(
        self,
        group_id: int,
        block_pos: int,
    ) -> list[torch.Tensor]:
        key = (group_id, block_pos)
        scratch_views = self._window_scratch_views.get(key)
        if scratch_views is None:
            layout = self.group_layouts[group_id]
            scratch_views = [
                torch.empty_like(tensor)
                for tensor in layout.extract_block_tensor_views([0])
            ]
            self._window_scratch_views[key] = scratch_views
        return scratch_views

    def _scratch_block_addrs(self, group_id: int, block_pos: int) -> np.ndarray:
        return np.asarray(
            [
                tensor.data_ptr()
                for tensor in self._scratch_block_tensor_views(group_id, block_pos)
            ],
            dtype=np.uint64,
        )

    def _extract_group_addrs(
        self,
        group_rows: KVCacheGroupRows,
        group_ids: tuple[int, ...],
        scratch_for_missing: bool = False,
    ) -> np.ndarray:
        rows: list[np.ndarray] = []
        for group_row in group_rows:
            row_parts: list[np.ndarray] = []
            for row_group_id, selected_ids in enumerate(group_row):
                group_id = group_ids[row_group_id]
                layout = self.group_layouts.get(group_id)
                if layout is None:
                    continue
                if not selected_ids:
                    continue
                for block_pos, block_id in enumerate(selected_ids):
                    if block_id < 0:
                        if not scratch_for_missing:
                            raise ValueError(
                                f"KV cache group {group_id} block "
                                f"position {block_pos} needs a scratch target."
                            )
                        row_parts.append(self._scratch_block_addrs(group_id, block_pos))
                    else:
                        row_parts.append(
                            layout.extract_block_addrs([block_id]).reshape(-1)
                        )
            if not row_parts:
                raise ValueError("KV cache pointer row is empty.")
            rows.append(np.concatenate(row_parts).astype(np.uint64, copy=False))
        if not rows:
            return np.empty((0, 0), dtype=np.uint64)
        return np.vstack(rows)

    def _select_group_block_ids(
        self,
        canonical_block_idx: int,
        blocks: "KVCacheBlocks",
        allow_null_tail: bool = False,
    ) -> KVCacheGroupRow:
        selected: list[list[int]] = []
        group_indices_by_group = self._group_indices(canonical_block_idx)
        for group_id, group_indices in enumerate(group_indices_by_group):
            group_selected: list[int] = []
            if group_id >= len(blocks.blocks):
                if group_indices:
                    raise ValueError(
                        f"KV cache group {group_id} is missing from "
                        f"KVCacheBlocks for canonical block {canonical_block_idx}."
                    )
                selected.append(group_selected)
                continue

            group_blocks = blocks.blocks[group_id]
            for group_block_idx in group_indices:
                if group_block_idx >= len(group_blocks):
                    raise ValueError(
                        f"KV cache group {group_id} block index "
                        f"{group_block_idx} is out of range "
                        f"(len={len(group_blocks)}) for canonical block "
                        f"{canonical_block_idx}."
                    )
                block = group_blocks[group_block_idx]
                if block.is_null:
                    if allow_null_tail and group_id in self.window_group_ids:
                        group_selected.append(-1)
                        continue
                    raise ValueError(
                        f"KV cache group {group_id} block index "
                        f"{group_block_idx} maps to a null HBM block for "
                        f"canonical block {canonical_block_idx}."
                    )
                group_selected.append(block.block_id)
            selected.append(group_selected)
        return tuple(selected)

    def _record_group_block_ids(
        self,
        req_meta: FAWARequestMeta,
        blocks: "KVCacheBlocks",
        end_block: int,
    ) -> None:
        for canonical_block_idx in range(end_block):
            if canonical_block_idx in req_meta.group_block_ids:
                continue
            allow_null_tail = canonical_block_idx < req_meta.total_hit_block_num - 1
            req_meta.group_block_ids[canonical_block_idx] = (
                self._select_group_block_ids(
                    canonical_block_idx,
                    blocks,
                    allow_null_tail=allow_null_tail,
                )
            )

    def _lookup_external_hit_blocks(self, external_keys: list[bytes]) -> int:
        if self.fa_store is None:
            raise RuntimeError("FA store is not initialized.")
        if self.wa_store is None:
            raise RuntimeError("WA store is not initialized.")
        fa_hit_blocks = self.fa_store.lookup_on_prefix(external_keys) + 1
        window_hit_blocks = self.wa_store.lookup_on_prefix(external_keys) + 1
        return min(fa_hit_blocks, window_hit_blocks)

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        assert num_computed_tokens % self.hash_block_size == 0
        hbm_hit_block_num = num_computed_tokens // self.hash_block_size
        canonical_hashes = self.generate_hash(
            self.hash_block_size, request.all_token_ids, self._seed
        )

        if self.persist_token_threshold > request.num_tokens:
            return 0, False

        external_keys = [
            self._block_key(block_hash)
            for block_hash in canonical_hashes[hbm_hit_block_num:]
        ]
        if not external_keys:
            return 0, False

        try:
            external_hit_blocks = self._lookup_external_hit_blocks(external_keys)
        except Exception as e:
            external_hit_blocks = 0
            logger.error(
                f"request {request.request_id} FAWA lookup error. "
                f"{type(e).__name__}: {e}"
            )

        total_hit_block_num = hbm_hit_block_num + external_hit_blocks
        external_hit_tokens = external_hit_blocks * self.hash_block_size
        num_total_hit_tokens = total_hit_block_num * self.hash_block_size
        if num_total_hit_tokens == request.num_tokens:
            external_hit_tokens -= 1

        self.requests_meta[request.request_id] = FAWARequestMeta(
            ucm_block_ids=canonical_hashes,
            hbm_hit_block_num=hbm_hit_block_num,
            total_hit_block_num=total_hit_block_num,
            num_token_ids=len(request.all_token_ids),
            token_processed=num_total_hit_tokens,
        )

        logger.info_once(
            f"FAWA request_id: {request.request_id}, "
            f"total_blocks_num: {len(canonical_hashes)}, "
            f"hit hbm: {hbm_hit_block_num}, "
            f"hit external: {external_hit_blocks}"
        )
        return external_hit_tokens, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        req_meta = self.requests_meta.get(request.request_id)
        if req_meta is None:
            return

        max_full_blocks = req_meta.num_token_ids // self.hash_block_size
        fa_blocks = min(
            len(blocks.blocks[group_id])
            for group_id in self.fa_group_ids
            if group_id < len(blocks.blocks)
        )
        end_block = min(max_full_blocks, fa_blocks)
        if end_block == 0:
            return

        try:
            self._record_group_block_ids(req_meta, blocks, end_block)
        except Exception as e:
            logger.error(
                f"request {request.request_id} record FAWA HBM-aligned "
                f"block ids failed. {type(e).__name__}: {e}"
            )
            raise

    def _make_dispatch_meta(
        self,
        request_id: str,
        req_meta: FAWARequestMeta,
        new_tokens: int,
        need_load: bool,
    ) -> FAWARequestDispatchMeta:
        load_keys: list[bytes] = []
        load_group_block_ids: KVCacheGroupRows = []
        if need_load and req_meta.total_hit_block_num > req_meta.hbm_hit_block_num:
            load_indices = list(
                range(req_meta.hbm_hit_block_num, req_meta.total_hit_block_num)
            )
            load_keys = [
                self._block_key(req_meta.ucm_block_ids[idx]) for idx in load_indices
            ]
            load_group_block_ids = [
                req_meta.group_block_ids[idx] for idx in load_indices
            ]

        dump_keys: list[bytes] = []
        dump_group_block_ids: KVCacheGroupRows = []
        if req_meta.token_processed < req_meta.num_token_ids:
            start_block = req_meta.token_processed // self.hash_block_size
            end_block = (req_meta.token_processed + new_tokens) // self.hash_block_size
            if end_block > start_block:
                dump_indices = list(range(start_block, end_block))
                dump_keys = [
                    self._block_key(req_meta.ucm_block_ids[idx]) for idx in dump_indices
                ]
                dump_group_block_ids = [
                    req_meta.group_block_ids[idx] for idx in dump_indices
                ]
            req_meta.token_processed += new_tokens

        return FAWARequestDispatchMeta(
            (load_keys, load_group_block_ids),
            (dump_keys, dump_group_block_ids),
        )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        requests_dispatch_meta: dict[str, FAWARequestDispatchMeta] = {}

        for request in scheduler_output.scheduled_new_reqs:
            req_meta = self.requests_meta.get(request.req_id)
            if req_meta:
                requests_dispatch_meta[request.req_id] = self._make_dispatch_meta(
                    request.req_id,
                    req_meta,
                    scheduler_output.num_scheduled_tokens[request.req_id],
                    True,
                )

        cached = scheduler_output.scheduled_cached_reqs
        for request_id in cached.req_ids:
            req_meta = self.requests_meta.get(request_id)
            if not req_meta:
                continue
            resumed = request_id in cached.resumed_req_ids
            requests_dispatch_meta[request_id] = self._make_dispatch_meta(
                request_id,
                req_meta,
                scheduler_output.num_scheduled_tokens[request_id],
                resumed,
            )

        for request_id in scheduler_output.finished_req_ids:
            self.requests_meta.pop(request_id, None)

        return UCMFAWAConnectorMetadata(requests_dispatch_meta)

    def _submit_load_task(
        self,
        request_id: str,
        label: str,
        store: UcmKVStoreBaseV1,
        keys: list[bytes],
        ptrs: np.ndarray,
    ) -> FAWALoadTask:
        shard_indexs = [0] * len(keys)
        task = store.load_data(keys, shard_indexs, ptrs)
        return FAWALoadTask(
            request_id=request_id,
            label=label,
            store=store,
            task=task,
        )

    def _wait_load_task(
        self,
        load_task: FAWALoadTask,
    ) -> None:
        try:
            load_task.store.wait(load_task.task)
        except Exception as e:
            logger.error(
                f"request {load_task.request_id} wait FAWA load "
                f"task label={load_task.label} error. {type(e).__name__}: {e}"
            )

    def _submit_dump_task(
        self,
        store: UcmKVStoreBaseV1,
        keys: list[bytes],
        ptrs: np.ndarray,
        event_handle,
    ) -> FAWADumpTask:
        shard_indexs = [0] * len(keys)
        task = store.dump_data(keys, shard_indexs, ptrs, event_handle)
        return FAWADumpTask(
            store=store,
            task=task,
        )

    def _wait_dump_task(self, dump_task: FAWADumpTask) -> None:
        dump_task.store.wait(dump_task.task)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, UCMFAWAConnectorMetadata)

        tasks: list[FAWALoadTask] = []
        for request_id, request in metadata.request_meta.items():
            keys, group_rows = request.load_block_ids
            if not keys:
                continue
            try:
                if self.fa_store is None:
                    raise RuntimeError("FA store is not initialized.")
                fa_ptrs = self._extract_group_addrs(
                    self._fa_rows(group_rows),
                    self.fa_group_ids,
                )
                tasks.append(
                    self._submit_load_task(
                        request_id,
                        "FA",
                        self.fa_store,
                        keys,
                        fa_ptrs,
                    )
                )

                if self.wa_store is None:
                    raise RuntimeError("WA store is not initialized.")
                window_keys = keys[-1:]
                window_rows = self._window_rows(group_rows[-1:])
                window_ptrs = self._extract_group_addrs(
                    window_rows,
                    self.window_group_ids,
                    scratch_for_missing=True,
                )
                tasks.append(
                    self._submit_load_task(
                        request_id,
                        "WA",
                        self.wa_store,
                        window_keys,
                        window_ptrs,
                    )
                )
            except Exception as e:
                logger.error(
                    f"request {request_id} submit FAWA load task "
                    f"error. {type(e).__name__}: {e}"
                )

        for load_task in tasks:
            self._wait_load_task(load_task)

    def wait_for_save(self) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, UCMFAWAConnectorMetadata)

        keys: list[bytes] = []
        group_rows: KVCacheGroupRows = []
        for request in metadata.request_meta.values():
            req_keys, req_group_rows = request.dump_block_ids
            if not req_keys:
                continue
            keys.extend(req_keys)
            group_rows.extend(req_group_rows)

        if not keys:
            return

        if self.tp_rank != 0:
            return

        try:
            event_handle = self._get_dump_event_handle()
            tasks: list[FAWADumpTask] = []
            if self.fa_store is None:
                raise RuntimeError("FA store is not initialized.")
            fa_ptrs = self._extract_group_addrs(
                self._fa_rows(group_rows),
                self.fa_group_ids,
            )
            tasks.append(
                self._submit_dump_task(
                    self.fa_store,
                    keys,
                    fa_ptrs,
                    event_handle,
                )
            )
            if self.wa_store is None:
                raise RuntimeError("WA store is not initialized.")
            window_ptrs = self._extract_group_addrs(
                self._window_rows(group_rows),
                self.window_group_ids,
            )
            tasks.append(
                self._submit_dump_task(
                    self.wa_store,
                    keys,
                    window_ptrs,
                    event_handle,
                )
            )
            for dump_task in tasks:
                self._wait_dump_task(dump_task)
        except Exception as e:
            logger.error(f"dump FAWA kv cache failed. {type(e).__name__}: {e}")
