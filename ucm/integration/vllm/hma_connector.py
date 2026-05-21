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


@dataclass(frozen=True)
class KVCacheGroupMeta:
    group_id: int
    token_block_size: int
    tail_blocks: int
    tail_tokens: int


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
        self.tensor_token_strides: np.ndarray
        self.tensor_sizes_per_token: np.ndarray
        self.tensor_block_sizes: np.ndarray
        self._build_layout()

    @staticmethod
    def _sort_key(item: tuple[str, torch.Tensor]) -> tuple[int, str]:
        name, _ = item
        return (extract_layer_index(name), name)

    def _build_layout(self) -> None:
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
            if tensor.dim() == 5:
                # [2, num_blocks, block_size, num_head, head_dim]
                handle_tensor(tensor[0], (-3, -2, -1), layer_name)
                handle_tensor(tensor[1], (-3, -2, -1), layer_name)
            elif tensor.dim() == 4:
                if tensor.shape[1] == 2:
                    # GPU kernels may register [num_blocks, 2, block_size, ...].
                    # Split the K/V axis before reading the token dimension.
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
        return (
            block_ids[:, None] * self.block_strides[None, :] + self.base_ptrs[None, :]
        ).astype(np.uint64, copy=False)

    def segment_tensor_size_list(
        self,
        logical_tokens: int,
        group_token_block_size: int,
    ) -> list[int]:

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


@dataclass
class FAWARequestMeta:
    ucm_block_ids: list[bytes] = field(default_factory=list)
    hbm_hit_block_num: int = 0
    total_hit_block_num: int = 0
    num_token_ids: int = 0
    vllm_block_ids: tuple[list[int], ...] = field(default_factory=tuple)
    token_processed: int = 0


@dataclass
class FAWARequestDispatchMeta:
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
    request_meta: dict[str, FAWARequestDispatchMeta] = field(default_factory=dict)


@dataclass
class FAWALoadTask:
    request_id: str
    label: str
    store: UcmKVStoreBaseV1
    task: Task
    key_count: int
    anchor_vllm_block_ids: set[int] = field(default_factory=set)


@dataclass
class FAWADumpTask:
    label: str
    store: UcmKVStoreBaseV1
    task: Task
    key_count: int


class UCMFAWAConnector(UCMDirectConnector):
    """UCM connector for mixed full-attention and window KV cache groups.

    Full-attention groups are stored once per reusable prefix block and are
    loaded for every external prefix hit. WA groups store the tail blocks
    needed at each prefix boundary, and only the final matched boundary is
    loaded.
    """

    DEFAULT_HASH_BLOCK_SIZE = 256
    ASCEND_DEFAULT_HASH_BLOCK_SIZE = 512

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
        self.fa_group_ids, self.window_group_ids = [], []
        self.group_metas: dict[int, KVCacheGroupMeta] = {}
        self._init_group_metas()
        self.fa_store: Optional[UcmKVStoreBaseV1] = None
        self.wa_store: Optional[UcmKVStoreBaseV1] = None
        self.requests_meta: dict[str, FAWARequestMeta] = {}
        if role == KVConnectorRole.SCHEDULER:
            self.store = self._create_fa_store(None)
            self.fa_store = self.store
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
            f"is_ascend_layout={self.is_ascend_layout}, "
            f"group_metas={group_meta_summary}"
        )
        logger.info("Init UCM FAWA connector.")

    @classmethod
    def can_handle_kv_cache_config(
        cls, kv_cache_config: Optional["KVCacheConfig"]
    ) -> bool:
        if kv_cache_config is None:
            return False

        kv_cache_groups = kv_cache_config.kv_cache_groups
        spec_names = set()
        for group_spec in kv_cache_groups:
            nested_specs = getattr(group_spec.kv_cache_spec, "kv_cache_specs", None)
            spec = (
                next(iter(nested_specs.values()))
                if nested_specs
                else group_spec.kv_cache_spec
            )
            spec_names.add(type(spec).__name__)
        # current only support for DeepSeekV4
        DS_V4_REQUIRED_SPECS = frozenset({"SlidingWindowMLASpec"})
        gpu_support = DS_V4_REQUIRED_SPECS.issubset(spec_names)
        if gpu_support:
            return True
        return cls.can_handle_ascend_kv_cache_config(kv_cache_config)

    @classmethod
    def can_handle_ascend_kv_cache_config(
        cls, kv_cache_config: Optional["KVCacheConfig"]
    ) -> bool:
        if kv_cache_config is None:
            return False
        kv_cache_groups = kv_cache_config.kv_cache_groups
        spec_names = set()
        for group_spec in kv_cache_groups:
            nested_specs = getattr(group_spec.kv_cache_spec, "kv_cache_specs", None)
            spec = (
                next(iter(nested_specs.values()))
                if nested_specs
                else group_spec.kv_cache_spec
            )
            spec_names.add(type(spec).__name__)
        ASCEND_REQUIRED_SPECS = frozenset(
            {"Compress4AttentionSpec", "C4IndexerSpec", "Compress128AttentionSpec"}
        )
        npu_support = type(kv_cache_groups[0]).__name__.startswith(
            "Ascend"
        ) and ASCEND_REQUIRED_SPECS.issubset(spec_names)
        return npu_support

    def _init_group_metas(self) -> None:
        if self.can_handle_ascend_kv_cache_config(self._kv_cache_config):
            self.is_ascend_layout = True
            self.hash_block_size = self.ASCEND_DEFAULT_HASH_BLOCK_SIZE

        groups = self._kv_cache_config.kv_cache_groups
        self.fa_group_ids, self.window_group_ids = [], []
        layer_compress_ratios = getattr(
            self._vllm_config.model_config.hf_config,
            "compress_ratios",
            None,
        )
        if layer_compress_ratios is None:
            raise "current only support DSV4"
        for group_id, group in enumerate(groups):
            kv_cache_spec = group.kv_cache_spec
            # handle attention window cache
            nested_specs = getattr(kv_cache_spec, "kv_cache_specs", None)
            spec = next(iter(nested_specs.values())) if nested_specs else kv_cache_spec
            window_size = getattr(spec, "sliding_window", None)
            compress_ratio = getattr(spec, "compress_ratio", 1)
            token_block_size = kv_cache_spec.block_size

            if self.is_ascend_layout:
                # for ascend bug, wait for ascend fix
                token_block_size = kv_cache_spec.block_size * compress_ratio

            if window_size is None:
                # hash_block_size must be an integral multiple of token_block_size.
                tail_tokens = self.hash_block_size
                self.fa_group_ids.append(group_id)
            else:
                tensor_name = group.layer_names[0]
                if type(spec).__name__ in ["SWAAttentionSpec"] or tensor_name.split(
                    "."
                )[-1] in ["swa_cache"]:
                    # for swa cache
                    tail_tokens = window_size
                else:
                    # for compressor state cache
                    layer_index = extract_layer_index(tensor_name)
                    tail_tokens = window_size - layer_compress_ratios[layer_index]

                tail_blocks = tail_tokens // token_block_size
                self.window_group_ids.append(group_id)

            tail_blocks = max(tail_tokens // token_block_size, 1)
            self.group_metas[group_id] = KVCacheGroupMeta(
                group_id=group_id,
                token_block_size=token_block_size,
                tail_blocks=tail_blocks,
                tail_tokens=tail_tokens,
            )

    def _create_fa_store(
        self,
        group_layouts: Optional[dict[int, KVCacheGroupLayout]],
        cpu_affinity_cores: Optional[list[int]] = None,
    ) -> UcmKVStoreBaseV1:
        tensor_size_list = None
        if self._role == KVConnectorRole.WORKER:
            if group_layouts is None:
                raise RuntimeError("Worker FA store needs layouts.")
            tensor_size_list = self._store_tensor_size_list(
                group_layouts,
                self.fa_group_ids,
            )
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
            tensor_size_list = self._store_tensor_size_list(
                group_layouts,
                self.window_group_ids,
            )
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
        # MLA ranks share one logical store buffer; non-MLA stores are per rank.
        config.setdefault("share_buffer_enable", self.is_mla)
        if isinstance(config.get("storage_backends"), str):
            config["storage_backends"] = [
                path for path in config["storage_backends"].split(":")
            ]
        config["unique_id"] = f"{self.engine_id}_fawa_{store_suffix}"
        self._namespace_storage_backends(config, store_suffix)
        dp_rank = self._vllm_config.parallel_config.data_parallel_rank
        config["posix_gc_enable"] = (
            self._role != KVConnectorRole.WORKER and dp_rank == 0
        )
        return name, module_path, config

    @staticmethod
    def _namespace_storage_backends(
        config: dict[str, object],
        store_suffix: str,
    ) -> None:
        backends = config.get("storage_backends")
        if not isinstance(backends, list):
            return
        namespaced_backends: list[str] = []
        for backend in backends:
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
        name, module_path, config = self._base_store_config(store_suffix)
        if self._role == KVConnectorRole.WORKER:
            if tensor_size_list is None:
                raise RuntimeError(f"Worker FAWA {label} store needs tensor sizes.")
            config["device_id"] = self.local_rank
            config["tensor_size_list"] = tensor_size_list
            # for io_direct support, shard_size & block_size should be aligned with 4KB
            aligned_size = 4096
            padded_size = ((sum(tensor_size_list) + aligned_size - 1) // aligned_size) * aligned_size
            config["shard_size"] = padded_size
            config["block_size"] = padded_size 
            # MLA stores aggregate TP shards under one logical rank group.
            config["local_rank_size"] = self.tp_size if self.is_mla else 1
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

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        self.kv_caches = kv_caches
        self.device = create_device()

        enable_affinity = os.getenv("VLLM_CPU_AFFINITY") == "1"
        worker_cores, store_cores = (
            self.device.split_cores(self.local_rank)
            if enable_affinity
            else (None, None)
        )

        if self.is_ascend_layout:
            # current for ascend, one layer_tensor_name per group_spec, multi tensors per layer_tensor_name
            next_tensor_index_by_layer: dict[str, int] = {}
            for group_id, group in enumerate(self._kv_cache_config.kv_cache_groups):
                kv_cache_spec_name = type(group.kv_cache_spec).__name__
                group_caches: dict[str, torch.Tensor] = {}
                for layer_name in group.layer_names:
                    tensor_count = 2 if kv_cache_spec_name == "C4IndexerSpec" else 1
                    start = next_tensor_index_by_layer.get(layer_name, 0)
                    end = start + tensor_count
                    next_tensor_index_by_layer[layer_name] = end
                    group_caches[layer_name] = tuple(kv_caches[layer_name][start:end])

                layout = KVCacheGroupLayout(group_caches)
                self.group_layouts[group_id] = layout
        else:
            for group_id, group_spec in enumerate(
                self._kv_cache_config.kv_cache_groups
            ):
                group_caches: dict[str, torch.Tensor] = {}
                for layer_name in group_spec.layer_names:
                    group_caches[layer_name] = kv_caches[layer_name]
                layout = KVCacheGroupLayout(group_caches)
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
        if self.fa_store is None:
            raise RuntimeError("FA store is not initialized.")
        if self.wa_store is None:
            raise RuntimeError("WA store is not initialized.")
        fa_hit_blocks = self.fa_store.lookup_on_prefix(external_keys) + 1
        if fa_hit_blocks <= 0:
            return 0

        # WA rows represent window boundary state, so they are not required to
        # form a prefix. Search only inside the FA-contiguous hit range and use
        # the latest boundary that exists.
        for hit_blocks in range(fa_hit_blocks, -1, -1):
            # TODO: Add Posix SpaceManager::LookupOnSuffix() for sparse WA boundary lookups, where only the latest existing key is needed.
            key = external_keys[hit_blocks - 1]
            if self.wa_store.lookup([key])[0]:
                return hit_blocks
        return 0

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        if num_computed_tokens % self.hash_block_size != 0:
            raise RuntimeError(
                f"FAWA requires aligned computed tokens, got "
                f"{num_computed_tokens} with block size {self.hash_block_size}."
            )
        hbm_hit_block_num = num_computed_tokens // self.hash_block_size
        canonical_hashes = self.generate_hash(
            self.hash_block_size, request.all_token_ids, self._seed
        )

        if self.persist_token_threshold > request.num_tokens:
            return 0, False

        external_keys = canonical_hashes[hbm_hit_block_num:]
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
        pass

    def _slice_group_block_ids(
        self,
        group_id: int,
        group_block_ids: list[int],
        window_boundary_token_idx: np.ndarray,
    ) -> list[int]:
        is_window_group = group_id in self.window_group_ids
        group_meta = self.group_metas[group_id]
        if is_window_group:
            if not group_meta.tail_tokens:
                return []
            # current only care the last hash block's wa cache
            boundary_block_idx = (
                window_boundary_token_idx[-1] // group_meta.token_block_size
            ) + 1
            return group_block_ids[
                boundary_block_idx - group_meta.tail_blocks : boundary_block_idx
            ]
        # for fa part, hash_block_size <= token_block_size, at most one block per hash block
        return np.array(group_block_ids)[
            window_boundary_token_idx // group_meta.token_block_size
        ].tolist()

    def _generate_dispatch_meta(
        self,
        req_meta: FAWARequestMeta,
        new_tokens: int,
        new_vllm_block_ids: tuple[list[int], ...],
        need_load: bool = True,
    ) -> FAWARequestDispatchMeta:
        """
        Request Blocks layout:
        ----------------------------------------------------------------------------------------------------
        | local_computed_block(HBM hit) | external_computed_block(external hit) | new_block(need to dump)  |
        ----------------------------------------------------------------------------------------------------
        |      hbm_hit_block_num        |                 LOAD                  |     new_blocks_num       |
        ----------------------------------------------------------------------------------------------------
        |                              total_hit_block_num                      |
        ----------------------------------------------------------------------------------------------------
        |                                         scheduled_block_num                                      |
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
                np.arange(load_start, load_end) * self.hash_block_size - 1
            )
            for group_id, group_block_ids in enumerate(all_group_block_ids):
                load_vllm_block_ids.append(
                    self._slice_group_block_ids(
                        group_id,
                        group_block_ids,
                        window_boundary_token_idx,
                    )
                )

        computed_end_token = min(
            req_meta.num_token_ids,
            req_meta.token_processed + new_tokens,
        )
        dump_start = req_meta.token_processed // self.hash_block_size
        dump_end = computed_end_token // self.hash_block_size
        dump_block_keys: list[bytes] = []
        dump_vllm_block_ids: list[list[int]] = []
        if dump_end > dump_start:
            dump_block_keys = req_meta.ucm_block_ids[dump_start:dump_end]
            window_boundary_token_idx = (
                np.arange(dump_start, dump_end) * self.hash_block_size - 1
            )
            for group_id, group_block_ids in enumerate(all_group_block_ids):
                dump_vllm_block_ids.append(
                    self._slice_group_block_ids(
                        group_id,
                        group_block_ids,
                        window_boundary_token_idx,
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
    ) -> KVConnectorMetadata:
        requests_dispatch_meta: dict[str, FAWARequestDispatchMeta] = {}
        # for new request, we need to load and dump
        for request in scheduler_output.scheduled_new_reqs:
            request_id, vllm_block_ids = request.req_id, request.block_ids
            req_meta = self.requests_meta.get(request_id)
            if req_meta:
                requests_dispatch_meta[request_id] = self._generate_dispatch_meta(
                    req_meta,
                    scheduler_output.num_scheduled_tokens[request_id],
                    tuple(vllm_block_ids),
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
                )

        for request_id in scheduler_output.finished_req_ids:
            self.requests_meta.pop(request_id, None)

        return UCMFAWAConnectorMetadata(requests_dispatch_meta)

    def update_connector_output(self, connector_output) -> None:
        return None

    def get_finished(
        self,
        finished_req_ids: set[str],
    ) -> tuple[set[str] | None, set[str] | None]:
        return None, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, object] | None]:
        return False, None

    def _submit_load_task(
        self,
        request_id: str,
        label: str,
        store: UcmKVStoreBaseV1,
        keys: list[bytes],
        ptrs: np.ndarray,
        anchor_vllm_block_ids: set[int],
    ) -> FAWALoadTask:
        shard_indices = [0] * len(keys)
        task = store.load_data(keys, shard_indices, ptrs)
        return FAWALoadTask(
            request_id=request_id,
            label=label,
            store=store,
            task=task,
            key_count=len(keys),
            anchor_vllm_block_ids=anchor_vllm_block_ids,
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
            self._invalid_block_ids.update(load_task.anchor_vllm_block_ids)

    def get_block_ids_with_load_errors(self) -> set[int]:
        """
        Get vLLM block IDs that failed to load through FAWA stores.

        Returns:
            Set of vLLM/HMA block IDs that encountered load errors.
            Empty set if no load errors occurred.
        """
        res = self._invalid_block_ids
        self._invalid_block_ids = set()
        return res

    @staticmethod
    def _first_group_anchor_ids(
        candidate_vllm_ids: tuple[list[int], ...],
    ) -> set[int]:
        if not candidate_vllm_ids:
            return set()
        return {block_id for block_id in candidate_vllm_ids[0] if block_id >= 0}

    def _first_group_anchor_ids_for_hash_range(
        self,
        candidate_vllm_ids: tuple[list[int], ...],
        hash_start: int,
        hash_end: int,
        candidate_hash_start: int,
    ) -> set[int]:
        if not candidate_vllm_ids or hash_end <= hash_start:
            return set()
        first_group_ids = candidate_vllm_ids[0]
        start = hash_start - candidate_hash_start
        end = hash_end - candidate_hash_start
        if start < 0 or end > len(first_group_ids):
            raise RuntimeError(
                f"FAWA load anchor range [{hash_start}, {hash_end}) is outside "
                f"candidate base={candidate_hash_start}, "
                f"candidates={len(first_group_ids)}."
            )
        return {block_id for block_id in first_group_ids[start:end] if block_id >= 0}

    def _submit_dump_task(
        self,
        label: str,
        store: UcmKVStoreBaseV1,
        keys: list[bytes],
        ptrs: np.ndarray,
        event_handle,
    ) -> FAWADumpTask:
        shard_indices = [0] * len(keys)
        task = store.dump_data(keys, shard_indices, ptrs, event_handle)
        return FAWADumpTask(
            label=label,
            store=store,
            task=task,
            key_count=len(keys),
        )

    def _wait_dump_task(self, dump_task: FAWADumpTask) -> None:
        dump_task.store.wait(dump_task.task)

    def _extract_fa_ptr(self, store_keys, hash_start, hash_end, candidate_vllm_ids):
        all_ptrs = []
        for group_id in self.fa_group_ids:
            layout = self.group_layouts.get(group_id)
            if layout is None:
                continue
            meta = self.group_metas[group_id]
            block_ids = np.asarray(candidate_vllm_ids[group_id], dtype=np.uint64)
            # current self.hash_block_size <= meta.token_block_size
            if self.hash_block_size == meta.token_block_size:
                # for gpu setting
                group_ptrs = layout.extract_addrs(block_ids)
            else:
                # for npu setting
                token_start = np.arange(hash_start, hash_end) * self.hash_block_size
                token_offsets = token_start % meta.token_block_size
                group_ptrs = layout.extract_addrs_with_offsets(
                    block_ids, meta.token_block_size, token_offsets
                )
            all_ptrs.append(group_ptrs)

        return np.concatenate(all_ptrs, axis=1)

    def _extract_wa_ptr(self, store_keys, vllm_ids):
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
                token_offsets = np.ones_like(block_ids) * (
                    meta.token_block_size - meta.tail_tokens
                )
                group_ptrs = layout.extract_addrs_with_offsets(
                    block_ids, meta.token_block_size, token_offsets
                )
            else:
                token_offsets = np.zeros_like(block_ids)
                group_ptrs = layout.extract_addrs(block_ids)
                group_ptrs = group_ptrs.reshape(len(store_keys), -1)

            all_ptrs.append(group_ptrs)

        return np.concatenate(all_ptrs, axis=1)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, UCMFAWAConnectorMetadata):
            raise RuntimeError(f"Unexpected FAWA metadata type: {type(metadata)}")

        tasks: list[FAWALoadTask] = []
        for request_id, request in metadata.request_meta.items():
            if not request.load_keys:
                continue
            group0_vllm_block_ids = set(request.load_vllm_block_ids[0])
            try:
                if self.fa_store is None:
                    raise RuntimeError("FA store is not initialized.")
                if self.wa_store is None:
                    raise RuntimeError("WA store is not initialized.")

                # FA groups are loaded for every external-hit canonical block.
                fa_ptrs = self._extract_fa_ptr(
                    request.load_keys,
                    request.load_hash_start,
                    request.load_hash_end,
                    request.load_vllm_block_ids,
                )
                tasks.append(
                    self._submit_load_task(
                        request_id,
                        "FA",
                        self.fa_store,
                        request.load_keys,
                        fa_ptrs,
                        group0_vllm_block_ids,
                    )
                )

                # WA groups only need the final matched boundary.
                window_keys = request.load_keys[-1:]
                window_ptrs = self._extract_wa_ptr(
                    window_keys,
                    request.load_vllm_block_ids,
                )
                tasks.append(
                    self._submit_load_task(
                        request_id,
                        "WA",
                        self.wa_store,
                        window_keys,
                        window_ptrs,
                        group0_vllm_block_ids,
                    )
                )
            except Exception as e:
                logger.error(
                    f"request {request_id} submit FAWA load task "
                    f"error. {type(e).__name__}: {e}"
                )
                self._invalid_block_ids.update(group0_vllm_block_ids)

        for load_task in tasks:
            self._wait_load_task(load_task)

    def wait_for_save(self) -> None:
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, UCMFAWAConnectorMetadata):
            raise RuntimeError(f"Unexpected FAWA metadata type: {type(metadata)}")

        if self.tp_rank != 0:
            return

        try:
            event_handle = self._get_dump_event_handle()
            if self.fa_store is None:
                raise RuntimeError("FA store is not initialized.")
            if self.wa_store is None:
                raise RuntimeError("WA store is not initialized.")

            fa_dump_keys: list[bytes] = []
            wa_dump_keys: list[bytes] = []
            fa_ptr_rows: list[np.ndarray] = []
            wa_ptr_rows: list[np.ndarray] = []
            for request in metadata.request_meta.values():
                if not request.dump_keys:
                    continue
                fa_dump_keys.extend(request.dump_keys)
                fa_ptr_rows.append(
                    self._extract_fa_ptr(
                        request.dump_keys,
                        request.dump_hash_start,
                        request.dump_hash_end,
                        request.dump_vllm_block_ids,
                    )
                )
                wa_dump_keys.extend(request.dump_keys[-1:])
                wa_ptr_rows.append(
                    self._extract_wa_ptr(
                        request.dump_keys[-1:],
                        request.dump_vllm_block_ids,
                    )
                )

            if not fa_dump_keys:
                return

            fa_ptrs = np.vstack(fa_ptr_rows)
            window_ptrs = np.vstack(wa_ptr_rows)
            tasks: list[FAWADumpTask] = []

            tasks.append(
                self._submit_dump_task(
                    "FA",
                    self.fa_store,
                    fa_dump_keys,
                    fa_ptrs,
                    event_handle,
                )
            )
            tasks.append(
                self._submit_dump_task(
                    "WA",
                    self.wa_store,
                    wa_dump_keys,
                    window_ptrs,
                    event_handle,
                )
            )
            for dump_task in tasks:
                self._wait_dump_task(dump_task)
        except Exception as e:
            logger.error(f"dump FAWA kv cache failed. {type(e).__name__}: {e}")
