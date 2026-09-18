import copy
import glob
import hashlib
import math
import os
import pickle
import re
import shutil
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from functools import wraps
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)

try:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorWorkerMetadata,
    )
except ImportError:
    KVConnectorWorkerMetadata = object

from vllm.distributed.parallel_state import get_world_group
from vllm.model_executor.models.utils import extract_layer_index
from vllm.platforms import current_platform
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput

from ucm.integration.vllm.device import create_device, get_current_device_id
from ucm.integration.vllm.metrics import (
    UCM_HAS_PROM_METRICS,
    UCMConnectorStats,
    UCMPromMetrics,
)
from ucm.integration.vllm.rank_consistency import RankConsistencyManager
from ucm.logger import init_logger
from ucm.metrics_config import (
    MULTIPROC_CONSUMER,
    VLLM_CONNECTOR_CONSUMER,
    consumer_enabled,
    get_vllm_connector_metric_definitions,
    load_launch_metrics_config,
    setup_ucm_metrics,
)
from ucm.metrics_dispatcher import get_metrics_dispatcher
from ucm.observability import PrometheusStatsLogger
from ucm.shared.metrics import ucmmetrics
from ucm.store.factory_v1 import UcmConnectorFactoryV1
from ucm.store.ucmstore_v1 import Task, UcmKVStoreBaseV1
from ucm.utils import Config

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
        KVConnectorPromMetrics,
        KVConnectorStats,
        PromMetric,
        PromMetricT,
    )
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

from ucm.sparse.state import has_ucm_sparse

logger = init_logger(__name__)


def _has_shared_indexer_layers(vllm_config: "VllmConfig") -> bool:
    model_config = getattr(vllm_config, "model_config", None)
    configs = (
        getattr(model_config, "hf_text_config", None),
        getattr(model_config, "hf_config", None),
        getattr(getattr(model_config, "hf_config", None), "text_config", None),
    )
    for config in configs:
        indexer_types = getattr(config, "indexer_types", None)
        if isinstance(indexer_types, (list, tuple)) and any(
            isinstance(indexer_type, str) and indexer_type.lower() == "shared"
            for indexer_type in indexer_types
        ):
            return True
    return False


def _short_list(values: list[int], limit: int = 12) -> list[int]:
    return values[:limit]


def _get_store_io_sizes(
    logical_shard_size: int,
    logical_block_size: int,
    alignment: int = 4096,
) -> tuple[int, int]:
    """Return aligned physical Store sizes."""
    if alignment <= 0:
        raise ValueError(f"Store I/O alignment must be positive, got {alignment}.")
    if logical_shard_size <= 0 or logical_block_size <= 0:
        raise ValueError(
            "Store shard_size and block_size must be positive, "
            f"got shard_size={logical_shard_size}, "
            f"block_size={logical_block_size}."
        )
    shard_count, remainder = divmod(logical_block_size, logical_shard_size)
    if remainder:
        raise ValueError(
            "Store block_size must be divisible by shard_size, "
            f"got shard_size={logical_shard_size}, "
            f"block_size={logical_block_size}."
        )

    physical_shard_size = (logical_shard_size + alignment - 1) // alignment * alignment
    return physical_shard_size, physical_shard_size * shard_count


def _get_store_gc_block_size(
    store_pipeline: str,
    tensor_size_list: list[int],
    store_shard_size: int,
    store_block_size: int,
) -> int:
    """Return the persisted file size used by scheduler-side Posix GC."""
    if store_pipeline != "YuanRong|Posix":
        return store_block_size
    shard_count, remainder = divmod(store_block_size, store_shard_size)
    if remainder:
        raise ValueError(
            "Store block_size must be divisible by shard_size, "
            f"got shard_size={store_shard_size}, block_size={store_block_size}."
        )
    object_size = sum(int(size) for size in tensor_size_list)
    if object_size <= 0:
        raise ValueError("tensor_size_list is required for YuanRong|Posix")
    return object_size * shard_count


_SHM_DIR = "/dev/shm"


def _check_shm_capacity(cache_buffer_capacity_gb: int) -> None:
    """Early-validate that /dev/shm can hold the shared-buffer store.

    With ``share_buffer_enable=True`` the cache buffer is backed by ``shm_open``
    in ``/dev/shm`` (a tmpfs with a fixed size limit). If the configured buffer
    capacity exceeds what ``/dev/shm`` can hold, allocation fails deep inside
    the C++ store; raise here instead with an actionable message.
    """
    if cache_buffer_capacity_gb <= 0:
        return
    try:
        shm_total = shutil.disk_usage(_SHM_DIR).total
    except OSError:
        # /dev/shm unavailable (e.g. non-Linux dev host); defer to the store.
        logger.debug("Skip /dev/shm capacity check: %s unavailable.", _SHM_DIR)
        return
    needed_bytes = cache_buffer_capacity_gb * (1 << 30)
    if shm_total < needed_bytes:
        raise RuntimeError(
            f"Shared-buffer cache requires {cache_buffer_capacity_gb}GB in {_SHM_DIR}, "
            f"but {_SHM_DIR} has only {shm_total >> 30}GB. "
            f"Either increase the size of {_SHM_DIR} (e.g. remount tmpfs with a "
            f"larger size= option) or decrease cache_buffer_capacity_gb."
        )


def _drop_null_vllm_blocks(
    ucm_block_ids: list[bytes],
    vllm_block_ids: list[int],
    context: str,
) -> tuple[list[bytes], list[int]]:
    if not ucm_block_ids or not vllm_block_ids:
        return ucm_block_ids, vllm_block_ids

    filtered_ucm_block_ids: list[bytes] = []
    filtered_vllm_block_ids: list[int] = []
    skipped = 0
    for ucm_block_id, vllm_block_id in zip(ucm_block_ids, vllm_block_ids):
        if vllm_block_id == 0:
            skipped += 1
            continue
        filtered_ucm_block_ids.append(ucm_block_id)
        filtered_vllm_block_ids.append(vllm_block_id)

    if skipped:
        logger.info(
            f"{context}: skipped {skipped} null vLLM block(s), "
            f"kept_vllm={_short_list(filtered_vllm_block_ids)}"
        )
    return filtered_ucm_block_ids, filtered_vllm_block_ids


def _record_counter(name: str, value: float = 1.0) -> None:
    ucmmetrics.update_stats({name: value})


def _record_connector_interface_duration(func):
    metric_name = f"connector_{func.__name__}_duration_ms"

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            duration_ms = (time.perf_counter() - start) * 1e3
            ucmmetrics.update_stats({metric_name: duration_ms})

    return wrapper


def _use_ucm_connector_cpu_affinity() -> bool:
    return (
        os.getenv("VLLM_CPU_AFFINITY") == "1"
        and getattr(current_platform, "device_type", None) != "npu"
    )


def _worker_generate_unique_id() -> str:
    """Worker-side: broadcast a uuid and write to a per-instance file."""
    world_group = get_world_group()
    if world_group.rank_in_group == 0:
        now = time.time()
        for f in glob.glob("/dev/shm/ucm_uniqueid_*"):
            try:
                if now - os.path.getmtime(f) > 600:
                    os.remove(f)
            except OSError:
                pass
        local_uid = uuid.uuid4().hex
    else:
        local_uid = None
    uid = world_group.broadcast_object(local_uid, src=0)
    path = f"/dev/shm/ucm_uniqueid_{os.getppid()}"
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(uid)
    os.replace(tmp, path)
    return uid


def _scheduler_read_unique_id() -> str:
    """Scheduler-side: read the uuid from the per-instance file."""
    for pid in (os.getpid(), os.getppid()):
        path = f"/dev/shm/ucm_uniqueid_{pid}"
        try:
            with open(path) as f:
                uid = f.read().strip()
            if uid:
                return uid
        except FileNotFoundError:
            continue
    raise RuntimeError(
        "scheduler-side UCM initialization failed: unique_id file not found"
    )


def _worker_publish_block_size(block_size: int, dp_rank: int) -> None:
    if dp_rank != 0 or get_world_group().rank_in_group != 0:
        return

    now = time.time()
    for stale in glob.glob("/dev/shm/ucm_blocksize_*"):
        try:
            if now - os.path.getmtime(stale) > 600:
                os.remove(stale)
        except OSError:
            pass

    path = f"/dev/shm/ucm_blocksize_{os.getppid()}"
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(str(int(block_size)))
    os.replace(tmp, path)
    logger.info(
        f"Published store block_size {block_size} to {path} "
        f"(pid={os.getpid()}, ppid={os.getppid()})."
    )


def _scheduler_read_block_size() -> int | None:
    for pid in (os.getpid(), os.getppid()):
        path = f"/dev/shm/ucm_blocksize_{pid}"
        try:
            with open(path) as f:
                content = f.read().strip()
        except (FileNotFoundError, OSError):
            continue
        try:
            block_size = int(content)
        except ValueError:
            logger.warning(
                f"block_size file {path} holds a non-integer value "
                f"{content!r}, fall back to manual estimate"
            )
            return None
        if block_size <= 0:
            logger.warning(
                f"block_size file {path} holds a non-positive value "
                f"{block_size}, fall back to manual estimate"
            )
            return None
        logger.info(
            f"Read worker-published store block_size {block_size} from {path} "
            f"(pid={os.getpid()}, ppid={os.getppid()})."
        )
        return block_size
    logger.warning(
        "block_size file not found "
        f"(looked for /dev/shm/ucm_blocksize_{os.getpid()} and "
        f"/dev/shm/ucm_blocksize_{os.getppid()}), fall back to manual estimate"
    )
    return None


@dataclass
class RequestMeta:
    ucm_block_ids: list[bytes] = field(default_factory=list)
    hbm_hit_block_num: int = 0
    # local_computed_block + external_computed_block
    total_hit_block_num: int = 0
    num_token_ids: int = 0
    vllm_block_ids: list[int] = field(default_factory=list)
    token_processed: int = 0


@dataclass
class RequestDispatchMeta:
    load_block_ids: tuple[
        list[bytes], list[int]
    ]  # [0] mean ucm_block_ids, [1] means vllm_block_ids
    dump_block_ids: tuple[list[bytes], list[int]]


@dataclass(frozen=True)
class KVCacheTensorInfo:
    ptr: int
    bytes_per_block: int
    buffer_size: int


@dataclass(frozen=True)
class SharedIndexerLayerInfo:
    layer_id: int
    sfa_tensors: tuple[KVCacheTensorInfo, ...]
    indexer: Optional[KVCacheTensorInfo]
    scale: Optional[KVCacheTensorInfo]


@dataclass(frozen=True)
class KVCacheSegment:
    ptr: int
    copy_size: int
    block_stride: int
    buffer_size: int


class KVCacheLayout:
    def __init__(
        self,
        kvcaches,
        ucm_config: dict,
        vllm_config: "VllmConfig",
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        # Direct mode stores only real tensors in flat arrays. Generic
        # layerwise mode intentionally supports regular tensor matrices only;
        # model families with checkpoint-level shared indexers use the
        # dedicated SharedIndexerKVCacheLayout below.
        self.base_ptrs: np.ndarray
        self.buffer_sizes: np.ndarray
        self.tensor_size_lists: np.ndarray
        self.block_stride_lists: np.ndarray
        self.use_layerwise = ucm_config.get("use_layerwise", True)
        self.kv_cache_config = kv_cache_config
        self.vllm_config = vllm_config
        self.pp_size = self.vllm_config.parallel_config.pipeline_parallel_size
        self.num_hidden_layers = getattr(
            self.vllm_config.model_config.hf_text_config, "num_hidden_layers", 0
        )
        if self.pp_size > 1 and self.num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be > 0 when pp_size > 1")
        self.layer_name_to_id = {
            name: extract_layer_index(name) for name in kvcaches.keys()
        }
        self.first_layer_id = next(iter(self.layer_name_to_id.values()))
        self.num_blocks = self.kv_cache_config.num_blocks
        self._build_layout(kvcaches)

    def _tensor_infos(
        self,
        layer_name: str,
        kv_layer,
    ) -> tuple[KVCacheTensorInfo, ...]:
        """Describe tensor components in one vLLM KV-cache entry."""
        tensor_infos = []

        def handle_tensor(tensor: torch.Tensor) -> None:
            bytes_per_block = math.prod(int(dim) for dim in tensor.shape[1:]) * int(
                tensor.element_size()
            )
            tensor_infos.append(
                KVCacheTensorInfo(
                    ptr=int(tensor[0].data_ptr()),
                    bytes_per_block=bytes_per_block,
                    buffer_size=int(tensor.shape[0]) * bytes_per_block,
                )
            )

        if isinstance(kv_layer, torch.Tensor):
            if kv_layer.dim() == 5 and kv_layer.shape[0] == 2:
                if kv_layer.shape[1] == 2 and kv_layer.shape[1] >= self.num_blocks:
                    logger.warning(
                        "Ambiguous KV cache layout with two leading dimensions "
                        "of size 2; treating it as K/V-first: layer=%s, "
                        "shape=%s, num_blocks=%s.",
                        layer_name,
                        tuple(kv_layer.shape),
                        self.num_blocks,
                    )
                # Legacy K/V-first layout: [2, physical_blocks, ...].
                handle_tensor(kv_layer[0])
                handle_tensor(kv_layer[1])
            else:
                # A block-first tensor stores one complete page per row.
                handle_tensor(kv_layer)
        elif isinstance(kv_layer, (tuple, list)):
            for tensor in kv_layer:
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError(
                        "KV cache component must be a tensor: "
                        f"layer={layer_name}, type={type(tensor)}."
                    )
                handle_tensor(tensor)
        else:
            raise TypeError(
                f"Unsupported KV cache type: layer={layer_name}, "
                f"type={type(kv_layer)}."
            )
        return tuple(tensor_infos)

    def _collect_tensor_rows(self, kvcaches):
        num_rows = len(set(self.layer_name_to_id.values()))
        tensor_rows = [[] for _ in range(num_rows)]
        row_layer_ids = [None for _ in range(num_rows)]

        for layer_name, kv_layer in kvcaches.items():
            local_layer_id = self.layer_name_to_id[layer_name] - self.first_layer_id
            row_layer_ids[local_layer_id] = self.layer_name_to_id[layer_name]
            tensor_rows[local_layer_id].extend(self._tensor_infos(layer_name, kv_layer))

        return tensor_rows, row_layer_ids

    @staticmethod
    def _unpack_tensor_rows(tensor_rows):
        ptr_rows = [[tensor.ptr for tensor in row] for row in tensor_rows]
        stride_rows = [
            [tensor.bytes_per_block for tensor in row] for row in tensor_rows
        ]
        buffer_size_rows = [
            [tensor.buffer_size for tensor in row] for row in tensor_rows
        ]
        return ptr_rows, stride_rows, buffer_size_rows

    def _build_layout(self, kvcaches):
        tensor_rows, row_layer_ids = self._collect_tensor_rows(kvcaches)
        raw_ptr_rows, stride_rows, buffer_size_rows = self._unpack_tensor_rows(
            tensor_rows
        )

        if not self.use_layerwise:
            # A direct transfer copies one whole KV-cache block. Keep only real
            # tensors and flatten all layers into that block; padding would add
            # fake tensors to the store metadata and copy path.
            self.base_ptrs = np.asarray(
                [ptr for row in raw_ptr_rows for ptr in row], dtype=np.uint64
            )
            self.tensor_size_lists = np.asarray(
                [stride for row in stride_rows for stride in row], dtype=np.uint64
            )
            self.buffer_sizes = np.asarray(
                [size for row in buffer_size_rows for size in row], dtype=np.uint64
            )
            self.block_stride_lists = self.tensor_size_lists
            logger.info(
                f"base_ptrs: {self.base_ptrs.shape}, "
                f"tensor_size_lists: {self.tensor_size_lists.shape}"
            )
            return

        if not tensor_rows or not tensor_rows[0]:
            raise ValueError("KV cache layout must contain at least one tensor")

        expected_strides = stride_rows[0]
        incompatible_rows = {
            row_layer_ids[row_id]: row
            for row_id, row in enumerate(stride_rows)
            if row != expected_strides
        }
        if incompatible_rows:
            raise ValueError(
                "Invalid generic KV cache layout for `use_layerwise: true` in "
                "ucm_config.yaml: every layer must have the same tensor count "
                "and per-block tensor sizes. "
                f"expected={expected_strides} from layer {row_layer_ids[0]}, "
                f"incompatible_layers={incompatible_rows}. Models with "
                "checkpoint-level shared indexers require "
                "SharedIndexerKVCacheLayout; otherwise set `use_layerwise: false`."
            )

        self.base_ptrs = np.asarray(raw_ptr_rows, dtype=np.uint64)
        self.tensor_size_lists = np.asarray(stride_rows, dtype=np.uint64)
        self.buffer_sizes = np.asarray(buffer_size_rows, dtype=np.uint64)
        self.block_stride_lists = self.tensor_size_lists

        logger.info(
            f"base_ptrs: {self.base_ptrs.shape}, tensor_size_lists: {self.tensor_size_lists.shape}"
        )

    def extract_block_addrs(
        self, vllm_block_ids: List[int], layer_first: bool = False
    ) -> np.ndarray:
        vllm_block_ids_np = np.array(vllm_block_ids, np.uint64)
        if not self.use_layerwise:
            if layer_first:
                raise ValueError(
                    "layer_first=True requires a layerwise KV cache layout"
                )
            return (
                vllm_block_ids_np[:, None] * self.block_stride_lists[None, :]
                + self.base_ptrs[None, :]
            )

        if layer_first:
            # (n_layers, num_blocks, n_ptrs)
            return (
                self.block_stride_lists[:, None, :] * vllm_block_ids_np[None, :, None]
                + self.base_ptrs[:, None, :]
            )
        return (
            vllm_block_ids_np[:, None, None] * self.block_stride_lists[None, :, :]
            + self.base_ptrs[None, :, :]
        )  # (num_blocks, n_layers, n_ptrs)

    @property
    def tensor_size_list(self) -> list[int]:
        return (
            self.tensor_size_lists.reshape(-1).tolist()
            if not self.use_layerwise
            else self.tensor_size_lists[0].tolist()
        )

    @property
    def shard_size(self) -> int:
        return int(
            self.tensor_size_lists.sum()
            if not self.use_layerwise
            else self.tensor_size_lists[0].sum()
        )

    @property
    def block_size(self) -> int:
        if self.pp_size > 1:
            return int(self.tensor_size_lists[0].sum() * self.num_hidden_layers)
        return int(self.tensor_size_lists.sum())


class SharedIndexerKVCacheLayout(KVCacheLayout):
    """Layerwise layout for checkpoint-level shared Indexer models.

    Missing semantic slots use ``base_ptr=0`` and ``block_stride=0``. This keeps
    their computed device address at nullptr for every block without adding
    mask behavior to the generic KV cache layout.
    """

    TENSOR_ROLE_PATTERNS = {
        "indexer": (
            re.compile(
                r"(?:^|\.)indexer(?:\.|$)",
                re.IGNORECASE,
            ),
        ),
    }
    DEFAULT_TENSOR_ROLE = "attention"

    @classmethod
    def supports(cls, vllm_config: "VllmConfig", ucm_config: dict) -> bool:
        device_type = getattr(current_platform, "device_type", None)
        return (
            bool(ucm_config.get("use_layerwise", True))
            and device_type in ("npu", "cuda")
            and _has_shared_indexer_layers(vllm_config)
        )

    @staticmethod
    def _ghost_segment(slot_size: int) -> KVCacheSegment:
        return KVCacheSegment(
            ptr=0,
            copy_size=int(slot_size),
            block_stride=0,
            buffer_size=0,
        )

    @staticmethod
    def _real_segment(
        ptr: int,
        copy_size: int,
        block_stride: int,
        buffer_size: int,
    ) -> KVCacheSegment:
        return KVCacheSegment(
            ptr=int(ptr),
            copy_size=int(copy_size),
            block_stride=int(block_stride),
            buffer_size=int(buffer_size),
        )

    @staticmethod
    def _whole_tensor_segment(tensor: KVCacheTensorInfo) -> KVCacheSegment:
        return KVCacheSegment(
            ptr=tensor.ptr,
            copy_size=tensor.bytes_per_block,
            block_stride=tensor.bytes_per_block,
            buffer_size=tensor.buffer_size,
        )

    @staticmethod
    def _validate_uniform_sizes(
        values: list[int], description: str, row_layer_ids: list[int]
    ) -> int:
        unique_values = sorted(set(values))
        if len(unique_values) != 1:
            raise ValueError(
                f"Shared Indexer KV cache layout requires one {description}, "
                f"but got {unique_values}; layers={row_layer_ids}."
            )
        return unique_values[0]

    @staticmethod
    def _layer_type(layer: SharedIndexerLayerInfo) -> str:
        if layer.scale is not None:
            return "li_c8"
        if layer.indexer is not None:
            return "bf16"
        return "shared"

    @classmethod
    def _cache_role(cls, layer_name: str) -> str:
        """Classify an attention or Indexer cache independently of platform."""
        path_components = [
            component for component in layer_name.lower().split(".") if component
        ]
        for index, component in enumerate(path_components[:-1]):
            if component == "layers" and path_components[index + 1].isdigit():
                path_components = path_components[index + 2 :]
                break
        semantic_name = ".".join(path_components)
        for role, patterns in cls.TENSOR_ROLE_PATTERNS.items():
            if any(pattern.search(semantic_name) for pattern in patterns):
                return role
        return cls.DEFAULT_TENSOR_ROLE

    def _collect_role_tensors(self, kvcaches, tensor_infos):
        """Group cache entries by layer and semantic role."""
        layer_tensors: dict[int, dict[str, Optional[tuple[KVCacheTensorInfo, ...]]]] = (
            {}
        )
        for layer_name, kv_layer in kvcaches.items():
            layer_id = self.layer_name_to_id[layer_name]
            layer = layer_tensors.setdefault(
                layer_id, {"attention": None, "indexer": None}
            )
            slot = self._cache_role(layer_name)
            if layer[slot] is not None:
                raise ValueError(
                    f"Duplicate Shared Indexer {slot} cache for "
                    f"layer {layer_id}: {layer_name}."
                )
            layer[slot] = tensor_infos(layer_name, kv_layer)
        return layer_tensors

    def _build_segment_rows(
        self,
        layers: list[SharedIndexerLayerInfo],
        layout_mode: str,
        *,
        indexer_size: int = 0,
        index_chunk_size: int = 0,
        scale_size: int = 0,
    ) -> list[list[KVCacheSegment]]:
        """Build fixed-width rows shared by CUDA and Ascend layouts."""
        segment_rows = []
        for layer in layers:
            segments = [
                self._whole_tensor_segment(tensor) for tensor in layer.sfa_tensors
            ]

            if layout_mode == "bf16":
                if layer.indexer is not None:
                    segments.append(self._whole_tensor_segment(layer.indexer))
                else:
                    segments.append(self._ghost_segment(indexer_size))
            elif layout_mode == "li_c8":
                if layer.scale is not None:
                    segments.extend(
                        [
                            self._whole_tensor_segment(layer.indexer),
                            self._whole_tensor_segment(layer.scale),
                        ]
                    )
                else:
                    segments.extend(
                        [
                            self._ghost_segment(index_chunk_size),
                            self._ghost_segment(scale_size),
                        ]
                    )
            elif layout_mode == "mixed" and layer.scale is not None:
                segments.extend(
                    [
                        self._real_segment(
                            ptr=layer.indexer.ptr,
                            copy_size=index_chunk_size,
                            block_stride=layer.indexer.bytes_per_block,
                            buffer_size=layer.indexer.buffer_size,
                        ),
                        self._ghost_segment(index_chunk_size),
                        self._whole_tensor_segment(layer.scale),
                    ]
                )
            elif layout_mode == "mixed" and layer.indexer is not None:
                if layer.indexer.bytes_per_block != 2 * index_chunk_size:
                    raise ValueError(
                        "Cannot split BF16 Indexer tensor into two C8-sized "
                        f"segments: bf16_size={layer.indexer.bytes_per_block}, "
                        f"c8_size={index_chunk_size}."
                    )
                segments.extend(
                    [
                        self._real_segment(
                            ptr=layer.indexer.ptr,
                            copy_size=index_chunk_size,
                            block_stride=layer.indexer.bytes_per_block,
                            buffer_size=layer.indexer.buffer_size,
                        ),
                        self._real_segment(
                            ptr=layer.indexer.ptr + index_chunk_size,
                            copy_size=index_chunk_size,
                            block_stride=layer.indexer.bytes_per_block,
                            buffer_size=0,
                        ),
                        self._ghost_segment(scale_size),
                    ]
                )
            elif layout_mode == "mixed":
                segments.extend(
                    [
                        self._ghost_segment(index_chunk_size),
                        self._ghost_segment(index_chunk_size),
                        self._ghost_segment(scale_size),
                    ]
                )
            else:
                raise ValueError(
                    f"Unsupported Shared Indexer layout mode: {layout_mode}."
                )
            segment_rows.append(segments)
        return segment_rows

    def _set_segment_rows(self, segment_rows: list[list[KVCacheSegment]]) -> None:
        """Materialize segment metadata in the arrays consumed by the Store."""
        self.base_ptrs = np.asarray(
            [[segment.ptr for segment in row] for row in segment_rows],
            dtype=np.uint64,
        )
        self.tensor_size_lists = np.asarray(
            [[segment.copy_size for segment in row] for row in segment_rows],
            dtype=np.uint64,
        )
        self.block_stride_lists = np.asarray(
            [[segment.block_stride for segment in row] for row in segment_rows],
            dtype=np.uint64,
        )
        self.buffer_sizes = np.asarray(
            [[segment.buffer_size for segment in row] for row in segment_rows],
            dtype=np.uint64,
        )

    def _build_layout(self, kvcaches):
        """Dispatch platform-specific collection into the shared row builder."""
        if not self.use_layerwise:
            super()._build_layout(kvcaches)
            return

        if getattr(current_platform, "device_type", None) == "cuda":
            self._build_cuda_layout(kvcaches)
        else:
            self._build_ascend_layout(kvcaches)

    def _build_cuda_layout(self, kvcaches) -> None:
        """Build CUDA rows as fixed ``[Attention, Indexer]`` shards.

        Layers without an independent Indexer keep the same copy schema by
        receiving a metadata-only ghost segment in the Indexer slot.
        """
        layer_tensors = self._collect_role_tensors(kvcaches, self._tensor_infos)

        row_layer_ids = sorted(layer_tensors)
        self.first_layer_id = row_layer_ids[0]

        layers = []
        for layer_id in row_layer_ids:
            attention = layer_tensors[layer_id]["attention"]
            if attention is None:
                raise ValueError(
                    "CUDA Shared Indexer KV cache layer has no Attention "
                    f"tensor: layer={layer_id}."
                )
            indexer_tensors = layer_tensors[layer_id]["indexer"]
            if indexer_tensors is not None and len(indexer_tensors) != 1:
                raise ValueError(
                    "CUDA Shared Indexer layer must have one Indexer tensor: "
                    f"layer={layer_id}, count={len(indexer_tensors)}."
                )
            layers.append(
                SharedIndexerLayerInfo(
                    layer_id=layer_id,
                    sfa_tensors=attention,
                    indexer=indexer_tensors[0] if indexer_tensors else None,
                    scale=None,
                )
            )

        attention_sizes = [tensor.bytes_per_block for tensor in layers[0].sfa_tensors]
        incompatible_attention_layers = {}
        for layer in layers[1:]:
            current_attention_sizes = [
                tensor.bytes_per_block for tensor in layer.sfa_tensors
            ]
            if current_attention_sizes != attention_sizes:
                incompatible_attention_layers[layer.layer_id] = current_attention_sizes
        if incompatible_attention_layers:
            raise ValueError(
                "CUDA Shared Indexer layers must have the same Attention slot "
                "count and per-block sizes: "
                f"expected={attention_sizes} from layer {layers[0].layer_id}, "
                f"incompatible_layers={incompatible_attention_layers}."
            )

        indexer_layers = [layer for layer in layers if layer.indexer is not None]
        if not indexer_layers:
            raise ValueError(
                "CUDA Shared Indexer KV cache layout did not find any "
                "independent Indexer layer."
            )
        indexer_size = self._validate_uniform_sizes(
            [layer.indexer.bytes_per_block for layer in indexer_layers],
            "CUDA Indexer block size",
            [layer.layer_id for layer in indexer_layers],
        )

        segment_rows = self._build_segment_rows(
            layers, layout_mode="bf16", indexer_size=indexer_size
        )
        self._set_segment_rows(segment_rows)
        logger.info(
            "CUDA Shared Indexer layerwise KV cache layout: "
            f"slot_sizes={attention_sizes + [indexer_size]}, "
            f"layer_types={[self._layer_type(layer) for layer in layers]}"
        )

    def _build_ascend_layout(self, kvcaches) -> None:
        """Build Ascend SFA C8/BF16/mixed layerwise layout by cache role.

        vLLM-Ascend used to append an optional Indexer cache after the MLA
        cache in one tuple. Newer versions register it under an independent
        ``*.indexer.*`` cache entry. Grouping only by numeric layer id makes
        the latter dependent on dictionary insertion order, so classify the
        entries before assembling the fixed UCM slot order.
        """

        additional_config = getattr(self.vllm_config, "additional_config", None) or {}
        sfa_c8 = bool(additional_config.get("enable_sparse_sfa_c8", False))
        li_c8 = bool(additional_config.get("enable_sparse_li_c8", False))
        base_tensor_count = 1 if sfa_c8 else 2
        max_extra_count = 2 if li_c8 else 1

        layer_tensors = self._collect_role_tensors(kvcaches, self._tensor_infos)

        if not layer_tensors:
            raise ValueError("KV cache layout must contain at least one tensor")

        row_layer_ids = sorted(layer_tensors)
        self.first_layer_id = row_layer_ids[0]
        layers = []
        for layer_id in row_layer_ids:
            attention = layer_tensors[layer_id]["attention"]
            if attention is None or len(attention) < base_tensor_count:
                count = 0 if attention is None else len(attention)
                raise ValueError(
                    "Ascend Shared Indexer layer has insufficient SFA base "
                    f"tensors: layer={layer_id}, expected={base_tensor_count}, "
                    f"actual={count}."
                )

            # Legacy versions keep Indexer/scale after the base SFA tensors in
            # the attention tuple. Newer versions expose them in an independent
            # Indexer entry. Seeing both would be ambiguous and must not result
            # in a silent duplicate store segment.
            legacy_extras = attention[base_tensor_count:]
            separate_indexer = layer_tensors[layer_id]["indexer"]
            if legacy_extras and separate_indexer is not None:
                raise ValueError(
                    "Ascend Shared Indexer layer exposes both legacy and "
                    f"separate Indexer caches: layer={layer_id}."
                )
            extras = separate_indexer if separate_indexer is not None else legacy_extras
            if len(extras) > max_extra_count:
                raise ValueError(
                    "Unsupported Ascend Shared Indexer cache components: "
                    f"layer={layer_id}, count={len(extras)}."
                )
            layers.append(
                SharedIndexerLayerInfo(
                    layer_id=layer_id,
                    sfa_tensors=tuple(attention[:base_tensor_count]),
                    indexer=extras[0] if extras else None,
                    scale=extras[1] if len(extras) == 2 else None,
                )
            )

        base_sizes = [tensor.bytes_per_block for tensor in layers[0].sfa_tensors]
        for layer in layers:
            current_sizes = [tensor.bytes_per_block for tensor in layer.sfa_tensors]
            if current_sizes != base_sizes:
                raise ValueError(
                    "Shared Indexer layers must have identical SFA base tensors: "
                    f"expected={base_sizes}, layer={layer.layer_id}, "
                    f"actual={current_sizes}."
                )

        c8_layers = [layer for layer in layers if layer.scale is not None]
        bf16_layers = [
            layer
            for layer in layers
            if layer.indexer is not None and layer.scale is None
        ]

        if c8_layers and bf16_layers:
            layout_mode = "mixed"
        elif c8_layers:
            layout_mode = "li_c8"
        elif bf16_layers:
            layout_mode = "bf16"
        else:
            raise ValueError(
                "Shared Indexer KV cache layout did not find any full " "Indexer layer."
            )

        indexer_size = 0
        index_chunk_size = 0
        scale_size = 0
        if c8_layers:
            index_chunk_size = self._validate_uniform_sizes(
                [layer.indexer.bytes_per_block for layer in c8_layers],
                "C8 Indexer block size",
                row_layer_ids,
            )
            scale_size = self._validate_uniform_sizes(
                [layer.scale.bytes_per_block for layer in c8_layers],
                "LI C8 scale block size",
                row_layer_ids,
            )
        if layout_mode == "mixed":
            slot_sizes = base_sizes + [
                index_chunk_size,
                index_chunk_size,
                scale_size,
            ]
        elif layout_mode == "li_c8":
            slot_sizes = base_sizes + [index_chunk_size, scale_size]
        else:
            indexer_size = self._validate_uniform_sizes(
                [layer.indexer.bytes_per_block for layer in bf16_layers],
                "BF16 Indexer block size",
                row_layer_ids,
            )
            slot_sizes = base_sizes + [indexer_size]

        segment_rows = self._build_segment_rows(
            layers,
            layout_mode,
            indexer_size=indexer_size,
            index_chunk_size=index_chunk_size,
            scale_size=scale_size,
        )
        self._set_segment_rows(segment_rows)

        logger.info(
            "Shared Indexer layerwise KV cache layout: "
            f"sfa_c8={sfa_c8}, li_c8={li_c8}, "
            f"layout_mode={layout_mode}, "
            f"slot_sizes={slot_sizes}, "
            f"layer_types={[self._layer_type(layer) for layer in layers]}"
        )


@dataclass
class UCMConnectorMetadata(KVConnectorMetadata):
    request_meta: dict[str, RequestDispatchMeta] = field(default_factory=dict)
    preempted_req_ids: set[str] = field(default_factory=set)


@dataclass
class PendingDumpTask:
    task: Task
    request_ids: set[str]
    event_handle: int = 0
    wait_for_save_start_ms: float = 0.0


class RequestHasher:
    """hash(md5) request to generate ucm block id"""

    def __init__(self, vllm_config, rank_id):
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
        self.meta_bytes = meta.encode("utf-8")

    def __call__(self, input_data) -> bytes:
        if isinstance(input_data, bytes):
            input_bytes = input_data
        else:
            input_bytes = pickle.dumps(input_data, protocol=pickle.HIGHEST_PROTOCOL)

        h = hashlib.md5(self.meta_bytes + input_bytes)
        return h.digest()


@dataclass
class UCMWorkerMetadata(KVConnectorWorkerMetadata):
    """Worker-to-scheduler metadata for load failures and Dump success."""

    is_mla: bool = False
    load_failed_reqs: set[str] = field(default_factory=set)
    missing_reqs: set[str] = field(default_factory=set)
    missing_blocks: set[bytes] = field(default_factory=set)
    dump_succeeded_blocks: set[bytes] = field(default_factory=set)

    def mark_failed(self, req_id: str) -> None:
        """Record a failed load request from this worker."""
        self.load_failed_reqs.add(req_id)

    def mark_missing(self, req_id: str, rank0_block_ids: list[bytes]) -> None:
        self.missing_reqs.add(req_id)
        self.missing_blocks.update(rank0_block_ids)

    def aggregate(self, other: Any) -> Any:
        assert isinstance(other, UCMWorkerMetadata)
        assert self.is_mla == other.is_mla

        self.load_failed_reqs.update(other.load_failed_reqs)
        self.missing_reqs.update(other.missing_reqs)
        self.missing_blocks.update(other.missing_blocks)
        # TODO: Support PP-aware Dump success aggregation.
        # for mla, blocks can only be dumped by 1 rank, for FAWA, the dump is distributed across ranks
        # for non-mla, blocks should be dumped by all ranks
        # so for mla, aggregation logic is union, for non-mla, aggregation is intersection
        if self.is_mla:
            self.dump_succeeded_blocks.update(other.dump_succeeded_blocks)
        else:
            self.dump_succeeded_blocks.intersection_update(other.dump_succeeded_blocks)
        return self


class UCMDirectConnector(KVConnectorBase_V1):
    """
    This connector means synchronize:
    load -> forward -> save

    Store operations that participate in rank consistency must use
    RankConsistencyManager. It owns StoreNotFoundError classification and
    cross-rank consistency metadata.
    """

    @staticmethod
    def _consistency_manager_enabled(launch_config: dict, is_mla: bool) -> bool:
        return launch_config.get("use_consistency_manager", not is_mla)

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self.use_layerwise = False
        self.kv_caches: dict[str, torch.Tensor] = {}
        self.local_rank = (
            -1 if role == KVConnectorRole.SCHEDULER else get_world_group().local_rank
        )
        self.device_id = (
            -1 if role == KVConnectorRole.SCHEDULER else get_current_device_id()
        )
        if role != KVConnectorRole.SCHEDULER:
            logger.info(
                "UCM worker device mapping: local_rank=%s, device_id=%s",
                self.local_rank,
                self.device_id,
            )
        self.tp_rank = self._vllm_config.parallel_config.rank
        self.block_size = self._vllm_config.cache_config.block_size
        self.is_mla = self._vllm_config.model_config.is_deepseek_mla
        self.num_layers = self._vllm_config.model_config.get_num_layers(
            self._vllm_config.parallel_config
        )
        self.tp_size = self._vllm_config.parallel_config.tensor_parallel_size
        self.kv_cache_dtype: torch.dtype = None
        self.num_head = vllm_config.model_config.get_num_kv_heads(
            vllm_config.parallel_config
        )
        self.head_size = vllm_config.model_config.get_head_size()
        self.element_size = vllm_config.model_config.dtype.itemsize
        self.use_compress = hasattr(
            self._vllm_config.model_config.hf_config, "compress_ratios"
        )
        self._kv_cache_config = kv_cache_config

        if current_platform.is_cuda_alike():
            logger.info("CUDA device is available.")
            torch_dev = torch
            dev_name = "cuda"
        elif current_platform.device_type == "npu":
            logger.info("NPU device is available.")
            torch_dev = torch.npu
            dev_name = "npu"
        elif current_platform.device_type == "cpu":
            logger.info("CPU device is available.")
            torch_dev = torch
            dev_name = "cpu"
        else:
            raise RuntimeError("Unsupported device platform for UCMDirectConnector.")

        if self.device_id >= 0:
            self.device = torch_dev.device(f"{dev_name}:{self.device_id}")

        self.store: UcmKVStoreBaseV1
        self.rope_store: Optional[UcmKVStoreBaseV1] = None

        # save block info, avoid hash request twice, and track them until request finished
        self.requests_meta: dict[str, RequestMeta] = {}

        ucm_config = Config(vllm_config.kv_transfer_config)
        self.engine_id = vllm_config.kv_transfer_config.engine_id.rsplit("_dp", 1)[0]
        self.launch_config = ucm_config.get_config()
        self.connector_configs = self.launch_config.get("ucm_connectors", [])
        assert len(self.connector_configs) > 0, "no storage connector name in config."
        share_buffer_enable = (
            self.connector_configs[0]
            .get("ucm_connector_config", {})
            .get("share_buffer_enable", self.is_mla)
        )
        if share_buffer_enable:
            if role == KVConnectorRole.WORKER:
                self.unique_id = _worker_generate_unique_id()
            else:
                self.unique_id = _scheduler_read_unique_id()
        else:
            self.unique_id = self.engine_id
        self.enable_event_sync = self.launch_config.get("enable_event_sync", True)
        self.enable_record_traces = self.launch_config.get(
            "enable_record_traces", False
        )
        self._skip_null_vllm_blocks = False

        self.chunk_size = self.block_size
        self.blocks_per_chunk = self.chunk_size // self.block_size
        self._other_rank_hashers: list[RequestHasher] = []

        defer_scheduler_store = getattr(self, "_defer_scheduler_store", False)
        if role == KVConnectorRole.SCHEDULER:
            self.request_hasher = RequestHasher(vllm_config, 0)
            self._other_rank_hashers = self._make_other_rank_hashers(vllm_config)
            self._seed = self.request_hasher("UCM_HASH_SEED")
            # init scheduler-side connector
            if not defer_scheduler_store:
                self.store = self._create_store(None)
        else:
            self.request_hasher = RequestHasher(
                vllm_config,
                self.tp_rank % self.tp_size,
            )
            self._connector_worker_meta = UCMWorkerMetadata(is_mla=self.is_mla)

        self._rank_consistency = RankConsistencyManager(
            is_scheduler=role == KVConnectorRole.SCHEDULER,
            use_consistency_manager=self._consistency_manager_enabled(
                self.launch_config, self.is_mla
            ),
        )

        self.persist_token_threshold = self.launch_config.get(
            "persist_token_threshold", 0
        )

        # invalid block ids due to load errors
        self._invalid_block_ids: set[int] = set()
        self._async_dump_req_ids: set[str] = set()
        self._pending_dump_tasks: list[PendingDumpTask] = []
        self.cp_world_size = 1
        self.hash_block_size = self.block_size
        self.block_size *= self.cp_world_size

    def get_block_size(self) -> int:
        return self.block_size

    def _get_full_hit_recompute_tokens(self) -> int:
        cached = getattr(self, "_recompute_tokens", None)
        if cached is not None:
            return cached
        if not self.use_layerwise:
            cached = 1
        else:
            speculative_config = getattr(self._vllm_config, "speculative_config", None)
            if speculative_config is None:
                cached = 2
            else:
                spec_token_num = getattr(
                    speculative_config, "num_speculative_tokens", 0
                )
                try:
                    spec_token_num = int(spec_token_num)
                except (TypeError, ValueError):
                    spec_token_num = 0
                cached = max(spec_token_num, 0) + 2
        self._recompute_tokens = cached
        return cached

    @staticmethod
    def _record_counter(name: str, value: float = 1.0) -> None:
        _record_counter(name, value)

    def _make_other_rank_hashers(self, vllm_config) -> list[RequestHasher]:
        if self.is_mla:
            return []
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        return [RequestHasher(vllm_config, rank_id) for rank_id in range(1, tp_size)]

    def _record_load_error(self, metric_name: str, block_ids: Any) -> None:
        invalid_blocks = set(block_ids)
        new_invalid_blocks = invalid_blocks - self._invalid_block_ids
        self._invalid_block_ids.update(invalid_blocks)
        ucmmetrics.update_stats(
            {
                metric_name: 1.0,
                "connector_load_invalid_requests_total": 1.0,
                "connector_load_invalid_blocks_total": float(len(new_invalid_blocks)),
            }
        )

    def generate_hash(
        self, block_size: int, token_ids: List[int], parent_block_hash_value: bytes
    ) -> list[bytes]:
        ret = []
        for start in range(0, len(token_ids), block_size):
            end = start + block_size
            block_token_ids = token_ids[start:end]
            # Do not hash the block if it is not full.
            if len(block_token_ids) < block_size:
                break

            block_token_ids_tuple = tuple(block_token_ids)
            hash_value = self.request_hasher(
                (parent_block_hash_value, block_token_ids_tuple)
            )
            parent_block_hash_value = hash_value
            ret.append(hash_value)

        return ret

    def _set_default_shm_buffer_capacity(self, config: dict[str, Any]) -> None:
        if not bool(config.get("share_buffer_enable", False)):
            return
        if config.get("cache_buffer_capacity_gb") is None:
            config["cache_buffer_capacity_gb"] = 128
            logger.info(
                "Set cache_buffer_capacity_gb to 128GB for shared-buffer store."
            )
        # The shared buffer is allocated via shm_open in /dev/shm; fail early
        # (before store creation) if the tmpfs cannot hold it.
        _check_shm_capacity(int(config["cache_buffer_capacity_gb"]))

    def _create_store(
        self,
        kv_cache_layout: Optional[KVCacheLayout],
        cpu_affinity_cores: Optional[list[int]] = None,
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
        config["unique_id"] = f"{self.unique_id}"
        if self._role == KVConnectorRole.WORKER:
            config["device_id"] = self.device_id
            tensor_size_list = kv_cache_layout.tensor_size_list * self.blocks_per_chunk
            logical_shard_size = kv_cache_layout.shard_size * self.blocks_per_chunk
            logical_block_size = kv_cache_layout.block_size * self.blocks_per_chunk
            store_shard_size, store_block_size = _get_store_io_sizes(
                logical_shard_size,
                logical_block_size,
            )
            config["tensor_size_list"] = tensor_size_list
            config["shard_size"] = store_shard_size
            config["block_size"] = store_block_size
            gc_block_size = _get_store_gc_block_size(
                str(config.get("store_pipeline", "")),
                tensor_size_list,
                store_shard_size,
                store_block_size,
            )
            self._publish_block_size(gc_block_size)
            if store_shard_size != logical_shard_size:
                logger.info(
                    "Pad Store sizes for I/O alignment: "
                    f"logical_shard_size={logical_shard_size}, "
                    f"store_shard_size={store_shard_size}, "
                    f"logical_block_size={logical_block_size}, "
                    f"store_block_size={store_block_size}."
                )
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

    @property
    def _dp_rank(self) -> int:
        return self._vllm_config.parallel_config.data_parallel_rank

    @property
    def _gc_owner(self) -> bool:
        """Whether this connector's store runs block GC.

        Only the DP0 scheduler does, which is also the only side that needs a
        real block_size.
        """
        return self._role != KVConnectorRole.WORKER and self._dp_rank == 0

    def _publish_block_size(self, block_size: int) -> None:
        _worker_publish_block_size(block_size, self._dp_rank)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        if has_ucm_sparse() and os.getenv("VLLM_HASH_ATTENTION") == "1":
            for layer_name, value in kv_caches.items():
                kv_cache, k_hash = value
                self.kv_caches[layer_name] = kv_cache
        else:
            self.kv_caches = kv_caches
        sample_kv_layer = next(iter(self.kv_caches.values()))
        if self.kv_cache_dtype is None:
            self.kv_cache_dtype = sample_kv_layer[0].dtype
        if isinstance(sample_kv_layer, torch.Tensor):
            logger.info(f"kv cache shape {sample_kv_layer.shape}")
        elif isinstance(sample_kv_layer, Tuple):
            # vllm_ascend >= 0.10.0 uses Tuple for kvcaches
            for i, tensor in enumerate(sample_kv_layer):
                logger.info(f"kv cache shape {i}: {tensor.shape}")
        layout_cls = (
            SharedIndexerKVCacheLayout
            if SharedIndexerKVCacheLayout.supports(
                self._vllm_config, self.launch_config
            )
            else KVCacheLayout
        )
        self.kv_cache_layout = layout_cls(
            self.kv_caches,
            self.launch_config,
            self._vllm_config,
            self._kv_cache_config,
        )
        self.block_data_size = self.kv_cache_layout.block_size
        self.layer_name_to_id = self.kv_cache_layout.layer_name_to_id
        self.layer_ids = sorted(set(self.layer_name_to_id.values()))
        self.first_layer_id = self.layer_ids[0]

        self.device = create_device()

        enable_affinity = _use_ucm_connector_cpu_affinity()
        worker_cores, store_cores = (
            self.device.split_cores(self.device_id) if enable_affinity else (None, None)
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

        if self.device is None:
            raise RuntimeError(f"Unsupported device platform for UCMDirectConnector.")

    def _prefetch_other_rank_hashes(self, rank0_block_ids: list[bytes]) -> None:
        if not self._other_rank_hashers or not rank0_block_ids:
            return

        other_rank_block_ids = [
            rank_hasher(block_id)
            for rank_hasher in self._other_rank_hashers
            for block_id in rank0_block_ids
        ]

        if other_rank_block_ids:
            self.store.prefetch(other_rank_block_ids)

    def _prefetch_direct_hit_key_hotness(
        self,
        hbm_hit_block_ids: list[bytes],
        all_hit_block_ids: list[bytes],
    ) -> None:
        """Best-effort GC hotness update for keys skipped by scheduler lookup.

        Rank 0 external keys are already touched by ``lookup_on_prefix``. The
        local-HBM prefix is not part of that lookup, while other TP ranks do not
        perform scheduler-side lookup at all, so update those two sets here.
        """

        if hbm_hit_block_ids:
            try:
                self.store.prefetch(hbm_hit_block_ids)
            except Exception as e:
                logger.warning(
                    "UCM rank-0 HBM hotness update failed. " f"{type(e).__name__}: {e}"
                )

        if all_hit_block_ids:
            try:
                self._prefetch_other_rank_hashes(all_hit_block_ids)
            except Exception as e:
                # Prefetch is only a GC hotness hint. A failure must not turn a
                # valid cache hit into a scheduler-side miss.
                logger.warning(
                    "UCM other-rank hotness update failed. " f"{type(e).__name__}: {e}"
                )

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        assert num_computed_tokens % self.block_size == 0
        hbm_hit_block_num = num_computed_tokens // self.block_size

        ucm_block_ids = self.generate_hash(
            self.hash_block_size, request.all_token_ids, self._seed
        )

        if (
            self.enable_record_traces
            and request.request_id not in self.requests_meta
            and len(ucm_block_ids) > 0
        ):
            hex_ucm_block_ids = [id.hex() for id in ucm_block_ids]
            logger.info_once(
                f"timestamp: {time.perf_counter()}, "
                f"input_length: {request.num_tokens}, "
                f"output_length: {request.max_tokens}, "
                f"ucm_block_ids: {hex_ucm_block_ids}"
            )

        # Skip persistence if token count is below the threshold
        if self.persist_token_threshold > request.num_tokens:
            logger.info_once(
                f"Skip persistence: req {request.request_id}, "
                f"input tokens ({request.num_tokens}) < threshold ({self.persist_token_threshold})."
            )
            return 0, False

        external_block_ids = ucm_block_ids[hbm_hit_block_num * self.cp_world_size :]
        external_hit_blocks = 0
        if external_block_ids:
            try:
                external_hit_hashes = (
                    self._rank_consistency.lookup_on_prefix(
                        self.store, external_block_ids
                    )
                    + 1
                )
                external_hit_blocks = external_hit_hashes // self.cp_world_size
            except Exception as e:
                logger.error(
                    f"request {request.request_id} look up error. {type(e).__name__}: {e}"
                )
                self._record_counter("connector_lookup_errors_total")

        hbm_hit_hashes = hbm_hit_block_num * self.cp_world_size
        total_hit_hashes = (
            hbm_hit_block_num + external_hit_blocks
        ) * self.cp_world_size
        self._prefetch_direct_hit_key_hotness(
            ucm_block_ids[:hbm_hit_hashes],
            ucm_block_ids[:total_hit_hashes],
        )

        logger.info_once(
            f"request_id: {request.request_id}, "
            f"total_blocks_num: {len(ucm_block_ids)}, "
            f"hit hbm: {hbm_hit_block_num * self.cp_world_size}, "
            f"hit external: {external_hit_blocks * self.cp_world_size}, "
            f"total tokens: {len(request.all_token_ids)}"
        )

        if not external_block_ids:
            return 0, False

        ucmmetrics.update_stats(
            {
                "interval_lookup_hit_rates": external_hit_blocks
                * self.cp_world_size
                / len(ucm_block_ids)
            },
        )

        total_hit_block_num = hbm_hit_block_num + external_hit_blocks

        external_hit_tokens = external_hit_blocks * self.block_size

        # Ensure vLLM recomputes enough tokens to avoid FULL cudagraph
        # (which skips layerwise hooks). To be removed when vLLM scheduler
        # handles this natively.
        num_total_hit_tokens = total_hit_block_num * self.block_size
        recompute_tokens = self._get_full_hit_recompute_tokens()
        actual_recompute_tokens = request.num_tokens - num_total_hit_tokens
        if actual_recompute_tokens < recompute_tokens:
            deficit = recompute_tokens - actual_recompute_tokens
            if external_hit_tokens < deficit:
                logger.error(
                    f"Full UCM cache hit fallback would make external hit tokens negative: "
                    f"request_id: {request.request_id}, "
                    f"external_hit_tokens: {external_hit_tokens}, "
                    f"recompute_tokens: {recompute_tokens}, "
                    f"num_total_hit_tokens: {num_total_hit_tokens}, "
                    f"request_tokens: {request.num_tokens}"
                )
            external_hit_tokens = max(0, external_hit_tokens - deficit)

        self.requests_meta[request.request_id] = RequestMeta(
            ucm_block_ids=ucm_block_ids,
            hbm_hit_block_num=hbm_hit_block_num,
            total_hit_block_num=total_hit_block_num,
            num_token_ids=len(request.all_token_ids),
            token_processed=hbm_hit_block_num * self.block_size + external_hit_tokens,
        )

        return external_hit_tokens, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        pass

    def _generate_dispatch_meta(
        self,
        req_meta: RequestMeta,
        new_tokens: int,
        vllm_block_ids: list[int],
        need_load: bool = True,
    ) -> RequestDispatchMeta:
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

        hbm_hit_block_num = req_meta.hbm_hit_block_num
        total_hit_block_num = req_meta.total_hit_block_num
        ucm_block_ids = req_meta.ucm_block_ids
        req_meta.vllm_block_ids.extend(vllm_block_ids)

        load_ucm_block_ids, load_vllm_block_ids = [], []
        dump_ucm_block_ids, dump_vllm_block_ids = [], []
        if need_load:
            load_ucm_block_ids = ucm_block_ids[
                hbm_hit_block_num
                * self.cp_world_size : total_hit_block_num
                * self.cp_world_size
            ]
            load_vllm_block_ids = vllm_block_ids[hbm_hit_block_num:total_hit_block_num]

        if req_meta.token_processed < req_meta.num_token_ids:
            start_idx = req_meta.token_processed // self.block_size
            end_idx = (req_meta.token_processed + new_tokens) // self.block_size
            dump_ucm_block_ids = ucm_block_ids[
                start_idx * self.cp_world_size : end_idx * self.cp_world_size
            ]
            dump_vllm_block_ids = req_meta.vllm_block_ids[start_idx:end_idx]
            req_meta.token_processed += new_tokens
            if req_meta.token_processed > req_meta.num_token_ids:
                logger.warning(
                    f"Processed tokens "
                    f"({req_meta.token_processed}) exceed total tokens "
                    f"({req_meta.num_token_ids}). Truncating dump_vllm_block_ids "
                    f"to the length of dump_ucm_block_ids."
                )
                dump_vllm_block_ids = dump_vllm_block_ids[: len(dump_ucm_block_ids)]

        return RequestDispatchMeta(
            (load_ucm_block_ids, load_vllm_block_ids),
            (dump_ucm_block_ids, dump_vllm_block_ids),
        )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        requests_dispatch_meta = {}
        # for new request, we need to load and dump
        for request in scheduler_output.scheduled_new_reqs:
            request_id, vllm_block_ids = request.req_id, request.block_ids[0]
            req_meta = self.requests_meta.get(request_id)
            if req_meta:
                requests_dispatch_meta[request_id] = self._generate_dispatch_meta(
                    req_meta,
                    scheduler_output.num_scheduled_tokens[request_id],
                    vllm_block_ids,
                )

        # for cached request, there are 3 situation:
        # 1. chunked prefill: we only need dump
        # 2. resumed: we need to handle like new request
        # 3. TODO decode stage: nothing happened
        scheduled_cached_reqs = scheduler_output.scheduled_cached_reqs
        if not isinstance(scheduled_cached_reqs, list):
            # >= 0.9.2
            for i, request_id in enumerate(scheduled_cached_reqs.req_ids):
                req_meta = self.requests_meta.get(request_id)
                if req_meta:
                    new_block_ids = []
                    if scheduled_cached_reqs.new_block_ids[i] != None:
                        new_block_ids = scheduled_cached_reqs.new_block_ids[i][0]
                    if hasattr(scheduled_cached_reqs, "resumed_from_preemption"):
                        resumed_from_preemption = (
                            scheduled_cached_reqs.resumed_from_preemption[i]
                        )
                    else:
                        resumed_from_preemption = (
                            request_id in scheduled_cached_reqs.resumed_req_ids
                        )
                    requests_dispatch_meta[request_id] = self._generate_dispatch_meta(
                        req_meta,
                        scheduler_output.num_scheduled_tokens[request_id],
                        new_block_ids,
                        resumed_from_preemption,
                    )
        else:
            for request in scheduled_cached_reqs:
                request_id = request.req_id
                req_meta = self.requests_meta.get(request_id)
                if req_meta:
                    requests_dispatch_meta[request_id] = self._generate_dispatch_meta(
                        req_meta,
                        scheduler_output.num_scheduled_tokens[request_id],
                        request.new_block_ids[0],
                        request.resumed_from_preemption,
                    )

        # clear finished request
        for request_id in scheduler_output.finished_req_ids:
            self.requests_meta.pop(request_id, None)

        self._track_async_dump_requests(requests_dispatch_meta)

        return UCMConnectorMetadata(
            requests_dispatch_meta,
            scheduler_output.preempted_req_ids or set(),
        )

    def _track_async_dump_requests(
        self,
        requests_dispatch_meta: dict[str, RequestDispatchMeta],
    ) -> None:
        self._async_dump_req_ids.update(
            request_id
            for request_id, dispatch_meta in requests_dispatch_meta.items()
            if len(dispatch_meta.dump_block_ids[0]) > 0
        )

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, UCMConnectorMetadata)

        request_to_task: dict[str, Task] = {}
        is_load = False
        num_loaded_block = 0
        request_to_load_blocks: dict[str, int] = {}
        for request_id, request in metadata.request_meta.items():
            if len(request.load_block_ids[0]) == 0:
                continue
            is_load = True
            num_loaded_block += len(request.load_block_ids[0])

            ucm_block_ids, vllm_block_ids = request.load_block_ids
            if self._skip_null_vllm_blocks:
                ucm_block_ids, vllm_block_ids = _drop_null_vllm_blocks(
                    ucm_block_ids,
                    vllm_block_ids,
                    f"UCM load request {request_id}",
                )
                if len(ucm_block_ids) == 0:
                    num_loaded_block -= len(request.load_block_ids[0])
                    continue
                num_loaded_block -= len(request.load_block_ids[0]) - len(ucm_block_ids)
            store_block_ids = ucm_block_ids
            if self.tp_rank != 0 and not self.is_mla:
                store_block_ids = [
                    self.request_hasher(block_id) for block_id in ucm_block_ids
                ]
            try:
                total_ptrs = self.kv_cache_layout.extract_block_addrs(vllm_block_ids)
                total_ptrs = total_ptrs.reshape(total_ptrs.shape[0], -1)
                shard_indexs = [0] * len(ucm_block_ids)
                task = self._rank_consistency.submit_load(
                    self.store,
                    {request_id: ucm_block_ids},
                    store_block_ids,
                    shard_indexs,
                    total_ptrs,
                )
                request_to_task[request_id] = task
                request_to_load_blocks[request_id] = len(ucm_block_ids)
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
                num_loaded_block -= len(ucm_block_ids)

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

        load_bytes = num_loaded_block * self.block_data_size
        if is_load:
            ucmmetrics.update_stats({"load_bytes_total": load_bytes})

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def _get_dump_event_handle(self) -> int:
        if not self.enable_event_sync:
            self.device.synchronize()
            return 0

        event_handle = self.device.get_event_handle()
        if event_handle == 0:
            self.device.synchronize()
        return event_handle

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        pass

    def _wait_pending_dump_task(self, pending_dump_task: PendingDumpTask) -> None:
        wait_start_ms = time.perf_counter() * 1000
        try:
            self._rank_consistency.wait_dump(pending_dump_task.task)
            wait_end_ms = time.perf_counter() * 1000
            stats = {}
            if pending_dump_task.wait_for_save_start_ms > 0:
                stats["save_duration"] = self._non_negative_ms(
                    wait_end_ms - pending_dump_task.wait_for_save_start_ms
                )
            stats["save_completion_wait_duration"] = self._non_negative_ms(
                wait_end_ms - wait_start_ms
            )
            if stats:
                ucmmetrics.update_stats(stats)
        finally:
            self._release_dump_event_handle(pending_dump_task)

    def _release_dump_event_handle(self, pending_dump_task: PendingDumpTask) -> None:
        if (
            self.enable_event_sync
            and pending_dump_task.event_handle
            and self.device is not None
        ):
            self.device.destroy_event_handle(pending_dump_task.event_handle)
            pending_dump_task.event_handle = 0

    def _poll_pending_dump_tasks(self) -> None:
        remaining_tasks: list[PendingDumpTask] = []
        for pending_dump_task in self._pending_dump_tasks:
            try:
                if not self.store.check(pending_dump_task.task):
                    remaining_tasks.append(pending_dump_task)
                    continue
            except Exception as e:
                logger.error(f"check dump kv cache failed. {type(e).__name__}: {e}")
                remaining_tasks.append(pending_dump_task)
                continue

            try:
                # Check does not consume the task handle. Wait is non-blocking
                # after completion and removes the task from the store.
                self._wait_pending_dump_task(pending_dump_task)
            except Exception as e:
                logger.error_limit(
                    f"wait for dump kv cache failed. {type(e).__name__}: {e}"
                )

        self._pending_dump_tasks = remaining_tasks

    def _flush_pending_dump_tasks(self, request_ids: Optional[set[str]] = None) -> None:
        pending_dump_tasks = self._pending_dump_tasks
        self._pending_dump_tasks = []
        remaining_tasks: list[PendingDumpTask] = []
        for pending_dump_task in pending_dump_tasks:
            if request_ids is not None and not (
                pending_dump_task.request_ids & request_ids
            ):
                remaining_tasks.append(pending_dump_task)
                continue
            try:
                self._wait_pending_dump_task(pending_dump_task)
            except Exception as e:
                logger.error_limit(
                    f"wait for dump kv cache failed. {type(e).__name__}: {e}"
                )
        self._pending_dump_tasks = remaining_tasks

    def handle_preemptions(
        self,
        kv_connector_metadata: KVConnectorMetadata | set[str],
    ) -> None:
        preempted_req_ids = (
            kv_connector_metadata
            if isinstance(kv_connector_metadata, set)
            else getattr(kv_connector_metadata, "preempted_req_ids", None)
        )
        if preempted_req_ids:
            self._flush_pending_dump_tasks(preempted_req_ids)

    def wait_for_save(self) -> None:
        # TODO support PP
        wait_for_save_start_ms = time.perf_counter() * 1000
        self._poll_pending_dump_tasks()

        metadata = self._get_connector_metadata()
        assert isinstance(metadata, UCMConnectorMetadata)
        metadata_dump_request_ids = {
            request_id
            for request_id, request in metadata.request_meta.items()
            if len(request.dump_block_ids[0]) > 0
        }
        self._async_dump_req_ids.update(metadata_dump_request_ids)

        if self.is_mla and self.tp_rank != 0:
            return

        is_save = False
        num_saved_block = 0
        total_ucm_block_ids, total_vllm_block_ids = [], []
        dump_request_ids: set[str] = set()
        block_ids_by_request: dict[str, set[bytes]] = {}
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
            is_save = True
            dump_request_ids.add(request_id)
            block_ids_by_request[request_id] = set(ucm_block_ids)
            num_saved_block += len(ucm_block_ids)
            store_block_ids = ucm_block_ids
            if self.tp_rank != 0:
                store_block_ids = [
                    self.request_hasher(block_id) for block_id in ucm_block_ids
                ]
            total_ucm_block_ids.extend(store_block_ids)
            total_vllm_block_ids.extend(vllm_block_ids)

        if is_save:
            event_handle = 0
            try:
                total_ptrs = self.kv_cache_layout.extract_block_addrs(
                    total_vllm_block_ids
                )
                total_ptrs = total_ptrs.reshape(total_ptrs.shape[0], -1)
                shard_indexs = [0] * len(total_ucm_block_ids)
                event_handle = self._get_dump_event_handle()
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
                return

            save_bytes = num_saved_block * self.block_data_size
            ucmmetrics.update_stats({"save_bytes_total": save_bytes})
            self._pending_dump_tasks.append(
                PendingDumpTask(
                    task=task,
                    request_ids=set(dump_request_ids),
                    event_handle=event_handle,
                    wait_for_save_start_ms=wait_for_save_start_ms,
                )
            )

    @staticmethod
    def _non_negative_ms(value: float) -> float:
        value = float(value)
        if not math.isfinite(value):
            return 0.0
        return max(value, 0.0)

    def clear_connector_metadata(self) -> None:
        super().clear_connector_metadata()

    def get_block_ids_with_load_errors(self) -> set[int]:
        """
        Get the set of block IDs that failed to load.

        Returns:
            Set of block IDs that encountered load errors.
            Empty set if no load errors occurred.
        """
        res = self._invalid_block_ids
        self._invalid_block_ids = set()
        return res

    def build_connector_worker_meta(self) -> UCMWorkerMetadata | None:
        """Return worker events accumulated since the last call."""
        if (
            not self._rank_consistency.enabled
            and not self._connector_worker_meta.load_failed_reqs
        ):
            return None
        self._rank_consistency.update_worker_meta(self._connector_worker_meta)
        meta = self._connector_worker_meta
        self._connector_worker_meta = UCMWorkerMetadata(is_mla=meta.is_mla)
        return meta

    def update_connector_output(self, connector_output: KVConnectorOutput):
        meta = getattr(connector_output, "kv_connector_worker_meta", None)
        if meta is None:
            return
        if not isinstance(meta, UCMWorkerMetadata):
            return
        self._rank_consistency.apply_worker_meta(meta)
        for req_id in meta.load_failed_reqs:
            req_meta = self.requests_meta.get(req_id)
            if req_id in meta.missing_reqs and req_meta is not None:
                # for block missing, try to fall back to recompute and continue dump to fix it
                logger.info(
                    f"Request {req_id} loaded an incomplete UCM block; "
                    "recompute and dump it."
                )
                req_meta.total_hit_block_num = req_meta.hbm_hit_block_num
                req_meta.token_processed = (
                    req_meta.hbm_hit_block_num * self.get_block_size()
                )
            else:
                # for other failure, pop metadata so won't dump for the request
                logger.info(f"Request {req_id} failed to load, skip caching.")
                self.requests_meta.pop(req_id, None)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        if request.request_id in self._async_dump_req_ids:
            self._async_dump_req_ids.discard(request.request_id)
            return True, None
        return False, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, object] | None]:
        if block_ids:
            return self.request_finished(request, block_ids[0])
        return self.request_finished(request, [])

    def get_finished(
        self,
        finished_req_ids: set[str],
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        async_finished_req_ids = finished_req_ids & self._async_dump_req_ids

        if async_finished_req_ids:
            remaining_tasks: list[PendingDumpTask] = []
            for pending_dump_task in self._pending_dump_tasks:
                if pending_dump_task.request_ids & async_finished_req_ids:
                    try:
                        self._wait_pending_dump_task(pending_dump_task)
                    except Exception as e:
                        logger.error_limit(
                            f"wait for dump kv cache failed. {type(e).__name__}: {e}"
                        )
                else:
                    remaining_tasks.append(pending_dump_task)
            self._pending_dump_tasks = remaining_tasks

        self._rank_consistency.finish_dump(async_finished_req_ids)
        self._async_dump_req_ids.difference_update(async_finished_req_ids)

        return async_finished_req_ids or None, None


class UCMLayerWiseConnector(UCMDirectConnector):
    """
    This Connector means overlap:
    load l0 -> forward l0 -> save l0
               load l1    -> forward l1 -> save l1
                             load l2    -> forward l2 -> save l2
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        # {layer_id: {request_id: Task}}
        self.load_tasks: dict[int, dict[str, Task]] = defaultdict(dict)
        self.use_layerwise = True
        self.need_load = False
        self.dump_total_ptrs: np.ndarray | None = None
        self.request_data: list[tuple[str, list, list, np.ndarray]] = []
        self._failure_req_ids: set[str] = set()
        # A layer can be visited more than once in one model-runner batch
        # (for example by speculative decoding). Persist the first visit only.
        # This state is reset at the start of every connector batch.
        self._dumped_layer_ids: set[int] = set()
        self._layerwise_load_start_by_layer: dict[int, float] = {}
        self._layerwise_layer_load_duration_sum_ms = 0.0
        self._layerwise_load_bytes = 0
        self._layerwise_save_bytes = 0
        logger.info("Init UCMLayerWiseConnector.")

    def _record_layerwise_load_duration(self, layer_id: int, load_end: float) -> bool:
        if layer_id not in self._layerwise_load_start_by_layer:
            return False
        load_start = self._layerwise_load_start_by_layer.pop(layer_id)
        duration_ms = self._non_negative_ms((load_end - load_start) * 1000)
        self._layerwise_layer_load_duration_sum_ms += duration_ms
        ucmmetrics.update_stats({"layerwise_layer_load_duration_ms": duration_ms})
        return True

    def _submit_layerwise_dump_task(
        self,
        layer_id: int,
        total_ucm_block_ids: list[bytes],
        total_vllm_block_ids: list[int],
        dump_request_ids: set[str],
    ) -> bool:
        """Submit a layerwise dump and attach it to the existing async lifecycle."""
        if not dump_request_ids:
            return False

        metadata = self._get_connector_metadata()
        block_ids_by_request = {
            request_id: set(metadata.request_meta[request_id].dump_block_ids[0])
            for request_id in dump_request_ids
        }
        local_layer_id = layer_id - self.first_layer_id
        if self.dump_total_ptrs is None:
            self.dump_total_ptrs = self.kv_cache_layout.extract_block_addrs(
                total_vllm_block_ids, layer_first=True
            )

        event_handle = 0
        try:
            layer_ptrs = np.ascontiguousarray(self.dump_total_ptrs[local_layer_id])
            shard_indexs = [layer_id] * len(total_ucm_block_ids)
            event_handle = self._get_dump_event_handle()
            task = self._rank_consistency.submit_dump(
                self.store,
                block_ids_by_request,
                total_ucm_block_ids,
                shard_indexs,
                layer_ptrs,
                event_handle,
            )
            self._pending_dump_tasks.append(
                PendingDumpTask(
                    task=task,
                    request_ids=set(dump_request_ids),
                    event_handle=event_handle,
                )
            )
            self._layerwise_save_bytes += (
                len(total_ucm_block_ids) * self.kv_cache_layout.shard_size
            )
            return True
        except Exception as e:
            logger.error(
                f"submit dump task for layer {layer_id} failed. {type(e).__name__}: {e}",
                exc_info=True,
            )
            self._record_counter("connector_dump_submit_errors_total")
            if self.enable_event_sync and event_handle and self.device is not None:
                self.device.destroy_event_handle(event_handle)
            return False

    def _submit_request_load_tasks_for_layer(
        self,
        layer_id: int,
        local_row: int,
        metadata: "UCMConnectorMetadata",
        load_start: Optional[float] = None,
    ) -> None:
        if load_start is None:
            load_start = time.perf_counter()
        submitted = False
        for (
            request_id,
            ucm_block_ids,
            store_block_ids,
            total_ptrs,
        ) in self.request_data:
            if request_id in self._failure_req_ids:
                continue
            try:
                shard_indexs = [layer_id] * len(ucm_block_ids)
                layer_ptrs = total_ptrs[local_row]
                task = self._rank_consistency.submit_load(
                    self.store,
                    {request_id: ucm_block_ids},
                    store_block_ids,
                    shard_indexs,
                    layer_ptrs,
                )
                self.load_tasks[layer_id][request_id] = task
                submitted = True
            except Exception as e:
                logger.error(
                    f"request {request_id} submit load task for layer {layer_id} "
                    f"error. {type(e).__name__}: {e}"
                )
                self._record_load_error(
                    "connector_load_submit_errors_total",
                    metadata.request_meta[request_id].load_block_ids[1]
                    + metadata.request_meta[request_id].dump_block_ids[1],
                )
                self._failure_req_ids.add(request_id)
                self._connector_worker_meta.mark_failed(request_id)
        if submitted:
            self._layerwise_load_start_by_layer[layer_id] = load_start

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        self._dumped_layer_ids.clear()
        first_layer_load_start = time.perf_counter()
        metadata = self._get_connector_metadata()
        self.load_tasks.clear()
        self.request_data.clear()
        self._failure_req_ids.clear()
        self.need_load = False
        self._layerwise_load_start_by_layer.clear()
        self._layerwise_layer_load_duration_sum_ms = 0.0
        self._layerwise_load_bytes = 0
        self._layerwise_save_bytes = 0

        for request_id, request in metadata.request_meta.items():
            if len(request.load_block_ids[0]) == 0:
                continue

            self.need_load = True
            ucm_block_ids = list(request.load_block_ids[0])
            vllm_block_ids = list(request.load_block_ids[1])
            store_block_ids = ucm_block_ids
            if self.tp_rank % self.tp_size != 0 and not self.is_mla:
                store_block_ids = [
                    self.request_hasher(block_id) for block_id in ucm_block_ids
                ]
            total_ptrs = self.kv_cache_layout.extract_block_addrs(
                vllm_block_ids, layer_first=True
            )
            self.request_data.append(
                (request_id, ucm_block_ids, store_block_ids, total_ptrs)
            )

        if self.need_load:
            self._submit_request_load_tasks_for_layer(
                self.first_layer_id,
                0,
                metadata,
                first_layer_load_start,
            )

    def wait_for_layer_load(self, layer_name: str) -> None:
        if not self._connector_metadata:
            return
        if not self.need_load:
            return
        metadata = self._get_connector_metadata()
        current_layer_id = self.layer_name_to_id[layer_name]

        # Pop before wait so MTP / rollback paths that revisit the same layer_name
        # do not call store.wait() again on already-completed handles.
        layer_tasks = self.load_tasks.pop(current_layer_id, {})
        for request_id, task in layer_tasks.items():
            try:
                self._rank_consistency.wait_load(task)
            except Exception as e:
                logger.error(
                    f"request {request_id} wait {layer_name} load failed. "
                    f"{type(e).__name__}: {e}"
                )
                self._record_load_error(
                    "connector_load_wait_errors_total",
                    metadata.request_meta[request_id].load_block_ids[1]
                    + metadata.request_meta[request_id].dump_block_ids[1],
                )
                self._connector_worker_meta.mark_failed(request_id)
                self._failure_req_ids.add(request_id)
            else:
                self._layerwise_load_bytes += (
                    len(metadata.request_meta[request_id].load_block_ids[0])
                    * self.kv_cache_layout.shard_size
                )

        wait_end = time.perf_counter()
        load_duration_recorded = self._record_layerwise_load_duration(
            current_layer_id, wait_end
        )

        next_layer_id = current_layer_id + 1
        has_next = next_layer_id in self.layer_ids
        if has_next:
            next_local_row = next_layer_id - self.first_layer_id
            self._submit_request_load_tasks_for_layer(
                next_layer_id, next_local_row, metadata
            )
        elif load_duration_recorded:
            duration_sum_ms = self._layerwise_layer_load_duration_sum_ms
            ucmmetrics.update_stats(
                {
                    "layerwise_batch_load_duration_sum_ms": duration_sum_ms,
                    "load_bytes_total": self._layerwise_load_bytes,
                }
            )
            self._layerwise_layer_load_duration_sum_ms = 0.0
            self._layerwise_load_bytes = 0

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

        metadata = self._get_connector_metadata()

        total_ucm_block_ids, total_vllm_block_ids = [], []
        dump_request_ids: set[str] = set()
        layer_id = self.layer_name_to_id[layer_name]
        if layer_id in self._dumped_layer_ids:
            logger.debug(
                f"Skip duplicate layerwise dump in the same batch: "
                f"layer_name={layer_name}, layer_id={layer_id}"
            )
            return

        for request_id, request in metadata.request_meta.items():
            if len(request.dump_block_ids[0]) == 0:
                continue

            dump_request_ids.add(request_id)
            ucm_block_ids, vllm_block_ids = request.dump_block_ids
            store_block_ids = ucm_block_ids
            if self.tp_rank % self.tp_size != 0:
                store_block_ids = [
                    self.request_hasher(block_id) for block_id in ucm_block_ids
                ]
            total_ucm_block_ids.extend(store_block_ids)
            total_vllm_block_ids.extend(vllm_block_ids)

        if dump_request_ids:
            submitted = self._submit_layerwise_dump_task(
                layer_id,
                total_ucm_block_ids,
                total_vllm_block_ids,
                dump_request_ids,
            )
            if submitted:
                self._dumped_layer_ids.add(layer_id)

    def wait_for_save(self) -> None:
        wait_for_save_start_ms = time.perf_counter() * 1000
        for pending_dump_task in self._pending_dump_tasks:
            if pending_dump_task.wait_for_save_start_ms <= 0:
                pending_dump_task.wait_for_save_start_ms = wait_for_save_start_ms
        self._poll_pending_dump_tasks()
        if self._layerwise_save_bytes > 0:
            ucmmetrics.update_stats({"save_bytes_total": self._layerwise_save_bytes})
            self._layerwise_save_bytes = 0
        if self._connector_metadata:
            metadata = self._get_connector_metadata()
            self._async_dump_req_ids.update(
                request_id
                for request_id, request in metadata.request_meta.items()
                if len(request.dump_block_ids[0]) > 0
            )

        self.dump_total_ptrs = None


class UCMCPConnector(UCMLayerWiseConnector):
    _defer_scheduler_store = True

    @staticmethod
    def _consistency_manager_enabled(launch_config: dict, is_mla: bool) -> bool:
        if launch_config.get("use_consistency_manager", False):
            raise ValueError(
                "Rank consistency manager is not supported by UCMCPConnector."
            )
        return False

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        self.use_layerwise = self.launch_config.get("use_layerwise", True)

        try:
            from vllm.distributed import get_dcp_group, get_pcp_group
        except ImportError as e:
            raise ImportError(
                "Please check if the current vLLM version supports DCP and PCP features."
            ) from e

        try:
            self.pcp_world_size = get_pcp_group().world_size
            self.pcp_rank = (
                get_pcp_group().rank_in_group if self.pcp_world_size > 1 else 0
            )
            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            # DCP might not be initialized in testing
            self.dcp_world_size = 1
            self.dcp_rank = 0
            self.pcp_world_size = 1
            self.pcp_rank = 0
        self.cp_world_size = (
            self._vllm_config.parallel_config.prefill_context_parallel_size
            * self._vllm_config.parallel_config.decode_context_parallel_size
        )
        self.current_rank = self.dcp_world_size * self.pcp_rank + self.dcp_rank
        old_tp_size = vllm_config.parallel_config.tensor_parallel_size
        logger.info(
            f"pcp_world_size: {self.pcp_world_size}, pcp_rank: {self.pcp_rank}, dcp_world_size: {self.dcp_world_size}, dcp_rank: {self.dcp_rank}"
        )

        self.tp_rank %= self.tp_size
        self.tp_rank //= self.dcp_world_size
        if not self.is_mla:
            vllm_config.parallel_config.tensor_parallel_size //= self.dcp_world_size

        if role == KVConnectorRole.SCHEDULER:
            self.request_hasher = RequestHasher(vllm_config, 0)
            self._other_rank_hashers = self._make_other_rank_hashers(vllm_config)
            self._seed = self.request_hasher("UCM_HASH_SEED")
            # init scheduler-side connector
            self.store = self._create_store(None)
        else:
            self.request_hasher = RequestHasher(vllm_config, self.tp_rank)
        vllm_config.parallel_config.tensor_parallel_size = old_tp_size
        self.block_size *= self.cp_world_size
        logger.info("Init UCMCPConnector.")

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        # When DCP/PCP features are enabled,
        # the blocks that each device can process are [current_rank :: cp_world_size],
        # where current_rank = self.dcp_world_size * self.pcp_rank + self.dcp_rank.
        for _, request in connector_metadata.request_meta.items():
            if len(request.load_block_ids[0]) > 0:
                ucm_block_ids, vllm_block_ids = request.load_block_ids
                ucm_block_ids = ucm_block_ids[self.current_rank :: self.cp_world_size]
                request.load_block_ids = (ucm_block_ids, vllm_block_ids)

            if len(request.dump_block_ids[0]) > 0:
                ucm_block_ids, vllm_block_ids = request.dump_block_ids
                ucm_block_ids = ucm_block_ids[self.current_rank :: self.cp_world_size]
                request.dump_block_ids = (ucm_block_ids, vllm_block_ids)
        super().bind_connector_metadata(connector_metadata)

    def start_load_kv(self, forward_context, **kwargs):
        if self.use_layerwise:
            super().start_load_kv(forward_context, **kwargs)
        else:
            super(UCMLayerWiseConnector, self).start_load_kv(forward_context, **kwargs)

    def wait_for_layer_load(self, layer_name: str) -> None:
        if self.use_layerwise:
            super().wait_for_layer_load(layer_name)
        else:
            pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        if self.use_layerwise:
            super().save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)
        else:
            pass

    def wait_for_save(self):
        if self.use_layerwise:
            super().wait_for_save()
        else:
            super(UCMLayerWiseConnector, self).wait_for_save()


class UCMPDConnector(UCMDirectConnector):
    """
    This Connector means overlap (especially for Decode Instance):
    step (req0,1,2) forward -> step (req0,1,2,3) forward
    load req3               -> load req4
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        raise NotImplementedError

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens.

        Returns:
            ids of requests that have finished asynchronous transfer
            (requests that previously returned True from request_finished()),
            tuple of (sending/saving ids, recving/loading ids).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """
        raise NotImplementedError


class UCMMockConnector(UCMDirectConnector):
    """
    This Connector can control hit ratio, for example: if your hit ratio is 100%,
    you can set "hit_ratio" by config or env_vars, then get_num_new_matched_tokens()
    will reduce hit_tokens under the hit_ratio you set.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        self._hit_ratio = float(self.launch_config["hit_ratio"])
        logger.info(f"hit_ratio: {self._hit_ratio}")

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        hit_tokens, _ = super().get_num_new_matched_tokens(request, num_computed_tokens)
        expect_hit_tokens = int(self._hit_ratio * request.num_prompt_tokens)
        if hit_tokens <= expect_hit_tokens:
            return hit_tokens, False
        expect_hit_block_num = expect_hit_tokens // self.block_size
        request_meta = self.requests_meta[request.request_id]
        request_meta.total_hit_block_num = expect_hit_block_num
        request_meta.hbm_hit_block_num = min(
            expect_hit_block_num, request_meta.hbm_hit_block_num
        )

        logger.info(
            "Hijacked By MockConnector,"
            f"request_id: {request.request_id}, "
            f"total_blocks_num: {len(request_meta.ucm_block_ids)}, "
            f"hit hbm: {request_meta.hbm_hit_block_num}, "
            f"hit external: {request_meta.total_hit_block_num - request_meta.hbm_hit_block_num}"
        )

        return expect_hit_block_num * self.block_size, False


class UCMLiteConnector(KVConnectorBase_V1):
    def __init__(
        self,
        vllm_config,
        role,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        self.block_size = vllm_config.cache_config.block_size
        self.hash_block_size = self.block_size
        self.requests_meta: dict[str, RequestMeta] = {}
        self.total_block_nums = 0

        self.request_hasher = RequestHasher(vllm_config, 0)
        self._seed = self.request_hasher("UCM_HASH_SEED")

        super().__init__(vllm_config, role, kv_cache_config)

        logger.info("Init UCMLiteConnector.")

    def get_block_size(self) -> int:
        return self.block_size

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        req_blocks_num = len(request.all_token_ids) // self.hash_block_size
        if req_blocks_num < 1:
            return 0, False
        if request.request_id not in self.requests_meta:
            hash_start = time.perf_counter()
            ucm_block_ids = self.generate_hash(
                self.hash_block_size, request.all_token_ids, self._seed
            )
            hash_end = time.perf_counter()
            hash_time_ms = (hash_end - hash_start) * 1000.0

            print_start = time.perf_counter()
            hex_ucm_block_ids = [b.hex() for b in ucm_block_ids]
            logger.info(
                f"timestamp: {time.perf_counter()}, "
                f"request_id: {request.request_id}, "
                f"input_length: {request.num_tokens}, "
                f"output_length: {request.max_tokens}, "
                f"ucm_block_ids: {hex_ucm_block_ids}"
            )
            print_time_ms = (time.perf_counter() - print_start) * 1000.0
            logger.info(
                f"request_id: {request.request_id}, "
                f"hash_time_ms: {hash_time_ms:.3f}, "
                f"print_time_ms: {print_time_ms:.3f}"
            )

            # store minimal RequestMeta for scheduler bookkeeping
            self.requests_meta[request.request_id] = RequestMeta(
                ucm_block_ids=ucm_block_ids,
                hbm_hit_block_num=0,
                total_hit_block_num=0,
                num_token_ids=len(request.all_token_ids),
                token_processed=0,
            )

            self.total_block_nums += req_blocks_num

        return 0, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        pass

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        for request_id in scheduler_output.finished_req_ids:
            self.requests_meta.pop(request_id, None)
        return UCMConnectorMetadata()

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        self.requests_meta.pop(request.request_id, None)
        return False, None

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        pass

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        pass

    def wait_for_save(self):
        pass

    def generate_hash(
        self,
        block_size: int,
        token_ids: List[int],
        parent_block_hash_value: bytes,
    ) -> list[bytes]:
        ret = []
        for start in range(0, len(token_ids), block_size):
            end = start + block_size
            block_token_ids = token_ids[start:end]
            # Do not hash the block if it is not full.
            if len(block_token_ids) < block_size:
                break

            block_token_ids_tuple = tuple(block_token_ids)
            hash_value = self.request_hasher(
                (parent_block_hash_value, block_token_ids_tuple)
            )
            parent_block_hash_value = hash_value
            ret.append(hash_value)

        return ret


class UCMConnector(KVConnectorBase_V1, SupportsHMA):
    @classmethod
    def requires_piecewise_for_cudagraph(cls, extra_config: dict[str, Any]) -> bool:
        from ucm.integration.vllm.inference_duration_monitor_connector import (
            inference_duration_monitor_enabled,
        )

        return inference_duration_monitor_enabled(extra_config)

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self.connector: KVConnectorBase_V1
        ucm_config = Config(vllm_config.kv_transfer_config)
        self.engine_id = vllm_config.kv_transfer_config.engine_id.rsplit("_dp", 1)[0]
        self.launch_config = ucm_config.get_config()
        self._worker_rank = (
            get_world_group().rank
            if role == KVConnectorRole.WORKER
            else "scheduler" if role == KVConnectorRole.SCHEDULER else None
        )
        if role == KVConnectorRole.WORKER and self._worker_rank == 0:
            logger.info(f"UCM worker rank 0 KV cache config: {kv_cache_config!r}")
        self._setup_ucm_metrics(vllm_config, role)
        logger.info(f"self.launch_config: {self.launch_config}")

        use_layerwise = (
            self.launch_config.get("use_layerwise", True)
            if self.launch_config is not None
            else False
        )

        use_lite = (
            self.launch_config.get("use_lite", False)
            if self.launch_config is not None
            else False
        )

        if use_lite:
            self.connector = UCMLiteConnector(vllm_config, role, kv_cache_config)
            return

        use_inference_duration_monitor = (
            self.launch_config.get("use_inference_duration_monitor", False)
            if self.launch_config is not None
            else False
        )
        if use_inference_duration_monitor:
            from ucm.integration.vllm.inference_duration_monitor_connector import (
                UCMInferenceDurationMonitorConnector,
            )

            self.connector = UCMInferenceDurationMonitorConnector(
                vllm_config, role, kv_cache_config
            )
            return

        pp_enabled = self._vllm_config.parallel_config.pipeline_parallel_size > 1
        if pp_enabled and not use_layerwise:
            raise RuntimeError(
                "Pipeline parallelism is not supported in UCMDirectConnector, please set use_layerwise=True."
            )

        use_ratio_rate = (
            self.launch_config is not None and "hit_ratio" in self.launch_config
        )

        use_cp_parallel = (
            hasattr(self._vllm_config.parallel_config, "prefill_context_parallel_size")
            and hasattr(
                self._vllm_config.parallel_config, "decode_context_parallel_size"
            )
            and self._vllm_config.parallel_config.prefill_context_parallel_size
            * self._vllm_config.parallel_config.decode_context_parallel_size
            > 1
        )

        from ucm.integration.vllm.hla_connector import (
            UCMHybridLinearAttentionConnector,
            UCMHybridLinearAttentionLayerWiseConnector,
        )
        from ucm.integration.vllm.hma_connector import UCMFAWAConnector

        use_hybrid_linear_attention = (
            UCMHybridLinearAttentionConnector.supports_kv_cache_layout(kv_cache_config)
        )
        use_hybrid_linear_attention_layerwise = (
            use_hybrid_linear_attention
            and use_layerwise
            and self.launch_config.get("hybrid_linear_attention_layerwise", True)
        )

        if UCMFAWAConnector.can_handle_kv_cache_config(kv_cache_config):
            self.connector = UCMFAWAConnector(vllm_config, role, kv_cache_config)
        elif use_ratio_rate:
            self.connector = UCMMockConnector(vllm_config, role, kv_cache_config)
        elif use_cp_parallel:
            self.connector = UCMCPConnector(vllm_config, role, kv_cache_config)
        elif use_hybrid_linear_attention_layerwise:
            self.connector = UCMHybridLinearAttentionLayerWiseConnector(
                vllm_config, role, kv_cache_config
            )
        elif use_hybrid_linear_attention:
            self.connector = UCMHybridLinearAttentionConnector(
                vllm_config, role, kv_cache_config
            )
        elif use_layerwise:
            self.connector = UCMLayerWiseConnector(vllm_config, role, kv_cache_config)
        else:
            self.connector = UCMDirectConnector(vllm_config, role, kv_cache_config)

    @_record_connector_interface_duration
    def get_block_size(self) -> int:
        return self.connector.get_block_size()

    def _setup_ucm_metrics(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        self._vllm_metrics_enabled = False
        self._vllm_metric_definitions = []
        self._metrics_dispatcher = None

        metrics_config_path = self.launch_config.get("metrics_config_path", "")
        self.metrics_config = load_launch_metrics_config(self.launch_config)
        if self.metrics_config and (
            consumer_enabled(self.metrics_config, MULTIPROC_CONSUMER)
            or consumer_enabled(self.metrics_config, VLLM_CONNECTOR_CONSUMER)
        ):
            setup_ucm_metrics(self.metrics_config)
            self._metrics_dispatcher = get_metrics_dispatcher(self.metrics_config)
        if (
            metrics_config_path
            and self.metrics_config
            and consumer_enabled(self.metrics_config, MULTIPROC_CONSUMER)
        ):
            worker_id = (
                f"{self.engine_id}_{get_world_group().rank}"
                if role == KVConnectorRole.WORKER
                else self.engine_id
            )
            self.stats_logger = PrometheusStatsLogger(
                vllm_config.model_config.served_model_name,
                worker_id,
                metrics_config_path,
            )
            logger.info(
                f"metrics_config_path: {metrics_config_path}, set worker_id: {worker_id}"
            )
        if self.metrics_config and consumer_enabled(
            self.metrics_config, VLLM_CONNECTOR_CONSUMER
        ):
            self._vllm_metric_definitions = get_vllm_connector_metric_definitions(
                self.metrics_config
            )
            self._vllm_metrics_enabled = bool(self._vllm_metric_definitions)

    @_record_connector_interface_duration
    def get_kv_connector_stats(self) -> Optional["KVConnectorStats"]:
        if not self._vllm_metrics_enabled:
            return None
        if self._metrics_dispatcher is None:
            return None
        self._metrics_dispatcher.drain_to_consumers()
        counter_stats, gauge_stats, histogram_stats = (
            self._metrics_dispatcher.get_stats_and_clear(VLLM_CONNECTOR_CONSUMER)
        )
        stats = UCMConnectorStats.from_ucm_snapshot(
            counter_stats,
            gauge_stats,
            histogram_stats,
            worker_rank=self._worker_rank,
            metric_definitions=self._vllm_metric_definitions,
        )
        return None if stats.is_empty() else stats

    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> Optional["KVConnectorStats"]:
        return UCMConnectorStats(data=data) if data is not None else UCMConnectorStats()

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: "VllmConfig",
        metric_types: dict[type["PromMetric"], type["PromMetricT"]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> Optional["KVConnectorPromMetrics"]:
        if not UCM_HAS_PROM_METRICS:
            return None
        config = load_launch_metrics_config(
            Config(vllm_config.kv_transfer_config).get_config()
        )
        if not config or not consumer_enabled(config, VLLM_CONNECTOR_CONSUMER):
            return None
        if not get_vllm_connector_metric_definitions(config):
            return None
        return UCMPromMetrics(
            vllm_config,
            metric_types,
            labelnames,
            per_engine_labelvalues,
        )

    @_record_connector_interface_duration
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the
            external KV cache beyond what is already computed.
        """
        external_hit_tokens, need_load = self.connector.get_num_new_matched_tokens(
            request, num_computed_tokens
        )
        self._record_prefix_cache_token_metrics(
            request,
            num_computed_tokens,
            external_hit_tokens,
        )
        return external_hit_tokens, need_load

    def _record_prefix_cache_token_metrics(
        self,
        request: "Request",
        num_computed_tokens: int,
        external_hit_tokens: int,
    ) -> None:
        total_tokens = getattr(request, "num_tokens", None)
        if total_tokens is None:
            total_tokens = len(getattr(request, "all_token_ids", ()))

        total_tokens = max(int(total_tokens), 0)
        gpu_hbm_hit_tokens = min(max(int(num_computed_tokens), 0), total_tokens)
        ucm_hit_tokens = min(
            max(int(external_hit_tokens), 0),
            max(total_tokens - gpu_hbm_hit_tokens, 0),
        )
        block_size = max(int(self.get_block_size()), 1)
        ucmmetrics.update_stats(
            {
                "total_prefix_query_tokens_total": total_tokens,
                "gpu_hbm_hit_tokens_total": gpu_hbm_hit_tokens,
                "ucm_hit_tokens_total": ucm_hit_tokens,
                "total_prefix_query_blocks_total": total_tokens // block_size,
                "gpu_hbm_hit_blocks_total": gpu_hbm_hit_tokens // block_size,
            }
        )

    @_record_connector_interface_duration
    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """
        Update KVConnector state after block allocation.
        """
        self.connector.update_state_after_alloc(request, blocks, num_external_tokens)

    @_record_connector_interface_duration
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """
        Initialize with the KV caches. Useful for pre-registering the
        KV Caches in the KVConnector (e.g. for NIXL).

        Args: kv_caches:
            dictionary of layer names, kv cache
        """
        self.connector.register_kv_caches(kv_caches)

    @_record_connector_interface_duration
    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """
        Build the connector metadata for this step.

        This function should NOT modify fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        return self.connector.build_connector_meta(scheduler_output)

    @_record_connector_interface_duration
    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        """Set the connector metadata from the scheduler.

        This function should be called by the model runner every time
        before the model execution. The metadata will be used for runtime
        KV cache loading and saving.

        Args:
            connector_metadata (dict): the connector metadata.
        """
        self.connector.bind_connector_metadata(connector_metadata)

    @_record_connector_interface_duration
    def handle_preemptions(self, kv_connector_metadata: KVConnectorMetadata):
        self.connector.handle_preemptions(kv_connector_metadata)

    @_record_connector_interface_duration
    def has_connector_metadata(self) -> bool:
        """Check whether the connector metadata is currently set.

        Returns:
            bool: True if connector metadata exists, False otherwise.
        """
        return self.connector.has_connector_metadata()

    @_record_connector_interface_duration
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        """
        Start loading the KV cache from the connector to vLLM's paged
        KV buffer. This is called from the forward context before the
        forward pass to enable async loading during model execution.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be
            the same.

        """
        self.connector.start_load_kv(forward_context, **kwargs)

    @_record_connector_interface_duration
    def wait_for_layer_load(self, layer_name: str) -> None:
        """
        Block until the KV for a specific layer is loaded into vLLM's
        paged buffer. This is called from within attention layer to ensure
        async copying from start_load_kv is complete.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        self.connector.wait_for_layer_load(layer_name)

    @_record_connector_interface_duration
    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        """
        Start saving the a layer of KV cache from vLLM's paged buffer
        to the connector. This is called from within attention layer to
        enable async copying during execution.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        self.connector.save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)

    @_record_connector_interface_duration
    def wait_for_save(self) -> None:
        """
        Block until all the save operations is done. This is called
        as the forward context exits to ensure that the async saving
        from save_kv_layer is complete before finishing the forward.

        This prevents overwrites of paged KV buffer before saving done.
        """
        self.connector.wait_for_save()

    @_record_connector_interface_duration
    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, object] | None]:
        if isinstance(self.connector, SupportsHMA):
            return self.connector.request_finished_all_groups(request, block_ids)
        if block_ids:
            return self.connector.request_finished(request, block_ids[0])
        return self.connector.request_finished(request, [])

    @_record_connector_interface_duration
    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, object] | None]:
        return self.connector.request_finished(request, block_ids)

    @_record_connector_interface_duration
    def get_finished(
        self,
        finished_req_ids: set[str],
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        return self.connector.get_finished(finished_req_ids)

    @_record_connector_interface_duration
    def build_connector_worker_meta(self):
        return self.connector.build_connector_worker_meta()

    @_record_connector_interface_duration
    def update_connector_output(self, connector_output: KVConnectorOutput):
        return self.connector.update_connector_output(connector_output)

    @_record_connector_interface_duration
    def clear_connector_metadata(self) -> None:
        """Clear the connector metadata.

        This function should be called by the model runner every time
        after the model execution.
        """
        self.connector.clear_connector_metadata()

    @_record_connector_interface_duration
    def get_block_ids_with_load_errors(self) -> set[int]:
        """
        Get the set of block IDs that failed to load.

        Returns:
            Set of block IDs that encountered load errors.
            Empty set if no load errors occurred.
        """
        return self.connector.get_block_ids_with_load_errors()
