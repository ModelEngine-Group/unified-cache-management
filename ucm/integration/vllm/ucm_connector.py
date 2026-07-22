import copy
import hashlib
import math
import os
import pickle
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
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

from ucm.integration.vllm.device import create_device
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


def _normalize_tensor_size_list(tensor_size_list: Any) -> list[int]:
    if isinstance(tensor_size_list, np.ndarray):
        return [int(v) for v in tensor_size_list.reshape(-1).tolist()]
    if isinstance(tensor_size_list, (list, tuple)):
        return [int(v) for v in tensor_size_list]
    return [int(tensor_size_list)]


def _short_list(values: list[int], limit: int = 12) -> list[int]:
    return values[:limit]


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


def _use_ucm_connector_cpu_affinity() -> bool:
    return (
        os.getenv("VLLM_CPU_AFFINITY") == "1"
        and getattr(current_platform, "device_type", None) != "npu"
    )


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


class KVCacheLayout:
    def __init__(
        self,
        kvcaches,
        ucm_config: dict,
        vllm_config: "VllmConfig",
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        # each row is a layer, each column is a tensor_size/ptr in the layer (e.g., k, v, rope, k_index)
        self.base_ptrs: np.ndarray  # (n_layers, n_ptrs）
        self.buffer_sizes: np.ndarray  # (n_layers, n_ptrs)
        self.tensor_size_lists: np.ndarray  # (n_layers, n_tensor_sizes)
        self.block_stride_lists: np.ndarray  # (n_layers, n_tensor_strides)
        self.use_layerwise = ucm_config.get("use_layerwise", False)
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

    def _build_layout(self, kvcaches):

        num_rows = len(set(self.layer_name_to_id.values()))
        raw_ptr_rows = [[] for _ in range(num_rows)]
        stride_rows = [[] for _ in range(num_rows)]
        buffer_size_rows = [[] for _ in range(num_rows)]

        for layer_name, kv_layer in kvcaches.items():
            ptrs = []
            strides = []
            buffer_sizes = []

            def handle_tensor(t: torch.Tensor, size_dims):
                ptrs.append(t[0].data_ptr())

                stride = math.prod([t.shape[i] for i in size_dims]) * t.element_size()
                strides.append(stride)
                buffer_sizes.append(int(t.shape[0]) * stride)

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
                        f"Unsupported kv cache tensor shape: {kv_layer.shape}"
                    )
            elif isinstance(kv_layer, Tuple):
                # vllm_ascend >= 0.10.0, ([num_blocks, block_size, num_head, head_dim], ...)
                for tensor in kv_layer:
                    handle_tensor(tensor, (-3, -2, -1))
            else:
                raise TypeError(f"Unsupported kv cache type: {type(kv_layer)}")

            local_layer_id = self.layer_name_to_id[layer_name] - self.first_layer_id
            raw_ptr_rows[local_layer_id].extend(ptrs)
            stride_rows[local_layer_id].extend(strides)
            buffer_size_rows[local_layer_id].extend(buffer_sizes)

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

    def extract_block_addrs_for_row(
        self, vllm_block_ids: List[int], row_id: int
    ) -> np.ndarray:
        vllm_block_ids_np = np.asarray(vllm_block_ids, dtype=np.uint64)
        return (
            vllm_block_ids_np[:, None] * self.block_stride_lists[row_id][None, :]
            + self.base_ptrs[row_id][None, :]
        )

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
        meta = (
            f"{vllm_config.model_config.model}:"
            f"{vllm_config.parallel_config.tensor_parallel_size}:"
            f"{vllm_config.model_config.dtype}:{rank_id}{spec_info}"
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
        else:
            raise RuntimeError("Unsupported device platform for UCMDirectConnector.")

        if self.local_rank >= 0:
            self.device = torch_dev.device(f"{dev_name}:{self.local_rank}")

        self.store: UcmKVStoreBaseV1
        self.rope_store: Optional[UcmKVStoreBaseV1] = None

        # save block info, avoid hash request twice, and track them until request finished
        self.requests_meta: dict[str, RequestMeta] = {}

        ucm_config = Config(vllm_config.kv_transfer_config)
        dp_engine_id_suffix = re.compile(r"_dp\d+$")
        self.engine_id = dp_engine_id_suffix.sub(
            "", vllm_config.kv_transfer_config.engine_id
        )
        self.launch_config = ucm_config.get_config()
        self.connector_configs = self.launch_config.get("ucm_connectors", [])
        self.enable_event_sync = self.launch_config.get("enable_event_sync", True)
        self.enable_record_traces = self.launch_config.get(
            "enable_record_traces", False
        )
        self._skip_null_vllm_blocks = False
        assert len(self.connector_configs) > 0, "no storage connector name in config."

        self.chunk_size = self.block_size
        self.blocks_per_chunk = self.chunk_size // self.block_size
        self._other_rank_hashers: list[RequestHasher] = []

        defer_scheduler_store = getattr(self, "_defer_scheduler_store", False)
        if role == KVConnectorRole.SCHEDULER:
            self.request_hasher = RequestHasher(vllm_config, 0)
            self._other_rank_hashers = self._make_other_rank_hashers(vllm_config)
            self._seed = self.request_hasher("UCM_HASH_SEED")
            # init scheduler-size connector
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
        return 1

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
        if config.get("cache_buffer_capacity_gb") is not None:
            return

        config["cache_buffer_capacity_gb"] = 128
        logger.info("Set cache_buffer_capacity_gb to 128GB for shared-buffer store.")

    def _create_store(
        self,
        kv_cache_layout: Optional[KVCacheLayout],
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
        config["unique_id"] = f"{self.engine_id}{unique_id_suffix}"
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
            config["local_rank_size"] = self.tp_size if self.is_mla else 1
            buffer_addrs = kv_cache_layout.base_ptrs.reshape(-1).tolist()
            buffer_sizes = kv_cache_layout.buffer_sizes.reshape(-1).tolist()
            gpu_kv_buffer_set = set()
            gpu_kv_buffer_addrs = []
            gpu_kv_buffer_sizes = []
            for addr, size in zip(buffer_addrs, buffer_sizes):
                key = (int(addr), int(size))
                if key in gpu_kv_buffer_set:
                    continue
                gpu_kv_buffer_set.add(key)
                gpu_kv_buffer_addrs.append(key[0])
                gpu_kv_buffer_sizes.append(key[1])
            config["gpu_kv_buffer_addrs"] = gpu_kv_buffer_addrs
            config["gpu_kv_buffer_sizes"] = gpu_kv_buffer_sizes
        else:
            config_base = self.block_size * self.element_size * self.head_size
            config["block_size"] = (
                config_base
                * self.num_layers
                * (1 if self.is_mla else self.num_head * 2)
                * self.blocks_per_chunk
            )
        dp_rank = self._vllm_config.parallel_config.data_parallel_rank
        config["posix_gc_enable"] = (
            self._role != KVConnectorRole.WORKER and dp_rank == 0
        )
        logger.info(f"create {name} with config: {config}")
        return UcmConnectorFactoryV1.create_connector(name, config, module_path)

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
        self.kv_cache_layout = KVCacheLayout(
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
            self.device.split_cores(self.local_rank)
            if enable_affinity
            else (None, None)
        )

        self.store = self._create_store(self.kv_cache_layout, store_cores)

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
                self._prefetch_other_rank_hashes(
                    external_block_ids[:external_hit_hashes]
                )
                external_hit_blocks = external_hit_hashes // self.cp_world_size
            except Exception as e:
                logger.error(
                    f"request {request.request_id} look up error. {type(e).__name__}: {e}"
                )
                self._record_counter("connector_lookup_errors_total")

        logger.info_once(
            f"request_id: {request.request_id}, "
            f"total_blocks_num: {len(ucm_block_ids)}, "
            f"hit hbm: {hbm_hit_block_num * self.cp_world_size}, "
            f"hit external: {external_hit_blocks * self.cp_world_size}"
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

        # When all the tokens are cached in ssd or hbm, recompute enough tokens
        # to keep layerwise loads on the prefill path when speculative decode is
        # enabled. This branch will be removed once vLLM scheduler provides a
        # better solution in the future.
        num_total_hit_tokens = total_hit_block_num * self.block_size
        if num_total_hit_tokens == request.num_tokens:
            recompute_tokens = self._get_full_hit_recompute_tokens()
            if external_hit_tokens < recompute_tokens:
                logger.error(
                    f"Full UCM cache hit fallback would make external hit tokens negative: "
                    f"request_id: {request.request_id}, "
                    f"external_hit_tokens: {external_hit_tokens}, "
                    f"recompute_tokens: {recompute_tokens}, "
                    f"num_total_hit_tokens: {num_total_hit_tokens}, "
                    f"request_tokens: {request.num_tokens}"
                )
            external_hit_tokens = max(0, external_hit_tokens - recompute_tokens)

        self.requests_meta[request.request_id] = RequestMeta(
            ucm_block_ids=ucm_block_ids,
            hbm_hit_block_num=hbm_hit_block_num,
            total_hit_block_num=total_hit_block_num,
            num_token_ids=len(request.all_token_ids),
            token_processed=num_total_hit_tokens,
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
        num_loaded_request = 0
        load_start_time = time.perf_counter() * 1000
        request_to_load_blocks: dict[str, int] = {}
        for request_id, request in metadata.request_meta.items():
            if len(request.load_block_ids[0]) == 0:
                continue
            is_load = True
            num_loaded_block += len(request.load_block_ids[0])
            num_loaded_request += 1

            ucm_block_ids, vllm_block_ids = request.load_block_ids
            if self._skip_null_vllm_blocks:
                ucm_block_ids, vllm_block_ids = _drop_null_vllm_blocks(
                    ucm_block_ids,
                    vllm_block_ids,
                    f"UCM load request {request_id}",
                )
                if len(ucm_block_ids) == 0:
                    num_loaded_block -= len(request.load_block_ids[0])
                    num_loaded_request -= 1
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

        load_end_time = time.perf_counter() * 1000
        load_duration_ms = load_end_time - load_start_time
        load_bytes = num_loaded_block * self.block_data_size
        load_speed = load_bytes / load_duration_ms / 1024 / 1024  # GB/s
        if is_load:
            load_stats = {
                "load_requests_num": num_loaded_request,
                "load_blocks_num": num_loaded_block,
                "load_duration": load_duration_ms,
                "load_speed": load_speed,
                "load_bytes_total": load_bytes,
            }
            ucmmetrics.update_stats(load_stats)

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
        num_saved_request = 0
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
            num_saved_request += 1
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
            save_stats = {
                "save_requests_num": num_saved_request,
                "save_blocks_num": num_saved_block,
                "save_bytes_total": save_bytes,
            }
            ucmmetrics.update_stats(save_stats)
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

    _BATCH_TOTAL_METRICS = {
        (False, False): "layerwise_batch_total_no_transfer_ms",
        (True, False): "layerwise_batch_total_load_only_ms",
        (False, True): "layerwise_batch_total_save_only_ms",
        (True, True): "layerwise_batch_total_load_save_ms",
    }
    _BATCH_LOAD_WAIT_METRICS = {
        (True, False): "layerwise_batch_load_wait_total_load_only_ms",
        (True, True): "layerwise_batch_load_wait_total_load_save_ms",
    }
    _BATCH_SAVE_TAIL_METRICS = {
        (False, True): "layerwise_batch_save_tail_save_only_ms",
        (True, True): "layerwise_batch_save_tail_load_save_ms",
    }

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
        self.is_save = False
        self.need_load = False
        self.dump_total_ptrs: np.ndarray | None = None
        self.request_data: list[tuple[str, list, list, np.ndarray]] = []
        self._failure_req_ids: set[str] = set()
        self._layerwise_prev_wait_end: Optional[float] = None
        self._layerwise_batch_start: Optional[float] = None
        self._layerwise_batch_wait_blocking_total_ms = 0.0
        # MTP layers can be revisited several times in one speculative decode
        # batch. Keep only the last metadata snapshot for each MTP layer and
        # submit its dump after the layerwise forward finishes.
        self._deferred_mtp_dumps: dict[
            int,
            tuple[
                list[bytes],
                list[int],
                set[str],
            ],
        ] = {}
        self._is_mtp = False
        self._num_mtp_layers = 0
        self._mtp_layer_start: Optional[int] = None
        self._mtp_layer_end: Optional[int] = None
        self._init_mtp_layerwise_dump_state()
        logger.info("Init UCMLayerWiseConnector.")

    def _get_full_hit_recompute_tokens(self) -> int:
        if not self.use_layerwise:
            return super()._get_full_hit_recompute_tokens()

        speculative_config = getattr(self._vllm_config, "speculative_config", None)
        if speculative_config is None:
            return 2

        spec_token_num = getattr(speculative_config, "num_speculative_tokens", 0)
        try:
            spec_token_num = int(spec_token_num)
        except (TypeError, ValueError):
            logger.warning(
                f"Invalid speculative token count: {spec_token_num}. "
                "Fallback to recomputing two tokens on full layerwise UCM cache hit."
            )
            spec_token_num = 0

        return max(spec_token_num, 0) + 2

    def _layerwise_batch_stats(
        self, total_end: float, save_tail_ms: Optional[float] = None
    ) -> dict[str, float]:
        if self._layerwise_batch_start is None:
            return {}

        batch_type = (self.need_load, self.is_save)
        batch_total_ms = (total_end - self._layerwise_batch_start) * 1000
        stats = {
            "layerwise_batch_total_ms": batch_total_ms,
            self._BATCH_TOTAL_METRICS[batch_type]: batch_total_ms,
        }

        load_wait_metric = self._BATCH_LOAD_WAIT_METRICS.get(batch_type)
        if load_wait_metric:
            stats[load_wait_metric] = self._layerwise_batch_wait_blocking_total_ms

        save_tail_metric = self._BATCH_SAVE_TAIL_METRICS.get(batch_type)
        if save_tail_metric and save_tail_ms is not None:
            stats["layerwise_save_tail_total_ms"] = save_tail_ms
            stats[save_tail_metric] = save_tail_ms

        self._layerwise_batch_start = None
        self._layerwise_batch_wait_blocking_total_ms = 0.0
        return stats

    def _init_mtp_layerwise_dump_state(self) -> None:
        """Cache whether MTP is enabled and how many MTP layers it has."""
        speculative_config = getattr(self._vllm_config, "speculative_config", None)
        if speculative_config is None:
            return

        mtp_method = getattr(speculative_config, "method", None)
        self._is_mtp = mtp_method == "mtp" or (
            isinstance(mtp_method, str) and mtp_method.endswith("_mtp")
        )
        if not self._is_mtp:
            return

        draft_model_config = getattr(speculative_config, "draft_model_config", None)
        draft_hf_config = getattr(draft_model_config, "hf_config", None)
        value = getattr(draft_hf_config, "n_predict", None)
        if value is not None:
            try:
                self._num_mtp_layers = max(int(value), 0)
            except (TypeError, ValueError) as e:
                logger.warning(
                    f"Failed to parse MTP n_predict from draft hf_config, "
                    f"n_predict={value}. "
                    f"{type(e).__name__}: {e}"
                )

    def _refresh_mtp_layer_range(self) -> None:
        """Resolve MTP layer ids from the registered KV cache layout."""
        self._mtp_layer_start = None
        self._mtp_layer_end = None
        if not self._is_mtp or self._num_mtp_layers <= 0:
            return

        num_hidden_layers = self.kv_cache_layout.num_hidden_layers
        if num_hidden_layers <= 0:
            logger.warning_once(
                "Skip deferred MTP layerwise dump because num_hidden_layers is unknown."
            )
            return

        # MTP layers are indexed after the base model layers, e.g. a 78-layer
        # model uses layer_id 78 for its first MTP layer.
        self._mtp_layer_start = num_hidden_layers
        self._mtp_layer_end = num_hidden_layers + self._num_mtp_layers

    def _is_mtp_layer(self, layer_id: int) -> bool:
        """Check whether a global layer id belongs to the MTP layer range."""
        return (
            self._mtp_layer_start is not None
            and self._mtp_layer_end is not None
            and self._mtp_layer_start <= layer_id < self._mtp_layer_end
        )

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        super().register_kv_caches(kv_caches)
        self._refresh_mtp_layer_range()

    def _submit_layerwise_dump_task(
        self,
        layer_id: int,
        total_ucm_block_ids: list[bytes],
        total_vllm_block_ids: list[int],
        dump_request_ids: set[str],
    ) -> None:
        """Submit a layerwise dump and attach it to the existing async lifecycle."""
        if not dump_request_ids:
            return

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
        except Exception as e:
            logger.error(
                f"submit dump task for layer {layer_id} failed. {type(e).__name__}: {e}"
            )
            self._record_counter("connector_dump_submit_errors_total")
            if self.enable_event_sync and event_handle and self.device is not None:
                self.device.destroy_event_handle(event_handle)

    def _flush_deferred_mtp_dumps(self) -> None:
        """Submit the last saved metadata snapshot for each deferred MTP layer."""
        if not self._deferred_mtp_dumps:
            return

        deferred_mtp_dumps = self._deferred_mtp_dumps
        self._deferred_mtp_dumps = {}
        for layer_id in sorted(deferred_mtp_dumps):
            (
                total_ucm_block_ids,
                total_vllm_block_ids,
                dump_request_ids,
            ) = deferred_mtp_dumps[layer_id]
            self._submit_layerwise_dump_task(
                layer_id,
                total_ucm_block_ids,
                total_vllm_block_ids,
                dump_request_ids,
            )

    def _submit_request_load_tasks_for_layer(
        self,
        layer_id: int,
        local_row: int,
        metadata: "UCMConnectorMetadata",
    ) -> None:
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

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        self._layerwise_batch_start = time.perf_counter()
        metadata = self._get_connector_metadata()
        self.load_tasks.clear()
        self.request_data.clear()
        self._failure_req_ids.clear()
        self.need_load = False
        self._layerwise_prev_wait_end = None
        self._layerwise_batch_wait_blocking_total_ms = 0.0

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
            first_submit_start = time.perf_counter()
            self._submit_request_load_tasks_for_layer(self.first_layer_id, 0, metadata)
            first_submit_end = time.perf_counter()
            n_reqs = len(self.request_data) - len(self._failure_req_ids)
            ucmmetrics.update_stats(
                {
                    "layerwise_first_layer_submit_ms": (
                        first_submit_end - first_submit_start
                    )
                    * 1000,
                    "layerwise_first_layer_requests": float(n_reqs),
                }
            )

    def wait_for_layer_load(self, layer_name: str) -> None:
        if not self._connector_metadata:
            return
        if not self.need_load:
            return
        metadata = self._get_connector_metadata()
        current_layer_id = self.layer_name_to_id[layer_name]

        wait_start = time.perf_counter()

        # Pop before wait so MTP / rollback paths that revisit the same layer_name
        # do not call store.wait() again on already-completed handles.
        layer_tasks = self.load_tasks.pop(current_layer_id, {})
        n_tasks = len(layer_tasks)
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

        wait_end = time.perf_counter()

        next_layer_id = current_layer_id + 1
        has_next = next_layer_id in self.layer_ids
        if has_next:
            next_local_row = next_layer_id - self.first_layer_id
            self._submit_request_load_tasks_for_layer(
                next_layer_id, next_local_row, metadata
            )

        blocking_ms = (wait_end - wait_start) * 1000
        self._layerwise_batch_wait_blocking_total_ms += blocking_ms
        stats = {
            "layerwise_wait_blocking_ms": blocking_ms,
            "layerwise_wait_tasks_count": float(n_tasks),
        }
        if self._layerwise_prev_wait_end is not None:
            stats["layerwise_inter_wait_interval_ms"] = (
                wait_start - self._layerwise_prev_wait_end
            ) * 1000
        if has_next:
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

        metadata = self._get_connector_metadata()

        submit_start = time.perf_counter()
        total_ucm_block_ids, total_vllm_block_ids = [], []
        dump_request_ids: set[str] = set()
        layer_id = self.layer_name_to_id[layer_name]
        for request_id, request in metadata.request_meta.items():
            if len(request.dump_block_ids[0]) == 0:
                continue

            self.is_save = True
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
            if self._is_mtp_layer(layer_id):
                self._deferred_mtp_dumps[layer_id] = (
                    list(total_ucm_block_ids),
                    list(total_vllm_block_ids),
                    set(dump_request_ids),
                )
                logger.debug(
                    f"Defer MTP layerwise dump: layer={layer_name}, "
                    f"layer_id={layer_id}, blocks={len(total_ucm_block_ids)}"
                )
            else:
                self._submit_layerwise_dump_task(
                    layer_id,
                    total_ucm_block_ids,
                    total_vllm_block_ids,
                    dump_request_ids,
                )
        if self.is_save:
            submit_end = time.perf_counter()
            ucmmetrics.update_stats(
                {"layerwise_save_submit_ms": (submit_end - submit_start) * 1000}
            )

    def wait_for_save(self) -> None:
        save_tail_start = time.perf_counter()
        wait_for_save_start_ms = save_tail_start * 1000
        self._flush_deferred_mtp_dumps()
        for pending_dump_task in self._pending_dump_tasks:
            if pending_dump_task.wait_for_save_start_ms <= 0:
                pending_dump_task.wait_for_save_start_ms = wait_for_save_start_ms
        self._poll_pending_dump_tasks()
        if self._connector_metadata:
            metadata = self._get_connector_metadata()
            self._async_dump_req_ids.update(
                request_id
                for request_id, request in metadata.request_meta.items()
                if len(request.dump_block_ids[0]) > 0
            )

        total_end = time.perf_counter()
        save_tail_ms = (total_end - save_tail_start) * 1000
        stats = self._layerwise_batch_stats(total_end, save_tail_ms)
        if stats:
            ucmmetrics.update_stats(stats)
        self.is_save = False
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
        self.use_layerwise = self.launch_config.get("use_layerwise", False)

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
            # init scheduler-size connector
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
        dp_engine_id_suffix = re.compile(r"_dp\d+$")
        self.engine_id = dp_engine_id_suffix.sub(
            "", vllm_config.kv_transfer_config.engine_id
        )
        self.launch_config = ucm_config.get_config()
        self._worker_rank = (
            get_world_group().rank
            if role == KVConnectorRole.WORKER
            else "scheduler" if role == KVConnectorRole.SCHEDULER else None
        )
        self._setup_ucm_metrics(vllm_config, role)
        logger.info(f"self.launch_config: {self.launch_config}")

        use_layerwise = (
            self.launch_config.get("use_layerwise", False)
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

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """
        Update KVConnector state after block allocation.
        """
        self.connector.update_state_after_alloc(request, blocks, num_external_tokens)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """
        Initialize with the KV caches. Useful for pre-registering the
        KV Caches in the KVConnector (e.g. for NIXL).

        Args: kv_caches:
            dictionary of layer names, kv cache
        """
        self.connector.register_kv_caches(kv_caches)

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

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        """Set the connector metadata from the scheduler.

        This function should be called by the model runner every time
        before the model execution. The metadata will be used for runtime
        KV cache loading and saving.

        Args:
            connector_metadata (dict): the connector metadata.
        """
        self.connector.bind_connector_metadata(connector_metadata)

    def handle_preemptions(self, kv_connector_metadata: KVConnectorMetadata):
        self.connector.handle_preemptions(kv_connector_metadata)

    def has_connector_metadata(self) -> bool:
        """Check whether the connector metadata is currently set.

        Returns:
            bool: True if connector metadata exists, False otherwise.
        """
        return self.connector.has_connector_metadata()

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

    def wait_for_save(self) -> None:
        """
        Block until all the save operations is done. This is called
        as the forward context exits to ensure that the async saving
        from save_kv_layer is complete before finishing the forward.

        This prevents overwrites of paged KV buffer before saving done.
        """
        self.connector.wait_for_save()

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

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, object] | None]:
        return self.connector.request_finished(request, block_ids)

    def get_finished(
        self,
        finished_req_ids: set[str],
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        return self.connector.get_finished(finished_req_ids)

    def build_connector_worker_meta(self):
        return self.connector.build_connector_worker_meta()

    def update_connector_output(self, connector_output: KVConnectorOutput):
        return self.connector.update_connector_output(connector_output)

    def clear_connector_metadata(self) -> None:
        """Clear the connector metadata.

        This function should be called by the model runner every time
        after the model execution.
        """
        self.connector.clear_connector_metadata()

    def get_block_ids_with_load_errors(self) -> set[int]:
        """
        Get the set of block IDs that failed to load.

        Returns:
            Set of block IDs that encountered load errors.
            Empty set if no load errors occurred.
        """
        return self.connector.get_block_ids_with_load_errors()
