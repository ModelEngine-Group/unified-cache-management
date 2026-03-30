import copy
import hashlib
import math
import os
import pickle
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import numpy as np
import torch
import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.parallel_state import get_tp_group, get_world_group
from vllm.distributed.utils import get_pp_indices
from vllm.model_executor.models.utils import extract_layer_index
from vllm.platforms import current_platform
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
)

from ucm.integration.vllm.device import create_device
from ucm.logger import init_logger
from ucm.observability import PrometheusStatsLogger
from ucm.shared.metrics import ucmmetrics
from ucm.store.factory_v1 import UcmConnectorFactoryV1
from ucm.store.ucmstore_v1 import Task, UcmKVStoreBaseV1
from ucm.utils import Config

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

from ucm.sparse.state import has_ucm_sparse

logger = init_logger(__name__)


@dataclass
class RequestMeta:
    ucm_block_ids: list[bytes] = field(default_factory=list)
    mamba_block_ids: list[bytes] = field(default_factory=list)
    hbm_hit_block_num: int = 0
    # local_computed_block + external_computed_block
    total_hit_block_num: int = 0
    num_token_ids: int = 0
    vllm_block_ids: list[int] = field(default_factory=list)
    token_processed: int = 0
    ucm_block_ids_by_group: list[list[bytes]] = field(default_factory=list)
    vllm_block_ids_by_group: list[list[int]] = field(default_factory=list)


class KVCacheType(Enum):
    ATTENTION = "attention"
    MAMBA = "mamba"
    UNKNOWN = "unknown"


@dataclass
class RequestDispatchMeta:
    load_block_ids: dict[KVCacheType, tuple[list[bytes], list[int]]]
    dump_block_ids: dict[KVCacheType, tuple[list[bytes], list[int]]]


@dataclass
class SingleLayout:
    base_ptrs: np.ndarray
    stride_lists: np.ndarray
    tensor_size_lists: np.ndarray

    def extract_block_addrs(
        self, vllm_block_ids: List[int], layer_first: bool = False
    ) -> np.ndarray:
        vllm_block_ids_np = np.array(vllm_block_ids, np.uint64)
        if layer_first:
            return (
                vllm_block_ids_np[None, :, None] * self.stride_lists[:, None, :]
                + self.base_ptrs[:, None, :]
            )
        return (
            vllm_block_ids_np[:, None, None] * self.stride_lists[None, :, :]
            + self.base_ptrs[None, :, :]
        )

    def tensor_size_list(self, use_layerwise: bool) -> list[int]:
        return (
            self.tensor_size_lists.reshape(-1).tolist()
            if not use_layerwise
            else self.tensor_size_lists[0].tolist()
        )

    def shard_size(self, use_layerwise: bool) -> int:
        return int(
            self.tensor_size_lists[0].sum()
            if use_layerwise
            else self.tensor_size_lists.sum()
        )

    def block_size(self, pp_size: int, num_hidden_layers: int) -> int:
        if pp_size > 1:
            return int(self.tensor_size_lists[0].sum() * num_hidden_layers)
        return int(self.tensor_size_lists.sum())


class KVCacheLayout:
    def __init__(
        self,
        vllm_config: "VllmConfig",
        ucm_config: Config,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ) -> None:
        self.use_layerwise = ucm_config.get_config().get("use_layerwise", False)
        self.kv_cache_config = kv_cache_config
        self.vllm_config = vllm_config
        self.pp_size = self.vllm_config.parallel_config.pipeline_parallel_size
        self.num_hidden_layers = getattr(
            self.vllm_config.model_config.hf_text_config, "num_hidden_layers", 0
        )
        self.pp_rank = (
            self.vllm_config.parallel_config.rank
            // self.vllm_config.parallel_config.tensor_parallel_size
        ) % self.vllm_config.parallel_config.pipeline_parallel_size
        start, end = get_pp_indices(self.num_hidden_layers, self.pp_rank, self.pp_size)
        self.local_num_hidden_layers = end - start
        if self.pp_size > 1 and self.num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be > 0 when pp_size > 1")
        self.cache_block_size = self.vllm_config.cache_config.block_size
        self.layouts: dict[KVCacheType, SingleLayout] = {}
        self.layer_name_to_id: dict[str, int] = {}
        self.first_layer_id: int = 0
        self.layer_ids: list[int] = []
        self.layer_name_to_group_id: dict[str, int] = {}
        self.layer_name_to_kv_cache_type: dict[str, KVCacheType] = {}
        self.layer_name_to_raw_tensor_idx: dict[str, int] = {}
        self.group_ids_by_kv_cache_type: dict[KVCacheType, list[int]] = defaultdict(
            list
        )
        self.kv_cache_types: list[KVCacheType] = [KVCacheType.ATTENTION]
        self.kernel_block_size_scale = 1
        if self.kv_cache_config is not None:
            self._initialize_kv_cache_config()

    @property
    def is_hybrid(self) -> bool:
        return len(self.kv_cache_types) > 1

    @property
    def default_kv_cache_type(self) -> KVCacheType:
        if KVCacheType.ATTENTION in self.kv_cache_types:
            return KVCacheType.ATTENTION
        return self.kv_cache_types[0]

    def _kv_cache_type_from_spec(self, kv_cache_spec: "KVCacheSpec") -> KVCacheType:
        if isinstance(kv_cache_spec, MambaSpec):
            return KVCacheType.MAMBA
        if isinstance(kv_cache_spec, AttentionSpec):
            return KVCacheType.ATTENTION
        return KVCacheType.UNKNOWN

    def _initialize_kv_cache_config(self):
        discovered_kv_cache_types: list[KVCacheType] = []
        for group_id, kv_cache_group_spec in enumerate(
            self.kv_cache_config.kv_cache_groups
        ):
            kv_cache_type = self._kv_cache_type_from_spec(
                kv_cache_group_spec.kv_cache_spec
            )
            if (
                kv_cache_type not in discovered_kv_cache_types
                and kv_cache_type != KVCacheType.UNKNOWN
            ):
                discovered_kv_cache_types.append(kv_cache_type)
            if kv_cache_type != KVCacheType.UNKNOWN:
                self.group_ids_by_kv_cache_type[kv_cache_type].append(group_id)
            for layer_name in kv_cache_group_spec.layer_names:
                self.layer_name_to_group_id[layer_name] = group_id
                self.layer_name_to_kv_cache_type[layer_name] = kv_cache_type
        if discovered_kv_cache_types:
            self.kv_cache_types = discovered_kv_cache_types

    def initialize_kv_cache_layout(self, kvcaches):
        if self.kv_cache_config is not None:
            self._build_layout_with_kv_cache_config(kvcaches)
        else:
            self._build_layout(kvcaches)
        self.layer_ids = list(sorted(set(self.layer_name_to_id.values())))
        self.first_layer_id = self.layer_ids[0]

    def _handle_kv_layer(self, kv_layer, kv_cache_type: KVCacheType):
        ptrs = []
        strides = []
        tensor_sizes = []

        def handle_kv_tensor(t: torch.Tensor):
            ptrs.append(t.data_ptr())
            strides.append(t.stride(0) * t.element_size())
            tensor_size = (
                math.prod([t.shape[i] for i in range(1, t.dim())]) * t.element_size()
            )
            if kv_cache_type == KVCacheType.ATTENTION:
                kernel_block_size = t.shape[1]
                self.kernel_block_size_scale = (
                    self.cache_block_size // kernel_block_size
                )
                tensor_size *= self.kernel_block_size_scale
            tensor_sizes.append(tensor_size)

        if isinstance(kv_layer, torch.Tensor):
            if kv_layer.dim() == 5 and kv_layer.shape[0] == 2:
                # full attention packed KV: [2, num_blocks, ...]
                handle_kv_tensor(kv_layer[0])
                handle_kv_tensor(kv_layer[1])
            else:
                handle_kv_tensor(kv_layer)
        elif isinstance(kv_layer, (tuple, list)):
            for tensor in kv_layer:
                handle_kv_tensor(tensor)
        else:
            raise TypeError(f"Unsupported kv cache type: {type(kv_layer)}")

        return ptrs, strides, tensor_sizes

    def _create_layout(
        self,
        raw_ptr_rows: list[list[int]],
        stride_rows: list[list[int]],
        tensor_size_rows: list[list[int]],
    ) -> SingleLayout:
        return SingleLayout(
            base_ptrs=np.asarray(raw_ptr_rows, dtype=np.uint64),
            stride_lists=np.asarray(stride_rows, dtype=np.uint64),
            tensor_size_lists=np.asarray(tensor_size_rows, dtype=np.uint64),
        )

    def _build_layout(self, kvcaches):
        raw_ptr_rows = []
        stride_rows = []
        tensor_size_rows = []
        kv_cache_type = self.default_kv_cache_type

        for layer_name, kv_layer in kvcaches.items():
            self.layer_name_to_id[layer_name] = extract_layer_index(layer_name)
            ptrs, strides, tensor_sizes = self._handle_kv_layer(kv_layer, kv_cache_type)
            raw_ptr_rows.append(ptrs)
            stride_rows.append(strides)
            tensor_size_rows.append(tensor_sizes)
            self.layer_name_to_kv_cache_type.setdefault(layer_name, kv_cache_type)
            raw_tensor_idx = len(raw_ptr_rows) - 1
            self.layer_name_to_raw_tensor_idx.setdefault(layer_name, raw_tensor_idx)

        self.layouts = {
            kv_cache_type: self._create_layout(
                raw_ptr_rows, stride_rows, tensor_size_rows
            )
        }
        self.kv_cache_types = [kv_cache_type]
        layout = self.layouts[kv_cache_type]

        logger.info(
            f"layout[{kv_cache_type}]: base_ptrs {layout.base_ptrs.shape}, "
            f"stride_lists {layout.stride_lists.shape}, "
            f"tensor_size_lists {layout.tensor_size_lists.shape}"
        )

    def _build_layout_with_kv_cache_config(self, kvcaches):
        raw_ptr_rows: dict[KVCacheType, list[list[int]]] = defaultdict(list)
        stride_rows: dict[KVCacheType, list[list[int]]] = defaultdict(list)
        tensor_size_rows: dict[KVCacheType, list[list[int]]] = defaultdict(list)

        for raw_tensor_idx, kv_cache_tensor in enumerate(
            self.kv_cache_config.kv_cache_tensors
        ):
            kv_cache_type_to_layer_names: dict[KVCacheType, list[str]] = defaultdict(
                list
            )
            for layer_name in kv_cache_tensor.shared_by:
                self.layer_name_to_id[layer_name] = extract_layer_index(layer_name)
                self.layer_name_to_raw_tensor_idx[layer_name] = raw_tensor_idx
                kv_cache_type = self.layer_name_to_kv_cache_type.get(
                    layer_name, KVCacheType.UNKNOWN
                )
                kv_cache_type_to_layer_names[kv_cache_type].append(layer_name)
            for (
                kv_cache_type,
                shared_layer_names,
            ) in kv_cache_type_to_layer_names.items():
                if kv_cache_type == KVCacheType.UNKNOWN:
                    continue
                kv_layer = kvcaches.get(shared_layer_names[0])
                if kv_layer is None:
                    raise KeyError(
                        f"Layer {shared_layer_names[0]} referenced by kv_cache_config "
                        "was not found in registered KV caches."
                    )
                ptrs, strides, tensor_sizes = self._handle_kv_layer(
                    kv_layer, kv_cache_type
                )
                raw_ptr_rows[kv_cache_type].append(ptrs)
                stride_rows[kv_cache_type].append(strides)
                tensor_size_rows[kv_cache_type].append(tensor_sizes)

        self.layouts = {}
        for kv_cache_type in self.kv_cache_types:
            self.layouts[kv_cache_type] = self._create_layout(
                raw_ptr_rows[kv_cache_type],
                stride_rows[kv_cache_type],
                tensor_size_rows[kv_cache_type],
            )

        for kv_cache_type, layout in self.layouts.items():
            logger.info(
                f"layout[{kv_cache_type}]: base_ptrs {layout.base_ptrs.shape}, "
                f"stride_lists {layout.stride_lists.shape}, "
                f"tensor_size_lists {layout.tensor_size_lists.shape}"
            )

    def get_layout(self, kv_cache_type: Optional[KVCacheType] = None) -> SingleLayout:
        resolved_kv_cache_type = kv_cache_type or self.default_kv_cache_type
        return self.layouts[resolved_kv_cache_type]

    def extract_block_addrs(
        self, vllm_block_ids: List[int], kv_cache_type: Optional[KVCacheType] = None
    ) -> np.ndarray:
        resolved_kv_cache_type = kv_cache_type or self.default_kv_cache_type
        if resolved_kv_cache_type == KVCacheType.ATTENTION:
            vllm_block_ids = [
                block_id * self.kernel_block_size_scale for block_id in vllm_block_ids
            ]
        return self.get_layout(kv_cache_type).extract_block_addrs(vllm_block_ids)

    def tensor_size_list(
        self, kv_cache_type: Optional[KVCacheType] = None
    ) -> list[int]:
        return self.get_layout(kv_cache_type).tensor_size_list(self.use_layerwise)

    def shard_size(self, kv_cache_type: Optional[KVCacheType] = None) -> int:
        return self.get_layout(kv_cache_type).shard_size(self.use_layerwise)

    def block_size(self, kv_cache_type: Optional[KVCacheType] = None) -> int:
        return self.get_layout(kv_cache_type).block_size(
            self.pp_size, self.num_hidden_layers
        )


@dataclass
class UCMConnectorMetadata(KVConnectorMetadata):
    request_meta: dict[str, RequestDispatchMeta] = field(default_factory=dict)


class RequestHasher:
    """hash(md5) request to generate ucm block id"""

    def __init__(self, vllm_config, rank_id):
        meta = f"{vllm_config.model_config.model}:{vllm_config.parallel_config.tensor_parallel_size}:{vllm_config.model_config.dtype}:{rank_id}"
        self.meta_bytes = meta.encode("utf-8")

    def __call__(self, input_data) -> bytes:
        if isinstance(input_data, bytes):
            input_bytes = input_data
        else:
            input_bytes = pickle.dumps(input_data, protocol=pickle.HIGHEST_PROTOCOL)

        h = hashlib.md5(self.meta_bytes + input_bytes)
        return h.digest()


class UCMDirectConnector(KVConnectorBase_V1, SupportsHMA):
    """
    This connector means synchronize:
    load -> forward -> save
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        super().__init__(
            vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config
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
        self.stores: dict[KVCacheType, UcmKVStoreBaseV1] = {}
        self.rope_store: Optional[UcmKVStoreBaseV1] = None

        # save block info, avoid hash request twice, and track them until request finished
        self.requests_meta: dict[str, RequestMeta] = {}

        ucm_config = Config(vllm_config.kv_transfer_config)
        self.engine_id = vllm_config.kv_transfer_config.engine_id
        self.launch_config = ucm_config.get_config()
        self.connector_configs = self.launch_config.get("ucm_connectors", [])
        self.enable_event_sync = self.launch_config.get("enable_event_sync", True)
        assert len(self.connector_configs) > 0, "no storage connector name in config."

        self.chunk_size = self.block_size
        self.blocks_per_chunk = self.chunk_size // self.block_size
        self.kv_cache_config = kv_cache_config
        self.kv_cache_layout = KVCacheLayout(
            self._vllm_config,
            ucm_config,
            getattr(self, "kv_cache_config", None),
        )

        if role == KVConnectorRole.SCHEDULER:
            self.request_hasher = RequestHasher(vllm_config, 0)
            self._seed = self.request_hasher("UCM_HASH_SEED")
            self.stores = self._create_stores(self.kv_cache_layout)
            self.store = self.stores[self.kv_cache_layout.default_kv_cache_type]
        else:
            self.request_hasher = RequestHasher(
                vllm_config, self.tp_rank % self.tp_size
            )

        self.metrics_config = self.launch_config.get("metrics_config_path", "")
        if self.metrics_config:
            worker_id = (
                get_world_group().rank
                if role == KVConnectorRole.WORKER
                else self.engine_id
            )
            self.stats_logger = PrometheusStatsLogger(
                vllm_config.model_config.served_model_name,
                worker_id,
                self.metrics_config,
            )
            logger.info(
                f"metrics_config_path: {self.metrics_config}, set worker_id: {worker_id}"
            )

        # invalid block ids due to load errors
        self._invalid_block_ids: set[int] = set()

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

    def _generate_mamba_block_ids(self, attn_block_id: bytes) -> list[bytes]:
        if not hasattr(self, "kv_cache_layout"):
            return []
        mamba_group_ids = self.kv_cache_layout.group_ids_by_kv_cache_type.get(
            KVCacheType.MAMBA, []
        )
        return [
            self.request_hasher((attn_block_id, f"group_{group_id}"))
            for group_id in mamba_group_ids
        ]

    def _validate_attn_hit_with_mamba(
        self, attn_block_id: bytes
    ) -> tuple[bool, list[bytes]]:
        mamba_store = self.stores.get(KVCacheType.MAMBA)
        if mamba_store is None:
            return True, []

        mamba_block_ids = self._generate_mamba_block_ids(attn_block_id)
        if not mamba_block_ids:
            return True, []

        lookup_result = mamba_store.lookup(mamba_block_ids)
        if all(lookup_result):
            return True, mamba_block_ids
        return False, []

    def _create_store(
        self,
        kv_cache_layout: Optional[KVCacheLayout],
        kv_cache_type: KVCacheType,
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
        if "storage_backends" in config:
            backends = [path for path in config["storage_backends"].split(":")]
            config["storage_backends"] = backends
        config["unique_id"] = f"{self.engine_id}:{kv_cache_type.value}"
        if self._role == KVConnectorRole.WORKER:
            config["device_id"] = self.local_rank
            config["tensor_size_list"] = (
                kv_cache_layout.tensor_size_list(kv_cache_type) * self.blocks_per_chunk
            )
            config["shard_size"] = (
                kv_cache_layout.shard_size(kv_cache_type) * self.blocks_per_chunk
            )
            config["block_size"] = (
                kv_cache_layout.block_size(kv_cache_type) * self.blocks_per_chunk
            )
            config["local_rank_size"] = self.tp_size if self.is_mla else 1
            if cpu_affinity_cores:
                config["cpu_affinity_cores"] = list(cpu_affinity_cores)
        logger.info(f"create {name} with config: {config}")
        return UcmConnectorFactoryV1.create_connector(name, config, module_path)

    def _create_stores(
        self,
        kv_cache_layout: Optional[KVCacheLayout],
        cpu_affinity_cores: Optional[list[int]] = None,
    ) -> dict[KVCacheType, UcmKVStoreBaseV1]:
        kv_cache_types = (
            kv_cache_layout.kv_cache_types
            if kv_cache_layout is not None
            else [KVCacheType.ATTENTION]
        )
        return {
            kv_cache_type: self._create_store(
                kv_cache_layout, kv_cache_type, cpu_affinity_cores
            )
            for kv_cache_type in kv_cache_types
        }

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
        self.kv_cache_layout.initialize_kv_cache_layout(self.kv_caches)
        self.block_data_size = self.kv_cache_layout.block_size()
        self.layer_name_to_id = self.kv_cache_layout.layer_name_to_id
        self.layer_ids = self.kv_cache_layout.layer_ids
        self.first_layer_id = self.kv_cache_layout.first_layer_id

        self.device = create_device()

        enable_affinity = os.getenv("VLLM_CPU_AFFINITY") == "1"
        worker_cores, store_cores = (
            self.device.split_cores(self.local_rank)
            if enable_affinity
            else (None, None)
        )

        self.stores: dict[KVCacheType, UcmKVStoreBaseV1] = self._create_stores(
            self.kv_cache_layout, store_cores
        )
        self.store = self.stores[self.kv_cache_layout.default_kv_cache_type]

        if worker_cores:
            try:
                os.sched_setaffinity(0, worker_cores)
                logger.info(f"[VLLM CPU Affinity] Worker bound to cores {worker_cores}")
            except Exception as e:
                logger.warning(f"Failed to bind worker: {e}")

        if self.device is None:
            raise RuntimeError(f"Unsupported device platform for UCMDirectConnector.")

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        assert num_computed_tokens % self.block_size == 0
        hbm_hit_block_num = num_computed_tokens // self.block_size

        ucm_block_ids = self.generate_hash(
            self.block_size, request.all_token_ids, self._seed
        )

        external_block_ids = ucm_block_ids[hbm_hit_block_num:]
        if not external_block_ids:
            return 0, False
        mamba_block_ids: list[bytes] = []
        try:
            if self.kv_cache_layout.is_hybrid:
                attn_store = self.stores.get(KVCacheType.ATTENTION, self.store)
                external_hit_blocks = (
                    attn_store.lookup_on_prefix(external_block_ids) + 1
                )
            else:
                external_hit_blocks = (
                    self.store.lookup_on_prefix(external_block_ids) + 1
                )
            if self.kv_cache_layout.is_hybrid and external_hit_blocks > 0:
                last_hit_attn_block_id = external_block_ids[external_hit_blocks - 1]
                mamba_hit, mamba_block_ids = self._validate_attn_hit_with_mamba(
                    last_hit_attn_block_id
                )
                if not mamba_hit:
                    external_hit_blocks = 0
        except RuntimeError as e:
            external_hit_blocks = 0
            logger.error(f"request {request.request_id} look up error. {e}")
        logger.info_once(
            f"request_id: {request.request_id}, "
            f"total_blocks_num: {len(ucm_block_ids)}, "
            f"hit hbm: {hbm_hit_block_num}, "
            f"hit external: {external_hit_blocks}"
        )
        if self.metrics_config:
            ucmmetrics.update_stats(
                {"interval_lookup_hit_rates": external_hit_blocks / len(ucm_block_ids)},
            )

        total_hit_block_num = hbm_hit_block_num + external_hit_blocks

        external_hit_tokens = external_hit_blocks * self.block_size

        # When all the tokens are cached in ssd or hbm,
        # we need to recompute the last token. This if condition will be removed
        # once vLLM scheduler provides a better solution in the future.
        num_total_hit_tokens = total_hit_block_num * self.block_size
        if num_total_hit_tokens == request.num_tokens:
            external_hit_tokens -= 1

        self.requests_meta[request.request_id] = RequestMeta(
            ucm_block_ids=ucm_block_ids,
            mamba_block_ids=mamba_block_ids,
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

    def _extend_vllm_block_ids_by_group(
        self, req_meta: RequestMeta, vllm_block_ids_by_group: list[list[int]]
    ) -> None:
        normalized_block_ids = [
            list(group_block_ids) for group_block_ids in vllm_block_ids_by_group
        ]
        if not req_meta.vllm_block_ids_by_group:
            req_meta.vllm_block_ids_by_group = [
                [] for _ in range(len(normalized_block_ids))
            ]
        for target, source in zip(
            req_meta.vllm_block_ids_by_group, normalized_block_ids
        ):
            target.extend(source)

    @staticmethod
    def _nonzero_block_ids(block_ids: list[int]) -> list[int]:
        return [block_id for block_id in block_ids if block_id != 0]

    def _generate_dispatch_meta(
        self,
        req_meta: RequestMeta,
        new_tokens: int,
        vllm_block_ids: list[list[int]],
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
        self._extend_vllm_block_ids_by_group(req_meta, vllm_block_ids)

        load_block_ids: dict[KVCacheType, tuple[list[bytes], list[int]]] = {}
        dump_block_ids: dict[KVCacheType, tuple[list[bytes], list[int]]] = {}

        group_ids_by_kv_cache_type = self.kv_cache_layout.group_ids_by_kv_cache_type
        attn_group_ids = group_ids_by_kv_cache_type.get(KVCacheType.ATTENTION, [])
        mamba_group_ids = group_ids_by_kv_cache_type.get(KVCacheType.MAMBA, [])
        assert (
            len(attn_group_ids) == 1
        ), "Current hybrid path expects exactly one attention group."
        attn_group_id = attn_group_ids[0]
        attn_vllm_block_ids = req_meta.vllm_block_ids_by_group[attn_group_id]

        if need_load:
            # Attention uses the dense block timeline directly.
            attn_load_ucm_block_ids = req_meta.ucm_block_ids[
                hbm_hit_block_num:total_hit_block_num
            ]
            attn_load_vllm_block_ids = attn_vllm_block_ids[
                hbm_hit_block_num:total_hit_block_num
            ]
            if attn_load_ucm_block_ids and attn_load_vllm_block_ids:
                load_block_ids[KVCacheType.ATTENTION] = (
                    list(attn_load_ucm_block_ids),
                    list(attn_load_vllm_block_ids),
                )

        if need_load and self.kv_cache_layout.is_hybrid and req_meta.mamba_block_ids:
            mamba_load_pairs: list[tuple[bytes, int]] = []
            # For Mamba, the penultimate non-zero block is the load target.
            for group_id, mamba_ucm_block_id in zip(
                mamba_group_ids, req_meta.mamba_block_ids
            ):
                nonzero_vllm_block_ids = self._nonzero_block_ids(
                    req_meta.vllm_block_ids_by_group[group_id]
                )
                if len(nonzero_vllm_block_ids) >= 2:
                    mamba_load_pairs.append(
                        (mamba_ucm_block_id, nonzero_vllm_block_ids[-1])
                    )
            if mamba_load_pairs:
                load_block_ids[KVCacheType.MAMBA] = (
                    [ucm_block_id for ucm_block_id, _ in mamba_load_pairs],
                    [vllm_block_id for _, vllm_block_id in mamba_load_pairs],
                )

        if req_meta.token_processed < req_meta.num_token_ids:
            start_idx = req_meta.token_processed // self.block_size
            end_idx = (req_meta.token_processed + new_tokens) // self.block_size
            attn_dump_ucm_block_ids = req_meta.ucm_block_ids[start_idx:end_idx]
            attn_dump_vllm_block_ids = attn_vllm_block_ids[start_idx:end_idx]

            if attn_dump_ucm_block_ids and attn_dump_vllm_block_ids:
                # Attention dump follows the newly scheduled dense blocks.
                dump_block_ids[KVCacheType.ATTENTION] = (
                    list(attn_dump_ucm_block_ids),
                    list(attn_dump_vllm_block_ids),
                )

                if self.kv_cache_layout.is_hybrid:
                    mamba_dump_pairs: list[tuple[bytes, int]] = []
                    last_attn_dump_block_id = attn_dump_ucm_block_ids[-1]
                    # Mamba dump keys are derived from the last attention dump key.
                    for group_id, mamba_ucm_block_id in zip(
                        mamba_group_ids,
                        self._generate_mamba_block_ids(last_attn_dump_block_id),
                    ):
                        nonzero_vllm_block_ids = self._nonzero_block_ids(
                            req_meta.vllm_block_ids_by_group[group_id]
                        )
                        if nonzero_vllm_block_ids:
                            mamba_dump_pairs.append(
                                (mamba_ucm_block_id, nonzero_vllm_block_ids[-1])
                            )
                    if mamba_dump_pairs:
                        dump_block_ids[KVCacheType.MAMBA] = (
                            [ucm_block_id for ucm_block_id, _ in mamba_dump_pairs],
                            [vllm_block_id for _, vllm_block_id in mamba_dump_pairs],
                        )

            req_meta.token_processed += new_tokens

        return RequestDispatchMeta(load_block_ids, dump_block_ids)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        requests_dispatch_meta = {}
        # for new request, we need to load and dump
        for request in scheduler_output.scheduled_new_reqs:
            request_id = request.req_id
            req_meta = self.requests_meta.get(request_id)
            if req_meta:
                requests_dispatch_meta[request_id] = self._generate_dispatch_meta(
                    req_meta,
                    scheduler_output.num_scheduled_tokens[request_id],
                    list(request.block_ids),
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
                    new_block_ids: list[list[int]] = []
                    if scheduled_cached_reqs.new_block_ids[i] is not None:
                        new_block_ids = list(scheduled_cached_reqs.new_block_ids[i])
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
                        list(request.new_block_ids),
                        request.resumed_from_preemption,
                    )

        # clear finished request
        for request_id in scheduler_output.finished_req_ids:
            self.requests_meta.pop(request_id, None)

        return UCMConnectorMetadata(requests_dispatch_meta)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, UCMConnectorMetadata)

        pending_tasks: list[tuple["UcmKVStoreBaseV1", str, KVCacheType, Task]] = []
        is_load = False
        num_loaded_block = 0
        num_loaded_request = 0
        load_start_time = time.perf_counter() * 1000
        for request_id, request in metadata.request_meta.items():
            if not request.load_block_ids:
                continue
            is_load = True
            for kv_cache_type, (
                load_ucm_ids,
                load_vllm_ids,
            ) in request.load_block_ids.items():
                store = self.stores.get(kv_cache_type, self.store)
                if not load_ucm_ids or not load_vllm_ids:
                    continue
                num_loaded_block += len(load_ucm_ids)
                num_loaded_request += 1

                if self.tp_rank != 0 and not self.is_mla:
                    for i, ucm_id in enumerate(load_ucm_ids):
                        load_ucm_ids[i] = self.request_hasher(ucm_id)

                total_ptrs = self.kv_cache_layout.extract_block_addrs(
                    load_vllm_ids, kv_cache_type
                )
                total_ptrs = total_ptrs.reshape(total_ptrs.shape[0], -1)
                shard_idxs = [0] * len(load_ucm_ids)
                try:
                    task = store.load_data(load_ucm_ids, shard_idxs, total_ptrs)
                    pending_tasks.append((store, request_id, kv_cache_type, task))
                except RuntimeError as e:
                    logger.error(f"request {request_id} submit load task error. {e}")
                    self._invalid_block_ids.update(load_vllm_ids)
                    num_loaded_block -= len(load_ucm_ids)

        for store, request_id, kv_cache_type, task in pending_tasks:
            try:
                store.wait(task)
            except RuntimeError as e:
                logger.error(f"request {request_id} wait load task error. {e}")
                self._invalid_block_ids.update(
                    metadata.request_meta[request_id].load_block_ids[kv_cache_type][1]
                )
                num_loaded_block -= len(
                    metadata.request_meta[request_id].load_block_ids[kv_cache_type][0]
                )

        load_end_time = time.perf_counter() * 1000
        load_speed = (
            num_loaded_block
            * self.block_data_size
            / (load_end_time - load_start_time)
            / 1024
            / 1024
        )  # GB/s
        if self.metrics_config and is_load:
            ucmmetrics.update_stats(
                {
                    "load_requests_num": num_loaded_request,
                    "load_blocks_num": num_loaded_block,
                    "load_duration": load_end_time - load_start_time,
                    "load_speed": load_speed,
                }
            )

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

    def wait_for_save(self) -> None:
        # TODO support PP
        if self.is_mla and self.tp_rank != 0:
            return

        metadata = self._get_connector_metadata()
        assert isinstance(metadata, UCMConnectorMetadata)

        dump_tasks: list[tuple["UcmKVStoreBaseV1", Task]] = []
        is_save = False
        num_saved_block = 0
        num_saved_request = 0
        total_ucm_block_ids: dict[KVCacheType, list[bytes]] = defaultdict(list)
        total_vllm_block_ids: dict[KVCacheType, list[int]] = defaultdict(list)
        for _, request in metadata.request_meta.items():
            if not request.dump_block_ids:
                continue
            for kv_cache_type, (
                ucm_block_ids,
                vllm_block_ids,
            ) in request.dump_block_ids.items():
                if not ucm_block_ids or not vllm_block_ids:
                    continue
                is_save = True
                num_saved_block += len(ucm_block_ids)
                num_saved_request += 1
                total_ucm_block_ids[kv_cache_type].extend(ucm_block_ids)
                total_vllm_block_ids[kv_cache_type].extend(vllm_block_ids)

        if is_save:
            save_start_time = time.perf_counter() * 1000
            for kv_cache_type, ucm_block_ids in total_ucm_block_ids.items():
                store = self.stores.get(kv_cache_type, self.store)
                ucm_ids = list(ucm_block_ids)
                if self.tp_rank != 0:
                    for i, ucm_block_id in enumerate(ucm_ids):
                        ucm_ids[i] = self.request_hasher(ucm_block_id)

                vllm_block_ids = total_vllm_block_ids[kv_cache_type]
                total_ptrs = self.kv_cache_layout.extract_block_addrs(
                    vllm_block_ids, kv_cache_type
                )
                total_ptrs = total_ptrs.reshape(total_ptrs.shape[0], -1)
                shard_indexs = [0] * len(ucm_ids)
                try:
                    event_handle = self._get_dump_event_handle()
                    task = store.dump_data(
                        ucm_ids, shard_indexs, total_ptrs, event_handle
                    )
                    dump_tasks.append((store, task))
                except RuntimeError as e:
                    logger.error(f"dump kv cache failed. {e}")
                    return

        if is_save and dump_tasks:
            try:
                for store, task in dump_tasks:
                    store.wait(task)
                save_end_time = time.perf_counter() * 1000
            except RuntimeError as e:
                logger.error(f"wait for dump kv cache failed.{e}")
                return

            save_speed = (
                num_saved_block
                * self.block_data_size
                / (save_end_time - save_start_time)
                / 1024
                / 1024
            )  # GB/s
            if self.metrics_config:
                ucmmetrics.update_stats(
                    {
                        "save_requests_num": num_saved_request,
                        "save_blocks_num": num_saved_block,
                        "save_duration": save_end_time - save_start_time,
                        "save_speed": save_speed,
                    },
                )

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

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        return False, None


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
        self.dump_tasks: dict[str, Task] = {}
        self.use_layerwise = True
        self.is_save = False
        self.need_load = False
        self.dump_total_ptrs: np.ndarray | None = None
        self.request_data: list[tuple[str, list, np.ndarray]] = []
        self._failure_req_ids: set[str] = set()
        logger.info("Init UCMLayerWiseConnector.")

    def _submit_request_load_tasks_for_layer(
        self,
        layer_id: int,
        local_row: int,
        metadata: "UCMConnectorMetadata",
    ) -> None:
        for request_id, ucm_block_ids, total_ptrs in self.request_data:
            if request_id in self._failure_req_ids:
                continue
            try:
                shard_indexs = [layer_id] * len(ucm_block_ids)
                layer_ptrs = total_ptrs[local_row]
                task = self.store.load_data(ucm_block_ids, shard_indexs, layer_ptrs)
                self.load_tasks[layer_id][request_id] = task
            except RuntimeError as e:
                logger.error(f"request {request_id} submit load task error. {e}")
                self._invalid_block_ids.update(
                    metadata.request_meta[request_id].load_block_ids[1]
                )
                self._failure_req_ids.add(request_id)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        metadata = self._get_connector_metadata()
        self.load_tasks.clear()
        self.request_data.clear()
        self._failure_req_ids.clear()
        self.need_load = False

        for request_id, request in metadata.request_meta.items():
            if len(request.load_block_ids[0]) == 0:
                continue

            self.need_load = True
            ucm_block_ids, vllm_block_ids = request.load_block_ids
            if self.tp_rank % self.tp_size != 0 and not self.is_mla:
                for i, ucm_block_id in enumerate(ucm_block_ids):
                    ucm_block_ids[i] = self.request_hasher(ucm_block_id)
            total_ptrs = self.kv_cache_layout.extract_block_addrs(
                vllm_block_ids, layer_first=True
            )
            self.request_data.append((request_id, ucm_block_ids, total_ptrs))

        if self.need_load:
            self._submit_request_load_tasks_for_layer(self.first_layer_id, 0, metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        if not self.need_load:
            return
        metadata = self._get_connector_metadata()
        current_layer_id = self.layer_name_to_id[layer_name]

        for request_id, task in self.load_tasks.get(current_layer_id, {}).items():
            try:
                self.store.wait(task)
            except RuntimeError as e:
                logger.error(f"request {request_id} wait {layer_name} load failed. {e}")
                self._invalid_block_ids.update(
                    metadata.request_meta[request_id].load_block_ids[1]
                )
                self._failure_req_ids.add(request_id)

        next_layer_id = current_layer_id + 1
        if next_layer_id not in self.layer_ids:
            return
        next_local_row = next_layer_id - self.first_layer_id

        self._submit_request_load_tasks_for_layer(
            next_layer_id, next_local_row, metadata
        )

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        # TODO support PP
        if self.is_mla and self.tp_rank % self.tp_size != 0:
            return

        metadata = self._get_connector_metadata()

        total_ucm_block_ids, total_vllm_block_ids = [], []
        layer_id = self.layer_name_to_id[layer_name]
        local_layer_id = layer_id - self.first_layer_id
        for _, request in metadata.request_meta.items():
            if len(request.dump_block_ids[0]) == 0:
                continue

            self.is_save = True
            ucm_block_ids, vllm_block_ids = request.dump_block_ids
            if self.tp_rank % self.tp_size != 0 and local_layer_id == 0:
                for i, ucm_block_id in enumerate(ucm_block_ids):
                    ucm_block_ids[i] = self.request_hasher(ucm_block_id)
            total_ucm_block_ids.extend(ucm_block_ids)
            total_vllm_block_ids.extend(vllm_block_ids)

        if self.is_save:
            if self.dump_total_ptrs is None:
                self.dump_total_ptrs = self.kv_cache_layout.extract_block_addrs(
                    total_vllm_block_ids, layer_first=True
                )
            shard_indexs = [layer_id] * len(total_ucm_block_ids)
            try:
                layer_ptrs = np.ascontiguousarray(self.dump_total_ptrs[local_layer_id])
                event_handle = self._get_dump_event_handle()
                task = self.store.dump_data(
                    total_ucm_block_ids, shard_indexs, layer_ptrs, event_handle
                )
                self.dump_tasks[layer_name] = task
            except RuntimeError as e:
                logger.error(f"submit dump task failed. {e}")

    def wait_for_save(self) -> None:
        if not self.is_save:
            return
        try:
            for layer_name in self.kv_caches:
                if layer_name in self.dump_tasks:
                    self.store.wait(self.dump_tasks[layer_name])
        except RuntimeError as e:
            logger.error(f"wait for dump kv cache failed. {e}")
        self.dump_tasks.clear()
        self.is_save = False
        self.dump_total_ptrs = None
        if self.enable_event_sync:
            self.device.destroy_event_handles()


class UCMCPConnector(UCMLayerWiseConnector):
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
            self._seed = self.request_hasher("UCM_HASH_SEED")
            # init scheduler-size connector
            self.store = self._create_store(None)
        else:
            self.request_hasher = RequestHasher(vllm_config, self.tp_rank)
        vllm_config.parallel_config.tensor_parallel_size = old_tp_size
        self.hash_block_size = self.block_size
        self.block_size *= self.cp_world_size
        logger.info("Init UCMCPConnector.")

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

        external_block_ids = ucm_block_ids[hbm_hit_block_num * self.cp_world_size :]
        if not external_block_ids:
            return 0, False
        try:
            external_hit_blocks = self.store.lookup_on_prefix(external_block_ids) + 1
            external_hit_blocks //= self.cp_world_size
        except RuntimeError as e:
            external_hit_blocks = 0
            logger.error(f"request {request.request_id} look up error. {e}")
        logger.info(
            f"request_id: {request.request_id}, "
            f"total_blocks_num: {len(ucm_block_ids)}, "
            f"hit hbm: {hbm_hit_block_num * self.cp_world_size}, "
            f"hit external: {external_hit_blocks * self.cp_world_size}"
        )
        if self.metrics_config:
            ucmmetrics.update_stats(
                {
                    "interval_lookup_hit_rates": external_hit_blocks
                    * self.cp_world_size
                    / len(ucm_block_ids)
                },
            )

        total_hit_block_num = hbm_hit_block_num + external_hit_blocks

        external_hit_tokens = external_hit_blocks * self.block_size

        # When all the tokens are cached in ssd or hbm,
        # we need to recompute the last token. This if condition will be removed
        # once vLLM scheduler provides a better solution in the future.
        num_total_hit_tokens = total_hit_block_num * self.block_size
        if num_total_hit_tokens == request.num_tokens:
            external_hit_tokens -= 1

        self.requests_meta[request.request_id] = RequestMeta(
            ucm_block_ids=ucm_block_ids,
            hbm_hit_block_num=hbm_hit_block_num,
            total_hit_block_num=total_hit_block_num,
            num_token_ids=len(request.all_token_ids),
            token_processed=num_total_hit_tokens,
        )

        return external_hit_tokens, False

    def _generate_dispatch_meta(
        self,
        req_meta: RequestMeta,
        new_tokens: int,
        vllm_block_ids: list[int],
        need_load: bool = True,
    ) -> RequestDispatchMeta:
        # Since the block_size on the scheduler side is multiplied by cp_world_size,
        # while the block_size on the UCM side remains unchanged,
        # the selected ucm_blocks need to be expanded by a factor of cp_world_size.
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

        return RequestDispatchMeta(
            (load_ucm_block_ids, load_vllm_block_ids),
            (dump_ucm_block_ids, dump_vllm_block_ids),
        )

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

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config, role)

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


class UCMConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        super().__init__(
            vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config
        )
        self.connector: KVConnectorBase_V1
        ucm_config = Config(vllm_config.kv_transfer_config)
        self.launch_config = ucm_config.get_config()
        logger.info(f"self.launch_config: {self.launch_config}")

        use_layerwise = (
            self.launch_config.get("use_layerwise", False)
            if self.launch_config is not None
            else False
        )
        pp_enabled = self._vllm_config.parallel_config.pipeline_parallel_size > 1
        if pp_enabled and not use_layerwise:
            raise RuntimeError(
                "Pipeline parallelism is not supported in UCMDirectConnector, please set use_layerwise=True."
            )
        if self.launch_config is not None and "hit_ratio" in self.launch_config:
            self.connector = UCMMockConnector(vllm_config, role, kv_cache_config)
        elif (
            hasattr(self._vllm_config.parallel_config, "prefill_context_parallel_size")
            and hasattr(
                self._vllm_config.parallel_config, "decode_context_parallel_size"
            )
            and self._vllm_config.parallel_config.prefill_context_parallel_size
            * self._vllm_config.parallel_config.decode_context_parallel_size
            > 1
        ):
            self.connector = UCMCPConnector(vllm_config, role, kv_cache_config)
        elif use_layerwise:
            self.connector = UCMLayerWiseConnector(vllm_config, role, kv_cache_config)
        else:
            self.connector = UCMDirectConnector(vllm_config, role, kv_cache_config)

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
        return self.connector.get_num_new_matched_tokens(request, num_computed_tokens)

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

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        return self.connector.request_finished_all_groups(request, block_ids)
