import copy
import hashlib
import math
import os
import pickle
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
from vllm.distributed.parallel_state import get_world_group
from vllm.model_executor.models.utils import extract_layer_index
from vllm.platforms import current_platform
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.outputs import KVConnectorOutput

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
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

from ucm.sparse.state import has_ucm_sparse

logger = init_logger(__name__)


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
class HMARequestMeta(RequestMeta):
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
      scheduler allocation snapshot by :meth:`UCMHMAConnector.update_state_after_alloc`
      and maintained by :meth:`UCMHMAConnector._generate_hma_dispatch_meta`.
      HMA dispatch later slices these per-group tables to build the flattened
      load/dump pairs consumed by the inherited I/O path.
    """

    group_ucm_block_ids: list[list[bytes]] = field(default_factory=list)
    group_vllm_block_ids: list[list[int]] = field(default_factory=list)


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
        self.layer_name_to_kv_cache_spec = layer_name_to_kv_cache_spec(kv_cache_config)
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


class UCMDirectConnector(KVConnectorBase_V1):
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
        self.engine_id = vllm_config.kv_transfer_config.engine_id
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

        defer_scheduler_store = getattr(self, "_defer_scheduler_store", False)
        if role == KVConnectorRole.SCHEDULER:
            self.request_hasher = RequestHasher(vllm_config, 0)
            self._seed = self.request_hasher("UCM_HASH_SEED")
            # init scheduler-size connector
            if not defer_scheduler_store:
                self.store = self._create_store(None)
        else:
            self.request_hasher = RequestHasher(
                vllm_config, self.tp_rank % self.tp_size
            )

        metrics_config = self.launch_config.get("metrics_config_path", "")
        if metrics_config:
            worker_id = (
                f"{self.engine_id}_{get_world_group().rank}"
                if role == KVConnectorRole.WORKER
                else self.engine_id
            )
            self.stats_logger = PrometheusStatsLogger(
                vllm_config.model_config.served_model_name,
                worker_id,
                metrics_config,
            )
            logger.info(
                f"metrics_config_path: {metrics_config}, set worker_id: {worker_id}"
            )

        self.persist_token_threshold = self.launch_config.get(
            "persist_token_threshold", 0
        )

        # invalid block ids due to load errors
        self._invalid_block_ids: set[int] = set()
        self.cp_world_size = 1
        self.hash_block_size = self.block_size
        self.block_size *= self.cp_world_size

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
        if "storage_backends" in config:
            backends = [path for path in config["storage_backends"].split(":")]
            config["storage_backends"] = backends
        config["unique_id"] = f"{self.engine_id}"
        if self._role == KVConnectorRole.WORKER:
            config["device_id"] = self.local_rank
            config["tensor_size_list"] = (
                kv_cache_layout.tensor_size_list * self.blocks_per_chunk
            )
            config["shard_size"] = kv_cache_layout.shard_size * self.blocks_per_chunk
            config["block_size"] = kv_cache_layout.block_size * self.blocks_per_chunk
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
            if cpu_affinity_cores:
                config["cpu_affinity_cores"] = list(cpu_affinity_cores)
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

        enable_affinity = os.getenv("VLLM_CPU_AFFINITY") == "1"
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
        if not external_block_ids:
            return 0, False
        try:
            external_hit_blocks = self.store.lookup_on_prefix(external_block_ids) + 1
            external_hit_blocks //= self.cp_world_size
        except Exception as e:
            external_hit_blocks = 0
            logger.error(
                f"request {request.request_id} look up error. {type(e).__name__}: {e}"
            )

        logger.info_once(
            f"request_id: {request.request_id}, "
            f"total_blocks_num: {len(ucm_block_ids)}, "
            f"hit hbm: {hbm_hit_block_num * self.cp_world_size}, "
            f"hit external: {external_hit_blocks * self.cp_world_size}"
        )
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

        return UCMConnectorMetadata(requests_dispatch_meta)

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
            if self.tp_rank != 0 and not self.is_mla:
                for i, ucm_block_id in enumerate(ucm_block_ids):
                    ucm_block_ids[i] = self.request_hasher(ucm_block_id)
            try:
                total_ptrs = self.kv_cache_layout.extract_block_addrs(vllm_block_ids)
                total_ptrs = total_ptrs.reshape(total_ptrs.shape[0], -1)
                shard_indexs = [0] * len(ucm_block_ids)
                task = self.store.load_data(ucm_block_ids, shard_indexs, total_ptrs)
                request_to_task[request_id] = task
                request_to_load_blocks[request_id] = len(ucm_block_ids)
            except Exception as e:
                logger.error(
                    f"request {request_id} submit load task error. {type(e).__name__}: {e}"
                )
                self._invalid_block_ids.update(
                    metadata.request_meta[request_id].load_block_ids[1]
                )
                num_loaded_block -= len(ucm_block_ids)

        for request_id, task in request_to_task.items():
            try:
                self.store.wait(task)
            except Exception as e:
                logger.error(
                    f"request {request_id} wait load task error. {type(e).__name__}: {e}"
                )
                self._invalid_block_ids.update(
                    metadata.request_meta[request_id].load_block_ids[1]
                )
                num_loaded_block -= request_to_load_blocks.get(request_id, 0)

        load_end_time = time.perf_counter() * 1000
        load_bytes = num_loaded_block * self.block_data_size
        load_speed = (
            load_bytes / (load_end_time - load_start_time) / 1024 / 1024
        )  # GB/s
        if is_load:
            ucmmetrics.update_stats(
                {
                    "load_requests_num": num_loaded_request,
                    "load_blocks_num": num_loaded_block,
                    "load_duration": load_end_time - load_start_time,
                    "load_speed": load_speed,
                    "load_bytes_total": load_bytes,
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

        dump_tasks: List[Task] = []
        is_save = False
        num_saved_block = 0
        num_saved_request = 0
        total_ucm_block_ids, total_vllm_block_ids = [], []
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
            num_saved_block += len(ucm_block_ids)
            num_saved_request += 1
            if self.tp_rank != 0:
                for i, ucm_block_id in enumerate(ucm_block_ids):
                    ucm_block_ids[i] = self.request_hasher(ucm_block_id)
            total_ucm_block_ids.extend(ucm_block_ids)
            total_vllm_block_ids.extend(vllm_block_ids)

        if is_save:
            try:
                total_ptrs = self.kv_cache_layout.extract_block_addrs(
                    total_vllm_block_ids
                )
                total_ptrs = total_ptrs.reshape(total_ptrs.shape[0], -1)
                shard_indexs = [0] * len(total_ucm_block_ids)
                event_handle = self._get_dump_event_handle()
                save_start_time = time.perf_counter() * 1000
                task = self.store.dump_data(
                    total_ucm_block_ids, shard_indexs, total_ptrs, event_handle
                )
                dump_tasks.append(task)
            except Exception as e:
                logger.error(f"dump kv cache failed. {type(e).__name__}: {e}")
                return

            try:
                for task in dump_tasks:
                    self.store.wait(task)
                save_end_time = time.perf_counter() * 1000
            except Exception as e:
                logger.error(f"wait for dump kv cache failed. {type(e).__name__}: {e}")
                return

            save_bytes = num_saved_block * self.block_data_size
            save_speed = (
                save_bytes / (save_end_time - save_start_time) / 1024 / 1024
            )  # GB/s
            ucmmetrics.update_stats(
                {
                    "save_requests_num": num_saved_request,
                    "save_blocks_num": num_saved_block,
                    "save_duration": save_end_time - save_start_time,
                    "save_speed": save_speed,
                    "save_bytes_total": save_bytes,
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
    ) -> tuple[bool, dict[str, object] | None]:
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
        self._layerwise_prev_wait_end: Optional[float] = None
        self._layerwise_batch_start: Optional[float] = None
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
            except Exception as e:
                logger.error(
                    f"request {request_id} submit load task error. {type(e).__name__}: {e}"
                )
                self._invalid_block_ids.update(
                    metadata.request_meta[request_id].load_block_ids[1]
                )
                self._failure_req_ids.add(request_id)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        self._layerwise_batch_start = time.perf_counter()
        metadata = self._get_connector_metadata()
        self.load_tasks.clear()
        self.request_data.clear()
        self._failure_req_ids.clear()
        self.need_load = False
        self._layerwise_prev_wait_end = None

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
                self.store.wait(task)
            except Exception as e:
                logger.error(
                    f"request {request_id} wait {layer_name} load failed. {type(e).__name__}: {e}"
                )
                self._invalid_block_ids.update(
                    metadata.request_meta[request_id].load_block_ids[1]
                )
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
            except Exception as e:
                logger.error(f"submit dump task failed. {type(e).__name__}: {e}")
        if self.is_save:
            submit_end = time.perf_counter()
            ucmmetrics.update_stats(
                {"layerwise_save_submit_ms": (submit_end - submit_start) * 1000}
            )

    def wait_for_save(self) -> None:
        if not self.is_save:
            total_end = time.perf_counter()
            if self._layerwise_batch_start is not None:
                batch_total_ms = (total_end - self._layerwise_batch_start) * 1000
                ucmmetrics.update_stats({"layerwise_batch_total_ms": batch_total_ms})
                self._layerwise_batch_start = None
            return
        total_start = time.perf_counter()
        try:
            for layer_name in self.kv_caches:
                if layer_name not in self.dump_tasks:
                    continue
                self.store.wait(self.dump_tasks[layer_name])
        except Exception as e:
            logger.error(f"wait for dump kv cache failed. {type(e).__name__}: {e}")
        total_end = time.perf_counter()
        stats = {"layerwise_save_tail_total_ms": (total_end - total_start) * 1000}
        if self._layerwise_batch_start is not None:
            stats["layerwise_batch_total_ms"] = (
                total_end - self._layerwise_batch_start
            ) * 1000
            self._layerwise_batch_start = None
        ucmmetrics.update_stats(stats)
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


class UCMLiteConnector(UCMDirectConnector):
    def __init__(
        self,
        vllm_config,
        role,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        ucm_config = Config(vllm_config.kv_transfer_config)
        launch_config = ucm_config.get_config()
        enable_record_traces = launch_config.get("enable_record_traces", False)
        persist_token_threshold = launch_config.get("persist_token_threshold", 0)
        vllm_config.kv_transfer_config.kv_connector_extra_config = {
            "ucm_connectors": [
                {
                    "ucm_connector_name": "UcmPipelineStore",
                    "ucm_connector_config": {
                        "store_pipeline": "Fake",
                        "share_buffer_enable": True,
                        "buffer_number": 244032232,
                    },
                }
            ],
            "enable_record_traces": enable_record_traces,
            "persist_token_threshold": persist_token_threshold,
            "use_lite": True,
        }
        super().__init__(vllm_config, role, kv_cache_config)
        self.total_block_nums = 0
        self.total_hit_block_nums = 0
        logger.info("Init UCMLiteConnector.")

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        super().get_num_new_matched_tokens(request, num_computed_tokens)

        external_hit_blocks = 0
        req_blocks_num = len(request.all_token_ids) // self.hash_block_size
        if req_blocks_num < 1:
            return 0, False
        self.total_block_nums += req_blocks_num
        if request.request_id in self.requests_meta:
            request_meta = self.requests_meta[request.request_id]
            external_hit_blocks = (
                request_meta.total_hit_block_num - request_meta.hbm_hit_block_num
            )
            need_dump_blks = request_meta.ucm_block_ids[
                request_meta.total_hit_block_num :
            ]
            shard_indexs = [0] * len(need_dump_blks)
            total_ptrs = [[0]] * len(need_dump_blks)
            try:
                task = self.store.dump_data(need_dump_blks, shard_indexs, total_ptrs)
                self.store.wait(task)
            except Exception as e:
                logger.error(
                    f"request {request.request_id} wait dump task error. {type(e).__name__}: {e}"
                )
            self.requests_meta[request.request_id] = RequestMeta()

        self.total_hit_block_nums += external_hit_blocks

        logger.info(
            f"req external hit rate: {(external_hit_blocks / req_blocks_num):.2f}, "
            f"total external hit rate: {(self.total_hit_block_nums / self.total_block_nums):.2f}"
        )
        return 0, False


def layer_name_to_kv_cache_spec(
    kv_cache_config: KVCacheConfig,
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

    if current_platform.device_type == "npu" and hasattr(spec, "compress_ratio"):
        block_size *= spec.compress_ratio

    return block_size


def is_mamba_align_kv_cache_spec(spec: KVCacheSpec) -> bool:
    if isinstance(spec, UniformTypeKVCacheSpecs):
        sample = next(iter(spec.kv_cache_specs.values()))
        return is_mamba_align_kv_cache_spec(sample)
    return isinstance(spec, MambaSpec) and spec.mamba_cache_mode == "align"


def sliding_window_from_kv_cache_spec(spec: KVCacheSpec) -> Optional[int]:
    """Return the sliding window size of a group spec, or None for full attention.

    A group is treated as full attention iff its sample spec exposes no
    ``sliding_window`` attribute or that attribute is ``None``.
    """
    if is_mamba_align_kv_cache_spec(spec):
        return block_size_from_kv_cache_spec(spec)
    if isinstance(spec, UniformTypeKVCacheSpecs):
        sample = next(iter(spec.kv_cache_specs.values()))
        return getattr(sample, "sliding_window", None)
    return getattr(spec, "sliding_window", None)


@dataclass
class GroupInfo:
    """Per-group metadata used by :class:`KVCacheGroupManager`."""

    group_id: int
    block_size: int
    # None for full-attention groups, otherwise the window length in tokens.
    sliding_window: Optional[int]
    layer_names: tuple[str, ...]
    # Independent hash chain seed per group (see ``KVCacheGroupManager``).
    seed: bytes
    is_mamba_align: bool = False

    @property
    def is_full_attention(self) -> bool:
        return self.sliding_window is None


class KVCacheGroupManager:
    """Group-aware hashing and lookup for hybrid (HMA) connectors.

    Splits ``kv_cache_config.kv_cache_groups`` into full-attention groups
    (one or more) and sliding-window groups, derives a per-group hash chain
    seed, and exposes a two-stage lookup that:

    1. For every full-attention group, hashes ``request.all_token_ids`` with
       that group's block size and runs ``store.lookup_on_prefix`` on the
       blocks beyond its own ``hbm_hit_block_num``. The candidate hits (in
       tokens) are min'd across full-attn groups and rounded down to
       ``lcm_block_size``.
    2. For each sliding-window group, re-hashes the same prefix with that
       group's own block size and verifies the last
       ``max(1, sliding_window // block_size)`` blocks all exist via
       ``store.lookup`` (when ``block_size > sliding_window`` — e.g. on
       Ascend — the last single block already covers the SW). If any
       sliding-window group fails this check, the whole external hit is
       downgraded to zero.
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
        # All groups whose spec has no sliding_window. Order follows group_id.
        self.full_attn_groups: list[GroupInfo] = []
        self.sliding_window_groups: list[GroupInfo] = []

        for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
            spec = group.kv_cache_spec
            block_size = block_size_from_kv_cache_spec(spec)
            sliding_window = sliding_window_from_kv_cache_spec(spec)
            # Mix group_id into the hash chain seed so two groups with the
            # same block_size do not collide in the underlying store.
            seed = request_hasher((b"UCM_GROUP_SEED", base_seed, group_id))
            info = GroupInfo(
                group_id=group_id,
                block_size=block_size,
                sliding_window=sliding_window,
                layer_names=tuple(group.layer_names),
                seed=seed,
                is_mamba_align=is_mamba_align_kv_cache_spec(spec),
            )
            self.groups_by_id.append(info)
            if info.is_full_attention:
                self.full_attn_groups.append(info)
            else:
                self.sliding_window_groups.append(info)

        assert len(self.full_attn_groups) >= 1, (
            "UCMHMAConnector expects at least one full-attention group in "
            "kv_cache_config.kv_cache_groups."
        )

        # Resume points must be aligned to the LCM of every group's
        # block_size so that per-group block accounting (including each
        # full-attn group's lookup result and every SW group's tail slice)
        # lands on a clean block boundary.
        all_block_sizes = [g.block_size for g in self.groups_by_id]
        self.lcm_block_size: int = math.lcm(*all_block_sizes)

        for g in self.groups_by_id:
            assert self.lcm_block_size % g.block_size == 0, (
                f"group {g.group_id} block_size={g.block_size} does not "
                f"divide LCM={self.lcm_block_size}"
            )
        for sw in self.sliding_window_groups:
            # The dump path stores only ``[B - sliding_window, B)`` for each
            # LCM boundary B. Requiring ``sliding_window <= lcm_block_size``
            # guarantees consecutive boundaries' tails do not overlap, so the
            # incremental dump can append each boundary's tail without
            # cross-boundary deduplication.
            assert sw.sliding_window <= self.lcm_block_size, (
                f"sliding window group {sw.group_id} sliding_window="
                f"{sw.sliding_window} > lcm_block_size="
                f"{self.lcm_block_size}; not supported."
            )
            # On some backends (e.g. Ascend) ``block_size`` can exceed
            # ``sliding_window``; in that case a single block already holds
            # more than a window of tokens, so we will treat the last block
            # as the SW tail and skip the divisibility check.
            if sw.block_size >= sw.sliding_window:
                continue
            if sw.sliding_window % sw.block_size != 0:
                raise ValueError(
                    f"Sliding window group {sw.group_id} sliding_window="
                    f"{sw.sliding_window} is not a multiple of block_size="
                    f"{sw.block_size}."
                )

        logger.info(
            "KVCacheGroupManager initialized: "
            f"lcm_block_size={self.lcm_block_size}, "
            f"full_attn_groups="
            f"{[(g.group_id, g.block_size) for g in self.full_attn_groups]}, "
            f"sliding_window_groups="
            f"{[(g.group_id, g.block_size, g.sliding_window, g.is_mamba_align) for g in self.sliding_window_groups]}"
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
    ) -> tuple[int, int]:
        """Two-stage HMA lookup using precomputed per-group hashes.

        ``group_block_ids`` must have one entry per group, indexed by the
        original ``group_id`` (see :meth:`compute_all_group_block_ids`).

        Stage 1 — every full-attention group runs ``lookup_on_prefix``
        beyond its own ``hbm_hit_block_num``; the candidate hits are taken
        as a min and rounded down to ``lcm_block_size`` so the final
        external hit is consistent across all full-attn groups and aligns
        to the kv-cache page granularity expected by the scheduler.

        Stage 2 — every sliding-window group must have the last
        ``sliding_window // block_size`` blocks before ``total_hit_tokens``
        present in the store; if any group fails, the whole external hit
        is downgraded to zero.

        Returns:
            Tuple of
            - ``external_hit_tokens``: tokens hit beyond ``num_computed_tokens``,
              aligned to ``lcm_block_size``. ``0`` if any check fails.
            - ``external_hit_lcm_blocks``: ``external_hit_tokens //
              lcm_block_size`` (also ``0`` on downgrade).
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
            return 0, 0

        # Stage 2: every SW group's tail window must be in the store.
        total_hit_tokens = num_computed_tokens + external_hit_tokens
        for sw in self.sliding_window_groups:
            if sw.is_mamba_align:
                mamba_state_hash = self.compute_mamba_align_state_hash(
                    sw, total_hit_tokens, group_block_ids
                )
                if mamba_state_hash is None:
                    logger.info(
                        f"mamba-align group {sw.group_id} state hash missing "
                        f"at total_hit_tokens={total_hit_tokens}, "
                        "downgrade external hit to 0."
                    )
                    return 0, 0
                try:
                    results = store.lookup([mamba_state_hash])
                except Exception as e:
                    logger.error(
                        f"mamba-align group {sw.group_id} lookup error. "
                        f"{type(e).__name__}: {e}"
                    )
                    return 0, 0
                if not all(results):
                    logger.info(
                        f"mamba-align group {sw.group_id} state miss: "
                        f"hits={results}, downgrade external hit to 0."
                    )
                    return 0, 0
                continue

            # When ``block_size > sliding_window`` (e.g. on Ascend) a single
            # block already covers more than a window of tokens, so we use
            # the last block as the SW tail (tail_count = 1). Otherwise
            # ``sliding_window`` is a multiple of ``block_size`` (validated
            # in __init__) and we take exactly ``sliding_window/block_size``
            # blocks.
            tail_count = max(1, sw.sliding_window // sw.block_size)
            min_required_tokens = tail_count * sw.block_size
            if total_hit_tokens < min_required_tokens:
                logger.info(
                    f"sliding window group {sw.group_id} tail check skipped: "
                    f"total_hit_tokens={total_hit_tokens} < "
                    f"min_required={min_required_tokens} "
                    f"(sliding_window={sw.sliding_window}, "
                    f"block_size={sw.block_size}), downgrade to 0."
                )
                return 0, 0

            sw_block_ids = group_block_ids[sw.group_id]
            # ``sw.block_size`` divides ``total_hit_tokens`` because
            # ``total_hit_tokens`` is a multiple of ``lcm_block_size`` and
            # ``lcm_block_size`` is divisible by every group's block_size
            # (validated in __init__).
            num_blocks_in_total_hit = total_hit_tokens // sw.block_size
            tail_block_ids = sw_block_ids[
                num_blocks_in_total_hit - tail_count : num_blocks_in_total_hit
            ]
            try:
                results = store.lookup(tail_block_ids)
            except Exception as e:
                logger.error(
                    f"sliding window group {sw.group_id} lookup error. "
                    f"{type(e).__name__}: {e}"
                )
                return 0, 0
            if not all(results):
                logger.info(
                    f"sliding window group {sw.group_id} tail miss: "
                    f"hits={results}, downgrade external hit to 0."
                )
                return 0, 0

        return external_hit_tokens, external_hit_tokens // self.lcm_block_size


class HMAKVCacheLayout(KVCacheLayout):
    def __init__(
        self,
        kvcaches,
        ucm_config: dict,
        vllm_config: "VllmConfig",
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(kvcaches, ucm_config, vllm_config, kv_cache_config)

    def _build_layout(self, kvcaches):
        base_ptrs = []
        buffer_size_rows = []
        tensor_size_lists = []
        block_stride_lists = []

        for raw_tensor in self.kv_cache_config.kv_cache_tensors:
            ptrs = []
            buffer_sizes = []
            tensor_sizes = []
            block_strides = []

            if raw_tensor.shared_by:
                sample_layer_name = raw_tensor.shared_by[0]
                kv_layer = kvcaches.get(sample_layer_name)
                if kv_layer is None:
                    logger.warning(
                        f"kv_layer {sample_layer_name} not found in kvcaches"
                    )
                    continue
                kv_cache_spec = self.layer_name_to_kv_cache_spec[sample_layer_name][0]
                if isinstance(kv_layer, torch.Tensor):
                    ptrs.append(kv_layer.data_ptr())
                    buffer_sizes.append(raw_tensor.size)
                    tensor_sizes.append(kv_cache_spec.page_size_bytes)
                    block_strides.append(kv_cache_spec.page_size_bytes)
                elif isinstance(kv_layer, (tuple, list)):
                    ptrs.append(kv_layer[0].data_ptr())
                    buffer_sizes.append(raw_tensor.size)
                    tensor_sizes.append(kv_cache_spec.page_size_bytes)
                    block_strides.append(kv_cache_spec.page_size_bytes)
                else:
                    logger.warning(f"unsupported kv_layer type: {type(kv_layer)}")

            if not ptrs and not tensor_sizes:
                continue

            base_ptrs.append(ptrs)
            buffer_size_rows.append(buffer_sizes)
            tensor_size_lists.append(tensor_sizes)
            block_stride_lists.append(block_strides)

        self.base_ptrs = np.asarray(base_ptrs, dtype=np.uint64)
        self.buffer_sizes = np.asarray(buffer_size_rows, dtype=np.uint64)
        self.tensor_size_lists = np.asarray(tensor_size_lists, dtype=np.uint64)
        self.block_stride_lists = np.asarray(block_stride_lists, dtype=np.uint64)

        logger.info(
            f"base_ptrs: {self.base_ptrs.shape}, tensor_size_lists: {self.tensor_size_lists.shape}"
        )


class AscendDSV4Layout(HMAKVCacheLayout):
    def __init__(
        self,
        kvcaches,
        ucm_config: dict,
        vllm_config: "VllmConfig",
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(kvcaches, ucm_config, vllm_config, kv_cache_config)
        self.indexer_scale_size_bytes = 0
        for _, layer_specs in self.layer_name_to_kv_cache_spec.items():
            for spec in layer_specs:
                if hasattr(spec, "indexer_scale_size_bytes"):
                    self.indexer_scale_size_bytes = spec.indexer_scale_size_bytes
                    break

    def _build_layout(self, kvcaches):
        self.indexer_scale_size_bytes = 0
        for _, layer_specs in self.layer_name_to_kv_cache_spec.items():
            for spec in layer_specs:
                if hasattr(spec, "indexer_scale_size_bytes"):
                    self.indexer_scale_size_bytes = spec.indexer_scale_size_bytes
                    break

        base_ptrs = []
        buffer_size_rows = []
        tensor_size_lists = []
        block_stride_lists = []

        for raw_tensor in self.kv_cache_config.kv_cache_tensors:
            ptrs = []
            buffer_sizes = []
            tensor_sizes = []
            block_strides = []
            kv_size = raw_tensor.size - self.indexer_scale_size_bytes * self.num_blocks

            if raw_tensor.shared_by:
                sample_layer_name = raw_tensor.shared_by[0]
                kv_layer = kvcaches.get(sample_layer_name)
                if kv_layer is None:
                    logger.warning(
                        f"kv_layer {sample_layer_name} not found in kvcaches"
                    )
                    continue
                kv_cache_specs = self.layer_name_to_kv_cache_spec[sample_layer_name]
                if isinstance(kv_layer, (tuple, list)):
                    ptrs.append(kv_layer[0].data_ptr())
                    buffer_sizes.append(kv_size)
                    tensor_sizes.append(
                        kv_cache_specs[0].page_size_bytes
                        - self.indexer_scale_size_bytes
                    )
                    block_strides.append(
                        kv_cache_specs[0].page_size_bytes
                        - self.indexer_scale_size_bytes
                    )
                    ptrs.append(kv_layer[0].data_ptr() + kv_size)
                    buffer_sizes.append(self.indexer_scale_size_bytes * self.num_blocks)
                    tensor_sizes.append(self.indexer_scale_size_bytes)
                    block_strides.append(self.indexer_scale_size_bytes)
                else:
                    logger.warning(f"unsupported kv_layer type: {type(kv_layer)}")

            if not ptrs and not tensor_sizes:
                continue

            base_ptrs.append(ptrs)
            buffer_size_rows.append(buffer_sizes)
            tensor_size_lists.append(tensor_sizes)
            block_stride_lists.append(block_strides)

        self.base_ptrs = np.asarray(base_ptrs, dtype=np.uint64)
        self.buffer_sizes = np.asarray(buffer_size_rows, dtype=np.uint64)
        self.tensor_size_lists = np.asarray(tensor_size_lists, dtype=np.uint64)
        self.block_stride_lists = np.asarray(block_stride_lists, dtype=np.uint64)

        logger.info(
            f"base_ptrs: {self.base_ptrs.shape}, tensor_size_lists: {self.tensor_size_lists.shape}"
        )


def _dtype_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _mamba_component_sizes(spec: MambaSpec) -> list[int]:
    return [
        math.prod(shape) * _dtype_size(dtype)
        for shape, dtype in zip(spec.shapes, spec.dtypes)
    ]


def _attention_component_sizes(spec: KVCacheSpec) -> tuple[int, int]:
    assert isinstance(spec, FullAttentionSpec)
    k_size = (
        spec.block_size * spec.num_kv_heads * spec.head_size * _dtype_size(spec.dtype)
    )
    head_size_v = getattr(spec, "head_size_v", spec.head_size)
    v_size = spec.block_size * spec.num_kv_heads * head_size_v * _dtype_size(spec.dtype)
    return k_size, v_size


class HybridLinearAttentionLayout(HMAKVCacheLayout):
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

    def _collect_shared_tensor_info(
        self,
        raw_tensor,
        kvcaches,
    ) -> tuple[list[KVCacheSpec], list[int]]:
        shared_specs: list[KVCacheSpec] = []
        shared_ptrs: list[int] = []
        for layer_name in raw_tensor.shared_by:
            kv_layer = kvcaches.get(layer_name)
            if kv_layer is None:
                continue
            shared_specs.extend(self.layer_name_to_kv_cache_spec[layer_name])
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
        mamba_sizes = _mamba_component_sizes(mamba_specs[0])
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
        k_size, v_size = _attention_component_sizes(attn_specs[0])
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

    def _build_layout(self, kvcaches):
        base_ptrs = []
        buffer_size_rows = []
        tensor_size_lists = []
        block_stride_lists = []

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

            mamba_specs = [s for s in shared_specs if isinstance(s, MambaSpec)]
            attn_specs = [s for s in shared_specs if isinstance(s, FullAttentionSpec)]
            if not mamba_specs or not attn_specs:
                self._append_contiguous_page_layout(
                    raw_tensor,
                    shared_ptrs,
                    base_ptrs,
                    buffer_size_rows,
                    tensor_size_lists,
                    block_stride_lists,
                )
                continue

            if current_platform.device_type == "npu":
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
            else:
                self._append_contiguous_page_layout(
                    raw_tensor,
                    shared_ptrs,
                    base_ptrs,
                    buffer_size_rows,
                    tensor_size_lists,
                    block_stride_lists,
                )

        self.base_ptrs = np.asarray(base_ptrs, dtype=np.uint64)
        self.buffer_sizes = np.asarray(buffer_size_rows, dtype=np.uint64)
        self.tensor_size_lists = np.asarray(tensor_size_lists, dtype=np.uint64)
        self.block_stride_lists = np.asarray(block_stride_lists, dtype=np.uint64)


class UCMHMAConnector(UCMDirectConnector, SupportsHMA):
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
            f"UCMHMAConnector initialized with use_layerwise={self.use_layerwise}"
        )

    def _create_kv_cache_layout(
        self, kv_caches: dict[str, torch.Tensor]
    ) -> KVCacheLayout:
        if current_platform.device_type == "npu" and self.use_compress:
            return AscendDSV4Layout(
                kv_caches,
                self.launch_config,
                self._vllm_config,
                self._kv_cache_config,
            )
        return HMAKVCacheLayout(
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
        # LCM-block units in HMA mode; per-group block ids/counts are derived
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

        external_hit_tokens, external_hit_lcm_blocks = (
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

        self.requests_meta[request.request_id] = HMARequestMeta(
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
        assert isinstance(req_meta, HMARequestMeta)
        block_ids = blocks.get_block_ids()
        if self.group_manager is not None:
            assert len(block_ids) == self.group_manager.num_groups, (
                f"allocated block group count {len(block_ids)} does not match "
                f"HMA group count {self.group_manager.num_groups}"
            )
        req_meta.group_vllm_block_ids = [list(group) for group in block_ids]

    def _generate_hma_dispatch_meta(
        self,
        req_meta: "HMARequestMeta",
        new_tokens: int,
        new_vllm_block_ids_per_group: tuple[list[int], ...],
        need_load: bool = True,
        request_id: str = "",
        incoming_block_ids_are_full: bool = False,
    ) -> RequestDispatchMeta:
        """Build a flat (ucm, vllm) block id pair list across all groups.

        The output ``RequestDispatchMeta`` keeps the same shape as the
        non-HMA path (``tuple[list[bytes], list[int]]``) so that
        ``start_load_kv`` / ``wait_for_save`` and the underlying store APIs
        do not need to know about groups. Per-group slices are concatenated
        in ascending ``group_id`` order, with ``ucm_block_ids[k]`` and
        ``vllm_block_ids[k]`` always referring to the same block.

        Layout per group within ``[token_processed, token_processed + new_tokens)``:
        - **load** (only when ``external_hit_blocks > 0`` and ``need_load``):
          - full-attn group: tokens ``[hbm_hit_tokens, total_hit_tokens)``
          - sliding-window group: the blocks covering tokens
            ``[total_hit_tokens - sliding_window, total_hit_tokens)``;
            ``start_blk`` is floor-rounded so when ``block_size >
            sliding_window`` (e.g. on Ascend) we naturally load the single
            last block. The SW window is reloaded every resume because
            older blocks are evicted by the SW manager.
        - **dump** of ``[token_processed, token_processed + new_tokens)``:
          - full-attn group: every newly-completed full block (the
            ``lookup_on_prefix`` chain needs every prefix block to be
            present).
          - sliding-window group: only the last
            ``max(1, sliding_window/block_size)`` blocks before each LCM
            boundary reached in this range. Lookup always resumes at LCM
            boundaries and stage-2 SW check only inspects those tails, so
            blocks between tails would be dead weight in the store.
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

        def extend_non_null(
            dst_ucm_block_ids: list[bytes],
            dst_vllm_block_ids: list[int],
            src_ucm_block_ids: list[bytes],
            src_vllm_block_ids: list[int],
        ) -> None:
            # Mamba align mode pads req block tables with vLLM's null block
            # (block_id=0). These are metadata placeholders, not physical pages
            # that should be loaded from or dumped to the store.
            for ucm_block_id, vllm_block_id in zip(
                src_ucm_block_ids, src_vllm_block_ids
            ):
                if vllm_block_id == 0:
                    continue
                dst_ucm_block_ids.append(ucm_block_id)
                dst_vllm_block_ids.append(vllm_block_id)

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
                    "HMA mamba-align state vLLM block missing: "
                    f"request_id={request_id}, group_id={gid}, reason={reason}, "
                    f"seq_len={seq_len}, state_idx={state_idx}, "
                    f"vllm_state_idx={vllm_state_idx}, "
                    f"num_vllm_blocks={len(req_meta.group_vllm_block_ids[gid])}"
                )
                return
            if vllm_block_id == 0:
                return
            if group.is_mamba_align:
                ucm_block_id = self.group_manager.compute_mamba_align_state_hash(
                    group, seq_len, req_meta.group_ucm_block_ids
                )
            else:
                try:
                    ucm_block_id = req_meta.group_ucm_block_ids[gid][state_idx]
                except IndexError:
                    logger.error(
                        "HMA state block UCM hash missing: "
                        f"request_id={request_id}, group_id={gid}, "
                        f"reason={reason}, seq_len={seq_len}, "
                        f"state_idx={state_idx}, "
                        f"num_ucm_blocks={len(req_meta.group_ucm_block_ids[gid])}"
                    )
                    return
            if ucm_block_id is None:
                logger.error(
                    "HMA mamba-align state hash missing: "
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
                if group.is_full_attention:
                    load_tok_start = hbm_hit_tokens
                else:
                    load_tok_start = total_hit_tokens - group.sliding_window
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
            # SW groups only need the tail at these boundaries because lookup
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
                    # Dump only the tail blocks at each LCM boundary reached
                    # in this range. Since ``sliding_window <=
                    # lcm_block_size`` (validated in ``KVCacheGroupManager``),
                    # consecutive boundaries' tails do not overlap and we can
                    # extend the lists without dedup.
                    if first_lcm_b > last_lcm_b:
                        continue
                    if group.is_mamba_align:
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
                        continue
                    tail_count = max(1, group.sliding_window // group.block_size)
                    b = first_lcm_b
                    while b <= last_lcm_b:
                        end_blk = b // group.block_size
                        start_blk = max(0, end_blk - tail_count)
                        if start_blk < end_blk:
                            extend_non_null(
                                dump_ucm_block_ids,
                                dump_vllm_block_ids,
                                req_meta.group_ucm_block_ids[gid][start_blk:end_blk],
                                req_meta.group_vllm_block_ids[gid][start_blk:end_blk],
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
            assert isinstance(req_meta, HMARequestMeta)
            requests_dispatch_meta[request_id] = self._generate_hma_dispatch_meta(
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
                assert isinstance(req_meta, HMARequestMeta)
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
                requests_dispatch_meta[request_id] = self._generate_hma_dispatch_meta(
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
                assert isinstance(req_meta, HMARequestMeta)
                requests_dispatch_meta[request_id] = self._generate_hma_dispatch_meta(
                    req_meta,
                    scheduler_output.num_scheduled_tokens[request_id],
                    request.new_block_ids,
                    request.resumed_from_preemption,
                    request_id=request_id,
                    incoming_block_ids_are_full=request.resumed_from_preemption,
                )

        for request_id in scheduler_output.finished_req_ids:
            self.requests_meta.pop(request_id, None)

        return UCMConnectorMetadata(requests_dispatch_meta)

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        return False, None


def use_hybrid_linear_attention_layout(
    kv_cache_config: Optional["KVCacheConfig"],
) -> bool:
    # Scheduler-side connector construction may happen before vLLM has built
    # the concrete KV cache config. In that phase there is no physical layout
    # to inspect yet, so this connector specialization cannot be selected.
    if kv_cache_config is None:
        return False

    if current_platform.device_type != "npu" and not current_platform.is_cuda_alike():
        return False

    layer_to_specs = layer_name_to_kv_cache_spec(kv_cache_config)
    for raw_tensor in kv_cache_config.kv_cache_tensors:
        shared_specs = [
            spec
            for layer_name in raw_tensor.shared_by
            for spec in layer_to_specs.get(layer_name, [])
        ]
        tensor_has_full_attention = any(
            isinstance(spec, FullAttentionSpec) for spec in shared_specs
        )
        tensor_has_mamba_align = any(
            isinstance(spec, MambaSpec) and spec.mamba_cache_mode == "align"
            for spec in shared_specs
        )
        if tensor_has_full_attention and tensor_has_mamba_align:
            return True

    return False


class UCMHybridLinearAttentionConnector(UCMHMAConnector):
    """Connector for full-attention + linear-attention hybrid layouts."""

    def _create_kv_cache_layout(
        self, kv_caches: dict[str, torch.Tensor]
    ) -> KVCacheLayout:
        return HybridLinearAttentionLayout(
            kv_caches,
            self.launch_config,
            self._vllm_config,
            self._kv_cache_config,
        )


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

        use_lite = (
            self.launch_config.get("use_lite", False)
            if self.launch_config is not None
            else False
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

        use_hma = (
            self._vllm_config.scheduler_config.disable_hybrid_kv_cache_manager is False
            or os.getenv("USE_MULTI_GROUPS_KV_CACHE") == "1"
        )

        use_hybrid_linear_attention = use_hybrid_linear_attention_layout(
            kv_cache_config
        )

        from ucm.integration.vllm.hma_connector import UCMFAWAConnector

        if UCMFAWAConnector.can_handle_kv_cache_config(kv_cache_config):
            self.connector = UCMFAWAConnector(vllm_config, role, kv_cache_config)
        elif use_lite:
            self.connector = UCMLiteConnector(vllm_config, role, kv_cache_config)
        elif use_ratio_rate:
            self.connector = UCMMockConnector(vllm_config, role, kv_cache_config)
        elif use_cp_parallel:
            self.connector = UCMCPConnector(vllm_config, role, kv_cache_config)
        elif use_layerwise:
            self.connector = UCMLayerWiseConnector(vllm_config, role, kv_cache_config)
        elif use_hybrid_linear_attention:
            self.connector = UCMHybridLinearAttentionConnector(
                vllm_config, role, kv_cache_config
            )
        # elif use_hma:
        #     self.connector = UCMHMAConnector(vllm_config, role, kv_cache_config)
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
