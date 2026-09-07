import copy
import hashlib
import math
import os
import pickle
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorWorkerMetadata,
    SupportsHMA,
)
from vllm.distributed.parallel_state import get_world_group
from vllm.model_executor.models.utils import extract_layer_index
from vllm.platforms import current_platform
from vllm.v1.core.sched.output import SchedulerOutput
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
        self, kvcaches, use_layerwise: bool, vllm_config: "VllmConfig"
    ) -> None:
        # each row is a layer, each column is a tensor_size/ptr in the layer (e.g., k, v, rope, k_index)
        self.base_ptrs: np.ndarray  # (n_layers, n_ptrs）
        self.tensor_size_lists: np.ndarray  # (n_layers, n_tensor_sizes)
        self.use_layerwise = use_layerwise
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
        self._build_layout(kvcaches)

    def _build_layout(self, kvcaches):

        num_rows = len(set(self.layer_name_to_id.values()))
        raw_ptr_rows = [[] for _ in range(num_rows)]
        stride_rows = [[] for _ in range(num_rows)]

        for layer_name, kv_layer in kvcaches.items():
            ptrs = []
            strides = []

            def handle_tensor(t: torch.Tensor, size_dims):
                ptrs.append(t[0].data_ptr())

                stride = math.prod([t.shape[i] for i in size_dims]) * t.element_size()
                strides.append(stride)

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

        self.base_ptrs = np.asarray(raw_ptr_rows, dtype=np.uint64)
        self.tensor_size_lists = np.asarray(stride_rows, dtype=np.uint64)

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
                self.tensor_size_lists[:, None, :] * vllm_block_ids_np[None, :, None]
                + self.base_ptrs[:, None, :]
            )
        return (
            vllm_block_ids_np[:, None, None] * self.tensor_size_lists[None, :, :]
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

        self.metrics_config = self.launch_config.get("metrics_config_path", "")
        if self.metrics_config:
            worker_id = (
                f"{self.engine_id}_{get_world_group().rank}"
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
        registrations: Optional[np.ndarray] = None,
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
        config.pop("gpu_kv_buffer_addrs", None)
        config.pop("gpu_kv_buffer_sizes", None)
        config.setdefault("share_buffer_enable", self.is_mla)
        if "storage_backends" in config:
            backends = [path for path in config["storage_backends"].split(":")]
            config["storage_backends"] = backends
        config["unique_id"] = f"{self.engine_id}"
        config["tensor_layout"] = "mla" if self.is_mla else "gqa"
        config["role"] = (
            "worker" if self._role == KVConnectorRole.WORKER else "scheduler"
        )
        if self._role == KVConnectorRole.WORKER:
            config["device_id"] = self.local_rank
            config["tensor_size_list"] = (
                kv_cache_layout.tensor_size_list * self.blocks_per_chunk
            )
            config["shard_size"] = kv_cache_layout.shard_size * self.blocks_per_chunk
            config["block_size"] = kv_cache_layout.block_size * self.blocks_per_chunk
            config["local_rank_size"] = self.tp_size if self.is_mla else 1
            if registrations is not None and len(registrations):
                config["gpu_kv_buffer_addrs"] = [
                    int(item["addr"]) for item in registrations
                ]
                config["gpu_kv_buffer_sizes"] = [
                    int(item["size"]) for item in registrations
                ]
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
            self.kv_caches, self.use_layerwise, self._vllm_config
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

        registrations = self._collect_kv_cache_registrations()
        self.store = self._create_store(
            self.kv_cache_layout, store_cores, registrations
        )

        if worker_cores:
            try:
                os.sched_setaffinity(0, worker_cores)
                logger.info(f"[VLLM CPU Affinity] Worker bound to cores {worker_cores}")
            except Exception as e:
                logger.warning(f"Failed to bind worker: {e}")

        if self.device is None:
            raise RuntimeError(f"Unsupported device platform for UCMDirectConnector.")

    def _collect_kv_cache_registrations(self):
        registrations: list[tuple[int, int]] = []
        unique_registrations: set[tuple[int, int]] = set()

        def add_tensor(tensor: torch.Tensor) -> None:
            addr = tensor.data_ptr()
            size = tensor.numel() * tensor.element_size()
            registration = (addr, size)
            if addr == 0 or size <= 0 or registration in unique_registrations:
                return
            registrations.append(registration)
            unique_registrations.add(registration)

        for kv_layer in self.kv_caches.values():
            if isinstance(kv_layer, torch.Tensor):
                if kv_layer.dim() == 5:
                    add_tensor(kv_layer[0])
                    add_tensor(kv_layer[1])
                elif kv_layer.dim() == 3:
                    add_tensor(kv_layer)
                else:
                    raise ValueError(
                        f"Unsupported kv cache tensor shape: {kv_layer.shape}"
                    )
            elif isinstance(kv_layer, Tuple):
                for tensor in kv_layer:
                    add_tensor(tensor)
            else:
                raise TypeError(f"Unsupported kv cache type: {type(kv_layer)}")

        return np.array(
            registrations,
            dtype=[("addr", np.uintp), ("size", np.uintp)],
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
        for request_id, request in metadata.request_meta.items():
            if len(request.load_block_ids[0]) == 0:
                continue
            is_load = True
            num_loaded_block += len(request.load_block_ids[0])
            num_loaded_request += 1

            ucm_block_ids, vllm_block_ids = request.load_block_ids
            if self.tp_rank != 0 and not self.is_mla:
                for i, ucm_block_id in enumerate(ucm_block_ids):
                    ucm_block_ids[i] = self.request_hasher(ucm_block_id)
            try:
                total_ptrs = self.kv_cache_layout.extract_block_addrs(vllm_block_ids)
                total_ptrs = total_ptrs.reshape(total_ptrs.shape[0], -1)
                shard_indexs = [0] * len(ucm_block_ids)
                task = self.store.load_data(ucm_block_ids, shard_indexs, total_ptrs)
                request_to_task[request_id] = task
            except Exception as e:
                logger.error(
                    f"request {request_id} submit load task error. {type(e).__name__}: {e}"
                )
                self._invalid_block_ids.update(
                    metadata.request_meta[request_id].load_block_ids[1]
                )
                num_loaded_block -= len(request.load_block_ids[0])

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
                num_loaded_block -= len(
                    metadata.request_meta[request_id].load_block_ids[0]
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

        dump_tasks: List[Task] = []
        is_save = False
        num_saved_block = 0
        num_saved_request = 0
        total_ucm_block_ids, total_vllm_block_ids = [], []
        for request_id, request in metadata.request_meta.items():
            if len(request.dump_block_ids[0]) == 0:
                continue
            is_save = True
            num_saved_block += len(request.dump_block_ids[0])
            num_saved_request += 1

            ucm_block_ids, vllm_block_ids = request.dump_block_ids
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
            except Exception as e:
                logger.error(
                    f"request {request_id} submit load task error. {type(e).__name__}: {e}"
                )
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
                self.store.wait(task)
            except Exception as e:
                logger.error(
                    f"request {request_id} wait {layer_name} load failed. {type(e).__name__}: {e}"
                )
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
        if not self._connector_metadata:
            return
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
            except Exception as e:
                logger.error(f"submit dump task failed. {type(e).__name__}: {e}")

    def wait_for_save(self) -> None:
        if not self.is_save:
            return
        try:
            for layer_name in self.kv_caches:
                if layer_name in self.dump_tasks:
                    self.store.wait(self.dump_tasks[layer_name])
        except Exception as e:
            logger.error(f"wait for dump kv cache failed. {type(e).__name__}: {e}")
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

        from ucm.integration.vllm.hma_connector import (
            UCMAscendFAWAConnector,
            UCMFAWAConnector,
        )

        use_fawa_store = UCMFAWAConnector.can_handle_kv_cache_config(
            kv_cache_config
        ) or UCMAscendFAWAConnector.can_handle_ascend_kv_cache_config(kv_cache_config)
        if use_fawa_store:
            connector_cls = (
                UCMAscendFAWAConnector
                if UCMAscendFAWAConnector.can_handle_ascend_kv_cache_config(
                    kv_cache_config
                )
                else UCMFAWAConnector
            )
            self.connector = connector_cls(vllm_config, role, kv_cache_config)
        elif use_lite:
            self.connector = UCMLiteConnector(vllm_config, role, kv_cache_config)
        elif use_ratio_rate:
            self.connector = UCMMockConnector(vllm_config, role, kv_cache_config)
        elif use_cp_parallel:
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

    def build_connector_worker_meta(self) -> KVConnectorWorkerMetadata | None:
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
