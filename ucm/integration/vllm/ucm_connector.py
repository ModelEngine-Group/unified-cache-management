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
import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.parallel_state import get_tp_group, get_world_group
from vllm.platforms import current_platform
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import SupportsHMA

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
    base_hash_ids: list[bytes] = field(default_factory=list)
    ucm_block_ids_by_group: tuple[list[bytes | None], ...] = ()
    vllm_block_ids_by_group: tuple[list[int], ...] = ()


@dataclass
class RequestDispatchMeta:
    load_block_ids: tuple[
        list[bytes], list[int]
    ]  # [0] mean ucm_block_ids, [1] means vllm_block_ids
    dump_block_ids: tuple[list[bytes], list[int]]


class KVCacheLayout:
    def __init__(
        self, kvcaches, use_layerwise: bool, vllm_config: "VllmConfig", group_size: int, tensor_sizes: int
    ) -> None:
        # each row is a layer, each column is a tensor_size/ptr in the layer (e.g., k, v, rope, k_index)
        self.base_ptrs: np.ndarray  # (n_layers, n_ptrs）
        self.tensor_size_lists: np.ndarray  # (n_layers, n_tensor_sizes)
        self.tensor_sizes = tensor_sizes
        self.group_size = group_size
        self.group_num = len(kvcaches) // self.group_size
        self.use_layerwise = use_layerwise
        self.vllm_config = vllm_config
        self.pp_size = self.vllm_config.parallel_config.pipeline_parallel_size
        self.num_hidden_layers = getattr(
            self.vllm_config.model_config.hf_text_config, "num_hidden_layers", 0
        )
        if self.pp_size > 1 and self.num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be > 0 when pp_size > 1")
        self._build_layout(kvcaches)

    def _build_layout(self, kvcaches):
        raw_ptr_rows = []
        stride_rows = []

        for idx, (_, kv_layer) in enumerate(kvcaches.items()):
            if idx >= self.group_size:
                break
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
            elif isinstance(kv_layer, list):
                ptrs = [kv_layer[0].data_ptr()]
                strides = [self.tensor_sizes]
            else:
                raise TypeError(f"Unsupported kv cache type: {type(kv_layer)}")

            raw_ptr_rows.append(ptrs)
            stride_rows.append(strides)

        self.base_ptrs = np.asarray(raw_ptr_rows, dtype=np.uint64)
        self.tensor_size_lists = np.asarray(stride_rows, dtype=np.uint64)

        logger.info(
            f"base_ptrs: {self.base_ptrs.shape}, tensor_size_lists: {self.tensor_size_lists.shape}"
        )

    def extract_block_addrs(self, vllm_block_ids: List[int], is_load: bool) -> np.ndarray:
        vllm_block_ids_np = np.array(vllm_block_ids, np.uint64)
        if is_load and self.group_num > 1:
            vllm_block_ids_np[:self.group_num-1] += np.uint64(len(vllm_block_ids))
        block_addrs = (
            vllm_block_ids_np[:, None, None] * self.tensor_size_lists[None, :, :]
            + self.base_ptrs[None, :, :]
        )  # (num_blocks, n_layers, n_ptrs)
        return block_addrs

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
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
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
        self.rope_store: Optional[UcmKVStoreBaseV1] = None

        # save block info, avoid hash request twice, and track them until request finished
        self.requests_meta: dict[str, RequestMeta] = {}

        ucm_config = Config(vllm_config.kv_transfer_config)
        self.engine_id = vllm_config.kv_transfer_config.engine_id
        self.launch_config = ucm_config.get_config()
        logger.info(f"self.launch_config: {self.launch_config}")
        self.connector_configs = self.launch_config.get("ucm_connectors", [])
        self.enable_event_sync = self.launch_config.get("enable_event_sync", True)
        assert len(self.connector_configs) > 0, "no storage connector name in config."

        self.chunk_size = self.block_size
        self.blocks_per_chunk = self.chunk_size // self.block_size

        if role == KVConnectorRole.SCHEDULER:
            self.request_hasher = RequestHasher(vllm_config, 0)
            self._seed = self.request_hasher("UCM_HASH_SEED")
            # init scheduler-size connector
            self.store = self._create_store(None)
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

        # invlalid block ids due to load errors
        self._invalid_block_ids: set[int] = set()

        # Hybrid-model helpers (group-aware).
        self.group_is_mamba: dict[int, bool] = {}
        self.layer_name_to_id: dict[str, int] = {}
        for gid, g in enumerate(kv_cache_config.kv_cache_groups):
            layer_names = list(getattr(g, "layer_names", []))
            for n in layer_names:
                self.layer_name_to_id[n] = int(n.split(".")[2])
            spec = getattr(g, "kv_cache_spec", None)
            self.group_is_mamba[gid] = (type(spec).__name__ == "MambaSpec" if spec is not None else False)
        self.group_size = len(kv_cache_config.kv_cache_tensors)
        self.group_num = len(self.group_is_mamba)            
        self.tensor_sizes = kv_cache_config.kv_cache_tensors[0].size // kv_cache_config.num_blocks

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

    def generate_hash_by_group(
        self, vllm_block_ids_by_group, base_hash_ids: list[bytes]
    ) -> tuple[list[bytes | None], ...]:
        ret: list[list[bytes | None]] = []
        for gid, vllm_list in enumerate(vllm_block_ids_by_group):
            group_hash_list = []
            suffix = gid.to_bytes(4, "big", signed=False)

            for i, bid in enumerate(vllm_list):
                if bid == 0 or i >= len(base_hash_ids):
                    group_hash_list.append(None)
                else:
                    group_hash_list.append(base_hash_ids[i] + suffix)
            ret.append(group_hash_list)
        return tuple(ret)

    def _rank_hash20(self, ucm_block_id: bytes) -> bytes:
        h = self.request_hasher(ucm_block_id)
        suffix = (
            ucm_block_id[16:20]
            if len(ucm_block_id) >= 20
            else (0).to_bytes(4, "big", signed=False)
        )
        return h + suffix

    def flatten_block_ids_by_group(
        self,
        block_ids_by_group: dict[int, list[Any]] | list[list[Any]] | tuple[list[Any], ...],
    ) -> list[Any]:
        flattened = []

        if isinstance(block_ids_by_group, dict):
            for gid in sorted(block_ids_by_group.keys()):
                for val in block_ids_by_group[gid]:
                    if val is None or val == 0:
                        continue
                    flattened.append(val)
        else:
            for _gid, row in enumerate(block_ids_by_group):
                for val in row:
                    if val is None or val == 0:
                        continue
                    flattened.append(val)

        return flattened

    def _create_store(
        self, kv_cache_layout: Optional[KVCacheLayout]
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
            self.kv_caches, self.use_layerwise, self._vllm_config, self.use_layerwise, self.group_size, self.tensor_sizes
        )
        self.block_data_size = self.kv_cache_layout.block_size
        self.first_layer_id = next(iter(self.layer_name_to_id.values()))
        self.store = self._create_store(self.kv_cache_layout)
        self.device = create_device()
        if self.device is None:
            raise RuntimeError(f"Unsupported device platform for UCMDirectConnector.")

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        assert num_computed_tokens % self.block_size == 0
        hbm_hit_block_num = num_computed_tokens // self.block_size

        base_hash_ids = self.generate_hash(
            self.block_size, request.all_token_ids, self._seed
        )
        group = self.group_num - 1
        suffix = group.to_bytes(4, "big", signed=False)
        ucm_block_ids = [h + suffix for h in base_hash_ids]

        external_block_ids = ucm_block_ids[hbm_hit_block_num:]
        if not external_block_ids:
            return 0, False
        try:
            external_hit_blocks = self.store.lookup_on_prefix(external_block_ids) + 1
        except RuntimeError as e:
            external_hit_blocks = 0
            logger.error(f"request {request.request_id} look up error. {e}")
        logger.info(
            f"request_id: {request.request_id}, "
            f"total_blocks_num: {len(ucm_block_ids)}, "
            f"hit hbm: {hbm_hit_block_num}, "
            f"hit external: {external_hit_blocks}"
        )
        if self.metrics_config:
            ucmmetrics.update_stats(
                {"interval_lookup_hit_rates": external_hit_blocks / len(base_hash_ids)},
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
            base_hash_ids=base_hash_ids,
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
        vllm_block_ids_by_group: Any,
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

        if len(req_meta.vllm_block_ids_by_group) == 0:
            req_meta.vllm_block_ids_by_group = tuple[list[int], ...]([[] for _ in range(self.group_num)])
        
        vllm_block_ids_by_group = list(vllm_block_ids_by_group)
        for target, source in zip(req_meta.vllm_block_ids_by_group, vllm_block_ids_by_group):
            target.extend(source)
        if len(req_meta.ucm_block_ids_by_group) == 0:
            req_meta.ucm_block_ids_by_group = self.generate_hash_by_group(
                req_meta.vllm_block_ids_by_group, req_meta.base_hash_ids
            )
        ucm_block_ids_by_group = req_meta.ucm_block_ids_by_group

        load_ucm_block_ids, load_vllm_block_ids = [], []
        dump_ucm_block_ids, dump_vllm_block_ids = [], []
        if need_load:
            load_ucm_block_ids = self.flatten_block_ids_by_group(
                [
                    req_meta.ucm_block_ids_by_group[gid][hbm_hit_block_num:total_hit_block_num]
                    for gid in range(self.group_num)
                ]
            )
            load_vllm_block_ids = self.flatten_block_ids_by_group(
                [row[hbm_hit_block_num:total_hit_block_num] for row in vllm_block_ids_by_group]
            )

        if req_meta.token_processed < req_meta.num_token_ids:
            start_idx = req_meta.token_processed // self.block_size
            end_idx = (req_meta.token_processed + new_tokens) // self.block_size
            dump_ucm_block_ids = self.flatten_block_ids_by_group(
                [
                    ucm_block_ids_by_group[gid][start_idx:end_idx]
                    for gid in range(self.group_num)
                ]
            )
            dump_vllm_block_ids = self.flatten_block_ids_by_group(
                [
                    req_meta.vllm_block_ids_by_group[gid][start_idx:end_idx]
                    for gid in range(self.group_num)
                ]
            )
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
            request_id, vllm_block_ids = request.req_id, request.block_ids
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
                        new_block_ids = scheduled_cached_reqs.new_block_ids[i]
                    if hasattr(scheduled_cached_reqs, "resumed_req_ids"):
                        resumed = request_id in scheduled_cached_reqs.resumed_req_ids
                    else:
                        resumed_list = getattr(
                            scheduled_cached_reqs, "resumed_from_preemption", None
                        )
                        resumed = bool(resumed_list[i]) if resumed_list is not None else False
                    requests_dispatch_meta[request_id] = self._generate_dispatch_meta(
                        req_meta,
                        scheduler_output.num_scheduled_tokens[request_id],
                        new_block_ids,
                        resumed,
                    )
        else:
            for request in scheduled_cached_reqs:
                request_id = request.req_id
                req_meta = self.requests_meta.get(request_id)
                if req_meta:
                    requests_dispatch_meta[request_id] = self._generate_dispatch_meta(
                        req_meta,
                        scheduler_output.num_scheduled_tokens[request_id],
                        request.new_block_ids,
                        request.resumed_from_preemption,
                    )

        # clear finished request
        for request_id in scheduler_output.finished_req_ids:
            self.requests_meta.pop(request_id, None)

        return UCMConnectorMetadata(requests_dispatch_meta)

    @staticmethod
    def _extract_layer_index(layer_name: str) -> Optional[int]:
        """
        Extract the layer index from the layer name.
        """
        for chunk in layer_name.split("."):
            if chunk.isdigit():
                return int(chunk)
        return None

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
                    ucm_block_ids[i] = self._rank_hash20(ucm_block_id)
            total_ptrs = self.kv_cache_layout.extract_block_addrs(vllm_block_ids, True)
            total_ptrs = total_ptrs.reshape(total_ptrs.shape[0], -1)
            shard_indexs = [0] * len(ucm_block_ids)
            try:
                task = self.store.load_data(ucm_block_ids, shard_indexs, total_ptrs)
                request_to_task[request_id] = task
            except RuntimeError as e:
                logger.error(f"request {request_id} submit load task error. {e}")
                self._invalid_block_ids.update(
                    metadata.request_meta[request_id].load_block_ids[1]
                )
                num_loaded_block -= len(request.load_block_ids[0])

        for request_id, task in request_to_task.items():
            try:
                self.store.wait(task)
            except RuntimeError as e:
                logger.error(f"request {request_id} wait load task error. {e}")
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
                    ucm_block_ids[i] = self._rank_hash20(ucm_block_id)
            total_ucm_block_ids.extend(ucm_block_ids)
            total_vllm_block_ids.extend(vllm_block_ids)

        if is_save:
            total_ptrs = self.kv_cache_layout.extract_block_addrs(total_vllm_block_ids, False)
            total_ptrs = total_ptrs.reshape(total_ptrs.shape[0], -1)
            shard_indexs = [0] * len(total_ucm_block_ids)
            try:
                event_handle = self._get_dump_event_handle()
                save_start_time = time.perf_counter() * 1000
                task = self.store.dump_data(
                    total_ucm_block_ids, shard_indexs, total_ptrs, event_handle
                )
                dump_tasks.append(task)
            except RuntimeError as e:
                logger.error(f"dump kv cache failed. {e}")
                return

            try:
                for task in dump_tasks:
                    self.store.wait(task)
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
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        self.load_tasks: dict[str, dict[int, Task]] = defaultdict(dict)
        self.dump_tasks: dict[str, Task] = {}
        self.use_layerwise = True
        self.is_save = False
        logger.info("Init UCMLayerWiseConnector.")

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        metadata = self._get_connector_metadata()
        self.load_tasks.clear()

        for request_id, request in metadata.request_meta.items():
            if len(request.load_block_ids[0]) == 0:
                continue

            ucm_block_ids, vllm_block_ids = request.load_block_ids
            if self.tp_rank % self.tp_size != 0 and not self.is_mla:
                for i, ucm_block_id in enumerate(ucm_block_ids):
                    ucm_block_ids[i] = self.request_hasher(ucm_block_id)
            try:
                total_ptrs = self.kv_cache_layout.extract_block_addrs(vllm_block_ids)

                for layer_name in self.kv_caches:
                    layer_id = self.layer_name_to_id[layer_name]
                    local_layer_id = layer_id - self.first_layer_id
                    shard_indexs = [layer_id] * len(ucm_block_ids)
                    layer_ptrs = np.ascontiguousarray(total_ptrs[:, local_layer_id, :])
                    task = self.store.load_data(ucm_block_ids, shard_indexs, layer_ptrs)
                    self.load_tasks[request_id][layer_name] = task
            except RuntimeError as e:
                logger.error(f"request {request_id} submit load task error. {e}")
                self._invalid_block_ids.update(
                    metadata.request_meta[request_id].load_block_ids[1]
                )

    def wait_for_layer_load(self, layer_name: str) -> None:
        metadata = self._get_connector_metadata()

        for request_id, tasks in self.load_tasks.items():
            try:
                if layer_name in tasks:
                    self.store.wait(tasks[layer_name])
            except RuntimeError as e:
                logger.error(f"request {request_id} wait load failed. {e}")
                self._invalid_block_ids.update(
                    metadata.request_meta[request_id].load_block_ids[1]
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
            total_ptrs = self.kv_cache_layout.extract_block_addrs(total_vllm_block_ids)
            shard_indexs = [layer_id] * len(total_ucm_block_ids)
            try:
                layer_ptrs = np.ascontiguousarray(total_ptrs[:, local_layer_id, :])
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
        if self.enable_event_sync:
            self.device.destroy_event_handles()


class UCMCPConnector(UCMLayerWiseConnector):
    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config, role)
        self.use_layerwise = (
            self._vllm_config.kv_transfer_config.kv_connector_extra_config.get(
                "use_layerwise", False
            )
        )

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

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config, role)
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
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        self.connector: KVConnectorBase_V1
        # TODO new conn by config
        use_layerwise = (
            self._vllm_config.kv_transfer_config.kv_connector_extra_config.get(
                "use_layerwise", False
            )
        )
        pp_enabled = self._vllm_config.parallel_config.pipeline_parallel_size > 1
        if pp_enabled and not use_layerwise:
            raise RuntimeError(
                "Pipeline parallelism is not supported in UCMDirectConnector, please set use_layerwise=True."
            )
        if (
            self._vllm_config.kv_transfer_config is not None
            and "hit_ratio"
            in self._vllm_config.kv_transfer_config.kv_connector_extra_config
        ):
            self.connector = UCMMockConnector(vllm_config, role)
        elif (
            hasattr(self._vllm_config.parallel_config, "prefill_context_parallel_size")
            and hasattr(
                self._vllm_config.parallel_config, "decode_context_parallel_size"
            )
            and self._vllm_config.parallel_config.prefill_context_parallel_size
            * self._vllm_config.parallel_config.decode_context_parallel_size
            > 1
        ):
            self.connector = UCMCPConnector(vllm_config, role)
        elif (
            self._vllm_config.kv_transfer_config is not None
            and self._vllm_config.kv_transfer_config.kv_connector_extra_config.get(
                "use_layerwise", False
            )
        ):
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
        return False, None