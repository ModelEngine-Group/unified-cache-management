import copy
import hashlib
import os
import pickle
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

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

from ucm.logger import init_logger
from ucm.shared.metrics import ucmmonitor
from ucm.shared.metrics.observability import UCMStatsLogger
from ucm.store.factory_v1 import UcmConnectorFactoryV1
from ucm.store.ucmstore_v1 import Task, UcmKVStoreBaseV1
from ucm.utils import Config

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

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


@dataclass
class UCMConnectorMetadata(KVConnectorMetadata):
    request_meta: dict[str, RequestDispatchMeta] = field(default_factory=dict)


class RequestHasher:
    """hash(md5) request to generate ucm block id"""

    def __init__(self, vllm_config, rank_id):
        meta = f"{vllm_config.model_config.model}:{vllm_config.parallel_config.world_size}:{vllm_config.model_config.dtype}:{rank_id}"
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

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        self.kv_caches: dict[str, torch.Tensor] = {}
        self.local_rank = (
            -1 if role == KVConnectorRole.SCHEDULER else get_world_group().local_rank
        )
        self.global_rank = self._vllm_config.parallel_config.rank
        self.block_size = self._vllm_config.cache_config.block_size
        self.is_mla = self._vllm_config.model_config.is_deepseek_mla
        self.is_dsa = False
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
        assert len(self.connector_configs) > 0, "no storage connector name in config."
        # TODO: haven't support broadcast
        self.load_only_first_rank = False
        if self.load_only_first_rank:
            if role == KVConnectorRole.WORKER:
                self.group_coordinator = get_tp_group()
                self.broadcast_fn = self.group_coordinator.broadcast
                self.broadcast_stream = torch.cuda.Stream()
        self.chunk_size = 1

        if role == KVConnectorRole.SCHEDULER:
            self.request_hasher = RequestHasher(vllm_config, 0)
            self._seed = self.request_hasher("UCM_HASH_SEED")
            # init scheduler-size connector
            self.store = self._create_store(None, None)
        else:
            self.request_hasher = RequestHasher(vllm_config, self.global_rank)

        self.metrics_config = self.launch_config.get("metrics_config_path", "")
        if self.metrics_config:
            self.stats_logger = UCMStatsLogger(
                vllm_config.model_config.served_model_name,
                self.global_rank,
                self.metrics_config,
            )
            self.monitor = ucmmonitor.StatsMonitor.get_instance()

        self.synchronize = (
            torch.cuda.synchronize
            if current_platform.is_cuda_alike()
            else torch.npu.synchronize
        )

        # invlalid block ids due to load errors
        self._invalid_block_ids: set[int] = set()

    def generate_hash(self, block_size: int, request: "Request") -> list[bytes]:
        token_ids = request.all_token_ids

        ret = []
        parent_block_hash_value = self._seed
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
        tensor_size: Optional[int],
        chunk_block_size: Optional[int],
        is_rope: bool = False,
    ) -> UcmKVStoreBaseV1:
        if len(self.connector_configs) != 1:
            raise RuntimeError(
                f"Expected exactly one connector config, "
                f"but got {len(self.connector_configs)}: "
                f"{self.connector_configs}"
            )

        store = None
        name = self.connector_configs[0]["ucm_connector_name"]
        config = copy.deepcopy(self.connector_configs[0]["ucm_connector_config"])
        config["storage_backends"] = self._generate_storage_backends(
            config["storage_backends"], is_rope
        )
        config["unique_id"] = (
            self.engine_id if not is_rope else self.engine_id + "_rope"
        )

        if self._role == KVConnectorRole.SCHEDULER:
            config["device_id"] = -1
            store = UcmConnectorFactoryV1.create_connector(name, config)
        else:
            config["device_id"] = self.local_rank
            config["tensor_size"] = tensor_size
            config["shard_size"] = chunk_block_size
            config["block_size"] = chunk_block_size
            if self.is_dsa or self.is_mla:
                config["share_buffer_enable"] = True
                config["local_rank_size"] = self.tp_size
            else:
                config["share_buffer_enable"] = False
            store = UcmConnectorFactoryV1.create_connector(name, config)

        return store

    def _generate_storage_backends(
        self, storage_backends: str, is_rope: bool = False
    ) -> List[str]:
        if is_rope:
            backends = [
                os.path.join(path, "rope") for path in storage_backends.split(":")
            ]
            os.makedirs(backends[0], exist_ok=True)
            return backends
        else:
            return [path for path in storage_backends.split(":") if path]

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        if os.environ["VLLM_HASH_ATTENTION"] == "1":
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
            # Since vllm_ascend >= 0.10.0, the MLA model's tensor shape has changed to Tuple
            # [(num_blocks, block_size, num_kv_heads, nope_dim/rope_dim)]
            # Currently, we treat it as GQA, dump rope_dim to a separate directory and use is_dsa to mark it
            for i, tensor in enumerate(sample_kv_layer):
                logger.info(f"kv cache shape {i}: {tensor.shape}")
            if self.is_mla:
                self.is_mla = False
                self.is_dsa = True
        logger.info(f"use mla: {self.is_mla}, use dsa: {self.is_dsa}")

        # Initialize KV cache base addresses
        k_ptrs, v_ptrs = [], []
        self.k_base_ptrs: np.ndarray
        self.v_base_ptrs: Optional[np.ndarray] = None
        for _, kv_layer in self.kv_caches.items():
            if len(sample_kv_layer) == 2:
                k_ptrs.append(kv_layer[0].data_ptr())
                v_ptrs.append(kv_layer[1].data_ptr())
            else:
                k_ptrs.append(kv_layer.data_ptr())
        self.k_base_ptrs = np.array(k_ptrs, dtype=np.uint64)
        self.v_base_ptrs = np.array(v_ptrs, dtype=np.uint64) if v_ptrs else None

        # init work-side connector
        tensor_size = (
            sample_kv_layer[0][0].numel() * sample_kv_layer[0][0].element_size()
            if not self.is_mla
            else sample_kv_layer[0].numel() * sample_kv_layer[0].element_size()
        )
        chunk_block_size = (
            tensor_size
            * self.num_layers
            * self.chunk_size
            * (1 if self.is_mla or self.is_dsa else 2)
        )
        self.block_stride = tensor_size

        self.block_data_size = chunk_block_size
        self.store = self._create_store(tensor_size, chunk_block_size)
        if self.is_dsa:
            rope_tensor_size = (
                sample_kv_layer[1][0].numel() * sample_kv_layer[1][0].element_size()
            )
            rope_chunk_block_size = rope_tensor_size * self.num_layers * self.chunk_size
            self.rope_store = self._create_store(
                rope_tensor_size, rope_chunk_block_size, True
            )
            self.rope_block_stride = rope_tensor_size
            self.block_data_size += rope_chunk_block_size

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        assert num_computed_tokens % self.block_size == 0
        hbm_hit_block_num = num_computed_tokens // self.block_size

        ucm_block_ids = self.generate_hash(self.block_size, request)

        external_block_ids = ucm_block_ids[hbm_hit_block_num:]
        if not external_block_ids:
            return 0, False

        lookup_results = self.store.lookup(external_block_ids)
        external_hit_blocks = 0
        for i, hit in enumerate(lookup_results):
            if not hit:
                break
            external_hit_blocks += 1
        logger.info(
            f"request_id: {request.request_id}, "
            f"total_blocks_num: {len(ucm_block_ids)}, "
            f"hit hbm: {hbm_hit_block_num}, "
            f"hit external: {external_hit_blocks}"
        )
        if self.metrics_config:
            self.monitor.update_stats(
                "ConnStats",
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
            load_ucm_block_ids = ucm_block_ids[hbm_hit_block_num:total_hit_block_num]
            load_vllm_block_ids = vllm_block_ids[hbm_hit_block_num:total_hit_block_num]

        if req_meta.token_processed < req_meta.num_token_ids:
            start_idx = req_meta.token_processed // self.block_size
            end_idx = (req_meta.token_processed + new_tokens) // self.block_size
            dump_ucm_block_ids = ucm_block_ids[start_idx:end_idx]
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
                    requests_dispatch_meta[request_id] = self._generate_dispatch_meta(
                        req_meta,
                        scheduler_output.num_scheduled_tokens[request_id],
                        new_block_ids,
                        scheduled_cached_reqs.resumed_from_preemption[i],
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

    @staticmethod
    def _extract_layer_index(layer_name: str) -> Optional[int]:
        """
        Extract the layer index from the layer name.
        """
        for chunk in layer_name.split("."):
            if chunk.isdigit():
                return int(chunk)
        return None

    def _get_tensors(
        self, vllm_block_id: int
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        GQA/MHA: one layer shape is (2, num_blocks, block_size, num_kv_heads, head_size)
        MLA: one layer shape is (num_blocks, block_size, head_size)
        """
        k_tensors, v_tensors = [], []
        for _, kv_layer in self.kv_caches.items():
            k_tensors.append(
                kv_layer[vllm_block_id] if self.is_mla else kv_layer[0][vllm_block_id]
            )
            if not self.is_mla:
                v_tensors.append(kv_layer[1][vllm_block_id])
        return k_tensors, v_tensors

    def _generate_task(
        self, vllm_block_ids: List[int], ucm_block_ids: List[bytes]
    ) -> Tuple[List[bytes], List[int], np.ndarray, np.ndarray]:
        block_addrs, rope_block_addrs = None, None
        vllm_block_ids_np = np.array(vllm_block_ids, np.uint64)
        k_addrs = (
            vllm_block_ids_np[:, None] * self.block_stride + self.k_base_ptrs[None, :]
        )
        num_blocks, num_layers = k_addrs.shape
        shard_indexs = [0] * num_blocks
        if self.v_base_ptrs is None:
            block_addrs = k_addrs
        elif self.is_dsa:
            v_addrs = (
                vllm_block_ids_np[:, None] * self.rope_block_stride
                + self.v_base_ptrs[None, :]
            )
            block_addrs = k_addrs
            rope_block_addrs = v_addrs
        else:
            v_addrs = (
                vllm_block_ids_np[:, None] * self.block_stride
                + self.v_base_ptrs[None, :]
            )
            block_addrs = np.empty((num_blocks, num_layers * 2), dtype=np.uint64)
            block_addrs[:, :num_layers] = k_addrs
            block_addrs[:, num_layers:] = v_addrs

        return ucm_block_ids, shard_indexs, block_addrs, rope_block_addrs

    def _broadcast(self, dst_tensor_addr: list[torch.Tensor]):
        rec_tensor: torch.Tensor = None
        with torch.cuda.stream(self.broadcast_stream):
            # TODO support broadcast when PP
            if self.global_rank == 0:
                tensor_to_broadcast = torch.stack(dst_tensor_addr, dim=0)
                self.broadcast_fn(tensor_to_broadcast, 0)
            else:
                shape = (len(dst_tensor_addr),) + dst_tensor_addr[0].shape
                # TODO create earlier
                rec_tensor = torch.empty(
                    shape, dtype=self.kv_cache_dtype, device=self.device
                )
                self.broadcast_fn(rec_tensor, 0)
        self.broadcast_stream.synchronize()
        if self.global_rank != 0 and rec_tensor is not None:
            for i, tensor in enumerate(dst_tensor_addr):
                tensor.copy_(rec_tensor[i])

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, UCMConnectorMetadata)

        request_to_task: dict[str, Optional[List[Task]]] = {}
        req_broadcast_addr = {}
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
            if self.global_rank != 0 and not self.is_mla and not self.is_dsa:
                for i, ucm_block_id in enumerate(ucm_block_ids):
                    ucm_block_ids[i] = self.request_hasher(ucm_block_id)
            block_ids, shard_indexs, total_tensors, rope_tensors = self._generate_task(
                vllm_block_ids, ucm_block_ids
            )
            if self.global_rank == 0 or not self.load_only_first_rank:
                task = self.store.load_data(block_ids, shard_indexs, total_tensors)
                request_to_task[request_id] = [task]
                if rope_tensors is not None and self.rope_store:
                    rope_task = self.rope_store.load_data(
                        block_ids, shard_indexs, rope_tensors
                    )
                    request_to_task[request_id].append(rope_task)
            else:
                request_to_task[request_id] = None

        for request_id, tasks in request_to_task.items():
            # TODO error handling
            if self.global_rank == 0 or not self.load_only_first_rank:
                try:
                    self.store.wait(tasks[0])
                    if len(tasks) > 1 and self.rope_store:
                        self.rope_store.wait(tasks[1])
                except RuntimeError as e:
                    logger.error("request {request_id} load kv cache failed.:", e)
                    self._invalid_block_ids.update(
                        metadata.request_meta[request_id].load_block_ids[1]
                    )
            if self.load_only_first_rank:
                self._broadcast(req_broadcast_addr[request_id])
        load_end_time = time.perf_counter() * 1000
        load_speed = (
            num_loaded_block
            * self.block_data_size
            / (load_end_time - load_start_time)
            / 1024
            / 1024
        )  # GB/s
        if self.metrics_config and is_load:
            self.monitor.update_stats(
                "ConnStats",
                {
                    "load_requests_num": num_loaded_request,
                    "load_blocks_num": num_loaded_block,
                    "load_duration": load_end_time - load_start_time,
                    "load_speed": load_speed,
                },
            )

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

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
        if (self.is_mla or self.is_dsa) and self.global_rank != 0:
            return
        if self.metrics_config or current_platform.device_type == "npu":
            # When use vllm_ascend, we should add synchronize here, otherwise accuracy problem will raise
            # This has already been fixed in the latest main branch of vllm_ascend,
            # so synchronize will no longer be needed in future versions.
            self.synchronize()

        metadata = self._get_connector_metadata()
        assert isinstance(metadata, UCMConnectorMetadata)

        request_to_task: dict[str, List[Task]] = {}
        is_save = False
        num_saved_block = 0
        num_saved_request = 0
        save_start_time = time.perf_counter() * 1000
        for request_id, request in metadata.request_meta.items():
            if len(request.dump_block_ids[0]) == 0:
                continue
            is_save = True
            num_saved_block += len(request.dump_block_ids[0])
            num_saved_request += 1

            ucm_block_ids, vllm_block_ids = request.dump_block_ids
            if self.global_rank != 0:
                for i, ucm_block_id in enumerate(ucm_block_ids):
                    ucm_block_ids[i] = self.request_hasher(ucm_block_id)
            block_ids, shard_indexs, total_tensors, rope_tensors = self._generate_task(
                vllm_block_ids, ucm_block_ids
            )
            task = self.store.dump_data(block_ids, shard_indexs, total_tensors)
            request_to_task[request_id] = [task]
            if rope_tensors is not None and self.rope_store:
                rope_task = self.rope_store.dump_data(
                    block_ids, shard_indexs, rope_tensors
                )
                request_to_task[request_id].append(rope_task)

        for request_id, tasks in request_to_task.items():
            try:
                self.store.wait(tasks[0])
                if len(tasks) > 1 and self.rope_store:
                    self.rope_store.wait(tasks[1])
            except RuntimeError as e:
                logger.error("request {request_id} dump kv cache failed.:", e)
        save_end_time = time.perf_counter() * 1000
        save_speed = (
            num_saved_block
            * self.block_data_size
            / (save_end_time - save_start_time)
            / 1024
            / 1024
        )  # GB/s
        if self.metrics_config and is_save:
            self.monitor.update_stats(
                "ConnStats",
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

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config, role)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        raise NotImplementedError

    def wait_for_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        raise NotImplementedError

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        raise NotImplementedError

    def wait_for_save(self) -> None:
        raise NotImplementedError


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


class UCMConnector(KVConnectorBase_V1):
    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        self.connector: KVConnectorBase_V1
        # TODO new conn by config
        if (
            self._vllm_config.kv_transfer_config is not None
            and "hit_ratio"
            in self._vllm_config.kv_transfer_config.kv_connector_extra_config
        ):
            self.connector = UCMMockConnector(vllm_config, role)
        else:
            self.connector = UCMDirectConnector(vllm_config, role)

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
