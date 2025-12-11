import copy
import hashlib
import itertools
import os
import pickle
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Dict, List, Optional

import torch
import yaml
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)
from vllm.distributed.parallel_state import get_tp_group, get_world_group
from vllm.platforms import current_platform
from vllm.v1.core.sched.output import SchedulerOutput

from ucm.logger import init_logger
from ucm.shared.metrics import ucmmonitor
from ucm.shared.metrics.observability import UCMStatsLogger
from ucm.store.factory import UcmConnectorFactory
from ucm.store.ucmstore import Task, UcmKVStoreBase
from ucm.utils import Config

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class RequestMeta:
    ucm_block_ids: list[str] = field(default_factory=list)
    hbm_hit_block_num: int = 0
    # local_computed_block + external_computed_block
    total_hit_block_num: int = 0
    num_token_ids: int = 0
    vllm_block_ids: list[int] = field(default_factory=list)
    token_processed: int = 0


@dataclass
class RequestDispatchMeta:
    load_block_ids: tuple[
        list[str], list[int]
    ]  # [0] mean ucm_block_ids, [1] means vllm_block_ids
    dump_block_ids: tuple[list[str], list[int]]


@dataclass
class UCMConnectorMetadata(KVConnectorMetadata):
    request_meta: dict[str, RequestDispatchMeta] = field(default_factory=dict)


class RequestHasher:
    """hash(md5) request to generate ucm block id"""

    _SEED_HASH = None

    def __init__(self, vllm_config, rank_id):
        meta = f"{vllm_config.model_config.model}:{vllm_config.parallel_config.world_size}:{vllm_config.model_config.dtype}:{rank_id}"
        self.meta_bytes = meta.encode("utf-8")

        if RequestHasher._SEED_HASH is None:
            RequestHasher._SEED_HASH = self("UCM_HASH_SEED")

    def __call__(self, input_data) -> int:
        if isinstance(input_data, str):
            input_bytes = input_data.encode("utf-8")
        else:
            input_bytes = pickle.dumps(input_data, protocol=pickle.HIGHEST_PROTOCOL)

        h = hashlib.md5(self.meta_bytes + input_bytes)
        return int.from_bytes(h.digest(), byteorder="big")


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
            self._layer_offset_cache = {}

        self.store: UcmKVStoreBase

        if role == KVConnectorRole.SCHEDULER:
            self.request_hasher = RequestHasher(vllm_config, 0)
        else:
            self.request_hasher = RequestHasher(vllm_config, self.global_rank)

        # save block info, avoid hash request twice, and track them until request finished
        self.requests_meta: dict[str, RequestMeta] = {}

        ucm_config = Config(vllm_config.kv_transfer_config)
        self.launch_config = ucm_config.get_config()

        self.load_only_first_rank: bool = (
            self.launch_config.get("load_only_first_rank", self.is_mla) and self.is_mla
        )
        if self.load_only_first_rank:
            if role == KVConnectorRole.WORKER:
                self.group_coordinator = get_tp_group()
                self.broadcast_fn = self.group_coordinator.broadcast
                self.broadcast_stream = torch.cuda.Stream()

        logger.info(f"self.launch_config: {self.launch_config}")
        connector_configs = self.launch_config.get("ucm_connectors", [])
        assert len(connector_configs) > 0, "no storage connector name in config."

        name = connector_configs[0].get("ucm_connector_name")
        config = connector_configs[0].get("ucm_connector_config") or {}
        config["device"] = self.local_rank
        config["role"] = "scheduler" if role == KVConnectorRole.SCHEDULER else "worker"
        element_size = vllm_config.model_config.dtype.itemsize
        single_head_dim = vllm_config.model_config.get_head_size()
        num_head_per_tp = vllm_config.model_config.get_num_kv_heads(
            vllm_config.parallel_config
        )
        total_tp_size = vllm_config.parallel_config.tensor_parallel_size
        num_layers = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        block_size_per_layer = self.block_size * element_size * single_head_dim
        config["kv_block_size"] = (
            block_size_per_layer
            * num_layers
            * (1 if self.is_mla else num_head_per_tp * 2)
        )
        config["io_size"] = block_size_per_layer * (
            1 if self.is_mla else num_head_per_tp
        )
        self.store = UcmConnectorFactory.create_connector(name, config)
        self.block_data_size = config["kv_block_size"]

        logger.info("init UCConnectorImpl, connector: %s", name)
        logger.info(
            "single file size = %d MB, io_size = %d KB,",
            config["kv_block_size"] / 1024 / 1024,
            config["io_size"] / 1024,
        )

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

    def generate_hash(self, block_size: int, request: "Request") -> list[str]:
        token_ids = request.all_token_ids

        ret = []
        parent_block_hash_value = RequestHasher._SEED_HASH
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
            ret.append(str(hash_value))

        return ret

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

    def _init_kv_caches_from_forward_context(self, forward_context: "ForwardContext"):
        if len(self.kv_caches) > 0:
            return
        for layer_name in forward_context.no_compile_layers:
            attn_layer = forward_context.no_compile_layers[layer_name]
            if not hasattr(attn_layer, "kv_cache"):
                continue

            if layer_name not in self.kv_caches:
                self.kv_caches[layer_name] = attn_layer.kv_cache[
                    forward_context.virtual_engine
                ]
        # Since vllm_ascend >= 0.10.0, the MLA model's tensor shape has changed to
        # (2, num_blocks, block_size, num_kv_heads, nope_dim/rope_dim).
        # Currently, we treat it as GQA, and use is_dsa to mark it,
        # which works but leads to space inefficiency.
        # TODO: Optimize this to avoid unnecessary space usage.
        sample_kv_layer = next(iter(self.kv_caches.values()))
        if self.is_mla and len(sample_kv_layer) == 2:
            self.is_mla = False
            self.is_dsa = True
        if self.kv_cache_dtype is None:
            self.kv_cache_dtype = sample_kv_layer[0].dtype

    @staticmethod
    def _extract_layer_index(layer_name: str) -> Optional[int]:
        """
        Extract the layer index from the layer name.
        """
        for chunk in layer_name.split("."):
            if chunk.isdigit():
                return int(chunk)
        return None

    def _precompute_layer_offsets(self):
        if not self.kv_caches:
            return

        sample_kv_layer = next(iter(self.kv_caches.values()))
        elem_size = sample_kv_layer[0].element_size()
        block_data_size = (
            sample_kv_layer[0].numel() if self.is_mla else sample_kv_layer[0][0].numel()
        ) * elem_size
        layer_data_size = block_data_size if self.is_mla else block_data_size * 2

        # precompute all layers offset
        for layer_name, _ in self.kv_caches.items():
            layer_id = self._extract_layer_index(layer_name)
            assert layer_id is not None
            k_offset = layer_data_size * layer_id
            v_offset = k_offset + block_data_size if not self.is_mla else 0
            self._layer_offset_cache[layer_name] = (k_offset, v_offset)

    def _get_tensor_and_offset(
        self, vllm_block_ids: list[int], kv_layer: torch.Tensor, layer_name: str
    ) -> tuple[list[torch.Tensor], list[int]]:
        """
        GQA/MHA: one layer shape is (2, num_blocks, block_size, num_kv_heads, head_size)
        MLA: one layer shape is (num_blocks, block_size, head_size)
        """
        k_tensors, k_offsets = [], []
        v_tensors, v_offsets = [], []
        k_offset, v_offset = self._layer_offset_cache[layer_name]

        for vllm_block_id in vllm_block_ids:
            k_tensors.append(
                kv_layer[vllm_block_id] if self.is_mla else kv_layer[0][vllm_block_id]
            )
            k_offsets.append(k_offset)
            if not self.is_mla:
                v_tensors.append(kv_layer[1][vllm_block_id])
                v_offsets.append(v_offset)
        return k_tensors + v_tensors, k_offsets + v_offsets

    def _generate_task(self, vllm_block_ids: List[int], ucm_block_ids: List[str]):
        if not self._layer_offset_cache:
            self._precompute_layer_offsets()

        num_layers = len(self.kv_caches)
        num_blocks_per_layer = len(vllm_block_ids)
        num_tensors_per_layer = num_blocks_per_layer * (1 if self.is_mla else 2)
        dst_tensor_addr = [None] * (num_layers * num_tensors_per_layer)
        ucm_offsets = [0] * (num_layers * num_tensors_per_layer)

        idx = 0
        for layer_name, one_layer_kv_cache in self.kv_caches.items():
            tensors, offsets = self._get_tensor_and_offset(
                vllm_block_ids, one_layer_kv_cache, layer_name
            )
            dst_tensor_addr[idx : idx + len(tensors)] = tensors
            ucm_offsets[idx : idx + len(offsets)] = offsets
            idx += len(tensors)

        repeat_times = len(self.kv_caches) * (1 if self.is_mla else 2)
        ucm_total_block_ids = ucm_block_ids * repeat_times

        assert len(ucm_total_block_ids) == len(ucm_offsets) == len(dst_tensor_addr)
        return ucm_total_block_ids, ucm_offsets, dst_tensor_addr

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

        self._init_kv_caches_from_forward_context(forward_context)

        request_to_task: dict[str, Optional[Task]] = {}
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
                    ucm_block_ids[i] = str(self.request_hasher(ucm_block_id))
            ucm_total_block_ids, ucm_offsets, dst_tensor_addr = self._generate_task(
                vllm_block_ids, ucm_block_ids
            )
            if self.global_rank == 0 or not self.load_only_first_rank:
                request_to_task[request_id] = self.store.load(
                    ucm_total_block_ids, ucm_offsets, dst_tensor_addr
                )
            else:
                request_to_task[request_id] = None
            req_broadcast_addr[request_id] = dst_tensor_addr

        for request_id, task in request_to_task.items():
            # TODO error handling
            if self.global_rank == 0 or not self.load_only_first_rank:
                if self.store.wait(task) != 0:
                    self._invalid_block_ids.update(
                        metadata.request_meta[request_id].load_block_ids[1]
                    )
                    logger.error(f"request {request_id} load kv cache failed.")
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
            # This has already been fixed in the latest main branch of vllm_ascend, so synchronize will no longer be needed in future versions.
            self.synchronize()

        metadata = self._get_connector_metadata()
        assert isinstance(metadata, UCMConnectorMetadata)

        request_to_task: dict[str, Task] = {}
        request_to_blocks: dict[str, list[str]] = {}
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
                    ucm_block_ids[i] = str(self.request_hasher(ucm_block_id))
            rets = self.store.create(ucm_block_ids)
            end = 0
            for i, ret in enumerate(rets):
                if ret != 0:
                    logger.error(
                        f"create blocks for {request_id} failed, block index: {i}, ret code: {ret}"
                    )
                    break
                end += 1

            if end == 0:
                continue
            ucm_block_ids = ucm_block_ids[:end]
            vllm_block_ids = vllm_block_ids[:end]
            ucm_total_block_ids, ucm_offsets, dst_tensor_addr = self._generate_task(
                vllm_block_ids, ucm_block_ids
            )
            request_to_task[request_id] = self.store.dump(
                ucm_total_block_ids, ucm_offsets, dst_tensor_addr
            )
            request_to_blocks[request_id] = ucm_block_ids

        for request_id, task in request_to_task.items():
            ucm_block_ids = request_to_blocks[request_id]
            if self.store.wait(task) == 0:
                self.store.commit(ucm_block_ids, True)
            else:
                logger.error(f"request {request_id} dump kv cache failed.")
                self.store.commit(ucm_block_ids, False)
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

    def get_kv_connector_stats(self) -> Optional[KVConnectorStats]:
        if not hasattr(self, "monitor") or self.monitor is None:
            return None

        # Get stats from monitor using get_stats_and_clear
        try:
            stats_dict = self.monitor.get_stats_and_clear("ConnStats")
            if not stats_dict:
                return None

            # Convert monitor stats to UCMKVConnectorStats format
            ucm_stats = UCMKVConnectorStats()
            # Record stats for this worker
            ucm_stats.record_stats(stats_dict, worker_rank=self.global_rank)

            if not ucm_stats.is_empty():
                return ucm_stats
            return None
        except Exception as e:
            logger.warning(f"Failed to get stats from monitor: {e}")
            return None

    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> Optional[KVConnectorStats]:
        return (
            UCMKVConnectorStats(data=data)
            if data is not None
            else UCMKVConnectorStats()
        )

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: "VllmConfig",
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> Optional[KVConnectorPromMetrics]:
        return UCMPromMetrics(
            vllm_config, metric_types, labelnames, per_engine_labelvalues
        )


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

    def get_kv_connector_stats(self) -> Optional[KVConnectorStats]:
        return self.connector.get_kv_connector_stats()

    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> Optional[KVConnectorStats]:
        return UCMDirectConnector.build_kv_connector_stats(data)

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> Optional[KVConnectorPromMetrics]:
        return UCMDirectConnector.build_prom_metrics(
            vllm_config, metric_types, labelnames, per_engine_labelvalues
        )


@dataclass
class UCMKVConnectorStats(KVConnectorStats):
    """Data structure: {worker_rank: {metric_name: [values]}}"""

    def reset(self):
        """Reset stats for all workers"""
        for worker_data in self.data.values():
            worker_data.clear()
        self.data.clear()

    def record_stats(self, stats_dict: dict[str, Any], worker_rank: int | None = None):
        """Record stats from monitor data for a specific worker"""
        worker_key = str(worker_rank) if worker_rank is not None else "0"

        if worker_key not in self.data:
            self.data[worker_key] = {}

        worker_data = self.data[worker_key]

        for key, value in stats_dict.items():
            if key not in worker_data:
                worker_data[key] = []

            if isinstance(value, list):
                worker_data[key].extend([float(v) for v in value])
            else:
                worker_data[key].append(float(value))

    def is_empty(self) -> bool:
        return all(
            all(len(v) == 0 for v in worker_data.values())
            for worker_data in self.data.values()
        )

    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
        """Aggregate stats from another worker, preserving per-worker separation"""
        if other.is_empty():
            return self

        assert isinstance(other, UCMKVConnectorStats), "Expected UCMKVConnectorStats"

        for worker_rank, worker_data in other.data.items():
            if worker_rank not in self.data:
                self.data[worker_rank] = copy.deepcopy(worker_data)
                continue

            for metric_name, values in worker_data.items():
                self.data[worker_rank].setdefault(metric_name, []).extend(values)

        return self

    def reduce(self) -> dict[str, int | float]:
        """Reduce the observations to representative values for CLI logging."""
        result = {}
        is_count_metric = lambda k: any(x in k for x in ["num", "requests", "blocks"])
        is_hit_rate_metric = lambda k: "hit_rate" in k or "hit_rates" in k

        for worker_rank, worker_data in sorted(self.data.items()):
            for metric_name, values in worker_data.items():
                if is_hit_rate_metric(metric_name) and worker_rank != "0":
                    continue

                suffix = "(total)" if is_count_metric(metric_name) else "(avg)"
                if is_hit_rate_metric(metric_name) and worker_rank == "0":
                    worker_key = f"{metric_name} {suffix}"
                else:
                    worker_key = f"worker_{worker_rank}_{metric_name} {suffix}"

                if not values:
                    result[worker_key] = 0 if is_count_metric(metric_name) else 0.0
                elif is_count_metric(metric_name):
                    result[worker_key] = int(sum(values))
                else:
                    result[worker_key] = round(sum(values) / len(values), 3)

        return result


class UCMPromMetrics(KVConnectorPromMetrics):
    """
    Prometheus metrics for UCM connector.
    Records metrics from self.monitor data based on metrics_configs.yaml configuration.
    """

    _config_cache: Dict[str, Dict[str, Any]] = {}

    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)

        config = self._load_config(vllm_config)
        prometheus_config = config.get("prometheus", {})

        self.metric_mappings: Dict[str, Dict[str, Any]] = {}
        self._init_metrics_from_config(
            labelnames, prometheus_config, per_engine_labelvalues
        )

    @staticmethod
    def _get_metrics_config_path(vllm_config: VllmConfig) -> str:
        """Get metrics_config_path from vllm_config"""
        if vllm_config.kv_transfer_config is None:
            return ""
        try:
            ucm_config = Config(vllm_config.kv_transfer_config)
            return ucm_config.get_config().get("metrics_config_path", "")
        except Exception:
            return ""

    def _load_config(self, vllm_config: VllmConfig) -> Dict[str, Any]:
        """Load configuration from YAML file with caching"""
        metrics_config_path = self._get_metrics_config_path(vllm_config)
        if not metrics_config_path:
            return {}

        if metrics_config_path in self._config_cache:
            return self._config_cache[metrics_config_path]

        try:
            with open(metrics_config_path, "r") as f:
                config = yaml.safe_load(f)
                if config:
                    self._config_cache[metrics_config_path] = config
                return config or {}
        except FileNotFoundError:
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML config file {metrics_config_path}: {e}")
            return {}

    def _init_metrics_from_config(
        self,
        labelnames: list[str],
        prometheus_config: Dict[str, Any],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        enabled = prometheus_config.get("enabled_metrics", {})
        metric_prefix = prometheus_config.get("metric_prefix", "ucm:")

        extended_labelnames = (
            labelnames + ["worker_id"] if "worker_id" not in labelnames else labelnames
        )

        metric_types_config = {
            "counter": (
                self._counter_cls,
                enabled.get("counters", True),
                prometheus_config.get("counters", []),
            ),
            "gauge": (
                self._gauge_cls,
                enabled.get("gauges", True),
                prometheus_config.get("gauges", []),
            ),
            "histogram": (
                self._histogram_cls,
                enabled.get("histograms", True),
                prometheus_config.get("histograms", []),
            ),
        }

        for metric_type, (
            metric_cls,
            is_enabled,
            metrics_list,
        ) in metric_types_config.items():
            if not is_enabled:
                continue

            for metric_cfg in metrics_list:
                if not (name := metric_cfg.get("name")):
                    continue

                doc = metric_cfg.get("documentation", "")
                prometheus_name = f"{metric_prefix}{name}"
                attr_name = f"{metric_type}_{name}"

                metric_kwargs = {
                    "name": prometheus_name,
                    "documentation": doc,
                    "labelnames": extended_labelnames,
                }
                if metric_type == "gauge":
                    metric_kwargs["multiprocess_mode"] = metric_cfg.get(
                        "multiprocess_mode", "live"
                    )
                elif metric_type == "histogram":
                    metric_kwargs["buckets"] = metric_cfg.get("buckets", [])

                metric = metric_cls(**metric_kwargs)
                setattr(self, attr_name, metric)
                self.metric_mappings[name] = {
                    "type": metric_type,
                    "attr": attr_name,
                    "extended_labelnames": extended_labelnames,
                }

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        """Record transfer statistics to Prometheus metrics based on configuration."""
        if transfer_stats_data and isinstance(transfer_stats_data, dict):
            first_key = next(iter(transfer_stats_data.keys()), None)
            if first_key and isinstance(transfer_stats_data[first_key], dict):
                for worker_rank, worker_stats in sorted(transfer_stats_data.items()):
                    self._observe_worker_stats(worker_stats, engine_idx, worker_rank)
                return

        # Fallback: treat as single worker (backward compatibility)
        self._observe_worker_stats(transfer_stats_data, engine_idx, "aggregated")

    def _observe_worker_stats(
        self, worker_stats: dict[str, Any], engine_idx: int, worker_rank: str
    ):
        """Helper method to observe stats for a specific worker"""
        if not worker_stats:
            return

        base_labelvalues = self._per_engine_labelvalues.get(engine_idx, [])
        extended_labelvalues = list(base_labelvalues) + [str(worker_rank)]

        for stat_name, value in worker_stats.items():
            try:
                if stat_name not in self.metric_mappings:
                    continue

                metric_mapped = self.metric_mappings[stat_name]
                base_metric = getattr(self, metric_mapped["attr"], None)
                if base_metric is None:
                    logger.warning(f"Metric {stat_name} not initialized.")
                    continue

                metric_type = metric_mapped["type"]
                extended_labelnames = metric_mapped.get("extended_labelnames", [])

                if "worker_id" in extended_labelnames:
                    per_engine_metric = base_metric.labels(*extended_labelvalues)
                else:
                    per_engine_metric = base_metric.labels(*base_labelvalues)

                values = value if isinstance(value, list) else [value]

                if metric_type == "counter":
                    for val in values:
                        if val >= 0:
                            per_engine_metric.inc(int(val))
                elif metric_type == "gauge":
                    if values:
                        per_engine_metric.set(float(values[-1]))
                elif metric_type == "histogram":
                    for val in values:
                        per_engine_metric.observe(float(val))

            except Exception as e:
                logger.warning(
                    f"Failed to log metric {stat_name} for worker {worker_rank}: {e}"
                )
