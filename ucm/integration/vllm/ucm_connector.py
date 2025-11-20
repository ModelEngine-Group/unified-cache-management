import hashlib
import itertools
import os
import pickle
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, List, Optional

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.parallel_state import get_tp_group, get_world_group
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request

from ucm.logger import init_logger
from ucm.store.factory import UcmConnectorFactory
from ucm.store.ucmstore import Task, UcmKVStoreBase
from ucm.utils import Config

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks

logger = init_logger(__name__)


@dataclass
class RequestMeta:
    ucm_block_ids: list[str] = field(default_factory=list)
    hbm_hit_block_num: int = 0
    # local_computed_block + external_computed_block
    total_hit_block_num: int = 0


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

    def __init__(self):
        if RequestHasher._SEED_HASH is None:
            RequestHasher._SEED_HASH = self._md5("UCM_HASH_SEED")

    @staticmethod
    def _md5(input_data) -> int:
        input_bytes = pickle.dumps(input_data, protocol=pickle.HIGHEST_PROTOCOL)
        md5_bytes = hashlib.md5(input_bytes).digest()
        return int.from_bytes(md5_bytes, byteorder="big")

    def __call__(self, block_size: int, request: "Request") -> list[str]:
        token_ids = request.all_token_ids

        ret = []
        parent_block_hash_value = None
        for start in range(0, len(token_ids), block_size):
            end = start + block_size
            block_token_ids = token_ids[start:end]
            # Do not hash the block if it is not full.
            if len(block_token_ids) < block_size:
                break

            if not parent_block_hash_value:
                parent_block_hash_value = RequestHasher._SEED_HASH

            block_token_ids_tuple = tuple(block_token_ids)
            hash_value = self._md5((parent_block_hash_value, block_token_ids_tuple))
            parent_block_hash_value = hash_value
            ret.append(str(hash_value))

        return ret


class UCMDirectConnector(KVConnectorBase_V1):
    """
    This connector means synchronize:
    load -> forward -> save
    """

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        self.kv_caches: dict[str, torch.Tensor] = {}
        self.rank = (
            -1 if role == KVConnectorRole.SCHEDULER else get_world_group().local_rank
        )
        self.block_size = self._vllm_config.cache_config.block_size
        self.is_mla = self._vllm_config.model_config.is_deepseek_mla

        self.store: UcmKVStoreBase

        self.request_hasher = RequestHasher()

        # save block info, avoid hash request twice, and track them until request finished
        self.requests_meta: dict[str, RequestMeta] = {}
        ucm_config = Config(vllm_config.kv_transfer_config)
        self.launch_config = ucm_config.get_config()

        if "ucm_connector_name" in self.launch_config:
            name = self.launch_config.get("ucm_connector_name")
            config = self.launch_config.get("ucm_connector_config") or {}
            config["device"] = self.rank
            config["role"] = (
                "scheduler" if role == KVConnectorRole.SCHEDULER else "worker"
            )
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
                * (1 if self.is_mla else num_head_per_tp * total_tp_size * 2)
            )
            config["io_size"] = block_size_per_layer * (
                1 if self.is_mla else num_head_per_tp
            )
            self.load_only_first_rank: bool = config.get(
                "load_only_first_rank", self.is_mla
            )
            if self.load_only_first_rank:
                if role == KVConnectorRole.WORKER:
                    self.group_coordinator = get_tp_group()
                    self.broadcast_fn = self.group_coordinator.broadcast
                    self.broadcast_stream = torch.cuda.Stream()
            self.store = UcmConnectorFactory.create_connector(name, config)

            logger.info("init UCConnectorImpl, connector: %s", name)
            logger.info(
                "single file size = %d MB, io_size = %d KB,",
                config["kv_block_size"] / 1024 / 1024,
                config["io_size"] / 1024,
            )
        else:
            raise TypeError(f"no storage connector name in config.")

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:

        assert num_computed_tokens % self.block_size == 0
        hbm_hit_block_num = num_computed_tokens // self.block_size

        ucm_block_ids = self.request_hasher(self.block_size, request)

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

        total_hit_block_num = hbm_hit_block_num + external_hit_blocks

        external_hit_tokens = external_hit_blocks * self.block_size

        # When all the tokens are cached in ssd or hbm,
        # we need to recompute the last token. This if condition will be removed
        # once vLLM scheduler provides a better solution in the future.
        if external_hit_tokens == request.num_prompt_tokens:
            external_hit_tokens -= 1

        self.requests_meta[request.request_id] = RequestMeta(
            ucm_block_ids=ucm_block_ids,
            hbm_hit_block_num=hbm_hit_block_num,
            total_hit_block_num=total_hit_block_num,
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

        new_blocks_num = new_tokens // self.block_size
        hbm_hit_block_num = req_meta.hbm_hit_block_num
        total_hit_block_num = req_meta.total_hit_block_num
        scheduled_block_num = total_hit_block_num + new_blocks_num
        ucm_block_ids = req_meta.ucm_block_ids

        dump_ucm_block_ids = ucm_block_ids[total_hit_block_num:scheduled_block_num]
        if need_load:
            dump_vllm_block_ids = vllm_block_ids[
                total_hit_block_num:scheduled_block_num
            ]
        else:
            dump_vllm_block_ids = vllm_block_ids

        # after this round, req_meta will be updated
        req_meta.total_hit_block_num = scheduled_block_num

        load_ucm_block_ids, load_vllm_block_ids = [], []
        if need_load:
            load_ucm_block_ids = ucm_block_ids[hbm_hit_block_num:total_hit_block_num]
            load_vllm_block_ids = vllm_block_ids[hbm_hit_block_num:total_hit_block_num]

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
                if scheduler_output.num_scheduled_tokens[request_id] == 1:
                    # decode stage
                    continue
                req_meta = self.requests_meta.get(request_id)
                if req_meta:
                    requests_dispatch_meta[request_id] = self._generate_dispatch_meta(
                        req_meta,
                        scheduler_output.num_scheduled_tokens[request_id],
                        scheduled_cached_reqs.new_block_ids[i][0],
                        scheduled_cached_reqs.resumed_from_preemption[i],
                    )
        else:
            for request in scheduled_cached_reqs:
                request_id = request.request_id
                if scheduler_output.num_scheduled_tokens[request_id] == 1:
                    # decode stage
                    continue
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

    @staticmethod
    def _extract_layer_index(layer_name: str) -> Optional[int]:
        """
        Extract the layer index from the layer name.
        """
        for chunk in layer_name.split("."):
            if chunk.isdigit():
                return int(chunk)
        return None

    def _data_offset(self, kv_layer, layer_id, is_v) -> int:
        """
        GQA/MHA: one layer shape is (2, num_blocks, block_size, num_kv_heads, head_size)
        MLA: one layer shape is (1, num_blocks, block_size, head_size)
        """
        elem_size = kv_layer[0].element_size()
        block_data_size = (
            kv_layer[0].numel() if self.is_mla else kv_layer[0][0].numel()
        ) * elem_size
        if is_v:
            return self._data_offset(kv_layer, layer_id, False) + block_data_size

        layer_data_size = block_data_size if self.is_mla else block_data_size * 2
        return layer_data_size * layer_id

    def _get_tensor_and_offset(
        self, vllm_block_ids: list[int], kv_layer: torch.Tensor, layer_name: str
    ) -> tuple[list[torch.Tensor], list[int]]:
        k_tensors, k_offsets = [], []
        v_tensors, v_offsets = [], []
        layer_id = self._extract_layer_index(layer_name)
        assert layer_id is not None

        for vllm_block_id in vllm_block_ids:
            offset = self._data_offset(kv_layer, layer_id, False)
            tensor = (
                kv_layer[vllm_block_id] if self.is_mla else kv_layer[0][vllm_block_id]
            )
            k_tensors.append(tensor)
            k_offsets.append(offset)
            if not self.is_mla:
                v_offset = self._data_offset(kv_layer, layer_id, True)
                v_tensors.append(kv_layer[1][vllm_block_id])
                v_offsets.append(v_offset)
        return k_tensors + v_tensors, k_offsets + v_offsets

    def _generate_task(
        self,
        vllm_block_ids,
        ucm_block_ids,
        func: Callable[[List[str], List[int], List[torch.Tensor]], Task],
    ) -> Task:
        dst_tensor_addr, ucm_offsets = [], []
        for layer_name, one_layer_kv_cache in self.kv_caches.items():
            addrs, offsets = self._get_tensor_and_offset(
                vllm_block_ids, one_layer_kv_cache, layer_name
            )
            dst_tensor_addr.extend(addrs)
            ucm_offsets.extend(offsets)
        ucm_total_block_ids = ucm_block_ids * len(self.kv_caches)
        assert len(ucm_total_block_ids) == len(ucm_offsets) == len(dst_tensor_addr)
        return func(ucm_total_block_ids, ucm_offsets, dst_tensor_addr)

    def _generate_load_task_for_broadcast(
        self,
        vllm_block_ids,
        ucm_block_ids,
        can_load: bool,
    ) -> tuple[Task, dict[str, torch.Tensor], int]:
        """
        Load or Dump func is only called in rank 0 in MLA;
        In rank != 0, worker will receive broadcast tensors from rank 0.
        """
        layer_to_tensors = {}
        total_block_num = len(ucm_block_ids)
        dst_tensor_addr, ucm_offsets = [], []
        for layer_name, one_layer_kv_cache in self.kv_caches.items():
            addrs, offsets = self._get_tensor_and_offset(
                vllm_block_ids, one_layer_kv_cache, layer_name
            )
            layer_to_tensors[layer_name] = addrs[:total_block_num]
            dst_tensor_addr.extend(addrs)
            ucm_offsets.extend(offsets)
        ucm_total_block_ids = ucm_block_ids * len(self.kv_caches)

        task = None
        if can_load:
            assert len(ucm_total_block_ids) == len(ucm_offsets) == len(dst_tensor_addr)
            task = self.store.load(ucm_total_block_ids, ucm_offsets, dst_tensor_addr)
        return task, layer_to_tensors, total_block_num

    def _broadcast_or_receive_blocks(
        self, layer_to_tensors: dict[str : torch.Tensor], total_block_num
    ):
        receive_dict = {}
        for layer_name, kv_layer in self.kv_caches.items():
            k_tensors = layer_to_tensors[layer_name][:total_block_num]
            if self.rank == 0:
                tensor_to_broadcast = torch.stack(k_tensors, dim=0)
                self.broadcast_fn(tensor_to_broadcast, 0)
            else:
                shape = (len(k_tensors),) + k_tensors[0].shape
                dtype = k_tensors[0].dtype
                rec_tensor = torch.empty(shape, dtype=dtype, device=f"cuda:{self.rank}")
                self.broadcast_fn(rec_tensor, 0)
                receive_dict[layer_name] = rec_tensor
        return receive_dict

    def _wait_for_broadcast(
        self,
        req_id: str,
        task: Task,
        layer_to_tensors: dict[str, torch.Tensor],
        total_block_num: int,
    ):
        if self.rank == 0:
            if self.store.wait(task) != 0:
                logger.error(f"request {req_id} load kv cache failed.")
                return
            logger.debug(
                f"request {req_id} load {total_block_num} blocks on rank {self.rank}"
            )
        with torch.cuda.stream(self.broadcast_stream):
            receive_dict = self._broadcast_or_receive_blocks(
                layer_to_tensors, total_block_num
            )
        self.broadcast_stream.synchronize()
        if self.rank > 0 and receive_dict:
            for layer_name, kv_layer in self.kv_caches.items():
                received_tensor = receive_dict[layer_name]
                for i in range(total_block_num):
                    layer_to_tensors[layer_name][i].copy_(received_tensor[i])
            logger.debug(
                f"request {req_id} receive broadcast {total_block_num} blocks on rank {self.rank}"
            )

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:

        metadata = self._get_connector_metadata()
        assert isinstance(metadata, UCMConnectorMetadata)

        self._init_kv_caches_from_forward_context(forward_context)

        request_to_task: dict[str, Optional[Task]] = {}
        req_to_layer = {}
        for request_id, request in metadata.request_meta.items():
            if len(request.load_block_ids[0]) == 0:
                continue

            ucm_block_ids, vllm_block_ids = request.load_block_ids
            if self.load_only_first_rank:
                can_load = self.rank == 0
                task, layer_to_tensors, total_block_num = (
                    self._generate_load_task_for_broadcast(
                        vllm_block_ids, ucm_block_ids, can_load
                    )
                )
                req_to_layer[request_id] = (layer_to_tensors, total_block_num)
            else:
                task = self._generate_task(
                    vllm_block_ids, ucm_block_ids, self.store.load
                )
            request_to_task[request_id] = task

        for req_id, task in request_to_task.items():
            if self.load_only_first_rank:
                layer_to_tensors, total_block_num = req_to_layer[req_id]
                self._wait_for_broadcast(
                    req_id, task, layer_to_tensors, total_block_num
                )
            else:
                if self.store.wait(task) != 0:
                    logger.error(f"request {req_id} load kv cache failed.")

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

        if self.load_only_first_rank and self.rank != 0:
            return

        metadata = self._get_connector_metadata()
        assert isinstance(metadata, UCMConnectorMetadata)

        request_to_task: dict[str, Task] = {}
        request_to_blocks: dict[str, list[str]] = {}
        for request_id, request in metadata.request_meta.items():
            if len(request.dump_block_ids[0]) == 0:
                continue

            ucm_block_ids, vllm_block_ids = request.dump_block_ids
            rets = self.store.create(ucm_block_ids)
            end = 0
            for i, ret in enumerate(rets):
                if ret != 0:
                    logger.error(
                        f"create blocks for {request_id} failed, block index: {i}, ret code: {ret}"
                    )
                    break
                end += 1

            ucm_block_ids = ucm_block_ids[:end]
            vllm_block_ids = vllm_block_ids[:end]
            request_to_task[request_id] = self._generate_task(
                vllm_block_ids, ucm_block_ids, self.store.dump
            )
            request_to_blocks[request_id] = ucm_block_ids

        for request_id, task in request_to_task.items():
            ucm_block_ids = request_to_blocks[request_id]
            if self.store.wait(task) == 0:
                self.store.commit(ucm_block_ids, True)
            else:
                logger.error(f"request {request_id} dump kv cache failed.")
                self.store.commit(ucm_block_ids, False)

    def clear_connector_metadata(self) -> None:
        super().clear_connector_metadata()


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
