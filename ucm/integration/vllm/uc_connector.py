#
# MIT License
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Adapted from lmcache/lmcache/integration/vllm/vllm_v1_adapter.py
#
import hashlib
import pickle
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Generator, List, Optional, Union

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.parallel_state import get_world_group
from vllm.v1.core.kv_cache_utils import hash_request_tokens
from vllm.v1.core.sched.output import SchedulerOutput

from ucm.logger import init_logger
from ucm.store.base import Task
from ucm.store.factory import UcmConnectorFactory

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

logger = init_logger(__name__)


class BlockOperation(Enum):
    NONE = "none"
    LOAD = "load"
    DUMP = "dump"


@dataclass
class RequestBlockInfo:
    # Hash values for all blocks
    block_hashes: list[str] = field(default_factory=list)
    # Operation type for each block
    block_operations: list[BlockOperation] = field(default_factory=list)
    # Next block position to process
    start_position: int = 0


@dataclass
class ReqMeta:
    request_id: str
    # list[(block_hash, vllm_block_id)]
    load_blocks: list[tuple[str, int]] = field(default_factory=list)
    # list[(block_hash, vllm_block_id)]
    dump_blocks: list[tuple[str, int]] = field(default_factory=list)
    # Whether use load_async
    load_async: bool = False


@dataclass
class UCConnectorV1Metadata(KVConnectorMetadata):
    requests: list[ReqMeta] = field(default_factory=list)


class UnifiedCacheConnectorV1(KVConnectorBase_V1):

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        self.block_size = vllm_config.cache_config.block_size
        self.use_layerwise = True
        self.kv_caches: dict[str, torch.Tensor] = {}
        self.total_tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.rank = (
            -1 if role == KVConnectorRole.SCHEDULER else get_world_group().local_rank
        )
        self.request_block_infos: dict[str, RequestBlockInfo] = {}
        # dump tasks record request -> block -> list[task]
        self.dump_tasks: dict[str, dict[str, List[Task]]] = {}
        self.layerwise_load_tasks: dict[str, dict[str, tuple[Task, Task]]] = {}
        self.is_mla = self._vllm_config.model_config.is_deepseek_mla
        self.num_layers = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.element_size = vllm_config.model_config.dtype.itemsize
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self._need_load_reqs: dict[str, Union[list[int], list[Task]]] = {}
        self._load_failed_reqs: set[str] = set()
        self._load_req_to_blocks: dict[str, set[int]] = {}
        if (
            self._vllm_config.kv_transfer_config is not None
            and "ucm_connector_name"
            in self._vllm_config.kv_transfer_config.kv_connector_extra_config
        ):
            name = self._vllm_config.kv_transfer_config.kv_connector_extra_config[
                "ucm_connector_name"
            ]
            config = {}
            if (
                "ucm_connector_config"
                in self._vllm_config.kv_transfer_config.kv_connector_extra_config
            ):
                config = self._vllm_config.kv_transfer_config.kv_connector_extra_config[
                    "ucm_connector_config"
                ]
            config["device"] = self.rank
            config["role"] = (
                "scheduler" if role == KVConnectorRole.SCHEDULER else "worker"
            )
            logger.info("init UCConnectorImpl, connector: %s", name)
            self.connector = UcmConnectorFactory.create_connector(name, config)
        else:
            raise TypeError(f"no storage connector.")
        if (
            self._vllm_config.kv_transfer_config is not None
            and "use_layerwise"
            in self._vllm_config.kv_transfer_config.kv_connector_extra_config
        ):
            self.use_layerwise = (
                self._vllm_config.kv_transfer_config.kv_connector_extra_config[
                    "use_layerwise"
                ]
            )

    def _init_kv_caches_from_forward_context(self, forward_context: "ForwardContext"):
        for layer_name in forward_context.no_compile_layers:
            attn_layer = forward_context.no_compile_layers[layer_name]
            if not hasattr(attn_layer, "kv_cache"):
                logger.debug("The layer %s does not have kv_cache, skip it", layer_name)
                continue

            if layer_name not in self.kv_caches:
                self.kv_caches[layer_name] = attn_layer.kv_cache[
                    forward_context.virtual_engine
                ]

    def DataOffset(self, kv_layer, rank, layer_id, is_v):
        # Non-MLA scene: one layer shape is (2, num_blocks, block_size, num_kv_heads, head_size)
        # MLA scene: one layer shape is (num_blocks, block_size, head_size)
        # Element size
        elem_size = kv_layer[0].element_size()
        logger.debug(
            f"total_tp_size = {self.total_tp_size},\n" f"element size = {elem_size}."
        )
        # One block size
        k_min_data_block_size = (
            kv_layer[0][0].numel() if not self.is_mla else kv_layer[0].numel()
        ) * elem_size
        v_min_data_block_size = (
            kv_layer[1][0].numel() if not self.is_mla else 0
        ) * elem_size
        # When tp > 1 layer_size = (k_min_data_block_size + v_min_data_block_size) * tp_size
        layer_size = (
            k_min_data_block_size + v_min_data_block_size
        ) * self.total_tp_size
        if is_v:
            # Offset of v = Offset of k + k_min_data_block_size
            return int(
                self.DataOffset(kv_layer, rank, layer_id, False) + k_min_data_block_size
            )
        if self.is_mla:
            return int(layer_size * layer_id)
        else:
            # Offset of k = layer_size * layer_id + layer_size / tp_size * current rank
            return int(
                layer_size * layer_id + layer_size / self.total_tp_size * self.rank
            )

    def get_tensor_and_offset_layerwise(
        self, vllm_block_ids: List[int], kv_layer: torch.Tensor, layer_name: str
    ) -> tuple[List[torch.Tensor], List[int]]:
        k_tensors = []
        k_offsets = []
        v_tensors = []
        v_offsets = []
        layer_id = self._extract_layer_index(layer_name)

        for blk_id in vllm_block_ids:
            k_data_offset = self.DataOffset(kv_layer, self.rank, layer_id, False)
            if self.is_mla:
                k_tensors.append(kv_layer[blk_id])
            else:
                k_tensors.append(kv_layer[0][blk_id])
            k_offsets.append(k_data_offset)
            if not self.is_mla:
                v_data_offset = self.DataOffset(kv_layer, self.rank, layer_id, True)
                v_tensors.append(kv_layer[1][blk_id])
                v_offsets.append(v_data_offset)
        return k_tensors + v_tensors, k_offsets + v_offsets

    # ==============================
    # Worker-side methods
    # ==============================
    def clear_connector_metadata(self) -> None:
        """Clear the connector metadata.

        This function should be called by the model runner every time
        after the model execution.
        """
        self._load_failed_reqs.clear()
        self._load_req_to_blocks.clear()
        super().clear_connector_metadata()

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
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, UCConnectorV1Metadata)

        if len(self.kv_caches) == 0:
            self._init_kv_caches_from_forward_context(forward_context)

        self.layerwise_load_tasks.clear()
        self.current_layer = 0
        for request in metadata.requests:
            if not request.load_blocks:
                continue

            storage_block_ids = [block[0] for block in request.load_blocks]
            vllm_block_ids = [block[1] for block in request.load_blocks]
            blocks_len = len(storage_block_ids)
            self._load_req_to_blocks.setdefault(request.request_id, set()).update(
                vllm_block_ids
            )
            for layer_name, kv_layer in self.kv_caches.items():
                tensors, offsets = self.get_tensor_and_offset_layerwise(
                    vllm_block_ids, kv_layer, layer_name
                )
                k_task_id = self.connector.load(
                    storage_block_ids, offsets[:blocks_len], tensors[:blocks_len]
                )
                v_task_id = None
                if not self.is_mla:
                    v_task_id = self.connector.load(
                        storage_block_ids,
                        offsets[blocks_len:],
                        tensors[blocks_len:],
                    )
                if request.request_id not in self.layerwise_load_tasks:
                    self.layerwise_load_tasks[request.request_id] = {}
                self.layerwise_load_tasks[request.request_id][layer_name] = (
                    k_task_id,
                    v_task_id,
                )

            if request.load_async and request.request_id in self.layerwise_load_tasks:
                for _, (k_task, v_task) in self.layerwise_load_tasks[
                    request.request_id
                ].items():
                    if request.request_id not in self._need_load_reqs:
                        self._need_load_reqs[request.request_id] = []
                    self._need_load_reqs[request.request_id].append(k_task)
                    if not self.is_mla:
                        self._need_load_reqs[request.request_id].append(v_task)
                continue

            if (
                not self.use_layerwise
                and request.request_id in self.layerwise_load_tasks
            ):
                for _, (k_task, v_task) in self.layerwise_load_tasks[
                    request.request_id
                ].items():
                    if self.connector.wait(k_task) != 0:
                        self._load_failed_reqs.add(request.request_id)
                        break
                    if v_task and self.connector.wait(v_task) != 0:
                        self._load_failed_reqs.add(request.request_id)
                        break

    def wait_for_layer_load(self, layer_name: str) -> None:
        """
        Block until the KV for a specific layer is loaded into vLLM's
        paged buffer. This is called from within attention layer to ensure
        async copying from start_load_kv is complete.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        if not self.use_layerwise:
            return
        if self.layerwise_load_tasks:
            logger.debug(f"Waiting for layer {self.current_layer} to be loaded")

        assert (
            self.current_layer < self.num_layers
        ), "The current layer should be less than total layers!"
        for request_id, layer_to_task in self.layerwise_load_tasks.items():
            if request_id in self._load_failed_reqs:
                continue
            k_task, v_task = layer_to_task[layer_name]
            if self.connector.wait(k_task) != 0:
                self._load_failed_reqs.add(request_id)
                logger.error(
                    f"Failed to load block for request {request_id} on layer {layer_name}"
                )
                continue
            if not self.is_mla:
                if self.connector.wait(v_task) != 0:
                    self._load_failed_reqs.add(request_id)
                    logger.error(
                        f"Failed to load block for request {request_id} on layer {layer_name}"
                    )
                    continue
            logger.debug(f"Load tasks for {request_id} on layer {layer_name} finished.")

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
        if self.is_mla and self.rank != 0:
            return
        self.current_layer += 1
        if hasattr(self, "kv_role") and self.kv_role == "kv_consumer":
            return

        if not self.use_layerwise:
            return

        metadata = self._get_connector_metadata()
        assert isinstance(metadata, UCConnectorV1Metadata)

        for request in metadata.requests:
            if not request.dump_blocks or request.load_async:
                continue

            # Extract storage block IDs and vLLM block IDs from dump_blocks, same for load_blocks
            # dump_blocks format: [(block_hash, vllm_block_id), ...]
            # Note: block_hash is the storage_block_id
            # Example: [("hash_123", 5), ("hash_456", 8), ("hash_789", 12)]
            # ["hash_123", "hash_456", "hash_789"]
            storage_block_ids = [block[0] for block in request.dump_blocks]
            vllm_block_ids = [block[1] for block in request.dump_blocks]  # [5, 8, 12]
            blocks_len = len(storage_block_ids)
            tensors, offsets = self.get_tensor_and_offset_layerwise(
                vllm_block_ids, kv_layer, layer_name
            )

            if kv_layer[0].device.type == "npu":
                torch.npu.current_stream().synchronize()
            elif kv_layer[0].device.type == "cuda":
                torch.cuda.current_stream().synchronize()

            for block_id, offset, tensor in zip(
                storage_block_ids, offsets[:blocks_len], tensors[:blocks_len]
            ):
                task = self.connector.dump([block_id], [offset], [tensor])
                self.dump_tasks.setdefault(request.request_id, {}).setdefault(
                    block_id, []
                ).append(task)
            if not self.is_mla:
                for block_id, offset, tensor in zip(
                    storage_block_ids, offsets[blocks_len:], tensors[blocks_len:]
                ):
                    task = self.connector.dump([block_id], [offset], [tensor])
                    self.dump_tasks.setdefault(request.request_id, {}).setdefault(
                        block_id, []
                    ).append(task)

    def wait_for_save(self) -> Optional[dict[str, list[str]]]:
        """
        Block until all the save operations is done. This is called
        as the forward context exits to ensure that the async saving
        from save_kv_layer is complete before finishing the forward.

        This prevents overwrites of paged KV buffer before saving done.
        """
        if hasattr(self, "kv_role") and self.kv_role == "kv_consumer":
            return
        # request id -> succeed dumped blocks
        success_dumped_blocks: dict[str, list[str]] = {}

        def wait_for_tasks():
            for request_id, block_dump_tasks in self.dump_tasks.items():
                for block_id, dump_tasks in block_dump_tasks.items():
                    if any(self.connector.wait(task) != 0 for task in dump_tasks):
                        continue
                    success_dumped_blocks.setdefault(request_id, []).append(block_id)

        metadata = self._get_connector_metadata()
        assert isinstance(metadata, UCConnectorV1Metadata)
        if self.use_layerwise:
            wait_for_tasks()
            # clear dump_tasks for all request
            self.dump_tasks.clear()
            return success_dumped_blocks if success_dumped_blocks else None

        for request in metadata.requests:
            if not request.dump_blocks:
                continue

            storage_block_ids = [block[0] for block in request.dump_blocks]
            vllm_block_ids = [block[1] for block in request.dump_blocks]
            blocks_len = len(storage_block_ids)
            for layer_name, kv_layer in self.kv_caches.items():
                tensors, offsets = self.get_tensor_and_offset_layerwise(
                    vllm_block_ids, kv_layer, layer_name
                )
                for block_id, offset, tensor in zip(
                    storage_block_ids, offsets[:blocks_len], tensors[:blocks_len]
                ):
                    task = self.connector.dump([block_id], [offset], [tensor])
                    self.dump_tasks.setdefault(request.request_id, {}).setdefault(
                        block_id, []
                    ).append(task)
                if not self.is_mla:
                    for block_id, offset, tensor in zip(
                        storage_block_ids,
                        offsets[blocks_len:],
                        tensors[blocks_len:],
                    ):
                        task = self.connector.dump([block_id], [offset], [tensor])
                        self.dump_tasks.setdefault(request.request_id, {}).setdefault(
                            block_id, []
                        ).append(task)
        wait_for_tasks()
        self.dump_tasks.clear()
        return success_dumped_blocks if success_dumped_blocks else None

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """Get the finished recving and sending requests."""
        done_recving: set[str] = set()
        for req_id, tasks in self._need_load_reqs.items():
            if req_id in self._load_failed_reqs:
                continue
            unfinished_tasks = []
            for task in tasks:
                ret = self.connector.check(task)
                if ret == -1:
                    unfinished_tasks.append(task)
                    continue
                elif ret == 0 and self.connector.wait(task) == 0:
                    continue
                self._load_failed_reqs.add(req_id)
                break
            if not unfinished_tasks:
                done_recving.add(req_id)
            self._need_load_reqs[req_id] = unfinished_tasks

        # remove the finished requests
        for req_id in list(done_recving):
            self._need_load_reqs.pop(req_id, None)

        return None, done_recving

    # ==============================
    # Scheduler-side methods
    # ==============================
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
        # When the request is preempt req, need to commit succeed dumped blocks
        # to avoid duplicate invoking create/commit funcs. Only preempt reqs
        # whose succeed_dumped_blocks is non-empty need this check.
        if hasattr(request, "succeed_dumped_blocks") and request.succeed_dumped_blocks:
            self.connector.commit(request.succeed_dumped_blocks, True)
            request.succeed_dumped_blocks.clear()

        def md5(input) -> int:
            input_bytes = pickle.dumps(input, protocol=pickle.HIGHEST_PROTOCOL)
            md5_bytes = hashlib.md5(input_bytes).digest()
            return int.from_bytes(md5_bytes, byteorder="big")

        assert num_computed_tokens % self.block_size == 0
        block_hash_types = hash_request_tokens(md5, self.block_size, request)
        block_hashes: List[str] = [str(x.hash_value) for x in block_hash_types]
        if not block_hashes:
            logger.debug("Maybe tokens too short to load.")
            return 0, False

        # Calculate start position (exclude blocks already in HBM)
        start_position = num_computed_tokens // self.block_size

        block_operations = [BlockOperation.NONE] * len(block_hashes)

        remain_hashes = block_hashes[start_position:]
        if not remain_hashes:
            # All blocks are in HBM
            return 0, False

        lookup_results = self.connector.lookup(remain_hashes)

        # Find the longest continuous match from the beginning
        num_lookup_hits = 0
        for i, hit in enumerate(lookup_results):
            if hit:
                num_lookup_hits += 1
                block_operations[start_position + i] = BlockOperation.LOAD
            else:
                # TODO we will fix hole match later
                break
        logger.info(
            f"\nnum_total_blocks: {len(block_hashes)}\n"
            f"\nnum_lookup_hits on hbm: {start_position}\n"
            f"\nnum_lookup_hits on storage except hbm: {num_lookup_hits}\n"
        )

        # Load async when Decode instance need to load
        if hasattr(self, "kv_role") and self.kv_role == "kv_consumer":
            # Only trigger 1 asynchronous KV transfer per request.
            if (
                request.kv_transfer_params
                and request.kv_transfer_params["load_async"] == False
            ):
                return 0, False
            request.kv_transfer_params = request.kv_transfer_params or {}
            request.kv_transfer_params["load_async"] = False
            if num_lookup_hits > 0:
                self.request_block_infos[request.request_id] = RequestBlockInfo(
                    block_hashes=block_hashes,
                    block_operations=block_operations,
                    start_position=start_position,
                )
                self._need_load_reqs[request.request_id] = []
                return num_lookup_hits * self.block_size, True

        # Create blocks for the remaining (unmatched) blocks
        if num_lookup_hits < len(remain_hashes):
            remaining_hashes = remain_hashes[num_lookup_hits:]
            create_results = self.connector.create(remaining_hashes)
            logger.info(f"\ncreate_results on storage: {create_results}\n")
            for j, ret in enumerate(create_results):
                idx = num_lookup_hits + j
                block_operations[start_position + idx] = (
                    BlockOperation.DUMP if ret == 0 else BlockOperation.NONE
                )

        # When all the tokens are cached in ssd or hbm,
        # we need to recompute the last token. This if condition will be removed
        # once vLLM's scheduler provides a better solution in the future.
        if (num_lookup_hits + start_position) * self.block_size == len(
            request.all_token_ids
        ):
            num_lookup_hits -= 1
            block_operations[-1] = BlockOperation.NONE

        self.request_block_infos[request.request_id] = RequestBlockInfo(
            block_hashes=block_hashes,
            block_operations=block_operations,
            start_position=start_position,
        )

        return num_lookup_hits * self.block_size, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """
        Update KVConnector state after block allocation.
        """
        if request.request_id in self._need_load_reqs:
            local_block_ids = (
                blocks.get_unhashed_block_ids() if num_external_tokens > 0 else []
            )
            self._need_load_reqs[request.request_id] = local_block_ids

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
        meta = UCConnectorV1Metadata()

        for req_id, block_ids in self._need_load_reqs.items():
            block_info = self.request_block_infos.get(req_id)
            if block_info:
                load_blocks, dump_blocks = self._extract_blocks(block_ids, block_info)
            meta.requests.append(
                ReqMeta(
                    request_id=req_id,
                    load_blocks=load_blocks,
                    dump_blocks=dump_blocks,
                    load_async=True,
                )
            )
        self._need_load_reqs.clear()

        for new_req in scheduler_output.scheduled_new_reqs:
            req_id = new_req.req_id
            vllm_block_ids = new_req.block_ids[0]

            block_info = self.request_block_infos.get(req_id)
            if block_info:
                load_blocks, dump_blocks = self._extract_blocks(
                    vllm_block_ids, block_info
                )
                if load_blocks or dump_blocks:
                    meta.requests.append(
                        ReqMeta(
                            request_id=req_id,
                            load_blocks=load_blocks,
                            dump_blocks=dump_blocks,
                        )
                    )

        # Process cached requests using iterator
        cached_request_data = scheduler_output.scheduled_cached_reqs

        # Adapted for vllm 0.9.1, 0.9.2 and later versions
        def get_requests():
            # 0.9.1
            if isinstance(cached_request_data, list):
                return [
                    (
                        request_data.req_id,
                        request_data.new_block_ids,
                    )
                    for request_data in cached_request_data
                ]
            # >= 0.9.2
            else:
                return [
                    (
                        req_id,
                        cached_request_data.new_block_ids[i],
                    )
                    for i, req_id in enumerate(cached_request_data.req_ids)
                ]

        # When prompt tokens > max_num_batched_tokens, request of running requests may need to save
        for req_id, new_block_ids in get_requests():
            block_info = self.request_block_infos.get(req_id)
            if block_info:
                load_blocks, dump_blocks = self._extract_blocks(
                    new_block_ids[0], block_info
                )
                if load_blocks or dump_blocks:
                    meta.requests.append(
                        ReqMeta(
                            request_id=req_id,
                            load_blocks=load_blocks,
                            dump_blocks=dump_blocks,
                        )
                    )

        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        block_info = self.request_block_infos.pop(request.request_id, None)
        if hasattr(request, "succeed_dumped_blocks") and request.succeed_dumped_blocks:
            logger.debug(f"commit {request.succeed_dumped_blocks} to True.")
            self.connector.commit(request.succeed_dumped_blocks, True)
        if block_info is not None:
            cancel_blocks = [
                block_info.block_hashes[i]
                for i, op in enumerate(block_info.block_operations)
                if op == BlockOperation.DUMP
                and hasattr(request, "succeed_dumped_blocks")
                and block_info.block_hashes[i] not in request.succeed_dumped_blocks
            ]
            if cancel_blocks:
                logger.warning(f"commit {cancel_blocks} to False.")
                self.connector.commit(cancel_blocks, False)
        return False, None

    def _extract_blocks(
        self, vllm_block_ids: list[int], block_info: RequestBlockInfo
    ) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
        """
        Extract blocks that need load and dump, block_info.start_position
        is the next block position to process, only return blocks that need
        processing, NONE blocks are ignored.
        """
        start_pos = block_info.start_position

        if start_pos >= len(block_info.block_operations):
            return [], []

        process_length = min(
            len(block_info.block_operations) - start_pos, len(vllm_block_ids)
        )
        ops = block_info.block_operations[start_pos : start_pos + process_length]
        hashes = block_info.block_hashes[start_pos : start_pos + process_length]
        vllm_ids = vllm_block_ids[:process_length]

        load_blocks = []
        dump_blocks = []
        for op, hash, vllm_id in zip(ops, hashes, vllm_ids):
            if op == BlockOperation.LOAD:
                load_blocks.append((hash, vllm_id))
            elif op == BlockOperation.DUMP:
                dump_blocks.append((hash, vllm_id))

        block_info.start_position += process_length
        return load_blocks, dump_blocks

    def get_block_ids_with_load_errors(self) -> set[int]:
        invalid_block_ids: set[int] = set()
        for req_id in self._load_failed_reqs:
            if req_id in self._load_req_to_blocks:
                invalid_block_ids.update(self._load_req_to_blocks[req_id])
        return invalid_block_ids

    @staticmethod
    def _extract_layer_index(layer_name: str) -> Optional[int]:
        """
        Extract the layer index from the layer name.
        """
        for chunk in layer_name.split("."):
            if chunk.isdigit():
                return int(chunk)
        return None
