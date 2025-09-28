import hashlib
import math
import pickle
from dataclasses import dataclass
from functools import cache
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from numpy.typing import NDArray
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer import get_kv_transfer_group
from vllm.forward_context import ForwardContext
from vllm.sequence import SequenceStage
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.request import Request

from ucm.integration.vllm.ucm_sparse.base import (
    INVALID_SLOT,
    UcmSparseBase,
    UcmSparseMetadata,
    UcmSparseRole,
)
from ucm.store.connector.ucmstore import Task, UcmKVStoreBase
from ucm.ucm_sparse.retrieval import retrieval_backend
from ucm.ucm_sparse.retrieval.retrieval_worker import RetrievalWorker

ReqType = Union[str, int]
HashType = Union[str, int]

data = None


class ReprePool:
    def __init__(self, num_slots):
        self.free_slots = set(range(num_slots))
        self.allocated = set()

    def allocate(self, num_new_slots):
        assert len(self.free_slots) >= num_new_slots, "Not enough free slots"
        allocated = list(self.free_slots)[:num_new_slots]
        self.free_slots.difference_update(allocated)
        self.allocated.update(allocated)
        return allocated

    def free(self, slots):
        self.free_slots.update(slots)
        self.allocated.difference_update(slots)


@dataclass
class ReqMeta:
    request_id: ReqType
    index_in_batch: int
    num_scheduled_tokens: int
    num_computed_tokens: int
    vllm_block_ids: list[int]
    query_start_loc: int
    prompt_token_ids: list[int]
    output_token_ids: list[int]

    @property
    def step(self) -> int:
        return self.num_output_tokens

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def stage(self) -> SequenceStage:
        return (
            SequenceStage.DECODE
            if self.num_output_tokens > 0
            else SequenceStage.PREFILL
        )

    @property
    def is_last_chunk(self) -> bool:
        return (
            self.num_computed_tokens + self.num_scheduled_tokens
            >= self.num_prompt_tokens
        )


@dataclass
class ESASparseMetaData(UcmSparseMetadata):
    requests: list[ReqMeta]
    finished_req_ids: List[ReqType]

    def __init__(self):
        self.requests = []
        self.finished_req_ids = []

    def add_request(
        self,
        request_id: ReqType,
        index_in_batch: int,
        num_scheduled_tokens: int,
        num_computed_tokens: int,
        vllm_block_ids: list[int],
        query_start_loc: int,
        prompt_token_ids: list[int],
        output_token_ids: list[int],
    ) -> None:

        meta = ReqMeta(
            request_id=request_id,
            index_in_batch=index_in_batch,
            num_scheduled_tokens=num_scheduled_tokens,
            num_computed_tokens=num_computed_tokens,
            vllm_block_ids=vllm_block_ids,
            query_start_loc=query_start_loc,
            prompt_token_ids=prompt_token_ids,
            output_token_ids=output_token_ids,
        )
        self.requests.append(meta)


@cache
def get_offset(block_shape, rank, tp_size, precision, layer_id, is_v, is_mla) -> int:
    block_size, num_key_heads_per_tp, head_size = block_shape
    k_min_data_block_size = block_size * num_key_heads_per_tp * head_size * precision
    v_min_data_block_size = k_min_data_block_size if not is_mla else 0
    layer_size = (k_min_data_block_size + v_min_data_block_size) * tp_size
    if is_mla:
        k_offset = layer_size * layer_id
    else:
        k_offset = layer_size * layer_id + layer_size // tp_size * rank
    v_offset = k_offset + k_min_data_block_size
    return v_offset if is_v else k_offset


@cache
def md5(input) -> int:
    input_bytes = pickle.dumps(input, protocol=pickle.HIGHEST_PROTOCOL)
    md5_bytes = hashlib.md5(input_bytes).digest()
    return int.from_bytes(md5_bytes, byteorder="big")


@cache
def block_hash_func(parent_block_hash, curr_block_token_ids):
    if not parent_block_hash:
        parent_block_hash = md5("UCMHASHSEED")
    curr_block_token_ids_tuple = tuple(curr_block_token_ids)
    return md5((parent_block_hash, curr_block_token_ids_tuple))


def task_hash_func(block_ids, store_type, tensor_type):
    return hash((tuple(block_ids), store_type, tensor_type))


class ReqStatePerLayer:
    # handle single request per layer

    def __init__(
        self,
        req_meta: ReqMeta,
        layer_name: str,
        rank: int,
        tp_size: int,
        store_instance: UcmKVStoreBase,
        vllm_config: VllmConfig,
        retrieval_worker: Optional[RetrievalWorker] = None,
        repre_pool: Optional[ReprePool] = None,
    ):
        self.layer_name = layer_name
        self.layer_id = int(layer_name.split(".")[2])
        self.slots = []
        self.slots_to_relative_indexes = {}
        self.repre_pool: ReprePool | None = repre_pool
        self.store_instance = store_instance
        self.retrieval_worker: Optional[RetrievalWorker] = retrieval_worker
        self.retrieval_task = None
        self.req_meta = req_meta
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self.k_cache = None
        self.v_cache = None
        self.rank = rank
        self.tp_size = tp_size
        self.tasks: Dict[str, Task] = {}
        self.esa_cfg = vllm_config.kv_transfer_config.kv_connector_extra_config[
            "ucm_sparse_config"
        ]["ESA"]
        self.indexes: Optional[NDArray[np.int64]] = None
        self.block_hashes = None
        self.pre_topk_block_hashes: Dict[int, str] = {}
        self.sparse_range: int = 0
        self.init_static_flag = False

        self.num_key_heads = vllm_config.model_config.get_num_kv_heads(
            vllm_config.parallel_config
        )
        self.head_size = vllm_config.model_config.get_head_size()
        self.sparse_range = self.get_sparse_prefill_range()

    def set_block_hashes(self, token_ids):
        if self.block_hashes is not None:
            return
        self.block_hashes = []
        parent_block_hash_value = None
        for start in range(0, len(token_ids), self.block_size):
            end = start + self.block_size
            block_token_ids = token_ids[start:end]
            if len(block_token_ids) < self.block_size:
                break
            curr_block_token_ids_tuple = tuple(block_token_ids)
            block_hash = block_hash_func(
                parent_block_hash_value, curr_block_token_ids_tuple
            )
            self.block_hashes.append(str(block_hash))
            parent_block_hash_value = block_hash

    def update_meta(self, req_meta: ReqMeta):
        self.req_meta = req_meta

    def launch_transfer_task(self, transfer_type, block_hashes, vllm_block_ids):
        fn = getattr(self.store_instance, transfer_type)
        length = len(block_hashes)
        block_shape = (self.block_size, self.num_key_heads, self.head_size)
        precision = self.k_cache.storage().element_size()
        # TODO: consider is_mla here
        is_mla = False

        block_shape = tuple(block_shape)
        offsets_k = [
            get_offset(
                block_shape,
                self.rank,
                self.tp_size,
                precision,
                self.layer_id,
                is_v=False,
                is_mla=is_mla,
            )
        ] * length
        offsets_v = [
            get_offset(
                block_shape,
                self.rank,
                self.tp_size,
                precision,
                self.layer_id,
                is_v=True,
                is_mla=is_mla,
            )
        ] * length

        key_src_tensors = [self.k_cache[id_] for id_ in vllm_block_ids]
        value_src_tensors = [self.v_cache[id_] for id_ in vllm_block_ids]

        task_k = fn(block_hashes, offsets_k, key_src_tensors)
        task_v = fn(block_hashes, offsets_v, value_src_tensors)

        task_k_hash = task_hash_func(block_hashes, transfer_type, "key")
        self.tasks[task_k_hash] = task_k
        task_v_hash = task_hash_func(block_hashes, transfer_type, "value")
        self.tasks[task_v_hash] = task_v

    def extract_block_repre(self, vllm_block_ids):
        return self.k_cache[vllm_block_ids].mean(1)

    def maybe_register_static_data(self, forward_context: ForwardContext):
        if self.init_static_flag:
            return
        attn = forward_context.no_compile_layers[self.layer_name]
        kv_cache = attn.kv_cache[forward_context.virtual_engine]
        # TODO not mla
        self.k_cache = kv_cache[0]
        self.v_cache = kv_cache[1]
        self.set_block_hashes(self.req_meta.prompt_token_ids)
        self.init_static_flag = True

    def wait_transfer_task_done(self):
        assert len(self.tasks) > 0
        for task_hash, task in self.tasks.items():
            # TODO: handle exceptions
            ret = self.store_instance.wait(task)
        self.tasks.clear()  # reset

    def start_retrieval(self, batch_query, forward_context):
        query_start_loc = self.req_meta.query_start_loc
        query_len = self.req_meta.num_scheduled_tokens
        query = batch_query[query_start_loc : query_start_loc + query_len]
        ntokens, num_q_heads, _ = query.shape
        if num_q_heads > self.num_key_heads:
            query = query.view(ntokens, self.num_key_heads, -1, self.head_size)
            query = query.mean(2)
        elif num_q_heads < self.num_key_heads:
            query = torch.repeat_interleave(query, self.num_key_heads // num_q_heads, 1)
        query_flat = query.reshape(query.shape[0], -1)
        top_k = int(self.sparse_range * self.esa_cfg["sparse_ratio"])
        indexes = [self.slots]
        self.retrieval_task = self.retrieval_worker.submit(
            query_flat, topk=top_k, indexes=indexes
        )

    def wait_retrieval_and_start_load(self):
        self.retrieval_worker.wait(self.retrieval_task)
        result = self.retrieval_worker.get_result(self.retrieval_task)
        choosed_slots = result["indices"][0]
        rel_block_ids = [self.slots_to_relative_indexes[int(e)] for e in choosed_slots]
        block_hashes = [self.block_hashes[id_] for id_ in rel_block_ids]
        top_k = int(self.sparse_range * self.esa_cfg["sparse_ratio"])
        sparse_vllm_block_ids = self.req_meta.vllm_block_ids[:top_k]

        # load delta
        diff_vllm_block_ids = set(sparse_vllm_block_ids)
        diff_block_hashes = set(block_hashes)
        if len(self.pre_topk_block_hashes) == 0:
            self.pre_topk_block_hashes = {
                blk_id: blk_hash
                for (blk_id, blk_hash) in zip(sparse_vllm_block_ids, block_hashes)
            }
        else:
            matched = {}
            for k in sparse_vllm_block_ids:
                if (
                    k in self.pre_topk_block_hashes
                    and self.pre_topk_block_hashes[k] in diff_block_hashes
                ):
                    matched[k] = self.pre_topk_block_hashes[k]
                    diff_vllm_block_ids.remove(k)
                    diff_block_hashes.remove(matched[k])
            self.pre_topk_block_hashes = matched
            for diff_blk_id, diff_blk_hash in zip(
                diff_vllm_block_ids, diff_block_hashes
            ):
                self.pre_topk_block_hashes[diff_blk_id] = diff_blk_hash

        self.launch_transfer_task(
            "load", list(diff_block_hashes), list(diff_vllm_block_ids)
        )
        self.retrieval_task = None

    def get_sparse_prefill_range(self):
        if (self.req_meta.num_prompt_tokens % self.block_size) == 0:
            sparse_range = (
                self.req_meta.num_prompt_tokens // self.block_size
                - self.esa_cfg["local_window_sz"]
            )
        else:
            sparse_range = math.floor(
                self.req_meta.num_prompt_tokens / self.block_size
            ) - (self.esa_cfg["local_window_sz"] - 1)
        return sparse_range

    def block_repre_data(self):
        vllm_block_ids = self.req_meta.vllm_block_ids
        vllm_block_ids_dump = vllm_block_ids[: self.sparse_range]
        repre = self.extract_block_repre(vllm_block_ids_dump)
        repre_flat = repre.reshape(repre.shape[0], -1)
        new_slots = self.repre_pool.allocate(self.sparse_range)
        og_len = len(self.slots)
        for i, slot in enumerate(new_slots):
            self.slots_to_relative_indexes[slot] = og_len + i
        self.slots.extend(new_slots)
        vals = repre_flat.to("cpu", non_blocking=True, dtype=torch.float32)
        data[self.layer_id][new_slots] = vals

    def attention_begin(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        forward_context: ForwardContext,
    ) -> None:
        self.maybe_register_static_data(forward_context)
        if self.req_meta.step % self.esa_cfg["retrieval_stride"] == 1:
            if self.req_meta.step == 1:
                self.start_retrieval(query, forward_context)
                self.wait_retrieval_and_start_load()
            self.wait_transfer_task_done()

    def attention_finished(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_output: torch.Tensor,
        forward_context: ForwardContext,
    ) -> None:
        should_save = (
            self.req_meta.stage == SequenceStage.PREFILL and self.req_meta.is_last_chunk
        )
        if should_save:
            self.block_repre_data()
        else:
            if self.req_meta.step == 0:
                return
            if self.req_meta.step % self.esa_cfg["retrieval_stride"] == 2:
                self.start_retrieval(query, forward_context)
            if self.req_meta.step % self.esa_cfg["retrieval_stride"] == 0:
                self.wait_retrieval_and_start_load()


class ESA(UcmSparseBase):
    # handle batch
    def __init__(self, vllm_config: VllmConfig, role: UcmSparseRole):
        super().__init__(vllm_config, role)
        self.req_states: dict[str, List[ReqStatePerLayer]] = {}
        self.rank = vllm_config.parallel_config.rank
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        if role == UcmSparseRole.WORKER:
            self.connector = get_kv_transfer_group().connector
        else:
            self.connector = None
        self.esa_cfg = vllm_config.kv_transfer_config.kv_connector_extra_config[
            "ucm_sparse_config"
        ]["ESA"]
        self.total_num_hidden_layers = (
            vllm_config.model_config.hf_config.num_hidden_layers
        )

        global data

        if data is None:
            parallel_config = vllm_config.parallel_config
            num_slots = (
                vllm_config.model_config.max_model_len
                * vllm_config.scheduler_config.max_num_seqs
                // vllm_config.cache_config.block_size
            )
            dim = (
                vllm_config.model_config.get_num_kv_heads(parallel_config)
                * vllm_config.model_config.get_head_size()
            )
            data = [
                torch.empty((num_slots, dim), dtype=torch.float32)
                for _ in range(self.total_num_hidden_layers)
            ]
            self.layer_pools: list[ReprePool] = [
                ReprePool(num_slots) for _ in range(self.total_num_hidden_layers)
            ]

        self.retrieval_workers: List[RetrievalWorker] = []
        for i in range(self.total_num_hidden_layers):
            backend_src = data[i]
            backend = retrieval_backend.RetrievalWorkerBackend(backend_src)
            self.retrieval_workers.append(RetrievalWorker(backend))

    def attention_begin(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_name: str,
        forward_context: ForwardContext,
    ) -> None:
        for req_meta in self._sparse_metadata.requests:
            layer_id = int(layer_name.split(".")[2])
            if req_meta.request_id not in self.req_states:
                if self.req_states.get(req_meta.request_id) is None:
                    self.req_states[req_meta.request_id] = [
                        None
                    ] * self.total_num_hidden_layers
            if self.req_states[req_meta.request_id][layer_id] is None:
                self.req_states[req_meta.request_id][layer_id] = ReqStatePerLayer(
                    req_meta,
                    layer_name,
                    self.rank,
                    self.tp_size,
                    self.connector,
                    self._vllm_config,
                    self.retrieval_workers[layer_id],
                    self.layer_pools[layer_id],
                )
            req_state = self.req_states[req_meta.request_id][layer_id]
            req_state.update_meta(req_meta)
            req_state.attention_begin(query, key, value, forward_context)

    def attention_finished(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_output: torch.Tensor,
        layer_name: str,
        forward_context: ForwardContext,
    ) -> None:
        for req_meta in self._sparse_metadata.requests:
            layer_id = int(layer_name.split(".")[2])
            if req_meta.request_id not in self.req_states:
                if self.req_states.get(req_meta.request_id) is None:
                    self.req_states[req_meta.request_id] = [
                        None
                    ] * self.total_num_hidden_layers
            if self.req_states[req_meta.request_id][layer_id] is None:
                self.req_states[req_meta.request_id][layer_id] = ReqStatePerLayer(
                    req_meta,
                    layer_name,
                    self.rank,
                    self.tp_size,
                    self.connector,
                    self._vllm_config,
                    self.retrieval_workers[layer_id],
                    self.layer_pools[layer_id],
                )
            req_state = self.req_states[req_meta.request_id][layer_id]
            req_state.update_meta(req_meta)
            req_state.attention_finished(
                query, key, value, attn_output, forward_context
            )

    def build_sparse_meta(
        self, scheduler_output, requests, input_batch, attn_metadata
    ) -> UcmSparseMetadata:
        sparse_meta = ESASparseMetaData()
        for (
            req_id,
            num_scheduled_tokens,
        ) in scheduler_output.num_scheduled_tokens.items():
            req_state = requests[req_id]
            if (
                len(req_state.prompt_token_ids)
                <= self._vllm_config.cache_config.block_size
            ):
                return

            if isinstance(attn_metadata, dict):
                attn_metadata = next(iter(attn_metadata.values()))
            sparse_meta.add_request(
                req_id,
                input_batch.req_id_to_index[req_id],
                num_scheduled_tokens,
                req_state.num_computed_tokens,
                req_state.block_ids[0],
                attn_metadata.query_start_loc[input_batch.req_id_to_index[req_id]],
                req_state.prompt_token_ids,
                req_state.output_token_ids,
            )
        self._sparse_metadata = sparse_meta

    def request_begin(self, request_id: ReqType, prompt_token_ids: List[int]):
        pass

    def request_finished_in_worker(self, request_id: ReqType):
        for layer_state in self.req_states[request_id]:
            layer_state.repre_pool.free(layer_state.slots)
        del self.req_states[request_id]

    def estimate_num_slots_sparsed(self, request: Request) -> int:
        if (
            request.num_output_tokens == 0
            or request.num_prompt_tokens
            < self._vllm_config.cache_config.block_size * self.esa_cfg["min_blocks"]
        ):
            return INVALID_SLOT
        prompt_len = request.num_prompt_tokens
        output_len = request.num_output_tokens
        block_size = self._vllm_config.cache_config.block_size
        if (flaw := prompt_len % block_size) == 0:
            sparse_range = prompt_len // block_size - self.esa_cfg["local_window_sz"]
            local_window = block_size * self.esa_cfg["local_window_sz"] + output_len
        else:
            sparse_range = math.floor(prompt_len / block_size) - (
                self.esa_cfg["local_window_sz"] - 1
            )
            local_window = (
                flaw + block_size * (self.esa_cfg["local_window_sz"] - 1) + output_len
            )
        return (
            int(sparse_range * self.esa_cfg["sparse_ratio"]) * block_size + local_window
        )

    def allocate_slots(
        self, request, num_slots_sparsed, coordinator, block_pool, kv_cache_groups
    ):
        block_size = self._vllm_config.cache_config.block_size
        num_blocks_need = math.ceil(num_slots_sparsed / block_size)
        allocated_blocks = coordinator.get_blocks(request.request_id)[0]
        returned_blocks = []
        kept_blocks = []
        num_blocks_original = len(allocated_blocks)
        for i, block in enumerate(allocated_blocks):
            if i >= num_blocks_original - num_blocks_need:
                kept_blocks.append(block)
            else:
                returned_blocks.append(block)
            block_pool._maybe_evict_cached_block(block)
        block_pool.free_blocks(returned_blocks)

        coordinator.single_type_managers[0].req_to_blocks[
            request.request_id
        ] = kept_blocks

        new_computed_block_list = tuple([] for _ in range(len(kv_cache_groups)))
        num_blocks_to_allocate = coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=num_slots_sparsed,
            new_computed_blocks=new_computed_block_list,
        )
        if num_blocks_to_allocate > block_pool.get_num_free_blocks():
            return None
        coordinator.allocate_new_blocks(request.request_id, num_slots_sparsed)
        return KVCacheBlocks(tuple([kept_blocks]))
