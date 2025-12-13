import hashlib
import math
import pickle
from collections import defaultdict
from dataclasses import dataclass
from functools import cache
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from numpy.typing import NDArray
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer import get_kv_transfer_group
from vllm.forward_context import ForwardContext
from vllm.sequence import SequenceStage
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.request import Request, RequestStatus

from ucm.integration.vllm.ucm_connector import RequestHasher
from ucm.sparse.base import (
    INVALID_SLOT,
    UcmSparseBase,
    UcmSparseMetadata,
    UcmSparseRole,
)
from ucm.sparse.esa.retrieval import retrieval_backend
from ucm.sparse.esa.retrieval.retrieval_worker import RetrievalWorker
from ucm.sparse.kvstar.utils import get_bind_cpus_for_rank
from ucm.store.ucmstore import Task, UcmKVStoreBase
from ucm.utils import Config

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
    is_preempt: bool
    ucm_block_hashes: list[str]

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def num_tokens(self) -> int:
        return self.num_prompt_tokens + self.num_output_tokens

    @property
    def is_last_chunk(self) -> bool:
        # NOTE: both decode and last chunk-prefill meet `self.num_computed_tokens + self.num_scheduled_tokens >= self.num_tokens`
        return self.num_computed_tokens + self.num_scheduled_tokens >= self.num_tokens


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
        is_preempt: bool,
        ucm_block_hashes: list[str],
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
            is_preempt=is_preempt,
            ucm_block_hashes=ucm_block_hashes,
        )
        self.requests.append(meta)


@cache
def get_offset(block_shape, rank, tp_size, precision, layer_id, is_v, is_mla) -> int:
    block_size, num_key_heads_per_tp, head_size = block_shape
    k_min_data_block_size = block_size * num_key_heads_per_tp * head_size * precision
    v_min_data_block_size = k_min_data_block_size if not is_mla else 0
    layer_size = (k_min_data_block_size + v_min_data_block_size) * (
        tp_size if not is_mla else 1
    )
    if is_mla:
        k_offset = layer_size * layer_id
    else:
        k_offset = layer_size * layer_id + layer_size // tp_size * rank
    v_offset = k_offset + k_min_data_block_size
    return v_offset if is_v else k_offset


@cache
def get_sparse_range(init_window_sz, local_window_sz, prompt_len, block_size):
    num_blocks_upper_bound = math.ceil(prompt_len / block_size)
    sparse_range = num_blocks_upper_bound - init_window_sz - local_window_sz
    return sparse_range


@cache
def compute_parent_block_hash(model_name, world_size, dtype, seed_rank=0) -> int:
    meta = f"{model_name}:{world_size}:{dtype}:{seed_rank}"
    meta_bytes = meta.encode("utf-8")
    h_seed = hashlib.md5(meta_bytes + b"UCM_HASH_SEED").digest()
    return int.from_bytes(h_seed, byteorder="big")


@cache
def compute_layer_offset(
    block_data_size: int,
    layer_id: int,
    is_v: bool,
    is_mla: bool,
) -> int:
    layer_data_size = block_data_size if is_mla else block_data_size * 2

    k_offset = layer_data_size * layer_id

    if is_mla:
        return k_offset

    v_offset = k_offset + block_data_size
    return v_offset if is_v else k_offset


def task_hash_func(block_ids, store_type, tensor_type):
    return hash((tuple(block_ids), store_type, tensor_type))


def diff_two_map(map1: dict, map2: dict):
    keys2 = map2.keys()
    values2 = map2.values()
    keys2_set = set(keys2)
    values2_set = set(values2)
    diff_map = {}
    updated_map = {}
    for k1, v1 in map1.items():
        if k1 in keys2 and v1 in values2:
            updated_map[k1] = v1
            keys2_set.remove(k1)
            values2_set.remove(v1)
    for k2, v2 in zip(keys2_set, values2_set):
        diff_map[k2] = v2
        updated_map[k2] = v2
    return updated_map, diff_map


class ReqStatePerLayer:
    # handle single request per layer
    def __init__(
        self,
        layer_name: str,
        rank: int,
        tp_size: int,
        store_instance: UcmKVStoreBase,
        vllm_config: VllmConfig,
        retrieval_worker: Optional[RetrievalWorker] = None,
        repre_pool: Optional[ReprePool] = None,
        esa_cfg: Optional[Dict[str, Any]] = None,
    ):
        self.layer_name = layer_name
        self.layer_id = int(layer_name.split(".")[2])
        self.slots = []
        self.slots_to_relative_indexes = {}
        self.repre_pool: ReprePool | None = repre_pool
        self.store_instance = store_instance
        self.retrieval_worker: Optional[RetrievalWorker] = retrieval_worker
        self.retrieval_task = None
        self.req_meta = None
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self.k_cache = None
        self.v_cache = None
        self.rank = rank
        self.tp_size = tp_size
        self.tasks: Dict[str, Task] = {}
        self.esa_cfg = esa_cfg
        self.indexes: Optional[NDArray[np.int64]] = None
        self.block_hashes = None
        self.pre_topk_block_hashes: Dict[int, str] = {}
        self.sparse_range: int = 0
        self.init_static_flag = False
        self.init_window = None
        self.local_window = None

        self.num_key_heads = vllm_config.model_config.get_num_kv_heads(
            vllm_config.parallel_config
        )
        self.head_size = vllm_config.model_config.get_head_size()
        self.is_mla = self.vllm_config.model_config.is_deepseek_mla
        self.step = 0

    def update_meta(self, req_meta: ReqMeta):
        self.req_meta = req_meta

    def launch_transfer_task(self, transfer_type, block_hashes, vllm_block_ids):
        fn = getattr(self.store_instance, transfer_type)
        length = len(block_hashes)
        precision = self.vllm_config.model_config.dtype.itemsize
        block_data_size = self.k_cache[0].numel() * precision

        offset_k = compute_layer_offset(
            block_data_size,
            self.layer_id,
            is_v=False,
            is_mla=self.is_mla,
        )
        offsets_k = [offset_k] * length

        key_src_tensors = [self.k_cache[id_] for id_ in vllm_block_ids]
        task_k = fn(block_hashes, offsets_k, key_src_tensors)
        task_k_hash = task_hash_func(block_hashes, transfer_type, "key")
        self.tasks[task_k_hash] = task_k

        if not self.is_mla:
            offset_v = compute_layer_offset(
                block_data_size,
                self.layer_id,
                is_v=True,
                is_mla=self.is_mla,
            )
            offsets_v = [offset_v] * length
            value_src_tensors = [self.v_cache[id_] for id_ in vllm_block_ids]
            task_v = fn(block_hashes, offsets_v, value_src_tensors)
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
        if self.is_mla:
            self.k_cache = kv_cache
        else:
            self.k_cache = kv_cache[0]
            self.v_cache = kv_cache[1]
        self.block_hashes = self.req_meta.ucm_block_hashes
        self.init_static_flag = True

    def wait_transfer_task_done(self):
        # assert len(self.tasks) > 0
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
            query = query.view(
                ntokens,
                self.num_key_heads,
                num_q_heads // self.num_key_heads,
                self.head_size,
            )
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
        vllm_block_ids = self.req_meta.vllm_block_ids[
            self.esa_cfg["init_window_sz"] : self.esa_cfg["init_window_sz"] + top_k
        ]
        ## 1. load delta
        target_map = {
            b_id: b_hash for b_id, b_hash in zip(vllm_block_ids, block_hashes)
        }
        self.pre_topk_block_hashes, diff_blocks = diff_two_map(
            self.pre_topk_block_hashes, target_map
        )
        if diff_blocks:
            self.launch_transfer_task(
                "load", list(diff_blocks.values()), list(diff_blocks.keys())
            )

        ## 2. load all
        # self.launch_transfer_task(
        #     "load", block_hashes, vllm_block_ids
        # )

        self.retrieval_task = None

    def block_repre_data(self):
        self.sparse_range = get_sparse_range(
            self.esa_cfg["init_window_sz"],
            self.esa_cfg["local_window_sz"],
            self.req_meta.num_prompt_tokens,
            self.block_size,
        )
        vllm_block_ids = self.req_meta.vllm_block_ids
        vllm_block_ids_dump = vllm_block_ids[
            self.esa_cfg["init_window_sz"] : self.esa_cfg["init_window_sz"]
            + self.sparse_range
        ]
        repre = self.extract_block_repre(vllm_block_ids_dump)
        repre_flat = repre.reshape(repre.shape[0], -1)
        new_slots = self.repre_pool.allocate(self.sparse_range)
        og_len = len(self.slots)
        for i, slot in enumerate(new_slots):
            self.slots_to_relative_indexes[slot] = og_len + i
        self.slots.extend(new_slots)
        vals = repre_flat.to("cpu", dtype=torch.float32)
        data[self.layer_id][new_slots] = vals
        # NOTE: in Preemption, local_window_start != -self.esa_cfg['local_window_sz']
        local_window_start = self.esa_cfg["init_window_sz"] + self.sparse_range

        if not self.is_mla:
            self.init_window = (
                self.k_cache[vllm_block_ids[: self.esa_cfg["init_window_sz"]]].clone(),
                self.v_cache[vllm_block_ids[: self.esa_cfg["init_window_sz"]]].clone(),
            )
            self.local_window = (
                self.k_cache[vllm_block_ids[local_window_start:]].clone(),
                self.v_cache[vllm_block_ids[local_window_start:]].clone(),
            )
        else:
            self.init_window = self.k_cache[
                vllm_block_ids[: self.esa_cfg["init_window_sz"]]
            ].clone()
            self.local_window = self.k_cache[
                vllm_block_ids[local_window_start:]
            ].clone()

    def attention_begin(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        forward_context: ForwardContext,
    ) -> None:
        self.maybe_register_static_data(forward_context)
        if self.step % self.esa_cfg["retrieval_stride"] == 1:
            if self.step == 1:
                vllm_block_ids = self.req_meta.vllm_block_ids
                # NOTE: in Preemption, local_window_start != -self.esa_cfg['local_window_sz']
                if not self.is_mla:
                    local_window_sz = self.local_window[0].shape[0]
                    self.k_cache[vllm_block_ids[: self.esa_cfg["init_window_sz"]]] = (
                        self.init_window[0]
                    )
                    self.v_cache[vllm_block_ids[: self.esa_cfg["init_window_sz"]]] = (
                        self.init_window[1]
                    )
                    self.k_cache[vllm_block_ids[-local_window_sz:]] = self.local_window[
                        0
                    ]
                    self.v_cache[vllm_block_ids[-local_window_sz:]] = self.local_window[
                        1
                    ]
                else:
                    local_window_sz = self.local_window.shape[0]
                    self.k_cache[vllm_block_ids[: self.esa_cfg["init_window_sz"]]] = (
                        self.init_window
                    )
                    self.k_cache[vllm_block_ids[-local_window_sz:]] = self.local_window
                self.start_retrieval(query, forward_context)
                self.wait_retrieval_and_start_load()
            if len(self.tasks) > 0:
                self.wait_transfer_task_done()

    def attention_finished(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_output: torch.Tensor,
        forward_context: ForwardContext,
    ) -> None:
        if self.step == 0:
            if self.req_meta.is_last_chunk:
                self.block_repre_data()
                self.step += 1
        else:
            if self.step % self.esa_cfg["retrieval_stride"] == 2:
                self.start_retrieval(query, forward_context)
            if self.step % self.esa_cfg["retrieval_stride"] == 0:
                self.wait_retrieval_and_start_load()
            self.step += 1


class ESA(UcmSparseBase):
    # handle batch
    def __init__(self, vllm_config: VllmConfig, role: UcmSparseRole):
        super().__init__(vllm_config, role)
        self.req_states: dict[str, List[ReqStatePerLayer]] = {}
        self.rank = vllm_config.parallel_config.rank
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        if role == UcmSparseRole.WORKER:
            self.connector = get_kv_transfer_group().connector.store
        else:
            self.connector = None
        self.esa_cfg = (
            Config(vllm_config.kv_transfer_config)
            .get_config()
            .get("ucm_sparse_config")
            .get("ESA")
        )
        self.total_num_hidden_layers = (
            vllm_config.model_config.hf_config.num_hidden_layers
        )
        self.is_mla = vllm_config.model_config.is_deepseek_mla
        self._sparse_metadata_prefill: ESASparseMetaData = ESASparseMetaData()
        self._sparse_metadata_decode: ESASparseMetaData = ESASparseMetaData()
        self._sparse_metadata: ESASparseMetaData = ESASparseMetaData()
        self.request_hasher = RequestHasher(vllm_config, 0)
        self.block_size = vllm_config.cache_config.block_size
        self.block_hashes: dict[int, dict[int, list[str]]] = {}
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

        self.local_tp_rank = vllm_config.parallel_config.rank
        self.total_tp_size = vllm_config.parallel_config.tensor_parallel_size
        ratio = 0.75

        bind_info_list, alloc_numa_ids = get_bind_cpus_for_rank(
            self.total_tp_size, self.local_tp_rank, ratio=ratio
        )

        bind_info_dict = defaultdict(list)
        for item in bind_info_list:
            bind_info_dict[item[1]].append(item[0])
        bind_info_dict = dict(bind_info_dict)

        self.retrieval_workers: List[RetrievalWorker] = []
        for i in range(self.total_num_hidden_layers):
            backend_src = data[i]
            backend = retrieval_backend.RetrievalWorkerBackend(
                backend_src, bind_info_dict
            )
            self.retrieval_workers.append(RetrievalWorker(backend))

        self.preempt_req_output_tokens: Dict[ReqType, int] = {}

    def get_or_create_layerwise_req_state(self, req_meta, layer_name):
        layer_id = int(layer_name.split(".")[2])
        if req_meta.is_preempt:
            layer_state = self.req_states[req_meta.request_id][layer_id]
            layer_state.repre_pool.free(layer_state.slots)
            self.req_states[req_meta.request_id][layer_id] = None
        if req_meta.request_id not in self.req_states:
            if self.req_states.get(req_meta.request_id) is None:
                self.req_states[req_meta.request_id] = [
                    None
                ] * self.total_num_hidden_layers
        if self.req_states[req_meta.request_id][layer_id] is None:
            self.req_states[req_meta.request_id][layer_id] = ReqStatePerLayer(
                layer_name,
                self.rank,
                self.tp_size,
                self.connector,
                self._vllm_config,
                self.retrieval_workers[layer_id],
                self.layer_pools[layer_id],
                self.esa_cfg,
            )
        return self.req_states[req_meta.request_id][layer_id]

    def create_req_state_attention_begin(
        self, req_meta, layer_name, query, key, value, forward_context
    ):
        req_state = self.get_or_create_layerwise_req_state(req_meta, layer_name)
        req_state.update_meta(req_meta)
        req_state.attention_begin(query, key, value, forward_context)

    def attention_begin(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_name: str,
        forward_context: ForwardContext,
        output: Optional[torch.Tensor] = None,
        phase: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.is_mla:
            for req_meta in self._sparse_metadata.requests:
                self.create_req_state_attention_begin(
                    req_meta, layer_name, query, key, value, forward_context
                )
        else:
            if phase == "prefill":
                for req_meta in self._sparse_metadata_prefill.requests:
                    self.create_req_state_attention_begin(
                        req_meta, layer_name, query, key, value, forward_context
                    )
            if phase == "decode":
                for req_meta in self._sparse_metadata_decode.requests:
                    self.create_req_state_attention_begin(
                        req_meta, layer_name, query, key, value, forward_context
                    )

        return query, key, value, output

    def update_req_state_attention_end(
        self, req_meta, layer_name, query, key, value, attn_output, forward_context
    ):
        layer_id = int(layer_name.split(".")[2])
        req_state = self.req_states[req_meta.request_id][layer_id]
        req_state.update_meta(req_meta)
        req_state.attention_finished(query, key, value, attn_output, forward_context)

    def attention_finished(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_output: torch.Tensor,
        layer_name: str,
        forward_context: ForwardContext,
        phase: Optional[str] = None,
    ) -> None:
        if not self.is_mla:
            for req_meta in self._sparse_metadata.requests:
                self.update_req_state_attention_end(
                    req_meta,
                    layer_name,
                    query,
                    key,
                    value,
                    attn_output,
                    forward_context,
                )
        else:
            if phase == "prefill":
                for req_meta in self._sparse_metadata_prefill.requests:
                    self.update_req_state_attention_end(
                        req_meta,
                        layer_name,
                        query,
                        key,
                        value,
                        attn_output,
                        forward_context,
                    )
            if phase == "decode":
                for req_meta in self._sparse_metadata_decode.requests:
                    self.update_req_state_attention_end(
                        req_meta,
                        layer_name,
                        query,
                        key,
                        value,
                        attn_output,
                        forward_context,
                    )

    def is_sparsed_request(self, req):
        return (
            len(req.prompt_token_ids)
            >= self._vllm_config.cache_config.block_size * self.esa_cfg["min_blocks"]
        )

    def set_block_hashes(self, req_id, token_ids):
        if req_id not in self.block_hashes:
            self.block_hashes[req_id] = {}

        if self.rank in self.block_hashes[req_id]:
            return

        self.block_hashes[req_id][self.rank] = []

        parent_block_hash_value = compute_parent_block_hash(
            self._vllm_config.model_config.model,
            self._vllm_config.parallel_config.world_size,
            self._vllm_config.model_config.dtype,
            seed_rank=0,
        )

        num_total_blocks = math.ceil(len(token_ids) / self.block_size)
        for start in range(0, len(token_ids), self.block_size):
            end = start + self.block_size
            block_idx = start // self.block_size
            if block_idx >= num_total_blocks - self.esa_cfg["local_window_sz"]:
                continue
            block_token_ids = token_ids[start:end]
            if len(block_token_ids) < self.block_size:
                break
            curr_block_token_ids_tuple = tuple(block_token_ids)
            hash_value = self.request_hasher(
                (parent_block_hash_value, curr_block_token_ids_tuple)
            )
            if block_idx >= self.esa_cfg["init_window_sz"]:
                self.block_hashes[req_id][self.rank].append(str(hash_value))

            parent_block_hash_value = hash_value

        if self.rank != 0 and not self.is_mla:
            self.newqrequest_hasher = RequestHasher(self._vllm_config, self.rank)
            for i, ucm_block_id in enumerate(self.block_hashes[req_id][self.rank]):
                self.block_hashes[req_id][self.rank][i] = str(
                    self.newqrequest_hasher(ucm_block_id)
                )

    def build_sparse_meta(
        self, scheduler_output, requests, input_batch, attn_metadata
    ) -> UcmSparseMetadata:
        self._sparse_metadata_prefill = ESASparseMetaData()
        self._sparse_metadata_decode = ESASparseMetaData()
        self._sparse_metadata = ESASparseMetaData()

        num_sched = scheduler_output.num_scheduled_tokens
        req_ids = list(getattr(input_batch, "req_ids", []))
        decode_ids = [rid for rid in req_ids if num_sched.get(rid, 0) == 1]
        decode_set = set(decode_ids)
        cached_reqs = scheduler_output.scheduled_cached_reqs
        preempt_reqs = set()
        if cached_reqs:
            for req, is_preempt in zip(
                cached_reqs.req_ids, cached_reqs.resumed_from_preemption
            ):
                if is_preempt:
                    preempt_reqs.add(req)
        for (
            req_id,
            num_scheduled_tokens,
        ) in scheduler_output.num_scheduled_tokens.items():
            req = requests[req_id]
            if not self.is_sparsed_request(req):
                continue
            self.set_block_hashes(int(req_id), req.prompt_token_ids)
            if isinstance(attn_metadata, dict):
                attn_metadata = next(iter(attn_metadata.values()))

            if not self.is_mla:
                self._sparse_metadata.add_request(
                    req_id,
                    input_batch.req_id_to_index[req_id],
                    num_scheduled_tokens,
                    req.num_computed_tokens,
                    req.block_ids[0],
                    attn_metadata.query_start_loc[input_batch.req_id_to_index[req_id]],
                    req.prompt_token_ids,
                    req.output_token_ids,
                    req_id in preempt_reqs,
                    self.block_hashes[int(req_id)][self.rank],
                )

            else:
                attn_metadata_prefill = getattr(attn_metadata, "prefill", None)
                attn_metadata_decode = getattr(attn_metadata, "decode", None)

                # 区分该req是在decode阶段还是prefill
                if req_id in decode_set:
                    if attn_metadata_decode:
                        req_id_to_index_decode = input_batch.req_id_to_index[req_id]
                        self._sparse_metadata_decode.add_request(
                            req_id,
                            req_id_to_index_decode,
                            num_scheduled_tokens,
                            req.num_computed_tokens,
                            req.block_ids[0],
                            attn_metadata.query_start_loc[req_id_to_index_decode],
                            req.prompt_token_ids,
                            req.output_token_ids,
                            req_id in preempt_reqs,
                            self.block_hashes[int(req_id)][self.rank],
                        )

                else:
                    req_id_to_index_prefill = (
                        input_batch.req_id_to_index[req_id] - attn_metadata.num_decodes
                    )
                    self._sparse_metadata_prefill.add_request(
                        req_id,
                        req_id_to_index_prefill,
                        num_scheduled_tokens,
                        req.num_computed_tokens,
                        req.block_ids[0],
                        attn_metadata_prefill.query_start_loc[req_id_to_index_prefill],
                        req.prompt_token_ids,
                        req.output_token_ids,
                        req_id in preempt_reqs,
                        self.block_hashes[int(req_id)][self.rank],
                    )

            # self._sparse_metadata = sparse_meta

    def request_begin(self, request_id: ReqType, prompt_token_ids: List[int]):
        pass

    def request_finished_in_worker(self, request_id: ReqType):
        if request_id not in self.req_states:
            return
        for layer_state in self.req_states[request_id]:
            layer_state.repre_pool.free(layer_state.slots)
        del self.req_states[request_id]

    def request_finished_in_scheduler(self, request_id: Union[int, str]):
        """
        This is called inside "Scheduler->finish_requests" function.
        Generate the metadata required by UcmSparse instance at worker-side.
        """
        pass

    def estimate_num_slots_sparsed(self, request: Request) -> int:
        if request.status == RequestStatus.PREEMPTED:
            self.preempt_req_output_tokens[request.request_id] = (
                request.num_output_tokens
            )

        if request.request_id in self.preempt_req_output_tokens:
            num_output_tokens = (
                request.num_output_tokens
                - self.preempt_req_output_tokens[request.request_id]
            )
        else:
            num_output_tokens = request.num_output_tokens

        if (
            request.num_computed_tokens == 0
            or num_output_tokens == 0
            or not self.is_sparsed_request(request)
        ):
            return INVALID_SLOT
        prompt_len = request.num_prompt_tokens
        output_len = request.num_output_tokens
        block_size = self._vllm_config.cache_config.block_size
        sparse_range = get_sparse_range(
            self.esa_cfg["init_window_sz"],
            self.esa_cfg["local_window_sz"],
            prompt_len,
            block_size,
        )
        if (flaw := prompt_len % block_size) == 0:
            local_window_tokens = block_size * self.esa_cfg["local_window_sz"]
        else:
            local_window_tokens = flaw + block_size * (
                self.esa_cfg["local_window_sz"] - 1
            )
        compressed_prompt_len = (
            self.esa_cfg["init_window_sz"] * block_size
            + int(sparse_range * self.esa_cfg["sparse_ratio"]) * block_size
            + local_window_tokens
        )
        return compressed_prompt_len + output_len

    def allocate_slots(self, kv_cache_manager, request, num_slots_sparsed):
        coordinator = kv_cache_manager.coordinator
        block_pool = kv_cache_manager.block_pool
        kv_cache_groups = kv_cache_manager.kv_cache_config.kv_cache_groups

        if request.request_id in self.preempt_req_output_tokens:
            # handle preempt: get the TRUE output_len
            num_output_tokens = (
                request.num_output_tokens
                - self.preempt_req_output_tokens[request.request_id]
            )
        else:
            num_output_tokens = request.num_output_tokens

        if num_output_tokens == 1:
            kv_cache_manager.free(request)

        new_computed_block_list = tuple([] for _ in range(len(kv_cache_groups)))
        num_blocks_to_allocate = coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=num_slots_sparsed,
            new_computed_blocks=new_computed_block_list,
        )
        manual_preempt = False
        # manual_preempt = (request.num_output_tokens % 10) == 0
        if manual_preempt or num_blocks_to_allocate > block_pool.get_num_free_blocks():
            return None
        coordinator.allocate_new_blocks(request.request_id, num_slots_sparsed)
        blocks = coordinator.single_type_managers[0].req_to_blocks[request.request_id]
        return KVCacheBlocks(tuple([blocks]))
