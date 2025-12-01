import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer import get_kv_transfer_group
from vllm.forward_context import ForwardContext
from vllm.v1.request import Request, RequestStatus

from ucm.integration.vllm.ucm_connector import RequestHasher
from ucm.logger import init_logger
from ucm.sparse.base import (
    INVALID_SLOT,
    UcmSparseBase,
    UcmSparseRole,
)
from ucm.sparse.esa.esa import (
    ESA,
    ESASparseMetaData,
    ReprePool,
    ReqStatePerLayer,
    get_sparse_range,
)
from ucm.sparse.kvcomp.hash_encoder import HashEncoder
from ucm.sparse.kvcomp.hash_retrieval import hash_retrieval_backend
from ucm.sparse.kvcomp.hash_retrieval.hash_retrieval_worker import HashRetrievalWorker
from ucm.sparse.kvcomp.kvcomp_config import KvCompConfig
from ucm.sparse.kvstar.utils import get_bind_cpus_for_rank
from ucm.sparse.state import get_ucm_sparse
from ucm.store.ucmstore import Task, UcmKVStoreBase
from ucm.utils import Config

logger = init_logger(__name__)

ReqType = Union[str, int]

data = None


class ReqStatePerLayerKvComp(ReqStatePerLayer):
    # handle single request per layer

    def __init__(
        self,
        layer_name: str,
        rank: int,
        tp_size: int,
        store_instance: UcmKVStoreBase,
        vllm_config: VllmConfig,
        retrieval_worker: Optional[HashRetrievalWorker] = None,
        repre_pool: Optional[ReprePool] = None,
        esa_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            layer_name,
            rank,
            tp_size,
            store_instance,
            vllm_config,
            retrieval_worker,
            repre_pool,
        )

        self.esa_cfg = esa_cfg
        # `retrieval_worker` 类型是 HashRetrievalWorker
        self.retrieval_worker = retrieval_worker

    def extract_block_repre(self, vllm_block_ids):
        ucm_sparse = get_ucm_sparse()
        hash_encoder = ucm_sparse.hash_encoder
        hashk_cache = hash_encoder.compute_hash(self.k_cache[vllm_block_ids])
        if self.is_mla:
            hashk_cache = hashk_cache.unsqueeze(-2)
        return hashk_cache

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
        ucm_sparse = get_ucm_sparse()
        hash_encoder = ucm_sparse.hash_encoder
        hash_query = hash_encoder.compute_hash(query)
        query_flat = hash_query.reshape(query.shape[0], -1)
        top_k = int(self.sparse_range * self.esa_cfg["sparse_ratio"])
        indexes = [self.slots]
        self.retrieval_task = self.retrieval_worker.submit(
            query_flat, topk=top_k, indexes=indexes
        )

    def block_repre_data(self):
        self.sparse_range = get_sparse_range(
            self.esa_cfg["init_window_sz"],
            self.esa_cfg["local_window_sz"],
            self.req_meta.num_prompt_tokens,
            self.block_size,
        )
        vllm_block_ids = self.req_meta.vllm_block_ids
        # torch.save({"k": self.k_cache[vllm_block_ids].cpu(), "v": self.v_cache[vllm_block_ids].cpu()},
        #            f"/home/heke/debug/{self.layer_id}.pkl")
        vllm_block_ids_dump = vllm_block_ids[
            self.esa_cfg["init_window_sz"] : self.esa_cfg["init_window_sz"]
            + self.sparse_range
        ]

        ######## 修改表征
        repre = self.extract_block_repre(vllm_block_ids_dump)
        repre_flat = repre.reshape(repre.shape[0], repre.shape[1], -1)
        new_slots = self.repre_pool.allocate(self.sparse_range)
        og_len = len(self.slots)
        for i, slot in enumerate(new_slots):
            self.slots_to_relative_indexes[slot] = og_len + i
        self.slots.extend(new_slots)
        vals = repre_flat.to("cpu", non_blocking=True, dtype=torch.uint8)
        data[self.layer_id][new_slots] = vals
        ##############

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


class KvComp(ESA):
    # handle batch
    def __init__(self, vllm_config: VllmConfig, role: UcmSparseRole):
        UcmSparseBase.__init__(self, vllm_config, role)
        self.req_states: dict[str, List[ReqStatePerLayerKvComp]] = {}
        self.rank = vllm_config.parallel_config.rank
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        if role == UcmSparseRole.WORKER:
            self.connector = get_kv_transfer_group().connector.store
        else:
            self.connector = None
        self.total_num_hidden_layers = (
            vllm_config.model_config.hf_config.num_hidden_layers
        )
        self.is_mla = vllm_config.model_config.is_deepseek_mla
        self._sparse_metadata_prefill: ESASparseMetaData = ESASparseMetaData()
        self._sparse_metadata_decode: ESASparseMetaData = ESASparseMetaData()
        self._sparse_metadata: ESASparseMetaData = ESASparseMetaData()
        self.esa_cfg = (
            Config(vllm_config.kv_transfer_config)
            .get_config()
            .get("ucm_sparse_config")
            .get("KvComp")
        )

        self.block_size = vllm_config.cache_config.block_size
        self.block_hashes: dict[int, dict[int, list[str]]] = {}
        self.request_hasher = RequestHasher(vllm_config, 0)
        self.num_kv_heads = vllm_config.model_config.get_num_kv_heads(
            vllm_config.parallel_config
        )
        self.hashk_cache = None

        kvcomp_config_path = vllm_config.kv_transfer_config.kv_connector_extra_config[
            "kvcomp_config_path"
        ]

        self.kvcomp_config = KvCompConfig.from_json(kvcomp_config_path)
        logger.info(f"read kvcomp config file : {kvcomp_config_path} ")

        assert (
            self.kvcomp_config.num_hidden_layers == self.total_num_hidden_layers
        ), f"kvcomp_config.num_hidden_layers {self.kvcomp_config.num_hidden_layers} \
             != vllm_config.model_config.hf_text_config.num_hidden_layers \
                {self.total_num_hidden_layers}"

        if hasattr(torch, "npu") and torch.npu.is_available():
            device = torch.device(f"npu:{self.rank}")
        elif hasattr(torch, "cuda") and torch.cuda.is_available():
            device = torch.device(f"cuda:{self.rank}")
        else:
            device = torch.device("cpu")

        self.hash_encoder = HashEncoder(
            input_dim=self.kvcomp_config.head_dim,
            hash_bits=self.kvcomp_config.hash_bits,
            dtype=vllm_config.model_config.dtype,
            device=device,
        )
        self.device = device

        global data

        if data is None:
            parallel_config = vllm_config.parallel_config
            num_slots = (
                vllm_config.model_config.max_model_len
                * vllm_config.scheduler_config.max_num_seqs
                // vllm_config.cache_config.block_size
            )
            block_size = vllm_config.cache_config.block_size
            dim = (
                vllm_config.model_config.get_num_kv_heads(parallel_config)
                * self.kvcomp_config.hash_bits  # 修改vllm_config.model_config.get_head_size()为hash_bits
                // 8
            )
            data = [
                torch.empty((num_slots, block_size, dim), dtype=torch.uint8)
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

        self.retrieval_workers: List[HashRetrievalWorker] = []
        for i in range(self.total_num_hidden_layers):
            backend_src = data[i]
            backend = hash_retrieval_backend.HashRetrievalWorkerBackend(
                backend_src, bind_info_dict
            )
            self.retrieval_workers.append(HashRetrievalWorker(backend))

        self.preempt_req_output_tokens: Dict[ReqType, int] = {}

    def get_or_create_layerwise_req_state(self, req_meta, layer_name):
        layer_id = int(layer_name.split(".")[2])
        if req_meta.is_preempt:
            print(
                f"preempt {req_meta.request_id}, layer_id: {layer_id}, {req_meta.num_output_tokens}"
            )
            layer_state = self.req_states[req_meta.request_id][layer_id]
            layer_state.repre_pool.free(layer_state.slots)
            self.req_states[req_meta.request_id][layer_id] = None
        if req_meta.request_id not in self.req_states:
            if self.req_states.get(req_meta.request_id) is None:
                self.req_states[req_meta.request_id] = [
                    None
                ] * self.total_num_hidden_layers
        if self.req_states[req_meta.request_id][layer_id] is None:
            self.req_states[req_meta.request_id][layer_id] = ReqStatePerLayerKvComp(
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

    def execute_begin(self, scheduler_output):
        if self.hashk_cache is None:
            print(
                " ========================== initialize hashk cache ========================== "
            )
            num_blocks = self._vllm_config.cache_config.num_gpu_blocks
            self.hashk_cache = [
                torch.empty(
                    (
                        num_blocks,
                        self.num_kv_heads,
                        self.block_size,
                        self.hash_encoder.hash_bits // 8,
                    ),
                    dtype=torch.uint8,
                    device=self.device,
                )
                for _ in range(self.total_num_hidden_layers)
            ]
