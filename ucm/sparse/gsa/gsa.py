import copy
import hashlib
import math
import pickle
import time
from dataclasses import dataclass
from functools import cache, wraps
from itertools import accumulate
from typing import Dict, List, Optional, Union

import torch

if hasattr(torch, "npu") and torch.npu.is_available():
    import torch_npu
    import ucm_custom_ops
    from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer import get_kv_transfer_group
from vllm.forward_context import (
    ForwardContext,
    get_forward_context,
)
from vllm.sequence import SequenceStage
from vllm.utils import make_tensor_with_pad, sha256
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import NONE_HASH
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request

from ucm.integration.vllm.ucm_connector import RequestHasher
from ucm.sparse.base import (
    INVALID_SLOT,
    UcmSparseBase,
    UcmSparseMetadata,
    UcmSparseRole,
)
from ucm.sparse.gsa.offload_ops import gsa_offload_ops
from ucm.sparse.gsa.prefetch.prefetch_engine import GSAPrefetchBase
from ucm.sparse.utils import (
    CUDA_TOPK,
    ENABLE_KVCOMP,
    MAX_BS,
    PTOPK_PREFETCH_ENABLE,
    SEG_PREFILL_THRESHOLD,
    gsa_config,
)

if ENABLE_KVCOMP:
    from ucm.sparse.kvcomp.hash_encoder import HashEncoder

ReqType = Union[str, int]


class GSAReqStat:
    def __init__(self, req_id, vllm_config: VllmConfig) -> None:
        self.req_id = req_id
        self.repre_slot_mapping = []
        self.calc_block_table = []
        self.calc_repre_slot_mapping = []
        self.include_mask = []
        self.exclude_mask = []
        self.blocks = []
        self.num_computed_tokens = 0
        self.num_scheduled_tokens = 0
        self.num_prompt_tokens = 0
        self.num_output_tokens = 0
        self.is_use_gsa = 0
        self.index_in_batch = 0
        self.remain_idx = None
        self.prefetch_idx = None
        self.topk_buf_tmp = None
        self.init_window_kv = None
        self.local_window_kv = []
        self.sparse_len = 0
        self.block_size = vllm_config.cache_config.block_size
        self.block_hashes = None
        self.num_prompt_blocks = 0
        self.remain_map = None
        self.prefetch_map = None
        self._vllm_config = vllm_config
        self.rank = vllm_config.parallel_config.rank
        self.use_mla = vllm_config.model_config.use_mla
        self.request_hasher = RequestHasher(vllm_config, self.rank)

    def step(self) -> int:
        return self.num_output_tokens

    def stage(self) -> SequenceStage:
        return (
            SequenceStage.DECODE
            if self.num_prompt_tokens <= self.num_computed_tokens
            else SequenceStage.PREFILL
        )

    def is_gsa(self) -> bool:
        return (
            self.num_prompt_tokens > SEG_PREFILL_THRESHOLD
            and self.stage() != SequenceStage.PREFILL
        )

    def is_last_chunk(self) -> bool:
        return (
            self.num_computed_tokens + self.num_scheduled_tokens
            == self.num_prompt_tokens
        )

    def get_seq_len(self) -> int:
        return self.num_computed_tokens + self.num_scheduled_tokens

    def set_block_hashes(self, token_ids):
        if self.block_hashes is not None:
            return
        self.block_hashes = []

        # (ldeng): why this is needed? The RequestHasher() is overlapped with compute_parent_block_hash()
        # parent_block_hash_value = compute_parent_block_hash(
        #     self._vllm_config.model_config.model,
        #     self._vllm_config.parallel_config.world_size,
        #     self._vllm_config.model_config.dtype,
        #     seed_rank=0,
        # )
        parent_block_hash_value = None

        for start in range(0, len(token_ids), self.block_size):
            end = start + self.block_size
            block_token_ids = token_ids[start:end]
            if len(block_token_ids) < self.block_size:
                break
            curr_block_token_ids_tuple = tuple(block_token_ids)
            if parent_block_hash_value is None:
                hash_value = self.request_hasher((curr_block_token_ids_tuple))
            else:
                hash_value = self.request_hasher(
                    (parent_block_hash_value, curr_block_token_ids_tuple)
                )
            # (ldeng): we forgot to update self.block_hashes here originally
            self.block_hashes.append(hash_value)
            parent_block_hash_value = hash_value

        # (ldeng): It is no longer needed now since we initialize the RequestHasher() with the rank before
        # if self.rank != 0 and not self.use_mla:
        #     self.newqrequest_hasher = RequestHasher(self._vllm_config, self.rank)
        #     for i, ucm_block_id in enumerate(self.block_hashes):
        #         self.block_hashes[i] = str(self.newqrequest_hasher(ucm_block_id))

    def add_req_new(
        self, num_scheduled_tokens, add_req_state, index_in_batch, offset
    ) -> None:
        self.blocks = [x for x in add_req_state.block_ids[0]]
        self.index_in_batch = index_in_batch
        self.num_computed_tokens = add_req_state.num_computed_tokens
        self.num_scheduled_tokens = num_scheduled_tokens
        self.num_prompt_tokens = len(add_req_state.prompt_token_ids)
        self.num_output_tokens = len(add_req_state.output_token_ids)
        self.num_prompt_blocks = math.ceil(self.num_prompt_tokens / self.block_size)
        self.is_use_gsa = (
            True if self.num_prompt_tokens > SEG_PREFILL_THRESHOLD else False
        )
        self._init_slot(offset)
        if len(self.repre_slot_mapping) > len(self.blocks):
            self.repre_slot_mapping = self.repre_slot_mapping[: len(self.blocks)]
        self.set_block_hashes(add_req_state.prompt_token_ids)

    def updata_req_state(
        self, num_scheduled_tokens, add_req_state, index_in_batch
    ) -> None:
        self.num_computed_tokens = add_req_state.num_computed_tokens
        self.num_scheduled_tokens = num_scheduled_tokens
        self.num_output_tokens = len(add_req_state.output_token_ids)
        self.index_in_batch = index_in_batch
        if self.stage() == SequenceStage.PREFILL:
            add_blocks = [x for x in add_req_state.block_ids[0] if x not in self.blocks]
            self.blocks = [x for x in add_req_state.block_ids[0]]
            self._update_slot(add_blocks)
        else:
            self._get_sparse_and_free_block()
            if len(add_req_state.block_ids[0]) != self.sparse_len:
                add_blocks = [add_req_state.block_ids[0][-1]]
                self.blocks += [add_req_state.block_ids[0][-1]]
                self.sparse_len = len(add_req_state.block_ids[0])
                self._update_slot(add_blocks)
            else:
                self.calc_block_table = []
                self.calc_repre_slot_mapping = []
        if len(self.repre_slot_mapping) > len(self.blocks):
            self.topk_buf_tmp = None
            self.repre_slot_mapping = self.repre_slot_mapping[: len(self.blocks)]

    def _get_sparse_and_free_block(self):
        if self.num_prompt_tokens != self.num_computed_tokens:
            self.remain_idx = None
            self.prefetch_idx = None
            return

        blocks_len = len(self.blocks)
        if self.num_prompt_tokens > SEG_PREFILL_THRESHOLD and PTOPK_PREFETCH_ENABLE:
            remain_len = gsa_config.compute_topk_len(blocks_len)
            if remain_len < blocks_len:
                prefetch_len = min(
                    gsa_config.num_prefetch_blocks, blocks_len - remain_len
                )
                req_idx_list = list(range(blocks_len))
                init_windows_size = gsa_config.init_windows_size
                self.remain_idx = (
                    req_idx_list[:init_windows_size]
                    + req_idx_list[init_windows_size - remain_len :]
                )
                self.prefetch_idx = req_idx_list[
                    init_windows_size
                    - remain_len
                    - prefetch_len : init_windows_size
                    - remain_len
                ]
                self.sparse_len = remain_len + prefetch_len
                return

        self.remain_idx = list(range(blocks_len))
        self.prefetch_idx = []
        self.sparse_len = blocks_len

    def _init_slot(self, offset: int) -> None:
        self.repre_slot_mapping = list(range(len(self.blocks)))
        self.repre_slot_mapping = [x + offset for x in self.repre_slot_mapping]
        if self.is_last_chunk():
            self.calc_block_table = [x for x in self.blocks[:-1]]
            self.calc_repre_slot_mapping = [x for x in self.repre_slot_mapping[:-1]]
        else:
            self.calc_block_table = [x for x in self.blocks]
            self.calc_repre_slot_mapping = [x for x in self.repre_slot_mapping]

        value = len(self.blocks)
        one_mask = [False] * value
        if value > 2:
            one_mask[0] = True
            one_mask[-1] = True
            one_mask[-2] = True
        else:
            one_mask = [True] * value
        self.include_mask = one_mask
        self.exclude_mask = [False] * value

    def _update_slot(
        self,
        add_blocks: List[int],
    ) -> None:
        add_len = len(add_blocks)
        for _ in range(add_len):
            self.repre_slot_mapping.append(self.repre_slot_mapping[-1] + 1)
            if len(self.include_mask) > 2:
                self.include_mask[-2] = False
                self.include_mask.append(True)
            else:
                self.include_mask.append(True)
            self.exclude_mask.append(False)
        if add_len > 0:
            if self.stage() == SequenceStage.PREFILL:
                if self.is_last_chunk():
                    self.calc_block_table = [x for x in add_blocks[:-1]]
                    self.calc_repre_slot_mapping = self.repre_slot_mapping[
                        add_len * -1 : -1
                    ]
                else:
                    self.calc_block_table = [x for x in add_blocks]
                    self.calc_repre_slot_mapping = self.repre_slot_mapping[
                        add_len * -1 :
                    ]
            else:
                self.calc_block_table = [self.blocks[-1]]
                self.calc_repre_slot_mapping = [self.repre_slot_mapping[-1]]
        else:
            self.calc_block_table = []
            self.calc_repre_slot_mapping = []


class GSAMetaData(UcmSparseMetadata):
    def __init__(self, vllm_config: VllmConfig):
        self.gsa_stats = {}
        self.block_size = vllm_config.cache_config.block_size
        self.device = vllm_config.device_config.device_type
        self.use_mla = vllm_config.model_config.use_mla
        self._vllm_config = vllm_config

    def get_model_input(
        self,
        scheduler_output: SchedulerOutput,
        topk_kpre_map,
        max_block_len,
        requests,
        input_batch,
        prefetch_engine,
    ) -> None:
        for index, req_id in enumerate(scheduler_output.scheduled_cached_reqs.req_ids):
            assert req_id in self.gsa_stats
            if scheduler_output.scheduled_cached_reqs.resumed_from_preemption[index]:
                del self.gsa_stats[req_id]
                prefetch_engine.del_finish_meta(req_id, False)
                self.gsa_stats[req_id] = GSAReqStat(req_id, self._vllm_config)
                self.gsa_stats[req_id].add_req_new(
                    scheduler_output.num_scheduled_tokens[req_id],
                    requests[req_id],
                    input_batch.req_id_to_index[req_id],
                    max_block_len * topk_kpre_map[req_id],
                )
            else:
                self.gsa_stats[req_id].updata_req_state(
                    scheduler_output.num_scheduled_tokens[req_id],
                    requests[req_id],
                    input_batch.req_id_to_index[req_id],
                )
        for new_req in scheduler_output.scheduled_new_reqs:
            if new_req.req_id in self.gsa_stats:
                del self.gsa_stats[new_req.req_id]
            self.gsa_stats[new_req.req_id] = GSAReqStat(
                new_req.req_id, self._vllm_config
            )
            self.gsa_stats[new_req.req_id].add_req_new(
                scheduler_output.num_scheduled_tokens[new_req.req_id],
                requests[new_req.req_id],
                input_batch.req_id_to_index[new_req.req_id],
                max_block_len * topk_kpre_map[new_req.req_id],
            )
        return self.trans_input_tensor(scheduler_output)

    def trans_input_tensor(self, scheduler_output: SchedulerOutput):
        calc_block_table = []
        model_input = {}
        calc_repre_slot_mappings = []
        batch_size = len(scheduler_output.num_scheduled_tokens.items())
        query_locals = [0] * (batch_size + 1)
        if self.use_mla:
            query_locals_prefill = [0] * (batch_size + 1)

        if ENABLE_KVCOMP:
            block_table_for_hamming = [None] * batch_size

        for req_id, num_tokens in scheduler_output.num_scheduled_tokens.items():
            req_in_batch = self.gsa_stats[req_id].index_in_batch
            calc_block_table += self.gsa_stats[req_id].calc_block_table
            calc_repre_slot_mappings += self.gsa_stats[req_id].calc_repre_slot_mapping
            query_locals[req_in_batch + 1] = scheduler_output.num_scheduled_tokens[
                req_id
            ]
            if self.use_mla and self.gsa_stats[req_id].stage() == SequenceStage.PREFILL:
                query_locals_prefill[req_in_batch + 1] = num_tokens

            if ENABLE_KVCOMP:
                block_table_for_hamming[req_in_batch] = self.gsa_stats[
                    req_id
                ].repre_slot_mapping

        query_locals = list(accumulate(query_locals))
        if self.use_mla:
            query_locals_prefill = list(accumulate(query_locals_prefill))
        model_input["calc_block_table"] = torch.tensor(
            calc_block_table, dtype=torch.int32, device="cpu"
        )
        model_input["calc_repre_slot_mapping"] = torch.tensor(
            calc_repre_slot_mappings, dtype=torch.int32, device="cpu"
        )
        model_input["query_locals"] = query_locals
        if self.use_mla:
            model_input["query_locals_prefill"] = query_locals_prefill

        if ENABLE_KVCOMP:
            model_input["block_table_for_hamming"] = make_tensor_with_pad(
                block_table_for_hamming, pad=0, dtype=torch.int32, device="cpu"
            ).to(device=self.device, non_blocking=True)
        return model_input


class TopKAndKpreManger:
    def __init__(self, max_num: int):
        self.cache_map = {}
        self.max_num = max_num
        self.free_cache = []
        for i in range(max_num):
            self.free_cache.append(i)

    def free(self, req_id: ReqType) -> bool:
        if self.cache_map[req_id] in self.free_cache:
            print("[GSA] ERROR free req_id is free cache")
            return False
        else:
            self.free_cache.append(self.cache_map[req_id])
            del self.cache_map[req_id]
            return True

    def alloc(self, req_id: ReqType) -> int:
        if self.free_cache != []:
            free_index = self.free_cache.pop(0)
            self.cache_map[req_id] = free_index
            return free_index
        else:
            return None

    def is_exist(self, req_id: ReqType) -> bool:
        if req_id in self.cache_map:
            return True
        else:
            return False


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


class TopkCal:
    def __init__(self, att_num_heads, kv_num_heads, head_size, kpre_caches, use_mla):
        self.att_num_heads = att_num_heads
        self.kv_num_heads = kv_num_heads
        self.head_size = head_size
        self.kpre_caches = kpre_caches
        self.topk_ratio = 0.3
        self.use_mla = use_mla

        if ENABLE_KVCOMP:
            self.device = kpre_caches[0].device
            # (TODO) support other dtypes, we simply use float16 here
            self.hash_encoder = HashEncoder(
                input_dim=head_size,
                hash_bits=head_size,
                dtype=torch.float16,
                device=self.device,
            )
            self.preserved_blocks = (
                gsa_config.kvcomp_preserve_blocks
            )

    def set_topk_param(self, repre_slot_mapping, include_mask, exclude_mask):
        self.repre_slot_mapping = repre_slot_mapping
        self.include_mask = include_mask
        self.exclude_mask = exclude_mask

    def set_topk_param_for_hamming(
        self, block_table_for_hamming, include_mask, exclude_mask, seq_lens_ori
    ):
        self.block_table_for_hamming = block_table_for_hamming
        self.include_mask = include_mask
        self.exclude_mask = exclude_mask
        self.seq_lens_for_hamming = seq_lens_ori
        self.max_seq_len_for_hamming = torch.max(self.seq_lens_for_hamming).item()
        self.batch_size = block_table_for_hamming.shape[0]
        self.top_k_for_hamming = torch.full(
            size=[self.batch_size],
            fill_value=self.preserved_blocks,
            dtype=torch.int32,
            device=self.device,
        )
        self.chunk_sizes_for_hamming = torch.full(
            size=[self.batch_size],
            fill_value=gsa_config.block_size,
            dtype=torch.int32,
            device=self.device,
        )

        if self.kv_num_heads == 1:
            self.hamming_output = torch.zeros(
                size=[self.batch_size, self.preserved_blocks],
                dtype=torch.int32,
                device=self.device,
            )
        else:
            self.hamming_output = torch.zeros(
                size=[self.batch_size, self.kv_num_heads, self.preserved_blocks],
                dtype=torch.int32,
                device=self.device,
            )

    def set_topk_caches(self, cal_topk_id, topk_caches, topk_len_list):
        self.cal_topk_id = cal_topk_id
        self.topk_caches = topk_caches
        self.topk_len_list = topk_len_list

    def cal_topk(self, intermediate_q, current_layer_id):
        bs = len(self.cal_topk_id)
        head_group_num = self.att_num_heads // self.kv_num_heads
        q_decode = intermediate_q[self.cal_topk_id]
        kpre_index = self.repre_slot_mapping.flatten()
        kpre_need = self.kpre_caches[current_layer_id][kpre_index]
        max_norm_num = kpre_need.shape[1]
        kpre_out = kpre_need.unsqueeze(2).expand(-1, -1, head_group_num, -1, -1)
        kpre_out = kpre_out.reshape(bs, -1, self.att_num_heads, self.head_size)
        blk_num = kpre_out.shape[1] // max_norm_num
        qk = torch.einsum("bij,bmij->bim", q_decode, kpre_out)
        attention_weights_without_norm, _ = torch.max(
            qk.reshape(bs, self.att_num_heads, blk_num, max_norm_num), dim=-1
        )
        dot_product_weights = attention_weights_without_norm.mean(1)
        dot_product_weights.masked_fill_(self.include_mask == 1, float("inf"))
        dot_product_weights.masked_fill_(self.exclude_mask == 1, float("-inf"))
        selected_block_nums = self.topk_len_list[0]
        _, top_indices = torch.topk(
            dot_product_weights, selected_block_nums, dim=-1, sorted=False
        )
        self.topk_caches[current_layer_id][self.cal_topk_id] = top_indices

    def cal_topk_for_hamming(self, intermediate_q, current_layer_id):
        q_decode = intermediate_q[self.cal_topk_id]
        block_table_decode = self.block_table_for_hamming[self.cal_topk_id]
        hashq = self.hash_encoder.compute_hash(q_decode)
        hashq = hashq.unsqueeze(2).contiguous()
        hashk_cache = self.kpre_caches[current_layer_id]


        # debug
        if current_layer_id == 0 and True:
            print(f"=======cal_topk_for_hamming=======")
            print(f"hashq.shape: {hashq.shape}")
            print(f"hashk_cache.shape: {hashk_cache.shape}")
            print(f"top_k_for_hamming.shape: {self.top_k_for_hamming.shape}")
            print(f"seq_lens_for_hamming.shape: {self.seq_lens_for_hamming.shape}")
            print(f"chunk_sizes_for_hamming.shape: {self.chunk_sizes_for_hamming.shape}")
            print(f"block_table_for_hamming.shape: {block_table_decode.shape}")
            print(f"hamming_output.shape: {self.hamming_output.shape}")
            print(f"top_k_for_hamming: {self.top_k_for_hamming}")
            print(f"seq_lens_for_hamming: {self.seq_lens_for_hamming}")
            print(f"chunk_sizes_for_hamming: {self.chunk_sizes_for_hamming}")
            print(f"max_seq_len_for_hamming: {self.max_seq_len_for_hamming}")
            print(f"block_table_for_hamming: {block_table_decode}")
            torch.npu.synchronize()
        

        ucm_custom_ops.hamming_dist_top_k(
            hashq_op=hashq,
            hashk_cache_op=hashk_cache,
            top_n_op=self.top_k_for_hamming,
            seq_len_op=self.seq_lens_for_hamming,
            chunk_size_op=self.chunk_sizes_for_hamming,
            max_seq_len=self.max_seq_len_for_hamming,
            sink=gsa_config.init_windows_size,
            recent=gsa_config.recent_window_size,
            block_table_op=block_table_decode,
            indices=self.hamming_output,
        )

        if current_layer_id == 0 and True:
            print(f"=======cal_topk_for_hamming=======")
            print(f"after ucm_custom_ops.hamming_dist_top_k, hamming_output: {self.hamming_output}")

        if self.kv_num_heads == 1:
            topk_indices = self.hamming_output
        else:
            # (ldeng) we use the first head's topk indices as the final topk indices now, need to support multi-head later
            topk_indices = self.hamming_output[:, 0, :]
        self.topk_caches[current_layer_id][self.cal_topk_id] = topk_indices


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


class GSA(UcmSparseBase):
    def __init__(self, vllm_config: VllmConfig, role: UcmSparseRole):
        super().__init__(vllm_config, role)
        self.rank = vllm_config.parallel_config.rank
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.device = vllm_config.device_config.device
        self.num_key_heads = vllm_config.model_config.get_num_kv_heads(
            vllm_config.parallel_config
        )
        self.head_size = vllm_config.model_config.get_head_size()
        self.use_mla = vllm_config.model_config.use_mla
        self.block_size = vllm_config.cache_config.block_size
        self.element_size = vllm_config.model_config.dtype.itemsize
        self.num_head = vllm_config.model_config.get_num_kv_heads(
            vllm_config.parallel_config
        )
        self.total_tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.att_num_heads = vllm_config.model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        self.dtype = vllm_config.model_config.dtype
        if PTOPK_PREFETCH_ENABLE:
            if role == UcmSparseRole.WORKER:
                self.connector = get_kv_transfer_group().connector.store
            else:
                self.connector = None
        self.is_python_load = not torch.cuda.is_available()
        if CUDA_TOPK:
            self.prefetch_engine = GSAPrefetchBase(
                vllm_config, 16, True, False, False, 1, self.is_python_load
            )
        else:
            self.prefetch_engine = GSAPrefetchBase(
                vllm_config, 16, True, True, False, 1, self.is_python_load
            )
        self.topk_kpre_manger = TopKAndKpreManger(MAX_BS)
        self.gsa_metadata = None
        self.model_input = None
        self.gsa_stats = {}
        self.init_topk_cal(vllm_config, self.prefetch_engine)
        self.decode_index = []
        self.copy_k_flag = [False] * self.layer_num
        gsa_config.set_config(self.block_size)
        self.task_load = {}

    def init_topk_cal(
        self,
        vllm_config: VllmConfig,
        prefetch_engine: GSAPrefetchBase,
    ) -> None:
        parallel_config = vllm_config.parallel_config
        block_size = vllm_config.cache_config.block_size
        att_num_heads = vllm_config.model_config.get_num_attention_heads(
            parallel_config
        )
        kv_num_heads = vllm_config.model_config.get_num_kv_heads(parallel_config)
        head_size = vllm_config.model_config.get_head_size()
        self.gsa_offload_ops = gsa_offload_ops.CalKpreAndTopk(
            self.layer_num, block_size, MAX_BS, att_num_heads, head_size
        )
        self.gsa_offload_ops.set_kpre_method_param(kv_num_heads, 1)
        self.gsa_offload_ops.set_kpre_cache(prefetch_engine.kpre_caches)
        self.is_cal_kpre = [False] * self.layer_num
        self.gsa_q_cache = torch.zeros(
            (
                self.layer_num,
                MAX_BS,
                att_num_heads,
                head_size,
            ),
            device=vllm_config.device_config.device,
            dtype=torch.float32,
        )
        if CUDA_TOPK:
            self.gsa_cuda_topk = TopkCal(
                att_num_heads,
                kv_num_heads,
                head_size,
                prefetch_engine.kpre_caches,
                self.use_mla,
            )

    def copy_q(self, query: torch.Tensor, current_layer_id: int) -> None:
        ids = [-1] * len(self.prefetch_engine.req_ids_bs)
        for req_id in self.prefetch_engine.req_ids_bs:
            req_meta = self.gsa_metadata.gsa_stats[req_id]
            if not self.use_mla:
                if req_meta.is_gsa():
                    index_in_batch = req_meta.index_in_batch
                    ids[index_in_batch] = (
                        self.model_input["query_locals"][index_in_batch + 1] - 1
                    )
            else:
                if req_meta.is_gsa():
                    index_in_batch = req_meta.index_in_batch
                    ids[index_in_batch] = 1
        if CUDA_TOPK:
            if not self.use_mla:
                if ENABLE_KVCOMP:
                    self.gsa_cuda_topk.cal_topk_for_hamming(
                        query[ids], current_layer_id
                    )
                else:
                    self.gsa_cuda_topk.cal_topk(
                        query[ids], current_layer_id
                    )  #####  todo 计算的ids
            else:
                self.gsa_cuda_topk.cal_topk(query, current_layer_id)
        else:
            if not self.use_mla:
                self.gsa_q_cache[current_layer_id][: len(ids)].copy_(query[ids])
            else:
                self.gsa_q_cache[current_layer_id][self.decode_index].copy_(query)
            is_cal_kpre = len(self.model_input["calc_block_table"]) > 0
            self.gsa_offload_ops.add_copy_req(
                is_cal_kpre, current_layer_id, ids, self.gsa_q_cache[current_layer_id]
            )

    def copy_k(self, layer_name: str, forward_context: ForwardContext) -> None:
        current_layer_id = int(layer_name.split(".")[2])
        block_ids = self.model_input["calc_block_table"]
        calc_repre_slot_mappings = self.model_input["calc_repre_slot_mapping"]
        if len(block_ids) > 0:
            attn = forward_context.no_compile_layers
            if not self.use_mla:
                if ENABLE_KVCOMP:
                    hashk_cache_out = self.gsa_cuda_topk.hash_encoder.compute_hash(
                        attn[layer_name].kv_cache[forward_context.virtual_engine][0][
                            block_ids
                        ]
                    )
                    hashk_cache_out = hashk_cache_out.transpose(1, 2).contiguous()
                else:
                    key_cache_mean_out = (
                        attn[layer_name]
                        .kv_cache[forward_context.virtual_engine][0][block_ids]
                        .mean(dim=1, keepdim=True)
                    )
            else:
                key_cache_mean_out = (
                    attn[layer_name]
                    .kv_cache[forward_context.virtual_engine][block_ids]
                    .mean(dim=1, keepdim=True)
                )
                if torch.cuda.is_available():
                    key_cache_mean_out = torch.unsqueeze(key_cache_mean_out, 1)
            if CUDA_TOPK:
                if ENABLE_KVCOMP:
                    self.prefetch_engine.kpre_caches[current_layer_id][
                        calc_repre_slot_mappings
                    ] = hashk_cache_out.clone()
                else:
                    self.prefetch_engine.kpre_caches[current_layer_id][
                        calc_repre_slot_mappings
                    ] = key_cache_mean_out.clone()
            else:
                self.prefetch_engine.kpre_caches[current_layer_id][
                    calc_repre_slot_mappings
                ] = key_cache_mean_out.to(dtype=torch.float32, device="cpu")
            if not self.use_mla:
                k_needed = attn[layer_name].kv_cache[forward_context.virtual_engine][0]
            else:
                k_needed = attn[layer_name].kv_cache[forward_context.virtual_engine]
            self.gsa_offload_ops.add_copy_req(
                True, current_layer_id, [], k_needed
            )  #####  todo  适配kcache形状

    def attention_begin(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_name: str,
        forward_context: ForwardContext,
        phase: Optional[str] = None,
    ) -> None:
        current_layer_id = int(layer_name.split(".")[2])
        if self.prefetch_engine.atb_gsa_enable and self.prefetch_engine.is_topk_cal:
            if not self.use_mla:
                self.copy_q(query, current_layer_id)
            else:
                if phase == "decode":
                    self.copy_q(query, current_layer_id)
        if isinstance(forward_context.attn_metadata, dict):
            attn_metadata = forward_context.attn_metadata[layer_name]
        else:
            attn_metadata = forward_context.attn_metadata
        if self.prefetch_engine.atb_gsa_enable:
            if not self.use_mla:
                if torch.cuda.is_available():
                    attn_metadata.block_table = self.model_input["block_tables_mp"][
                        current_layer_id
                    ]
                    attn_metadata.seq_lens = self.model_input["gsa_seq_len"][
                        current_layer_id
                    ]
                else:
                    attn_metadata.block_tables[
                        : len(self.prefetch_engine.req_ids_bs)
                    ] = self.model_input["block_tables_mp"][current_layer_id]
                    attn_metadata.seq_lens.copy_(
                        self.model_input["gsa_seq_len"][current_layer_id]
                    )
            else:
                if phase == "decode":
                    if torch.cuda.is_available():
                        attn_metadata.decode.block_table = self.model_input[
                            "block_tables_mp"
                        ][current_layer_id][self.decode_index]
                        attn_metadata.decode.seq_lens = self.model_input["gsa_seq_len"][
                            current_layer_id
                        ][self.decode_index]
                    else:
                        attn_metadata.decode.block_table[
                            : len(self.prefetch_engine.req_ids_bs)
                        ].copy_(
                            self.model_input["block_tables_mp"][current_layer_id][
                                self.decode_index
                            ]
                        )
                        attn_metadata.decode.seq_lens.copy_(
                            self.model_input["gsa_seq_len"][current_layer_id][
                                self.decode_index
                            ]
                        )

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
        current_layer_id = int(layer_name.split(".")[2])
        if not self.copy_k_flag[current_layer_id]:
            self.copy_k(layer_name, forward_context)
            self.copy_k_flag[current_layer_id] = True
        if self.use_mla and torch.cuda.is_available():
            return
        for req_id in self.prefetch_engine.req_ids_bs:
            assert req_id in self.gsa_metadata.gsa_stats
            req_meta = self.gsa_metadata.gsa_stats[req_id]
            if (
                req_meta.is_last_chunk()
                and req_meta.num_prompt_tokens > SEG_PREFILL_THRESHOLD
                and PTOPK_PREFETCH_ENABLE
            ):
                blocks_len = len(self.gsa_metadata.gsa_stats[req_id].blocks)
                remain_len = gsa_config.compute_topk_len(blocks_len)
                prefetch_len = min(
                    gsa_config.num_prefetch_blocks, blocks_len - remain_len
                )

                if ENABLE_KVCOMP:
                    topk_value = self.last_chunk_topk_cal_for_hamming(
                        req_meta, query, current_layer_id, remain_len + prefetch_len
                    )
                else:
                    topk_value = self.last_chunk_topk_cal(
                        req_meta, query, current_layer_id, remain_len + prefetch_len
                    )

                if self.gsa_metadata.gsa_stats[req_id].remain_map == None:
                    self.gsa_metadata.gsa_stats[req_id].remain_map = [
                        None
                    ] * self.layer_num
                    self.gsa_metadata.gsa_stats[req_id].prefetch_map = [
                        None
                    ] * self.layer_num

                self.kvcache_init_last_chunk(
                    forward_context, layer_name, topk_value, req_id
                )

                if self.gsa_metadata.gsa_stats[req_id].topk_buf_tmp == None:
                    self.gsa_metadata.gsa_stats[req_id].topk_buf_tmp = torch.zeros(
                        (self.layer_num, len(topk_value)),
                        dtype=torch.int32,
                        device="cpu",
                    )
                self.gsa_metadata.gsa_stats[req_id].topk_buf_tmp[
                    current_layer_id
                ] = topk_value

    def last_chunk_topk_cal(self, req_meta, query, current_layer_id, first_topk_len):
        index_in_batch = req_meta.index_in_batch
        bs = 1
        if not self.use_mla:
            cal_topk_id = [self.model_input["query_locals"][index_in_batch + 1] - 1]
        else:
            cal_topk_id = [
                self.model_input["query_locals_prefill"][index_in_batch + 1] - 1
            ]
        head_group_num = self.att_num_heads // self.num_key_heads
        q_decode = query[cal_topk_id]

        include_mask = torch.tensor(
            req_meta.include_mask, dtype=torch.uint8, device=self.device
        )
        exclude_mask = torch.tensor(
            req_meta.exclude_mask, dtype=torch.uint8, device=self.device
        )
        if CUDA_TOPK:
            kpre_index = torch.tensor(
                req_meta.repre_slot_mapping, dtype=torch.int32, device=self.device
            )
            kpre_need = self.prefetch_engine.kpre_caches[current_layer_id][kpre_index]
        else:
            kpre_index = torch.tensor(
                req_meta.repre_slot_mapping, dtype=torch.int32, device="cpu"
            )
            kpre_need = self.prefetch_engine.kpre_caches[current_layer_id][
                kpre_index
            ].to(device=self.device, dtype=self.dtype)

        max_norm_num = kpre_need.shape[1]
        kpre_out = kpre_need.unsqueeze(2).expand(-1, -1, head_group_num, -1, -1)
        kpre_out = kpre_out.reshape(bs, -1, self.att_num_heads, self.head_size)
        blk_num = kpre_out.shape[1] // max_norm_num
        qk = torch.einsum("bij,bmij->bim", q_decode, kpre_out)
        attention_weights_without_norm, _ = torch.max(
            qk.reshape(bs, self.att_num_heads, blk_num, max_norm_num), dim=-1
        )
        dot_product_weights = attention_weights_without_norm.mean(1)
        dot_product_weights.masked_fill_(include_mask == 1, float("inf"))
        dot_product_weights.masked_fill_(exclude_mask == 1, float("-inf"))
        _, top_indices = torch.topk(dot_product_weights, first_topk_len, dim=-1)
        return top_indices[0].cpu()

    def last_chunk_topk_cal_for_hamming(
        self, req_meta, query, current_layer_id, first_topk_len
    ):
        assert ENABLE_KVCOMP
        assert CUDA_TOPK

        index_in_batch = req_meta.index_in_batch
        bs = 1
        if not self.use_mla:
            cal_topk_id = [self.model_input["query_locals"][index_in_batch + 1] - 1]
        else:
            cal_topk_id = [
                self.model_input["query_locals_prefill"][index_in_batch + 1] - 1
            ]
        q_decode = query[cal_topk_id]
        hashq = self.gsa_cuda_topk.hash_encoder.compute_hash(q_decode)
        hashq = hashq.unsqueeze(2).contiguous()
        hashk_cache = self.prefetch_engine.kpre_caches[current_layer_id]
        hamming_output = torch.zeros(
            size=[bs, self.gsa_cuda_topk.preserved_blocks],
            dtype=torch.int32,
            device=self.device,
        )

        top_k_for_hamming = torch.tensor(
            [first_topk_len], dtype=torch.int32, device=self.device
        )

        seq_lens_for_hamming = torch.tensor(
            [req_meta.get_seq_len()], dtype=torch.int32, device=self.device
        )
        chunk_sizes_for_hamming = torch.tensor(
            [gsa_config.block_size], dtype=torch.int32, device=self.device
        )
        max_seq_len_for_hamming = req_meta.get_seq_len()

        block_table_for_hamming = torch.tensor(
            req_meta.repre_slot_mapping, dtype=torch.int32, device=self.device
        )

        # debug
        if self.rank == 0 and True:
            print(f"=======last_chunk_topk_cal_for_hamming=======")
            print(f"hashq.shape: {hashq.shape}")
            print(f"hashk_cache.shape: {hashk_cache.shape}")
            print(f"top_k_for_hamming.shape: {top_k_for_hamming.shape}")
            print(f"seq_lens_for_hamming.shape: {seq_lens_for_hamming.shape}")
            print(f"chunk_sizes_for_hamming.shape: {chunk_sizes_for_hamming.shape}")
            print(f"block_table_for_hamming.shape: {block_table_for_hamming.shape}")
            print(f"hamming_output.shape: {hamming_output.shape}")
            print(f"top_k_for_hamming: {top_k_for_hamming}")
            print(f"seq_lens_for_hamming: {seq_lens_for_hamming}")
            print(f"chunk_sizes_for_hamming: {chunk_sizes_for_hamming}")
            print(f"max_seq_len_for_hamming: {max_seq_len_for_hamming}")
            print(f"block_table_for_hamming: {block_table_for_hamming}")
            print(f"hamming_output: {hamming_output}")
            torch.npu.synchronize()

        ucm_custom_ops.hamming_dist_top_k(
            hashq_op=hashq,
            hashk_cache_op=hashk_cache,
            top_n_op=top_k_for_hamming,
            seq_len_op=seq_lens_for_hamming,
            chunk_size_op=chunk_sizes_for_hamming,
            max_seq_len=max_seq_len_for_hamming,
            sink=gsa_config.init_windows_size,
            recent=gsa_config.recent_window_size,
            block_table_op=block_table_for_hamming,
            indices=hamming_output,
        )

        if self.rank == 0 and True:
            print(f"=======last_chunk_topk_cal_for_hamming=======")
            print(f"after ucm_custom_ops.hamming_dist_top_k, hamming_output: {hamming_output}")

        if self.kv_num_heads == 1:
            return hamming_output[0].cpu()
        else:
            # (ldeng) we use the first head's topk indices as the final topk indices now, need to support multi-head later
            return hamming_output[0, 0, :].cpu()

    def kvcache_init_last_chunk(
        self, forward_context: ForwardContext, layer_name, topk_value, req_id
    ):
        current_layer_id = int(layer_name.split(".")[2])
        blocks_len = len(self.gsa_metadata.gsa_stats[req_id].blocks)
        remain_len = gsa_config.compute_topk_len(blocks_len)
        prefetch_len = min(gsa_config.num_prefetch_blocks, blocks_len - remain_len)
        req_idx_list = list(range(blocks_len))
        init_windows_size = gsa_config.init_windows_size
        remain_idx = (
            req_idx_list[:init_windows_size]
            + req_idx_list[init_windows_size - remain_len - prefetch_len :]
        )
        assert len(remain_idx) == len(topk_value)
        mv_map, remain_map, prefetch_map = self.get_mv_map(
            self.gsa_metadata.gsa_stats[req_id].blocks,
            remain_idx,
            topk_value.tolist(),
            remain_len,
        )
        self.gsa_metadata.gsa_stats[req_id].remain_map[current_layer_id] = remain_map
        self.gsa_metadata.gsa_stats[req_id].prefetch_map[
            current_layer_id
        ] = prefetch_map
        if not self.use_mla:
            layer_k_cache = forward_context.no_compile_layers[layer_name].kv_cache[
                forward_context.virtual_engine
            ][0]
            layer_v_cache = forward_context.no_compile_layers[layer_name].kv_cache[
                forward_context.virtual_engine
            ][1]
        else:
            layer_k_cache = forward_context.no_compile_layers[layer_name].kv_cache[
                forward_context.virtual_engine
            ]
        for block_id in mv_map:
            layer_k_cache[mv_map[block_id]].copy_(layer_k_cache[block_id])
            if not self.use_mla:
                layer_v_cache[mv_map[block_id]].copy_(layer_v_cache[block_id])

    def get_mv_map(self, blocks, remain_idxs, topk_values, remain_len):
        mv_map = {}
        free_block = []
        hit_block = []
        miss_block = []
        remain_map = {}
        prefetch_map = {}
        new_block = [None] * len(topk_values)
        for index, idx in enumerate(topk_values):
            if idx in remain_idxs:
                new_block[index] = blocks[idx]
                hit_block.append(idx)
            else:
                miss_block.append(idx)

        for idx in remain_idxs:
            if idx not in hit_block:
                free_block.append(idx)

        for index in range(len(new_block)):
            if new_block[index] == None:
                one_free_idx = free_block.pop(0)
                new_block[index] = blocks[one_free_idx]
                idx = topk_values[index]
                mv_map[blocks[idx]] = blocks[one_free_idx]

        for index in range(len(new_block)):
            idx = topk_values[index]
            if index < remain_len:
                remain_map[idx] = new_block[index]
            else:
                prefetch_map[idx] = new_block[index]
        return mv_map, remain_map, prefetch_map

    def build_gsa_metadata(
        self, scheduler_output: SchedulerOutput, requests, input_batch
    ) -> GSAMetaData:
        for req_id, _ in scheduler_output.num_scheduled_tokens.items():
            if not self.topk_kpre_manger.is_exist(req_id):
                index = self.topk_kpre_manger.alloc(req_id)
                assert index != None
        gsa_meta = GSAMetaData(self._vllm_config)
        gsa_meta.gsa_stats = self.gsa_stats
        self.model_input = gsa_meta.get_model_input(
            scheduler_output,
            self.topk_kpre_manger.cache_map,
            self.prefetch_engine.max_block_len,
            requests,
            input_batch,
            self.prefetch_engine,
        )
        self.gsa_stats = gsa_meta.gsa_stats
        return gsa_meta

    def execute_begin(self, scheduler_output: SchedulerOutput):
        self.copy_k_flag = [False] * self.layer_num
        batch_size = len(scheduler_output.num_scheduled_tokens.items())
        req_ids = [0] * batch_size
        block_table_ori = [0] * batch_size
        topk_kpre_maps = [0] * batch_size
        for req_id, _ in scheduler_output.num_scheduled_tokens.items():
            req_in_batch = self.gsa_metadata.gsa_stats[req_id].index_in_batch
            req_ids[req_in_batch] = req_id
            block_table_ori[req_in_batch] = self.gsa_metadata.gsa_stats[req_id].blocks
            topk_kpre_maps[req_in_batch] = self.topk_kpre_manger.cache_map[req_id]

        is_topk_done = self.gsa_offload_ops.is_calculate_finish()
        self.prefetch_engine.model_input_deal(
            req_ids,
            block_table_ori,
            topk_kpre_maps,
            self.model_input,
            self.gsa_metadata,
            is_topk_done,
        )
        self.gsa_stats = self.gsa_metadata.gsa_stats
        self._start_topk_cal()

    def execute_finished(self):
        kv_caches = [None] * self.layer_num
        forward_context = get_forward_context()
        attn = forward_context.no_compile_layers
        for layer_name in attn.keys():
            if self.use_mla and "mlp.experts" in layer_name:
                continue
            kv_cache = attn[layer_name].kv_cache[forward_context.virtual_engine]
            layer_id = int(layer_name.split(".")[2])
            kv_caches[layer_id] = kv_cache
        if PTOPK_PREFETCH_ENABLE:
            if self.is_python_load:
                is_prefetch_done = self.check_transfer_task_done()
            else:
                is_prefetch_done = (
                    self.prefetch_engine.prefetch_engine_c.get_prefetch_status()
                )
            all_free_block_ids, all_miss_ids = self.prefetch_engine.deal_async_prefetch(
                is_prefetch_done,
                self.gsa_metadata,
                kv_caches,
                self.connector.cc_store(),
            )
            if self.is_python_load:
                self.launch_transfer_task(all_free_block_ids, all_miss_ids, kv_caches)
        else:
            self.prefetch_engine.deal_async_prefetch(
                False, self.gsa_metadata, kv_caches, None
            )

    def launch_transfer_task(self, all_free_block_ids, all_miss_ids, kv_caches):
        if all_free_block_ids == None:
            return
        fn = getattr(self.connector, "load")
        precision = self.element_size
        if self.use_mla:
            block_data_size = kv_caches[0].numel() * precision
        else:
            block_data_size = kv_caches[0][0].numel() * precision

        offsets_k = []
        key_src_tensors = []
        block_hashes = []

        for req_id in all_free_block_ids.keys():
            req_block_hash = self.gsa_metadata.gsa_stats[req_id].block_hashes
            for layer_id in range(self.layer_num):
                length = len(all_free_block_ids[req_id][layer_id])
                if length == 0:
                    continue

                offset_k = compute_layer_offset(
                    block_data_size,
                    layer_id,
                    is_v=False,
                    is_mla=self.use_mla,
                )
                offsets_k += [offset_k] * length
                block_hashes += [
                    req_block_hash[i] for i in all_miss_ids[req_id][layer_id]
                ]

                if not self.use_mla:
                    key_src_tensors += [
                        kv_caches[layer_id][0][_id]
                        for _id in all_free_block_ids[req_id][layer_id]
                    ]
                    offset_v = compute_layer_offset(
                        block_data_size,
                        layer_id,
                        is_v=True,
                        is_mla=self.use_mla,
                    )
                    offsets_k += [offset_v] * length
                    block_hashes += [
                        req_block_hash[i] for i in all_miss_ids[req_id][layer_id]
                    ]
                    key_src_tensors += [
                        kv_caches[layer_id][1][_id]
                        for _id in all_free_block_ids[req_id][layer_id]
                    ]
                else:
                    key_src_tensors += [
                        kv_caches[layer_id][_id]
                        for _id in all_free_block_ids[req_id][layer_id]
                    ]

        task_all = fn(block_hashes, offsets_k, key_src_tensors)
        task_all_hash = task_hash_func(block_hashes, "load", "value")
        self.task_load[task_all_hash] = task_all

    def check_transfer_task_done(self) -> bool:
        if len(self.task_load) == 0:
            return True

        for task_hash, task in self.task_load.items():
            ret = self.connector.check(task)
            if not ret:
                return False
        self.task_load.clear()
        return True

    def build_sparse_meta(
        self, scheduler_output: SchedulerOutput, requests, input_batch, attn_metadata
    ) -> None:
        self.gsa_metadata = self.build_gsa_metadata(
            scheduler_output, requests, input_batch
        )
        num_sched = scheduler_output.num_scheduled_tokens
        req_ids = list(getattr(input_batch, "req_ids", []))
        self.decode_index = [
            input_batch.req_id_to_index[rid]
            for rid in req_ids
            if num_sched.get(rid, 0) == 1
        ]

    def request_begin(self, request_id: ReqType, prompt_token_ids: List[int]):
        pass

    def request_finished_in_scheduler(self, request_id: ReqType):
        pass

    def request_finished_in_worker(self, request_id: ReqType):
        if self.topk_kpre_manger.is_exist(request_id):
            self.topk_kpre_manger.free(request_id)
        if request_id in self.gsa_stats:
            del self.gsa_stats[request_id]
        self.prefetch_engine.del_finish_meta(request_id)

    def update_state_after_alloc(self, request: Request, num_blocks: int):
        pass

    def estimate_num_slots_sparsed(self, request: Request) -> int:
        if not PTOPK_PREFETCH_ENABLE:
            return INVALID_SLOT
        if (
            request.num_output_tokens == 0
            or request.num_prompt_tokens < self.block_size
        ):
            return INVALID_SLOT
        if request.num_prompt_tokens <= SEG_PREFILL_THRESHOLD:
            return INVALID_SLOT
        block_size = self._vllm_config.cache_config.block_size
        num_prompt_blocks = math.ceil(request.num_prompt_tokens / block_size)
        num_all_blocks = math.ceil(request.num_tokens / block_size)
        topk_len = gsa_config.compute_topk_len(num_prompt_blocks)
        prefetch_len = min(gsa_config.num_prefetch_blocks, num_prompt_blocks - topk_len)
        num_sparse_blocks = num_all_blocks - num_prompt_blocks + topk_len + prefetch_len
        flaw = request.num_tokens % block_size
        if flaw:
            flaw = block_size - flaw
        num_tokens_sparsed = num_sparse_blocks * block_size - flaw
        return num_tokens_sparsed

    def _start_topk_cal(self) -> None:
        if self.prefetch_engine.atb_gsa_enable and self.prefetch_engine.is_topk_cal:
            cal_topk_id = []
            is_decode = []
            topk_len_list = []
            repre_slot_mappings = []
            repre_slot_mappings_all = []
            include_masks = []
            exclude_masks = []

            if ENABLE_KVCOMP:
                seq_lens_ori = []

            for req_id in self.prefetch_engine.req_ids_bs:
                req_meta = self.gsa_metadata.gsa_stats[req_id]
                if req_meta.is_gsa():
                    cal_topk_id.append(req_meta.index_in_batch)
                    is_decode.append(True)
                    one_topk_len = (
                        gsa_config.compute_topk_len(len(req_meta.blocks))
                        + gsa_config.num_prefetch_blocks
                    )
                    topk_len_list.append(one_topk_len)
                    if CUDA_TOPK:
                        include_masks.append(req_meta.include_mask)
                        exclude_masks.append(req_meta.exclude_mask)
                        repre_slot_mappings.append(req_meta.repre_slot_mapping)

                    if ENABLE_KVCOMP:
                        seq_lens_ori.append(req_meta.get_seq_len())
                else:
                    is_decode.append(False)
                repre_slot_mappings_all.append(req_meta.repre_slot_mapping)

            if ENABLE_KVCOMP and len(topk_len_list) != 0:
                seq_lens_ori = torch.tensor(seq_lens_ori, dtype=torch.int32).to(
                    device=self.device, non_blocking=True
                )

            if CUDA_TOPK and len(topk_len_list) != 0:
                topk_len_list = [max(topk_len_list)] * len(topk_len_list)
                repre_slot_mappings = make_tensor_with_pad(
                    repre_slot_mappings, pad=0, dtype=torch.int32, device=self.device
                )
                include_masks = make_tensor_with_pad(
                    include_masks, pad=False, dtype=torch.uint8, device=self.device
                )
                exclude_masks = make_tensor_with_pad(
                    exclude_masks, pad=True, dtype=torch.uint8, device=self.device
                )
            self.gsa_offload_ops.set_common_param(cal_topk_id, is_decode)
            if len(self.model_input["calc_block_table"]) != 0:
                self.gsa_offload_ops.set_kpre_param(
                    self.model_input["calc_block_table"], []
                )

            if CUDA_TOPK and len(topk_len_list) != 0:
                if ENABLE_KVCOMP:
                    self.gsa_cuda_topk.set_topk_param_for_hamming(
                        repre_slot_mappings,
                        include_masks,
                        exclude_masks,
                        seq_lens_ori,
                    )
                else:
                    self.gsa_cuda_topk.set_topk_param(
                        repre_slot_mappings,
                        include_masks,
                        exclude_masks,
                    )
                self.gsa_cuda_topk.set_topk_caches(
                    cal_topk_id, self.model_input["topk_caches"], topk_len_list
                )
            else:
                self.gsa_offload_ops.set_topk_param(repre_slot_mappings_all)
                self.gsa_offload_ops.set_topk_cache(
                    self.model_input["topk_caches"], topk_len_list
                )

    def allocate_slots(self, kv_cache_manager, request, num_slots_sparsed):
        coordinator = kv_cache_manager.coordinator
        block_pool = kv_cache_manager.block_pool
        kv_cache_groups = kv_cache_manager.kv_cache_config.kv_cache_groups
        if (
            request.num_prompt_tokens + 1 == request.num_tokens
            and request.num_tokens % self.block_size == 1
        ):
            num_blocks_need = math.ceil(num_slots_sparsed / self.block_size) - 1
        else:
            num_blocks_need = math.ceil(num_slots_sparsed / self.block_size)
        allocated_blocks = coordinator.get_blocks(request.request_id)[0]
        returned_blocks = []
        kept_blocks = []
        num_blocks_original = len(allocated_blocks)
        init_windows_size = gsa_config.init_windows_size
        for i, block in enumerate(allocated_blocks):
            if (
                i >= num_blocks_original - num_blocks_need + init_windows_size
                or i < init_windows_size
            ):
                kept_blocks.append(block)
            else:
                returned_blocks.append(block)
                block.ref_cnt = 1
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
