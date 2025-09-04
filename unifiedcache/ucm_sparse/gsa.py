import math
import time
from dataclasses import dataclass
from functools import wraps
from typing import Dict, List, Union

import torch
from vllm.config import VllmConfig
from vllm.forward_context import ForwardContext
from vllm.sequence import SequenceStage
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request

from unifiedcache.integration.vllm.ucm_sparse.base import (
    INVALID_SLOT,
    UcmSparseBase,
    UcmSparseMetadata,
    UcmSparseRole,
)
from unifiedcache.ucm_sparse.prefetch_engine import GSAPrefetchBase
from unifiedcache.ucm_sparse.utils import (MAX_TOPK_LEN, compute_topk_len,
                                           SEG_PREFILL_THRESHOLD, LOCAL_WINDOW_SZ)
from unifiedcache.ucm_connector.base import Task, UcmKVStoreBase
from unifiedcache.ucm_connector.factory import UcmConnectorFactory
from vllm.utils import make_tensor_with_pad
import os
import sys
from vllm.forward_context import set_forward_context, get_forward_context
import copy

path = os.getcwd()
sys.path.append(os.path.join(path, 'unifiedcache/csrc/gsaoffloadops'))
import gsa_offload_ops as GsaOffloadOps



def stat(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.call_count += 1
        start = time.perf_counter_ns()
        result = func(*args, **kwargs)
        end = time.perf_counter_ns()
        cost = end - start
        wrapper.time_costs.append(cost)
        return result

    wrapper.call_count = 0
    wrapper.time_costs = []
    return wrapper


ReqType = Union[str, int]
HashType = Union[str, int]

# TODO: add ESA specific config in kv_transfer_config -> extra_config
INIT_WINDOW_SZ = 1
SPARSE_RATIO = 0.3
RETRIEVAL_STRIDE = 4


class GSAReqStat:
    def __init__(
        self,
        req_id,
    ) -> None:
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

    def step(self) -> int:
        return self.num_output_tokens
    
    def stage(self) -> SequenceStage:
        return (
            SequenceStage.DECODE
            if self.num_prompt_tokens <= self.num_computed_tokens
            else SequenceStage.PREFILL
        )
    
    def is_gsa(self) -> bool:
        return ( self.num_prompt_tokens > SEG_PREFILL_THRESHOLD 
                and self.stage() != SequenceStage.PREFILL
        )
    
    def is_last_chunk(self) -> bool:
        return (
            self.num_computed_tokens + self.num_scheduled_tokens 
            >= self.num_prompt_tokens 
        ) 
    
    def _init_slot(self, offset: int) -> None:
        self.repre_slot_mapping = list(range(len(self.blocks)))
        self.repre_slot_mapping = [x + offset for x in self.repre_slot_mapping]

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
    
    def get_seq_len(self) -> int:
        return self.num_computed_tokens + self.num_scheduled_tokens

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
            self.calc_block_table = [x for x in add_blocks]
            self.calc_repre_slot_mapping = self.repre_slot_mapping[add_len * -1 :]
        else:
            self.calc_block_table = []
            self.calc_repre_slot_mapping = []
    
    def add_req_new(
        self,
        num_scheduled_tokens,
        add_req_state,
        index_in_batch,
        offset
    ) -> None:
        self.blocks = [x for x in add_req_state.block_ids[0]]
        self.index_in_batch = index_in_batch
        self._init_slot(offset)
        self.num_computed_tokens = add_req_state.num_computed_tokens
        self.num_scheduled_tokens = num_scheduled_tokens
        self.num_prompt_tokens = len(add_req_state.prompt_token_ids)
        self.num_output_tokens = len(add_req_state.output_token_ids)
        self.is_use_gsa = True if self.num_prompt_tokens > SEG_PREFILL_THRESHOLD else False
    
    def updata_req_state(
        self,
        num_scheduled_tokens,
        add_req_state
    ) -> None:
        self.num_computed_tokens = add_req_state.num_computed_tokens
        self.num_scheduled_tokens = num_scheduled_tokens
        self.num_output_tokens = len(add_req_state.output_token_ids)
        if self.stage() == SequenceStage.PREFILL:
            add_blocks = [x for x in add_req_state.block_ids[0] if x not in self.blocks]
            self.blocks = [x for x in add_req_state.block_ids[0]]
            self._update_slot(add_blocks)
        else:
            self._get_sparse_and_free_block()
            if len(add_req_state.block_ids[0]) != self.sparse_len:
                add_blocks = [add_req_state.block_ids[0][-1]]
                self._update_slot(add_blocks)
                self.blocks += add_blocks
                self.calc_block_table.append(self.blocks[-2])
                self.calc_repre_slot_mapping.append(self.repre_slot_mapping[-2])
                self.sparse_len = len(add_req_state.block_ids[0])
            else:
                self.calc_block_table = []
                self.calc_repre_slot_mapping = []

    def _get_sparse_and_free_block(self):
        if self.num_prompt_tokens == self.num_computed_tokens:
            blocks_len = len(self.blocks)
            if self.num_prompt_tokens > SEG_PREFILL_THRESHOLD:
                remain_len = compute_topk_len(blocks_len)
                prefetch_len = MAX_TOPK_LEN - remain_len + 1
                self.remain_idx = []
                self.prefetch_idx = []
                for i in range(blocks_len):
                    if len(self.remain_idx) < (remain_len):
                        self.remain_idx.append(i)
                    elif len(self.prefetch_idx) < prefetch_len:
                        self.prefetch_idx.append(i)
                self.sparse_len = remain_len + prefetch_len
            else:
                self.remain_idx = list(range(blocks_len))
                self.prefetch_idx = []
                self.sparse_len = blocks_len
            return
        else:
            self.remain_idx = None
            self.prefetch_idx = None
        
class GSAMetaData(UcmSparseMetadata):
    def __init__(
        self,
        block_size,
        device,
    ):
        self.gsa_stats = {}
        self.block_size = block_size
        self.device = device

    def get_model_input(
        self,
        scheduler_output: SchedulerOutput,
        topk_kpre_map,
        max_block_len,
        requests,
        input_batch
    ) -> None:
        for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
            assert req_id in self.gsa_stats
            self.gsa_stats[req_id].updata_req_state(
                scheduler_output.num_scheduled_tokens[req_id],
                requests[req_id])
        for new_req in scheduler_output.scheduled_new_reqs:
            self.gsa_stats[new_req.req_id] = GSAReqStat(new_req.req_id)
            self.gsa_stats[new_req.req_id].add_req_new(
                scheduler_output.num_scheduled_tokens[new_req.req_id],
                requests[new_req.req_id],
                input_batch.req_id_to_index[new_req.req_id],
                max_block_len * topk_kpre_map[new_req.req_id]
            )
        return self.trans_input_tensor(scheduler_output)

    def trans_input_tensor(self, scheduler_output: SchedulerOutput):
        repre_slot_mapping = []
        calc_block_table = []
        calc_repre_slot_mapping = []
        include_mask = []
        exclude_mask = []
        model_input = {}
        for req_id, _ in scheduler_output.num_scheduled_tokens.items():
            repre_slot_mapping.append(self.gsa_stats[req_id].repre_slot_mapping)
            include_mask.append(self.gsa_stats[req_id].include_mask)
            exclude_mask.append(self.gsa_stats[req_id].exclude_mask)
            calc_block_table += self.gsa_stats[req_id].calc_block_table
            calc_repre_slot_mapping += self.gsa_stats[req_id].calc_repre_slot_mapping

        model_input["include_mask"] = make_tensor_with_pad(
            include_mask,
            pad=False,
            dtype=torch.uint8,
            device=self.device
        )
        model_input["exclude_mask"] = make_tensor_with_pad(
            exclude_mask,
            pad=True,
            dtype=torch.uint8,
            device=self.device
        )
        model_input["repre_slot_mapping"] = make_tensor_with_pad(
            repre_slot_mapping,
            pad=0,
            dtype=torch.int32,
            device=self.device
        )
        model_input["calc_block_table"] = torch.tensor(
            calc_block_table,
            dtype=torch.int32,
            device=self.device
        )
        model_input["calc_repre_slot_mapping"] = torch.tensor(
            calc_repre_slot_mapping,
            dtype=torch.int32,
            device=self.device
        )
        return model_input

@dataclass
class ReqMeta:
    request_id: ReqType
    index_in_batch: int
    num_prompt_tokens: int
    num_output_tokens: int
    num_scheduled_tokens: int
    num_computed_tokens: int
    num_sparsed_tokens: int
    vllm_block_ids: list[int]

    @property
    def step(self) -> int:
        return self.num_output_tokens

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
class TopkSparseMetaData(UcmSparseMetadata):
    requests: list[ReqMeta]

    def __init__(self):
        self.requests = []

    def add_request(
        self,
        request_id: ReqType,
        index_in_batch: int,
        num_prompt_tokens: int,
        num_output_tokens: int,
        num_scheduled_tokens: int,
        num_computed_tokens: int,
        num_sparsed_tokens: int,
        vllm_block_ids: list[int],
    ) -> None:
        meta = ReqMeta(
            request_id=request_id,
            index_in_batch=index_in_batch,
            num_prompt_tokens=num_prompt_tokens,
            num_output_tokens=num_output_tokens,
            num_scheduled_tokens=num_scheduled_tokens,
            num_computed_tokens=num_computed_tokens,
            num_sparsed_tokens=num_sparsed_tokens,
            vllm_block_ids=vllm_block_ids,
        )
        self.requests.append(meta)

class TopKAndKpreManger:
    def __init__(
        self,
        max_num: int):
        self.cache_map = {}
        self.max_num = max_num
        self.free_cache = []
        for i in range(max_num):
            self.free_cache.append(i)

    def free(self, 
             req_id: ReqType
    ) -> bool:
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

class GSA(UcmSparseBase):
    # handle batch
    def __init__(self, vllm_config: VllmConfig, role: UcmSparseRole):
        super().__init__(vllm_config, role)
        self.rank = vllm_config.parallel_config.rank
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.device = vllm_config.device_config.device_type
        self.num_key_heads = vllm_config.model_config.get_num_kv_heads(vllm_config.parallel_config)
        self.head_size = vllm_config.model_config.get_head_size()
        config = {"max_cache_size": 5368709120, "device": self.rank, "role": "worker"}
        self.connector = UcmConnectorFactory.create_connector("UcmDram", config)
        self.prefetch_engine = GSAPrefetchBase(vllm_config, 16, True, True, False, 1)
        self.topk_kpre_manger = TopKAndKpreManger(vllm_config.scheduler_config.max_num_seqs)
        self.block_size = vllm_config.cache_config.block_size
        self.k_cache = {}
        self.v_cache = {}
        self.tasks_dump = {}
        self.tasks_load = {}
        self.gsa_metadata = None
        self.model_input = None
        self.gsa_stats = {}
        self._sparse_metadata = None
        self.init_topk_cal(vllm_config, self.prefetch_engine)
    
    def init_topk_cal(
        self,
        vllm_config: VllmConfig,
        prefetch_engine: GSAPrefetchBase,
    ) -> None:
        parallel_config = vllm_config.parallel_config
        self.layer_num = vllm_config.model_config.get_num_layers_by_block_type(parallel_config)
        block_size = vllm_config.cache_config.block_size
        att_num_heads = vllm_config.model_config.get_num_attention_heads(parallel_config)
        kv_num_heads = vllm_config.model_config.get_num_kv_heads(parallel_config)
        head_size = vllm_config.model_config.get_head_size()
        max_model_len = vllm_config.model_config.max_model_len
        self.gsa_offload_ops = GsaOffloadOps.CalKpreAndTopk(self.layer_num, block_size,
            vllm_config.scheduler_config.max_num_seqs, att_num_heads, head_size)
        self.gsa_offload_ops.set_kpre_method_param(int(max_model_len / block_size), kv_num_heads, 1)
        self.gsa_offload_ops.set_kpre_cache(prefetch_engine.kpre_caches)
    
    @classmethod
    def req_state_hash(cls, req_id, layer_name):
        return hash((req_id, layer_name))

    @classmethod
    def block_hash(cls, request_id, block_id):
        return f"req_{request_id}_blk_{block_id}"

    def launch_transfer_task(self, transfer_type, block_hashes, vllm_block_ids, layer_id):
        fn = getattr(self.connector, transfer_type)
        length = len(block_hashes)
        block_shape = (self.block_size, self.num_key_heads, self.head_size)
        precision = self.k_cache[layer_id].untyped_storage().element_size()
        # TODO: consider is_mla here
        is_mla = False
        offsets_k = [
            get_offset(
                block_shape,
                self.rank,
                self.tp_size,
                precision,
                layer_id,
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
                layer_id,
                is_v=True,
                is_mla=is_mla,
            )
        ] * length
        key_src_tensors = [self.k_cache[layer_id][id_] for id_ in vllm_block_ids]
        value_src_tensors = [self.v_cache[layer_id][id_] for id_ in vllm_block_ids]
        task_k = fn(block_hashes, offsets_k, key_src_tensors)
        task_v = fn(block_hashes, offsets_v, value_src_tensors)
        task_k_hash = self.task_hash(block_hashes, transfer_type, "key")
        task_v_hash = self.task_hash(block_hashes, transfer_type, "value")
        if transfer_type == "dump":
            self.tasks_dump[task_k_hash] = task_k
            self.tasks_dump[task_v_hash] = task_v
        if transfer_type == "load":
            self.tasks_load[task_k_hash] = task_k
            self.tasks_load[task_v_hash] = task_v

    @classmethod
    def task_hash(cls, block_ids, store_type, tensor_type):
        return hash((tuple(block_ids), store_type, tensor_type))
    
    def copy_q(
        self,
        req_meta: ReqMeta,
        query: torch.Tensor,
        layer_name: str,
        forward_context: ForwardContext,
    ) -> None:
        current_layer_id = int(layer_name.split(".")[2])
        if isinstance(forward_context.attn_metadata, dict):
            attn_metadata = forward_context.attn_metadata[layer_name]
        else:
            attn_metadata = forward_context.attn_metadata
        index_in_batch = req_meta.index_in_batch
        query_start_loc = attn_metadata.query_start_loc[index_in_batch]
        query_len = req_meta.num_scheduled_tokens
        current_query = query[query_start_loc: query_start_loc + query_len]
        last_query = current_query[-1].cpu()
        temp = last_query.to(torch.float32)
        self.gsa_offload_ops.q_cache[current_layer_id][index_in_batch] = temp

    def attention_begin(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_name: str,
        forward_context: ForwardContext,
    ) -> None:
        current_layer_id = int(layer_name.split(".")[2])
        if self.prefetch_engine.atb_gsa_enable and self.prefetch_engine.is_topk_cal:
            for req_meta in self._sparse_metadata.requests:
                if req_meta.stage == SequenceStage.DECODE:
                    self.copy_q(req_meta, query, layer_name, forward_context)
            self.gsa_offload_ops.set_topk_data_ready(current_layer_id)
        
        if isinstance(forward_context.attn_metadata, dict):
            attn_metadata = forward_context.attn_metadata[layer_name]
            block_tables = attn_metadata.block_table
        else:
            attn_metadata = forward_context.attn_metadata
            block_tables = attn_metadata.block_tables
        if self.prefetch_engine.atb_gsa_enable:
            for index, req_meta in enumerate(self._sparse_metadata.requests):
                if not self.prefetch_engine.is_gsa_req_id[req_meta.request_id]:
                    continue
                if req_meta.stage == SequenceStage.DECODE:
                    index_in_batch = req_meta.index_in_batch
                    block_len = self.model_input["block_tables_mp"][current_layer_id][index].shape[0]
                    block_tables[index_in_batch].zero_()
                    block_tables[index_in_batch][:block_len].copy_(
                        self.model_input["block_tables_mp"][current_layer_id][index])
                    attn_metadata.seq_lens[index_in_batch].copy_(self.model_input["gsa_seq_len"][current_layer_id][index])

    def attention_finished(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_output: torch.Tensor,
        layer_name: str,
        forward_context: ForwardContext,
    ) -> None:
        self.maybe_register_kv_cache(forward_context, layer_name)
        current_layer_id = int(layer_name.split(".")[2])
        block_ids = self.model_input["calc_block_table"]
        if len(block_ids) > 0:
            attn = forward_context.no_compile_layers
            k_needed = attn[layer_name].kv_cache[forward_context.virtual_engine][0][block_ids].cpu()
            temp_k_cache = k_needed.to(torch.float32).permute(0, 2, 1, 3)
            self.gsa_offload_ops.k_cache[current_layer_id][:len(block_ids)] = temp_k_cache
            self.gsa_offload_ops.set_kpre_data_ready(current_layer_id)
        
        block_hashes = []
        block_ids = []
        for req_id in self.prefetch_engine.req_ids_bs:
            offset = self.prefetch_engine.max_block_len * self.topk_kpre_manger.cache_map[req_id]
            block_hashes += [f"{self.block_hash(req_id, id_ - offset)}" 
                                for id_ in self.gsa_metadata.gsa_stats[req_id].calc_repre_slot_mapping]
            block_ids += self.gsa_metadata.gsa_stats[req_id].calc_block_table
        self.launch_transfer_task("dump", block_hashes, block_ids, current_layer_id)
    
    def wait_all_task_done(self, transfer_type):
        if transfer_type == "dump":
            for task_hash, task in self.tasks_dump.items():
                ret = self.connector.wait(task)
            self.tasks_dump.clear()
        else:
            for task_hash, task in self.tasks_load.items():
                ret = self.connector.wait(task)
            self.tasks_load.clear()
    
    def maybe_register_kv_cache(
        self,
        forward_context: ForwardContext,
        layer_name
    ) -> None:
        current_layer_id = int(layer_name.split(".")[2])
        attn = forward_context.no_compile_layers[self.layer_name]
        kv_cache = attn.kv_cache[forward_context.virtual_engine]
        # TODO: consider is_mla here
        self.k_cache[current_layer_id] = kv_cache[0]
        self.v_cache[current_layer_id] = kv_cache[1]
        self.block_size = self.k_cache[current_layer_id].shape[1]
        self.num_key_heads = self.k_cache[current_layer_id].shape[2]
        self.head_size = self.k_cache[current_layer_id].shape[3]

    def build_gsa_metadata(
        self,
        scheduler_output: SchedulerOutput,
        requests,
        input_batch
    ) -> GSAMetaData:
        for req_id, _ in scheduler_output.num_scheduled_tokens.items():
            if not self.topk_kpre_manger.is_exist(req_id):
                index = self.topk_kpre_manger.alloc(req_id)
                assert index != None
        gsa_meta = GSAMetaData(self.block_size, self.device)
        gsa_meta.gsa_stats = self.gsa_stats
        self.model_input = gsa_meta.get_model_input(
            scheduler_output, self.topk_kpre_manger.cache_map,
            self.prefetch_engine.max_block_len,
            requests, input_batch)
        self.gsa_stats = gsa_meta.gsa_stats
        return gsa_meta
    
    def _start_topk_cal(self) -> None:
        cal_topk_id = []
        is_decode = []
        topk_len_list = []
        repre_slot_mappings = []
        calc_block_tables = []
        calc_repre_slot_mappings = []
        for req_meta in self._sparse_metadata.requests:
            if req_meta.stage == SequenceStage.DECODE:
                cal_topk_id.append(req_meta.index_in_batch)
                is_decode.append(True)
                one_topk_len = compute_topk_len(len(self.gsa_metadata.gsa_stats[req_meta.request_id].blocks))
                topk_len_list.append(one_topk_len)
            else:
                is_decode.append(False)
            repre_slot_mappings.append(self.gsa_metadata.gsa_stats[req_meta.request_id].repre_slot_mapping)
            calc_block_tables += self.gsa_metadata.gsa_stats[req_meta.request_id].calc_block_table
            calc_repre_slot_mappings += self.gsa_metadata.gsa_stats[req_meta.request_id].calc_repre_slot_mapping
        self.gsa_offload_ops.set_common_param(cal_topk_id, is_decode)
        if len(calc_block_tables) != 0:
            self.gsa_offload_ops.set_kpre_param(calc_block_tables, calc_repre_slot_mappings)
        if self.prefetch_engine.atb_gsa_enable and self.prefetch_engine.is_topk_cal:
            self.gsa_offload_ops.set_topk_param(repre_slot_mappings)
            self.gsa_offload_ops.set_topk_cache(self.model_input["topk_caches"], topk_len_list)

    def execute_begin(self, scheduler_output: SchedulerOutput):
        req_ids = []
        block_table_ori = []
        topk_kpre_maps = []
        for req_id, _ in scheduler_output.num_scheduled_tokens.items():
            req_ids.append(req_id)
            block_table_ori.append(self.gsa_metadata.gsa_stats[req_id].blocks)
            topk_kpre_maps.append(self.topk_kpre_manger.cache_map[req_id])
        is_topk_done = self.gsa_offload_ops.is_calculate_finish()
        self.prefetch_engine.model_input_del(req_ids, block_table_ori, 
                                             topk_kpre_maps, self.model_input,
                                             self.gsa_metadata, is_topk_done)
        self._start_topk_cal()

    def execute_finished(self):
        forward_context = get_forward_context()
        attn = forward_context.no_compile_layers

        all_need_load_block, all_miss_idx = self.prefetch_engine.deal_async_prefetch(
            self.rank, self.gsa_metadata)
        self.gsa_stats = self.gsa_metadata.gsa_stats
        self.wait_all_task_done("dump")
        self._gsa_sparse_local_kv()
        if all_need_load_block != None:
            self.wait_all_task_done("load")
            for layer_name in attn.keys():
                layer_id = int(layer_name.split(".")[2])
                self.k_cache[layer_id] = attn[layer_name].kv_cache[forward_context.virtual_engine][0]
                self.v_cache[layer_id] = attn[layer_name].kv_cache[forward_context.virtual_engine][1]
                block_hashes_load = []
                block_ids_load = []
                for index, req_id in enumerate(self.prefetch_engine.req_ids_bs):
                    load_len = len(all_need_load_block[index][layer_id])
                    block_hashes_load += [f"{self.block_hash(req_id, id_)}" 
                                for id_ in all_miss_idx[index][layer_id][:load_len]]
                    block_ids_load += all_need_load_block[index][layer_id]
                self.launch_transfer_task("load", block_hashes_load, block_ids_load, layer_id)
    
    def _gsa_sparse_local_kv(
        self,
    ) -> None:
        forward_context = get_forward_context()
        attn = forward_context.no_compile_layers
        for req_id in self.prefetch_engine.req_ids_bs:
            assert req_id in self.gsa_metadata.gsa_stats
            if self.gsa_metadata.gsa_stats[req_id].stage() == SequenceStage.PREFILL and \
                self.gsa_metadata.gsa_stats[req_id].is_last_chunk():
                local_blocks = self.gsa_metadata.gsa_stats[req_id].blocks[LOCAL_WINDOW_SZ * -1:]
                k_cache = {}
                v_cache = {}
                for layer_name in attn.keys():
                    k_cache[layer_name] = []
                    v_cache[layer_name] = []
                    for block in local_blocks:
                        k_cache[layer_name].append(attn[layer_name].kv_cache[forward_context.virtual_engine][0][block].clone())
                        v_cache[layer_name].append(attn[layer_name].kv_cache[forward_context.virtual_engine][1][block].clone())
                self.gsa_metadata.gsa_stats[req_id].local_window_kv.append(k_cache)
                self.gsa_metadata.gsa_stats[req_id].local_window_kv.append(v_cache)
    
    def _init_sparse_local_kv(
        self,
        scheduler_output: SchedulerOutput,
        requests
    ) -> None:
        forward_context = get_forward_context()
        attn = forward_context.no_compile_layers
        for req_id, _ in scheduler_output.num_scheduled_tokens.items():
            if req_id in self.gsa_metadata.gsa_stats and \
                self.gsa_metadata.gsa_stats[req_id].num_computed_tokens == self.gsa_metadata.gsa_stats[req_id].num_prompt_tokens:
                assert self.gsa_metadata.gsa_stats[req_id].remain_idx != None
                local_window = self.gsa_metadata.gsa_stats[req_id].remain_idx[LOCAL_WINDOW_SZ * -1:]
                req_blocks = requests[req_id].block_ids[0]
                local_blocks = [req_blocks[x] for x in local_window]
                for layer_name in attn.keys():
                    for index, block in enumerate(local_blocks):
                        attn[layer_name].kv_cache[forward_context.virtual_engine][0][block].copy_(
                            self.gsa_metadata.gsa_stats[req_id].local_window_kv[0][layer_name][index]
                        )
                        attn[layer_name].kv_cache[forward_context.virtual_engine][1][block].copy_(
                            self.gsa_metadata.gsa_stats[req_id].local_window_kv[1][layer_name][index]
                        )
                for i in range(LOCAL_WINDOW_SZ):
                    self.gsa_metadata.gsa_stats[req_id].remain_idx[i - LOCAL_WINDOW_SZ] = \
                        len(self.gsa_metadata.gsa_stats[req_id].blocks) - LOCAL_WINDOW_SZ + i

    def build_sparse_meta(
        self,
        scheduler_output: SchedulerOutput,
        requests,
        input_batch,
    ) -> None:
        sparse_meta = TopkSparseMetaData()
        for (
            req_id,
            num_scheduled_tokens,
        ) in scheduler_output.num_scheduled_tokens.items():
            req_state = requests[req_id]
            sparse_meta.add_request(
                req_id,
                input_batch.req_id_to_index[req_id],
                len(req_state.prompt_token_ids),
                len(req_state.output_token_ids),
                num_scheduled_tokens,
                req_state.num_computed_tokens,
                scheduler_output.req_sparsed_slots[req_id],
                req_state.block_ids[0]
            )
        self._sparse_metadata = sparse_meta
        self.gsa_metadata = self.build_gsa_metadata(scheduler_output, requests, input_batch)
        self._init_sparse_local_kv(scheduler_output, requests)

    def request_begin(self, request_id: ReqType, prompt_token_ids: List[int]):
        pass

    def request_finished_in_scheduler(self, request_id: ReqType):
        pass

    def request_finished_in_worker(self, request_id: ReqType):
        self.topk_kpre_manger.free(request_id)
        if request_id in self.gsa_stats:
            del self.gsa_stats[request_id]
        self.prefetch_engine.del_finish_meta(request_id)

    def update_state_after_alloc(self, request: Request, num_blocks: int):
        pass

    def estimate_num_slots_sparsed(self, request: Request) -> int:
        if request.num_output_tokens == 0:
            return INVALID_SLOT
        if request.num_prompt_tokens <= SEG_PREFILL_THRESHOLD:
            return INVALID_SLOT
        block_size = self._vllm_config.cache_config.block_size
        num_prompt_blocks = math.ceil(request.num_prompt_tokens / block_size)
        num_all_blocks = math.ceil(request.num_tokens / block_size)
        topk_len = compute_topk_len(num_prompt_blocks)
        prefetch_len = MAX_TOPK_LEN - topk_len + 1
        num_sparse_blocks = num_all_blocks - num_prompt_blocks + topk_len + prefetch_len
        flaw = request.num_tokens % block_size
        if flaw:
            flaw = block_size - flaw
        num_tokens_sparsed = num_sparse_blocks * block_size - flaw
        return num_tokens_sparsed