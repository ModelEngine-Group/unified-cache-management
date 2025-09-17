# Copyright (c) Huawei Technologies,. Ltd 2004-2025. All rights reserved

import torch
import math
import numpy as np
from typing import List, Optional, Tuple, Dict

from ucm.ucm_sparse.utils import (MAX_TOPK_LEN, compute_topk_len, 
                                           SEG_PREFILL_THRESHOLD, 
                                           PTOPK_PREFETCH_ENABLE, 
                                           VLLM_CUDA_MEM_ALIGN_KV_CACHE, 
                                           align_to_256bytes, LOCAL_WINDOW_SZ)

from vllm.sequence import SequenceStage


from vllm.utils import  is_pin_memory_available
from vllm.config import VllmConfig

from ucm.ucm_sparse import gsa_prefetch

class GSAPrefetchBase:
    def __init__(
        self,
        vllm_config: VllmConfig,
        async_thread: int,
        is_log: bool,
        is_cpu_topk: bool = False,
        is_max_norm: bool = False,
        max_norm_num: int = 1,
        is_prefetch: Optional[bool] = True,
        head_num: Optional[int] = None,
        is_mutli_head: Optional[bool] = None,
    ) -> None:
        self.is_cpu_topk = is_cpu_topk
        self.is_max_norm = is_max_norm
        self.async_thread = async_thread
        self.use_mla = vllm_config.model_config.use_mla
        self.is_prefetch = is_prefetch
        self.num_attention_layers = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.max_bs = vllm_config.scheduler_config.max_num_seqs
        self.is_log = is_log
        self.max_block_len = math.ceil(vllm_config.model_config.max_model_len / vllm_config.cache_config.block_size)
        self.block_size = vllm_config.cache_config.block_size
        self.device_config = vllm_config.device_config
        self.num_kv_heads = vllm_config.model_config.get_num_kv_heads(vllm_config.parallel_config)
        self.head_size = vllm_config.model_config.get_head_size()
        self.dtype = vllm_config.model_config.dtype
        self.align_cache = vllm_config.model_config.use_mla and VLLM_CUDA_MEM_ALIGN_KV_CACHE
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        
        self.sp_max_len = self.max_block_len
        if self.is_max_norm:
            self.kpre_shape = (self.max_bs * self.max_block_len, 1, self.num_kv_heads, self.head_size)
        else:
            self.kpre_shape = (self.max_bs * self.max_block_len, max_norm_num, self.num_kv_heads, self.head_size)
        self.topk_shape = (self.num_attention_layers, self.max_bs, self.max_block_len)
        if self.is_cpu_topk:
            self.kpre_caches, self.use_topk_caches = self._init_kpre_and_topk_cache("cpu", torch.float32, torch.int32)
        else:
            self.kpre_caches, self.use_topk_caches = self._init_kpre_and_topk_cache(self.device_config.device, self.dtype, torch.int64)
        self._init_tensor()
        self.prefetch_engine_c = gsa_prefetch.GSAPrefetchEngineC(self.prefetch_blocks,
                                                      self.m_load_success_list,
                                                      self.prefetch_block_len,
                                                      self.block_table_len, self.is_log)
        
        self.prefetch_space = 0
        self.num_token = 0
        self.step_time = 0
        self.is_prefetch_flag = False
        self.is_topk_cal = False
        self.select_bs_index = None
        self.open_gsa = True
        self.atb_gsa_enable = True
        self.ptopk_prefetch_enable = True
        self.req_ids_bs = []

        self.block_map_flag = {}
        self.block_table_flag = {}

        self.is_mutli_head = is_mutli_head
        self.head_num = head_num
        self.atten_score = []

        self.is_gsa_req_id = {}
        self.min_gsa_len = math.ceil(SEG_PREFILL_THRESHOLD / self.block_size)

        self.topk_buf_tmp = None
        self.topk_bs = []
    
    def model_input_del(
        self,
        req_ids,
        block_table_ori,
        topk_kpre_maps,
        gsa_model_input,
        gsa_metadata,
        is_topk_done,
    ) -> None:
        self.step_time += 1
        self.select_bs_index = topk_kpre_maps
        self.block_table_list_bs = block_table_ori
        self.req_ids_bs = req_ids
        self._get_run_type(gsa_metadata)
        self._set_req_stat(gsa_metadata)

        if self.atb_gsa_enable:
            block_table_index = torch.tensor(self.select_bs_index, device="cpu")
            self.topk_len = compute_topk_len(self._get_max_block_len())
            topk_buf_tmp = self.use_topk_caches[:, block_table_index.cpu(), :]
            topk_buf_tmp = topk_buf_tmp[:, :, :self.topk_len]
            self._topk_insert_last_idx(gsa_metadata)
            if self.ptopk_prefetch_enable:
                self._first_topk_deal(gsa_metadata)
                self._gsa_block_len_pre(gsa_metadata)
            else:
                self._no_gsa_input_deal(gsa_metadata)
            block_table_tmp = self.use_block_table[:, block_table_index, :].to(self.device_config.device)
            gen_len_tmp = self.gsa_seq_len[:, self.select_bs_index].to(self.device_config.device)

            list_topk_buf = list(topk_buf_tmp.unbind(dim=0))
            list_block_table = list(block_table_tmp.unbind(dim=0))
            gsa_len_list = list(gen_len_tmp.unbind(dim=0))
            self.is_topk_cal = is_topk_done and self.prefetch_space == 3
            gsa_model_input["topk_caches"] = list_topk_buf
            gsa_model_input["kpre_caches"] = self.kpre_caches
            gsa_model_input["is_topk"] = self.is_topk_cal
            gsa_model_input["block_tables_mp"] = list_block_table
            gsa_model_input["gsa_seq_len"] = gsa_len_list

            if self.is_topk_cal:
                for index, topk_info in enumerate(self.topk_bs):
                    if topk_info[1]: 
                        if topk_info[0] in gsa_metadata.gsa_stats:
                            gsa_metadata.gsa_stats[topk_info[0]].topk_buf_tmp = \
                                self.topk_buf_tmp[:, index, :topk_info[2]].clone()
                self.topk_bs = []
                for index, req_id in enumerate(self.req_ids_bs):
                    one_block_len = len(gsa_metadata.gsa_stats[req_id].blocks)
                    one_topk_len = compute_topk_len(one_block_len)
                    self.topk_bs.append([req_id, gsa_metadata.gsa_stats[req_id].stage() \
                                         != SequenceStage.PREFILL, one_topk_len])
                self.topk_buf_tmp = topk_buf_tmp
            
        gsa_model_input["atb_gsa_enable"] = self.atb_gsa_enable
        
    def deal_async_prefetch(
        self,
        rank,
        gsa_metadata,
    ) -> None:        
        if self.atb_gsa_enable:
            if self.ptopk_prefetch_enable:
                if self.prefetch_space >= 5:
                    tmp = self.use_block_table
                    self.use_block_table = self.m_load_success_list
                    self.m_load_success_list = tmp

                    tmp = self.use_block_table_len
                    self.use_block_table_len = self.block_table_len
                    self.block_table_len = tmp

                    self._swap_block_table_tensor(self.select_bs_index, gsa_metadata)
                    self.prefetch_engine_c.set_blocks_table_info(
                        self.m_load_success_list,
                        self.block_table_len,
                        self.prefetch_topk_buf[:, :len(self.select_bs_index), :],
                        self.step_time)

                    topk_len_list = []
                    req_id_list = []
                    for req_id in self.req_ids_bs:
                        req_id_list.append(int(req_id))
                        if not self.is_gsa_req_id[req_id]:
                            topk_len_list.append(0)
                            continue
                        else:
                            if gsa_metadata.gsa_stats[req_id].topk_buf_tmp != None:
                                topk_len_list.append(len(gsa_metadata.gsa_stats[req_id].topk_buf_tmp[0]))
                            else:
                                topk_len_list.append(0)
                    self.prefetch_engine_c.run_async_prefetch_bs(req_id_list, topk_len_list,
                                                                 self.select_bs_index, rank)
                    self.is_prefetch_flag = True
                    self.prefetch_space = 0
                else:
                    self.prefetch_space += 1
            self.num_token += 1

    def del_finish_meta(
        self,
        del_req
    ) -> None:
        if del_req in self.block_map_flag:
            del self.block_map_flag[del_req]
        if del_req in self.block_table_flag:
            del self.block_table_flag[del_req]
        if del_req in self.is_gsa_req_id:
            del self.is_gsa_req_id[del_req]
        if self.ptopk_prefetch_enable:
            self.prefetch_engine_c.del_blocks_map(int(del_req))
    
    def _init_tensor(self):
        device = "cpu"
        self.prefetch_blocks = torch.zeros((self.num_attention_layers, self.max_bs, int(self.sp_max_len)),
                                         dtype=torch.int32,
                                         pin_memory=is_pin_memory_available(),
                                         device=device)
        self.m_load_success_list = torch.zeros((self.num_attention_layers, self.max_bs, int(self.sp_max_len)),
                                         dtype=torch.int32,
                                         pin_memory=False,
                                         device=device)
        self.use_block_table = torch.zeros((self.num_attention_layers, self.max_bs, int(self.sp_max_len)),
                                         dtype=torch.int32,
                                         pin_memory=False,
                                         device=device)
        self.prefetch_block_len = torch.zeros((self.num_attention_layers, self.max_bs),
                                         dtype=torch.int32,
                                         pin_memory=is_pin_memory_available(),
                                         device=device)
        self.gsa_seq_len = torch.zeros((self.num_attention_layers, self.max_bs),
                                         dtype=torch.int32,
                                         pin_memory=is_pin_memory_available(),
                                         device=device)
        self.block_table_len = torch.zeros((self.num_attention_layers, self.max_bs),
                                         dtype=torch.int32,
                                         pin_memory=is_pin_memory_available(),
                                         device=device)
        self.use_block_table_len = torch.zeros((self.num_attention_layers, self.max_bs),
                                         dtype=torch.int32,
                                         pin_memory=is_pin_memory_available(),
                                         device=device)
        self.prefetch_topk_buf = torch.zeros((self.num_attention_layers, self.max_bs, int(self.sp_max_len)),
                                         dtype=torch.int64,
                                         pin_memory=is_pin_memory_available(),
                                         device=device)
        
    def _init_kpre_and_topk_cache(
        self,
        device,
        krepre_type,
        topk_type
    ) -> Tuple[List[torch.tensor], torch.tensor]:
        kpre_caches = []
        pin_memory = is_pin_memory_available() if device == "cpu" else False

        use_topk_caches = torch.zeros(
                                self.topk_shape,
                                dtype=topk_type,
                                pin_memory=pin_memory,
                                device=device)
        for _ in range(self.num_attention_layers):
            if self.align_cache:
                entry_shape = self.kpre_shape[2:]
                entry_size = np.prod(entry_shape)
                alloc_entry_size = align_to_256bytes(entry_size, krepre_type)
                alloc_shape = (*self.kpre_shape[:2], alloc_entry_size)
            else:
                alloc_shape = self.kpre_shape
            one_kpre_value = torch.zeros(
                alloc_shape,
                dtype=krepre_type,
                pin_memory=pin_memory,
                device=device)
            if self.align_cache:
                one_kpre_value = one_kpre_value[..., :entry_size]
            kpre_caches.append(one_kpre_value)
        
        return kpre_caches, use_topk_caches
    
    def _first_topk_deal(
        self, 
        gsa_metadata
    ) -> None:
        for index, req_id in enumerate(self.req_ids_bs):
            if gsa_metadata.gsa_stats[req_id].remain_idx != None:
                bs_index = self.select_bs_index[index]
                real_length = len(gsa_metadata.gsa_stats[req_id].blocks)
                block_table_list = self.block_table_list_bs[index][:real_length]
                remain_index = gsa_metadata.gsa_stats[req_id].remain_idx
                prefetch_idx = gsa_metadata.gsa_stats[req_id].prefetch_idx
                assert len(remain_index) < self.sp_max_len
                
                self.prefetch_block_len[:, bs_index] = len(prefetch_idx)
                self.block_table_len[:, bs_index] = len(remain_index)
                self.use_block_table_len[:, bs_index] = len(remain_index)

                prefetch_blocks_list = [block_table_list[x] for x in prefetch_idx]
                self.prefetch_blocks[:, bs_index, :len(prefetch_blocks_list)] = torch.tensor(prefetch_blocks_list, dtype=torch.int32)
                topk_block_list = [block_table_list[x] for x in remain_index]
                topk_block_tensor = torch.tensor(topk_block_list, dtype=torch.int32, device="cpu")

                if gsa_metadata.gsa_stats[req_id].num_prompt_tokens <= SEG_PREFILL_THRESHOLD:
                    block_table_list_input = block_table_list
                else:
                    block_table_list_input = [x for x in block_table_list]
                    remain_all_len = len(prefetch_idx) + len(remain_index)
                    block_table_list_input[-1 * LOCAL_WINDOW_SZ:] = \
                        block_table_list[remain_all_len - LOCAL_WINDOW_SZ:remain_all_len]
                    
                    block_table_list_input[remain_all_len - LOCAL_WINDOW_SZ:real_length - LOCAL_WINDOW_SZ] = \
                        block_table_list[remain_all_len - real_length:]
                    
                    remain_index[-1 * LOCAL_WINDOW_SZ:] = list(range(real_length))[-1 * LOCAL_WINDOW_SZ:]
                input_idxs = prefetch_idx + remain_index
                self.use_block_table[:, bs_index, :len(topk_block_list)] = topk_block_tensor
                self.m_load_success_list[:, bs_index, :len(topk_block_list)] = topk_block_tensor
                if self.is_gsa_req_id[req_id]:
                    self.prefetch_engine_c.set_blocks_map(int(req_id), block_table_list_input,
                                                          input_idxs)
    
    def _gsa_block_len_pre(
        self,
        gsa_metadata,
    ) -> None:
        self.gsa_seq_len.copy_(self.use_block_table_len)
        for index, req_id in enumerate(self.req_ids_bs):
            bs_index = self.select_bs_index[index]
            remain_slot = gsa_metadata.gsa_stats[req_id].get_seq_len() % self.block_size
            if gsa_metadata.gsa_stats[req_id].stage() == SequenceStage.DECODE:
                if remain_slot == 0:
                    self.gsa_seq_len[:, bs_index].mul_(self.block_size)
                elif remain_slot == 1:
                    self.gsa_seq_len[:, bs_index].mul_(self.block_size).add_(remain_slot)
                    last_block = gsa_metadata.gsa_stats[req_id].blocks[-1]
                    for layer_id in range(self.num_attention_layers):
                        indices = self.use_block_table_len[layer_id][bs_index].item()
                        assert indices < self.sp_max_len
                        self.use_block_table[layer_id][bs_index][indices] = last_block
                        self.use_block_table_len[layer_id][bs_index].add_(1)
                    if req_id not in self.block_table_flag:
                        self.block_map_flag[req_id] = []
                        self.block_table_flag[req_id] = []
                    self.block_table_flag[req_id].append(last_block)
                    self.block_map_flag[req_id].append([len(gsa_metadata.gsa_stats[req_id].blocks) - 1, last_block])
                else:
                    self.gsa_seq_len[:, bs_index].add_(-1).mul_(self.block_size).add_(remain_slot)
            else:
                self.block_map_flag[req_id] = []
                self.block_table_flag[req_id] = []
                self.gsa_seq_len[:, bs_index] = gsa_metadata.gsa_stats[req_id].get_seq_len()
                self.use_block_table[:, bs_index, :len(gsa_metadata.gsa_stats[req_id].blocks)] = \
                    torch.tensor(gsa_metadata.gsa_stats[req_id].blocks, dtype=torch.int32, device="cpu")
    
    def _topk_insert_last_idx(
        self,
        gsa_metadata
    ) -> None:
        for index in range(len(self.req_ids_bs)):
            req_id = self.req_ids_bs[index]
            if gsa_metadata.gsa_stats[req_id].topk_buf_tmp != None:
                last_idx = len(gsa_metadata.gsa_stats[req_id].blocks) - 1
                if last_idx not in gsa_metadata.gsa_stats[req_id].topk_buf_tmp:
                     gsa_metadata.gsa_stats[req_id].topk_buf_tmp = torch.nn.functional.pad(
                         gsa_metadata.gsa_stats[req_id].topk_buf_tmp, (0, 1), value=last_idx)

    def _swap_block_table_tensor(
        self,
        bs_index_list: List[int],
        gsa_metadata,
    ) -> None:
        for index, bs_index in enumerate(bs_index_list):
            req_id = self.req_ids_bs[index]
            if req_id in self.block_map_flag:
                for block_mp_add in self.block_map_flag[req_id]:
                    self.prefetch_engine_c.add_blocks_map(int(req_id), block_mp_add[0], block_mp_add[1])
                self.block_map_flag[req_id].clear()

            if req_id in self.block_table_flag.keys():
                for block_table_add in self.block_table_flag[req_id]:
                    for layer_id in range(self.num_attention_layers):
                        indices = self.use_block_table_len[layer_id][bs_index].item()
                        assert indices < self.sp_max_len
                        self.use_block_table[layer_id][bs_index][indices] = block_table_add
                        self.use_block_table_len[layer_id][bs_index].add_(1)
                self.block_table_flag[req_id].clear()
            if gsa_metadata.gsa_stats[req_id].topk_buf_tmp != None:
                self.prefetch_topk_buf[:, index, :len(gsa_metadata.gsa_stats[req_id].topk_buf_tmp[0])].copy_(
                    gsa_metadata.gsa_stats[req_id].topk_buf_tmp)

    def _get_run_type(
        self,
        gsa_metadata,
    ) -> None:
        self.open_gsa = False
        for req_id in self.req_ids_bs:
            if gsa_metadata.gsa_stats[req_id].stage() == SequenceStage.DECODE:
                self.open_gsa = True
                break
        if self.open_gsa:
            self.atb_gsa_enable = True
            self.ptopk_prefetch_enable = True and PTOPK_PREFETCH_ENABLE
        else:
            self.atb_gsa_enable = False
            self.ptopk_prefetch_enable = False

    def _set_req_stat(
        self,
        gsa_metadata,
    ) -> None:
        for req_id in self.req_ids_bs:
            if req_id in self.is_gsa_req_id.keys():
                if gsa_metadata.gsa_stats[req_id].stage() != SequenceStage.PREFILL:
                    if gsa_metadata.gsa_stats[req_id].is_gsa():
                        self.is_gsa_req_id[req_id] = True
            else:
                if gsa_metadata.gsa_stats[req_id].is_gsa():
                    self.is_gsa_req_id[req_id] = True
                else:
                    self.is_gsa_req_id[req_id] = False
    
    def _get_max_block_len(self) -> int:
        max_len = 0
        for blocks in self.block_table_list_bs:
            max_len = max(max_len, len(blocks))
        return max_len
    
    def _no_gsa_input_deal(
        self,
        gsa_metadata,
    ) -> None:
        for index, req_id in enumerate(self.req_ids_bs):
            bs_index = self.select_bs_index[index]
            one_block_table = torch.tensor(self.block_table_list_bs[index],
                                           dtype=torch.int32, device="cpu")
            if self.is_gsa_req_id[req_id] and gsa_metadata.gsa_stats[req_id].topk_buf_tmp != None:
                remain_slot = gsa_metadata.gsa_stats[req_id].get_seq_len() % self.block_size
                one_topk_len = len(gsa_metadata.gsa_stats[req_id].topk_buf_tmp[0])
                for layer_id in range(self.num_attention_layers):
                    self.use_block_table[layer_id][bs_index][:one_topk_len] = \
                        one_block_table[gsa_metadata.gsa_stats[req_id].topk_buf_tmp[layer_id]]
                self.gsa_seq_len[:, bs_index].fill_(0)
                if remain_slot == 0:
                    self.gsa_seq_len[:, bs_index].add_(one_topk_len * self.block_size)
                else:
                    self.gsa_seq_len[:, bs_index].add_(one_topk_len * self.block_size - self.block_size + remain_slot)
            else:
                self.gsa_seq_len[:, bs_index] = gsa_metadata.gsa_stats[req_id].get_seq_len()
                self.use_block_table[:, bs_index, :len(gsa_metadata.gsa_stats[req_id].blocks)] = one_block_table
    