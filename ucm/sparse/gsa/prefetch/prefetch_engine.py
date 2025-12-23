# Copyright (c) Huawei Technologies,. Ltd 2004-2025. All rights reserved

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.sequence import SequenceStage
from vllm.utils import is_pin_memory_available

from ucm.sparse.gsa.prefetch import gsa_prefetch
from ucm.sparse.utils import (
    ENABLE_KVCOMP,
    MAX_BS,
    PTOPK_PREFETCH_ENABLE,
    VLLM_CUDA_MEM_ALIGN_KV_CACHE,
    align_to_256bytes,
    gsa_config,
)


class GSAPrefetchBase:
    def __init__(
        self,
        vllm_config: VllmConfig,
        async_thread: int,
        is_log: bool,
        is_cpu_topk: bool = False,
        is_max_norm: bool = False,
        max_norm_num: int = 1,
        is_python_load: bool = False,
        is_prefetch: Optional[bool] = True,
        head_num: Optional[int] = None,
        is_mutli_head: Optional[bool] = None,
    ) -> None:
        self.rank = vllm_config.parallel_config.rank
        self.is_cpu_topk = is_cpu_topk
        self.is_max_norm = is_max_norm
        self.async_thread = async_thread
        self.use_mla = vllm_config.model_config.use_mla
        self.is_prefetch = is_prefetch
        self.num_attention_layers = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.max_bs = MAX_BS
        self.is_log = is_log
        self.max_block_len = math.ceil(
            vllm_config.model_config.max_model_len / vllm_config.cache_config.block_size
        )
        self.block_size = vllm_config.cache_config.block_size
        self.device_config = vllm_config.device_config
        self.num_kv_heads = vllm_config.model_config.get_num_kv_heads(
            vllm_config.parallel_config
        )
        self.head_size = vllm_config.model_config.get_head_size()
        self.dtype = vllm_config.model_config.dtype
        self.align_cache = (
            vllm_config.model_config.use_mla and VLLM_CUDA_MEM_ALIGN_KV_CACHE
        )
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size

        self.sp_max_len = self.max_block_len

        if ENABLE_KVCOMP:
            assert (
                is_cpu_topk == False
            ), "KVComp requires is_cpu_topk to be False (i.e., CUDA_TOPK to be True)"
            self.kpre_shape = (
                self.max_bs * self.max_block_len,
                self.num_kv_heads,
                self.block_size,
                self.head_size // 8,
            )
        else:  # original GSA logic
            if self.is_max_norm:
                self.kpre_shape = (
                    self.max_bs * self.max_block_len,
                    1,
                    self.num_kv_heads,
                    self.head_size,
                )
            else:
                self.kpre_shape = (
                    self.max_bs * self.max_block_len,
                    max_norm_num,
                    self.num_kv_heads,
                    self.head_size,
                )
        self.topk_shape = (self.num_attention_layers, self.max_bs, self.max_block_len)
        if self.is_cpu_topk:
            self.kpre_caches, self.use_topk_caches = self._init_kpre_and_topk_cache(
                "cpu", torch.float32, torch.int32
            )
        else:
            if ENABLE_KVCOMP:
                self.kpre_caches, self.use_topk_caches = self._init_kpre_and_topk_cache(
                    self.device_config.device, torch.uint8, torch.int32
                )
            else:
                self.kpre_caches, self.use_topk_caches = self._init_kpre_and_topk_cache(
                    self.device_config.device, self.dtype, torch.int64
                )
        self._init_tensor()
        kv_shape = [self.block_size, self.num_kv_heads, self.head_size]
        self.is_python_load = is_python_load
        self.prefetch_engine_c = gsa_prefetch.GSAPrefetchEngineC(
            self.m_load_success_list,
            self.block_table_len,
            kv_shape,
            self.use_mla,
            self.is_log,
            self.tp_size,
            self.rank,
            self.is_python_load,
        )

        self.topk_space = 0
        self.step_time = 0
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

        self.topk_buf_tmp = None
        self.topk_bs = []
        self.is_topk_update = False

    def model_input_deal(
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
            # CPU tensor used only for CPU indexing operations, no need for pinned memory
            block_table_index = torch.tensor(self.select_bs_index, device="cpu")
            self.topk_len = (
                gsa_config.compute_topk_len(self._get_max_block_len(gsa_metadata))
                # + gsa_config.num_prefetch_blocks
            )
            # self.use_topk_caches is in CPU under KVComp
            topk_buf_tmp = self.use_topk_caches[:, block_table_index, :]
            topk_buf_tmp = topk_buf_tmp[:, :, : self.topk_len]
            self.is_topk_cal = is_topk_done and self.topk_space % 1 == 0
            if self.is_topk_cal:
                self._topk_tmp_deal(gsa_metadata, topk_buf_tmp)
                self.is_topk_update = True

            self._topk_insert_last_idx(gsa_metadata)
            if self.ptopk_prefetch_enable:
                self._first_topk_deal(gsa_metadata)
                self._gsa_block_len_pre(gsa_metadata)
            else:
                self._no_gsa_input_deal(gsa_metadata)
            # self.use_block_table in cpu
            block_table_tmp = self.use_block_table[:, block_table_index, :]
            if torch.cuda.is_available():
                gen_len_tmp = self.gsa_seq_len[:, self.select_bs_index].to(
                    self.device_config.device
                )
            else:
                gen_len_tmp = self.gsa_seq_len[:, self.select_bs_index]

            list_topk_buf = list(topk_buf_tmp.unbind(dim=0))
            list_block_table = list(block_table_tmp.unbind(dim=0))
            # Convert each tensor in list_block_table to NPU tensor asynchronously
            list_block_table_device = [
                t.to(self.device_config.device, non_blocking=True)
                for t in list_block_table
            ]
            gsa_len_list = list(gen_len_tmp.unbind(dim=0))
            gsa_model_input["topk_caches"] = list_topk_buf
            gsa_model_input["kpre_caches"] = self.kpre_caches
            gsa_model_input["is_topk"] = self.is_topk_cal
            gsa_model_input["block_tables_mp"] = list_block_table_device
            gsa_model_input["gsa_seq_len"] = gsa_len_list
        gsa_model_input["atb_gsa_enable"] = self.atb_gsa_enable

    def _topk_tmp_deal(self, gsa_metadata, topk_buf_tmp):
        for index, topk_info in enumerate(self.topk_bs):
            if topk_info[1] and topk_info[0] in gsa_metadata.gsa_stats:
                if not self.is_cpu_topk:
                    # Use non_blocking transfer to reduce synchronization latency
                    gsa_metadata.gsa_stats[topk_info[0]].topk_buf_tmp = (
                        self.topk_buf_tmp[:, index, : topk_info[2]].to(
                            "cpu", non_blocking=True
                        )
                    )
                else:
                    gsa_metadata.gsa_stats[topk_info[0]].topk_buf_tmp = (
                        self.topk_buf_tmp[:, index, : topk_info[2]].clone()
                    )
        self.topk_bs = []
        for index, req_id in enumerate(self.req_ids_bs):
            one_topk_len = (
                gsa_config.compute_topk_len(len(gsa_metadata.gsa_stats[req_id].blocks))
                # + gsa_config.num_prefetch_blocks
            )
            self.topk_bs.append(
                [
                    req_id,
                    gsa_metadata.gsa_stats[req_id].is_gsa(),
                    one_topk_len,
                ]
            )
        self.topk_buf_tmp = topk_buf_tmp

    def deal_async_prefetch(self, is_prefetch_done, gsa_metadata, kvcache, store_ptr, is_topk_update_np=None):
        self.topk_space += 1
        all_free_block_ids = None
        all_miss_ids = None
        prefetch_threshold = 10 # the minimum number of layers with topk indices updated under QS feature for prefetching
        if not self.atb_gsa_enable:
            return all_free_block_ids, all_miss_ids
        if is_prefetch_done and self.ptopk_prefetch_enable and self.is_topk_update:
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
                self.prefetch_topk_buf[:, : len(self.select_bs_index), :],
                self.step_time,
            )
            topk_len_list = []
            req_id_list = []
            for req_id in self.req_ids_bs:
                index_in_batch = gsa_metadata.gsa_stats[req_id].index_in_batch
                req_id_list.append(req_id)
                if not self.is_gsa_req_id[req_id]:
                    print(f"req_id: {req_id} with index_in_batch: {index_in_batch} is not a GSA request")
                    topk_len_list.append(0)
                    continue
                # Note: is_topk_update_np is updated asynchronously via NPU stream in cal_topk_for_hamming.
                # The stream is synchronized per-layer before updating the numpy array, so by the time
                # this function is called (in execute_end, after all layers are processed), all updates are complete.
                elif is_topk_update_np[:,index_in_batch].sum() < prefetch_threshold: 
                    print(f"req_id: {req_id} with index_in_batch: {index_in_batch} does not have enough (<{prefetch_threshold}) layers with topk indices updated under QS feature")
                    topk_len_list.append(0)
                    continue
                else:
                    if gsa_metadata.gsa_stats[req_id].topk_buf_tmp != None:
                        topk_len_list.append(
                            len(gsa_metadata.gsa_stats[req_id].topk_buf_tmp[0])
                        )
                    else:
                        topk_len_list.append(0)
            self.prefetch_engine_c.run_async_prefetch_bs(
                req_id_list, topk_len_list, self.select_bs_index, kvcache, store_ptr
            )
            self.is_topk_update = False
            if self.is_python_load:
                all_free_block_ids = self.prefetch_engine_c.obtain_load_blocks()
                all_miss_ids = self.prefetch_engine_c.obtain_miss_idxs()
        return all_free_block_ids, all_miss_ids

    def del_finish_meta(self, del_req, flag: bool = True) -> None:
        if del_req in self.block_map_flag:
            del self.block_map_flag[del_req]
        if del_req in self.block_table_flag:
            del self.block_table_flag[del_req]
        if del_req in self.is_gsa_req_id:
            del self.is_gsa_req_id[del_req]
        if PTOPK_PREFETCH_ENABLE and flag:
            self.prefetch_engine_c.del_blocks_map(del_req)

    def _init_tensor(self):
        device = "cpu"
        """
        self.prefetch_blocks = torch.zeros(
            (self.num_attention_layers, self.max_bs, int(self.sp_max_len)),
            dtype=torch.int32,
            pin_memory=is_pin_memory_available(),
            device=device,
        )
        """
        self.m_load_success_list = torch.zeros(
            (self.num_attention_layers, self.max_bs, int(self.sp_max_len)),
            dtype=torch.int32,
            pin_memory=False,
            device=device,
        )
        self.use_block_table = torch.zeros(
            (self.num_attention_layers, self.max_bs, int(self.sp_max_len)),
            dtype=torch.int32,
            pin_memory=False,
            device=device,
        )
        """
        self.prefetch_block_len = torch.zeros(
            (self.num_attention_layers, self.max_bs),
            dtype=torch.int32,
            pin_memory=is_pin_memory_available(),
            device=device,
        )
        """
        self.gsa_seq_len = torch.zeros(
            (self.num_attention_layers, self.max_bs),
            dtype=torch.int32,
            pin_memory=is_pin_memory_available(),
            device=device,
        )
        self.block_table_len = torch.zeros(
            (self.num_attention_layers, self.max_bs),
            dtype=torch.int32,
            pin_memory=is_pin_memory_available(),
            device=device,
        )
        self.use_block_table_len = torch.zeros(
            (self.num_attention_layers, self.max_bs),
            dtype=torch.int32,
            pin_memory=is_pin_memory_available(),
            device=device,
        )
        self.prefetch_topk_buf = torch.zeros(
            (self.num_attention_layers, self.max_bs, int(self.sp_max_len)),
            dtype=torch.int64,
            pin_memory=is_pin_memory_available(),
            device=device,
        )

    def _init_kpre_and_topk_cache(
        self, device, krepre_type, topk_type
    ) -> Tuple[List[torch.tensor], torch.tensor]:
        kpre_caches = []
        pin_memory = is_pin_memory_available() if device == "cpu" else False
        # we put use_topk_caches in CPU under KVComp for better performance
        use_topk_caches = torch.zeros(
            self.topk_shape, dtype=topk_type, pin_memory=pin_memory, device="cpu"
        )
        for _ in range(self.num_attention_layers):
            if self.align_cache:
                entry_shape = self.kpre_shape[2:]
                entry_size = np.prod(entry_shape)
                alloc_entry_size = align_to_256bytes(entry_size, krepre_type)
                alloc_shape = (*self.kpre_shape[:2], alloc_entry_size)
            else:
                alloc_shape = self.kpre_shape
            one_kpre_value = torch.zeros(
                alloc_shape, dtype=krepre_type, pin_memory=pin_memory, device=device
            )
            if self.align_cache:
                one_kpre_value = one_kpre_value[..., :entry_size]
            # we put kpre_caches in NPU under KVComp
            kpre_caches.append(one_kpre_value)

        return kpre_caches, use_topk_caches

    def _first_topk_deal(self, gsa_metadata) -> None:
        for index, req_id in enumerate(self.req_ids_bs):
            if gsa_metadata.gsa_stats[req_id].remain_idx == None:
                continue

            bs_index = self.select_bs_index[index]
            if gsa_metadata.gsa_stats[req_id].remain_map != None:
                topk_block_list_all = []
                # prefetch_blocks_list_all = []
                for layer_id in range(self.num_attention_layers):
                    topk_block_list = sorted(
                        list(
                            gsa_metadata.gsa_stats[req_id].remain_map[layer_id].values()
                        )
                    )
                    # prefetch_blocks_list = list(
                    #    gsa_metadata.gsa_stats[req_id].prefetch_map[layer_id].values()
                    # )
                    topk_block_list_all.append(topk_block_list)
                    # prefetch_blocks_list_all.append(prefetch_blocks_list)
                # CPU tensor used only for CPU-to-CPU assignment, no need for pinned memory
                topk_block_tensor = torch.tensor(
                    topk_block_list_all, dtype=torch.int32, device="cpu"
                )
                # prefetch_block_tensor = torch.tensor(
                #    prefetch_blocks_list_all, dtype=torch.int32
                # )
            else:
                real_length = len(gsa_metadata.gsa_stats[req_id].blocks)
                block_table_list = self.block_table_list_bs[index][:real_length]
                remain_index = gsa_metadata.gsa_stats[req_id].remain_idx
                prefetch_idx = gsa_metadata.gsa_stats[req_id].prefetch_idx
                assert len(remain_index) < self.sp_max_len

                # prefetch_blocks_list = [block_table_list[x] for x in prefetch_idx]
                topk_block_list = [block_table_list[x] for x in remain_index]
                # CPU tensor used only for CPU-to-CPU assignment, no need for pinned memory
                topk_block_tensor = torch.tensor(
                    topk_block_list, dtype=torch.int32, device="cpu"
                )
                # prefetch_block_tensor = torch.tensor(
                #    prefetch_blocks_list, dtype=torch.int32
                # )

            # self.prefetch_block_len[:, bs_index] = len(prefetch_blocks_list)
            self.block_table_len[:, bs_index] = len(topk_block_list)
            self.use_block_table_len[:, bs_index] = len(topk_block_list)

            # self.prefetch_blocks[:, bs_index, : len(prefetch_blocks_list)] = (
            #    prefetch_block_tensor
            # )
            self.use_block_table[:, bs_index, : len(topk_block_list)] = (
                topk_block_tensor
            )
            self.m_load_success_list[:, bs_index, : len(topk_block_list)] = (
                topk_block_tensor
            )
            max_idx = len(gsa_metadata.gsa_stats[req_id].block_hashes)
            if self.is_gsa_req_id[req_id]:
                if gsa_metadata.gsa_stats[req_id].remain_map != None:
                    self.prefetch_engine_c.set_blocks_map_multilayer(
                        req_id,
                        gsa_metadata.gsa_stats[req_id].remain_map,
                        gsa_metadata.gsa_stats[req_id].prefetch_map,
                        gsa_metadata.gsa_stats[req_id].block_hashes,
                        max_idx,
                    )
                else:
                    self.prefetch_engine_c.set_blocks_map(
                        req_id,
                        block_table_list,
                        remain_index,
                        prefetch_idx,
                        gsa_metadata.gsa_stats[req_id].block_hashes,
                        max_idx,
                    )

    def _gsa_block_len_pre(
        self,
        gsa_metadata,
    ) -> None:
        # CPU-to-CPU copy, no need for non_blocking
        self.gsa_seq_len.copy_(self.use_block_table_len)
        for index, req_id in enumerate(self.req_ids_bs):
            bs_index = self.select_bs_index[index]
            remain_slot = gsa_metadata.gsa_stats[req_id].get_seq_len() % self.block_size
            if gsa_metadata.gsa_stats[req_id].stage() == SequenceStage.DECODE:
                if remain_slot == 0:
                    self.gsa_seq_len[:, bs_index].mul_(self.block_size)
                elif remain_slot == 1:
                    self.gsa_seq_len[:, bs_index].mul_(self.block_size).add_(
                        remain_slot
                    )
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
                    self.block_map_flag[req_id].append(
                        [len(gsa_metadata.gsa_stats[req_id].blocks) - 1, last_block]
                    )
                else:
                    self.gsa_seq_len[:, bs_index].add_(-1).mul_(self.block_size).add_(
                        remain_slot
                    )
            else:
                self.block_map_flag[req_id] = []
                self.block_table_flag[req_id] = []
                self.gsa_seq_len[:, bs_index] = gsa_metadata.gsa_stats[
                    req_id
                ].get_seq_len()
                self.use_block_table[
                    :, bs_index, : len(gsa_metadata.gsa_stats[req_id].blocks)
                ] = torch.tensor(
                    gsa_metadata.gsa_stats[req_id].blocks,
                    dtype=torch.int32,
                    device="cpu",
                )

    def _topk_insert_last_idx(self, gsa_metadata) -> None:
        for index in range(len(self.req_ids_bs)):
            req_id = self.req_ids_bs[index]
            if gsa_metadata.gsa_stats[req_id].topk_buf_tmp == None:
                continue

            last_idx = len(gsa_metadata.gsa_stats[req_id].blocks) - 1

            if last_idx in gsa_metadata.gsa_stats[req_id].topk_buf_tmp:
                continue

            gsa_metadata.gsa_stats[req_id].topk_buf_tmp = torch.nn.functional.pad(
                gsa_metadata.gsa_stats[req_id].topk_buf_tmp,
                (0, 1),
                value=last_idx,
            )

    def _swap_block_table_tensor(
        self,
        bs_index_list: List[int],
        gsa_metadata,
    ) -> None:
        for index, bs_index in enumerate(bs_index_list):
            req_id = self.req_ids_bs[index]
            if req_id in self.block_map_flag:
                for block_mp_add in self.block_map_flag[req_id]:
                    self.prefetch_engine_c.add_blocks_map(
                        req_id, block_mp_add[0], block_mp_add[1]
                    )
                self.block_map_flag[req_id].clear()

            if req_id in self.block_table_flag.keys():
                for block_table_add in self.block_table_flag[req_id]:
                    for layer_id in range(self.num_attention_layers):
                        indices = self.use_block_table_len[layer_id][bs_index].item()
                        assert indices < self.sp_max_len
                        self.use_block_table[layer_id][bs_index][
                            indices
                        ] = block_table_add
                        self.use_block_table_len[layer_id][bs_index].add_(1)
                self.block_table_flag[req_id].clear()

            if gsa_metadata.gsa_stats[req_id].topk_buf_tmp != None:
                # CPU-to-CPU copy, no need for non_blocking
                self.prefetch_topk_buf[
                    :, index, : len(gsa_metadata.gsa_stats[req_id].topk_buf_tmp[0])
                ].copy_(gsa_metadata.gsa_stats[req_id].topk_buf_tmp)

    def _get_run_type(
        self,
        gsa_metadata,
    ) -> None:
        self.open_gsa = False
        for req_id in self.req_ids_bs:
            if gsa_metadata.gsa_stats[req_id].is_gsa():
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

    def _get_max_block_len(self, gsa_metadata) -> int:
        max_len = 0
        for req_id in self.req_ids_bs:
            if self.is_gsa_req_id[req_id]:
                max_len = max(max_len, len(gsa_metadata.gsa_stats[req_id].blocks))
        return max_len

    def _no_gsa_input_deal(
        self,
        gsa_metadata,
    ) -> None:
        for index, req_id in enumerate(self.req_ids_bs):
            bs_index = self.select_bs_index[index]
            # CPU tensor used only for CPU-to-CPU assignment, no need for pinned memory
            one_block_table = torch.tensor(
                self.block_table_list_bs[index], dtype=torch.int32, device="cpu"
            )
            if (
                self.is_gsa_req_id[req_id]
                and gsa_metadata.gsa_stats[req_id].topk_buf_tmp != None
            ):
                if torch.max(gsa_metadata.gsa_stats[req_id].topk_buf_tmp) > (
                    len(self.block_table_list_bs[index]) - 1
                ):
                    self.gsa_seq_len[:, bs_index] = gsa_metadata.gsa_stats[
                        req_id
                    ].get_seq_len()
                    self.use_block_table[:, bs_index, :].fill_(0)
                    self.use_block_table[
                        :, bs_index, : len(gsa_metadata.gsa_stats[req_id].blocks)
                    ] = one_block_table
                    continue
                remain_slot = (
                    gsa_metadata.gsa_stats[req_id].get_seq_len() % self.block_size
                )
                one_topk_len = len(gsa_metadata.gsa_stats[req_id].topk_buf_tmp[0])
                for layer_id in range(self.num_attention_layers):
                    self.use_block_table[layer_id][bs_index][:one_topk_len] = (
                        one_block_table[
                            gsa_metadata.gsa_stats[req_id].topk_buf_tmp[layer_id]
                        ]
                    )
                self.gsa_seq_len[:, bs_index].fill_(0)
                if remain_slot == 0:
                    self.gsa_seq_len[:, bs_index].add_(one_topk_len * self.block_size)
                else:
                    self.gsa_seq_len[:, bs_index].add_(
                        one_topk_len * self.block_size - self.block_size + remain_slot
                    )
            else:
                self.gsa_seq_len[:, bs_index] = gsa_metadata.gsa_stats[
                    req_id
                ].get_seq_len()
                self.use_block_table[
                    :, bs_index, : len(gsa_metadata.gsa_stats[req_id].blocks)
                ] = one_block_table
