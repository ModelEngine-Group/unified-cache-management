# isort: skip_file
# fmt: off

# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

# This file is based on mindie_llm/text_generator/plugins/prefix_cache/prefix_cache_plugin.py
# Modifications Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Modified by nrj on 2026-03-10
# You can use this file according to the terms and conditions of the Mulan PSL v2.
 
import threading
import queue
import numpy as np
from mindie_llm.text_generator.mempool.uc_utils import KVPtrComputer
from .prefix_cache_preprocess import PrefixCachePreprocess
from ..plugin import Plugin
from ..plugin_manager import MemPoolType
from ....modeling.backend_type import BackendType
from ....utils.env import ENV
from ....utils.log.logging import logger, print_log

from ucm.integration.mindie import uc_hash_ext

UPDATE_INTERVAL = 2 * 60

HASH_SHIFT_LEFT = 6
HASH_SHIFT_RIGHT = 2
INVALID_HASH_VALUE = 0
EXTRA_HASH = INVALID_HASH_VALUE
MEM_POOL_ROLE_KEY = "worker"


def cpp_style_hash(value):
    """_summary_
    Simulate the default hash algorithm in C++. C++ typically returns the value itself directly (for integers).
    """
    if isinstance(value, int):
        return value
    else:
        if isinstance(value, str):
            hash_value = 0
            for char in value:
                hash_value = (hash_value * 31 + ord(char)) & 0xFFFFFFFFFFFFFFFF
            return hash_value
    return hash(value)


def hash_combine(seed, token_id):
    seed ^= (cpp_style_hash(token_id) + 0x9e3779b97f4a7c15 + (seed << HASH_SHIFT_LEFT) + (seed >> HASH_SHIFT_RIGHT))
    seed = 1 if seed == INVALID_HASH_VALUE else seed
    return seed % 2**64


class PrefixCachePlugin(Plugin):
    def __init__(self, generator_backend, kvcache_settings, infer_context, output_filter, plugin_data_param, **kwargs):
        super().__init__()
        self.generator_backend = generator_backend
        self.model_wrapper = self.generator_backend.model_wrapper
        self.kvcache_settings = kvcache_settings
        self.infer_context = infer_context
        self.output_filter = output_filter
        self.plugin_data_param = plugin_data_param
        self.model_name = self.generator_backend.model_name
        self.sp_size = self.infer_context.spcp_parallel_info.sp_size
        self.sp_rank = self.infer_context.spcp_parallel_info.sp_rank
        self.cp_size = self.infer_context.spcp_parallel_info.cp_size
        self.cp_rank = self.infer_context.spcp_parallel_info.cp_rank
        self.scp_size = self.infer_context.spcp_parallel_info.scp_size
        self.scp_rank = self.infer_context.spcp_parallel_info.scp_rank
        self.tp_size = 1
        self.tp_rank = 0
        self.rank = generator_backend.rank
        self.device_id = self.generator_backend.npu_device_id
        if hasattr(self.model_wrapper, "mapping"):
            if self.model_wrapper.mapping.attn_tp.group_size > 1:
                self.tp_size = self.model_wrapper.mapping.attn_tp.group_size
                self.tp_rank = self.model_wrapper.mapping.attn_tp.rank

        ## for model inputs preprocess
        self.prefix_cache_preprocess = PrefixCachePreprocess(
            self.infer_context, self.cp_size, self.scp_size, self.scp_rank
        )

        ## for hit rate log info
        self.total_token_num = 0
        self.total_local_matched_token_num = 0
        self.total_remote_matched_token_num = 0

        ## for mindspore
        self.attention_mask = AttentionMask(None)
        self.model_name = self.generator_backend.model_name
        self.is_300i = False
        if self.generator_backend.backend_type == BackendType.ATB:
            self.is_300i = self.model_wrapper.model_runner.soc_info.is_300i()

        ## for kvcache pool
        self.mempool_type = MemPoolType.DISABLED
        self.num_put_layers = self.kvcache_settings.num_layers
        if len(self.generator_backend.kv_pool_backend) != 0 and len(self.generator_backend.kv_pool_config_path) != 0:
            self.mempool_type = \
                MemPoolType.ASYNC_WRITE if self.generator_backend.kv_pool_async_write else MemPoolType.SYNC_WRITE
            from mindie_llm.text_generator.mempool import MemPool
            self.m_store = MemPool.create_pool(
                backend=self.generator_backend.kv_pool_backend,
                config_path=self.generator_backend.kv_pool_config_path,
                role=MEM_POOL_ROLE_KEY,
                device_id=self.device_id,
                kv_caches=self.generator_backend.cache_pool.npu_cache
            )
            if self.m_store is None:
                self.mempool_type = MemPoolType.DISABLED
        if self.mempool_type != MemPoolType.DISABLED:
            self.kvptr_computer = KVPtrComputer(self.generator_backend.cache_pool.npu_cache)
            self._init_prefix_key_cache()
            logger.info("Init mem pool successfully.")
        if self.mempool_type == MemPoolType.ASYNC_WRITE:
            self.put_input_queue = queue.Queue()
            self.put_task_queue = queue.Queue()
            self.put_prefix_kvcache_thread = threading.Thread(target=self._put_prefix_kvcache_thread, daemon=True)
            self.put_prefix_kvcache_thread.start()
            self.save_event = threading.Event()
            self.save_event.set()  # True
            self.save_timeout = 10  # 落盘等待时限,10s
            logger.info("Create prefix cache async save threads successfully.")

        self.is_300i = False
        if self.generator_backend.backend_type == BackendType.ATB:
            self.is_300i = self.model_wrapper.model_runner.soc_info.is_300i()
        logger.info("Create prefix cache plugin successfully.")

    @staticmethod
    def enable_local_prefixcache(input_metadata):
        return input_metadata.is_prefill and input_metadata.computed_blocks is not None

    @staticmethod
    def enable_remmote_prefixcache(input_metadata):
        return input_metadata.is_prefill and input_metadata.remote_computed_blocks is not None

    # 第一个插件类函数：输入构造
    def model_inputs_update(self, model_inputs, input_metadata, sampling_metadata, cache_ids, input_len_mask, **kwargs):
        (q_len, spec_mask) = input_len_mask

        if input_metadata.batch_is_prefill is not None: # prefix_cache + splitfuse
            batch_size = input_metadata.batch_is_prefill.shape[0]
        else:
            batch_size = input_metadata.batch_size

        local_has_prefixcache = self.enable_local_prefixcache(input_metadata)
        remote_has_prefixcache = self.enable_remmote_prefixcache(input_metadata)
        # prefill阶段有缓存时，用decode并行解码模式代替prefill
        if local_has_prefixcache or remote_has_prefixcache:
            self.get_prefix_kvcache_from_mempool(input_metadata)
            model_inputs = self.prefix_cache_preprocess.update_infer_input(model_inputs, input_metadata)
            q_len, spec_mask = self.get_extra_infer_input(model_inputs, batch_size, q_len, spec_mask)

        if input_metadata.is_prefill and self.rank == 0:
            self.total_token_num += input_metadata.total_seq_num
            local_matched_token_num = 0
            remote_matched_token_num = 0
            if input_metadata.computed_blocks is not None:
                local_matched_token_num = input_metadata.max_block_size * np.sum(input_metadata.computed_blocks)
                self.total_local_matched_token_num += local_matched_token_num
            if input_metadata.remote_computed_blocks is not None:
                remote_matched_token_num = input_metadata.max_block_size * np.sum(input_metadata.remote_computed_blocks)
                remote_matched_token_num -= local_matched_token_num
                self.total_remote_matched_token_num += remote_matched_token_num
            local_cache_hit_rate = local_matched_token_num / input_metadata.total_seq_num * 100
            remote_cache_hit_rate = remote_matched_token_num / input_metadata.total_seq_num * 100
            total_local_cache_hit_rate = self.total_local_matched_token_num / self.total_token_num * 100
            total_remote_cache_hit_rate = self.total_remote_matched_token_num / self.total_token_num * 100
            print_log(self.rank, logger.info, f'Prefix Cache Reporter: '
                      f'#batchsize: {batch_size}, '
                      f'#batched-tokens: {input_metadata.total_seq_num}, '
                      f'#local cached-tokens: {local_matched_token_num}, '
                      f'#local cache hit rate: {round(local_cache_hit_rate, 3)}%, '
                      f'#remote cached-tokens: {remote_matched_token_num}, '
                      f'#remote cache hit rate: {round(remote_cache_hit_rate, 3)}%, '
                      f'#cache hit rate: {round(local_cache_hit_rate + remote_cache_hit_rate, 3)}%')
            print_log(self.rank, logger.info, f'Prefix Cache Global Reporter: '
                      f'#total prefill tokens: {self.total_token_num}, '
                      f'#total local matched tokens: {self.total_local_matched_token_num}, '
                      f'#total local cached hit rate: {round(total_local_cache_hit_rate, 3)}%, '
                      f'#total remote matched tokens: {self.total_remote_matched_token_num}, '
                      f'#total remote cached hit rate: {round(total_remote_cache_hit_rate, 3)}%, '
                      f'#total cached hit rate: {round(total_local_cache_hit_rate + total_remote_cache_hit_rate, 3)}%')

        input_len_mask = (q_len, spec_mask)

        return model_inputs, input_len_mask

    def get_extra_infer_input(self, model_inputs, batch_size, q_len, attention_mask):
        q_lens_list = q_len
        attention_mask = attention_mask
        if model_inputs.is_prefill:  # decode并行解码模式，需要构造q_lens和spec_mask参数
            if model_inputs.query_length is not None:
                q_lens_list = model_inputs.query_length.tolist()
            else:
                q_lens_list = [1] * batch_size

            kv_dtype = self.kvcache_settings.dtype
            if self.generator_backend.backend_type == BackendType.MS:
                if model_inputs.query_length is None:
                    return q_lens_list, [[0]] * batch_size
                seq_len = model_inputs.max_seq_len
                atten_mask = self.attention_mask.get_attn_mask(seq_len)
            elif self.is_300i: # In 300I, the construction of mask is different from A2 and A3
                kv_device = self.model_wrapper.device
                atten_mask = self.model_wrapper.model_runner.attn_mask.get_attn_mask(model_inputs.max_seq_len,
                                                                                    kv_dtype, kv_device)
                if atten_mask[0][1] > 0:
                    atten_mask = atten_mask * -10000.0
            else:
                attention_mask = self.model_wrapper.model_runner.attn_mask.get_splitfuse_mask(self.model_wrapper.device)
                return q_lens_list, attention_mask

            req_mask_list = []
            for i in range(batch_size):
                start = model_inputs.context_length[i] - q_lens_list[i]
                end = model_inputs.context_length[i]
                req_mask_list.append(atten_mask[start:end])
            if self.generator_backend.backend_type == BackendType.MS:
                attention_mask = np.concatenate(req_mask_list, axis=0)
            else:
                import torch
                attention_mask = torch.cat(req_mask_list, 0)
        return q_lens_list, attention_mask

    def sample_preprocess(self, logits, result, sampling_metadata, input_metadata):
        return logits

    def plugin_verify(self, sampling_output, cache_ids, result):
        pass

    def plugin_cache_update(self, cache_ids, sampling_output, la_cache_input, is_prefill=False):
        pass

    def plugin_cache_clear(self, cache_ids, finish_reason):
        pass

    def hash_block(self, prefix_hash_value, block_token_ids):
        seed = INVALID_HASH_VALUE
        if prefix_hash_value != INVALID_HASH_VALUE:
            seed = hash_combine(seed, prefix_hash_value)
        for token_id in block_token_ids:
            seed = hash_combine(seed, token_id)
        seed = hash_combine(seed, EXTRA_HASH)
        return seed

    def _init_prefix_key_cache(self):
        if self.scp_size > 1:
            self._prefix_key_suffix = f"_{self.scp_rank}_{self.scp_size}_{self.model_name}"
        else:
            self._prefix_key_suffix = f"_{self.tp_rank}_{self.tp_size}_{self.model_name}"

    def get_prefix_keys(self, hash_value):
        return str(hash_value) + self._prefix_key_suffix

    def get_prefix_kvcache_from_mempool(self, input_metadata):
        if self.mempool_type == MemPoolType.DISABLED:
            return

        attn_dp_rank = self.generator_backend.mapping.attn_dp.rank
        max_block_size = input_metadata.max_block_size
        batch_block_tables = input_metadata.batch_block_tables
        computed_blocks = input_metadata.computed_blocks
        remote_computed_blocks = input_metadata.remote_computed_blocks
        if computed_blocks is None:
            if self.scp_size > 1:
                computed_blocks = np.zeros((input_metadata.batch_size, self.scp_size), dtype=np.int64)
            else:
                computed_blocks = np.zeros(input_metadata.batch_size, dtype=np.int64)

        if self.scp_size > 1:
            computed_blocks = np.sum(computed_blocks, axis=1)                   # shape: [batch_size]
            remote_computed_blocks = np.sum(remote_computed_blocks, axis=1)     # shape: [batch_size]

        batch_input_ids_offset = 0  # 每个请求的input ids开始索引
        prefix_keys = []        # shape： [all_requests_kyes_num] 
        req_block_ids = []
        for i in range(input_metadata.batch_size):
            seq_len = int(input_metadata.batch_seq_len[i])
            if input_metadata.batch_dp_rank_ids[i] != attn_dp_rank:
                batch_input_ids_offset += seq_len
                continue
            full_blocks = (seq_len - 1) // max_block_size if max_block_size != 0 else 0
            if full_blocks <= 0:
                batch_input_ids_offset += seq_len
                continue

            computed = int(computed_blocks[i])
            remote_computed = int(remote_computed_blocks[i])
            if remote_computed <= computed or computed >= full_blocks:
                batch_input_ids_offset += seq_len
                continue
            remote_computed = min(remote_computed, full_blocks)
            start = batch_input_ids_offset
            flat = input_metadata.input_ids[start : start + remote_computed * max_block_size]
            prefix_hashes = uc_hash_ext.hash_prefix(INVALID_HASH_VALUE, flat, max_block_size, 0, remote_computed)

            # computed blocks: scp_rank = computed % scp_size
            # [computed, remote_computed):   scp_rank = (scp_rank + 1) % scp_size
            if self.scp_size > 1:
                # get first block which belong to current scp_rank
                first = computed + ((self.scp_rank - (computed % self.scp_size)) % self.scp_size)
                local_idx = 0 
                for b in range(first, remote_computed, self.scp_size):
                    prefix_keys.append(self.get_prefix_keys(prefix_hashes[b]))
                    req_block_ids.append(int(batch_block_tables[i, local_idx]))
                    local_idx += 1
            else:
                for b in range(computed, remote_computed):
                    prefix_keys.append(self.get_prefix_keys(prefix_hashes[b]))
                    req_block_ids.append(int(batch_block_tables[i, b]))

            batch_input_ids_offset += seq_len
        
        if not prefix_keys:
            return

        kvcache_tensors = self.kvptr_computer.ptrs_for_blocks_np(req_block_ids)
        self.m_store.get(prefix_keys, kvcache_tensors)

    def async_put_prefix_kvcache_to_mempool(self, input_metadata, cache_ids):
        if self.mempool_type == MemPoolType.DISABLED or not input_metadata.is_prefill:
            return
        self.put_input_queue.put((input_metadata, cache_ids))

    def put_prefix_kvcache_put_task_queue(self, input_metadata, cache_ids):
        only_save_kv = False
        remote_computed_blocks = input_metadata.remote_computed_blocks
        if remote_computed_blocks is None:
            only_save_kv = True
        elif self.scp_size == 1:
            attn_dp_rank = self.generator_backend.mapping.attn_dp.rank
            cur_dp_remote_blocks_hits = 0
            for batch_dp_rank_id, num_computed_blocks in zip(input_metadata.batch_dp_rank_ids, remote_computed_blocks):
                if attn_dp_rank == batch_dp_rank_id:
                    cur_dp_remote_blocks_hits += num_computed_blocks
            only_save_kv = cur_dp_remote_blocks_hits == 0
        for layer_id in range(self.num_put_layers):
            self.put_task_queue.put((layer_id == 0, layer_id == (self.num_put_layers - 1), only_save_kv))

    def put_prefix_kvcache_to_mempool(self, input_metadata, cache_ids):
        if self.mempool_type == MemPoolType.DISABLED or not input_metadata.is_prefill or \
            sum(input_metadata.batch_dp_rank_ids == self.generator_backend.mapping.attn_dp.rank) <= 0:
            return

        attn_dp_rank = self.generator_backend.mapping.attn_dp.rank
        batch_dp_rank_ids = input_metadata.batch_dp_rank_ids
        if sum(batch_dp_rank_ids == attn_dp_rank) <= 0:
            return

        batch_input_ids = self.infer_context.get_all_input_ids(cache_ids)
        batch_seq_lens = self.infer_context.get_seq_lens(cache_ids)
        remote_computed_blocks = input_metadata.remote_computed_blocks
        if remote_computed_blocks is None:
            if self.scp_size > 1:
                remote_computed_blocks = np.zeros((input_metadata.batch_size, self.scp_size), dtype=np.int64)
            else:
                remote_computed_blocks = np.zeros(input_metadata.batch_size, dtype=np.int64)

        if self.scp_size > 1:
            remote_computed_blocks = np.sum(remote_computed_blocks, axis=1)     # shape: [batch_size]

        max_block_size = input_metadata.max_block_size
        batch_block_tables = input_metadata.batch_block_tables
        prefix_keys = []        # shape： [all_requests_kyes_num]
        req_block_ids = []
        for i in range(input_metadata.batch_size):
            if batch_dp_rank_ids[i] != attn_dp_rank:
                continue

            seq_len = int(batch_seq_lens[i])
            if (max_block_size != 0) and ((seq_len - 1) < max_block_size):
                continue

            full_blocks = (seq_len - 1) // max_block_size
            if full_blocks <= 0:
                continue

            computed = int(remote_computed_blocks[i])
            if computed >= full_blocks:
                continue
            
            flat = batch_input_ids[i, : full_blocks * max_block_size]
            prefix_hashes = uc_hash_ext.hash_prefix(INVALID_HASH_VALUE, flat, max_block_size, 0, full_blocks)
            if self.scp_size > 1:
                first = computed + ((self.scp_rank - (computed % self.scp_size)) % self.scp_size)
                local_idx = 0
                for b in range(first, full_blocks, self.scp_size):
                    prefix_keys.append(self.get_prefix_keys(prefix_hashes[b]))
                    req_block_ids.append(int(batch_block_tables[i, local_idx]))
                    local_idx += 1
            else:
                for b in range(computed, full_blocks):
                    prefix_keys.append(self.get_prefix_keys(prefix_hashes[b]))
                    req_block_ids.append(int(batch_block_tables[i, b]))
            
        if not prefix_keys:
            return
        kvcache_tensors = self.kvptr_computer.ptrs_for_blocks_np(np.asarray(req_block_ids, dtype=np.intp))
        self.m_store.put(prefix_keys, kvcache_tensors)

    def _put_prefix_kvcache_thread(self):
        import torch
        torch.npu.set_device(f"npu:{self.device_id}")
        stream = torch.npu.Stream()
        torch.npu.set_stream(stream)
        logger.info("Create _put_prefix_kvcache_thread")
        while True:
            input_metadata, cache_ids = self.put_input_queue.get()
            self.put_prefix_kvcache_put_task_queue(input_metadata, cache_ids)
            while not self.put_task_queue.empty():
                is_first, is_last, only_save_kv = self.put_task_queue.get()
                if is_first:
                    self.save_event.clear()  # False
                pipe_key = self.model_wrapper.generate_mem_pool_event_key(only_save_kv)
                self.model_wrapper.model_runner.wait_event(pipe_key)
                if is_last:
                    self.put_prefix_kvcache_to_mempool(input_metadata, cache_ids)
                    self.save_event.set()  # True


class AttentionMask:
    def __init__(self, atten_mask):
        self._seq_len_cache = 0
        self.atten_mask_cache = atten_mask

    def update_attn_cache(self, seq_len):
        if seq_len > self._seq_len_cache:
            self._seq_len_cache = seq_len
            bias_cache = np.tril(np.ones((seq_len, seq_len), dtype=np.bool_))
            bias_cache = ~bias_cache
            mask_value = np.finfo(np.float32).min
            atten_mask = np.ma.masked_array(np.zeros((seq_len, seq_len)), mask=bias_cache).filled(mask_value)
            self.atten_mask_cache = atten_mask

    def get_attn_mask(self, max_s: int):
        self.update_attn_cache(max_s)
        return self.atten_mask_cache[:max_s, :max_s]
