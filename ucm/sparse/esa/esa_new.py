# TODO: handle preemption
# TODO: init ESA before warmup to make profile_run right!!!
# TODO: reduce memory usage
# TODO: interface of esa_retrieval


import numpy as np
import torch
import math
import time
from vllm.config import VllmConfig
import torch.cuda.nvtx as nvtx
from ucm.sparse.base import (
    INVALID_SLOT,
    UcmSparseBase,
    UcmSparseMetadata,
    UcmSparseRole,
)
import sys
sys.path.append("/root/hk/cuda")
import esa_interface as esa_lib
esa_retrieval = esa_lib.esa_retrieval
esa_repre = esa_lib.esa_repre
esa_copy = esa_lib.esa_copy
esa_scatter_copy = esa_lib.esa_scatter_copy


class ReprePool:
    def __init__(self, capability):
        self.capability = capability
        self.repre_blocks = list(range(capability-1, -1, -1))

    def allocate(self, num_blocks):
        assert len(self.repre_blocks) >= num_blocks, f"reprePool has no blocks, increase the capability: {self.capability}"
        res = []
        for _ in range(num_blocks):
            res.append(self.repre_blocks.pop())
        return res

    def free(self, blocks):
        for e in blocks:
            self.repre_blocks.append(e)

class ESA(UcmSparseBase):
    # handle batch
    def __init__(self, vllm_config: VllmConfig, role: UcmSparseRole):
        super().__init__(vllm_config, role)
        parallel_config = vllm_config.parallel_config
        model_config = vllm_config.model_config
        self.total_num_hidden_layers = model_config.hf_config.num_hidden_layers
        self.block_size = vllm_config.cache_config.block_size
        self.device = vllm_config.device_config.device
        self.dtype = model_config.dtype

        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        max_block_per_seq = math.ceil(model_config.max_model_len / vllm_config.cache_config.block_size)
        max_num_blocks =  max_block_per_seq * max_num_seqs
        shape = (1000, self.block_size, model_config.get_num_kv_heads(parallel_config), model_config.get_head_size()) # TODO:从config里拿到实际的blocks数量*3
        self.host_kv_cache = [
            torch.zeros(shape, dtype=self.dtype, device="cpu", pin_memory=True)
            for _ in range(self.total_num_hidden_layers)
        ]
        shape = (max_num_blocks, model_config.get_num_kv_heads(parallel_config), model_config.get_head_size())
        self.repre_cache = [
            torch.zeros(shape, dtype=self.dtype, device=self.device)
            for _ in range(self.total_num_hidden_layers)
        ]
        self.repre_pool = ReprePool(max_num_blocks)
        self.row_index_pool = ReprePool(max_num_seqs)

        ########################
        # req states 
        # TODO: clear in request_finished
        self.req_row_id = dict()
        self.req_cols = dict()
        self.req_step = dict()
        self.req_prefill_repre_blocks = torch.zeros(max_num_seqs, max_block_per_seq, dtype=torch.int32, device="cpu", pin_memory=True)
        self.req_prefill_repre_blocks_np = self.req_prefill_repre_blocks.numpy()
        ########################

        self.req_topk_block_tables = dict()
        self.req_topk_repre_indexes = dict()

        # TODO: use 2d array. req_id -> row_id -> |..layer0..|..layer1..|..layern..|
        # self.req_topk_block_tables = torch.zeros(max_num_seqs, max_block_per_seq, dtype=torch.int32, device="cpu", pin_memory=True)
        # self.req_topk_repre_indexes = torch.zeros(max_num_seqs, max_block_per_seq, dtype=torch.int32, device="cpu", pin_memory=True)
        # self.req_topk_block_tables_np = self.req_topk_block_tables.numpy()
        # self.req_topk_repre_indexes_np = self.req_topk_repre_indexes.numpy()
        
        ########################
        # retrieval input and output
        self.size_of_int32 = 4
        self.retrieval_input = esa_lib.RetrievalInputTensor()
        self.retrieval_output = esa_lib.RetrievalOutputTensor()
        self.retrieval_output.score = torch.zeros(max_num_blocks, dtype=self.dtype, device=self.device)
        self.retrieval_output.score_cpu = torch.zeros(max_num_blocks, dtype=self.dtype, device="cpu", pin_memory=True)
        self.retrieval_output.score_sorted_cpu = torch.zeros(max_num_blocks, dtype=self.dtype, device="cpu", pin_memory=True)
        self.retrieval_output.index_sorted_cpu = torch.zeros(max_num_blocks, dtype=torch.int32, device="cpu", pin_memory=True)
         ########################

        ########################
        # batch dynamic metadata
        # prefill
        self.prefill_repre_index_cpu = torch.zeros(max_num_blocks, dtype=torch.int32, device="cpu", pin_memory=True)
        self.prefill_repre_index = torch.zeros(max_num_blocks, dtype=torch.int32, device=self.device)
        self.prefill_block_tables_cpu = torch.zeros(max_num_blocks, dtype=torch.int32, device="cpu", pin_memory=True)
        self.prefill_block_tables = torch.zeros(max_num_blocks, dtype=torch.int32, device=self.device)
        self.prefill_num_blocks = 0     
        self.has_prefill = False
        # decode
        self.decode_q_index_cpu = torch.zeros(max_num_blocks, dtype=torch.int32, device="cpu", pin_memory=True)
        self.decode_q_index = torch.zeros(max_num_blocks, dtype=torch.int32, device=self.device)
        self.decode_repre_index_cpu = torch.zeros(max_num_blocks, dtype=torch.int32, device="cpu", pin_memory=True)
        self.decode_repre_index = torch.zeros(max_num_blocks, dtype=torch.int32, device=self.device)
        self.decode_batch_offset_cpu = torch.zeros(max_num_seqs, dtype=torch.int32, device="cpu", pin_memory=True)
        self.decode_batch_offset = torch.zeros(max_num_seqs, dtype=torch.int32, device=self.device)
        self.decode_retrieval_batch = 0
        self.decode_retrieval_s_len = 0
        self.has_decode = False        
        ########################


    def get_kv_cache(self, forward_context, layer_name):
        attn = forward_context.no_compile_layers[layer_name]
        kv_cache = attn.kv_cache[forward_context.virtual_engine]
        return kv_cache

    def get_layer_id(self, layer_name):
        layer_id = int(layer_name.split(".")[2])
        return layer_id

    def get_diff_blocks_and_index(self, keys, old_values, new_values):
        equal = new_values[:, None] == old_values[None, :]
        rows = np.sum(equal, 1)
        cols = np.sum(equal, 0)
        remain_keys = keys[cols == 0]
        remain_new_values = new_values[rows == 0]
        return remain_keys, remain_new_values

    def build_sparse_meta(
        self, scheduler_output, requests, input_batch, attn_metadata
        ):
        with nvtx.range(f"build_sparse_meta"):
            if isinstance(attn_metadata, dict):
                attn_metadata = next(iter(attn_metadata.values()))
            self.attn_metadata = attn_metadata
            self.has_prefill = False
            self.has_decode = False
            self.decode_retrieval_batch = 0
            self.decode_retrieval_s_len = 0
            prefill_repre_index_offset = 0
            decode_repre_index_offset = 0
            decode_batch_offset_index = 0
            for (req_id, num_scheduled_tokens) in scheduler_output.num_scheduled_tokens.items():
                req = requests[req_id]
                is_decode = len(req.output_token_ids) > 0 # 抢占时不成立, FIXME

                # construct metadata for prefill batch
                is_last_chunk = (not is_decode) and (req.num_computed_tokens + num_scheduled_tokens >= req.num_prompt_tokens)
                if is_last_chunk:
                    self.has_prefill = True
                    row_id = self.row_index_pool.allocate(1)[0]
                    self.req_row_id[req_id] = row_id
                    prompt_len = len(req.prompt_token_ids)
                    prompt_blocks = math.ceil(prompt_len / self.block_size) # 包括最后一个不满的block
                    assert prompt_blocks == len(req.block_ids[0])
                    self.req_cols[req_id] = prompt_blocks
                    repre_blocks = self.repre_pool.allocate(prompt_blocks)
                    self.req_prefill_repre_blocks_np[row_id][:prompt_blocks] = repre_blocks
                    for i, b in enumerate(repre_blocks):
                        self.prefill_repre_index_cpu[prefill_repre_index_offset + i] = b
                    for i, b in enumerate(req.block_ids[0]):
                        self.prefill_block_tables_cpu[prefill_repre_index_offset + i] = b
                    prefill_repre_index_offset += prompt_blocks

                # construct metadata for decode batch
                if is_decode:

                    ################ TODO: HERE，检索需要排除掉local window， 需要记录topk长度等
                    # if req_id not in self.req_topk_block_tables:
                    #     print("init topk @decode step 1")
                    #     decode_blocks = len(req.block_ids[0])
                    #     layer_block_tables = [] # NOTE: 默认逐层block_tables可以不一样，留给以后做逐层topk
                    #     for i in range(self.total_num_hidden_layers):
                    #         layer_block_tables.append(np.array(req.block_ids[0])) # TODO: handle local window
                    #     self.req_topk_block_tables[req_id] = layer_block_tables                        

                    #     layer_repre_indexes = []
                    #     for i in range(self.total_num_hidden_layers):
                    #         layer_repre_indexes.append(np.array([-1 for _ in range(decode_blocks)]))
                    #     self.req_topk_repre_indexes = layer_repre_indexes

                    self.has_decode = True
                    assert req_id in self.req_row_id, f"req {req_id} does not has repre_blocks"
                    row_id = self.req_row_id[req_id]
                    cols = self.req_cols[req_id]
                    repre_blocks = self.req_prefill_repre_blocks_np[row_id][:cols]
                    req_index = input_batch.req_id_to_index[req_id]
                    for i, b in enumerate(repre_blocks):
                        self.decode_repre_index_cpu[decode_repre_index_offset + i] = b
                        self.decode_q_index_cpu[decode_repre_index_offset + i] = req_index
                    self.decode_batch_offset_cpu[decode_batch_offset_index:decode_batch_offset_index+1] = decode_repre_index_offset
                    decode_batch_offset_index += 1
                    self.decode_retrieval_batch += 1
                    decode_repre_index_offset += len(repre_blocks)


                        

            self.prefill_num_blocks = prefill_repre_index_offset
            if self.has_prefill:
                self.prefill_repre_index[:prefill_repre_index_offset].copy_(self.prefill_repre_index_cpu[:prefill_repre_index_offset], True)
                self.prefill_block_tables[:prefill_repre_index_offset].copy_(self.prefill_block_tables_cpu[:prefill_repre_index_offset], True)
                # bytes = math.ceil(prefill_offset / 8) * 8 * self.size_of_int32 # 对齐32bytes
                # esa_copy(self.repre_index_cpu, self.repre_index, bytes)
                # esa_copy(self.block_tables_cpu, self.block_tables, bytes)

            if self.has_decode:
                self.handles = []
                self.decode_retrieval_s_len = decode_repre_index_offset
                self.decode_repre_index[:decode_repre_index_offset].copy_(self.decode_repre_index_cpu[:decode_repre_index_offset], True)
                self.decode_q_index[:decode_repre_index_offset].copy_(self.decode_q_index_cpu[:decode_repre_index_offset], True)
                self.decode_batch_offset_cpu[decode_batch_offset_index:decode_batch_offset_index+1] = decode_repre_index_offset
                decode_batch_offset_index += 1
                self.decode_batch_offset[:decode_batch_offset_index].copy_(self.decode_batch_offset_cpu[:decode_batch_offset_index], True)

                # bytes = math.ceil(repre_index_offset / 8) * 8 * self.size_of_int32
                # esa_copy(self.repre_index_cpu, self.repre_index, bytes)
                # esa_copy(self.q_index_cpu, self.q_index, bytes)
                # bytes = math.ceil(batch_offset_index / 8) * 8 * self.size_of_int32
                # esa_copy(self.batch_offset_cpu, self.batch_offset, bytes)

    def attention_begin(
        self,
        query,
        key,
        value,
        layer_name,
        forward_context,
        phase = None,
    ) -> None:
        return
        # if not self.has_decode:
        #     return
        # with nvtx.range(f"retrieval"):
        #     layer_id = self.get_layer_id(layer_name)
        #     self.retrieval_input.query = query
        #     self.retrieval_input.repre_cache = self.repre_cache[layer_id]
        #     self.retrieval_input.q_index = self.decode_q_index
        #     self.retrieval_input.repre_index = self.decode_repre_index
        #     self.retrieval_input.repre_index_cpu = self.decode_repre_index_cpu
        #     self.retrieval_input.batch_offset = self.decode_batch_offset_cpu
        #     self.retrieval_input.batch = self.decode_retrieval_batch
        #     self.retrieval_input.s = self.decode_retrieval_s_len
        #     h = esa_retrieval(self.retrieval_input, self.retrieval_output)
        #     self.handles.append(h)

    def attention_finished(
        self,
        query,
        key,
        value,
        attn_output,
        layer_name,
        forward_context,
        phase = None,
    ) -> None:
        if self.has_prefill:
            with nvtx.range(f"dump_kv_and_compute_repre"):
                layer_id = self.get_layer_id(layer_name)
                k_cache, _ = self.get_kv_cache(forward_context, layer_name)
                esa_repre(k_cache.flatten(-2, -1), self.repre_cache[layer_id].flatten(-2, -1),
                          self.prefill_block_tables[:self.prefill_num_blocks], self.prefill_repre_index[:self.prefill_num_blocks])
                esa_scatter_copy(k_cache.flatten(-3), self.host_kv_cache[layer_id].flatten(-3),
                                 self.prefill_block_tables[:self.prefill_num_blocks], self.prefill_repre_index[:self.prefill_num_blocks])


        if self.has_decode:
            with nvtx.range(f"retrieval"):
                layer_id = self.get_layer_id(layer_name)
                self.retrieval_input.query = query
                self.retrieval_input.repre_cache = self.repre_cache[layer_id]
                self.retrieval_input.q_index = self.decode_q_index
                self.retrieval_input.repre_index = self.decode_repre_index
                self.retrieval_input.repre_index_cpu = self.decode_repre_index_cpu
                self.retrieval_input.batch_offset = self.decode_batch_offset_cpu
                self.retrieval_input.batch = self.decode_retrieval_batch
                self.retrieval_input.s = self.decode_retrieval_s_len
                h = esa_retrieval(self.retrieval_input, self.retrieval_output)
                self.handles.append(h)

    def _wait(self, h, timeout_in_seconds):
        deadline = time.time() + timeout_in_seconds
        while time.time() < deadline:
            ready = esa_lib.esa_retrieval_poll(h)
            if ready == 1:
                break
            time.sleep(1 / 1e6) # 1us
        assert esa_lib.esa_retrieval_cleanup(h) == 1


    def execute_finished(self):
        """
        This is called at the end of "ModelRunner->execute_model" function.
        """
        return
        if self.has_decode:
            for h in self.handles:
                self._wait(h, 10)

    def estimate_num_slots_sparsed(self, request) -> int:
        return INVALID_SLOT
