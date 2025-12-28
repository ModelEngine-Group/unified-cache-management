from typing import Any, Dict, List, Optional, Union

import torch
if hasattr(torch,'npu') and torch.npu.is_available():
    import torch_npu
    import ucm_custom_ops
    from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm import _custom_ops as ops
from vllm.attention.ops.flashmla import get_mla_metadata
from vllm.config import VllmConfig
from vllm.forward_context import ForwardContext
from vllm.v1.attention.backends.mla.common import MLACommonMetadata
from vllm.v1.request import Request, RequestStatus
from vllm.v1.core.sched.output import SchedulerOutput

from ucm.logger import init_logger
from ucm.sparse.base import (
    INVALID_SLOT,
    UcmSparseBase,
    UcmSparseRole,
)
if hasattr(torch, 'cuda') and torch.cuda.is_available():
    from ucm.sparse.kvcomp.hamming_topk import cuda_hamming_topk, fake_hamming_topk
    from ucm.sparse.kvcomp.hash_encoder import reshape_and_cache_khash_triton
from ucm.sparse.kvcomp.hash_encoder import HashEncoder
from ucm.sparse.kvcomp.kvcomp_config import KvCompConfig
from ucm.utils import Config

logger = init_logger(__name__)

ReqType = Union[str, int]


class KvCompOnDevice(UcmSparseBase):
    # handle batch
    def __init__(self, vllm_config: VllmConfig, role: UcmSparseRole):
        super().__init__(vllm_config, role)
        self.rank = vllm_config.parallel_config.rank
        self.is_mla = vllm_config.model_config.is_deepseek_mla

        if vllm_config.device_config.device_type == "cuda":
            self.is_cuda = True
            self.device = torch.device(f"cuda:{self.rank}")
        elif vllm_config.device_config.device_type == "npu":
            self.is_cuda = False
            self.device = torch.device(f"npu:{self.rank}")
        else:
            raise ValueError(f"Unsupported device type: {vllm_config.device_config.device_type}")

        self.num_q_heads = vllm_config.model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        self.num_key_heads = vllm_config.model_config.get_num_kv_heads(
            vllm_config.parallel_config
        )
        self.block_size = vllm_config.cache_config.block_size

        self.kvcompOnDevice_cfg = (
            Config(vllm_config.kv_transfer_config)
            .get_config()
            .get("ucm_sparse_config")
            .get("KvCompOnDevice")
        )

        kvcompOnDevice_config_path = self.kvcompOnDevice_cfg[
            "kvcompOnDevice_config_path"
        ]
        self.kvcompOnDevice_config = KvCompConfig.from_json(kvcompOnDevice_config_path)
        logger.info(f"read kvcomp config file : {kvcompOnDevice_config_path} ")
        self.hash_topk_tokens = self.kvcompOnDevice_config.vllm_hash_attention_topk
        self.hash_rollback_layers = (
            self.kvcompOnDevice_config.vllm_hash_attention_rollback_layers
        )
        self.hash_skip_layers = (
            self.kvcompOnDevice_config.vllm_hash_attention_skip_layers
        )

        self.seq_len_threshhold = (
            self.kvcompOnDevice_config.seq_len_threshhold
        )

        if role == UcmSparseRole.WORKER:
            if self.is_cuda: # cuda only variables
                device_properties = torch.cuda.get_device_properties(self.device)
                num_sms = device_properties.multi_processor_count

                if not vllm_config.model_config.enforce_eager:
                    self.cg_buf_topk_tile_scheduler_metadata = torch.zeros(
                        (num_sms, 8),
                        device=self.device,
                        dtype=torch.int32,
                    )
                    self.cg_buf_topk_num_splits = torch.empty(
                        (vllm_config.scheduler_config.max_num_seqs + 1),
                        device=self.device,
                        dtype=torch.int32,
                    )

            self.ori_seq_lens_decode = None
            self.ori_block_table_decode = None
            self.origin_tile_scheduler_metadata = None # for MLA
            self.origin_num_splits = None # for MLA

            # for GQA
            self.topk_block_table = None
            self.topk_seq_lens = None
            self.topk_seq_lens_qwen = None
            self.decode_mask = None

            self._k_scale = torch.tensor(1.0, dtype=torch.float32)

            if self.is_mla:
                logger.info("KvCompOnDevice initialized with MLA model config")
                self.hash_reduction_head_num = (
                    self.kvcompOnDevice_config.vllm_hash_attention_reduction_head_num
                )
                self.kv_lora_rank = getattr(
                    vllm_config.model_config.hf_text_config, "kv_lora_rank", None
                )
                self.qk_rope_head_dim = getattr(
                    vllm_config.model_config.hf_text_config, "qk_rope_head_dim", None
                )
                self.hash_encoder_nope = HashEncoder(
                    input_dim=self.kv_lora_rank,
                    hash_bits=self.kv_lora_rank,
                    dtype=vllm_config.model_config.dtype,
                    device=self.device,
                )

                self.hash_encoder_rope = HashEncoder(
                    input_dim=self.qk_rope_head_dim,
                    hash_bits=self.qk_rope_head_dim,
                    dtype=vllm_config.model_config.dtype,
                    device=self.device,
                )
            else:
                logger.info("KvCompOnDevice initialized with non-MLA model config")
                self.head_dim = vllm_config.model_config.get_head_size()
                self.hash_encoder = HashEncoder(
                    input_dim=self.head_dim,
                    hash_bits=self.head_dim,
                    dtype=vllm_config.model_config.dtype,
                    device=self.device,
                )

                if not self.is_cuda: # NPU only variables
                    self.decode_mask_npu = None
                    self.is_tensor_computed = False
                    self.max_batch_size = self.kvcompOnDevice_cfg["max_batch_size"]
                    if self.max_batch_size is None:
                        self.max_batch_size = vllm_config.scheduler_config.max_num_seqs
                    else:
                        self.max_batch_size = min(self.max_batch_size, vllm_config.scheduler_config.max_num_seqs)

                    self.chunk_sizes_for_hamming_full = torch.full([self.max_batch_size],
                                                               fill_value=self.block_size,
                                                               dtype=torch.int32,
                                                               device=self.device)
                    self.topk_for_hamming_full = torch.full([self.max_batch_size],
                                                        fill_value=self.hash_topk_tokens // self.block_size,
                                                        dtype=torch.int32,
                                                        device=self.device)
                    self.topk_for_hamming_full_cpu = torch.full([self.max_batch_size],
                                                        fill_value=self.hash_topk_tokens // self.block_size,
                                                        dtype=torch.int32,
                                                        device='cpu')
                    self.seq_lens_for_hamming = torch.zeros([self.max_batch_size],
                                                               dtype=torch.int32,
                                                               device=self.device)
                    self.hamming_output = torch.zeros([self.max_batch_size, self.hash_topk_tokens // self.block_size],
                                                               dtype=torch.int32,
                                                               device=self.device)

    def hash_code(
        self,
        nope: Optional[torch.Tensor] = None,
        rope: Optional[torch.Tensor] = None,
        reduction_head_num: int = 1,
        query: Optional[torch.Tensor] = None,
    ):
        if self.is_mla:
            if nope is None or rope is None:
                raise ValueError("MLA mode requires `nope` and `rope`.")
            if reduction_head_num > 1:
                # reduce heads: [T, H, D] -> [T, H/reduce, D]
                nope = nope.view(
                    nope.shape[0],
                    reduction_head_num,
                    nope.shape[1] // reduction_head_num,
                    nope.shape[2],
                ).mean(dim=1)
                rope = rope.view(
                    rope.shape[0],
                    reduction_head_num,
                    rope.shape[1] // reduction_head_num,
                    rope.shape[2],
                ).mean(dim=1)
            hash_nope = self.hash_encoder_nope.compute_hash(nope)
            hash_rope = self.hash_encoder_rope.compute_hash(rope)
            return hash_nope.view(torch.bfloat16), hash_rope.view(torch.bfloat16)

        # ---- GQA mode ----
        else:
            if query is None:
                raise ValueError("GQA mode requires `query`.")
            if self.num_q_heads > self.num_key_heads:
                query = query.view(
                    query.shape[0],
                    self.num_key_heads,
                    self.num_q_heads // self.num_key_heads,
                    query.shape[2],
                ).mean(2)
            elif self.num_q_heads < self.num_key_heads:
                query = torch.repeat_interleave(
                    query, self.num_key_heads // self.num_q_heads, dim=1
                )

            return self.hash_encoder.compute_hash(query).view(torch.bfloat16)

    def get_layer_attn_metadata(self, forward_context: ForwardContext, layer_name: str):
        attn_meta = forward_context.attn_metadata
        return attn_meta[layer_name] if isinstance(attn_meta, dict) else attn_meta

    def get_layer_state(self, layer_name: str):
        layer_id = int(layer_name.split(".")[2])
        is_rollback_layer = layer_id in self.hash_rollback_layers
        is_skip_hash_layer = (
            layer_id < len(self.hash_skip_layers) and self.hash_skip_layers[layer_id]
        )
        return is_rollback_layer, is_skip_hash_layer

    def attention_begin(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_name: str,
        forward_context: ForwardContext,
        output: Optional[torch.Tensor] = None,
        phase: Optional[str] = None,
        k_hash: Optional[torch.Tensor] = None,
        decode_ql_nope: Optional[torch.Tensor] = None,
        decode_q_pe: Optional[torch.Tensor] = None,
    ):
        attn_metadata = self.get_layer_attn_metadata(forward_context, layer_name)
        # TODO: Should mark MTP layer as rollback layer
        is_rollback_layer, is_skip_hash_layer = self.get_layer_state(layer_name)

        if not is_rollback_layer and not is_skip_hash_layer:
            if self.is_mla:
                k_c_normed_hash, k_pe_hash = self.hash_code(nope=key, rope=value)
                ops.concat_and_cache_mla(
                    k_c_normed_hash,
                    k_pe_hash.squeeze(1),
                    k_hash,
                    attn_metadata.slot_mapping.flatten(),
                    kv_cache_dtype="auto",
                    scale=self._k_scale,
                )
            else: # GQA
                if self.is_cuda:
                    k_hash_compute = self.hash_encoder.compute_hash(key).view(
                        torch.bfloat16
                    )
                    reshape_and_cache_khash_triton(
                        k_hash_compute,
                        attn_metadata.slot_mapping.flatten(),
                        k_hash,
                        block_size=self.block_size,
                    )
                else: # NPU
                    if not self.is_tensor_computed:
                        decode_req_ids = torch.nonzero(
                                self.decode_mask, as_tuple=False
                        ).flatten()
                        decode_req_ids_npu = torch.nonzero(
                                self.decode_mask_npu, as_tuple=False
                        ).flatten()
                        batch_size_for_hamming = self.decode_mask.sum().item()
                        self.query_lens_device = attn_metadata.query_lens_device[decode_req_ids_npu]
                        self.topk_for_hamming = self.topk_for_hamming_full[:batch_size_for_hamming]
                        self.chunk_size_for_hamming = self.chunk_sizes_for_hamming[:batch_size_for_hamming]
                        self.seq_lens_for_hamming = attn_metadata.seq_lens_device[decode_req_ids_npu]
                        self.max_seq_len_for_hamming = torch.max(attn_metadata.seq_lens[decode_req_ids]).item()
                        self.is_tensor_computed = True
                        
                    k_hash_compute = self.hash_encoder.compute_hash(key)
                    k_hash_compute = k_hash_compute.transpose(0,1).reshape(-1, k_hash_compute.shape[-1]).contiguous()
                    ucm_custom_ops.reshape_and_cache_bnsd(
                        k_hash_compute,
                        k_hash,
                        attn_metadata.slot_mapping,
                        attn_metadata.query_lens_device, # need to modify attention_v1.py in vllm-asecnd
                        k_hash,
                    )
        if self.is_mla:
            if phase == "decode":
                if not is_rollback_layer:
                    if is_skip_hash_layer:
                        assert attn_metadata.decode.topk_block_table is not None
                        block_table = attn_metadata.decode.topk_block_table
                    else:
                        q_nope_hash, q_rope_hash = self.hash_code(
                            nope=decode_ql_nope,
                            rope=decode_q_pe,
                            reduction_head_num=self.hash_reduction_head_num,
                        )
                        q_hash = torch.cat([q_nope_hash, q_rope_hash], dim=-1)
                        topk_token = self.hash_topk_tokens
                        block_table = cuda_hamming_topk(
                            q_hash.unsqueeze(1),
                            k_hash.unsqueeze(1),
                            attn_metadata.decode.block_table,
                            attn_metadata.decode.seq_lens,
                            topk_token=topk_token,
                            sink_token=64,
                            recent_token=512,
                        )
                        attn_metadata.decode.topk_block_table = block_table

                    seq_lens = attn_metadata.decode.topk_seq_lens
                    tile_scheduler_metadata = (
                        attn_metadata.decode.topk_tile_scheduler_metadata
                    )
                    num_splits = attn_metadata.decode.topk_num_splits

                    self.ori_block_table_decode = attn_metadata.decode.block_table
                    self.ori_seq_lens_decode = attn_metadata.decode.seq_lens
                    self.origin_tile_scheduler_metadata = (
                        attn_metadata.decode.tile_scheduler_metadata
                    )
                    self.origin_num_splits = attn_metadata.decode.num_splits

                    attn_metadata.decode.block_table = block_table
                    attn_metadata.decode.seq_lens = seq_lens
                    attn_metadata.decode.tile_scheduler_metadata = (
                        tile_scheduler_metadata
                    )
                    attn_metadata.decode.num_splits = num_splits
        else: #GQA
            q_start = attn_metadata.query_start_loc
            if self.decode_mask.any():  # 有decode阶段的req
                if not is_rollback_layer:
                    if is_skip_hash_layer:
                        # 跳层 使用上一个topk结果
                        attn_metadata.block_tables = self.topk_block_table
                        attn_metadata.seq_lens = self.topk_seq_lens
                    else:
                        if self.is_cuda:

                            decode_req_ids = torch.nonzero(
                                self.decode_mask, as_tuple=False
                            ).flatten()
                            decode_token_idx = q_start[:-1].index_select(0, decode_req_ids)
                            q_decode = query.index_select(0, decode_token_idx)
                            q_hash = self.hash_code(query=q_decode)

                            topk_token = self.hash_topk_tokens

                            block_table_decode = attn_metadata.block_table.index_select(
                                0, decode_req_ids
                            )
                            seq_len_decode = self.ori_seq_lens_decode.index_select(
                                0, decode_req_ids
                            )
                        
                            block_table_decode = cuda_hamming_topk(
                                q_hash.unsqueeze(1),
                                k_hash.unsqueeze(1),
                                block_table_decode,
                                seq_len_decode,
                                topk_token=topk_token,
                                sink_token=64,
                                recent_token=512,
                            )
                            # update topk_block_table
                            topk = block_table_decode.shape[1]
                            attn_metadata.block_table[decode_req_ids, :topk] = (
                                block_table_decode
                            )
                            attn_metadata.block_table[decode_req_ids, topk:] = 0

                            attn_metadata.seq_lens[self.decode_mask] = (
                                self.topk_seq_lens_qwen
                            )
                        else: # NPU


                            decode_req_ids = torch.nonzero(
                                self.decode_mask_npu, as_tuple=False
                            ).flatten()
                            decode_token_idx = q_start[:-1].index_select(0, decode_req_ids)
                            q_decode = query.index_select(0, decode_token_idx)

                            q_hash = self.self.hash_encoder.compute_hash(q_decode).unsqueeze(2).contiguous()

            
                            block_table_decode = attn_metadata.block_table.index_select(
                                0, decode_req_ids
                            )

                            ucm_custom_ops.hamming_dist_top_k(
                                q_hash,
                                k_hash,
                                self.topk_for_hamming,
                                self.seq_lens_for_hamming,
                                self.chunk_size_for_hamming,
                                self.max_seq_len_for_hamming,
                                self.hamming_keep_chunks_head,
                                self.hamming_keep_chunks_tail,
                                0, #support_offload is disabled
                                block_table_decode,
                                self.hamming_output[:len(decode_req_ids)]
                            )
                            topk = self.hamming_output.shape[1]
                            attn_metadata.block_table[decode_req_ids,:topk] = self.hamming_output[:len(decode_req_ids)]
                            attn_metadata.block_table[decode_req_ids,topk:] = 0
                            attn_metadata.seq_lens[self.decode_mask] = self.seq_lens_for_hamming

                        # topk for skip layer
                        self.topk_block_table = attn_metadata.block_table
                        self.topk_seq_lens = attn_metadata.seq_lens

        return query, key, value, output

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
        attn_metadata = self.get_layer_attn_metadata(forward_context, layer_name)
        if self.is_mla:
            if phase == "decode":
                # TODO: Should mark MTP layer as rollback layer
                is_rollback_layer, is_skip_hash_layer = self.get_layer_state(layer_name)
                if not is_rollback_layer:
                    attn_metadata.decode.block_table = self.ori_block_table_decode
                    attn_metadata.decode.seq_lens = self.ori_seq_lens_decode
                    attn_metadata.decode.tile_scheduler_metadata = (
                        self.origin_tile_scheduler_metadata
                    )
                    attn_metadata.decode.num_splits = self.origin_num_splits
        else:  # 判断req decode阶段
            if self.decode_mask.any():
                attn_metadata.block_table = self.ori_block_table_decode
                attn_metadata.seq_lens = self.ori_seq_lens_decode

    def request_begin(self, request_id: ReqType, prompt_token_ids: List[int]):
        pass

    def request_finished_in_scheduler(self, request_id: Union[int, str]):
        """
        This is called inside "Scheduler->finish_requests" function.
        Generate the metadata required by UcmSparse instance at worker-side.
        """
        pass

    def execute_begin(self, scheduler_output: SchedulerOutput):
        self.is_tensor_computed = False

    def estimate_num_slots_sparsed(self, request: Request) -> int:
        return INVALID_SLOT

    def initialize_kv_hash_cache_tensors(self, kv_caches, device):
        dtype = torch.bfloat16
        for layer_name, kv_cache in kv_caches.items():
            khash_cache_shape = list((kv_cache if self.is_mla else kv_cache[0]).shape)
            khash_cache_shape[-1] //= dtype.itemsize * 8
            khash_cache = torch.zeros(khash_cache_shape, dtype=dtype, device=device)
            kv_caches[layer_name] = (kv_cache, khash_cache)

    def initialize_kv_hash_cache_tensors_npu(self, kv_caches, device):
        for layer_name, kv_cache in kv_caches.items():
            is_rollback_layer, is_skip_hash_layer = self.get_layer_state(layer_name)
            k_cache_shape = kv_cache[0].shape
            khash_cache_shape = (k_cache_shape[0], k_cache_shape[2], k_cache_shape[1], self.hash_encoder.hash_bits // 8)
            if not is_rollback_layer and not is_skip_hash_layer:
                khash_cache = torch.empty(khash_cache_shape, dtype=torch.uint8, device=device)
            else:
                khash_cache = None
            kv_caches[layer_name] = (kv_cache, khash_cache)

    def build_decode_hash(self, seq_lens):
        from ucm.sparse.kvcomp.hamming_topk import update_seq_lens

        topk_seq_lens = update_seq_lens(
            seq_lens,
            topk_token=self.hash_topk_tokens,
            block_size=self.block_size,
        )
        topk_tile_scheduler_metadata, topk_num_splits = get_mla_metadata(
            topk_seq_lens,
            self.num_q_heads,
            1,
        )
        return topk_seq_lens, topk_tile_scheduler_metadata, topk_num_splits

    def build_decode_attention_meta(self, query_start_loc, seq_lens, block_table):

        from ucm.sparse.kvcomp.hamming_topk import update_seq_lens

        q_lens = query_start_loc[1:] - query_start_loc[:-1]
        self.decode_mask = q_lens == 1

        self.ori_seq_lens_decode = seq_lens.clone()
        self.ori_block_table_decode = block_table.clone()
        if self.decode_mask.any():
            decode_seq_lens = seq_lens[self.decode_mask]
            self.topk_seq_lens_qwen = update_seq_lens(
                decode_seq_lens,
                topk_token=self.hash_topk_tokens,
                block_size=self.block_size,
            )
        return self.decode_mask, self.topk_seq_lens_qwen
    
    def build_decode_attention_meta_npu(self, query_lens, seq_lens, block_table):

        from ucm.sparse.kvcomp.hamming_topk import update_seq_lens

        # self.decode_mask is on cpu in vllm-asencd under NPU device
        self.decode_mask = (query_lens == 1) & (seq_lens >= self.seq_len_threshhold)
        self.decode_mask = self.decode_mask.pin_memory()
        
        self.ori_seq_lens_decode = seq_lens.clone()
        self.ori_block_table_decode = block_table.clone()

        self.decode_mask_npu = self.decode_mask.to(self.device, non_blocking=True)
        
        if self.decode_mask.any():
            decode_seq_lens = seq_lens[self.decode_mask]
            self.topk_seq_lens_qwen = update_seq_lens(
                decode_seq_lens,
                topk_token=self.hash_topk_tokens,
                block_size=self.block_size,
            )

    def maybe_init_cudagraph_buffers_for_topk(self, n, tile_scheduler_metadata):
        sm_parts = tile_scheduler_metadata.size(0)
        topk_tile_scheduler_metadata_view = self.cg_buf_topk_tile_scheduler_metadata[
            :sm_parts
        ]
        topk_tile_scheduler_metadata_view.copy_(topk_tile_scheduler_metadata)
        topk_tile_scheduler_metadata = topk_tile_scheduler_metadata_view

        topk_num_splits_view = self.cg_buf_topk_num_splits[:n]
        topk_num_splits_view.copy_(topk_num_splits)
        self.cg_buf_topk_num_splits[n:].fill_(topk_num_splits[-1])
        topk_num_splits = topk_num_splits_view
        return topk_tile_scheduler_metadata, topk_num_splits
