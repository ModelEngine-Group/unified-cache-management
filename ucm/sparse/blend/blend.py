from typing import TYPE_CHECKING, Dict, List, Optional, Union, Tuple

import torch
import time
from sympy import false
from torch import Tensor

from ucm.logger import init_logger
from dataclasses import dataclass, field

from ucm.store.dramstore.dramstore_connector import device

logger = init_logger(__name__)

from vllm.forward_context import ForwardContext
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request
from vllm.config import VllmConfig
from ucm.sparse.base import (
    INVALID_SLOT,
    UcmSparseBase,
    UcmSparseMetadata,
    UcmSparseRole,
)
from ucm.sparse.blend.chunk_processor import ChunkMetaData
from ucm.sparse.utils import round_up

def get_num_blks(num_tokens, block_size):
    return (num_tokens + block_size - 1) // block_size
    
@dataclass
class ReqMeta:
    req_idx: int = 0
    need_blend: bool = false

    prefix_len: int = 0
    prefix_blk_len: int = 0

    chunks_len: int = 0
    chunks_blk_len: int = 0

    suffix_len: int = 0
    suffix_blk_len: int = 0

    chunk_hit_mask: List[bool] = field(default_factory=list)

    chunk_hit_blk_len : int = 0


@dataclass
class BlendMetaData(UcmSparseMetadata):
    requests: list[ReqMeta] = field(default_factory=list)
    compute_mask: Tensor = None
    chunk_blks_hit_mask: Tensor = None
    query_lens: Tensor = None
    blend_start_req_idx: int = 0
    need_re_index: bool = False

    def add_request(
        self,
        idx: int,
        chunks_load_meta: list[ChunkMetaData],
        seq_lens: Tensor,
        block_size: int
    ) -> None:
        if chunks_load_meta:
            hit_mask = []
            req_idx = idx + self.blend_start_req_idx
            for meta in chunks_load_meta:
                hit_mask.extend(meta.store_hits)
            reqMeta = ReqMeta(
                req_idx=req_idx,
                prefix_len=chunks_load_meta[0].start_idx_in_req,
                prefix_blk_len=get_num_blks(chunks_load_meta[0].start_idx_in_req, block_size),
                chunks_len=len(hit_mask) * block_size,
                chunks_blk_len=len(hit_mask),
                chunk_hit_mask=hit_mask,
                chunk_hit_blk_len=sum(hit_mask)
            )
            reqMeta.need_blend = reqMeta.chunk_hit_blk_len > 0
            reqMeta.suffix_len = seq_lens[req_idx].item() - reqMeta.prefix_len - reqMeta.chunks_len
            reqMeta.suffix_blk_len = get_num_blks(reqMeta.suffix_len, block_size)

            self.requests.append(reqMeta)

    def reset_compute_mask(self) -> None:
        self.compute_mask.fill_(False)
        # for decode req in the front of the batch
        self.compute_mask[:self.blend_start_req_idx] = True

    def update_req_compute_mask(self,req_query_start, req_chunk_end,  req_query_end, chunk_hit_mask, top_k_indices):
        chunks = self.compute_mask[req_query_start:req_chunk_end]
        chunks = chunks.reshape(len(chunk_hit_mask), -1)

        # for chunk block cache miss part, just recompute
        chunks.masked_fill_(~chunk_hit_mask.unsqueeze(1), True)

        flat = chunks.view(-1)
        # for chunk block cache hit part, just recompute HKVD(highest KV deviation) tokens
        flat[top_k_indices] = True

        # for question part, default
        self.compute_mask[req_chunk_end:req_query_end].fill_(True)


class Blend(UcmSparseBase):
    def __init__(self, vllm_config: VllmConfig, role: UcmSparseRole):
        super().__init__(vllm_config, role)
        self.blend_config = vllm_config.kv_transfer_config.kv_connector_extra_config[
            "ucm_sparse_config"
        ]["Blend"]

        max_model_len = vllm_config.model_config.max_model_len
        self.block_size = vllm_config.cache_config.block_size

        self.device = vllm_config.device_config.device
        self.forward_mask = torch.zeros(max_model_len, device=self.device).bool()
        self.mask_idx = torch.arange(round_up(max_model_len, self.block_size), device=self.device)
        self.mask_idx = self.mask_idx.reshape(-1, self.block_size)

        # for multi batch, ignore the decode-stage req at the beginning
        self.blend_start_req_idx = 0

        self.compute_meta = self.blend_config['compute_meta']
        self.blend_req_metas: BlendMetaData = BlendMetaData(
            need_re_index=False,
            chunk_blks_hit_mask = torch.zeros(round_up(max_model_len, self.block_size), device=self.device).bool()
        )
        self.attn_metadata = None

    def reset_blend_meta(self):
        self.blend_req_metas.need_re_index = False
        self.blend_req_metas.requests=[]


    def build_sparse_meta(
            self, scheduler_output, requests, input_batch, attn_metadata
    ) -> UcmSparseMetadata:

        if isinstance(attn_metadata, dict):
            attn_metadata = next(iter(attn_metadata.values()))
        self.attn_metadata = attn_metadata

        # current not support chunk prefill
        # for multi req in one batch, we should discard the decode req
        self.reset_blend_meta()
        self.blend_req_metas.compute_mask = self.forward_mask[:attn_metadata.query_start_loc[-1]]

        self.blend_req_metas.query_lens = attn_metadata.query_start_loc[1:] - attn_metadata.query_start_loc[:-1]
        self.blend_req_metas.blend_start_req_idx = len(scheduler_output.scheduled_cached_reqs.req_ids)

        for idx, req in enumerate(scheduler_output.kv_connector_metadata.requests):
            self.blend_req_metas.add_request(idx, req.rag_chunks_load_meta, attn_metadata.seq_lens, self.block_size)

        return self.blend_req_metas

    def update_attn_metadata(self):
        # update attn_metadata, cause we sparse the prefill tokens
        self.attn_metadata.slot_mapping = self.attn_metadata.slot_mapping[self.blend_req_metas.compute_mask]
        self.attn_metadata.query_start_loc[1:] = torch.cumsum(self.blend_req_metas.query_lens, dim=0)
        self.attn_metadata.max_query_len = self.blend_req_metas.query_lens.max().item()
        self.attn_metadata.num_actual_tokens = self.blend_req_metas.query_lens.sum().item()

    def estimate_num_slots_sparsed(self, request: Request) -> int:
        """
        This is called by "Scheduler->schedule" function to estimate the number of required blocks.
        """
        return INVALID_SLOT

    def request_begin(self, request_id: Union[int, str], prompt_token_ids: List[int]):
        pass

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
        attn = forward_context.no_compile_layers[layer_name]
        kv_cache = attn.kv_cache[forward_context.virtual_engine]
        start_time = time.perf_counter()
        if layer_name in self.compute_meta.keys():
            need_update = False
            self.blend_req_metas.reset_compute_mask()

            # maybe we can use triton kernel
            for req_meta in self.blend_req_metas.requests:
                req_idx = req_meta.req_idx
                req_query_start = self.attn_metadata.query_start_loc[req_idx].item()
                req_query_end = self.attn_metadata.query_start_loc[req_idx + 1].item()

                if not req_meta.need_blend:
                    self.blend_req_metas.compute_mask[req_query_start:req_query_end].fill_(True)
                    continue
                req_chunk_end = req_query_start + req_meta.chunks_len

                # HBM prefix cache is not supported now
                # UC store prefix cache can be fully reused
                his_vllm_blk_ids = self.attn_metadata.block_table[req_idx][req_meta.prefix_blk_len:
                                                                           req_meta.prefix_blk_len + req_meta.chunks_blk_len]
                # only compute topk of chunk's hits block
                chunk_hit_mask = self.blend_req_metas.chunk_blks_hit_mask[:len(req_meta.chunk_hit_mask)]
                src = torch.as_tensor(req_meta.chunk_hit_mask, dtype=chunk_hit_mask.dtype, device=chunk_hit_mask.device)
                chunk_hit_mask.copy_(src)

                his_vllm_blk_ids = his_vllm_blk_ids[chunk_hit_mask]
                his_k = kv_cache[0, his_vllm_blk_ids]
                candidate_len = req_meta.chunk_hit_blk_len * self.block_size
                his_k = his_k.reshape(candidate_len, -1)

                req_key = key[req_query_start:req_chunk_end]

                # req_key does not contain prefix cache
                golden_k = req_key.reshape(req_meta.chunks_blk_len, self.block_size, -1)[chunk_hit_mask]
                golden_k = golden_k.reshape(candidate_len, -1)

                diff_k = torch.sum((his_k - golden_k).abs(), dim=[1])
                topK_num = int(candidate_len * self.compute_meta[layer_name]["ratio"])

                topK_indices = torch.topk(diff_k, k=topK_num).indices

                # get origin idx in req_key
                topK_indices = self.mask_idx[:req_meta.chunks_blk_len][chunk_hit_mask].reshape(-1)[topK_indices]

                # update compute_mask
                self.blend_req_metas.update_req_compute_mask(req_query_start, req_chunk_end, req_query_end,
                                                             chunk_hit_mask, topK_indices)

                self.blend_req_metas.query_lens[req_idx] -= (candidate_len - topK_num)
                need_update = True

            if need_update:
                logger.info(f"[blend] compute_mask time: {(time.perf_counter() - start_time) * 1000}ms")
                self.blend_req_metas.need_re_index = True
                self.update_attn_metadata()

                indexed_query = query[self.blend_req_metas.compute_mask]
                indexed_key = key[self.blend_req_metas.compute_mask]
                indexed_value = value[self.blend_req_metas.compute_mask]
                indexed_output = None
                if output is not None:
                    indexed_output = output[:self.blend_req_metas.compute_mask.sum()]
                logger.info(f"[blend]  compute_mask time + index time: {(time.perf_counter() - start_time) * 1000}ms")
                return indexed_query, indexed_key, indexed_value, indexed_output
        return query, key, value, output

    def attention_end(self, output: torch.Tensor, layer_name: str):
        if layer_name in self.compute_meta.keys() and self.blend_req_metas.need_re_index:
            return output[:self.attn_metadata.num_actual_tokens]
        return output

    def self_attention_finished(
            self,
            residual: torch.Tensor,
            hidden_states: torch.Tensor
    ) -> torch.Tensor:
        if len(residual) != len(hidden_states):
            return self.index_tensor(residual)
        return residual

    def layer_finished(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor
    ) -> torch.Tensor:
        if len(positions) != len(hidden_states):
            return self.index_tensor(positions)
        return positions

    def execute_finished(self, logits_indices :torch.Tensor):
        if self.blend_req_metas.need_re_index:
            return self.attn_metadata.query_start_loc[1:] - 1
        return logits_indices

    def index_tensor(self, tensor: torch.Tensor):
        if self.blend_req_metas.need_re_index:
            return tensor[self.blend_req_metas.compute_mask]
        return tensor
