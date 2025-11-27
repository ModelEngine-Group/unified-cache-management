from dataclasses import dataclass, field


from ucm.logger import init_logger
from typing import List, Optional, Tuple, Any

from ucm.sparse.blend.blockwise_rope import block_wise_rope_forward
from ucm.sparse.blend.utils import get_rotary_emb_ops

logger = init_logger(__name__)

import itertools

@dataclass
class ChunkMetaData:
    # [start, start + len)
    start_idx_in_req: int
    chunk_tokens_len: int

    start_idx_in_req_blks: int
    chunk_blks_len: int

    cached_start_position: int

    vllm_blk_ids: List[int] = field(default_factory=list)
    chunk_blks_hash : List[str] = field(default_factory=list)
    store_hits: List[bool] = field(default_factory=list)

    @property
    def end_idx_in_req(self) -> int:
        return self.start_idx_in_req + self.chunk_tokens_len

    @property
    def end_idx_in_req_blks(self) -> int:
        return self.start_idx_in_req_blks + self.chunk_blks_len

    @property
    def cached_end_position(self) -> int:
        return self.cached_start_position + self.chunk_tokens_len

    @property
    def position_offset(self) -> int:
        return self.start_idx_in_req - self.cached_start_position

    @property
    def hits_vllm_blk_ids(self) -> List[int]:
        return list(itertools.compress(self.vllm_blk_ids, self.store_hits))

    @property
    def hits_chunk_blks_hash(self) -> List[str]:
        return list(itertools.compress(self.chunk_blks_hash, self.store_hits))



def hash_token_ids(
        hash_function: Any, block_size: int, token_ids: list[int], parent_block_hash_value = None
    ) -> List[str]:
    """
    process token_ids into prefix blk hashes with parent_block_hash_value
    """
    ret = []

    if not parent_block_hash_value:
        parent_block_hash_value = hash_function("UCMHASHSEED")

    for start in range(0, len(token_ids), block_size):
        end = start + block_size
        block_token_ids = token_ids[start:end]
        # Do not hash the block if it is not full.
        if len(block_token_ids) < block_size:
            break

        block_token_ids_tuple = tuple(block_token_ids)
        hash_value = hash_function(
            (parent_block_hash_value, block_token_ids_tuple)
        )
        parent_block_hash_value = str(hash_value)
        ret.append(parent_block_hash_value)

    return ret


class ChunkProcessor():
    """
    """

    def __init__(self, config):
        self.block_size = config['block_size']

        self.chunk_end_token_id = config['chunk_end_token_id']
        self.hash_function = None
        self.rotary_emb: Optional[callable] = None

    def update_meta4_partial_pc(self, rag_chunk_meta: ChunkMetaData, num_pc_part_blks: int) -> None:
        rag_chunk_meta.start_idx_in_req += num_pc_part_blks * self.block_size
        rag_chunk_meta.chunk_tokens_len -= num_pc_part_blks * self.block_size

        rag_chunk_meta.start_idx_in_req_blks += num_pc_part_blks
        rag_chunk_meta.chunk_blks_len -= num_pc_part_blks

        rag_chunk_meta.chunk_blks_hash = rag_chunk_meta.chunk_blks_hash[num_pc_part_blks:]
        rag_chunk_meta.cached_start_position += num_pc_part_blks * self.block_size

    def get_stage(self,all_prefill_tokens):
        build_cache = True
        if all_prefill_tokens[-1] == self.chunk_end_token_id and len(all_prefill_tokens) % self.block_size == 0:
            return build_cache
        return not build_cache

    def process_request(
            self,
            request,
            hash_function,
            rag_start_blk_idx,
    ) -> Tuple[List[ChunkMetaData], bool] :
        """
        Process the request to split prompt tokens into RAG chunks and construct metadata.

        Args:
            request: Request object containing prompt_token_ids.
            hash_function: Function used to compute prefix hash in each rag chunk, should be in line with kv store.
            rag_start_blk_idx: Start idx of vllm blocks where prefix cache matches end.

        Returns:
            Tuple of:
            - rag_chunks_meta: List of ChunkMetaData parsed from req.
            - is_build_cache: whether current req is in build cache stage .
        """
        if self.hash_function is None:
            self.hash_function = hash_function
        all_prefill_tokens = request.prompt_token_ids
        rag_chunks_meta: List[ChunkMetaData] = []

        # first judge current req is whether in build chunk cache stage or in use chunk cache stage
        # for future work, this two stage should not be exposed to user
        is_build_cache = self.get_stage(all_prefill_tokens)
        if is_build_cache:
            chunk_blks_hash = hash_token_ids(hash_function, self.block_size, all_prefill_tokens)
            chunk_tokens_len = len(all_prefill_tokens)
            chunk_blks_len = len(all_prefill_tokens) // self.block_size

            rag_chunk_meta = ChunkMetaData(
                start_idx_in_req=0,
                chunk_tokens_len=chunk_tokens_len,
                start_idx_in_req_blks=0,
                chunk_blks_len=chunk_blks_len,
                chunk_blks_hash=chunk_blks_hash,
                cached_start_position=0)
            return [rag_chunk_meta] , is_build_cache

        start_blk_idx = 0
        start_token_dix = 0
        for end_blk_idx, end_token_idx in enumerate(range(self.block_size - 1, len(all_prefill_tokens), self.block_size)):
            # only compare the last token id in each blk
            if all_prefill_tokens[end_token_idx] == self.chunk_end_token_id:
                chunk_token_ids = all_prefill_tokens[start_token_dix: end_token_idx + 1]
                chunk_blks_hash = hash_token_ids(hash_function, self.block_size, chunk_token_ids)
                chunk_blks_len = end_blk_idx - start_blk_idx + 1
                chunk_tokens_len = chunk_blks_len * self.block_size

                rag_chunk_meta = ChunkMetaData(
                    start_idx_in_req=start_token_dix,
                    chunk_tokens_len=chunk_tokens_len,
                    start_idx_in_req_blks=start_blk_idx,
                    chunk_blks_len=chunk_blks_len,
                    chunk_blks_hash = chunk_blks_hash,
                    cached_start_position=0)

                # update for next rag chunk
                start_blk_idx = end_blk_idx + 1
                start_token_dix = end_token_idx + 1

                if rag_chunk_meta.end_idx_in_req_blks <= rag_start_blk_idx:
                    # current rag chunk is fully in prefix cache, no need to process
                    continue

                # rag chunk is partly in prefix cache
                num_pc_part_blks = 0
                if rag_chunk_meta.start_idx_in_req_blks < rag_start_blk_idx:
                    num_pc_part_blks = rag_start_blk_idx - rag_chunk_meta.start_idx_in_req_blks
                    # blend stage, rag chunk in PC is no need to recompute
                    self.update_meta4_partial_pc(rag_chunk_meta, num_pc_part_blks)
                rag_chunks_meta.append(rag_chunk_meta)

        return rag_chunks_meta, is_build_cache

    def merge_chunks(self, old_chunk_meta: ChunkMetaData, chunk_meta:ChunkMetaData):
        # current we use a fix pattern(end with a fix token id) to recognize the text token chunk
        # in some special situation, one text chunk maybe split as multi text chunk, so we should merge them into one
        old_chunk_meta.chunk_tokens_len += chunk_meta.chunk_tokens_len
        old_chunk_meta.chunk_blks_len += chunk_meta.chunk_blks_len
        old_chunk_meta.chunk_blks_hash += chunk_meta.chunk_blks_hash
        old_chunk_meta.store_hits += chunk_meta.store_hits

    def setup_rotary_emb(self, model):
        self.rotary_emb = get_rotary_emb_ops(model)

    def process_chunk_cache(self, k_cache, vllm_ids, positions):
        """
        post process loaded chunk kcache
        """
        if self.rotary_emb is None:
            raise "Please call setup_rotary_emb first."
        # triton kernl for block-wise delta rope
        block_wise_rope_forward(k_cache, vllm_ids, positions, self.rotary_emb.cos_sin_cache)