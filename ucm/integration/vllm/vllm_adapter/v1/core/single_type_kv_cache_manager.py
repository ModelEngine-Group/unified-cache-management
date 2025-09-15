from vllm.v1.request import Request
from vllm.v1.core.kv_cache_utils import BlockHash

def cache_blocks(self, request: Request, block_hashes: list[BlockHash],
                    num_tokens: int) -> None:
    """
    Cache the blocks for the request.

    Args:
        request: The request.
        block_hashes: The block hashes of the request.
        num_tokens: The total number of tokens that need to be cached 
            (including tokens that are already cached).
    """
    num_cached_blocks = self.num_cached_block[request.request_id]
    num_full_blocks = num_tokens // self.block_size
    
    if num_cached_blocks >= num_full_blocks:
        return

    self.block_pool.cache_full_blocks(
        request=request,
        blocks=self.req_to_blocks[request.request_id],
        block_hashes=block_hashes,
        num_cached_blocks=num_cached_blocks,
        num_full_blocks=num_full_blocks,
        block_size=self.block_size,
        kv_cache_group_id=self.kv_cache_group_id,
        hash_fn=self.caching_hash_fn,
    )

    self.num_cached_block[request.request_id] = num_full_blocks
