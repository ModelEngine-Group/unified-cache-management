from typing import Optional

from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.request import Request


class KVCacheManager:

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: Optional[KVCacheBlocks] = None,
        num_lookahead_tokens: int = 0,
        delay_cache_blocks: bool = False,
        num_encoder_tokens: int = 0,
    ) -> Optional[KVCacheBlocks]:
        """Add slots for a request with new tokens to append.

        Args:
            request: The request to allocate slots.
            num_new_tokens: The number of tokens to allocate, including external
                tokens. Note that this does not include tokens that have
                already been computed locally (i.e. new_computed_blocks).
            num_new_computed_tokens: The number of new computed tokens just
                hitting the prefix caching, excluding external tokens.
            new_computed_blocks: The cached blocks for the above new computed
                tokens.
            num_lookahead_tokens: The number of speculative tokens to allocate.
                This is used by spec decode proposers with kv-cache such
                as eagle.
            delay_cache_blocks: Whether to skip caching the blocks. This is
                used by P/D when allocating blocks used in a KV transfer
                which will complete in a future step.

        Blocks layout:
        ```
        -----------------------------------------------------------------------
        | < computed > | < new computed > |    < new >    | < pre-allocated > |
        -----------------------------------------------------------------------
        |                  < required >                   |
        --------------------------------------------------
        |                    < full >                  |
        ------------------------------------------------
                                          | <new full> |
                                          --------------
        ```
        The following *_blocks are illustrated in this layout.

        Returns:
            A list of new allocated blocks.
        """
        if num_new_tokens == 0:
            raise ValueError("num_new_tokens must be greater than 0")

        if new_computed_blocks is not None:
            new_computed_block_list = new_computed_blocks.blocks
        else:
            new_computed_block_list = tuple(
                [] for _ in range(len(self.kv_cache_config.kv_cache_groups))
            )

        self.coordinator.remove_skipped_blocks(
            request.request_id, request.num_computed_tokens
        )

        num_computed_tokens = request.num_computed_tokens + num_new_computed_tokens
        num_tokens_need_slot = min(
            num_computed_tokens + num_new_tokens + num_lookahead_tokens,
            self.max_model_len,
        )

        num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=num_tokens_need_slot,
            new_computed_blocks=new_computed_block_list,
            num_encoder_tokens=num_encoder_tokens,
        )

        if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
            return None

        if self.enable_caching:
            self.block_pool.touch(new_computed_block_list)
        else:
            assert not any(new_computed_block_list), (
                "Computed blocks should be empty when " "prefix caching is disabled"
            )

        self.coordinator.save_new_computed_blocks(
            request.request_id, new_computed_block_list
        )

        new_blocks = self.coordinator.allocate_new_blocks(
            request.request_id, num_tokens_need_slot, num_encoder_tokens
        )

        if not self.enable_caching or delay_cache_blocks:
            return KVCacheBlocks(new_blocks)

        num_tokens_to_cache = min(
            num_computed_tokens + num_new_tokens, request.num_tokens
        )
        self.coordinator.cache_blocks(request, num_tokens_to_cache)

        return KVCacheBlocks(new_blocks)
