# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from typing import Optional, Union

from vllm.logger import init_logger

from ucm.integration.vllm.ucm_sparse.base import INVALID_SLOT

# from vllm.v1.request import Request
from ucm.integration.vllm.vllm_adapter.v1.request import Request

logger = init_logger(__name__)

import vllm.v1.core.kv_cache_manager as vllm_v1_kv_cache_manager
from vllm.v1.core.kv_cache_manager import KVCacheBlocks


class KVCacheManager(vllm_v1_kv_cache_manager.KVCacheManager):

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: Optional[KVCacheBlocks] = None,
        num_draft_tokens: int = 0,
        num_lookahead_tokens: int = 0,
        delay_cache_blocks: bool = False,
        num_slots_sparsed: Union[None, int] = None,
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

        if num_slots_sparsed != INVALID_SLOT:
            self.block_size = self.kv_cache_config.kv_cache_groups[
                0
            ].kv_cache_spec.block_size
            num_blocks_need = math.ceil(num_slots_sparsed / self.block_size)
            allocated_blocks = self.coordinator.get_blocks(request.request_id)[0]
            returned_blocks = []
            sparsed_blocks = []
            for i, block in enumerate(allocated_blocks):
                if i < num_blocks_need:
                    sparsed_blocks.append(block)
                else:
                    returned_blocks.append(block)
                self.block_pool._maybe_evict_cached_block(block)
            self.block_pool.free_blocks(returned_blocks)
            self.coordinator.single_type_managers[0].req_to_blocks[
                request.request_id
            ] = sparsed_blocks
            new_computed_block_list = tuple(
                [] for _ in range(len(self.kv_cache_config.kv_cache_groups))
            )
            num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
                request_id=request.request_id,
                num_tokens=num_slots_sparsed,
                new_computed_blocks=new_computed_block_list,
            )
            if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
                return None
            new_blocks = self.coordinator.allocate_new_blocks(
                request.request_id, num_slots_sparsed
            )
            return KVCacheBlocks(tuple([sparsed_blocks]))

        if new_computed_blocks is not None:
            new_computed_block_list = new_computed_blocks.blocks
        else:
            new_computed_block_list = tuple(
                [] for _ in range(len(self.kv_cache_config.kv_cache_groups))
            )

        # Free the blocks that are skipped during the attention computation
        # (e.g., tokens outside the sliding window).
        # We can do this even if we cannot schedule this request due to
        # insufficient free blocks.
        # Should call this function before allocating new blocks to reduce
        # the number of evicted blocks.
        self.coordinator.remove_skipped_blocks(
            request.request_id, request.num_computed_tokens
        )

        # The number of computed tokens is the number of computed tokens plus
        # the new prefix caching hits
        num_computed_tokens = request.num_computed_tokens + num_new_computed_tokens
        num_tokens_need_slot = min(
            num_computed_tokens + num_new_tokens + num_lookahead_tokens,
            self.max_model_len,
        )

        num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=num_tokens_need_slot,
            new_computed_blocks=new_computed_block_list,
        )

        if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
            # Cannot allocate new blocks
            return None

        # Touch the computed blocks to make sure they won't be evicted.
        if self.enable_caching:
            self.block_pool.touch(new_computed_block_list)
        else:
            assert not any(new_computed_block_list), (
                "Computed blocks should be empty when " "prefix caching is disabled"
            )

        # Append the new computed blocks to the request blocks until now to
        # avoid the case where the new blocks cannot be allocated.
        self.coordinator.save_new_computed_blocks(
            request.request_id, new_computed_block_list
        )

        new_blocks = self.coordinator.allocate_new_blocks(
            request.request_id, num_tokens_need_slot
        )

        # P/D: delay caching blocks if we have to recv from
        # remote. Update state for locally cached blocks.
        if not self.enable_caching or delay_cache_blocks:
            return KVCacheBlocks(new_blocks)

        # Speculated tokens might be rejected in the future, so we does
        # not cache any speculated tokens. We only cache blocks with
        # generated (accepted) tokens.
        self.coordinator.cache_blocks(
            request,
            self.req_to_block_hashes[request.request_id],
            num_computed_tokens + num_new_tokens - num_draft_tokens,
        )

        return KVCacheBlocks(new_blocks)


vllm_v1_kv_cache_manager.KVCacheManager = KVCacheManager
