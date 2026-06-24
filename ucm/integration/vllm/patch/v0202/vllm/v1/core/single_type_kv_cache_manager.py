from ucm.logger import init_logger

logger = init_logger(__name__)


class SingleTypeKVCacheManager:
    def free(self, request_id: str) -> None:
        """
        Free the blocks for the request.

        Args:
            request_id: The request ID.
        """
        # Default to [] in case a request is freed (aborted) before alloc.
        req_blocks = self.req_to_blocks.pop(request_id, [])

        # Free blocks in reverse order so that the tail blocks are freed first.
        # Deduplicate while preserving reverse order.
        seen_blocks = set()
        ordered_blocks = []

        for block in reversed(req_blocks):
            if block in seen_blocks:
                continue
            seen_blocks.add(block)
            ordered_blocks.append(block)

        self.block_pool.free_blocks(ordered_blocks)
        self.num_cached_block.pop(request_id, None)

    def remove_skipped_blocks(
        self, request_id: str, total_computed_tokens: int
    ) -> None:
        """
        Remove and free the blocks that are no longer needed for attention computation.
        The removed blocks should be replaced by null_block.

        This function depends on `get_num_skipped_tokens`, which need to be implemented
        differently for each attention type.

        Args:
            request_id: The request ID.
            total_computed_tokens: The total number of computed tokens, including
                local computed tokens and external computed tokens.
        """
        # Remove the blocks that will be skipped during attention computation.
        num_skipped_tokens = self.get_num_skipped_tokens(total_computed_tokens)
        if num_skipped_tokens <= 0:
            # This indicates that ALL tokens are inside attention window.
            # Thus we do not need to free any blocks outside attention window.
            # A typical case is full attention that we never free any token
            # before the request is finished.
            return
        blocks = self.req_to_blocks[request_id]
        num_skipped_blocks = num_skipped_tokens // self.block_size
        # `num_skipped_tokens` may include tokens that haven't been allocated yet
        # (e.g., when the attention window moves into the external computed tokens
        # range), so we must cap to the number of blocks that currently exist for
        # this request.
        num_skipped_blocks = min(num_skipped_blocks, len(blocks))
        removed_blocks = []
        # Because the block starts from index 0, the num_skipped_block-th block
        # corresponds to index num_skipped_blocks - 1.
        removed_block_ids: set[int] = set()
        duplicate_removed_block_ids: set[int] = set()

        for i in range(num_skipped_blocks - 1, -1, -1):
            block = blocks[i]
            if block == self._null_block:
                break

            blocks[i] = self._null_block
            if block.block_id in removed_block_ids:
                duplicate_removed_block_ids.add(block.block_id)
                continue

            removed_block_ids.add(block.block_id)
            removed_blocks.append(block)

        if duplicate_removed_block_ids:
            logger.warning(
                "Duplicate physical KV blocks found in remove_skipped_blocks: "
                "request_id=%s, total_computed_tokens=%d, "
                "num_skipped_tokens=%d, num_skipped_blocks=%d, "
                "duplicate_block_ids=%s",
                request_id,
                total_computed_tokens,
                num_skipped_tokens,
                num_skipped_blocks,
                sorted(duplicate_removed_block_ids),
            )

        if removed_blocks:
            self.block_pool.free_blocks(removed_blocks)
