from ucm.logger import init_logger

logger = init_logger(__name__)


class BlockPool:
    def free_blocks(self, ordered_blocks) -> None:
        """Free physical KV blocks without corrupting the free-block queue."""

        # UCM double-free guard patch start: filter duplicate/null blocks.
        blocks = []
        seen_block_ids: set[int] = set()
        duplicate_block_ids: set[int] = set()

        for block in ordered_blocks:
            # The shared null block does not participate in reference counting
            # and must never be inserted into the free-block queue.
            if block.is_null:
                continue

            block_id = block.block_id
            if block_id in seen_block_ids:
                duplicate_block_ids.add(block_id)
                continue

            seen_block_ids.add(block_id)
            blocks.append(block)

        if duplicate_block_ids:
            logger.warning(
                "Duplicate physical KV blocks passed to BlockPool.free_blocks; "
                "freeing each block once: block_ids=%s",
                sorted(duplicate_block_ids),
            )
        # UCM double-free guard patch end.

        freed_blocks = []
        for block in blocks:
            # UCM double-free guard patch start: warn and skip invalid frees.
            if block.ref_cnt <= 0:
                logger.warning(
                    "Skip freeing KV block with non-positive ref_cnt: "
                    "block_id=%d, ref_cnt=%d",
                    block.block_id,
                    block.ref_cnt,
                )
                continue

            if block.prev_free_block is not None or block.next_free_block is not None:
                logger.warning(
                    "Skip freeing KV block that is already linked in the free "
                    "queue: block_id=%d, ref_cnt=%d, prev_block_id=%s, "
                    "next_block_id=%s",
                    block.block_id,
                    block.ref_cnt,
                    (
                        block.prev_free_block.block_id
                        if block.prev_free_block is not None
                        else None
                    ),
                    (
                        block.next_free_block.block_id
                        if block.next_free_block is not None
                        else None
                    ),
                )
                continue
            # UCM double-free guard patch end.

            block.ref_cnt -= 1
            if block.ref_cnt == 0:
                freed_blocks.append(block)

        self.free_block_queue.append_n(freed_blocks)
