from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import FullAttentionSpec

from ucm.logger import init_logger

logger = init_logger(__name__)


class CompressAttentionManager:
    def allocate_new_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks,
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        """Prevent inconsistent compressed hits from corrupting the free queue."""
        if request_id in self.num_cached_block:
            assert len(new_computed_blocks) == 0
            return

        req_blocks = self.req_to_blocks[request_id]
        assert len(req_blocks) == 0

        num_total_computed_tokens = (
            num_local_computed_tokens + num_external_computed_tokens
        ) // self.compress_ratio
        num_skipped_tokens = self.get_num_skipped_tokens(num_total_computed_tokens)
        num_skipped_blocks = num_skipped_tokens // self.block_size
        if num_skipped_blocks > 0:
            new_computed_blocks = new_computed_blocks[num_skipped_blocks:]
            num_external_computed_tokens = min(
                num_total_computed_tokens - num_skipped_tokens,
                num_external_computed_tokens,
            )

        if self.enable_caching:
            self.block_pool.touch(new_computed_blocks)
        else:
            assert not any(
                new_computed_blocks
            ), "Computed blocks should be empty when prefix caching is disabled"

        req_blocks.extend([self._null_block] * num_skipped_blocks)
        req_blocks.extend(new_computed_blocks)
        self.num_cached_block[request_id] = len(req_blocks)

        if num_external_computed_tokens > 0:
            target_blocks = cdiv(num_total_computed_tokens, self.block_size)
            num_new_blocks = target_blocks - len(req_blocks)
            if num_new_blocks <= 0:
                if num_new_blocks < 0:
                    logger.warning(
                        "Skip negative Ascend compressed-attention KV allocation: "
                        "request_id=%s, num_new_blocks=%d, target_blocks=%d, "
                        "req_blocks=%d, compress_ratio=%d, block_size=%d, "
                        "compressed_total_tokens=%d, num_skipped_blocks=%d, "
                        "kv_cache_group_id=%s, spec=%s",
                        request_id,
                        num_new_blocks,
                        target_blocks,
                        len(req_blocks),
                        self.compress_ratio,
                        self.block_size,
                        num_total_computed_tokens,
                        num_skipped_blocks,
                        getattr(self, "kv_cache_group_id", None),
                        type(self.kv_cache_spec).__name__,
                    )
                return

            allocated_blocks = self.block_pool.get_new_blocks(num_new_blocks)
            req_blocks.extend(allocated_blocks)
            if type(self.kv_cache_spec) is FullAttentionSpec:
                self.new_block_ids.extend(b.block_id for b in allocated_blocks)
