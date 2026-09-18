from vllm.v1.core.kv_cache_utils import BlockHashListWithBlockSize
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec


def _effective_block_size(kv_cache_spec: KVCacheSpec) -> int:
    return kv_cache_spec.block_size * max(
        getattr(kv_cache_spec, "compress_ratio", 1), 1
    )


class AscendHybridKVCacheCoordinator:
    def find_longest_cache_hit(
        self,
        block_hashes,
        max_cache_hit_length: int,
    ):
        """Patch Ascend hybrid prefix cache to truncate every full-attention group."""

        def _get_block_hashes(kv_cache_spec: KVCacheSpec):
            if kv_cache_spec.block_size == self.hash_block_size:
                return block_hashes
            return BlockHashListWithBlockSize(
                block_hashes, self.hash_block_size, kv_cache_spec.block_size
            )

        num_groups = len(self.kv_cache_config.kv_cache_groups)
        hit_length = max_cache_hit_length
        hit_blocks_by_group = [None] * num_groups

        # Simple hybrid (1 full attn + 1 other): one iteration suffices.
        # Full attn is always first if it exists.
        is_simple_hybrid = len(self.attention_groups) == 2 and isinstance(
            self.attention_groups[0][0], FullAttentionSpec
        )

        eagle_verified: set[int] = set()

        while True:
            curr_hit_length = hit_length

            for idx, (spec, group_ids, manager_cls) in enumerate(self.attention_groups):
                effective_block_size = _effective_block_size(spec)
                cached_blocks = hit_blocks_by_group[group_ids[0]]
                if isinstance(spec, FullAttentionSpec) and cached_blocks is not None:
                    # Full attention is downward-closed. Reuse the original
                    # lookup and align to this group's effective token block.
                    curr_hit_length = (
                        curr_hit_length // effective_block_size * effective_block_size
                    )
                    continue

                use_eagle = (
                    idx in self.eagle_attn_group_indices and idx not in eagle_verified
                )

                _max_length = curr_hit_length
                if use_eagle:
                    _max_length = min(
                        curr_hit_length + spec.block_size, max_cache_hit_length
                    )
                hit_blocks = manager_cls.find_longest_cache_hit(
                    block_hashes=_get_block_hashes(spec),
                    max_length=_max_length,
                    kv_cache_group_ids=group_ids,
                    block_pool=self.block_pool,
                    kv_cache_spec=spec,
                    use_eagle=use_eagle,
                    alignment_tokens=self.lcm_block_size,
                )
                _new_hit_length = len(hit_blocks[0]) * effective_block_size
                if use_eagle:
                    eagle_verified.add(idx)
                elif _new_hit_length < curr_hit_length:
                    eagle_verified.clear()
                curr_hit_length = _new_hit_length
                for group_id, blocks in zip(group_ids, hit_blocks):
                    hit_blocks_by_group[group_id] = blocks

            if curr_hit_length >= hit_length:
                break
            hit_length = curr_hit_length
            if is_simple_hybrid:
                break

        # UCM patch for vllm-ascend 0.20.2:
        # vllm-ascend originally truncated only attention_groups[0]. DeepSeek
        # Ascend layouts can contain multiple compressed full-attention groups,
        # so each full-attention group must be truncated to the final shared
        # hit_length using its effective token block size.
        for spec, group_ids, _ in self.attention_groups:
            if not isinstance(spec, FullAttentionSpec):
                continue
            num_blocks = hit_length // _effective_block_size(spec)
            for group_id in group_ids:
                if (blocks := hit_blocks_by_group[group_id]) is not None:
                    del blocks[num_blocks:]

        return (
            tuple(
                blocks if blocks is not None else [] for blocks in hit_blocks_by_group
            ),
            hit_length,
        )
