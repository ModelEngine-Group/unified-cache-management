from vllm.v1.core.kv_cache_utils import BlockHashListWithBlockSize
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec, MambaSpec


def _effective_block_size(kv_cache_spec: KVCacheSpec) -> int:
    block_size = kv_cache_spec.block_size
    if hasattr(kv_cache_spec, "compress_ratio"):
        compress_ratio = kv_cache_spec.compress_ratio or 1
        compress_ratio = compress_ratio if compress_ratio >= 1 else 1
        block_size *= compress_ratio
    return block_size


class AscendHybridKVCacheCoordinator:
    def find_longest_cache_hit(
        self,
        block_hashes,
        max_cache_hit_length: int,
    ):
        """Patch Ascend hybrid prefix cache to truncate every full-attention group."""

        def _get_block_hashes(kv_cache_spec: KVCacheSpec):
            target_block_size = kv_cache_spec.block_size
            if not isinstance(kv_cache_spec, MambaSpec) and self.dcp_world_size * self.pcp_world_size > 1:
                target_block_size *= self.dcp_world_size * self.pcp_world_size
            if target_block_size == self.hash_block_size:
                return block_hashes
            return BlockHashListWithBlockSize(
                block_hashes, self.hash_block_size, target_block_size
            )

        num_groups = len(self.kv_cache_config.kv_cache_groups)
        hit_length = max_cache_hit_length
        hit_blocks_by_group = [None] * num_groups

        is_simple_hybrid = len(self.attention_groups) == 2 and isinstance(
            self.attention_groups[0][0], FullAttentionSpec
        )

        eagle_verified: set[int] = set()

        while True:
            curr_hit_length = hit_length

            for idx, (spec, group_ids, manager_cls) in enumerate(self.attention_groups):
                effective_block_size = self._get_effective_block_size(spec)
                cached_blocks = hit_blocks_by_group[group_ids[0]]
                if isinstance(spec, FullAttentionSpec) and cached_blocks is not None:
                    num_blocks = curr_hit_length // effective_block_size
                    curr_hit_length = num_blocks * effective_block_size
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
                    dcp_world_size=self.dcp_world_size,
                    pcp_world_size=self.pcp_world_size,
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

        for spec, group_ids, _ in self.attention_groups:
            if not isinstance(spec, FullAttentionSpec):
                continue
            num_blocks = hit_length // self._get_effective_block_size(spec)
            for group_id in group_ids:
                if (blocks := hit_blocks_by_group[group_id]) is not None:
                    del blocks[num_blocks:]

        return (
            tuple(
                blocks if blocks is not None else [] for blocks in hit_blocks_by_group
            ),
            hit_length,
        )
