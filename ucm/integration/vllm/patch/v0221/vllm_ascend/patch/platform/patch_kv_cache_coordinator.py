from vllm.v1.core.kv_cache_utils import BlockHashListWithBlockSize
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec, MambaSpec


def _find_longest_cache_hit(
    coordinator,
    block_hashes,
    max_cache_hit_length: int,
    *,
    skip_mamba: bool,
    eagle_kwarg_name: str,
):
    """Run the v0.22.1 lookup and truncate every full-attention group."""

    def _get_block_hashes(kv_cache_spec: KVCacheSpec):
        target_block_size = kv_cache_spec.block_size
        if (
            not isinstance(kv_cache_spec, MambaSpec)
            and coordinator.dcp_world_size * coordinator.pcp_world_size > 1
        ):
            target_block_size *= coordinator.dcp_world_size * coordinator.pcp_world_size
        if target_block_size == coordinator.hash_block_size:
            return block_hashes
        return BlockHashListWithBlockSize(
            block_hashes, coordinator.hash_block_size, target_block_size
        )

    num_groups = len(coordinator.kv_cache_config.kv_cache_groups)
    hit_length = max_cache_hit_length
    hit_blocks_by_group = [None] * num_groups

    is_simple_hybrid = len(coordinator.attention_groups) == 2 and isinstance(
        coordinator.attention_groups[0][0], FullAttentionSpec
    )
    eagle_verified: set[int] = set()

    while True:
        curr_hit_length = hit_length
        for idx, (spec, group_ids, manager_cls) in enumerate(
            coordinator.attention_groups
        ):
            if skip_mamba and isinstance(spec, MambaSpec):
                if hit_blocks_by_group[group_ids[0]] is None:
                    for group_id in group_ids:
                        hit_blocks_by_group[group_id] = []
                continue

            effective_block_size = coordinator._get_effective_block_size(spec)
            cached_blocks = hit_blocks_by_group[group_ids[0]]
            if isinstance(spec, FullAttentionSpec) and cached_blocks is not None:
                curr_hit_length = (
                    curr_hit_length // effective_block_size * effective_block_size
                )
                continue

            use_eagle = (
                idx in coordinator.eagle_attn_group_indices
                and idx not in eagle_verified
            )
            max_length = curr_hit_length
            if use_eagle:
                max_length = min(
                    curr_hit_length + spec.block_size, max_cache_hit_length
                )

            eagle_kwarg = {eagle_kwarg_name: use_eagle}
            hit_blocks = manager_cls.find_longest_cache_hit(
                block_hashes=_get_block_hashes(spec),
                max_length=max_length,
                kv_cache_group_ids=group_ids,
                block_pool=coordinator.block_pool,
                kv_cache_spec=spec,
                **eagle_kwarg,
                alignment_tokens=coordinator.lcm_block_size,
                dcp_world_size=coordinator.dcp_world_size,
                pcp_world_size=coordinator.pcp_world_size,
            )
            new_hit_length = len(hit_blocks[0]) * effective_block_size
            if use_eagle:
                eagle_verified.add(idx)
            elif new_hit_length < curr_hit_length:
                eagle_verified.clear()
            curr_hit_length = new_hit_length
            for group_id, blocks in zip(group_ids, hit_blocks):
                hit_blocks_by_group[group_id] = blocks

        if curr_hit_length >= hit_length:
            break
        hit_length = curr_hit_length
        if is_simple_hybrid:
            break

    # DeepSeek-V4 can have multiple compressed full-attention groups (for
    # example c4 and c128). Every one of them must agree with the final shared
    # token hit length, using its own effective token block size.
    for spec, group_ids, _ in coordinator.attention_groups:
        if not isinstance(spec, FullAttentionSpec):
            continue
        num_blocks = hit_length // coordinator._get_effective_block_size(spec)
        for group_id in group_ids:
            if (blocks := hit_blocks_by_group[group_id]) is not None:
                del blocks[num_blocks:]

    return (
        tuple(blocks if blocks is not None else [] for blocks in hit_blocks_by_group),
        hit_length,
    )


class AscendHybridKVCacheCoordinator:
    def find_longest_cache_hit(
        self,
        block_hashes,
        max_cache_hit_length: int,
    ):
        return _find_longest_cache_hit(
            self,
            block_hashes,
            max_cache_hit_length,
            skip_mamba=False,
            eagle_kwarg_name="use_eagle",
        )

    def find_longest_cache_hit_per_group(
        self,
        block_hashes,
        max_cache_hit_length: int,
    ):
        return _find_longest_cache_hit(
            self,
            block_hashes,
            max_cache_hit_length,
            skip_mamba=True,
            eagle_kwarg_name="drop_eagle_block",
        )
