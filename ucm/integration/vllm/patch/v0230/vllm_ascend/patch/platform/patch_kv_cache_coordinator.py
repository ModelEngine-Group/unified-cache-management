"""Hybrid prefix-cache result correction for vLLM-Ascend 0.23.0rc1."""

from functools import wraps

from vllm.v1.kv_cache_interface import FullAttentionSpec


def _truncate_full_attention_groups(
    coordinator,
    hit_blocks_by_group,
    hit_length: int,
) -> None:
    """Align every full-attention group with the final shared hit length."""
    for spec, group_ids, _ in coordinator.attention_groups:
        if not isinstance(spec, FullAttentionSpec):
            continue

        num_blocks = hit_length // coordinator._get_effective_block_size(spec)
        for group_id in group_ids:
            del hit_blocks_by_group[group_id][num_blocks:]


def wrap_full_attention_cache_hit_lookup(original_method):
    """Preserve the rc1 lookup algorithm and correct all returned full groups."""

    @wraps(original_method)
    def wrapped(coordinator, *args, **kwargs):
        hit_blocks_by_group, hit_length = original_method(
            coordinator, *args, **kwargs
        )
        _truncate_full_attention_groups(
            coordinator,
            hit_blocks_by_group,
            hit_length,
        )
        return hit_blocks_by_group, hit_length

    return wrapped
