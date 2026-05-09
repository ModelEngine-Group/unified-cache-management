def update_from_output(
    self,
    num_cached_tokens: int,
    num_external_computed_tokens: int,
    prompt_len: int,
) -> None:
    """Update stats from a prefill output."""
    recomputed = 1 if (num_cached_tokens + 1 == prompt_len) else 0

    self.computed += prompt_len - num_cached_tokens
    self.external_kv_transfer += num_external_computed_tokens
    self.local_cache_hit += max(
        0, num_cached_tokens + recomputed - num_external_computed_tokens
    )
    self.cached_tokens += num_cached_tokens
    self.recomputed_tokens += recomputed
    self.total += prompt_len
