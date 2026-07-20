from ucm.integration.vllm.patch.utils import patch_or_inject, when_imported


@when_imported("vllm.v1.core.sched.scheduler")
def patch_core_sched_scheduler(mod):
    """Wrap Scheduler._update_requests_with_invalid_blocks for KV load-failure
    recovery.

    For single-group models: delegate to the original method unchanged.

    For multi-group (HMA) models: the stock vLLM method crashes because it
    tuple-unpacks ``(req_block_ids,) = get_block_ids(req_id)`` which expects
    exactly one group.  We fix this transparently by temporarily shrinking
    ``get_block_ids`` to return only group 0 (full-attention) as a 1-tuple
    while the original method runs, then restore it.  After the call we also
    merge ``invalid_block_ids`` into ``blocks_to_evict`` so that invalid
    blocks from *all* groups are evicted, not just group 0.

    Both paths then apply UCM-specific post-processing (clearing
    ``num_output_placeholders`` for affected requests).
    """

    original = getattr(mod.Scheduler, "_update_requests_with_invalid_blocks", None)

    if original is not None:

        def wrapped_update(self, requests, invalid_block_ids, *args, **kwargs):
            num_groups = getattr(self.kv_cache_manager, "num_kv_cache_groups", 1)

            if num_groups > 1:
                orig_get_block_ids = self.kv_cache_manager.get_block_ids

                def _single_group_get_block_ids(req_id):
                    groups = orig_get_block_ids(req_id)
                    if len(groups) > 1:
                        return (groups[0],)
                    return groups

                self.kv_cache_manager.get_block_ids = _single_group_get_block_ids
                try:
                    result = original(
                        self, requests, invalid_block_ids, *args, **kwargs
                    )
                finally:
                    self.kv_cache_manager.get_block_ids = orig_get_block_ids

                if result and len(result) >= 3:
                    blocks_to_evict = set(result[2])
                    blocks_to_evict.update(invalid_block_ids)
                    result = (result[0], result[1], blocks_to_evict)
            else:
                result = original(self, requests, invalid_block_ids, *args, **kwargs)

            return result

        patch_or_inject(
            mod.Scheduler,
            "_update_requests_with_invalid_blocks",
            wrapped_update,
        )
