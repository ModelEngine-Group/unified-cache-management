"""KV load-failure recovery with group 0 block IDs."""

from __future__ import annotations

import inspect
from textwrap import dedent

from ucm.integration.vllm.patch.utils import patch_or_inject, when_imported


@when_imported("vllm.v1.core.sched.scheduler")
def patch_core_sched_scheduler(mod):
    """Wrap Scheduler._update_requests_with_invalid_blocks for KV load-failure
    recovery.

    For single-group models: delegate to the original method unchanged.

    For multi-group (HMA) models: the stock vLLM method crashes because it
    tuple-unpacks ``(req_block_ids,) = get_block_ids(req_id)`` which expects
    exactly one group.  We fix this transparently by temporarily shrinking
    ``get_block_ids`` to return only group 0 as a 1-tuple
    while the original method runs, then restore it.  After the call we also
    merge ``invalid_block_ids`` into ``blocks_to_evict`` so that all
    reported invalid blocks are included in eviction.

    Allow the outer multi-group check to pass only for policy=fail. Keep
    native request identification and finishing; multi-group recompute is
    still unsupported.
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

    original_handle = getattr(mod.Scheduler, "_handle_invalid_blocks", None)
    if original_handle is None:
        return
    source_lines, first_line = inspect.getsourcelines(original_handle)
    source = dedent("".join(source_lines))
    multi_group_check = "if len(self.kv_cache_config.kv_cache_groups) > 1:"
    if multi_group_check not in source:
        # Older vLLM versions do not have the outer rejection.
        return
    # Change only the guard and retain the installed vLLM method's body.
    source = source.replace(
        multi_group_check,
        "if len(self.kv_cache_config.kv_cache_groups) > 1 "
        "and self.recompute_kv_load_failures:",
        1,
    )
    namespace = {}
    exec(
        compile(
            "\n" * (first_line - 1) + source,
            original_handle.__code__.co_filename,
            "exec",
        ),
        original_handle.__globals__,
        namespace,
    )
    patch_or_inject(
        mod.Scheduler, "_handle_invalid_blocks", namespace["_handle_invalid_blocks"]
    )
