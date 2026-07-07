from ucm.integration.vllm.patch.utils import patch_or_inject, when_imported


@when_imported("vllm.v1.core.sched.scheduler")
def patch_core_sched_scheduler(mod):
    """Wrap Scheduler._update_requests_with_invalid_blocks for KV load-failure
    recovery. Delegates to each version's own implementation, then applies
    UCM-specific post-processing."""

    # Capture the original method if it exists; use UCM fallback otherwise.
    original = getattr(mod.Scheduler, "_update_requests_with_invalid_blocks", None)

    if original is not None:

        def wrapped_update(self, requests, *args, **kwargs):
            # Delegate to the version-specific implementation
            result = original(self, requests, *args, **kwargs)

            # UCM post-processing: track requests with KV load failures
            if result:
                affected_req_ids = result[0]
                for request in requests:
                    if request.request_id in affected_req_ids:
                        request.num_output_placeholders = 0

            return result

        patch_or_inject(
            mod.Scheduler,
            "_update_requests_with_invalid_blocks",
            wrapped_update,
        )
