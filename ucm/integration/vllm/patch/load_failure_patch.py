from ucm.integration.vllm.patch.utils import patch_or_inject, when_imported


def _multi_group_failures(scheduler, requests, invalid_block_ids):
    """Identify failed requests without mapping recurrent state to token offsets."""
    null_block = scheduler.kv_cache_manager.block_pool.null_block.block_id
    invalid = set(invalid_block_ids) - {null_block}
    affected = set()
    blocks_to_evict = set()
    for request in requests:
        groups = scheduler.kv_cache_manager.get_block_ids(request.request_id)
        blocks = {block for group in groups for block in group} - {null_block}
        if blocks & invalid:
            affected.add(request.request_id)
            # Any group can depend on the failed state. Evict the entire
            # request's cached blocks, including this forward's new blocks.
            blocks_to_evict.update(blocks)
    return affected, blocks_to_evict


@when_imported("vllm.v1.core.sched.scheduler")
def patch_core_sched_scheduler(mod):
    """Handle multi-group load failures with the native request-level fail path.

    New engines reject multi-group invalid block reports at the outer handler;
    older engines assume group 0 in the inner updater. The fail-policy wrapper
    scans every group without interpreting state blocks as token positions.

    Single-group handling and the historical inner compatibility wrapper remain
    unchanged. This patch does not enable multi-group recompute, which is still
    rejected by UCMConnector's constructor.
    """

    original_schedule = getattr(mod.Scheduler, "schedule", None)
    if original_schedule is not None:

        def wrapped_schedule(self, *args, **kwargs):
            output = original_schedule(self, *args, **kwargs)
            if len(self.kv_cache_config.kv_cache_groups) > 1 and not getattr(
                self, "recompute_kv_load_failures", False
            ):
                metadata = getattr(output, "kv_connector_metadata", None)
                pool = self.kv_cache_manager.block_pool
                snapshots = []
                for req_id, dispatch in getattr(metadata, "request_meta", {}).items():
                    targets = set(dispatch.load_block_ids[1]) - {
                        pool.null_block.block_id
                    }
                    if not targets:
                        continue
                    groups = self.kv_cache_manager.get_block_ids(req_id)
                    ids = {bid for group in groups for bid in group} - {
                        pool.null_block.block_id
                    }
                    snapshots.append(
                        (targets, [(bid, pool.blocks[bid].block_hash) for bid in ids])
                    )
                output._ucm_load_failure_blocks = snapshots
            return output

        patch_or_inject(mod.Scheduler, "schedule", wrapped_schedule)

    original_output = getattr(mod.Scheduler, "update_from_output", None)
    if original_output is not None:

        def wrapped_output(
            self, scheduler_output, model_runner_output, *args, **kwargs
        ):
            output = getattr(model_runner_output, "kv_connector_output", None)
            invalid = getattr(output, "invalid_block_ids", None)
            if (
                invalid
                and len(self.kv_cache_config.kv_cache_groups) > 1
                and not getattr(self, "recompute_kv_load_failures", False)
            ):
                # EngineCore processes aborts before this call. The step's
                # snapshot survives request removal and includes both external
                # load targets and blocks allocated for this forward.
                pool = self.kv_cache_manager.block_pool
                null_id = pool.null_block.block_id
                failed = set(invalid) - {null_id}
                evict = set()
                for targets, blocks in getattr(
                    scheduler_output, "_ucm_load_failure_blocks", ()
                ):
                    if targets & failed:
                        for bid, block_hash in blocks:
                            # Cancellation frees references but preserves hashes.
                            # A different hash means this physical ID was reused.
                            if pool.blocks[bid].block_hash == block_hash:
                                evict.add(bid)
                if evict:
                    self.kv_cache_manager.evict_blocks(evict)
            return original_output(
                self, scheduler_output, model_runner_output, *args, **kwargs
            )

        patch_or_inject(mod.Scheduler, "update_from_output", wrapped_output)

    original_handle = getattr(mod.Scheduler, "_handle_invalid_blocks", None)
    if original_handle is not None:

        def wrapped_handle(self, invalid_block_ids, *args, **kwargs):
            num_groups = len(self.kv_cache_config.kv_cache_groups)
            if num_groups <= 1 or getattr(self, "recompute_kv_load_failures", False):
                return original_handle(self, invalid_block_ids, *args, **kwargs)

            # New engines reject flat invalid_block_ids before reaching
            # _update_requests_with_invalid_blocks. Handle the fail policy here
            # instead; no token rollback or Mamba checkpoint selection is needed.
            # Flat IDs may alias across groups: conservatively fail extra users
            # of such an ID rather than silently accepting a failed state load.
            waiting = (
                req
                for req in self.requests.values()
                if req.status == mod.RequestStatus.WAITING_FOR_REMOTE_KVS
            )
            async_failed, _ = _multi_group_failures(self, waiting, invalid_block_ids)
            sync_failed, blocks = _multi_group_failures(
                self, self.running, invalid_block_ids
            )
            if blocks:
                self.kv_cache_manager.evict_blocks(blocks)
            failed = async_failed | sync_failed
            if failed:
                mod.logger.error(
                    "Failing %d hybrid request(s) due to KV load failure "
                    "(failure_policy=fail). Request IDs: %s",
                    len(failed),
                    failed,
                )
            # The native update_from_output consumer discards this step's output
            # and terminates these requests using the engine's normal error path.
            return failed

        patch_or_inject(mod.Scheduler, "_handle_invalid_blocks", wrapped_handle)

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
