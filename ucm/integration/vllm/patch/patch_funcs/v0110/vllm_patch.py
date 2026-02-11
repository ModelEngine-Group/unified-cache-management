#
# MIT License
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

"""Monkey patches for vLLM 0.11.0 (UCM)."""

from __future__ import annotations

from ucm.logger import init_logger

from .patch_txn import PatchTxn

logger = init_logger(__name__)


def _apply_vllm_patches() -> None:
    """Apply all patches for vLLM 0.11.0."""
    txn = PatchTxn()
    try:
        _patch_kv_connector_base_v1(txn)
        _patch_attention_layer(txn)
        _patch_kv_connector_output(txn)
        _patch_cached_request_data(txn)
        _patch_kv_output_aggregator(txn)
        _patch_block_pool(txn)
        _patch_single_type_kv_cache_manager(txn)
        _patch_scheduler(txn)
        _patch_gpu_model_runner(txn)
        _patch_gpu_worker(txn)
        _patch_kv_connector_model_runner_mixin(txn)
    except Exception as e:
        txn.rollback()
        PatchTxn.rollback_all()
        logger.warning(f"vLLM 0.11.0 patches failed to apply: {e}")
        raise
    else:
        txn.commit()
        logger.info(f"vLLM 0.11.0 patches applied successfully")


def _rollback_vllm_patches() -> None:
    """Rollback all recorded vLLM 0.11.0 monkey patches."""
    PatchTxn.rollback_all()


def _patch_kv_connector_base_v1(txn: PatchTxn | None = None) -> None:
    """Add get_block_ids_with_load_errors to KVConnectorBase_V1."""
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1 import base as base_mod

        if getattr(base_mod, "__ucm_patched__", False):
            return

        KVConnectorBase_V1 = base_mod.KVConnectorBase_V1

        def get_block_ids_with_load_errors(self) -> set[int]:
            """Get the set of block IDs that failed to load."""
            return set()

        # Mandatory patch: keep it out of rollback tracking.
        KVConnectorBase_V1.get_block_ids_with_load_errors = (
            get_block_ids_with_load_errors
        )

        # add has_connector_metadata to KVConnectorBase_V1
        def has_connector_metadata(self) -> bool:
            """Check whether the connector metadata is currently set.

            Returns:
                bools: True if connector metadata exists, False otherwise.
            """
            return self._connector_metadata is not None

        KVConnectorBase_V1.has_connector_metadata = has_connector_metadata
        base_mod.__ucm_patched__ = True
    except Exception as e:
        logger.warning(f"Could not patch KVConnectorBase_V1: {e}")
        raise


def _patch_attention_layer(txn: PatchTxn | None = None) -> None:
    """Patch attention layer"""
    try:
        # IMPORTANT: Rebinding names imported with
        # `from vllm.attention.layer import foo` only updates local variables in
        # this patch module, NOT the original `vllm.attention.layer` module.
        # We must patch the module attributes directly.
        import vllm.attention.layer as layer_mod

        if getattr(layer_mod, "__ucm_patched__", False):
            return

        def patched_wait_for_kv_layer_from_connector(layer_name: str) -> None:
            # Keep imports local to reduce import-time coupling across vLLM versions.
            from vllm.distributed.kv_transfer import (
                get_kv_transfer_group,
                has_kv_transfer_group,
                is_v1_kv_transfer_group,
            )
            from vllm.forward_context import get_forward_context

            if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
                return

            connector = get_kv_transfer_group()
            # Guard for older connectors to avoid AttributeError.
            if (
                hasattr(connector, "has_connector_metadata")
                and not connector.has_connector_metadata()
            ):
                return

            attn_metadata = get_forward_context().attn_metadata
            if attn_metadata is None:
                return
            assert isinstance(attn_metadata, dict)
            connector.wait_for_layer_load(layer_name)

        def patched_maybe_save_kv_layer_to_connector(
            layer_name: str,
            kv_cache_layer: "list[object]",
        ) -> None:
            from vllm.distributed.kv_transfer import (
                get_kv_transfer_group,
                has_kv_transfer_group,
                is_v1_kv_transfer_group,
            )
            from vllm.forward_context import get_forward_context

            if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
                return

            connector = get_kv_transfer_group()
            if (
                hasattr(connector, "has_connector_metadata")
                and not connector.has_connector_metadata()
            ):
                return

            attn_metadata = get_forward_context().attn_metadata
            if attn_metadata is None:
                return
            assert isinstance(attn_metadata, dict)
            connector.save_kv_layer(
                layer_name, kv_cache_layer, attn_metadata[layer_name]
            )

        # Mandatory patch: keep it out of rollback tracking.
        layer_mod.wait_for_kv_layer_from_connector = (
            patched_wait_for_kv_layer_from_connector
        )
        layer_mod.maybe_save_kv_layer_to_connector = (
            patched_maybe_save_kv_layer_to_connector
        )
        layer_mod.__ucm_patched__ = True
    except Exception as e:
        logger.warning(f"Could not patch attention layer: {e}")
        raise


def _patch_kv_connector_output(txn: PatchTxn | None = None) -> None:
    """Add invalid_block_ids to KVConnectorOutput and update is_empty()."""
    txn = txn or PatchTxn()
    try:
        import functools

        from vllm.v1 import outputs as outputs_mod

        KVConnectorOutput = outputs_mod.KVConnectorOutput
        if getattr(KVConnectorOutput, "__ucm_patched__", False):
            return

        # 1. Wrap original __init__ to add invalid_block_ids parameter
        original_init = KVConnectorOutput.__init__

        @functools.wraps(original_init)
        def patched_init(self, *args, invalid_block_ids=None, **kwargs):
            original_init(self, *args, **kwargs)
            self.invalid_block_ids = (
                invalid_block_ids if invalid_block_ids is not None else set()
            )

        txn.set_attr(KVConnectorOutput, "__init__", patched_init)

        # 2. Replace is_empty method to include invalid_block_ids check
        def patched_is_empty(self):
            return (
                not self.finished_sending
                and not self.finished_recving
                and not self.kv_connector_stats
                and not getattr(self, "invalid_block_ids", None)
            )

        txn.set_attr(KVConnectorOutput, "is_empty", patched_is_empty)

        # 3. Update __annotations__ for type hints
        if hasattr(KVConnectorOutput, "__annotations__"):
            txn.set_dict_item(
                KVConnectorOutput.__annotations__, "invalid_block_ids", set[int]
            )
        txn.set_attr(KVConnectorOutput, "__ucm_patched__", True)  # type: ignore[attr-defined]
    except Exception as e:
        logger.warning(f"Could not patch KVConnectorOutput: {e}")
        raise


def _patch_cached_request_data(txn: PatchTxn | None = None) -> None:
    """Add num_output_tokens to CachedRequestData (keep all original fields)."""
    txn = txn or PatchTxn()
    try:
        from dataclasses import dataclass, field
        from typing import Optional

        from vllm.v1.core.sched import output as output_mod

        CachedRequestData = output_mod.CachedRequestData
        if getattr(CachedRequestData, "__ucm_patched__", False):
            return

        @dataclass
        class CachedRequestDataPatched:
            req_ids: list[str]
            resumed_from_preemption: list[bool]
            new_token_ids: list[list[int]]
            new_block_ids: list[Optional[tuple[list[int], ...]]]
            num_computed_tokens: list[int]
            # Default so vLLM's _make_cached_request_data (5 args) can still construct; we overwrite in patched_schedule.
            num_output_tokens: list[int] = field(default_factory=list)

            @property
            def num_reqs(self) -> int:
                return len(self.req_ids)

            @classmethod
            def make_empty(cls) -> "CachedRequestDataPatched":
                return cls(
                    req_ids=[],
                    resumed_from_preemption=[],
                    new_token_ids=[],
                    new_block_ids=[],
                    num_computed_tokens=[],
                    num_output_tokens=[],
                )

        CachedRequestDataPatched.__qualname__ = "CachedRequestData"
        CachedRequestDataPatched.__module__ = output_mod.__name__
        txn.set_attr(output_mod, "CachedRequestData", CachedRequestDataPatched)
        txn.set_attr(CachedRequestDataPatched, "__ucm_patched__", True)  # type: ignore[attr-defined]
    except Exception as e:
        logger.warning(f"Could not patch CachedRequestData: {e}")
        raise


def _patch_kv_output_aggregator(txn: PatchTxn | None = None) -> None:
    """Patch KVOutputAggregator.aggregate to aggregate invalid_block_ids."""
    txn = txn or PatchTxn()
    try:
        from vllm.distributed.kv_transfer.kv_connector import utils as utils_mod

        KVOutputAggregator = utils_mod.KVOutputAggregator
        if getattr(KVOutputAggregator, "__ucm_patched__", False):
            return

        original_aggregate = KVOutputAggregator.aggregate

        def patched_aggregate(self, outputs, output_rank: int = 0):
            result = original_aggregate(self, outputs, output_rank)
            invalid_block_ids: set[int] = set()
            for model_runner_output in outputs:
                out = model_runner_output.kv_connector_output
                if out is not None:
                    invalid_block_ids |= getattr(out, "invalid_block_ids", set())
            # invalid_block_ids belongs on KVConnectorOutput (result.kv_connector_output), not on ModelRunnerOutput
            if result.kv_connector_output is not None:
                setattr(
                    result.kv_connector_output, "invalid_block_ids", invalid_block_ids
                )
            # if invalid_block_ids:
            #    logger.warning(
            #        f"[kv-load] aggregate invalid_block_ids={len(invalid_block_ids)} "
            #        f"sample={list(invalid_block_ids)[:10]}"
            #    )
            return result

        txn.set_attr(KVOutputAggregator, "aggregate", patched_aggregate)
        txn.set_attr(KVOutputAggregator, "__ucm_patched__", True)  # type: ignore[attr-defined]
    except Exception as e:
        logger.warning(f"Could not patch KVOutputAggregator: {e}")
        raise


def _patch_block_pool(txn: PatchTxn | None = None) -> None:
    """Change BlockPool.cache_full_blocks condition from == to >=."""
    txn = txn or PatchTxn()
    try:
        from vllm.v1.core import block_pool as block_pool_mod

        BlockPool = block_pool_mod.BlockPool
        if getattr(BlockPool.cache_full_blocks, "__ucm_patched__", False):
            return

        original_cache_full_blocks = BlockPool.cache_full_blocks

        def patched_cache_full_blocks(
            self,
            request,
            blocks,
            num_cached_blocks,
            num_full_blocks,
            block_size,
            kv_cache_group_id,
        ):
            if num_cached_blocks >= num_full_blocks:
                return
            return original_cache_full_blocks(
                self,
                request,
                blocks,
                num_cached_blocks,
                num_full_blocks,
                block_size,
                kv_cache_group_id,
            )

        txn.set_attr(patched_cache_full_blocks, "__ucm_patched__", True)
        txn.set_attr(BlockPool, "cache_full_blocks", patched_cache_full_blocks)
    except Exception as e:
        logger.warning(f"Could not patch BlockPool: {e}")
        raise


def _patch_single_type_kv_cache_manager(txn: PatchTxn | None = None) -> None:
    """Add early return when num_cached_blocks >= num_full_blocks in cache_blocks."""
    txn = txn or PatchTxn()
    try:
        from vllm.v1.core import single_type_kv_cache_manager as stkvm_mod

        SingleTypeKVCacheManager = stkvm_mod.SingleTypeKVCacheManager
        if getattr(SingleTypeKVCacheManager.cache_blocks, "__ucm_patched__", False):
            return

        original_cache_blocks = SingleTypeKVCacheManager.cache_blocks

        def patched_cache_blocks(self, request, num_tokens):
            num_cached_blocks = self.num_cached_block[request.request_id]
            num_full_blocks = num_tokens // self.block_size
            if num_cached_blocks >= num_full_blocks:
                return
            return original_cache_blocks(self, request, num_tokens)

        txn.set_attr(patched_cache_blocks, "__ucm_patched__", True)
        txn.set_attr(SingleTypeKVCacheManager, "cache_blocks", patched_cache_blocks)
    except Exception as e:
        logger.warning(f"Could not patch SingleTypeKVCacheManager: {e}")
        raise


def _patch_scheduler(txn: PatchTxn | None = None) -> None:
    """Patch Scheduler for load-failure recovery."""
    txn = txn or PatchTxn()
    try:
        from collections.abc import Iterable

        from vllm.v1.core.sched.output import CachedRequestData
        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm.v1.request import Request, RequestStatus

        if getattr(Scheduler, "__ucm_patched__", False):
            return

        orig_init = Scheduler.__init__

        def patched_init(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            if not hasattr(self, "failed_recving_kv_req_ids"):
                self.failed_recving_kv_req_ids: set[str] = set()

        txn.set_attr(Scheduler, "__init__", patched_init)

        def _ensure_failed_recving_kv_req_ids(self: Scheduler) -> set[str]:
            """Ensure `failed_recving_kv_req_ids` exists even for pre-existing instances.

            NOTE: Monkeypatching `Scheduler.__init__` only affects instances created
            after the patch is applied. Engine cores may already have created the
            Scheduler, so we must lazily initialize the attribute on first use.
            """
            s = getattr(self, "failed_recving_kv_req_ids", None)
            if s is None:
                s = set()
                setattr(self, "failed_recving_kv_req_ids", s)
            return s

        def _update_requests_with_invalid_blocks(
            self: Scheduler,
            requests: Iterable[Request],
            invalid_block_ids: set[int],
        ) -> tuple[set[str], int]:
            """
            Identify and update requests affected by invalid KV cache blocks.

            This method scans the given requests, detects those with invalid blocks
            and adjusts their `num_computed_tokens` to the longest valid prefix.
            For observability, it also accumulates the total number of tokens that
            will need to be recomputed across all affected requests.

            Args:
                requests: The set of requests to scan for invalid blocks.
                invalid_block_ids: IDs of invalid blocks.

            Returns:
                tuple:
                    - affected_req_ids (set[str]): IDs of requests impacted by
                    invalid blocks.
                    - total_affected_tokens (int): Total number of tokens that must
                    be recomputed across all affected requests (for observability).
            """
            affected_req_ids: set[str] = set()
            total_affected_tokens = 0
            failed_recving_kv_req_ids = _ensure_failed_recving_kv_req_ids(self)
            # If a block is invalid and shared by multiple requests in the batch,
            # these requests must be rescheduled, but only the first will recompute
            # it. This set tracks blocks already marked for recomputation.
            marked_invalid_block_ids: set[int] = set()
            for request in requests:
                is_affected = False
                marked_invalid_block = False
                req_id = request.request_id
                # TODO (davidb): add support for hybrid memory allocator
                (req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)
                # We iterate only over blocks that may contain externally computed
                # tokens
                if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                    # Async loading. If num_computed_tokens is set it implies we
                    # already processed some block failures for it in a prior step
                    req_num_computed_tokens = (
                        request.num_computed_tokens
                        if req_id in failed_recving_kv_req_ids
                        else len(req_block_ids) * self.block_size
                    )
                else:
                    # Sync loading. num_computed_tokens includes new tokens
                    req_num_computed_tokens = request.num_cached_tokens

                prev_num_computed_tokens = request.num_computed_tokens
                req_num_computed_blocks = (
                    req_num_computed_tokens + self.block_size - 1
                ) // self.block_size
                for idx, block_id in zip(range(req_num_computed_blocks), req_block_ids):
                    if block_id not in invalid_block_ids:
                        continue

                    is_affected = True

                    if block_id in marked_invalid_block_ids:
                        # This invalid block is shared with a previous request
                        # and was already marked for recomputation.
                        # This means this request can still consider this block
                        # as computed when rescheduled.
                        # Currently this only applies to sync loading; Async
                        # loading does not yet support block sharing
                        continue

                    marked_invalid_block_ids.add(block_id)

                    if marked_invalid_block:
                        # This request has already marked an invalid block for
                        # recomputation and updated its num_computed_tokens.
                        continue

                    marked_invalid_block = True
                    # Truncate the computed tokens at the first failed block
                    request.num_computed_tokens = idx * self.block_size
                    total_affected_tokens += (
                        req_num_computed_tokens - request.num_computed_tokens
                    )

                if is_affected:
                    if not marked_invalid_block:
                        # All invalid blocks of this request are shared with
                        # previous requests and will be recomputed by them.
                        # Revert to considering only cached tokens as computed.
                        # Currently this only applies to sync loading; Async
                        # loading does not yet support block sharing
                        total_affected_tokens += (
                            request.num_computed_tokens - request.num_cached_tokens
                        )
                        request.num_computed_tokens = request.num_cached_tokens

                    affected_req_ids.add(request.request_id)

            return affected_req_ids, total_affected_tokens

        def _handle_invalid_blocks(
            self: Scheduler, invalid_block_ids: set[int]
        ) -> set[str]:
            failed_recving_kv_req_ids = _ensure_failed_recving_kv_req_ids(self)
            total_requests_to_reschedule = 0
            total_tokens_to_reschedule = 0

            # logger.warning(
            #    f"[kv-load] scheduler handle invalid_block_ids={len(invalid_block_ids)} "
            #    f"sample={list(invalid_block_ids)[:10]}"
            # )

            # --- Handle async KV loads (WAITING_FOR_REMOTE_KVS) ---
            async_load_reqs = (
                req
                for req in self.waiting
                if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
            )
            async_affected_req_ids, num_tokens_to_reschedule = (
                self._update_requests_with_invalid_blocks(
                    async_load_reqs, invalid_block_ids
                )
            )

            total_requests_to_reschedule += len(async_affected_req_ids)
            total_tokens_to_reschedule += num_tokens_to_reschedule

            # Mark requests with async KV load failures; they will be rescheduled
            # once loading completes.
            failed_recving_kv_req_ids |= async_affected_req_ids
            if async_affected_req_ids:
                logger.warning(
                    f"[kv-load] async affected reqs (WAITING_FOR_REMOTE_KVS): "
                    f"count={len(async_affected_req_ids)} sample={list(async_affected_req_ids)[:10]}"
                )

            # --- Handle sync KV loads (running requests) ---
            sync_affected_req_ids, num_tokens_to_reschedule = (
                self._update_requests_with_invalid_blocks(
                    self.running, invalid_block_ids
                )
            )
            if sync_affected_req_ids and self.connector is not None:
                # Keep UCM request metadata consistent with rescheduled requests.
                # self.connector may be either the real connector implementation
                # or a wrapper that stores the implementation in `.connector`.
                connector_impl = getattr(self.connector, "connector", self.connector)
                requests_meta = getattr(connector_impl, "requests_meta", None)
                if isinstance(requests_meta, dict):
                    for req_id in sync_affected_req_ids:
                        requests_meta.pop(req_id, None)
                        logger.warning(f"Removed request metadata for request {req_id}")

            total_requests_to_reschedule += len(sync_affected_req_ids)
            total_tokens_to_reschedule += num_tokens_to_reschedule

            if total_requests_to_reschedule:
                logger.warning(
                    f"Recovered from KV load failure: {total_requests_to_reschedule} request(s) "
                    f"rescheduled ({total_tokens_to_reschedule} tokens affected)."
                )
                # if sync_affected_req_ids:
                #    logger.warning(
                #        f"[kv-load] sync affected running reqs (skip output update this step): "
                #        f"count={len(sync_affected_req_ids)} sample={list(sync_affected_req_ids)[:10]}"
                #    )

            # Return the IDs of affected running requests to skip in
            # update_from_output.
            return sync_affected_req_ids

        txn.set_attr(
            Scheduler,
            "_update_requests_with_invalid_blocks",
            _update_requests_with_invalid_blocks,
        )
        txn.set_attr(Scheduler, "_handle_invalid_blocks", _handle_invalid_blocks)

        def patched_make_cached_request_data(
            self: Scheduler,
            running_reqs: list[Request],
            resumed_reqs: list[Request],
            num_scheduled_tokens: dict[str, int],
            spec_decode_tokens: dict[str, list[int]],
            req_to_new_blocks: dict[str, KVCacheBlocks],
        ) -> CachedRequestData:
            req_ids: list[str] = []
            new_token_ids: list[list[int]] = []
            new_block_ids: list[Optional[tuple[list[int], ...]]] = []
            num_computed_tokens: list[int] = []
            num_output_tokens: list[int] = []

            import itertools

            use_connector = self.connector is not None
            for req in itertools.chain(running_reqs, resumed_reqs):
                req_id = req.request_id
                req_ids.append(req_id)
                num_tokens = num_scheduled_tokens[req_id] - len(
                    spec_decode_tokens.get(req_id, ())
                )
                if self.use_pp:
                    # When using PP, the scheduler sends the sampled tokens back,
                    # because there's no direct communication between the first-
                    # stage worker and the last-stage worker. Otherwise, we don't
                    # need to send the sampled tokens back because the model runner
                    # will cache them.
                    token_ids = req.all_token_ids[
                        req.num_computed_tokens : req.num_computed_tokens + num_tokens
                    ]
                    new_token_ids.append(token_ids)
                elif use_connector:
                    # When using a KVConnector, we add a placeholder to avoid index
                    # out of bounds errors. TODO: Remove this once the KVConnector
                    # is updated to handle token IDs properly.
                    new_token_ids.append([])
                new_block_ids.append(
                    req_to_new_blocks[req_id].get_block_ids(allow_none=True)
                )
                num_computed_tokens.append(req.num_computed_tokens)
                num_output_tokens.append(len(req.output_token_ids))
            # Because resumed_reqs is usually empty, it is more efficient to do
            # in-place appending so that we don't need to allocate a new list.
            resumed_from_preemption = [False] * len(running_reqs)
            resumed_from_preemption += [True] * len(resumed_reqs)

            return CachedRequestData(
                req_ids=req_ids,
                resumed_from_preemption=resumed_from_preemption,
                new_token_ids=new_token_ids,
                new_block_ids=new_block_ids,
                num_computed_tokens=num_computed_tokens,
                num_output_tokens=num_output_tokens,
            )

        txn.set_attr(
            Scheduler, "_make_cached_request_data", patched_make_cached_request_data
        )
        from collections import defaultdict
        from typing import Optional

        from vllm.v1.core.sched.output import SchedulerOutput
        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm.v1.core.sched.utils import check_stop, remove_all
        from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
        from vllm.v1.outputs import ModelRunnerOutput
        from vllm.v1.request import Request

        def patched_update_from_output(
            self,
            scheduler_output: SchedulerOutput,
            model_runner_output: ModelRunnerOutput,
        ) -> dict[int, EngineCoreOutputs]:

            sampled_token_ids = model_runner_output.sampled_token_ids
            logprobs = model_runner_output.logprobs
            prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens
            pooler_outputs = model_runner_output.pooler_output
            num_nans_in_logits = model_runner_output.num_nans_in_logits
            kv_connector_output = model_runner_output.kv_connector_output

            outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
            from vllm.v1.spec_decode.metrics import SpecDecodingStats

            spec_decoding_stats: Optional[SpecDecodingStats] = None
            kv_connector_stats = (
                kv_connector_output.kv_connector_stats if kv_connector_output else None
            )

            failed_kv_load_req_ids = None
            if kv_connector_output and kv_connector_output.invalid_block_ids:
                # These blocks contain externally computed tokens that failed to
                # load. Identify affected requests and adjust their computed token
                # count to trigger recomputation of the invalid blocks.
                # logger.warning(
                #    f"[kv-load] scheduler received invalid_block_ids={len(kv_connector_output.invalid_block_ids)} "
                #    f"sample={list(kv_connector_output.invalid_block_ids)[:10]}"
                # )
                failed_kv_load_req_ids = self._handle_invalid_blocks(
                    kv_connector_output.invalid_block_ids
                )
            # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
            # the below loop can be a performance bottleneck. We should do our best
            # to avoid expensive operations inside the loop.
            stopped_running_reqs: set[Request] = set()
            stopped_preempted_reqs: set[Request] = set()
            for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
                assert num_tokens_scheduled > 0
                if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:
                    # Skip requests that were recovered from KV load failure
                    continue
                request = self.requests.get(req_id)
                if request is None:
                    # The request is already finished. This can happen if the
                    # request is aborted while the model is executing it (e.g.,
                    # in pipeline parallelism).
                    continue

                req_index = model_runner_output.req_id_to_index[req_id]
                generated_token_ids = (
                    sampled_token_ids[req_index] if sampled_token_ids else []
                )

                scheduled_spec_token_ids = (
                    scheduler_output.scheduled_spec_decode_tokens.get(req_id)
                )
                if scheduled_spec_token_ids:
                    num_draft_tokens = len(scheduled_spec_token_ids)
                    num_accepted = len(generated_token_ids) - 1
                    num_rejected = num_draft_tokens - num_accepted
                    # num_computed_tokens represents the number of tokens
                    # processed in the current step, considering scheduled
                    # tokens and rejections. If some tokens are rejected,
                    # num_computed_tokens is decreased by the number of rejected
                    # tokens.
                    if request.num_computed_tokens > 0:
                        request.num_computed_tokens -= num_rejected
                    if request.num_output_tokens > 0:
                        request.num_output_tokens -= num_rejected
                    spec_decoding_stats = self.make_spec_decoding_stats(
                        spec_decoding_stats,
                        num_draft_tokens=num_draft_tokens,
                        num_accepted_tokens=num_accepted,
                    )

                stopped = False
                new_logprobs = None
                new_token_ids = generated_token_ids
                kv_transfer_params = None
                status_before_stop = request.status

                # Check for stop and update request status.
                if new_token_ids:
                    new_token_ids, stopped = self._update_request_with_output(
                        request, new_token_ids
                    )

                # Stop checking for pooler models.
                pooler_output = None
                if pooler_outputs:
                    pooler_output = pooler_outputs[req_index]
                    stopped = check_stop(request, self.max_model_len, pooler_output)

                if stopped:
                    kv_transfer_params = self._free_request(request)
                    if status_before_stop == RequestStatus.RUNNING:
                        stopped_running_reqs.add(request)
                    else:
                        stopped_preempted_reqs.add(request)

                # Extract sample logprobs if needed.
                if (
                    request.sampling_params is not None
                    and request.sampling_params.logprobs is not None
                    and logprobs
                ):
                    # NOTE: once we support N tokens per step (spec decode),
                    # the outer lists can be of length > 1.
                    new_logprobs = logprobs.slice(req_index, req_index + 1)

                if new_token_ids and self.structured_output_manager.should_advance(
                    request
                ):
                    # NOTE: structured_output_request
                    # should not be None if use_structured_output, we have
                    # checked above, so safe to ignore type warning
                    request.structured_output_request.grammar.accept_tokens(  # type: ignore[union-attr]
                        req_id, new_token_ids
                    )

                if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                    request.num_nans_in_logits = num_nans_in_logits[req_id]

                # Get prompt logprobs for this request.
                prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
                if new_token_ids or pooler_output is not None or kv_transfer_params:

                    # Add EngineCoreOutput for this Request.
                    outputs[request.client_index].append(
                        EngineCoreOutput(
                            request_id=req_id,
                            new_token_ids=new_token_ids,
                            finish_reason=request.get_finished_reason(),
                            new_logprobs=new_logprobs,
                            new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                            pooling_output=pooler_output,
                            stop_reason=request.stop_reason,
                            events=request.take_events(),
                            kv_transfer_params=kv_transfer_params,
                            trace_headers=request.trace_headers,
                            num_cached_tokens=request.num_cached_tokens,
                        )
                    )
                else:
                    # Invariant: EngineCore returns no partial prefill outputs.
                    assert not prompt_logprobs_tensors

            # Remove the stopped requests from the running and waiting queues.
            if stopped_running_reqs:
                self.running = remove_all(self.running, stopped_running_reqs)
            if stopped_preempted_reqs:
                # This is a rare case and unlikely to impact performance.
                self.waiting.remove_requests(stopped_preempted_reqs)

            # KV Connector: update state for finished KV Transfers.
            if model_runner_output.kv_connector_output:
                self._update_from_kv_xfer_finished(
                    model_runner_output.kv_connector_output
                )

            # Create EngineCoreOutputs for all clients that have requests with
            # outputs in this step.
            engine_core_outputs = {
                client_index: EngineCoreOutputs(outputs=outs)
                for client_index, outs in outputs.items()
            }

            finished_req_ids = self.finished_req_ids_dict
            if finished_req_ids:
                # Include ids of requests that finished since last outputs
                # were sent.
                for client_index, finished_set in finished_req_ids.items():
                    # Set finished request set in EngineCoreOutputs for this client.
                    if (eco := engine_core_outputs.get(client_index)) is not None:
                        eco.finished_requests = finished_set
                    else:
                        engine_core_outputs[client_index] = EngineCoreOutputs(
                            finished_requests=finished_set
                        )
                finished_req_ids.clear()

            if (
                stats := self.make_stats(spec_decoding_stats, kv_connector_stats)
            ) is not None:
                # Return stats to only one of the front-ends.
                if (eco := next(iter(engine_core_outputs.values()), None)) is None:
                    # We must return the stats even if there are no request
                    # outputs this step.
                    engine_core_outputs[0] = eco = EngineCoreOutputs()
                eco.scheduler_stats = stats

            return engine_core_outputs

        txn.set_attr(Scheduler, "update_from_output", patched_update_from_output)

        def patched_update_waiting_for_remote_kv(self, request: Request) -> bool:
            """
            KV Connector: check if the request_id is finished_recving.

            The finished_recving_kv_req_ids list is populated
            on the previous steps()'s update_from_output based
            on the worker side connector.

            When the kv transfer is ready, we cache the blocks
            and the request state will be moved back to WAITING from
            WAITING_FOR_REMOTE_KV.
            """
            assert self.connector is not None
            failed_recving_kv_req_ids = _ensure_failed_recving_kv_req_ids(self)
            if request.request_id not in self.finished_recving_kv_req_ids:
                return False

            if request.request_id in failed_recving_kv_req_ids:
                # Request had KV load failures; num_computed_tokens was already
                # updated in _update_requests_with_invalid_blocks
                if request.num_computed_tokens:
                    # Cache any valid computed tokens.
                    # logger.warning(
                    #    f"[kv-load] async recv finished (had failures): cache valid prefix. "
                    #    f"req_id={request.request_id} num_computed_tokens={request.num_computed_tokens}"
                    # )
                    self.kv_cache_manager.cache_blocks(
                        request, request.num_computed_tokens
                    )
                else:
                    # No valid computed tokens, release allocated blocks.
                    # There may be a local cache hit on retry.
                    # logger.warning(
                    #    f"[kv-load] async recv finished (had failures): free blocks (no valid prefix). "
                    #    f"req_id={request.request_id}"
                    # )
                    self.kv_cache_manager.free(request)

                failed_recving_kv_req_ids.discard(request.request_id)
            else:
                # Now that the blocks are ready, actually cache them.
                (block_ids,) = self.kv_cache_manager.get_block_ids(request.request_id)
                num_computed_tokens = len(block_ids) * self.block_size
                # Handle the case where num request tokens less than one block.
                num_computed_tokens = min(num_computed_tokens, request.num_tokens)
                if num_computed_tokens == request.num_tokens:
                    num_computed_tokens -= 1
                # This will cache the blocks iff caching is enabled.
                self.kv_cache_manager.cache_blocks(request, num_computed_tokens)

                # Update the request state for scheduling.
                request.num_computed_tokens = num_computed_tokens

            # Return that we are ready.
            self.finished_recving_kv_req_ids.remove(request.request_id)
            return True

        txn.set_attr(
            Scheduler,
            "_update_waiting_for_remote_kv",
            patched_update_waiting_for_remote_kv,
        )
        txn.set_attr(Scheduler, "__ucm_patched__", True)  # type: ignore[attr-defined]

    except Exception as e:
        logger.warning(f"Could not patch Scheduler: {e}")
        raise


def _patch_gpu_model_runner(txn: PatchTxn | None = None) -> None:
    """Patch _update_states method in GPUModelRunner."""
    txn = txn or PatchTxn()
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner

        if getattr(GPUModelRunner, "__ucm_patched__", False):
            return

        def patched_update_states(self, scheduler_output: "SchedulerOutput") -> None:
            """Update the cached states and the persistent batch with the scheduler
            output.

            The updated states are used by the `_prepare_inputs` function to create
            the input GPU tensors for the model.

            The SamplingMetadata is updated and copied to the GPU if there is a
            new/resumed/paused/finished request in the batch.
            """
            from typing import cast

            import torch
            from vllm.distributed.parallel_state import get_pp_group
            from vllm.sampling_params import SamplingType
            from vllm.v1.worker.gpu_input_batch import CachedRequestState
            from vllm.v1.worker.gpu_model_runner import VllmModelForPooling

            # Remove finished requests from the cached states.
            for req_id in scheduler_output.finished_req_ids:
                self.requests.pop(req_id, None)
            # Remove the finished requests from the persistent batch.
            # NOTE(woosuk): There could be an edge case where finished_req_ids and
            # scheduled_req_ids overlap. This happens when a request is aborted and
            # then resubmitted with the same ID. In this case, we treat them as two
            # distinct requests - clearing the cached states for the first request
            # and handling the second as a new request.
            for req_id in scheduler_output.finished_req_ids:
                self.input_batch.remove_request(req_id)

            # Free the cached encoder outputs.
            for mm_hash in scheduler_output.free_encoder_mm_hashes:
                self.encoder_cache.pop(mm_hash, None)

            # Remove the unscheduled requests from the persistent batch.
            # NOTE(woosuk): The unscheduled requests are either preempted requests
            # or running requests that are not scheduled in this step. We remove
            # them from the persistent batch but keep their cached states since
            # they will be scheduled again sometime in the future.
            scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
            cached_req_ids = self.input_batch.req_id_to_index.keys()
            unscheduled_req_ids = cached_req_ids - scheduled_req_ids
            # NOTE(woosuk): The persistent batch optimization assumes that
            # consecutive batches contain mostly the same requests. If batches
            # have low request overlap (e.g., alternating between two distinct
            # sets of requests), this optimization becomes very inefficient.
            for req_id in unscheduled_req_ids:
                self.input_batch.remove_request(req_id)

            reqs_to_add: list[CachedRequestState] = []
            # Add new requests to the cached states.
            for new_req_data in scheduler_output.scheduled_new_reqs:
                req_id = new_req_data.req_id
                sampling_params = new_req_data.sampling_params
                pooling_params = new_req_data.pooling_params

                if (
                    sampling_params
                    and sampling_params.sampling_type == SamplingType.RANDOM_SEED
                ):
                    generator = torch.Generator(device=self.device)
                    generator.manual_seed(sampling_params.seed)
                else:
                    generator = None

                if self.is_pooling_model:
                    assert pooling_params is not None
                    task = pooling_params.task
                    assert task is not None, "You did not set `task` in the API"

                    model = cast(VllmModelForPooling, self.get_model())
                    to_update = model.pooler.get_pooling_updates(task)
                    to_update.apply(pooling_params)

                req_state = CachedRequestState(
                    req_id=req_id,
                    prompt_token_ids=new_req_data.prompt_token_ids,
                    prompt_embeds=new_req_data.prompt_embeds,
                    mm_features=new_req_data.mm_features,
                    sampling_params=sampling_params,
                    pooling_params=pooling_params,
                    generator=generator,
                    block_ids=new_req_data.block_ids,
                    num_computed_tokens=new_req_data.num_computed_tokens,
                    output_token_ids=[],
                    lora_request=new_req_data.lora_request,
                )
                self.requests[req_id] = req_state

                # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
                if self.uses_mrope:
                    self._init_mrope_positions(req_state)

                reqs_to_add.append(req_state)

            # Update the states of the running/resumed requests.
            is_last_rank = get_pp_group().is_last_rank
            req_data = scheduler_output.scheduled_cached_reqs
            for i, req_id in enumerate(req_data.req_ids):
                req_state = self.requests[req_id]
                num_computed_tokens = req_data.num_computed_tokens[i]
                new_block_ids = req_data.new_block_ids[i]
                resumed_from_preemption = req_data.resumed_from_preemption[i]
                num_output_tokens = req_data.num_output_tokens[i]
                req_index = self.input_batch.req_id_to_index.get(req_id)

                # Update the cached states.
                req_state.num_computed_tokens = num_computed_tokens

                if not is_last_rank:
                    # When using PP, the scheduler sends the sampled tokens back,
                    # because there's no direct communication between the first-
                    # stage worker and the last-stage worker.
                    new_token_ids = req_data.new_token_ids[i]
                    # Add the sampled token(s) from the previous step (if any).
                    # This doesn't include "unverified" tokens like spec tokens.
                    num_new_tokens = (
                        num_computed_tokens + len(new_token_ids) - req_state.num_tokens
                    )
                    if num_new_tokens == 1:
                        # Avoid slicing list in most common case.
                        req_state.output_token_ids.append(new_token_ids[-1])
                    elif num_new_tokens > 0:
                        req_state.output_token_ids.extend(
                            new_token_ids[-num_new_tokens:]
                        )
                elif num_output_tokens < len(req_state.output_token_ids):
                    # Some output tokens were discarded due to a sync-KV-load
                    # failure. Align the cached state.
                    # logger.warning(
                    #    f"[kv-load] align output tokens after sync load failure: "
                    #    f"req_id={req_id} output_token_ids {len(req_state.output_token_ids)}->{num_output_tokens}"
                    # )
                    del req_state.output_token_ids[num_output_tokens:]
                    if req_index is not None:
                        end_idx = (
                            self.input_batch.num_prompt_tokens[req_index]
                            + num_output_tokens
                        )
                        self.input_batch.num_tokens[req_index] = end_idx
                        self.input_batch.num_tokens_no_spec[req_index] = end_idx

                # Update the block IDs.
                if not resumed_from_preemption:
                    if new_block_ids is not None:
                        # Append the new blocks to the existing block IDs.
                        for block_ids, new_ids in zip(
                            req_state.block_ids, new_block_ids
                        ):
                            block_ids.extend(new_ids)
                else:
                    assert new_block_ids is not None
                    # The request is resumed from preemption.
                    # Replace the existing block IDs with the new ones.
                    req_state.block_ids = new_block_ids

                if req_index is None:
                    # The request is not in the persistent batch.
                    # The request was either preempted and resumed later, or was not
                    # scheduled in the previous step and needs to be added again.
                    reqs_to_add.append(req_state)
                    continue

                # Update the persistent batch.
                self.input_batch.num_computed_tokens_cpu[req_index] = (
                    num_computed_tokens
                )
                if new_block_ids is not None:
                    self.input_batch.block_table.append_row(new_block_ids, req_index)

                # For the last rank, we don't need to update the token_ids_cpu
                # because the sampled tokens are already cached.
                if not is_last_rank:
                    # Add new_token_ids to token_ids_cpu.
                    start_token_index = num_computed_tokens
                    end_token_index = num_computed_tokens + len(new_token_ids)
                    self.input_batch.token_ids_cpu[
                        req_index, start_token_index:end_token_index
                    ] = new_token_ids
                    self.input_batch.num_tokens_no_spec[req_index] = end_token_index
                    self.input_batch.num_tokens[req_index] = end_token_index

                # Add spec_token_ids to token_ids_cpu.
                spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(
                    req_id, ()
                )
                if spec_token_ids:
                    num_spec_tokens = len(spec_token_ids)
                    start_index = self.input_batch.num_tokens_no_spec[req_index]
                    end_token_index = start_index + num_spec_tokens
                    self.input_batch.token_ids_cpu[
                        req_index, start_index:end_token_index
                    ] = spec_token_ids
                    # NOTE(woosuk): `num_tokens` here may include spec tokens.
                    self.input_batch.num_tokens[req_index] += num_spec_tokens

            # Add the new or resumed requests to the persistent batch.
            # The smaller empty indices are filled first.
            for request in reqs_to_add:
                self.input_batch.add_request(request)

            # Condense the batched states if there are gaps left by removed requests
            self.input_batch.condense()
            # Allow attention backend to reorder the batch, potentially
            self._may_reorder_batch(scheduler_output)
            # Refresh batch metadata with any pending updates.
            self.input_batch.refresh_metadata()

        txn.set_attr(GPUModelRunner, "_update_states", patched_update_states)
        txn.set_attr(GPUModelRunner, "__ucm_patched__", True)  # type: ignore[attr-defined]
    except Exception as e:
        logger.warning(f"Could not patch GPUModelRunner: {e}")
        raise


def _patch_gpu_worker(txn: PatchTxn | None = None) -> None:
    """Patch Worker to use kv_connector_output.is_empty()."""
    txn = txn or PatchTxn()
    try:
        import copy
        from typing import Optional, Union

        import torch
        from vllm.v1.worker import gpu_worker as gpu_worker_mod

        Worker = gpu_worker_mod.Worker
        if getattr(Worker, "__ucm_patched__", False):
            return

        @torch.inference_mode()
        def patched_execute_model(
            self,
            scheduler_output: "SchedulerOutput",
        ) -> Optional[Union["ModelRunnerOutput", "AsyncModelRunnerOutput"]]:
            """Execute model with kv_connector_output.is_empty() check."""
            from vllm.distributed.parallel_state import (
                get_pp_group,
                get_tp_group,
            )
            from vllm.sequence import IntermediateTensors
            from vllm.v1.outputs import (
                EMPTY_MODEL_RUNNER_OUTPUT,
                AsyncModelRunnerOutput,
                ModelRunnerOutput,
            )
            from vllm.v1.worker.utils import is_residual_scattered_for_sp

            intermediate_tensors = None
            forward_pass = scheduler_output.total_num_scheduled_tokens > 0
            num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
            num_input_tokens = self.model_runner._get_num_input_tokens(
                num_scheduled_tokens
            )
            all_gather_tensors = {
                "residual": not is_residual_scattered_for_sp(
                    self.vllm_config, num_input_tokens
                )
            }
            if forward_pass and not get_pp_group().is_first_rank:
                intermediate_tensors = IntermediateTensors(
                    get_pp_group().recv_tensor_dict(
                        all_gather_group=get_tp_group(),
                        all_gather_tensors=all_gather_tensors,
                    )
                )

            output = self.model_runner.execute_model(
                scheduler_output, intermediate_tensors
            )
            if isinstance(output, (ModelRunnerOutput, AsyncModelRunnerOutput)):
                return output
            assert isinstance(output, IntermediateTensors)

            parallel_config = self.vllm_config.parallel_config
            assert (
                parallel_config.distributed_executor_backend != ("external_launcher")
                and not get_pp_group().is_last_rank
            )

            get_pp_group().send_tensor_dict(
                output.tensors,
                all_gather_group=get_tp_group(),
                all_gather_tensors=all_gather_tensors,
            )

            kv_connector_output = output.kv_connector_output
            if not kv_connector_output:
                return None

            # In case of PP with kv transfer, we need to pass through the
            # kv_connector_output
            if kv_connector_output.is_empty():
                return EMPTY_MODEL_RUNNER_OUTPUT

            output = copy.copy(EMPTY_MODEL_RUNNER_OUTPUT)
            output.kv_connector_output = kv_connector_output
            return output

        txn.set_attr(Worker, "execute_model", patched_execute_model)
        txn.set_attr(Worker, "__ucm_patched__", True)  # type: ignore[attr-defined]
    except Exception as e:
        logger.warning(f"Could not patch Worker: {e}")
        raise


def _patch_kv_connector_model_runner_mixin(txn: PatchTxn | None = None) -> None:
    """Set output.invalid_block_ids in KVConnectorModelRunnerMixin."""
    txn = txn or PatchTxn()
    try:
        from contextlib import contextmanager
        from typing import Generator

        from vllm.v1.worker import kv_connector_model_runner_mixin as mixin_mod

        KVConnectorModelRunnerMixin = mixin_mod.KVConnectorModelRunnerMixin
        if getattr(KVConnectorModelRunnerMixin, "__ucm_patched__", False):
            return

        @staticmethod
        @contextmanager
        def patched_get_kv_connector_output(
            scheduler_output: "SchedulerOutput",
            wait_for_save: bool = True,
        ) -> Generator["KVConnectorOutput", None, None]:
            """Get KVConnectorOutput with invalid_block_ids populated."""
            # Keep imports local to reduce import-time coupling across vLLM versions.
            from vllm.distributed.kv_transfer import get_kv_transfer_group
            from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
            from vllm.forward_context import get_forward_context
            from vllm.v1.outputs import KVConnectorOutput

            output = KVConnectorOutput()

            # Update KVConnector with the KVConnector metadata forward().
            kv_connector = get_kv_transfer_group()
            assert isinstance(kv_connector, KVConnectorBase)
            assert scheduler_output.kv_connector_metadata is not None
            kv_connector.bind_connector_metadata(scheduler_output.kv_connector_metadata)

            # Background KV cache transfers happen here.
            # These transfers are designed to be async and the requests
            # involved may be disjoint from the running requests.
            # Do this here to save a collective_rpc.
            kv_connector.start_load_kv(get_forward_context())
            try:
                yield output
            finally:
                if wait_for_save:
                    kv_connector.wait_for_save()

                output.finished_sending, output.finished_recving = (
                    kv_connector.get_finished(scheduler_output.finished_req_ids)
                )
                output.invalid_block_ids = kv_connector.get_block_ids_with_load_errors()
                # if output.invalid_block_ids:
                #    logger.warning(
                #        f"[kv-load] model-runner sees invalid_block_ids={len(output.invalid_block_ids)} "
                #        f"sample={list(output.invalid_block_ids)[:10]}"
                #    )

                output.kv_connector_stats = (
                    KVConnectorModelRunnerMixin.get_kv_connector_stats()
                )
                kv_connector.clear_connector_metadata()

        # Directly replace the whole method (same style as other patches).
        txn.set_attr(
            KVConnectorModelRunnerMixin,
            "_get_kv_connector_output",
            patched_get_kv_connector_output,
        )
        txn.set_attr(KVConnectorModelRunnerMixin, "__ucm_patched__", True)  # type: ignore[attr-defined]
    except Exception as e:
        logger.warning(f"Could not patch KVConnectorModelRunnerMixin: {e}")
        raise
