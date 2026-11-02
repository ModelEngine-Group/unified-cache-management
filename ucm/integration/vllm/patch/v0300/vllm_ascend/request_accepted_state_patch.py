# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Preserve per-request accepted-token counts across scheduling pauses.

Root cause (vllm-ascend 0.30.0, hybrid GDN "align" + MTP + async
scheduling): the accepted-token count doubles as the address of a
request's settled recurrent state (slot = state_idx + accepted - 1).
The count only lives in batch-row-indexed transients -- the GPU
postprocess D2H-copies it in the executed batch's row order, and the
next step's ``_sync_num_accepted_tokens`` attributes the values via
``prev_req_id_to_index``.  When ``defer_block_free`` turns an
allocation failure into a scheduling pause, the paused request's
InputBatch row and prev-mapping entry are deleted while its mamba state
stays parked at the old slot.  On return the count defaults to 1 and
the next forward reads the stale slot-0 state, silently corrupting
generation (repetition loops / garbled output).

Fix (equivalent to the validated preserve-request-accepted-state.diff,
see UCM_MTP_Bug_Investigation_20261031 section 11): harvest the D2H
values into a per-request dict at the event sync that follows each
postprocess submit, and publish from that dict in current batch order.
Entries are invalidated on finished/preempted/resumed/new lifecycle
events so a prior lifetime's pending D2H cannot restore stale values.

Gating: active only for async + speculative + hybrid +
need_accepted_tokens + num_spec_tokens > 0 + mamba_cache_mode ==
"align" (identical to the source diff's
``_use_mamba_accepted_state_cache``).  Every other configuration keeps
the stock vllm-ascend behavior verbatim.

If the installed vllm-ascend has drifted from the validated shape
(missing/renamed methods or changed source markers), the patch logs an
error and stays inactive rather than wrapping an implementation it does
not understand.

Known uncovered edge (same as the source diff): a KV-load-failure rewind
re-truncates ``num_computed_tokens`` while leaving the request RUNNING, so it
carries no finished/preempted/resumed/new marker and the cached count is not
invalidated.  It is only reachable with ``kv_load_failure_policy='recompute'``;
the default ``'fail'`` fails the request (already invalidated), and UCMConnector
refuses ``'recompute'`` for hybrid multi-group KV.  If multi-group recompute is
ever supported, invalidate on
``scheduled num_computed_tokens < worker-side num_computed_tokens``.

Retire this module once vllm-ascend carries the native fix: it detects
``NPUModelRunner._use_mamba_accepted_state_cache`` and no-ops.
"""

import inspect
import sys
from functools import wraps

import numpy as np

from ucm.integration.vllm.patch.utils import when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)

_RUNNER_MODULE = "vllm_ascend.worker.model_runner_v1"
_TARGET_CLASS = "NPUModelRunner"

# Exact parameter tuples of the three methods this patch wraps.  Any drift
# across vllm-ascend revisions skips the install with an error log instead
# of half-applying a correctness fix.
_EXPECTED_PARAMS = {
    "_update_states": ("self", "scheduler_output"),
    "_update_states_after_model_execute": (
        "self",
        "output_token_ids",
        "scheduler_output",
    ),
    "_sync_num_accepted_tokens": ("self", "num_reqs", "has_prev_mapping"),
}

# Source markers proving the wrapped implementation still has the shape this
# patch was validated against (ascend 0.30.0, commit 8fde6d4e).
_SOURCE_MARKERS = {
    "_update_states_after_model_execute": (
        "num_accepted_tokens_cpu_tensor=self.num_accepted_tokens.cpu",
        "postprocess_mamba_align_gpu",
    ),
    "_sync_num_accepted_tokens": (
        "num_accepted_tokens.np",
        "prev_positions.np",
        "has_prev_mapping",
    ),
}


def _install_refused(reason: str) -> None:
    logger.error(
        "UCM request-accepted state patch NOT installed on %s: %s "
        "The fix is INACTIVE on this build; hybrid (mamba align) + MTP + "
        "async workloads keep the pause-time accepted-count loss (silent "
        "precision corruption). Re-verify this patch against the installed "
        "vllm-ascend.",
        _TARGET_CLASS,
        reason,
    )


@when_imported(_RUNNER_MODULE)
def patch_npu_model_runner_accepted_state(mod):
    runner_cls = getattr(mod, _TARGET_CLASS, None)
    if runner_cls is None:
        _install_refused(f"{_RUNNER_MODULE} does not expose {_TARGET_CLASS}.")
        return

    # The native (source-diff) fix is already present: stand down.
    if hasattr(runner_cls, "_use_mamba_accepted_state_cache"):
        return

    if getattr(runner_cls._update_states, "_ucm_accepted_state_patched", False):
        return

    for name, params in _EXPECTED_PARAMS.items():
        fn = getattr(runner_cls, name, None)
        if fn is None:
            _install_refused(f"{_TARGET_CLASS}.{name} is missing.")
            return
        got = tuple(inspect.signature(fn).parameters)
        if got != params:
            _install_refused(
                f"{_TARGET_CLASS}.{name} signature {got} != expected {params}."
            )
            return
    for name, markers in _SOURCE_MARKERS.items():
        try:
            src = inspect.getsource(getattr(runner_cls, name))
        except (OSError, TypeError) as exc:
            _install_refused(f"cannot read source of {_TARGET_CLASS}.{name} ({exc}).")
            return
        for marker in markers:
            if marker not in src:
                _install_refused(
                    f"{_TARGET_CLASS}.{name} no longer contains the "
                    f"validated marker {marker!r}."
                )
                return

    orig_update_states = runner_cls._update_states
    orig_after_execute = runner_cls._update_states_after_model_execute
    orig_sync = runner_cls._sync_num_accepted_tokens

    # ------------------------------------------------------------------
    # Helpers (attached to the runner class; all of them are no-ops unless
    # the gate below is active).
    # ------------------------------------------------------------------

    def _ucm_accepted_state_active(self) -> bool:
        return (
            self.use_async_scheduling
            and self.speculative_config is not None
            and self.model_config.is_hybrid
            and self.need_accepted_tokens
            and self.num_spec_tokens > 0
            and self.cache_config.mamba_cache_mode == "align"
        )

    def _ucm_ensure_accepted_state(self) -> None:
        if not hasattr(self, "_ucm_accepted_by_request"):
            self._ucm_accepted_by_request: dict = {}
            self._ucm_pending_accepted_rows = None

    def _ucm_invalidate_accepted_state(self, scheduler_output) -> None:
        self._ucm_ensure_accepted_state()
        invalidated = set(scheduler_output.finished_req_ids)
        invalidated.update(scheduler_output.preempted_req_ids or ())
        invalidated.update(scheduler_output.scheduled_cached_reqs.resumed_req_ids)
        invalidated.update(req.req_id for req in scheduler_output.scheduled_new_reqs)
        for req_id in invalidated:
            self._ucm_accepted_by_request.pop(req_id, None)
            # A prior lifetime's D2H may still be pending; it must not
            # restore the invalidated value when the event is synchronized.
            if self._ucm_pending_accepted_rows is not None:
                self._ucm_pending_accepted_rows.pop(req_id, None)

    def _ucm_record_pending_accepted_rows(self, num_reqs: int) -> None:
        self._ucm_ensure_accepted_state()
        # The postprocess writes its D2H in this full batch order,
        # independently of later InputBatch row removal/compaction.
        self._ucm_pending_accepted_rows = {
            req_id: row
            for row, req_id in enumerate(self.input_batch.req_ids[:num_reqs])
        }

    # ------------------------------------------------------------------
    # Wrappers.
    # ------------------------------------------------------------------

    @wraps(orig_update_states)
    def _update_states(self, scheduler_output):
        if self._ucm_accepted_state_active():
            # Invalidate before the original applies lifecycle changes; a
            # stale pending D2H from a finished/preempted lifetime must not
            # repopulate the cache at the next event sync.
            self._ucm_invalidate_accepted_state(scheduler_output)
        return orig_update_states(self, scheduler_output)

    @wraps(orig_after_execute)
    def _update_states_after_model_execute(self, output_token_ids, scheduler_output):
        if self._ucm_accepted_state_active():
            # Gate on => the original reaches the align postprocess and
            # submits the async D2H; snapshot req->row BEFORE that submit so
            # the values stay attributable once the event is synchronized.
            self._ucm_record_pending_accepted_rows(int(output_token_ids.size(0)))
        return orig_after_execute(self, output_token_ids, scheduler_output)

    @wraps(orig_sync)
    def _sync_num_accepted_tokens(self, num_reqs: int, has_prev_mapping: bool) -> None:
        if not self._ucm_accepted_state_active():
            return orig_sync(self, num_reqs, has_prev_mapping)

        self._ucm_ensure_accepted_state()
        accepted = self.num_accepted_tokens.np
        # The caller has already synchronized num_accepted_tokens_event, so
        # the D2H values are final for the pending rows.  Harvest every
        # submitted row (including requests paused in the current batch)
        # before the shared buffer can be reordered/overwritten.
        pending = self._ucm_pending_accepted_rows
        if pending is not None:
            for req_id, row in pending.items():
                self._ucm_accepted_by_request[req_id] = int(accepted[row])
            self._ucm_pending_accepted_rows = None
        accepted[:num_reqs] = np.fromiter(
            (
                self._ucm_accepted_by_request.get(req_id, 1)
                for req_id in self.input_batch.req_ids[:num_reqs]
            ),
            dtype=np.int32,
            count=num_reqs,
        )
        self.input_batch.num_accepted_tokens_cpu[:num_reqs] = accepted[:num_reqs]

    runner_cls._ucm_accepted_state_active = _ucm_accepted_state_active
    runner_cls._ucm_ensure_accepted_state = _ucm_ensure_accepted_state
    runner_cls._ucm_invalidate_accepted_state = _ucm_invalidate_accepted_state
    runner_cls._ucm_record_pending_accepted_rows = _ucm_record_pending_accepted_rows
    runner_cls._update_states = _update_states
    runner_cls._update_states_after_model_execute = _update_states_after_model_execute
    runner_cls._sync_num_accepted_tokens = _sync_num_accepted_tokens
    _update_states._ucm_accepted_state_patched = True


# ``when_imported``'s immediate-install path is gated by a module-level
# ``_ucm_patched`` flag (ucm/integration/vllm/patch/utils.py), which another
# patch registering a hook on the same module can already have consumed. In
# that case the decorator above silently does not run for an already-imported
# runner module, and a correctness fix would end up inactive. Run the installer
# explicitly for whatever is already imported; it is idempotent.
_imported_runner_module = sys.modules.get(_RUNNER_MODULE)
if _imported_runner_module is not None:
    patch_npu_model_runner_accepted_state(_imported_runner_module)
