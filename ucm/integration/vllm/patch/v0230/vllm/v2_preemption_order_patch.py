"""Run handle_preemptions before Model Runner V2 rewrites KV blocks.

The scheduler frees a preempted request's blocks right away, and they can be
reallocated in the same step. Since vLLM 0.23.0, V2 ``execute_model`` zeroes
new blocks, applies CoW block copies and copies Mamba state in
``update_requests`` / ``preprocess_state``. ``handle_preemptions`` is only
called later in ``pre_forward``, so those writes can race with the preempted
request's in-flight async dump.

This patch calls ``handle_preemptions`` at the start of ``execute_model``,
matching the V1 runner and vllm-ascend's V2 runner. The later call in
``pre_forward`` finds nothing left to wait for.
"""

from ucm.integration.vllm.patch.utils import when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


@when_imported("vllm.v1.worker.gpu.model_runner")
def patch_v2_preemption_order(mod):
    """Wrap GPUModelRunner.execute_model to flush preempted saves first."""
    try:
        from vllm.distributed.kv_transfer import (
            get_kv_transfer_group,
            has_kv_transfer_group,
        )
    except ImportError:
        logger.warning("Skip V2 preemption order patch: kv_transfer not found")
        return

    runner_cls = getattr(mod, "GPUModelRunner", None)
    if runner_cls is None:
        logger.warning("Skip V2 preemption order patch: GPUModelRunner not found")
        return

    original_execute_model = runner_cls.execute_model
    if getattr(original_execute_model, "_ucm_preemption_order_patched", False):
        return

    def patched_execute_model(self, scheduler_output, *args, **kwargs):
        if has_kv_transfer_group():
            kv_connector_metadata = scheduler_output.kv_connector_metadata
            if kv_connector_metadata is not None:
                get_kv_transfer_group().handle_preemptions(kv_connector_metadata)
        return original_execute_model(self, scheduler_output, *args, **kwargs)

    patched_execute_model._ucm_preemption_order_patched = True
    runner_cls.execute_model = patched_execute_model
    logger.info(
        "UCM V2 preemption order patch applied: handle_preemptions runs "
        "before GPUModelRunner.execute_model updates KV blocks"
    )
