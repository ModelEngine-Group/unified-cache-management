from ucm.integration.vllm.patch.utils import (
    patch_dataclass_fields,
    patch_or_inject,
    when_imported,
)
from ucm.logger import init_logger

logger = init_logger(__name__)


@when_imported("vllm.attention.layer")
def patch_attention_layer(mod):
    logger.debug(f"Patched {mod} called")

    from ucm.integration.vllm.patch.v0110.vllm.pc.attention import layer

    patch_or_inject(
        mod, "wait_for_kv_layer_from_connector", layer.wait_for_kv_layer_from_connector
    )
    patch_or_inject(
        mod, "maybe_save_kv_layer_to_connector", layer.maybe_save_kv_layer_to_connector
    )


@when_imported("vllm.distributed.kv_transfer.kv_connector.v1.base")
def patch_kv_connector_base_v1(mod):
    logger.debug(f"Patched {mod} called")

    from ucm.integration.vllm.patch.v0110.vllm.pc.v1.distributed.kv_connector import (
        base,
    )

    patch_or_inject(
        mod.KVConnectorBase_V1,
        "get_block_ids_with_load_errors",
        base.KVConnectorBase_V1.get_block_ids_with_load_errors,
    )
    patch_or_inject(
        mod.KVConnectorBase_V1,
        "has_connector_metadata",
        base.KVConnectorBase_V1.has_connector_metadata,
    )


@when_imported("vllm.v1.core.sched.output")
def patch_scheduler_output(mod):
    logger.debug(f"Patched {mod} called")

    from ucm.integration.vllm.patch.v0110.vllm.pc.v1.core.sched import output

    patch_dataclass_fields(
        mod.SchedulerOutput,
        output.SchedulerOutput,
    )


@when_imported("vllm.v1.core.sched.output")
def patch_cached_request_data(mod):
    logger.debug(f"Patched {mod} called")

    from ucm.integration.vllm.patch.v0110.vllm.pc.v1.core.sched import (
        cached_request_data,
    )

    patch_or_inject(
        mod,
        "CachedRequestData",
        cached_request_data.CachedRequestData,
    )


@when_imported("vllm.v1.outputs")
def patch_outputs(mod):
    logger.debug(f"Patched {mod} called")

    from ucm.integration.vllm.patch.v0110.vllm.pc.v1 import outputs

    patch_dataclass_fields(
        mod.KVConnectorOutput,
        outputs.KVConnectorOutput,
    )
    patch_or_inject(
        mod.KVConnectorOutput,
        "is_empty",
        outputs.KVConnectorOutput.is_empty,
    )


@when_imported("vllm.v1.core.block_pool")
def patch_block_pool(mod):
    logger.debug(f"Patched {mod} called")

    from ucm.integration.vllm.patch.v0110.vllm.pc.v1.core import block_pool

    patch_or_inject(
        mod.BlockPool,
        "cache_full_blocks",
        block_pool.BlockPool.cache_full_blocks,
    )


@when_imported("vllm.v1.core.single_type_kv_cache_manager")
def patch_single_type_kv_cache_manager(mod):
    logger.debug(f"Patched {mod} called")

    from ucm.integration.vllm.patch.v0110.vllm.pc.v1.core import (
        single_type_kv_cache_manager,
    )

    patch_or_inject(
        mod.SingleTypeKVCacheManager,
        "cache_blocks",
        single_type_kv_cache_manager.SingleTypeKVCacheManager.cache_blocks,
    )


@when_imported("vllm.v1.worker.gpu_worker")
def patch_worker_gpu_worker(mod):
    logger.debug(f"Patched {mod} called")
    from ucm.integration.vllm.patch.v0110.vllm.pc.v1.worker import gpu_worker

    patch_or_inject(
        mod.Worker,
        "execute_model",
        gpu_worker.Worker.execute_model,
    )


@when_imported("vllm.v1.worker.gpu_model_runner")
def patch_worker_gpu_model_runner(mod):
    logger.debug(f"Patched {mod} called")

    from ucm.integration.vllm.patch.v0110.vllm.pc.v1.worker import gpu_model_runner

    patch_or_inject(
        mod.GPUModelRunner,
        "_update_states",
        gpu_model_runner.GPUModelRunner._update_states,
    )


@when_imported("vllm.v1.core.sched.scheduler")
def patch_core_sched_scheduler(mod):
    """Patch Scheduler for load-failure recovery."""
    logger.debug(f"Patched {mod} called")

    from ucm.integration.vllm.patch.v0110.vllm.pc.v1.core.sched import scheduler

    patch_or_inject(mod.Scheduler, "__init__", scheduler.Scheduler.__init__)
    patch_or_inject(
        mod.Scheduler,
        "_update_requests_with_invalid_blocks",
        scheduler.Scheduler._update_requests_with_invalid_blocks,
    )
    patch_or_inject(
        mod.Scheduler,
        "_handle_invalid_blocks",
        scheduler.Scheduler._handle_invalid_blocks,
    )
    patch_or_inject(
        mod.Scheduler,
        "_make_cached_request_data",
        scheduler.Scheduler._make_cached_request_data,
    )
    patch_or_inject(
        mod.Scheduler,
        "update_from_output",
        scheduler.Scheduler.update_from_output,
    )
    patch_or_inject(
        mod.Scheduler,
        "_update_waiting_for_remote_kv",
        scheduler.Scheduler._update_waiting_for_remote_kv,
    )


@when_imported("vllm.distributed.kv_transfer.kv_connector.utils")
def patch_kv_connector_utils(mod):
    logger.debug(f"Patched {mod} called")

    from ucm.integration.vllm.patch.v0110.vllm.pc.v1.distributed.kv_connector import (
        utils,
    )

    patch_or_inject(
        mod.KVOutputAggregator,
        "aggregate",
        utils.KVOutputAggregator.aggregate,
    )


@when_imported("vllm.v1.worker.kv_connector_model_runner_mixin")
def patch_kv_connector_model_runner_mixin(mod):
    logger.debug(f"Patched {mod} called")

    from ucm.integration.vllm.patch.v0110.vllm.pc.v1.worker import (
        kv_connector_model_runner_mixin,
    )

    patch_or_inject(
        mod.KVConnectorModelRunnerMixin,
        "_get_kv_connector_output",
        kv_connector_model_runner_mixin._get_kv_connector_output,
    )
