import torch

from ucm.integration.vllm.patch.utils import (
    PatchOpProxy,
    patch_dataclass_fields,
    patch_or_inject,
    when_imported,
)
from ucm.logger import init_logger

logger = init_logger(__name__)


@when_imported("vllm.attention.layer")
def patch_attention_layer(mod):
    logger.debug(f"Patched {mod} called")

    from ucm.integration.vllm.patch.v0110.sparse.attention import layer

    torch.ops.vllm.unified_attention_with_output = PatchOpProxy(
        torch.ops.vllm.unified_attention_with_output,
        layer.unified_attention_with_output,
    )
    torch.ops.vllm.unified_attention = PatchOpProxy(
        torch.ops.vllm.unified_attention, layer.unified_attention
    )
    patch_or_inject(mod, "unified_attention", layer.unified_attention)
    patch_or_inject(
        mod, "unified_attention_with_output", layer.unified_attention_with_output
    )


@when_imported("vllm.model_executor.models.llama")
def patch_llama_model(mod):
    logger.debug(f"Patched {mod} called")

    from ucm.integration.vllm.patch.v0110.sparse.model_executor.models import llama

    patch_or_inject(mod.LlamaDecoderLayer, "forward", llama.LlamaDecoderLayer.forward)
    patch_or_inject(mod.LlamaModel, "forward", llama.LlamaModel.forward)


@when_imported("vllm.model_executor.models.qwen2")
def patch_qwen2_model(mod):
    logger.debug(f"Patched {mod} called")

    from ucm.integration.vllm.patch.v0110.sparse.model_executor.models import qwen2

    patch_or_inject(mod.Qwen2DecoderLayer, "forward", qwen2.Qwen2DecoderLayer.forward)
    patch_or_inject(mod.Qwen2Model, "forward", qwen2.Qwen2Model.forward)


@when_imported("vllm.v1.attention.backends.mla.common")
def patch_common_attention_backend(mod):
    logger.debug(f"Patched {mod} called")

    from ucm.integration.vllm.patch.v0110.sparse.v1.attention.backends.mla import common

    patch_or_inject(mod.MLACommonImpl, "forward", common.MLACommonImpl.forward)


@when_imported("vllm.v1.attention.backends.mla.flashmla")
def patch_flashmla_attention_backend(mod):
    logger.debug(f"Patched {mod} called")

    from ucm.integration.vllm.patch.v0110.sparse.v1.attention.backends.mla import (
        flashmla,
    )

    patch_dataclass_fields(
        mod.FlashMLADecodeMetadata,
        flashmla.FlashMLADecodeMetadata,
    )

    patch_or_inject(
        mod.FlashMLAMetadataBuilder,
        "_build_decode",
        flashmla.FlashMLAMetadataBuilder._build_decode,
    )


@when_imported("vllm.v1.core.kv_cache_coordinator")
def patch_kv_cache_coordinator(mod):
    logger.debug(f"Patched {mod} called")

    from ucm.integration.vllm.patch.v0110.sparse.v1.core import kv_cache_coordinator

    patch_or_inject(
        mod.KVCacheCoordinator,
        "get_num_blocks_to_allocate",
        kv_cache_coordinator.KVCacheCoordinator.get_num_blocks_to_allocate,
    )


@when_imported("vllm.v1.core.kv_cache_manager")
def patch_kv_cache_manager(mod):
    logger.debug(f"Patched {mod} called")

    from ucm.integration.vllm.patch.v0110.sparse.v1.core import kv_cache_manager

    patch_or_inject(
        mod.KVCacheManager,
        "allocate_slots",
        kv_cache_manager.KVCacheManager.allocate_slots,
    )


@when_imported("vllm.v1.core.kv_cache_utils")
def patch_kv_cache_utils(mod):
    logger.debug(f"Patched {mod} called")

    from ucm.integration.vllm.patch.v0110.sparse.v1.core import kv_cache_utils

    patch_or_inject(
        mod,
        "get_kv_cache_config_from_groups",
        kv_cache_utils.get_kv_cache_config_from_groups,
    )


@when_imported("vllm.v1.core.sched.output")
def patch_scheduler_output(mod):
    logger.debug(f"Patched {mod} called")

    from ucm.integration.vllm.patch.v0110.sparse.v1.core.sched import output

    patch_dataclass_fields(
        mod.SchedulerOutput,
        output.SchedulerOutput,
    )


@when_imported("vllm.v1.worker.block_table")
def patch_worker_block_table(mod):
    logger.debug(f"Patched {mod} called")

    from ucm.integration.vllm.patch.v0110.sparse.v1.worker import block_table

    patch_or_inject(mod.BlockTable, "append_row", block_table.BlockTable.append_row)
    patch_or_inject(
        mod.MultiGroupBlockTable,
        "reset_row",
        block_table.MultiGroupBlockTable.reset_row,
    )


@when_imported("vllm.v1.worker.gpu_worker")
def patch_worker_gpu_worker(mod):
    logger.debug(f"Patched {mod} called")

    from ucm.integration.vllm.patch.v0110.sparse.v1.worker import gpu_worker

    patch_or_inject(
        mod,
        "init_worker_distributed_environment",
        gpu_worker.init_worker_distributed_environment,
    )


@when_imported("vllm.v1.worker.gpu_model_runner")
def patch_worker_gpu_model_runner(mod):
    logger.debug(f"Patched {mod} called")

    from ucm.integration.vllm.patch.v0110.sparse.v1.worker import gpu_model_runner

    patch_or_inject(
        mod.GPUModelRunner,
        "_update_states",
        gpu_model_runner.GPUModelRunner._update_states,
    )
    patch_or_inject(
        mod.GPUModelRunner,
        "_prepare_inputs",
        gpu_model_runner.GPUModelRunner._prepare_inputs,
    )
    patch_or_inject(
        mod.GPUModelRunner,
        "execute_model",
        gpu_model_runner.GPUModelRunner.execute_model,
    )
    patch_or_inject(
        mod.GPUModelRunner,
        "maybe_execute_ucm_sparse_begin",
        gpu_model_runner.GPUModelRunner.maybe_execute_ucm_sparse_begin,
    )
    patch_or_inject(
        mod.GPUModelRunner,
        "maybe_execute_ucm_sparse_finished",
        gpu_model_runner.GPUModelRunner.maybe_execute_ucm_sparse_finished,
    )
    patch_or_inject(
        mod.GPUModelRunner,
        "ucm_sparse_request_finished_in_worker",
        gpu_model_runner.GPUModelRunner.ucm_sparse_request_finished_in_worker,
    )
    patch_or_inject(
        mod.GPUModelRunner,
        "ucm_sparse_update_states",
        gpu_model_runner.GPUModelRunner.ucm_sparse_update_states,
    )
    patch_or_inject(
        mod.GPUModelRunner,
        "initialize_kv_cache_tensors",
        gpu_model_runner.GPUModelRunner.initialize_kv_cache_tensors,
    )


@when_imported("vllm.v1.core.sched.scheduler")
def patch_core_sched_scheduler(mod):
    logger.debug(f"Patched {mod} called")

    from ucm.integration.vllm.patch.v0110.sparse.v1.core.sched import scheduler

    patch_or_inject(mod.Scheduler, "__init__", scheduler.Scheduler.__init__)
    patch_or_inject(mod.Scheduler, "schedule", scheduler.Scheduler.schedule)
    patch_or_inject(mod.Scheduler, "add_request", scheduler.Scheduler.add_request)
    patch_or_inject(mod.Scheduler, "_free_request", scheduler.Scheduler._free_request)
