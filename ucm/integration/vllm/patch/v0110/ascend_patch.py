import torch

from ucm.integration.vllm.patch.utils import (
    PatchOpProxy,
    patch_dataclass_fields,
    patch_or_inject,
    when_imported,
)
from ucm.logger import init_logger

logger = init_logger(__name__)

@when_imported("vllm_ascend.attention.attention_v1")
def patch_attention_v1(mod):
    logger.debug(f"Patched {mod} called")

    from ucm.integration.vllm.patch.v0110.ascend.attention import attention_v1

    patch_dataclass_fields(mod.AscendMetadata,
        attention_v1.AscendMetadata)
    patch_or_inject(mod.AscendAttentionMetadataBuilder,
        "build", attention_v1.AscendAttentionMetadataBuilder.build)
    patch_or_inject(mod.AscendAttentionBackendImpl,
        "forward", attention_v1.AscendAttentionBackendImpl.forward)
    torch.ops.vllm.unified_attention_with_output = PatchOpProxy(
        torch.ops.vllm.unified_ascend_attention_with_output,
        attention_v1.unified_ascend_attention_with_output,
    )

@when_imported("vllm_ascend.attention.mla_v1")
def patch_mla_v1(mod):
    logger.debug(f"Patched {mod} called")

    from ucm.integration.vllm.patch.v0110.ascend.attention import mla_v1

    patch_dataclass_fields(mod.AscendMLADecodeMetadata, 
        mla_v1.AscendMLADecodeMetadata)
    patch_dataclass_fields(mod.AscendMLAMetadata, 
        mla_v1.AscendMLAMetadata)
    patch_or_inject(mod.AscendMLAMetadataBuilder,
        "build", mla_v1.AscendMLAMetadataBuilder.build)
    patch_dataclass_fields(mod.PrefillMLAPreprocessResult,
        mla_v1.PrefillMLAPreprocessResult)
    patch_or_inject(mod.AscendMLAImpl,
        "_mla_preprocess", mla_v1.AscendMLAImpl._mla_preprocess)
    patch_or_inject(mod.AscendMLAImpl,
        "forward", mla_v1.AscendMLAImpl.forward)

@when_imported("vllm_ascend.worker.block_table")
def patch_block_table(mod):
    logger.debug(f"Patched {mod} called")

    from ucm.integration.vllm.patch.v0110.ascend.worker import block_table

    patch_or_inject(mod.BlockTable,
        "reset_now", block_table.BlockTable.reset_row)
    patch_or_inject(mod.MultiGroupBlockTable,
        "reset_now", block_table.MultiGroupBlockTable.reset_row)

@when_imported("vllm_ascend.worker.model_runner_v1")
def patch_model_runner_v1(mod):
    logger.debug(f"Patched {mod} called")

    from ucm.integration.vllm.patch.v0110.ascend.worker import model_runner_v1

    patch_or_inject(mod.NPUModelRunner,
        "_update_states", model_runner_v1.NPUModelRunner._update_states)
    patch_or_inject(mod.NPUModelRunner,
        "_prepare_inputs", model_runner_v1.NPUModelRunner._prepare_inputs)
    patch_or_inject(mod.NPUModelRunner,
        "execute_model", model_runner_v1.NPUModelRunner.execute_model)
    patch_or_inject(mod.NPUModelRunner,
        "maybe_execute_ucm_sparse_begin",
        model_runner_v1.NPUModelRunner.maybe_execute_ucm_sparse_begin)
    patch_or_inject(mod.NPUModelRunner,
        "maybe_execute_ucm_sparse_finished",
        model_runner_v1.NPUModelRunner.maybe_execute_ucm_sparse_finished)
    patch_or_inject(mod.NPUModelRunner,
        "ucm_sparse_request_finished_in_worker",
        model_runner_v1.NPUModelRunner.ucm_sparse_request_finished_in_worker)
    patch_or_inject(mod.NPUModelRunner,
        "ucm_sparse_update_states",
        model_runner_v1.NPUModelRunner.ucm_sparse_update_states)
    patch_or_inject(mod.NPUModelRunner,
        "initialize_kv_cache_tensors_deepseek_mla",
        model_runner_v1.NPUModelRunner.initialize_kv_cache_tensors_deepseek_mla)
    patch_or_inject(mod.NPUModelRunner,
        "initialize_kv_cache_tensors",
        model_runner_v1.NPUModelRunner.initialize_kv_cache_tensors)

@when_imported("vllm_ascend.worker.worker_v1")
def patch_worker_v1(mod):
    logger.debug(f"Patched {mod} called")

    from ucm.integration.vllm.patch.v0110.ascend.worker import worker_v1

    patch_or_inject(mod.NPUWorker,
        "_init_worker_distributed_environment",
        worker_v1.NPUWorker._init_worker_distributed_environment)
