from ucm.integration.vllm.patch.utils import (
    PatchOpProxy,
    patch_dataclass_fields,
    patch_or_inject,
    when_imported,
)
from ucm.logger import init_logger

logger = init_logger(__name__)


@when_imported("vllm_ascend.attention.utils")
def patch_ascend_attention_layer(mod):
    logger.debug(f"Patched {mod} called")

    from ucm.integration.vllm.patch.v0110.vllm_ascend.pc.attention import layer

    patch_or_inject(
        mod, "wait_for_kv_layer_from_connector", layer.wait_for_kv_layer_from_connector
    )
    patch_or_inject(
        mod, "maybe_save_kv_layer_to_connector", layer.maybe_save_kv_layer_to_connector
    )


@when_imported("vllm_ascend.worker.model_runner_v1")
def patch_worker_npu_model_runner(mod):
    logger.debug(f"Patched {mod} called")

    from ucm.integration.vllm.patch.v0110.vllm_ascend.pc.v1.worker import (
        npu_model_runner,
    )

    patch_or_inject(
        mod.NPUModelRunner,
        "execute_model",
        npu_model_runner.NPUModelRunner.execute_model,
    )
    patch_or_inject(
        mod.NPUModelRunner,
        "_update_states",
        npu_model_runner.NPUModelRunner._update_states,
    )


@when_imported("vllm_ascend.worker.worker_v1")
def patch_worker_npu_worker(mod):
    logger.debug(f"Patched {mod} called")
    from ucm.integration.vllm.patch.v0110.vllm_ascend.pc.v1.worker import npu_worker

    patch_or_inject(
        mod.NPUWorker,
        "execute_model",
        npu_worker.NPUWorker.execute_model,
    )
