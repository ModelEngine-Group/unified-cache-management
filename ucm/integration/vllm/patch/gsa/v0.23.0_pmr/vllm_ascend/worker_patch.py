"""GSA NPUWorker patches for vLLM-Ascend 0.23.0."""

from __future__ import annotations

from types import FunctionType

from ucm.integration.vllm.patch.utils import patch_or_inject, when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


def init_device(self):
    from ucm.sparse.state import has_ucm_sparse
    from ucm.sparse.utils import PTOPK_PREFETCH_ENABLE

    # NOTE: KEEP device the member of `NPUWorker`, as it will be checked
    # in ray scenario. see https://github.com/vllm-project/vllm/pull/26845
    # for more details
    self.device = self._init_device()
    # Initialize workspace manager
    num_ubatches = 1
    init_workspace_manager(self.device, num_ubatches)
    # Init ModelRunner here, so that we have access to self.device.
    if self.use_v2_model_runner:
        logger.warning(
            "npu model runner v2 is in developing, some features doesn't work for now."
        )
        from vllm_ascend.worker.v2.model_runner import (
            NPUModelRunner as NPUModelRunnerV2,
        )

        self.model_runner = NPUModelRunnerV2(self.vllm_config, self.device)
    else:
        self.model_runner = NPUModelRunner(self.vllm_config, self.device)

    if (
        hasattr(self.vllm_config.model_config, "hf_text_config")
        and hasattr(self.vllm_config.model_config.hf_text_config, "index_topk")
        and has_ucm_sparse()
        and PTOPK_PREFETCH_ENABLE
    ):
        self.maybe_init_ucm_sparse_dsa(self.model_runner)

    if self.rank == 0:
        # If usage stat is enabled, collect relevant info.
        report_usage_stats(self.vllm_config)


def execute_model(
    self,
    scheduler_output: "SchedulerOutput",
) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
    from ucm.pmr.perf_counter import PerfCounters

    self.log_memory_stats()
    # enable msMonitor to monitor the performance of vllm-ascend
    if get_ascend_config().msmonitor_use_daemon:
        dp.step()

    if self._pp_send_work:
        for handle in self._pp_send_work:
            handle.wait()
        self._pp_send_work = []

    intermediate_tensors = None
    forward_pass = scheduler_output.total_num_scheduled_tokens > 0
    if forward_pass and not get_pp_group().is_first_rank:
        # If flashcomm1 is used, this all_gather_group parameter needs to be removed, otherwise
        # it will conflict with the all-gather operation in flashcomm1.
        if enable_sp():
            all_gather_group = None
        else:
            all_gather_group = get_tp_group()
        tensor_dict, comm_handles, comm_postprocess = get_pp_group().irecv_tensor_dict(
            all_gather_group=all_gather_group
        )
        assert tensor_dict is not None
        intermediate_tensors = AsyncIntermediateTensors(
            tensor_dict,
            comm_handles=comm_handles,
            comm_postprocess=comm_postprocess,
        )

    if self.profiler is not None:
        self.profiler.step()

    start_time = time.perf_counter()
    output = self.model_runner.execute_model(scheduler_output, intermediate_tensors)
    if self.is_driver_worker:
        PerfCounters.get_inst().update(
            "--- worker.execute_model", (time.perf_counter() - start_time)
        )

    if isinstance(output, (ModelRunnerOutput, AsyncModelRunnerOutput, NoneType)):
        return output

    assert isinstance(output, IntermediateTensors)
    parallel_config = self.vllm_config.parallel_config
    assert (
        parallel_config.distributed_executor_backend != ("external_launcher")
        and not get_pp_group().is_last_rank
    )
    # If flashcomm1 is used, this all_gather_group parameter needs to be removed, otherwise
    # it will conflict with the all-gather operation in flashcomm1.
    if enable_sp():
        all_gather_group = None
    else:
        all_gather_group = get_tp_group()
    self._pp_send_work = get_pp_group().isend_tensor_dict(
        output.tensors,
        all_gather_group=all_gather_group,
    )

    kv_connector_output = output.kv_connector_output
    if not kv_connector_output:
        return None

    # In case of PP with kv transfer, we need to pass through the
    # kv_connector_output
    if (
        not kv_connector_output.finished_sending
        and not kv_connector_output.finished_recving
    ):
        return EMPTY_MODEL_RUNNER_OUTPUT
    output = copy.copy(EMPTY_MODEL_RUNNER_OUTPUT)
    output.kv_connector_output = kv_connector_output
    return output


def _init_worker_distributed_environment(self) -> None:
    """Initialize the distributed environment."""
    from ucm.sparse.state import ensure_ucm_sparse_initialized

    init_batch_invariance()
    init_distributed_environment(
        self.parallel_config.world_size,
        self.rank,
        self.distributed_init_method,
        self.local_rank,
        "hccl",
    )
    ensure_model_parallel_initialized(
        self.parallel_config.tensor_parallel_size,
        self.parallel_config.pipeline_parallel_size,
        self.parallel_config.prefill_context_parallel_size,
        self.parallel_config.decode_context_parallel_size,
    )
    init_ascend_model_parallel(self.parallel_config)
    ensure_ec_transfer_initialized(self.vllm_config)
    ensure_ucm_sparse_initialized(self.vllm_config)


def maybe_init_ucm_sparse_dsa(self, model_runner):
    from ucm.sparse.state import get_ucm_sparse, has_ucm_sparse

    if not has_ucm_sparse():
        return
    ucm_sparse = get_ucm_sparse()
    ucm_sparse.init_dsa_data(model_runner)


@when_imported("vllm_ascend.worker.worker")
def _patch(mod):
    try:
        import os as _os
        import time as _time

        if not hasattr(mod, "os"):
            setattr(mod, "os", _os)
        if not hasattr(mod, "time"):
            setattr(mod, "time", _time)
    except Exception as e:
        logger.warning(f"GSA worker os/time injection failed {e}")

    if hasattr(mod.NPUWorker, "init_device"):
        _patched = FunctionType(
            init_device.__code__,
            vars(mod),
            name=init_device.__name__,
            argdefs=init_device.__defaults__,
            closure=init_device.__closure__,
        )
        _patched.__annotations__ = init_device.__annotations__
        _patched.__kwdefaults__ = init_device.__kwdefaults__
        patch_or_inject(mod.NPUWorker, "init_device", _patched)
        logger.info("UCM GSA NPUWorker.init_device patch applied")

    if hasattr(mod.NPUWorker, "execute_model"):
        _patched = FunctionType(
            execute_model.__code__,
            vars(mod),
            name=execute_model.__name__,
            argdefs=execute_model.__defaults__,
            closure=execute_model.__closure__,
        )
        _patched.__annotations__ = execute_model.__annotations__
        _patched.__kwdefaults__ = execute_model.__kwdefaults__
        patch_or_inject(mod.NPUWorker, "execute_model", _patched)
        logger.info("UCM GSA NPUWorker.execute_model patch applied")

    if hasattr(mod.NPUWorker, "_init_worker_distributed_environment"):
        _patched = FunctionType(
            _init_worker_distributed_environment.__code__,
            vars(mod),
            name=_init_worker_distributed_environment.__name__,
            argdefs=_init_worker_distributed_environment.__defaults__,
            closure=_init_worker_distributed_environment.__closure__,
        )
        _patched.__annotations__ = _init_worker_distributed_environment.__annotations__
        _patched.__kwdefaults__ = _init_worker_distributed_environment.__kwdefaults__
        patch_or_inject(mod.NPUWorker, "_init_worker_distributed_environment", _patched)
        logger.info(
            "UCM GSA NPUWorker._init_worker_distributed_environment patch applied"
        )

    if not hasattr(mod.NPUWorker, "maybe_init_ucm_sparse_dsa"):
        _patched = FunctionType(
            maybe_init_ucm_sparse_dsa.__code__,
            vars(mod),
            name=maybe_init_ucm_sparse_dsa.__name__,
            argdefs=maybe_init_ucm_sparse_dsa.__defaults__,
            closure=maybe_init_ucm_sparse_dsa.__closure__,
        )
        _patched.__annotations__ = maybe_init_ucm_sparse_dsa.__annotations__
        _patched.__kwdefaults__ = maybe_init_ucm_sparse_dsa.__kwdefaults__
        patch_or_inject(mod.NPUWorker, "maybe_init_ucm_sparse_dsa", _patched)
        logger.info("UCM GSA NPUWorker.maybe_init_ucm_sparse_dsa injected")
