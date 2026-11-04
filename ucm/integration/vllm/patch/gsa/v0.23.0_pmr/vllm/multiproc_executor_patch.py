"""Add PerfCounters to vLLM 0.23.0 WorkerProc."""

from __future__ import annotations

from types import FunctionType

from ucm.integration.vllm.patch.utils import patch_or_inject, when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


def __init__(
    self,
    vllm_config: VllmConfig,
    local_rank: int,
    rank: int,
    distributed_init_method: str,
    input_shm_handle: Handle,
    shared_worker_lock: LockType,
    is_driver_worker: bool,
):
    from ucm.pmr.perf_counter import PerfCounters

    self.rank = rank
    wrapper = WorkerWrapperBase(rpc_rank=local_rank, global_rank=rank)
    # TODO: move `init_worker` to executor level as a collective rpc call
    all_kwargs: list[dict] = [{} for _ in range(vllm_config.parallel_config.world_size)]
    all_kwargs[local_rank] = {
        "vllm_config": vllm_config,
        "local_rank": local_rank,
        "rank": rank,
        "distributed_init_method": distributed_init_method,
        "is_driver_worker": is_driver_worker,
        "shared_worker_lock": shared_worker_lock,
    }
    wrapper.init_worker(all_kwargs)
    self.worker = wrapper

    if is_driver_worker:
        PerfCounters.get_inst().start()

    self.setup_proc_title_and_log_prefix(
        enable_ep=vllm_config.parallel_config.enable_expert_parallel
    )

    # Load model
    self.worker.init_device()
    # Update process title now that parallel groups are initialized
    self.setup_proc_title_and_log_prefix(
        enable_ep=vllm_config.parallel_config.enable_expert_parallel
    )
    if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
        self.worker.elastic_ep_execute("load_model")
    else:
        self.worker.load_model()

    scheduler_config = vllm_config.scheduler_config
    self.use_async_scheduling = scheduler_config.async_scheduling
    if self.use_async_scheduling:
        self.async_output_queue: queue.Queue = queue.Queue()
        self.async_output_copy_thread = Thread(
            target=self.async_output_busy_loop,
            daemon=True,
            name="WorkerAsyncOutputCopy",
        )
        self.async_output_copy_thread.start()

    # Set block size based on the attention backends
    current_platform.update_block_size_for_backend(vllm_config)

    # Initialize message queues after init_device() since multi-node setups
    # (nnodes_within_dp > 1) require distributed groups to be initialized
    self._init_message_queues(input_shm_handle, vllm_config)

    # Enable environment variable cache (e.g. assume no more
    # environment variable overrides after this point)
    enable_envs_cache()


def monitor_death_pipe(self, death_pipe, shutdown_requested: threading.Event):
    from ucm.pmr.perf_counter import PerfCounters

    if death_pipe is None:
        return

    def death_pipe_monitor(queues_to_shutdown: list[MessageQueue]):
        try:
            # This will block until parent process exits (pipe closes)
            death_pipe.recv()
        except EOFError:
            import pickle

            with open("/tmp/worker.pkl", "wb") as f:
                pickle.dump(PerfCounters.get_inst(), f)
            PerfCounters.get_inst().print()
            logger.info_once("Parent process exited, terminating worker queues")
            shutdown_requested.set()
            for mq in queues_to_shutdown:
                if mq is not None:
                    mq.shutdown()
        except Exception as e:
            logger.warning("Death monitoring error: %s", e)

    # Pass queue references directly to avoid gc issues if passing self
    Thread(
        target=death_pipe_monitor,
        args=([self.rpc_broadcast_mq, self.worker_response_mq],),
        daemon=True,
        name="DeathPipeMonitor",
    ).start()


@when_imported("vllm.v1.executor.multiproc_executor")
def _patch(mod):
    if hasattr(mod.WorkerProc, "__init__"):
        _patched = FunctionType(
            __init__.__code__,
            vars(mod),
            name=__init__.__name__,
            argdefs=__init__.__defaults__,
            closure=__init__.__closure__,
        )
        _patched.__annotations__ = __init__.__annotations__
        _patched.__kwdefaults__ = __init__.__kwdefaults__
        patch_or_inject(mod.WorkerProc, "__init__", _patched)
        logger.info("UCM GSA WorkerProc.__init__ patch applied")

    if hasattr(mod.WorkerProc, "monitor_death_pipe"):
        _patched = FunctionType(
            monitor_death_pipe.__code__,
            vars(mod),
            name=monitor_death_pipe.__name__,
            argdefs=monitor_death_pipe.__defaults__,
            closure=monitor_death_pipe.__closure__,
        )
        _patched.__annotations__ = monitor_death_pipe.__annotations__
        _patched.__kwdefaults__ = monitor_death_pipe.__kwdefaults__
        patch_or_inject(mod.WorkerProc, "monitor_death_pipe", _patched)
        logger.info("UCM GSA WorkerProc.monitor_death_pipe patch applied")
