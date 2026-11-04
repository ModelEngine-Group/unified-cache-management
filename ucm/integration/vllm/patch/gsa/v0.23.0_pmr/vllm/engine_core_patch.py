"""GSA EngineCore patches for vLLM 0.23.0."""

from __future__ import annotations

from types import FunctionType

from ucm.integration.vllm.patch.utils import patch_or_inject, when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


def __init__(
    self,
    vllm_config: VllmConfig,
    executor_class: type[Executor],
    log_stats: bool,
    executor_fail_callback: Callable | None = None,
    include_finished_set: bool = False,
):
    # plugins need to be loaded at the engine/scheduler level too
    from vllm.plugins import load_general_plugins

    from ucm.sparse.utils import IS_OPEN_GSA, rm_shared_memory

    load_general_plugins()

    self.vllm_config = vllm_config
    if not vllm_config.parallel_config.data_parallel_rank_local:
        logger.info(
            "Initializing a V1 LLM engine (v%s) with config: %s",
            VLLM_VERSION,
            vllm_config,
        )

    if IS_OPEN_GSA:
        rm_shared_memory(vllm_config.parallel_config.data_parallel_rank)

    self.log_stats = log_stats

    # Setup Model.
    self.model_executor = executor_class(vllm_config)
    if executor_fail_callback is not None:
        self.model_executor.register_failure_callback(executor_fail_callback)

    self.available_gpu_memory_for_kv_cache = -1

    if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
        self._eep_scale_up_before_kv_init()

    # Setup KV Caches and update CacheConfig after profiling.
    kv_cache_config = self._initialize_kv_caches(vllm_config)
    self.structured_output_manager = StructuredOutputManager(vllm_config)

    # Setup scheduler.
    Scheduler = vllm_config.scheduler_config.get_scheduler_cls()

    if len(kv_cache_config.kv_cache_groups) == 0:  # noqa: SIM102
        # Encoder models without KV cache don't support
        # chunked prefill. But do SSM models?
        if vllm_config.scheduler_config.enable_chunked_prefill:
            logger.warning("Disabling chunked prefill for model without KVCache")
            vllm_config.scheduler_config.enable_chunked_prefill = False

    scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(
        kv_cache_config, vllm_config
    )

    self.scheduler: SchedulerInterface = Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=self.structured_output_manager,
        include_finished_set=include_finished_set,
        log_stats=self.log_stats,
        block_size=scheduler_block_size,
        hash_block_size=hash_block_size,
    )
    self.use_spec_decode = vllm_config.speculative_config is not None
    if self.scheduler.connector is not None:  # type: ignore
        self.model_executor.init_kv_output_aggregator(self.scheduler.connector)  # type: ignore

    mm_registry = MULTIMODAL_REGISTRY
    self.mm_receiver_cache = mm_registry.engine_receiver_cache_from_config(vllm_config)

    # If a KV connector is initialized for scheduler, we want to collect
    # handshake metadata from all workers so the connector in the scheduler
    # will have the full context
    kv_connector = self.scheduler.get_kv_connector()
    if kv_connector is not None:
        # Collect and store KV connector xfer metadata from workers
        # (after KV cache registration)
        xfer_handshake_metadata = (
            self.model_executor.get_kv_connector_handshake_metadata()
        )

        if xfer_handshake_metadata:
            # xfer_handshake_metadata is list of dicts from workers
            # Each dict already has structure {(pp_rank, tp_rank): metadata}
            # Merge all worker dicts into a single dict
            content: dict[tuple[int, int], Any] = {}
            for worker_dict in xfer_handshake_metadata:
                if worker_dict is not None:
                    content.update(worker_dict)
            kv_connector.set_xfer_handshake_metadata_pp_aware(content)

    # Setup batch queue for pipeline parallelism.
    # Batch queue for scheduled batches. This enables us to asynchronously
    # schedule and execute batches, and is required by pipeline parallelism
    # to eliminate pipeline bubbles.
    self.batch_queue_size = vllm_config.max_concurrent_batches
    self.batch_queue: (
        deque[tuple[Future[ModelRunnerOutput], SchedulerOutput, Future[Any]]] | None
    ) = None
    if self.batch_queue_size > 1:
        logger.debug("Batch queue is enabled with size %d", self.batch_queue_size)
        self.batch_queue = deque(maxlen=self.batch_queue_size)

    self.is_ec_consumer = (
        vllm_config.ec_transfer_config is None
        or vllm_config.ec_transfer_config.is_ec_consumer
    )
    self.is_pooling_model = vllm_config.model_config.runner_type == "pooling"

    self.request_block_hasher: Callable[[Request], list[BlockHash]] | None = None
    if vllm_config.cache_config.enable_prefix_caching or kv_connector is not None:
        caching_hash_fn = get_hash_fn_by_name(
            vllm_config.cache_config.prefix_caching_hash_algo
        )
        init_none_hash(caching_hash_fn)

        self.request_block_hasher = get_request_block_hasher(
            hash_block_size, caching_hash_fn
        )

    self.step_fn = self.step if self.batch_queue is None else self.step_with_batch_queue
    self.async_scheduling = vllm_config.scheduler_config.async_scheduling

    self.aborts_queue = queue.Queue[list[str]]()

    self._idle_state_callbacks: list[Callable] = []

    # Mark the startup heap as static so that it's ignored by GC.
    # Reduces pause times of oldest generation collections.
    freeze_gc_heap()
    # If enable, attach GC debugger after static variable freeze.
    maybe_attach_gc_debug_callback()
    # Enable environment variable cache (e.g. assume no more
    # environment variable overrides after this point)
    enable_envs_cache()


def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
    """Schedule, execute, and make output.

    Returns tuple of outputs and a flag indicating whether the model
    was executed.
    """

    # Check for any requests remaining in the scheduler - unfinished,
    # or finished and not yet removed from the batch.
    if not self.scheduler.has_requests():
        return {}, False
    scheduler_output = self.scheduler.schedule()
    future = self.model_executor.execute_model(scheduler_output, non_block=True)
    grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
    with (
        self.log_error_detail(scheduler_output),
        self.log_iteration_details(scheduler_output),
    ):
        model_output = future.result()
        if model_output is None:
            model_output = self.model_executor.sample_tokens(grammar_output)

    if (
        not self.async_scheduling
        and self.use_spec_decode
        and scheduler_output.total_num_scheduled_tokens > 0
    ):
        # Take the draft token ids.
        draft_token_ids = self.model_executor.take_draft_token_ids()
        if draft_token_ids is not None:
            self.scheduler.update_draft_token_ids(draft_token_ids)

    # Before processing the model output, process any aborts that happened
    # during the model execution.
    self._process_aborts_queue()
    engine_core_outputs = self.scheduler.update_from_output(
        scheduler_output, model_output
    )

    return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0


def post_step(self, model_executed: bool) -> None:
    return
    # When using async scheduling we can't get draft token ids in advance,
    # so we update draft token ids in the worker process and don't
    # need to update draft token ids here.
    if not self.async_scheduling and self.use_spec_decode and model_executed:
        # Take the draft token ids.
        draft_token_ids = self.model_executor.take_draft_token_ids()
        if draft_token_ids is not None:
            self.scheduler.update_draft_token_ids(draft_token_ids)


def run_engine_core(*args, dp_rank: int = 0, local_dp_rank: int = 0, **kwargs):
    """Launch EngineCore busy loop in background process."""
    from ucm.pmr.perf_counter import PerfCounters

    # Ensure we can serialize transformer config after spawning
    maybe_register_config_serialize_by_value()

    engine_core: EngineCoreProc | None = None
    signal_callback: SignalCallback | None = None
    try:
        vllm_config: VllmConfig = kwargs["vllm_config"]
        parallel_config: ParallelConfig = vllm_config.parallel_config
        data_parallel = parallel_config.data_parallel_size > 1 or dp_rank > 0
        if data_parallel:
            parallel_config.data_parallel_rank_local = local_dp_rank
            process_title = f"EngineCore_DP{dp_rank}"
        else:
            process_title = "EngineCore"
        set_process_title(process_title)
        maybe_init_worker_tracer("vllm.engine_core", "engine_core", process_title)
        decorate_logs()
        if parallel_config.numa_bind:
            numa_utils.log_current_affinity_state(process_title)

        if data_parallel and vllm_config.kv_transfer_config is not None:
            # modify the engine_id and append the local_dp_rank to it to ensure
            # that the kv_transfer_config is unique for each DP rank.
            vllm_config.kv_transfer_config.engine_id = (
                f"{vllm_config.kv_transfer_config.engine_id}_dp{local_dp_rank}"
            )
            logger.debug(
                "Setting kv_transfer_config.engine_id to %s",
                vllm_config.kv_transfer_config.engine_id,
            )

        parallel_config.data_parallel_index = dp_rank
        if data_parallel and vllm_config.model_config.is_moe:
            # Set data parallel rank for this engine process.
            parallel_config.data_parallel_rank = dp_rank
            engine_core = DPEngineCoreProc(*args, **kwargs)
        else:
            # Non-MoE DP ranks are completely independent, so treat like DP=1.
            # Note that parallel_config.data_parallel_index will still reflect
            # the original DP rank.
            parallel_config.data_parallel_size = 1
            parallel_config.data_parallel_size_local = 1
            parallel_config.data_parallel_rank = 0
            engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)

        assert engine_core is not None

        def wakeup_engine():
            # Wakes up idle engine via input_queue when shutdown is requested
            # Not safe in a signal handler - we may interrupt the main thread
            # while it is holding the non-reentrant input_queue.mutex
            engine_core.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))

        signal_callback = SignalCallback(wakeup_engine)

        def signal_handler(signum, frame):
            signal_name = signal.Signals(signum).name
            logger.info(
                "[shutdown] EngineCore: trigger received signal=%s",
                signal_name,
            )
            engine_core.shutdown_state = EngineShutdownState.REQUESTED
            signal_callback.trigger()

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        PerfCounters.get_inst().start()

        engine_core.run_busy_loop()

    except SystemExit:
        logger.info_once("[shutdown] EngineCore: exiting busy loop")
        raise
    except Exception as e:
        if engine_core is None:
            logger.exception("EngineCore failed to start.")
        else:
            logger.exception("EngineCore encountered a fatal error.")
            engine_core._send_engine_dead()
        raise e
    finally:
        if "speculatorsearch" in sys.modules:
            sys.modules["speculatorsearch"].SpeculatorPrintFtdsInfo()
        dump_and_spawn()
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        if signal_callback is not None:
            signal_callback.stop()
        if engine_core is not None:
            engine_core.shutdown()


def _process_input_queue(self):
    """Exits when an engine step needs to be performed."""

    waited = False
    while not self.has_work() and self.is_running():
        # Notify callbacks waiting for engine to become idle.
        self._notify_idle_state_callbacks()
        if self.input_queue.empty():
            # Drain aborts queue; all aborts are also processed via input_queue.
            with self.aborts_queue.mutex:
                self.aborts_queue.queue.clear()
            if logger.isEnabledFor(DEBUG):
                logger.debug("EngineCore waiting for work.")
                waited = True
        block = self.process_input_queue_block
        try:
            req = self.input_queue.get(block=block)
            self._handle_client_request(*req)
        except queue.Empty:
            break
        if not block:
            break

    if waited:
        logger.debug("EngineCore loop active.")

    # Handle any more client requests.
    while not self.input_queue.empty():
        req = self.input_queue.get_nowait()
        self._handle_client_request(*req)


def dump_and_spawn():
    """主进程：1. 立即 dump 数据  2. 启动子进程做长任务"""
    from ucm.pmr.perf_counter import PerfCounters

    if os.path.exists(DONE_FLAG):
        os.remove(DONE_FLAG)  # 重新跑
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)  # 重新跑
    with open(DATA_FILE, "wb") as f:
        pickle.dump(PerfCounters.get_inst(), f)
    # 子进程跑长任务（会等它完成）
    import multiprocessing as mp
    import shlex
    import subprocess

    cmd = f"nohup python -u {script_path}"
    subprocess.Popen(shlex.split(cmd), shell=False)


def long_task_sub():
    """子进程：读数据 → 耗时计算 → 写结果 → 写完成标志"""
    if os.path.exists(DONE_FLAG):
        return  # 已完成
    if not os.path.exists(DATA_FILE):
        return  # 无数据
    with open(DATA_FILE, "rb") as f:
        perfCounter = pickle.load(f)
        perfCounter.print()
    return


@when_imported("vllm.v1.engine.core")
def _patch(mod):
    # Inject module-level constants needed by dump_and_spawn / long_task_sub.
    import os
    import pickle
    import subprocess
    import sys
    import time

    from ucm.pmr.perf_counter import PerfCounters

    _SUB_CODE = """
DATA_FILE = "/tmp/job_data.pkl"
DONE_FLAG = "/tmp/job_done.flag"
import sys, pickle, os, time
def run():
    if os.path.exists(DONE_FLAG):
        return
    if not os.path.exists(DATA_FILE):
        return
    with open(DATA_FILE, "rb") as f:
        perfCounter = pickle.load(f)
        perfCounter.print()
    return
run()
"""
    _script_path = "/tmp/nohup_script.py"
    for _name, _val in {
        "subprocess": subprocess,
        "sys": sys,
        "os": os,
        "time": time,
        "pickle": pickle,
        "PerfCounters": PerfCounters,
        "DATA_FILE": "/tmp/job_data.pkl",
        "DONE_FLAG": "/tmp/job_done.flag",
        "LOG_FILE": "nohup.out",
        "SUB_CODE": _SUB_CODE,
        "script_path": _script_path,
    }.items():
        if not hasattr(mod, _name):
            setattr(mod, _name, _val)
    try:
        with open(_script_path, "w") as _f:
            _f.write(_SUB_CODE)
        os.chmod(_script_path, 0o755)
    except Exception as e:
        logger.warning(f"GSA engine_core nohup script write failed: {e}")

    if hasattr(mod.EngineCore, "__init__"):
        _patched = FunctionType(
            __init__.__code__,
            vars(mod),
            name=__init__.__name__,
            argdefs=__init__.__defaults__,
            closure=__init__.__closure__,
        )
        _patched.__annotations__ = __init__.__annotations__
        _patched.__kwdefaults__ = __init__.__kwdefaults__
        patch_or_inject(mod.EngineCore, "__init__", _patched)
        logger.info("UCM GSA EngineCore.__init__ patch applied")

    if hasattr(mod.EngineCore, "step"):
        _patched = FunctionType(
            step.__code__,
            vars(mod),
            name=step.__name__,
            argdefs=step.__defaults__,
            closure=step.__closure__,
        )
        _patched.__annotations__ = step.__annotations__
        _patched.__kwdefaults__ = step.__kwdefaults__
        patch_or_inject(mod.EngineCore, "step", _patched)
        logger.info("UCM GSA EngineCore.step patch applied")

    if hasattr(mod.EngineCore, "post_step"):
        _patched = FunctionType(
            post_step.__code__,
            vars(mod),
            name=post_step.__name__,
            argdefs=post_step.__defaults__,
            closure=post_step.__closure__,
        )
        _patched.__annotations__ = post_step.__annotations__
        _patched.__kwdefaults__ = post_step.__kwdefaults__
        patch_or_inject(mod.EngineCore, "post_step", _patched)
        logger.info("UCM GSA EngineCore.post_step patch applied")

    if hasattr(mod.EngineCoreProc, "run_engine_core"):
        _patched = FunctionType(
            run_engine_core.__code__,
            vars(mod),
            name=run_engine_core.__name__,
            argdefs=run_engine_core.__defaults__,
            closure=run_engine_core.__closure__,
        )
        _patched.__annotations__ = run_engine_core.__annotations__
        _patched.__kwdefaults__ = run_engine_core.__kwdefaults__
        patch_or_inject(mod.EngineCoreProc, "run_engine_core", _patched)
        logger.info("UCM GSA EngineCoreProc.run_engine_core patch applied")

    if hasattr(mod.EngineCoreProc, "_process_input_queue"):
        _patched = FunctionType(
            _process_input_queue.__code__,
            vars(mod),
            name=_process_input_queue.__name__,
            argdefs=_process_input_queue.__defaults__,
            closure=_process_input_queue.__closure__,
        )
        _patched.__annotations__ = _process_input_queue.__annotations__
        _patched.__kwdefaults__ = _process_input_queue.__kwdefaults__
        patch_or_inject(mod.EngineCoreProc, "_process_input_queue", _patched)
        logger.info("UCM GSA EngineCoreProc._process_input_queue patch applied")

    if not hasattr(mod, "dump_and_spawn"):
        _patched = FunctionType(
            dump_and_spawn.__code__,
            vars(mod),
            name=dump_and_spawn.__name__,
            argdefs=dump_and_spawn.__defaults__,
            closure=dump_and_spawn.__closure__,
        )
        _patched.__annotations__ = dump_and_spawn.__annotations__
        _patched.__kwdefaults__ = dump_and_spawn.__kwdefaults__
        mod.dump_and_spawn = _patched
        logger.info("UCM GSA dump_and_spawn injected")

    if not hasattr(mod, "long_task_sub"):
        _patched = FunctionType(
            long_task_sub.__code__,
            vars(mod),
            name=long_task_sub.__name__,
            argdefs=long_task_sub.__defaults__,
            closure=long_task_sub.__closure__,
        )
        _patched.__annotations__ = long_task_sub.__annotations__
        _patched.__kwdefaults__ = long_task_sub.__kwdefaults__
        mod.long_task_sub = _patched
        logger.info("UCM GSA long_task_sub injected")
