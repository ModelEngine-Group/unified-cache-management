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


@when_imported("vllm.v1.engine.core")
def _patch(mod):
    # Inject module-level constants needed by dump_and_spawn / long_task_sub.
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
