"""Guard counter_prompt_tokens_cached increment in vLLM 0.23.0 metrics."""

from __future__ import annotations

from types import FunctionType

from ucm.integration.vllm.patch.utils import patch_or_inject, when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


def record(
    self,
    scheduler_stats: SchedulerStats | None,
    iteration_stats: IterationStats | None,
    mm_cache_stats: MultiModalCacheStats | None = None,
    engine_idx: int = 0,
):
    """Log to prometheus."""
    if scheduler_stats is not None:
        self.gauge_scheduler_running[engine_idx].set(scheduler_stats.num_running_reqs)
        total_waiting = (
            scheduler_stats.num_waiting_reqs + scheduler_stats.num_skipped_waiting_reqs
        )
        self.gauge_scheduler_waiting[engine_idx].set(total_waiting)
        self.gauge_waiting_by_reason[WAITING_REASON_CAPACITY][engine_idx].set(
            scheduler_stats.num_waiting_reqs
        )
        self.gauge_waiting_by_reason[WAITING_REASON_DEFERRED][engine_idx].set(
            scheduler_stats.num_skipped_waiting_reqs
        )
        self.gauge_kv_cache_usage[engine_idx].set(scheduler_stats.kv_cache_usage)

        self.counter_prefix_cache_queries[engine_idx].inc(
            scheduler_stats.prefix_cache_stats.queries
        )
        self.counter_prefix_cache_hits[engine_idx].inc(
            scheduler_stats.prefix_cache_stats.hits
        )

        if scheduler_stats.connector_prefix_cache_stats is not None:
            self.counter_connector_prefix_cache_queries[engine_idx].inc(
                scheduler_stats.connector_prefix_cache_stats.queries
            )
            self.counter_connector_prefix_cache_hits[engine_idx].inc(
                scheduler_stats.connector_prefix_cache_stats.hits
            )

        if scheduler_stats.spec_decoding_stats is not None:
            self.spec_decoding_prom.observe(
                scheduler_stats.spec_decoding_stats, engine_idx
            )

        if scheduler_stats.kv_connector_stats is not None:
            self.kv_connector_prom.observe(
                scheduler_stats.kv_connector_stats, engine_idx
            )

        if scheduler_stats.perf_stats is not None:
            self.perf_metrics_prom.observe(scheduler_stats.perf_stats, engine_idx)

        if self.kv_cache_metrics_enabled and scheduler_stats.kv_cache_eviction_events:
            lifetime_hist = self.histogram_kv_block_lifetime[engine_idx]
            idle_hist = self.histogram_kv_block_idle_before_evict[engine_idx]
            reuse_hist = self.histogram_kv_block_reuse_gap[engine_idx]

            for event in scheduler_stats.kv_cache_eviction_events:
                lifetime_hist.observe(event.lifetime_seconds)
                idle_hist.observe(event.idle_seconds)
                for gap in event.reuse_gaps_seconds:
                    reuse_hist.observe(gap)

        if self.gauge_lora_info is not None:
            running_lora_adapters = ",".join(
                scheduler_stats.running_lora_adapters.keys()
            )
            waiting_lora_adapters = ",".join(
                scheduler_stats.waiting_lora_adapters.keys()
            )
            lora_info_labels = {
                self.labelname_running_lora_adapters: running_lora_adapters,
                self.labelname_waiting_lora_adapters: waiting_lora_adapters,
                self.labelname_max_lora: self.max_lora,
            }
            self.gauge_lora_info.labels(**lora_info_labels).set_to_current_time()

    if mm_cache_stats is not None:
        self.counter_mm_cache_queries[engine_idx].inc(mm_cache_stats.queries)
        self.counter_mm_cache_hits[engine_idx].inc(mm_cache_stats.hits)

    if iteration_stats is None:
        return
    if envs.VLLM_COMPUTE_NANS_IN_LOGITS:
        self.counter_corrupted_requests[engine_idx].inc(
            iteration_stats.num_corrupted_reqs
        )
    self.counter_num_preempted_reqs[engine_idx].inc(iteration_stats.num_preempted_reqs)
    self.counter_prompt_tokens[engine_idx].inc(iteration_stats.num_prompt_tokens)
    # Labeled prompt token counters by source
    pts = iteration_stats.prompt_token_stats
    for source in PromptTokenStats.ALL_SOURCES:
        self.counter_prompt_tokens_by_source[source][engine_idx].inc(
            pts.get_by_source(source)
        )
    if pts.cached_tokens > 0:
        self.counter_prompt_tokens_cached[engine_idx].inc(pts.cached_tokens)
    self.counter_generation_tokens[engine_idx].inc(
        iteration_stats.num_generation_tokens
    )
    self.histogram_iteration_tokens[engine_idx].observe(
        iteration_stats.prompt_token_stats.computed
        + iteration_stats.num_generation_tokens
    )

    for max_gen_tokens in iteration_stats.max_num_generation_tokens_iter:
        self.histogram_max_num_generation_tokens_request[engine_idx].observe(
            max_gen_tokens
        )
    for n_param in iteration_stats.n_params_iter:
        self.histogram_n_request[engine_idx].observe(n_param)
    for ttft in iteration_stats.time_to_first_tokens_iter:
        self.histogram_time_to_first_token[engine_idx].observe(ttft)
    for itl in iteration_stats.inter_token_latencies_iter:
        self.histogram_inter_token_latency[engine_idx].observe(itl)

    for finished_request in iteration_stats.finished_requests:
        self.counter_request_success[finished_request.finish_reason][engine_idx].inc()
        self.histogram_e2e_time_request[engine_idx].observe(
            finished_request.e2e_latency
        )
        self.histogram_queue_time_request[engine_idx].observe(
            finished_request.queued_time
        )
        self.histogram_prefill_time_request[engine_idx].observe(
            finished_request.prefill_time
        )
        self.histogram_inference_time_request[engine_idx].observe(
            finished_request.inference_time
        )
        self.histogram_decode_time_request[engine_idx].observe(
            finished_request.decode_time
        )
        # Calculate prefill KV compute (excludes cached tokens)
        prefill_kv_computed = finished_request.num_prompt_tokens - max(
            finished_request.num_cached_tokens, 0
        )
        self.histogram_prefill_kv_computed_request[engine_idx].observe(
            prefill_kv_computed
        )
        self.histogram_num_prompt_tokens_request[engine_idx].observe(
            finished_request.num_prompt_tokens
        )
        self.histogram_num_generation_tokens_request[engine_idx].observe(
            finished_request.num_generation_tokens
        )
        self.histogram_request_time_per_output_token[engine_idx].observe(
            finished_request.mean_time_per_output_token
        )
        if finished_request.max_tokens_param:
            self.histogram_max_tokens_request[engine_idx].observe(
                finished_request.max_tokens_param
            )


@when_imported("vllm.v1.metrics.loggers")
def _patch(mod):
    if hasattr(mod.PrometheusStatLogger, "record"):
        _patched = FunctionType(
            record.__code__,
            vars(mod),
            name=record.__name__,
            argdefs=record.__defaults__,
            closure=record.__closure__,
        )
        _patched.__annotations__ = record.__annotations__
        _patched.__kwdefaults__ = record.__kwdefaults__
        patch_or_inject(mod.PrometheusStatLogger, "record", _patched)
        logger.info("UCM GSA PrometheusStatLogger.record patch applied")
