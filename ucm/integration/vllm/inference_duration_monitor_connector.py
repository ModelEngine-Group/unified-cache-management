"""No-I/O vLLM connector for forward and per-layer block-window timing."""

import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Optional

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)

try:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorWorkerMetadata,
    )
except ImportError:
    KVConnectorWorkerMetadata = object

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput

from ucm.integration.vllm.device import Device, create_device
from ucm.logger import init_logger
from ucm.utils import Config

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


def inference_duration_monitor_enabled(extra_config: dict[str, Any]) -> bool:
    """Resolve the monitor switch from inline or YAML-backed UCM config."""
    kv_transfer_config = SimpleNamespace(kv_connector_extra_config=extra_config)
    launch_config = Config(kv_transfer_config).get_config() or {}
    return bool(launch_config.get("use_inference_duration_monitor", False))


@dataclass
class InferenceDurationMonitorMetadata(KVConnectorMetadata):
    """Scheduler-to-worker metadata for a no-I/O monitor step."""

    preempted_req_ids: set[str] = field(default_factory=set)
    scheduled_reqs: int = 0
    new_reqs: int = 0
    new_reqs_with_computed_tokens: int = 0
    scheduled_tokens: int = 0
    total_num_computed_tokens: int = 0
    fake_hit: int = 0
    dp_rank: int = 0
    scheduler_iteration_id: int = -1
    forward_id: int = -1

    @property
    def should_collect_duration(self) -> bool:
        return self.new_reqs > 0 and self.scheduled_tokens > 0


@dataclass
class DurationStats:
    """Mergeable duration summary in milliseconds."""

    count: int = 0
    sum_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0

    def observe(self, value_ms: float) -> None:
        value_ms = float(value_ms)
        self.count += 1
        self.sum_ms += value_ms
        self.min_ms = min(self.min_ms, value_ms)
        self.max_ms = max(self.max_ms, value_ms)

    def aggregate(self, other: "DurationStats") -> None:
        if other.count == 0:
            return
        self.count += other.count
        self.sum_ms += other.sum_ms
        self.min_ms = min(self.min_ms, other.min_ms)
        self.max_ms = max(self.max_ms, other.max_ms)

    @property
    def avg_ms(self) -> float:
        return self.sum_ms / self.count if self.count else 0.0


@dataclass
class InferenceDurationMonitorWorkerMetadata(KVConnectorWorkerMetadata):
    """Per-forward timing data aggregated across one DP engine's workers."""

    duration_stats: dict[str, DurationStats] = field(default_factory=dict)
    worker_ranks: set[int] = field(default_factory=set)
    fake_hit: int = 0
    dp_rank: int = 0
    scheduler_iteration_id: int = -1
    forward_id: int = -1

    def aggregate(self, other: Any) -> Any:
        assert isinstance(other, InferenceDurationMonitorWorkerMetadata)
        assert self.dp_rank == other.dp_rank
        assert self.scheduler_iteration_id == other.scheduler_iteration_id
        assert self.forward_id == other.forward_id
        for name, other_stats in other.duration_stats.items():
            self.duration_stats.setdefault(name, DurationStats()).aggregate(other_stats)
        self.worker_ranks.update(other.worker_ranks)
        self.fake_hit = other.fake_hit
        return self


class UCMInferenceDurationMonitorConnector(KVConnectorBase_V1, SupportsHMA):
    """Measure forward and per-layer block windows without performing KV I/O.

    Each per-layer window starts immediately before one attention invocation
    and ends immediately before the next layer's attention invocation. It thus
    approximates a complete transformer-block compute window.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ) -> None:
        if getattr(vllm_config, "speculative_config", None) is not None:
            raise ValueError(
                "UCM inference duration monitoring does not support speculative "
                "decoding because its timing hooks do not isolate the target model "
                "forward. Disable speculative decoding before enabling the monitor."
            )
        if getattr(vllm_config.cache_config, "enable_prefix_caching", False):
            raise ValueError(
                "UCM inference duration monitoring requires HBM prefix caching to "
                "be disabled so computed tokens represent simulated external KV "
                "hits only. Disable prefix caching before enabling the monitor."
            )
        super().__init__(vllm_config, role, kv_cache_config)
        parallel_config = vllm_config.parallel_config
        self._dp_rank = int(getattr(parallel_config, "data_parallel_rank", 0))
        self._model_rank = int(getattr(parallel_config, "rank", 0))
        launch_config = Config(vllm_config.kv_transfer_config).get_config() or {}
        self._fake_hit_ratio = float(
            launch_config.get("inference_duration_monitor_fake_hit_ratio", 0.0)
        )
        self._current_fake_hit: int = 0
        self._scheduler_iteration_id: int = 0
        self._current_dp_rank: int = self._dp_rank
        self._current_scheduler_iteration_id: int = -1
        self._next_forward_id: int = 0
        self._current_forward_id: int = -1
        self._device: Optional[Device] = None
        self._collect_current_forward = False
        self._inference_start_time: Optional[float] = None
        self._active_attention_events: dict[str, Any] = {}
        self._pending_attention_events: list[tuple[str, Any]] = []
        self._duration_stats: dict[str, DurationStats] = {}
        self._pending_worker_metadata: Optional[
            InferenceDurationMonitorWorkerMetadata
        ] = None
        logger.info(
            "Init UCMInferenceDurationMonitorConnector (no KV I/O, "
            "fake_hit_ratio=%.2f).",
            self._fake_hit_ratio,
        )

    @classmethod
    def requires_piecewise_for_cudagraph(cls, extra_config: dict[str, Any]) -> bool:
        del extra_config
        return True

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        del kv_caches
        self._device = self._create_device()

    @staticmethod
    def _extract_layer_index(layer_name: str) -> Optional[int]:
        for part in layer_name.split("."):
            if part.isdigit():
                return int(part)
        return None

    def get_block_size(self) -> int:
        return self._vllm_config.cache_config.block_size

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        request_tokens = getattr(request, "num_tokens", None)
        if request_tokens is None:
            request_tokens = len(getattr(request, "all_token_ids", ()))
        total = max(int(request_tokens), 0)
        local_hit = max(int(num_computed_tokens), 0)
        fake_hit = min(int(total * self._fake_hit_ratio), max(total - local_hit, 0))
        if fake_hit <= 0:
            return 0, False
        return fake_hit, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        del request, blocks, num_external_tokens

    @staticmethod
    def _scheduled_request_ids(scheduler_output: SchedulerOutput) -> list[str]:
        request_ids = [
            request.req_id for request in scheduler_output.scheduled_new_reqs
        ]
        cached_reqs = scheduler_output.scheduled_cached_reqs
        if isinstance(cached_reqs, list):
            request_ids.extend(request.req_id for request in cached_reqs)
        else:
            request_ids.extend(getattr(cached_reqs, "req_ids", ()))
        return request_ids

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        scheduler_iteration_id = self._scheduler_iteration_id
        self._scheduler_iteration_id += 1
        scheduled_request_ids = list(
            getattr(scheduler_output, "num_scheduled_tokens", {}).keys()
        )
        if not scheduled_request_ids:
            scheduled_request_ids = self._scheduled_request_ids(scheduler_output)
        new_requests = scheduler_output.scheduled_new_reqs
        new_request_num_computed_tokens = [
            max(
                int(
                    getattr(
                        request,
                        "num_computed_tokens",
                        0,
                    )
                ),
                0,
            )
            for request in new_requests
        ]
        new_reqs_with_computed_tokens = sum(
            num_computed_tokens > 0
            for num_computed_tokens in new_request_num_computed_tokens
        )
        total_num_computed_tokens = sum(new_request_num_computed_tokens)
        scheduled_tokens = sum(
            int(tokens)
            for tokens in getattr(scheduler_output, "num_scheduled_tokens", {}).values()
        )
        forward_id = -1
        if scheduled_tokens > 0:
            forward_id = self._next_forward_id
            self._next_forward_id += 1
        logger.info(
            "Inference duration scheduler stats: dp_rank=%d, "
            "scheduler_iteration_id=%d, forward_id=%d, rank=%d, "
            "scheduled_reqs=%d, new_reqs=%d, "
            "scheduled_tokens=%d",
            self._dp_rank,
            scheduler_iteration_id,
            forward_id,
            self._model_rank,
            len(scheduled_request_ids),
            len(new_requests),
            scheduled_tokens,
        )
        return InferenceDurationMonitorMetadata(
            preempted_req_ids=scheduler_output.preempted_req_ids or set(),
            scheduled_reqs=len(scheduled_request_ids),
            new_reqs=len(new_requests),
            new_reqs_with_computed_tokens=new_reqs_with_computed_tokens,
            scheduled_tokens=scheduled_tokens,
            total_num_computed_tokens=total_num_computed_tokens,
            fake_hit=total_num_computed_tokens,
            dp_rank=self._dp_rank,
            scheduler_iteration_id=scheduler_iteration_id,
            forward_id=forward_id,
        )

    @staticmethod
    def _create_device() -> Device:
        device = create_device()
        if device is None:
            raise RuntimeError(
                "Unsupported device platform for inference duration monitoring."
            )
        return device

    def _get_device(self) -> Device:
        if self._device is None:
            self._device = self._create_device()
        return self._device

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        del forward_context, kwargs
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, InferenceDurationMonitorMetadata)
        self._collect_current_forward = metadata.should_collect_duration
        self._current_fake_hit = metadata.fake_hit
        self._current_dp_rank = metadata.dp_rank
        self._current_scheduler_iteration_id = metadata.scheduler_iteration_id
        self._current_forward_id = metadata.forward_id
        self._inference_start_time = None
        self._active_attention_events.clear()
        self._pending_attention_events.clear()
        self._duration_stats.clear()
        if not self._collect_current_forward:
            return
        self._get_device().synchronize()
        self._inference_start_time = time.perf_counter()

    def wait_for_layer_load(self, layer_name: str) -> None:
        if not self._collect_current_forward:
            return
        try:
            self._active_attention_events[layer_name] = (
                self._get_device().record_timing_event()
            )
        except Exception as error:
            logger.warning(
                "Failed to record attention start event for %s: %s",
                layer_name,
                error,
            )

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: Any,
        **kwargs: Any,
    ) -> None:
        del kv_layer, attn_metadata, kwargs
        if not self._collect_current_forward:
            return
        start_event = self._active_attention_events.pop(layer_name, None)
        if start_event is None:
            return
        self._pending_attention_events.append((layer_name, start_event))

    def _observe_duration(self, name: str, value_ms: float) -> None:
        self._duration_stats.setdefault(name, DurationStats()).observe(value_ms)

    def wait_for_save(self) -> None:
        if self._inference_start_time is None:
            self._collect_current_forward = False
            return
        device = self._get_device()

        block_end_event = None
        if self._pending_attention_events:
            try:
                block_end_event = device.record_timing_event()
            except Exception as error:
                logger.warning(
                    "Failed to record block end event: %s",
                    error,
                )

        device.synchronize()
        elapsed_ms = (time.perf_counter() - self._inference_start_time) * 1000
        self._inference_start_time = None
        self._collect_current_forward = False
        self._observe_duration("forward", elapsed_ms)

        if not self._pending_attention_events:
            logger.warning_once(
                "Inference duration monitor observed no attention hooks. "
                "Per-layer block-window timing is unavailable when the active model "
                "execution path bypasses KV connector layer hooks."
            )

        num_events = len(self._pending_attention_events)
        for i in range(num_events):
            layer_name, start_event = self._pending_attention_events[i]
            try:
                if i < num_events - 1:
                    _, next_start_event = self._pending_attention_events[i + 1]
                    block_ms = device.elapsed_time_ms(start_event, next_start_event)
                elif block_end_event is not None:
                    block_ms = device.elapsed_time_ms(start_event, block_end_event)
                else:
                    continue
            except Exception as error:
                logger.warning(
                    "Failed to read block duration for %s: %s",
                    layer_name,
                    error,
                )
                continue
            layer_idx = self._extract_layer_index(layer_name)
            scope_name = str(layer_idx) if layer_idx is not None else layer_name
            self._observe_duration(f"block_layer:{scope_name}", block_ms)

        self._active_attention_events.clear()
        self._pending_attention_events.clear()
        self._pending_worker_metadata = InferenceDurationMonitorWorkerMetadata(
            duration_stats=self._duration_stats,
            worker_ranks={self._model_rank},
            fake_hit=self._current_fake_hit,
            dp_rank=self._current_dp_rank,
            scheduler_iteration_id=self._current_scheduler_iteration_id,
            forward_id=self._current_forward_id,
        )
        self._duration_stats = {}

    def build_connector_worker_meta(
        self,
    ) -> Optional[InferenceDurationMonitorWorkerMetadata]:
        metadata = self._pending_worker_metadata
        self._pending_worker_metadata = None
        return metadata

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        metadata = getattr(connector_output, "kv_connector_worker_meta", None)
        if not isinstance(metadata, InferenceDurationMonitorWorkerMetadata):
            return
        for name in sorted(metadata.duration_stats):
            stats = metadata.duration_stats[name]
            if stats.count == 0:
                continue
            logger.info(
                "Inference duration aggregate: dp_rank=%d, "
                "scheduler_iteration_id=%d, forward_id=%d, workers=%d, "
                "scope=%s, fake_hit=%d, count=%d, avg_ms=%.3f, "
                "min_ms=%.3f, max_ms=%.3f",
                metadata.dp_rank,
                metadata.scheduler_iteration_id,
                metadata.forward_id,
                len(metadata.worker_ranks),
                name,
                metadata.fake_hit,
                stats.count,
                stats.avg_ms,
                stats.min_ms,
                stats.max_ms,
            )

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        del request, block_ids
        return False, None
