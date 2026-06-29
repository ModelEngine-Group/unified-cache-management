import copy
import math
from dataclasses import dataclass, field
from typing import Any

from vllm.config import VllmConfig

from ucm.logger import init_logger
from ucm.metrics_config import (
    MetricDefinition,
    get_vllm_connector_metric_definitions,
    load_launch_metrics_config,
)

try:
    from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
        KVConnectorPromMetrics,
        KVConnectorStats,
        PromMetric,
        PromMetricT,
    )

    UCM_HAS_PROM_METRICS = True
except ImportError:
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
            KVConnectorStats,
        )
    except ImportError:

        @dataclass
        class KVConnectorStats:
            data: dict[str, Any] = field(default_factory=dict)

            def is_empty(self) -> bool:
                return not self.data

    KVConnectorPromMetrics = object
    PromMetric = Any
    PromMetricT = Any
    UCM_HAS_PROM_METRICS = False

logger = init_logger(__name__)


@dataclass
class UCMConnectorStats(KVConnectorStats):
    worker_rank: int | str | None = None

    def __post_init__(self):
        if self.worker_rank is not None:
            self.worker_rank = str(self.worker_rank)
        if not self.data:
            self.reset()

    @classmethod
    def from_ucm_snapshot(
        cls,
        counter_stats: dict[str, int | float],
        gauge_stats: dict[str, int | float],
        histogram_stats: dict[str, Any],
        worker_rank: int | str | None,
        metric_definitions: list[MetricDefinition],
    ) -> "UCMConnectorStats":
        stats = cls(worker_rank=worker_rank)
        rank = stats._rank_key()
        definitions_by_name = {
            definition.name: definition
            for definition in metric_definitions
            if definition.vllm_connector_enabled
        }

        for metric_name, value in counter_stats.items():
            definition = definitions_by_name.get(metric_name)
            if definition is None or definition.metric_type != "counter":
                continue
            value_float = stats._finite(value)
            if value_float is None:
                continue
            stats.data["counters_by_rank"].setdefault(rank, {})[
                metric_name
            ] = value_float

        for metric_name, value in gauge_stats.items():
            definition = definitions_by_name.get(metric_name)
            if definition is None or definition.metric_type != "gauge":
                continue
            value_float = stats._finite(value)
            if value_float is None:
                continue
            stats.data["gauges_by_rank"].setdefault(rank, {})[metric_name] = value_float

        for metric_name, value in histogram_stats.items():
            definition = definitions_by_name.get(metric_name)
            if definition is None or definition.metric_type != "histogram":
                continue
            histogram = _histogram_snapshot(value)
            if histogram is None:
                continue
            stats.data["histograms_by_rank"].setdefault(rank, {})[
                metric_name
            ] = histogram

        return stats

    def reset(self):
        self.data: dict[str, Any] = {
            "counters_by_rank": {},
            "gauges_by_rank": {},
            "histograms_by_rank": {},
        }

    def record(
        self,
        observations: dict[str, int | float],
        metric_types: dict[str, str] | None = None,
        worker_rank: int | str | None = None,
    ) -> None:
        rank = self._rank_key(worker_rank)
        metric_types = metric_types or {}
        for metric_name, value in observations.items():
            value_float = self._finite(value)
            if value_float is None:
                continue
            metric_type = metric_types.get(metric_name, "histogram")
            if metric_type == "counter":
                self.data["counters_by_rank"].setdefault(rank, {})[
                    metric_name
                ] = value_float
            elif metric_type == "gauge":
                self.data["gauges_by_rank"].setdefault(rank, {})[
                    metric_name
                ] = value_float
            else:
                rank_data = self.data["histograms_by_rank"].setdefault(rank, {})
                rank_data.setdefault(metric_name, []).append(value_float)

    def clone_and_reset(self) -> "UCMConnectorStats":
        snapshot = copy.deepcopy(self.data)
        self.reset()
        return UCMConnectorStats(data=snapshot, worker_rank=self.worker_rank)

    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
        if other.is_empty():
            return self

        for section in ("counters_by_rank", "gauges_by_rank"):
            target = self.data.setdefault(section, {})
            for rank, rank_data in other.data.get(section, {}).items():
                target.setdefault(str(rank), {}).update(rank_data)

        histograms_by_rank = self.data.setdefault("histograms_by_rank", {})
        other_histograms = other.data.get("histograms_by_rank", {})
        for rank, rank_data in other_histograms.items():
            accumulator = histograms_by_rank.setdefault(str(rank), {})
            for metric_name, value in rank_data.items():
                _merge_histogram_value(accumulator, metric_name, value)
        return self

    def reduce(self) -> dict[str, int | float]:
        return {}

    def is_empty(self) -> bool:
        for section in (
            "counters_by_rank",
            "gauges_by_rank",
            "histograms_by_rank",
        ):
            for rank_data in self.data.get(section, {}).values():
                if rank_data:
                    return False
        return True

    def _rank_key(self, worker_rank: int | str | None = None) -> str:
        if worker_rank is not None:
            return str(worker_rank)
        if self.worker_rank is not None:
            return str(self.worker_rank)
        return "unknown"

    def _finite(self, value: int | float) -> float | None:
        value_float = float(value)
        if not math.isfinite(value_float):
            return None
        return value_float


if UCM_HAS_PROM_METRICS:

    class UCMPromMetrics(KVConnectorPromMetrics):
        def __init__(
            self,
            vllm_config: VllmConfig,
            metric_types: dict[type[PromMetric], type[PromMetricT]],
            labelnames: list[str],
            per_engine_labelvalues: dict[int, list[object]],
        ):
            super().__init__(
                vllm_config, metric_types, labelnames, per_engine_labelvalues
            )
            config = _metrics_config_from_vllm_config(vllm_config)
            definitions = get_vllm_connector_metric_definitions(config)
            self._definitions = {
                definition.name: definition for definition in definitions
            }
            self._metrics_by_name: dict[str, PromMetricT] = {}
            self._labeled_metrics: dict[tuple[int, str, str], PromMetricT] = {}
            counts = {"counter": 0, "gauge": 0, "histogram": 0}

            for definition in definitions:
                self._metrics_by_name[definition.name] = self._create_metric(
                    definition,
                    labelnames + ["worker_rank"],
                )
                counts[definition.metric_type] += 1
            logger.info(
                f"UCM metrics vllm_connector path enabled: "
                f"total={len(definitions)}, counters={counts['counter']}, "
                f"gauges={counts['gauge']}, histograms={counts['histogram']}, "
                f"labels={labelnames + ['worker_rank']}"
            )

        def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
            self._observe_counters(transfer_stats_data, engine_idx)
            self._observe_gauges(transfer_stats_data, engine_idx)
            self._observe_histograms(transfer_stats_data, engine_idx)

        def _create_metric(
            self,
            definition: MetricDefinition,
            labelnames: list[str],
        ) -> PromMetricT:
            metric_kwargs: dict[str, Any] = {
                "name": definition.vllm_connector_name,
                "documentation": definition.documentation,
                "labelnames": labelnames,
            }
            if definition.metric_type == "histogram":
                metric_kwargs["buckets"] = list(definition.vllm_connector_buckets)
                return self._histogram_cls(**metric_kwargs)
            if definition.metric_type == "counter":
                return self._counter_cls(**metric_kwargs)
            return self._gauge_cls(**metric_kwargs)

        def _observe_counters(
            self, transfer_stats_data: dict[str, Any], engine_idx: int
        ) -> None:
            for worker_rank, rank_data in transfer_stats_data.get(
                "counters_by_rank", {}
            ).items():
                for metric_name, value in rank_data.items():
                    definition = self._definition(metric_name, "counter")
                    if definition is None:
                        continue
                    value_float = _finite(value)
                    if value_float is None or value_float < 0:
                        continue
                    self._metric(engine_idx, worker_rank, metric_name).inc(value_float)

        def _observe_gauges(
            self, transfer_stats_data: dict[str, Any], engine_idx: int
        ) -> None:
            for worker_rank, rank_data in transfer_stats_data.get(
                "gauges_by_rank", {}
            ).items():
                for metric_name, value in rank_data.items():
                    definition = self._definition(metric_name, "gauge")
                    if definition is None:
                        continue
                    value_float = _finite(value)
                    if value_float is None:
                        continue
                    self._metric(engine_idx, worker_rank, metric_name).set(value_float)

        def _observe_histograms(
            self, transfer_stats_data: dict[str, Any], engine_idx: int
        ) -> None:
            for worker_rank, rank_data in transfer_stats_data.get(
                "histograms_by_rank", {}
            ).items():
                for metric_name, value in rank_data.items():
                    definition = self._definition(metric_name, "histogram")
                    if definition is None:
                        continue
                    histogram = self._metric(engine_idx, worker_rank, metric_name)
                    if isinstance(value, list):
                        for observation in value:
                            value_float = _finite(observation)
                            if value_float is not None:
                                histogram.observe(
                                    value_float * definition.vllm_connector_value_scale
                                )
                    elif isinstance(value, dict):
                        self._update_histogram_snapshot(histogram, value, definition)

        def _update_histogram_snapshot(
            self,
            histogram: PromMetricT,
            value: dict[str, Any],
            definition: MetricDefinition,
        ) -> None:
            bucket_counts = list(value.get("bucket_counts", []))
            sum_delta = (
                float(value.get("sum", 0.0)) * definition.vllm_connector_value_scale
            )
            buckets = getattr(histogram, "_buckets", None)
            metric_sum = getattr(histogram, "_sum", None)
            if (
                buckets is None
                or metric_sum is None
                or len(bucket_counts) != len(buckets)
            ):
                return
            for bucket, count in zip(buckets, bucket_counts):
                if count:
                    bucket.inc(count)
            if sum_delta:
                metric_sum.inc(sum_delta)

        def _definition(
            self, metric_name: str, metric_type: str
        ) -> MetricDefinition | None:
            definition = self._definitions.get(metric_name)
            if definition is None or definition.metric_type != metric_type:
                return None
            if metric_name not in self._metrics_by_name:
                return None
            return definition

        def _metric(
            self, engine_idx: int, worker_rank: int | str, metric_name: str
        ) -> PromMetricT:
            worker_rank = str(worker_rank)
            key = (engine_idx, worker_rank, metric_name)
            if key not in self._labeled_metrics:
                self._labeled_metrics[key] = self._metrics_by_name[metric_name].labels(
                    *(self._engine_labelvalues[engine_idx] + [worker_rank])
                )
            return self._labeled_metrics[key]

        @property
        def _engine_labelvalues(self) -> dict[int, list[object]]:
            if hasattr(self, "per_engine_labelvalues"):
                return self.per_engine_labelvalues
            return self._per_engine_labelvalues

else:

    class UCMPromMetrics:
        def __init__(self, *args: Any, **kwargs: Any):
            raise RuntimeError("vLLM connector Prometheus metrics are unavailable.")


def _metrics_config_from_vllm_config(vllm_config: VllmConfig) -> dict[str, Any]:
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    launch_config = getattr(kv_transfer_config, "launch_config", None)
    if launch_config is not None:
        return load_launch_metrics_config(launch_config)
    try:
        from ucm.utils import Config

        return load_launch_metrics_config(Config(kv_transfer_config).get_config())
    except Exception:
        return {}


def _finite(value: Any) -> float | None:
    value_float = float(value)
    if not math.isfinite(value_float):
        return None
    return value_float


def _histogram_snapshot(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        bucket_counts = value.get("bucket_counts", [])
        sum_delta = value.get("sum", 0.0)
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        bucket_counts, sum_delta = value
    else:
        bucket_counts = getattr(value, "bucketCounts", None)
        sum_delta = getattr(value, "sum", None)
    if bucket_counts is None or sum_delta is None:
        return None
    return {
        "bucket_counts": [int(count) for count in bucket_counts],
        "sum": float(sum_delta),
    }


def _merge_histogram_value(
    accumulator: dict[str, Any], metric_name: str, value: Any
) -> None:
    if isinstance(value, list):
        accumulator.setdefault(metric_name, []).extend(value)
        return
    if not isinstance(value, dict):
        return
    target = accumulator.setdefault(
        metric_name,
        {"bucket_counts": [0] * len(value.get("bucket_counts", [])), "sum": 0.0},
    )
    if isinstance(target, list):
        target.append(float(value.get("sum", 0.0)))
        return
    target_counts = target.setdefault(
        "bucket_counts", [0] * len(value.get("bucket_counts", []))
    )
    value_counts = list(value.get("bucket_counts", []))
    if len(target_counts) < len(value_counts):
        target_counts.extend([0] * (len(value_counts) - len(target_counts)))
    for index, count in enumerate(value_counts):
        target_counts[index] += int(count)
    target["sum"] = float(target.get("sum", 0.0)) + float(value.get("sum", 0.0))
