from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping

from .config import metric_specs
from .promql import PromqlEvaluator
from .storage import MetricsStore


@dataclass(frozen=True)
class QueryRow:
    metric: str
    group: dict[str, str]
    values: dict[str, float]
    unit: str = ""
    start_ms: int | None = None
    end_ms: int | None = None


class QueryEngine:
    Row = QueryRow

    def __init__(self, store: MetricsStore):
        self.store = store

    def query_config(
        self,
        config: dict,
        window_seconds: float,
        end_ms: int | None = None,
        start_ms: int | None = None,
        aggr_by_seconds: float | None = None,
        limit: int | None = None,
        tag_filters: Mapping[str, str] | None = None,
    ) -> list[QueryRow]:
        window_ms = int(window_seconds * 1000)
        if start_ms is not None:
            end_ms = start_ms + window_ms
        elif end_ms is None:
            end_ms = self.store.latest_ts_ms() or int(time.time() * 1000)
            start_ms = end_ms - window_ms
        else:
            start_ms = end_ms - window_ms
        if aggr_by_seconds is not None:
            rows = self._query_bucketed(
                config, start_ms, end_ms, aggr_by_seconds, tag_filters
            )
            if limit is not None:
                rows = rows[:limit]
            return rows
        rows = self._query_window(config, start_ms, end_ms, tag_filters)
        rows.sort(key=lambda row: _primary_value(row), reverse=True)
        if limit is not None:
            rows = rows[:limit]
        return rows

    def _query_bucketed(
        self,
        config: dict,
        start_ms: int,
        end_ms: int,
        aggr_by_seconds: float,
        tag_filters: Mapping[str, str] | None,
    ) -> list[QueryRow]:
        bucket_ms = int(aggr_by_seconds * 1000)
        if bucket_ms <= 0:
            raise ValueError("aggr_by_seconds must be positive")
        rows: list[QueryRow] = []
        bucket_start = start_ms
        while bucket_start < end_ms:
            bucket_end = min(bucket_start + bucket_ms, end_ms)
            bucket_rows = self._query_window(
                config, bucket_start, bucket_end, tag_filters
            )
            bucket_rows.sort(key=lambda row: (row.metric, tuple(row.group.items())))
            rows.extend(
                QueryRow(
                    metric=row.metric,
                    group=row.group,
                    values=row.values,
                    unit=row.unit,
                    start_ms=bucket_start,
                    end_ms=bucket_end,
                )
                for row in bucket_rows
            )
            bucket_start = bucket_end
        return rows

    def _query_window(
        self,
        config: dict,
        start_ms: int,
        end_ms: int,
        tag_filters: Mapping[str, str] | None,
    ) -> list[QueryRow]:
        rows: list[QueryRow] = []
        for spec in metric_specs(config):
            metric_type = spec.get("type", "gauge")
            if metric_type == "promql" or "expr" in spec:
                rows.extend(self._promql_rows(spec, start_ms, end_ms, tag_filters))
            elif metric_type == "counter":
                rows.extend(self._counter_rows(spec, start_ms, end_ms, tag_filters))
            elif metric_type == "histogram":
                rows.extend(self._histogram_rows(spec, start_ms, end_ms, tag_filters))
            else:
                rows.extend(self._gauge_rows(spec, start_ms, end_ms, tag_filters))
        return rows

    def _promql_rows(
        self,
        spec: dict,
        start_ms: int,
        end_ms: int,
        tag_filters: Mapping[str, str] | None,
    ) -> list[QueryRow]:
        aggregate = spec["aggregate"]
        scale = float(spec.get("scale", 1.0))
        value_name = spec.get("value", "value")
        evaluator = PromqlEvaluator(self.store, start_ms, end_ms, tag_filters)
        grouped: dict[tuple[tuple[str, str], ...], list[float]] = defaultdict(list)
        for point in evaluator.evaluate(spec["expr"]).values():
            value = point.value * scale
            if not math.isfinite(value):
                continue
            grouped[_group_key(point.labels, spec.get("group_by", []))].append(value)
        return [
            QueryRow(
                spec["name"],
                dict(group),
                {value_name: _aggregate(values, aggregate)},
                spec.get("unit", ""),
            )
            for group, values in grouped.items()
        ]

    def _counter_rows(
        self,
        spec: dict,
        start_ms: int,
        end_ms: int,
        tag_filters: Mapping[str, str] | None,
    ) -> list[QueryRow]:
        op = spec.get("op", "rate")
        aggregate = spec["aggregate"]
        scale = float(spec.get("scale", 1.0))
        grouped: dict[tuple[tuple[str, str], ...], list[float]] = defaultdict(list)
        for series in self.store.list_series(_source_name(spec)):
            if not _matches_tag_filters(series["labels"], tag_filters):
                continue
            samples = self.store.samples_for_series(int(series["id"]), start_ms, end_ms)
            delta = _counter_delta(samples)
            if delta is None:
                continue
            elapsed = max((samples[-1][0] - samples[0][0]) / 1000.0, 0.001)
            value = delta / elapsed if op == "rate" else delta
            grouped[_group_key(series["labels"], spec.get("group_by", []))].append(
                value * scale
            )
        key_name = "rate" if op == "rate" else "increase"
        return [
            QueryRow(
                spec["name"],
                dict(group),
                {key_name: _aggregate(values, aggregate)},
                spec.get("unit", ""),
            )
            for group, values in grouped.items()
        ]

    def _gauge_rows(
        self,
        spec: dict,
        start_ms: int,
        end_ms: int,
        tag_filters: Mapping[str, str] | None,
    ) -> list[QueryRow]:
        aggregate = spec["aggregate"]
        scale = float(spec.get("scale", 1.0))
        grouped: dict[tuple[tuple[str, str], ...], list[float]] = defaultdict(list)
        for series in self.store.list_series(_source_name(spec)):
            if not _matches_tag_filters(series["labels"], tag_filters):
                continue
            samples = self.store.samples_for_series(int(series["id"]), start_ms, end_ms)
            if not samples:
                continue
            grouped[_group_key(series["labels"], spec.get("group_by", []))].append(
                samples[-1][1] * scale
            )
        return [
            QueryRow(
                spec["name"],
                dict(group),
                {spec.get("op", "last"): _aggregate(values, aggregate)},
                spec.get("unit", ""),
            )
            for group, values in grouped.items()
        ]

    def _histogram_rows(
        self,
        spec: dict,
        start_ms: int,
        end_ms: int,
        tag_filters: Mapping[str, str] | None,
    ) -> list[QueryRow]:
        name = spec["name"]
        source_name = _source_name(spec)
        group_by = spec.get("group_by", [])
        scale = float(spec.get("scale", 1.0))
        bucket_values: dict[tuple[tuple[str, str], ...], dict[float, float]] = (
            defaultdict(lambda: defaultdict(float))
        )
        sum_values: dict[tuple[tuple[str, str], ...], float] = defaultdict(float)
        count_values: dict[tuple[tuple[str, str], ...], float] = defaultdict(float)

        for series in self.store.list_series(f"{source_name}_bucket"):
            labels = dict(series["labels"])
            if not _matches_tag_filters(labels, tag_filters):
                continue
            if "le" not in labels:
                continue
            le = _parse_le(labels["le"])
            samples = self.store.samples_for_series(int(series["id"]), start_ms, end_ms)
            delta = _counter_delta(samples)
            if delta is None:
                continue
            labels.pop("le", None)
            bucket_values[_group_key(labels, group_by)][le] += delta

        for suffix, target in (("_sum", sum_values), ("_count", count_values)):
            for series in self.store.list_series(f"{source_name}{suffix}"):
                if not _matches_tag_filters(series["labels"], tag_filters):
                    continue
                samples = self.store.samples_for_series(
                    int(series["id"]), start_ms, end_ms
                )
                delta = _counter_delta(samples)
                if delta is None:
                    continue
                target[_group_key(series["labels"], group_by)] += delta

        rows: list[QueryRow] = []
        for group, buckets in bucket_values.items():
            values: dict[str, float] = {}
            count = count_values.get(group) or buckets.get(math.inf, 0.0)
            for quantile in spec.get("quantiles", []):
                value = histogram_quantile(float(quantile), buckets, count)
                if value is not None:
                    values[_quantile_name(float(quantile))] = value * scale
            if spec.get("avg", False) and count > 0:
                values["avg"] = (sum_values.get(group, 0.0) / count) * scale
            if values:
                rows.append(QueryRow(name, dict(group), values, spec.get("unit", "")))
        return rows


def histogram_quantile(
    quantile: float, cumulative_buckets: dict[float, float], count: float | None = None
) -> float | None:
    if not cumulative_buckets:
        return None
    buckets = sorted(cumulative_buckets.items(), key=lambda item: item[0])
    total = float(count if count is not None else cumulative_buckets.get(math.inf, 0.0))
    if total <= 0:
        return None
    rank = quantile * total
    previous_bound: float | None = None
    previous_count = 0.0
    for upper_bound, cumulative_count in buckets:
        if cumulative_count < rank:
            previous_bound = upper_bound
            previous_count = cumulative_count
            continue
        if math.isinf(upper_bound):
            return previous_bound
        if previous_bound is None:
            lower_bound = 0.0 if upper_bound > 0 else upper_bound
        else:
            lower_bound = previous_bound
        bucket_count = cumulative_count - previous_count
        if bucket_count <= 0:
            return upper_bound
        return lower_bound + (upper_bound - lower_bound) * (
            (rank - previous_count) / bucket_count
        )
    last_finite = [bound for bound, _ in buckets if not math.isinf(bound)]
    return last_finite[-1] if last_finite else None


def _counter_delta(samples: list[tuple[int, float]]) -> float | None:
    if len(samples) < 2:
        return None
    total = 0.0
    previous = samples[0][1]
    for _, value in samples[1:]:
        if value >= previous:
            total += value - previous
        else:
            total += max(value, 0.0)
        previous = value
    return total


def _group_key(labels: object, group_by: Iterable[str]) -> tuple[tuple[str, str], ...]:
    label_map = labels if isinstance(labels, dict) else {}
    return tuple((name, str(label_map.get(name, ""))) for name in group_by)


def _matches_tag_filters(labels: object, tag_filters: Mapping[str, str] | None) -> bool:
    if not tag_filters:
        return True
    label_map = labels if isinstance(labels, dict) else {}
    return all(
        str(label_map.get(name, "")) == value for name, value in tag_filters.items()
    )


def _source_name(spec: dict) -> str:
    return str(spec.get("source", spec["name"]))


def _parse_le(value: str) -> float:
    if value == "+Inf":
        return math.inf
    return float(value)


def _quantile_name(quantile: float) -> str:
    percentile = quantile * 100
    if abs(percentile - round(percentile)) < 1e-9:
        return f"p{int(round(percentile))}"
    return f"p{percentile:g}"


def _aggregate(values: list[float], aggregate: str) -> float:
    if aggregate == "sum":
        return sum(values)
    if aggregate == "avg":
        return sum(values) / len(values)
    if aggregate == "max":
        return max(values)
    raise ValueError(f"Unsupported aggregate: {aggregate}")


def _primary_value(row: QueryRow) -> float:
    for key in ("p99", "p95", "p90", "p50", "avg", "rate", "increase", "last"):
        if key in row.values:
            return row.values[key]
    return next(iter(row.values.values()), 0.0)
