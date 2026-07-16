from __future__ import annotations

import math
from collections import defaultdict
from typing import Mapping

from .config import metric_specs
from .parser import MetricSample
from .promql import PromqlEvaluator
from .query import (
    QueryRow,
    _aggregate,
    _group_key,
    _matches_tag_filters,
    _parse_le,
    _quantile_name,
    _source_name,
    histogram_quantile,
)


class SnapshotQueryEngine:
    def __init__(self, samples: list[MetricSample]):
        self.store = SnapshotStore(samples)

    def query_config(
        self,
        config: dict,
        tag_filters: Mapping[str, str] | None = None,
    ) -> list[QueryRow]:
        rows: list[QueryRow] = []
        for spec in metric_specs(config):
            metric_type = spec.get("type", "gauge")
            if metric_type == "promql" or "expr" in spec:
                rows.extend(self._promql_rows(spec, tag_filters))
            elif metric_type == "histogram":
                rows.extend(self._histogram_rows(spec, tag_filters))
            else:
                rows.extend(self._current_rows(spec, tag_filters))
        return rows

    def _promql_rows(
        self,
        spec: dict,
        tag_filters: Mapping[str, str] | None,
    ) -> list[QueryRow]:
        aggregate = spec["aggregate"]
        scale = float(spec.get("scale", 1.0))
        value_name = spec.get("value", "value")
        evaluator = PromqlEvaluator(
            self.store,
            0,
            0,
            tag_filters,
            instant_counters=True,
        )
        grouped: dict[tuple[tuple[str, str], ...], list[float]] = defaultdict(list)
        for point in evaluator.evaluate(spec["expr"]).values():
            value = point.value * scale
            if math.isfinite(value):
                grouped[_group_key(point.labels, spec.get("group_by", []))].append(
                    value
                )
        return [
            QueryRow(
                spec["name"],
                dict(group),
                {value_name: _aggregate(values, aggregate)},
                spec.get("unit", ""),
            )
            for group, values in grouped.items()
        ]

    def _current_rows(
        self,
        spec: dict,
        tag_filters: Mapping[str, str] | None,
    ) -> list[QueryRow]:
        aggregate = spec["aggregate"]
        scale = float(spec.get("scale", 1.0))
        grouped: dict[tuple[tuple[str, str], ...], list[float]] = defaultdict(list)
        for series in self.store.list_series(_source_name(spec)):
            if not _matches_tag_filters(series["labels"], tag_filters):
                continue
            samples = self.store.samples_for_series(int(series["id"]), 0, 0)
            if not samples or not math.isfinite(samples[-1][1]):
                continue
            grouped[_group_key(series["labels"], spec.get("group_by", []))].append(
                samples[-1][1] * scale
            )
        value_name = spec.get("op", "value")
        return [
            QueryRow(
                spec["name"],
                dict(group),
                {value_name: _aggregate(values, aggregate)},
                spec.get("unit", ""),
            )
            for group, values in grouped.items()
        ]

    def _histogram_rows(
        self,
        spec: dict,
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
            if not _matches_tag_filters(labels, tag_filters) or "le" not in labels:
                continue
            samples = self.store.samples_for_series(int(series["id"]), 0, 0)
            if not samples or not math.isfinite(samples[-1][1]):
                continue
            le = _parse_le(labels.pop("le"))
            bucket_values[_group_key(labels, group_by)][le] += samples[-1][1]

        for suffix, target in (("_sum", sum_values), ("_count", count_values)):
            for series in self.store.list_series(f"{source_name}{suffix}"):
                if not _matches_tag_filters(series["labels"], tag_filters):
                    continue
                samples = self.store.samples_for_series(int(series["id"]), 0, 0)
                if not samples or not math.isfinite(samples[-1][1]):
                    continue
                target[_group_key(series["labels"], group_by)] += samples[-1][1]

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


class SnapshotStore:
    def __init__(self, samples: list[MetricSample]):
        self.series: list[dict[str, object]] = []
        self.values: dict[int, float] = {}
        series_ids: dict[tuple[str, tuple[tuple[str, str], ...]], int] = {}
        for sample in samples:
            key = (sample.name, tuple(sorted(sample.labels.items())))
            series_id = series_ids.get(key)
            if series_id is None:
                series_id = len(self.series) + 1
                series_ids[key] = series_id
                self.series.append(
                    {
                        "id": series_id,
                        "name": sample.name,
                        "labels": dict(sample.labels),
                    }
                )
            self.values[series_id] = sample.value

    def list_series(self, name: str) -> list[dict[str, object]]:
        return [series for series in self.series if series["name"] == name]

    def samples_for_series(
        self,
        series_id: int,
        _start_ms: int,
        _end_ms: int,
    ) -> list[tuple[int, float]]:
        if series_id not in self.values:
            return []
        return [(0, self.values[series_id])]
