#
# MIT License
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

import threading
from copy import deepcopy
from typing import Any

from ucm.metrics_config import (
    MULTIPROC_CONSUMER,
    VLLM_CONNECTOR_CONSUMER,
    MetricDefinition,
    consumer_enabled,
    get_metric_definitions,
)
from ucm.shared.metrics import ucmmetrics

CONSUMERS = (MULTIPROC_CONSUMER, VLLM_CONNECTOR_CONSUMER)
_DISPATCHER: "MetricsDispatcher | None" = None
_DISPATCHER_LOCK = threading.Lock()


class MetricsDispatcher:
    def __init__(self, config: dict[str, Any] | None):
        self.config = config or {}
        self._definitions = get_metric_definitions(self.config)
        self._definitions_by_name = {
            definition.name: definition for definition in self._definitions
        }
        self._enabled = {
            consumer: consumer_enabled(self.config, consumer) for consumer in CONSUMERS
        }
        self._buffers = {consumer: self._empty_stats() for consumer in CONSUMERS}
        self._lock = threading.Lock()

    def drain_to_consumers(self) -> None:
        with self._lock:
            counter_stats, gauge_stats, histogram_stats = (
                ucmmetrics.get_all_stats_and_clear()
            )
            if not counter_stats and not gauge_stats and not histogram_stats:
                return

            for consumer in CONSUMERS:
                if not self._enabled[consumer]:
                    continue
                self._merge_stats(
                    self._buffers[consumer],
                    counter_stats,
                    gauge_stats,
                    histogram_stats,
                    consumer,
                )

    def get_stats_and_clear(self, consumer: str):
        if consumer not in CONSUMERS:
            raise ValueError(f"Unsupported metrics consumer: {consumer}")
        with self._lock:
            snapshot = deepcopy(self._buffers[consumer])
            self._buffers[consumer] = self._empty_stats()
        return snapshot

    def _merge_stats(
        self,
        buffer,
        counter_stats: dict[str, float],
        gauge_stats: dict[str, float],
        histogram_stats: dict[str, Any],
        consumer: str,
    ) -> None:
        counters, gauges, histograms = buffer
        for metric_name, value in counter_stats.items():
            definition = self._consumer_definition(metric_name, "counter", consumer)
            if definition is not None:
                counters[metric_name] = counters.get(metric_name, 0.0) + float(value)

        for metric_name, value in gauge_stats.items():
            definition = self._consumer_definition(metric_name, "gauge", consumer)
            if definition is not None:
                gauges[metric_name] = float(value)

        for metric_name, value in histogram_stats.items():
            definition = self._consumer_definition(metric_name, "histogram", consumer)
            if definition is not None:
                self._merge_histogram(histograms, metric_name, value)

    def _consumer_definition(
        self, metric_name: str, metric_type: str, consumer: str
    ) -> MetricDefinition | None:
        definition = self._definitions_by_name.get(metric_name)
        if definition is None or definition.metric_type != metric_type:
            return None
        if (
            consumer == VLLM_CONNECTOR_CONSUMER
            and not definition.vllm_connector_enabled
        ):
            return None
        return definition

    def _merge_histogram(self, histograms: dict[str, Any], metric_name: str, value):
        bucket_counts, sum_delta = self._histogram_tuple(value)
        current_counts, current_sum = histograms.get(
            metric_name, ([0] * len(bucket_counts), 0.0)
        )
        if len(current_counts) < len(bucket_counts):
            current_counts.extend([0] * (len(bucket_counts) - len(current_counts)))
        for index, count in enumerate(bucket_counts):
            current_counts[index] += int(count)
        histograms[metric_name] = (current_counts, current_sum + float(sum_delta))

    def _histogram_tuple(self, value):
        if isinstance(value, dict):
            return list(value.get("bucket_counts", [])), float(value.get("sum", 0.0))
        if isinstance(value, (tuple, list)) and len(value) == 2:
            bucket_counts, sum_delta = value
            return list(bucket_counts), float(sum_delta)
        return list(getattr(value, "bucketCounts", [])), float(
            getattr(value, "sum", 0.0)
        )

    def _empty_stats(self):
        return ({}, {}, {})


def get_metrics_dispatcher(config: dict[str, Any] | None) -> MetricsDispatcher:
    global _DISPATCHER
    with _DISPATCHER_LOCK:
        if _DISPATCHER is None:
            _DISPATCHER = MetricsDispatcher(config)
        return _DISPATCHER
