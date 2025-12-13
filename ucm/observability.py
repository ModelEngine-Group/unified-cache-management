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
# Be impressed by https://github.com/LMCache/LMCache/blob/dev/lmcache/observability.py

import os
import threading
import time
from dataclasses import dataclass
from typing import Any, List

import prometheus_client
import yaml

from ucm.logger import init_logger
from ucm.shared.metrics import ucmmonitor

logger = init_logger(__name__)


@dataclass
class UCMEngineMetadata:
    """Metadata for UCM engine"""

    model_name: str
    worker_id: str


class PrometheusLogger:
    _gauge_cls = prometheus_client.Gauge
    _counter_cls = prometheus_client.Counter
    _histogram_cls = prometheus_client.Histogram

    def __init__(self, metadata: UCMEngineMetadata, config: dict[str, Any]):
        # Ensure PROMETHEUS_MULTIPROC_DIR is set before any metric registration
        prometheus_config = config.get("prometheus", {})
        multiproc_dir = prometheus_config.get("multiproc_dir", "/vllm-workspace")
        if "PROMETHEUS_MULTIPROC_DIR" not in os.environ:
            os.environ["PROMETHEUS_MULTIPROC_DIR"] = multiproc_dir
            if not os.path.exists(multiproc_dir):
                os.makedirs(multiproc_dir, exist_ok=True)

        self.metadata = metadata
        self.config = config
        self.labels = self._metadata_to_labels(metadata)
        labelnames = list(self.labels.keys())

        # Initialize metrics based on configuration
        self._init_metrics_from_config(labelnames, prometheus_config)

    def _init_metrics_from_config(
        self, labelnames: List[str], prometheus_config: dict[str, Any]
    ):
        """Initialize metrics based on configuration"""
        enabled = prometheus_config.get("enabled_metrics", {})

        # Get metric name prefix from config (e.g., "ucm:")
        # If not specified, use empty string
        metric_prefix = prometheus_config.get("metric_prefix", "ucm:")

        # Store metric mapping: metric_name -> (metric_type, attribute_name, stats_field_name)
        # This mapping will be used in log_prometheus to dynamically log metrics
        self.metric_mappings: dict[str, dict[str, str]] = {}

        # Initialize counters
        if enabled.get("counters", True):
            counters = prometheus_config.get("counters", [])
            for counter_cfg in counters:
                name = counter_cfg.get("name")
                doc = counter_cfg.get("documentation", "")
                # Prometheus metric name with prefix
                prometheus_name = f"{metric_prefix}{name}" if metric_prefix else name
                # Internal attribute name for storing the metric object
                attr_name = f"counter_{name}"

                if not hasattr(self, attr_name):
                    setattr(
                        self,
                        attr_name,
                        self._counter_cls(
                            name=prometheus_name,
                            documentation=doc,
                            labelnames=labelnames,
                        ),
                    )
                    # Store mapping for dynamic logging
                    self.metric_mappings[name] = {
                        "type": "counter",
                        "attr": attr_name,
                    }

        # Initialize gauges
        if enabled.get("gauges", True):
            gauges = prometheus_config.get("gauges", [])
            for gauge_cfg in gauges:
                name = gauge_cfg.get("name")
                doc = gauge_cfg.get("documentation", "")
                multiprocess_mode = gauge_cfg.get("multiprocess_mode", "live")
                # Prometheus metric name with prefix
                prometheus_name = f"{metric_prefix}{name}" if metric_prefix else name
                # Internal attribute name
                attr_name = f"gauge_{name}"

                if not hasattr(self, attr_name):
                    setattr(
                        self,
                        attr_name,
                        self._gauge_cls(
                            name=prometheus_name,
                            documentation=doc,
                            labelnames=labelnames,
                            multiprocess_mode=multiprocess_mode,
                        ),
                    )
                    # Store mapping for dynamic logging
                    self.metric_mappings[name] = {
                        "type": "gauge",
                        "attr": attr_name,
                    }

        # Initialize histograms
        if enabled.get("histograms", True):
            histograms = prometheus_config.get("histograms", [])
            for hist_cfg in histograms:
                name = hist_cfg.get("name")
                doc = hist_cfg.get("documentation", "")
                buckets = hist_cfg.get("buckets", [])
                # Prometheus metric name with prefix
                prometheus_name = f"{metric_prefix}{name}" if metric_prefix else name
                # Internal attribute name
                attr_name = f"histogram_{name}"

                if not hasattr(self, attr_name):
                    setattr(
                        self,
                        attr_name,
                        self._histogram_cls(
                            name=prometheus_name,
                            documentation=doc,
                            labelnames=labelnames,
                            buckets=buckets,
                        ),
                    )
                    # Store mapping for dynamic logging
                    self.metric_mappings[name] = {
                        "type": "histogram",
                        "attr": attr_name,
                    }

    def _set_gauge(self, gauge, data: List) -> None:
        # Convenience function for logging to gauge.
        if not data:
            return
        gauge.labels(**self.labels).set(data)

    def _inc_counter(self, counter, data: List) -> None:
        # Convenience function for logging to counter.
        # Prevent ValueError from negative increment
        counter.labels(**self.labels).inc(sum(data))

    def _observe_histogram(self, histogram, data: List) -> None:
        # Convenience function for logging to histogram.
        for value in data:
            histogram.labels(**self.labels).observe(value)

    def update_stats(self, stats: dict[str, List]):
        """Log metrics to Prometheus based on configuration file"""
        # Dynamically log metrics based on what's configured in YAML
        for stat_name, value in stats.items():
            try:
                metric_mapped = self.metric_mappings[stat_name]
                if metric_mapped is None:
                    logger.debug(f"Stat {stat_name} not initialized.")
                    continue
                metric_obj = getattr(self, metric_mapped["attr"], None)
                metric_type = metric_mapped["type"]

                # Log based on metric type
                if metric_type == "counter":
                    self._set_gauge(metric_obj, value)
                elif metric_type == "gauge":
                    self._inc_counter(metric_obj, value)
                elif metric_type == "histogram":
                    # Histograms expect a list
                    self._observe_histogram(metric_obj, value)
                else:
                    logger.error(f"Not found metric type for {stat_name}")
            except Exception:
                logger.debug(f"Failed to log metric {stat_name}")

    @staticmethod
    def _metadata_to_labels(metadata: UCMEngineMetadata):
        return {
            "model_name": metadata.model_name,
            "worker_id": metadata.worker_id,
        }


class UCMStatsLogger:
    def __init__(self, model_name: str, rank: int, config_path: str = ""):
        # Create metadata
        self.metadata = UCMEngineMetadata(
            model_name=str(model_name), worker_id=str(rank)
        )
        # Load configuration
        config = self._load_config(config_path)
        self.log_interval = config.get("log_interval", 10)
        self.prometheus_logger = PrometheusLogger(self.metadata, config)
        self.is_running = True

        self.thread = threading.Thread(target=self.log_worker, daemon=True)
        self.thread.start()

    def _load_config(self, config_path: str) -> dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                if config is None:
                    logger.warning(
                        f"Config file {config_path} is empty, using defaults"
                    )
                    return {}
                return config
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML config file {config_path}: {e}")
            return {}

    def log_worker(self):
        while self.is_running:
            stats = ucmmonitor.get_all_stats_and_clear().data
            self.prometheus_logger.update_stats(stats)
            time.sleep(self.log_interval)

    def shutdown(self):
        self.is_running = False
        self.thread.join()

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
