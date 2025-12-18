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
from typing import Any, Union

import yaml
from prometheus_client import Counter, Gauge, Histogram

from ucm.logger import init_logger
from ucm.shared.metrics import ucmmetrics

logger = init_logger(__name__)


class PrometheusStatsLogger:

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

    def __init__(self, model_name, worker_id, config_path):
        """
        Load metrics config from YAML file (config_path),
        register metrics using prometheus_client, and start a thread to get updated metrics.
        """
        # Load metrics config
        self.config = self._load_config(config_path)
        self.log_interval = self.config.get("log_interval", 10)

        multiproc_dir = self.config.get("multiproc_dir", "/vllm-workspace")
        if "PROMETHEUS_MULTIPROC_DIR" not in os.environ:
            os.environ["PROMETHEUS_MULTIPROC_DIR"] = multiproc_dir
            if not os.path.exists(multiproc_dir):
                os.makedirs(multiproc_dir, exist_ok=True)

        self.labels = {
            "model_name": model_name,
            "worker_id": worker_id,
        }
        self.labelnames = list(self.labels.keys())

        self.metric_type_config = {
            "counter": (Counter, {}),
            "gauge": (Gauge, {"multiprocess_mode": "all"}),
            "histogram": (Histogram, {"buckets": []}),
        }
        # Initialize metrics based on config
        self._init_metrics_from_config()

        # Start thread to update metrics
        self.is_running = True
        self.thread = threading.Thread(target=self.update_stats_loop, daemon=True)
        self.thread.start()

    def _register_metrics_by_type(self, metric_type):
        """
        Register metrics by different metric types.
        """
        metric_cls, default_kwargs = self.metric_type_config[metric_type]
        cfg_list = self.config.get(metric_type, [])

        for cfg in cfg_list:
            name = cfg.get("name")
            doc = cfg.get("documentation", "")
            # Prometheus metric name with prefix
            prometheus_name = f"{self.metric_prefix}{name}"
            ucmmetrics.create_stats(name, metric_type)

            metric_kwargs = {
                "name": prometheus_name,
                "documentation": doc,
                "labelnames": self.labelnames,
                **default_kwargs,
                **{k: v for k, v in cfg.items() if k in default_kwargs},
            }

            self.metric_mappings[name] = metric_cls(**metric_kwargs)

    def _init_metrics_from_config(self):
        """Initialize metrics based on config"""
        # Get metric name prefix from config (e.g., "ucm:")
        self.metric_prefix = self.config.get("metric_prefix", "ucm:")

        # Store metric mapping: metric_name -> Union[Counter, Gauge, Histogram]
        # This mapping will be used in update_stats to dynamically log metrics
        self.metric_mappings: dict[str, Union[Counter, Gauge, Histogram]] = {}

        for metric_type in self.metric_type_config.keys():
            self._register_metrics_by_type(metric_type)

    def _update_counter(self, metric, value):
        if value < 0:
            return
        metric.inc(value)

    def _update_gauge(self, metric, value):
        metric.set(value)

    def _update_histogram(self, metric, value):
        for data in value:
            metric.observe(data)

    def _update_with_func(self, update_func, stats: dict[str, Any], op_desc: str):
        """
        Generic update for Prometheus metrics: match metrics by name, bind labels,
        and update values via the specified function (update_func).
        """
        for stat_name, value in stats.items():
            if stat_name not in self.metric_mappings:
                logger.error(f"Metric {stat_name} not found")
                continue

            metric = self.metric_mappings[stat_name]
            try:
                metric_with_labels = metric.labels(**self.labels)
                update_func(metric_with_labels, value)
            except AttributeError as e:
                logger.error(f"Metric {stat_name} does not support {op_desc}: {e}")
            except Exception as e:
                logger.debug(f"Failed to {op_desc} {stat_name}: {e}")

    def update_stats(self, counter_stats, gauge_stats, histogram_stats):
        """
        Update all Prometheus metrics (Counter/Gauge/Histogram) with given stats.
        """
        update_tasks = [
            (self._update_counter, counter_stats, "increment"),
            (self._update_gauge, gauge_stats, "set"),
            (self._update_histogram, histogram_stats, "observe"),
        ]
        for update_func, stats, op_desc in update_tasks:
            self._update_with_func(update_func, stats, op_desc)

    def update_stats_loop(self):
        """
        Periodically update Prometheus metrics in a loop until stopped.
        """
        while self.is_running:
            counter_stats, gauge_stats, histogram_stats = (
                ucmmetrics.get_all_stats_and_clear()
            )
            self.update_stats(counter_stats, gauge_stats, histogram_stats)
            time.sleep(self.log_interval)

    def shutdown(self):
        self.is_running = False
        self.thread.join()

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
