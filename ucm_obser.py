import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import prometheus_client
import yaml

# Third Party
from prometheus_client import REGISTRY

"""====================================="""
import UCMStatsMonitor

"""====================================="""

from vllm.distributed.parallel_state import get_world_group

from ucm.logger import init_logger

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

    def __init__(self, metadata: UCMEngineMetadata, config_path: str):
        # Load configuration from YAML file
        config = self._load_config(config_path)

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

    def _load_config(self, config_path: str) -> Dict[str, Any]:
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

    def _init_metrics_from_config(
        self, labelnames: List[str], prometheus_config: Dict[str, Any]
    ):
        """Initialize metrics based on configuration"""
        enabled = prometheus_config.get("enabled_metrics", {})

        # Get metric name prefix from config (e.g., "ucm:")
        # If not specified, use empty string
        metric_prefix = prometheus_config.get("metric_prefix", "ucm:")

        # Store metric mapping: metric_name -> (metric_type, attribute_name, stats_field_name)
        # This mapping will be used in log_prometheus to dynamically log metrics
        self.metric_mappings: Dict[str, Dict[str, str]] = {}

        # Initialize counters
        if enabled.get("counters", True):
            counters = prometheus_config.get("counters", [])
            for counter_cfg in counters:
                name = counter_cfg.get("name")
                doc = counter_cfg.get("documentation", "")
                # Use name directly as stats field name (since YAML name matches stats attribute)
                stats_field = name
                # Prometheus metric name with prefix
                prometheus_name = f"{metric_prefix}{name}" if metric_prefix else name
                # Internal attribute name for storing the metric object
                attr_name = f"counter_{name.replace(':', '_').replace('-', '_')}"

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
                        "stats_field": stats_field,
                    }

        # Initialize gauges
        if enabled.get("gauges", True):
            gauges = prometheus_config.get("gauges", [])
            for gauge_cfg in gauges:
                name = gauge_cfg.get("name")
                doc = gauge_cfg.get("documentation", "")
                multiprocess_mode = gauge_cfg.get("multiprocess_mode", "livemostrecent")
                # Use name directly as stats field name
                stats_field = name
                # Prometheus metric name with prefix
                prometheus_name = f"{metric_prefix}{name}" if metric_prefix else name
                # Internal attribute name
                attr_name = f"gauge_{name.replace(':', '_').replace('-', '_')}"

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
                        "stats_field": stats_field,
                    }

        # Initialize histograms
        if enabled.get("histograms", True):
            histograms = prometheus_config.get("histograms", [])
            for hist_cfg in histograms:
                name = hist_cfg.get("name")
                doc = hist_cfg.get("documentation", "")
                buckets = hist_cfg.get("buckets", [])
                # Use name directly as stats field name
                stats_field = name
                # Prometheus metric name with prefix
                prometheus_name = f"{metric_prefix}{name}" if metric_prefix else name
                # Internal attribute name
                attr_name = f"histogram_{name.replace(':', '_').replace('-', '_')}"

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
                        "stats_field": stats_field,
                    }

    def _log_gauge(self, gauge, data: Union[int, float]) -> None:
        # Convenience function for logging to gauge.
        gauge.labels(**self.labels).set(data)

    def _log_counter(self, counter, data: Union[int, float]) -> None:
        # Convenience function for logging to counter.
        # Prevent ValueError from negative increment
        if data < 0:
            return
        counter.labels(**self.labels).inc(data)

    def _log_histogram(self, histogram, data: Union[List[int], List[float]]) -> None:
        # Convenience function for logging to histogram.
        for value in data:
            histogram.labels(**self.labels).observe(value)

    def log_prometheus(self, stats: Any):
        """Log metrics to Prometheus based on configuration file"""
        # Dynamically log metrics based on what's configured in YAML
        for metric_name, mapping in self.metric_mappings.items():
            try:
                metric_type = mapping["type"]
                attr_name = mapping["attr"]
                stats_field = mapping["stats_field"]

                # Get the metric object
                metric_obj = getattr(self, attr_name, None)
                if metric_obj is None:
                    logger.warning(
                        f"Metric {metric_name} not initialized (attr: {attr_name})"
                    )
                    continue

                # Get the stats value
                stats_value = getattr(stats, stats_field, None)
                if stats_value is None:
                    # Try to get with default value for missing fields
                    continue

                # Log based on metric type
                if metric_type == "counter":
                    self._log_counter(metric_obj, stats_value)
                elif metric_type == "gauge":
                    self._log_gauge(metric_obj, stats_value)
                elif metric_type == "histogram":
                    # Histograms expect a list
                    if not isinstance(stats_value, list):
                        if stats_value:
                            stats_value = [stats_value]
                        else:
                            stats_value = []
                    self._log_histogram(metric_obj, stats_value)
            except Exception as e:
                logger.warning(f"Failed to log metric {metric_name}: {e}")

    @staticmethod
    def _metadata_to_labels(metadata: UCMEngineMetadata):
        return {
            "model_name": metadata.model_name,
            "worker_id": metadata.worker_id,
        }

    _instance = None

    @staticmethod
    def GetOrCreate(
        metadata: UCMEngineMetadata,
        config_path: str = "/vllm-workspace/metrics_configs.yaml",
    ) -> "PrometheusLogger":
        if PrometheusLogger._instance is None:
            PrometheusLogger._instance = PrometheusLogger(metadata, config_path)
        # assert PrometheusLogger._instance.metadata == metadata, \
        #    "PrometheusLogger instance already created with different metadata"
        if PrometheusLogger._instance.metadata != metadata:
            logger.error(
                "PrometheusLogger instance already created with"
                "different metadata. This should not happen except "
                "in test"
            )
        return PrometheusLogger._instance

    @staticmethod
    def GetInstance() -> "PrometheusLogger":
        assert (
            PrometheusLogger._instance is not None
        ), "PrometheusLogger instance not created yet"
        return PrometheusLogger._instance

    @staticmethod
    def GetInstanceOrNone() -> Optional["PrometheusLogger"]:
        """
        Returns the singleton instance of PrometheusLogger if it exists,
        otherwise returns None.
        """
        return PrometheusLogger._instance


class UCMStatsLogger:
    def __init__(
        self,
        vllm_config: "VllmConfig",
        log_interval: int,
        config_path: str = "/vllm-workspace/metrics_configs.yaml",
    ):
        # Parse model_name and worker_id from vllm_config
        model_name = getattr(vllm_config, "model_name", None)
        worker_id = get_world_group().local_rank
        # Create metadata
        self.metadata = UCMEngineMetadata(
            model_name=str(model_name), worker_id=str(worker_id)
        )

        self.log_interval = log_interval
        self.prometheus_logger = PrometheusLogger.GetOrCreate(
            self.metadata, config_path
        )
        self.is_running = True

        self.thread = threading.Thread(target=self.log_worker, daemon=True)
        self.thread.start()

    def log_worker(self):
        while self.is_running:
            # Use UCMStatsMonitor.get_states_and_clear() from external import
            stats = UCMStatsMonitor.get_states_and_clear()
            self.prometheus_logger.log_prometheus(stats)
            time.sleep(self.log_interval)

    def shutdown(self):
        self.is_running = False
        self.thread.join()
