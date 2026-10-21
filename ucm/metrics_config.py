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

from dataclasses import dataclass
from typing import Any

from ucm.default_metrics_config import get_default_metrics_config
from ucm.logger import init_logger
from ucm.shared.metrics import ucmmetrics

logger = init_logger(__name__)

METRIC_TYPES = ("counter", "gauge", "histogram")
VLLM_EXCLUDED_METRICS = {"interval_lookup_hit_rates"}
MULTIPROC_CONSUMER = "multiproc"
VLLM_CONNECTOR_CONSUMER = "vllm_connector"
ENABLE_METRICS_CONFIG_KEY = "enable_metrics"


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    metric_type: str
    documentation: str = ""
    buckets: tuple[float, ...] = ()
    vllm_connector_name: str = ""
    vllm_connector_buckets: tuple[float, ...] = ()
    vllm_connector_value_scale: float = 1.0
    vllm_connector_enabled: bool = True
    multiprocess_mode: str = ""
    # Names carried as a wire-name suffix instead of a static label value, e.g.
    # ["slot_size"] makes "drampool_buffer_pool_usage_ratio_4096" export as
    # drampool_buffer_pool_usage_ratio{slot_size="4096"}. At most one label is
    # supported today; the suffix must be all digits.
    dynamic_labels: tuple[str, ...] = ()


def load_metrics_config(config_path: str) -> dict[str, Any]:
    if not config_path:
        return {}
    try:
        import yaml

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found")
        return {}
    except ImportError as e:
        logger.error(f"PyYAML is required to read metrics config {config_path}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error loading metrics config file {config_path}: {e}")
        return {}
    if config is None:
        return {}
    if not isinstance(config, dict):
        logger.error(f"Metrics config {config_path} must be a YAML mapping")
        return {}
    return config


def load_launch_metrics_config(launch_config: dict[str, Any] | None) -> dict[str, Any]:
    if not metrics_enabled(launch_config):
        return {}
    if not launch_config:
        return get_default_metrics_config()
    inline_config = launch_config.get("metrics_config")
    if isinstance(inline_config, dict):
        return inline_config
    config_path = launch_config.get("metrics_config_path", "")
    if config_path:
        return load_metrics_config(config_path)
    return get_default_metrics_config()


def metrics_enabled(launch_config: dict[str, Any] | None) -> bool:
    return _as_bool((launch_config or {}).get(ENABLE_METRICS_CONFIG_KEY), True)


def consumer_enabled(
    config: dict[str, Any] | None, consumer: str, default: bool = True
) -> bool:
    consumers = (config or {}).get("consumers")
    if not isinstance(consumers, dict):
        return default
    return _as_bool(consumers.get(consumer), False)


def get_metric_definitions(config: dict[str, Any] | None) -> list[MetricDefinition]:
    if not config:
        return []

    definitions: list[MetricDefinition] = []
    for metric_type in METRIC_TYPES:
        for item in config.get(metric_type, []) or []:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if not name:
                continue
            buckets = tuple(float(bucket) for bucket in item.get("buckets", []) or [])
            vllm_connector_scale = _vllm_connector_value_scale(name, item)
            vllm_connector_buckets = item.get("vllm_connector_buckets")
            if vllm_connector_buckets is None:
                vllm_connector_buckets = tuple(
                    bucket * vllm_connector_scale for bucket in buckets
                )
            else:
                vllm_connector_buckets = tuple(
                    float(bucket) for bucket in vllm_connector_buckets
                )
            vllm_connector_enabled = _metric_vllm_connector_enabled(name, item)
            definitions.append(
                MetricDefinition(
                    name=name,
                    metric_type=metric_type,
                    documentation=item.get("documentation", ""),
                    buckets=buckets,
                    vllm_connector_name=_vllm_connector_metric_name(
                        name,
                        item,
                        vllm_connector_prefix(config),
                    ),
                    vllm_connector_buckets=tuple(vllm_connector_buckets),
                    vllm_connector_value_scale=vllm_connector_scale,
                    vllm_connector_enabled=vllm_connector_enabled,
                    multiprocess_mode=(
                        str(item.get("multiprocess_mode", ""))
                        if metric_type == "gauge"
                        else ""
                    ),
                    dynamic_labels=tuple(item.get("dynamic_labels", []) or []),
                )
            )
    return definitions


def get_vllm_connector_metric_definitions(
    config: dict[str, Any] | None,
) -> list[MetricDefinition]:
    return [
        definition
        for definition in get_metric_definitions(config)
        if definition.vllm_connector_enabled
    ]


def dynamic_label_bases(
    definitions: list[MetricDefinition],
) -> list[tuple[str, MetricDefinition]]:
    """Base names that carry one dynamic label as a wire-name suffix."""
    return [
        (definition.name, definition)
        for definition in definitions
        if len(definition.dynamic_labels) == 1
    ]


def split_dynamic_metric_name(
    metric_name: str, bases: list[tuple[str, MetricDefinition]]
) -> tuple[MetricDefinition, str, str] | None:
    """Resolve a suffixed wire name to (definition, label_name, label_value).

    "drampool_buffer_pool_usage_ratio_4096" with base
    "drampool_buffer_pool_usage_ratio" declaring dynamic_labels=["slot_size"]
    resolves to (definition, "slot_size", "4096"). Returns None when the name
    is not a dynamic-label variant, so callers fall back to exact matching.
    """
    for base_name, definition in bases:
        prefix = base_name + "_"
        if metric_name.startswith(prefix):
            value = metric_name[len(prefix) :]
            if value.isdigit():
                return definition, definition.dynamic_labels[0], value
    return None


def setup_ucm_metrics(config: dict[str, Any] | None) -> list[MetricDefinition]:
    definitions = get_metric_definitions(config)
    if not definitions:
        return []
    ucmmetrics.set_up()
    for definition in definitions:
        ucmmetrics.create_stats(
            definition.name,
            definition.metric_type,
            list(definition.buckets),
        )
    enabled_consumers = [
        consumer
        for consumer in (MULTIPROC_CONSUMER, VLLM_CONNECTOR_CONSUMER)
        if consumer_enabled(config, consumer)
    ]
    counts = _metric_definition_counts(definitions)
    logger.info(
        f"UCM metrics enabled for "
        f"{', '.join(enabled_consumers) if enabled_consumers else 'no consumers'}: "
        f"total={len(definitions)}, counters={counts['counter']}, "
        f"gauges={counts['gauge']}, histograms={counts['histogram']}"
    )
    return definitions


def multiproc_metric_name(
    config: dict[str, Any] | None, metric_name: str, default_prefix: str = "ucm:"
) -> str:
    return f"{multiproc_prefix(config, default_prefix)}{metric_name}"


def multiproc_prefix(
    config: dict[str, Any] | None, default_prefix: str = "ucm:"
) -> str:
    config = config or {}
    return config.get("multiproc_prefix", config.get("metric_prefix", default_prefix))


def vllm_connector_prefix(
    config: dict[str, Any] | None, default_prefix: str = "ucm:"
) -> str:
    return (config or {}).get("vllm_connector_prefix", default_prefix)


def _metric_definition_counts(
    definitions: list[MetricDefinition],
) -> dict[str, int]:
    return {
        metric_type: sum(
            1 for definition in definitions if definition.metric_type == metric_type
        )
        for metric_type in METRIC_TYPES
    }


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _metric_vllm_connector_enabled(name: str, item: dict[str, Any]) -> bool:
    if "vllm_connector_enabled" in item:
        return _as_bool(item.get("vllm_connector_enabled"), True)
    return name not in VLLM_EXCLUDED_METRICS


def _vllm_connector_value_scale(name: str, item: dict[str, Any]) -> float:
    if "vllm_connector_value_scale" in item:
        return float(item["vllm_connector_value_scale"])
    return 1.0


def _vllm_connector_metric_name(name: str, item: dict[str, Any], prefix: str) -> str:
    configured = item.get("vllm_connector_name")
    if configured:
        return _ensure_prefix(str(configured), prefix)
    return f"{prefix}{name}"


def _ensure_prefix(name: str, prefix: str) -> str:
    if name.startswith(prefix):
        return name
    if ":" in name:
        name = name.split(":", 1)[1]
    return f"{prefix}{name}"
