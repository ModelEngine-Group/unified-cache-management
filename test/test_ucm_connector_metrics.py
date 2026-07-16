import ast
import importlib
import json
import math
import re
import sys
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


class FakeValue:
    def __init__(self):
        self.value = 0

    def inc(self, value):
        self.value += value


class FakeMetric:
    created = {}

    def __init__(self, name, documentation, labelnames=None, buckets=None, **kwargs):
        self.name = name
        self.documentation = documentation
        self.labelnames = list(labelnames or [])
        self.buckets = list(buckets or [])
        self.children = {}
        self.observations = []
        self.increments = []
        self.set_values = []
        self.labelvalues = ()
        self._init_storage()
        self.__class__.created[name] = self

    def _init_storage(self):
        pass

    def labels(self, *labelvalues, **labelkwargs):
        if labelkwargs:
            labelvalues = tuple(labelkwargs[name] for name in self.labelnames)
        child = self.__class__.__new__(self.__class__)
        child.name = self.name
        child.documentation = self.documentation
        child.labelnames = self.labelnames
        child.buckets = self.buckets
        child.children = {}
        child.observations = []
        child.increments = []
        child.set_values = []
        child.labelvalues = tuple(labelvalues)
        child._init_storage()
        self.children[child.labelvalues] = child
        return child

    def observe(self, value):
        self.observations.append(value)

    def inc(self, value):
        self.increments.append(value)

    def set(self, value):
        self.set_values.append(value)


class FakeGauge(FakeMetric):
    created = {}


class FakeCounter(FakeMetric):
    created = {}


class FakeHistogram(FakeMetric):
    created = {}

    def _init_storage(self):
        self._upper_bounds = list(self.buckets)
        if not self._upper_bounds or not math.isinf(self._upper_bounds[-1]):
            self._upper_bounds.append(math.inf)
        self._buckets = [FakeValue() for _ in self._upper_bounds]
        self._sum = FakeValue()


class FakeThread:
    def __init__(self, target):
        self.target = target
        self.started = False
        self.joined = False

    def start(self):
        self.started = True

    def join(self):
        self.joined = True


def _install_package(name, path):
    parent_name, _, child_name = name.rpartition(".")
    if parent_name and parent_name not in sys.modules:
        _install_package(parent_name, path.parent)
    module = ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module
    if parent_name:
        setattr(sys.modules[parent_name], child_name, module)
    return module


def _install_module(name, **attrs):
    parent_name, _, child_name = name.rpartition(".")
    if parent_name and parent_name not in sys.modules:
        try:
            importlib.import_module(parent_name)
        except ModuleNotFoundError:
            _install_module(parent_name)
    module = ModuleType(name)
    for attr, value in attrs.items():
        setattr(module, attr, value)
    sys.modules[name] = module
    if parent_name:
        setattr(sys.modules[parent_name], child_name, module)
    return module


class KVConnectorRole(Enum):
    WORKER = "worker"
    SCHEDULER = "scheduler"


class KVConnectorBase_V1:
    def __init__(self, vllm_config=None, role=None, kv_cache_config=None):
        self._vllm_config = vllm_config
        self._role = role
        self._kv_cache_config = kv_cache_config

    def clear_connector_metadata(self):
        pass

    def get_kv_connector_stats(self):
        return None


class SupportsHMA:
    pass


class KVConnectorMetadata:
    pass


@dataclass
class KVConnectorStats:
    data: dict = field(default_factory=dict)

    def is_empty(self):
        return not self.data


class KVConnectorPromMetrics:
    def __init__(self, vllm_config, metric_types, labelnames, per_engine_labelvalues):
        self._kv_transfer_config = vllm_config.kv_transfer_config
        self._gauge_cls = metric_types[FakeGauge]
        self._counter_cls = metric_types[FakeCounter]
        self._histogram_cls = metric_types[FakeHistogram]
        self._labelnames = labelnames
        self.per_engine_labelvalues = per_engine_labelvalues


class FakeUcmMetrics:
    def __init__(self):
        self.created = []
        self.updated = []
        self.setup_calls = 0
        self.drained = []
        self.snapshot = ({}, {}, {})
        self.on_drain = None

    def set_up(self, *args, **kwargs):
        self.setup_calls += 1

    def create_stats(self, name, metric_type, buckets=None):
        self.created.append((name, metric_type, tuple(buckets or ())))

    def update_stats(self, stats):
        self.updated.append(stats)

    def get_all_stats_and_clear(self):
        self.drained.append("all")
        if self.on_drain is not None:
            self.on_drain()
        snapshot = self.snapshot
        self.snapshot = ({}, {}, {})
        return snapshot


class _Logger:
    def info(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def warning_once(self, *args, **kwargs):
        pass


class CaptureLogger:
    def __init__(self):
        self.infos = []

    def info(self, message, *args, **kwargs):
        if args:
            message = message % args
        self.infos.append(str(message))

    def debug(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def warning_once(self, *args, **kwargs):
        pass


class FakeConfig:
    def __init__(self, kv_transfer_config):
        self._config = getattr(kv_transfer_config, "launch_config", {})

    def get_config(self):
        return self._config


fake_ucmmetrics = FakeUcmMetrics()

GRAFANA_VLLM_UCM_TAG = "ucm-vllm-connector-metrics"
GRAFANA_UCM_DASHBOARDS = [
    "grafana_connector.json",
    "grafana_layerwise.json",
    "grafana_pipeline_store.json",
    "grafana_mooncake.json",
]


def _install_stubs():
    _install_module(
        "numpy",
        ndarray=object,
        bool_=bool,
        uint64="uint64",
        int64="int64",
        zeros=lambda shape, *args, **kwargs: [
            [0 for _ in range(shape[1])] for _ in range(shape[0])
        ],
        vstack=lambda rows: list(rows),
        concatenate=lambda arrays, axis=0: [
            item for array in arrays for item in list(array)
        ],
        asarray=lambda values, dtype=None: values,
        arange=lambda *args, **kwargs: list(range(*args)),
        isscalar=lambda value: isinstance(value, (int, float, bool, str, bytes)),
    )
    _install_package("ucm", REPO_ROOT / "ucm")
    _install_package("ucm.integration", REPO_ROOT / "ucm" / "integration")
    _install_package("ucm.integration.vllm", REPO_ROOT / "ucm" / "integration" / "vllm")
    _install_module("torch", Tensor=type("Tensor", (), {}))
    _install_module(
        "prometheus_client",
        Counter=FakeCounter,
        Gauge=FakeGauge,
        Histogram=FakeHistogram,
    )
    _install_module("wrapt", ObjectProxy=object)
    _install_module("vllm.config", VllmConfig=type("VllmConfig", (), {}))
    _install_module(
        "vllm.distributed.kv_transfer.kv_connector.v1.base",
        KVConnectorBase_V1=KVConnectorBase_V1,
        KVConnectorMetadata=KVConnectorMetadata,
        KVConnectorRole=KVConnectorRole,
        SupportsHMA=SupportsHMA,
    )
    _install_module(
        "vllm.distributed.kv_transfer.kv_connector.v1.metrics",
        KVConnectorPromMetrics=KVConnectorPromMetrics,
        KVConnectorStats=KVConnectorStats,
        PromMetric=object,
        PromMetricT=object,
    )
    _install_module(
        "vllm.distributed.parallel_state",
        get_world_group=lambda: SimpleNamespace(local_rank=0, rank=0),
    )
    _install_module(
        "vllm.model_executor.models.utils",
        extract_layer_index=lambda layer_name: 0,
    )
    _install_module(
        "vllm.platforms",
        current_platform=SimpleNamespace(
            is_cuda_alike=lambda: True,
            device_type="cuda",
        ),
    )
    _install_module(
        "vllm.v1.core.sched.output",
        SchedulerOutput=type("SchedulerOutput", (), {}),
    )
    _install_module(
        "vllm.v1.kv_cache_interface",
        FullAttentionSpec=type("FullAttentionSpec", (), {}),
        KVCacheConfig=type("KVCacheConfig", (), {}),
        KVCacheSpec=type("KVCacheSpec", (), {}),
        MambaSpec=type("MambaSpec", (), {}),
        SlidingWindowSpec=type("SlidingWindowSpec", (), {}),
        UniformTypeKVCacheSpecs=type("UniformTypeKVCacheSpecs", (), {}),
    )
    _install_module(
        "vllm.v1.outputs",
        KVConnectorOutput=type("KVConnectorOutput", (), {}),
    )
    _install_module(
        "ucm.integration.vllm.device",
        create_device=lambda *args, **kwargs: None,
    )
    _install_module("ucm.logger", init_logger=lambda name: _Logger())
    _install_module("ucm.shared.metrics", ucmmetrics=fake_ucmmetrics)
    _install_module(
        "ucm.store.factory_v1",
        UcmConnectorFactoryV1=type("UcmConnectorFactoryV1", (), {}),
    )
    _install_module(
        "ucm.store.ucmstore_v1",
        Task=type("Task", (), {}),
        UcmKVStoreBaseV1=type("UcmKVStoreBaseV1", (), {}),
    )
    _install_module("ucm.utils", Config=FakeConfig)
    _install_module("ucm.sparse.state", has_ucm_sparse=lambda *args, **kwargs: False)


_install_stubs()

import ucm.integration.vllm.ucm_connector as ucm_connector_module
from ucm.default_metrics_config import DEFAULT_METRICS_CONFIG
from ucm.integration.vllm.metrics import UCMConnectorStats, UCMPromMetrics
from ucm.integration.vllm.ucm_connector import (
    PendingDumpTask,
    UCMConnector,
    UCMDirectConnector,
)
from ucm.metrics_config import (
    consumer_enabled,
    get_metric_definitions,
    get_vllm_connector_metric_definitions,
    load_launch_metrics_config,
    metrics_enabled,
    multiproc_metric_name,
    setup_ucm_metrics,
    vllm_connector_prefix,
)
from ucm.metrics_dispatcher import get_metrics_dispatcher


def _metric_types():
    return {
        FakeGauge: FakeGauge,
        FakeCounter: FakeCounter,
        FakeHistogram: FakeHistogram,
    }


def _strip_yaml_comment(line):
    in_quote = False
    previous = ""
    for index, char in enumerate(line):
        if char == '"' and previous != "\\":
            in_quote = not in_quote
        if char == "#" and not in_quote:
            return line[:index].rstrip()
        previous = char
    return line.rstrip()


def _parse_yaml_scalar(value):
    value = value.strip()
    if not value:
        return ""
    if value in {"true", "false"}:
        return value == "true"
    if value[0] in {'"', "'", "["}:
        return ast.literal_eval(value)
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _load_example_metrics_config_without_yaml():
    path = REPO_ROOT / "examples" / "metrics" / "metrics_configs.yaml"
    lines = path.read_text(encoding="utf-8").splitlines()
    config = {}
    section = None
    current_item = None
    index = 0

    while index < len(lines):
        raw = _strip_yaml_comment(lines[index])
        stripped = raw.strip()
        if not stripped:
            index += 1
            continue

        indent = len(raw) - len(raw.lstrip())
        if indent == 0:
            key, _, value = stripped.partition(":")
            if value.strip():
                config[key] = _parse_yaml_scalar(value)
                section = None
            else:
                config[key] = [] if key in {"counter", "gauge", "histogram"} else {}
                section = key
            current_item = None
            index += 1
            continue

        if section == "consumers" and indent == 2:
            key, _, value = stripped.partition(":")
            config[section][key] = _parse_yaml_scalar(value)
            index += 1
            continue

        if section in {"counter", "gauge", "histogram"}:
            if stripped.startswith("- name:"):
                current_item = {"name": _parse_yaml_scalar(stripped.split(":", 1)[1])}
                config[section].append(current_item)
                index += 1
                continue
            if current_item is not None:
                key, _, value = stripped.partition(":")
                if key == "buckets":
                    bucket_text = value.strip()
                    while "]" not in bucket_text:
                        index += 1
                        bucket_text += " " + _strip_yaml_comment(lines[index]).strip()
                    current_item[key] = _parse_yaml_scalar(bucket_text)
                else:
                    current_item[key] = _parse_yaml_scalar(value)
        index += 1

    return config


def _metrics_config(consumers=None):
    return {
        "multiproc_prefix": "ucm_multiproc:",
        "vllm_connector_prefix": "ucm:",
        "consumers": consumers or {"multiproc": True, "vllm_connector": True},
        "counter": [
            {
                "name": "load_bytes_total",
                "documentation": "Total load bytes.",
            }
        ],
        "gauge": [
            {
                "name": "cache_lookup_hit_rate",
                "documentation": "Latest cache lookup hit rate.",
            }
        ],
        "histogram": [
            {
                "name": "load_duration",
                "documentation": "Load duration in ms.",
                "buckets": [50, 100],
            },
            {
                "name": "cache_load_duration_ms",
                "documentation": "Cache load duration in ms.",
                "buckets": [1, 5],
            },
            {
                "name": "interval_lookup_hit_rates",
                "documentation": "Prefer vLLM external prefix cache metrics.",
                "buckets": [0.1, 0.5, 1.0],
            },
        ],
    }


def _vllm_config(config=None, launch_config=None):
    launch_config = dict(launch_config or {})
    if config is not None:
        launch_config["metrics_config"] = config
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector="UCM",
            engine_id="engine-0",
            launch_config=launch_config,
        ),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=True),
    )


def _reset_fakes():
    fake_ucmmetrics.created.clear()
    fake_ucmmetrics.updated.clear()
    fake_ucmmetrics.setup_calls = 0
    fake_ucmmetrics.drained.clear()
    fake_ucmmetrics.snapshot = ({}, {}, {})
    fake_ucmmetrics.on_drain = None
    FakeGauge.created = {}
    FakeCounter.created = {}
    FakeHistogram.created = {}
    import ucm.metrics_dispatcher as dispatcher_module

    dispatcher_module._DISPATCHER = None


def test_config_definitions_register_enable_list_and_metric_names():
    _reset_fakes()
    config = _metrics_config()

    definitions = get_metric_definitions(config)
    by_name = {definition.name: definition for definition in definitions}
    setup_ucm_metrics(config)

    assert consumer_enabled(config, "multiproc")
    assert consumer_enabled(config, "vllm_connector")
    assert not consumer_enabled({"consumers": {"vllm_connector": True}}, "multiproc")
    assert consumer_enabled({"consumers": {"vllm_connector": True}}, "vllm_connector")
    assert (
        multiproc_metric_name(config, "load_bytes_total")
        == "ucm_multiproc:load_bytes_total"
    )
    assert vllm_connector_prefix({}) == "ucm:"
    assert list(by_name) == [
        "load_bytes_total",
        "cache_lookup_hit_rate",
        "load_duration",
        "cache_load_duration_ms",
        "interval_lookup_hit_rates",
    ]
    assert by_name["load_duration"].vllm_connector_name == "ucm:load_duration"
    assert by_name["load_duration"].vllm_connector_buckets == (50, 100)
    assert by_name["load_duration"].vllm_connector_value_scale == 1.0
    assert (
        by_name["cache_load_duration_ms"].vllm_connector_name
        == "ucm:cache_load_duration_ms"
    )
    assert by_name["interval_lookup_hit_rates"].vllm_connector_enabled is False
    assert fake_ucmmetrics.created == [
        ("load_bytes_total", "counter", ()),
        ("cache_lookup_hit_rate", "gauge", ()),
        ("load_duration", "histogram", (50, 100)),
        ("cache_load_duration_ms", "histogram", (1, 5)),
        ("interval_lookup_hit_rates", "histogram", (0.1, 0.5, 1.0)),
    ]


def test_default_metrics_config_matches_example_yaml():
    assert DEFAULT_METRICS_CONFIG == _load_example_metrics_config_without_yaml()


def test_launch_metrics_config_defaults_to_builtin_metrics_when_path_is_missing():
    _reset_fakes()

    config = load_launch_metrics_config({})
    definitions = get_metric_definitions(config)
    setup_ucm_metrics(config)

    assert config == DEFAULT_METRICS_CONFIG
    assert definitions
    assert consumer_enabled(config, "vllm_connector")
    assert not consumer_enabled(config, "multiproc")
    assert fake_ucmmetrics.setup_calls == 1
    assert {name for name, _, _ in fake_ucmmetrics.created} == {
        definition.name for definition in definitions
    }


def test_launch_metrics_config_uses_explicit_file_as_enable_list(monkeypatch):
    import ucm.metrics_config as metrics_config

    custom_config = {
        "vllm_connector_prefix": "ucm:",
        "consumers": {"vllm_connector": True},
        "counter": [
            {
                "name": "only_this_counter_total",
                "documentation": "Only configured metric.",
            }
        ],
    }
    monkeypatch.setattr(metrics_config, "load_metrics_config", lambda _: custom_config)

    config = load_launch_metrics_config({"metrics_config_path": "metrics.yaml"})
    definitions = get_metric_definitions(config)

    assert [definition.name for definition in definitions] == [
        "only_this_counter_total"
    ]


def test_launch_metrics_config_respects_enable_metrics_switch():
    assert metrics_enabled({}) is True
    assert metrics_enabled({"enable_metrics": False}) is False
    assert load_launch_metrics_config({"enable_metrics": False}) == {}
    assert (
        load_launch_metrics_config(
            {"enable_metrics": False, "metrics_config": _metrics_config()}
        )
        == {}
    )


def test_setup_ucm_metrics_logs_registered_metrics(monkeypatch):
    _reset_fakes()
    import ucm.metrics_config as metrics_config

    capture_logger = CaptureLogger()
    monkeypatch.setattr(metrics_config, "logger", capture_logger)

    setup_ucm_metrics(_metrics_config())

    assert capture_logger.infos == [
        "UCM metrics enabled for multiproc, vllm_connector: "
        "total=5, counters=1, gauges=1, histograms=3"
    ]


def test_scheduler_side_metrics_are_configured_for_vllm_connector():
    config = load_launch_metrics_config({})
    definitions = {
        definition.name for definition in get_vllm_connector_metric_definitions(config)
    }

    assert {
        "total_prefix_query_tokens_total",
        "gpu_hbm_hit_tokens_total",
        "ucm_hit_tokens_total",
        "connector_lookup_errors_total",
        "fawa_scheduler_lookup_external_hit_blocks_ms",
        "fawa_scheduler_get_num_new_matched_tokens_ms",
    } <= definitions


def test_dispatcher_fans_out_single_core_drain_to_independent_consumers():
    _reset_fakes()
    config = _metrics_config()
    fake_ucmmetrics.snapshot = (
        {"load_bytes_total": 4096.0, "not_configured": 99.0},
        {"cache_lookup_hit_rate": 0.5},
        {
            "load_duration": ([0, 1, 0], 75.0),
            "interval_lookup_hit_rates": ([1, 0, 0, 0], 0.1),
        },
    )
    dispatcher = get_metrics_dispatcher(config)

    dispatcher.drain_to_consumers()
    multiproc_stats = dispatcher.get_stats_and_clear("multiproc")
    vllm_stats = dispatcher.get_stats_and_clear("vllm_connector")

    assert fake_ucmmetrics.drained == ["all"]
    assert multiproc_stats[0] == {"load_bytes_total": 4096.0}
    assert multiproc_stats[1] == {"cache_lookup_hit_rate": 0.5}
    assert multiproc_stats[2]["load_duration"] == ([0, 1, 0], 75.0)
    assert multiproc_stats[2]["interval_lookup_hit_rates"] == ([1, 0, 0, 0], 0.1)
    assert vllm_stats[0] == {"load_bytes_total": 4096.0}
    assert vllm_stats[1] == {"cache_lookup_hit_rate": 0.5}
    assert vllm_stats[2] == {"load_duration": ([0, 1, 0], 75.0)}
    assert dispatcher.get_stats_and_clear("multiproc") == ({}, {}, {})


def test_dispatcher_accumulates_deltas_and_keeps_gauge_latest():
    _reset_fakes()
    dispatcher = get_metrics_dispatcher(_metrics_config())

    fake_ucmmetrics.snapshot = (
        {"load_bytes_total": 100.0},
        {"cache_lookup_hit_rate": 0.25},
        {"load_duration": ([1, 0, 0], 50.0)},
    )
    dispatcher.drain_to_consumers()
    fake_ucmmetrics.snapshot = (
        {"load_bytes_total": 50.0},
        {"cache_lookup_hit_rate": 0.75},
        {"load_duration": ([0, 2, 0], 125.0)},
    )
    dispatcher.drain_to_consumers()

    counters, gauges, histograms = dispatcher.get_stats_and_clear("vllm_connector")

    assert counters == {"load_bytes_total": 150.0}
    assert gauges == {"cache_lookup_hit_rate": 0.75}
    assert histograms == {"load_duration": ([1, 2, 0], 175.0)}


def test_dispatcher_lock_covers_core_drain_and_fanout():
    _reset_fakes()
    dispatcher = get_metrics_dispatcher(_metrics_config())
    fake_ucmmetrics.snapshot = (
        {"load_bytes_total": 100.0},
        {},
        {"load_duration": ([1, 0, 0], 50.0)},
    )
    reader_snapshot = None
    reader_started = threading.Event()
    reader_thread = None

    def read_consumer():
        nonlocal reader_snapshot
        reader_started.set()
        reader_snapshot = dispatcher.get_stats_and_clear("vllm_connector")

    def read_while_core_drain_is_active():
        nonlocal reader_thread
        reader_thread = threading.Thread(target=read_consumer)
        reader_thread.start()
        reader_started.wait(timeout=1)

    fake_ucmmetrics.on_drain = read_while_core_drain_is_active

    dispatcher.drain_to_consumers()
    reader_thread.join(timeout=1)

    assert reader_snapshot == (
        {"load_bytes_total": 100.0},
        {},
        {"load_duration": ([1, 0, 0], 50.0)},
    )
    assert dispatcher.get_stats_and_clear("vllm_connector") == ({}, {}, {})


def test_dispatcher_disabled_consumer_does_not_store_snapshot():
    _reset_fakes()
    config = _metrics_config({"multiproc": False, "vllm_connector": True})
    fake_ucmmetrics.snapshot = (
        {"load_bytes_total": 10.0},
        {},
        {"load_duration": ([1, 0, 0], 50.0)},
    )
    dispatcher = get_metrics_dispatcher(config)

    dispatcher.drain_to_consumers()

    assert dispatcher.get_stats_and_clear("multiproc") == ({}, {}, {})
    assert dispatcher.get_stats_and_clear("vllm_connector")[0] == {
        "load_bytes_total": 10.0
    }


def test_dispatcher_rejects_invalid_consumer_name():
    _reset_fakes()
    dispatcher = get_metrics_dispatcher(_metrics_config())

    with pytest.raises(ValueError):
        dispatcher.get_stats_and_clear("legacy")


def test_stats_from_ucm_snapshot_preserves_metric_types_and_worker_rank():
    definitions = get_vllm_connector_metric_definitions(_metrics_config())
    stats = UCMConnectorStats.from_ucm_snapshot(
        counter_stats={"load_bytes_total": 2048.0, "not_configured": 99.0},
        gauge_stats={"cache_lookup_hit_rate": 0.75},
        histogram_stats={
            "load_duration": ([1, 2, 0], 150.0),
            "interval_lookup_hit_rates": ([1, 0, 0, 0], 0.1),
        },
        worker_rank=7,
        metric_definitions=definitions,
    )

    assert stats.worker_rank == "7"
    assert stats.data["counters_by_rank"]["7"]["load_bytes_total"] == 2048.0
    assert stats.data["gauges_by_rank"]["7"]["cache_lookup_hit_rate"] == 0.75
    assert stats.data["histograms_by_rank"]["7"]["load_duration"] == {
        "bucket_counts": [1, 2, 0],
        "sum": 150.0,
    }
    assert "not_configured" not in stats.data["counters_by_rank"]["7"]
    assert "interval_lookup_hit_rates" not in stats.data["histograms_by_rank"]["7"]


def test_stats_record_aggregate_clone_and_reset_preserve_worker_rank():
    rank0 = UCMConnectorStats(worker_rank=0)
    rank1 = UCMConnectorStats(worker_rank=1)

    rank0.record({"load_duration": 50.0}, {"load_duration": "histogram"})
    rank1.aggregate(
        UCMConnectorStats(
            data={
                "counters_by_rank": {"1": {"load_bytes_total": 10.0}},
                "gauges_by_rank": {"1": {"cache_lookup_hit_rate": 0.5}},
                "histograms_by_rank": {
                    "1": {"load_duration": {"bucket_counts": [0, 1], "sum": 75.0}}
                },
            }
        )
    )

    rank0.aggregate(rank1)
    snapshot = rank0.clone_and_reset()

    assert rank0.is_empty()
    assert snapshot.data["histograms_by_rank"]["0"]["load_duration"] == [50.0]
    assert snapshot.data["counters_by_rank"]["1"]["load_bytes_total"] == 10.0
    assert snapshot.data["gauges_by_rank"]["1"]["cache_lookup_hit_rate"] == 0.5
    assert snapshot.data["histograms_by_rank"]["1"]["load_duration"] == {
        "bucket_counts": [0, 1],
        "sum": 75.0,
    }
    assert snapshot.worker_rank == "0"


def test_stats_reduce_skips_ucm_cli_summary():
    stats = UCMConnectorStats(
        data={
            "counters_by_rank": {"0": {"load_bytes_total": 10.0}},
            "gauges_by_rank": {"0": {"cache_lookup_hit_rate": 0.5}},
            "histograms_by_rank": {
                "0": {
                    "load_duration": {"bucket_counts": [1, 0], "sum": 50.0},
                    "cache_load_duration_ms": {
                        "bucket_counts": [0, 1],
                        "sum": 75.0,
                    },
                }
            },
        }
    )

    assert stats.reduce() == {}


def test_prom_metrics_register_vllm_connector_prefixed_metrics():
    _reset_fakes()
    import ucm.integration.vllm.metrics as metrics_module

    capture_logger = CaptureLogger()
    metrics_module.logger = capture_logger
    prom = UCMPromMetrics(
        _vllm_config(_metrics_config()),
        _metric_types(),
        ["model_name", "engine"],
        {0: ["model-a", "0"]},
    )

    prom.observe(
        {
            "counters_by_rank": {"7": {"load_bytes_total": 2048.0}},
            "gauges_by_rank": {"7": {"cache_lookup_hit_rate": 0.75}},
            "histograms_by_rank": {
                "7": {"load_duration": {"bucket_counts": [1, 2, 0], "sum": 150.0}}
            },
        },
        engine_idx=0,
    )

    assert all(name.startswith("ucm:") for name in FakeCounter.created)
    assert all(name.startswith("ucm:") for name in FakeGauge.created)
    assert all(name.startswith("ucm:") for name in FakeHistogram.created)
    assert capture_logger.infos == [
        "UCM metrics vllm_connector path enabled: "
        "total=4, counters=1, gauges=1, histograms=2, "
        "labels=['model_name', 'engine', 'worker_rank']"
    ]

    counter = FakeCounter.created["ucm:load_bytes_total"]
    gauge = FakeGauge.created["ucm:cache_lookup_hit_rate"]
    histogram = FakeHistogram.created["ucm:load_duration"]

    assert counter.labelnames == ["model_name", "engine", "worker_rank"]
    assert counter.children[("model-a", "0", "7")].increments == [2048.0]
    assert gauge.children[("model-a", "0", "7")].set_values == [0.75]
    histogram_child = histogram.children[("model-a", "0", "7")]
    assert [bucket.value for bucket in histogram_child._buckets] == [1, 2, 0]
    assert histogram_child._sum.value == 150.0


def test_prom_metrics_keeps_ucm_duration_observations_in_ms():
    _reset_fakes()
    prom = UCMPromMetrics(
        _vllm_config(_metrics_config()),
        _metric_types(),
        ["model_name", "engine"],
        {0: ["model-a", "0"]},
    )

    prom.observe({"histograms_by_rank": {"3": {"load_duration": [50.0]}}})

    histogram = FakeHistogram.created["ucm:load_duration"]
    assert histogram.children[("model-a", "0", "3")].observations == [50.0]


def test_ucm_connector_builds_stats_and_respects_vllm_connector_switch():
    data = {"histograms_by_rank": {"3": {"load_duration": [2.0]}}}

    stats = UCMConnector.build_kv_connector_stats(data)
    prom = UCMConnector.build_prom_metrics(
        _vllm_config(_metrics_config({"multiproc": True, "vllm_connector": True})),
        _metric_types(),
        ["model_name", "engine"],
        {0: ["model-a", "0"]},
    )
    disabled = UCMConnector.build_prom_metrics(
        _vllm_config(_metrics_config({"multiproc": True, "vllm_connector": False})),
        _metric_types(),
        ["model_name", "engine"],
        {0: ["model-a", "0"]},
    )
    default_prom = UCMConnector.build_prom_metrics(
        _vllm_config(),
        _metric_types(),
        ["model_name", "engine"],
        {0: ["model-a", "0"]},
    )
    disabled_by_launch_config = UCMConnector.build_prom_metrics(
        _vllm_config(launch_config={"enable_metrics": False}),
        _metric_types(),
        ["model_name", "engine"],
        {0: ["model-a", "0"]},
    )

    assert isinstance(stats, UCMConnectorStats)
    assert stats.data == data
    assert isinstance(prom, UCMPromMetrics)
    assert isinstance(default_prom, UCMPromMetrics)
    assert disabled is None
    assert disabled_by_launch_config is None


def test_ucm_connector_metrics_registration_is_owned_by_outer_connector():
    class CustomInnerConnector(UCMDirectConnector):
        pass

    prom = UCMConnector.build_prom_metrics(
        _vllm_config(),
        _metric_types(),
        ["model_name", "engine"],
        {0: ["model-a", "0"]},
    )
    stats = UCMConnector.build_kv_connector_stats(
        {"counters_by_rank": {"0": {"load_bytes_total": 1.0}}}
    )

    assert "get_kv_connector_stats" in UCMConnector.__dict__
    assert "build_kv_connector_stats" in UCMConnector.__dict__
    assert "build_prom_metrics" in UCMConnector.__dict__
    assert not hasattr(CustomInnerConnector, "build_prom_metrics")
    assert isinstance(prom, UCMPromMetrics)
    assert isinstance(stats, UCMConnectorStats)
    assert stats.data["counters_by_rank"]["0"]["load_bytes_total"] == 1.0


def test_ucm_connector_prefers_lite_when_lite_and_fawa_are_both_enabled(monkeypatch):
    class FakeInnerConnector(KVConnectorBase_V1):
        pass

    class FakeFawaConnector(KVConnectorBase_V1):
        @classmethod
        def can_handle_kv_cache_config(cls, kv_cache_config):
            return True

    monkeypatch.setattr(UCMConnector, "_setup_ucm_metrics", lambda *args: None)
    monkeypatch.setattr(ucm_connector_module, "UCMLiteConnector", FakeInnerConnector)
    monkeypatch.setitem(
        sys.modules,
        "ucm.integration.vllm.hma_connector",
        SimpleNamespace(UCMFAWAConnector=FakeFawaConnector),
    )

    connector = UCMConnector(
        _vllm_config(launch_config={"use_lite": True}),
        KVConnectorRole.SCHEDULER,
        kv_cache_config=object(),
    )

    assert type(connector.connector) is FakeInnerConnector


def test_ucm_connector_drains_dispatcher_vllm_connector_snapshot():
    _reset_fakes()
    config = _metrics_config()
    dispatcher = get_metrics_dispatcher(config)
    fake_ucmmetrics.snapshot = (
        {"load_bytes_total": 4096.0},
        {"cache_lookup_hit_rate": 0.5},
        {"load_duration": ([0, 1, 0], 75.0)},
    )
    connector = object.__new__(UCMConnector)
    connector.connector = SimpleNamespace(
        get_kv_connector_stats=lambda: pytest.fail(
            "inner connector metrics should not be used"
        )
    )
    connector._vllm_metrics_enabled = True
    connector._vllm_metric_definitions = get_vllm_connector_metric_definitions(config)
    connector._metrics_dispatcher = dispatcher
    connector._worker_rank = 5

    stats = connector.get_kv_connector_stats()

    assert fake_ucmmetrics.drained == ["all"]
    assert stats.data["counters_by_rank"]["5"]["load_bytes_total"] == 4096.0
    assert stats.data["gauges_by_rank"]["5"]["cache_lookup_hit_rate"] == 0.5
    assert stats.data["histograms_by_rank"]["5"]["load_duration"] == {
        "bucket_counts": [0, 1, 0],
        "sum": 75.0,
    }
    assert connector.get_kv_connector_stats() is None


def test_ucm_connector_drains_scheduler_vllm_connector_snapshot():
    _reset_fakes()
    connector = object.__new__(UCMConnector)
    connector.launch_config = {}
    connector.engine_id = "engine-0"
    connector._worker_rank = "scheduler"

    connector._setup_ucm_metrics(_vllm_config(), KVConnectorRole.SCHEDULER)
    fake_ucmmetrics.snapshot = (
        {
            "total_prefix_query_tokens_total": 2048,
            "gpu_hbm_hit_tokens_total": 512,
            "ucm_hit_tokens_total": 384,
        },
        {},
        {"fawa_scheduler_get_num_new_matched_tokens_ms": ([0, 1], 12.0)},
    )

    stats = connector.get_kv_connector_stats()

    assert stats.data["counters_by_rank"]["scheduler"] == {
        "total_prefix_query_tokens_total": 2048.0,
        "gpu_hbm_hit_tokens_total": 512.0,
        "ucm_hit_tokens_total": 384.0,
    }
    assert stats.data["histograms_by_rank"]["scheduler"][
        "fawa_scheduler_get_num_new_matched_tokens_ms"
    ] == {
        "bucket_counts": [0, 1],
        "sum": 12.0,
    }


def test_ucm_connector_records_prefix_cache_token_counters():
    _reset_fakes()
    connector = object.__new__(UCMConnector)
    connector.connector = SimpleNamespace(
        get_num_new_matched_tokens=lambda request, num_computed_tokens: (384, False)
    )
    request = SimpleNamespace(num_tokens=2048)

    assert connector.get_num_new_matched_tokens(request, 512) == (384, False)

    assert fake_ucmmetrics.updated == [
        {
            "total_prefix_query_tokens_total": 2048,
            "gpu_hbm_hit_tokens_total": 512,
            "ucm_hit_tokens_total": 384,
        }
    ]


def test_vllm_ascend_scheduler_patch_collects_scheduler_only_stats():
    from ucm.integration.vllm.patch import scheduler_metrics_patch

    class Stats:
        def __init__(self, name, empty=False):
            self.name = name
            self.empty = empty

        def is_empty(self):
            return self.empty

    class Connector:
        def __init__(self, stats):
            self.stats = stats
            self.calls = 0

        def get_kv_connector_stats(self):
            self.calls += 1
            return self.stats

    class Scheduler:
        def __init__(self, connector):
            self.connector = connector

        def make_stats(
            self,
            spec_decoding_stats=None,
            kv_connector_stats=None,
            cudagraph_stats=None,
            perf_stats=None,
        ):
            return kv_connector_stats

    scheduler_metrics_patch.patch_recompute_scheduler_cls(Scheduler)

    scheduler_stats = Stats("scheduler")
    scheduler = Scheduler(Connector(scheduler_stats))
    assert scheduler.make_stats(kv_connector_stats=None) is scheduler_stats
    assert scheduler.connector.calls == 1

    worker_stats = Stats("worker")
    scheduler = Scheduler(Connector(scheduler_stats))
    assert scheduler.make_stats(kv_connector_stats=worker_stats) is worker_stats
    assert scheduler.connector.calls == 0


def test_vllm_ascend_scheduler_patch_entrypoints_are_versioned():
    root = REPO_ROOT / "ucm" / "integration" / "vllm" / "patch"
    common_import = "ucm.integration.vllm.patch.scheduler_metrics_patch"

    assert not (root / "vllm_ascend").exists()
    assert common_import in (
        root / "v0180" / "vllm_ascend" / "ucm_connector_patch.py"
    ).read_text(encoding="utf-8")
    assert common_import in (
        root / "v0191" / "vllm_ascend" / "pc_ascend_patch.py"
    ).read_text(encoding="utf-8")
    assert common_import in (
        root / "v0202" / "vllm_ascend" / "ascend_hybrid_cache_patch.py"
    ).read_text(encoding="utf-8")


def test_direct_connector_get_finished_records_async_durations():
    _reset_fakes()
    import ucm.integration.vllm.ucm_connector as ucm_connector_module

    class Store:
        def __init__(self):
            self.waited = []

        def wait(self, task):
            self.waited.append(task)

    class Device:
        def __init__(self):
            self.destroyed = []

        def destroy_event_handle(self, event_handle):
            self.destroyed.append(event_handle)

    connector = object.__new__(UCMDirectConnector)
    connector.store = Store()
    connector.enable_event_sync = True
    connector.device = Device()
    task = object()
    pending = PendingDumpTask(
        task=task,
        request_ids={"req-1"},
        event_handle=7,
        wait_for_save_start_ms=900.0,
    )
    connector._pending_dump_tasks = [pending]
    connector._async_dump_req_ids = {"req-1"}

    times = iter([1.0, 1.025])
    original_perf_counter = ucm_connector_module.time.perf_counter
    ucm_connector_module.time.perf_counter = lambda: next(times)
    try:
        finished, skipped = connector.get_finished({"req-1"})
    finally:
        ucm_connector_module.time.perf_counter = original_perf_counter

    assert finished == {"req-1"}
    assert skipped is None
    assert connector._pending_dump_tasks == []
    assert connector._async_dump_req_ids == set()
    assert connector.store.waited == [task]
    assert connector.device.destroyed == [7]
    assert pending.event_handle == 0
    assert fake_ucmmetrics.updated == [
        {
            "save_duration": 125.0,
            "save_completion_wait_duration": 25.0,
        }
    ]


def test_direct_connector_poll_records_zero_completion_wait_duration():
    _reset_fakes()
    import ucm.integration.vllm.ucm_connector as ucm_connector_module

    class Store:
        def __init__(self):
            self.checked = []
            self.waited = []

        def check(self, task):
            self.checked.append(task)
            return True

        def wait(self, task):
            self.waited.append(task)

    connector = object.__new__(UCMDirectConnector)
    connector.store = Store()
    connector.enable_event_sync = False
    connector.device = None
    task = object()
    connector._pending_dump_tasks = [
        PendingDumpTask(
            task=task,
            request_ids={"req-1"},
            wait_for_save_start_ms=900.0,
        )
    ]

    times = iter([1.0, 1.0])
    original_perf_counter = ucm_connector_module.time.perf_counter
    ucm_connector_module.time.perf_counter = lambda: next(times)
    try:
        connector._poll_pending_dump_tasks()
    finally:
        ucm_connector_module.time.perf_counter = original_perf_counter

    assert connector._pending_dump_tasks == []
    assert connector.store.checked == [task]
    assert connector.store.waited == [task]
    assert fake_ucmmetrics.updated == [
        {
            "save_duration": 100.0,
            "save_completion_wait_duration": 0.0,
        }
    ]


def test_multiproc_logger_uses_prefix_and_dispatcher_snapshot(tmp_path):
    _reset_fakes()
    import ucm.observability as observability

    config = _metrics_config()
    config["multiproc_dir"] = str(tmp_path)
    capture_logger = CaptureLogger()
    observability._metric_mappings.clear()
    observability.load_metrics_config = lambda _: config
    observability.logger = capture_logger
    observability.threading.Thread = FakeThread
    original_sleep = observability.time.sleep
    logger = observability.PrometheusStatsLogger("model-a", "worker-0", "unused.yaml")
    fake_ucmmetrics.snapshot = (
        {"load_bytes_total": 1024.0},
        {"cache_lookup_hit_rate": 0.25},
        {"load_duration": ([1, 0, 0], 50.0)},
    )

    observability.time.sleep = lambda _: setattr(logger, "is_running", False)
    try:
        logger.update_stats_loop()
    finally:
        observability.time.sleep = original_sleep

    counter = FakeCounter.created["ucm_multiproc:load_bytes_total"]
    gauge = FakeGauge.created["ucm_multiproc:cache_lookup_hit_rate"]
    histogram = FakeHistogram.created["ucm_multiproc:load_duration"]

    assert logger.thread.started
    assert any(
        "UCM metrics multiproc path enabled: total=5, counters=1, gauges=1, "
        "histograms=3, prefix=ucm_multiproc:, labels=['model_name', 'worker_id']"
        in message
        for message in capture_logger.infos
    )
    assert counter.children[("model-a", "worker-0")].increments == [1024.0]
    assert gauge.children[("model-a", "worker-0")].set_values == [0.25]
    histogram_child = histogram.children[("model-a", "worker-0")]
    assert [bucket.value for bucket in histogram_child._buckets] == [1, 0, 0]
    assert histogram_child._sum.value == 50.0


def test_multiproc_logger_respects_consumer_switch():
    _reset_fakes()
    import ucm.observability as observability

    observability._metric_mappings.clear()
    observability.load_metrics_config = lambda _: _metrics_config(
        {"multiproc": False, "vllm_connector": True}
    )
    logger = observability.PrometheusStatsLogger("model-a", "worker-0", "unused.yaml")

    assert not hasattr(logger, "thread")
    assert FakeCounter.created == {}


def test_ucm_connector_get_kv_connector_stats_skips_inner_when_disabled():
    connector = object.__new__(UCMConnector)
    connector.connector = SimpleNamespace(
        get_kv_connector_stats=lambda: pytest.fail(
            "inner connector metrics should not be used"
        )
    )
    connector._vllm_metrics_enabled = False

    assert connector.get_kv_connector_stats() is None


def test_example_metrics_config_defaults_to_vllm_connector_metrics():
    text = (REPO_ROOT / "examples" / "metrics" / "metrics_configs.yaml").read_text(
        encoding="utf-8"
    )
    assert "connector_task_" not in text
    assert 'name: "save_completion_wait_duration"' in text
    assert 'name: "cache_d2h_callback_wait_ms"' not in text
    assert 'name: "save_speed"' not in text
    assert '# multiproc_dir: "/vllm-workspace"' in text
    assert '# multiproc_prefix: "ucm_multiproc:"' in text
    assert 'vllm_connector_prefix: "ucm:"' in text
    assert "# multiproc: true" in text
    assert re.search(r"^\s+vllm_connector:\s+true$", text, re.MULTILINE)
    assert not re.search(r"^\s+multiproc:\s+true$", text, re.MULTILINE)


def test_connector_dashboard_direct_connector_layout_and_metrics():
    dashboard_path = REPO_ROOT / "examples" / "metrics" / "grafana_connector.json"
    text = dashboard_path.read_text(encoding="utf-8")
    dashboard = json.loads(text)
    panels = dashboard["panels"]
    titles = [panel.get("title", "") for panel in panels]

    assert "_seconds" not in text
    assert "1000 *" not in text
    assert "save_speed" not in text
    assert panels[0]["title"] == "Connector Prefix Cache Hit Rate"
    assert panels[0]["gridPos"] == {"h": 8, "w": 24, "x": 0, "y": 0}
    assert "Direct Connector" in titles
    assert not any("Requests Rate" in title for title in titles)
    assert not any("Size Distribution" in title for title in titles)

    expected_direct_titles = [
        "Direct Connector",
        "Connector Load Bandwidth (aggregated)",
        "Connector Dump Bandwidth (aggregated)",
        "Connector Load Duration",
        "Connector Dump Duration",
        "Connector Load Speed (per task)",
        "Connector Dump Completion Wait Duration",
    ]
    expected_fawa_titles = [
        "FAWA Connector",
        "FAWA Scheduler Lookup Duration",
        "FAWA Scheduler Match Duration",
        "FAWA Worker Load Duration",
        "FAWA Worker Wait Load Tasks Duration",
        "FAWA Worker Dump Duration",
    ]
    direct_start = titles.index("Direct Connector")
    assert titles[direct_start:] == expected_direct_titles + expected_fawa_titles

    by_title = {panel["title"]: panel for panel in panels}
    assert by_title["Direct Connector"]["type"] == "row"
    assert by_title["Direct Connector"]["gridPos"] == {
        "h": 1,
        "w": 24,
        "x": 0,
        "y": 8,
    }
    assert by_title["FAWA Connector"]["type"] == "row"
    assert by_title["FAWA Connector"]["gridPos"] == {
        "h": 1,
        "w": 24,
        "x": 0,
        "y": 33,
    }

    expected_fawa_metrics = {
        "FAWA Scheduler Lookup Duration": "fawa_scheduler_lookup_external_hit_blocks_ms",
        "FAWA Scheduler Match Duration": "fawa_scheduler_get_num_new_matched_tokens_ms",
        "FAWA Worker Load Duration": "fawa_worker_start_load_kv_ms",
        "FAWA Worker Wait Load Tasks Duration": "fawa_worker_wait_wait_all_load_task_ms",
        "FAWA Worker Dump Duration": "fawa_worker_wait_for_save_ms",
    }
    for title, metric in expected_fawa_metrics.items():
        panel = by_title[title]
        assert panel["fieldConfig"]["defaults"]["unit"] == "ms"
        assert panel["type"] == "timeseries"
        exprs = [target["expr"] for target in panel["targets"]]
        assert all(f"ucm:{metric}" in expr for expr in exprs)
        assert any("_bucket" in expr for expr in exprs)
        assert any("_sum" in expr and "_count" in expr for expr in exprs)

    occupied = set()
    for panel in panels:
        grid = panel["gridPos"]
        for x in range(grid["x"], grid["x"] + grid["w"]):
            for y in range(grid["y"], grid["y"] + grid["h"]):
                cell = (x, y)
                assert cell not in occupied
                occupied.add(cell)


def test_ucm_dashboards_reference_configured_vllm_connector_metrics():
    metrics_text = (
        REPO_ROOT / "examples" / "metrics" / "metrics_configs.yaml"
    ).read_text(encoding="utf-8")
    configured_names = set(
        re.findall(r'^\s*-\s+name:\s+"([^"]+)"', metrics_text, re.MULTILINE)
    )
    expected_vllm_names = set()
    for name in configured_names:
        if name == "interval_lookup_hit_rates":
            continue
        expected_vllm_names.add(f"ucm:{name}")

    for filename in GRAFANA_UCM_DASHBOARDS:
        dashboard_text = (REPO_ROOT / "examples" / "metrics" / filename).read_text(
            encoding="utf-8"
        )
        assert "vllm:ucm_" not in dashboard_text
        referenced = set(re.findall(r"ucm:[A-Za-z0-9_]+", dashboard_text))
        referenced = {
            re.sub(r"_(bucket|sum|count)$", "", metric) for metric in referenced
        }

        assert referenced <= expected_vllm_names
        assert not any(
            metric.startswith("ucm:connector_task_") for metric in referenced
        )


def test_vllm_dashboard_uses_combined_prefix_cache_hit_rate_breakdown():
    dashboard = json.loads(
        (REPO_ROOT / "examples" / "metrics" / "grafana_vllm.json").read_text(
            encoding="utf-8"
        )
    )
    panels = {panel["title"]: panel for panel in dashboard["panels"]}

    assert "GPU Prefix Cache Hit Rate" not in panels
    assert "External Connector Hit Rate" not in panels

    panel = panels["KV Cache Hit Rate Breakdown"]
    assert panel["fieldConfig"]["defaults"]["unit"] == "percentunit"
    assert panel["fieldConfig"]["defaults"]["min"] == 0
    assert panel["fieldConfig"]["defaults"]["max"] == 1
    assert panel["fieldConfig"]["defaults"]["custom"]["stacking"] == {
        "group": "A",
        "mode": "none",
    }
    assert panel["gridPos"] == {"h": 8, "w": 24, "x": 0, "y": 36}
    assert len(panel["targets"]) == 2

    expected = [
        (
            "GPU Prefix Cache",
            "ucm:gpu_hbm_hit_tokens_total",
            "ucm:total_prefix_query_tokens_total",
        ),
        (
            "Connector Prefix Cache",
            "ucm:ucm_hit_tokens_total",
            "ucm:total_prefix_query_tokens_total",
        ),
    ]
    for target, (legend, hits, queries) in zip(panel["targets"], expected):
        assert target["legendFormat"] == legend
        expr = target["expr"]
        assert hits in expr
        assert queries in expr
        assert "vllm:prefix_cache_queries_total" not in expr
        assert "vllm:prefix_cache_hits_total" not in expr
        assert "vllm:external_prefix_cache_hits_total" not in expr
        assert 'model_name="$model_name"' in expr
        assert 'job=~"$job"' in expr
        assert 'instance="$instance"' in expr
        assert "clamp_min" in expr


def test_vllm_dashboard_shows_time_range_token_and_prefix_totals():
    dashboard = json.loads(
        (REPO_ROOT / "examples" / "metrics" / "grafana_vllm.json").read_text(
            encoding="utf-8"
        )
    )
    panels = {panel["title"]: panel for panel in dashboard["panels"]}

    expected = [
        ("Total Input Tokens", "vllm:prompt_tokens_total", 0, 5),
        ("Total Output Tokens", "vllm:generation_tokens_total", 5, 5),
        ("Prefix Cache Query Tokens", "ucm:total_prefix_query_tokens_total", 10, 5),
        ("GPU/HBM Prefix Hit Tokens", "ucm:gpu_hbm_hit_tokens_total", 15, 5),
        ("UCM Prefix Hit Tokens", "ucm:ucm_hit_tokens_total", 20, 4),
    ]

    for title, metric, x, w in expected:
        panel = panels[title]
        assert panel["type"] == "stat"
        assert panel["gridPos"] == {"h": 4, "w": w, "x": x, "y": 0}
        assert panel["fieldConfig"]["defaults"]["unit"] == "short"
        assert panel["fieldConfig"]["defaults"]["decimals"] == 2
        assert len(panel["targets"]) == 1
        target = panel["targets"][0]
        assert target["instant"] is True
        assert target["range"] is False
        expr = target["expr"]
        assert expr == (
            f'sum(increase({metric}{{model_name="$model_name", '
            'job=~"$job", instance="$instance", engine=~"$engine"}[$__range]))'
        )
        assert "$__rate_interval" not in expr

    pie = panels["Prefix Cache Query Breakdown"]
    assert pie["type"] == "piechart"
    assert pie["gridPos"] == {"h": 8, "w": 24, "x": 0, "y": 4}
    assert pie["options"]["displayLabels"] == ["name", "value", "percent"]
    assert pie["options"]["legend"]["displayMode"] == "table"
    assert pie["options"]["legend"]["values"] == ["value", "percent"]
    pie_targets = {target["legendFormat"]: target["expr"] for target in pie["targets"]}
    assert set(pie_targets) == {
        "GPU/HBM Prefix Hit Tokens",
        "UCM Prefix Hit Tokens",
        "Misses",
    }
    assert pie_targets["GPU/HBM Prefix Hit Tokens"] == (
        'sum(increase(ucm:gpu_hbm_hit_tokens_total{model_name="$model_name", '
        'job=~"$job", instance="$instance", engine=~"$engine"}[$__range]))'
    )
    assert pie_targets["UCM Prefix Hit Tokens"] == (
        'sum(increase(ucm:ucm_hit_tokens_total{model_name="$model_name", '
        'job=~"$job", instance="$instance", engine=~"$engine"}[$__range]))'
    )
    assert "ucm:total_prefix_query_tokens_total" in pie_targets["Misses"]
    assert "ucm:gpu_hbm_hit_tokens_total" in pie_targets["Misses"]
    assert "ucm:ucm_hit_tokens_total" in pie_targets["Misses"]
    assert "vllm:prefix_cache_queries_total" not in pie_targets["Misses"]
    assert "vllm:external_prefix_cache_hits_total" not in pie_targets["Misses"]
    assert pie_targets["Misses"].startswith("clamp_min(")
    assert "$__rate_interval" not in pie_targets["Misses"]

    assert panels["E2E Request Latency"]["gridPos"]["y"] == 12


def test_vllm_dashboard_uses_engine_filter_and_aggregates_engine_series():
    dashboard = json.loads(
        (REPO_ROOT / "examples" / "metrics" / "grafana_vllm.json").read_text(
            encoding="utf-8"
        )
    )
    variables = {var["name"]: var for var in dashboard["templating"]["list"]}

    engine = variables["engine"]
    assert engine["includeAll"] is True
    assert engine["allValue"] == ".*"
    assert engine["current"] == {"selected": True, "text": "All", "value": "$__all"}
    assert (
        engine["definition"]
        == 'label_values(vllm:num_requests_running{job=~"$job", model_name="$model_name"}, engine)'
    )

    instance = variables["instance"]
    assert 'engine=~"$engine"' in instance["definition"]
    assert 'engine=~"$engine"' in instance["query"]["query"]

    panels = {panel["title"]: panel for panel in dashboard["panels"]}
    e2e_average = next(
        target
        for target in panels["E2E Request Latency"]["targets"]
        if target["legendFormat"] == "Average"
    )["expr"]
    assert "sum(rate(vllm:e2e_request_latency_seconds_sum" in e2e_average
    assert "sum(rate(vllm:e2e_request_latency_seconds_count" in e2e_average

    scheduler_exprs = {
        target["legendFormat"]: target["expr"]
        for target in panels["Scheduler State"]["targets"]
    }
    assert scheduler_exprs["Num Running"].startswith("sum(")
    assert scheduler_exprs["Num Waiting"].startswith("sum(")

    cache_usage_expr = panels["Cache Utilization"]["targets"][0]["expr"]
    assert cache_usage_expr.startswith("avg(")

    for panel in dashboard["panels"]:
        for target in panel.get("targets", []):
            expr = target.get("expr", "")
            if "vllm:" in expr:
                assert 'engine=~"$engine"' in expr


def test_grafana_dashboards_use_isolated_vllm_ucm_identity():
    expected = {
        "grafana_connector.json": (
            "vLLM - UCM Connector (vLLM Metrics)",
            "ucm-vllm-connector-overview",
        ),
        "grafana_layerwise.json": (
            "vLLM - UCM Layerwise (vLLM Metrics)",
            "ucm-vllm-layerwise",
        ),
        "grafana_pipeline_store.json": (
            "vLLM - UCM Cache / Posix Store (vLLM Metrics)",
            "ucm-vllm-pipeline-store",
        ),
        "grafana_mooncake.json": (
            "vLLM - UCM Mooncake Store (vLLM Metrics)",
            "ucm-vllm-mooncake-store",
        ),
        "grafana_vllm.json": (
            "vLLM (UCM Metrics)",
            "ucm-vllm-overview",
        ),
    }
    old_uids = {
        "ucm-connector-overview",
        "ucm-layerwise",
        "ucm-pipeline-store",
        "vllm-overview",
    }

    for filename, (title, uid) in expected.items():
        dashboard = json.loads(
            (REPO_ROOT / "examples" / "metrics" / filename).read_text(encoding="utf-8")
        )

        assert dashboard["title"] == title
        assert dashboard["uid"] == uid
        assert dashboard["id"] is None
        assert dashboard["uid"] not in old_uids
        assert GRAFANA_VLLM_UCM_TAG in dashboard["tags"]
        assert "ucm" not in dashboard["tags"]
        assert "(tag: ucm)" not in dashboard.get("description", "")
        if "other UCM dashboards" in dashboard.get("description", ""):
            assert f"(tag: {GRAFANA_VLLM_UCM_TAG})" in dashboard["description"]
        for link in dashboard.get("links", []):
            if link.get("type") == "dashboards":
                assert link["title"] == "Other UCM vLLM metrics dashboards"
                assert link["tags"] == [GRAFANA_VLLM_UCM_TAG]


def test_ucm_dashboards_use_engine_and_worker_rank_filters():
    for filename in GRAFANA_UCM_DASHBOARDS:
        dashboard_text = (REPO_ROOT / "examples" / "metrics" / filename).read_text(
            encoding="utf-8"
        )
        dashboard = json.loads(dashboard_text)
        variables = {item["name"]: item for item in dashboard["templating"]["list"]}
        variable_names = [item["name"] for item in dashboard["templating"]["list"]]

        assert variable_names.index("perWorker") < variable_names.index("engine")
        assert variables["perWorker"]["current"]["text"] == "Aggregated"
        assert variables["perWorker"]["current"]["value"] == "model_name"
        assert variables["perWorker"]["options"] == [
            {
                "selected": True,
                "text": "Aggregated",
                "value": "model_name",
            },
            {
                "selected": False,
                "text": "Per Worker",
                "value": "model_name, engine, worker_rank",
            },
        ]
        assert "${perWorker:raw}" in dashboard_text
        assert variables["engine"]["includeAll"] is True
        assert variables["engine"]["allValue"] == ".*"
        assert "label_values" in variables["engine"]["definition"]
        assert "engine" in variables["engine"]["definition"]
        assert variables["worker_rank"]["includeAll"] is True
        assert variables["worker_rank"]["allValue"] == ".*"
        assert 'engine=~"$engine"' in variables["worker_rank"]["definition"]

        exprs = [
            target["expr"]
            for panel in dashboard["panels"]
            for target in panel.get("targets", [])
            if "expr" in target
        ]
        legends = [
            target["legendFormat"]
            for panel in dashboard["panels"]
            for target in panel.get("targets", [])
            if "legendFormat" in target and "ucm:" in target.get("expr", "")
        ]
        assert legends
        for legend in legends:
            if legend == "{{le}}":
                continue
            assert "engine={{engine}}" not in legend
            assert "worker={{worker_rank}}" not in legend
            assert "Aggregated" not in legend
            assert "${perWorker:text}" not in legend
            assert "{{engine}}" in legend
            assert "{{worker_rank}}" in legend

        for expr in exprs:
            if "ucm:" not in expr:
                continue
            assert 'engine=~"$engine"' in expr
            assert 'worker_rank=~"$worker_rank"' in expr
            if "sum by (" in expr:
                assert (
                    "sum by (${perWorker:raw})" in expr
                    or "sum by (le, ${perWorker:raw})" in expr
                    or "sum by (le)" in expr
                )

        if filename == "grafana_connector.json":
            external_prefix_exprs = [
                expr for expr in exprs if "vllm:external_prefix_cache_" in expr
            ]
            assert external_prefix_exprs
            for expr in external_prefix_exprs:
                assert 'engine=~"$engine"' in expr
                assert 'worker_rank=~"$worker_rank"' not in expr
                assert "sum by (model_name)" in expr


def test_mooncake_dashboard_covers_configured_mooncake_metrics():
    dashboard_path = REPO_ROOT / "examples" / "metrics" / "grafana_mooncake.json"
    dashboard_text = dashboard_path.read_text(encoding="utf-8")
    dashboard = json.loads(dashboard_text)
    panels = {panel["title"]: panel for panel in dashboard["panels"]}

    assert dashboard["title"] == "vLLM - UCM Mooncake Store (vLLM Metrics)"
    assert dashboard["uid"] == "ucm-vllm-mooncake-store"
    assert "Mooncake Store" in panels
    assert "Mooncake Load Hit / Miss Shards" in panels
    assert "Mooncake Dump Existing / Missing Shards" in panels
    assert "Mooncake Load Stage Avg Breakdown" in panels
    assert "Mooncake Dump Stage Avg Breakdown" in panels
    assert "Mooncake Error Rate" in panels
    assert "mooncake_h2d_bandwidth_gbps" not in dashboard_text
    assert "mooncake_d2h_bandwidth_gbps" not in dashboard_text

    metrics_text = (
        REPO_ROOT / "examples" / "metrics" / "metrics_configs.yaml"
    ).read_text(encoding="utf-8")
    configured_mooncake = {
        name
        for name in re.findall(
            r'^\s*-\s+name:\s+"(mooncake_[^"]+)"', metrics_text, re.MULTILINE
        )
        if name not in {"mooncake_h2d_bandwidth_gbps", "mooncake_d2h_bandwidth_gbps"}
    }
    referenced = {
        re.sub(r"_(bucket|sum|count)$", "", metric)
        for metric in re.findall(r"ucm:(mooncake_[A-Za-z0-9_]+)", dashboard_text)
    }

    assert configured_mooncake <= referenced


def test_layerwise_dashboard_hides_no_transfer_and_uses_rate_interval_for_breakdown():
    dashboard = json.loads(
        (REPO_ROOT / "examples" / "metrics" / "grafana_layerwise.json").read_text(
            encoding="utf-8"
        )
    )
    panels = {panel["title"]: panel for panel in dashboard["panels"]}
    all_panels = []
    for panel in dashboard["panels"]:
        all_panels.append(panel)
        all_panels.extend(panel.get("panels", []))

    assert "Layerwise / Batch Total - No Transfer" not in panels
    assert all("No Transfer" not in panel["title"] for panel in all_panels)

    batch_mix = panels["Layerwise / Batch Mix Rate"]
    assert all(
        "layerwise_batch_total_no_transfer_ms" not in target["expr"]
        for target in batch_mix["targets"]
    )
    assert all(
        "no transfer" not in target.get("legendFormat", "")
        for target in batch_mix["targets"]
    )

    load_only = panels["Layerwise / Load Only Batch Avg Breakdown"]
    save_only = panels["Layerwise / Save Only Batch Avg Breakdown"]
    load_save = panels["Layerwise / Load + Save Batch Avg Breakdown"]
    for panel in (load_only, save_only, load_save):
        for target in panel["targets"]:
            assert "[40s]" not in target["expr"]
            assert "[$__rate_interval]" in target["expr"]


def test_layerwise_dashboard_uses_batch_duration_and_dump_completion_wait():
    dashboard = json.loads(
        (REPO_ROOT / "examples" / "metrics" / "grafana_layerwise.json").read_text(
            encoding="utf-8"
        )
    )
    all_panels = []
    for panel in dashboard["panels"]:
        all_panels.append(panel)
        all_panels.extend(panel.get("panels", []))
    panels = {panel["title"]: panel for panel in all_panels}

    assert "Layerwise / Batch Total Duration (all batches)" not in panels
    assert "Layerwise / wait_for_save Total (save batches only)" not in panels
    assert all("Batch Total" not in panel["title"] for panel in all_panels)

    assert "Layerwise / Batch Duration - Load Only" in panels
    assert "Layerwise / Batch Duration - Save Only" in panels
    assert "Layerwise / Batch Duration - Load + Save" in panels

    panel = panels["Layerwise / Dump Completion Wait Duration"]
    assert panel["gridPos"] == {"h": 8, "w": 12, "x": 0, "y": 64}
    assert all(
        "ucm:save_completion_wait_duration" in target["expr"]
        for target in panel["targets"]
    )
    assert all(
        "ucm:layerwise_save_tail_total_ms" not in target["expr"]
        for target in panel["targets"]
    )


def test_layerwise_wait_for_save_records_save_tail_and_completion_start():
    source = (
        REPO_ROOT / "ucm" / "integration" / "vllm" / "ucm_connector.py"
    ).read_text(encoding="utf-8")

    assert "save_tail_start = time.perf_counter()" in source
    assert "wait_for_save_start_ms = save_tail_start * 1000" in source
    assert "pending_dump_task.wait_for_save_start_ms = wait_for_save_start_ms" in source
    assert "save_tail_ms = (total_end - save_tail_start) * 1000" in source
    assert "self._layerwise_batch_stats(total_end, save_tail_ms)" in source
    assert 'stats["layerwise_save_tail_total_ms"] = save_tail_ms' in source


def test_cache_load_h2d_duration_records_stream_synchronize_only():
    header = (REPO_ROOT / "ucm" / "store" / "cache" / "cc" / "load_queue.h").read_text(
        encoding="utf-8"
    )
    source = (REPO_ROOT / "ucm" / "store" / "cache" / "cc" / "load_queue.cc").read_text(
        encoding="utf-8"
    )

    assert "double h2dBatchStartTp_{0.0};" not in header
    assert "firstH2dReadyTp" not in header
    assert "firstH2dReadyTp" not in source
    assert "double h2dBatchStartTp = 0.0;" not in source
    assert "double& h2dBatchStartTp" not in header
    assert "double& h2dBatchStartTp" not in source
    assert "h2dBatchStartTp_" not in source
    assert "auto tpH2dSyncStart = NowTime::Now();" in source
    assert "auto h2dSyncMs = (NowTime::Now() - tpH2dSyncStart) * 1e3;" in source
    assert "cache_h2d_bandwidth_gbps" not in source
    assert "auto h2dSyncMs = (NowTime::Now() - tpH2dSubmitted) * 1e3;" not in source


def test_cache_dump_d2h_metrics_require_event_ready_timestamp():
    source = (REPO_ROOT / "ucm" / "store" / "cache" / "cc" / "dump_queue.cc").read_text(
        encoding="utf-8"
    )

    stream_header = (REPO_ROOT / "ucm" / "shared" / "trans" / "stream.h").read_text(
        encoding="utf-8"
    )
    cuda_header = (
        REPO_ROOT / "ucm" / "shared" / "trans" / "cuda" / "cuda_stream.h"
    ).read_text(encoding="utf-8")
    cuda_source = (
        REPO_ROOT / "ucm" / "shared" / "trans" / "cuda" / "cuda_stream.cc"
    ).read_text(encoding="utf-8")
    ascend_header = (
        REPO_ROOT / "ucm" / "shared" / "trans" / "ascend" / "ascend_stream.h"
    ).read_text(encoding="utf-8")
    ascend_source = (
        REPO_ROOT / "ucm" / "shared" / "trans" / "ascend" / "ascend_stream.cc"
    ).read_text(encoding="utf-8")

    assert "struct StreamEventTimer" not in stream_header
    assert "RecordEventTimerStart" not in stream_header
    assert "RecordEventTimerEnd" not in stream_header
    assert "EventElapsedTimeMs" not in stream_header
    assert "DestroyEventTimer" not in stream_header

    assert "RecordEventTimerStart" not in cuda_header
    assert "RecordEventTimerEnd" not in cuda_header
    assert "EventElapsedTimeMs" not in cuda_header
    assert "DestroyEventTimer" not in cuda_header
    assert "CudaEventTimer" not in cuda_source
    assert "cudaEventCreate(&eventTimer->start)" not in cuda_source
    assert "cudaEventRecord(eventTimer->start, stream_)" not in cuda_source
    assert "cudaEventRecord(eventTimer->end, stream_)" not in cuda_source
    assert "cudaEventSynchronize(eventTimer->end)" not in cuda_source
    assert (
        "cudaEventElapsedTime(&elapsedMs, eventTimer->start, eventTimer->end)"
        not in cuda_source
    )

    assert "RecordEventTimerStart" not in ascend_header
    assert "RecordEventTimerEnd" not in ascend_header
    assert "EventElapsedTimeMs" not in ascend_header
    assert "DestroyEventTimer" not in ascend_header
    assert "AscendEventTimer" not in ascend_source
    assert "aclrtCreateEvent(&eventTimer->start)" not in ascend_source
    assert "aclrtRecordEvent(eventTimer->start, stream_)" not in ascend_source
    assert "aclrtRecordEvent(eventTimer->end, stream_)" not in ascend_source
    assert "aclrtSynchronizeEvent(eventTimer->end)" not in ascend_source
    assert (
        "aclrtEventElapsedTime(&elapsedMs, eventTimer->start, eventTimer->end)"
        not in ascend_source
    )

    assert "eventReadyTp->store(NowTime::Now(), std::memory_order_release);" in source
    assert "auto ready = eventReadyTp->load(std::memory_order_acquire);" in source
    assert "auto tpSyncStart = NowTime::Now();" in source
    assert "auto tpSyncStream = NowTime::Now();" in source
    assert "auto d2hMs = std::max(0.0, tpSyncStream - tpSyncStart) * 1e3;" in source
    assert "auto tpBackendSubmitStart = NowTime::Now();" in source
    assert "(tpEnd - tpBackendSubmitStart) * 1e3" in source
    assert 'NAME_TO_METRIC_ID("cache_d2h_callback_wait_ms")' not in source
    assert "auto dumpStartTp = NowTime::Now();" in source
    assert "std::shared_ptr<std::atomic<double>> d2hEndTp" not in source
    assert "auto endCbStatus = stream.AppendCallback" not in source
    assert "d2hEndTp" not in source
    assert "auto tp = NowTime::Now();" not in source
    assert "auto d2hStartTp = tpMakeBuffer;" not in source
    assert "d2hStartTp = std::max(d2hStartTp, ready);" not in source
    assert "tpSyncStream - ready" not in source
    assert "end - ready" not in source
    assert "Trans::StreamEventTimer d2hEventTimer" not in source
    assert "RecordEventTimerStart" not in source
    assert "RecordEventTimerEnd" not in source
    assert "EventElapsedTimeMs" not in source
    assert "DestroyEventTimer" not in source
    assert "auto copyStream = stream.NextStream();" not in source
    assert "DeviceToHostAsync(stream, shard.addrs.data(), host)" in source
    assert "if (eventReadyTp && copiedShards > 0)" not in source
    assert "cache_d2h_bandwidth_gbps" not in source

    d2h_duration_pos = source.index('NAME_TO_METRIC_ID("cache_d2h_duration_ms")')
    sync_start_pos = source.index("auto tpSyncStart = NowTime::Now()")
    sync_pos = source.index("auto s = stream.Synchronize()")
    sync_end_pos = source.index("auto tpSyncStream = NowTime::Now()")
    d2h_ms_pos = source.index(
        "auto d2hMs = std::max(0.0, tpSyncStream - tpSyncStart) * 1e3;"
    )
    backend_submit_start_pos = source.index(
        "auto tpBackendSubmitStart = NowTime::Now()"
    )
    prereq_block_pos = source.index("if (eventReadyTp)")
    ready_load_pos = source.index("auto ready = eventReadyTp->load", prereq_block_pos)
    backend_submit_pos = source.index(
        'NAME_TO_METRIC_ID("cache_dump_backend_submit_duration_ms")'
    )

    assert sync_start_pos < sync_pos < sync_end_pos < backend_submit_start_pos
    assert sync_end_pos < d2h_ms_pos < d2h_duration_pos < backend_submit_pos
    assert prereq_block_pos < ready_load_pos < backend_submit_pos


def test_h2d_d2h_bandwidth_metrics_are_not_configured_or_recorded():
    metrics_text = (
        REPO_ROOT / "examples" / "metrics" / "metrics_configs.yaml"
    ).read_text(encoding="utf-8")
    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            REPO_ROOT / "ucm" / "store" / "cache" / "cc" / "load_queue.cc",
            REPO_ROOT / "ucm" / "store" / "cache" / "cc" / "dump_queue.cc",
            REPO_ROOT / "ucm" / "store" / "mooncakestore" / "cc" / "load_queue.cc",
            REPO_ROOT / "ucm" / "store" / "mooncakestore" / "cc" / "dump_queue.cc",
            REPO_ROOT
            / "ucm"
            / "store"
            / "mooncakestore"
            / "cc"
            / "share_load_queue.cc",
        )
    )
    removed_metrics = {
        "cache_h2d_bandwidth_gbps",
        "cache_d2h_bandwidth_gbps",
        "mooncake_h2d_bandwidth_gbps",
        "mooncake_d2h_bandwidth_gbps",
    }

    for metric in removed_metrics:
        assert metric not in metrics_text
        assert metric not in sources


def test_mooncake_store_links_metrics_target_for_metrics_api_includes():
    mooncake_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (REPO_ROOT / "ucm" / "store" / "mooncakestore" / "cc").glob("*.cc")
    )
    cmake_text = (
        REPO_ROOT / "ucm" / "store" / "mooncakestore" / "CMakeLists.txt"
    ).read_text(encoding="utf-8")

    assert '#include "metrics_api.h"' in mooncake_sources
    assert re.search(
        r"target_link_libraries\(\s*mooncakestore\s+PUBLIC[\s\S]*?\bmetrics\b",
        cmake_text,
    )


def test_pipeline_dashboard_orders_cache_bandwidth_rows():
    dashboard = json.loads(
        (REPO_ROOT / "examples" / "metrics" / "grafana_pipeline_store.json").read_text(
            encoding="utf-8"
        )
    )
    panels = {panel["title"]: panel for panel in dashboard["panels"]}

    assert panels["Cache Load Bandwidth (aggregated)"]["gridPos"]["y"] == 33
    assert panels["Cache Dump Bandwidth (aggregated)"]["gridPos"]["y"] == 33
    assert panels["Cache Load Bandwidth (per task)"]["gridPos"]["y"] == 41
    assert panels["Cache Dump Bandwidth (per task)"]["gridPos"]["y"] == 41
    assert "Cache Load H2D Bandwidth (per task)" not in panels
    assert "Cache Dump D2H Bandwidth (per task)" not in panels
    assert panels["Cache Load Backend Wait Duration"]["gridPos"] == {
        "h": 8,
        "w": 12,
        "x": 0,
        "y": 65,
    }
    assert panels["Cache Dump Wait Compute Duration"]["gridPos"] == {
        "h": 8,
        "w": 12,
        "x": 12,
        "y": 65,
    }
    assert panels["Cache Load H2D Duration"]["gridPos"]["y"] == 73
    assert "Cache Dump D2H Duration (include wait compute)" in panels
    assert "Cache Dump D2H Duration" not in panels
