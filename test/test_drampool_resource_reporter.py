#
# MIT License
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
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

"""Tests for the DramPool snapshot resource reporter."""

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class FakeMetrics:
    def __init__(self):
        self.clear()

    def clear(self):
        self.counters = {}
        self.gauges = {}
        self.histograms = {}

    def update_stats(self, values):
        for name, value in values.items():
            if name.endswith("_total"):
                self.counters[name] = self.counters.get(name, 0) + value
            else:
                self.gauges[name] = value

    def merge_histogram_stats(self, values):
        for name, (counts, total) in values.items():
            current = self.histograms.get(name)
            if current is None:
                self.histograms[name] = (list(counts), total)
                continue
            current_counts, current_total = current
            self.histograms[name] = (
                [left + right for left, right in zip(current_counts, counts)],
                current_total + total,
            )

    def get_all_stats_and_clear(self):
        result = (self.counters, self.gauges, self.histograms)
        self.clear()
        return result


native = FakeMetrics()
logger_module = ModuleType("ucm.logger")
logger_module.init_logger = logging.getLogger
sys.modules["ucm.logger"] = logger_module
package = ModuleType("ucm")
package.__path__ = [str(ROOT / "ucm")]
sys.modules["ucm"] = package
sys.modules["ucm.shared.metrics.ucmmetrics"] = native
import ucm.shared.metrics

ucm.shared.metrics.ucmmetrics = native
from ucm.store.dram import resource_reporter as reporter

NAME = "drampool_load_duration_ms"
COUNTER = "drampool_load_requests_total"
GAUGE = "drampool_used_bytes"


@pytest.fixture(autouse=True)
def metrics():
    native.clear()
    yield
    native.clear()


def record(tick=41, counts=None, total=12000, counter=36):
    counts = [10, 20, 5, 1] if counts is None else counts
    return {
        "event": "drampool_metrics_snapshot",
        "timestamp": 1788825600 + tick,
        "counters": {COUNTER: counter},
        "gauges": {GAUGE: 4096},
        "histograms": {
            NAME: {
                "upper_bounds": [100, 500, 1000],
                "bucket_counts": counts,
                "count": sum(counts),
                "sum": total,
                "unit": "ms",
            }
        },
    }


def parse(value):
    return reporter.parse_drampool_resource_snapshot(json.dumps(value))


def test_snapshot_difference_is_published_to_metrics():
    previous = parse(record())
    current = parse(record(43, counts=[12, 23, 6, 1], total=13900, counter=42))
    counters, gauges, histograms = reporter.snapshot_deltas(current, previous)
    assert histograms == {NAME: ([2, 3, 1, 0], 1900)}
    native.merge_histogram_stats(histograms)
    native.update_stats(counters | gauges)
    got_counter, got_gauge, got_hist = native.get_all_stats_and_clear()
    assert got_counter[COUNTER] == 6
    assert got_gauge[GAUGE] == 4096
    assert got_hist[NAME] == ([2, 3, 1, 0], 1900)
    assert reporter.snapshot_deltas(current, current)[2][NAME] == (
        [0, 0, 0, 0],
        0,
    )
    assert reporter.snapshot_deltas(previous, current)[2][NAME] == (
        [10, 20, 5, 1],
        12000,
    )


def test_baseline_restart_and_large_integer_counter():
    previous = parse(record(counter=2**60))
    assert reporter.snapshot_deltas(previous, None)[2][NAME] == (
        [0, 0, 0, 0],
        0,
    )
    current = parse(record(42, counter=2**60 + 1))
    assert reporter.snapshot_deltas(current, previous)[0][COUNTER] == 1
    reset = parse(record(1, counts=[1, 0, 0, 0], total=50, counter=2))
    assert reporter.snapshot_deltas(reset, previous)[2][NAME] == (
        [1, 0, 0, 0],
        50,
    )
    assert reporter.snapshot_deltas(reset, previous)[0][COUNTER] == 2
    # A reset that has already exceeded the old totals cannot be detected.
    assert (
        reporter.snapshot_deltas(parse(record(counter=2**60 + 100)), previous)[0][
            COUNTER
        ]
        == 100
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("bucket_counts", [-1, 20, 5, 1]),
        ("bucket_counts", [10.5, 20, 5, 1]),
        ("bucket_counts", [2**64, 20, 5, 1]),
        ("count", 37),
        ("upper_bounds", [100, 100, 1000]),
        ("sum", float("nan")),
        ("sum", -1),
    ],
)
def test_invalid_histogram(field, value):
    data = record()
    data["histograms"][NAME][field] = value
    with pytest.raises(ValueError):
        parse(data)


def test_histogram_decrease_resets_entire_distribution():
    current = parse(record(counts=[9, 22, 5, 1], total=13000))
    assert reporter.snapshot_deltas(current, parse(record()))[2][NAME] == (
        [9, 22, 5, 1],
        13000,
    )


def test_histogram_same_counts_with_changed_sum_is_a_reset():
    current = parse(record(counts=[10, 20, 5, 1], total=12001))
    assert reporter.snapshot_deltas(current, parse(record()))[2][NAME] == (
        [10, 20, 5, 1],
        12001,
    )


def make_reporter(tmp_path, monkeypatch):
    reader = reporter.DramPoolResourceReporter(
        str(tmp_path / "metrics.log"), shared_memory_dir=str(tmp_path)
    )
    reader.state_path = tmp_path / "state.json"
    monkeypatch.setattr(reader, "_try_become_leader", lambda: True)
    return reader


def write_record(reader, data, tail=b""):
    reader.log_path.write_bytes(json.dumps(data).encode() + b"\n" + tail)


def test_file_baseline_partial_tail_rotation_and_state(tmp_path, monkeypatch):
    reader = make_reporter(tmp_path, monkeypatch)
    write_record(reader, record(), b'{"incomplete":')
    reader._collect_once()
    native.get_all_stats_and_clear()
    reader.log_path.rename(tmp_path / "old.log")
    write_record(reader, record(43, counts=[12, 23, 6, 1], total=13900, counter=42))
    reader._collect_once()
    assert native.get_all_stats_and_clear()[2][NAME] == ([2, 3, 1, 0], 1900)
    reader._collect_once()
    assert native.get_all_stats_and_clear()[2][NAME] == ([0, 0, 0, 0], 0)
    assert reader._read_state() == parse(
        record(43, counts=[12, 23, 6, 1], total=13900, counter=42)
    )


def test_restart_with_same_histogram_counts_advances_state(tmp_path, monkeypatch):
    reader = make_reporter(tmp_path, monkeypatch)
    previous = record()
    write_record(reader, previous)
    reader._collect_once()
    native.get_all_stats_and_clear()

    current = record(43, total=12001)
    write_record(reader, current)
    reader._collect_once()
    assert native.get_all_stats_and_clear()[2][NAME] == (
        current["histograms"][NAME]["bucket_counts"],
        current["histograms"][NAME]["sum"],
    )
    assert reader._read_state() == parse(current)

    reader._collect_once()
    assert native.get_all_stats_and_clear()[2][NAME] == ([0, 0, 0, 0], 0)


def test_state_write_failure_reuses_persisted_baseline(tmp_path, monkeypatch):
    reader = make_reporter(tmp_path, monkeypatch)
    write_record(reader, record())
    reader._collect_once()
    native.get_all_stats_and_clear()
    write_record(reader, record(43, counts=[12, 23, 6, 1], total=13900, counter=42))

    def fail(snapshot):
        raise OSError("disk full")

    monkeypatch.setattr(reader, "_write_state", fail)
    reader._collect_once()
    assert native.get_all_stats_and_clear()[2][NAME][0] == [2, 3, 1, 0]
    reader._collect_once()
    assert native.get_all_stats_and_clear()[2][NAME][0] == [2, 3, 1, 0]
    assert reader._read_state() == parse(record())


def test_empty_and_oversized_file(tmp_path, monkeypatch):
    reader = make_reporter(tmp_path, monkeypatch)
    for content in [b"", b"x" * (2 * reporter.MAX_RECORD_BYTES + 1)]:
        reader.log_path.write_bytes(content)
        with pytest.raises(ValueError):
            reader._read_latest_complete_line()


def test_scalar_failure_leaves_state_for_next_round(tmp_path, monkeypatch):
    reader = make_reporter(tmp_path, monkeypatch)
    write_record(reader, record())
    reader._collect_once()
    native.get_all_stats_and_clear()
    write_record(reader, record(43, counts=[12, 23, 6, 1], total=13900, counter=42))
    update = native.update_stats

    def fail(values):
        if COUNTER in values:
            raise RuntimeError("temporary scalar failure")
        update(values)

    monkeypatch.setattr(native, "update_stats", fail)
    reader._collect_once()
    assert native.get_all_stats_and_clear()[2][NAME][0] == [2, 3, 1, 0]
    monkeypatch.setattr(native, "update_stats", update)
    reader._collect_once()
    c, _, h = native.get_all_stats_and_clear()
    assert c[COUNTER] == 6 and h[NAME][0] == [2, 3, 1, 0]
    assert reader._read_state().timestamp == record(43)["timestamp"]


def test_real_prometheus_export(monkeypatch):
    """Real reporter delta -> C++ -> dispatcher -> connector -> Prometheus text."""
    from dataclasses import dataclass, field
    from functools import partial
    from types import SimpleNamespace

    prometheus = pytest.importorskip("prometheus_client")
    registry = prometheus.CollectorRegistry()
    vllm = ModuleType("vllm")
    vllm.__path__ = []
    config_module = ModuleType("vllm.config")
    config_module.VllmConfig = object
    bridge = ModuleType("vllm.distributed.kv_transfer.kv_connector.v1.metrics")

    @dataclass
    class Stats:
        data: dict = field(default_factory=dict)

    class PromMetrics:
        def __init__(self, config, metric_types, labelnames, per_engine_labelvalues):
            self._counter_cls = partial(prometheus.Counter, registry=registry)
            self._gauge_cls = partial(prometheus.Gauge, registry=registry)
            self._histogram_cls = partial(prometheus.Histogram, registry=registry)
            self.per_engine_labelvalues = per_engine_labelvalues

    bridge.KVConnectorStats = Stats
    bridge.KVConnectorPromMetrics = PromMetrics
    bridge.PromMetric = bridge.PromMetricT = object
    for name, module in [
        ("vllm", vllm),
        ("vllm.config", config_module),
        (bridge.__name__, bridge),
    ]:
        monkeypatch.setitem(sys.modules, name, module)
    from ucm.integration.vllm.metrics import UCMConnectorStats, UCMPromMetrics
    from ucm.metrics_config import get_metric_definitions
    from ucm.metrics_dispatcher import MetricsDispatcher

    config = {
        "consumers": {"vllm_connector": True},
        "histogram": [
            {
                "name": NAME,
                "buckets": [100, 500, 1000],
                "vllm_connector_name": "drampool_load_duration_seconds",
                "vllm_connector_value_scale": 1e-3,
            }
        ],
    }
    delta = reporter.snapshot_deltas(
        parse(record(43, counts=[12, 23, 6, 1], total=13900)), parse(record())
    )
    native.merge_histogram_stats(delta[2])
    dispatcher = MetricsDispatcher(config)
    dispatcher.drain_to_consumers()
    c, g, h = dispatcher.get_stats_and_clear("vllm_connector")
    stats = UCMConnectorStats.from_ucm_snapshot(
        c,
        g,
        h,
        "scheduler",
        get_metric_definitions(config),
    )
    vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(launch_config={"metrics_config": config})
    )
    prom = UCMPromMetrics(vllm_config, {}, ["model_name"], {0: ["test-model"]})
    prom.observe(stats.data)
    samples = [s for metric in registry.collect() for s in metric.samples]
    buckets = [s for s in samples if s.name.endswith("_bucket")]
    assert [s.value for s in buckets] == [2, 5, 6, 6]
    assert next(s.value for s in samples if s.name.endswith("_sum")) == pytest.approx(
        1.9
    )
    assert next(s.value for s in samples if s.name.endswith("_count")) == 6


@pytest.mark.skipif(os.name != "posix", reason="real flock election requires POSIX")
def test_real_flock_excludes_other_processes(tmp_path):
    readers = [
        reporter.DramPoolResourceReporter(
            str(tmp_path / "metrics.log"),
            shared_memory_dir=str(tmp_path),
        )
        for _ in range(2)
    ]
    try:
        assert readers[0]._try_become_leader()
        # An independent process must also be excluded by the actual OS lock.
        probe = """
import fcntl, sys
with open(sys.argv[1], 'a+') as lock:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(7)
"""
        lock_path = readers[0]._lock_file.name
        assert subprocess.run([sys.executable, "-c", probe, lock_path]).returncode == 7
        assert not readers[1]._try_become_leader()
        readers[0]._lock_file.close()
        readers[0]._lock_file = None
        assert subprocess.run([sys.executable, "-c", probe, lock_path]).returncode == 0
        assert readers[1]._try_become_leader()
    finally:
        for reader in readers:
            if reader._lock_file:
                reader._lock_file.close()


@pytest.mark.parametrize(
    "overrides",
    [
        {"drampool_resource_log_path": ""},
        {"device_id": 0},
        {"drampool_resource_metrics_enable": False},
    ],
)
def test_reporter_role_and_enable_guards(overrides):
    assert (
        reporter.start_drampool_resource_reporter(
            {"drampool_resource_log_path": "/not-opened/metrics.log", **overrides}
        )
        is None
    )


def test_reporter_failure_updates_drampool_error_counter(tmp_path, monkeypatch):
    reader = make_reporter(tmp_path, monkeypatch)

    def fail_election():
        raise OSError("lock unavailable")

    monkeypatch.setattr(reader, "_try_become_leader", fail_election)
    reader._run()

    counters, _, _ = native.get_all_stats_and_clear()
    assert counters["drampool_resource_log_read_errors_total"] == 1


@pytest.mark.skipif(os.name != "posix", reason="reporter lifecycle uses POSIX flock")
def test_reporter_thread_elects_once_and_loser_exits(tmp_path, monkeypatch):
    readers = [
        reporter.DramPoolResourceReporter(
            str(tmp_path / "metrics.log"),
            interval_sec=1,
            shared_memory_dir=str(tmp_path),
        )
        for _ in range(2)
    ]

    def wait_until(predicate):
        deadline = time.monotonic() + 5
        while not predicate() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert predicate()

    try:
        readers[0].start()
        wait_until(lambda: readers[0]._lock_file is not None)
        write_record(readers[0], record())
        wait_until(lambda: readers[0].state_path and readers[0].state_path.exists())
        readers[1].start()
        readers[1]._thread.join(timeout=5)
        assert not readers[1]._thread.is_alive()
        assert readers[1]._lock_file is None
        readers[0].stop()
        assert not readers[0]._thread.is_alive()
        assert readers[0]._lock_file is None
        # Stopping the leader does not restart the losing reporter.
        assert not readers[1]._thread.is_alive()
        assert readers[1]._lock_file is None
    finally:
        for reader in readers:
            reader.stop()
