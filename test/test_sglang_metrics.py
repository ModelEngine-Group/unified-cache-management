"""Test real multiprocess Prometheus export without native UCM, SGLang or GPUs.

Run with pytest --noconftest test/test_sglang_metrics.py. Each case uses a fresh
interpreter because Prometheus selects its storage mode at import time.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = r"""
import logging, os, sys, time, types
from pathlib import Path
import yaml

def module(name, **attrs):
    result = types.ModuleType(name)
    result.__dict__.update(attrs)
    sys.modules[name] = result

module("ucm", __path__=[str(Path.cwd() / "ucm")])
module("ucm.shared", __path__=[str(Path.cwd() / "ucm/shared")])
module("ucm.logger", init_logger=logging.getLogger)
snapshot = ({}, {}, {})
registered, events = [], []
drains = 0
def drain():
    global snapshot, drains
    result, snapshot = snapshot, ({}, {}, {})
    drains += 1
    return result
module("ucm.shared.metrics", ucmmetrics=types.SimpleNamespace(
    set_up=lambda: None, create_stats=lambda *a: registered.append(a),
    get_all_stats_and_clear=drain))
module("torch")
for name in ("sglang", "sglang.srt", "sglang.srt.distributed"):
    module(name, __path__=[])
module("sglang.srt.distributed.parallel_state",
       get_world_group=lambda: types.SimpleNamespace(local_rank=0))
class Store:
    def close(self):
        events.append("store.close")
def create(*args):
    events.append("store.create")
    return Store()
module("ucm.store.factory_v1", UcmConnectorFactoryV1=types.SimpleNamespace(create_connector=create))
from ucm.integration.sglang import ucm_connector as c

config = {"log_interval": 0.02,
    "consumers": {"multiproc": True, "vllm_connector": False},
    "counter": [{"name": "posix_s2h_bytes_total"}],
    "gauge": [{"name": "posix_store_used_bytes", "multiprocess_mode": "livemostrecent"}],
    "histogram": [{"name": "posix_load_task_duration_ms", "buckets": [1, 10]}]}
config_path = Path(sys.argv[1]) / "metrics.yaml"
def write_config():
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
write_config()
launch = {"metrics_config_path": str(config_path), "ucm_connector_name": "FakeStore",
          "ucm_connector_config": {"storage_backends": "/tmp/fake-store"}}
storage = types.SimpleNamespace(model_name="test-model", is_mla_model=False,
    tp_rank=0, tp_size=2, pp_rank=0, pp_size=1, dp_rank=0, attn_cp_rank=0,
    enable_storage_metrics=True, extra_config={"kv_connector_extra_config": launch})
pool = types.SimpleNamespace(dtype="f16", page_size=1, kv_buffer=object(),
                             get_size_per_token=lambda: 128)
def start():
    return c.SglangUcmConnector.from_hicache(storage, pool)
def scrape():
    from prometheus_client import CollectorRegistry, generate_latest, multiprocess
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    return generate_latest(registry).decode()
def value(name, **labels):
    from prometheus_client.parser import text_string_to_metric_families
    matches = [s.value for f in text_string_to_metric_families(scrape()) for s in f.samples
               if s.name == name and all(s.labels.get(k) == v for k, v in labels.items())]
    assert len(matches) == 1, (name, labels, matches)
    return matches[0]
"""


def run_child(tmp_path, code, *, multiprocess=True):
    env = os.environ.copy()
    env.pop("prometheus_multiproc_dir", None)
    env.pop("PROMETHEUS_MULTIPROC_DIR", None)
    if multiprocess:
        env["PROMETHEUS_MULTIPROC_DIR"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, "-c", BOOTSTRAP + "\n" + textwrap.dedent(code), str(tmp_path)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_two_workers_existing_thread_and_asgi_endpoint(tmp_path):
    for rank, amount in [(0, 4096), (1, 8192)]:
        run_child(
            tmp_path,
            f"""
            storage.tp_rank = {rank}
            def create_with_metrics(*args):
                global snapshot
                assert registered, "Register metrics before creating stores"
                snapshot = ({{"posix_s2h_bytes_total": {amount}}},
                            {{"posix_store_used_bytes": {amount}}},
                            {{"posix_load_task_duration_ms": ([1, 1, 0], 6.0)}})
                return Store()
            c.UcmConnectorFactoryV1.create_connector = create_with_metrics
            connector = start()
            try:
                from prometheus_client import Counter
                Counter("sglang:generated_tokens_total", "Generated tokens").inc(3)
                deadline = time.monotonic() + 5
                while (snapshot[0] or drains < 3) and time.monotonic() < deadline:
                    time.sleep(0.01)
                assert not snapshot[0] and drains >= 3
                assert value("ucm:posix_s2h_bytes_total", worker_id="dp0_tp{rank}_pp0_attn_cp0") == {amount}
                try:
                    start()
                except RuntimeError as error:
                    assert "only one" in str(error)
                else:
                    raise AssertionError("Duplicate logger accepted")
            finally:
                connector.close()
            assert not connector._metrics_logger.thread.is_alive()
        """,
        )
    assert len(list(tmp_path.glob("counter_*.db"))) == 2
    run_child(
        tmp_path,
        """
        import asyncio
        from prometheus_client import CollectorRegistry, make_asgi_app, multiprocess
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        responses = []
        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}
        async def send(message):
            responses.append(message)
        asyncio.run(make_asgi_app(registry=registry)(
            {"type": "http", "method": "GET", "path": "/metrics", "headers": [],
             "query_string": b""}, receive, send))
        assert responses[0]["status"] == 200
        text = b"".join(r.get("body", b"") for r in responses).decode()
        assert "sglang:generated_tokens_total 6.0" in text
        assert "ucm:posix_s2h_bytes_total" in text
        worker = "dp0_tp1_pp0_attn_cp0"
        assert value("ucm:posix_s2h_bytes_total", worker_id=worker) == 8192
        assert value("ucm:posix_store_used_bytes", worker_id=worker) == 8192
        assert value("ucm:posix_load_task_duration_ms_count", worker_id=worker) == 2
        assert value("ucm:posix_load_task_duration_ms_sum", worker_id=worker) == 6
        assert value("ucm:posix_load_task_duration_ms_bucket", worker_id=worker, le="10.0") == 2
        assert scrape() == scrape()
    """,
    )


@pytest.mark.parametrize("disable", ["sglang", "ucm"])
def test_disabled_does_not_import_prometheus(tmp_path, disable):
    run_child(
        tmp_path,
        f"""
        if {disable!r} == "sglang":
            storage.enable_storage_metrics = False
        else:
            launch["enable_metrics"] = False
        launch.pop("metrics_config_path")
        connector = start()
        assert connector._metrics_logger is None
        assert "prometheus_client" not in sys.modules
        connector.close()
    """,
        multiprocess=False,
    )


@pytest.mark.parametrize("early_import", [False, True])
def test_multiprocess_environment_errors(tmp_path, early_import):
    code = ""
    if early_import:
        code = f'from prometheus_client import Counter\nos.environ["PROMETHEUS_MULTIPROC_DIR"] = {str(tmp_path)!r}\n'
    run_child(
        tmp_path,
        code
        + """
try:
    start()
except RuntimeError as error:
    assert "PROMETHEUS_MULTIPROC_DIR" in str(error)
else:
    raise AssertionError("Invalid multiprocess environment accepted")
assert events == []
""",
        multiprocess=False,
    )


def test_vllm_only_yaml_rejected(tmp_path):
    run_child(
        tmp_path,
        """
        config["consumers"] = {"vllm_connector": True}
        write_config()
        try:
            start()
        except ValueError as error:
            assert "enable multiproc" in str(error)
        else:
            raise AssertionError("Wrong consumer config accepted")
        assert events == []
    """,
    )


@pytest.mark.parametrize("contents", [None, "{}", "counter: ["])
def test_invalid_explicit_config_does_not_fall_back(tmp_path, contents):
    run_child(
        tmp_path,
        f"""
        config_path.unlink()
        if {contents!r} is not None:
            config_path.write_text({contents!r}, encoding="utf-8")
        try:
            start()
        except ValueError as error:
            assert "No UCM metrics defined" in str(error)
        else:
            raise AssertionError("Invalid explicit config silently replaced by defaults")
        assert events == []
        """,
    )


def test_lifecycle_creation_failure_hybrid_pools_and_close_failure(tmp_path):
    run_child(
        tmp_path,
        """
        class Logger:
            def shutdown(self):
                events.append("shutdown")
        def create_logger(*args):
            events.append("logger.create")
            return Logger()
        c._create_metrics_logger = create_logger
        connector = start()
        assert events == ["logger.create", "store.create"]
        connector.pool_components = {"mamba": [(Store(), 1)], "swa": [(Store(), 1)]}
        connector.close()
        connector.close()
        assert events == ["logger.create", "store.create", "store.close",
                          "store.close", "store.close", "shutdown"]
        events.clear()
        def fail(*args):
            raise ValueError("store creation failed")
        c.UcmConnectorFactoryV1.create_connector = fail
        try:
            start()
        except ValueError:
            pass
        else:
            raise AssertionError("Expected store creation failure")
        assert events == ["logger.create", "shutdown"]
        events.clear()
        pool.kv_buffer = None
        connector = start()
        assert events == ["logger.create"]
        class BrokenStore:
            def close(self):
                events.append("broken.close")
                raise RuntimeError("close failed")
        connector.pool_components = {"pool": [(BrokenStore(), 1), (Store(), 1)]}
        try:
            connector.close()
        except RuntimeError:
            pass
        else:
            raise AssertionError("Expected store close failure")
        assert events == ["logger.create", "broken.close", "store.close", "shutdown"]
    """,
    )


IO_BOOTSTRAP = r"""
import bisect, threading
from ucm.integration.sglang import metrics as io_metrics
from ucm.shared.metrics import ucmmetrics

config = yaml.safe_load(Path("examples/metrics/metrics_configs_sglang.yaml").read_text(encoding="utf-8"))
assert config["consumers"] == {"multiproc": True, "vllm_connector": False}
config["log_interval"] = 0.01
write_config()
lock = threading.Lock()
observations = []
defs = {m["name"]: (typ, m) for typ in ("counter", "gauge", "histogram") for m in config[typ]}
def update(values):
    with lock:
        observations.append(dict(values))
        for name, v in values.items():
            typ, definition = defs[name]
            if typ == "counter":
                snapshot[0][name] = snapshot[0].get(name, 0) + v
            elif typ == "histogram":
                bounds = definition["buckets"] + [float("inf")]
                counts, total = snapshot[2].get(name, ([0] * len(bounds), 0))
                counts[bisect.bisect_left(bounds, v)] += 1
                snapshot[2][name] = (counts, total + v)
def io_drain():
    with lock:
        return drain()
ucmmetrics.update_stats = update
ucmmetrics.get_all_stats_and_clear = io_drain

class IoStore(Store):
    phase = None
    def load_data(self, *args):
        if self.phase == "submit":
            raise RuntimeError("submit failed")
        return object()
    dump_data = load_data
    def wait(self, task):
        if self.phase == "wait":
            raise RuntimeError("wait failed")
store = IoStore()
def make_connector():
    c.UcmConnectorFactoryV1.create_connector = lambda *args: store
    connector = start()
    connector._generate_task = lambda keys, indices: (keys, [0] * len(keys), [[1, 2]] * len(keys))
    return connector
def clock_calls(count):
    # Exactly 10ms per call, independent of the periodic export thread.
    ticks = iter(v for i in range(count) for v in (i, i + 0.01))
    io_metrics.time = types.SimpleNamespace(perf_counter=lambda: next(ticks))
"""


def test_builtin_config_exports_without_yaml_and_preserves_vllm(tmp_path):
    run_child(
        tmp_path,
        IO_BOOTSTRAP
        + r"""
from copy import deepcopy
from ucm.default_metrics_config import DEFAULT_METRICS_CONFIG, get_default_metrics_config
from ucm.integration.sglang.default_metrics_config import get_default_sglang_metrics_config
import ucm.metrics_config as metrics_config

baseline = deepcopy(DEFAULT_METRICS_CONFIG)
expected = yaml.safe_load(Path("examples/metrics/metrics_configs_sglang.yaml").read_text(encoding="utf-8"))
assert get_default_sglang_metrics_config() == expected
modified = get_default_sglang_metrics_config()
modified["counter"][0]["name"] = "changed"
modified["histogram"][-1]["buckets"].append(999999)
modified["consumers"]["multiproc"] = False
assert get_default_sglang_metrics_config() == expected
assert DEFAULT_METRICS_CONFIG == baseline == get_default_metrics_config()

launch.pop("metrics_config_path")
config_path.unlink()
def reject_yaml(*args, **kwargs):
    raise AssertionError("Built-in metrics must not read YAML")
metrics_config.load_metrics_config = reject_yaml
yaml.safe_load = reject_yaml
connector = make_connector()
try:
    assert connector._metrics_logger.config == expected
    clock_calls(1)
    assert connector.batch_get_v1(["a"], None) == [True]
    deadline = time.monotonic() + 7
    while time.monotonic() < deadline:
        try:
            if value("ucm:sglang_connector_load_bytes_total") == 128:
                break
        except AssertionError:
            pass
        time.sleep(0.02)
    assert value("ucm:sglang_connector_load_requests_total") == 1
    assert value("ucm:sglang_connector_load_bytes_total") == 128
    assert value("ucm:sglang_connector_load_pages_count") == 1
    assert DEFAULT_METRICS_CONFIG == baseline
finally:
    connector.close()
""",
    )


def test_linker_async_adapter_exports_through_shared_multiprocess_registry(tmp_path):
    run_child(
        tmp_path,
        IO_BOOTSTRAP
        + r"""
import numpy as np
module("ucm.store.ucmstore_v1", Task=object, UcmKVStoreBaseV1=object)
from ucm.integration.sglang.ucm_direct_store_adapter import UcmDirectStoreAdapter, UcmLayerTransfer

metrics_logger = io_metrics.create_metrics_logger(
    enabled=True, model_name="linker-model",
    ranks={"dp_rank": 1, "tp_rank": 2, "pp_rank": 3, "attn_cp_rank": 0},
    launch_config=launch)
adapter = UcmDirectStoreAdapter(connector_name="test", module_path=None, store_config={}, identity={})
adapter.metrics_enabled = True
adapter._ensure_store_schema = lambda *a, **kw: store
transfer = UcmLayerTransfer(
    store_group="kv", source_pool="kv", pools=("kv",), layers=(0, 1),
    shard_size=8, tensor_size_list=(8,), page_keys=("a", "b"),
    addresses=np.array([[[100],[108]],[[200],[208]]], dtype=np.uint64),
    fragments=((8,0),(8,0)))
operation = io_metrics.LinkerOperation.begin(True, "offload")
operation.start()
tasks = adapter.submit_offload([transfer])
assert not any("sglang_connector_dump_requests_total" in s for s in observations)
adapter.wait(tasks)
operation.finish(True)
io_metrics.shutdown_metrics_logger(metrics_logger)
assert not metrics_logger.thread.is_alive()
labels = {"model_name": "linker-model", "worker_id": "dp1_tp2_pp3_attn_cp0"}
assert value("ucm:sglang_connector_dump_requests_total", **labels) == 1
assert value("ucm:sglang_connector_dump_bytes_total", **labels) == 32
assert value("ucm:sglang_connector_dump_duration_ms_count", **labels) == 1
assert value("ucm:sglang_linker_offload_completed_total", **labels) == 1
assert value("ucm:sglang_linker_offload_duration_ms_count", **labels) == 1
assert "sglang:prefetch_pgs" not in scrape()
""",
    )


def test_io_samples_export_across_tp_pp_workers(tmp_path):
    for tp, pp in ((0, 0), (1, 1)):
        run_child(
            tmp_path,
            IO_BOOTSTRAP
            + f"""
storage.tp_rank, storage.pp_rank, storage.pp_size = {tp}, {pp}, 2
# Keep PP-scoped test storage paths inside the pytest directory.
launch["ucm_connector_config"]["storage_backends"] = str(Path(sys.argv[1]) / "store")
connector = make_connector()
try:
    clock_calls(3)
    assert connector.batch_get_v1(["a", "b"], None) == [True, True]
    assert connector.batch_get_v1(["c"], None) == [True]
    assert connector.batch_set_v1(["a", "b"], None) == [True, True]
    assert len(observations) == 3
    worker = "dp0_tp{tp}_pp{pp}_attn_cp0"
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            if value("ucm:sglang_connector_dump_bytes_total", worker_id=worker) == 256:
                break
        except AssertionError:
            pass
        time.sleep(0.01)
    assert value("ucm:sglang_connector_load_requests_total", worker_id=worker) == 2
    assert value("ucm:sglang_connector_load_pages_count", worker_id=worker) == 2
    assert value("ucm:sglang_connector_load_pages_sum", worker_id=worker) == 3
    assert value("ucm:sglang_connector_load_bytes_total", worker_id=worker) == 384
    assert abs(value("ucm:sglang_connector_load_duration_ms_sum", worker_id=worker) - 20) < 1e-6
    assert abs(value("ucm:sglang_connector_load_bandwidth_gbps_sum", worker_id=worker) - 384/0.01/1e9) < 1e-12
    assert value("ucm:sglang_connector_dump_bytes_total", worker_id=worker) == 256
    # Repeated collection must not replay the same observations.
    time.sleep(0.03)
    assert value("ucm:sglang_connector_load_requests_total", worker_id=worker) == 2
finally:
    connector.close()
""",
        )


def test_io_failures_disabled_empty_and_logical_calls(tmp_path):
    run_child(
        tmp_path,
        IO_BOOTSTRAP
        + r"""
connector = make_connector()
try:
    clock_calls(4)
    for is_set in (False, True):
        call = connector.batch_set_v1 if is_set else connector.batch_get_v1
        prefix = "sglang_connector_dump" if is_set else "sglang_connector_load"
        store.phase = "submit"
        try:
            call(["a"], None)
        except RuntimeError as exc:
            assert str(exc) == "submit failed"
        else:
            raise AssertionError("submission exception swallowed")
        assert observations[-1][prefix + "_submit_errors_total"] == 1
        store.phase = "wait"
        assert call(["a"], None) == [False]
        assert observations[-1][prefix + "_wait_errors_total"] == 1
    assert all(not any(k.endswith(("_bytes_total", "_bandwidth_gbps", "_pages")) for k in v) for v in observations)
    assert connector.batch_get_v1([], None) == []
    pool.kv_buffer = None
    connector.store = None
    assert connector.batch_get_v1(["a"], None) == [True]
    assert connector.batch_set_v1(["a"], None) == [True]
    pool.kv_buffer = object()
    connector.store = store
    store.phase = None
    logger = connector._metrics_logger
    connector._metrics_logger = None
    assert connector.batch_get_v1(["a"], None) == [True]
    connector._metrics_logger = logger
    assert len(observations) == 4  # No extra clock read or sample for skipped/disabled calls.
finally:
    connector.close()
""",
    )


def test_io_v2_pool_sizes_flattened_and_recording_failure(tmp_path):
    run_child(
        tmp_path,
        IO_BOOTSTRAP
        + r"""
connector = make_connector()
try:
    clock_calls(5)
    transfer = types.SimpleNamespace(name="swa", keys=["a", "b"])
    connector.pool_components["swa"] = [(store, 48)]
    connector._transfer_meta = lambda t: (t.keys, [[1, 2], [3, 4]], [[48, 48], [48, 48]])
    assert connector.batch_io_v2([transfer], False) == {"swa": [True, True]}
    assert observations[-1]["sglang_connector_load_bytes_total"] == 192
    store.phase = "submit"
    assert connector.batch_io_v2([transfer], True) == {"swa": [False, False]}
    assert observations[-1]["sglang_connector_dump_submit_errors_total"] == 1
    store.phase = None
    page = types.SimpleNamespace(numel=lambda: 80, element_size=lambda: 1, data_ptr=lambda: 7)
    restored = []
    connector.registered_pools["mamba"] = types.SimpleNamespace(
        page_size=1, get_data_page=lambda *a, **k: page,
        get_dummy_flat_data_page=lambda: page,
        set_from_flat_data_page=lambda *a: restored.append(a))
    connector.pool_components["mamba"] = [(store, 80)]
    connector.flattened_pools.add("mamba")
    flat = types.SimpleNamespace(name="mamba", keys=["a"], host_indices=[types.SimpleNamespace(item=lambda: 0)])
    assert connector.batch_io_v2([flat], True) == {"mamba": [True]}
    assert observations[-1]["sglang_connector_dump_bytes_total"] == 80
    assert connector.batch_io_v2([flat], False) == {"mamba": [True]}
    assert len(restored) == 1
    assert observations[-1]["sglang_connector_load_bytes_total"] == 80
    # A telemetry failure must not fail a successful storage operation.
    ucmmetrics.update_stats = lambda values: (_ for _ in ()).throw(RuntimeError("metrics unavailable"))
    assert connector.batch_get_v1(["a"], None) == [True]
finally:
    connector.close()
""",
    )
