"""Test real multiprocess Prometheus export without native UCM, SGLang or GPUs.

Run with pytest --noconftest test/test_sglang_linker_export.py. Each case uses a fresh
interpreter because Prometheus selects its storage mode at import time.
"""

import os


import subprocess


import sys


import textwrap


from pathlib import Path

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
            base = name.rsplit("_", 1)[0]
            typ, definition = defs[name] if name in defs else defs[base]
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
"""


def test_linker_async_adapter_exports_through_shared_multiprocess_registry(tmp_path):
    run_child(
        tmp_path,
        IO_BOOTSTRAP + r"""
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
adapter.wait(tasks)
operation.finish(True)
io_metrics.record_metrics({"sglang_linker_load_layer_duration_ms_3": 12.5})
io_metrics.shutdown_metrics_logger(metrics_logger)
assert not metrics_logger.thread.is_alive()
labels = {"model_name": "linker-model", "worker_id": "dp1_tp2_pp3_attn_cp0"}
assert value("ucm:sglang_connector_dump_requests_total", **labels) == 1
assert value("ucm:sglang_connector_dump_bytes_total", **labels) == 32
assert value("ucm:sglang_linker_offload_completed_total", **labels) == 1
assert value("ucm:sglang_linker_offload_duration_ms_count", **labels) == 1
assert value("ucm:sglang_adapter_interface_submit_offload_duration_ms_count", **labels) == 1
assert value("ucm:sglang_adapter_interface_wait_duration_ms_count", **labels) == 1
assert value("ucm:sglang_linker_load_layer_duration_ms_sum", layer="3", **labels) == 12.5
assert "sglang:prefetch_pgs" not in scrape()
""",
    )
