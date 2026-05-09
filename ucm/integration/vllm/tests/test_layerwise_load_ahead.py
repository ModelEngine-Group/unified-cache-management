from __future__ import annotations

import importlib.util
import sys
import types
from collections import defaultdict
from pathlib import Path

_MISSING = object()
_STUBBED_MODULE_NAMES = [
    "torch",
    "numpy",
    "vllm",
    "vllm.config",
    "vllm.distributed",
    "vllm.distributed.kv_transfer",
    "vllm.distributed.kv_transfer.kv_connector",
    "vllm.distributed.kv_transfer.kv_connector.v1",
    "vllm.distributed.kv_transfer.kv_connector.v1.base",
    "vllm.distributed.parallel_state",
    "vllm.model_executor",
    "vllm.model_executor.models",
    "vllm.model_executor.models.utils",
    "vllm.platforms",
    "vllm.v1",
    "vllm.v1.core",
    "vllm.v1.core.sched",
    "vllm.v1.core.sched.output",
    "ucm",
    "ucm.integration",
    "ucm.integration.vllm",
    "ucm.integration.vllm.device",
    "ucm.logger",
    "ucm.observability",
    "ucm.shared",
    "ucm.shared.metrics",
    "ucm.store",
    "ucm.store.factory_v1",
    "ucm.store.ucmstore_v1",
    "ucm.utils",
    "ucm.sparse",
    "ucm.sparse.state",
]


def _module(name: str, *, package: bool = False) -> types.ModuleType:
    module = types.ModuleType(name)
    if package:
        module.__path__ = []
    sys.modules[name] = module
    return module


def _install_dependency_stubs() -> dict[str, object]:
    previous_modules = {
        name: sys.modules.get(name, _MISSING) for name in _STUBBED_MODULE_NAMES
    }
    for name in _STUBBED_MODULE_NAMES:
        sys.modules.pop(name, None)

    torch = _module("torch")
    torch.Tensor = type("Tensor", (), {})
    torch.dtype = type("dtype", (), {})

    numpy = _module("numpy")
    numpy.ndarray = list
    numpy.uint64 = int
    numpy.asarray = lambda value, dtype=None: value
    numpy.ascontiguousarray = lambda value: value

    _module("vllm", package=True)
    config = _module("vllm.config")
    config.VllmConfig = type("VllmConfig", (), {})

    _module("vllm.distributed", package=True)
    _module("vllm.distributed.kv_transfer", package=True)
    _module("vllm.distributed.kv_transfer.kv_connector", package=True)
    _module("vllm.distributed.kv_transfer.kv_connector.v1", package=True)
    base = _module("vllm.distributed.kv_transfer.kv_connector.v1.base")

    class KVConnectorBase_V1:
        def __init__(self, vllm_config=None, role=None):
            self._vllm_config = vllm_config
            self._role = role
            self._connector_metadata = None

        def _get_connector_metadata(self):
            return self._connector_metadata

        def bind_connector_metadata(self, connector_metadata):
            self._connector_metadata = connector_metadata

        def clear_connector_metadata(self):
            self._connector_metadata = None

    base.KVConnectorBase_V1 = KVConnectorBase_V1
    base.KVConnectorMetadata = type("KVConnectorMetadata", (), {})
    base.KVConnectorRole = types.SimpleNamespace(SCHEDULER="scheduler", WORKER="worker")

    parallel_state = _module("vllm.distributed.parallel_state")
    parallel_state.get_world_group = lambda: types.SimpleNamespace(local_rank=0, rank=0)

    _module("vllm.model_executor", package=True)
    _module("vllm.model_executor.models", package=True)
    model_utils = _module("vllm.model_executor.models.utils")
    model_utils.extract_layer_index = lambda name: int(name.rsplit(".", 1)[1])

    platforms = _module("vllm.platforms")
    platforms.current_platform = types.SimpleNamespace(
        is_cuda_alike=lambda: False,
        device_type="cpu",
    )

    _module("vllm.v1", package=True)
    _module("vllm.v1.core", package=True)
    _module("vllm.v1.core.sched", package=True)
    sched_output = _module("vllm.v1.core.sched.output")
    sched_output.SchedulerOutput = type("SchedulerOutput", (), {})

    _module("ucm", package=True)
    _module("ucm.integration", package=True)
    _module("ucm.integration.vllm", package=True)
    device = _module("ucm.integration.vllm.device")
    device.create_device = lambda: None

    logger_module = _module("ucm.logger")

    class DummyLogger:
        def info(self, *args, **kwargs):
            pass

        def info_once(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

    logger_module.init_logger = lambda name=None: DummyLogger()

    observability = _module("ucm.observability")
    observability.PrometheusStatsLogger = type("PrometheusStatsLogger", (), {})

    _module("ucm.shared", package=True)
    metrics = _module("ucm.shared.metrics")
    metrics.ucmmetrics = types.SimpleNamespace()

    _module("ucm.store", package=True)
    factory = _module("ucm.store.factory_v1")
    factory.UcmConnectorFactoryV1 = type("UcmConnectorFactoryV1", (), {})
    store_base = _module("ucm.store.ucmstore_v1")
    store_base.Task = type("Task", (), {})
    store_base.UcmKVStoreBaseV1 = type("UcmKVStoreBaseV1", (), {})

    utils = _module("ucm.utils")
    utils.Config = type("Config", (), {})

    _module("ucm.sparse", package=True)
    sparse_state = _module("ucm.sparse.state")
    sparse_state.has_ucm_sparse = lambda: False
    return previous_modules


def _restore_dependency_stubs(previous_modules: dict[str, object]) -> None:
    for name in _STUBBED_MODULE_NAMES:
        previous_module = previous_modules[name]
        if previous_module is _MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous_module


def _load_connector_module():
    previous_modules = _install_dependency_stubs()
    module_name = "ucm.integration.vllm.ucm_connector"
    previous_connector_module = sys.modules.get(module_name, _MISSING)
    try:
        sys.modules.pop(module_name, None)
        module_path = (
            Path(__file__).resolve().parents[4]
            / "ucm"
            / "integration"
            / "vllm"
            / "ucm_connector.py"
        )
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        if previous_connector_module is _MISSING:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_connector_module
        _restore_dependency_stubs(previous_modules)


class FakeTask:
    def __init__(self, layer_id: int, block_ids: tuple[bytes, ...]):
        self.layer_id = layer_id
        self.block_ids = block_ids


class FakeStore:
    def __init__(self, fail_block_ids: set[bytes] | None = None):
        self.fail_block_ids = fail_block_ids or set()
        self.load_attempts = []
        self.loads = []
        self.waited_layers = []

    def load_data(self, block_ids, shard_indexes, layer_ptrs):
        layer_id = shard_indexes[0]
        block_ids_tuple = tuple(block_ids)
        self.load_attempts.append((layer_id, block_ids_tuple))
        if self.fail_block_ids.intersection(block_ids_tuple):
            raise RuntimeError("load submit failed")

        task = FakeTask(layer_id, block_ids_tuple)
        self.loads.append(
            {
                "layer_id": layer_id,
                "block_ids": block_ids_tuple,
                "ptrs": list(layer_ptrs),
                "task": task,
            }
        )
        return task

    def wait(self, task):
        self.waited_layers.append(task.layer_id)


class FakeKVCacheLayout:
    def __init__(self, layer_count: int):
        self.layer_count = layer_count

    def extract_block_addrs(self, vllm_block_ids, layer_first=False):
        assert layer_first is True
        rows = []
        for local_row in range(self.layer_count):
            rows.append([[local_row * 1000 + block_id] for block_id in vllm_block_ids])
        return rows


def _make_metadata(*, include_failing_request: bool = False):
    request_meta = {
        "ok": types.SimpleNamespace(
            load_block_ids=([b"ok-block"], [7]),
            dump_block_ids=([], []),
        )
    }
    if include_failing_request:
        request_meta["bad"] = types.SimpleNamespace(
            load_block_ids=([b"bad-block"], [99]),
            dump_block_ids=([], []),
        )
    return types.SimpleNamespace(request_meta=request_meta)


def _make_connector(module, *, load_ahead: int, store: FakeStore | None = None):
    connector = module.UCMLayerWiseConnector.__new__(module.UCMLayerWiseConnector)
    connector.load_tasks = defaultdict(dict)
    connector.store = store or FakeStore()
    connector.request_data = []
    connector._failure_req_ids = set()
    connector._invalid_block_ids = set()
    connector.layerwise_load_ahead = load_ahead
    connector.layer_ids = [10, 11, 12, 13, 14]
    connector.layer_name_to_id = {
        f"layer.{layer_id}": layer_id for layer_id in connector.layer_ids
    }
    connector.first_layer_id = 10
    connector.kv_cache_layout = FakeKVCacheLayout(len(connector.layer_ids))
    connector.tp_rank = 0
    connector.tp_size = 1
    connector.is_mla = False
    connector.request_hasher = lambda block_id: block_id
    connector.need_load = False
    connector._connector_metadata = _make_metadata()
    return connector


def _loaded_layers(store: FakeStore) -> list[int]:
    return [load["layer_id"] for load in store.loads]


def test_layerwise_load_ahead_one_preserves_single_layer_submission():
    module = _load_connector_module()
    store = FakeStore()
    connector = _make_connector(module, load_ahead=1, store=store)

    connector.start_load_kv(None)
    assert _loaded_layers(store) == [10]

    connector.wait_for_layer_load("layer.10")
    assert store.waited_layers == [10]
    assert _loaded_layers(store) == [10, 11]


def test_layerwise_load_ahead_prefetches_window_and_refills_by_layer_order():
    module = _load_connector_module()
    store = FakeStore()
    connector = _make_connector(module, load_ahead=3, store=store)

    connector.start_load_kv(None)
    assert _loaded_layers(store) == [10, 11, 12]

    connector.wait_for_layer_load("layer.10")
    assert store.waited_layers == [10]
    assert _loaded_layers(store) == [10, 11, 12, 13]

    connector.wait_for_layer_load("layer.11")
    assert store.waited_layers == [10, 11]
    assert _loaded_layers(store) == [10, 11, 12, 13, 14]

    for layer_id in [12, 13, 14]:
        connector.wait_for_layer_load(f"layer.{layer_id}")

    assert store.waited_layers == [10, 11, 12, 13, 14]
    assert _loaded_layers(store) == [10, 11, 12, 13, 14]


def test_layerwise_load_ahead_skips_failed_request_in_future_layers():
    module = _load_connector_module()
    store = FakeStore(fail_block_ids={b"bad-block"})
    connector = _make_connector(module, load_ahead=3, store=store)
    connector._connector_metadata = _make_metadata(include_failing_request=True)

    connector.start_load_kv(None)

    assert connector._failure_req_ids == {"bad"}
    assert connector._invalid_block_ids == {99}
    assert _loaded_layers(store) == [10, 11, 12]
    assert [
        layer_id
        for layer_id, block_ids in store.load_attempts
        if block_ids == (b"bad-block",)
    ] == [10]

    connector.wait_for_layer_load("layer.10")
    connector.wait_for_layer_load("layer.11")

    assert _loaded_layers(store) == [10, 11, 12, 13, 14]
    assert [
        layer_id
        for layer_id, block_ids in store.load_attempts
        if block_ids == (b"bad-block",)
    ] == [10]
