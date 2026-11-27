"""Consumer wait timing preserves compatibility/native counter semantics."""

import types
from concurrent.futures import Future
from unittest.mock import Mock, patch

import pytest
from common.sglang_utils import counter_metrics_module as counters
from common.sglang_utils import load_source, metrics_module

CompatibilityCounter = load_source(
    "ucm/integration/sglang/patch/v0520/layerwise_load_counter.py",
    Future=Future,
    record_layer_wait=counters.record_layer_wait,
).LayerWiseLoadCounter


class NativeCounter:
    # Native counter swallows Future failure, then executes its ready callback.
    def __init__(self):
        self.consumer_index = 0
        self.futures = {0: [Future()]}
        self.callback = Mock()

    def wait_until(self, threshold):
        futures = self.futures.get(self.consumer_index)
        if futures is None:
            return
        futures[threshold].exception()
        self.callback()
        return "native-result"


@pytest.fixture(autouse=True)
def active_metrics_logger():
    with patch.object(metrics_module, "_active_metrics_logger", object()):
        yield


@pytest.fixture
def native_module():
    class Counter(NativeCounter):
        pass

    module = types.SimpleNamespace(LayerWiseLoadCounter=Counter)
    with (
        patch.object(counters, "version", return_value="0.5.22"),
        patch.object(
            counters.importlib, "import_module", return_value=module
        ) as importer,
    ):
        yield module, importer


@pytest.mark.parametrize(
    "version,enabled",
    [
        ("0.5.19", False),
        ("0.5.20", False),
        ("0.5.21.post1", False),
        ("0.5.22", True),
        (" 0.5.22rc1+build ", True),
        ("0.5.22.post1+build", True),
        ("0.5.23", True),
        ("0.6.0", True),
        ("unknown", False),
        ("0.5", False),
    ],
)
def test_native_patch_changes_only_wait_until(version, enabled, native_module):
    module, importer = native_module
    counter_type = module.LayerWiseLoadCounter
    original_init = counter_type.__init__
    original_wait = counter_type.wait_until
    original_members = dict(vars(counter_type))
    existing = counter_type()
    with patch.object(counters, "version", return_value=version):
        counters.install_counter_metrics()
        installed = counter_type.wait_until
        counters.install_counter_metrics()
    assert module.LayerWiseLoadCounter is counter_type
    assert counter_type.__init__ is original_init
    assert counter_type.wait_until is installed
    assert (installed is not original_wait) is enabled
    assert existing.wait_until.__func__ is installed
    assert {
        name: value
        for name, value in vars(counter_type).items()
        if name != "wait_until"
    } == {
        name: value for name, value in original_members.items() if name != "wait_until"
    }
    if enabled:
        assert installed.__wrapped__ is original_wait
        assert importer.call_args.args == (
            "sglang.srt.mem_cache.unified_cache.unified_cache_linker",
        )
    else:
        importer.assert_not_called()


def test_missing_version_skips_native_patch(native_module):
    module, importer = native_module
    original = module.LayerWiseLoadCounter.wait_until
    with patch.object(counters, "version", side_effect=counters.PackageNotFoundError):
        counters.install_counter_metrics()
    assert module.LayerWiseLoadCounter.wait_until is original
    importer.assert_not_called()


def test_missing_native_counter_does_not_install_a_replacement(native_module):
    module, _ = native_module
    del module.LayerWiseLoadCounter
    counters.install_counter_metrics()
    assert not hasattr(module, "LayerWiseLoadCounter")


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("failed", [False, True])
def test_wait_timing_preserves_failure_cleanup_and_callbacks(
    native, failed, native_module
):
    module, _ = native_module
    counters.install_counter_metrics()
    counter = module.LayerWiseLoadCounter() if native else CompatibilityCounter(1)
    if not native:
        counter.set_consumer(counter.update_producer())
    future = counter.futures[0][0]
    if failed:
        future.set_exception(RuntimeError("load failed"))
    else:
        future.set_result(None)
    samples = []
    with (
        patch.object(metrics_module, "record_metrics", samples.append),
        patch.object(
            metrics_module,
            "time",
            types.SimpleNamespace(perf_counter=Mock(side_effect=[10, 10.25])),
        ),
    ):
        if failed and not native:
            with pytest.raises(RuntimeError, match="load failed"):
                counter.wait_until(0)
        else:
            result = counter.wait_until(0)
            if native:
                assert result == "native-result"
    assert samples == [{"sglang_linker_load_layer_wait_duration_ms_0": 250}]
    if native:
        counter.callback.assert_called_once_with()
    else:
        assert counter.futures == {}  # final-layer cleanup is unchanged


def test_disabled_inactive_invalid_calls_do_not_sample(native_module):
    module, _ = native_module
    counters.install_counter_metrics()
    counter = module.LayerWiseLoadCounter()
    counter.futures[0][0].set_result(None)
    with patch.object(
        metrics_module,
        "time",
        types.SimpleNamespace(perf_counter=Mock(side_effect=AssertionError("clock"))),
    ):
        with patch.object(metrics_module, "_active_metrics_logger", None):
            assert counter.wait_until(0) == "native-result"
        with pytest.raises(IndexError):
            counter.wait_until(4)
        counter.consumer_index = -1
        assert counter.wait_until(0) is None


@pytest.mark.parametrize("version", ["0.5.20", "0.5.21"])
def test_compatibility_patch_is_not_double_wrapped(version, native_module):
    counter = CompatibilityCounter(2)
    original = counter.wait_until.__func__
    with patch.object(counters, "version", return_value=version):
        counters.install_counter_metrics()
    assert counter.wait_until.__func__ is original
    assert getattr(original, "_ucm_wait_metrics", False)


def test_actual_wait_blocks_until_complete():
    import threading

    counter = CompatibilityCounter(2)
    counter.set_consumer(counter.update_producer())
    finished = threading.Event()
    samples = []

    def consume():
        counter.wait_until(0)
        finished.set()

    with patch.object(metrics_module, "record_metrics", samples.append):
        thread = threading.Thread(target=consume)
        thread.start()
        try:
            assert not finished.wait(0.02)
            counter.complete(0, 0)
            thread.join(2)
            assert not thread.is_alive()
        finally:
            counter.complete(0, 0)
            thread.join(2)
    assert len(samples) == 1
    assert samples[0]["sglang_linker_load_layer_wait_duration_ms_0"] >= 0


def test_consumer_wait_exports_layer_label(tmp_path):
    from test_sglang_linker_export import IO_BOOTSTRAP, run_child

    run_child(
        tmp_path,
        IO_BOOTSTRAP + r"""
from concurrent.futures import Future
from unittest.mock import patch
from ucm.integration.sglang.patch.layerwise_counter_metrics import install_counter_metrics
from ucm.integration.sglang.default_metrics_config import get_default_sglang_metrics_config
expected = yaml.safe_load(Path("examples/metrics/metrics_configs_sglang.yaml").read_text(encoding="utf-8"))
assert get_default_sglang_metrics_config() == expected
logger = io_metrics.create_metrics_logger(enabled=True, model_name="counter", ranks={}, launch_config=launch)
class Native:
    def __init__(self):
        self.consumer_index = 0
        self.futures = {0: [Future(), Future()]}
        self.futures[0][1].set_result(None)
    def wait_until(self, threshold):
        self.futures[self.consumer_index][threshold].exception()
counter = Native()
native_module = types.SimpleNamespace(LayerWiseLoadCounter=Native)
with (
    patch("ucm.integration.sglang.patch.layerwise_counter_metrics.version", return_value="0.5.22"),
    patch("ucm.integration.sglang.patch.layerwise_counter_metrics.importlib.import_module", return_value=native_module),
):
    install_counter_metrics()
    install_counter_metrics()
assert native_module.LayerWiseLoadCounter is Native
assert io_metrics.counter_metrics_enabled()
ticks = iter([10, 10.125])
io_metrics.time = types.SimpleNamespace(perf_counter=lambda: next(ticks))
counter.wait_until(1)
io_metrics.shutdown_metrics_logger(logger)
assert not io_metrics.counter_metrics_enabled()
counter.wait_until(1)  # disabled: no clock read or extra sample after shutdown
assert value("ucm:sglang_linker_load_layer_wait_duration_ms_count", layer="1") == 1
assert value("ucm:sglang_linker_load_layer_wait_duration_ms_sum", layer="1") == 125
""",
    )
