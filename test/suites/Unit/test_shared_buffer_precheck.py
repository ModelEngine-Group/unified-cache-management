"""Unit tests for the launch-time shared-buffer resource precheck.

Follows the source-extraction pattern of test_kv_cache_layout.py so the
tests run without vllm/torch installed (e.g. on a dev host).
"""

import ast
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
CONNECTOR_PATH = REPO_ROOT / "ucm" / "integration" / "vllm" / "ucm_connector.py"
ENGINE_CORE_PATCH_PATH = (
    REPO_ROOT / "ucm" / "integration" / "vllm" / "patch" / "engine_core_startup_shutdown_patch.py"
)


class FakeLogger:
    def __init__(self):
        self.messages = []

    def debug(self, *args, **kwargs):
        self.messages.append(args[0] if args else None)

    def info(self, *args, **kwargs):
        self.messages.append(args[0] if args else None)

    def warning(self, *args, **kwargs):
        self.messages.append(args[0] if args else None)

    def error(self, *args, **kwargs):
        self.messages.append(args[0] if args else None)


class FakeShutil:
    """shutil stand-in whose disk_usage result is swapped per test."""

    usage = None
    raise_oserror = False

    @staticmethod
    def disk_usage(path):
        if FakeShutil.raise_oserror or FakeShutil.usage is None:
            raise OSError(f"cannot stat {path}")
        return FakeShutil.usage


class FakeMeminfoFile:
    def __init__(self, lines):
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def __iter__(self):
        return iter(self._lines)


class FakeOpen:
    """open() stand-in serving only /proc/meminfo."""

    lines = ["MemAvailable: 1048576 kB\n"]
    raise_oserror = False

    def __call__(self, path, *args, **kwargs):
        if path == "/proc/meminfo":
            if FakeOpen.raise_oserror:
                raise OSError("no /proc/meminfo")
            return FakeMeminfoFile(FakeOpen.lines)
        raise OSError(f"unexpected open({path!r})")


class FakeConfig:
    """Config stand-in; launch_config is swapped per test."""

    launch_config = {}
    instances = []

    def __init__(self, kv_transfer_config):
        self.kv_transfer_config = kv_transfer_config
        FakeConfig.instances.append(self)

    def get_config(self):
        return FakeConfig.launch_config


def _load_module_symbols(path, selected_names, namespace):
    source = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    selected_nodes = []
    found = set()
    for node in tree.body:
        if getattr(node, "name", None) in selected_names:
            selected_nodes.append(node)
            found.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in selected_names:
                    selected_nodes.append(node)
                    found.add(target.id)
    missing = selected_names - found
    assert not missing, f"symbols not found in {path}: {sorted(missing)}"
    module = ast.Module(body=selected_nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def _reset_fakes():
    FakeShutil.usage = None
    FakeShutil.raise_oserror = False
    FakeOpen.lines = ["MemAvailable: 1048576 kB\n"]
    FakeOpen.raise_oserror = False
    FakeConfig.launch_config = {}
    FakeConfig.instances = []


GIB = 1 << 30


def _set_resources(shm_total_gib=512, shm_free_gib=512, mem_available_gib=512):
    FakeShutil.usage = SimpleNamespace(
        total=shm_total_gib * GIB, free=shm_free_gib * GIB
    )
    FakeOpen.lines = [f"MemAvailable: {mem_available_gib << 20} kB\n"]


def _make_vllm_config(
    kv_connector="UCMConnector",
    kv_transfer_config="sentinel",
    is_mla=True,
):
    if kv_connector is not None:
        kvt = SimpleNamespace(kv_connector=kv_connector)
    else:
        kvt = kv_transfer_config
    return SimpleNamespace(
        kv_transfer_config=kvt,
        model_config=SimpleNamespace(is_deepseek_mla=is_mla),
    )


SYMBOLS = _load_module_symbols(
    CONNECTOR_PATH,
    {
        "_SHM_DIR",
        "_DEFAULT_SHARED_BUFFER_CAPACITY_GB",
        "_check_shm_capacity",
        "_check_host_mem_available",
        "_precheck_shared_buffer_capacity",
        "precheck_shared_buffer_resources",
    },
    {
        "Any": Any,
        "shutil": FakeShutil,
        "logger": FakeLogger(),
        "open": FakeOpen(),
        "Config": FakeConfig,
    },
)
precheck_shared_buffer_capacity = SYMBOLS["_precheck_shared_buffer_capacity"]
precheck_shared_buffer_resources = SYMBOLS["precheck_shared_buffer_resources"]
DEFAULT_CAPACITY_GB = SYMBOLS["_DEFAULT_SHARED_BUFFER_CAPACITY_GB"]


PATCH_SYMBOLS = _load_module_symbols(
    ENGINE_CORE_PATCH_PATH,
    {"_extract_vllm_config", "_run_shared_buffer_precheck"},
    {"logger": FakeLogger()},
)
extract_vllm_config = PATCH_SYMBOLS["_extract_vllm_config"]
run_shared_buffer_precheck = PATCH_SYMBOLS["_run_shared_buffer_precheck"]


class SharedBufferCapacityGateTest(unittest.TestCase):
    def setUp(self):
        _reset_fakes()
        _set_resources()

    def test_capacity_exceeding_shm_total_raises(self):
        _set_resources(shm_total_gib=128, shm_free_gib=128)
        configs = [
            {"ucm_connector_config": {"cache_buffer_capacity_gb": 156}}
        ]
        with self.assertRaises(RuntimeError) as ctx:
            precheck_shared_buffer_capacity(configs, is_mla=True)
        self.assertIn("requires 156GB in /dev/shm", str(ctx.exception))
        self.assertIn("has only 128GB", str(ctx.exception))

    def test_free_space_shortfall_raises(self):
        _set_resources(shm_total_gib=512, shm_free_gib=64)
        configs = [
            {"ucm_connector_config": {"cache_buffer_capacity_gb": 128}}
        ]
        with self.assertRaises(RuntimeError) as ctx:
            precheck_shared_buffer_capacity(configs, is_mla=True)
        self.assertIn("requires 128GB free in /dev/shm", str(ctx.exception))

    def test_host_memory_shortfall_raises(self):
        _set_resources(shm_total_gib=512, shm_free_gib=512, mem_available_gib=64)
        configs = [
            {"ucm_connector_config": {"cache_buffer_capacity_gb": 128}}
        ]
        with self.assertRaises(RuntimeError) as ctx:
            precheck_shared_buffer_capacity(configs, is_mla=True)
        self.assertIn("MemAvailable", str(ctx.exception))

    def test_sufficient_resources_pass(self):
        configs = [
            {"ucm_connector_config": {"cache_buffer_capacity_gb": 128}}
        ]
        precheck_shared_buffer_capacity(configs, is_mla=True)  # no raise

    def test_default_capacity_checked_when_unset(self):
        _set_resources(shm_total_gib=64, shm_free_gib=64)
        configs = [{"ucm_connector_config": {}}]
        with self.assertRaises(RuntimeError) as ctx:
            precheck_shared_buffer_capacity(configs, is_mla=True)
        self.assertIn(f"requires {DEFAULT_CAPACITY_GB}GB", str(ctx.exception))

    def test_share_buffer_disabled_skips(self):
        _set_resources(shm_total_gib=1, shm_free_gib=1)
        configs = [
            {
                "ucm_connector_config": {
                    "cache_buffer_capacity_gb": 128,
                    "share_buffer_enable": False,
                }
            }
        ]
        precheck_shared_buffer_capacity(configs, is_mla=True)  # no raise

    def test_non_mla_defaults_to_disabled(self):
        _set_resources(shm_total_gib=1, shm_free_gib=1)
        configs = [{"ucm_connector_config": {"cache_buffer_capacity_gb": 128}}]
        precheck_shared_buffer_capacity(configs, is_mla=False)  # no raise

    def test_empty_connector_configs_is_noop(self):
        precheck_shared_buffer_capacity([], is_mla=True)  # no raise


class LaunchTimePrecheckTest(unittest.TestCase):
    def setUp(self):
        _reset_fakes()
        _set_resources()

    def _set_launch_config(self, ucm_connector_config=None, **extra):
        connector = {"ucm_connector_name": "UcmPipelineStore"}
        if ucm_connector_config is not None:
            connector["ucm_connector_config"] = ucm_connector_config
        FakeConfig.launch_config = {"ucm_connectors": [connector], **extra}

    def test_resource_shortfall_raises(self):
        _set_resources(shm_total_gib=128, shm_free_gib=128)
        self._set_launch_config({"cache_buffer_capacity_gb": 156})
        with self.assertRaises(RuntimeError):
            precheck_shared_buffer_resources(_make_vllm_config())

    def test_sufficient_resources_pass(self):
        self._set_launch_config({"cache_buffer_capacity_gb": 128})
        precheck_shared_buffer_resources(_make_vllm_config())  # no raise

    def test_kv_transfer_disabled_is_noop(self):
        vllm_config = SimpleNamespace(
            kv_transfer_config=None, model_config=SimpleNamespace()
        )
        precheck_shared_buffer_resources(vllm_config)
        self.assertEqual(FakeConfig.instances, [])

    def test_non_ucm_connector_is_skipped(self):
        self._set_launch_config({"cache_buffer_capacity_gb": 4096})
        _set_resources(shm_total_gib=1, shm_free_gib=1)
        precheck_shared_buffer_resources(
            _make_vllm_config(kv_connector="SharedStorageConnector")
        )
        self.assertEqual(FakeConfig.instances, [])

    def test_use_lite_dispatch_is_skipped(self):
        self._set_launch_config(
            {"cache_buffer_capacity_gb": 4096}, use_lite=True
        )
        _set_resources(shm_total_gib=1, shm_free_gib=1)
        precheck_shared_buffer_resources(_make_vllm_config())  # no raise

    def test_monitor_dispatch_is_skipped(self):
        self._set_launch_config(
            {"cache_buffer_capacity_gb": 4096},
            use_inference_duration_monitor=True,
        )
        _set_resources(shm_total_gib=1, shm_free_gib=1)
        precheck_shared_buffer_resources(_make_vllm_config())  # no raise

    def test_malformed_capacity_is_deferred_to_connector(self):
        self._set_launch_config({"cache_buffer_capacity_gb": "not-a-number"})
        precheck_shared_buffer_resources(_make_vllm_config())  # no raise

    def test_config_parse_error_is_swallowed(self):
        class ExplodingConfig:
            def __init__(self, kv_transfer_config):
                raise ValueError("bad yaml")

        SYMBOLS["Config"] = ExplodingConfig
        try:
            precheck_shared_buffer_resources(_make_vllm_config())  # no raise
        finally:
            SYMBOLS["Config"] = FakeConfig

    def test_missing_ucm_connectors_is_noop(self):
        FakeConfig.launch_config = {"ucm_connector_name": "UcmNfsStore"}
        precheck_shared_buffer_resources(_make_vllm_config())  # no raise


class EngineCorePatchPrecheckTest(unittest.TestCase):
    def setUp(self):
        _reset_fakes()
        _set_resources()

    def test_extract_vllm_config_from_positional_args(self):
        vllm_config = _make_vllm_config()
        result = extract_vllm_config(vllm_config, "usage-context", object(), True)
        self.assertIs(result, vllm_config)

    def test_extract_vllm_config_from_kwargs(self):
        vllm_config = _make_vllm_config()
        result = extract_vllm_config(
            "usage-context", object(), True, vllm_config=vllm_config
        )
        self.assertIs(result, vllm_config)

    def test_extract_vllm_config_missing(self):
        self.assertIsNone(extract_vllm_config("a", "b"))

    def _install_entry(self, impl):
        module = SimpleNamespace(precheck_shared_buffer_resources=impl)
        self._saved = sys.modules.get("ucm.integration.vllm.ucm_connector")
        sys.modules["ucm.integration.vllm.ucm_connector"] = module

    def tearDown(self):
        if hasattr(self, "_saved"):
            if self._saved is None:
                sys.modules.pop("ucm.integration.vllm.ucm_connector", None)
            else:
                sys.modules["ucm.integration.vllm.ucm_connector"] = self._saved

    def test_precheck_runs_before_original(self):
        calls = []

        def entry(vllm_config):
            calls.append("precheck")

        self._install_entry(entry)
        run_shared_buffer_precheck(_make_vllm_config())
        self.assertEqual(calls, ["precheck"])

    def test_resource_error_propagates(self):
        def entry(vllm_config):
            raise RuntimeError("Shared-buffer cache requires 156GB in /dev/shm")

        self._install_entry(entry)
        with self.assertRaises(RuntimeError):
            run_shared_buffer_precheck(_make_vllm_config())

    def test_unexpected_errors_are_swallowed(self):
        def entry(vllm_config):
            raise ValueError("connector import blew up")

        self._install_entry(entry)
        run_shared_buffer_precheck(_make_vllm_config())  # no raise

    def test_no_vllm_config_is_noop(self):
        run_shared_buffer_precheck("usage-context", object(), True)  # no raise


if __name__ == "__main__":
    unittest.main()
