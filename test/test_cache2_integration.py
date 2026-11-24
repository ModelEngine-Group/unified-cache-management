"""CPU-only coverage of Cache2 geometry and absolute MLA dump ownership."""

import ast
import shutil
import subprocess
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_cpu_module(relative_path):
    path = ROOT / relative_path
    source = ast.parse(path.read_text(encoding="utf-8"))
    source.body = [
        n
        for n in source.body
        if not (isinstance(n, ast.ImportFrom) and (n.module or "").startswith("ucm."))
    ]
    module = ModuleType(path.stem)
    module.__file__ = str(path)
    module.init_logger = Mock(return_value=Mock())
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


cache_config = load_cpu_module("ucm/integration/vllm/cache_config.py")


def connector_type():
    path = ROOT / "ucm/integration/vllm/ucm_connector.py"
    source = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in source.body if getattr(n, "name", "") == "UCMDirectConnector")
    names = {"_get_mla_dump_slice", "_select_mla_dump_blocks", "_get_rank_dump_blocks"}
    methods = [n for n in cls.body if getattr(n, "name", "") in names]
    namespace = {}
    exec(
        compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace
    )
    return type("DumpSelection", (), {name: namespace[name] for name in names})


@pytest.fixture(autouse=True)
def clear_logger():
    cache_config.logger.reset_mock()


@pytest.mark.parametrize(
    "platform", ["ascend", "ascend-a3", "ascend-a5", "cuda", "maca", "musa", "simu"]
)
@pytest.mark.parametrize("unit_tests", [False, True])
def test_native_build_selects_cache_sources_and_matching_tests(
    platform, unit_tests, tmp_path
):
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is needed to check build selection")
    source = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    marker = "set(UCM_RUNTIME_ASCEND_FAMILY OFF)"
    selection = marker + source.split(marker, 1)[1].split("if(BUILD_UNIT_TESTS)", 1)[0]
    capabilities = {
        "UCM_RUNTIME_ASCEND_FAMILY": platform.startswith("ascend"),
        "UCM_RUNTIME_ASCEND_IO_AGGREGATION": platform == "ascend",
        "UCM_RUNTIME_ASCEND_SDMA_DIRECT": platform == "ascend-a3",
    }
    capability_checks = "".join(
        f'if(NOT "${{{name}}}" STREQUAL "{("ON" if enabled else "OFF")}")\n'
        f'  message(FATAL_ERROR "Unexpected {name} for {platform}")\nendif()\n'
        for name, enabled in capabilities.items()
    )
    script = tmp_path / "cache-selection.cmake"
    no_target_operations = "\n".join(
        f"function({name})\nendfunction()"
        for name in (
            "set_target_properties",
            "target_compile_definitions",
            "target_link_libraries",
            "install",
        )
    )
    script.write_text(
        f'set(RUNTIME_ENVIRONMENT "{platform}")\n'
        + f"set(BUILD_UNIT_TESTS {'ON' if unit_tests else 'OFF'})\n"
        + selection
        + capability_checks
        + f'\nset(UCM_ROOT_DIR "{ROOT.as_posix()}")\n'
        + "function(add_library name kind)\n"
        '  set(CACHE_SOURCES "${ARGN}" PARENT_SCOPE)\nendfunction()\n'
        + no_target_operations
        + f'\ninclude("{(ROOT / "ucm/store/cache/CMakeLists.txt").as_posix()}")\n'
        + 'if(NOT CACHE_SOURCES)\n  message(FATAL_ERROR "No Cache sources")\nendif()\n'
        + "foreach(source IN LISTS CACHE_SOURCES)\n"
        '  if(NOT source MATCHES "/v2/")\n'
        '    message(FATAL_ERROR "Wrong Cache source: ${source}")\n'
        "  endif()\nendforeach()\n"
        + f'file(GLOB_RECURSE UCMSTORE_TEST_SOURCE_FILES "{ROOT.as_posix()}/ucm/store/test/case/cache/*.cc")\n'
        + '\nif(NOT UCMSTORE_TEST_SOURCE_FILES)\nmessage(FATAL_ERROR "No Cache tests")\nendif()\n'
        + "foreach(source IN LISTS UCMSTORE_TEST_SOURCE_FILES)\n"
        + 'if(NOT source MATCHES "/cache/v2/")\n'
        + 'message(FATAL_ERROR "Wrong Cache test: ${source}")\nendif()\nendforeach()\n'
        + 'message(STATUS "CACHE_SOURCES_VALIDATED")\n',
        encoding="utf-8",
    )
    result = subprocess.run(
        [cmake, "-P", str(script)],
        cwd=ROOT / "ucm/store/cache",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CACHE_SOURCES_VALIDATED" in result.stdout


@pytest.mark.parametrize(
    "platform", ["ascend", "ascend-a3", "ascend-a5", "cuda", "maca", "musa", "simu"]
)
def test_p2p_build_does_not_enable_unsupported_a5(platform, tmp_path):
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is needed to check platform support")
    source = (ROOT / "ucm/transport/p2p/CMakeLists.txt").read_text(encoding="utf-8")
    guard = source.split("if(NOT DEFINED ASCEND_ROOT)", 1)[0]
    script = tmp_path / "p2p-selection.cmake"
    script.write_text(
        f'set(RUNTIME_ENVIRONMENT "{platform}")\n'
        + guard
        + '\nmessage(STATUS "P2P_ENABLED")\n',
        encoding="utf-8",
    )
    result = subprocess.run(
        [cmake, "-P", str(script)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert ("P2P_ENABLED" in result.stdout) is (
        platform in {"ascend", "ascend-a3", "simu"}
    )


def test_native_cache_tests_cover_partitioned_implementation():
    directory = ROOT / "ucm/store/test/case/cache"
    cc_tests = list(directory.glob("*.cc"))
    v2_tests = list((directory / "v2").glob("*.cc"))
    assert not cc_tests
    assert v2_tests
    assert all("cache/cc/" not in path.read_text(encoding="utf-8") for path in v2_tests)


@pytest.mark.parametrize(
    "platform", ["ascend", "ascend-a3", "ascend-a5", "cuda", "maca", "musa", "simu"]
)
def test_cache2_applies_capacity_and_numa_on_all_platforms(platform, monkeypatch):
    monkeypatch.setenv("PLATFORM", platform)
    config = {"cache_buffer_capacity_gb": 128, "share_buffer_enable": True}
    device = Mock()
    device.get_numa_node.return_value = 3
    geometry = parallel(data_parallel_size_local=2)
    cache_config.configure_cache_capacity(config, geometry)
    cache_config.configure_cache_numa(config, device, 0, geometry, 0)
    assert config == {
        "cache_buffer_capacity_gb": 64,
        "share_buffer_enable": True,
        "cache_detected_numa_node": 3,
        "cache_fallback_numa_rank": 0,
        "cache_fallback_numa_rank_count": 16,
    }
    device.get_numa_node.assert_called_once_with(0)
    cache_config.logger.warning_once.assert_not_called()


def parallel(**kwargs):
    return NS(
        **dict(
            {
                "world_size": 8,
                "local_world_size": 8,
                "tensor_parallel_size": 8,
                "pipeline_parallel_size": 1,
                "data_parallel_rank_local": 0,
                "data_parallel_size_local": 1,
            },
            **kwargs,
        )
    )


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("dp_size", [1, 2, 4])
def test_cache_capacity_is_a_node_budget_for_gqa_and_mla(shared, dp_size):
    config = {"cache_buffer_capacity_gb": 128, "share_buffer_enable": shared}
    cache_config.configure_cache_capacity(
        config, parallel(data_parallel_size_local=dp_size)
    )
    assert config["cache_buffer_capacity_gb"] == 128 // dp_size
    assert config["cache_buffer_capacity_gb"] * dp_size == 128
    cache_config.logger.warning_once.assert_not_called()


@pytest.mark.parametrize("shared", [False, True])
def test_cache_capacity_uses_160gb_default_before_dp_division(shared):
    config = {"share_buffer_enable": shared}
    cache_config.configure_cache_capacity(config, parallel(data_parallel_size_local=2))
    assert config["cache_buffer_capacity_gb"] == 160 // 2


def test_cache_capacity_rounds_down_without_exceeding_node_budget():
    config = {"cache_buffer_capacity_gb": 128}
    cache_config.configure_cache_capacity(config, parallel(data_parallel_size_local=3))
    assert config["cache_buffer_capacity_gb"] == 42


def test_cache_capacity_zero_keeps_cpp_default_before_dp_division():
    config = {"cache_buffer_capacity_gb": 0, "share_buffer_enable": True}
    cache_config.configure_cache_capacity(config, parallel(data_parallel_size_local=2))
    assert config["cache_buffer_capacity_gb"] == 80


def test_hybrid_lb_uses_local_dp_size_not_global_dp_size():
    config = {"cache_buffer_capacity_gb": 128}
    cache_config.configure_cache_capacity(
        config,
        parallel(
            data_parallel_size=8,
            data_parallel_size_local=2,
            data_parallel_hybrid_lb=True,
        ),
    )
    assert config["cache_buffer_capacity_gb"] == 64
    cache_config.logger.warning_once.assert_not_called()


@pytest.mark.parametrize("stores", [1, 2])
def test_external_lb_keeps_per_dp_budget_and_warns(stores):
    config = {"cache_buffer_capacity_gb": 128}
    cache_config.configure_cache_capacity(
        config,
        parallel(
            data_parallel_external_lb=True,
            data_parallel_size=8,
            data_parallel_size_local=1,
        ),
        stores_per_dp=stores,
    )
    assert config["cache_buffer_capacity_gb"] == 128 // stores
    message = cache_config.logger.warning_once.call_args.args[0]
    assert "manually set" in message
    assert "actual local DP count" in message


@pytest.mark.parametrize("dp_size", [0, -1])
def test_invalid_local_dp_size_is_rejected(dp_size):
    with pytest.raises(ValueError, match="Invalid Cache capacity partition"):
        cache_config.configure_cache_capacity(
            {"cache_buffer_capacity_gb": 128},
            parallel(data_parallel_size_local=dp_size),
        )


def test_tiny_capacity_does_not_round_to_zero_and_activate_cpp_default():
    with pytest.raises(ValueError, match="set at least 4GB"):
        cache_config.configure_cache_capacity(
            {"cache_buffer_capacity_gb": 2},
            parallel(data_parallel_size_local=2),
            stores_per_dp=2,
        )


@pytest.mark.parametrize(
    "hma,capacity,dp_size,expected",
    [
        (False, 128, 2, 64),
        (False, None, 2, 80),
        (False, 0, 2, 80),
        (True, 128, 2, 32),
        (True, None, 8, 10),
        (True, 160, 8, 10),
        (True, 0, 8, 10),
    ],
)
@pytest.mark.parametrize("shared", [False, True])
def test_connector_capacity_hooks_divide_dp_and_fawa_once(
    hma, capacity, dp_size, expected, shared
):
    relative = "hma_connector.py" if hma else "ucm_connector.py"
    path = ROOT / "ucm/integration/vllm" / relative
    source = ast.parse(path.read_text(encoding="utf-8"))
    method = next(
        node
        for node in ast.walk(source)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_set_default_shm_buffer_capacity"
    )
    namespace = {
        "Any": object,
        "configure_cache_capacity": cache_config.configure_cache_capacity,
    }
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    connector = NS(
        _vllm_config=NS(parallel_config=parallel(data_parallel_size_local=dp_size))
    )
    config = {"share_buffer_enable": shared}
    if capacity is not None:
        config["cache_buffer_capacity_gb"] = capacity
    namespace[method.name](connector, config)
    assert config["cache_buffer_capacity_gb"] == expected


@pytest.mark.parametrize("world,local,count", [(8, 8, 8), (16, 8, 8), (8, 16, 8)])
def test_cache_domain_counts_only_local_workers(world, local, count):
    assert (
        cache_config.cache_rank_count(
            parallel(world_size=world, local_world_size=local)
        )
        == count
    )


def test_affinity_works_without_worker_geometry():
    config = {}
    cache_config.configure_cache_numa(config, NS(get_numa_node=lambda _: 6), 0, NS(), 0)
    assert config == {"cache_detected_numa_node": 6}


@pytest.mark.parametrize("affinity", [None, 6])
def test_dp_isolation_does_not_restart_numa_slices(affinity):
    results = []
    for dp in range(2):
        for rank in range(8):
            config = {}
            cache_config.configure_cache_numa(
                config,
                NS(get_numa_node=lambda _: affinity),
                dp * 8 + rank,
                parallel(data_parallel_rank_local=dp, data_parallel_size_local=2),
                rank,
            )
            worker = config["cache_fallback_numa_rank"]
            count = config["cache_fallback_numa_rank_count"]
            results.append(worker * 8 // count)
    assert results == [n for n in range(8) for _ in range(2)]


@pytest.mark.parametrize("worker", [False, True])
@pytest.mark.parametrize("local_dp_size", [1, 2])
def test_ec_store_configures_capacity_and_local_geometry(worker, local_dp_size):
    path = ROOT / "ucm/integration/vllm/ucm_ec_connector.py"
    method = next(
        node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef) and node.name == "create_ucm_ec_store"
    )
    roles = NS(WORKER="worker", SCHEDULER="scheduler")
    device = NS(get_numa_node=lambda _: 6)
    factory = Mock(side_effect=lambda _name, config: config)
    namespace = {
        "Any": object,
        "VllmConfig": object,
        "EncoderCacheLayout": object,
        "UcmKVStoreBaseV1": object,
        "ECConnectorRole": roles,
        "UcmConnectorFactoryV1": NS(create_connector=factory),
        "cache_rank_count": cache_config.cache_rank_count,
        "configure_cache_capacity": cache_config.configure_cache_capacity,
        "configure_cache_numa": cache_config.configure_cache_numa,
        "create_device": lambda: device,
        "_get_store_io_sizes": lambda shard, block: (shard, block),
        "_get_store_gc_block_size": lambda _pipeline, _sizes, _shard, block: block,
        "logger": Mock(),
    }
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    config = namespace[method.name](
        vllm_config=NS(
            parallel_config=parallel(
                local_world_size=4,
                data_parallel_rank=0,
                data_parallel_size_local=local_dp_size,
            )
        ),
        role=roles.WORKER if worker else roles.SCHEDULER,
        layout=NS(chunk_bytes=4096),
        ec_config={
            "ucm_connector_name": "UcmPipelineStore",
            "ucm_connector_config": {},
            "encoder_cache_config": {"store_unique_id": "engine.ec.dp0"},
        },
        device_id=5 if worker else -1,
    )
    assert config["cache_buffer_capacity_gb"] == 160 // local_dp_size
    assert config["local_rank_size"] == 4
    assert config["unique_id"] == "engine.ec.dp0"
    assert config["block_size"] == 4096
    if worker:
        assert config["cache_detected_numa_node"] == 6
        assert config["tensor_size_list"] == [4096]
    else:
        assert "cache_detected_numa_node" not in config
    factory.assert_called_once()


@pytest.mark.parametrize("size", [1, 4, 8, 16])
@pytest.mark.parametrize("start,length", [(0, 0), (0, 512), (3, 2), (7, 79), (16, 1)])
def test_mla_dump_absolute_ownership_and_complete_coverage(size, start, length):
    cls = connector_type()
    selected = []
    keys = list(range(start, start + length))
    blocks = list(range(1000, 1000 + length))
    for rank in range(size):
        connector = cls()
        connector.tp_size, connector.tp_rank, connector.is_mla = size, rank, True
        rows, addresses = connector._get_rank_dump_blocks(
            NS(dump_block_ids=(keys, blocks), dump_block_start=start)
        )
        assert all(key % size == rank for key in rows)
        assert all(
            address == 1000 + key - start for key, address in zip(rows, addresses)
        )
        selected.extend(rows)
    assert sorted(selected) == keys


def test_gqa_keeps_rank_specific_records_instead_of_tp_striping():
    connector = connector_type()()
    connector.is_mla, connector.tp_rank, connector.tp_size = False, 7, 8
    assert connector._get_rank_dump_blocks(
        NS(dump_block_ids=([b"a", b"b"], [4, 5]))
    ) == ([b"a", b"b"], [4, 5])


def test_mla_owner_is_stable_across_replay_chunks():
    connector = connector_type()()
    connector.tp_size, connector.tp_rank = 8, 3
    complete = list(range(78))[connector._get_mla_dump_slice(0)]
    chunked = []
    for start, end in [(0, 7), (7, 31), (31, 78)]:
        chunked.extend(list(range(start, end))[connector._get_mla_dump_slice(start)])
    assert chunked == complete


def test_wa_tail_blocks_follow_whole_key_rows():
    import numpy as np

    connector = connector_type()()
    connector.tp_size, connector.tp_rank = 4, 2
    key_slice = connector._get_mla_dump_slice(3)
    rows = np.arange(8 * 3).reshape(-1, 3)[key_slice]
    assert rows.tolist() == [[9, 10, 11], [21, 22, 23]]
