"""CPU-only coverage of Cache2 geometry and absolute MLA dump ownership."""

import ast
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

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
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


capabilities = load_cpu_module("ucm/store/cache/capabilities.py")
cache_config = load_cpu_module("ucm/integration/vllm/cache_config.py")
cache_config.uses_partitioned_cache = capabilities.uses_partitioned_cache


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
def clear_capabilities():
    capabilities.uses_partitioned_cache.cache_clear()
    yield
    capabilities.uses_partitioned_cache.cache_clear()


@pytest.mark.parametrize("version, expected", [(1, False), (2, True)])
def test_installed_core_not_platform_environment(version, expected, monkeypatch):
    monkeypatch.setenv("PLATFORM", "ascend-a5" if version == 1 else "cuda")
    library = NS(UcmCacheCoreVersion=Mock(return_value=version))
    with (
        patch.object(Path, "exists", return_value=True),
        patch.object(capabilities.ctypes, "CDLL", return_value=library),
    ):
        assert capabilities.uses_partitioned_cache() is expected
        assert library.UcmCacheCoreVersion.argtypes == []


def test_old_wheel_without_version_symbol_retains_legacy_behavior():
    with (
        patch.object(Path, "exists", return_value=True),
        patch.object(capabilities.ctypes, "CDLL", return_value=NS()),
    ):
        assert not capabilities.uses_partitioned_cache()


def test_missing_library_retains_legacy_behavior():
    with patch.object(Path, "exists", return_value=False):
        assert not capabilities.uses_partitioned_cache()


def test_broken_installed_library_is_not_silently_treated_as_legacy():
    with (
        patch.object(Path, "exists", return_value=True),
        patch.object(
            capabilities.ctypes, "CDLL", side_effect=OSError("unresolved symbol")
        ),
    ):
        with pytest.raises(OSError, match="unresolved symbol"):
            capabilities.uses_partitioned_cache()


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


@pytest.mark.parametrize("world,local,count", [(8, 8, 8), (16, 8, 8), (8, 16, 8)])
def test_cache_domain_counts_only_local_workers(world, local, count):
    assert (
        cache_config.cache_rank_count(
            parallel(world_size=world, local_world_size=local)
        )
        == count
    )


def test_affinity_short_circuits_fallback():
    config = {}
    with patch.object(cache_config, "uses_partitioned_cache", return_value=True):
        cache_config.configure_cache_numa(
            config, NS(get_numa_node=lambda _: 6), 0, NS(), 0
        )
    assert config == {"cache_detected_numa_node": 6}


def test_dp_isolation_does_not_restart_numa_slices():
    results = []
    with patch.object(cache_config, "uses_partitioned_cache", return_value=True):
        for dp in range(2):
            for rank in range(8):
                config = {}
                cache_config.configure_cache_numa(
                    config,
                    NS(get_numa_node=lambda _: None),
                    dp * 8 + rank,
                    parallel(data_parallel_rank_local=dp, data_parallel_size_local=2),
                    rank,
                )
                worker = config["cache_fallback_numa_rank"]
                count = config["cache_fallback_numa_rank_count"]
                results.append(worker * 8 // count)
    assert results == [n for n in range(8) for _ in range(2)]


def test_legacy_core_does_not_probe_numa():
    device = Mock()
    config = {}
    with patch.object(cache_config, "uses_partitioned_cache", return_value=False):
        cache_config.configure_cache_numa(config, device, 0, NS(), 0)
    assert config == {}
    device.get_numa_node.assert_not_called()


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
