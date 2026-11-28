"""Check exported byte offsets against real CPU tensor views and file bytes."""

import ast
import copy
import ctypes
import importlib.util
import json
import math
import struct
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Optional
from unittest.mock import Mock

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
MODULE = ROOT / "ucm/integration/vllm/kv_cache_metadata.py"
spec = importlib.util.spec_from_file_location("kv_metadata_under_test", MODULE)
metadata = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metadata)


@pytest.fixture
def hma():
    path = ROOT / "ucm/integration/vllm/hma_connector.py"
    tree = ast.parse(path.read_text())
    connector = next(
        node for node in tree.body if getattr(node, "name", None) == "UCMFAWAConnector"
    )
    nodes = [
        node
        for node in tree.body
        if getattr(node, "name", None)
        in ("KVCacheGroupLayout", "select_transfer_views")
    ]
    nodes += [
        node
        for node in connector.body
        if getattr(node, "name", None) in ("_store_tensor_size_list", "_create_store")
    ]
    namespace = {
        "torch": torch,
        "np": np,
        "math": math,
        "Optional": Optional,
        "Sequence": Sequence,
        "Tuple": tuple,
        "extract_layer_index": lambda name: int(name.split(".")[2]),
        "logger": Mock(),
        "UcmKVStoreBaseV1": object,
    }
    exec(  # noqa: S102
        compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace
    )
    return NS(**namespace)


def connector_for(
    hma, caches, *, whole_page=None, tail_tokens=8, tail_blocks=1, label="FA"
):
    specs = {
        name: NS(
            dtype=(value[0] if isinstance(value, tuple) else value).dtype, block_size=8
        )
        for name, value in caches.items()
    }
    group = NS(layer_names=list(caches), kv_cache_spec=NS(kv_cache_specs=specs))
    selected = {
        name: tensor
        for name, tensor in caches.items()
        if whole_page is None or name in whole_page
    }
    layout = hma.KVCacheGroupLayout(
        selected,
        is_ascend_layout=True,
        expected_block_size=8,
        whole_page_sizes_by_layer=whole_page,
    )
    connector = NS(
        kv_caches=caches,
        group_layouts={0: layout},
        group_metas={
            0: NS(token_block_size=8, tail_tokens=tail_tokens, tail_blocks=tail_blocks)
        },
        _kv_cache_config=NS(kv_cache_groups=[group]),
        fa_group_ids=[0] if label == "FA" else [],
        window_group_ids=[0] if label == "WA" else [],
        hash_block_size=8,
        file_size={label: 4096},
        engine_id="engine",
        transient_group_ids={1},
        _vllm_config=NS(
            model_config=NS(model="deepseek", dtype=torch.bfloat16),
            cache_config=NS(cache_dtype="auto"),
            attention_config=NS(indexer_kv_dtype="int8"),
            parallel_config=NS(rank=0, data_parallel_rank=0, tensor_parallel_size=1),
        ),
    )
    sizes = hma._store_tensor_size_list(
        connector,
        connector.group_layouts,
        connector.fa_group_ids if label == "FA" else connector.window_group_ids,
    )
    config = {
        "tensor_size_list": sizes,
        "store_pipeline": "Cache|Posix",
        "storage_backends": ["/cache/fawa_" + label.lower()],
    }
    return connector, config


def record(tmp_path, data, payload, padding=0xFF):
    path = tmp_path / "committed-key"
    path.write_bytes(
        payload + bytes([padding]) * (data["store"]["record_size_bytes"] - len(payload))
    )
    return path


def test_v4_mixed_dtypes_follow_real_store_order(hma, tmp_path):
    main = torch.ones((2, 8, 1, 2), dtype=torch.bfloat16)
    main[0, 3, 0, 1] = float("nan")
    key = torch.full((2, 8, 1, 2), -1, dtype=torch.int8)
    scale = torch.ones((2, 8, 1, 1), dtype=torch.float16)
    scale[0, 5] = float("inf")
    caches = {"model.layers.0.attn": main, "model.layers.0.indexer": (key, scale)}
    connector, config = connector_for(hma, caches)
    data = metadata.build_fawa_metadata(connector, "FA", config)
    payload = b"".join(
        ctypes.string_at(int(ptr), size)
        for ptr, size in zip(
            connector.group_layouts[0].base_ptrs, config["tensor_size_list"]
        )
    )
    result = metadata.inspect_file(data, record(tmp_path, data, payload))
    assert result["status"] == "nan_found"
    assert (result["nan_count"], result["inf_count"], result["integer_bytes"]) == (
        1,
        1,
        16,
    )
    assert result["issues"][0]["first_nan_byte"] == 14
    assert result["issues"][1]["first_inf_byte"] == 58
    assert result["opaque_bytes"] == 0
    assert result["padding_bytes"] == 4096 - len(payload)
    assert [segment["size_bytes"] for segment in data["segments"]] == config[
        "tensor_size_list"
    ]


def test_v41_owner_page_includes_indexer_scale_and_ignores_padding(hma, tmp_path):
    raw = torch.full((2 * 4096 + 16,), 0xFF, dtype=torch.uint8)
    main = torch.as_strided(
        raw.view(torch.bfloat16), (2, 8, 1, 2), (2048, 2, 2, 1), storage_offset=8
    )
    key = torch.as_strided(
        raw.view(torch.int8), (2, 8, 1, 2), (4096, 2, 2, 1), storage_offset=48
    )
    scale = torch.as_strided(
        raw.view(torch.float16), (2, 8, 1, 1), (2048, 1, 1, 1), storage_offset=32
    )
    main.fill_(1)
    key.fill_(-1)
    scale.fill_(1)
    scale[0, 4] = float("nan")
    caches = {
        "model.layers.0.long_kv_cache": main,
        "model.layers.0.indexer.k_cache": (key, scale),
    }
    connector, config = connector_for(
        hma, caches, whole_page={"model.layers.0.long_kv_cache": 4096}
    )
    data = metadata.build_fawa_metadata(connector, "FA", config)
    result = metadata.inspect_file(
        data, record(tmp_path, data, ctypes.string_at(main.data_ptr(), 4096))
    )
    assert config["tensor_size_list"] == [4096]
    assert result["nan_count"] == 1
    assert result["issues"][0]["first_nan_byte"] == 56
    assert result["issues"][0]["name"].endswith("indexer.k_cache")
    assert result["checked_float_bytes"] == 48
    assert result["integer_bytes"] == 16
    assert result["padding_bytes"] == 4032
    assert result["opaque_bytes"] == 0


def test_full_alias_retains_typed_value_and_scale_regions(hma, tmp_path):
    raw = torch.zeros((2 * 32,), dtype=torch.uint8)
    key = torch.as_strided(raw.view(torch.int8), (2, 8, 1, 2), (32, 2, 2, 1))
    scale = torch.as_strided(
        raw.view(torch.float16), (2, 8, 1, 1), (16, 1, 1, 1), storage_offset=8
    )
    full = raw.view(2, 8, 1, 4)
    scale[0, 2] = float("nan")
    connector, config = connector_for(
        hma, {"model.layers.0.indexer": (key, scale, full)}
    )
    data = metadata.build_fawa_metadata(connector, "FA", config)
    assert config["tensor_size_list"] == [32]
    result = metadata.inspect_file(
        data, record(tmp_path, data, bytes(raw[:32].tolist()))
    )
    assert result["nan_count"] == 1
    assert result["issues"][0]["first_nan_byte"] == 20
    assert result["opaque_bytes"] == 0


def test_wa_partial_tail_and_repeated_blocks(hma, tmp_path):
    tensor = torch.ones((2, 8, 1, 1), dtype=torch.float32)
    tensor[0, 0] = float("nan")
    tensor[0, 6] = float("nan")
    connector, config = connector_for(
        hma, {"model.layers.0.state": tensor}, tail_tokens=4, label="WA"
    )
    data = metadata.build_fawa_metadata(connector, "WA", config)
    result = metadata.inspect_file(
        data, record(tmp_path, data, ctypes.string_at(tensor.data_ptr() + 16, 16))
    )
    assert result["nan_count"] == 1
    assert result["issues"][0]["first_nan_byte"] == 8
    connector, config = connector_for(
        hma, {"model.layers.0.state": tensor}, tail_tokens=16, tail_blocks=2, label="WA"
    )
    data = metadata.build_fawa_metadata(connector, "WA", config)
    result = metadata.inspect_file(
        data, record(tmp_path, data, ctypes.string_at(tensor.data_ptr(), 64))
    )
    assert [s["file_offset_bytes"] for s in data["segments"]] == [0, 32]
    assert result["nan_count"] == 2


def test_opaque_packed_bytes_do_not_report_a_clean_file(hma, tmp_path):
    tensor = torch.full((2, 8, 1, 4), 255, dtype=torch.uint8)
    connector, config = connector_for(hma, {"model.layers.0.packed": tensor})
    data = metadata.build_fawa_metadata(connector, "FA", config)
    result = metadata.inspect_file(data, record(tmp_path, data, bytes([255]) * 32))
    assert result["status"] == "incomplete"
    assert (
        result["nan_count"],
        result["opaque_bytes"],
        result["checked_float_bytes"],
    ) == (0, 32, 0)


def test_size_errors_and_metadata_integrity(hma, tmp_path):
    tensor = torch.ones((2, 8, 1, 1), dtype=torch.bfloat16)
    connector, config = connector_for(hma, {"model.layers.0.attn": tensor})
    data = metadata.build_fawa_metadata(connector, "FA", config)
    path = tmp_path / "truncated-key"
    path.write_bytes(b"\x00")
    assert metadata.inspect_file(data, path)["status"] == "invalid_size"
    path = record(tmp_path, data, b"\x00" * 16)
    bad = copy.deepcopy(data)
    bad["segments"][0]["regions"][0]["size_bytes"] += 2
    with pytest.raises(ValueError):
        metadata.inspect_file(bad, path)
    config["store_pipeline"] = "Cache|Compress|Posix"
    with pytest.raises(ValueError, match="uncompressed"):
        metadata.inspect_file(
            metadata.build_fawa_metadata(connector, "FA", config), path
        )
    with pytest.raises(ValueError, match="completed"):
        metadata.inspect_file(data, tmp_path / "key.tmp")


def test_cli_atomic_export_and_no_nan_result(hma, tmp_path):
    tensor = torch.ones((2, 8, 1, 1), dtype=torch.bfloat16)
    connector, config = connector_for(hma, {"model.layers.0.attn": tensor})
    path = metadata.write_fawa_metadata(connector, "FA", config, tmp_path / "metadata")
    data = json.loads(path.read_text())
    payload = ctypes.string_at(tensor.data_ptr(), 16)
    binary = record(tmp_path, data, payload)
    command = [sys.executable, str(MODULE), "--metadata", str(path), str(binary)]
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    binary.write_bytes(struct.pack("<H", 0x7FC0) + binary.read_bytes()[2:])
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    assert result.returncode == 1
    assert json.loads(result.stdout)["nan_count"] == 1
    metadata.write_fawa_metadata(connector, "FA", config, path.parent)
    assert list(path.parent.iterdir()) == [path]


@pytest.fixture
def directory_scan(hma, tmp_path):
    tensor = torch.ones((2, 8, 1, 1), dtype=torch.bfloat16)
    connector, config = connector_for(hma, {"model.layers.0.attn": tensor})
    json_path = metadata.write_fawa_metadata(
        connector, "FA", config, tmp_path / "metadata"
    )
    root = tmp_path / "data"
    root.mkdir()
    payload = ctypes.string_at(tensor.data_ptr(), 16) + bytes(4096 - 16)
    return json_path, root, payload


def test_cli_recursively_scans_nested_files_and_skips_tmp(directory_scan):
    json_path, root, payload = directory_scan
    nested = root / "ab" / "cd"
    nested.mkdir(parents=True)
    (root / "key-one").write_bytes(payload)
    (nested / "key-two").write_bytes(payload)
    (nested / "writing.tmp").write_bytes(b"unfinished")
    (nested / "parent-link").symlink_to(root, target_is_directory=True)
    command = [sys.executable, str(MODULE), "--metadata", str(json_path), str(root)]
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert result.returncode == 0, result.stderr
    assert len(rows) == 1
    assert rows[-1] == {
        "type": "summary",
        "scanned_files": 2,
        "skipped_tmp_files": 1,
        "nan_count": 0,
        "inf_count": 0,
        "opaque_bytes": 0,
        "directory_errors": 0,
        "statuses": {"no_nan_in_typed_regions": 2},
    }


def test_cli_accepts_files_and_multiple_directories_together(directory_scan, tmp_path):
    json_path, root, payload = directory_scan
    explicit = tmp_path / "explicit-key"
    explicit.write_bytes(payload)
    (root / "first-key").write_bytes(payload)
    another = tmp_path / "another-directory"
    another.mkdir()
    (another / "second-key").write_bytes(payload)
    command = [
        sys.executable,
        str(MODULE),
        "--metadata",
        str(json_path),
        str(explicit),
        str(root),
        str(another),
    ]
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert result.returncode == 0, result.stderr
    assert len(rows) == 1
    assert rows[-1]["scanned_files"] == 3
    assert rows[-1]["statuses"] == {"no_nan_in_typed_regions": 3}


@pytest.mark.parametrize("status", ["nan_found", "invalid_size"])
def test_cli_directory_continues_after_an_invalid_file(directory_scan, status):
    json_path, root, payload = directory_scan
    (root / "a-good").write_bytes(payload)
    bad = struct.pack("<H", 0x7FC0) + payload[2:] if status == "nan_found" else b"short"
    (root / "b-bad").write_bytes(bad)
    (root / "c-good").write_bytes(payload)
    command = [sys.executable, str(MODULE), "--metadata", str(json_path), str(root)]
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert result.returncode == 1
    assert len(rows) == 2
    assert rows[0]["file"] == str(root / "b-bad")
    assert rows[0]["status"] == status
    assert rows[-1]["scanned_files"] == 3
    assert rows[-1]["statuses"] == {"no_nan_in_typed_regions": 2, status: 1}
    assert rows[-1]["nan_count"] == int(status == "nan_found")


@pytest.mark.parametrize("tmp_only", [False, True])
def test_cli_directory_requires_at_least_one_checked_file(directory_scan, tmp_only):
    json_path, root, _ = directory_scan
    if tmp_only:
        (root / "writing.tmp").write_bytes(b"unfinished")
    command = [sys.executable, str(MODULE), "--metadata", str(json_path), str(root)]
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    summary = json.loads(result.stdout)
    assert result.returncode == 1
    assert summary["scanned_files"] == 0
    assert summary["skipped_tmp_files"] == int(tmp_only)
    assert summary["statuses"] == {}


def test_cli_reports_directory_traversal_errors(directory_scan, monkeypatch, capsys):
    json_path, root, payload = directory_scan
    (root / "good-key").write_bytes(payload)

    def walk(path, *, onerror):
        onerror(PermissionError(13, "Permission denied", str(path / "private")))
        yield str(path), [], ["good-key"]

    monkeypatch.setattr(metadata.os, "walk", walk)
    monkeypatch.setattr(
        sys, "argv", [str(MODULE), "--metadata", str(json_path), str(root)]
    )
    assert metadata.main() == 1
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(rows) == 2
    assert rows[0]["status"] == "error"
    assert rows[-1]["directory_errors"] == 1
    assert rows[-1]["scanned_files"] == 1


@pytest.mark.parametrize("enabled", [False, True])
def test_store_startup_exports_only_when_enabled(hma, tmp_path, monkeypatch, enabled):
    tensor = torch.ones((2, 8, 1, 1), dtype=torch.bfloat16)
    connector, config = connector_for(hma, {"model.layers.0.attn": tensor})
    connector._role = "worker"
    connector._base_store_config = lambda _: ("UcmPipelineStore", None, dict(config))
    connector._get_world_size = lambda: 1
    connector._configure_numa_placement = lambda _: None
    connector._set_default_shm_buffer_capacity = lambda _: None
    connector._publish_block_size = lambda *_args, **_kwargs: None
    connector._summarize_store_config = lambda value: value
    connector.device_id = 0
    connector.is_v41_layout = True
    created = object()
    hma._create_store.__globals__.update(
        KVConnectorRole=NS(WORKER="worker"),
        UcmConnectorFactoryV1=NS(create_connector=lambda *_: created),
        round_up=lambda n, a: (n + a - 1) // a * a,
        _get_store_gc_block_size=lambda _, _sizes, _shard, block: block,
    )
    monkeypatch.setitem(sys.modules, "ucm.integration.vllm.kv_cache_metadata", metadata)
    monkeypatch.delenv("UCM_KV_CACHE_METADATA_DIR", raising=False)
    if enabled:
        monkeypatch.setenv("UCM_KV_CACHE_METADATA_DIR", str(tmp_path))
    assert (
        hma._create_store(connector, "FA", "fa", config["tensor_size_list"]) is created
    )
    files = list(tmp_path.glob("*.json"))
    assert len(files) == int(enabled)
    if enabled:
        assert (
            json.loads(files[0].read_text())["segments"][0]["regions"][0]["dtype"]
            == "torch.bfloat16"
        )


@pytest.mark.parametrize(
    "dtype,code,values,nans,infs",
    [
        ("torch.bfloat16", "H", [0, 0x7F80, 0xFF80, 0x7F81, 0xFFC0], 2, 2),
        ("torch.float16", "H", [0, 0x7C00, 0xFC00, 0x7C01, 0xFE00], 2, 2),
        (
            "torch.float32",
            "I",
            [0, 0x7F800000, 0xFF800000, 0x7F800001, 0xFFC00000],
            2,
            2,
        ),
        (
            "torch.float64",
            "Q",
            [0, 0x7FF0000000000000, 0xFFF0000000000000, 0x7FF0000000000001],
            1,
            2,
        ),
        ("torch.float8_e4m3fn", "B", [0, 0x7E, 0x7F, 0xFF], 2, 0),
        ("torch.float8_e5m2", "B", [0, 0x7C, 0xFC, 0x7D, 0xFF], 2, 2),
        ("torch.float8_e4m3fnuz", "B", [0, 0x7F, 0x80, 0xFF], 1, 0),
        ("torch.float8_e5m2fnuz", "B", [0, 0x7F, 0x80, 0xFF], 1, 0),
        ("torch.float8_e8m0fnu", "B", [0, 0x80, 0xFE, 0xFF], 1, 0),
    ],
)
def test_float_bit_patterns_across_chunk_boundary(
    tmp_path, dtype, code, values, nans, infs
):
    payload = bytes(1 << 20) + b"".join(
        struct.pack("<" + code, value) for value in values
    )
    size = len(payload)
    data = {
        "schema_version": 1,
        "byte_order": "little",
        "store": {
            "label": "FA",
            "raw_posix_supported": True,
            "record_size_bytes": size,
            "payload_size_bytes": size,
            "trailing_padding_bytes": 0,
        },
        "segments": [
            {
                "group_id": 0,
                "tail_block": 0,
                "file_offset_bytes": 0,
                "size_bytes": size,
                "opaque_bytes": 0,
                "padding_bytes": 0,
                "regions": [
                    {
                        "offset_bytes": 0,
                        "size_bytes": size,
                        "dtype": dtype,
                        "name": "test",
                        "component": "0",
                    }
                ],
            }
        ],
    }
    result = metadata.inspect_file(data, record(tmp_path, data, payload))
    assert (result["nan_count"], result["inf_count"]) == (nans, infs)
    assert result["status"] == "nan_found"


def test_packed_float8_views_are_opaque_even_with_a_float_dtype(hma, tmp_path):
    tensor = torch.zeros((2, 8, 1, 4), dtype=torch.float8_e4m3fn)
    connector, config = connector_for(hma, {"model.layers.0.packed": tensor})
    spec = connector._kv_cache_config.kv_cache_groups[0].kv_cache_spec.kv_cache_specs[
        "model.layers.0.packed"
    ]
    spec.model_version = "deepseek_v4"
    data = metadata.build_fawa_metadata(connector, "FA", config)
    assert (
        metadata.inspect_file(data, record(tmp_path, data, bytes(32)))["status"]
        == "incomplete"
    )
