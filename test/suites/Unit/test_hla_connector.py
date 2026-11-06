"""CPU-only checks for HLA decisions and transfer metadata.

Like test_kv_cache_layout.py, load production methods via AST to avoid
requiring torch, vLLM or NPU libraries. These tests do not emulate kernels
or establish the safety of every model-selection/state-copy configuration.
"""

import ast
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, List, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
HLA_PATH = ROOT / "ucm/integration/vllm/hla_connector.py"


class KVCacheSpec(SimpleNamespace):
    pass


class FullAttentionSpec(KVCacheSpec):
    pass


class MLAAttentionSpec(FullAttentionSpec):
    pass


class MambaSpec(KVCacheSpec):
    pass


class UniformTypeKVCacheSpecs(KVCacheSpec):
    pass


class FakeTensor:
    def __init__(self, ptr, shape, strides, element_size=2):
        self.ptr = ptr
        self.shape = shape
        self.strides = strides
        self.item_size = element_size

    def numel(self):
        return math.prod(self.shape)

    def dim(self):
        return len(self.shape)

    def stride(self, dim):
        return self.strides[dim]

    def element_size(self):
        return self.item_size

    def data_ptr(self):
        return self.ptr


@pytest.fixture
def hla():
    functions = {
        "_kv_cache_tensor_layers",
        "layer_name_to_kv_cache_spec",
        "block_size_from_kv_cache_spec",
        "is_mamba_align_kv_cache_spec",
        "participates_in_prefix_caching",
        "extend_non_null",
        "kv_cache_layout_name",
        "is_cuda_blnhc_layout",
        "is_npu_layer_major_blnhc",
        "npu_flash_next_store_columns",
        "npu_flash_next_column_owners",
        "ple_stage_row_map",
        "group_state_flags",
        "state_group_ids",
        "blnhc_layer_role",
        "blnhc_group_role",
    }
    methods = {
        "KVCacheGroupManager": {
            "__init__",
            "num_groups",
            "compute_block_hashes",
            "compute_all_group_block_ids",
        },
        "HybridLinearAttentionLayout": {
            "_tokens_per_state",
            "_has_glm53_layout",
            "_build_layout",
            "_build_glm53_ascend_layout",
            "_validate_glm53_ascend_descriptors",
            "_build_glm53_ascend_row",
            "_finalize_glm53_ascend_columns",
            "_finalize_layout_arrays",
            "_extract_group_addrs",
        },
        "UCMHybridLinearAttentionConnector": {
            "_scope_blocks",
            "_append_mamba_align_state_block",
            "_generate_hla_dispatch_meta",
            "build_connector_meta",
            "start_load_kv",
            "_mark_load_failed",
        },
        "UCMHybridLinearAttentionLayerWiseConnector": {
            "_submit_layer_load",
            "_plan_row_saves",
            "wait_for_save",
            "_is_full_cudagraph",
            "start_load_kv",
            "_submit_request_load_tasks_for_row",
            "_wait_row_load",
            "_mark_load_failed",
            "_record_layerwise_load_bytes",
        },
    }
    nodes = []
    found_unhashed_roles = False
    for node in ast.parse(HLA_PATH.read_text(encoding="utf-8-sig")).body:
        if getattr(node, "name", None) in functions | {
            "GroupInfo",
            "HLARequestMeta",
            "HLARequestDispatchMeta",
        }:
            nodes.append(node)
        elif isinstance(node, ast.ClassDef) and node.name in methods:
            node.bases = []
            node.body = [
                item
                for item in node.body
                if getattr(item, "name", None) in methods[node.name]
            ]
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_BLNHC_UNHASHED_ROLES"
            for target in node.targets
        ):
            nodes.append(node)
            found_unhashed_roles = True
    if not found_unhashed_roles:
        raise AssertionError(
            "_BLNHC_UNHASHED_ROLES assignment missing from hla_connector.py"
        )
    connector_path = ROOT / "ucm/integration/vllm/ucm_connector.py"
    base_nodes = [
        node
        for node in ast.parse(connector_path.read_text(encoding="utf-8-sig")).body
        if isinstance(node, ast.ClassDef)
        and node.name
        in {
            "KVCacheSegment",
            "RequestMeta",
            "RequestDispatchMeta",
            "UCMConnectorMetadata",
        }
    ]
    for node in base_nodes:
        if node.name == "UCMConnectorMetadata":
            node.bases = []
    nodes = base_nodes + nodes
    namespace = {
        "__name__": __name__,
        "math": math,
        "time": time,
        "np": np,
        "defaultdict": defaultdict,
        "dataclass": dataclass,
        "field": field,
        "Callable": Callable,
        "List": List,
        "Optional": Optional,
        "KVCacheSpec": KVCacheSpec,
        "FullAttentionSpec": FullAttentionSpec,
        "MLAAttentionSpec": MLAAttentionSpec,
        "MambaSpec": MambaSpec,
        "UniformTypeKVCacheSpecs": UniformTypeKVCacheSpecs,
        "torch": SimpleNamespace(Tensor=FakeTensor),
        "logger": MagicMock(),
        "ucmmetrics": MagicMock(),
        "current_platform": SimpleNamespace(
            device_type="npu", is_cuda_alike=lambda: False
        ),
    }
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future, *nodes], type_ignores=[])
    )
    exec(compile(module, str(HLA_PATH), "exec"), namespace)
    namespace["UCMHybridLinearAttentionLayerWiseConnector"]._mark_load_failed = (
        namespace["UCMHybridLinearAttentionConnector"]._mark_load_failed
    )
    return SimpleNamespace(**namespace)


def group(spec, *names):
    return SimpleNamespace(kv_cache_spec=spec, layer_names=list(names))


@pytest.fixture
def direct_load_connector(hla):
    connector = hla.UCMHybridLinearAttentionConnector()
    unscoped = [b"kv0", b"kv1", b"state", b"state"]
    scoped = [b"kv0", b"kv1", b"state@1", b"state@1"]
    blocks = [10, 11, 20, 21]
    groups = [0, 0, 1, 1]
    dispatch = SimpleNamespace(
        load_full_attn_count=2, load_block_ids=(unscoped, blocks)
    )
    metadata = hla.UCMConnectorMetadata(request_meta={"req": dispatch})
    connector._get_connector_metadata = lambda: metadata
    connector._reset_transfer_state = lambda: None
    connector._prepare_load_plan = lambda _: None
    connector.request_data = [("req", unscoped, scoped, blocks, groups)]
    connector.is_mla = True
    connector.need_load = True
    connector.layer_order = ["mla", "state"]
    connector.store = object()
    connector._failure_req_ids = set()
    connector._invalid_block_ids = set()
    connector._connector_worker_meta = SimpleNamespace(mark_failed=MagicMock())
    connector._load_bytes_total = 0
    connector.device = SimpleNamespace(synchronize=MagicMock())
    layout = hla.HybridLinearAttentionLayout.__new__(hla.HybridLinearAttentionLayout)
    layout.direct_group_bytes = {0: 64, 1: 16}
    layout.extract_block_addrs = lambda ids, gids: np.array(
        list(zip(ids, gids)), dtype=np.uint64
    )
    connector.kv_cache_layout = layout
    return connector


@pytest.mark.parametrize(
    "failure", [None, "submit_mla", "submit_state", "wait_mla", "wait_state"]
)
def test_direct_load_splits_shared_and_private_tasks_and_drains_failures(
    hla, direct_load_connector, failure
):
    connector = direct_load_connector
    submissions = []
    waits = []

    def submit(store, requests, keys, shards, ptrs, *, distribute_backend_load):
        label = "mla" if distribute_backend_load else "state"
        assert store is connector.store
        submissions.append((label, requests, keys, shards, ptrs.tolist()))
        if failure == f"submit_{label}":
            raise RuntimeError("submit failed")
        return SimpleNamespace(label=label)

    def wait(task):
        waits.append(task.label)
        if failure == f"wait_{task.label}":
            raise RuntimeError("wait failed")

    connector._rank_consistency = SimpleNamespace(submit_load=submit, wait_load=wait)
    connector.start_load_kv(SimpleNamespace())
    connector.device.synchronize.assert_called_once()
    assert submissions[0] == (
        "mla",
        {"req": [b"kv0", b"kv1"]},
        [b"kv0", b"kv1"],
        [0, 0],
        [[10, 0], [11, 0]],
    )
    if failure != "submit_mla":
        assert submissions[1] == (
            "state",
            {"req": [b"state", b"state"]},
            [b"state@1", b"state@1"],
            [0, 0],
            [[20, 1], [21, 1]],
        )
    expected_waits = [] if failure == "submit_mla" else ["mla"]
    if failure not in ("submit_mla", "submit_state"):
        expected_waits.append("state")
    assert waits == expected_waits
    stats = hla.ucmmetrics.update_stats.call_args.args[0]
    assert stats["load_requests_num"] == (0 if failure else 1)
    if failure:
        assert connector._failure_req_ids == {"req"}
        assert connector._invalid_block_ids == {10, 11, 20, 21}
    else:
        assert stats["load_blocks_num"] == 4
        assert stats["load_bytes_total"] == 160
        assert not connector._failure_req_ids


@pytest.mark.parametrize("shared", [False, True])
def test_direct_load_skips_empty_task_type(hla, direct_load_connector, shared):
    connector = direct_load_connector
    start, end = (0, 2) if shared else (2, 4)
    request_id, unscoped, scoped, blocks, groups = connector.request_data[0]
    connector.request_data = [
        (
            request_id,
            unscoped[start:end],
            scoped[start:end],
            blocks[start:end],
            groups[start:end],
        )
    ]
    dispatch = connector._get_connector_metadata().request_meta[request_id]
    dispatch.load_full_attn_count = 2 if shared else 0
    connector._rank_consistency = SimpleNamespace(
        submit_load=MagicMock(return_value=object()), wait_load=MagicMock()
    )
    connector.start_load_kv(SimpleNamespace())
    connector._rank_consistency.submit_load.assert_called_once()
    assert connector._rank_consistency.submit_load.call_args.kwargs == {
        "distribute_backend_load": shared
    }
    connector._rank_consistency.wait_load.assert_called_once()


def test_gqa_direct_load_keeps_all_rank_scoped_keys_in_one_task(direct_load_connector):
    connector = direct_load_connector
    connector.is_mla = False
    connector._rank_consistency = SimpleNamespace(
        submit_load=MagicMock(return_value=object()), wait_load=MagicMock()
    )
    connector.start_load_kv(SimpleNamespace())
    connector._rank_consistency.submit_load.assert_called_once()
    call = connector._rank_consistency.submit_load.call_args
    assert call.args[2] == [b"kv0", b"kv1", b"state@1", b"state@1"]
    assert call.kwargs == {"distribute_backend_load": False}
    connector._rank_consistency.wait_load.assert_called_once()


@pytest.mark.parametrize(
    "is_mla,group_id,expected", [(True, 0, True), (True, 1, False), (False, 0, False)]
)
def test_layerwise_load_distributes_only_shared_mla_group(
    hla, is_mla, group_id, expected
):
    connector = hla.UCMHybridLinearAttentionLayerWiseConnector()
    connector.is_mla = is_mla
    connector.layer_order = ["layer"]
    connector.layer_to_group = {"layer": group_id}
    connector.layer_to_slot = {"layer": 0}
    connector._submitted_positions = set()
    connector._submitted_group_slots = set()
    connector._failure_req_ids = set()
    connector._load_bytes_by_pair = defaultdict(int)
    connector.load_tasks = defaultdict(dict)
    connector._kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            group(MLAAttentionSpec()),
            group(MambaSpec(mamba_cache_mode="align")),
        ]
    )
    connector._state_group_ids = hla.state_group_ids(connector._kv_cache_config)
    metadata = hla.UCMConnectorMetadata(request_meta={})
    connector._get_connector_metadata = lambda: metadata
    connector.request_data = [("req", [b"key"], [b"scoped"], [10], [group_id])]
    connector.store = object()
    layout = hla.HybridLinearAttentionLayout.__new__(hla.HybridLinearAttentionLayout)
    layout.extract_layer_addrs = lambda name, blocks: np.array([[10]], dtype=np.uint64)
    layout.layer_column_bytes = {"layer": 16}
    connector.kv_cache_layout = layout
    task = object()
    connector._rank_consistency = SimpleNamespace(
        submit_load=MagicMock(return_value=task)
    )
    connector._submit_layer_load(0)
    assert connector._rank_consistency.submit_load.call_args.kwargs == {
        "distribute_backend_load": expected
    }
    assert connector.load_tasks[(group_id, 0)]["req"] is task
    assert connector._load_bytes_by_pair[(group_id, 0)] == 16


@pytest.mark.parametrize(
    "fields,expected",
    [
        ({"layers": ["new"]}, ["new"]),
        ({"shared_by": ["old"]}, ["old"]),
        ({"layers": ["new"], "shared_by": ["old"]}, ["new"]),
        ({"layers": [], "shared_by": ["old"]}, []),
        ({"layers": None, "shared_by": ["old"]}, ["old"]),
        ({"shared_by": None}, []),
        ({"layers": None, "shared_by": None}, []),
        ({"layers": ("a", "b")}, ("a", "b")),
        ({}, []),
    ],
)
def test_tensor_layer_names_preserve_values_and_handle_none(hla, fields, expected):
    result = hla._kv_cache_tensor_layers(SimpleNamespace(**fields))
    assert result == expected
    assert isinstance(result, type(expected))


@pytest.mark.parametrize(
    "layer_name,expected",
    [
        ("model.layers.0.linear_attn.raw_key", "raw_key"),
        # Engine QSA owner name: dotted segment is raw_key_cache, not raw_key.
        ("model.layers.0.indexer.raw_key_cache", "raw_key"),
        ("model.layers.0.linear_attn.k_compress", "compressed"),
        ("model.layers.0.linear_attn.ple_state", "ple"),
        ("model.layers.0.self_attn.attn", ""),
        ("model.layers.0.linear_attn.raw_keypad", ""),
    ],
)
def test_blnhc_layer_role(hla, layer_name, expected):
    assert hla.blnhc_layer_role(layer_name) == expected


def test_blnhc_group_role_requires_every_layer(hla):
    assert hla.blnhc_group_role(["model.layers.0.linear_attn.ple_state"]) == "ple"
    assert (
        hla.blnhc_group_role(
            [
                "model.layers.0.self_attn.attn",
                "model.layers.0.linear_attn.k_compress",
            ]
        )
        == ""
    )


@pytest.mark.parametrize(
    "fields,expected",
    [
        ({}, True),
        ({"prefix_cacheable": False}, False),
        ({"participates_in_prefix_caching": False}, False),
        ({"prefix_cacheable": True, "participates_in_prefix_caching": False}, False),
        ({"prefix_cacheable": False, "participates_in_prefix_caching": True}, False),
        ({"prefix_cacheable": True, "participates_in_prefix_caching": True}, True),
    ],
)
def test_prefix_eligibility_checks_both_descriptor_versions(hla, fields, expected):
    assert hla.participates_in_prefix_caching(KVCacheSpec(**fields)) is expected


def test_nested_uniform_prefix_eligibility(hla):
    tail = KVCacheSpec(prefix_cacheable=False)
    nested = UniformTypeKVCacheSpecs(kv_cache_specs={"tail": tail})
    outer = UniformTypeKVCacheSpecs(
        kv_cache_specs={"attention": FullAttentionSpec(), "nested": nested}
    )
    assert not hla.participates_in_prefix_caching(outer)
    tail.prefix_cacheable = True
    assert hla.participates_in_prefix_caching(outer)


def test_tail_is_excluded_from_hashes_and_lcm_without_renumbering(hla):
    config = SimpleNamespace(
        kv_cache_groups=[
            group(FullAttentionSpec(block_size=4), "mla"),
            group(KVCacheSpec(block_size=7, prefix_cacheable=False), "tail"),
            group(MambaSpec(block_size=6, mamba_cache_mode="align"), "gdn"),
        ]
    )
    hasher = MagicMock(return_value=b"seed")
    hash_request = MagicMock(return_value=[b"a", b"b", b"c"])
    hasher.make_request_block_hasher.return_value = hash_request
    manager = hla.KVCacheGroupManager(config, hasher, b"base")
    assert manager.lcm_block_size == 12
    assert [g.group_id for g in manager.full_attn_groups] == [0]
    assert [g.group_id for g in manager.non_prefix_groups] == [1]
    assert [g.group_id for g in manager.state_groups] == [2]
    assert manager.compute_all_group_block_ids(
        SimpleNamespace(all_token_ids=list(range(12)))
    ) == [[b"a", b"b", b"c"], [], [b"", b""]]
    hash_request.assert_called_once()


@pytest.mark.parametrize(
    "fields,expected",
    [
        ({}, 1),
        ({"compress_ratio": 4}, 4),
        ({"tokens_per_state": 2, "compress_ratio": 4}, 2),
    ],
)
def test_compression_field_fallback(hla, fields, expected):
    assert (
        hla.HybridLinearAttentionLayout._tokens_per_state(KVCacheSpec(**fields))
        == expected
    )


def _glm_detection_layout(hla, hf_fields):
    specs = {
        "mla": MLAAttentionSpec(tokens_per_state=1),
        "indexer": MLAAttentionSpec(tokens_per_state=4),
    }
    config = SimpleNamespace(
        kv_cache_tensors=[SimpleNamespace(layers=["mla", "indexer", "state"])],
        kv_cache_groups=[
            group(UniformTypeKVCacheSpecs(kv_cache_specs=specs), *specs),
            group(MambaSpec(mamba_cache_mode="align"), "state"),
        ],
    )
    layout = hla.HybridLinearAttentionLayout()
    layout.kv_cache_config = config
    layout.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(**hf_fields))
    )
    return layout


@pytest.mark.parametrize(
    "fields,expected",
    [
        ({"architectures": ["Glm5NextForConditionalGeneration"]}, True),
        ({"architectures": ["OtherModel", "Glm5NextForConditionalGeneration"]}, True),
        ({"architectures": ["Glm5NextForCausalLM"]}, False),
        ({"architectures": ["OtherHybridForConditionalGeneration"]}, False),
        ({"architectures": ["Glm5NextForConditionalGenerationExtra"]}, False),
        ({"architectures": []}, False),
        ({"architectures": None}, False),
        ({}, False),
    ],
)
def test_glm_detection_uses_exact_supported_architecture(hla, fields, expected):
    layout = _glm_detection_layout(hla, fields)
    assert layout._has_glm53_layout(layout.kv_cache_config) is expected


@pytest.mark.parametrize("tag", [None, "other_model", "glm5_next"])
def test_similar_structure_and_spec_tags_do_not_override_other_architecture(hla, tag):
    layout = _glm_detection_layout(
        hla, {"architectures": ["OtherHybridForConditionalGeneration"]}
    )
    specs = layout.kv_cache_config.kv_cache_groups[0].kv_cache_spec.kv_cache_specs
    for spec in specs.values():
        spec.model_version = tag
    assert not layout._has_glm53_layout(layout.kv_cache_config)


@pytest.mark.parametrize(
    "device,expected", [("npu", True), ("cuda", True), ("cpu", False)]
)
def test_glm_detection_preserves_platform_guard(hla, device, expected):
    layout = _glm_detection_layout(
        hla, {"architectures": ["Glm5NextForConditionalGeneration"]}
    )
    hla.current_platform.device_type = device
    hla.current_platform.is_cuda_alike = lambda: device == "cuda"
    assert layout._has_glm53_layout(layout.kv_cache_config) is expected


def test_glm_detection_requires_allocations(hla):
    layout = _glm_detection_layout(
        hla, {"architectures": ["Glm5NextForConditionalGeneration"]}
    )
    layout.kv_cache_config.kv_cache_tensors = []
    assert not layout._has_glm53_layout(layout.kv_cache_config)


def test_supported_architecture_selects_ascend_layout(hla):
    layout = _glm_detection_layout(
        hla, {"architectures": ["Glm5NextForConditionalGeneration"]}
    )
    layout._build_glm53_ascend_layout = MagicMock()
    caches = {"mla": object()}
    layout._build_layout(caches)
    layout._build_glm53_ascend_layout.assert_called_once_with(caches)


def test_supported_architecture_explicitly_rejects_cuda(hla):
    layout = _glm_detection_layout(
        hla, {"architectures": ["Glm5NextForConditionalGeneration"]}
    )
    hla.current_platform.device_type = "cuda"
    hla.current_platform.is_cuda_alike = lambda: True
    with pytest.raises(
        NotImplementedError, match="CUDA KV cache layout is not supported"
    ):
        layout._build_layout({})


@pytest.mark.parametrize(
    "architecture", ["OtherHybridForConditionalGeneration", "Glm5NextForCausalLM"]
)
def test_other_architectures_keep_generic_cuda_path(hla, architecture):
    layout = _glm_detection_layout(hla, {"architectures": [architecture]})
    hla.current_platform.device_type = "cuda"
    hla.current_platform.is_cuda_alike = lambda: True
    layout._build_glm53_ascend_layout = MagicMock()
    # Exercise the generic dispatch branch without requiring physical tensors.
    layout._collect_shared_tensor_info = MagicMock(return_value=([], []))
    layout._build_layout({})
    layout._collect_shared_tensor_info.assert_called_once()
    layout._build_glm53_ascend_layout.assert_not_called()


def test_empty_rope_is_skipped_before_contiguity_checks(hla):
    layout = hla.HybridLinearAttentionLayout()
    layout.num_blocks = 2
    key = FakeTensor(1000, (4, 2, 4), (8, 4, 1))
    rope = FakeTensor(2000, (4, 2, 0), (8, 4, 1))
    rows = layout._build_glm53_ascend_row(
        "mla",
        {"mla": (key, rope)},
        SimpleNamespace(size=64),
        KVCacheSpec(page_size_bytes=32),
    )
    assert len(rows) == 1
    segment = rows[0]
    assert (
        segment.ptr,
        segment.copy_size,
        segment.block_stride,
        segment.buffer_size,
    ) == (
        1000,
        32,
        32,
        64,
    )


def test_padded_state_copies_only_live_bytes(hla):
    layout = hla.HybridLinearAttentionLayout()
    layout.num_blocks = 2
    rows = layout._build_glm53_ascend_row(
        "gdn",
        {"gdn": FakeTensor(1000, (2, 2, 3), (16, 3, 1))},
        SimpleNamespace(size=64),
        KVCacheSpec(page_size_bytes=32),
    )
    assert (rows[0].copy_size, rows[0].block_stride, rows[0].buffer_size) == (
        12,
        32,
        44,
    )


def test_nonempty_noncontiguous_view_is_rejected(hla):
    layout = hla.HybridLinearAttentionLayout()
    layout.num_blocks = 2
    with pytest.raises(ValueError, match="Non-contiguous Ascend KV block"):
        layout._build_glm53_ascend_row(
            "mla",
            {"mla": FakeTensor(1000, (2, 2, 3), (16, 4, 1))},
            SimpleNamespace(size=64),
            KVCacheSpec(page_size_bytes=32),
        )


@pytest.mark.parametrize("compressed_first", [False, True])
def test_glm_families_preserve_column_order_and_pad_rows(hla, compressed_first):
    specs = {
        "mla0": MLAAttentionSpec(page_size_bytes=16, tokens_per_state=1),
        "mla1": MLAAttentionSpec(page_size_bytes=16, tokens_per_state=1),
        "index": MLAAttentionSpec(page_size_bytes=16, tokens_per_state=4),
    }
    names = ["index", "mla0", "mla1"] if compressed_first else ["mla0", "index", "mla1"]
    layout = hla.HybridLinearAttentionLayout()
    layout.num_blocks = 2
    layout.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            group(UniformTypeKVCacheSpecs(kv_cache_specs=specs), *names),
            group(KVCacheSpec(page_size_bytes=16, prefix_cacheable=False), "tail"),
            group(MambaSpec(page_size_bytes=16), "gdn"),
        ],
        kv_cache_tensors=[
            SimpleNamespace(layers=[name], size=32, block_stride=16)
            for name in [*names, "tail", "gdn"]
        ],
    )

    def tensor(ptr):
        return FakeTensor(ptr, (2, 2), (8, 1))

    layout._build_glm53_ascend_layout(
        {
            "mla0": tensor(1000),
            "mla1": tensor(3000),
            "index": tensor(2000),
            "gdn": (tensor(5000), tensor(5004)),
        }
    )
    first = [2000, 1000] if compressed_first else [1000, 2000]
    second = [0, 3000] if compressed_first else [3000, 0]
    assert layout.base_ptrs.tolist() == [*first, 5000, 5004, *second, 0, 0]
    assert layout.row_tensor_size_lists == [[4, 4, 4, 4], [4, 4, 4, 4]]
    assert layout.row_shard_sizes == [16, 16]
    assert set(layout.group_layouts) == {0, 2}
    assert layout.preload_all_rows and layout.save_after_forward
    assert layout.layer_name_to_row["mla1"] == 1
    assert "tail" not in layout.layer_name_to_row
    addresses = layout._extract_group_addrs([1, 1], [0, 2])
    assert addresses[0].tolist() == [
        first[0] + 16,
        first[1] + 16,
        0,
        0,
        *[value + 16 if value else 0 for value in second],
        0,
        0,
    ]
    assert addresses[1].tolist() == [0, 0, 5016, 5020, 0, 0, 0, 0]
    np.testing.assert_array_equal(
        layout._extract_group_addrs([1, 1], [0, 2], layout.row_slices[1]),
        addresses[:, 4:],
    )


def test_group_address_validation_and_empty_batch(hla):
    layout = hla.HybridLinearAttentionLayout()
    layout.base_ptrs = np.array([1000], dtype=np.uint64)
    layout.group_layouts = {0: (layout.base_ptrs, np.array([16], dtype=np.uint64))}
    with pytest.raises(ValueError, match="lengths differ"):
        layout._extract_group_addrs([1], [])
    with pytest.raises(ValueError, match="No hybrid physical layout"):
        layout._extract_group_addrs([1], [5])
    empty = layout._extract_group_addrs([], [])
    assert empty.shape == (0, 1)
    assert empty.flags.c_contiguous


@pytest.fixture
def save_connector(hla):
    connector = hla.UCMHybridLinearAttentionLayerWiseConnector()
    connector._metadata_factory = hla.UCMConnectorMetadata
    specs = {
        "attn0": FullAttentionSpec(),
        "state2": MambaSpec(),
        "attn3": FullAttentionSpec(),
        "index4": MLAAttentionSpec(tokens_per_state=4),
        "state9": MambaSpec(),
        "attn8": FullAttentionSpec(),
        "tail100": KVCacheSpec(prefix_cacheable=False),
    }
    connector._kv_cache_config = SimpleNamespace(
        kv_cache_groups=[group(spec, name) for name, spec in specs.items()]
    )
    connector.kv_cache_layout = SimpleNamespace(save_after_forward=False)
    connector.layer_name_to_id = {
        "attn0": 0,
        "state2": 2,
        "attn3": 3,
        "index4": 4,
        "state9": 9,
        "attn8": 8,
        "tail100": 100,
    }
    connector.layer_name_to_row = {
        "attn0": 0,
        "state2": 0,
        "attn3": 1,
        "index4": 1,
        "state9": 2,
        "attn8": 3,
        "tail100": 0,
    }
    connector.row_ids = [0, 1, 2, 3]
    connector.commit_row_id = 3
    return connector


def test_row_saves_wait_for_last_component_and_exclude_commit(save_connector):
    # index4 cannot be a callback; tail100 must not delay row 0.
    # state9 has no subsequent callback, so row 2 stays for post-forward.
    assert save_connector._plan_row_saves() == {"attn3": [0], "attn8": [1]}


def test_no_attention_callback_leaves_rows_for_post_forward(save_connector):
    save_connector._kv_cache_config.kv_cache_groups = [
        group(MambaSpec(), name) for name in save_connector.layer_name_to_row
    ]
    assert save_connector._plan_row_saves() == {}


def test_glm_saves_are_all_deferred(save_connector):
    save_connector.kv_cache_layout.save_after_forward = True
    assert save_connector._plan_row_saves() == {}


@pytest.mark.parametrize("already_saved", [set(), {0, 1}])
def test_post_forward_backfills_rows_before_commit(save_connector, already_saved):
    connector = save_connector
    events = []
    metadata = connector._metadata_factory(request_meta={})
    connector._connector_metadata = metadata
    connector._get_connector_metadata = lambda: metadata
    connector._dumped_row_ids = set(already_saved)
    connector.dump_tasks = {}
    connector._dump_transfer_data = (None, None, None, {"req"})
    connector._layerwise_save_bytes = 0
    connector.is_save = True
    connector.enable_event_sync = False
    connector._rank_consistency = SimpleNamespace(
        finish_dump=lambda requests: events.append(("finish", requests))
    )

    def submit(row, meta):
        assert meta is metadata
        events.append(("submit", row))
        connector._dumped_row_ids.add(row)

    connector._submit_dump_row = submit
    connector._wait_dump_rows = lambda rows: events.append(("wait", rows))
    connector.wait_for_save()
    assert events == [
        *(("submit", row) for row in [0, 1, 2] if row not in already_saved),
        ("wait", [0, 1, 2]),
        ("submit", 3),
        ("wait", [3]),
        ("finish", {"req"}),
    ]
    assert connector._dump_transfer_data is None
    assert not connector.is_save


@pytest.mark.parametrize("is_mla", [False, True])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("is_dump", [False, True])
def test_rank_scoping_keeps_keys_blocks_and_groups_paired(hla, is_mla, rank, is_dump):
    connector = hla.UCMHybridLinearAttentionConnector()
    connector.tp_rank = rank
    connector.tp_size = 2
    connector.is_mla = is_mla
    connector.request_hasher = lambda key: b"rank1:" + key
    result = connector._scope_blocks([b"mla", b"gdn"], [11, 22], [0, 2], 1, is_dump)
    selected = [1] if is_mla and rank == 1 and is_dump else [0, 1]
    expected_keys = [b"mla", b"gdn"]
    expected_scoped = [
        b"mla" if is_mla or rank == 0 else b"rank1:mla",
        b"gdn" if rank == 0 else b"rank1:gdn",
    ]
    assert result == tuple(
        [values[index] for index in selected]
        for values in [expected_keys, expected_scoped, [11, 22], [0, 2]]
    )


def test_null_filter_keeps_group_pairing(hla):
    keys, blocks, groups = [], [], []
    hla.extend_non_null(keys, blocks, groups, [b"a", b"b", b"c"], [3, 0, 5], 7)
    assert (keys, blocks, groups) == ([b"a", b"c"], [3, 5], [7, 7])


@pytest.fixture
def state_connector(hla):
    connector = hla.UCMHybridLinearAttentionConnector()
    connector.is_mla = True
    connector.group_manager = SimpleNamespace(
        num_groups=2,
        lcm_block_size=4,
        groups_by_id=[
            SimpleNamespace(
                block_size=4,
                is_full_attention=True,
                is_mamba_align=False,
                is_state_hash=False,
            ),
            SimpleNamespace(
                block_size=4,
                is_full_attention=False,
                is_mamba_align=True,
                is_state_hash=True,
            ),
        ],
        compute_mamba_align_state_hash=MagicMock(return_value=b"checkpoint"),
    )
    return connector


@pytest.mark.parametrize(
    "seq_len,scheduled,table,reason,expected",
    [
        (8, 1, [21, 22, 23, 99, 100], "load", [22, 23]),
        (8, 4, [21, 22, 23, 99, 100], "load", [22, 23]),
        (8, 0, [21, 22, 99, 100], "load", [22]),
        (7, 1, [21, 22, 99, 100], "load", [22]),
        (8, 1, [21, 22, 22, 99, 100], "load", [22]),
        (8, 1, [21, 0, 23, 99, 100], "load", [23]),
        (8, 1, [21, 22, 0, 99, 100], "load", [22]),
        (8, 1, [21, 0, 0, 99, 100], "load", []),
        (8, 4, [21, 22, 23, 99, 100], "dump", [22]),
    ],
)
def test_state_restore_selects_checkpoint_and_running_not_scratch(
    hla, state_connector, seq_len, scheduled, table, reason, expected
):
    req_meta = hla.HLARequestMeta(
        group_vllm_block_ids=[[11, 12, 13], table],
        group_ucm_block_ids=[[b"a", b"b", b"c"], []],
    )
    keys, blocks, groups = [], [], []
    state_connector._append_mamba_align_state_block(
        keys, blocks, groups, req_meta, "req", 1, seq_len, reason, scheduled
    )
    assert blocks == expected
    assert keys == [b"checkpoint"] * len(expected)
    assert groups == [1] * len(expected)
    assert 99 not in blocks and 100 not in blocks
    if expected:
        state_connector.group_manager.compute_mamba_align_state_hash.assert_called_once_with(
            state_connector.group_manager.groups_by_id[1],
            seq_len,
            req_meta.group_ucm_block_ids,
        )
    else:
        state_connector.group_manager.compute_mamba_align_state_hash.assert_not_called()


@pytest.mark.parametrize(
    "failure", ["missing_checkpoint", "missing_running", "missing_hash"]
)
def test_incomplete_state_restore_does_not_append_partial_metadata(
    hla, state_connector, failure
):
    tables = {
        "missing_checkpoint": [21],
        "missing_running": [21, 22],
        "missing_hash": [21, 22, 23],
    }
    if failure == "missing_hash":
        state_connector.group_manager.compute_mamba_align_state_hash.return_value = None
    req_meta = hla.HLARequestMeta(
        group_vllm_block_ids=[[], tables[failure]],
        group_ucm_block_ids=[[], []],
    )
    keys, blocks, groups = [b"existing"], [7], [0]
    state_connector._append_mamba_align_state_block(
        keys, blocks, groups, req_meta, "req", 1, 8, "load", 1
    )
    assert (keys, blocks, groups) == ([b"existing"], [7], [0])
    hla.logger.error.assert_called_once()


def _restore_request_meta(hla, *, hbm_hits=0):
    return hla.HLARequestMeta(
        hbm_hit_block_num=hbm_hits,
        total_hit_block_num=2,
        num_token_ids=40,
        token_processed=8,
        group_ucm_block_ids=[
            [b"fa0", b"fa1", b"fa2", b"fa3"],
            [b"", b"", b"", b""],
        ],
        group_vllm_block_ids=[
            [11, 12, 13, 14],
            [21, 22, 23, 99, 100],
        ],
    )


@pytest.mark.parametrize("format_name", ["legacy_list", "resumed_flags", "resumed_set"])
@pytest.mark.parametrize("resumed", [False, True])
def test_cached_request_only_reloads_on_resume(
    hla, state_connector, format_name, resumed
):
    req_meta = _restore_request_meta(hla)
    state_connector.requests_meta = {"req": req_meta}
    if format_name == "legacy_list":
        cached = [
            SimpleNamespace(
                req_id="req",
                new_block_ids=(
                    tuple(list(ids) for ids in req_meta.group_vllm_block_ids)
                    if resumed
                    else ([], [])
                ),
                resumed_from_preemption=resumed,
            )
        ]
    else:
        cached = SimpleNamespace(
            req_ids=["req"],
            new_block_ids=[
                (
                    tuple(list(ids) for ids in req_meta.group_vllm_block_ids)
                    if resumed
                    else None
                )
            ],
        )
        if format_name == "resumed_flags":
            cached.resumed_from_preemption = [resumed]
        else:
            cached.resumed_req_ids = {"req"} if resumed else set()
    output = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached,
        num_scheduled_tokens={"req": 1},
        finished_req_ids=set(),
        preempted_req_ids=set(),
    )
    dispatch = state_connector.build_connector_meta(output).request_meta["req"]
    if resumed:
        assert dispatch.load_block_ids == (
            [b"fa0", b"fa1", b"checkpoint", b"checkpoint"],
            [11, 12, 22, 23],
        )
        assert dispatch.load_group_ids == [0, 0, 1, 1]
        assert not dispatch.load_async
    else:
        # Keep a positive external-hit count to ensure the continuation
        # gate, rather than the absence of a hit, prevents a reload.
        assert dispatch.load_block_ids == ([], [])
        assert dispatch.load_group_ids == []
        state_connector.group_manager.compute_mamba_align_state_hash.assert_not_called()


@pytest.mark.parametrize("hbm_hits", [0, 1, 2])
def test_new_request_restores_only_external_hit_increment(
    hla, state_connector, hbm_hits
):
    req_meta = _restore_request_meta(hla, hbm_hits=hbm_hits)
    state_connector.requests_meta = {"req": req_meta}
    output = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(req_id="req", block_ids=([11, 12, 13], [21, 22, 23, 99]))
        ],
        scheduled_cached_reqs=[],
        num_scheduled_tokens={"req": 1},
        finished_req_ids=set(),
        preempted_req_ids=set(),
    )
    dispatch = state_connector.build_connector_meta(output).request_meta["req"]
    if hbm_hits < 2:
        expected_attention_keys = [b"fa0", b"fa1"][hbm_hits:]
        expected_attention_blocks = [11, 12][hbm_hits:]
        assert dispatch.load_block_ids == (
            expected_attention_keys + [b"checkpoint", b"checkpoint"],
            expected_attention_blocks + [22, 23],
        )
        assert dispatch.load_group_ids == [0] * (2 - hbm_hits) + [1, 1]
    else:
        assert dispatch.load_block_ids == ([], [])
        state_connector.group_manager.compute_mamba_align_state_hash.assert_not_called()


@pytest.mark.parametrize(
    "new_tokens,expected",
    [
        (8, [16]),
        (14, []),
    ],
)
def test_dump_writes_state_only_when_step_ends_on_lcm_boundary(
    hla, state_connector, new_tokens, expected
):
    req_meta = _restore_request_meta(hla)
    state_connector._generate_hla_dispatch_meta(
        req_meta, new_tokens, ([], []), request_id="req", need_load=False
    )
    seq_lens = [
        call.args[1]
        for call in (
            state_connector.group_manager.compute_mamba_align_state_hash.call_args_list
        )
    ]
    assert seq_lens == expected


@pytest.mark.parametrize("row_count", [2, 3])
def test_glm_preload_waits_all_state_rows_before_returning(
    hla, state_connector, row_count
):
    dispatch = state_connector._generate_hla_dispatch_meta(
        _restore_request_meta(hla), 1, ([], []), request_id="req"
    )
    metadata = hla.UCMConnectorMetadata(request_meta={"req": dispatch})
    connector = hla.UCMHybridLinearAttentionLayerWiseConnector()
    connector._get_connector_metadata = lambda: metadata
    connector.tp_rank = 0
    connector.tp_size = 1
    connector.is_mla = True
    connector._scope_blocks = state_connector._scope_blocks
    state_connector.tp_rank = 0
    state_connector.tp_size = 1
    connector.row_ids = list(range(row_count))
    connector._load_prefetch_rows = 0
    connector._row_shard_size = 16
    connector.load_tasks = defaultdict(dict)
    connector.request_data = []
    connector._failure_req_ids = set()
    connector._submitted_load_rows = set()
    connector._dumped_row_ids = set()
    connector._load_block_counts = {}
    connector._invalid_block_ids = set()
    connector._connector_worker_meta = SimpleNamespace(mark_failed=MagicMock())
    connector.store = object()
    events = []
    memory = {
        row: {22: b"old-checkpoint", 23: b"old-running", 99: b"scratch"}
        for row in connector.row_ids
    }

    def addresses(blocks, row, group_ids):
        assert blocks == [11, 12, 22, 23]
        assert group_ids == [0, 0, 1, 1]
        return np.array([[block] for block in blocks], dtype=np.uint64)

    connector.kv_cache_layout = SimpleNamespace(
        preload_all_rows=True, extract_block_addrs_for_row=addresses
    )
    connector.device = SimpleNamespace(synchronize=lambda: events.append(("sync",)))

    def submit(store, requests, keys, shards, ptrs):
        assert store is connector.store
        assert requests == {"req": dispatch.load_block_ids[0]}
        row = shards[0]
        assert shards == [row] * len(keys)
        events.append(("submit", row))
        return SimpleNamespace(row=row, keys=list(keys), blocks=ptrs[:, 0].tolist())

    def wait(task):
        events.append(("wait", task.row))
        # Fake the external DMA: pre-existing destination contents are
        # replaced only when this row's load is completed.
        for key, block in zip(task.keys, task.blocks):
            if block in memory[task.row]:
                memory[task.row][block] = key

    connector._rank_consistency = SimpleNamespace(submit_load=submit, wait_load=wait)
    connector.start_load_kv(SimpleNamespace())
    assert events == [
        ("sync",),
        *(("submit", row) for row in connector.row_ids),
        *(("wait", row) for row in connector.row_ids),
    ]
    assert not connector.load_tasks
    assert connector._layerwise_load_bytes == row_count * 4 * 16
    assert connector._layerwise_load_bytes_recorded
    # A consumer proceeding after start_load_kv sees both restored slots
    # for every row, while the unused speculative tail remains untouched.
    for values in memory.values():
        assert values == {22: b"checkpoint", 23: b"checkpoint", 99: b"scratch"}


def test_npu_flash_next_whole_store_columns(hla):
    columns, shard, block = hla.npu_flash_next_store_columns(
        12, 524288, 32768, 184320, False
    )
    assert columns == [524288] * 12 + [32768] * 12 + [184320]
    assert shard == block == 6868992


def test_npu_flash_next_layerwise_columns(hla):
    columns, shard, block = hla.npu_flash_next_store_columns(
        12, 524288, 32768, 184320, True
    )
    assert columns == [524288, 32768]
    assert shard == 557056
    assert block == 6684672


def test_npu_flash_next_column_owners(hla):
    whole = hla.npu_flash_next_column_owners(12, 0, [1, 2, 3], 4, False)
    assert whole[0] == {(index, 0) for index in range(24)}
    assert whole[1] == {(index, 0) for index in range(12)}
    assert whole[4] == {(24, 0)}
    layerwise = hla.npu_flash_next_column_owners(12, 0, [1], 4, True)
    assert layerwise[0] == {(type_id, slot) for type_id in (0, 1) for slot in range(12)}
    assert layerwise[1] == {(0, slot) for slot in range(12)}
    assert layerwise[4] == {(0, 0)}


def test_ple_stage_row_map_skips_null_and_duplicates(hla):
    assert hla.ple_stage_row_map([0, 7, 7, 3, 0]) == {7: 0, 3: 1}
    assert hla.ple_stage_row_map([]) == {}


def test_non_mamba_ple_is_a_state_group(hla):
    config = SimpleNamespace(
        kv_cache_layout="BLNHC",
        kv_cache_groups=[
            group(FullAttentionSpec(), "layers.0.self_attn"),
            group(FullAttentionSpec(), "layers.1.ple"),
        ],
        kv_cache_tensors=[
            SimpleNamespace(
                layers=["layers.0.self_attn", "layers.1.linear"],
                layer_stride=200,
                block_stride=50,
            )
        ],
    )
    assert hla.state_group_ids(config) == {1}
    assert hla.group_state_flags(config, config.kv_cache_groups[1]) == (False, True)


def test_is_npu_layer_major_blnhc(hla):
    def config(name, tensors):
        return SimpleNamespace(kv_cache_layout=name, kv_cache_tensors=tensors)

    multi = SimpleNamespace(
        layers=["a", "b"], layer_stride=2104881696, block_stride=524288
    )
    assert hla.is_npu_layer_major_blnhc(config("BLNHC", [multi]))
    block_first = SimpleNamespace(
        layers=["a", "b"], layer_stride=100, block_stride=524288
    )
    assert not hla.is_npu_layer_major_blnhc(config("BLNHC", [block_first]))
    assert not hla.is_npu_layer_major_blnhc(config("LBNHC", [multi]))
    ple_only = SimpleNamespace(
        layers=["ple"], layer_stride=695070720, block_stride=184320
    )
    assert not hla.is_npu_layer_major_blnhc(config("BLNHC", [ple_only]))


class _FakeCUDAGraphMode:
    """Mirrors vLLM CUDAGraphMode.has_full_cudagraphs() semantics."""

    def __init__(self, name: str):
        self.name = name

    def has_full_cudagraphs(self) -> bool:
        return self.name in {"FULL", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE"}


def _layerwise_start_load_fixture(hla, cudagraph_mode: str):
    connector = hla.UCMHybridLinearAttentionLayerWiseConnector()
    connector.need_load = True
    connector.layer_order = ["l0", "l1", "l2", "l3"]
    connector._vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            cudagraph_mode=_FakeCUDAGraphMode(cudagraph_mode)
        )
    )
    connector.device = SimpleNamespace(synchronize=MagicMock())
    connector._reset_transfer_state = MagicMock()
    connector._prepare_load_plan = MagicMock()
    connector._get_connector_metadata = MagicMock(
        return_value=hla.UCMConnectorMetadata(request_meta={})
    )
    submitted: list[int] = []
    drained: list[bool] = []
    connector._submit_layer_load = lambda pos: submitted.append(pos)
    connector._drain_unwaited_loads = lambda: drained.append(True)
    connector._record_load_bytes = MagicMock()
    return connector, submitted, drained


@pytest.mark.parametrize(
    "cudagraph_mode",
    ["FULL", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE"],
)
def test_layerwise_full_cudagraph_loads_all_positions_before_forward(
    hla, cudagraph_mode
):
    connector, submitted, drained = _layerwise_start_load_fixture(hla, cudagraph_mode)
    connector.start_load_kv(SimpleNamespace())
    assert submitted == [0, 1, 2, 3]
    assert drained == [True]
    connector._record_load_bytes.assert_called_once()


@pytest.mark.parametrize("cudagraph_mode", ["NONE", "PIECEWISE"])
def test_layerwise_non_full_keeps_depth2_prefetch(hla, cudagraph_mode):
    connector, submitted, drained = _layerwise_start_load_fixture(hla, cudagraph_mode)
    connector.start_load_kv(SimpleNamespace())
    assert submitted == [0, 1]
    assert drained == []
    connector._record_load_bytes.assert_not_called()
