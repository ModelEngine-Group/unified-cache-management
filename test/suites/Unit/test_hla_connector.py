"""CPU-only checks for HLA decisions and transfer metadata.

Like test_kv_cache_layout.py, load production methods via AST to avoid
requiring torch, vLLM or NPU libraries. These tests do not emulate kernels
or establish the safety of every model-selection/state-copy configuration.
"""

import ast
import math
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
        },
        "UCMHybridLinearAttentionLayerWiseConnector": {
            "_plan_row_saves",
            "wait_for_save",
            "start_load_kv",
            "_submit_request_load_tasks_for_row",
            "_wait_row_load",
            "_mark_load_failed",
            "_record_layerwise_load_bytes",
        },
    }
    nodes = []
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
    return SimpleNamespace(**namespace)


def group(spec, *names):
    return SimpleNamespace(kv_cache_spec=spec, layer_names=list(names))


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
            SimpleNamespace(block_size=4, is_full_attention=True, is_mamba_align=False),
            SimpleNamespace(block_size=4, is_full_attention=False, is_mamba_align=True),
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
