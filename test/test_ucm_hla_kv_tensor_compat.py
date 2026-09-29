"""KVCacheTensor ``shared_by`` -> ``layers`` rename compatibility tests.

vLLM PR #51718 (KV-cache layout refactor) renamed ``KVCacheTensor.shared_by``
to ``layers`` with no deprecation period. These tests pin the connector
behavior for both interfaces: the compat accessor must resolve the shared
layer names from either field, and the HLA layout / connector-selection
logic must produce identical results for old-style and new-style KV cache
tensors.
"""

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("vllm")

from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheTensor,
    MambaSpec,
)

import ucm.integration.vllm.hla_connector as hla_connector
from ucm.integration.vllm.hla_connector import (
    HybridLinearAttentionLayout,
    UCMHybridLinearAttentionConnector,
    kv_cache_tensor_layer_names,
)

NUM_BLOCKS = 4
RAW_SIZE = 4096
PAGE_SIZE = RAW_SIZE // NUM_BLOCKS

ATTN_LAYER = "model.layers.0.self_attn.attn"
ATTN_LAYER_2 = "model.layers.1.self_attn.attn"
MAMBA_LAYER = "model.layers.0.mamba"

# FullAttentionSpec(block_size=16, num_kv_heads=2, head_size=8, uint8)
# -> k_size = v_size = 16 * 2 * 8 = 256 bytes per block.
K_SIZE = 256
V_SIZE = 256
# MambaSpec components: conv = prod((16, 2, 3)) = 96, ssm = prod((16, 2, 5)) = 160.
CONV_SIZE = 96
SSM_SIZE = 160
MIDDLE_SIZE = max(K_SIZE, SSM_SIZE)
TAIL_SIZE = PAGE_SIZE - CONV_SIZE - MIDDLE_SIZE


def _attn_spec():
    return FullAttentionSpec(
        block_size=16, num_kv_heads=2, head_size=8, dtype=torch.uint8
    )


def _mamba_spec():
    return MambaSpec(
        block_size=16,
        shapes=((16, 2, 3), (16, 2, 5)),
        dtypes=(torch.uint8, torch.uint8),
        mamba_cache_mode="align",
    )


def _old_tensor(size=RAW_SIZE, layers=(ATTN_LAYER, MAMBA_LAYER)):
    return SimpleNamespace(size=size, shared_by=list(layers))


def _new_tensor(size=RAW_SIZE, layers=(ATTN_LAYER, MAMBA_LAYER)):
    return SimpleNamespace(
        size=size,
        layers=list(layers),
        layer_stride=PAGE_SIZE * NUM_BLOCKS,
        block_stride=PAGE_SIZE,
        offset=0,
        host_resident=False,
    )


def _groups():
    return [
        SimpleNamespace(layer_names=[ATTN_LAYER], kv_cache_spec=_attn_spec()),
        SimpleNamespace(layer_names=[ATTN_LAYER_2], kv_cache_spec=_attn_spec()),
        SimpleNamespace(layer_names=[MAMBA_LAYER], kv_cache_spec=_mamba_spec()),
    ]


def _kv_cache_config(tensors):
    return SimpleNamespace(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=tensors,
        kv_cache_groups=_groups(),
    )


def _kvcaches():
    return {
        ATTN_LAYER: torch.zeros(NUM_BLOCKS, 256, dtype=torch.uint8),
        ATTN_LAYER_2: torch.zeros(NUM_BLOCKS, 256, dtype=torch.uint8),
        MAMBA_LAYER: torch.zeros(NUM_BLOCKS, 256, dtype=torch.uint8),
    }


def _fake_platform(device_type):
    return SimpleNamespace(device_type=device_type, is_cuda_alike=lambda: True)


def _build_layout(monkeypatch, tensors, device_type, kvcaches):
    layout = HybridLinearAttentionLayout.__new__(HybridLinearAttentionLayout)
    layout.kv_cache_config = _kv_cache_config(tensors)
    layout.num_blocks = NUM_BLOCKS
    monkeypatch.setattr(hla_connector, "current_platform", _fake_platform(device_type))
    layout._build_layout(kvcaches)
    return layout


def _assert_layouts_equal(old_layout, new_layout):
    for attr in (
        "base_ptrs",
        "buffer_sizes",
        "tensor_size_lists",
        "block_stride_lists",
    ):
        assert np.array_equal(
            getattr(old_layout, attr), getattr(new_layout, attr)
        ), attr
    assert old_layout.row_tensor_size_lists == new_layout.row_tensor_size_lists
    assert old_layout.row_shard_sizes == new_layout.row_shard_sizes
    assert old_layout.layer_name_to_row == new_layout.layer_name_to_row
    assert [s.start for s in old_layout.row_slices] == [
        s.start for s in new_layout.row_slices
    ]
    assert [s.stop for s in old_layout.row_slices] == [
        s.stop for s in new_layout.row_slices
    ]


def test_kv_cache_tensor_layer_names_reads_new_layers_field():
    tensor = _new_tensor(layers=(ATTN_LAYER, MAMBA_LAYER))
    assert kv_cache_tensor_layer_names(tensor) == [ATTN_LAYER, MAMBA_LAYER]


def test_kv_cache_tensor_layer_names_falls_back_to_legacy_shared_by():
    tensor = _old_tensor(layers=(ATTN_LAYER, MAMBA_LAYER))
    assert kv_cache_tensor_layer_names(tensor) == [ATTN_LAYER, MAMBA_LAYER]


def test_kv_cache_tensor_layer_names_prefers_layers_when_both_fields_exist():
    tensor = SimpleNamespace(layers=["new"], shared_by=["old"])
    assert kv_cache_tensor_layer_names(tensor) == ["new"]


def test_kv_cache_tensor_layer_names_with_real_vllm_kv_cache_tensor():
    field_names = {f.name for f in dataclasses.fields(KVCacheTensor)}
    if "shared_by" in field_names:
        tensor = KVCacheTensor(size=64, shared_by=["layer.a", "layer.b"])
    else:
        tensor = KVCacheTensor(
            size=64,
            layers=["layer.a", "layer.b"],
            layer_stride=32,
            block_stride=32,
        )
    assert kv_cache_tensor_layer_names(tensor) == ["layer.a", "layer.b"]


@pytest.mark.parametrize("device_type", ["cuda", "npu"])
def test_hybrid_layout_identical_for_old_and_new_tensor_api(monkeypatch, device_type):
    kvcaches = _kvcaches()
    old_layout = _build_layout(monkeypatch, [_old_tensor()], device_type, kvcaches)
    new_layout = _build_layout(monkeypatch, [_new_tensor()], device_type, kvcaches)

    _assert_layouts_equal(old_layout, new_layout)
    assert old_layout.layer_name_to_row == {ATTN_LAYER: 0, MAMBA_LAYER: 0}

    if device_type == "cuda":
        assert old_layout.tensor_size_lists.tolist() == [PAGE_SIZE]
        assert old_layout.row_shard_sizes == [PAGE_SIZE]
    else:
        assert old_layout.tensor_size_lists.tolist() == [
            CONV_SIZE,
            MIDDLE_SIZE,
            TAIL_SIZE,
        ]
        assert old_layout.row_shard_sizes == [PAGE_SIZE]
        base_ptrs = old_layout.base_ptrs.tolist()
        assert base_ptrs[1] - base_ptrs[0] == CONV_SIZE * NUM_BLOCKS
        assert base_ptrs[2] - base_ptrs[1] == MIDDLE_SIZE * NUM_BLOCKS


def test_attn_only_layout_identical_for_old_and_new_tensor_api(monkeypatch):
    kvcaches = _kvcaches()
    layers = (ATTN_LAYER, ATTN_LAYER_2)
    old_layout = _build_layout(
        monkeypatch, [_old_tensor(layers=layers)], "npu", kvcaches
    )
    new_layout = _build_layout(
        monkeypatch, [_new_tensor(layers=layers)], "npu", kvcaches
    )

    _assert_layouts_equal(old_layout, new_layout)
    assert old_layout.layer_name_to_row == {ATTN_LAYER: 0, ATTN_LAYER_2: 0}
    conv_padding = PAGE_SIZE - K_SIZE - V_SIZE
    assert old_layout.tensor_size_lists.tolist() == [
        conv_padding,
        K_SIZE,
        V_SIZE,
    ]
    assert old_layout.row_shard_sizes == [PAGE_SIZE]


def test_collect_shared_tensor_info_skips_layers_missing_from_kvcaches():
    layout = HybridLinearAttentionLayout.__new__(HybridLinearAttentionLayout)
    layout.kv_cache_config = _kv_cache_config([])
    kvcaches = {ATTN_LAYER: torch.zeros(NUM_BLOCKS, 8, dtype=torch.uint8)}

    specs_old, ptrs_old = layout._collect_shared_tensor_info(_old_tensor(), kvcaches)
    specs_new, ptrs_new = layout._collect_shared_tensor_info(_new_tensor(), kvcaches)

    assert len(specs_old) == len(specs_new) == 1
    assert isinstance(specs_old[0], FullAttentionSpec)
    assert ptrs_old == ptrs_new == [kvcaches[ATTN_LAYER].data_ptr()]


def test_supports_kv_cache_layout_accepts_both_tensor_apis(monkeypatch):
    monkeypatch.setattr(hla_connector, "current_platform", _fake_platform("cuda"))
    for tensor in (_old_tensor(), _new_tensor()):
        assert UCMHybridLinearAttentionConnector.supports_kv_cache_layout(
            _kv_cache_config([tensor])
        )


def test_supports_kv_cache_layout_rejects_non_hla_layouts(monkeypatch):
    monkeypatch.setattr(hla_connector, "current_platform", _fake_platform("cuda"))
    supports = UCMHybridLinearAttentionConnector.supports_kv_cache_layout

    assert not supports(None)
    assert not supports(_kv_cache_config([]))
    attn_only_old = _old_tensor(layers=(ATTN_LAYER, ATTN_LAYER_2))
    attn_only_new = _new_tensor(layers=(ATTN_LAYER,))
    assert not supports(_kv_cache_config([attn_only_old]))
    assert not supports(_kv_cache_config([attn_only_new]))
