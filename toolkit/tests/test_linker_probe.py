"""Necessary address-contract checks; no inference model or SGLang required."""

from types import SimpleNamespace

import pytest
from ucm_toolkit.tools.sglang_model_check.linker_probe import transfer_views


def test_transfer_views_uses_native_addresses():
    torch = pytest.importorskip("torch")
    buffer = torch.arange(32, dtype=torch.uint8)
    entry = SimpleNamespace(
        kv_buffer=[buffer],
        layer_mapping={0: 0},
        prepare_locations=lambda indices: indices,
        get_prepared_layer_range_meta=lambda locations, layer: (
            [[buffer.data_ptr() + 8]],
            [[4]],
            [[0]],
        ),
    )
    group = SimpleNamespace(
        entry_map={"kv": entry},
        resolve_transfers=lambda transfers: transfers,
    )
    views = transfer_views(group, [SimpleNamespace(name="kv", host_indices=[1])])
    assert len(views) == 1
    assert views[0].tolist() == [8, 9, 10, 11]
    views[0].zero_()
    assert buffer[8:12].tolist() == [0, 0, 0, 0]
    assert buffer[12].item() == 12


def test_transfer_views_rejects_invalid_addresses():
    torch = pytest.importorskip("torch")
    buffer = torch.zeros(8, dtype=torch.uint8)
    entry = SimpleNamespace(
        kv_buffer=[buffer],
        layer_mapping={0: 0},
        prepare_locations=lambda indices: indices,
        get_prepared_layer_range_meta=lambda locations, layer: (
            [[buffer.data_ptr() + 9]],
            [[1]],
            [[0]],
        ),
    )
    group = SimpleNamespace(
        entry_map={"kv": entry},
        resolve_transfers=lambda transfers: transfers,
    )
    with pytest.raises(ValueError, match="outside registered"):
        transfer_views(group, [SimpleNamespace(name="kv", host_indices=[1])])
