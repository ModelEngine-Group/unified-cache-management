"""Preserve the ranges of K/V views interleaved by kernel row."""

from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.stage(0), pytest.mark.feature("connector_v2")]


@pytest.mark.parametrize("with_value", [False, True], ids=["key-only", "key-and-value"])
def test_interleaved_kernel_rows(with_value):
    import torch

    from ucm.integration.vllm.v2.cache_view import build_layer_view

    rows, tokens, channels, blocks = 12, 128, 256, 2
    storage = torch.empty(
        (blocks * rows * 2, 1, tokens, channels), dtype=torch.bfloat16
    )
    key, value = storage[::2], storage[1::2]
    layer = SimpleNamespace(
        num_blocks=blocks,
        storage_block_size=rows * tokens,
        attention_view_order="BHNC",
        layer_name="layer.3.attn",
        layer_index=3,
    )
    view = build_layer_view(
        (key, value) if with_value else key,
        layer,
        state_snapshot=False,
        device_type="npu",
    )
    row_bytes = tokens * channels * key.element_size()
    stride = rows * 2 * row_bytes
    assert not view.supports_partial_tokens
    if with_value:
        assert len(view.segments) == 1
        segment = view.segments[0]
        assert (
            segment.base_ptr,
            segment.payload_bytes,
            segment.block_stride_bytes,
        ) == (
            storage.data_ptr(), stride, stride
        )
    else:
        assert len(view.segments) == rows
        assert [segment.base_ptr for segment in view.segments] == [
            key.data_ptr() + row * 2 * row_bytes for row in range(rows)
        ]
        assert all(segment.payload_bytes == row_bytes for segment in view.segments)
        assert all(segment.block_stride_bytes == stride for segment in view.segments)
