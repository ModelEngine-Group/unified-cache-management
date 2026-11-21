"""Verify the bytes at the addresses passed to the fixed-size Store."""

import ctypes
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ucm.integration.vllm.partial_tail import PartialTailBuffer


@pytest.mark.parametrize("layerwise", [False, True])
def test_tail_copies_only_valid_tokens_to_registered_pages(layerwise):
    # Physical HND order, logical NHD view. Include separate K and V pages.
    caches = tuple(torch.full((3, 2, 128, 4), 73.0).transpose(1, 2) for _ in range(2))
    for index, tensor in enumerate(caches):
        tensor[2, :20] = index + 1
    base = np.array([t.data_ptr() for t in caches], dtype=np.uint64)
    if layerwise:
        base = base.reshape(1, -1)
    page_bytes = caches[0][0].nbytes
    layout = SimpleNamespace(
        use_layerwise=layerwise,
        base_ptrs=base,
        tensor_size_lists=np.full_like(base, page_bytes),
        block_stride_lists=np.full_like(base, page_bytes),
    )
    buf = PartialTailBuffer({"layer.0": caches}, layout, 128)
    addresses = buf.stage(2, 20, 0 if layerwise else None)
    for index, addr in enumerate(addresses[0]):
        assert (int(addr), page_bytes) in buf.registrations
        raw = ctypes.string_at(int(addr), page_bytes)
        stored = np.frombuffer(raw, dtype=np.float32).reshape(2, 128, 4)
        assert np.all(stored[:, :20] == index + 1)
        assert np.all(stored[:, 20:] == 0)
        assert torch.all(caches[index][2, 20:] == 73)
