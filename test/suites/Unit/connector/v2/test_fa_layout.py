"""Full-attention transfer ranges on CPU-backed vLLM MRV2 cache tensors."""

import numpy as np
import pytest

pytestmark = [pytest.mark.stage(0), pytest.mark.feature("connector_v2")]

BLOCK_TOKENS = 16
CHUNK_TOKENS = 64
LAYER_NAMES = ("model.layers.0.self_attn", "model.layers.1.self_attn")


@pytest.fixture(
    params=[
        ("LBHNC", True, False),
        ("BLHNC", True, False),
        ("BLHNC", False, False),
        ("LBHNC", True, True),
        ("BLHNC", True, True),
    ]
)
def fa_cache(request):
    import torch
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheTensor,
        compute_layout_strides,
    )
    from vllm.v1.kv_cache_layout import KVCacheLayout
    from vllm.v1.worker.utils import allocate_kv_cache

    layout_name, reverse_layers, padded = request.param
    fa = FullAttentionSpec(
        block_size=BLOCK_TOKENS,
        num_kv_heads=2,
        head_size=64,
        dtype=torch.float16,
        page_size_padded=12288 if padded else None,
    )
    packing = KVCacheLayout[layout_name]
    num_blocks = 16
    # Layer First keeps model order. Block First preserves physical placement.
    layers = list(reversed(LAYER_NAMES)) if reverse_layers else list(LAYER_NAMES)
    layer_stride, block_stride, *_ = compute_layout_strides(
        fa, num_blocks, len(layers), packing
    )
    if padded and layout_name == "BLHNC":
        # Also leave unused space after this group's last physical page.
        block_stride += 4096
    tensor = KVCacheTensor(
        size=(
            num_blocks * block_stride
            if layout_name == "BLHNC"
            else num_blocks * len(layers) * fa.page_size_bytes
        ),
        layers=layers,
        layer_stride=layer_stride,
        block_stride=block_stride,
    )
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[tensor],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=layers, kv_cache_spec=fa)],
        kv_cache_layout=packing.name,
    )
    return config, allocate_kv_cache(config, torch.device("cpu"), packing)


def _allocate_external_prefix(config, num_tokens):
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.request import Request

    request = Request(
        request_id="cold-fa",
        prompt_token_ids=list(range(num_tokens + 1)),
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
    )
    manager = KVCacheManager(
        config,
        max_model_len=request.num_prompt_tokens,
        scheduler_block_size=BLOCK_TOKENS,
        hash_block_size=BLOCK_TOKENS,
        enable_caching=False,
    )
    # No local prefix. Reserve the external prefix plus one token for prefill.
    allocated = manager.allocate_slots(
        request, num_new_tokens=1, num_external_computed_tokens=num_tokens
    )
    assert allocated is not None
    (block_ids,) = manager.get_block_ids_for_computed_tokens(
        request.request_id, num_tokens
    )
    assert len(block_ids) == num_tokens // BLOCK_TOKENS
    return block_ids


@pytest.mark.parametrize("layer_id", [None, 0, 1], ids=["bulk", "layer0", "layer1"])
def test_full_chunk_transfer_ranges(fa_cache, layer_id):
    from ucm.integration.vllm.v2 import (
        ParallelRank,
        ParallelTopology,
        RestorePolicy,
        UCMKVCacheLayout,
        parse_kv_cache_config,
    )

    config, views = fa_cache
    spec = parse_kv_cache_config(
        config,
        scheduler_block_size=BLOCK_TOKENS,
        chunk_size=CHUNK_TOKENS,
        hf_text_config=None,
        device_type="cpu",
    )
    layout = UCMKVCacheLayout(
        spec, views, kv_cache_config=config, num_hidden_layers=len(LAYER_NAMES)
    )
    layout.build_store_layouts(ParallelTopology(), ParallelRank())

    # Two complete chunks use different device pages and the same record offsets.
    block_ids = _allocate_external_prefix(config, 2 * CHUNK_TOKENS)
    store_layout = layout.store_layouts[0]
    block_first = config.kv_cache_layout == "BLHNC"
    if block_first:
        assert store_layout.group_layout.payload_bytes.size == 1
        assert not store_layout.group_layout.supports_partial_tokens
        if layer_id is not None:
            with pytest.raises(ValueError, match="whole-group IO"):
                store_layout.group_layout.segment_mask(layer_id)
            return
    offsets, ptrs, sizes = store_layout.resolve_transfer_ranges(
        np.asarray(block_ids, dtype=np.uint64),
        np.array([0, CHUNK_TOKENS], dtype=np.uint64),
        segment_mask=store_layout.group_layout.segment_mask(layer_id),
    )

    # Walk the native tensor pages in record order, including layers omitted
    # from a layerwise transfer when advancing the offset.
    blocks_per_chunk = CHUNK_TOKENS // BLOCK_TOKENS
    expected_rows = []
    for start in range(0, len(block_ids), blocks_per_chunk):
        row = []
        offset = 0
        for block_id in block_ids[start : start + blocks_per_chunk]:
            if block_first:
                descriptor = config.kv_cache_tensors[0]
                base = views[descriptor.layers[0]][block_id].data_ptr()
                width = len(descriptor.layers) * descriptor.layer_stride
                row.append((offset, base, width))
                offset += width
                continue
            for index, name in enumerate(LAYER_NAMES):
                page = views[name][block_id]
                page_bytes = page.numel() * page.element_size()
                if layer_id is None or index == layer_id:
                    row.append((offset, page.data_ptr(), page_bytes))
                offset += page_bytes
        expected_rows.append(row)
        assert layout.store_sizes_by_policy == {RestorePolicy.FA: offset}

    expected = np.asarray(expected_rows, dtype=np.uint64)
    assert layout.store_offsets_by_group == {0: 0}
    np.testing.assert_array_equal(offsets, expected[:, :, 0])
    np.testing.assert_array_equal(ptrs, expected[:, :, 1])
    np.testing.assert_array_equal(sizes, expected[:, :, 2])
