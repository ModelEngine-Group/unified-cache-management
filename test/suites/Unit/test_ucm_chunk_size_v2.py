"""Global token chunk sizing and one-time logical/physical slice validation."""

from unittest.mock import patch

import numpy as np
import pytest
from test_ucm_connector_v2 import (
    AscendSlidingWindowMLASpec,
    FakeTensor,
    FullAttentionSpec,
    MambaSpec,
    MLAAttentionSpec,
    UnsupportedSpec,
    config,
    group,
    make_cache_layout,
    parse_spec,
)

from ucm.integration.vllm.v2.connector_worker import UCMConnectorWorker
from ucm.integration.vllm.v2.layout import UCMKVCacheLayout
from ucm.integration.vllm.v2.runtime import ParallelRank, ParallelTopology


def attention_config(*sizes):
    return config(
        *(
            group([f"model.layers.{i}.attn"], FullAttentionSpec(size))
            for i, size in enumerate(sizes)
        )
    )


@pytest.mark.parametrize("dcp,expected,blocks", [(1, 384, (3, 2)), (8, 3072, (3, 2))])
def test_default_resolves_unequal_nondividing_group_sizes_after_dcp(
    dcp, expected, blocks
):
    spec = parse_spec(
        attention_config(128, 192), scheduler_block_size=384 * dcp, dcp_size=dcp
    )
    assert spec.chunk_size == expected
    assert spec.blocks_per_chunk == blocks
    assert [g.block_size for g in spec.groups] == [128, 192]
    assert [g.block_token_span for g in spec.groups] == [128 * dcp, 192 * dcp]


def test_state_constrains_default_boundary_and_still_stores_one_snapshot():
    native = attention_config(128, 192)
    native.kv_cache_groups.append(group(["model.layers.2.mixer"], MambaSpec(2048)))
    native.prefix_cacheable_group_ids = (0, 1, 2)
    spec = parse_spec(native, scheduler_block_size=6144, dcp_size=8)
    assert spec.chunk_size == 6144
    assert spec.blocks_per_chunk == (6, 4, 1)
    assert spec.groups[2].block_token_span == 2048


def test_scratch_and_zero_tail_do_not_inflate_default():
    native = config(
        group(["model.layers.0.attn"], FullAttentionSpec(128)),
        group(["model.layers.1.scratch"], UnsupportedSpec(7, prefix_cacheable=False)),
        group(["model.layers.2.swa_cache"], AscendSlidingWindowMLASpec(192, 1, 0)),
    )
    spec = parse_spec(native, scheduler_block_size=2688)
    assert spec.chunk_size == 128
    assert spec.blocks_per_chunk == (1, 0, 0)


@pytest.mark.parametrize(
    "chunk,blocks", [(64, (1, 1)), (128, (1, 1)), (256, (2, 1)), (512, (4, 2))]
)
def test_explicit_non_dcp_chunk_satisfies_each_fa_group(chunk, blocks):
    spec = parse_spec(
        attention_config(128, 256), scheduler_block_size=256, chunk_size=chunk
    )
    assert spec.chunk_size == chunk
    assert spec.blocks_per_chunk == blocks


@pytest.mark.parametrize("chunk", [0, -64, 96, 192, 384])
def test_explicit_chunk_rejects_invalid_size_or_group_alignment(chunk):
    with pytest.raises(ValueError, match="chunk_size"):
        parse_spec(
            attention_config(128, 256), scheduler_block_size=256, chunk_size=chunk
        )


@pytest.mark.parametrize("dcp", [2, 8])
@pytest.mark.parametrize("chunk", [64, 128, 256])
def test_dcp_does_not_accept_native_only_alignment(dcp, chunk):
    with pytest.raises(ValueError, match="whole distributed KV pages"):
        parse_spec(
            attention_config(128, 256),
            scheduler_block_size=256 * dcp,
            chunk_size=chunk,
            dcp_size=dcp,
        )


@pytest.mark.parametrize(
    "native", [MambaSpec(256), AscendSlidingWindowMLASpec(256, 1, 512)]
)
def test_boundary_group_forbids_smaller_fa_chunk(native):
    with pytest.raises(ValueError, match="Group 1 block span 256"):
        parse_spec(
            config(
                group(["model.layers.0.attn"], FullAttentionSpec(128)),
                group(["model.layers.1.cache"], native),
            ),
            scheduler_block_size=256,
            chunk_size=128,
        )


def test_slice_aligns_all_compressed_members_in_the_same_group():
    native = config(
        group(
            ["model.layers.0.attn", "model.layers.1.attn"],
            {
                "model.layers.0.attn": MLAAttentionSpec(256, tokens_per_state=64),
                "model.layers.1.attn": MLAAttentionSpec(256, tokens_per_state=128),
            },
        )
    )
    with pytest.raises(ValueError, match="tokens_per_state=128"):
        parse_spec(native, scheduler_block_size=256, chunk_size=64)
    assert (
        parse_spec(native, scheduler_block_size=256, chunk_size=128).chunk_size == 128
    )


def test_logical_slice_is_checked_against_actual_hbm_capability_at_registration():
    native = attention_config(128)
    spec = parse_spec(
        native, scheduler_block_size=128, chunk_size=64, device_type="cpu"
    )
    # Dense HNC supports a whole page; its first half is not one token range.
    with pytest.raises(ValueError, match="not token-contiguous"):
        make_cache_layout(
            spec,
            {"model.layers.0.attn": FakeTensor(1000, (8, 2, 128, 3), (768, 384, 3, 1))},
            kv_cache_config=native,
            num_hidden_layers=4,
        )


def test_valid_slice_templates_preserve_offsets_for_two_group_block_sizes():
    native = attention_config(128, 256)
    spec = parse_spec(
        native, scheduler_block_size=256, chunk_size=64, device_type="cpu"
    )
    layout = make_cache_layout(
        spec,
        {
            "model.layers.0.attn": FakeTensor(1000, (8, 128, 2), (256, 2, 1)),
            "model.layers.1.attn": FakeTensor(9000, (8, 256, 3), (768, 3, 1)),
        },
        kv_cache_config=native,
        num_hidden_layers=4,
    )
    for gid, expected_ptrs, sizes in [
        (0, [[1512], [1640]], [[128], [128]]),
        (1, [[10536], [10728]], [[192], [192]]),
    ]:
        store = layout.store_layouts[gid]
        _, ptrs, actual_sizes = store.resolve_matrices(
            np.array([2, 2], dtype=np.uint64), np.array([0, 64], dtype=np.uint64)
        )
        assert ptrs.tolist() == expected_ptrs
        assert actual_sizes.tolist() == sizes
    from ucm.integration.vllm.v2.connector_worker import UCMTransferBuilder
    from ucm.integration.vllm.v2.layout import RestorePolicy

    assert UCMTransferBuilder(layout).record_nbytes[RestorePolicy.FA] == 320


def test_layout_has_one_explicit_initialization_path():
    native = attention_config(128)
    spec = parse_spec(native, scheduler_block_size=128, device_type="cpu")
    layout = UCMKVCacheLayout(
        spec,
        {"model.layers.0.attn": FakeTensor(1000, (8, 128, 2), (256, 2, 1))},
        kv_cache_config=native,
        num_hidden_layers=4,
    )
    assert not hasattr(layout, "store_layouts")
    layout.build_store_layouts(
        ParallelTopology(),
        ParallelRank(),
        ((0, layout.export_layout_metadata(1), None),),
    )
    assert layout.store_layouts[0].chunk_segment_bytes.tolist() == [[256]]


def test_pp1_preserves_original_registration_error():
    worker = UCMConnectorWorker.__new__(UCMConnectorWorker)
    worker.spec, worker._kv_cache_config, worker._num_hidden_layers = None, None, 4
    worker.parallel_topology = ParallelTopology()
    worker.use_layerwise = False
    original = ValueError("invalid native tensor view")
    with patch(
        "ucm.integration.vllm.v2.connector_worker.UCMKVCacheLayout",
        side_effect=original,
    ):
        with pytest.raises(ValueError) as caught:
            worker.register_kv_caches({})
    assert caught.value is original


@pytest.mark.parametrize("dcp", [1, 8])
def test_parser_passes_actual_dcp_to_the_scheduler_group_rule(dcp):
    from test_ucm_connector_v2 import (
        KpoolTailSpec,
        dcp_world_size_for_kv_cache_spec,
    )

    native = config(
        group(
            ["model.layers.0.attn", "model.layers.1.attn"],
            {
                "model.layers.0.attn": MLAAttentionSpec(128),
                "model.layers.1.attn": FullAttentionSpec(128),
            },
        ),
        group(["model.layers.2.mixer"], MambaSpec(128)),
        group(["model.layers.3.pool"], KpoolTailSpec(16, 32)),
    )
    with patch(
        "vllm.v1.core.kv_cache_utils.dcp_world_size_for_kv_cache_spec",
        wraps=dcp_world_size_for_kv_cache_spec,
    ) as native_rule:
        spec = parse_spec(native, scheduler_block_size=128 * dcp, dcp_size=dcp)
    assert native_rule.call_count == 3
    assert [call.args for call in native_rule.call_args_list] == [
        (g.kv_cache_spec, dcp) for g in native.kv_cache_groups
    ]
    assert [g.block_token_span for g in spec.groups] == [128 * dcp, 128, 16]
    assert [g.dcp_sharded for g in spec.groups] == [dcp > 1, False, False]
    assert spec.chunk_size == 128 * dcp


@pytest.mark.parametrize(
    "partial,states,chunk,error",
    [(False, 8, 4, "not token-contiguous"), (True, 2, 2, "align with stored states")],
)
def test_pp_slice_respects_other_stages_physical_block_constraints(
    partial, states, chunk, error
):
    from ucm.integration.vllm.v2.layout.group import GroupLayoutMetadata

    native = attention_config(8)
    spec = parse_spec(
        native, scheduler_block_size=8, chunk_size=chunk, device_type="cpu"
    )
    layout = UCMKVCacheLayout(
        spec,
        {"model.layers.0.attn": FakeTensor(1000, (8, 8, 2), (16, 2, 1))},
        kv_cache_config=native,
        num_hidden_layers=4,
    )
    metadata = (
        (0, layout.export_layout_metadata(1), None),
        (1, {0: GroupLayoutMetadata((16,), (states,), partial, 1)}, None),
    )
    with pytest.raises(ValueError, match=error):
        layout.build_store_layouts(
            ParallelTopology(pp_size=2), ParallelRank(), metadata
        )
