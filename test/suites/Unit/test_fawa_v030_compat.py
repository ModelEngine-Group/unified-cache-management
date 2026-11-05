"""CPU regression coverage for Ascend V4/V4.1 release layout transitions."""

import ast
import copy
import ctypes
from types import SimpleNamespace as NS

import numpy as np
import pytest
import test_ucm_v41_compat as compat
from test_ucm_v41_compat import (
    HMA_PATH,
    _hash_config,
    _load_hash_helpers,
    _load_hma_methods,
)


def load_function(path, name, namespace):
    node = next(
        n for n in ast.parse(path.read_text()).body if getattr(n, "name", None) == name
    )
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace[name]


@pytest.mark.parametrize("is_v41", [False, True])
@pytest.mark.parametrize("tp_rank", [0, 5, 7])
def test_scheduler_worker_keys_follow_layout_not_package_version(
    is_v41, tp_rank, monkeypatch
):
    def reject_package_version(_):
        pytest.fail("FAWA layout identity must not depend on package versions")

    monkeypatch.setattr("importlib.metadata.version", reject_package_version)
    ns = _load_hash_helpers()
    ns.update(copy=copy, KVConnectorRole=NS(SCHEDULER="scheduler"))
    load_function(HMA_PATH, "_ascend_fawa_hash_suffix", ns)
    method = _load_hma_methods({"_init_fawa_request_hasher"})[
        "_init_fawa_request_hasher"
    ]
    method.__globals__.update(ns)
    if is_v41:
        methods, worker = compat.UCMV41CompatibilityTest._ascend_v41_connector()
        for group in worker._kv_cache_config.kv_cache_groups:
            nested = getattr(group.kv_cache_spec, "kv_cache_specs", None)
            for spec in nested.values() if nested else [group.kv_cache_spec]:
                spec.head_size, spec.dtype = 512, "bfloat16"
        methods["_init_group_metas"](worker)
        config = _hash_config(ratios=(1, 2), block_size=128)
    else:
        main = NS(head_size=640, dtype="uint8", block_size=512)
        index = NS(
            head_size=128,
            dtype="int8",
            block_size=512,
            scale_dim=1,
            scale_dtype="float16",
        )
        worker = NS(
            is_v41_layout=False,
            is_ascend_layout=True,
            ascend_base_block_size=128,
            whole_page_sizes={},
            group_metas={0: NS(token_block_size=512, tail_blocks=1, tail_tokens=512)},
            _kv_cache_config=NS(
                kv_cache_groups=[
                    NS(
                        layer_names=["layers.0.main", "layers.0.index"],
                        kv_cache_spec=NS(
                            block_size=512,
                            kv_cache_specs={
                                "layers.0.main": main,
                                "layers.0.index": index,
                            },
                        ),
                    )
                ]
            ),
        )
        config = _hash_config(ratios=(4, 128), block_size=128)
    worker.tp_rank, worker.tp_size, worker.is_mla = tp_rank, 8, True

    def seeds():
        # Match the real scheduler's deepcopy and representative-spec selection.
        scheduler = copy.deepcopy(worker)
        for group in scheduler._kv_cache_config.kv_cache_groups:
            nested = getattr(group.kv_cache_spec, "kv_cache_specs", None)
            if nested:
                group.kv_cache_spec = next(iter(nested.values()))
        scheduler_config = copy.deepcopy(config)
        scheduler_config.cache_config.block_size = 32
        worker_block_size = config.cache_config.block_size
        method(worker, config, "worker")
        method(scheduler, scheduler_config, "scheduler")
        assert config.cache_config.block_size == worker_block_size
        assert scheduler_config.cache_config.block_size == 32
        assert worker.request_hasher.seed == scheduler._seed
        return scheduler._seed

    before = seeds()
    # The observed rc2 -> final V4 transition must re-key even when the package
    # metadata is unavailable. Equal geometry keeps its identity across releases.
    representative = next(
        iter(
            worker._kv_cache_config.kv_cache_groups[
                0
            ].kv_cache_spec.kv_cache_specs.values()
        )
    )
    representative.head_size = 608 if not is_v41 else 544
    after = seeds()
    assert before != after
    representative.unrelated_runner_setting = "changed"
    assert seeds() == after

    config.cache_config.block_size = 64
    assert seeds() == after  # Mutable engine config is not physical geometry.
    worker.ascend_base_block_size = 64
    assert seeds() != after
    config.cache_config.block_size = worker.ascend_base_block_size = 128
    config.cache_config.cache_dtype = "bfloat16"
    assert seeds() != after
    config.cache_config.cache_dtype = "fp8_ds_mla"
    assert seeds() == after

    for field, value in (
        ("dtype", "float16"),
        ("num_kv_heads", 2),
        ("cache_sparse_sfa_turboquant", True),
    ):
        existed = hasattr(representative, field)
        original = getattr(representative, field, None)
        setattr(representative, field, value)
        assert seeds() != after
        if existed:
            setattr(representative, field, original)
        else:
            delattr(representative, field)
        assert seeds() == after
    representative.scale_dim, representative.scale_dtype = 1, "float16"
    with_scales = seeds()
    assert with_scales != after
    representative.scale_dtype = "float32"
    assert seeds() != with_scales
    del representative.scale_dim, representative.scale_dtype
    assert seeds() == after
    worker.whole_page_sizes = {"page": 77824}
    assert seeds() != after


@pytest.mark.parametrize("width", [640, 608])
@pytest.mark.parametrize("logical_spec", [False, True])
def test_ascend_size_estimate_uses_payload_and_skips_zero_tail(width, logical_spec):
    methods = _load_hma_methods({"_derive_v41_file_sizes", "_specs_for_group"})
    # Mixed C4 main/indexer group; C128 stores 4 states per 512-token hash.
    main = NS(
        block_size=512 if logical_spec else 128,
        ratio=4,
        real_page_size_bytes=128 * width,
    )
    index = NS(
        block_size=main.block_size,
        ratio=4,
        real_page_size_bytes=128 * 130,
        unpadded_page_size_bytes=128 * 128,
    )
    c128 = NS(
        block_size=16384 if logical_spec else 128,
        ratio=128,
        real_page_size_bytes=128 * width,
    )
    swa = NS(block_size=128, ratio=1, real_page_size_bytes=128 * width)
    empty = NS(block_size=32)  # No payload attributes should be inspected.
    # Missing scale planes must change the 4KB-aligned result, not disappear
    # into rounding slack (17 * 256 scale bytes is greater than one 4KB page).
    index_layers = 17
    c4_specs = {"main": main, **{f"index.{i}": index for i in range(index_layers)}}
    groups = [
        NS(
            layer_names=list(c4_specs),
            kv_cache_spec=NS(kv_cache_specs=c4_specs),
        ),
        NS(layer_names=["c128"], kv_cache_spec=c128),
        NS(layer_names=["swa"], kv_cache_spec=swa),
        NS(layer_names=["state"], kv_cache_spec=empty),
    ]
    connector = NS(
        is_ascend_layout=True,
        is_ascend_v41_layout=False,
        whole_page_sizes={},
        transient_group_ids=set(),
        fa_group_ids=[0, 1],
        window_group_ids=[2, 3],
        group_metas={
            0: NS(tail_tokens=512, tail_blocks=1),
            1: NS(tail_tokens=512, tail_blocks=1),
            2: NS(tail_tokens=256, tail_blocks=2),
            3: NS(tail_tokens=0, tail_blocks=1),
        },
        _specs_for_group=methods["_specs_for_group"],
        ascend_block_geometry=lambda spec: NS(
            logical=spec.block_size if logical_spec else spec.block_size * spec.ratio
        ),
    )
    result = methods["_derive_v41_file_sizes"](connector, groups)
    rounded = lambda size: (size + 4095) // 4096 * 4096
    assert result == {
        "FA": rounded(128 * width + index_layers * 128 * 130 + 4 * width),
        "WA": rounded(256 * width),
    }


@pytest.mark.parametrize("width", [640, 608])
@pytest.mark.parametrize(
    "logical,tail,offset", [(512, 512, 0), (16384, 512, 512), (128, 128, 0)]
)
def test_fp8_segment_roundtrip_preserves_padding(width, logical, tail, offset):
    class Tensor:
        dtype = "uint8"

        def __init__(self):
            self.shape = (2, 128, 1, width)
            self.strides = (128 * width + 4096, width, width, 1)
            self.raw = np.arange(2 * self.strides[0], dtype=np.uint8)

        def __getitem__(self, index):
            return self

        def data_ptr(self):
            return self.raw.ctypes.data

        def dim(self):
            return 4

        def element_size(self):
            return 1

        def stride(self, index=None):
            return self.strides if index is None else self.strides[index]

    from typing import Sequence

    ns = _load_hma_methods(set())
    ns.update(torch=NS(Tensor=Tensor), Tuple=tuple, Sequence=Sequence)
    load_function(HMA_PATH, "KVCacheGroupLayout", ns)
    tensor = Tensor()
    layout = ns["KVCacheGroupLayout"](
        {"layers.0.attn": tensor}, is_ascend_layout=True, expected_block_size=128
    )
    size = layout.segment_tensor_size_list(tail, logical)[0]
    assert size == tail * 128 // logical * width
    address = int(
        layout.extract_addrs_with_offsets(
            np.array([1], dtype=np.uint64), logical, np.array([offset], dtype=np.uint64)
        )[0, 0]
    )
    start = tensor.strides[0] + offset * 128 // logical * width
    assert address == tensor.data_ptr() + start
    before = tensor.raw.copy()
    saved = ctypes.string_at(address, size)
    tensor.raw[start : start + size] = 0
    ctypes.memmove(address, saved, size)
    np.testing.assert_array_equal(tensor.raw, before)
