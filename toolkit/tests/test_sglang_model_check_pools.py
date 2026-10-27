"""Offline hybrid planning and transfer regression tests (no SGLang/Torch required)."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ucm_toolkit.tools.sglang_model_check.config import CheckConfig
from ucm_toolkit.tools.sglang_model_check.pool_probe import (
    PoolProbeUnavailable,
    PoolSpec,
    build_pool_specs,
    expected_prefixes,
)
from ucm_toolkit.tools.sglang_model_check.runner import (
    CheckFailure,
    _roundtrip,
    _roundtrip_v1,
    _roundtrip_v2,
    run,
)


DTYPE = SimpleNamespace(itemsize=2)


def model_config():
    return SimpleNamespace(
        attention_arch=SimpleNamespace(name="MHA"),
        num_hidden_layers=4,
        full_attention_layer_ids=[0, 2], swa_attention_layer_ids=[1, 3],
        head_dim=8, v_head_dim=8, swa_head_dim=4, swa_v_head_dim=4,
        get_num_kv_heads=lambda tp: 2, get_swa_num_kv_heads=lambda tp: 1,
        sliding_window_size=3, dtype=DTYPE,
    )


def linear_config():
    return SimpleNamespace(
        full_attention_layer_ids=[0, 2],
        mamba2_cache_params=SimpleNamespace(
            layers=[1, 3],
            shape=SimpleNamespace(temporal=(2, 3, 4), conv=[(8, 2), (4, 2)]),
            dtype=SimpleNamespace(temporal=SimpleNamespace(itemsize=4), conv=DTYPE),
            is_kda=False,
        ),
    )


class PoolPlanTest(unittest.TestCase):
    def test_mla_state_plan_uses_latent_width_not_mha_head_geometry(self):
        model = model_config()
        model.attention_arch = SimpleNamespace(name="MLA")
        model.kv_lora_rank, model.qk_rope_head_dim = 192, 64
        model.head_dim, model.v_head_dim = 72, 128
        del model.get_num_kv_heads
        kv, state = build_pool_specs(model, ("kv", "mamba"), 64, "cuda",
                                     linear_config=linear_config())
        self.assertEqual((kv.kind, kv.layers, kv.kv_cache_dim), ("mla", (0, 2), 256))
        self.assertEqual(kv.bytes_per_slot(DTYPE, 64), 2 * 64 * 256 * 2)
        self.assertEqual(kv.report()["kv_cache_dim"], 256)
        self.assertEqual((state.kind, state.layers), ("mamba", (1, 3)))

    def test_mla_external_attention_plan_keeps_disjoint_state_layers(self):
        model = model_config()
        model.attention_arch = SimpleNamespace(name="MLA")
        # No MHA geometry should be read for a native MLA child.
        del model.get_num_kv_heads
        kv, state = build_pool_specs(model, ("kv", "mamba"), 256, "cuda",
                                     linear_config=linear_config(), external_attention=True)
        self.assertEqual((kv.kind, kv.layers), ("native", (0, 2)))
        self.assertEqual((state.kind, state.layers, state.window_pages), ("mamba", (1, 3), 1))
        invalid = linear_config()
        invalid.full_attention_layer_ids = [0, 1]
        with self.assertRaisesRegex(PoolProbeUnavailable, "complete disjoint"):
            build_pool_specs(model, ("kv", "mamba"), 256, "cuda",
                             linear_config=invalid, external_attention=True)

    def test_swa_reuses_separate_layer_and_head_geometry(self):
        kv, swa = build_pool_specs(model_config(), ("kv", "swa"), 2, "cuda")
        self.assertEqual(kv.layers, (0, 2))
        self.assertEqual(swa.layers, (1, 3))
        self.assertEqual((kv.head_num, kv.head_dim), (2, 8))
        self.assertEqual((swa.head_num, swa.head_dim, swa.window_pages), (1, 4, 2))

    def test_mamba_reuses_config_shapes_and_both_dtypes(self):
        state_config = linear_config()
        kv, state = build_pool_specs(model_config(), ("kv", "mamba"), 64, "cuda",
                                     linear_config=state_config)
        self.assertEqual(kv.layers, (0, 2))
        self.assertEqual(state.temporal_shape, (2, 3, 4))
        self.assertEqual(state.conv_shapes, ((8, 2), (4, 2)))
        self.assertIs(state.temporal_dtype, state_config.mamba2_cache_params.dtype.temporal)
        self.assertEqual(state.bytes_per_slot(DTYPE, 64), 2 * (24 * 4 + 24 * 2))
        self.assertEqual(state.report()["page_size"], 1)

    def test_npu_state_layout_transposes_standard_conv_only(self):
        state_config = linear_config()
        state = build_pool_specs(model_config(), ("kv", "mamba"), 64, "ascend",
                                 linear_config=state_config)[-1]
        self.assertEqual(state.conv_shapes, ((2, 8), (2, 4)))
        state_config.mamba2_cache_params.is_kda = True
        state = build_pool_specs(model_config(), ("kv", "mamba"), 64, "ascend",
                                 linear_config=state_config)[-1]
        self.assertEqual(state.conv_shapes, ((8, 2), (4, 2)))

    def test_combined_three_way_partition(self):
        config = model_config()
        config.swa_attention_layer_ids = [1]
        state = linear_config()
        state.full_attention_layer_ids = [0]
        state.mamba2_cache_params.layers = [2, 3]
        specs = build_pool_specs(config, ("kv", "mamba", "swa"), 2, "cuda",
                                 linear_config=state)
        self.assertEqual([spec.name for spec in specs], ["kv", "swa", "mamba"])

    def test_incomplete_and_overlapping_partitions_are_inconclusive(self):
        for full in ([0], [0, 1, 2]):
            config = model_config()
            config.full_attention_layer_ids = full
            with self.assertRaisesRegex(PoolProbeUnavailable, "partition"):
                build_pool_specs(config, ("kv", "swa"), 2, "cuda")

    def test_runtime_pools_report_specific_missing_descriptor(self):
        with self.assertRaisesRegex(PoolProbeUnavailable, "indexer"):
            build_pool_specs(object(), ("kv", "indexer"), 64, "cuda")

    def test_compressed_swa_is_not_misrepresented_as_plain_swa(self):
        config = model_config()
        config.is_hybrid_swa_compress = True
        with self.assertRaisesRegex(PoolProbeUnavailable, "compressed"):
            build_pool_specs(config, ("kv", "swa"), 64, "cuda")

    def test_cpu_state_is_not_assumed_to_use_cuda_layout(self):
        with self.assertRaisesRegex(PoolProbeUnavailable, "AMX"):
            build_pool_specs(model_config(), ("kv", "mamba"), 64, "cpu",
                             linear_config=linear_config())

    def test_window_must_be_known_and_positive(self):
        for window in (None, 0, -1, [16, 32]):
            config = model_config()
            config.sliding_window_size = window
            with self.assertRaisesRegex(PoolProbeUnavailable, "window"):
                build_pool_specs(config, ("kv", "swa"), 64, "cuda")

    def test_pure_swa_has_logical_anchor_and_real_side_pool(self):
        config = model_config()
        config.full_attention_layer_ids = []
        config.swa_attention_layer_ids = [0, 1, 2, 3]
        anchor, swa = build_pool_specs(config, ("kv", "swa"), 64, "cuda")
        self.assertEqual(anchor.kind, "logical")
        self.assertEqual(anchor.bytes_per_slot(DTYPE, 64), 0)
        self.assertEqual(swa.layers, (0, 1, 2, 3))

    def test_pure_linear_state_has_logical_anchor(self):
        state = linear_config()
        state.full_attention_layer_ids = []
        state.mamba2_cache_params.layers = [0, 1, 2, 3]
        anchor, mamba = build_pool_specs(model_config(), ("kv", "mamba"), 64, "cuda",
                                        linear_config=state)
        self.assertEqual(anchor.kind, "logical")
        self.assertEqual(mamba.kind, "mamba")

    def test_short_prefix_uses_clipped_window(self):
        self.assertEqual(expected_prefixes(5, {"swa": {0, 1, 3, 4}}, {"swa": 2}),
                         [1, 2, 5])

    def test_sparse_mult_pool_intersection_is_not_min_of_maxima(self):
        self.assertEqual(expected_prefixes(
            8, {"a": {3, 7}, "b": {5}}, {"a": 1, "b": 1}), [])

    def test_aggregate_memory_limit_precedes_allocation(self):
        spec = PoolSpec("mamba", "mamba", (0,), temporal_shape=(1000000,),
                        conv_shapes=((1, 1),), temporal_dtype=DTYPE, conv_dtype=DTYPE)
        with patch("ucm_toolkit.tools.sglang_model_check.runner._hybrid_specs",
                   return_value=[spec]):
            with self.assertRaises(CheckFailure) as raised:
                _roundtrip(CheckConfig(model="model", max_host_mib=1),
                           model_config(), SimpleNamespace(is_hybrid=True,
                               pool_names=("kv", "mamba"), is_mla=False))
        self.assertEqual(raised.exception.code, "CHECKER_MEMORY_LIMIT")


class _Indices(list):
    def __add__(self, offset):
        return _Indices(value + offset for value in self)


class _Tensor:
    def __init__(self, values=None):
        self.values = list(values or [0, 0])

    def clone(self):
        return _Tensor(self.values)

    def zero_(self):
        self.values[:] = [0] * len(self.values)

    def numel(self):
        return len(self.values)

    def element_size(self):
        return 4


class _Transfer(SimpleNamespace):
    pass


class PrimaryTransferTest(unittest.TestCase):
    def exercise(self, corrupt=False):
        pages, size = 3, 4
        memory = {page: [_Tensor(), _Tensor()] for page in range(pages * 2)}
        objects = {}

        class Storage:
            def batch_set_v1(self, keys, indices):
                for key, offset in zip(keys, indices[::size]):
                    objects[key] = [component.clone() for component in memory[offset // size]]
                return [True] * len(keys)

            def batch_exists(self, keys):
                for index, key in enumerate(keys):
                    if key not in objects:
                        return index
                return len(keys)

            def batch_get_v1(self, keys, indices):
                for key, offset in zip(keys, indices[::size]):
                    for src, dst in zip(objects[key], memory[offset // size]):
                        dst.values[:] = src.values
                        if corrupt:
                            dst.values[-1] = -1
                return [True] * len(keys)

        torch = SimpleNamespace(
            int64="int64", arange=lambda start, stop, **kwargs: _Indices(range(start, stop)),
            equal=lambda left, right: left.values == right.values,
        )
        with patch.dict(sys.modules, {"torch": torch}), patch(
            "ucm_toolkit.tools.sglang_model_check.runner._pool_page_views",
            side_effect=lambda pool, name, page, mla: memory[page],
        ), patch("ucm_toolkit.tools.sglang_model_check.runner._fill_pattern",
                 side_effect=lambda tensor, seed: tensor.values.__setitem__(slice(None), [seed, seed + 1])):
            return _roundtrip_v1(Storage(), SimpleNamespace(page_size=size),
                                 [f"key-{i}" for i in range(pages)],
                                 SimpleNamespace(pages=pages), True)

    def test_primary_component_roundtrip(self):
        self.assertEqual(self.exercise()["bytes_compared"], 3 * 2 * 2 * 4)

    def test_primary_corruption_cannot_pass(self):
        with self.assertRaises(CheckFailure) as raised:
            self.exercise(corrupt=True)
        self.assertEqual(raised.exception.code, "ROUNDTRIP_MISMATCH")


class HybridTransferTest(unittest.TestCase):
    """Run the checker orchestration with independent memory-backed page IO.

    This tests index units, sparse descriptors and corruption detection. It is
    not a replacement for running native SGLang pools and PosixStore on target.
    """

    def exercise(self, corrupt=False, bad_lookup=False, logical=False, indexer=False, dsa_state=False):
        pages = 5
        specs = [PoolSpec("kv", "logical" if logical else "mha", (0,)),
                 PoolSpec("swa", "mha", (1,), window_pages=2),
                 PoolSpec("mamba", "mamba", (2,), window_pages=1)]
        pools = {name: SimpleNamespace(page_size=size) for name, size in
                 (("kv", 4), ("swa", 4), ("mamba", 1))}
        if indexer:
            specs.append(PoolSpec("indexer", "native", (0,), storage_page_size=3))
            pools["indexer"] = SimpleNamespace(page_size=3)
        if dsa_state:
            specs = [spec for spec in specs if spec.name != "swa"]
            del pools["swa"]
        memory = {(name, page): [_Tensor(), _Tensor()] for name in pools
                  for page in range(pages * 2)}
        objects = {}

        class Storage:
            def batch_set_v2(self, transfers):
                result = {}
                for transfer in transfers:
                    size = pools[transfer.name].page_size
                    assert len(transfer.host_indices) == len(transfer.keys) * size
                    for key, offset in zip(transfer.keys, transfer.host_indices[::size]):
                        objects[transfer.name, key] = [t.clone() for t in memory[transfer.name, offset // size]]
                    result[transfer.name] = [True] * len(transfer.keys)
                return result

            def batch_exists_v2(self, keys, transfers):
                valid = []
                for end in range(1, len(keys) + 1):
                    if any(key.startswith("missing-") for key in keys[:end]):
                        continue
                    if all(all((t.name, key) in objects
                               for key in keys[(max(0, end - len(t.keys)) if t.hit_policy == "trailing" else 0):end])
                           for t in transfers):
                        valid.append(end)
                return SimpleNamespace(kv_hit_pages=max(valid) if valid else 0,
                                       restorable_prefix_pages=[] if bad_lookup else valid)

            def batch_get_v2(self, transfers):
                result = {}
                for transfer in transfers:
                    size = pools[transfer.name].page_size
                    for key, offset in zip(transfer.keys, transfer.host_indices[::size]):
                        for src, dst in zip(objects[transfer.name, key], memory[transfer.name, offset // size]):
                            dst.values[:] = src.values
                            if corrupt:
                                dst.values[0] = -123
                    result[transfer.name] = [True] * len(transfer.keys)
                return result

        fake_torch = SimpleNamespace(
            int64="int64", tensor=lambda values, **kwargs: _Indices(values),
            equal=lambda left, right: left.values == right.values,
        )
        with patch.dict(sys.modules, {
            "torch": fake_torch,
            "sglang.srt.mem_cache.hicache_storage": SimpleNamespace(
                PoolTransfer=_Transfer,
                PoolHitPolicy=SimpleNamespace(TRAILING_PAGES="trailing", ALL_PAGES="all")),
        }), patch("ucm_toolkit.tools.sglang_model_check.runner._roundtrip_v1",
                  return_value={"bytes_compared": 0}), patch(
            "ucm_toolkit.tools.sglang_model_check.runner._pool_name", side_effect=lambda name: name
        ), patch("ucm_toolkit.tools.sglang_model_check.runner._pool_page_views",
                 side_effect=lambda pool, name, page, mla: memory[name, page]), patch(
            "ucm_toolkit.tools.sglang_model_check.runner._fill_pattern",
            side_effect=lambda tensor, seed: tensor.values.__setitem__(slice(None), [seed, seed + 1]),
        ):
            return _roundtrip_v2(Storage(), pools, [f"key-{i}" for i in range(pages)],
                                 SimpleNamespace(pages=pages, page_size=4), dsa_state, specs)

    def test_sparse_windows_and_checkpoint_index_units(self):
        result = self.exercise()
        self.assertEqual(result["restorable_prefix_pages"], [2, 5])
        self.assertEqual(result["persisted_pages"]["mamba"], [1, 4])
        self.assertEqual(result["persisted_pages"]["swa"], [0, 1, 3, 4])

    def test_logical_anchor_requires_both_full_and_trailing_payloads(self):
        result = self.exercise(logical=True, indexer=True)
        self.assertTrue(result["logical_anchor"])
        self.assertEqual(result["restorable_prefix_pages"], [2, 5])
        self.assertEqual(result["persisted_pages"]["indexer"], [0, 1, 2, 3, 4])
        self.assertEqual(result["pages_written"], 0)

    def test_mla_state_joint_lookup_with_and_without_indexer(self):
        for indexer in (False, True):
            with self.subTest(indexer=indexer):
                result = self.exercise(indexer=indexer, dsa_state=True)
                names = {"kv", "mamba"} | ({"indexer"} if indexer else set())
                self.assertEqual(set(result["pool_names"]), names)
                self.assertEqual(result["restorable_prefix_pages"], [2, 5])
                self.assertEqual(result["persisted_pages"]["mamba"], [1, 4])
                if indexer:
                    self.assertEqual(result["persisted_pages"]["indexer"], list(range(5)))
                for options, code in (({"corrupt": True}, "ROUNDTRIP_MISMATCH"),
                                      ({"bad_lookup": True}, "LOOKUP_CONTRACT_MISMATCH")):
                    with self.assertRaises(CheckFailure) as raised:
                        self.exercise(indexer=indexer, dsa_state=True, **options)
                    self.assertEqual(raised.exception.code, code)


@unittest.skipUnless(
    os.environ.get("UCM_SGLANG_MODEL_CHECK_TEST_MODEL")
    and os.environ.get("UCM_SGLANG_MODEL_CHECK_TEST_BACKENDS"),
    "set a local hybrid model and a dedicated Posix backend for native integration tests",
)
class NativeHybridRoundtripTest(unittest.TestCase):
    """Opt-in real SGLang pools + production UCM; never a mock success claim."""

    def test_real_hybrid_pools_both_layouts(self):
        for layout in ("page_first", "page_first_direct"):
            with self.subTest(layout=layout):
                result = run(CheckConfig(
                    model=os.environ["UCM_SGLANG_MODEL_CHECK_TEST_MODEL"],
                    mode="roundtrip", skip_meta_model=True,
                    pages=5, page_size=int(os.environ.get("UCM_SGLANG_MODEL_CHECK_TEST_PAGE_SIZE", "0")), layout=layout,
                    kv_cache_dtype=os.environ.get("UCM_SGLANG_MODEL_CHECK_TEST_KV_DTYPE", "auto"),
                    max_host_mib=int(os.environ.get("UCM_SGLANG_MODEL_CHECK_TEST_MAX_HOST_MIB", "2048")),
                    storage_backends=os.environ["UCM_SGLANG_MODEL_CHECK_TEST_BACKENDS"],
                ))
                self.assertTrue(result.cache_requirements.get("is_hybrid"), result.to_dict(True))
                self.assertTrue(result.supported, result.to_dict(True))
                self.assertEqual(result.validation["storage_io"], "passed")
                self.assertIn(result.roundtrip["storage_api"], ("v2", "v1_integrated_payload"))
                self.assertTrue(result.roundtrip["bytes_compared"] > 0)


if __name__ == "__main__":
    unittest.main()
