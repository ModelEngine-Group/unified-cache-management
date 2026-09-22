"""CPU-only protocol tests; runnable without importing the UCM/vLLM runtime.

    python -m unittest discover -s test/suites/Unit/connector -p test_ucm_key_protocol.py -v
"""

import ast
import hashlib
import importlib.util
import pickle
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[4]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "ucm/integration/vllm" / filename
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


hashes = load_module("_ucm_protocol_hasher", "request_hasher.py")
layout = load_module("_ucm_protocol_layout", "fawa_layout.py")


# Execute the actual pure fingerprint builders without importing the GPU connector.
_connector_tree = ast.parse(
    (ROOT / "ucm/integration/vllm/ucm_connector.py").read_text(encoding="utf-8-sig")
)
exec(
    compile(
        ast.Module(
            body=[
                n
                for n in _connector_tree.body
                if isinstance(n, ast.FunctionDef)
                and n.name in {"build_request_fingerprint", "kv_layout_namespace"}
            ],
            type_ignores=[],
        ),
        "connector_fingerprint",
        "exec",
    )
)


def config(tp=4):
    return NS(
        model_config=NS(model="org/model", dtype="bfloat16"),
        parallel_config=NS(tensor_parallel_size=tp),
        cache_config=NS(cache_dtype="auto", block_size=4),
        speculative_config=None,
        additional_config={},
    )


def request(tokens=range(12), **kwargs):
    return NS(all_token_ids=list(tokens), **kwargs)


def make_hasher(kv_config=None):
    fingerprint = build_request_fingerprint(config()) + pickle.dumps(
        kv_layout_namespace(config(), kv_config), protocol=4
    )
    return hashes.RequestHasher(fingerprint)


class KeyProtocolTests(unittest.TestCase):
    def setUp(self):
        self.helper = patch.object(hashes, "generate_block_hash_extra_keys", None)
        self.helper.start()
        self.addCleanup(self.helper.stop)

    def test_key_from_digest_wire_format(self):
        self.assertEqual(
            hashes.key_from_digest(bytes(range(16))),
            bytes.fromhex("000102030405060708090a0b0c0d0000"),
        )

    def test_all_scopes_unique_and_rank_round_trip(self):
        digest = hashlib.md5(b"payload").digest()
        keys = set()
        for group in range(16):
            canonical = hashes.set_group_id(hashes.key_from_digest(digest), group)
            for rank in range(16):
                key = hashes.set_rank_id(canonical, rank)
                keys.add(key)
                self.assertEqual(key[:14], digest[:14])
                self.assertEqual(key[14:], bytes((0, group * 16 + rank)))
                self.assertEqual(hashes.set_rank_id(key, 0), canonical)
                self.assertEqual(hashes.set_rank_id(key, rank), key)
        self.assertEqual(len(keys), 256)

    def test_rank_update_preserves_every_other_bit(self):
        key = bytes(range(14)) + bytes((0xAB, 0xCD))
        updated = hashes.set_rank_id(key, 2)
        self.assertEqual(updated[:15], key[:15])
        self.assertEqual(updated[15], 0xC2)
        self.assertEqual(hashes.set_rank_id(updated, 13), key)
        with self.assertRaises(ValueError):
            hashes.set_rank_id(b"short", 0)

    def test_group_update_preserves_rank_and_reserved_bits(self):
        key = bytes(range(14)) + bytes((0xAB, 0xCD))
        updated = hashes.set_group_id(key, 2)
        self.assertEqual(updated[:15], key[:15])
        self.assertEqual(updated[15], 0x2D)
        self.assertEqual(hashes.set_group_id(updated, 12), key)

    def test_invalid_scope_rejected(self):
        for value in (-1, 16, 32, 1.5):
            with self.assertRaises(ValueError):
                hashes.set_group_id(bytes(16), value)
            with self.assertRaises(ValueError):
                hashes.set_rank_id(bytes(16), value)
        with self.assertRaises(ValueError):
            hashes.set_group_id(b"short", 0)

    def test_base_hash_chain_is_independent_of_group_and_rank(self):
        hasher = make_hasher()
        block_hasher = hasher.make_request_block_hasher(4)
        digests = block_hasher(request())
        parent = hasher.seed
        for i, digest in enumerate(digests):
            parent = hasher((parent, tuple(range(i * 4, (i + 1) * 4)), None))
            self.assertEqual(digest, hashes.key_from_digest(parent))
            for group, rank in ((0, 0), (2, 3), (15, 15)):
                key = hashes.set_rank_id(
                    hashes.set_group_id(hashes.key_from_digest(digest), group), rank
                )
                self.assertEqual(key[:14], parent[:14])
        self.assertEqual(digests, block_hasher(request()))
        self.assertEqual(len(block_hasher(request(range(6)))), 1)
        self.assertNotEqual(
            digests[-1], hasher.make_request_block_hasher(6)(request())[-1]
        )

    def test_request_semantics_and_failed_helper(self):
        hasher = make_hasher().make_request_block_hasher(4)
        with self.assertRaises(hashes.RequestHashError):
            hasher(request(cache_salt="tenant"))

        def extras(req, start, end, cursor):
            return ((req.cache_salt,) if start >= 4 else None), cursor

        with patch.object(hashes, "generate_block_hash_extra_keys", extras):
            a = hasher(request(cache_salt="a"))
            b = hasher(request(cache_salt="b"))
            self.assertEqual(a[0], b[0])
            self.assertNotEqual(a[1:], b[1:])
            self.assertEqual(a, hasher(request(cache_salt="a")))
        with patch.object(
            hashes, "generate_block_hash_extra_keys", side_effect=ValueError
        ):
            with self.assertRaises(hashes.RequestHashError):
                hasher(request())

    def test_actual_kv_and_scale_dtype_separate_namespaces(self):
        def make(kv_dtype, scale_dtype):
            spec = NS(block_size=4, dtype=kv_dtype, scale_dtype=scale_dtype)
            cfg = NS(kv_cache_groups=[NS(layer_names=["layer.0"], kv_cache_spec=spec)])
            return make_hasher(cfg).make_request_block_hasher(4)(request())

        self.assertNotEqual(make("bf16", "fp32"), make("fp8", "fp32"))
        self.assertNotEqual(make("fp8", "fp16"), make("fp8", "fp32"))

    def test_legacy_algorithm_unchanged_and_new_namespace_separate(self):
        old = hashes.RequestHasher(build_request_fingerprint(config(), 2))
        data = (b"parent", (1, 2, 3), None)
        expected = hashlib.md5(
            old.meta_bytes + pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        ).digest()
        self.assertEqual(old(data), expected)
        plain = hashes.RequestHasher(
            build_request_fingerprint(config())
        ).make_request_block_hasher(4)(request())
        current = make_hasher().make_request_block_hasher(4)(request())
        self.assertNotEqual(plain, current)


class BlockGeometryTests(unittest.TestCase):
    def test_old_and_new_compressed_specs(self):
        # Both versions inherit the same misleading property; the runner
        # convention, not this value, decides what block_size means.
        class Spec:
            def __init__(self, size, ratio):
                self.block_size, self.compress_ratio = size, ratio

            @property
            def storage_block_size(self):
                return self.block_size // self.compress_ratio

        for physical in (32, 64, 128):
            for ratio in (1, 4, 128):
                for logical_mode in (False, True):
                    spec = Spec(physical * ratio if logical_mode else physical, ratio)
                    self.assertEqual(
                        layout.ascend_block_geometry(
                            spec, block_size_is_logical=logical_mode
                        ),
                        layout.BlockGeometry(physical * ratio, physical),
                    )

    def test_runtime_convention_detection(self):
        for logical_mode in (False, True):
            module = NS()
            if logical_mode:
                module.get_storage_block_size = lambda spec: spec.storage_block_size
            with patch.dict(
                sys.modules, {"vllm_ascend.core.kv_cache_interface": module}
            ):
                spec = NS(
                    block_size=512 if logical_mode else 128,
                    compress_ratio=4,
                    storage_block_size=128 if logical_mode else 32,
                )
                self.assertEqual(
                    layout.ascend_block_geometry(spec), layout.BlockGeometry(512, 128)
                )

    def test_wrapper_members_and_invalid_geometry(self):
        old = NS(block_size=128, compress_ratio=4)
        wrapped = NS(kv_cache_specs={"a": old, "b": old})
        self.assertEqual(
            layout.ascend_block_geometry(wrapped, block_size_is_logical=False),
            layout.BlockGeometry(512, 128),
        )
        for spec in (
            NS(block_size=0, compress_ratio=4),
            NS(kv_cache_specs={"a": old, "b": NS(block_size=64, compress_ratio=4)}),
        ):
            with self.assertRaises(ValueError):
                layout.ascend_block_geometry(spec, block_size_is_logical=False)
        with self.assertRaises(ValueError):
            layout.ascend_block_geometry(
                NS(block_size=513, compress_ratio=4), block_size_is_logical=True
            )


class FullViewTests(unittest.TestCase):
    def setUp(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy needed for byte-level alias tests")
        self.np = np

    def views(self, block_size, padding=0):
        np = self.np
        stride = block_size * 132 + padding
        root = np.arange(stride * 3 + 16, dtype=np.int64).astype(np.uint8)

        class Tensor:
            device = "cpu"

            def __init__(self, array):
                self.array = array
                self.shape = array.shape

            def dim(self):
                return self.array.ndim

            def stride(self, dim=None):
                strides = tuple(
                    value // self.element_size() for value in self.array.strides
                )
                return strides if dim is None else strides[dim]

            @property
            def dtype(self):
                return self.array.dtype

            def element_size(self):
                return self.array.itemsize

            def data_ptr(self):
                return self.array.ctypes.data

            def untyped_storage(self):
                return NS(data_ptr=lambda: root.ctypes.data)

            def __getitem__(self, item):
                return Tensor(self.array[item])

            def numel(self):
                return self.array.size

            def is_contiguous(self):
                return self.array.flags.c_contiguous

        key = Tensor(
            np.ndarray(
                (3, block_size, 1, 128),
                dtype=np.uint8,
                buffer=root,
                offset=16,
                strides=(stride, 128, 128, 1),
            )
        )
        scale = Tensor(
            np.ndarray(
                (3, block_size, 1, 1),
                dtype=np.float32,
                buffer=root,
                offset=16 + block_size * 128,
                strides=(stride, 4, 4, 4),
            )
        )
        full = Tensor(
            np.ndarray(
                (3, block_size, 1, 132),
                dtype=np.uint8,
                buffer=root,
                offset=16,
                strides=(stride, 132, 132, 1),
            )
        )
        return key, scale, full

    def test_full_page_round_trip_restores_both_aliases(self):
        for size in (32, 64, 128):
            for padding in (0, 256):
                key, scale, full = self.views(size, padding)
                selected, whole_page = layout.select_transfer_views((key, scale, full))
                self.assertTrue(whole_page)
                self.assertEqual(selected, (full,))
                for block in (2, 0):
                    key_bytes = key.array[block].tobytes()
                    scale_bytes = scale.array[block].tobytes()
                    saved = selected[0].array[block].tobytes()
                    self.assertEqual(saved, key_bytes + scale_bytes)
                    selected[0].array[block].fill(0)
                    selected[0].array[block].flat[:] = self.np.frombuffer(
                        saved, dtype=self.np.uint8
                    )
                    self.assertEqual(key.array[block].tobytes(), key_bytes)
                    self.assertEqual(scale.array[block].tobytes(), scale_bytes)

    def test_split_views_preserved_and_bad_overlap_rejected(self):
        key, scale, full = self.views(128)
        self.assertEqual(
            layout.select_transfer_views((key, scale)), ((key, scale), False)
        )
        scale.device = "different-device"
        with self.assertRaises(ValueError):
            layout.select_transfer_views((key, scale, full))


if __name__ == "__main__":
    unittest.main()
