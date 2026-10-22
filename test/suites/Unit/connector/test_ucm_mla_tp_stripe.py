"""MLA dump striping across TP ranks; CPU-only, no UCM/vLLM runtime needed.

    python -m unittest discover -s test/suites/Unit/connector \\
        -p test_ucm_mla_tp_stripe.py -v

The helpers under test live in connector modules that import the GPU/vLLM
runtime at import time, so each one is extracted from its source file with
``ast`` and executed standalone, then driven through a fake connector.
"""

import ast
import dataclasses
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover - environment without numpy
    np = None
    _HAS_NUMPY = False

ROOT = Path(__file__).resolve().parents[4]
UCM = ROOT / "ucm/integration/vllm/ucm_connector.py"
HMA = ROOT / "ucm/integration/vllm/hma_connector.py"
HLA = ROOT / "ucm/integration/vllm/hla_connector.py"


def _parse(path):
    return ast.parse(path.read_text(encoding="utf-8-sig"))


def _globals():
    return {"np": np} if _HAS_NUMPY else {}


def _top_functions(path, names):
    """Pull selected module-level functions out of *path* unchanged."""
    wanted = set(names)
    body = [
        n
        for n in _parse(path).body
        if isinstance(n, ast.FunctionDef) and n.name in wanted
    ]
    missing = wanted - {n.name for n in body}
    if missing:
        raise AssertionError(f"missing functions in {path.name}: {sorted(missing)}")
    namespace = _globals()
    exec(compile(ast.Module(body=body, type_ignores=[]), path.name, "exec"), namespace)
    return namespace


def _method(path, class_name, method_name):
    """Pull a single method out of a class as a plain function."""
    for node in _parse(path).body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    namespace = _globals()
                    exec(
                        compile(
                            ast.Module(body=[item], type_ignores=[]),
                            path.name,
                            "exec",
                        ),
                        namespace,
                    )
                    return namespace[method_name]
    raise AssertionError(f"missing {class_name}.{method_name} in {path.name}")


class _FakeConn:
    """Minimal stand-in for the connector state the helpers read."""

    def __init__(self, rank, size, hashed=None):
        self.tp_rank = rank
        self.tp_size = size
        self.hashed = [] if hashed is None else hashed

    def request_hasher(self, block_id):
        self.hashed.append(block_id)
        return block_id + b"#rank"


stripe_tail_blocks = _top_functions(HMA, ["stripe_tail_blocks"])["stripe_tail_blocks"]
_select_mla_dump_blocks = _method(UCM, "UCMDirectConnector", "_select_mla_dump_blocks")
_extract_fa_ptr = _method(HMA, "UCMFAWAConnector", "_extract_fa_ptr")
_mla_split_scope = _method(HLA, "UCMHybridLinearAttentionConnector", "_mla_split_scope")

needs_numpy = unittest.skipUnless(_HAS_NUMPY, "NumPy needed for striping tests")


@needs_numpy
class SelectMlaDumpBlocksTests(unittest.TestCase):
    """`_select_mla_dump_blocks`: absolute-index striping, keys paired to addresses."""

    def select(self, rank, size, keys, ids, start):
        return _select_mla_dump_blocks(_FakeConn(rank, size), keys, ids, start)

    def owners(self, size, keys, ids, start):
        owners = set()
        for rank in range(size):
            if self.select(rank, size, keys, ids, start)[0]:
                owners.add(rank)
        return owners

    def test_full_batch_is_a_partition(self):
        size = 16
        keys = [f"k{i}".encode() for i in range(size)]
        ids = list(range(1000, 1000 + size))
        seen = {}
        for rank in range(size):
            got_keys, got_ids = self.select(rank, size, keys, ids, 0)
            self.assertEqual(len(got_keys), 1)
            self.assertEqual(len(got_keys), len(got_ids))
            seen[got_keys[0]] = got_ids[0]
        self.assertEqual(set(seen), set(keys))
        for index, key in enumerate(keys):
            self.assertEqual(seen[key], ids[index])

    def test_every_block_has_exactly_one_writer_over_a_request(self):
        # A 100-block request dumped in 4-block batches: ownership must be
        # disjoint and complete across ranks and batches, however the batches
        # split the request.
        size = 16
        keys = [f"k{i}".encode() for i in range(100)]
        ids = list(range(100, 200))
        claimed = {}
        for start in range(0, 100, 4):
            for rank in range(size):
                got_keys, got_ids = self.select(
                    rank, size, keys[start : start + 4], ids[start : start + 4], start
                )
                for key, block_id in zip(got_keys, got_ids):
                    self.assertNotIn(key, claimed)
                    absolute = int(key[1:])
                    self.assertEqual(absolute % size, rank)
                    # The vLLM address travels with its own UCM key.
                    self.assertEqual(block_id, ids[absolute])
                    claimed[key] = rank
        self.assertEqual(len(claimed), 100)

    def test_short_batch_rotates_among_ranks(self):
        # Batches shorter than tp_size must not always land on the low ranks.
        size = 16
        keys = [f"k{i}".encode() for i in range(3)]
        ids = [10, 11, 12]
        self.assertEqual(self.owners(size, keys, ids, start=0), {0, 1, 2})
        self.assertEqual(self.owners(size, keys, ids, start=13), {13, 14, 15})

    def test_no_low_rank_bias_over_a_long_request(self):
        # Position 0 always goes to rank 0, so the count per rank only evens out
        # if ownership is anchored on the absolute index rather than restarting
        # at 0 for every batch.
        size = 16
        keys = [f"k{i}".encode() for i in range(64)]
        ids = list(range(64))
        counts = {rank: 0 for rank in range(size)}
        for start in range(0, 64, 4):
            for rank in range(size):
                got_keys, _ = self.select(
                    rank, size, keys[start : start + 4], ids[start : start + 4], start
                )
                counts[rank] += len(got_keys)
        self.assertEqual(set(counts.values()), {4})

    def test_mla_keys_are_stored_verbatim_without_a_rank_suffix(self):
        # MLA keys are shared across ranks; the dump path must not hash them per
        # rank. A hashing connector that is never called proves the pass-through.
        size = 4
        keys = [f"k{i}".encode() for i in range(4)]
        ids = [0, 1, 2, 3]
        for rank in range(size):
            conn = _FakeConn(rank, size, hashed=[])
            got_keys, got_ids = _select_mla_dump_blocks(conn, keys, ids, 0)
            self.assertEqual(conn.hashed, [])
            self.assertIn(got_keys[0], keys)
            self.assertNotIn(b"#rank", got_keys[0])
            self.assertEqual(len(got_keys), len(got_ids))

    def test_empty_batch_yields_no_writer(self):
        self.assertEqual(_select_mla_dump_blocks(_FakeConn(7, 16), [], [], 0), ([], []))

    def test_rank_wraps_into_tp_size(self):
        keys = [f"k{i}".encode() for i in range(8)]
        ids = list(range(8))
        self.assertEqual(
            _select_mla_dump_blocks(_FakeConn(16, 16), keys, ids, 0),
            _select_mla_dump_blocks(_FakeConn(0, 16), keys, ids, 0),
        )


@needs_numpy
class StripeTailBlocksTests(unittest.TestCase):
    """`stripe_tail_blocks`: whole tail groups follow the shared key slice."""

    def test_selection_covers_whole_groups(self):
        blocks_per_key = 3
        groups = list(range(8 * blocks_per_key))
        # Keys 1 and 5 are selected, so groups 1 and 5 must follow whole.
        self.assertEqual(
            stripe_tail_blocks(groups, slice(1, None, 4), blocks_per_key),
            [3, 4, 5, 15, 16, 17],
        )

    def test_tail_group_matches_the_selected_keys(self):
        blocks_per_key = 2
        size = 4
        keys = list(range(12))
        fa_ids = [100 + i for i in range(12)]
        wa_ids = [400 + i for i in range(12 * blocks_per_key)]
        got_keys, got_fa_ids = _select_mla_dump_blocks(
            _FakeConn(2, size), keys, fa_ids, 0
        )
        # The FAWA path recomputes the same slice from rank and block start.
        key_slice = slice(2, None, size)
        self.assertEqual(got_keys, [2, 6, 10])
        self.assertEqual(got_fa_ids, fa_ids[key_slice])
        # Keys 2, 6, 10 own whole tail groups, in key order.
        self.assertEqual(
            stripe_tail_blocks(wa_ids, key_slice, blocks_per_key),
            [404, 405, 412, 413, 420, 421],
        )


@needs_numpy
class FawaHashIndexTests(unittest.TestCase):
    """FA pointer rows track the canonical index, not the batch offset."""

    class _Layout:
        def __init__(self):
            self.offset_calls = []
            self.addr_calls = []

        def extract_addrs(self, block_ids):
            self.addr_calls.append([int(b) for b in block_ids])
            return np.asarray(block_ids).reshape(-1, 1)

        def extract_addrs_with_offsets(self, block_ids, token_block, offsets):
            self.offset_calls.append([int(o) for o in offsets])
            return np.asarray(offsets).reshape(-1, 1)

    def fake_conn(self, hash_block_size, token_block_size):
        layout = self._Layout()
        conn = NS(
            fa_group_ids=[0],
            group_layouts={0: layout},
            group_metas={0: NS(token_block_size=token_block_size)},
            hash_block_size=hash_block_size,
        )
        return conn, layout

    def test_offsets_follow_the_striped_canonical_indices(self):
        # Ascend layout: one tensor block packs several canonical hash blocks,
        # so the row offset must come from the canonical index of each striped
        # key rather than from its position inside the batch. The batch covers
        # absolute blocks 6..11 of the request, and this rank owns 6 and 10.
        size, start, end = 4, 6, 12
        hash_block_size, token_block_size = 2, 8
        conn, layout = self.fake_conn(hash_block_size, token_block_size)
        conn.tp_rank, conn.tp_size = 2, size
        keys = list(range(start, end))
        ids = list(range(start, end))
        got_keys, got_ids = _select_mla_dump_blocks(conn, keys, ids, start)
        # The FA hash positions use the same selection over the absolute range.
        key_slice = slice((2 - start) % size, None, size)
        indices = list(range(start, end))[key_slice]
        self.assertEqual(got_keys, [6, 10])
        self.assertEqual(indices, [6, 10])
        self.assertEqual(got_ids, [6, 10])
        self.assertEqual([index % size for index in indices], [2, 2])
        _extract_fa_ptr(conn, got_keys, indices, {0: got_ids})
        expected = [(index * hash_block_size) % token_block_size for index in indices]
        self.assertEqual(layout.offset_calls, [expected])
        # Addresses come from the striped candidate ids, not the batch head.
        self.assertEqual(layout.addr_calls, [])

    def test_one_tensor_block_per_hash_block_skips_offsets(self):
        conn, layout = self.fake_conn(hash_block_size=8, token_block_size=8)
        _extract_fa_ptr(conn, [0, 1], [0, 1], {0: [7, 9]})
        self.assertEqual(layout.addr_calls, [[7, 9]])
        self.assertEqual(layout.offset_calls, [])


class HlaMlaSplitTests(unittest.TestCase):
    """HLA `_mla_split_scope`: MLA striped, KDA replicated, load untouched."""

    def split(self, rank, size, mla, kda, positions, is_dump):
        conn = _FakeConn(rank, size)
        ucm = mla + kda
        vllm = [900 + i for i in range(len(ucm))]
        return _mla_split_scope(conn, ucm, vllm, len(mla), is_dump, positions)

    def test_dump_stripes_mla_by_absolute_position(self):
        size = 4
        mla = [b"m0", b"m1", b"m2", b"m3", b"m4", b"m5"]
        kda = [b"d0", b"d1"]
        positions = list(range(6))
        seen = {}
        for rank in range(size):
            _, scoped_ucm, scoped_vllm = self.split(
                rank, size, mla, kda, positions, is_dump=True
            )
            mla_keys = [k for k in scoped_ucm if k.startswith(b"m")]
            for key in mla_keys:
                self.assertNotIn(key, seen)
                seen[key] = rank
            # Every rank still dumps the replicated KDA blocks, after the MLA
            # prefix, and keys and addresses stay the same length.
            kda_part = scoped_ucm[len(mla_keys) :]
            self.assertEqual(len(kda_part), len(kda))
            self.assertTrue(all(k.startswith(b"d") for k in kda_part))
            self.assertEqual(len(scoped_ucm), len(scoped_vllm))
        self.assertEqual(
            seen, {b"m0": 0, b"m1": 1, b"m2": 2, b"m3": 3, b"m4": 0, b"m5": 1}
        )

    def test_short_hla_batch_rotates_away_from_low_ranks(self):
        size = 16
        mla = [b"m0", b"m1", b"m2"]
        owners = set()
        for rank in range(size):
            _, scoped_ucm, _ = self.split(
                rank, size, mla, [], [13, 14, 15], is_dump=True
            )
            if scoped_ucm:
                owners.add(rank)
        self.assertEqual(owners, {13, 14, 15})

    def test_mla_key_and_address_stay_paired(self):
        size = 4
        mla = [b"m0", b"m1", b"m2", b"m3"]
        conn = _FakeConn(1, size)
        _, scoped_ucm, scoped_vllm = _mla_split_scope(
            conn, mla, [900, 901, 902, 903], 4, True, [0, 1, 2, 3]
        )
        self.assertEqual(scoped_ucm, [b"m1"])
        self.assertEqual(scoped_vllm, [901])

    def test_load_path_keeps_every_mla_block(self):
        # Load behaviour is unchanged: each rank still loads the full MLA set.
        size = 4
        mla = [b"m0", b"m1", b"m2", b"m3"]
        for rank in range(size):
            _, scoped_ucm, scoped_vllm = _mla_split_scope(
                _FakeConn(rank, size), mla, [10, 11, 12, 13], 4, False, None
            )
            self.assertEqual(scoped_ucm, mla)
            self.assertEqual(scoped_vllm, [10, 11, 12, 13])

    def test_ranks_partition_the_mla_prefix(self):
        # 16 ranks, 32 MLA blocks: each block is claimed by exactly one rank and
        # the union covers every block. KDA stays replicated on all ranks.
        size = 16
        mla = [f"m{i}".encode() for i in range(32)]
        kda = [b"d0"]
        positions = list(range(32))
        claimed = {}
        for rank in range(size):
            _, scoped_ucm, _ = self.split(rank, size, mla, kda, positions, True)
            mla_keys = [k for k in scoped_ucm if k.startswith(b"m")]
            self.assertEqual(len(mla_keys), 2)
            for key in mla_keys:
                self.assertNotIn(key, claimed)
                claimed[key] = rank
        self.assertEqual(len(claimed), 32)
        self.assertEqual(set(claimed), set(mla))

    def test_empty_positions_select_no_mla_on_dump(self):
        # The builder records one position per MLA dump entry, so an empty
        # position list means the MLA prefix is empty; nothing is dropped.
        _, scoped_ucm, _ = self.split(0, 4, [], [b"d0"], [], is_dump=True)
        self.assertEqual(scoped_ucm, [b"d0"])

    def test_positions_shorter_than_prefix_never_over_selects(self):
        # Inconsistent input must not raise or duplicate: only the indexed
        # entries can be claimed, and each is claimed at most once.
        size = 4
        mla = [b"m0", b"m1", b"m2", b"m3"]
        claimed = []
        for rank in range(size):
            _, scoped_ucm, _ = self.split(rank, size, mla, [], [0, 1], True)
            mla_keys = [k for k in scoped_ucm if k.startswith(b"m")]
            self.assertLessEqual(len(mla_keys), 1)
            claimed.extend(mla_keys)
        self.assertEqual(sorted(claimed), [b"m0", b"m1"])

    def test_non_rank0_kda_is_hashed_and_rank0_kda_is_shared(self):
        size = 4
        kda = [b"d0", b"d1"]
        rank0 = self.split(0, size, [], kda, [], is_dump=True)
        other = self.split(2, size, [], kda, [], is_dump=True)
        self.assertEqual(rank0[1], kda)
        self.assertEqual(other[1], [b"d0#rank", b"d1#rank"])


class MetaFieldTests(unittest.TestCase):
    """The new metadata fields are keyword-only and defaulted."""

    def _classes(self):
        wanted = ("RequestDispatchMeta", "HLARequestDispatchMeta")
        body = []
        for path in (UCM, HLA):
            for node in _parse(path).body:
                if isinstance(node, ast.ClassDef) and node.name in wanted:
                    body.append(node)
        namespace = {"dataclass": dataclasses.dataclass, "field": dataclasses.field}
        exec(compile(ast.Module(body=body, type_ignores=[]), "meta", "exec"), namespace)
        return namespace

    def test_positional_construction_still_works(self):
        meta = self._classes()["RequestDispatchMeta"](("load", [1]), ("dump", [2]))
        self.assertEqual(meta.dump_block_start, 0)

    def test_dump_block_start_is_keyword_only(self):
        namespace = self._classes()
        with self.assertRaises(TypeError):
            namespace["RequestDispatchMeta"](("l", []), ("d", []), 5)
        meta = namespace["RequestDispatchMeta"](
            ("l", []), ("d", []), dump_block_start=5
        )
        self.assertEqual(meta.dump_block_start, 5)

    def test_hla_meta_defaults_mla_positions(self):
        namespace = self._classes()
        meta = namespace["HLARequestDispatchMeta"](("l", []), ("d", []))
        self.assertEqual(meta.mla_dump_positions, [])
        meta = namespace["HLARequestDispatchMeta"](
            ("l", []), ("d", []), mla_dump_positions=[1, 2]
        )
        self.assertEqual(meta.mla_dump_positions, [1, 2])


if __name__ == "__main__":
    unittest.main()
