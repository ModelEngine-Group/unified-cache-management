import ast
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[3]
CONNECTOR_PATH = REPO_ROOT / "ucm" / "integration" / "vllm" / "blend_connector.py"


@dataclass
class RequestDispatchMeta:
    # Mirrors ucm/integration/vllm/ucm_connector.py's RequestDispatchMeta
    # (load_block_ids, dump_block_ids), which BlendRequestDispatchMeta
    # extends; not itself part of the code under test.
    load_block_ids: tuple
    dump_block_ids: tuple


class FakeRankConsistency:
    """Stands in for UCMBlendConnector._rank_consistency: looks each hash up
    in a fixed hit-set instead of a real store, mirroring lookup_all's
    (store, hashes) -> List[bool] signature."""

    def __init__(self, hit_hashes):
        self.hit_hashes = set(hit_hashes)

    def lookup_all(self, store, hashes):
        return [h in self.hit_hashes for h in hashes]


def _load_blend_symbols():
    source = CONNECTOR_PATH.read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    top_level_names = {
        "ChunkMetaData",
        "BlendStage",
        "BlendRequestMeta",
        "BlendRequestDispatchMeta",
    }
    selected_nodes = [
        node for node in tree.body if getattr(node, "name", None) in top_level_names
    ]

    method_names = {"_get_req_chunk_hit", "_generate_blend_dispatch_meta"}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "UCMBlendConnector":
            selected_nodes.extend(
                item
                for item in node.body
                if isinstance(item, ast.FunctionDef) and item.name in method_names
            )

    module = ast.Module(body=selected_nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "List": list,
        "Self": None,
        "Tuple": tuple,
        "dataclass": __import__("dataclasses").dataclass,
        "field": __import__("dataclasses").field,
        "Enum": __import__("enum").Enum,
        "auto": __import__("enum").auto,
        "itertools": __import__("itertools"),
        "RequestDispatchMeta": RequestDispatchMeta,
    }
    exec(compile(module, str(CONNECTOR_PATH), "exec"), namespace)
    return namespace


BLEND_SYMBOLS = _load_blend_symbols()
ChunkMetaData = BLEND_SYMBOLS["ChunkMetaData"]
BlendStage = BLEND_SYMBOLS["BlendStage"]
BlendRequestMeta = BLEND_SYMBOLS["BlendRequestMeta"]
get_req_chunk_hit = BLEND_SYMBOLS["_get_req_chunk_hit"]
generate_blend_dispatch_meta = BLEND_SYMBOLS["_generate_blend_dispatch_meta"]


def _build_connector(rank_consistency, block_size=16):
    return SimpleNamespace(
        _rank_consistency=rank_consistency, store=None, block_size=block_size
    )


def _build_two_chunk_request(block_size=16):
    # chunk0: global blocks 0-1, chunk1: global blocks 2-4. req_chunks_hashes
    # is the dense concatenation _process_req builds: no gaps, same order.
    chunk0_hashes = [f"chunk0-blk{i}" for i in range(2)]
    chunk1_hashes = [f"chunk1-blk{i}" for i in range(3)]
    req_chunks_hashes = chunk0_hashes + chunk1_hashes
    # prefix_block_hashes is the chained hash over the whole 5-block request,
    # a different hash space from the per-chunk hashes above.
    prefix_block_hashes = [f"prefix-chain-blk{i}" for i in range(5)]

    chunk0 = ChunkMetaData(
        start_token_dix=0,
        chunk_tokens_len=2 * block_size,
        start_blk_idx=0,
        chunk_blks_len=2,
        cached_start_position=0,
        chunk_blks_hash=list(chunk0_hashes),
    )
    chunk1 = ChunkMetaData(
        start_token_dix=2 * block_size,
        chunk_tokens_len=3 * block_size,
        start_blk_idx=2,
        chunk_blks_len=3,
        cached_start_position=0,
        chunk_blks_hash=list(chunk1_hashes),
    )
    return prefix_block_hashes, [chunk0, chunk1], req_chunks_hashes


class GetReqChunkHitPartialPrefixOverlapTest(unittest.TestCase):
    def test_prefix_hit_run_crossing_chunk_boundary_does_not_corrupt_chunk(self):
        prefix_block_hashes, req_chunks_meta, req_chunks_hashes = (
            _build_two_chunk_request()
        )
        # Prefix hit run covers global blocks 0,1,2 (pc_hit_blocks=3): it
        # extends past chunk0's own 2 blocks into chunk1's first block.
        connector = _build_connector(
            FakeRankConsistency(hit_hashes=prefix_block_hashes[:3])
        )
        chunk0 = req_chunks_meta[0]

        pc_hit_blocks, _ = get_req_chunk_hit(
            connector,
            BlendStage.CACHE_BLEND,
            prefix_block_hashes,
            req_chunks_meta,
            req_chunks_hashes,
        )

        self.assertEqual(pc_hit_blocks, 3)
        self.assertGreaterEqual(
            chunk0.chunk_blks_len,
            0,
            "chunk0.chunk_blks_len went negative: it was trimmed by the full "
            "prefix-hit run without clamping to its own length",
        )
        self.assertNotIn(
            chunk0,
            req_chunks_meta,
            "chunk0 is entirely covered by the prefix-cache hit but was not "
            "popped from req_chunks_meta",
        )

    def test_prefix_hit_run_crossing_chunk_boundary_does_not_double_load_block(self):
        prefix_block_hashes, req_chunks_meta, req_chunks_hashes = (
            _build_two_chunk_request()
        )
        ucm_block_hashs = list(prefix_block_hashes)
        vllm_block_ids = [100, 101, 102, 103, 104]
        connector = _build_connector(
            FakeRankConsistency(hit_hashes=prefix_block_hashes[:3])
        )

        pc_hit_blocks, _ = get_req_chunk_hit(
            connector,
            BlendStage.CACHE_BLEND,
            prefix_block_hashes,
            req_chunks_meta,
            req_chunks_hashes,
        )

        req_meta = BlendRequestMeta(
            ucm_block_hashs=ucm_block_hashs,
            pc_hit_block_num=pc_hit_blocks,
            chunks_meta=req_chunks_meta,
            blend_stage=BlendStage.CACHE_BLEND,
        )
        dispatch = generate_blend_dispatch_meta(
            connector, req_meta, new_tokens=0, vllm_block_ids=vllm_block_ids
        )
        _, load_vllm_ids = dispatch.load_block_ids

        # Block 102 (global block 2, chunk1's first block) is already served
        # by the prefix-path slice (vllm_block_ids[:pc_hit_block_num]); it
        # must not also be dispatched again via chunk1's own hit list.
        self.assertEqual(
            load_vllm_ids.count(102),
            1,
            f"block 102 dispatched for LOAD {load_vllm_ids.count(102)} times "
            f"in {load_vllm_ids}: once via the prefix path and again via "
            f"chunk1, since chunk1 was never trimmed for the overlap",
        )

    def test_prefix_hit_within_first_chunk_still_trims_it(self):
        # Regression guard for the ordinary case this function already
        # handled correctly: pc_hit_blocks smaller than chunk0's own length.
        prefix_block_hashes, req_chunks_meta, req_chunks_hashes = (
            _build_two_chunk_request()
        )
        connector = _build_connector(
            FakeRankConsistency(hit_hashes=prefix_block_hashes[:1])
        )
        chunk0 = req_chunks_meta[0]

        pc_hit_blocks, _ = get_req_chunk_hit(
            connector,
            BlendStage.CACHE_BLEND,
            prefix_block_hashes,
            req_chunks_meta,
            req_chunks_hashes,
        )

        self.assertEqual(pc_hit_blocks, 1)
        self.assertIn(chunk0, req_chunks_meta)
        self.assertEqual(chunk0.chunk_blks_len, 1)
        self.assertEqual(chunk0.start_blk_idx, 1)


if __name__ == "__main__":
    unittest.main()
