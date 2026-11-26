"""CPU tests for the FAWA partial-group slice bound check (#1647).

A dispatch range computed from the requested number of blocks can run one
block past the blocks a group actually holds under preemption /
fragmented-block conditions (``index 65 is out of bounds for axis 0 with
size 65``). The slice must clamp to the last available block instead of
raising, because an exception here escapes the scheduler thread and kills
EngineCore.
"""

import ast
import copy
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
CONNECTOR_PATH = REPO_ROOT / "ucm" / "integration" / "vllm" / "hma_connector.py"


def _load_slice_symbols():
    """Load only the slice helpers, without importing vLLM or torch."""

    tree = ast.parse(CONNECTOR_PATH.read_text(encoding="utf-8-sig"))
    meta = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "KVCacheGroupMeta"
    )
    connector_cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "UCMFAWAConnector"
    )
    # Keep only the methods under test and drop the vLLM base classes.
    stripped_cls = copy.copy(connector_cls)
    stripped_cls.bases = []
    stripped_cls.keywords = []
    stripped_cls.body = [
        node
        for node in connector_cls.body
        if getattr(node, "name", None)
        in {"_slice_group_block_ids", "_clamp_group_block_indices"}
    ]
    assert {n.name for n in stripped_cls.body} == {
        "_slice_group_block_ids",
        "_clamp_group_block_indices",
    }
    module = ast.Module(body=[meta, stripped_cls], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"dataclass": dataclass, "logger": Mock(), "np": np}
    exec(compile(module, str(CONNECTOR_PATH), "exec"), namespace)
    return namespace


SYMBOLS = _load_slice_symbols()
KVCacheGroupMeta = SYMBOLS["KVCacheGroupMeta"]
Connector = SYMBOLS["UCMFAWAConnector"]
LOGGER = SYMBOLS["logger"]


def _connector(group_metas, window_group_ids=(), transient_group_ids=()):
    connector = object.__new__(Connector)
    connector.group_metas = group_metas
    connector.window_group_ids = set(window_group_ids)
    connector.transient_group_ids = set(transient_group_ids)
    return connector


def _fa_connector(token_block_size=32):
    return _connector(
        {0: KVCacheGroupMeta(0, token_block_size, 0, 0)},
    )


def _boundary_token_idx(first_block, last_block, hash_block_size=32):
    """Last-token index of each canonical hash block, as the callers build it."""
    return np.arange(first_block + 1, last_block + 1) * hash_block_size - 1


def test_fa_slice_at_boundary_clamps_instead_of_raising():
    """Regression test for #1647: index 65 on a 65-element array."""
    connector = _fa_connector()
    group_block_ids = list(range(65))
    # Dispatch range covers 66 canonical blocks while the group holds 65.
    boundary_idx = _boundary_token_idx(0, 66)
    LOGGER.warning.reset_mock()
    result = connector._slice_group_block_ids(
        0, group_block_ids, boundary_idx, fetch_wa_block_wise=False
    )
    # Length is preserved so ids stay aligned with the canonical block keys;
    # the one over-long index repeats the last available block.
    assert result == list(range(65)) + [64]
    assert LOGGER.warning.called


def test_fa_slice_within_bounds_is_unchanged():
    connector = _fa_connector()
    group_block_ids = list(range(65))
    boundary_idx = _boundary_token_idx(0, 65)
    result = connector._slice_group_block_ids(
        0, group_block_ids, boundary_idx, fetch_wa_block_wise=False
    )
    assert result == list(range(65))


def test_fa_slice_with_empty_group_returns_empty():
    connector = _fa_connector()
    result = connector._slice_group_block_ids(
        0, [], _boundary_token_idx(0, 3), fetch_wa_block_wise=False
    )
    assert result == []


def test_slice_with_empty_boundaries_returns_empty():
    connector = _fa_connector()
    result = connector._slice_group_block_ids(
        0, list(range(65)), np.empty(0, dtype=np.int64), fetch_wa_block_wise=False
    )
    assert result == []


def test_transient_group_returns_empty():
    connector = _connector({0: KVCacheGroupMeta(0, 32, 0, 0)}, transient_group_ids=(0,))
    result = connector._slice_group_block_ids(
        0, list(range(65)), _boundary_token_idx(0, 3), fetch_wa_block_wise=False
    )
    assert result == []


def test_window_group_without_tail_returns_empty():
    connector = _connector({0: KVCacheGroupMeta(0, 32, 2, 0)}, window_group_ids=(0,))
    result = connector._slice_group_block_ids(
        0, list(range(65)), _boundary_token_idx(0, 3), fetch_wa_block_wise=False
    )
    assert result == []


def test_window_chunk_wise_slice_clamps_instead_of_raising():
    connector = _connector({0: KVCacheGroupMeta(0, 32, 2, 16)}, window_group_ids=(0,))
    # Final boundary implies block 66 while the group holds 65 blocks.
    boundary_idx = _boundary_token_idx(0, 66)
    result = connector._slice_group_block_ids(
        0, list(range(65)), boundary_idx, fetch_wa_block_wise=False
    )
    assert result == [64, 64]


def test_window_block_wise_slice_clamps_instead_of_raising():
    connector = _connector({0: KVCacheGroupMeta(0, 32, 2, 16)}, window_group_ids=(0,))
    # One tail row per canonical boundary; the boundary implies block 66
    # while the group holds 65 blocks.
    boundary_idx = np.array([66 * 32 - 1], dtype=np.int64)
    result = connector._slice_group_block_ids(
        0, list(range(65)), boundary_idx, fetch_wa_block_wise=True
    )
    assert result == [64, 64]
