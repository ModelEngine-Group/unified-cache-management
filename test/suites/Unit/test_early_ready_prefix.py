"""Exercise the real Early prefix-query method without native extensions or vLLM."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def ready(monkeypatch):
    root = Path(__file__).parents[3]
    # Only native construction/I/O is substituted. The lookup context and
    # EarlyPipelineStore implementation are loaded directly from this checkout.
    for name in (
        "ucm",
        "ucm.integration",
        "ucm.integration.vllm",
        "ucm.store",
        "ucm.store.pipeline",
    ):
        package = ModuleType(name)
        package.__path__ = [str(root.joinpath(*name.split(".")))]
        monkeypatch.setitem(sys.modules, name, package)

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, root / path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    lookup = load(
        "ucm.integration.vllm.early_lookup", "ucm/integration/vllm/early_lookup.py"
    )

    class NativeStore:
        def lookup_on_prefix(self, block_ids):
            self.native_queries.append(block_ids)
            return 1

    native = ModuleType("ucm.store.pipeline.connector")
    native.UcmPipelineStore = NativeStore
    native.UcmPipelineStoreBuilder = SimpleNamespace(register=lambda *args: None)
    native.ucmpipelinestore = SimpleNamespace()
    monkeypatch.setitem(sys.modules, native.__name__, native)
    early = load(
        "ucm.store.pipeline.early_connector", "ucm/store/pipeline/early_connector.py"
    )
    store = object.__new__(early.EarlyPipelineStore)
    store.native_queries = []
    job = SimpleNamespace(
        state="READY", hit=12, blocks=[n.to_bytes(16, "big") for n in range(20)]
    )
    connector = SimpleNamespace(store=store)

    def query(blocks, *, hbm=0, hbm_only=False, owner=connector, active_job=job):
        context = lookup.LookupContext(owner, object(), active_job, hbm, hbm_only)
        token = lookup.current_lookup.set(context)
        try:
            return store.lookup_on_prefix(blocks)
        finally:
            lookup.current_lookup.reset(token)

    return SimpleNamespace(store=store, job=job, query=query, lookup=lookup)


@pytest.mark.parametrize("hbm", [0, 2, 5])
def test_waiting_retry_counts_from_query_start_instead_of_hbm(ready, hbm):
    assert ready.query(ready.job.blocks[hbm:], hbm=hbm) == 12 - hbm - 1
    # Native resume accounting starts at the previous hit minus one block.
    # Four retries must still report the same protected prefix, not grow it.
    for _ in range(4):
        assert ready.query(ready.job.blocks[11:], hbm=hbm) == 0
    assert ready.store.native_queries == []


@pytest.mark.parametrize("start", [12, 13, 19, 20])
def test_query_outside_protected_prefix_reports_no_hit(ready, start):
    assert ready.query(ready.job.blocks[start:]) == -1


def test_rank_filtered_query_is_bounded_by_its_own_length(ready):
    assert ready.query(ready.job.blocks[4:7], hbm=2) == 2


def test_mismatched_block_stops_the_contiguous_prefix(ready):
    blocks = ready.job.blocks[8:10] + [b"unknown"] + ready.job.blocks[11:]
    assert ready.query(blocks, hbm=2) == 1


@pytest.mark.parametrize("blocks", [[], [b"unknown"]])
def test_empty_or_unknown_query_reports_no_hit(ready, blocks):
    assert ready.query(blocks) == -1
    assert ready.store.native_queries == []


def test_query_before_hbm_boundary_is_not_reused(ready):
    assert ready.query(ready.job.blocks[:2], hbm=5) == -1


def test_missing_prepared_block_list_reports_no_hit(ready):
    ready.job.blocks = None
    assert ready.query([b"unknown"]) == -1


def test_protected_count_cannot_exceed_prepared_blocks(ready):
    ready.job.hit = 100
    assert ready.query(ready.job.blocks[18:] + [b"unknown"]) == 1


def test_swe_boundary_repeated_query_keeps_1037_protected_blocks(ready):
    ready.job.blocks = [n.to_bytes(16, "big") for n in range(1039)]
    ready.job.hit = 1037
    assert ready.query(ready.job.blocks) == 1036
    for _ in range(3):
        assert ready.query(ready.job.blocks[1036:]) == 0
    assert ready.store.native_queries == []


def test_hbm_only_does_not_query_external_storage(ready):
    assert ready.query(ready.job.blocks, hbm_only=True) == -1
    assert ready.store.native_queries == []


@pytest.mark.parametrize("state", ["PENDING", "FAILED", "CANCELLED"])
def test_non_ready_states_keep_native_lookup(ready, state):
    ready.job.state = state
    assert ready.query(ready.job.blocks) == 1
    assert ready.store.native_queries == [ready.job.blocks]


def test_missing_job_keeps_native_lookup(ready):
    assert ready.query(ready.job.blocks, active_job=None) == 1
    assert ready.store.native_queries == [ready.job.blocks]


def test_other_connector_context_keeps_native_lookup(ready):
    assert ready.query(ready.job.blocks, owner=SimpleNamespace(store=object())) == 1
    assert ready.store.native_queries == [ready.job.blocks]


def test_no_lookup_context_keeps_native_lookup(ready):
    assert ready.lookup.current_lookup.get() is None
    assert ready.store.lookup_on_prefix(ready.job.blocks) == 1
    assert ready.store.native_queries == [ready.job.blocks]
