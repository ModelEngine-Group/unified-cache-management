"""Request logging without SGLang, GPUs, native UCM or metrics exporters."""

import logging
import types
from collections import OrderedDict
from unittest.mock import Mock

import pytest
import test_sglang_load_workers as workers
from common.sglang_utils import linker_module


@pytest.fixture
def linker(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=linker_module.logger.name)
    monkeypatch.setattr(
        linker_module, "PoolName", types.SimpleNamespace(KV="kv"), raising=False
    )
    obj = object.__new__(linker_module.UcmDirectLinker)
    obj._lookup_owner = True
    obj._lookup_progress = OrderedDict()
    obj._pending_loads = {}
    obj._dp_rank, obj._attn_dp_rank, obj._tp_rank = 1, 2, 3
    obj.page_size = 64
    obj._expand = lambda transfers, **kwargs: transfers
    obj._page_transfers = lambda transfers, **kwargs: [
        types.SimpleNamespace(
            store_group="kv",
            source_pool="kv",
            pools=("kv",),
            layers=(0,),
            hit_policy="all_pages",
            page_keys=tuple(transfers[0].keys),
        )
    ]
    obj.adapter = types.SimpleNamespace(lookup=Mock())
    return obj


def transfers():
    return [types.SimpleNamespace(name="kv", keys=["a", "b", "c", "d"])]


@pytest.mark.parametrize("boundaries, hit_pages", [([], 0), ([1, 4], 4)])
def test_lookup_preserves_boundaries_without_duplicate_logs(
    linker, monkeypatch, caplog, boundaries, hit_pages
):
    linker.adapter.lookup.return_value = boundaries
    caplog.set_level(logging.DEBUG, logger=linker_module.logger.name)
    clock = Mock(side_effect=AssertionError("lookup logging must not read clock"))
    monkeypatch.setattr(
        linker_module, "time", types.SimpleNamespace(perf_counter=clock)
    )
    for _ in range(2):
        assert linker.lookup("request-a", transfers()) == boundaries
    assert not caplog.records
    clock.assert_not_called()
    assert linker.adapter.lookup.call_count == 2
    assert linker.adapter.lookup.call_args.kwargs["resume_from_pages"] == hit_pages


def test_non_owner_and_empty_lookup_do_not_report_hits(linker, caplog):
    linker._lookup_owner = False
    assert linker.lookup("request-a", transfers()) == [1, 2, 3, 4]
    linker._lookup_owner = True
    assert linker.lookup("request-b", []) == []
    assert linker.lookup("request-c", [types.SimpleNamespace(name="kv", keys=[])]) == []
    linker.adapter.lookup.assert_not_called()
    assert not caplog.records


def test_lookup_failure_does_not_report_success(linker, caplog):
    linker.adapter.lookup.side_effect = RuntimeError("lookup failed")
    with pytest.raises(RuntimeError, match="lookup failed"):
        linker.lookup("request-a", transfers())
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.ERROR
    assert caplog.records[0].exc_info
    assert "lookup_failed request_id=request-a" in caplog.records[0].getMessage()
    assert not linker._lookup_progress


def test_disabled_info_does_not_read_clock(linker, monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger=linker_module.logger.name)
    clock = Mock(side_effect=AssertionError("logging clock must remain unused"))
    monkeypatch.setattr(
        linker_module, "time", types.SimpleNamespace(perf_counter=clock)
    )
    linker.adapter.lookup.return_value = [4]
    assert linker.lookup("request-a", transfers()) == [4]
    workers.LoadWorkerTest().run_worker()
    clock.assert_not_called()
    assert not caplog.records


def test_load_queue_and_rejection_logs(linker, caplog):
    caplog.set_level(logging.DEBUG, logger=linker_module.logger.name)
    assert not linker.load("empty", [])
    assert linker.load("request-a", transfers())
    assert not caplog.records
    with pytest.raises(RuntimeError, match="already queued"):
        linker.load("request-a", transfers())
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelno == logging.ERROR
    assert "request_id=request-a reason=already_queued" in record.getMessage()
    assert "dp_rank=1 attn_dp_rank=2 tp_rank=3" in record.getMessage()


def test_load_resolution_failure_preserves_exception(linker, caplog):
    linker._expand = Mock(side_effect=RuntimeError("bad transfers"))
    with pytest.raises(RuntimeError, match="bad transfers"):
        linker.load("request-a", transfers())
    assert not linker._pending_loads
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelno == logging.ERROR and record.exc_info
    assert "request_id=request-a stage=resolve_transfers" in record.getMessage()


@pytest.mark.parametrize(
    "failure, stage",
    [
        ({}, None),
        ({"fail_wait": 1}, "wait"),
        ({"fail_submit": 2}, "submit"),
        ({"fail_check": 0}, "check"),
    ],
)
def test_load_only_logs_failures_after_cleanup(monkeypatch, caplog, failure, stage):
    caplog.set_level(logging.DEBUG, logger=linker_module.logger.name)
    clock = Mock(side_effect=AssertionError("load logging must not read clock"))
    monkeypatch.setattr(
        linker_module, "time", types.SimpleNamespace(perf_counter=clock)
    )
    # Existing fixture verifies that every submitted task is reaped, completion
    # ordering is preserved and the batch result reaches the consumer.
    events = workers.LoadWorkerTest().run_worker(**failure)
    success = not failure
    assert events[-1] == ("result", success)
    clock.assert_not_called()
    assert len(caplog.records) == (0 if success else 1)
    if not success:
        errors = [record for record in caplog.records if record.exc_info]
        assert len(errors) == 1
        assert errors[0].levelno == logging.ERROR
        assert "request_ids=['req']" in errors[0].getMessage()
        assert f"stage={stage}" in errors[0].getMessage()
        assert "load_batch_id=0" in errors[0].getMessage()
