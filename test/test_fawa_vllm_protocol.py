"""CPU protocol tests against real vLLM source, without importing GPU packages.

Run with VLLM_SOURCE_DIR pointing to a vLLM checkout. VLLM_SOURCE_REF optionally
selects a git revision (e.g. the revision used by a reviewer).
"""

import ast
import os
import subprocess
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_ucm_connector_metrics import UCMConnector, _async_fawa_worker


def _engine_protocol():
    source_dir = os.environ.get("VLLM_SOURCE_DIR")
    if not source_dir:
        pytest.skip("set VLLM_SOURCE_DIR to exercise the installed engine's protocol")
    namespace = dict(
        __name__=__name__,
        dataclass=dataclass,
        field=field,
        chain=chain,
        logger=Mock(),
        RequestStatus=SimpleNamespace(WAITING_FOR_REMOTE_KVS="waiting"),
    )
    for path, class_name, methods in (
        ("vllm/v1/outputs.py", "KVConnectorOutput", None),
        (
            "vllm/distributed/kv_transfer/kv_connector/utils.py",
            "KVOutputAggregator",
            {"__init__", "aggregate"},
        ),
        ("vllm/v1/worker/gpu/kv_connector.py", "ActiveKVConnector", {"post_forward"}),
        (
            "vllm/v1/core/sched/scheduler.py",
            "Scheduler",
            {"_handle_failed_recving", "_handle_invalid_blocks"},
        ),
    ):
        ref = os.environ.get("VLLM_SOURCE_REF")
        source = (
            subprocess.check_output(
                ["git", "-C", source_dir, "show", f"{ref}:{path}"], text=True
            )
            if ref
            else (Path(source_dir) / path).read_text()
        )
        cls = next(
            n for n in ast.parse(source).body if getattr(n, "name", None) == class_name
        )
        if methods is not None:
            cls.bases = []
            cls.body = [n for n in cls.body if getattr(n, "name", None) in methods]
            assert {n.name for n in cls.body} == methods
        # Execute the unmodified methods with lightweight dependency objects.
        module = ast.parse("from __future__ import annotations")
        module.body.append(cls)
        exec(compile(ast.fix_missing_locations(module), path, "exec"), namespace)
    return SimpleNamespace(**namespace)


@pytest.mark.parametrize("recompute", [False, True])
def test_fawa_failure_survives_worker_output_tp_aggregation_and_scheduler(recompute):
    engine = _engine_protocol()
    workers = [_async_fawa_worker(), _async_fawa_worker()]
    adapters = []
    for worker in workers:
        worker.start_load_kv(None)
        outer = object.__new__(UCMConnector)
        outer.connector = worker
        outer.get_kv_connector_stats = Mock(return_value=None)
        outer.get_kv_connector_kv_cache_events = Mock(return_value=None)
        outer.build_connector_worker_meta = Mock(return_value=None)
        outer.wait_for_save = Mock()
        adapter = object.__new__(engine.ActiveKVConnector)
        adapter._disabled = False
        adapter._pending_load_kwargs = None
        adapter.kv_connector = outer
        adapters.append(adapter)

    def aggregate_step():
        outputs = [
            SimpleNamespace(kv_connector_output=adapter.post_forward(set()))
            for adapter in adapters
        ]
        return aggregator.aggregate(outputs).kv_connector_output

    aggregator = engine.KVOutputAggregator(expected_finished_count=2)
    first = list(workers[0]._load_wait_executor.futures.values())
    second = list(workers[1]._load_wait_executor.futures.values())
    first[0].set_exception(TimeoutError("FA timeout after drain"))
    # WA is still writing: even the failed worker must not report completion.
    output = aggregate_step()
    assert not output.finished_recving and not output.failed_recving
    first[1].set_result(None)
    output = aggregate_step()
    # Rank 0 has failed and drained, but rank 1 still owns its HBM buffers.
    assert not output.finished_recving and not output.failed_recving
    assert not output.invalid_block_ids
    assert aggregator._failed_recving_pending == {"req"}
    for future in second:
        future.set_result(None)
    output = aggregate_step()
    assert output.finished_recving == output.failed_recving == {"req"}
    assert output.invalid_block_ids == set()

    scheduler = object.__new__(engine.Scheduler)
    request = SimpleNamespace(status="waiting", num_computed_tokens=128)
    scheduler.requests = {"req": request}
    scheduler.recompute_kv_load_failures = recompute
    scheduler.failed_recving_kv_req_ids = set()
    scheduler.kv_cache_config = SimpleNamespace(kv_cache_groups=[object(), object()])
    affected = scheduler._handle_failed_recving(output.failed_recving)
    if recompute:
        # Exercise the engine's recompute consumer independently; UCM's existing
        # constructor guard against multi-group recompute is not changed here.
        assert affected == set()
        assert request.num_computed_tokens == 0
        assert scheduler.failed_recving_kv_req_ids == {"req"}
    else:
        assert affected == {"req"}  # update_from_output terminates this request.
    # The explicit rejection was added after some engines already supported
    # failed_recving. Only assert that diagnostic on engines that implement it.
    if any(
        isinstance(value, str) and "multiple KV cache groups" in value
        for value in scheduler._handle_invalid_blocks.__code__.co_consts
    ):
        with pytest.raises(RuntimeError, match="multiple KV cache groups"):
            scheduler._handle_invalid_blocks({10}, {})
    assert not aggregate_step().failed_recving
