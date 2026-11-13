"""CPU checks for request-level hybrid load failure handling.

Run without the GPU integration conftest:
    python -m pytest --confcutdir=test/suites/Unit \
        test/suites/Unit/test_hybrid_load_failure_patch.py

Set VLLM_SOURCE_DIR to exercise real, locally available scheduler revisions too.
No engine import, remote access or GPU is required.
"""

import ast
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[3]
PATCH = ROOT / "ucm/integration/vllm/patch/load_failure_patch.py"
STATUS = NS(RUNNING="running", WAITING_FOR_REMOTE_KVS="receiving")


def apply_patch(scheduler_cls):
    # Execute production functions with only the import-hook plumbing replaced.
    tree = ast.parse(PATCH.read_text(encoding="utf-8-sig"))
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    namespace = {
        "when_imported": lambda name: lambda fn: fn,
        "patch_or_inject": setattr,
    }
    exec(compile(tree, str(PATCH), "exec"), namespace)
    namespace["patch_core_sched_scheduler"](
        NS(Scheduler=scheduler_cls, RequestStatus=STATUS, logger=Mock())
    )


def scheduler(blocks, receiving=(), groups=2, recompute=False, cls=None):
    if cls is None:

        class Scheduler:
            def _handle_invalid_blocks(self, *args, **kwargs):
                return self.original(*args, **kwargs)

        cls = Scheduler
    apply_patch(cls)
    obj = cls()
    obj.original = Mock(return_value={"native"})
    obj.kv_cache_config = NS(kv_cache_groups=[object()] * groups)
    obj.recompute_kv_load_failures = recompute
    obj.requests = {
        key: NS(
            request_id=key,
            status=(
                STATUS.WAITING_FOR_REMOTE_KVS if key in receiving else STATUS.RUNNING
            ),
            num_computed_tokens=128,
        )
        for key in blocks
    }
    obj.running = [req for key, req in obj.requests.items() if key not in receiving]
    obj.kv_cache_manager = NS(
        num_kv_cache_groups=groups,
        block_pool=NS(null_block=NS(block_id=0)),
        get_block_ids=Mock(side_effect=lambda key: blocks[key]),
        evict_blocks=Mock(),
    )
    return obj


@pytest.mark.parametrize("bad", [10, 20])
@pytest.mark.parametrize("reverse", [False, True])
def test_attention_or_state_failure_evicts_all_groups(bad, reverse):
    groups = ([10, 11], [0, 20, 21])
    if reverse:
        groups = groups[::-1]
    obj = scheduler({"failed": groups, "healthy": ([30], [40])})
    assert obj._handle_invalid_blocks({bad}, {"failed": 1}) == {"failed"}
    obj.kv_cache_manager.evict_blocks.assert_called_once_with({10, 11, 20, 21})
    # The patch returns IDs to the engine; it must not free in-flight buffers,
    # change request state or roll back Mamba using an attention token offset.
    assert obj.requests["failed"].num_computed_tokens == 128
    assert obj.requests["healthy"].status == STATUS.RUNNING
    obj.original.assert_not_called()
    # The same scheduler remains usable for the next healthy step.
    assert obj._handle_invalid_blocks(set(), {"healthy": 1}) == set()


def test_shared_failed_blocks_report_every_owner():
    obj = scheduler({"a": ([10], [20]), "b": ([11], [20]), "ok": ([30], [40])})
    assert obj._handle_invalid_blocks({20}, {}) == {"a", "b"}
    obj.kv_cache_manager.evict_blocks.assert_called_once_with({10, 11, 20})


def test_flat_id_alias_is_conservative_across_groups():
    obj = scheduler({"a": ([10], [20]), "b": ([20], [30])})
    assert obj._handle_invalid_blocks({20}, {}) == {"a", "b"}


def test_empty_attention_group_does_not_hide_state_failure():
    obj = scheduler({"a": ([], [20])})
    assert obj._handle_invalid_blocks({20}, {}) == {"a"}


def test_null_and_unknown_blocks_do_not_fail_requests():
    obj = scheduler({"a": ([10], [0, 20]), "b": ([11], [0, 21])})
    assert obj._handle_invalid_blocks({0, 99}, {}) == set()
    obj.kv_cache_manager.evict_blocks.assert_not_called()


def test_receiving_and_running_failures_are_reported_together():
    obj = scheduler({"recv": ([10], [20]), "run": ([30], [40])}, receiving={"recv"})
    assert obj._handle_invalid_blocks({20, 40}, {}) == {"recv", "run"}
    # Async receive blocks are not yet cached; native finish_requests owns
    # their transfer lifetime, while the sync cache needs explicit eviction.
    obj.kv_cache_manager.evict_blocks.assert_called_once_with({30, 40})
    assert obj.requests["recv"].status == STATUS.WAITING_FOR_REMOTE_KVS


@pytest.mark.parametrize("groups,recompute", [(1, False), (1, True), (2, True)])
def test_other_policies_and_single_group_delegate(groups, recompute):
    obj = scheduler({"req": ([10],)}, groups=groups, recompute=recompute)
    assert obj._handle_invalid_blocks({10}, {"req": 1}) == {"native"}
    obj.original.assert_called_once_with({10}, {"req": 1})


@pytest.mark.parametrize(
    "ref", ["v0.18.0", "v0.20.2", "v0.27.1", "v0.30.0", "v0.31.0", "v0.32.0"]
)
def test_real_scheduler_failure_protocol(ref):
    source_dir = os.environ.get("VLLM_SOURCE_DIR")
    if not source_dir:
        pytest.skip("set VLLM_SOURCE_DIR for local engine source protocol checks")
    source = subprocess.check_output(
        [
            "git",
            "-C",
            source_dir,
            "show",
            f"refs/tags/{ref}:vllm/v1/core/sched/scheduler.py",
        ],
        text=True,
        encoding="utf-8",
    )
    cls = next(
        n for n in ast.parse(source).body if getattr(n, "name", None) == "Scheduler"
    )
    output_method = next(
        n for n in cls.body if getattr(n, "name", None) == "update_from_output"
    )
    cls.bases = []
    cls.body = [
        n
        for n in cls.body
        if getattr(n, "name", None)
        in {"_handle_invalid_blocks", "_update_requests_with_invalid_blocks"}
    ]
    module = ast.parse("from __future__ import annotations")
    module.body.append(cls)
    namespace = {"RequestStatus": STATUS, "logger": Mock()}
    exec(compile(ast.fix_missing_locations(module), ref, "exec"), namespace)
    obj = scheduler(
        {"failed": ([10], [20]), "healthy": ([30], [40])}, cls=namespace["Scheduler"]
    )
    failed = obj._handle_invalid_blocks({20}, {"failed": 1, "healthy": 1})
    assert failed == {"failed"}

    # Execute the engine's unchanged terminal-error consumer (not the entire
    # model-output pipeline). Older engines finish inline; newer ones merge
    # load failures into error_req_ids first.
    consumers = [
        n
        for n in output_method.body
        if isinstance(n, ast.If)
        and (
            ast.unparse(n.test)
            == "failed_kv_load_req_ids and (not self.recompute_kv_load_failures)"
            or ast.unparse(n.test) == "error_req_ids"
        )
    ]
    assert consumers, f"No native request-level fail consumer found in {ref}"
    request = obj.requests["failed"]
    request.client_index = 0
    request.trace_headers = None
    request.num_cached_tokens = 0
    request.get_finished_reason = lambda: "error"
    request.take_events = lambda: []
    obj.finish_requests = Mock(return_value=[request])
    outputs = {0: []}
    env = dict(
        self=obj,
        failed_kv_load_req_ids=failed,
        error_req_ids=set(),
        outputs=outputs,
        RequestStatus=NS(FINISHED_ERROR="error"),
        EngineCoreOutput=NS,
    )
    exec(compile(ast.Module(body=consumers, type_ignores=[]), ref, "exec"), env)
    obj.finish_requests.assert_called_once_with({"failed"}, "error")
    assert [out.request_id for out in outputs[0]] == ["failed"]
    assert outputs[0][0].new_token_ids == []
    assert outputs[0][0].finish_reason == "error"
