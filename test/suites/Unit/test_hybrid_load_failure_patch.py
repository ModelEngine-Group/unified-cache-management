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
from collections import defaultdict
from dataclasses import dataclass, field
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


def test_missing_failure_policy_uses_multi_group_fail_path():
    obj = scheduler({"failed": ([10], [20]), "healthy": ([30], [40])})
    del obj.recompute_kv_load_failures

    assert obj._handle_invalid_blocks({20}, {}) == {"failed"}
    obj.kv_cache_manager.evict_blocks.assert_called_once_with({10, 20})
    obj.original.assert_not_called()


def test_missing_failure_policy_preserves_single_group_handler():
    obj = scheduler({"req": ([10],)}, groups=1)
    del obj.recompute_kv_load_failures

    assert obj._handle_invalid_blocks({10}, {}) == {"native"}
    obj.original.assert_called_once_with({10}, {})


def real_block_pool(ref):
    source_dir = os.environ.get("VLLM_SOURCE_DIR")
    if not source_dir:
        pytest.skip("set VLLM_SOURCE_DIR for real BlockPool checks")
    source = subprocess.check_output(
        [
            "git",
            "-C",
            source_dir,
            "show",
            f"refs/tags/{ref}:vllm/v1/core/block_pool.py",
        ],
        text=True,
        encoding="utf-8",
    )

    class Block(NS):
        def reset_hash(self):
            self.block_hash = None

    namespace = {
        "KVCacheBlock": Block,
        "make_block_hash_with_group_id": lambda value, group: (value, group),
    }
    module = ast.parse("from __future__ import annotations")
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef) and node.name == "BlockHashToBlockMap":
            module.body.append(node)
        if isinstance(node, ast.ClassDef) and node.name == "BlockPool":
            node.body = [
                n
                for n in node.body
                if getattr(n, "name", None)
                in {
                    "free_blocks",
                    "evict_blocks",
                    "get_cached_block",
                    "_maybe_evict_cached_block",
                    "_remove_cached_block_hashes",
                    "_emit_block_removed_events",
                }
            ]
            module.body.append(node)
    exec(compile(module, f"{ref}/block_pool.py", "exec"), namespace)
    pool = namespace["BlockPool"]()
    pool.blocks = [
        Block(
            block_id=i, block_hash=(bytes([i]), 0), ref_cnt=1, is_null=i == 0, pool=None
        )
        for i in range(5)
    ]
    pool.null_block = pool.blocks[0]
    pool.enable_caching = True
    pool.enable_kv_cache_events = False
    pool.metrics_collector = None
    pool.free_block_queue = Mock()
    pool.cached_block_hashes_by_block = {}
    pool.cached_block_hash_to_block = namespace["BlockHashToBlockMap"]()
    for block in pool.blocks[1:]:
        pool.cached_block_hash_to_block.insert(block.block_hash, block)

    return pool


@pytest.mark.parametrize("ref", ["v0.23.0", "v0.30.0", "v0.32.0"])
def test_sync_failure_invalidates_real_prefix_cache(ref):
    pool = real_block_pool(ref)
    obj = scheduler({"failed": ([1, 2], [3]), "healthy": ([4], [])})
    obj.kv_cache_manager.block_pool = pool
    obj.kv_cache_manager.evict_blocks = pool.evict_blocks
    assert all(pool.get_cached_block(bytes([i]), [0]) for i in (1, 2, 3, 4))
    assert obj._handle_invalid_blocks({3}, {"failed": 1, "healthy": 1}) == {"failed"}
    # Native request finishing drops ownership after the hash entries are evicted.
    pool.free_blocks(pool.blocks[1:4])
    assert all(pool.get_cached_block(bytes([i]), [0]) is None for i in (1, 2, 3))
    assert pool.get_cached_block(bytes([4]), [0]) == [pool.blocks[4]]


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


def connector_plan(kind):
    """Use the production metadata schemas, without importing accelerator code."""
    namespace = {"__name__": __name__, "dataclass": dataclass, "field": field}
    module = ast.parse("from __future__ import annotations")
    for filename, names in (
        ("ucm_connector.py", {"RequestDispatchMeta", "UCMConnectorMetadata"}),
        ("hla_connector.py", {"HLARequestDispatchMeta"}),
        ("hma_connector.py", {"FAWARequestDispatchMeta", "UCMFAWAConnectorMetadata"}),
    ):
        path = ROOT / "ucm/integration/vllm" / filename
        for node in ast.parse(path.read_text(encoding="utf-8-sig")).body:
            if isinstance(node, ast.ClassDef) and node.name in names:
                if node.name.endswith("ConnectorMetadata"):
                    node.bases = []
                module.body.append(node)
    exec(compile(module, "connector_metadata", "exec"), namespace)
    if kind.endswith("hla"):
        plan = namespace["HLARequestDispatchMeta"](
            load_block_ids=([b"attention", b"state"], [1, 3]),
            dump_block_ids=([b"new"], [2]),
            load_group_ids=[0, 1],
            dump_group_ids=[0],
        )
        metadata = namespace["UCMConnectorMetadata"](request_meta={"failed": plan})
        invalid = {3}  # State-only failure; group 0 must not hide it.
    else:
        plan = namespace["FAWARequestDispatchMeta"](
            load_keys=[b"prefix"],
            load_vllm_block_ids=([1], [3]),
            dump_keys=[b"new"],
            dump_vllm_block_ids=([2], []),
        )
        metadata = namespace["UCMFAWAConnectorMetadata"](request_meta={"failed": plan})
        invalid = {3}
    if kind.startswith("multi_"):
        # MultiConnector's envelope has child metadata, not request_meta.
        metadata = NS(metadata=(metadata,))
    return metadata, invalid


@pytest.mark.parametrize("kind", ["hla", "fawa", "multi_hla", "multi_fawa"])
def test_normal_schedule_preserves_connector_plans(kind):
    metadata, _ = connector_plan(kind)

    class Scheduler:
        def schedule(self):
            return NS(kv_connector_metadata=metadata)

        def update_from_output(self, step, output):
            return output

        def _update_requests_with_invalid_blocks(self, *args):
            return "native updater"

        def _handle_invalid_blocks(self, *args):
            return set()

    originals = {
        name: getattr(Scheduler, name)
        for name in (
            "schedule",
            "update_from_output",
            "_update_requests_with_invalid_blocks",
        )
    }
    obj = scheduler({"failed": ([1, 2], [3])}, cls=Scheduler)
    assert obj.schedule().kv_connector_metadata is metadata
    for name, original in originals.items():
        assert getattr(Scheduler, name) is original


@pytest.mark.parametrize("kind", ["hla", "fawa", "multi_hla", "multi_fawa"])
def test_real_v032_output_isolates_sync_failure_and_invalidates_cache(kind):
    """Run the whole native output method with stubbed model/request services."""
    pool = real_block_pool("v0.32.0")
    source = subprocess.check_output(
        [
            "git",
            "-C",
            os.environ["VLLM_SOURCE_DIR"],
            "show",
            "refs/tags/v0.32.0:vllm/v1/core/sched/scheduler.py",
        ],
        text=True,
        encoding="utf-8",
    )
    cls = next(
        n for n in ast.parse(source).body if getattr(n, "name", None) == "Scheduler"
    )
    cls.bases = []
    cls.body = [
        n
        for n in cls.body
        if getattr(n, "name", None)
        in {
            "update_from_output",
            "_handle_invalid_blocks",
            "_update_requests_with_invalid_blocks",
        }
    ]
    module = ast.parse("from __future__ import annotations")
    module.body.append(cls)
    namespace = dict(
        defaultdict=defaultdict,
        RequestStatus=STATUS,
        logger=Mock(),
        EngineCoreOutput=NS,
        EngineCoreOutputs=NS,
        remove_all=lambda requests, removed: [r for r in requests if r not in removed],
    )
    STATUS.FINISHED_ERROR = "error"
    exec(compile(module, "v0.32.0/scheduler.py", "exec"), namespace)
    obj = scheduler(
        {"failed": ([1, 2], [3]), "healthy": ([4], [])}, cls=namespace["Scheduler"]
    )
    obj.kv_cache_manager.block_pool = pool
    obj.kv_cache_manager.evict_blocks = pool.evict_blocks
    obj.kv_cache_manager.take_events = lambda: None

    class Request(NS):
        __hash__ = object.__hash__

        def is_finished(self):
            return self.status in ("error", "stop")

        def get_finished_reason(self):
            return self.status if self.is_finished() else None

        def take_events(self):
            return []

        def take_prefill_stats(self):
            return None

    obj.requests = {
        key: Request(
            request_id=key,
            status=STATUS.RUNNING,
            num_in_flight_tokens=1,
            num_stale_output_tokens=0,
            has_encoder_inputs=False,
            sampling_params=None,
            client_index=0,
            stop_reason=None,
            trace_headers=None,
            spec_decode_metrics=None,
            num_nans_in_logits=0,
        )
        for key in ("failed", "healthy")
    }
    obj.running = list(obj.requests.values())
    obj.defer_block_free = False
    obj.perf_metrics = None
    obj.aux_output_connector = None
    obj.return_sampling_mask = False
    obj.ec_connector = obj.connector = None
    obj.grammar_compile_error_reqs = set()
    obj.encoder_cache_mismatch_reqs = set()
    obj.finished_req_ids_dict = {}
    obj.make_stats = lambda *args, **kwargs: None
    obj.structured_output_manager = NS(accept_tokens=lambda *args: True)
    obj._handle_stopped_request = lambda req: True
    obj._update_from_kv_xfer_finished = Mock()

    def complete(request, tokens, **kwargs):
        request.status = "stop"
        return tokens, True

    def free_request(request):
        ids = (1, 2, 3) if request.request_id == "failed" else (4,)
        pool.free_blocks([pool.blocks[i] for i in ids])
        obj.requests.pop(request.request_id)
        return None, None

    def finish_requests(ids, status):
        requests = [obj.requests[key] for key in ids]
        for request in requests:
            request.status = status
            free_request(request)
            obj.running.remove(request)
        return requests

    obj._update_request_with_output = Mock(side_effect=complete)
    obj._free_request = free_request
    obj.finish_requests = Mock(side_effect=finish_requests)
    metadata, invalid = connector_plan(kind)
    step = NS(
        kv_connector_metadata=metadata,
        num_scheduled_tokens={"failed": 1, "healthy": 1},
        scheduled_spec_decode_tokens={},
    )
    output = NS(
        sampled_token_ids=[[101], [202]],
        req_id_to_index={"failed": 0, "healthy": 1},
        logprobs=None,
        prompt_logprobs_dict={},
        prompt_token_id_logprobs_dict={},
        pooler_output=None,
        num_nans_in_logits=None,
        ec_connector_output=None,
        cudagraph_stats=None,
        kv_connector_output=NS(
            invalid_block_ids=invalid, failed_recving=set(), kv_connector_stats=None
        ),
    )
    results = obj.update_from_output(step, output)[0].outputs
    by_id = {result.request_id: result for result in results}
    assert by_id["failed"].finish_reason == "error"
    assert by_id["failed"].new_token_ids == []
    assert by_id["healthy"].finish_reason == "stop"
    assert by_id["healthy"].new_token_ids == [202]
    obj.finish_requests.assert_called_once_with({"failed"}, "error")
    assert obj._update_request_with_output.call_count == 1
    assert not obj.requests and not obj.running
    assert all(pool.get_cached_block(bytes([i]), [0]) is None for i in (1, 2, 3))
    assert pool.get_cached_block(bytes([4]), [0]) == [pool.blocks[4]]


@pytest.mark.parametrize(
    "ref",
    [
        "v0.18.0",
        "v0.20.2",
        "v0.23.0",
        "v0.27.1",
        "v0.30.0",
        "v0.31.0",
        "v0.32.0",
    ],
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
