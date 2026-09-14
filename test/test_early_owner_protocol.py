"""Owner protocol and real worker threads; storage and accelerator are mocked."""
import importlib
from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

package = ModuleType("owner_protocol_source")
package.__path__ = [str(Path(__file__).resolve().parents[1] / "ucm/integration/vllm")]
sys.modules[package.__name__] = package
owner = importlib.import_module(package.__name__ + ".early_owner")
consumer = importlib.import_module(package.__name__ + ".early_consumer")
early = importlib.import_module(package.__name__ + ".early_prefetch")
from test_early_batch_pipeline import Control, Capability, blocks, result, wait


def capability(dp=0, rank=0, domain="native-object"):
    return dict(version=1, ready=True, dp=dp, rank=rank, domain=domain, layout="same-layout",
                native=dict(nodes=132, reservable_nodes=112, shard_bytes=4, block_bytes=8),
                task_shards=2, dump_task_shards=6 if rank == 0 else 0,
                max_jobs=4, max_candidates=0, layers=[0, 1])


class Cluster:
    def __init__(self, dp=1, tp=2, capacity=True):
        self.out = []
        self.b = owner.OwnerBroker(lambda peer, msg: self.out.append((peer, msg)), tp, dp_size=dp)
        self.sequence = [0] * dp
        for d in range(dp):
            self.b.receive(f"s{d}", dict(op="hello", session="scheduler"))
            for r in range(tp):
                cap = capability(d, r) if capacity else {"reason": "private L1"}
                self.b.receive(f"w{d}.{r}", dict(op="hello", session="worker", capability=cap))
        self.out.clear()

    def job(self, key="g1", dp=0, count=4):
        self.sequence[dp] += 1
        self.b.receive(f"s{dp}", dict(op="job", key=key, dp=dp, blocks=blocks(count),
                                     request="reusable-id", sequence=self.sequence[dp]))
        return self.b.jobs[key]

    def message(self, job, op, peer=None, **kwargs):
        self.b.receive(peer or job["owner"], dict(op=op, key=job["key"], attempt=job["attempt"], **kwargs))

    def ready(self, job):
        self.message(job, "owner_answer", state="READY", hit=len(job["blocks"]))

    def drain(self, job):
        self.b.receive(job["client"], dict(op="finish", key=job["key"]))
        for peer in self.b.workers(job["dp"]):
            self.message(job, "quiesced", peer)
        self.message(job, "owner_drained", read_bytes=0, wasted_read_bytes=0)


def test_single_owner_round_robin_and_dp_isolation():
    c = Cluster(dp=2)
    jobs = [c.job(str(i), i // 2) for i in range(4)]
    assert [j["owner"] for j in jobs] == ["w0.0", "w0.1", "w1.0", "w1.1"]
    for job in jobs:
        grants = [(p, m) for p, m in c.out if m.get("key") == job["key"] and m["op"] in ("observe", "owner_reserve")]
        assert {p for p, _ in grants} == c.b.workers(job["dp"])
        assert sum(m["op"] == "owner_reserve" for _, m in grants) == 1
    assert len({j["attempt"] for j in jobs}) == 4
    assert len(c.b.domains) == 1  # DP-local requests, one physical budget


@pytest.mark.parametrize("change", ["missing", "private", "layout", "domain", "membership", "version"])
def test_unverified_handshake_uses_unchanged_broadcast(change):
    c = Cluster()
    caps = c.b.capabilities
    if change == "missing":
        caps.pop("w0.1")
    else:
        caps["w0.1"].update({"private": {"ready": False}, "layout": {"layout": "other"},
            "domain": {"domain": "other-object"}, "membership": {"dp": 1}, "version": {"version": 0}}[change])
    job = c.job()
    assert "owner" not in job
    assert len([m for _, m in c.out if m["op"] == "reserve"]) == 2


def test_shared_capacity_charged_before_work_and_returned_after_drain_only():
    c = Cluster(dp=2)
    a, b = c.job("a", 0, 30), c.job("b", 1, 20)
    assert a["budget"] is b["budget"] and a["budget"]["used"] == 400
    rejected = c.job("c", 0, 1)
    assert rejected["status"] == "FAILED" and "owner" not in rejected
    c.b.receive(a["client"], dict(op="cancel", key="a"))
    c.message(a, "quiesced", "w0.0")
    c.message(a, "owner_drained")  # other consumer not quiescent
    assert a["budget"]["used"] == 400
    c.message(a, "quiesced", "w0.1")
    assert a["budget"]["used"] == 400
    c.message(a, "owner_drained")
    assert b["budget"]["used"] == 160
    c.message(a, "owner_drained")  # replay cannot decrement twice
    c.drain(b)
    assert b["budget"]["used"] == 0
    assert c.job("new", count=50).get("owner")


def test_owner_job_and_candidate_limits_choose_another_owner_or_fallback():
    c = Cluster()
    for cap in c.b.capabilities.values():
        cap.update(max_jobs=1, max_candidates=4)
    assert c.job("a")["owner"] == "w0.0"
    assert c.job("b")["owner"] == "w0.1"
    assert c.job("c")["status"] == "FAILED"
    assert c.job("too-large", count=5)["status"] == "FAILED"


def test_ready_waits_for_owner_only_but_release_requires_every_consumer():
    c = Cluster()
    job = c.job()
    c.message(job, "owner_answer", "w0.1", state="READY", hit=4)
    assert job["status"] == "PENDING"
    c.ready(job)
    assert job["status"] == "READY"
    c.message(job, "consumed", "w0.0")
    c.message(job, "consumed", "w0.0")
    assert not job.get("retiring")
    c.message(job, "consumed", "w1.0")
    assert not job.get("retiring")
    c.message(job, "consumed", "w0.1")
    assert job["retiring"]
    assert not any(m["op"] == "release_owner" for _, m in c.out)
    c.drain(job)
    assert job["key"] not in c.b.jobs


def test_stale_attempt_reused_request_id_and_out_of_order_confirmation():
    c = Cluster()
    old = c.job("old")
    c.message(old, "consumed")  # PENDING cannot be consumed
    assert not old["consumers"]
    c.drain(old)
    new = c.job("new")
    c.ready(new)
    c.b.receive(new["owner"], dict(op="consumed", key="new", attempt=old["attempt"]))
    c.message(old, "quiesced")
    c.message(old, "owner_drained")
    assert not new["consumers"] and new["budget"]["used"] == new["charge"]
    c.b.receive("s1", dict(op="cancel", key="new"))
    assert new["status"] == "READY"


@pytest.mark.parametrize("failure", ["worker", "scheduler", "session", "capability", "load"])
def test_domain_loss_never_returns_credits_or_reassigns(failure):
    c = Cluster()
    job = c.job()
    c.ready(job)
    if failure in ("worker", "scheduler"):
        c.b.peers["w0.1" if failure == "worker" else "s0"] = time.monotonic() - 61
        c.b.expire()
    elif failure == "session":
        c.b.receive("w0.0", dict(op="hello", session="restarted"))
    elif failure == "capability":
        cap = dict(c.b.capabilities["w0.0"], domain="replaced-object")
        c.b.receive("w0.0", dict(op="hello", session="worker", capability=cap))
    else:
        c.b.receive("w0.1", dict(op="domain_fault", reason="failed native Wait"))
    assert c.b.fenced
    c.drain(job)
    assert job["budget"]["used"] == job["charge"]
    assert any(m["op"] == "domain_lost" for _, m in c.out)


def ticket(key="g", expected=None, grant=1):
    return dict(key=key, attempt="a-" + key, owner="w0.0", grant=grant,
                expected=blocks(2) if expected is None else expected)


def start_load(observer, request, ids, shard):
    token = consumer.load_scope.set(observer.scope([request]))
    try:
        return observer.begin(np.frombuffer(b"".join(ids), dtype=np.uint8), np.full(len(ids), shard))
    finally:
        consumer.load_scope.reset(token)


def test_layers_chunks_hbm_subset_and_two_requests_with_same_prefix():
    sent, faults = [], []
    observer = consumer.Consumer("w0.1", sent.append, faults.append)
    a, b = ticket(expected=blocks(3)[1:]), ticket("other", expected=blocks(3)[1:])
    observer.bind({"r": a, "same-prefix": b}, [0, 1])
    for layer in [0, 1]:
        for block in a["expected"]:
            receipt = start_load(observer, "r", [block], layer)
            assert not sent  # bind and submit cannot release protection
            observer.complete(receipt, 100, True)
    assert [m["key"] for m in sent] == ["g"]
    assert not observer.generations[("other", "a-other")]["completed"]
    assert not faults


def test_retire_waits_for_accepted_tasks_then_rejects_late_loads_and_receipts():
    sent, faults = [], []
    observer = consumer.Consumer("w0.1", sent.append, faults.append)
    t = ticket()
    observer.bind({"r": t}, [0, 1])
    token = start_load(observer, "r", blocks(1), 0)
    observer.retire(t["key"], t["attempt"])
    assert not sent
    with pytest.raises(RuntimeError):
        start_load(observer, "r", blocks(1), 1)
    observer.complete(token, 1, True)
    observer.complete(token, 1, True)
    assert [m["op"] for m in sent] == ["quiesced"]
    observer.forget(t["key"], t["attempt"])
    assert not observer.generations
    assert not faults


def test_no_consumer_and_failed_wait_have_different_release_semantics():
    sent, faults = [], []
    observer = consumer.Consumer("w0.1", sent.append, faults.append)
    observer.bind({"empty": ticket(expected=[]), "r": ticket("needed")}, [0, 1])
    assert [m["op"] for m in sent] == ["consumed"]
    token = start_load(observer, "r", blocks(1), 0)
    observer.complete(token, 2, False)
    assert len(faults) == 1
    assert len(sent) == 1


def test_task_scope_wraps_same_native_consistency_and_preserves_errors():
    observer = consumer.Consumer("w0.1", Mock(), Mock())
    observer.bind({"r": ticket()}, [0])
    native = Mock()
    native.submit_load.side_effect = lambda *a, **kw: consumer.load_scope.get()
    wrapper = consumer.LoadConsistency(native, observer)
    assert wrapper.submit_load("store", {"r": blocks(1)}) == (("g", "a-g"),)
    assert not consumer.load_scope.get()
    assert wrapper.wait_load is native.wait_load
    native.submit_load.side_effect = RuntimeError("native error")
    with pytest.raises(RuntimeError, match="native error"):
        wrapper.submit_load("store", {"r": blocks(1)})
    assert not consumer.load_scope.get()


def make_worker(rank):
    store = Mock()
    connector = SimpleNamespace(store=store, tp_rank=rank, tp_size=2, is_mla=True,
        _rank_consistency=Mock(), unique_id="owner-test",
        _vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_index=0)))
    original = owner.worker_capability
    owner.worker_capability = lambda c: capability(rank=c.tp_rank)
    try:
        w = early.EarlyWorker(connector, Control)
    finally:
        owner.worker_capability = original
    handles = iter(range(100))
    def reserve(ids):
        r = result(next(handles), len(ids), 0)
        r["batch"].snapshot = lambda: dict(l2_read_bytes=0, completed_read_bytes=0)
        return r
    w.store.reserve_prefix.side_effect = reserve
    return w


def test_actual_workers_only_owner_reserves_and_handoff_does_not_release():
    workers = [make_worker(0), make_worker(1)]
    try:
        for rank, w in enumerate(workers):
            w.receive(dict(op="owner_reserve" if rank == 0 else "observe", key="g", attempt="a-g",
                           grant=1, owner="w0.0", blocks=blocks(96), request="r", io_quota=8))
        wait(lambda: workers[0].jobs["g"].ready)
        assert workers[0].store.reserve_prefix.call_count == 2
        workers[1].store.reserve_prefix.assert_not_called()
        meta = SimpleNamespace(early_keys={"r": "g"}, early_owners={"r": ticket(expected=blocks(96))},
                               request_meta={"r": SimpleNamespace(load_block_ids=(blocks(96), []))})
        for w in workers:
            w.bind(meta)
            w.loaded(["r"])  # first native last-layer hook is not whole-generation release
            w.store.release_reservation.assert_not_called()
        workers[0].store.set_load_reservations.assert_called_with([0, 1])
        workers[1].store.set_load_reservations.assert_called_with([])
        for w in workers:
            w.receive(dict(op="retire", key="g", attempt="a-g"))
        workers[0].store.release_reservation.assert_not_called()
        workers[0].receive(dict(op="release_owner", key="g", attempt="a-g"))
        wait(lambda: any(m["op"] == "owner_drained" for m in workers[0].control.messages))
        assert workers[0].store.release_reservation.call_count == 2
        for w in workers:
            w.receive(dict(op="forget", key="g", attempt="a-g"))
            w.bind(meta)  # delayed decode metadata is harmless; new native load is forbidden
            with pytest.raises(RuntimeError):
                start_load(w.owner_protocol.consumer, "r", blocks(1), 0)
            assert not w.owner_protocol.consumer.generations
    finally:
        for w in workers:
            w.close()


def _transport_worker(endpoint, dp, rank, commands, events):
    """Independent process, real ZMQ; accelerator completion is explicit mock."""
    import queue
    import threading
    from owner_protocol_source.early_control import Control as RealControl
    connected = threading.Event()
    holder = {}
    def receive(message):
        events.put((f"w{dp}.{rank}", message))
        if message["op"] == "owner_reserve":
            connected.wait(5)
            holder["control"].send(dict(op="owner_answer", key=message["key"], attempt=message["attempt"],
                                        state="READY", hit=len(message["blocks"])))
        elif message["op"] == "retire":
            holder["control"].send(dict(op="quiesced", key=message["key"], attempt=message["attempt"]))
        elif message["op"] == "release_owner":
            holder["control"].send(dict(op="owner_drained", key=message["key"], attempt=message["attempt"]))
    control = RealControl(endpoint, f"w{dp}.{rank}", receive,
                          hello={"capability": capability(dp, rank)})
    holder["control"] = control
    connected.set()
    try:
        while True:
            try:
                message = commands.get(timeout=10)
            except queue.Empty:
                return
            if message is None:
                return
            control.send(message)
    finally:
        control.close()


def test_real_zmq_owner_dispatch_four_worker_processes_two_dp(tmp_path):
    import multiprocessing as mp
    import queue
    import uuid
    from owner_protocol_source.early_control import Control as RealControl
    ctx = mp.get_context("spawn")
    events = ctx.Queue()
    endpoint = "ipc://" + str(tmp_path / (uuid.uuid4().hex + ".sock"))
    received = [[], []]
    schedulers = [RealControl(endpoint, "s0", received[0].append, dict(tp_size=2, dp_size=2)),
                  RealControl(endpoint, "s1", received[1].append)]
    pipes = [ctx.Queue() for _ in range(4)]
    processes = [ctx.Process(target=_transport_worker, args=(endpoint, i // 2, i % 2, pipes[i], events))
                 for i in range(4)]
    try:
        for process in processes:
            process.start()
        # Give hello packets time to reach the real ROUTER before submitting.
        # An early probe takes legacy fallback if incomplete; it is terminated
        # before the next probe, so it cannot contaminate capacity accounting.
        deadline = time.monotonic() + 10
        ready = False
        sequence = 0
        while time.monotonic() < deadline and not ready:
            sequence += 1
            key = "handshake-" + str(sequence)
            schedulers[0].send(dict(op="job", dp=0, key=key, request=key,
                                   blocks=blocks(1), sequence=sequence))
            time.sleep(.15)
            assigned = [m for m in received[0] if m["op"] == "owner_assigned" and m["key"] == key]
            schedulers[0].send(dict(op="cancel", key=key))
            ready = bool(assigned)
        assert ready, "native capability handshake not completed"
        for dp in range(2):
            schedulers[dp].send(dict(op="job", dp=dp, key=f"target-{dp}", request="same-request-id",
                                    blocks=blocks(2), sequence=sequence + 1))
        wait(lambda: all(any(m["op"] == "result" and m.get("state") == "READY" and
                            m["key"] == f"target-{dp}" for m in received[dp]) for dp in range(2)))
        all_events = []
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            try:
                all_events.append(events.get(timeout=.1))
            except queue.Empty:
                if sum(m["op"] in ("observe", "owner_reserve") and m["key"].startswith("target-")
                       for _, m in all_events) == 4:
                    break
        for dp in range(2):
            target = [(p, m) for p, m in all_events if m.get("key") == f"target-{dp}" and
                      m["op"] in ("observe", "owner_reserve")]
            assert {p for p, _ in target} == {f"w{dp}.0", f"w{dp}.1"}
            assert sum(m["op"] == "owner_reserve" for _, m in target) == 1
            for peer, message in target:
                pipes[dp*2 + int(peer[-1])].put(dict(op="consumed", key=message["key"], attempt=message["attempt"]))
        forgotten = set()
        deadline = time.monotonic() + 3
        while len(forgotten) < 4 and time.monotonic() < deadline:
            try:
                peer, message = events.get(timeout=.1)
                if message["op"] == "forget" and message["key"].startswith("target-"):
                    forgotten.add(peer)
            except queue.Empty:
                pass
        assert len(forgotten) == 4
    finally:
        for pipe in pipes:
            pipe.put(None)
        for process in processes:
            process.join(12)
            if process.is_alive():
                process.terminate()  # only this test's child, after failure
                process.join(2)
            assert process.exitcode == 0
        for scheduler in schedulers:
            scheduler.close()


def test_reused_request_id_native_finish_cannot_retire_new_generation():
    w = make_worker(1)
    try:
        w.receive(dict(op="observe", key="new", attempt="a-new", grant=2, owner="w0.0",
                       blocks=blocks(2), request="reused", io_quota=8))
        t = ticket("new", grant=2)
        w.bind(SimpleNamespace(early_keys={"reused": "new"}, early_owners={"reused": t},
                              request_meta={"reused": SimpleNamespace(load_block_ids=(blocks(2), []))}))
        w.finished(["reused"])  # old callback lacks generation; cannot authorize retirement
        assert not w.owner_protocol.consumer.generations[("new", "a-new")]["stopped"]
        w.receive(dict(op="retire", key="new", attempt="a-new"))
        w.receive(dict(op="forget", key="new", attempt="a-new"))
    finally:
        w.close()


def test_owner_cancel_drains_accepted_io_before_budget_ack():
    import threading
    w = make_worker(0)
    gate = threading.Event()
    cap = Capability(4, gate=gate)
    cap.snapshot = lambda: dict(l2_read_bytes=cap.reads, completed_read_bytes=cap.reads - cap.pending)
    r = result(9, 4, 4)
    r["batch"] = cap
    w.store.reserve_prefix.side_effect = lambda ids: r
    w.store.release_reservation.side_effect = lambda handle: setattr(cap, "cancelled", True)
    try:
        w.receive(dict(op="owner_reserve", key="cancel", attempt="a-cancel", grant=1,
                       owner="w0.0", blocks=blocks(4), request="r", io_quota=1))
        wait(lambda: cap.pending == 1)
        w.receive(dict(op="retire", key="cancel", attempt="a-cancel", reason="cancel"))
        w.receive(dict(op="release_owner", key="cancel", attempt="a-cancel"))
        time.sleep(.02)
        assert cap.protected_bytes and not any(m["op"] == "owner_drained" for m in w.control.messages)
        before = cap.reads
        gate.set()
        wait(lambda: any(m["op"] == "owner_drained" for m in w.control.messages))
        assert cap.reads == before and cap.drained.is_set()
        receipt = next(m for m in w.control.messages if m["op"] == "owner_drained")
        assert receipt["read_bytes"] == receipt["wasted_read_upper_bound_bytes"] == before
        w.receive(dict(op="forget", key="cancel", attempt="a-cancel"))
    finally:
        gate.set()
        w.close()


def test_normal_shutdown_uses_acknowledged_drain_instead_of_fencing():
    import threading
    w = make_worker(0)
    closer = None
    try:
        w.receive(dict(op="owner_reserve", key="shutdown", attempt="a-shutdown", grant=1,
                       owner="w0.0", blocks=blocks(2), request="r", io_quota=1))
        wait(lambda: w.jobs["shutdown"].ready)
        closer = threading.Thread(target=w.close)
        closer.start()
        wait(lambda: any(m["op"] == "worker_retire" for m in w.control.messages))
        w.store.release_reservation.assert_not_called()
        w.receive(dict(op="retire", key="shutdown", attempt="a-shutdown", reason="worker shutdown"))
        w.receive(dict(op="release_owner", key="shutdown", attempt="a-shutdown"))
        wait(lambda: any(m["op"] == "owner_drained" for m in w.control.messages))
        w.receive(dict(op="forget", key="shutdown", attempt="a-shutdown"))
        closer.join(2)
        assert not closer.is_alive() and w.owner_protocol.fatal is None
    finally:
        if closer is None:
            w.close()
        else:
            closer.join(3)


@pytest.mark.parametrize("field,value,ready", [("is_mla", False, False), ("cp_world_size", 2, False),
    ("layer_ids", [0, 2], False), ("is_mla", True, True)])
def test_worker_capability_checks_actual_native_layout_and_configuration(field, value, ready):
    info = dict(backing="verified-by-native-provider", nodes=120, reservable_nodes=100,
                shard_bytes=4, block_bytes=8)
    store = SimpleNamespace(config=dict(share_buffer_enable=True, shard_size=4, block_size=8,
                                        tensor_size_list=[4]), domain_info=lambda: info)
    c = SimpleNamespace(store=store, launch_config={}, is_mla=True, cp_world_size=1,
        layer_ids=[0, 1], use_layerwise=True, unique_id="native", hash_block_size=4, block_size=4,
        blocks_per_chunk=1, generate_hash=lambda *a: [b"a" * 16], _seed=b"seed", tp_rank=1, tp_size=2,
        _vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_index=0),
            model_config=SimpleNamespace(max_model_len=32), scheduler_config=SimpleNamespace(max_num_seqs=3)))
    setattr(c, field, value)
    cap = owner.worker_capability(c)
    assert cap.get("ready", False) is ready
    if ready:
        assert (cap["dp"], cap["rank"], cap["native"]) == (0, 1, info)
        assert (cap["task_shards"], cap["dump_task_shards"]) == (8, 0)
        c.tp_rank = 0
        assert owner.worker_capability(c)["dump_task_shards"] == 24
        c.launch_config["enable_early_owner"] = False
        assert not owner.worker_capability(c).get("ready", False)


def test_control_loss_fences_even_rank_without_local_owner_jobs_after_domain_activation():
    w = make_worker(1)
    w.receive(dict(op="domain_active"))
    w.receive(dict(op="lost", reason="transport unavailable"))
    assert w.owner_protocol.fatal
    with pytest.raises(RuntimeError):
        w.owner_protocol.consumer.begin(np.zeros(0, dtype=np.uint8), [])
    w.close()


def test_unsupported_path_preserves_broker_ready_invalidation_on_peer_loss():
    c = Cluster(capacity=False)
    job = c.job()
    for peer in c.b.workers(0):
        c.b.receive(peer, dict(op="answer", key=job["key"], state="READY", hit=4))
    assert job["status"] == "READY"
    c.b.peers["w0.1"] = time.monotonic() - 61
    c.b.expire()
    assert job["status"] == "FAILED"
    assert any(m["op"] == "result" and m["state"] == "FAILED" for _, m in c.out)


def test_fenced_domain_rejects_late_hello_and_new_job_explicitly():
    c = Cluster()
    c.job()
    c.b.poison("unrecoverable native references")
    c.out.clear()
    c.b.receive("s0", dict(op="hello", session="replacement-scheduler"))
    c.b.receive("s0", dict(op="job", key="late-new-job", dp=0, blocks=blocks(2), sequence=10))
    assert len([m for p, m in c.out if p == "s0" and m["op"] == "domain_lost"]) == 2
    assert "late-new-job" not in c.b.jobs


def test_worker_capability_rejects_unrepresented_pipeline_parallel_consumers():
    info = dict(backing="native-backing", nodes=120, reservable_nodes=100, shard_bytes=4, block_bytes=8)
    c = SimpleNamespace(store=SimpleNamespace(config=dict(share_buffer_enable=True), domain_info=lambda: info),
        launch_config={}, is_mla=True, cp_world_size=1,
        _vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(pipeline_parallel_size=2)))
    assert "PP consumer cohort" in owner.worker_capability(c)["reason"]


def test_unscheduled_cancel_forget_clears_request_mapping_without_touching_reused_id():
    w = make_worker(1)
    try:
        for grant, key in [(1, "old"), (2, "new")]:
            w.receive(dict(op="observe", key=key, attempt="a-" + key, grant=grant, owner="w0.0",
                           request="reused", io_quota=8))
        w.receive(dict(op="retire", key="old", attempt="a-old"))
        w.receive(dict(op="forget", key="old", attempt="a-old"))
        w.finished(["reused"])
        assert w.request_keys == {"reused": "new"}
        w.receive(dict(op="retire", key="new", attempt="a-new"))
        w.receive(dict(op="forget", key="new", attempt="a-new"))
        assert not w.request_keys and not w.jobs
    finally:
        w.close()


@pytest.mark.parametrize("nodes,reservable,load,dump", [(132, 112, 2, 6), (132, 132, 16, 6), (8, 8, 4, 8)])
def test_domain_budget_leaves_full_native_partial_task_headroom(nodes, reservable, load, dump):
    c = Cluster(dp=2)
    for cap in c.b.capabilities.values():
        cap["native"].update(nodes=nodes, reservable_nodes=reservable)
        cap.update(task_shards=load, dump_task_shards=dump if cap["rank"] == 0 else 0)
    expected_slots = max(0, min(reservable - 2 * dump, nodes - 4 * load - 2 * dump))
    job = c.job(count=max(1, expected_slots // 2))
    budget = next(iter(c.b.domains.values()))
    assert budget["limit"] == expected_slots * 4
    assert budget["native_load_headroom_bytes"] == 4 * load * 4
    assert budget["native_dump_headroom_bytes"] == 2 * dump * 4
    assert budget["io_quota"] * 4 <= budget["io_limit"] <= budget["limit"]
    if expected_slots < 2:
        assert job["status"] == "FAILED" and "owner" not in job
        assert not any(m["op"] in ("reserve", "owner_reserve") for _, m in c.out)
    else:
        assert job.get("owner")
        assert c.job("overflow", count=1)["status"] == "FAILED"
        c.drain(job)
        assert budget["used"] == 0


@pytest.mark.parametrize("operation,limit", [("load", 2), ("dump", 6), ("dump", 0)])
def test_unscoped_native_io_cannot_exceed_advertised_headroom(operation, limit):
    poison = Mock()
    state = consumer.Consumer("w0.0", Mock(), poison)
    # No owner job/request scope is needed: native fallback must obey the same
    # physical-domain bound while another request's owner is holding pins.
    state.max_task_shards = 2
    state.max_dump_task_shards = limit if operation == "dump" else 6
    state.check_task(operation, limit)
    poison.assert_not_called()
    with pytest.raises(RuntimeError, match="headroom"):
        if operation == "load":
            state.begin(np.zeros(0, dtype=np.uint8), [0] * (limit + 1))
        else:
            state.check_task(operation, limit + 1)
    poison.assert_called_once()
    with pytest.raises(RuntimeError, match="fenced"):
        state.check_task("dump", 0)
