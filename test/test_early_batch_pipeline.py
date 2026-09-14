"""Real worker threads/control callbacks, mocked storage; no SSD/NPU claims."""
import importlib
from pathlib import Path
import sys
import threading
import time
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

package = ModuleType("batch_pipeline_source")
package.__path__ = [str(Path(__file__).resolve().parents[1] / "ucm/integration/vllm")]
sys.modules[package.__name__] = package
early = importlib.import_module(package.__name__ + ".early_prefetch")


def wait(predicate):
    deadline = time.monotonic() + 3
    while not predicate():
        assert time.monotonic() < deadline, "bounded worker test wait expired"
        time.sleep(.002)


class Control:
    def __init__(self, *args, **kwargs):
        self.messages = []

    def send(self, message):
        self.messages.append(message)

    def close(self):
        pass


@pytest.fixture
def worker():
    store = Mock()
    connector = SimpleNamespace(store=store, tp_rank=0, tp_size=8, is_mla=True,
        unique_id="batch-unit", request_hasher=Mock(side_effect=lambda block: bytes(reversed(block))),
        _vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_index=0)))
    instance = early.EarlyWorker(connector, Control)
    yield instance
    instance.close()


def blocks(count):
    return [i.to_bytes(16, "little") for i in range(count)]


class Capability:
    shard_bytes = 1
    window_bytes = 2
    protection_limit_bytes = 10000
    protected_bytes = 100

    def __init__(self, work, submit=lambda: None, gate=None):
        self.work, self.submit, self.gate = work, submit, gate
        self.pending = self.reads = self.steps = 0
        self.cancelled = False
        self.drained = threading.Event()

    def step(self, available):
        self.steps += 1
        advanced = False
        if self.pending and (self.gate is None or self.gate.is_set()):
            self.work -= self.pending
            self.pending = 0
            advanced = True
        if self.cancelled and not self.pending:
            self.protected_bytes = 0
            self.drained.set()
            raise RuntimeError("cancelled after draining")
        if available and self.work > self.pending and not self.cancelled:
            self.submit()
            self.reads += 1
            if self.gate is None or self.gate.is_set():
                self.work -= 1
            else:
                self.pending += 1
            advanced = True
        done = self.work == 0
        return dict(done=done, advanced=advanced, inflight_bytes=self.pending,
                    can_submit=self.work > self.pending and not self.cancelled,
                    remaining_l2_work=self.work, l2_read_bytes=self.reads,
                    l2_read_shards=self.reads, coalesced_shards=0)


def result(handle, hit, work):
    return dict(handle=handle, total_hit_blocks=hit, remaining_l2_work=work,
                batch=Capability(work))


def test_first_batch_reads_before_next_reserve_finishes(worker):
    reading = threading.Event()
    second_entered = threading.Event()
    allow_second = threading.Event()
    calls = []

    def reserve(ids):
        calls.append(ids)
        if len(calls) == 2:
            second_entered.set()
            assert reading.wait(2), "reserve still precedes all reads"
            assert allow_second.wait(2)
        reservation = result(len(calls), len(ids), len(ids))
        reservation["batch"].submit = reading.set
        return reservation

    worker.store.reserve_prefix.side_effect = reserve
    worker.receive(dict(op="reserve", key="overlap", request="r", blocks=blocks(96)))
    try:
        assert second_entered.wait(1)
        assert reading.wait(1)
        assert not worker.jobs["overlap"].ready  # no premature whole-request READY
        assert not worker.control.messages
    finally:
        allow_second.set()
    wait(lambda: worker.jobs["overlap"].ready)
    assert [len(ids) for ids in calls] == [32, 64]
    assert worker.control.messages[-1]["hit"] == 96
    worker.bind(SimpleNamespace(early_keys={"r": "overlap"}, request_meta={
        "r": SimpleNamespace(load_block_ids=([b"a" * 16], []))}))
    worker.store.set_load_reservations.assert_called_with([1, 2])
    worker.loaded({"r"})
    assert [call.args[0] for call in worker.store.release_reservation.call_args_list] == [1, 2]


def test_all_l1_and_short_prefix_never_use_data_queue(worker):
    worker.store.reserve_prefix.side_effect = [result(1, 32, 0), result(2, 7, 0)]
    worker.receive(dict(op="reserve", key="short", blocks=blocks(512)))
    wait(lambda: worker.jobs["short"].ready)
    assert worker.control.messages[-1]["hit"] == 39
    assert worker.store.reserve_prefix.call_count == 2
    worker.store.prefetch_reserved.assert_not_called()
    assert not worker.preparing and not worker.selected


def test_late_reservation_after_cancel_is_released_without_more_candidates(worker):
    entered, finish = threading.Event(), threading.Event()

    def reserve(ids):
        entered.set()
        assert finish.wait(2)
        return result(7, len(ids), len(ids))

    worker.store.reserve_prefix.side_effect = reserve
    worker.receive(dict(op="reserve", key="cancel", blocks=blocks(512)))
    try:
        assert entered.wait(1)
        worker.receive(dict(op="release", key="cancel"))
    finally:
        finish.set()
    wait(lambda: worker.store.release_reservation.called)
    worker.store.release_reservation.assert_called_once_with(7)
    worker.store.prefetch_reserved.assert_not_called()
    assert not worker.control.messages


def test_later_lookup_failure_releases_earlier_batches_and_falls_back(worker):
    worker.store.reserve_prefix.side_effect = [result(1, 32, 0), RuntimeError("L2 failed")]
    worker.receive(dict(op="reserve", key="failed", blocks=blocks(96)))
    wait(lambda: worker.control.messages)
    assert worker.control.messages[-1]["state"] == "FAILED"
    worker.store.release_reservation.assert_called_once_with(1)
    assert not worker.jobs["failed"].ready


@pytest.mark.parametrize("mla, rank", [(True, 7), (False, 0), (False, 7)])
def test_each_block_identity_is_computed_once_across_batches(worker, mla, rank):
    worker.connector.is_mla, worker.connector.tp_rank = mla, rank
    handles = iter(range(1, 10))
    worker.store.reserve_prefix.side_effect = lambda ids: result(next(handles), len(ids), 0)
    message = dict(op="reserve", key="once", blocks=blocks(96))
    worker.receive(message)
    worker.receive(message)
    wait(lambda: worker.jobs["once"].ready)
    assert worker.store.reserve_prefix.call_count == 2
    assert worker.connector.request_hasher.call_count == (96 if not mla and rank else 0)
    submitted = [block for call in worker.store.reserve_prefix.call_args_list for block in call.args[0]]
    assert submitted == ([bytes(reversed(block)) for block in message["blocks"]]
                         if not mla and rank else message["blocks"])


def test_hrrn_service_estimate_is_remaining_shards_not_total_prefix():
    long = early.WorkerBatch(0, 0, result(1, 512, 1), entered=0)
    short = early.WorkerBatch(0, 0, result(2, 2, 4), entered=0)
    assert early.EarlyWorker.ratio(long, 10) == 11
    assert early.EarlyWorker.ratio(long, 10) > early.EarlyWorker.ratio(short, 10)


def enqueue(worker, key, handle, capability, age=0):
    reservation = result(handle, 32, capability.work)
    reservation["batch"] = capability
    batch = early.WorkerBatch(0, 0, reservation, entered=time.monotonic() - age)
    job = early.WorkerJob(key, [], batches=[batch], reserve_done=True)
    worker.jobs[key] = job
    worker.preparing.append((job, batch))
    return job, batch


def test_pending_long_read_does_not_block_another_request(worker):
    gate = threading.Event()
    submitted = []
    long = Capability(4, submit=lambda: submitted.append("long"), gate=gate)
    short = Capability(1, submit=lambda: submitted.append("short"))
    with worker.condition:
        a, _ = enqueue(worker, "long", 1, long, age=100)
        b, _ = enqueue(worker, "short", 2, short)
        worker.condition.notify_all()
    try:
        wait(lambda: b.ready)
        assert not a.ready
        assert submitted[:2] == ["long", "short"]
        assert long.pending + short.pending <= 2
    finally:
        gate.set()
    wait(lambda: a.ready)
    worker.store.prefetch_reserved.assert_not_called()  # no synchronous whole-job waits


def test_window_is_shared_and_shutdown_drains_without_new_submissions(worker):
    gates = [threading.Event(), threading.Event()]
    submitted = []
    capabilities = [Capability(4, submit=lambda i=i: submitted.append(i), gate=gates[i])
                    for i in range(2)]
    worker.store.release_reservation.side_effect = lambda handle: setattr(capabilities[handle - 1], "cancelled", True)
    with worker.condition:
        for i, cap in enumerate(capabilities):
            enqueue(worker, str(i), i + 1, cap, age=20 - i)
        worker.condition.notify_all()
    closer = None
    try:
        wait(lambda: sum(cap.pending for cap in capabilities) == 2)
        time.sleep(.02)
        assert len(submitted) == 2  # the window is per worker, not per request
        closer = threading.Thread(target=worker.close)
        closer.start()
        wait(lambda: all(cap.cancelled for cap in capabilities))
        assert closer.is_alive()
        assert all(cap.protected_bytes for cap in capabilities)
        assert len(submitted) == 2
    finally:
        for gate in gates:
            gate.set()
        if closer is not None:
            closer.join(2)
    assert closer is not None and not closer.is_alive()
    assert all(cap.drained.is_set() and cap.protected_bytes == 0 for cap in capabilities)
    assert len(submitted) == 2


def test_new_arrivals_cannot_keep_an_existing_request_out_of_later_rounds(worker):
    # Exercise the scheduling rule independently of I/O timing: one old long
    # request and newly arriving short requests participate in finite rounds.
    with worker.condition:
        _, old = enqueue(worker, "old", 1, Capability(100), age=20)
        _, first = enqueue(worker, "first", 2, Capability(1), age=10)
        remaining = set()
        worker._choose(remaining)
        key, batch = worker.selected.pop(0)
        remaining.discard(key.key)
        worker.preparing.append((key, batch))
        for i in range(10):
            enqueue(worker, "new" + str(i), i + 3, Capability(1), age=100)
        worker._choose(remaining)
        assert worker.selected[0][1] is old  # finish original round before arrivals
        worker.preparing.clear()
        worker.selected.clear()


def test_finish_and_load_failure_are_not_reported_as_completed_handoff(worker, monkeypatch):
    logger = Mock()
    monkeypatch.setattr(early, "log", logger)
    with worker.condition:
        for key, handle in [("finished", 1), ("failed", 2), ("loaded", 3)]:
            worker.jobs[key] = early.WorkerJob(key, [], batches=[
                early.WorkerBatch(0, 0, result(handle, 1, 0), done=True)])
            worker.request_keys[key] = key
    worker.finished(["finished"])
    worker.loaded(["failed"], reason="load_failure")
    worker.loaded(["loaded"])
    releases = [call.args for call in logger.info.call_args_list
                if call.args[0].startswith("UCM_EARLY released")]
    assert [args[3] for args in releases] == ["finish", "load_failure", "handoff"]
    assert not worker.jobs
    assert worker.store.release_reservation.call_count == 3
