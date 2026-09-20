"""Broker safety/eligibility, with real generation and DP protocol messages."""
from test_early_owner_protocol import Cluster
from owner_protocol_source.early_capacity import ExecutionWindow


def test_four_slots_two_three_block_jobs_wait_without_partial_deadlock():
    w = ExecutionWindow(4)
    w.add("a", {1, 2, 3}); w.add("b", {4, 5, 6})
    assert w.acquire("a", {1, 2})
    assert not w.acquire("b", {4, 5})
    assert w.claims["b"].held == set()
    assert w.acquire("a", {3})
    w.complete("a")
    # One block may safely advance while all of a waits for delayed consumers.
    assert w.acquire("b", {4})
    assert not w.acquire("b", {5, 6})
    w.release("a")
    assert w.acquire("b", {5, 6})
    w.release("b")
    assert w.used == 0


def test_overlap_is_union_charged_and_parallel_completion_is_preserved():
    w = ExecutionWindow(4)
    w.add("a", {1, 2, 3}); w.add("b", {2, 3, 4})
    assert w.acquire("a", {1, 2}) and w.acquire("b", {2, 3})
    assert w.acquire("a", {3}) and w.acquire("b", {4})
    assert w.used == 4
    w.release("a")
    assert w.used == 3
    w.release("b")
    assert not w.used


def small_cluster():
    c = Cluster(dp=2)
    for cap in c.b.capabilities.values():
        cap["native"].update(nodes=6, reservable_nodes=4, shard_bytes=8, block_bytes=8)
    return c


def request(c, j, index, begin, count):
    c.message(j, "execution_request", batch=index, begin=begin, count=count)


def grants(c, j):
    return [m for p, m in c.out if m["op"] == "execution_grant" and m["key"] == j["key"]]


def test_cross_dp_capacity_returns_after_actual_drain_not_ready_or_one_consumer():
    c = small_cluster()
    a, b = c.job("a", 0, 3), c.job("b", 1, 3, offset=10)
    request(c, a, 0, 0, 2); request(c, b, 0, 0, 2)
    assert grants(c, a) and not grants(c, b)
    request(c, a, 1, 2, 1)
    c.ready(a)
    assert not grants(c, b)
    c.message(a, "consumed", "w0.0")
    assert not grants(c, b)
    c.drain(a)
    assert grants(c, b)[0]["count"] == 2
    request(c, b, 1, 2, 1)
    c.ready(b); c.drain(b)
    assert not b["domain"]["execution"].used


def test_short_confirmed_hit_not_candidate_size_controls_eligibility():
    c = small_cluster(); j = c.job(count=100)
    request(c, j, 0, 0, 2)
    assert not grants(c, j)
    assert j["domain"]["execution"].used == 0
    c.message(j, "lookup_complete", hit=2)
    assert grants(c, j)[0]["count"] == 2
    assert not grants(c, j)[0]["exceptional"]


def test_intrinsically_oversized_hit_has_explicit_safe_bound():
    c = small_cluster(); j = c.job(count=6)
    request(c, j, 0, 0, 6)
    assert not grants(c, j)
    c.message(j, "lookup_complete", hit=6)
    grant = grants(c, j)[0]
    assert grant["count"] == grant["end"] == 4 and grant["exceptional"]
    assert j["domain"]["execution"].used == 4


def test_late_wrong_generation_and_duplicate_permission_are_idempotent():
    c = small_cluster(); j = c.job(count=3)
    c.b.receive(j["owner"], dict(op="execution_request", key=j["key"], attempt="old", batch=0, begin=0, count=3))
    assert not grants(c, j)
    request(c, j, 0, 0, 3); request(c, j, 0, 0, 3)
    assert len(grants(c, j)) == 2
    assert j["domain"]["execution"].used == 3
    c.drain(j)
    request(c, j, 0, 0, 3)
    assert j["key"] not in c.b.jobs

class DeferredCapability:
    """Storage mock; C++ tests separately cover physical pins and I/O drain."""
    def __init__(self, hit, work, on_read=lambda: None):
        from test_early_batch_pipeline import Capability
        self.inner = Capability(work, submit=on_read)
        self.inner.hit = hit
        self.hit, self.work, self.activated = hit, work, False

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def limit(self, count):
        self.hit = self.inner.hit = count
        self.work = min(self.work, count)
        self.inner.work = self.work

    def activate(self):
        self.activated = True
        return dict(done=not self.work, total_hit_blocks=self.hit, remaining_l2_work=self.work)

    def step(self, available):
        assert self.activated
        return self.inner.step(available)


def test_worker_deferred_pipeline_reads_before_tail_query_and_fast_path_skips_data():
    import threading
    import time
    from test_early_owner_protocol import make_worker, blocks
    from test_early_batch_pipeline import wait
    c = Cluster()
    for cap in c.b.capabilities.values():
        cap["native"].update(nodes=1024, reservable_nodes=1024)
    workers = [make_worker(0), make_worker(1)]
    reading, tail, allow_tail, stop = [threading.Event() for _ in range(4)]
    errors, reservations = [], []
    lock = threading.Lock()
    def pump():
        outgoing, incoming = 0, [0, 0]
        try:
            while not stop.is_set():
                with lock:
                    while outgoing < len(c.out):
                        peer, message = c.out[outgoing]; outgoing += 1
                        if peer.startswith("w"):
                            workers[int(peer[-1])].receive(message)
                    for rank, worker in enumerate(workers):
                        while incoming[rank] < len(worker.control.messages):
                            message = worker.control.messages[incoming[rank]]; incoming[rank] += 1
                            c.b.receive(f"w0.{rank}", message)
                time.sleep(.001)
        except Exception as error:
            errors.append(error)
    def reserve(ids, *, deferred=False):
        assert deferred
        if reservations:
            tail.set()
            assert reading.wait(2), "first I/O still waits for whole-prefix lookup"
            assert allow_tail.wait(2)
        cap = DeferredCapability(len(ids), 1 if not reservations else 0, reading.set)
        r = dict(handle=len(reservations), total_hit_blocks=len(ids), remaining_l2_work=cap.work, batch=cap)
        reservations.append(r)
        return r
    workers[0].store.reserve_prefix.side_effect = reserve
    workers[1].store.reserve_prefix.side_effect = lambda ids, **kw: dict(
        handle=9, total_hit_blocks=len(ids), remaining_l2_work=0,
        batch=DeferredCapability(len(ids), 0))
    thread = threading.Thread(target=pump)
    thread.start()
    try:
        with lock:
            j = c.job("pipeline", count=96)
        assert tail.wait(2) and reading.wait(2)
        assert j["status"] == "PENDING"
        allow_tail.set()
        wait(lambda: j["status"] == "READY")
        assert j["hit"] == 96 and len(reservations) == 2
        assert reservations[1]["batch"].steps == 0
        with lock:
            c.b.receive(j["client"], dict(op="cancel", key=j["key"]))
        wait(lambda: j["key"] not in c.b.jobs)
        with lock:
            fast = c.job("fast", count=4)
        wait(lambda: fast["status"] == "READY")
        batch = workers[1].jobs["fast"].batches[0]
        assert batch.activated and batch.done and not batch.started
        assert batch.reservation["batch"].steps == 0
        with lock:
            c.b.receive(fast["client"], dict(op="cancel", key=fast["key"]))
        wait(lambda: fast["key"] not in c.b.jobs)
        assert not errors
    finally:
        allow_tail.set()
        stop.set(); thread.join(3)
        for worker in workers:
            # Only test-owned jobs on failed assertions; production uses the broker retire handshake.
            with worker.condition:
                for job in worker.jobs.values():
                    worker._release(job)
                worker.jobs.clear()
            worker.close()


def test_worker_capacity_wait_yields_and_event_resumes_without_shortening():
    import time
    from test_early_owner_protocol import make_worker, blocks
    from test_early_batch_pipeline import wait
    worker = make_worker(0)
    blocked = DeferredCapability(3, 3)
    ordinary = DeferredCapability(1, 1)
    state = {"free": False, "epoch": 0}
    original_step = blocked.step
    def step(available):
        if not state["free"]:
            return dict(done=False, advanced=False, can_submit=False, waiting_capacity=True,
                        wait_epoch=state["epoch"], inflight_bytes=0, remaining_l2_work=3)
        return original_step(available)
    blocked.step = step
    caps = iter([blocked, ordinary])
    def reserve(ids, **kwargs):
        cap = next(caps)
        return dict(handle=cap.hit, total_hit_blocks=cap.hit, remaining_l2_work=cap.work, batch=cap)
    worker.store.reserve_prefix.side_effect = reserve
    try:
        for key, count, grant in [("wait", 3, 1), ("other", 1, 2)]:
            worker.receive(dict(op="owner_reserve", key=key, attempt="a-" + key, grant=grant,
                                owner="w0.0", blocks=blocks(count), io_quota=2, deferred=True))
            wait(lambda: any(m["op"] == "execution_request" and m["key"] == key
                             for m in worker.control.messages))
            worker.receive(dict(op="execution_grant", key=key, attempt="a-" + key,
                                batch=0, count=count, end=count, exceptional=False))
        wait(lambda: worker.jobs["wait"].batches[0].waiting_capacity)
        wait(lambda: worker.jobs["other"].ready)
        assert not worker.jobs["wait"].ready
        # One notification opportunity, no timer-based allocator retry.
        with worker.condition:
            state.update(free=True, epoch=1)
            worker.release_serial += 1
            worker.condition.notify_all()
        start = time.monotonic()
        wait(lambda: worker.jobs["wait"].ready)
        assert time.monotonic() - start < .5
        reply = next(m for m in worker.control.messages if m["op"] == "owner_answer" and m["key"] == "wait")
        assert reply["hit"] == 3 and not reply["capacity_truncated"]
    finally:
        with worker.condition:
            for job in worker.jobs.values():
                worker._release(job)
            worker.jobs.clear()
        worker.close()


def test_release_before_lookup_pause_publication_resumes_once():
    from types import SimpleNamespace
    from test_early_owner_protocol import make_worker, early, blocks
    from test_early_batch_pipeline import wait
    worker = make_worker(0)
    observations = iter([True, False])
    class RacingBatch:
        @property
        def waiting_capacity(self):
            return next(observations, False)
        done = True
        started = False
        discarded = False
        reservation = dict(total_hit_blocks=32, handle=1)
    job = early.WorkerJob("pause", blocks(96), deferred=True,
                          lookup_cursor=32, lookup_count=64, batches=[RacingBatch()])
    worker.jobs[job.key] = job
    worker.store.reserve_prefix.side_effect = lambda ids, **kwargs: dict(
        handle=2, total_hit_blocks=0, remaining_l2_work=0, batch=DeferredCapability(0, 0))
    try:
        # First capacity observation says wait, release occurs before finally
        # publishes the continuation. No later notification is necessary.
        worker._reserve(job)
        wait(lambda: job.lookup_finished)
        worker.store.reserve_prefix.assert_called_once()
        assert not job.lookup_paused
    finally:
        with worker.condition:
            worker._release(job)
            worker.jobs.clear()
        worker.close()
