"""Broker protocol/byte-accounting tests; native pin/I/O tests run separately."""
import logging

import pytest

from test_early_owner_protocol import Cluster


def test_two_owners_share_full_prefix_credit_until_last_drain():
    c = Cluster(dp=2)
    a = c.job("a", 0, 40)
    b = c.job("b", 1, 40)
    assert a["owner"] != b["owner"]
    assert a["budget"] is b["budget"]
    assert a["budget"]["used"] == 320  # 640 under per-request charging
    c.ready(a)
    c.ready(b)
    c.drain(a)
    assert b["budget"]["used"] == 320
    # Retried completion cannot release the surviving consumer's credits.
    c.message(a, "owner_drained")
    assert b["budget"]["used"] == 320
    c.drain(b)
    assert b["budget"]["used"] == 0
    assert not b["budget"]["claims"]


def test_short_ready_returns_only_unshared_tail_before_consumer_drain():
    c = Cluster()
    a = c.job("a", count=10, confirm=False)
    b = c.job("b", count=10, offset=5)
    assert a["budget"]["used"] == 10 * 8
    assert c.confirm(a, 4)
    c.message(a, "owner_answer", state="READY", hit=4)
    assert a["budget"]["used"] == 14 * 8
    assert a["charge"] == 4 * 8
    c.message(a, "owner_answer", state="READY", hit=0)  # duplicate/out of order
    assert a["budget"]["used"] == 14 * 8
    c.message(b, "owner_answer", state="FAILED", reason="NoSpace after I/O")
    assert b["retiring"] and b["budget"]["used"] == 14 * 8
    c.drain(b)
    assert a["budget"]["used"] == 4 * 8
    c.drain(a)
    assert a["budget"]["used"] == 0


def test_zero_hit_frees_memory_credit_but_preserves_owner_work_limit():
    c = Cluster()
    for cap in c.b.capabilities.values():
        cap["max_jobs"] = 1
    a, b = c.job("a", count=40, confirm=False), c.job("b", count=40, confirm=False)
    c.message(a, "owner_answer", state="READY", hit=0)
    c.message(b, "owner_answer", state="READY", hit=0)
    assert a["budget"]["used"] == 0
    assert c.job("bounded", count=1)["status"] == "FAILED"
    c.drain(a)
    assert c.job("next", count=40)["owner"] == a["owner"]
    c.drain(b)


def test_short_hit_large_candidates_are_not_rejected_before_lookup(caplog):
    c = Cluster()
    a = c.job("a", count=40)
    with caplog.at_level(logging.INFO):
        job = c.job("long-candidate", count=200, offset=40, confirm=False)
        assert job["owner"]
        assert job["budget"]["used"] == 320
        assert c.confirm(job, 2)
    assert job["budget"]["used"] == 336
    assert "hit=2 granted=True" in caplog.text
    assert not c.confirm(job, 20, begin=2, index=1)
    assert job["budget"]["used"] == 336
    c.message(job, "owner_answer", state="FAILED", reason="confirmed capacity denied")
    assert job["retiring"] and job["budget"]["used"] == 336
    c.drain(job)
    c.drain(a)
    assert job["budget"]["used"] == 0


def test_old_attempt_and_non_owner_cannot_shrink_current_claims():
    c = Cluster()
    old = c.job("old")
    c.drain(old)
    job = c.job("current", count=10)
    c.b.receive(job["owner"], dict(op="owner_answer", key=job["key"],
                attempt=old["attempt"], state="READY", hit=0))
    other = next(iter(c.b.workers(0) - {job["owner"]}))
    c.message(job, "owner_answer", other, state="READY", hit=0)
    assert job["budget"]["used"] == 80
    c.b.receive(job["client"], dict(op="cancel", key=job["key"]))
    c.message(job, "owner_answer", state="READY", hit=0)
    assert job["budget"]["used"] == 80
    c.drain(job)
    assert job["budget"]["used"] == 0


def test_separate_physical_domains_do_not_share_credit():
    c = Cluster(dp=2)
    for peer in c.b.workers(1):
        c.b.capabilities[peer]["domain"] = "other-native-object"
    a, b = c.job("a", 0, 40), c.job("b", 1, 40)
    assert a["budget"] is not b["budget"]
    assert a["budget"]["used"] == b["budget"]["used"] == 320
    c.drain(a)
    assert b["budget"]["used"] == 320
    c.drain(b)


@pytest.mark.parametrize("op", ["cancel", "finish"])
def test_shared_claim_is_retained_through_cancellation_and_partial_quiescence(op):
    c = Cluster()
    a, b = c.job("a", count=40), c.job("b", count=40)
    c.b.receive(a["client"], dict(op=op, key=a["key"]))
    c.message(a, "quiesced", "w0.0")
    c.message(a, "owner_drained")
    c.drain(b)
    assert a["budget"]["used"] == 320
    c.message(a, "quiesced", "w0.1")
    c.message(a, "owner_drained")
    assert a["budget"]["used"] == 0


def test_domain_fault_retains_unique_claims_even_with_late_drain():
    c = Cluster()
    a, b = c.job("a", count=40), c.job("b", count=40)
    c.b.receive("w0.1", dict(op="domain_fault", reason="unknown I/O drain"))
    c.drain(a)
    c.drain(b)
    assert c.b.fenced
    assert a["budget"]["used"] == 320
    assert all(count == 2 for count in a["budget"]["claims"].values())


def test_previous_protocol_version_uses_legacy_path():
    c = Cluster()
    c.b.capabilities["w0.1"]["version"] = 1
    job = c.job()
    assert "owner" not in job
    assert sum(m["op"] == "reserve" for _, m in c.out) == 2


@pytest.mark.parametrize("hit", [0, 2])
def test_worker_zero_hit_is_not_reported_as_l1_fast_path(hit, caplog):
    from test_early_owner_protocol import make_worker, blocks, result, wait
    w = make_worker(0)
    def reserve(ids):
        r = result(17, hit, 0)
        r["batch"].snapshot = lambda: dict(l2_read_bytes=0, completed_read_bytes=0)
        return r
    w.store.reserve_prefix.side_effect = reserve
    try:
        with caplog.at_level(logging.INFO):
            w.receive(dict(op="owner_reserve", key="g", attempt="a-g", grant=1,
                           owner="w0.0", blocks=blocks(2), request="r", io_quota=8))
            wait(lambda: any(m["op"] == "owner_answer" for m in w.control.messages))
            # The answer is enqueued before the diagnostic record.
            wait(lambda: "outcome=" in caplog.text)
        assert ("UCM_EARLY fast_path " in caplog.text) == bool(hit)
        assert ("outcome=l1_ready" if hit else "outcome=zero_hit") in caplog.text
        w.receive(dict(op="retire", key="g", attempt="a-g"))
        w.receive(dict(op="release_owner", key="g", attempt="a-g"))
        wait(lambda: any(m["op"] == "owner_drained" for m in w.control.messages))
        w.receive(dict(op="forget", key="g", attempt="a-g"))
    finally:
        w.close()


@pytest.mark.parametrize("level", [logging.INFO, logging.DEBUG])
def test_detailed_logging_does_not_change_native_completion_ack(level, caplog):
    from test_early_owner_protocol import consumer, ticket, blocks, start_load
    sent, faults = [], []
    observer = consumer.Consumer("w0.0", sent.append, faults.append)
    with caplog.at_level(level):
        observer.bind({"r": ticket()}, [0, 1])
        for layer in (0, 1):
            token = start_load(observer, "r", blocks(2), layer)
            observer.submitted(token, layer)
            observer.complete(token, layer, True)
    assert [m["op"] for m in sent] == ["consumed"]
    assert not faults
    assert ("native_load_submitted" in caplog.text) == (level == logging.DEBUG)
    assert ("native_load_completed" in caplog.text) == (level == logging.DEBUG)


def test_candidate_probe_holds_no_domain_credit_and_batch_grants_are_idempotent():
    c = Cluster()
    job = c.job(count=200, confirm=False)
    assert job["budget"]["used"] == 0 and not job["claims"]
    assert c.confirm(job, 2)
    assert c.confirm(job, 2)
    assert job["budget"]["used"] == 16
    assert c.confirm(job, 0, begin=2, index=1)
    c.message(job, "owner_answer", state="READY", hit=2)
    assert job["status"] == "READY"
    c.drain(job)
    assert job["budget"]["used"] == 0


def test_retired_or_stale_attempt_cannot_acquire_new_batch_credit():
    c = Cluster()
    job = c.job(count=200, confirm=False)
    assert c.confirm(job, 2)
    c.b.receive(job["owner"], dict(op="confirm_batch", key=job["key"], attempt="old", batch=1, begin=2, count=10))
    assert job["confirmed"] == 2
    c.b.receive(job["client"], dict(op="cancel", key=job["key"]))
    assert not c.confirm(job, 10, begin=2, index=1)
    assert job["budget"]["used"] == 16
    c.drain(job)
    assert job["budget"]["used"] == 0


def test_capacity_rejected_receipt_does_not_claim_success_or_release_other_h2d():
    from test_early_owner_protocol import consumer, ticket, blocks, start_load
    sent, faults = [], []
    observer = consumer.Consumer("w0.0", sent.append, faults.append)
    observer.bind({"r": ticket()}, [0, 1])
    active = start_load(observer, "r", blocks(1), 0)
    rejected = start_load(observer, "r", blocks(1), 1)
    observer.rejected(rejected)
    observer.rejected(rejected)
    assert [m["op"] for m in sent] == ["worker_retire"]
    observer.retire("g", "a-g")
    assert len(sent) == 1
    observer.complete(active, 3, True)
    assert [m["op"] for m in sent] == ["worker_retire", "quiesced"]
    assert not faults


def test_worker_does_not_submit_io_before_confirmed_credit_and_drains_denial():
    from test_early_owner_protocol import make_worker, blocks, result, wait
    w = make_worker(0)
    w.control.send = w.control.messages.append  # real grant remains asynchronous
    w.store.reserve_prefix.side_effect = lambda ids: result(23, 2, 2)
    try:
        w.receive(dict(op="owner_reserve", key="g", attempt="a-g", grant=1,
                       owner="w0.0", blocks=blocks(200), request="r", io_quota=8))
        wait(lambda: any(m["op"] == "confirm_batch" for m in w.control.messages))
        job = w.jobs["g"]
        assert len(job.batches) == 1 and job.batches[0].reservation["batch"].reads == 0
        w.receive(dict(op="batch_grant", key="g", attempt="old", batch=0, granted=True))
        assert not job.batch_grants
        w.receive(dict(op="batch_grant", key="g", attempt="a-g", batch=0, granted=False))
        wait(lambda: any(m["op"] == "owner_answer" for m in w.control.messages))
        answer = next(m for m in w.control.messages if m["op"] == "owner_answer")
        assert answer["state"] == "FAILED" and job.batches[0].reservation["batch"].reads == 0
        w.store.release_reservation.assert_called_with(23)
        # Mock completed/no-I/O snapshot; actual drain is covered in C++.
        job.batches[0].reservation["batch"].snapshot = lambda: dict(l2_read_bytes=0, completed_read_bytes=0)
        w.receive(dict(op="retire", key="g", attempt="a-g"))
        w.receive(dict(op="release_owner", key="g", attempt="a-g"))
        wait(lambda: any(m["op"] == "owner_drained" for m in w.control.messages))
        w.receive(dict(op="forget", key="g", attempt="a-g"))
    finally:
        w.close()
