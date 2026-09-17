"""Native-allocation owner results; no capacity approval ledger."""
import logging
import pytest
from test_early_owner_protocol import Cluster


def test_long_candidates_and_overlapping_owners_need_no_capacity_approval():
    c = Cluster(dp=2)
    a, b = c.job("a", 0, 200), c.job("b", 1, 200)
    assert a["domain"] is b["domain"]
    assert "claims" not in a and "used" not in a["domain"]
    c.message(a, "owner_answer", state="READY", hit=2, capacity_truncated=True)
    c.message(b, "owner_answer", state="READY", hit=1, capacity_truncated=False)
    assert a["status"] == b["status"] == "READY"
    assert a["hit"] == 2 and a["capacity_truncated"]
    assert not any(m["op"] in ("confirm_batch", "batch_grant") for _, m in c.out)
    c.drain(a)
    assert b["domain"]["owners"][b["owner"]]["jobs"] == 1
    c.drain(b)


@pytest.mark.parametrize("hit", [-1, 201, True, None])
def test_owner_rejects_invalid_prefix_result(hit):
    c = Cluster()
    job = c.job(count=200)
    c.message(job, "owner_answer", state="READY", hit=hit, capacity_truncated=True)
    assert job["retiring"]
    assert any(m.get("state") == "FAILED" for _, m in c.out)


def test_capacity_zero_is_ready_without_requery_and_late_result_is_ignored():
    c = Cluster()
    job = c.job(count=200)
    c.message(job, "owner_answer", state="READY", hit=0, capacity_truncated=True)
    assert job["status"] == "READY" and job["hit"] == 0
    c.message(job, "owner_answer", state="READY", hit=200)
    assert job["hit"] == 0
    c.drain(job)


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
