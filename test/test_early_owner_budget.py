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
    a = c.job("a", count=10)
    b = c.job("b", count=10, offset=5)
    assert a["budget"]["used"] == 15 * 8
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
    a, b = c.job("a", count=40), c.job("b", count=40)
    c.message(a, "owner_answer", state="READY", hit=0)
    c.message(b, "owner_answer", state="READY", hit=0)
    assert a["budget"]["used"] == 0
    assert c.job("bounded", count=1)["status"] == "FAILED"
    c.drain(a)
    assert c.job("next", count=40)["owner"] == a["owner"]
    c.drain(b)


def test_capacity_rejection_precedes_reserve_and_explains_byte_limit(caplog):
    c = Cluster()
    c.job("a", count=40)
    with caplog.at_level(logging.INFO):
        rejected = c.job("distinct", count=40, offset=40)
    assert rejected["status"] == "FAILED"
    assert not any(m["op"] == "owner_reserve" and m.get("key") == "distinct" for _, m in c.out)
    assert "reason=domain_bytes" in caplog.text
    assert "wasted_read_bytes=0" in caplog.text
    c.job("shared", count=40)
    with caplog.at_level(logging.INFO):
        assert c.job("work-limit", count=40)["status"] == "FAILED"
    assert "reason=owner_work_limits" in caplog.text


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
