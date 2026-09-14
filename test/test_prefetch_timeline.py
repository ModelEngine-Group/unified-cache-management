from analyze_prefetch_timeline import elapsed, records, same_clock, summarize


def test_rank_and_reservation_ids_join_without_cross_process_clock_assumptions():
    text = """
UCM_EARLY submitted key=A request=req dp=0 ns=1000000
UCM_EARLY lookup_enqueued key=A rank=0 ns=2000000
UCM_EARLY lookup_start key=A rank=0 ns=9000000
UCM_EARLY batch_reserved key=A rank=0 batch=0 reservation=1 pid=10 start_ns=9000000 end_ns=11000000
UCM_EARLY batch_start key=A rank=0 batch=0 reservation=1 ns=13000000
UCM_EARLY batch_reserved key=A rank=0 batch=1 reservation=2 pid=10 start_ns=11000000 end_ns=30000000
UCM_EARLY batch_reserved key=B rank=1 batch=0 reservation=1 pid=11 start_ns=9000000 end_ns=12000000
UCM_EARLY stage=prefetch reservation=1 first_io_ns=15000000 last_io_ns=40000000 [10,19]
UCM_EARLY stage=reserve reservation=1 end_ns=11000000 [10,19]
UCM_EARLY stage=reserve reservation=2 end_ns=30000000 [10,20]
UCM_EARLY stage=reserve reservation=1 pin_ms=2 [11,20]
UCM_EARLY rank_ready key=A rank=0 ns=45000000
UCM_EARLY ready key=A dp=0 ns=50000000
UCM_EARLY handoff key=A dp=0 ns=59000000 native_scheduled_ns=55000000
"""
    rows = list(records(text))
    report = summarize(rows + rows)  # repeated log capture does not count twice
    rank = report["jobs"]["A"]["ranks"]["0"]
    assert rank["lookup_queue_ms"] == 7
    assert rank["batches"]["0"]["data_queue_ms"] == 2
    assert rank["first_io_lead_over_reserve_done_ms"] == 15
    assert report["jobs"]["A"]["submitted_to_all_ready_ms"] is None
    assert report["jobs"]["B"]["ranks"]["1"]["batches"]["0"]["stages"]["reserve"]["pin_ms"] == "2"
    verified = summarize(rows, cross_process=True)
    assert verified["jobs"]["A"]["submitted_to_all_ready_ms"] == 49
    assert verified["jobs"]["A"]["ready_to_native_scheduled_ms"] == 5
    assert not report["unmatched_stages"]


def test_joint_handoff_is_not_counted_once_per_consumer_and_missing_records_stay_missing():
    rows = records("""
UCM_EARLY batch_reserved key=A rank=0 batch=0 reservation=1 pid=10
UCM_EARLY batch_reserved key=B rank=0 batch=0 reservation=2 pid=10
UCM_EARLY stage=load_handles reservation=0 load_reservations=1,2 shards=78 handles=77 [10,10]
UCM_EARLY stage=reserve reservation=1 pin_ms=3 [12,12]
""")
    report = summarize(rows)
    assert len(report["joint_loads"]) == 1
    assert len(report["joint_loads"][0]["members"]) == 2
    assert report["joint_loads"][0]["native_get_fallback_shards"] == 1
    assert len(report["unmatched_stages"]) == 1
    assert elapsed(None, 4) is None
    assert elapsed(5, 4) is None


def test_clock_proof_rejects_missing_workers_or_different_namespaces():
    proof = dict(complete=True, expected_counts={role: 1 for role in ("client", "scheduler", "broker", "worker")},
                 processes=[dict(role=role, boot_id="boot", time_namespace="time")
                 for role in ("client", "scheduler", "broker", "worker")])
    assert same_clock(proof)
    proof["processes"][-1]["time_namespace"] = "other"
    assert not same_clock(proof)
    assert not same_clock({})
