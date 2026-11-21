import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "ucm/pd/summarize_benchmark.py"


def row(rid, start, end, tokens, kv, ttft, tpot, status="ok"):
    record = dict(
        request_id=rid,
        backend="mooncake",
        status=status,
        kv_status="observed" if kv != "NA" else "missing_decode",
        output_tokens=tokens,
        kv_handoff_ms=kv,
        ttft_ms=ttft,
        tpot_ms=tpot,
    )
    if start is not None:
        record.update(proxy_start_ns=start * 10**9, proxy_end_ns=end * 10**9)
    return (
        "INFO pd_request_timing "
        + " ".join(f"{k}={v}" for k, v in record.items())
        + "\n"
    )


def summarize(path, *args):
    return subprocess.run(
        [sys.executable, SCRIPT, path, *args], capture_output=True, text=True
    )


def fields(result):
    assert result.returncode == 0, result.stderr
    return dict(item.split("=") for item in result.stdout.strip().split()[1:])


def test_concurrent_window_and_successful_request_percentiles(tmp_path):
    log = tmp_path / "proxy.log"
    # The failed request ends last. A's KV join arrives after B's.
    log.write_text(
        row("b", 1, 4, 30, 30, 300, 40)
        + row("a", 0, 2, 10, 10, 100, 20)
        + row("c", 2, 5, "NA", "NA", 999, "NA", status="interrupted")
    )
    archive = tmp_path / "round1.log"
    result = summarize(log, "--expected-requests", "3", "--archive", archive)
    out = fields(result)
    assert out["successful"] == "2" and out["failed"] == "1"
    assert float(out["duration_s"]) == 5 and float(out["qps"]) == 0.4
    assert out["output_tokens"] == "40" and float(out["output_tps"]) == 8
    for name, expected in [
        ("kv_handoff", (20, 29, 29.8)),
        ("ttft", (200, 290, 298)),
        ("tpot", (30, 39, 39.8)),
    ]:
        assert out[f"{name}_n"] == "2"
        assert tuple(
            float(out[f"{name}_p{p}_ms"]) for p in (50, 95, 99)
        ) == pytest.approx(expected)
    assert not log.exists() and archive.read_text().endswith(result.stdout)
    assert fields(summarize(archive)) == out  # prior round summaries are ignored


def test_missing_values_and_legacy_clock_coverage(tmp_path):
    log = tmp_path / "proxy.log"
    log.write_text(
        row("a", None, None, "NA", "NA", "NA", "NA") + row("b", 1, 3, 1, 0, 8, "NA")
    )
    out = fields(summarize(log))
    assert out["qps"] == out["duration_s"] == out["output_tps"] == "NA"
    assert (
        out["output_tokens_n"] == out["known_output_tokens"] == out["window_n"] == "1"
    )
    assert out["tpot_n"] == "0" and out["tpot_p95_ms"] == "NA"
    assert float(out["kv_handoff_p99_ms"]) == 0 and out["kv_handoff_n"] == "1"
    out = fields(summarize(log, "--duration-s", "4"))
    assert (
        float(out["qps"]) == 0.5
        and out["output_tps"] == "NA"
        and out["duration_source"] == "client"
    )


def test_incomplete_round_and_existing_archive_do_not_move_or_overwrite_logs(tmp_path):
    log, archive = tmp_path / "proxy.log", tmp_path / "round.log"
    log.write_text(row("a", 1, 2, 10, 5, 50, 9))
    original = log.read_bytes()
    args = ("--wait-seconds", "0", "--archive", archive)
    assert summarize(log, *args, "--expected-requests", "2").returncode
    assert not archive.exists() and log.read_bytes() == original
    archive.write_text("keep")
    assert summarize(log, *args, "--expected-requests", "1").returncode
    assert archive.read_text() == "keep" and log.read_bytes() == original
    with log.open("a") as out:
        out.write("pd_proxy_log_gap dropped=1\n")
    fresh_archive = tmp_path / "incomplete.log"
    result = summarize(log, "--archive", fresh_archive)
    assert result.returncode and "coverage is unknown" in result.stderr
    assert log.exists() and not fresh_archive.exists()
