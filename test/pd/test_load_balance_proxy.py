import asyncio
import csv
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from contextlib import ExitStack, contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

PD = Path(__file__).resolve().parents[2] / "ucm" / "pd"


def send_through_proxy(
    monkeypatch,
    backend,
    request_id="ucm-test",
    payload=None,
    reports=None,
    ucm_pd=False,
):
    # The proxy also runs standalone on a host without the UCM engine libraries.
    monkeypatch.syspath_prepend(str(PD))
    import ucm_pd_disaggregation_example as proxy

    argv = ["proxy", "--ucm-pd"] if ucm_pd else ["proxy"]
    if reports is not None:
        argv += [
            "--timing-backend",
            "ucm-async" if ucm_pd else "mooncake",
            "--prefiller-logs",
            "p.log",
            "--decoder-logs",
            "d.log",
        ]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(proxy, "global_args", proxy.parse_args())
    scheduler = proxy.SharedProxyScheduler([("prefill", 8000)], [("decode", 8000)])
    if reports is not None:
        scheduler._timing_collector = SimpleNamespace(
            register=lambda key: None, finish=lambda key, report: reports.append(report)
        )
    runtime = proxy.WorkerRuntime(scheduler)
    monkeypatch.setattr(proxy, "runtime", runtime)

    async def run():
        transport = httpx.MockTransport(backend)
        async with (
            httpx.AsyncClient(
                base_url="http://prefill", transport=transport
            ) as prefill,
            httpx.AsyncClient(base_url="http://decode", transport=transport) as decode,
            httpx.AsyncClient(
                base_url="http://proxy", transport=httpx.ASGITransport(app=proxy.app)
            ) as client,
        ):

            async def get_client(role, key):
                return prefill if role == proxy.ServerRole.PREFILL else decode

            monkeypatch.setattr(runtime, "get_client", get_client)
            body = payload or {"prompt": "hello", "max_tokens": 8}
            return await client.post(
                "/v1/chat/completions" if "messages" in body else "/v1/completions",
                json=body,
                headers={"X-Request-Id": request_id} if request_id else {},
            )

    return asyncio.run(run()), scheduler


def sse(choice):
    return (
        b"data: "
        + json.dumps({"choices": [choice]}, ensure_ascii=False).encode()
        + b"\r\n\r\n"
    )


def test_coalesced_usage_rewrite_preserves_other_sse_fields(monkeypatch):
    first = b": heartbeat\r\n\r\n" + sse({"text": "hello"})
    fields = b"id: 9\r\nevent: message\r\n"
    usage = b'data: {"choices":[],\r\ndata: "usage":{"completion_tokens":1}}\r\n\r\n'
    done = b"data: [DONE]\n\n"

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield first + fields + usage[:16]
            yield usage[16:] + done

    def backend(request):
        if request.url.host == "prefill":
            return httpx.Response(
                200, json={"usage": {"prompt_tokens_details": {"cached_tokens": 9}}}
            )
        return httpx.Response(200, stream=Stream())

    response, _ = send_through_proxy(
        monkeypatch, backend, payload={"prompt": "p", "stream": True}
    )
    response.raise_for_status()
    prefix = first + fields + b"data: "
    assert response.content.startswith(prefix)
    assert response.content.endswith(b"\r\n\r\n" + done)
    rewritten = json.loads(
        response.content.removeprefix(prefix).removesuffix(b"\r\n\r\n" + done)
    )
    assert rewritten["usage"] == {
        "completion_tokens": 1,
        "prompt_tokens_details": {"cached_tokens": 9},
    }


def test_first_token_is_logged_once_across_recompute(monkeypatch, caplog):
    caplog.set_level("INFO", logger="ucm_pd_disaggregation_example")
    replies = iter([{"text": "one"}, {"text": "two"}])
    closed = []

    class Stream(httpx.AsyncByteStream):
        async def aclose(self):
            closed.append(self)

        async def __aiter__(self):
            choice = next(replies)
            if choice["text"] == "one":
                output = sse(choice)
                yield sse({"delta": {"role": "assistant"}}) + output[:12]
                assert not any("decode_first_token" in msg for msg in caplog.messages)
                yield (
                    output[12:]
                    + sse({"text": "", "stop_reason": "recomputed"})
                    + sse({"text": "obsolete"})
                    + b"data: [DONE]\n\n"
                )
            else:
                yield sse(choice)
            yield b"data: [DONE]\n\n"

    def backend(request):
        if request.url.host == "prefill":
            if json.loads(request.content)["prompt"] != "p":
                assert len(closed) == 1
            return httpx.Response(200, json={})
        return httpx.Response(200, stream=Stream())

    reports = []
    response, _ = send_through_proxy(
        monkeypatch, backend, payload={"prompt": "p", "stream": True}, reports=reports
    )
    assert "two" in response.text
    assert "obsolete" not in response.text and "recomputed" not in response.text
    assert response.content.count(b"[DONE]") == 1
    assert sum("decode_first_token" in msg for msg in caplog.messages) == 1
    assert len(reports) == 1 and reports[0]["recomputes"] == 1
    assert reports[0]["status"] == "ok"
    assert reports[0]["tpot_ms"] is None and reports[0]["ttft_ms"] is not None


@pytest.mark.parametrize("ucm_pd", [False, True])
def test_recompute_uses_only_current_handoff(monkeypatch, ucm_pd):
    first_handoff = {"remote_block_ids": [7], "do_remote_prefill": False}
    p_replies = iter([{"kv_transfer_params": first_handoff}, {}])
    d_bodies = []

    def backend(request):
        if request.url.host == "prefill":
            return httpx.Response(200, json=next(p_replies))
        d_bodies.append(json.loads(request.content))
        stop = "recomputed" if len(d_bodies) == 1 else "stop"
        return httpx.Response(
            200,
            json={
                "choices": [{"text": "x", "stop_reason": stop}],
                "usage": {"completion_tokens": 1},
            },
        )

    body = {"prompt": "hello", "max_tokens": 8}
    if ucm_pd:
        body["kv_transfer_params"] = {"client_hint": "keep"}
    response, _ = send_through_proxy(monkeypatch, backend, payload=body, ucm_pd=ucm_pd)
    response.raise_for_status()
    first, second = d_bodies
    role = {"do_remote_prefill": True} if ucm_pd else {}
    assert first["kv_transfer_params"] == {**first_handoff, **role}
    if ucm_pd:
        assert second["kv_transfer_params"] == {
            "client_hint": "keep",
            "do_remote_prefill": True,
        }
    else:
        assert "kv_transfer_params" not in second


def test_nonstream_usage_and_retry_have_no_first_token(monkeypatch):
    attempts = []

    class JsonBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"choices":[{"text":"ok"}],'
            yield b'"usage":{"completion_tokens":2}}'

    def backend(request):
        if request.url.host == "prefill":
            return httpx.Response(200, json={"usage": {"prompt_tokens": 5}})
        attempts.append(1)
        if len(attempts) == 1:
            return httpx.Response(503, json={"error": "retry"})
        return httpx.Response(
            200, headers={"content-type": "application/json"}, stream=JsonBody()
        )

    reports = []
    response, _ = send_through_proxy(monkeypatch, backend, reports=reports)
    response.raise_for_status()
    assert response.json()["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    (row,) = reports
    assert row["status"] == "ok" and row["http_retries"] == 1
    assert (row["prompt_tokens"], row["output_tokens"], row["count_source"]) == (
        5,
        2,
        "usage",
    )
    assert (
        row["ttft_ms"] is row["tpot_ms"] is row["p_response_to_first_token_ms"] is None
    )
    assert (
        row["prefill_http_ms"] > 0
        and row["decode_http_ms"] > 0
        and row["output_tps"] > 0
    )


def test_failed_recompute_releases_both_reservations(monkeypatch):
    calls = []

    def backend(request):
        calls.append((request.url.host, request.headers["X-Request-Id"]))
        if len(calls) == 1:
            return httpx.Response(200, json={})
        if request.url.host == "decode":
            return httpx.Response(
                200, json={"choices": [{"stop_reason": "recomputed", "text": "x"}]}
            )
        assert json.loads(request.content)["prompt"] == "hellox"
        return httpx.Response(400, json={"error": {"message": "failed after reassign"}})

    _, scheduler = send_through_proxy(
        monkeypatch, backend, request_id=None, ucm_pd=True
    )
    assert [host for host, _ in calls] == ["prefill", "decode", "prefill"]
    assert calls[0][1] and len({request_id for _, request_id in calls}) == 1
    assert scheduler.request_num == 0
    assert scheduler.prefillers["prefill:8000"].active_kv_cache == 0
    assert scheduler.decoders["decode:8000"].active_tokens == 0


def test_per_request_rates_use_tokens_and_token_arrivals(monkeypatch):
    monkeypatch.syspath_prepend(str(PD))
    import ucm_pd_disaggregation_example as proxy

    metric = proxy.RequestTiming(
        "/completions", {"request_id": "r", "stream": True}, 10
    )
    metric.prefill_response({"usage": {"prompt_tokens": 20}}, 10.1)
    metric.first_d_send = 10.15
    metric.observe({"choices": [{"token_ids": [10, 11]}]}, 10.25, "token_ids")
    metric.observe(
        {"choices": [{"token_ids": [12, 13, 14], "finish_reason": "stop"}]},
        10.75,
        "token_ids",
    )
    monkeypatch.setattr(proxy.time, "perf_counter", lambda: 11)
    row = metric.summary("ok")
    assert row["output_tokens"] == 5 and row["count_source"] == "token_ids"
    assert row["ttft_ms"] == 250 and row[
        "p_response_to_first_token_ms"
    ] == pytest.approx(150)
    assert row["tpot_ms"] == 125 and row["decode_tps"] == 8 and row["output_tps"] == 5
    metric.recomputes = 1
    assert metric.summary("ok")["tpot_ms"] is None


def test_text_events_without_usage_do_not_become_token_counts(monkeypatch):
    monkeypatch.syspath_prepend(str(PD))
    from ucm_pd_disaggregation_example import RequestTiming

    metric = RequestTiming("/completions", {"request_id": "r", "stream": True}, 1.0)
    metric.prefill_response({}, 1.05)
    metric.first_d_send = 1.1
    metric.observe({"choices": [], "usage": {"completion_tokens": 0}}, 1.12)
    metric.observe(
        {"choices": [{"text": "several words", "finish_reason": "stop"}]}, 1.2, "text"
    )
    row = metric.summary("ok")
    assert row["ttft_ms"] == pytest.approx(200)
    assert row["output_tokens"] is row["tpot_ms"] is row["output_tps"] is None
    metric.observe({"choices": [{"stop_reason": "recomputed"}]}, 1.3)
    assert metric.summary("ok")["status"] == "unresolved_recompute"


def append(path, line):
    with path.open("a") as out:
        print(line, file=out)


@contextmanager
def running_backend(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("mode", ["async", "sync", "legacy-sync"])
def test_comparison_joins_engine_ids_and_rejects_incomplete_handoffs(tmp_path, mode):
    sync = mode != "async"
    script = tmp_path / "ucm_pd_disaggregation_example.py"
    script.write_bytes((PD / script.name).read_bytes())
    mp, md, up, ud = (tmp_path / name for name in ("mp", "md", "up", "ud"))
    samples = ["ok", "missing", "retry", "failed"]
    mp.write_text(
        "".join(
            f"1790000000.010000 INFO Delaying free of 4 blocks for request cmpl-{s}-0-11223344\n"
            for s in samples
        )
    )
    md.write_text(
        "".join(
            f"1790000000.020000 DEBUG Finished recving KV transfer for request cmpl-{s}-0-55667788\n"
            for s in samples
        )
        + "ERROR Failed to transfer KV cache for request cmpl-failed-0-11223344: timeout\n"
    )
    up.write_text(
        "".join(
            f"ucm_pd_prefill_done request_id=cmpl-{s}-0-aabbccdd ts_ns=1790000000010000000\n"
            for s in samples
        )
        + "ucm_pd_prefill_done request_id=cmpl-retry-0-00ff00ff ts_ns=1790000000040000000\n"
    )
    if sync:
        count = " workers=2" if mode == "sync" else ""
        ud.write_text(
            "".join(
                f"ucm_pd_sync_load_done request_id=cmpl-{s}-0-ff112233 rank={r} ts_ns={1790000000020000000 + r * 4000000}{count}\n"
                for s in samples
                for r in ([0] if s == "missing" else [1, 0])
            )
        )
    else:
        ud.write_text(
            "".join(
                f"1790000000.024000 DEBUG Finished recving KV transfer for request cmpl-{s}-0-ff112233\n"
                for s in samples
                if s != "missing"
            )
        )
    args = [
        sys.executable,
        "-I",
        "-S",  # The copied script can compare logs with just the standard library.
        script,
        "compare-handoff",
        "--mooncake-p",
        mp,
        "--mooncake-d",
        md,
        "--ucm-p",
        up,
        "--ucm-d",
        ud,
    ]
    if mode == "sync":
        args.append("--ucm-sync")
    elif mode == "legacy-sync":
        args.extend(["--ucm-sync-ranks", "0", "1"])
    result = subprocess.run(args, capture_output=True, text=True, check=True)
    rows = {r["request_id"]: r for r in csv.DictReader(result.stdout.splitlines())}
    ucm_column = "ucm_sync_rank_ms" if sync else "ucm_scheduler_ms"
    assert rows["cmpl-ok-0"]["mooncake_scheduler_ms"] == "10.000"
    assert rows["cmpl-ok-0"][ucm_column] == "14.000"
    assert rows["cmpl-ok-0"]["ucm_minus_mooncake_ms"] == "4.000"
    assert rows["cmpl-missing-0"]["ucm_status"] == (
        "rank_set_mismatch" if sync else "missing_decode"
    )
    assert rows["cmpl-retry-0"]["ucm_status"] == "ambiguous_attempts"
    assert rows["cmpl-failed-0"]["mooncake_status"] == "failed_or_aborted"
    for key in ("missing", "retry", "failed"):
        assert rows[f"cmpl-{key}-0"]["ucm_minus_mooncake_ms"] == ""


def test_sync_timing_requires_every_worker_but_uses_only_actual_load_times(monkeypatch):
    monkeypatch.syspath_prepend(str(PD))
    import ucm_pd_disaggregation_example as proxy

    def result(rank, outcome, ns, workers):
        line = (
            f"ucm_pd_sync_load_{outcome} request_id=r rank={rank} ts_ns={ns}{workers}"
        )
        return proxy.parse_handoff_event(line, proxy.UCM_RANK)[1]

    start = [proxy.HandoffEvent("r", 1_000_000, None)]
    loaded = result(4, "done", 3_000_000, " workers=2")
    idle = result(5, "skipped", 8_000_000, " workers=2")
    assert proxy.handoff_duration(start, [loaded]) == (None, "rank_set_mismatch")
    assert proxy.handoff_duration(start, [idle, loaded]) == (2.0, "observed")
    assert proxy.handoff_duration(start, [idle, idle]) == (None, "repeated_completion")
    other_idle = result(4, "skipped", 3_000_000, " workers=2")
    assert proxy.handoff_duration(start, [other_idle, idle]) == (
        None,
        "no_external_load",
    )
    legacy = [result(4, "done", 3_000_000, "")]
    assert proxy.handoff_duration(start, legacy) == (None, "missing_worker_count")
    assert proxy.handoff_duration(start, legacy, {4}) == (2.0, "observed")


@pytest.mark.parametrize(
    "mode,workers", [("mooncake", 1), ("ucm-async", 1), ("ucm-sync", 2)]
)
def test_live_proxy_joins_late_engine_logs_across_rounds(tmp_path, mode, workers):
    p_log, d_log, proxy_log = (
        tmp_path / name for name in ("p.log", "d.log", "proxy.log")
    )
    p_log.touch()
    d_log.touch()
    p_ns = 1_789_948_800_000_000_000  # 2026-09-21 00:00:00 UTC

    def scheduler_event(ms, message):
        return json.dumps(
            {"asctime": f"2026-09-21 00:00:00,{ms:03}", "message": message}
        )

    class Engine(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            rid = "chatcmpl-" + self.headers["X-Request-Id"]
            if body["max_tokens"] == 1:
                assert body["kv_transfer_params"]["do_remote_decode"]
                key = rid + "-aaaabbbb"
                if mode == "mooncake":
                    append(
                        p_log,
                        scheduler_event(
                            0, f"Delaying free of 3 blocks for request {key}"
                        ),
                    )
                    reply = {"kv_transfer_params": {"remote_request_id": key}}
                else:
                    append(p_log, f"ucm_pd_prefill_done request_id={key} ts_ns={p_ns}")
                    reply = {}
                content = json.dumps(reply).encode()
                media_type = "application/json"
            else:
                if mode == "mooncake":
                    assert body["kv_transfer_params"] == {
                        "remote_request_id": rid + "-aaaabbbb"
                    }
                else:
                    assert body["kv_transfer_params"] == {"do_remote_prefill": True}
                    if mode == "ucm-sync":
                        append(
                            d_log,
                            f"ucm_pd_sync_load_done request_id={rid}-bbbbcccc rank=0 ts_ns={p_ns + 4_000_000} workers=2",
                        )
                content = b'data: {"choices":[{"delta":{"content":"X"},"token_ids":[1]}]}\n\ndata: [DONE]\n\n'
                media_type = "text/event-stream"
            self.send_response(200)
            self.send_header("Content-Type", media_type)
            self.end_headers()
            self.wfile.write(content)

    with ExitStack() as stack:
        engine_port = stack.enter_context(running_backend(Engine))
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        console = tmp_path / "console.log"
        cmd = [
            sys.executable,
            PD / "ucm_pd_disaggregation_example.py",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--workers",
            str(workers),
            "--prefiller-ports",
            str(engine_port),
            "--decoder-ports",
            str(engine_port),
            "--proxy-log",
            proxy_log,
            "--prefiller-logs",
            p_log,
            "--decoder-logs",
            d_log,
        ]
        if mode != "mooncake":
            config = tmp_path / "pd.yaml"
            async_load = "true" if mode == "ucm-async" else "false"
            config.write_text(f"use_request_async_load: {async_load}\n")
            cmd += ["--ucm-config", config]
        logfile = stack.enter_context(console.open("w"))
        proc = subprocess.Popen(
            cmd, stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            client = stack.enter_context(
                httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=4)
            )
            for _ in range(150):
                assert proc.poll() is None, console.read_text()
                try:
                    if client.get("/healthcheck").is_success:
                        break
                except httpx.ConnectError:
                    pass
                time.sleep(0.05)
            else:
                pytest.fail(console.read_text())

            archive = tmp_path / "round1.log"
            for rid in ("first", "second"):
                response = client.post(
                    "/v1/chat/completions",
                    headers={"X-Request-Id": rid},
                    json={
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": True,
                        "max_tokens": 8,
                    },
                )
                response.raise_for_status()
                assert b"[DONE]" in response.content
                # The last D event becomes visible only after HTTP has finished.
                key = f"chatcmpl-{rid}-bbbbcccc"
                if mode == "ucm-sync":
                    append(
                        d_log,
                        f"ucm_pd_sync_load_done request_id={key} rank=1 ts_ns={p_ns + 14_000_000} workers=2",
                    )
                else:
                    append(
                        d_log,
                        scheduler_event(
                            14, f"Finished recving KV transfer for request {key}"
                        ),
                    )

                for _ in range(100):
                    if (
                        proxy_log.exists()
                        and f"pd_request_timing request_id={rid} "
                        in proxy_log.read_text()
                    ):
                        break
                    time.sleep(0.05)
                else:
                    pytest.fail(f"No timing row for {rid}.\n{console.read_text()}")
                (line,) = [
                    s
                    for s in proxy_log.read_text().splitlines()
                    if "pd_request_timing " in s
                ]
                fields = dict(
                    item.split("=", 1)
                    for item in line.split("pd_request_timing ")[1].split()
                )
                assert fields["request_id"] == rid and fields["backend"] == mode
                assert (
                    fields["kv_handoff_ms"] == "14.000"
                    and fields["kv_status"] == "observed"
                )
                assert (
                    float(fields["p_response_to_first_token_ms"]) > 0
                    and float(fields["ttft_ms"]) > 0
                )

                if rid == "first":
                    subprocess.run(
                        [
                            sys.executable,
                            PD / "summarize_benchmark.py",
                            proxy_log,
                            "--expected-requests",
                            "1",
                            "--archive",
                            archive,
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    d_log.rename(tmp_path / "d.log.1")
                    d_log.touch()
            assert (
                "request_id=first " in archive.read_text()
                and "request_id=second " not in archive.read_text()
            )
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
