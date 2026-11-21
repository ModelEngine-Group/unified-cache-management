"""Summarize a completed P/D benchmark from pd_request_timing lines in a proxy log.

Run after the benchmark client has finished and before starting the next round.
Print one summary per backend and append it to the measured log. With --archive,
move the log with its summary to a new file on the same filesystem. Start the
proxy with --proxy-log to let it reopen the active path on its next write.
"""

import argparse
import fcntl
import json
import math
import os
import re
import sys
import time
from array import array
from datetime import datetime
from pathlib import Path

TIMING = b"pd_request_timing "
PREFILL = b"PD prefill_response request_id="
FIELD = re.compile(r'(?:^| )([a-z_0-9]+)=("(?:\\.|[^"\\])*"(?= |$)|\S+)')


class BenchmarkRun:
    def __init__(self):
        self.first_start_ns = self.last_end_ns = None
        self.requests = self.successful = self.timed = 0
        self.token_samples = self.known_tokens = 0
        self.latencies = {name: array("d") for name in ("kv_handoff", "ttft", "tpot")}

    def add(self, row):
        self.requests += 1
        if "proxy_start_ns" in row and "proxy_end_ns" in row:
            start, end = int(row["proxy_start_ns"]), int(row["proxy_end_ns"])
            if end < start:
                raise ValueError("request end precedes start")
            self.first_start_ns = (
                start
                if self.first_start_ns is None
                else min(self.first_start_ns, start)
            )
            self.last_end_ns = (
                end if self.last_end_ns is None else max(self.last_end_ns, end)
            )
            self.timed += 1
        if row["status"] != "ok":
            return
        self.successful += 1
        if row["output_tokens"] != "NA":
            tokens = int(row["output_tokens"])
            if tokens < 0:
                raise ValueError("negative output token count")
            self.known_tokens += tokens
            self.token_samples += 1
        for name, samples in self.latencies.items():
            if name == "kv_handoff" and row["kv_status"] != "observed":
                continue
            raw = row[f"{name}_ms"]
            if raw != "NA":
                value = float(raw)
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"invalid {name}_ms: {raw}")
                samples.append(value)

    def summary(self, backend, duration_s=None):
        seconds = duration_s
        if seconds is None and self.timed and self.timed == self.requests:
            seconds = (self.last_end_ns - self.first_start_ns) / 1e9
        tokens = self.known_tokens if self.token_samples == self.successful else None
        fields = {
            "backend": backend,
            "requests": self.requests,
            "successful": self.successful,
            "failed": self.requests - self.successful,
            "duration_source": (
                "client"
                if duration_s is not None
                else ("proxy" if seconds is not None else None)
            ),
            "duration_s": seconds,
            "qps": self.successful / seconds if seconds else None,
            "output_tokens": tokens,
            "output_tps": tokens / seconds if seconds and tokens is not None else None,
            "output_tokens_n": self.token_samples,
            "known_output_tokens": self.known_tokens,
            "window_n": self.timed,
        }
        for name, samples in self.latencies.items():
            values = sorted(samples)
            fields[f"{name}_n"] = len(values)
            for p in (50, 95, 99):
                quantile = None
                if values:
                    position = (len(values) - 1) * p / 100
                    lo, hi = math.floor(position), math.ceil(position)
                    quantile = values[lo] + (values[hi] - values[lo]) * (position - lo)
                fields[f"{name}_p{p}_ms"] = quantile
        return "pd_benchmark_summary " + " ".join(
            f"{key}={format_value(value)}" for key, value in fields.items()
        )


def format_value(value):
    if value is None:
        return "NA"
    return f"{value:.6f}" if isinstance(value, float) else str(value)


def read_round(log, expected, wait_seconds):
    """Consume each complete record once; late KV joins may follow the HTTP response."""
    runs, seen, pending = {}, set(), set()
    partial = b""
    deadline = None
    lineno = 0
    while True:
        chunk = log.readline()
        if chunk:
            partial += chunk
            if not partial.endswith(b"\n"):
                continue
            line, partial = partial, b""
            lineno += 1
            if (
                line.startswith(b"pd_proxy_log_gap ")
                or b"P/D timing queue overflow: observations dropped" in line
            ):
                raise ValueError(
                    f"{log.name}:{lineno}: proxy reported lost log records; round coverage is unknown"
                )
            if TIMING in line:
                try:
                    row = dict(
                        FIELD.findall(line.partition(TIMING)[2].decode().rstrip())
                    )
                    rid = row["request_id"]
                    # format_field JSON-quotes values containing whitespace.
                    if rid.startswith('"') and rid.endswith('"'):
                        decoded = json.loads(rid)
                        if any(c.isspace() for c in decoded):
                            rid = decoded
                    key = row["backend"], rid
                    if key in seen:
                        raise ValueError(f"duplicate timing row for {key}")
                    seen.add(key)
                    pending.discard(rid)
                    if key[0] not in runs:
                        runs[key[0]] = BenchmarkRun()
                    runs[key[0]].add(row)
                except (KeyError, ValueError) as exc:
                    raise ValueError(f"{log.name}:{lineno}: {exc}") from exc
            elif PREFILL in line:
                rid = line.partition(PREFILL)[2].rpartition(b" prefiller=")[0].decode()
                if not any((backend, rid) in seen for backend in runs):
                    pending.add(rid)
            continue

        if expected is not None and len(seen) > expected:
            raise ValueError(
                f"expected {expected} requests, found {len(seen)}; check the round boundary"
            )
        if (
            seen
            and not partial
            and (expected is not None or not pending)
            and (expected is None or len(seen) == expected)
        ):
            return runs
        if not seen and not pending and not partial and expected is None:
            raise ValueError("no pd_request_timing records in the log")
        if deadline is None:
            deadline = time.monotonic() + wait_seconds
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError(
                f"incomplete round: rows={len(seen)}, expected={expected}, pending_prefills={len(pending)}"
            )
        time.sleep(min(0.1, remaining))


def summarize(path, archive, expected, wait_seconds, duration_s):
    path = path.resolve(strict=True)
    # Keep the same inode locked and open for atomic append, even after rename.
    with open(
        path, "r+b", opener=lambda name, flags: os.open(name, flags | os.O_APPEND)
    ) as log:
        # Serialize independent summarizers. This advisory lock is never acquired by the proxy.
        fcntl.flock(log, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runs = read_round(log, expected, wait_seconds)
        if duration_s is not None and len(runs) != 1:
            raise ValueError("--duration-s needs a single backend in the log")
        lines = [run.summary(name, duration_s) for name, run in sorted(runs.items())]
        if not os.path.samestat(os.fstat(log.fileno()), path.stat()):
            raise ValueError(f"{path} rotated while reading; use the matching archive")
        if archive == "":
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            archive = path.with_name(f"{path.stem}.{stamp}{path.suffix}")
        if archive is not None:
            archive = Path(archive)
            # Reserve the name exclusively. rename then moves the live inode atomically.
            with archive.open("xb"):
                pass
            try:
                path.rename(archive)
            except OSError:
                archive.unlink(missing_ok=True)
                raise
        result = ("\n".join(lines) + "\n").encode()
        size = log.seek(0, os.SEEK_END)
        log.seek(max(0, size - len(result)))
        if log.read() != result:
            log.write(result)
            log.flush()
    return lines, archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="Proxy log for this benchmark round.")
    parser.add_argument(
        "--archive",
        nargs="?",
        const="",
        metavar="PATH",
        help="Rotate after summarizing; default to a timestamped sibling. Never overwrite an archive.",
    )
    parser.add_argument(
        "--expected-requests",
        type=int,
        metavar="N",
        help="Wait for exactly N logical request rows (including failures).",
    )
    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=60,
        help="Maximum wait at EOF for outstanding rows.",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        help="Use the client's measured round duration, e.g. for older logs.",
    )
    args = parser.parse_args()
    if args.expected_requests is not None and args.expected_requests <= 0:
        parser.error("--expected-requests must be positive")
    if not math.isfinite(args.wait_seconds) or args.wait_seconds < 0:
        parser.error("--wait-seconds must be nonnegative and finite")
    if args.duration_s is not None and (
        not math.isfinite(args.duration_s) or args.duration_s <= 0
    ):
        parser.error("--duration-s must be positive and finite")
    try:
        lines, archive = summarize(
            args.log,
            args.archive,
            args.expected_requests,
            args.wait_seconds,
            args.duration_s,
        )
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")
    print(*lines, sep="\n")
    if archive is not None:
        print(f"Archived to {archive}", file=sys.stderr)


if __name__ == "__main__":
    main()
