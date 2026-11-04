from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terminal_view_metrics.collector import collect_loop, scrape_urls
from terminal_view_metrics.config import (
    apply_config_param_overrides,
    list_preset_configs,
    load_config,
    parse_duration_seconds,
    parse_time_ms,
    resolve_config_path,
)
from terminal_view_metrics.query import QueryEngine
from terminal_view_metrics.render import render_json, render_table
from terminal_view_metrics.snapshot import SnapshotQueryEngine
from terminal_view_metrics.storage import MetricsStore

DEFAULT_DB = "/tmp/ucm_metrics.db"
DEFAULT_PID_FILE = "/tmp/ucm_metrics.pid"
DEFAULT_LOG_FILE = "/tmp/terminal_metrics.log"
PRIVATE_WORKER_COMMAND = "__collect_worker"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == PRIVATE_WORKER_COMMAND:
        args = _build_worker_parser().parse_args(argv[1:])
        return int(_cmd_collect_worker(args))
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect and query Prometheus metrics in SQLite"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", help="Scrape once and show config values")
    _add_check_args(check)
    check.set_defaults(func=_cmd_check)

    start = subparsers.add_parser("start", help="Start background metrics collection")
    _add_collect_args(start)
    start.add_argument("--pid-file", default=DEFAULT_PID_FILE)
    start.add_argument("--log-file", default=DEFAULT_LOG_FILE)
    start.set_defaults(func=_cmd_start)

    stop = subparsers.add_parser("stop", help="Stop background metrics collection")
    stop.add_argument("--pid-file", default=DEFAULT_PID_FILE)
    stop.set_defaults(func=_cmd_stop)

    status = subparsers.add_parser("status", help="Show background collector status")
    status.add_argument("--pid-file", default=DEFAULT_PID_FILE)
    status.set_defaults(func=_cmd_status)

    query = subparsers.add_parser("query", help="Query collected metrics once")
    _add_query_args(query)
    query.add_argument("--format", choices=["table", "json"], default="table")
    query.set_defaults(func=_cmd_query)

    presets = subparsers.add_parser("list-configs", help="List bundled config presets")
    presets.set_defaults(func=_cmd_list_configs)

    clean = subparsers.add_parser("clean", help="Clear the metrics SQLite database")
    clean.add_argument("--db", default=DEFAULT_DB)
    clean.set_defaults(func=_cmd_clean)

    return parser


def _add_collect_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--url",
        action="append",
        required=True,
        help="Prometheus /metrics URL; can be repeated",
    )
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--interval", default="5s")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional scrape filter config; omit it to collect all metrics",
    )
    parser.add_argument("--retention", default="24h")
    parser.add_argument("--timeout", default="5s")


def _add_check_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--url",
        action="append",
        required=True,
        help="Prometheus /metrics URL; can be repeated",
    )
    parser.add_argument("--config", default="metrics_lite")
    _add_config_param_args(parser)
    parser.add_argument("--timeout", default="5s")
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Only check series with this Prometheus label value; can be repeated",
    )
    parser.add_argument("--format", choices=["table", "json"], default="table")
    parser.add_argument("--limit", type=int, default=None)


def _add_query_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--config", default="metrics_lite")
    _add_config_param_args(parser)
    parser.add_argument("--window", default="5m")
    parser.add_argument(
        "--aggr-by",
        default=None,
        help="Group the query window into smaller intervals, for example 1m",
    )
    parser.add_argument(
        "--start-time",
        default=None,
        help="Window start time: epoch seconds, epoch milliseconds, or local ISO time",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Only query series with this Prometheus label value; can be repeated",
    )
    parser.add_argument("--limit", type=int, default=None)


def _add_config_param_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a parameter declared by the selected config; can be repeated",
    )


def _build_worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    _add_collect_args(parser)
    return parser


def _cmd_collect_worker(args: argparse.Namespace) -> int:
    collect_loop(
        urls=args.url,
        db_path=args.db,
        interval_seconds=parse_duration_seconds(args.interval),
        config_path=resolve_config_path(args.config) if args.config else None,
        retention_seconds=parse_duration_seconds(args.retention),
        timeout_seconds=parse_duration_seconds(args.timeout),
    )
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    config = apply_config_param_overrides(
        load_config(resolve_config_path(args.config)),
        args.config_param,
    )
    samples, failures = scrape_urls(args.url, parse_duration_seconds(args.timeout))
    if failures:
        for url, exc in failures:
            print(f"scrape failed for {url}: {exc}", file=sys.stderr)
        return 1
    rows = SnapshotQueryEngine(samples).query_config(
        config,
        tag_filters=_parse_tag_filters(args.tag),
    )
    if args.format == "json":
        print(render_json(rows, args.limit))
    else:
        print(render_table(rows, args.limit))
    return 0


def _cmd_start(args: argparse.Namespace) -> int:
    pid_file = Path(args.pid_file)
    if pid_file.exists():
        pid = pid_file.read_text(encoding="utf-8").strip()
        if pid and _is_alive(int(pid)):
            print(f"collector already running: pid={pid}")
            return 0
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        PRIVATE_WORKER_COMMAND,
        "--db",
        args.db,
        "--interval",
        args.interval,
        "--retention",
        args.retention,
        "--timeout",
        args.timeout,
    ]
    for url in args.url:
        command.extend(["--url", url])
    if args.config:
        command.extend(["--config", str(resolve_config_path(args.config))])
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a", encoding="utf-8")
    kwargs = {
        "stdout": log_file,
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
    }
    if os.name == "nt":
        kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        )
    else:
        kwargs["start_new_session"] = True
    process = subprocess.Popen(command, **kwargs)
    pid_file.write_text(str(process.pid), encoding="utf-8")
    print(f"collector started: pid={process.pid} db={args.db} log={log_path}")
    return 0


def _cmd_stop(args: argparse.Namespace) -> int:
    pid_file = Path(args.pid_file)
    if not pid_file.exists():
        print("collector is not running")
        return 0
    pid = int(pid_file.read_text(encoding="utf-8").strip())
    if _is_alive(pid):
        os.kill(pid, signal.SIGTERM)
        for _ in range(30):
            if not _is_alive(pid):
                break
            time.sleep(0.1)
    pid_file.unlink(missing_ok=True)
    print(f"collector stopped: pid={pid}")
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    pid_file = Path(args.pid_file)
    if not pid_file.exists():
        print("collector is not running")
        return 1
    pid = int(pid_file.read_text(encoding="utf-8").strip())
    if _is_alive(pid):
        print(f"collector running: pid={pid}")
        return 0
    print(f"collector pid file exists but process is not running: pid={pid}")
    return 1


def _cmd_query(args: argparse.Namespace) -> int:
    rows = _query_rows(args)
    if args.format == "json":
        print(render_json(rows, args.limit))
    else:
        print(render_table(rows, args.limit))
    return 0


def _cmd_clean(args: argparse.Namespace) -> int:
    store = MetricsStore(args.db)
    try:
        store.clear()
    finally:
        store.close()
    print(f"database cleaned: {args.db}")
    return 0


def _cmd_list_configs(_args: argparse.Namespace) -> int:
    for path in list_preset_configs():
        print(f"{path.stem}\t{path}")
    return 0


def _query_rows(args: argparse.Namespace):
    config = apply_config_param_overrides(
        load_config(resolve_config_path(args.config)),
        args.config_param,
    )
    store = MetricsStore(args.db)
    try:
        return QueryEngine(store).query_config(
            config=config,
            window_seconds=parse_duration_seconds(args.window),
            start_ms=parse_time_ms(args.start_time) if args.start_time else None,
            aggr_by_seconds=(
                parse_duration_seconds(args.aggr_by) if args.aggr_by else None
            ),
            limit=args.limit,
            tag_filters=_parse_tag_filters(args.tag),
        )
    finally:
        store.close()


def _parse_tag_filters(items: list[str]) -> dict[str, str]:
    filters: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--tag expects KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--tag expects KEY=VALUE, got: {item}")
        filters[key] = value
    return filters


def _is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
