from __future__ import annotations

import argparse
import ast
import gzip
import json
import math
import random
import re
import sys
import urllib.parse
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

GIB = 1024**3
PREFIX_CACHE_QUERIES_METRICS = (
    "vllm:prefix_cache_queries_total",
    "prefix_cache_queries_total",
)
PREFIX_CACHE_HITS_METRICS = (
    "vllm:prefix_cache_hits_total",
    "prefix_cache_hits_total",
)
DRAM_POOL_MODES = ("per-node", "shared")

TRACE_RE = re.compile(
    r"timestamp:\s*(?P<timestamp>\d+(?:\.\d+)?),\s*"
    r"(?:request_id:\s*(?P<request_id>[^,]+),\s*)?"
    r"input_length:\s*(?P<input_length>\d+),\s*"
    r"output_length:\s*(?P<output_length>\d+),\s*"
    r"ucm_block_ids:\s*(?P<ucm_block_ids>\[.*?\])"
)
SYSTEM_TIME_RE = re.compile(r"^\[(?P<system_time>\d{4}-\d{2}-\d{2} [^\]]+)\]")
AVAILABLE_KV_RE = re.compile(
    r"\b(?:available|current)[_\s-]*(?:kv[_\s-]*)?cache[_\s-]*memory\b"
    r"[^0-9]*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>[kmgt]?i?b|bytes?)?",
    re.IGNORECASE,
)
TP_SIZE_RE = re.compile(
    r"(?:['\"]?tensor[_-]parallel[_-]size['\"]?\s*[:=]\s*|"
    r"--tensor[-_]parallel[-_]size\s+)"
    r"(?P<value>\d+)",
    re.IGNORECASE,
)
PROM_SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


@dataclass
class TraceRecord:
    timestamp: float
    input_length: int
    output_length: int
    hash_ids: list[str]
    source: str
    request_id: str | None = None
    system_time: str | None = None

    def to_json(self) -> dict:
        data = {
            "timestamp": self.timestamp,
            "input_length": self.input_length,
            "output_length": self.output_length,
            "hash_ids": self.hash_ids,
            "source": self.source,
        }
        if self.request_id:
            data["request_id"] = self.request_id
        if self.system_time:
            data["system_time"] = self.system_time
        return data


@dataclass
class LogFacts:
    log_files: list[str]
    records: list[TraceRecord]
    available_kv_cache_memory_bytes: list[int]
    tensor_parallel_sizes: list[int]
    parse_errors: list[str]


class LogLineParseError(ValueError):
    pass


@dataclass(frozen=True)
class CacheEntry:
    producer_index: int


class RequestGroups:
    def __init__(self) -> None:
        self.parent: list[int] = []
        self.first_timestamp: list[float] = []
        self.last_hit_timestamp: list[float | None] = []

    def add(self, timestamp: float) -> int:
        index = len(self.parent)
        self.parent.append(index)
        self.first_timestamp.append(timestamp)
        self.last_hit_timestamp.append(None)
        return index

    def find(self, index: int) -> int:
        parent = self.parent[index]
        if parent != index:
            self.parent[index] = self.find(parent)
        return self.parent[index]

    def union_roots(self, roots: Iterable[int]) -> int:
        root_set = {self.find(root) for root in roots}
        if not root_set:
            raise ValueError("cannot union empty request group")
        root = min(root_set, key=lambda item: self.first_timestamp[item])
        for item in root_set:
            if item == root:
                continue
            self.parent[item] = root
            item_last_hit = self.last_hit_timestamp[item]
            if item_last_hit is not None:
                root_last_hit = self.last_hit_timestamp[root]
                self.last_hit_timestamp[root] = (
                    item_last_hit
                    if root_last_hit is None
                    else max(root_last_hit, item_last_hit)
                )
        return root

    def record_hit(self, root: int, timestamp: float) -> None:
        root = self.find(root)
        last_hit = self.last_hit_timestamp[root]
        self.last_hit_timestamp[root] = (
            timestamp if last_hit is None else max(last_hit, timestamp)
        )

    def lifetimes(self) -> list[float]:
        values: list[float] = []
        for index, last_hit in enumerate(self.last_hit_timestamp):
            if self.find(index) != index or last_hit is None:
                continue
            values.append(last_hit - self.first_timestamp[index])
        return values


class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = max(0, int(capacity))
        self.items: OrderedDict[str, CacheEntry] = OrderedDict()

    def touch(self, key: str) -> bool:
        return self.get(key) is not None

    def get(self, key: str) -> CacheEntry | None:
        if self.capacity <= 0 or key not in self.items:
            return None
        self.items.move_to_end(key)
        return self.items[key]

    def put(self, key: str, entry: CacheEntry) -> None:
        if self.capacity <= 0:
            return
        if key in self.items:
            self.items[key] = entry
            self.items.move_to_end(key)
        else:
            self.items[key] = entry
        while len(self.items) > self.capacity:
            self.items.popitem(last=False)


def iter_log_files(log_path: Path) -> list[Path]:
    if log_path.is_file():
        return [log_path]

    patterns = ("*.log", "*.log.*", "*.log.gz")
    files: dict[Path, None] = {}
    for pattern in patterns:
        for path in log_path.rglob(pattern):
            if path.is_file():
                files[path] = None
    return sorted(files)


def open_log_file(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return path.open("r", encoding="utf-8", errors="ignore")


def parse_bytes(value: str, unit: str | None) -> int:
    number = float(value)
    if not unit:
        return int(number)
    multipliers = {
        "b": 1,
        "byte": 1,
        "bytes": 1,
        "kb": 1024,
        "kib": 1024,
        "mb": 1024**2,
        "mib": 1024**2,
        "gb": 1024**3,
        "gib": 1024**3,
        "tb": 1024**4,
        "tib": 1024**4,
    }
    return int(number * multipliers.get(unit.lower(), 1))


def parse_is_mla(value: str) -> bool:
    return value.lower() == "true"


def line_excerpt(line: str, column: int | None = None, limit: int = 500) -> str:
    text = line.rstrip("\r\n")
    if len(text) <= limit:
        return repr(text)

    center = len(text) // 2 if column is None else max(0, column - 1)
    start = max(0, min(center - limit // 2, len(text) - limit))
    end = start + limit
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return repr(prefix + text[start:end] + suffix)


def line_parse_error(
    stage: str,
    source: str,
    line_number: int | None,
    line: str,
    cause: Exception | str,
    column: int | None = None,
) -> LogLineParseError:
    location = f"{source}:{line_number}" if line_number is not None else source
    column_text = f"\n  column: {column}" if column is not None else ""
    return LogLineParseError(
        "log line parse failed"
        f"\n  stage: {stage}"
        f"\n  source: {location}"
        f"{column_text}"
        f"\n  cause: {cause}"
        f"\n  line excerpt: {line_excerpt(line, column)}"
    )


def print_parse_error_summary(parse_errors: list[str]) -> None:
    if not parse_errors:
        return
    print(f"Log line parse failures: {len(parse_errors)}", file=sys.stderr)
    for index, error in enumerate(parse_errors, start=1):
        print(f"\nParse failure {index}:\n{error}", file=sys.stderr)
    sys.stderr.flush()


def parse_trace_line(
    line: str, source: str, line_number: int | None = None
) -> TraceRecord | None:
    match = TRACE_RE.search(line)
    if not match:
        if "timestamp:" in line and "ucm_block_ids:" in line:
            column = line.find("ucm_block_ids:") + 1
            raise line_parse_error(
                "trace.pattern_match",
                source,
                line_number,
                line,
                "trace marker found but required fields or list terminator are invalid",
                column,
            )
        return None
    try:
        hash_ids = ast.literal_eval(match.group("ucm_block_ids"))
    except (SyntaxError, ValueError) as exc:
        offset = exc.offset if isinstance(exc, SyntaxError) else None
        column = match.start("ucm_block_ids") + (offset or 1)
        raise line_parse_error(
            "trace.ucm_block_ids.literal_eval",
            source,
            line_number,
            line,
            exc,
            column,
        ) from exc
    if not isinstance(hash_ids, list):
        column = match.start("ucm_block_ids") + 1
        raise line_parse_error(
            "trace.ucm_block_ids.validation",
            source,
            line_number,
            line,
            f"expected list, got {type(hash_ids).__name__}",
            column,
        )

    system_time_match = SYSTEM_TIME_RE.search(line)
    request_id = match.group("request_id")
    return TraceRecord(
        timestamp=float(match.group("timestamp")),
        input_length=int(match.group("input_length")),
        output_length=int(match.group("output_length")),
        hash_ids=[str(item) for item in hash_ids],
        source=source,
        request_id=request_id.strip() if request_id else None,
        system_time=(
            system_time_match.group("system_time") if system_time_match else None
        ),
    )


def collect_log_facts(log_path: Path) -> LogFacts:
    if not log_path.exists():
        raise ValueError(f"log scan failed: path does not exist: {log_path}")
    if not log_path.is_file() and not log_path.is_dir():
        raise ValueError(
            f"log scan failed: path is not a file or directory: {log_path}"
        )

    log_files = iter_log_files(log_path)
    if not log_files:
        raise ValueError(
            f"log scan failed: no log files found in directory: {log_path}"
        )

    print(f"Log files to parse ({len(log_files)}):")
    for path in log_files:
        print(f"  {path}")
    sys.stdout.flush()

    records: list[TraceRecord] = []
    available_memory: list[int] = []
    tensor_parallel_sizes: list[int] = []
    parse_errors: list[str] = []
    for path in log_files:
        with open_log_file(path) as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    record = parse_trace_line(line, str(path), line_number)
                except LogLineParseError as exc:
                    parse_errors.append(str(exc))
                    record = None
                if record is not None:
                    records.append(record)

                for match in AVAILABLE_KV_RE.finditer(line):
                    available_memory.append(
                        parse_bytes(match.group("value"), match.group("unit"))
                    )
                for match in TP_SIZE_RE.finditer(line):
                    tensor_parallel_sizes.append(int(match.group("value")))

    print_parse_error_summary(parse_errors)

    if not records:
        raise ValueError("log validation failed: no trace records found in log files")
    if not available_memory:
        raise ValueError(
            "log validation failed: available kv cache memory was not found "
            "in log files"
        )
    if not tensor_parallel_sizes:
        raise ValueError(
            "log validation failed: tensor_parallel_size was not found in log files"
        )

    records.sort(key=lambda item: item.timestamp)
    return LogFacts(
        log_files=[str(path) for path in log_files],
        records=records,
        available_kv_cache_memory_bytes=available_memory,
        tensor_parallel_sizes=tensor_parallel_sizes,
        parse_errors=parse_errors,
    )


def resolve_tp_size(facts: LogFacts) -> int:
    values = set(facts.tensor_parallel_sizes)
    if len(values) != 1:
        raise ValueError(
            "conflicting tensor_parallel_size values found in log files: "
            + ", ".join(str(value) for value in sorted(values))
        )
    tp_size = next(iter(values))
    if tp_size <= 0:
        raise ValueError("tensor_parallel_size must be > 0")
    return tp_size


def resolve_gpu_cache_bytes(facts: LogFacts, is_mla: bool, tp_size: int) -> int:
    min_available = min(facts.available_kv_cache_memory_bytes)
    return min_available if is_mla else min_available * tp_size


def parse_prometheus_samples(metrics_text: str) -> dict[str, float]:
    samples: dict[str, float] = {}
    for raw_line in metrics_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = PROM_SAMPLE_RE.match(line)
        if not match:
            continue
        name = match.group("name")
        samples[name] = samples.get(name, 0.0) + float(match.group("value"))
    return samples


def metrics_url_from_service_url(service_url: str) -> str:
    normalized = service_url.strip()
    if not normalized:
        raise ValueError("service_url is empty")
    if "://" not in normalized:
        normalized = "http://" + normalized
    parsed = urllib.parse.urlparse(normalized)
    if parsed.scheme == "file":
        return normalized
    stripped = normalized.rstrip("/")
    if stripped.endswith("/metrics"):
        return stripped
    return stripped + "/metrics"


def first_metric_value(samples: dict[str, float], names: tuple[str, ...]) -> float:
    for name in names:
        if name in samples:
            return samples[name]
    raise ValueError("required metrics missing from /metrics: " + " or ".join(names))


def fetch_service_hit_rate(service_url: str, timeout: float) -> dict:
    metrics_url = metrics_url_from_service_url(service_url)
    with urllib.request.urlopen(metrics_url, timeout=timeout) as response:
        text = response.read().decode("utf-8", errors="replace")
    samples = parse_prometheus_samples(text)

    queries = first_metric_value(samples, PREFIX_CACHE_QUERIES_METRICS)
    hits = first_metric_value(samples, PREFIX_CACHE_HITS_METRICS)
    hit_rate = hits / queries if queries > 0 else 0.0
    return {
        "service_url": service_url,
        "metrics_url": metrics_url,
        "prefix_cache_hits_total": hits,
        "prefix_cache_queries_total": queries,
        "actual_kv_cache_hit_rate": hit_rate,
    }


def block_token_weights(record: TraceRecord) -> list[int]:
    block_count = len(record.hash_ids)
    if block_count == 0:
        return []
    base = record.input_length // block_count
    remainder = record.input_length % block_count
    return [base + 1 if index < remainder else base for index in range(block_count)]


def _rate(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def nearest_percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = math.ceil(len(sorted_values) * percentile / 100) - 1
    return sorted_values[max(0, min(index, len(sorted_values) - 1))]


def request_lifetime_stats(values: list[float]) -> dict:
    return {
        "request_lifetime_sample_count": len(values),
        "average_request_lifetime_seconds": (
            sum(values) / len(values) if values else 0.0
        ),
        "p90_request_lifetime_seconds": nearest_percentile(values, 90),
        "p95_request_lifetime_seconds": nearest_percentile(values, 95),
    }


def simulate_cache_hit_rate(
    records: Iterable[TraceRecord],
    gpu_capacity_blocks: int,
    dram_capacity_blocks: int,
    fs_capacity_blocks: int,
    num_nodes: int = 1,
    random_seed: int | None = 0,
    dram_pool_mode: str = "per-node",
) -> dict:
    if num_nodes < 1:
        raise ValueError("num_nodes must be >= 1")
    if gpu_capacity_blocks < 0 or dram_capacity_blocks < 0 or fs_capacity_blocks < 0:
        raise ValueError("cache capacities must be >= 0")
    if dram_pool_mode not in DRAM_POOL_MODES:
        raise ValueError("dram_pool_mode must be one of: " + ", ".join(DRAM_POOL_MODES))

    rng = random.Random(random_seed)
    gpu_caches = [LRUCache(gpu_capacity_blocks) for _ in range(num_nodes)]
    if dram_pool_mode == "shared":
        shared_dram_cache = LRUCache(dram_capacity_blocks)
        dram_caches = [shared_dram_cache for _ in range(num_nodes)]
    else:
        dram_caches = [LRUCache(dram_capacity_blocks) for _ in range(num_nodes)]
    fs_cache = LRUCache(fs_capacity_blocks)

    total_tokens = 0
    gpu_hit_tokens = 0
    dram_hit_tokens = 0
    fs_hit_tokens = 0
    miss_tokens = 0
    request_groups = RequestGroups()

    for record in records:
        request_index = request_groups.add(record.timestamp)
        total_tokens += record.input_length
        if not record.hash_ids:
            miss_tokens += record.input_length
            continue

        node_index = rng.randrange(num_nodes) if num_nodes > 1 else 0
        gpu_cache = gpu_caches[node_index]
        dram_cache = dram_caches[node_index]
        prefix_available = True
        hit_roots: set[int] = set()

        for block_id, weight in zip(record.hash_ids, block_token_weights(record)):
            gpu_entry = gpu_cache.get(block_id) if prefix_available else None
            if gpu_entry is not None:
                gpu_hit_tokens += weight
                hit_roots.add(request_groups.find(gpu_entry.producer_index))
                continue

            dram_entry = dram_cache.get(block_id) if prefix_available else None
            if dram_entry is not None:
                dram_hit_tokens += weight
                hit_roots.add(request_groups.find(dram_entry.producer_index))
                gpu_cache.put(block_id, dram_entry)
                continue

            fs_entry = fs_cache.get(block_id) if prefix_available else None
            if fs_entry is not None:
                fs_hit_tokens += weight
                hit_roots.add(request_groups.find(fs_entry.producer_index))
                dram_cache.put(block_id, fs_entry)
                gpu_cache.put(block_id, fs_entry)
            else:
                miss_tokens += weight
                prefix_available = False

        if hit_roots:
            root = request_groups.union_roots([request_index, *hit_roots])
            request_groups.record_hit(root, record.timestamp)

        entry = CacheEntry(producer_index=request_index)
        for block_id in record.hash_ids:
            fs_cache.put(block_id, entry)
            dram_cache.put(block_id, entry)
            gpu_cache.put(block_id, entry)

    total_hit_tokens = gpu_hit_tokens + dram_hit_tokens + fs_hit_tokens
    return {
        "total_tokens": total_tokens,
        "gpu_hit_tokens": gpu_hit_tokens,
        "dram_hit_tokens": dram_hit_tokens,
        "fs_hit_tokens": fs_hit_tokens,
        "miss_tokens": miss_tokens,
        "total_hit_tokens": total_hit_tokens,
        "hit_rate": _rate(total_hit_tokens, total_tokens),
        **request_lifetime_stats(request_groups.lifetimes()),
    }


def total_request_tokens(records: Iterable[TraceRecord]) -> int:
    return sum(record.input_length for record in records)


def unique_block_count(records: Iterable[TraceRecord]) -> int:
    return len({block_id for record in records for block_id in record.hash_ids})


def percent(value: float) -> float:
    return value * 100


def write_trace(records: Iterable[TraceRecord], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_json(), ensure_ascii=False) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze theoretical UCM KV cache hit-rate uplift from logs."
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        required=True,
        help="Log file or directory containing log files.",
    )
    parser.add_argument(
        "--block-kv-cache-size",
        "--block-bytes",
        dest="block_kv_cache_size",
        type=int,
        required=True,
    )
    parser.add_argument("--is-mla", choices=("true", "false"), required=True)
    parser.add_argument("--dram-pool-size-gb", type=float, required=True)
    parser.add_argument(
        "--dram-pool-mode",
        choices=DRAM_POOL_MODES,
        default="per-node",
        help=(
            "Interpret --dram-pool-size-gb as capacity per node or as one "
            "shared capacity across all nodes."
        ),
    )
    parser.add_argument("--fs-pool-size-gb", type=float, required=True)
    parser.add_argument("--service-url")
    parser.add_argument("--metrics-timeout", type=float, default=5.0)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--trace-output", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.block_kv_cache_size <= 0:
        raise ValueError("block kv cache size must be > 0")
    if args.dram_pool_size_gb < 0 or args.fs_pool_size_gb < 0:
        raise ValueError("pool sizes must be >= 0")
    if args.num_nodes < 1:
        raise ValueError("num_nodes must be >= 1")


def build_analysis(args: argparse.Namespace, facts: LogFacts | None = None) -> dict:
    validate_args(args)
    is_mla = parse_is_mla(args.is_mla)
    if facts is None:
        facts = collect_log_facts(args.log_dir)
    tp_size = resolve_tp_size(facts)
    gpu_kv_cache_bytes = resolve_gpu_cache_bytes(facts, is_mla, tp_size)
    block_bytes = args.block_kv_cache_size
    gpu_capacity_blocks = gpu_kv_cache_bytes // block_bytes
    dram_capacity_blocks = int(args.dram_pool_size_gb * GIB) // block_bytes
    dram_total_capacity_blocks = (
        dram_capacity_blocks * args.num_nodes
        if args.dram_pool_mode == "per-node"
        else dram_capacity_blocks
    )
    fs_capacity_blocks = int(args.fs_pool_size_gb * GIB) // block_bytes
    unique_blocks = unique_block_count(facts.records)

    theoretical_max = simulate_cache_hit_rate(
        records=facts.records,
        gpu_capacity_blocks=unique_blocks,
        dram_capacity_blocks=unique_blocks,
        fs_capacity_blocks=unique_blocks,
        num_nodes=args.num_nodes,
        random_seed=args.random_seed,
        dram_pool_mode=args.dram_pool_mode,
    )
    hbm = simulate_cache_hit_rate(
        records=facts.records,
        gpu_capacity_blocks=gpu_capacity_blocks,
        dram_capacity_blocks=0,
        fs_capacity_blocks=0,
        num_nodes=args.num_nodes,
        random_seed=args.random_seed,
        dram_pool_mode=args.dram_pool_mode,
    )
    hbm_dram = simulate_cache_hit_rate(
        records=facts.records,
        gpu_capacity_blocks=gpu_capacity_blocks,
        dram_capacity_blocks=dram_capacity_blocks,
        fs_capacity_blocks=0,
        num_nodes=args.num_nodes,
        random_seed=args.random_seed,
        dram_pool_mode=args.dram_pool_mode,
    )
    hbm_dram_fs = simulate_cache_hit_rate(
        records=facts.records,
        gpu_capacity_blocks=gpu_capacity_blocks,
        dram_capacity_blocks=dram_capacity_blocks,
        fs_capacity_blocks=fs_capacity_blocks,
        num_nodes=args.num_nodes,
        random_seed=args.random_seed,
        dram_pool_mode=args.dram_pool_mode,
    )
    service_metrics = (
        fetch_service_hit_rate(args.service_url, args.metrics_timeout)
        if args.service_url
        else None
    )

    request_count = len(facts.records)
    request_tokens = total_request_tokens(facts.records)
    analysis = {
        "total_request_count": request_count,
        "total_request_token_count": request_tokens,
        "average_request_token_count": request_tokens / request_count,
        "theoretical_max_kv_cache_hit_rate_percent": percent(
            theoretical_max["hit_rate"]
        ),
        "hbm_theoretical_hit_rate_percent": percent(hbm["hit_rate"]),
        "hbm_dram_pool_theoretical_hit_rate_percent": percent(hbm_dram["hit_rate"]),
        "hbm_dram_fs_pool_theoretical_hit_rate_percent": percent(
            hbm_dram_fs["hit_rate"]
        ),
        "request_lifetime_sample_count": theoretical_max[
            "request_lifetime_sample_count"
        ],
        "average_request_lifetime_seconds": theoretical_max[
            "average_request_lifetime_seconds"
        ],
        "p90_request_lifetime_seconds": theoretical_max["p90_request_lifetime_seconds"],
        "p95_request_lifetime_seconds": theoretical_max["p95_request_lifetime_seconds"],
    }
    if service_metrics is not None:
        analysis["service_actual_kv_cache_hit_rate_percent"] = percent(
            service_metrics["actual_kv_cache_hit_rate"]
        )

    return {
        "inputs": {
            "log_dir": str(args.log_dir),
            "block_kv_cache_size": block_bytes,
            "is_mla": is_mla,
            "dram_pool_size_gb": args.dram_pool_size_gb,
            "dram_pool_mode": args.dram_pool_mode,
            "fs_pool_size_gb": args.fs_pool_size_gb,
            "service_url": args.service_url,
        },
        "derived": {
            "log_files": facts.log_files,
            "parse_failure_count": len(facts.parse_errors),
            "parse_failures": facts.parse_errors,
            "available_kv_cache_memory_bytes": facts.available_kv_cache_memory_bytes,
            "tp_size": tp_size,
            "gpu_kv_cache_bytes": gpu_kv_cache_bytes,
            "gpu_capacity_blocks": gpu_capacity_blocks,
            "dram_capacity_blocks": dram_capacity_blocks,
            "dram_total_capacity_blocks": dram_total_capacity_blocks,
            "fs_capacity_blocks": fs_capacity_blocks,
            "unique_block_count": unique_blocks,
            "num_nodes": args.num_nodes,
            "metrics": service_metrics,
        },
        "analysis": analysis,
        "simulation_details": {
            "theoretical_max": theoretical_max,
            "hbm": hbm,
            "hbm_dram": hbm_dram,
            "hbm_dram_fs": hbm_dram_fs,
        },
    }


def print_summary(result: dict) -> None:
    analysis = result["analysis"]
    derived = result["derived"]
    inputs = result["inputs"]
    hbm_bytes = derived["gpu_kv_cache_bytes"]
    print("Trace cache hit rate analysis")
    print(f"  Total request count: {analysis['total_request_count']}")
    print(f"  Total request token count: {analysis['total_request_token_count']}")
    print(
        "  Average tokens per request: "
        f"{analysis['average_request_token_count']:.2f}"
    )
    print("  Total HBM available KV cache size: " f"{hbm_bytes / 1024**3:.2f} GiB")
    print(f"  TP size: {derived['tp_size']}")
    if inputs["dram_pool_mode"] == "per-node":
        print(f"  DRAM pool size per node: {inputs['dram_pool_size_gb']:.2f} GiB")
    else:
        print(f"  Shared DRAM pool size: {inputs['dram_pool_size_gb']:.2f} GiB")
    print(f"  FS pool size: {inputs['fs_pool_size_gb']:.2f} GiB")
    print(
        "  Theoretical max KV cache hit rate: "
        f"{analysis['theoretical_max_kv_cache_hit_rate_percent']:.6f}%"
    )
    if "service_actual_kv_cache_hit_rate_percent" in analysis:
        print(
            "  Service actual KV cache hit rate: "
            f"{analysis['service_actual_kv_cache_hit_rate_percent']:.6f}%"
        )
    print(
        "  HBM theoretical hit rate: "
        f"{analysis['hbm_theoretical_hit_rate_percent']:.6f}%"
    )
    print(
        "  HBM + DRAM pool theoretical hit rate: "
        f"{analysis['hbm_dram_pool_theoretical_hit_rate_percent']:.6f}%"
    )
    print(
        "  HBM + DRAM pool + FS pool theoretical hit rate: "
        f"{analysis['hbm_dram_fs_pool_theoretical_hit_rate_percent']:.6f}%"
    )
    print(
        "  Request lifetime sample count: "
        f"{analysis['request_lifetime_sample_count']}"
    )
    print(
        "  Average request lifetime: "
        f"{analysis['average_request_lifetime_seconds']:.6f} s"
    )
    print(
        "  P90 request lifetime: " f"{analysis['p90_request_lifetime_seconds']:.6f} s"
    )
    print(
        "  P95 request lifetime: " f"{analysis['p95_request_lifetime_seconds']:.6f} s"
    )


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        facts = collect_log_facts(args.log_dir)
        result = build_analysis(args, facts)
        if args.trace_output:
            write_trace(facts.records, args.trace_output)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        print_summary(result)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
