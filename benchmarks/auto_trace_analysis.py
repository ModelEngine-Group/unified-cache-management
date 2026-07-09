from __future__ import annotations

import argparse
import ast
import gzip
import json
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
GPU_HIT_METRIC = "ucm:gpu_hbm_hit_tokens_total"
UCM_HIT_METRIC = "ucm:ucm_hit_tokens_total"
TOTAL_QUERY_METRIC = "ucm:total_prefix_query_tokens_total"

TRACE_RE = re.compile(
    r"timestamp:\s*(?P<timestamp>\d+(?:\.\d+)?),\s*"
    r"(?:request_id:\s*(?P<request_id>[^,]+),\s*)?"
    r"input_length:\s*(?P<input_length>\d+),\s*"
    r"output_length:\s*(?P<output_length>\d+),\s*"
    r"ucm_block_ids:\s*(?P<ucm_block_ids>\[.*?\])"
)
SYSTEM_TIME_RE = re.compile(r"^\[(?P<system_time>\d{4}-\d{2}-\d{2} [^\]]+)\]")
AVAILABLE_KV_RE = re.compile(
    r"\bavailable[_\s-]*(?:kv[_\s-]*)?cache[_\s-]*memory\b"
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


class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = max(0, int(capacity))
        self.items: OrderedDict[str, None] = OrderedDict()

    def touch(self, key: str) -> bool:
        if self.capacity <= 0 or key not in self.items:
            return False
        self.items.move_to_end(key)
        return True

    def put(self, key: str) -> None:
        if self.capacity <= 0:
            return
        if key in self.items:
            self.items.move_to_end(key)
        else:
            self.items[key] = None
        while len(self.items) > self.capacity:
            self.items.popitem(last=False)


def iter_log_files(log_dir: Path) -> list[Path]:
    patterns = ("*.log", "*.log.*", "*.log.gz")
    files: dict[Path, None] = {}
    for pattern in patterns:
        for path in log_dir.rglob(pattern):
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


def parse_trace_line(line: str, source: str) -> TraceRecord | None:
    match = TRACE_RE.search(line)
    if not match:
        return None
    try:
        hash_ids = ast.literal_eval(match.group("ucm_block_ids"))
    except (SyntaxError, ValueError):
        raise ValueError(f"failed to parse ucm_block_ids in {source}")
    if not isinstance(hash_ids, list):
        raise ValueError(f"ucm_block_ids is not a list in {source}")

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


def collect_log_facts(log_dir: Path) -> LogFacts:
    if not log_dir.exists() or not log_dir.is_dir():
        raise ValueError(f"log directory does not exist: {log_dir}")

    log_files = iter_log_files(log_dir)
    if not log_files:
        raise ValueError(f"no log files found in log directory: {log_dir}")

    records: list[TraceRecord] = []
    available_memory: list[int] = []
    tensor_parallel_sizes: list[int] = []
    for path in log_files:
        with open_log_file(path) as handle:
            for line in handle:
                record = parse_trace_line(line, str(path))
                if record is not None:
                    records.append(record)

                for match in AVAILABLE_KV_RE.finditer(line):
                    available_memory.append(
                        parse_bytes(match.group("value"), match.group("unit"))
                    )
                for match in TP_SIZE_RE.finditer(line):
                    tensor_parallel_sizes.append(int(match.group("value")))

    if not records:
        raise ValueError("no trace records found in log files")
    if not available_memory:
        raise ValueError("available kv cache memory was not found in log files")
    if not tensor_parallel_sizes:
        raise ValueError("tensor_parallel_size was not found in log files")

    records.sort(key=lambda item: item.timestamp)
    return LogFacts(
        log_files=[str(path) for path in log_files],
        records=records,
        available_kv_cache_memory_bytes=available_memory,
        tensor_parallel_sizes=tensor_parallel_sizes,
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


def fetch_service_hit_rate(service_url: str, timeout: float) -> dict:
    metrics_url = metrics_url_from_service_url(service_url)
    with urllib.request.urlopen(metrics_url, timeout=timeout) as response:
        text = response.read().decode("utf-8", errors="replace")
    samples = parse_prometheus_samples(text)
    missing = [
        name
        for name in (GPU_HIT_METRIC, UCM_HIT_METRIC, TOTAL_QUERY_METRIC)
        if name not in samples
    ]
    if missing:
        raise ValueError(
            "required metrics missing from /metrics: " + ", ".join(missing)
        )

    total = samples[TOTAL_QUERY_METRIC]
    if total <= 0:
        raise ValueError(f"{TOTAL_QUERY_METRIC} must be > 0")

    gpu_hit = samples[GPU_HIT_METRIC]
    ucm_hit = samples[UCM_HIT_METRIC]
    return {
        "service_url": service_url,
        "metrics_url": metrics_url,
        "gpu_hbm_hit_tokens_total": gpu_hit,
        "ucm_hit_tokens_total": ucm_hit,
        "total_prefix_query_tokens_total": total,
        "actual_kv_cache_hit_rate": (gpu_hit + ucm_hit) / total,
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


def simulate_cache_hit_rate(
    records: Iterable[TraceRecord],
    gpu_capacity_blocks: int,
    dram_capacity_blocks: int,
    fs_capacity_blocks: int,
    num_nodes: int = 1,
    random_seed: int | None = 0,
) -> dict:
    if num_nodes < 1:
        raise ValueError("num_nodes must be >= 1")
    if gpu_capacity_blocks < 0 or dram_capacity_blocks < 0 or fs_capacity_blocks < 0:
        raise ValueError("cache capacities must be >= 0")

    rng = random.Random(random_seed)
    gpu_caches = [LRUCache(gpu_capacity_blocks) for _ in range(num_nodes)]
    dram_caches = [LRUCache(dram_capacity_blocks) for _ in range(num_nodes)]
    fs_cache = LRUCache(fs_capacity_blocks)

    total_tokens = 0
    gpu_hit_tokens = 0
    dram_hit_tokens = 0
    fs_hit_tokens = 0
    miss_tokens = 0

    for record in records:
        total_tokens += record.input_length
        if not record.hash_ids:
            miss_tokens += record.input_length
            continue

        node_index = rng.randrange(num_nodes) if num_nodes > 1 else 0
        gpu_cache = gpu_caches[node_index]
        dram_cache = dram_caches[node_index]
        prefix_available = True

        for block_id, weight in zip(record.hash_ids, block_token_weights(record)):
            if prefix_available and gpu_cache.touch(block_id):
                gpu_hit_tokens += weight
            elif prefix_available and dram_cache.touch(block_id):
                dram_hit_tokens += weight
                gpu_cache.put(block_id)
            elif prefix_available and fs_cache.touch(block_id):
                fs_hit_tokens += weight
                dram_cache.put(block_id)
                gpu_cache.put(block_id)
            else:
                miss_tokens += weight
                prefix_available = False

        for block_id in record.hash_ids:
            fs_cache.put(block_id)
            dram_cache.put(block_id)
            gpu_cache.put(block_id)

    total_hit_tokens = gpu_hit_tokens + dram_hit_tokens + fs_hit_tokens
    return {
        "total_tokens": total_tokens,
        "gpu_hit_tokens": gpu_hit_tokens,
        "dram_hit_tokens": dram_hit_tokens,
        "fs_hit_tokens": fs_hit_tokens,
        "miss_tokens": miss_tokens,
        "total_hit_tokens": total_hit_tokens,
        "hit_rate": _rate(total_hit_tokens, total_tokens),
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
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument(
        "--block-kv-cache-size",
        "--block-bytes",
        dest="block_kv_cache_size",
        type=int,
        required=True,
    )
    parser.add_argument("--is-mla", choices=("true", "false"), required=True)
    parser.add_argument("--dram-pool-size-gb", type=float, required=True)
    parser.add_argument("--fs-pool-size-gb", type=float, required=True)
    parser.add_argument("--service-url", required=True)
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


def build_analysis(args: argparse.Namespace) -> dict:
    validate_args(args)
    is_mla = parse_is_mla(args.is_mla)
    facts = collect_log_facts(args.log_dir)
    tp_size = resolve_tp_size(facts)
    gpu_kv_cache_bytes = resolve_gpu_cache_bytes(facts, is_mla, tp_size)
    block_bytes = args.block_kv_cache_size
    gpu_capacity_blocks = gpu_kv_cache_bytes // block_bytes
    dram_capacity_blocks = int(args.dram_pool_size_gb * GIB) // block_bytes
    fs_capacity_blocks = int(args.fs_pool_size_gb * GIB) // block_bytes
    unique_blocks = unique_block_count(facts.records)

    theoretical_max = simulate_cache_hit_rate(
        records=facts.records,
        gpu_capacity_blocks=unique_blocks,
        dram_capacity_blocks=unique_blocks,
        fs_capacity_blocks=unique_blocks,
        num_nodes=args.num_nodes,
        random_seed=args.random_seed,
    )
    hbm_dram = simulate_cache_hit_rate(
        records=facts.records,
        gpu_capacity_blocks=gpu_capacity_blocks,
        dram_capacity_blocks=dram_capacity_blocks,
        fs_capacity_blocks=0,
        num_nodes=args.num_nodes,
        random_seed=args.random_seed,
    )
    hbm_dram_fs = simulate_cache_hit_rate(
        records=facts.records,
        gpu_capacity_blocks=gpu_capacity_blocks,
        dram_capacity_blocks=dram_capacity_blocks,
        fs_capacity_blocks=fs_capacity_blocks,
        num_nodes=args.num_nodes,
        random_seed=args.random_seed,
    )
    service_metrics = fetch_service_hit_rate(args.service_url, args.metrics_timeout)

    request_count = len(facts.records)
    request_tokens = total_request_tokens(facts.records)
    analysis = {
        "total_request_count": request_count,
        "total_request_token_count": request_tokens,
        "average_request_token_count": request_tokens / request_count,
        "theoretical_max_kv_cache_hit_rate_percent": percent(
            theoretical_max["hit_rate"]
        ),
        "service_actual_kv_cache_hit_rate_percent": percent(
            service_metrics["actual_kv_cache_hit_rate"]
        ),
        "hbm_dram_pool_theoretical_hit_rate_percent": percent(hbm_dram["hit_rate"]),
        "hbm_dram_fs_pool_theoretical_hit_rate_percent": percent(
            hbm_dram_fs["hit_rate"]
        ),
    }

    return {
        "inputs": {
            "log_dir": str(args.log_dir),
            "block_kv_cache_size": block_bytes,
            "is_mla": is_mla,
            "dram_pool_size_gb": args.dram_pool_size_gb,
            "fs_pool_size_gb": args.fs_pool_size_gb,
            "service_url": args.service_url,
        },
        "derived": {
            "log_files": facts.log_files,
            "available_kv_cache_memory_bytes": facts.available_kv_cache_memory_bytes,
            "tp_size": tp_size,
            "gpu_kv_cache_bytes": gpu_kv_cache_bytes,
            "gpu_capacity_blocks": gpu_capacity_blocks,
            "dram_capacity_blocks": dram_capacity_blocks,
            "fs_capacity_blocks": fs_capacity_blocks,
            "unique_block_count": unique_blocks,
            "num_nodes": args.num_nodes,
            "metrics": service_metrics,
        },
        "analysis": analysis,
        "simulation_details": {
            "theoretical_max": theoretical_max,
            "hbm_dram": hbm_dram,
            "hbm_dram_fs": hbm_dram_fs,
        },
    }


def print_summary(result: dict) -> None:
    analysis = result["analysis"]
    derived = result["derived"]
    inputs = result["inputs"]
    hbm_bytes = derived["gpu_kv_cache_bytes"]
    print("Trace cache uplift analysis")
    print(f"  total request count: {analysis['total_request_count']}")
    print(f"  total request token count: {analysis['total_request_token_count']}")
    print(
        "  average tokens per request: "
        f"{analysis['average_request_token_count']:.2f}"
    )
    print(
        "  total hbm available kv cache size: "
        f"{hbm_bytes} bytes ({hbm_bytes / 1024**3:.2f} GiB)"
    )
    print(f"  tp size: {derived['tp_size']}")
    print(f"  dram pool size: {inputs['dram_pool_size_gb']:.2f} GiB")
    print(f"  fs pool size: {inputs['fs_pool_size_gb']:.2f} GiB")
    print(
        "  theoretical max kv cache hit rate: "
        f"{analysis['theoretical_max_kv_cache_hit_rate_percent']:.6f}%"
    )
    print(
        "  service actual kv cache hit rate: "
        f"{analysis['service_actual_kv_cache_hit_rate_percent']:.6f}%"
    )
    print(
        "  hbm + dram_pool theoretical hit rate: "
        f"{analysis['hbm_dram_pool_theoretical_hit_rate_percent']:.6f}%"
    )
    print(
        "  hbm + dram_pool + fs_pool theoretical hit rate: "
        f"{analysis['hbm_dram_fs_pool_theoretical_hit_rate_percent']:.6f}%"
    )


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        result = build_analysis(args)
        if args.trace_output:
            write_trace(collect_log_facts(args.log_dir).records, args.trace_output)
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
