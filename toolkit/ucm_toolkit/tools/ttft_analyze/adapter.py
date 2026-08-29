"""ttft-analyze adapter."""

from __future__ import annotations

import argparse

from ...registry import ToolAdapter
from . import kv_size, model


def _arch_label(arch: kv_size.ModelArchitecture) -> str:
    arch_type = kv_size.detect_architecture(arch)
    if arch_type == "dsa":
        return "DSA"
    if arch_type == "mla":
        return "MLA"
    if arch.num_key_value_heads == arch.num_attention_heads:
        return "MHA"
    if arch.num_key_value_heads == 1:
        return "MQA"
    return "GQA"


class TtftAnalyzeTool(ToolAdapter):
    """Adapter for UCM prefix-cache TTFT estimation."""

    name = "ttft-analyze"
    aliases = ("ttft_analyze",)
    description = "Estimate TTFT for UCM prefix-cache hits."
    buildable = False

    def add_run_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--model-dir",
            required=True,
            help="model directory; input length derives the kv-cache size.",
        )
        parser.add_argument(
            "--posix-bw",
            type=float,
            required=True,
            help="POSIX storage read bandwidth (GB/s).",
        )
        parser.add_argument(
            "--h2d-bw",
            type=float,
            required=True,
            help="H2D transfer bandwidth (GB/s).",
        )
        parser.add_argument(
            "--input-len",
            type=int,
            required=True,
            help="input sequence length (assumed prefix-hit length).",
        )
        parser.add_argument(
            "--ttft-prefill",
            type=float,
            required=True,
            help="Full Prefill TTFT at this input length (ms).",
        )
        parser.add_argument(
            "--ttft-hbm",
            type=float,
            required=True,
            help="Full HBM Prefix Cache TTFT at this input length (ms).",
        )
        parser.add_argument(
            "--tp",
            type=int,
            default=1,
            help="tensor-parallel card count (default 1).",
        )

    def _build_run_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            prog="ucm-toolkit run ttft-analyze",
            description="Estimate TTFT for UCM prefix-cache hits.",
        )
        self.add_run_args(parser)
        return parser

    def run(self, tool_args: list[str]) -> int:
        parser = self._build_run_parser()
        args = parser.parse_args(tool_args)
        if args.input_len <= 0:
            parser.error("--input-len must be positive")
        if args.tp < 1:
            parser.error("--tp must be at least 1")

        arch = kv_size.load_model_architecture(args.model_dir)
        cache_total = kv_size.kv_cache_bytes(arch, args.input_len)
        cache_per_card = kv_size.per_card_cache_bytes(arch, args.input_len, args.tp)

        inputs = model.TtftInputs(
            cache_size_bytes=cache_per_card,
            tp=args.tp,
            posix_bw_gbps=args.posix_bw,
            h2d_bw_gbps=args.h2d_bw,
            ttft_prefill_ms=args.ttft_prefill,
            ttft_hbm_ms=args.ttft_hbm,
        )
        layered = model.analyze(inputs, "layered")
        full = model.analyze(inputs, "full")

        _render(
            arch=arch,
            arch_label=_arch_label(arch),
            cache_total=cache_total,
            cache_per_card=cache_per_card,
            args=args,
            layered=layered,
            full=full,
        )
        return 0

    def doctor(self, args: argparse.Namespace | None = None) -> int:
        print(f"{self.name}: no environment checks")
        return 0


def _render(
    arch: kv_size.ModelArchitecture,
    arch_label: str,
    cache_total: int,
    cache_per_card: float,
    args: argparse.Namespace,
    layered: model.TtftBreakdown,
    full: model.TtftBreakdown,
) -> None:
    cache_total_gb = cache_total / 1e9
    cache_per_card_gb = cache_per_card / 1e9
    print("UCM Prefix Cache TTFT estimate")
    print("=" * 72)
    print(f"model-dir   {args.model_dir}")
    print(f"arch        {arch_label}")
    print(f"dtype       {arch.dtype}")
    print(f"input-len   {args.input_len}")
    print(f"tp          {args.tp}")
    print(f"kv-cache    {cache_total} bytes ({cache_total_gb:.3f} GB) total")
    print(f"per-card    {cache_per_card:.0f} bytes ({cache_per_card_gb:.3f} GB)")
    print(f"posix-bw    {args.posix_bw:.3f} GB/s (total)")
    print(f"h2d-bw      {args.h2d_bw:.3f} GB/s (per-card)")
    print()

    headers = [
        "mode",
        "TTFT(ms)",
        "storage-read(ms)",
        "H2D(ms)",
        "compute(ms)",
        "bottleneck",
    ]
    aligns = ["<", ">", ">", ">", ">", "<"]
    rows = [
        [
            mode,
            f"{result.ttft_ucm_ms:.3f}",
            f"{result.storage_read_ms:.3f}",
            f"{result.h2d_ms:.3f}",
            f"{args.ttft_hbm:.3f}",
            result.bottleneck,
        ]
        for mode, result in (("layered", layered), ("full", full))
    ]
    _render_table(headers, rows, aligns)
    print()

    print(f"vs Full Prefill ({args.ttft_prefill:.1f} ms):")
    for mode, result in (("layered", layered), ("full", full)):
        print(f"  {mode:<9} {_vs_prefill_text(result.ttft_ucm_ms, args.ttft_prefill)}")
    print(f"vs Full HBM ({args.ttft_hbm:.1f} ms):")
    for mode, result in (("layered", layered), ("full", full)):
        print(f"  {mode:<9} {_vs_hbm_text(result.ttft_ucm_ms, args.ttft_hbm)}")


def _render_table(headers: list[str], rows: list[list[str]], aligns: list[str]) -> None:
    widths = [
        max(len(headers[i]), max(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]

    def fmt(cells: list[str]) -> str:
        return "  ".join(
            f"{cells[i]:{aligns[i]}{widths[i]}}" for i in range(len(cells))
        )

    header_line = fmt(headers)
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        print(fmt(row))


def _vs_prefill_text(ttft_ucm_ms: float, ttft_prefill_ms: float) -> str:
    speedup = ttft_prefill_ms / ttft_ucm_ms
    if speedup >= 1:
        return f"{speedup:.2f}x faster"
    return f"{1 / speedup:.2f}x slower"


def _vs_hbm_text(ttft_ucm_ms: float, ttft_hbm_ms: float) -> str:
    ratio = ttft_ucm_ms / ttft_hbm_ms
    return f"{ratio:.2f}x slower"
