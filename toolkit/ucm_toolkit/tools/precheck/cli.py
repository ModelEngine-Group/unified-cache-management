"""precheck command-line interface (UCM environment pre-check, RFC #1208).

Runs a fixed set of environment checks and prints a pass/warn/fail report with
remediation advice for failing items. Mounted as ``ucm-toolkit run precheck``.

Examples::

    ucm-toolkit run precheck                          # all checks, no mount path
    ucm-toolkit run precheck --mount-path /mnt/ucm_cache
    ucm-toolkit run precheck --mount-path /mnt/ucm_cache --quick --engines aio
    ucm-toolkit run precheck --only kernel --only accelerator_driver
    ucm-toolkit run precheck --json --skip-bandwidth
    ucm-toolkit run precheck --config precheck.json --strict
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from . import __version__
from .bandwidth import check_bandwidth
from .checks import (
    CHECK_NAMES,
    check_accelerator_driver,
    check_aio_resources,
    check_kernel_version,
    check_memory_shm,
    check_serving_stack,
    check_uc_manager_version,
)
from .config import PrecheckConfig
from .parseutil import parse_int_list, parse_size_list, parse_str_list
from .reporter import (
    FAIL,
    INFO,
    STATUS_SKIP,
    WARN,
    CheckResult,
    overall_failed,
    render_json,
    render_text,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="precheck",
        description="UCM environment pre-check (versions, kernel, memory, "
        "storage bandwidth).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"precheck {__version__}")
    p.add_argument(
        "--config",
        metavar="FILE",
        help="load thresholds/matrix from a JSON or YAML file " "(YAML needs PyYAML)",
    )
    p.add_argument(
        "--mount-path",
        metavar="PATH",
        help="UCM storage mount point (required for the bandwidth " "benchmark)",
    )

    bw = p.add_argument_group("bandwidth matrix")
    bw.add_argument(
        "--shard-sizes",
        metavar="LIST",
        help="comma-separated shard sizes, e.g. '180k,1m'",
    )
    bw.add_argument(
        "--workers", metavar="LIST", help="comma-separated worker counts, e.g. '1,16'"
    )
    bw.add_argument(
        "--engines",
        metavar="LIST",
        help="comma-separated IO engines, subset of 'psync,aio' (default psync)",
    )
    bw.add_argument(
        "--modes",
        metavar="LIST",
        help="comma-separated phases to run per combo: subset of "
        "'dump,read,mix' (default all; mix = read-heavy 1:rw_ratio)",
    )
    bw.add_argument(
        "--threshold",
        type=float,
        metavar="GB",
        help="minimum best aggregate bandwidth in GB/s "
        f"(default {PrecheckConfig().bandwidth.threshold_gb})",
    )
    bw.add_argument(
        "--block-number", type=int, metavar="N", help="blocks per dump/load sweep"
    )
    bw.add_argument(
        "--dump-epochs", type=int, metavar="N", help="dump sweeps per combo"
    )
    bw.add_argument(
        "--load-epochs", type=int, metavar="N", help="load sweeps per combo"
    )
    bw.add_argument(
        "--mixed-epochs",
        type=int,
        metavar="N",
        help="read-heavy mixed sweeps per combo (1 dump + rw_ratio loads each)",
    )
    bw.add_argument(
        "--rw-ratio",
        type=int,
        metavar="N",
        help="reads per write in the mixed phase (read-heavy; 0 disables mixed, "
        f"default {PrecheckConfig().bandwidth.rw_ratio})",
    )
    bw.add_argument(
        "--quick",
        action="store_true",
        help="halve dump/load/mixed epochs for a faster run",
    )
    bw.add_argument(
        "--skip-bandwidth",
        action="store_true",
        help="skip the (slow) bandwidth benchmark entirely",
    )

    thr = p.add_argument_group("thresholds")
    thr.add_argument(
        "--kernel-min",
        metavar="VER",
        help="minimum kernel version, strictly greater (default '5.10')",
    )
    thr.add_argument(
        "--cuda-min-compute-cap",
        type=float,
        metavar="N",
        help="minimum CUDA compute capability (default 8.0)",
    )
    thr.add_argument(
        "--ascend-min-hdk",
        metavar="VER",
        help="minimum Ascend HDK version, strictly greater " "(default '25.2.0')",
    )

    sel = p.add_argument_group("selection / output")
    sel.add_argument(
        "--only",
        action="append",
        metavar="CHECK",
        help="run only this check (repeatable); choices: " + ", ".join(CHECK_NAMES),
    )
    sel.add_argument(
        "--skip", action="append", metavar="CHECK", help="skip this check (repeatable)"
    )
    sel.add_argument(
        "--strict",
        action="store_true",
        help="treat warnings as failures for the exit code",
    )
    sel.add_argument(
        "--no-color", action="store_true", help="disable ANSI color output"
    )
    sel.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    sel.add_argument("--verbose", action="store_true", help="show extra detail")
    return p


CHECK_FUNCS = {
    "serving_stack": lambda cfg: check_serving_stack(),
    "uc_manager": lambda cfg: check_uc_manager_version(),
    "accelerator_driver": lambda cfg: check_accelerator_driver(cfg),
    "kernel": lambda cfg: check_kernel_version(cfg),
    "memory_shm": lambda cfg: check_memory_shm(cfg),
    "aio_resources": lambda cfg: check_aio_resources(cfg),
    "bandwidth": lambda cfg: check_bandwidth(cfg),
}

# Order: cheap info checks first, hard checks, then the slow bandwidth last.
DEFAULT_ORDER: List[str] = [
    "serving_stack",
    "uc_manager",
    "accelerator_driver",
    "kernel",
    "memory_shm",
    "aio_resources",
    "bandwidth",
]


def _resolve_selection(
    only: Optional[List[str]], skip: Optional[List[str]], skip_bandwidth: bool
) -> List[str]:
    selected = list(DEFAULT_ORDER)
    if only:
        only_lower = [o.lower() for o in only]
        unknown = [o for o in only_lower if o not in CHECK_NAMES]
        if unknown:
            raise SystemExit(
                f"unknown --only check(s): {', '.join(unknown)}; "
                f"choices: {', '.join(CHECK_NAMES)}"
            )
        selected = [o for o in DEFAULT_ORDER if o in only_lower]
    if skip:
        skip_lower = [s.lower() for s in skip]
        unknown = [s for s in skip_lower if s not in CHECK_NAMES]
        if unknown:
            raise SystemExit(
                f"unknown --skip check(s): {', '.join(unknown)}; "
                f"choices: {', '.join(CHECK_NAMES)}"
            )
        selected = [o for o in selected if o not in skip_lower]
    if skip_bandwidth and "bandwidth" in selected:
        selected.remove("bandwidth")
    return selected


def build_config(args: argparse.Namespace) -> PrecheckConfig:
    # Base = shipped precheck.defaults.json (code constants as fallback);
    # a user --config layers on top.
    base = PrecheckConfig.default()
    if args.config:
        base = PrecheckConfig.from_file(args.config)
    engines = parse_str_list(args.engines)
    if engines is not None:
        invalid = [e for e in engines if e not in ("psync", "aio")]
        if invalid:
            raise SystemExit(
                f"invalid engine(s): {', '.join(invalid)}; " "choices: psync, aio"
            )
    modes = parse_str_list(args.modes)
    if modes is not None:
        invalid = [m for m in modes if m not in ("dump", "read", "mix")]
        if invalid:
            raise SystemExit(
                f"invalid mode(s): {', '.join(invalid)}; " "choices: dump, read, mix"
            )
    cfg = base.apply_overrides(
        mount_path=args.mount_path,
        shard_sizes=parse_size_list(args.shard_sizes),
        worker_counts=parse_int_list(args.workers),
        engines=engines,
        modes=modes,
        threshold_gb=args.threshold,
        block_number=args.block_number,
        dump_epochs=args.dump_epochs,
        load_epochs=args.load_epochs,
        mixed_epochs=args.mixed_epochs,
        rw_ratio=args.rw_ratio,
        kernel_min=args.kernel_min,
        cuda_min_compute_cap=args.cuda_min_compute_cap,
        ascend_min_hdk=args.ascend_min_hdk,
        quick=args.quick,
    )
    cfg.verbose = args.verbose
    return cfg


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    # parse_args SystemExit (bad flag / --help) propagates with the argparse
    # convention (2 for usage errors, 0 for --help).
    args = parser.parse_args(argv)

    try:
        cfg = build_config(args)
        selected = _resolve_selection(args.only, args.skip, args.skip_bandwidth)
    except SystemExit as exc:
        # Config/selection usage errors -> exit code 2.
        msg = str(exc.code) if exc.code not in (None, 0, 2) else ""
        print(
            f"error: {msg}" if msg else "error: invalid configuration", file=sys.stderr
        )
        return 2
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    results: List[CheckResult] = []
    for name in selected:
        try:
            res = CHECK_FUNCS[name](cfg)
        except Exception as exc:  # a check must never abort the whole run
            res = CheckResult(
                name=name,
                severity=FAIL,
                status="FAIL",
                value="-",
                detail=f"check raised {type(exc).__name__}: {exc}",
                raw={"error": str(exc)},
            )
        results.append(res)

    color = (not args.no_color) and sys.stdout.isatty()
    if args.json:
        print(render_json(results))
    else:
        print(render_text(results, color=color))

    return 1 if overall_failed(results, strict=args.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
