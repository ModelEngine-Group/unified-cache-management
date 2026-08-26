"""Configuration for the pre-check tool.

All tunable thresholds and the bandwidth matrix live in a shipped JSON file
(``precheck.defaults.json`` next to this module), so a version update is a
data-only change — no code edit needed. :class:`PrecheckConfig` layers, in
increasing precedence: code constants (fallback if the JSON is absent) -> the
shipped defaults JSON -> a ``--config`` user file -> CLI flags.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import List, Optional

from .parseutil import parse_size

# --- Fallback defaults (used only when precheck.defaults.json is missing) ---
DEFAULT_SHARD_SIZES: List[int] = [184320, 8388608]  # 180 KiB, 8 MiB
DEFAULT_WORKER_COUNTS: List[int] = [1, 8, 16]
DEFAULT_ENGINES: List[str] = ["psync", "aio"]
DEFAULT_MODES: List[str] = ["dump", "read", "mix"]

DEFAULT_KERNEL_MIN: str = "5.10"
DEFAULT_CUDA_MIN_COMPUTE_CAP: float = 8.0
DEFAULT_ASCEND_MIN_HDK: str = "25.2.0"
DEFAULT_SHM_MIN_GIB: float = 512.0
DEFAULT_BANDWIDTH_THRESHOLD_GB: float = 8.0

DEFAULT_BLOCK_NUMBER: int = 32
DEFAULT_DUMP_EPOCHS: int = 8
DEFAULT_LOAD_EPOCHS: int = 8
DEFAULT_SHARD_NUMBER: int = 1
DEFAULT_MIXED_EPOCHS: int = 8
DEFAULT_RW_RATIO: int = 4
DEFAULT_BARRIER_TIMEOUT: int = 60
DEFAULT_COMBO_TIMEOUT: int = 120
DEFAULT_AIO_QUEUE_DEPTH: int = 4096

_DEFAULTS_FILE = Path(__file__).resolve().parent / "precheck.defaults.json"


@dataclass
class BandwidthConfig:
    shard_sizes: List[int] = field(default_factory=lambda: list(DEFAULT_SHARD_SIZES))
    worker_counts: List[int] = field(
        default_factory=lambda: list(DEFAULT_WORKER_COUNTS)
    )
    engines: List[str] = field(default_factory=lambda: list(DEFAULT_ENGINES))
    modes: List[str] = field(default_factory=lambda: list(DEFAULT_MODES))
    block_number: int = DEFAULT_BLOCK_NUMBER
    shard_number: int = DEFAULT_SHARD_NUMBER
    dump_epochs: int = DEFAULT_DUMP_EPOCHS
    load_epochs: int = DEFAULT_LOAD_EPOCHS
    mixed_epochs: int = DEFAULT_MIXED_EPOCHS
    rw_ratio: int = DEFAULT_RW_RATIO
    barrier_timeout: int = DEFAULT_BARRIER_TIMEOUT
    combo_timeout: int = DEFAULT_COMBO_TIMEOUT
    aio_queue_depth: int = DEFAULT_AIO_QUEUE_DEPTH
    threshold_gb: float = DEFAULT_BANDWIDTH_THRESHOLD_GB


@dataclass
class PrecheckConfig:
    mount_path: Optional[str] = None
    kernel_min: str = DEFAULT_KERNEL_MIN
    cuda_min_compute_cap: float = DEFAULT_CUDA_MIN_COMPUTE_CAP
    ascend_min_hdk: str = DEFAULT_ASCEND_MIN_HDK
    shm_min_gib: float = DEFAULT_SHM_MIN_GIB
    bandwidth: BandwidthConfig = field(default_factory=BandwidthConfig)
    verbose: bool = False

    # ----- file loading -----

    @classmethod
    def _load_defaults_dict(cls) -> dict:
        """Read the shipped precheck.defaults.json; {} if missing/unreadable."""
        try:
            with open(_DEFAULTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f) or {}
        except (OSError, ValueError):
            return {}

    @classmethod
    def default(cls) -> "PrecheckConfig":
        """Runtime defaults: code constants merged with the shipped JSON."""
        return cls.from_dict(cls._load_defaults_dict())

    @classmethod
    def from_dict(
        cls, data: dict, base: Optional["PrecheckConfig"] = None
    ) -> "PrecheckConfig":
        cfg = base if base is not None else cls()
        if not isinstance(data, dict):
            return cfg
        cfg.mount_path = data.get("mount_path", cfg.mount_path)
        cfg.kernel_min = str(data.get("kernel_min", cfg.kernel_min))
        cfg.cuda_min_compute_cap = float(
            data.get("cuda_min_compute_cap", cfg.cuda_min_compute_cap)
        )
        cfg.ascend_min_hdk = str(data.get("ascend_min_hdk", cfg.ascend_min_hdk))
        cfg.shm_min_gib = float(data.get("shm_min_gib", cfg.shm_min_gib))

        bw = data.get("bandwidth", {})
        if isinstance(bw, dict):
            b = cfg.bandwidth
            if "shard_sizes" in bw:
                b.shard_sizes = [parse_size(x) for x in bw["shard_sizes"]]
            if "worker_counts" in bw:
                b.worker_counts = [int(x) for x in bw["worker_counts"]]
            if "engines" in bw:
                b.engines = [str(x) for x in bw["engines"]]
            if "modes" in bw:
                b.modes = [str(x) for x in bw["modes"]]
            if "block_number" in bw:
                b.block_number = int(bw["block_number"])
            if "shard_number" in bw:
                b.shard_number = int(bw["shard_number"])
            if "dump_epochs" in bw:
                b.dump_epochs = int(bw["dump_epochs"])
            if "load_epochs" in bw:
                b.load_epochs = int(bw["load_epochs"])
            if "mixed_epochs" in bw:
                b.mixed_epochs = int(bw["mixed_epochs"])
            if "rw_ratio" in bw:
                b.rw_ratio = int(bw["rw_ratio"])
            if "barrier_timeout" in bw:
                b.barrier_timeout = int(bw["barrier_timeout"])
            if "combo_timeout" in bw:
                b.combo_timeout = int(bw["combo_timeout"])
            if "aio_queue_depth" in bw:
                b.aio_queue_depth = int(bw["aio_queue_depth"])
            if "threshold_gb" in bw:
                b.threshold_gb = float(bw["threshold_gb"])
        return cfg

    @classmethod
    def from_file(cls, path: str) -> "PrecheckConfig":
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        ext = os.path.splitext(path)[1].lower()
        if ext in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "YAML config requires PyYAML; install it or use a .json "
                    "config file."
                ) from exc
            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text) if text.strip() else {}
        # Layer the user file on top of the shipped defaults.
        return cls.from_dict(data, base=cls.default())

    # ----- CLI overrides -----

    def apply_overrides(
        self,
        mount_path: Optional[str] = None,
        shard_sizes: Optional[List[int]] = None,
        worker_counts: Optional[List[int]] = None,
        engines: Optional[List[str]] = None,
        modes: Optional[List[str]] = None,
        threshold_gb: Optional[float] = None,
        block_number: Optional[int] = None,
        dump_epochs: Optional[int] = None,
        load_epochs: Optional[int] = None,
        mixed_epochs: Optional[int] = None,
        rw_ratio: Optional[int] = None,
        kernel_min: Optional[str] = None,
        cuda_min_compute_cap: Optional[float] = None,
        ascend_min_hdk: Optional[str] = None,
        shm_min_gib: Optional[float] = None,
        quick: bool = False,
    ) -> "PrecheckConfig":
        cfg = replace(self)
        if mount_path is not None:
            cfg.mount_path = mount_path
        if kernel_min is not None:
            cfg.kernel_min = kernel_min
        if cuda_min_compute_cap is not None:
            cfg.cuda_min_compute_cap = cuda_min_compute_cap
        if ascend_min_hdk is not None:
            cfg.ascend_min_hdk = ascend_min_hdk
        if shm_min_gib is not None:
            cfg.shm_min_gib = shm_min_gib

        b = replace(cfg.bandwidth)
        if shard_sizes is not None:
            b.shard_sizes = shard_sizes
        if worker_counts is not None:
            b.worker_counts = worker_counts
        if engines is not None:
            b.engines = engines
        if modes is not None:
            b.modes = modes
        if threshold_gb is not None:
            b.threshold_gb = threshold_gb
        if block_number is not None:
            b.block_number = block_number
        if dump_epochs is not None:
            b.dump_epochs = dump_epochs
        if load_epochs is not None:
            b.load_epochs = load_epochs
        if mixed_epochs is not None:
            b.mixed_epochs = mixed_epochs
        if rw_ratio is not None:
            b.rw_ratio = rw_ratio
        if quick:
            # Cut epochs in half for a faster (still representative) run.
            b.dump_epochs = max(1, b.dump_epochs // 2)
            b.load_epochs = max(1, b.load_epochs // 2)
            b.mixed_epochs = max(1, b.mixed_epochs // 2)
        cfg.bandwidth = b
        return cfg
