"""Storage bandwidth benchmark for the posix UCM store.

Reuses the ``posixstore_aio_test.py`` worker pattern (mmap-aligned buffers,
``UcmPipelineStore`` with ``store_pipeline="Posix"``, ``dump_data``/``load_data``
+ ``wait``) but runs a configurable matrix of shard sizes × worker counts × IO
engines and picks the best configuration.

numpy and the ``ucm`` C++ store are imported lazily so that the rest of the
pre-check tool remains importable (and unit-testable) without them. The
benchmark is Linux-only (posix mmap + the ucm ``.so``).
"""

from __future__ import annotations

import importlib.machinery
import multiprocessing
import os
import secrets
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .config import PrecheckConfig
from .reporter import STATUS_PASS, STATUS_SKIP, STATUS_WARN, WARN, CheckResult


@dataclass
class ComboResult:
    shard_size: int
    worker_count: int
    engine: str
    dump_bw: float = 0.0  # mean dump GB/s
    load_bw: float = 0.0  # mean load GB/s
    comprehensive: float = 0.0  # mean(dump, load)
    ok: bool = True
    error: str = ""
    # Stability stats across all epoch samples (populated by _run_combo).
    dump_bws: List[float] = field(default_factory=list)
    load_bws: List[float] = field(default_factory=list)
    dump_std: float = 0.0
    load_std: float = 0.0
    dump_min: float = 0.0
    dump_max: float = 0.0
    load_min: float = 0.0
    load_max: float = 0.0
    n: int = 0  # number of epoch samples per direction
    # Read-heavy mixed workload (1 dump + rw_ratio loads per epoch).
    mixed_bw: float = 0.0  # mean mixed throughput GB/s (business bandwidth)
    mixed_bws: List[float] = field(default_factory=list)
    mixed_std: float = 0.0
    mixed_min: float = 0.0
    mixed_max: float = 0.0
    rw_ratio: int = 0

    def label(self) -> str:
        return (
            f"shard={self.shard_size}B workers={self.worker_count} "
            f"engine={self.engine}"
        )


# ---------------------------------------------------------------------------
# Worker-side helpers (run in spawned processes)
# ---------------------------------------------------------------------------


def _create_worker(device_id: int, params: dict):
    """Build a UcmPipelineStore for one worker. Imports ucm lazily."""
    from ucm.store.factory_v1 import UcmConnectorFactoryV1  # noqa: lazy

    shard = params["shard_size"]
    shard_number = params["shard_number"]
    config = {
        "store_pipeline": "Posix",
        "posix_io_engine": params["engine"],
        "storage_backends": [params["mount_path"]],
        "tensor_size": shard,
        "shard_size": shard,
        # Explicit block_size satisfies the C++ validation
        # (blockSize >= shardSize and blockSize % shardSize == 0);
        # the reference script omitted this and relied on a 0 default.
        "block_size": shard * shard_number,
        "device_id": device_id,
    }
    return UcmConnectorFactoryV1.create_connector(
        "UcmPipelineStore", config, "ucm.store.pipeline.connector"
    )


def _make_array(size: int, alignment: int = 262144):
    """Page-aligned buffer via anonymous mmap (mirrors the reference script)."""
    import mmap

    import numpy as np  # noqa: lazy

    total_bytes = size  # uint8 -> itemsize == 1
    mm = mmap.mmap(-1, total_bytes + alignment)
    raw = np.frombuffer(mm, dtype=np.uint8, count=total_bytes + alignment)
    raw_ptr = raw.__array_interface__["data"][0]
    aligned = (raw_ptr + alignment - 1) & ~(alignment - 1)
    offset = aligned - raw_ptr
    arr = raw[offset : offset + total_bytes].view(dtype=np.uint8)
    return arr, mm


def _dump(epoch, device_id, worker, block_ids, block_ptr, params):
    shard = params["shard_size"]
    shard_number = params["shard_number"]
    block_number = params["block_number"]
    total_size = shard * shard_number * block_number
    costs = []
    for i in range(shard_number):
        idxes = [i for _ in range(block_number)]
        ptrs = [[ptr + i * shard] for ptr in block_ptr]
        t0 = time.perf_counter()
        task = worker.dump_data(block_ids, idxes, ptrs)
        worker.wait(task)
        costs.append(time.perf_counter() - t0)
    total_cost = sum(costs)
    bw = total_size / total_cost / 1e9 if total_cost > 0 else 0.0
    if params.get("verbose"):
        print(
            f"epoch={epoch:03}, worker={device_id:02}, "
            f"dump=[{shard} x {block_number} x {shard_number}], "
            f"cost={total_cost * 1e3:.3f}ms, bw={bw:.3f}GB/s.",
            flush=True,
        )
    return bw


def _load(epoch, device_id, worker, block_ids, block_ptr, params):
    shard = params["shard_size"]
    shard_number = params["shard_number"]
    block_number = params["block_number"]
    total_size = shard * shard_number * block_number
    costs = []
    for i in range(shard_number):
        idxes = [i for _ in range(block_number)]
        ptrs = [[ptr + i * shard] for ptr in block_ptr]
        t0 = time.perf_counter()
        task = worker.load_data(block_ids, idxes, ptrs)
        worker.wait(task)
        costs.append(time.perf_counter() - t0)
    total_cost = sum(costs)
    bw = total_size / total_cost / 1e9 if total_cost > 0 else 0.0
    if params.get("verbose"):
        print(
            f"epoch={epoch:03}, worker={device_id:02}, "
            f"load=[{shard} x {block_number} x {shard_number}], "
            f"cost={total_cost * 1e3:.3f}ms, bw={bw:.3f}GB/s.",
            flush=True,
        )
    return bw


def _mixed(epoch, device_id, worker, block_ids, block_ptr, params):
    """Read-heavy mixed epoch: 1 dump sweep + rw_ratio load sweeps, timed together.

    Loads read the blocks just written (cache-warm), mimicking a real KV-cache
    access pattern (write once, fetch many). The combined throughput =
    (1 + rw_ratio) * sweep_bytes / elapsed.
    """
    shard = params["shard_size"]
    shard_number = params["shard_number"]
    block_number = params["block_number"]
    rw_ratio = params["rw_ratio"]
    sweep_bytes = shard * shard_number * block_number

    def sweep(load):
        for i in range(shard_number):
            idxes = [i for _ in range(block_number)]
            ptrs = [[ptr + i * shard] for ptr in block_ptr]
            task = (
                worker.load_data(block_ids, idxes, ptrs)
                if load
                else worker.dump_data(block_ids, idxes, ptrs)
            )
            worker.wait(task)

    t0 = time.perf_counter()
    sweep(load=False)  # 1 write
    for _ in range(rw_ratio):  # rw_ratio reads
        sweep(load=True)
    elapsed = time.perf_counter() - t0
    bw = (1 + rw_ratio) * sweep_bytes / elapsed / 1e9 if elapsed > 0 else 0.0
    if params.get("verbose"):
        print(
            f"epoch={epoch:03}, worker={device_id:02}, "
            f"mixed(1:{rw_ratio})=[{shard} x {block_number} x {shard_number}], "
            f"cost={elapsed * 1e3:.3f}ms, bw={bw:.3f}GB/s.",
            flush=True,
        )
    return bw


def _worker_loop(device_id, barrier, result_queue, params: dict):
    """Worker process: create store, run phases, push results to the queue.

    Mirrors the original ``posixstore_aio_test.py`` pattern — a simple barrier
    start gate, independent epoch loop, results collected locally and pushed
    once at the end. No ``Manager`` server process (which adds IPC overhead and
    potential deadlock paths under heavy I/O).
    """
    # Silence the ucm/torch C++ layer's stderr noise (FunctionLoader
    # LD_PRELOAD warnings, torch_npu autoload chatter, glog W-lines) by
    # redirecting the actual fd 2 to /dev/null. Using os.dup2 (not just
    # sys.stderr = ...) because C++ code writes to fd 2 directly, bypassing
    # Python's sys.stderr. Our own worker messages go to stdout (not stderr)
    # so they survive. With --verbose, keep stderr visible for debugging.
    if not params.get("verbose"):
        try:
            _devnull_fd = os.open(os.devnull, os.O_WRONLY)
            os.dup2(_devnull_fd, 2)
            os.close(_devnull_fd)
        except Exception:
            pass

    dump_bws: List[float] = []
    load_bws: List[float] = []
    mixed_bws: List[float] = []

    try:
        store = _create_worker(device_id, params)
    except Exception as exc:  # pragma: no cover - depends on ucm build
        result_queue.put(("error", device_id, str(exc)))
        try:
            barrier.abort()
        except Exception:
            pass
        return

    block_number = params["block_number"]
    block_ids = [secrets.token_bytes(16) for _ in range(block_number)]
    shard_number = params["shard_number"]
    shard = params["shard_size"]
    block_data = []
    mmap_handles = []
    for _ in range(block_number):
        arr, mm = _make_array(shard * shard_number)
        block_data.append(arr)
        mmap_handles.append(mm)
    block_ptr = [arr.ctypes.data for arr in block_data]

    rw_ratio = params.get("rw_ratio", 0)
    mixed_epochs = params.get("mixed_epochs", 0)
    modes = params.get("modes", [])
    run_dump = "dump" in modes
    run_load = "read" in modes
    run_mixed = "mix" in modes and rw_ratio > 0 and mixed_epochs > 0
    btimeout = params.get("barrier_timeout", _DEFAULT_BARRIER_TIMEOUT)
    try:
        # Start gate: all workers begin I/O concurrently. A peer that failed
        # store creation calls barrier.abort(), waking survivors instantly.
        if not _barrier_wait(barrier, device_id, btimeout):
            result_queue.put(("error", device_id, "barrier broken"))
            return
        # Per-epoch barriers (mirrors the original posixstore_aio_test.py):
        # all workers sync after each epoch so the kernel aio queue is drained
        # before the next burst of submissions. Without this, independent
        # workers can overflow the kernel's aio-nr limit at high worker counts,
        # causing worker.wait(task) to block in uninterruptible D state.
        if run_dump:
            for epoch in range(params["dump_epochs"]):
                dump_bws.append(
                    _dump(epoch, device_id, store, block_ids, block_ptr, params)
                )
                if not _barrier_wait(barrier, device_id, btimeout):
                    result_queue.put(("error", device_id, "barrier broken during dump"))
                    return
        if run_load:
            for epoch in range(params["load_epochs"]):
                load_bws.append(
                    _load(epoch, device_id, store, block_ids, block_ptr, params)
                )
                if not _barrier_wait(barrier, device_id, btimeout):
                    result_queue.put(("error", device_id, "barrier broken during load"))
                    return
        if run_mixed:
            for epoch in range(mixed_epochs):
                mixed_bws.append(
                    _mixed(epoch, device_id, store, block_ids, block_ptr, params)
                )
                if not _barrier_wait(barrier, device_id, btimeout):
                    result_queue.put(
                        ("error", device_id, "barrier broken during mixed")
                    )
                    return
        result_queue.put(
            ("ok", device_id, {"dump": dump_bws, "load": load_bws, "mixed": mixed_bws})
        )
    except Exception as exc:  # pragma: no cover - depends on env
        result_queue.put(("error", device_id, str(exc)))
        try:
            barrier.abort()
        except Exception:
            pass
    finally:
        for mm in mmap_handles:
            try:
                mm.close()
            except Exception:
                pass


# Backstop: max seconds to wait at the start barrier. Normally a peer that
# fails store creation calls barrier.abort(), which wakes survivors instantly;
# this timeout only bites if abort() never reaches the survivors. Configurable
# via precheck.defaults.json (bandwidth.barrier_timeout).
_DEFAULT_BARRIER_TIMEOUT = 60


def _barrier_wait(barrier, device_id: int, timeout: int) -> bool:
    """Wait at the barrier with a timeout. Return False if it breaks/times out."""
    try:
        barrier.wait(timeout=timeout)
        return True
    except Exception as exc:  # BrokenBarrierError on timeout or a peer aborting
        print(f"[worker {device_id}] barrier broken: {exc}", flush=True)
        return False


def _terminate_proc(p):
    """Terminate a hung worker process: SIGTERM, then SIGKILL as fallback."""
    for method in ("terminate", "kill"):
        try:
            getattr(p, method)()
            p.join(timeout=5)
        except Exception:
            pass
        if not p.is_alive():
            break


# ---------------------------------------------------------------------------
# Matrix runner
# ---------------------------------------------------------------------------


def _run_combo(
    shard: int, workers: int, engine: str, cfg: PrecheckConfig
) -> ComboResult:
    """Run one (shard, workers, engine) benchmark and return its result."""
    bw = cfg.bandwidth
    modes = bw.modes
    params = {
        "shard_size": shard,
        "shard_number": bw.shard_number,
        "block_number": bw.block_number,
        "dump_epochs": bw.dump_epochs,
        "load_epochs": bw.load_epochs,
        "mixed_epochs": bw.mixed_epochs,
        "rw_ratio": bw.rw_ratio,
        "engine": engine,
        "mount_path": cfg.mount_path,
        "modes": modes,
        "barrier_timeout": bw.barrier_timeout,
        "combo_timeout": bw.combo_timeout,
        "verbose": cfg.verbose,
    }
    combo = ComboResult(
        shard_size=shard,
        worker_count=workers,
        engine=engine,
        rw_ratio=bw.rw_ratio if ("mix" in modes and bw.mixed_epochs > 0) else 0,
    )
    try:
        result_queue: multiprocessing.Queue = multiprocessing.Queue()
        barrier = multiprocessing.Barrier(workers)
        procs = []
        for i in range(workers):
            p = multiprocessing.Process(
                target=_worker_loop,
                args=(i, barrier, result_queue, params),
            )
            procs.append(p)
            p.start()
        combo_timeout = params.get("combo_timeout", 300)
        deadline = time.perf_counter() + combo_timeout
        timed_out = False
        for p in procs:
            remaining = max(1.0, deadline - time.perf_counter())
            p.join(timeout=remaining)
            if p.is_alive():
                timed_out = True
                _terminate_proc(p)
        if timed_out:
            combo.ok = False
            combo.error = (
                f"combo timed out after {combo_timeout}s "
                f"(a worker may be stuck on aio I/O; try psync or "
                f"raise fs.aio-max-nr)"
            )
            return combo
        # Drain the queue — each worker pushed one result dict at the end.
        # No Manager server process: workers collected samples locally and
        # pushed once, minimising IPC (mirrors the original test script).
        run_dump = "dump" in modes
        run_load = "read" in modes
        run_mixed = "mix" in modes and bw.rw_ratio > 0 and bw.mixed_epochs > 0
        dumps: List[float] = []
        loads: List[float] = []
        mixeds: List[float] = []
        errors: List[str] = []
        while not result_queue.empty():
            try:
                tag, _wid, payload = result_queue.get_nowait()
            except Exception:
                break
            if tag == "error":
                errors.append(str(payload))
                continue
            if tag == "ok":
                dumps.extend(payload.get("dump", []))
                loads.extend(payload.get("load", []))
                mixeds.extend(payload.get("mixed", []))
        if not (run_dump or run_load or run_mixed):
            combo.ok = False
            combo.error = "no bandwidth modes selected"
            return combo
        if run_dump:
            if not dumps:
                combo.ok = False
                combo.error = "dump phase produced no samples"
                return combo
            combo.dump_bw = _mean(dumps)
            combo.dump_bws = dumps
            combo.dump_std = _std(dumps)
            combo.dump_min, combo.dump_max = _minmax(dumps)
        if run_load:
            if not loads:
                combo.ok = False
                combo.error = "read phase produced no samples"
                return combo
            combo.load_bw = _mean(loads)
            combo.load_bws = loads
            combo.load_std = _std(loads)
            combo.load_min, combo.load_max = _minmax(loads)
        # comprehensive only meaningful when both dump and read ran.
        if run_dump and run_load:
            combo.comprehensive = (combo.dump_bw + combo.load_bw) / 2.0
        combo.n = len(dumps) or len(loads) or len(mixeds)
        if run_mixed:
            if not mixeds:
                combo.ok = False
                combo.error = "mixed phase produced no samples"
                return combo
            combo.mixed_bw = _mean(mixeds)
            combo.mixed_bws = mixeds
            combo.mixed_std = _std(mixeds)
            combo.mixed_min, combo.mixed_max = _minmax(mixeds)
    except Exception as exc:  # pragma: no cover - depends on env
        combo.ok = False
        combo.error = f"{type(exc).__name__}: {exc}"
    return combo


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: List[float]) -> float:
    """Population standard deviation (pure-python; main process has no numpy)."""
    if not xs:
        return 0.0
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def _minmax(xs: List[float]) -> Tuple[float, float]:
    return (min(xs), max(xs)) if xs else (0.0, 0.0)


# Each metric: (attribute on ComboResult, display label).
_METRICS = [
    ("dump_bw", "dump"),
    ("load_bw", "load"),
    ("comprehensive", "comprehensive"),
    ("mixed_bw", "mixed"),
]


def headline_bw(c: ComboResult) -> float:
    """Aggregate store throughput across all workers (prefers mixed, then
    comprehensive, then dump/load).  Used for the single "best" combo in raw
    output.  Per-metric status uses :func:`best_per_metric` instead."""
    if c.mixed_bw > 0:
        per_worker = c.mixed_bw
    elif c.comprehensive > 0:
        per_worker = c.comprehensive
    elif c.dump_bw > 0:
        per_worker = c.dump_bw
    elif c.load_bw > 0:
        per_worker = c.load_bw
    else:
        per_worker = 0.0
    return per_worker * c.worker_count


def pick_best(combos: List[ComboResult]) -> Optional[ComboResult]:
    """Return the combo with the highest headline bandwidth (ok only)."""
    ok = [c for c in combos if c.ok]
    return max(ok, key=headline_bw) if ok else None


def best_per_metric(
    combos: List[ComboResult],
) -> List[Tuple[str, ComboResult, float]]:
    """For each metric that ran, return ``(label, best_combo, aggregate_gb)``.

    The best combo for a metric is the one with the highest aggregate
    (per-worker BW × worker count) among ok combos.  Different metrics may
    have different best combos.
    """
    result: List[Tuple[str, ComboResult, float]] = []
    for attr, label in _METRICS:
        ok = [c for c in combos if c.ok and getattr(c, attr, 0.0) > 0]
        if ok:
            best = max(ok, key=lambda c: getattr(c, attr, 0.0) * c.worker_count)
            agg = getattr(best, attr, 0.0) * best.worker_count
            result.append((label, best, agg))
    return result


def bandwidth_status(
    metric_best: List[Tuple[str, ComboResult, float]], threshold: float
) -> str:
    """PASS only if every metric's best aggregate meets the threshold."""
    if not metric_best:
        return STATUS_WARN
    return (
        STATUS_PASS
        if all(agg >= threshold for _, _, agg in metric_best)
        else STATUS_WARN
    )


def bandwidth_detail(
    metric_best: List[Tuple[str, ComboResult, float]], threshold: float
) -> str:
    if not metric_best:
        return "no benchmark combo completed successfully"
    lines = []
    for label, combo, agg in metric_best:
        mark = "PASS" if agg >= threshold else "WARN"
        lines.append(f"  {label:15s} best={agg:.3f} GB/s ({combo.label()})  [{mark}]")
    return "\n".join(lines)


def _format_bw(x: float) -> str:
    return f"{x:.3f}"


def _render_matrix(combos: List[ComboResult]) -> str:
    """Detailed per-combo bandwidth report: per-worker mean/std/min/max + aggregate.
    Only phases that actually ran (dump/read/mix per --modes) are printed."""
    lines = ["bandwidth matrix:"]
    for c in combos:
        if not c.ok:
            lines.append(f"{c.label()}  ERR: {c.error}")
            continue
        w = c.worker_count
        lines.append(
            f"{_human_bytes(c.shard_size)}  workers={w}  engine={c.engine}  "
            f"(n={c.n} samples/direction, per-worker below)"
        )
        agg_parts = []
        if c.dump_bw > 0:
            lines.append(
                f"  dump  avg={c.dump_bw:.3f}  std={c.dump_std:.3f}  "
                f"min={c.dump_min:.3f}  max={c.dump_max:.3f}  GB/s"
            )
            agg_parts.append(f"dump={c.dump_bw * w:.3f}")
        if c.load_bw > 0:
            lines.append(
                f"  load  avg={c.load_bw:.3f}  std={c.load_std:.3f}  "
                f"min={c.load_min:.3f}  max={c.load_max:.3f}  GB/s"
            )
            agg_parts.append(f"load={c.load_bw * w:.3f}")
        if c.comprehensive > 0:
            lines.append(f"  comprehensive = {c.comprehensive:.3f} GB/s")
            agg_parts.append(f"comp={c.comprehensive * w:.3f}")
        if c.mixed_bw > 0:
            lines.append(
                f"  mixed avg={c.mixed_bw:.3f}  std={c.mixed_std:.3f}  "
                f"min={c.mixed_min:.3f}  max={c.mixed_max:.3f}  GB/s  "
                f"(read-heavy 1:{c.rw_ratio})"
            )
            agg_parts.append(f"mixed={c.mixed_bw * w:.3f}")
        if agg_parts:
            lines.append(f"  aggregate ({w}w): " + "  ".join(agg_parts) + "  GB/s")
    return "\n".join(lines)


def _human_bytes(n: int) -> str:
    for unit, factor in (("GiB", 1024**3), ("MiB", 1024**2), ("KiB", 1024)):
        if n >= factor and n % factor == 0:
            return f"{n // factor}{unit}"
    return f"{n}B"


def _path_resolves_to_repo_root(path: str, repo_root: str) -> bool:
    """Return whether a sys.path entry points at the ucm repo root."""
    try:
        candidate = os.path.realpath(path or os.getcwd())
    except OSError:
        return False
    return candidate == repo_root


def _read_aio_limits() -> Tuple[int, int]:
    """Read kernel aio limits from /proc/sys/fs/.

    Returns (aio_max_nr, aio_nr). Both 0 if the files don't exist (non-Linux
    or aio not enabled).
    """
    try:
        with open("/proc/sys/fs/aio-max-nr") as f:
            max_nr = int(f.read().strip())
        with open("/proc/sys/fs/aio-nr") as f:
            nr = int(f.read().strip())
        return max_nr, nr
    except (OSError, ValueError):
        return 0, 0


# ---------------------------------------------------------------------------
# Engine probe (avoids indefinite hang on hosts where aio blocks)
# ---------------------------------------------------------------------------


def _probe_worker(params: dict):
    """Probe worker: create store, dump one shard sweep, exit 0 on success.

    Mirrors the original ``posixstore_aio_test.py`` dump call: ``block_number``
    block IDs, ``idxes`` and ``ptrs`` lists of equal length, each ``ptr`` wrapped
    in a list (``[[ptr], [ptr], ...]``).
    """
    if not params.get("verbose"):
        try:
            _devnull_fd = os.open(os.devnull, os.O_WRONLY)
            os.dup2(_devnull_fd, 2)
            os.close(_devnull_fd)
        except Exception:
            pass
    try:
        store = _create_worker(0, params)
        shard = params["shard_size"]
        block_number = min(params["block_number"], 4)
        block_ids = [secrets.token_bytes(16) for _ in range(block_number)]
        mmap_handles = []
        block_ptr = []
        for _ in range(block_number):
            arr, mm = _make_array(shard)
            block_ptr.append(arr.ctypes.data)
            mmap_handles.append(mm)
        idxes = [0 for _ in range(block_number)]
        ptrs = [[ptr] for ptr in block_ptr]
        task = store.dump_data(block_ids, idxes, ptrs)
        store.wait(task)
    except Exception:
        sys.exit(1)
    finally:
        for mm in mmap_handles:
            try:
                mm.close()
            except Exception:
                pass
    sys.exit(0)


def _probe_engine(
    engine: str, mount_path: str, shard_size: int, block_number: int, timeout: int = 15
) -> bool:
    """Quick probe: can this engine create a store and complete one dump?

    Returns True if usable, False if it blocks or fails.  This prevents the
    indefinite hang that occurs when ``worker.wait(task)`` blocks on an
    uninterruptible kernel aio syscall (D state — even SIGKILL cannot
    interrupt it).
    """
    params = {
        "shard_size": shard_size,
        "shard_number": 1,
        "block_number": min(block_number, 4),
        "engine": engine,
        "mount_path": mount_path,
    }
    try:
        p = multiprocessing.Process(target=_probe_worker, args=(params,))
        p.start()
        p.join(timeout=timeout)
        if p.is_alive():
            _terminate_proc(p)
            return False
        return p.exitcode == 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Public check entry
# ---------------------------------------------------------------------------


def check_bandwidth(cfg: PrecheckConfig) -> CheckResult:
    """Run the bandwidth matrix; warn if the best comprehensive BW < threshold."""
    if os.name != "posix":
        return CheckResult(
            name="bandwidth",
            severity=WARN,
            status=STATUS_SKIP,
            value="-",
            detail="skipped: bandwidth benchmark is Linux/posix-only",
            remediation="run on a Linux host (the ucm posix store needs posix mmap)",
            raw={"platform": os.name},
        )
    if not cfg.mount_path:
        return CheckResult(
            name="bandwidth",
            severity=WARN,
            status=STATUS_SKIP,
            value="-",
            detail="skipped: provide --mount-path (or mount_path in config) to run",
            remediation="set --mount-path to the UCM cache mount point",
            raw={},
        )
    if not os.path.isdir(cfg.mount_path):
        return CheckResult(
            name="bandwidth",
            severity=WARN,
            status=STATUS_WARN,
            value="-",
            threshold=f">= {cfg.bandwidth.threshold_gb} GB/s",
            detail=f"mount path does not exist or is not a directory: "
            f"{cfg.mount_path}",
            remediation="point --mount-path at an existing writable directory",
            raw={"mount_path": cfg.mount_path},
        )

    # Suppress C++ noise: prevent torch_npu autoload (FunctionLoader warnings,
    # which can interfere with the posix store in container environments where
    # the NPU driver is only partially mounted) and glog W-lines. The posix
    # store uses libaio directly and does not need torch_npu. Worker stderr is
    # also redirected to /dev/null via os.dup2 (see _worker_loop).
    for _k, _v in (
        ("TORCH_DEVICE_BACKEND_AUTOLOAD", "0"),
        ("GLOG_minloglevel", "2"),
        ("GLOG_stderrthreshold", "2"),
    ):
        os.environ.setdefault(_k, _v)

    # Verify ucm is importable before spawning workers.
    # The source-tree ucm/ (incomplete — missing compiled artifacts like
    # ucmlogger) can shadow the installed package at site-packages when the
    # repo root is on sys.path (CWD, PYTHONPATH, or an editable-install .pth).
    # The ucm_patch.pth boot hook (installed with ucm) imports
    # ucm.integration.* at Python startup; with the repo root on sys.path that
    # boot import loads the INCOMPLETE source tree and leaves its submodules
    # cached in sys.modules.  So: drop repo-root entries from sys.path AND
    # purge any cached ucm.* submodules (only if ucm is still importable
    # without the repo root), forcing a fresh import from site-packages.
    # Workers (forked) inherit the fix.
    _repo_root = os.path.realpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), *([".."] * 4))
    )
    _saved_path = sys.path[:]
    _clean_path = [
        p for p in sys.path if not _path_resolves_to_repo_root(p, _repo_root)
    ]
    if len(_clean_path) != len(sys.path) and (
        importlib.machinery.PathFinder.find_spec("ucm", _clean_path) is not None
    ):
        sys.path[:] = _clean_path
        for _m in [k for k in list(sys.modules) if k == "ucm" or k.startswith("ucm.")]:
            del sys.modules[_m]
    try:
        import ucm  # noqa: F401
    except Exception as exc:
        sys.path[:] = _saved_path  # restore for error handling
        # Distinguish "ucm not installed" from "installed but import fails"
        # (e.g. the source tree shadows site-packages, or a dependency like
        # wrapt is missing).
        ucm_installed = True
        try:
            from importlib.metadata import version as _pkg_version

            _pkg_version("uc-manager")
        except Exception:
            ucm_installed = False
        if ucm_installed:
            detail = (
                f"ucm is installed (site-packages) but import failed "
                f"({type(exc).__name__}: {exc}); "
                f"if running from the ucm source tree, the source ucm/ "
                f"may shadow the installed package — try running from a "
                f"different directory"
            )
            remediation = (
                f"fix the import error ({exc}); "
                f"common cause: source tree shadows installed package "
                f"or missing dependency (pip install wrapt)"
            )
        else:
            detail = f"ucm not installed ({type(exc).__name__}: {exc})"
            remediation = "install ucm (with the built C++ posix store) to benchmark"
        return CheckResult(
            name="bandwidth",
            severity=WARN,
            status=STATUS_SKIP,
            value="-",
            threshold=f">= {cfg.bandwidth.threshold_gb} GB/s",
            detail=detail,
            remediation=remediation,
            raw={"mount_path": cfg.mount_path},
        )

    bw = cfg.bandwidth
    # Probe engines: aio may fail or block on some hosts.  Probe each non-
    # psync engine with a single-worker, single-block dump; skip engines that
    # fail or block.
    probed_engines: List[str] = []
    for engine in bw.engines:
        if engine == "psync":
            probed_engines.append(engine)
            continue
        print(f"[bw] probing {engine}...", flush=True)
        if _probe_engine(engine, cfg.mount_path, bw.shard_sizes[0], bw.block_number):
            probed_engines.append(engine)
        else:
            print(
                f"[bw] {engine} probe failed — skipping {engine} combos "
                f"(engine may block on this disk; use psync)",
                flush=True,
            )
    if not probed_engines:
        probed_engines = ["psync"]

    # Check aio resource limits before the matrix. Each aio store context
    # calls io_setup(queue_depth) which reserves queue_depth events from the
    # kernel's aio pool (aio-max-nr). If N workers × queue_depth exceeds the
    # available pool (aio-max-nr - aio-nr), the later workers' io_setup fails
    # with EAGAIN and the combo deadlocks. Cap the aio worker count to what
    # the kernel can actually support and warn if the user requested more.
    aio_max_nr, aio_nr = _read_aio_limits()
    max_aio_workers = 0  # 0 = unknown / unlimited (let it try)
    if aio_max_nr > 0 and "aio" in probed_engines:
        available = aio_max_nr - aio_nr
        qd = bw.aio_queue_depth
        max_aio_workers = max(0, available // qd) if qd > 0 else 0
        max_requested = max(bw.worker_counts)
        if max_aio_workers < max_requested:
            print(
                f"[bw] aio: {available} events available "
                f"(aio-max-nr={aio_max_nr}, aio-nr={aio_nr}), "
                f"each context needs {qd}, "
                f"max {max_aio_workers} aio workers "
                f"(requested up to {max_requested}); "
                f"raise aio-max-nr: echo 1048576 > /proc/sys/fs/aio-max-nr",
                flush=True,
            )

    combos: List[ComboResult] = []
    for shard in bw.shard_sizes:
        for workers in bw.worker_counts:
            for engine in probed_engines:
                if (
                    engine != "psync"
                    and max_aio_workers > 0
                    and workers > max_aio_workers
                ):
                    combos.append(
                        ComboResult(
                            shard_size=shard,
                            worker_count=workers,
                            engine=engine,
                            ok=False,
                            error=(
                                f"skipped: only {max_aio_workers} aio workers "
                                f"available (aio-max-nr={aio_max_nr})"
                            ),
                        )
                    )
                    continue
                print(
                    f"[bw] shard={_human_bytes(shard)} workers={workers} "
                    f"engine={engine} ...",
                    flush=True,
                )
                combo = _run_combo(shard, workers, engine, cfg)
                combos.append(combo)

    metric_best = best_per_metric(combos)

    matrix_str = _render_matrix(combos)
    threshold = bw.threshold_gb
    print("\n" + matrix_str, flush=True)

    raw = {
        "mount_path": cfg.mount_path,
        "engines_probed": probed_engines,
        "engines_skipped": [e for e in bw.engines if e not in probed_engines],
        "matrix": [
            {
                "shard_size": c.shard_size,
                "worker_count": c.worker_count,
                "engine": c.engine,
                "dump_gb": round(c.dump_bw, 3),
                "dump_std": round(c.dump_std, 3),
                "dump_min": round(c.dump_min, 3),
                "dump_max": round(c.dump_max, 3),
                "load_gb": round(c.load_bw, 3),
                "load_std": round(c.load_std, 3),
                "load_min": round(c.load_min, 3),
                "load_max": round(c.load_max, 3),
                "comprehensive_gb": round(c.comprehensive, 3),
                "mixed_gb": round(c.mixed_bw, 3),
                "mixed_std": round(c.mixed_std, 3),
                "mixed_min": round(c.mixed_min, 3),
                "mixed_max": round(c.mixed_max, 3),
                "rw_ratio": c.rw_ratio,
                "n": c.n,
                "ok": c.ok,
                "error": c.error,
                "dump_series": [round(v, 3) for v in c.dump_bws],
                "load_series": [round(v, 3) for v in c.load_bws],
                "mixed_series": [round(v, 3) for v in c.mixed_bws],
            }
            for c in combos
        ],
        "threshold_gb": threshold,
    }

    if not metric_best:
        sys.path[:] = _saved_path
        return CheckResult(
            name="bandwidth",
            severity=WARN,
            status=STATUS_WARN,
            value="-",
            threshold=f">= {threshold} GB/s",
            detail="no benchmark combo completed successfully",
            remediation="check ucm store health and that the C++ posix .so is built",
            raw=raw,
        )

    raw["best_per_metric"] = [
        {
            "metric": label,
            "aggregate_gb": round(agg, 3),
            "shard_size": combo.shard_size,
            "worker_count": combo.worker_count,
            "engine": combo.engine,
        }
        for label, combo, agg in metric_best
    ]

    value = (
        "  ".join(f"{label}={_format_bw(agg)}" for label, _, agg in metric_best)
        + " GB/s"
    )
    status = bandwidth_status(metric_best, threshold)
    detail = bandwidth_detail(metric_best, threshold)
    failed = [label for label, _, agg in metric_best if agg < threshold]
    remediation = ""
    if failed:
        remediation = (
            f"metrics below threshold ({', '.join(failed)}): "
            "check the mount-point disk/NVMe throughput; try the aio engine, "
            "raise --workers, or confirm io_direct/O_DIRECT settings"
        )
    sys.path[:] = _saved_path
    return CheckResult(
        name="bandwidth",
        severity=WARN,
        status=status,
        value=value,
        threshold=f">= {threshold} GB/s",
        detail=detail,
        remediation=remediation,
        raw=raw,
    )
