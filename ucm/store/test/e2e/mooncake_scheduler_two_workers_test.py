# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
"""Mooncake ``UcmMooncakeStoreV1`` e2e: 1 scheduler + 2 workers.

Key reliability behaviors:

- If **any one child fails / exits non-zero**, the parent IMMEDIATELY terminates
  the remaining children (no 5-min barrier wait).
- Parent installs signal (SIGINT/SIGTERM) + ``atexit`` handlers so Ctrl+C reaps
  every child; still-alive children get ``SIGKILL``.
- Parent is set as a process group leader (``os.setpgrp``) on Linux, so a single
  ``kill -TERM -<PGID>`` can take everything down.
- Child wraps ``store.shutdown()`` in a timeout-guarded thread and exits via
  ``os._exit`` to avoid hanging in Mooncake / Ascend native teardown.

Run (no pytest)::

    PYTHONPATH=. python ucm/store/test/e2e/mooncake_scheduler_two_workers_test.py

Env (all optional):

- ``UCM_MOONCAKE_MASTER``             default 127.0.0.1:50088
- ``UCM_MOONCAKE_METADATA``           default P2PHANDSHAKE
- ``UCM_MOONCAKE_LOCAL_HOSTNAME``     default 127.0.0.1
- ``UCM_MOONCAKE_GLOBAL_SEGMENT``     default 256MB
- ``UCM_MOONCAKE_LOCAL_BUFFER``       default 256MB
- ``UCM_MOONCAKE_TEST_BARRIER_TIMEOUT`` default 300  (seconds)
- ``UCM_MOONCAKE_TEST_JOIN_TIMEOUT``    default 600  (seconds, per child)
- ``UCM_MOONCAKE_TEST_SHUTDOWN_TIMEOUT`` default 60  (seconds, child-side)
"""
from __future__ import annotations

import atexit
import multiprocessing
import os
import secrets
import signal
import sys
import threading
import time
import traceback
from threading import BrokenBarrierError
from typing import Any, Optional

import numpy as np
import torch

from ucm.store.factory_v1 import UcmConnectorFactoryV1

ROLE_SCHEDULER = 0
ROLE_WORKER0 = 1
ROLE_WORKER1 = 2
ROLE_NAMES = {
    ROLE_SCHEDULER: "scheduler",
    ROLE_WORKER0: "worker0",
    ROLE_WORKER1: "worker1",
}
_NUM_PARTIES = 3

# Parent-only bookkeeping so signal handlers / atexit can reap children.
_CHILD_PROCS: list[multiprocessing.Process] = []


# ---------------------------------------------------------------------------
# Configs + helpers (shared by child roles)
# ---------------------------------------------------------------------------


def _mooncake_base_config() -> dict[str, Any]:
    return {
        "protocol": "ascend",
        "local_hostname": os.environ.get("UCM_MOONCAKE_LOCAL_HOSTNAME", "127.0.0.1"),
        "metadata_server": os.environ.get("UCM_MOONCAKE_METADATA", "P2PHANDSHAKE"),
        "master_server_address": os.environ.get(
            "UCM_MOONCAKE_MASTER", "127.0.0.1:50088"
        ),
        "device_name": "",
        "global_segment_size": os.environ.get("UCM_MOONCAKE_GLOBAL_SEGMENT", "256MB"),
        "local_buffer_size": os.environ.get("UCM_MOONCAKE_LOCAL_BUFFER", "256MB"),
        "executor_workers": 1,
        "unique_id": os.environ.get("UCM_MOONCAKE_UNIQUE_ID", "mooncake_e2e_test"),
    }


def _scheduler_config(base: dict[str, Any]) -> dict[str, Any]:
    cfg = {k: v for k, v in base.items() if k != "device_id"}
    cfg.pop("device_id", None)
    return cfg


def _make_block_tensor(row_idx: int, numel: int, device_id: int) -> torch.Tensor:
    return torch.full(
        (numel,),
        fill_value=float(row_idx + 1),
        dtype=torch.bfloat16,
        device=f"npu:{device_id}",
    )


def _tensor_rows_to_addr_array(tensors: list[list[torch.Tensor]]) -> np.ndarray:
    return np.asarray(
        [[int(t.data_ptr()) for t in row] for row in tensors],
        dtype=np.uint64,
    )


def _worker_config(
    base: dict[str, Any], device_id: int, tensors: list[list[torch.Tensor]]
) -> dict[str, Any]:
    ptrs: list[int] = []
    sizes: list[int] = []
    for row in tensors:
        for t in row:
            ptrs.append(int(t.data_ptr()))
            sizes.append(int(t.numel() * t.element_size()))
    sample = tensors[0][0]
    tensor_size_list = (int(sample.numel() * sample.element_size()),)
    cfg = dict(base)
    cfg["device_id"] = device_id
    cfg["tensor_size_list"] = tensor_size_list
    cfg["register_buffer_ptrs"] = tuple(ptrs)
    cfg["register_buffer_sizes"] = tuple(sizes)
    return cfg


# ---------------------------------------------------------------------------
# Child side
# ---------------------------------------------------------------------------


def _barrier_wait(barrier: multiprocessing.Barrier, label: str) -> None:
    timeout = float(os.environ.get("UCM_MOONCAKE_TEST_BARRIER_TIMEOUT", "300"))
    try:
        barrier.wait(timeout=timeout)
    except BrokenBarrierError as exc:
        raise RuntimeError(
            f"barrier '{label}' broken/timeout ({timeout}s): a peer likely crashed."
        ) from exc


def _shutdown_store_best_effort(store: Any, role: int) -> None:
    timeout = float(os.environ.get("UCM_MOONCAKE_TEST_SHUTDOWN_TIMEOUT", "60"))
    done = threading.Event()

    def _run() -> None:
        try:
            if hasattr(torch, "npu"):
                try:
                    torch.npu.synchronize()
                except Exception:
                    pass
            store.shutdown()
        except BaseException as exc:  # noqa: BLE001
            print(
                f"[{ROLE_NAMES.get(role, role)}] shutdown error: {exc}", file=sys.stderr
            )
        finally:
            done.set()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    if not done.wait(timeout=timeout):
        print(
            f"[{ROLE_NAMES.get(role, role)}] store.shutdown() exceeded {timeout}s; "
            "forcing child exit.",
            file=sys.stderr,
        )


def _mooncake_party(
    role: int,
    barrier: multiprocessing.Barrier,
    block_ids: list[bytes],
    tensor_numel: int,
) -> None:
    store = None
    exit_code = 0
    label = ROLE_NAMES.get(role, str(role))
    try:
        base = _mooncake_base_config()
        if role == ROLE_SCHEDULER:
            store = UcmConnectorFactoryV1.create_connector(
                "UcmMooncakeStoreV1", _scheduler_config(base)
            )
        elif role == ROLE_WORKER0:
            torch.npu.set_device(0)
            src_tensors = [
                [_make_block_tensor(i, tensor_numel, 0)] for i in range(len(block_ids))
            ]
            cfg = _worker_config(base, 0, src_tensors)
            store = UcmConnectorFactoryV1.create_connector("UcmMooncakeStoreV1", cfg)
        else:
            torch.npu.set_device(1)
            dst_tensors = [
                [torch.zeros((tensor_numel,), dtype=torch.bfloat16, device="npu:1")]
                for _ in block_ids
            ]
            cfg = _worker_config(base, 1, dst_tensors)
            store = UcmConnectorFactoryV1.create_connector("UcmMooncakeStoreV1", cfg)

        _barrier_wait(barrier, "after_setup")

        if role == ROLE_WORKER0:
            shard_indexes = [0] * len(block_ids)
            src_addrs = _tensor_rows_to_addr_array(src_tensors)
            task = store.dump_data(block_ids, shard_indexes, src_addrs)
            store.wait(task)

        _barrier_wait(barrier, "after_worker0_dump")

        if role == ROLE_WORKER1:
            shard_indexes = [0] * len(block_ids)
            dst_addrs = _tensor_rows_to_addr_array(dst_tensors)
            task = store.load_data(block_ids, shard_indexes, dst_addrs)
            store.wait(task)
            for i, row in enumerate(dst_tensors):
                expected = torch.full(
                    (tensor_numel,), float(i + 1), dtype=torch.bfloat16
                )
                if not torch.allclose(row[0].cpu(), expected):
                    raise AssertionError(
                        f"load mismatch at block row {i}: got {row[0].cpu()[:8]}, "
                        f"expected {expected[:8]}"
                    )

        _barrier_wait(barrier, "after_worker1_load")

        if role == ROLE_SCHEDULER:
            hits = store.lookup(block_ids)
            if not all(hits):
                raise AssertionError(f"scheduler lookup expected all True, got {hits}")
            prefix_idx = store.lookup_on_prefix(block_ids)
            if prefix_idx != len(block_ids) - 1:
                raise AssertionError(
                    f"lookup_on_prefix expected {len(block_ids) - 1}, got {prefix_idx}"
                )

        _barrier_wait(barrier, "after_scheduler_lookup")
        print(f"[{label}] OK", file=sys.stderr)
    except BaseException:
        exit_code = 1
        print(f"[{label}] FAILED:", file=sys.stderr)
        traceback.print_exc()
    finally:
        if store is not None:
            _shutdown_store_best_effort(store, role)
        # Bypass interpreter cleanup to guarantee the process dies promptly even
        # if Mooncake / Ascend native state is still fussy.
        os._exit(exit_code)


# ---------------------------------------------------------------------------
# Parent side: supervisor with fail-fast
# ---------------------------------------------------------------------------


def _kill_one(p: multiprocessing.Process, hard: bool = False) -> None:
    if not p.is_alive():
        return
    if hard and sys.platform != "win32":
        try:
            os.kill(p.pid, signal.SIGKILL)  # type: ignore[attr-defined]
        except (ProcessLookupError, PermissionError):
            pass
    else:
        try:
            p.terminate()
        except Exception:
            pass


def _terminate_all_children(reason: Optional[str] = None) -> None:
    if reason:
        print(f"[mooncake_e2e] terminating children: {reason}", file=sys.stderr)

    alive = [p for p in _CHILD_PROCS if p is not None and p.is_alive()]
    for p in alive:
        _kill_one(p, hard=False)
    deadline = time.monotonic() + 5.0
    for p in alive:
        remaining = max(0.0, deadline - time.monotonic())
        p.join(timeout=remaining)

    still = [p for p in _CHILD_PROCS if p is not None and p.is_alive()]
    for p in still:
        _kill_one(p, hard=True)
    for p in still:
        p.join(timeout=3)


def _signal_exit(signum: int, _frame: Any) -> None:
    _terminate_all_children(f"received signal {signum}")
    os._exit(128 + signum)


def _install_parent_cleanup() -> None:
    atexit.register(lambda: _terminate_all_children("parent atexit"))
    sigs = [signal.SIGINT]
    if sys.platform != "win32":
        sigs.append(signal.SIGTERM)
    for sig in sigs:
        try:
            signal.signal(sig, _signal_exit)
        except (OSError, ValueError):
            pass


def _become_process_group_leader() -> None:
    if sys.platform == "win32":
        return
    try:
        os.setpgrp()
    except OSError:
        pass


def _supervise(workers: list[multiprocessing.Process]) -> list[tuple[str, int]]:
    """Wait for all children.

    Return list of (role_name, exitcode) for failed children. If any child
    exits with non-zero code while others are still running, terminate the rest
    **immediately** (no barrier wait).
    """
    join_timeout = float(os.environ.get("UCM_MOONCAKE_TEST_JOIN_TIMEOUT", "600"))
    poll_interval = 0.5
    deadline = time.monotonic() + join_timeout

    failed: list[tuple[str, int]] = []
    aborted = False

    while True:
        alive_any = False
        for p in workers:
            if p.exitcode is None and p.is_alive():
                alive_any = True
                continue
            # Process has exited -> record if failure
            if getattr(p, "_ucm_reaped", False):
                continue
            p._ucm_reaped = True  # type: ignore[attr-defined]
            role_name = p.name or "unknown"
            if p.exitcode not in (0, None):
                failed.append((role_name, int(p.exitcode)))
                if not aborted:
                    aborted = True
                    _terminate_all_children(
                        f"{role_name} exited with code {p.exitcode}; aborting peers"
                    )

        if not alive_any:
            break

        if time.monotonic() > deadline:
            _terminate_all_children(f"join timeout ({join_timeout}s)")
            # give SIGKILL a moment, then still loop to mark remaining as failed
            deadline = time.monotonic() + 10.0  # grace period
            for p in workers:
                if p.exitcode is None and p.is_alive():
                    _kill_one(p, hard=True)

        time.sleep(poll_interval)

    for p in workers:
        if not getattr(p, "_ucm_reaped", False):
            role_name = p.name or "unknown"
            if p.exitcode not in (0, None):
                failed.append((role_name, int(p.exitcode or -1)))

    return failed


def _run_multiprocess_e2e(block_count: int = 6, tensor_numel: int = 256) -> None:
    ctx = multiprocessing.get_context("spawn")
    block_ids = [secrets.token_bytes(16) for _ in range(block_count)]
    barrier = ctx.Barrier(_NUM_PARTIES)

    _CHILD_PROCS.clear()
    workers: list[multiprocessing.Process] = []
    for role in (ROLE_SCHEDULER, ROLE_WORKER0, ROLE_WORKER1):
        p = ctx.Process(
            target=_mooncake_party,
            args=(role, barrier, block_ids, tensor_numel),
            name=ROLE_NAMES[role],
        )
        workers.append(p)
        p.start()
    _CHILD_PROCS.extend(workers)

    try:
        failed = _supervise(workers)
    finally:
        _terminate_all_children("final cleanup")
        _CHILD_PROCS.clear()

    if failed:
        details = ", ".join(f"{name}(code={code})" for name, code in failed)
        raise RuntimeError(f"mooncake e2e failed: {details}")


def _ascend_two_npu_available() -> bool:
    if not hasattr(torch, "npu"):
        return False
    try:
        return int(torch.npu.device_count()) >= 2
    except Exception:
        return False


def run_mooncake_scheduler_two_workers_main() -> None:
    if not hasattr(torch, "npu"):
        print(
            "SKIP: torch.npu not available (need Ascend environment).",
            file=sys.stderr,
        )
        sys.exit(0)
    if not _ascend_two_npu_available():
        print(
            "SKIP: need at least 2 NPUs (worker0=npu:0, worker1=npu:1).",
            file=sys.stderr,
        )
        sys.exit(0)

    _become_process_group_leader()
    _install_parent_cleanup()

    if sys.platform != "win32":
        pgid = os.getpgrp()
        print(
            f"[mooncake_e2e] parent pid={os.getpid()} pgrp={pgid} "
            f"(group kill: kill -TERM -{pgid})",
            file=sys.stderr,
        )

    time.sleep(0.3)
    _run_multiprocess_e2e()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        run_mooncake_scheduler_two_workers_main()
    except BaseException:
        _terminate_all_children("parent exception")
        raise
