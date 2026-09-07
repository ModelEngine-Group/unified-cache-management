# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without including without limitation the rights to use, copy,
# modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
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
"""DramStore bench, modelled on ``cache_on_empty_test.py``.

Self-contained: loads the ``ucmpipelinestore`` pybind module by path
(without the ``ucm`` package init's vllm dependency) and stacks DramStore
via ``PipelineStore.Stack("Dram", libdramstore.so, config)``. KV buffers
are host ``ctypes`` regions that DramStore registers and the ``drampool``
daemon reads (dump) / writes (load) over the configured one-sided transport.

Prerequisites (not started by this script):
  * A ``drampool`` daemon at ``--node-control-endpoints`` started with a yaml whose
    ``transport.endpoints[].one_sided`` matches ``--node-transport-manager-ids``.
  * ``ucmpipelinestore*.so`` + ``libdramstore.so`` + ``libucm_p2p_transport.so``
    on ``--so-dir``/``--lib-dir`` (and LD_LIBRARY_PATH pointing at lib-dir).
  * The chosen NIC reachable from both this host and the drampool host.

Run (from a node with the built modules):
  python3 dramstore_bench_test.py --so-dir /tmp --lib-dir /tmp \
    --local-host <client-ip> \
    --node-control-endpoints <pool-ip>:9000 \
    --node-transport-manager-ids <pool-ip>:4501
"""
import argparse
import array
import ctypes
import importlib.util
import os
import secrets
import statistics
import sys
import time

import numpy as np

# --- ACL runtime helpers for NPU device-memory KV buffers ---------------------
# hixl one-sided RDMA targets NPU device memory (the production KV cache lives
# on the NPU). The bench therefore allocates src/dst KV buffers with aclrtMalloc
# and copies data via aclrtMemcpy. aclInit + aclrtSetDevice(deviceId)
# are already done by the DramStore Stack path (dram_store.cc Compose), so by the
# time these helpers run the caller thread already has device 0 set.
_ACL_MEM_MALLOC_NORMAL = 2
_ACL_MEMCPY_HOST_TO_DEVICE = 1
_ACL_MEMCPY_DEVICE_TO_HOST = 2
_ACL = None


def _acl():
    global _ACL
    if _ACL is None:
        lib = ctypes.CDLL("libascendcl.so")
        lib.aclrtMalloc.restype = ctypes.c_int32
        lib.aclrtMalloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        lib.aclrtFree.restype = ctypes.c_int32
        lib.aclrtFree.argtypes = [ctypes.c_void_p]
        lib.aclrtMemcpy.restype = ctypes.c_int32
        lib.aclrtMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        lib.aclrtMemset.restype = ctypes.c_int32
        lib.aclrtMemset.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int32,
            ctypes.c_size_t,
        ]
        _ACL = lib
    return _ACL


def _check_acl(ret, what):
    if ret != 0:
        raise RuntimeError(f"{what} failed: aclError={ret}")


def device_to_bytes(ptr, size):
    buf = (ctypes.c_char * size)()
    _check_acl(
        _acl().aclrtMemcpy(
            buf, size, ctypes.c_void_p(ptr), size, _ACL_MEMCPY_DEVICE_TO_HOST
        ),
        "aclrtMemcpy D2H",
    )
    return buf.raw


def load_ucmpipelinestore(so_dir):
    """Load the ucmpipelinestore pybind .so by path, avoiding the ucm package init."""
    candidates = []
    for name in os.listdir(so_dir):
        if name.startswith("ucmpipelinestore") and name.endswith(".so"):
            candidates.append(os.path.join(so_dir, name))
    if not candidates:
        raise FileNotFoundError(f"ucmpipelinestore*.so not found in {so_dir}")
    path = sorted(candidates, key=os.path.getmtime)[-1]
    spec = importlib.util.spec_from_file_location("ucmpipelinestore", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_ucmdramstore(so_dir):
    # Backward-compat alias; the DramStore direct-link pybind was merged into
    # the pipeline pattern. Callers now load ucmpipelinestore and Stack DramStore.
    return load_ucmpipelinestore(so_dir)


class BareDramStore:
    """Thin wrapper over ucmpipelinestore.PipelineStore with DramStore stacked;
    mirrors the pipeline connector's Dram path."""

    def __init__(self, config, ucmpipelinestore, libdramstore_path=None):
        self._ds = ucmpipelinestore.PipelineStore()
        if libdramstore_path is None:
            # Testbed convention: libdramstore.so lives next to the pybind .so.
            libdramstore_path = os.path.join(
                os.path.dirname(getattr(ucmpipelinestore, "__file__", "") or ""),
                "libdramstore.so",
            )
        self._ds.Stack("Dram", libdramstore_path, dict(config))

    def lookup(self, block_ids):
        res = self._ds.Lookup(b"".join(block_ids))
        return [bool(b) for b in res]

    def lookup_on_prefix(self, block_ids):
        return self._ds.LookupOnPrefix(b"".join(block_ids))

    def wait(self, task_id):
        self._ds.Wait(task_id)

    def dump(self, block_ids, shard_index, addrs):
        return self._ds.Dump(
            b"".join(block_ids), array.array("Q", shard_index), addrs, 0
        )

    def load(self, block_ids, shard_index, addrs):
        return self._ds.Load(b"".join(block_ids), array.array("Q", shard_index), addrs)


def cmp_and_print_diff(dst_ptrs_sizes, expected_bytes):
    for r, ((ptr, size), want) in enumerate(zip(dst_ptrs_sizes, expected_bytes)):
        got = device_to_bytes(ptr, size)
        if got != want:
            for c, (xa, xb) in enumerate(zip(got, want)):
                if xa != xb:
                    print(f"DIFF at [{r}][{c}]  a={xa} b={xb}")
                    assert False
            if len(got) != len(want):
                print(f"DIFF at [{r}] length {len(got)} vs {len(want)}")
                assert False


def make_dram_config(args, role: str) -> dict:
    if role == "worker":
        local_control = f"{args.local_host}:{args.local_control_port}"
        local_mgr = f"{args.local_host}:{args.local_manager_port}"
    else:
        local_control = f"{args.local_host}:{args.local_control_port + 1}"
        local_mgr = f"{args.local_host}:{args.local_manager_port + 1}"
    return {
        "local_control_endpoint": local_control,
        "local_host": args.local_host,
        "local_transport_manager_id": local_mgr,
        "device_id": args.device_id,
        "router_type": args.router_type,
        "tensor_size_list": [args.tensor_size] * args.layer_size,
        "max_io_entries": 10000,
        "transport_worker_count": 1,
        "node_control_endpoints": args.node_control_endpoints,
        "node_transport_manager_ids": args.node_transport_manager_ids,
        "lookup_timeout_ms": args.lookup_timeout_ms,
        "dump_timeout_ms": args.dump_timeout_ms,
        "load_timeout_ms": args.load_timeout_ms,
    }


class Region:
    """A KV buffer region: `ptr` is a usable local pointer (fill/verify); `addr`
    is the value passed as the Operation's remote_addr. For NPU device memory
    these are the same (the device pointer returned by aclrtMalloc)."""

    __slots__ = ("ptr", "size", "addr", "_base")

    def __init__(self, ptr, size, addr):
        self.ptr = ptr
        self.size = size
        self.addr = addr


def build_device_regions(tensor_size, shards, request_size):
    """NPU device-memory model: aclrtMalloc'd buffers; ptr == addr == the device
    pointer. Only the integer device addresses are held in Region; the ACL
    runtime reclaims them on process exit."""

    total = tensor_size * shards * request_size
    print(
        f"[debug] build_device_regions: tensor_size={tensor_size} shards={shards} request_size={request_size} total={total}",
        file=sys.stderr,
    )
    base = ctypes.c_void_p()
    _check_acl(
        _acl().aclrtMalloc(ctypes.byref(base), total, _ACL_MEM_MALLOC_NORMAL),
        "aclrtMalloc",
    )

    regions = []
    for i in range(request_size * shards):
        addr = base.value + i * tensor_size
        regions.append(Region(addr, tensor_size, addr))
    regions[0]._base = base
    return regions, total


def fill_random(regions):
    for r in regions:
        data = secrets.token_bytes(r.size)
        host = ctypes.create_string_buffer(data, r.size)
        _check_acl(
            _acl().aclrtMemcpy(
                ctypes.c_void_p(r.ptr), r.size, host, r.size, _ACL_MEMCPY_HOST_TO_DEVICE
            ),
            "aclrtMemcpy H2D fill",
        )


def zero(regions):
    for r in regions:
        _check_acl(
            _acl().aclrtMemset(ctypes.c_void_p(r.ptr), r.size, 0, r.size),
            "aclrtMemset zero",
        )


def region_addrs(regions, shards):
    """Return a 2-D uint64 ndarray of shape (n_blocks, shards) for the C++ binding."""
    flat = [r.addr for r in regions]
    n_blocks = len(flat) // shards
    return np.array(flat, dtype=np.uint64).reshape(n_blocks, shards)


def e2e_test(worker, scheduler, src_regs, dst_regs, args):
    chunk_block_ids = [secrets.token_bytes(16) for _ in range(args.request_size)]
    shard_indexes = [0 for _ in range(args.request_size)]
    shards = args.layer_size * args.chunk_size

    founds = scheduler.lookup(chunk_block_ids)
    assert not any(founds), "expected all-miss before dump"

    fill_random(src_regs)
    zero(dst_regs)
    expected = [device_to_bytes(r.ptr, r.size) for r in src_regs]
    print(
        f"[debug] src[0][:4]={[b for b in expected[0][:4]]} ptr={hex(src_regs[0].ptr)} addr={hex(src_regs[0].addr)}",
        file=sys.stderr,
    )

    t0 = time.perf_counter()
    task = worker.dump(chunk_block_ids, shard_indexes, region_addrs(src_regs, shards))
    worker.wait(task)
    dump_dt = time.perf_counter() - t0

    founds = scheduler.lookup(chunk_block_ids)
    assert all(founds), "expected all-hit after dump"

    t0 = time.perf_counter()
    task = worker.load(chunk_block_ids, shard_indexes, region_addrs(dst_regs, shards))
    worker.wait(task)
    load_dt = time.perf_counter() - t0

    dst_flat = [(r.ptr, r.size) for r in dst_regs]
    print(
        f"[debug] dst first bytes: {[device_to_bytes(r.ptr, 1)[0] for r in dst_regs[:8]]}",
        file=sys.stderr,
    )
    cmp_and_print_diff(dst_flat, expected)

    per_block = args.tensor_size * args.layer_size * args.chunk_size
    return dump_dt, load_dt, args.request_size * per_block


def main():
    parser = argparse.ArgumentParser(description="DramStore bench")
    parser.add_argument(
        "--so-dir", default="/tmp", help="dir containing ucmdramstore*.so"
    )
    parser.add_argument("--tensor-size", type=int, default=4096, help="bytes per shard")
    parser.add_argument("--layer-size", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument("--request-size", type=int, default=32, help="blocks per batch")
    parser.add_argument("--batch-number", type=int, default=32)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--router-type", default="ring_hash")
    parser.add_argument("--local-host", default="127.0.0.1")
    parser.add_argument("--local-control-port", type=int, default=4702)
    parser.add_argument("--local-manager-port", type=int, default=4502)
    parser.add_argument(
        "--node-control-endpoints", nargs="+", default=["127.0.0.1:9000"]
    )
    parser.add_argument(
        "--node-transport-manager-ids", nargs="+", default=["127.0.0.1:4501"]
    )
    parser.add_argument(
        "--kv-cache-memory-type", default="host", choices=["host", "device"]
    )
    parser.add_argument("--lookup-timeout-ms", type=int, default=5000)
    parser.add_argument("--dump-timeout-ms", type=int, default=2000)
    parser.add_argument("--load-timeout-ms", type=int, default=2000)
    args = parser.parse_args()

    args.shard_size = args.tensor_size * args.layer_size * args.chunk_size
    args.block_size = args.shard_size
    os.environ.setdefault("UC_LOGGER_LEVEL", "info")

    ucmdramstore = load_ucmdramstore(args.so_dir)
    shards = args.layer_size * args.chunk_size
    count = args.request_size * shards
    # NPU device memory: aclrtMalloc'd buffers; ptr == addr == device pointer.
    src_regs, src_total = build_device_regions(
        args.tensor_size, shards, args.request_size
    )
    dst_regs, dst_total = build_device_regions(
        args.tensor_size, shards, args.request_size
    )
    config = make_dram_config(args, "worker")
    config["gpu_kv_buffer_addrs"] = [int(src_regs[0].addr), int(dst_regs[0].addr)]
    config["gpu_kv_buffer_sizes"] = [int(src_total), int(dst_total)]
    worker = BareDramStore(config, ucmdramstore)
    scheduler = worker  # single-peer: worker does lookup+dump+load, one hixl engine
    print(
        f"[debug] registered {len(src_regs) + len(dst_regs)} regions", file=sys.stderr
    )

    per_block = args.tensor_size * args.layer_size * args.chunk_size
    print(
        f"[bench] blocks/batch={args.request_size} shards={shards} per_shard={args.tensor_size} B "
        f"per_block={per_block} B"
        f"batches={args.batch_number} (warmup={args.warmup_batches})"
    )

    dump_times, load_times, data_bytes = [], [], 0
    time.sleep(5)  # let hixl re-exchange/re-connect churn settle
    for i in range(args.batch_number + args.warmup_batches):
        dt, lt, db = e2e_test(worker, scheduler, src_regs, dst_regs, args)
        if i < args.warmup_batches:
            continue
        dump_times.append(dt)
        load_times.append(lt)
        data_bytes = db

    n = len(dump_times)
    dump_avg = statistics.mean(dump_times)
    load_avg = statistics.mean(load_times)
    dump_bw = data_bytes / dump_avg / 1e6 if dump_avg > 0 else 0.0
    load_bw = data_bytes / load_avg / 1e6 if load_avg > 0 else 0.0

    print("\n================ DramStore bench ================")
    print(f"batches measured : {n}")
    print(f"data per batch   : {data_bytes} B ({data_bytes / 1e6:.2f} MB)")
    print(f"dump avg latency : {dump_avg * 1000:.2f} ms   bw {dump_bw:.2f} MB/s")
    print(f"load avg latency : {load_avg * 1000:.2f} ms   bw {load_bw:.2f} MB/s")
    if n > 1:
        print(
            f"dump p50/p99     : {statistics.median(dump_times) * 1000:.2f}"
            f" / {sorted(dump_times)[int(0.99 * (n - 1))] * 1000:.2f} ms"
        )
        print(
            f"load p50/p99     : {statistics.median(load_times) * 1000:.2f}"
            f" / {sorted(load_times)[int(0.99 * (n - 1))] * 1000:.2f} ms"
        )
    print("round-trip       : {:.2f} ms".format((dump_avg + load_avg) * 1000))


if __name__ == "__main__":
    main()
