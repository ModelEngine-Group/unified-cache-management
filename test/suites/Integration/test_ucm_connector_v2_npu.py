"""Real-Ascend probe for the connector-v2 worker layout and file Proxy.

This intentionally reuses model-check's production NPUModelRunner cache
construction, but does not use the legacy connector or its Store. The probe
builds complete v2 records from real NPU tensors, persists them through the
standalone file Proxy, and compares source/target blocks byte-for-byte.
"""

from __future__ import annotations

import gc
import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
for package_name, package_path in (
    ("ucm", REPO_ROOT / "ucm"),
    ("ucm.integration", REPO_ROOT / "ucm" / "integration"),
    ("ucm.integration.vllm", REPO_ROOT / "ucm" / "integration" / "vllm"),
):
    package = types.ModuleType(package_name)
    package.__path__ = [str(package_path)]
    sys.modules.setdefault(package_name, package)

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from ucm.integration.vllm.v2.ucm_connector import UCMConnector
from ucm.integration.vllm.v2.ucm_kv_cache import UCMTransferBuilder
from ucm.integration.vllm.v2.ucm_proxy import (
    SimpleFileUCMProxy,
    TorchTensorByteAccess,
    UCMProxyAdapter,
)
from ucm.integration.vllm.v2.ucm_scheduler import (
    RequestDispatchMeta,
    UCMConnectorMetadata,
    UCMGroupDispatchPlan,
    dispatch_routes,
)

if (
    os.environ.get("UCM_MODEL_CHECK_MODEL", "").rstrip("/").endswith("glm52")
    and "UCM_MODEL_CHECK_ADDITIONAL_CONFIG" not in os.environ
):
    os.environ["UCM_MODEL_CHECK_ADDITIONAL_CONFIG"] = (
        '{"enable_sparse_sfa_c8":true,"enable_sparse_li_c8":true}'
    )

from ucm_toolkit.tools.model_check import ascend


def _metadata(layout, kind, groups, key, block_base, attribute):
    group_block_ids = []
    cursor = block_base
    for group in groups:
        group_block_ids.append(np.arange(cursor, cursor + group.tail_blocks, dtype=np.uint64))
        cursor += group.tail_blocks
    plan = UCMGroupDispatchPlan(
        kind, (key,), 0, layout.spec.ucm_cache_block_size, tuple(group_block_ids)
    )
    return UCMConnectorMetadata(
        requests={"v2-npu-probe": RequestDispatchMeta(
            "v2-npu-probe", **{attribute: (plan,)}
        )}
    )


def _exercise_route(layout, proxy, byte_access, kind, groups, sequence):
    # Groups share a native block pool: use distinct IDs across groups and
    # disjoint source/target windows, rather than overwriting shared slots.
    window_blocks = sum(group.tail_blocks for group in groups)
    for group in groups:
        if group.group_id in layout.group_layouts:
            if 2 * window_blocks > layout.group_layouts[group.group_id].num_blocks:
                raise ValueError("NPU probe needs disjoint source/target route windows")
    key = sequence.to_bytes(16, "little")
    dump_meta = _metadata(layout, kind, groups, key, 0, "dump_plans")
    load_meta = _metadata(layout, kind, groups, key, window_blocks, "load_plans")
    (dump_transfer,) = UCMTransferBuilder(layout).build_dump_transfers(dump_meta)
    (load_transfer,) = UCMTransferBuilder(layout).build_load_transfers(load_meta)
    source_ptrs = dump_transfer.ptrs.reshape(-1)
    target_ptrs = load_transfer.ptrs.reshape(-1)
    sizes = dump_transfer.sizes.reshape(-1)
    np.testing.assert_array_equal(sizes, load_transfer.sizes.reshape(-1))

    expected = []
    for index, (ptr, size) in enumerate(zip(source_ptrs, sizes, strict=True)):
        ptr, size = int(ptr), int(size)
        pattern = bytes(
            (sequence * 29 + index * 17 + byte_index) % 251
            for byte_index in range(251)
        )
        payload = (pattern * ((size + len(pattern) - 1) // len(pattern)))[:size]
        byte_access.write(ptr, payload)
        expected.append(payload)
    torch.npu.synchronize()
    proxy.submit("dump", dump_transfer)
    proxy.commit(dump_transfer.keys)

    for ptr, size in zip(target_ptrs, sizes, strict=True):
        byte_access.write(int(ptr), bytes(int(size)))
    torch.npu.synchronize()
    proxy.submit("load", load_transfer)

    for index, (ptr, size) in enumerate(zip(target_ptrs, sizes, strict=True)):
        actual = byte_access.read(int(ptr), int(size))
        if actual != expected[index]:
            raise AssertionError(
                f"NPU byte mismatch for route={kind}, segment={index}, size={size}"
            )
    return len(sizes), int(sizes.sum())


def main() -> int:
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ascend.visible_devices
    active_device = torch.device("npu:0")
    __import__("torch_npu")
    torch.npu.set_device(active_device)
    fixture = None
    vllm_config = None
    try:
        vllm_config = ascend.make_config()
        fixture = ascend.make_cache(vllm_config, active_device)
        byte_access = TorchTensorByteAccess()
        byte_access.register_tensors(fixture.kv_caches)

        total_segments = 0
        total_bytes = 0
        with tempfile.TemporaryDirectory(prefix="ucm-v2-proxy-") as directory:
            vllm_config.kv_transfer_config.kv_connector_extra_config[
                "v2_storage_path"
            ] = directory
            worker = UCMConnector(
                vllm_config, KVConnectorRole.WORKER, fixture.kv_cache_config
            )
            worker.register_kv_caches(fixture.kv_caches)
            layout = worker.layout
            adapter = UCMProxyAdapter(SimpleFileUCMProxy(directory, byte_access))
            routes = dispatch_routes(layout.spec)
            for sequence, (kind, groups) in enumerate(routes, 1):
                segments, transferred = _exercise_route(
                    layout, adapter, byte_access, kind, groups, sequence
                )
                total_segments += segments
                total_bytes += transferred
                print(
                    "[ucm-v2-npu] PASS "
                    f"route={kind} segments={segments} bytes={transferred}",
                    flush=True,
                )

        print(
            "[ucm-v2-npu] PASS real NPU layout/file-proxy round-trip: "
            f"routes={len(routes)} "
            f"segments={total_segments}, bytes={total_bytes}",
            flush=True,
        )
        return 0
    finally:
        del fixture, vllm_config
        gc.collect()
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )
        from vllm_ascend.distributed.parallel_state import (
            destroy_ascend_model_parallel,
        )

        destroy_ascend_model_parallel()
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.npu.synchronize(active_device)
        torch.npu.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
