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

from ucm.integration.vllm.v2.ucm_kv_cache import (
    UCMKVCacheLayout,
    parse_kv_cache_config,
)
from ucm.integration.vllm.v2.ucm_proxy import (
    SimpleFileUCMProxy,
    TorchTensorByteAccess,
    UCMProxyAdapter,
)
from ucm.integration.vllm.v2.ucm_scheduler import (
    RequestDispatchMeta,
    UCMConnectorMetadata,
    UCMGroupBlockIds,
    UCMGroupDispatchPlan,
)

if (
    os.environ.get("UCM_MODEL_CHECK_MODEL", "").rstrip("/").endswith("glm52")
    and "UCM_MODEL_CHECK_ADDITIONAL_CONFIG" not in os.environ
):
    os.environ["UCM_MODEL_CHECK_ADDITIONAL_CONFIG"] = (
        '{"enable_sparse_sfa_c8":true,"enable_sparse_li_c8":true}'
    )

from ucm_toolkit.tools.model_check import ascend


def _metadata(
    group_id: int,
    key: bytes,
    token_count: int,
    block_id: int,
    attribute: str,
) -> UCMConnectorMetadata:
    plan = UCMGroupDispatchPlan(
        group_id,
        (key,),
        0,
        0,
        token_count,
        (UCMGroupBlockIds(group_id, 0, (block_id,)),),
    )
    kwargs = {attribute: (plan,)}
    return UCMConnectorMetadata(
        requests={"v2-npu-probe": RequestDispatchMeta("v2-npu-probe", **kwargs)}
    )


def _exercise_group(
    layout: UCMKVCacheLayout,
    proxy: UCMProxyAdapter,
    byte_access: TorchTensorByteAccess,
    group_id: int,
    sequence: int,
) -> tuple[int, int]:
    group = layout.spec.groups[group_id]
    key = sequence.to_bytes(16, "little")
    dump_meta = _metadata(
        group_id, key, group.token_block_size, 0, "dump_plans"
    )
    load_meta = _metadata(
        group_id, key, group.token_block_size, 1, "load_plans"
    )
    dump_batch = layout.build_dump_batches(dump_meta)
    load_batch = layout.build_load_batches(load_meta)
    if not dump_batch.block_ids or len(dump_batch.sizes) != len(load_batch.sizes):
        raise AssertionError(f"empty or asymmetric v2 batch for group {group_id}")

    expected: list[bytes] = []
    for index, (ptr, size) in enumerate(
        zip(dump_batch.ptrs, dump_batch.sizes, strict=True)
    ):
        pattern = bytes(
            (sequence * 29 + index * 17 + byte_index) % 251
            for byte_index in range(251)
        )
        payload = (pattern * ((size + len(pattern) - 1) // len(pattern)))[:size]
        byte_access.write(ptr, payload)
        expected.append(payload)
    torch.npu.synchronize()
    proxy.dump(
        dump_batch.block_ids,
        dump_batch.offsets,
        dump_batch.ptrs,
        dump_batch.sizes,
    )

    for ptr, size in zip(load_batch.ptrs, load_batch.sizes, strict=True):
        byte_access.write(ptr, bytes(size))
    torch.npu.synchronize()
    proxy.load(
        load_batch.block_ids,
        load_batch.offsets,
        load_batch.ptrs,
        load_batch.sizes,
    )

    for index, (ptr, size) in enumerate(
        zip(load_batch.ptrs, load_batch.sizes, strict=True)
    ):
        actual = byte_access.read(ptr, size)
        if actual != expected[index]:
            raise AssertionError(
                f"NPU byte mismatch for group={group_id}, segment={index}, size={size}"
            )
    return len(load_batch.sizes), sum(load_batch.sizes)


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
        parsed = parse_kv_cache_config(
            fixture.kv_cache_config,
            scheduler_block_size=ascend.block_size,
        )
        layout = UCMKVCacheLayout(
            parsed,
            fixture.kv_caches,
            num_blocks=int(fixture.kv_cache_config.num_blocks),
        )
        byte_access = TorchTensorByteAccess()
        byte_access.register_tensors(fixture.kv_caches)

        group_ids = sorted(set(fixture.layer_to_group.values()))

        total_segments = 0
        total_bytes = 0
        with tempfile.TemporaryDirectory(prefix="ucm-v2-proxy-") as directory:
            adapter = UCMProxyAdapter(SimpleFileUCMProxy(directory, byte_access))
            for sequence, group_id in enumerate(group_ids, 1):
                segments, transferred = _exercise_group(
                    layout,
                    adapter,
                    byte_access,
                    group_id,
                    sequence,
                )
                total_segments += segments
                total_bytes += transferred
                print(
                    "[ucm-v2-npu] PASS "
                    f"group={group_id} "
                    f"segments={segments} bytes={transferred}",
                    flush=True,
                )

        print(
            "[ucm-v2-npu] PASS real NPU layout/file-proxy round-trip: "
            f"mode={parsed.mode}, groups={len(group_ids)}, "
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
