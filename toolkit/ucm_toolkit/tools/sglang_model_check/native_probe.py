"""Bounded CPU cache construction using SGLang's production pool providers.

No Engine, Scheduler, weights or device copies. Physical format decisions stay
in the installed SGLang: device descriptions are materialized on CPU and fed to
its real host builders. Unsupported versions retain their exact error reason.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from math import prod
import inspect
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from .pool_probe import PoolProbeUnavailable, PoolSpec


class ProbeMemoryLimit(PoolProbeUnavailable):
    pass


@dataclass
class NativePools:
    pools: dict[str, Any] = field(default_factory=dict)
    specs: list[PoolSpec] = field(default_factory=list)
    keepalive: list[Any] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)
    layer_ids: dict[str, tuple[int, ...]] = field(default_factory=dict)

    def close(self):
        try:
            for pool in self.pools.values():
                destroy = getattr(pool, "destroy", None)
                if callable(destroy):
                    destroy()
        finally:
            self.pools.clear()
            self.keepalive.clear()


def native_family(requirements: Any) -> str | None:
    names = set(requirements.pool_names)
    if any(name.startswith("deepseek_v4_") for name in names):
        return "deepseek_v4"
    if "indexer" in names:
        return "dsa" if requirements.is_mla else "minimax_sparse"
    return None


@contextmanager
def cpu_allocation_budget(torch: Any, limit: int):
    """Bound common CPU factories and reject hidden accelerator allocations.

    Counts cumulative allocations conservatively. It is not an OS RSS limit;
    library workspaces and allocations outside these factories are not covered.
    The caller reserves additional space for IO staging and snapshots.
    """
    used = [0]

    def factory(original, like=False, full=False):
        def allocate(*args, **kwargs):
            source = args[0] if like else None
            device = kwargs.get("device", getattr(source, "device", "cpu"))
            if str(device).split(":")[0] != "cpu":
                raise PoolProbeUnavailable(f"cache constructor requested real device allocation: {device}")
            if like:
                shape = tuple(source.shape)
            else:
                shape = kwargs.get("size", args[0] if args else ())
                if isinstance(shape, int):
                    shape = args[:1] if full else args
                shape = tuple(shape)
            dtype = kwargs.get("dtype", getattr(source, "dtype", None)) or torch.get_default_dtype()
            size = prod(shape) * dtype.itemsize
            if used[0] + size > limit:
                raise ProbeMemoryLimit(f"native CPU cache allocation budget exceeded: {used[0] + size} > {limit} bytes")
            used[0] += size
            return original(*args, **kwargs)
        return allocate

    with ExitStack() as stack:
        for name in ("empty", "zeros", "ones", "full", "rand", "randn",
                     "empty_like", "zeros_like", "ones_like"):
            stack.enter_context(patch.object(
                torch, name, factory(getattr(torch, name), like=name.endswith("_like"), full=name == "full")
            ))
        yield used


def _host_from_device(device_pool, page_size, layout, *, is_mla):
    """Reuse the installed production host class and captured buffer_info."""
    from sglang.srt.mem_cache.pool_host.mha import get_mha_host_pool_cls
    from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost

    common = dict(
        host_to_device_ratio=1.0, host_size=0,
        page_size=page_size, layout=layout, pin_memory=False, device="cpu",
        allocator_type="default",
    )
    describe = getattr(device_pool, "get_device_pool_infos", None)
    if is_mla:
        pool_cls = MLATokenToKVPoolHost
    else:
        # The MHA selector changed its argument together with the constructors.
        from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost

        if "device_pool" in inspect.signature(MHATokenToKVPoolHost).parameters:
            pool_cls = get_mha_host_pool_cls(device_pool)
        elif callable(describe):
            pool_cls = get_mha_host_pool_cls(describe()[0].buffer_info)
        else:
            raise PoolProbeUnavailable("native MHA pool cannot expose buffer_info for the installed Host API")
    parameters = inspect.signature(pool_cls).parameters
    if "device_pool" in parameters:
        if is_mla:
            common["override_kv_cache_dim"] = device_pool.kv_cache_dim
        return pool_cls(device_pool=device_pool, **common)
    if "buffer_info" not in parameters or not callable(describe):
        raise PoolProbeUnavailable(
            f"{pool_cls.__name__} requires buffer_info, but "
            f"{type(device_pool).__name__} has no compatible device-buffer descriptor API"
        )
    common.update(buffer_info=describe()[0].buffer_info,
                  device_capacity_tokens=device_pool.size,
                  start_layer=device_pool.start_layer, end_layer=device_pool.end_layer)
    if is_mla:
        common.update(target_layer_num=device_pool.layer_num,
                      override_kv_cache_dim=device_pool.kv_cache_dim)
    return pool_cls(**common)


def _describe(bundle, window, primary_page_size):
    import torch
    from ucm.integration.sglang.ucm_connector import SglangUcmConnector

    for name, pool in bundle.pools.items():
        if any(spec.name == name for spec in bundle.specs):
            continue
        if name == "kv" and getattr(pool, "kv_buffer", None) is None:
            bundle.specs.append(PoolSpec(name, "logical", (),
                storage_page_size=pool.page_size, source="sglang_native"))
            continue
        ptrs, sizes = pool.get_page_buffer_meta(torch.arange(pool.page_size, dtype=torch.int64))
        _, normalized = SglangUcmConnector._flatten_page_meta(ptrs, sizes, 1)
        trailing = 0
        if name == "swa":
            trailing = (window + primary_page_size - 1) // primary_page_size
        elif name.endswith("_state"):
            trailing = 1
        bundle.specs.append(PoolSpec(
            name, "native", bundle.layer_ids.get(name, tuple(range(pool.layer_num))), window_pages=trailing,
            storage_page_size=pool.page_size, page_bytes=sum(normalized[0]),
            source="sglang_native",
        ))


def _build_dsa(bundle, model, config, platform, dtype, *, full_layers=None):
    from sglang.srt.configs.model_config import (
        get_dsa_index_head_dim, get_dsa_index_kpool, get_dsa_index_kpool_compress,
        dsa_layer_skips_topk,
    )
    from sglang.srt.mem_cache.kv_cache_configurator import calculate_mla_kv_cache_dim
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
    from sglang.srt.mem_cache.pool_host.dsa import DSAIndexerPoolHost

    layers = tuple(range(model.num_hidden_layers)) if full_layers is None else tuple(full_layers)
    size = config.page_size * (config.pages * 2 + 2)
    kwargs = dict(
        size=size, page_size=config.page_size, dtype=dtype,
        kv_lora_rank=model.kv_lora_rank, qk_rope_head_dim=model.qk_rope_head_dim,
        layer_num=len(layers), device="cpu", enable_memory_saver=False,
        start_layer=0, end_layer=len(layers),
        index_head_dim=get_dsa_index_head_dim(model.hf_config),
        kv_cache_dim=calculate_mla_kv_cache_dim(model_config=model, kv_cache_dtype=dtype),
    )
    if platform == "ascend":
        from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool

        kwargs.update(index_size=size, index_page_size=config.page_size)
        device_pool = NPUMLATokenToKVPool(**kwargs)
    else:
        from sglang.srt.platforms import current_platform

        pool_cls = current_platform.get_kv_pool_cls(kind="dsa") or DSATokenToKVPool
        kwargs.update(
            index_kpool=get_dsa_index_kpool(model.hf_config),
            index_kpool_compress=get_dsa_index_kpool_compress(model.hf_config),
            max_running_requests=1,
            skip_topk_layers=[dsa_layer_skips_topk(model.hf_config, i)
                              for i in layers],
        )
        device_pool = pool_cls(**kwargs)
    bundle.keepalive.append(device_pool)
    bundle.pools["kv"] = _host_from_device(device_pool, config.page_size, config.layout, is_mla=True)
    declarations = device_pool.host_pool_decls()
    for decl in declarations:
        if decl.is_primary:
            continue
        name = str(decl.pool_name)
        decl.host_pool_builder.validate(decl=decl, transfer_page_size=config.page_size,
                                        packed_draft_device_pools=())
        # Use the real declared builder, disabling pinning only at its actual
        # DSA host constructor; no transfer kernel is invoked in this probe.
        import sglang.srt.mem_cache.pool_host.dsa as dsa_module

        class CpuIndexer(DSAIndexerPoolHost):
            def __init__(self, *args, **kwargs):
                kwargs["pin_memory"] = False
                super().__init__(*args, **kwargs)

        with patch.object(dsa_module, "DSAIndexerPoolHost", CpuIndexer):
            bundle.pools[name] = decl.host_pool_builder.build(
                decl=decl, anchor_host=bundle.pools["kv"], allocator_type="default",
                packed_draft_device_pools=(),
            )
    if "indexer" not in bundle.pools:
        # NPU MLA embeds index K/scale in the primary captured buffer_info.
        host = bundle.pools["kv"]
        if getattr(host, "index_k_buffer", None) is None:
            raise PoolProbeUnavailable("DSA native host declarations did not expose indexer payload")
        bundle.report["integrated_payloads"] = {"kv": ["kv", "indexer"]}
    for name in bundle.pools:
        bundle.layer_ids[name] = layers


def _build_dsa_recurrent(bundle, model, config, platform, dtype):
    """Mirror HybridLinearKVPool's compact DSA child plus persistent state.

    No engine wrapper or request allocator is needed for Host-storage IO; the
    global-to-compact layer map is retained in the report instead of executing
    model-forward/transfer dispatch through that wrapper.
    """
    from .pool_probe import build_pool_specs
    from .runner import _make_mamba_host_pool

    specs = build_pool_specs(model, ("kv", "mamba"), config.page_size,
                             platform, external_attention=True)
    primary, state = specs
    if primary.kind == "logical" or not primary.layers:
        raise PoolProbeUnavailable("sparse recurrent model has no physical attention layers")
    _build_dsa(bundle, model, config, platform, dtype, full_layers=primary.layers)
    bundle.pools["mamba"] = _make_mamba_host_pool(state, config)
    bundle.specs.append(state)
    bundle.layer_ids["mamba"] = state.layers
    bundle.report["hybrid_layer_mapping"] = {
        "attention_model_layers": list(primary.layers),
        "state_model_layers": list(state.layers),
        "scope": "config partition and native child pools; engine wrapper dispatch not executed",
    }


def _build_minimax(bundle, model, config, platform, dtype):
    from sglang.srt.configs.model_config import (
        get_minimax_sparse_attention_config, get_minimax_sparse_layer_ids,
        get_minimax_sparse_disable_value_layer_ids,
    )
    from sglang.srt.mem_cache.memory_pool import MiniMaxSparseKVPool
    from sglang.srt.mem_cache.pool_host.mha import MHATokenToKOnlyPoolHost

    cfg = get_minimax_sparse_attention_config(model.hf_config)
    dense, sparse = get_minimax_sparse_layer_ids(cfg)
    disabled = get_minimax_sparse_disable_value_layer_ids(cfg)
    if set(sparse) - set(disabled):
        raise PoolProbeUnavailable("production MiniMax HiCache rejects value-bearing sparse index layers")
    pool_cls = MiniMaxSparseKVPool
    if platform == "ascend":
        from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMiniMaxSparseKVPool
        pool_cls = NPUMiniMaxSparseKVPool
    device_pool = pool_cls(
        size=config.page_size * (config.pages * 2 + 2), page_size=config.page_size,
        dtype=dtype, index_dtype=model.dtype,
        head_num=model.get_num_kv_heads(1), head_dim=model.head_dim,
        idx_head_dim=cfg["sparse_index_dim"], dense_layer_ids=dense,
        sparse_layer_ids=sparse, disable_value_sparse_layer_ids=disabled,
        device="cpu", enable_memory_saver=False, start_layer=0, end_layer=model.num_hidden_layers,
    )
    bundle.keepalive.append(device_pool)
    bundle.pools["kv"] = _host_from_device(device_pool.main_pool, config.page_size,
                                          config.layout, is_mla=False)
    if device_pool.index_k_pool is not None:
        bundle.pools["indexer"] = MHATokenToKOnlyPoolHost(
            device_pool.index_k_pool, bundle.pools["kv"], config.layout,
            pin_memory=False, device="cpu", allocator_type="default",
        )


def _dsv4_main_layer_ratios(model):
    """Select main-model layers, excluding reserved/draft configuration slots.

    This probe builds start_layer=0, end_layer=num_hidden_layers and does not
    construct an MTP worker. Extra configuration entries are not extra active
    cache layers; missing main-layer entries must not be silently padded.
    """
    ratios = list(model.compress_ratios)
    num_layers = model.num_hidden_layers
    if len(ratios) < num_layers:
        raise PoolProbeUnavailable(
            f"compress_ratios has {len(ratios)} entries, "
            f"but {num_layers} main model layers are required"
        )
    return ratios[:num_layers]


def _dsv4_low_ratio_regions_compat(assembler, device_pool, page_size, ratios):
    """Older V4 assemblers lack the V4.1 C1/C2 region-discovery helper.

    Only omit discovery when neither active layer descriptions nor physical
    pools indicate low-ratio payloads. Do not silently certify untested pools.
    """
    discover = getattr(assembler, "_dsv4_low_ratio_regions", None)
    if callable(discover):
        return discover(device_pool, page_size)
    active = set(ratios) & {1, 2}
    for field in ("sources_by_ratio", "kv_pools", "index_pools"):
        pools = getattr(device_pool, field, {}) or {}
        active.update(ratio for ratio in (1, 2) if pools.get(ratio))
    if active:
        raise PoolProbeUnavailable(
            "installed SGLang lacks _dsv4_low_ratio_regions, but active "
            f"C1/C2 payloads require discovery (ratios={sorted(active)}); "
            "this checker cannot validate those pools on this SGLang version"
        )
    return []


def _build_dsv4(bundle, model, config, platform, dtype):
    from sglang.srt.mem_cache import deepseek_v4_memory_pool as native
    from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as assembler
    from sglang.srt.mem_cache.kv_cache_configurator import _get_dsv4_compress_state_dtypes
    from sglang.srt.mem_cache.memory_pool_host import (
        LogicalHostPool, DeepSeekV4PagedHostPool, DeepSeekV4StateHostPool,
    )

    if platform != "ascend" and config.page_size != 256:
        raise PoolProbeUnavailable("paged DeepSeek V4 requires --page-size 256 on CUDA/ROCm")
    pages = config.pages * 2 + 2
    size = pages * config.page_size
    ratios = _dsv4_main_layer_ratios(model)
    cls = native.DeepSeekV4TokenToKVPool
    layout_args = {}
    if platform == "ascend":
        from sglang.srt.hardware_backend.npu.dsv4.dsv4_memory_pool import DSV4NPUTokenToKVPool
        cls = DSV4NPUTokenToKVPool
    else:
        main_layout, compressed_layout = native.select_dsv4_kv_layout()
        layout_args = dict(kv_layout=main_layout, compressed_kv_layout=compressed_layout)
    c4_dtype, c128_dtype = _get_dsv4_compress_state_dtypes()
    device_pool = cls(
        max_num_reqs=1, num_req_slots=2,
        swa_size=size, c4_size=size // 4,
        c128_size=max(1, size // 128),
        c4_state_pool_size=pages * native.resolve_compress_state_ring_size(4),
        c128_state_pool_size=2 * native.resolve_compress_state_ring_size(128),
        page_size=config.page_size, swa_page_size=config.page_size,
        dtype=dtype, c4_state_dtype=c4_dtype, c128_state_dtype=c128_dtype,
        qk_nope_head_dim=model.qk_nope_head_dim, qk_rope_head_dim=model.qk_rope_head_dim,
        indexer_head_dim=model.index_head_dim, layer_num=model.num_hidden_layers,
        device="cpu", enable_memory_saver=False, compression_ratios=ratios,
        sliding_window=model.window_size, start_layer=0, end_layer=model.num_hidden_layers,
        kv_source_layers=list(getattr(model.hf_config, "kv_source_layer_ids", ())),
        full_size=size, **layout_args,
    )
    bundle.keepalive.append(device_pool)
    bundle.pools["kv"] = LogicalHostPool(pages * config.page_size, config.page_size,
                                          layout=config.layout)

    def paged(name, buffers, item_bytes, slot_size, aligned=False):
        if not buffers:
            return
        bundle.pools[str(name)] = DeepSeekV4PagedHostPool(
            pool_name=str(name), device_buffers=buffers, item_bytes=int(item_bytes),
            num_host_pages=pages, slot_page_size=slot_size,
            layout=config.layout, device="cpu", pin_memory=False,
            allocator_type="default", page_aligned_only=aligned,
        )

    mappings = assembler._resolve_deepseek_v4_layer_mappings(device_pool)
    if device_pool.swa_kv_pool is not None and not device_pool._unified_kv:
        swa = device_pool.swa_kv_pool
        assembler._require_single_row_dsv4_swa_pages(
            logical_page_size=device_pool.swa_page_size, physical_page_size=swa.page_size,
            consumer="offline model probe",
        )
        paged("swa", swa.kv_buffer, swa.bytes_per_page_padded, device_pool.swa_page_size,
              assembler._dsv4_page_aligned_only(swa))
    for ratio, layer_map in ((4, mappings.c4), (128, mappings.c128)):
        if not layer_map:
            continue
        buffers, item_bytes = assembler._dsv4_compressed_region_buffers(device_pool, ratio)
        source_pool = device_pool.kv_pools[ratio]
        slot_size = getattr(source_pool, "kernel_page_size", config.page_size) if ratio == 128 else config.page_size
        paged(f"deepseek_v4_c{ratio}", buffers, item_bytes, slot_size,
              device_pool._unified_kv or assembler._dsv4_page_aligned_only(source_pool))
        rope = assembler._dsv4_rope_sibling(device_pool, ratio)
        if rope is not None:
            rope_name, rope_buffers, rope_bytes = rope
            paged(rope_name, rope_buffers, rope_bytes, slot_size, True)
    if mappings.c4:
        for region in assembler._dsv4_indexer_regions(device_pool, config.page_size):
            paged(region.name, region.device_buffers, region.item_bytes,
                  region.slot_page_size, region.page_aligned_only)
        if not device_pool._unified_kv:
            for name, state_pools in (
                ("deepseek_v4_c4_state", device_pool.compress_state_pools),
                ("deepseek_v4_c4_indexer_state", device_pool.indexer_compress_state_pools),
            ):
                bundle.pools[name] = DeepSeekV4StateHostPool(
                    pool_name=name,
                    state_pools=[state_pools[i] for i in mappings.c4_state_global_layers],
                    num_host_pages=pages, swa_page_size=device_pool.swa_page_size,
                    layout=config.layout, pin_memory=False, device="cpu", allocator_type="default",
                )
    for name, source_pool, buffers, _ in _dsv4_low_ratio_regions_compat(
        assembler, device_pool, config.page_size, ratios
    ):
        paged(name, buffers, buffers[0][0].nbytes, config.page_size, True)
    bundle.report.update(compress_ratios=ratios, unified_kv=device_pool._unified_kv)


def build_native_pools(model, requirements, config, platform, dtype):
    import torch

    family = native_family(requirements)
    combined = "mamba" in requirements.pool_names
    if combined and family != "dsa":
        raise PoolProbeUnavailable("combined sparse-indexer + recurrent-state assembly needs a unified native provider")
    bundle = NativePools(report={"family": family, "kv_cache_dtype": str(dtype)})
    try:
        with cpu_allocation_budget(torch, config.max_host_mib * 1024 * 1024 // 8) as used:
            builder = (_build_dsa_recurrent if combined else
                       {"deepseek_v4": _build_dsv4, "dsa": _build_dsa,
                        "minimax_sparse": _build_minimax}[family])
            builder(bundle, model, config, platform, dtype)
            _describe(bundle, int(getattr(model, "window_size", 1)), config.page_size)
        bundle.report["constructor_factory_bytes"] = used[0]
        bundle.report["dsv4_attn_backend"] = config.dsv4_attn_backend if family == "deepseek_v4" else None
        bundle.report["fp4_indexer"] = config.fp4_indexer if family == "deepseek_v4" else None
        bundle.report["coverage"] = "host_storage_contract_only"
        bundle.report["page_mapping_scope"] = "per_pool_physical_pages; engine transfer scheduling not executed"
        if len(bundle.pools) < 2 and not bundle.report.get("integrated_payloads"):
            raise PoolProbeUnavailable("native producer did not expose physical side payloads")
        return bundle
    except Exception:
        bundle.close()
        raise
