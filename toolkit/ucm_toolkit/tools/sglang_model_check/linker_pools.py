"""Reuse SGLang's device-pool dispatcher, request pool and allocator."""

import inspect
from copy import copy
from types import SimpleNamespace

from .native_probe import _dsv4_main_layer_ratios, cache_builder, native_family
from .runner import CheckFailure, _native_cache_dtype


def _assign_supported(builder, **values):
    """Set only attributes declared by this SGLang's slots dataclass."""
    fields = getattr(type(builder), "__dataclass_fields__", None)
    if fields is None:
        for name, value in values.items():
            setattr(builder, name, value)
        return
    for name, value in values.items():
        if name in fields:
            setattr(builder, name, value)


def _struct_kwargs(struct_cls, values):
    """Drop keyword arguments the installed msgspec struct does not declare."""
    fields = getattr(struct_cls, "__struct_fields__", None)
    if fields is None:
        return values
    return {name: value for name, value in values.items() if name in fields}


def build_linker_pools(config, model, requirements, environment, layer_range=None):
    from sglang.srt.configs.hybrid_arch import (
        hybrid_gdn_config,
        hybrid_kda_config,
        mambaish_config,
    )
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
        resolve_compress_state_ring_size,
    )
    from sglang.srt.mem_cache.kv_cache_configurator import (
        _get_dsv4_compress_state_dtypes,
        _PoolSizes,
    )
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    family = native_family(requirements)
    model = copy(model)
    pages = config.pages + 2
    size = pages * config.page_size
    model.context_len = size
    if family == "deepseek_v4":
        model.compress_ratios = _dsv4_main_layer_ratios(model)
    start, end = layer_range or (0, model.num_hidden_layers)
    dtype = (
        _native_cache_dtype(model, config, family, environment["platform"])
        if family
        else model.dtype
    )
    builder = cache_builder(
        model, config, dtype, start, end, is_mla=requirements.is_mla
    )
    try:
        from sglang.srt.runtime_context import get_server_args

        server_args = get_server_args()
    except Exception:
        server_args = None
    # KVCacheConfigurator is version-specific (slots dataclass). Only assign
    # attributes this installation declares; newer fields are optional extras.
    _assign_supported(
        builder,
        device=environment["device_name"],
        gpu_id=0,
        attn_dp_size=1,
        pp_size=1,
        pp_group=None,
        server_args=server_args,
        is_hybrid_swa_compress=False,
        is_hybrid_swa_mtp_draft=False,
        mambaish_config=mambaish_config(model),
        hybrid_gdn_config=hybrid_gdn_config(model),
        hybrid_kda_config=hybrid_kda_config(model),
        spec_algorithm=SpeculativeAlgorithm.NONE,
        is_hybrid_swa=bool(getattr(model, "sliding_window_size", None)),
        sliding_window_size=getattr(model, "sliding_window_size", None),
        draft_parallel_layout=None,
        draft_model_idx=None,
        draft_owns_its_depth=False,
        draft_swa_full_capacity=False,
        modelslim_cache_layout=None,
        kv_shard_rank=None,
        kv_shard_size=1,
        kv_shard_spec=None,
        kv_shard_group=None,
        _unified_plan=None,
        _unified_sized=None,
        token_to_kv_pool_allocator=None,
        model=SimpleNamespace(quant_config=None),
        spec_aux_config=SimpleNamespace(),
    )
    request_pool = builder._build_req_to_token_pool(max_num_reqs=2)
    state = getattr(request_pool, "state", None)
    c4_dtype, c128_dtype = _get_dsv4_compress_state_dtypes()
    sizes = _PoolSizes(
        **_struct_kwargs(
            _PoolSizes,
            dict(
                max_total_num_tokens=size,
                max_running_requests=2,
                full_max_total_num_tokens=size,
                swa_max_total_num_tokens=size,
                c4_max_total_num_tokens=max(size // 4, config.page_size),
                c128_max_total_num_tokens=max(size // 128, config.c128_page_size),
                c4_state_pool_size=pages * resolve_compress_state_ring_size(4),
                c128_state_pool_size=pages * resolve_compress_state_ring_size(128),
                c4_state_dtype=c4_dtype,
                c128_state_dtype=c128_dtype,
            ),
        )
    )
    pool_kwargs = dict(
        sizes=sizes,
        is_dsa_model=family == "dsa",
        req_to_token_pool=request_pool,
    )

    def model_kind_kwargs(build):
        if "is_compressed_kv_model" in inspect.signature(build).parameters:
            from sglang.srt.configs.model_config import uses_compressed_kv_attention

            return {
                "is_compressed_kv_model": uses_compressed_kv_attention(model.hf_config)
            }
        if "is_dsv4_model" in inspect.signature(build).parameters:
            return {"is_dsv4_model": family == "deepseek_v4"}
        raise CheckFailure(
            "CHECKER_RUNTIME_POOL_PROBE_REQUIRED",
            "linker_pool_construction",
            "Unrecognized SGLang pool factory signature: "
            + str(inspect.signature(build)),
        )

    pool_kwargs.update(model_kind_kwargs(builder._build_token_to_kv_pool))
    # Unified-recurrent SGLang versions pass an external state pool here; older
    # releases build it inside the pool factory and must not receive it.
    if "state" in inspect.signature(builder._build_token_to_kv_pool).parameters:
        pool_kwargs["state"] = state
    pool = builder._build_token_to_kv_pool(**pool_kwargs)
    allocator = builder._build_token_to_kv_pool_allocator(
        sizes=sizes,
        token_to_kv_pool=pool,
        req_to_token_pool=request_pool,
        token_to_kv_pool_allocator=None,
        **model_kind_kwargs(builder._build_token_to_kv_pool_allocator),
    )
    if state is not None:
        try:
            allocator.state = state
        except AttributeError:
            pass
    return pool, allocator, request_pool
