"""GSA attention patches for vLLM 0.23.0."""

from __future__ import annotations

from types import FunctionType

from vllm.v1.attention.backend import AttentionType

from ucm.integration.vllm.patch.utils import patch_or_inject, when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


def unified_attention_with_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
    output_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
    kv_cache_dummy_dep: torch.Tensor | None = None,
) -> None:
    # kv_cache_dummy_dep is not used but accepting it creates a data dependency
    # that ensures torch.compile preserves ordering between KV cache update and
    # attention forward.
    del kv_cache_dummy_dep
    layer_name = _resolve_layer_name(layer_name)
    attn_metadata, self, kv_cache, _ = get_attention_context(layer_name)

    if not self.use_mla:
        forward_context: ForwardContext = get_forward_context()
        if attn_metadata is not None:
            if os.getenv("VLLM_HASH_ATTENTION") == "1":
                kv_cache, k_hash = kv_cache
            else:
                k_hash = None
            query, _, _, _ = maybe_execute_sparse_attention_begin(
                query, key, value, layer_name, forward_context, output, k_hash=k_hash
            )

    self.impl.forward(
        self,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=output,
        output_scale=output_scale,
        output_block_scale=output_block_scale,
    )

    if not self.use_mla:
        maybe_execute_sparse_attention_finished(
            query, key, value, output, layer_name, forward_context
        )


def __init__(
    self,
    num_heads: int,
    head_size: int,
    scale: float,
    num_kv_heads: int | None = None,
    alibi_slopes: list[float] | None = None,
    use_alibi_sqrt: bool | None = None,
    cache_config: CacheConfig | None = None,
    quant_config: QuantizationConfig | None = None,
    logits_soft_cap: float | None = None,
    per_layer_sliding_window: int | None = None,
    prefix: str = "",
    attn_type: str = AttentionType.DECODER,
    kv_sharing_target_layer_name: str | None = None,
    attn_backend: type[AttentionBackend] | None = None,
    head_size_v: int | None = None,
    **extra_impl_args,
) -> None:
    """
    The KV cache is stored inside this class and is accessed via
    `self.kv_cache`.
    """
    super(Attention, self).__init__()
    sliding_window: int | None
    if per_layer_sliding_window is not None:
        # per-layer sliding window
        sliding_window = per_layer_sliding_window
    elif cache_config is not None:
        # model-level sliding window
        sliding_window = cache_config.sliding_window
    else:
        sliding_window = None

    vllm_config = get_current_vllm_config()
    if cache_config is not None:
        kv_cache_dtype = cache_config.cache_dtype
        calculate_kv_scales = cache_config.calculate_kv_scales
    else:
        kv_cache_dtype = "auto"
        calculate_kv_scales = False

    # llm-compressor models declare an FP8 KV-cache scheme in their
    # checkpoint config. Honor it only when the user did not explicitly
    # pick a kv_cache_dtype; an explicit choice (e.g. bfloat16) must win.
    # The "auto" case is normally resolved upstream in
    # resolve_kv_cache_dtype_string, but we re-apply here defensively in
    # case anything bypassed that path.
    kv_cache_scheme = getattr(quant_config, "kv_cache_scheme", None)
    if kv_cache_scheme is not None and kv_cache_dtype == "auto":
        kv_cache_dtype = "fp8"
        calculate_kv_scales = False
        if cache_config is not None:
            cache_config.cache_dtype = "fp8"
            cache_config.calculate_kv_scales = False

    # Check if per-head quant scales are required based on kv_cache_scheme
    use_per_head_quant_scales = (
        kv_cache_scheme is not None and kv_cache_scheme.get("strategy") == "attn_head"
    )
    self.use_mla = vllm_config.model_config.use_mla

    # Skip quantization for specified layers
    if cache_config is not None and cache_config.kv_cache_dtype_skip_layers:
        from vllm.model_executor.models.utils import extract_layer_index

        skip = False
        # Check attention type
        if (
            sliding_window is not None
            and "sliding_window" in cache_config.kv_cache_dtype_skip_layers
        ):
            skip = True
        # Check layer index
        layer_idx = extract_layer_index(prefix)
        if str(layer_idx) in cache_config.kv_cache_dtype_skip_layers:
            skip = True
        if skip:
            kv_cache_dtype = "auto"
            calculate_kv_scales = False
        logger.debug(
            "Layer %s: kv_cache_dtype=%s, sliding_window=%s",
            prefix,
            kv_cache_dtype,
            sliding_window,
        )

    self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
        kv_cache_dtype, vllm_config.model_config
    )
    self.kv_cache_dtype = kv_cache_dtype
    self.calculate_kv_scales = calculate_kv_scales
    if num_kv_heads is None:
        num_kv_heads = num_heads
    assert (
        num_heads % num_kv_heads == 0
    ), f"num_heads ({num_heads}) is not divisible by num_kv_heads ({num_kv_heads})"
    self.quant_config = quant_config
    self.layer_name = prefix

    self.num_heads = num_heads
    self.head_size = head_size
    self.head_size_v = self.head_size if head_size_v is None else head_size_v
    self.num_kv_heads = num_kv_heads
    self.sliding_window = sliding_window
    self.has_sink = extra_impl_args.get("sinks") is not None

    # NOTE: model_config may be None during certain tests
    model_config = vllm_config.model_config
    self.use_mm_prefix = model_config is not None and model_config.is_mm_prefix_lm

    # During model initialization, the default dtype is set as the model
    # weight and activation dtype.
    dtype = torch.get_default_dtype()
    if attn_backend is None:
        self.attn_backend = get_attn_backend(
            head_size,
            dtype,
            kv_cache_dtype,
            use_mla=False,
            has_sink=self.has_sink,
            use_mm_prefix=self.use_mm_prefix,
            use_per_head_quant_scales=use_per_head_quant_scales,
            attn_type=attn_type,
        )
    else:
        self.attn_backend = attn_backend
    backend_supports_alibi_sqrt = self.attn_backend.supports_alibi_sqrt()
    use_alibi_sqrt = use_alibi_sqrt if use_alibi_sqrt else False
    if use_alibi_sqrt and not backend_supports_alibi_sqrt:
        raise ValueError(
            f"use_alibi_sqrt is not supported by backend "
            f"{self.attn_backend.get_name()}."
        )
    self.use_alibi_sqrt = bool(use_alibi_sqrt)
    if backend_supports_alibi_sqrt:
        extra_impl_args["use_alibi_sqrt"] = self.use_alibi_sqrt
    # prefix caching + batch invariance is currently not supported for
    # FLASHINFER and TRITON_MLA.
    if (
        cache_config is not None
        and cache_config.enable_prefix_caching
        and envs.VLLM_BATCH_INVARIANT
        and (
            self.attn_backend.get_name() == "FLASHINFER"
            or self.attn_backend.get_name() == "TRITON_MLA"
        )
    ):
        logger.warning_once(
            "Disabling prefix caching for FLASHINFER/TRITON_MLA "
            "with batch invariance, as it is not yet supported.",
        )
        cache_config.enable_prefix_caching = False

    if extra_impl_args.get("chunk_lookback", -1) > -1:
        assert self.attn_backend.get_name() == "TRITON_ATTN", (
            f"Chunked attention with lookback requires the Triton backend, "
            f"but got {self.attn_backend.get_name()}."
        )

    if self.attn_backend.get_name() == "FLEX_ATTENTION":
        block_m = vllm_config.attention_config.flex_attn_block_m
        block_n = vllm_config.attention_config.flex_attn_block_n

        if envs.VLLM_BATCH_INVARIANT and cache_config is not None:
            if block_m is not None and block_m > cache_config.block_size:
                raise ValueError(
                    f"flex_attn_block_m ({block_m}) must be "
                    f"<= cache block size ({cache_config.block_size}) for "
                    f"batch invariance"
                )
            if block_n is not None and block_n > cache_config.block_size:
                raise ValueError(
                    f"flex_attn_block_n ({block_n}) must be "
                    f"<= cache block size ({cache_config.block_size}) for "
                    f"batch invariance"
                )

        if block_m is not None:
            extra_impl_args.setdefault("block_m", block_m)
        if block_n is not None:
            extra_impl_args.setdefault("block_n", block_n)

    impl_cls = self.attn_backend.get_impl_cls()
    self.impl = impl_cls(  # type: ignore[assignment]  # impl_cls always returns an AttentionImpl subclass
        num_heads,
        head_size,
        scale,
        num_kv_heads,
        alibi_slopes,
        sliding_window,
        kv_cache_dtype,
        logits_soft_cap,
        attn_type,
        kv_sharing_target_layer_name,
        **extra_impl_args,
    )
    self.backend = AttentionBackendEnum[self.attn_backend.get_name()]
    self.dtype = dtype

    # For cuda-alike (CUDA and ROCM) and cpu platforms, we control how
    # torch.compile works by registering the attention as one giant
    # opaque custom op. For other platforms, we directly call them
    # and let torch.compile handle them.
    self.use_direct_call = not current_platform.opaque_attention_op()

    compilation_config = vllm_config.compilation_config
    if prefix in compilation_config.static_forward_context:
        raise ValueError(f"Duplicate layer name: {prefix}")
    compilation_config.static_forward_context[prefix] = self
    self.attn_type = attn_type

    if kv_sharing_target_layer_name is not None:
        validate_kv_sharing_target(
            prefix,
            kv_sharing_target_layer_name,
            compilation_config.static_forward_context,
        )
    self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

    # use a placeholder kv cache tensor during init, which will be replaced
    # by bind_kv_cache
    # this variable will not be accessed if use_direct_call is True
    self.kv_cache = torch.tensor([])

    # Initialize KV cache quantization attributes
    _init_kv_cache_quant(self, quant_config, prefix)

    # for attn backends supporting query quantization
    self.query_quant = None
    if (
        self.impl.supports_quant_query_input
        and (self.kv_cache_dtype.startswith("fp8") or self.kv_cache_dtype == "nvfp4")
        and not self.kv_cache_dtype.endswith("per_token_head")
    ):
        is_per_head = (
            hasattr(self, "q_scale") and self.q_scale.numel() == self.num_kv_heads
        )
        block_size = self.head_size * self.num_heads // self.num_kv_heads
        self.query_quant = QuantFP8(
            static=True,
            group_shape=(
                GroupShape(-1, block_size) if is_per_head else GroupShape.PER_TENSOR
            ),
        )


def maybe_execute_sparse_attention_begin(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: str,
    forward_context: ForwardContext,
    output: Optional[torch.Tensor] = None,
    indexer_cache: Optional[torch.Tensor] = None,
    indexer_scale: Optional[torch.Tensor] = None,
    vllm_indexer_cache: Optional[torch.Tensor] = None,
    vllm_indexer_scale: Optional[torch.Tensor] = None,
    phase: Optional[str] = None,
    k_hash: Optional[torch.Tensor] = None,
    decode_ql_nope: Optional[torch.Tensor] = None,
    decode_q_pe: Optional[torch.Tensor] = None,
    has_indexer: Optional[bool] = True,
    use_li_c8: Optional[bool] = False,
):
    from ucm.sparse.state import get_ucm_sparse, has_ucm_sparse

    if not has_ucm_sparse():
        return query, key, value, output

    ucm_sparse = get_ucm_sparse()

    attn_metadata = forward_context.attn_metadata
    if attn_metadata is None:
        return query, key, value, output

    return ucm_sparse.attention_begin(
        query,
        key,
        value,
        layer_name,
        forward_context,
        indexer_cache,
        indexer_scale,
        vllm_indexer_cache,
        vllm_indexer_scale,
        output,
        phase,
        k_hash,
        decode_ql_nope,
        decode_q_pe,
        has_indexer,
        use_li_c8,
    )


def maybe_execute_sparse_attention_finished(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_output: torch.Tensor,
    layer_name: str,
    forward_context: ForwardContext,
    phase: Optional[str] = None,
    has_indexer: Optional[bool] = True,
    use_li_c8: Optional[bool] = False,
):
    from ucm.sparse.state import get_ucm_sparse, has_ucm_sparse

    if not has_ucm_sparse():
        return

    ucm_sparse = get_ucm_sparse()

    attn_metadata = forward_context.attn_metadata
    if attn_metadata is None:
        return

    ucm_sparse.attention_finished(
        query,
        key,
        value,
        attn_output,
        layer_name,
        forward_context,
        phase,
        has_indexer,
        use_li_c8,
    )


@when_imported("vllm.model_executor.layers.attention.attention")
def _patch(mod):
    try:
        import os as _os

        if not hasattr(mod, "os"):
            setattr(mod, "os", _os)
    except Exception as e:
        logger.warning(f"GSA attention os injection failed {e}")

    if hasattr(mod, "unified_attention_with_output"):
        _patched = FunctionType(
            unified_attention_with_output.__code__,
            vars(mod),
            name=unified_attention_with_output.__name__,
            argdefs=unified_attention_with_output.__defaults__,
            closure=unified_attention_with_output.__closure__,
        )
        _patched.__annotations__ = unified_attention_with_output.__annotations__
        _patched.__kwdefaults__ = unified_attention_with_output.__kwdefaults__
        patch_or_inject(mod, "unified_attention_with_output", _patched)
        logger.info("UCM GSA unified_attention_with_output patch applied")

    if hasattr(mod.Attention, "__init__"):
        _patched = FunctionType(
            __init__.__code__,
            vars(mod),
            name=__init__.__name__,
            argdefs=__init__.__defaults__,
            closure=__init__.__closure__,
        )
        _patched.__annotations__ = __init__.__annotations__
        _patched.__kwdefaults__ = __init__.__kwdefaults__
        patch_or_inject(mod.Attention, "__init__", _patched)
        logger.info("UCM GSA Attention.__init__ patch applied")

    if not hasattr(mod, "maybe_execute_sparse_attention_begin"):
        _patched = FunctionType(
            maybe_execute_sparse_attention_begin.__code__,
            vars(mod),
            name=maybe_execute_sparse_attention_begin.__name__,
            argdefs=maybe_execute_sparse_attention_begin.__defaults__,
            closure=maybe_execute_sparse_attention_begin.__closure__,
        )
        _patched.__annotations__ = maybe_execute_sparse_attention_begin.__annotations__
        _patched.__kwdefaults__ = maybe_execute_sparse_attention_begin.__kwdefaults__
        mod.maybe_execute_sparse_attention_begin = _patched
        logger.info("UCM GSA maybe_execute_sparse_attention_begin injected")

    if not hasattr(mod, "maybe_execute_sparse_attention_finished"):
        _patched = FunctionType(
            maybe_execute_sparse_attention_finished.__code__,
            vars(mod),
            name=maybe_execute_sparse_attention_finished.__name__,
            argdefs=maybe_execute_sparse_attention_finished.__defaults__,
            closure=maybe_execute_sparse_attention_finished.__closure__,
        )
        _patched.__annotations__ = (
            maybe_execute_sparse_attention_finished.__annotations__
        )
        _patched.__kwdefaults__ = maybe_execute_sparse_attention_finished.__kwdefaults__
        mod.maybe_execute_sparse_attention_finished = _patched
        logger.info("UCM GSA maybe_execute_sparse_attention_finished injected")
