"""Isolated implementation of the SGLang/UCM compatibility check."""

from __future__ import annotations

import contextlib
import inspect
import socket
import traceback
import uuid
from types import SimpleNamespace
from dataclasses import replace
from typing import Any, Iterator

from .compatibility import (
    find_missing_capabilities,
    infer_capabilities,
    requirements_from_facts,
)
from .config import CheckConfig
from .result import CheckResult, emit_result


class CheckFailure(RuntimeError):
    def __init__(self, code: str, stage: str, reason: str):
        super().__init__(reason)
        self.code = code
        self.stage = stage
        self.reason = reason


def _model_facts(model_config: Any) -> dict[str, Any]:
    from .sglang_compat import collect_model_facts

    return collect_model_facts(model_config)


def _redirect_device(value: Any) -> Any:
    if isinstance(value, str) and value.split(":", 1)[0] in {
        "cuda",
        "npu",
        "xpu",
        "musa",
    }:
        return "meta"
    try:
        if getattr(value, "type", None) in {"cuda", "npu", "xpu", "musa"}:
            return "meta"
    except Exception:
        pass
    return value


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextlib.contextmanager
def _single_rank_model_parallel() -> Iterator[None]:
    """Initialize the one-rank groups required by SGLang model constructors."""
    import torch
    from sglang.srt.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    owns_environment = not torch.distributed.is_initialized()
    try:
        if owns_environment:
            init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=0,
                distributed_init_method=f"tcp://127.0.0.1:{_free_local_port()}",
                backend="gloo",
            )
            initialize_model_parallel(backend="gloo")
        yield
    finally:
        if owns_environment and torch.distributed.is_initialized():
            cleanup_dist_env_and_memory()


@contextlib.contextmanager
def _meta_device_guard(torch: Any) -> Iterator[None]:
    """Redirect explicit accelerator allocations made during model construction."""
    factories = ("empty", "zeros", "ones", "full", "rand", "randn", "arange")
    originals: dict[str, Any] = {}

    for name in factories:
        original = getattr(torch, name)
        originals[name] = original

        def wrapper(*args: Any, __original: Any = original, **kwargs: Any) -> Any:
            if "device" in kwargs:
                kwargs["device"] = _redirect_device(kwargs["device"])
            return __original(*args, **kwargs)

        setattr(torch, name, wrapper)

    original_module_to = torch.nn.Module.to
    original_tensor_to = torch.Tensor.to

    def module_to(module: Any, *args: Any, **kwargs: Any) -> Any:
        if args:
            args = (_redirect_device(args[0]), *args[1:])
        if "device" in kwargs:
            kwargs["device"] = _redirect_device(kwargs["device"])
        return original_module_to(module, *args, **kwargs)

    def tensor_to(tensor: Any, *args: Any, **kwargs: Any) -> Any:
        if args:
            args = (_redirect_device(args[0]), *args[1:])
        if "device" in kwargs:
            kwargs["device"] = _redirect_device(kwargs["device"])
        return original_tensor_to(tensor, *args, **kwargs)

    torch.nn.Module.to = module_to
    torch.Tensor.to = tensor_to
    try:
        yield
    finally:
        torch.nn.Module.to = original_module_to
        torch.Tensor.to = original_tensor_to
        for name, original in originals.items():
            setattr(torch, name, original)


def _check_meta_model(model_config: Any) -> Any:
    import torch
    from sglang.srt.configs.load_config import LoadConfig
    from sglang.srt.model_loader.loader import (
        _get_quantization_config,
        _initialize_model,
    )
    from sglang.srt.model_executor.model_runner_components.layer_setup import (
        resolve_layer_indices,
    )
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
    from .sglang_compat import server_args_guard

    load_config = LoadConfig()
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(model_config.dtype)
        with server_args_guard():
            quant_config = _get_quantization_config(model_config, load_config)
            with _single_rank_model_parallel():
                with torch.device("meta"), _meta_device_guard(torch):
                    model = _initialize_model(model_config, load_config, quant_config)
                layer_args = dict(
                    model=model, model_config=model_config,
                    is_draft_worker=False, spec_algorithm=SpeculativeAlgorithm.NONE,
                )
                parameters = inspect.signature(resolve_layer_indices).parameters
                resolve_layer_indices(**{
                    key: value for key, value in layer_args.items() if key in parameters
                })
                return model
    except Exception as exc:
        raise CheckFailure(
            "META_MODEL_INIT_UNSUPPORTED",
            "meta_model_init",
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        ) from exc
    finally:
        torch.set_default_dtype(old_dtype)


def _make_storage_config(config: CheckConfig, requirements: Any) -> Any:
    from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig

    extra = {
        "kv_connector_extra_config": {
            "ucm_connector_name": config.connector_name,
            "ucm_connector_module_path": config.connector_module_path,
            "ucm_connector_config": {
                "storage_backends": config.storage_backends,
            },
        }
    }
    return HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=requirements.is_mla,
        enable_storage_metrics=False,
        is_page_first_layout=config.layout.startswith("page_first"),
        model_name=config.model,
        extra_config=extra,
    )


def _make_host_pool(
    model_config: Any, requirements: Any, config: CheckConfig, pool_spec: Any = None
) -> Any:
    import torch

    layer_num = int(
        max(
            getattr(model_config, "num_hidden_layers", 0),
            getattr(model_config, "num_attention_layers", 0),
        )
    )
    if pool_spec is not None:
        layer_num = len(pool_spec.layers)
    if layer_num <= 0:
        raise CheckFailure("KV_POOL_UNSUPPORTED", "host_pool", "invalid layer count")

    device_pool = SimpleNamespace(
        store_dtype=model_config.dtype,
        size=config.page_size * (config.pages * 2 + 1),
        start_layer=0,
        end_layer=layer_num,
        layer_num=layer_num,
        device="cpu",
        layer_shard_enabled=False,
        layer_shard_size=layer_num,
    )
    if requirements.is_mla:
        from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost

        device_pool.kv_lora_rank = int(model_config.kv_lora_rank)
        device_pool.qk_rope_head_dim = int(model_config.qk_rope_head_dim)
        device_pool.index_head_dim = getattr(model_config, "index_head_dim", None)
        pool_cls = MLATokenToKVPoolHost
    else:
        from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost

        device_pool.head_num = int(
            getattr(model_config, "num_key_value_heads", None)
            or model_config.num_attention_heads
        )
        device_pool.head_dim = int(model_config.head_dim)
        device_pool.v_head_dim = int(getattr(model_config, "v_head_dim", None) or model_config.head_dim)
        if pool_spec is not None:
            device_pool.head_num = pool_spec.head_num
            device_pool.head_dim = pool_spec.head_dim
            device_pool.v_head_dim = pool_spec.v_head_dim
        if device_pool.head_dim != device_pool.v_head_dim:
            raise CheckFailure(
                "UCM_POOL_GEOMETRY_UNSUPPORTED", "host_pool",
                "UCM primary v1 stores equal-size K/V components; asymmetric K/V requires a production adapter change",
            )
        pool_cls = MHATokenToKVPoolHost

    width = (
        device_pool.kv_lora_rank + device_pool.qk_rope_head_dim
        if requirements.is_mla
        else 2 * device_pool.head_num * device_pool.head_dim
    )
    # Cover device-description CPU buffers, host buffers, source snapshots and
    # comparison temporaries. This is a conservative estimate, not RSS accounting.
    estimated = device_pool.size * layer_num * width * model_config.dtype.itemsize * 8
    if estimated > config.max_host_mib * 1024 * 1024:
        raise CheckFailure(
            "CHECKER_MEMORY_LIMIT", "host_pool",
            f"estimated probe memory {estimated} bytes exceeds --max-host-mib={config.max_host_mib}; reduce --pages/--page-size",
        )

    try:
        kwargs = dict(
            host_to_device_ratio=1.0,
            host_size=0,
            page_size=config.page_size,
            layout=config.layout,
            pin_memory=False,
            device="cpu",
            allocator_type="default",
        )
        parameters = inspect.signature(pool_cls).parameters
        if "device_pool" in parameters:
            if requirements.is_mla:
                from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

                device_pool = MLATokenToKVPool(
                    size=device_pool.size, page_size=config.page_size,
                    dtype=model_config.dtype, kv_lora_rank=device_pool.kv_lora_rank,
                    qk_rope_head_dim=device_pool.qk_rope_head_dim, layer_num=layer_num,
                    device="cpu", enable_memory_saver=False,
                    start_layer=0, end_layer=layer_num,
                )
            else:
                # Legacy Host constructors read native row geometry and pointer
                # tables. Use the real CPU pool, not a partial attribute mock.
                from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

                device_pool = MHATokenToKVPool(
                    size=device_pool.size, page_size=config.page_size,
                    dtype=model_config.dtype, head_num=device_pool.head_num,
                    head_dim=device_pool.head_dim, layer_num=layer_num,
                    device="cpu", enable_memory_saver=False,
                    start_layer=0, end_layer=layer_num, enable_alt_stream=False,
                )
            return pool_cls(device_pool=device_pool, **kwargs)
        if "buffer_info" not in parameters:
            raise CheckFailure("CHECKER_HOST_POOL_API_UNSUPPORTED", "host_pool",
                               f"unknown constructor for {pool_cls.__name__}")
        # New SGLang separates physical buffer descriptions from pool capacity.
        # These CPU buffers describe geometry only; no device transfer is tested.
        common = dict(device_capacity_tokens=device_pool.size,
                      start_layer=0, end_layer=layer_num)
        if requirements.is_mla:
            from sglang.srt.mem_cache.pool_host.mla import MLAHostAllocation

            info = MLAHostAllocation(
                dtype=model_config.dtype,
                kv_lora_rank=device_pool.kv_lora_rank,
                qk_rope_head_dim=device_pool.qk_rope_head_dim,
                kv_cache_dim=width,
                size_per_token=layer_num * width * model_config.dtype.itemsize,
            )
            common["target_layer_num"] = layer_num
        else:
            from sglang.srt.mem_cache.device_pool_info import (
                KVBufferLayout, KVBuffers, MHABufferInfo,
            )

            shape = (device_pool.size, device_pool.head_num, device_pool.head_dim)
            k = tuple(torch.empty(shape, dtype=model_config.dtype) for _ in range(layer_num))
            v = tuple(torch.empty_like(item) for item in k)
            info = MHABufferInfo(
                buffers=KVBuffers(
                    k=k, v=v, layout=KVBufferLayout.NHD,
                    k_data_ptrs=torch.tensor([item.data_ptr() for item in k], dtype=torch.uint64),
                    v_data_ptrs=torch.tensor([item.data_ptr() for item in v], dtype=torch.uint64),
                ),
                page_size=config.page_size, head_num=device_pool.head_num,
                head_dim=device_pool.head_dim, v_head_dim=device_pool.v_head_dim,
                dtype=model_config.dtype, device=torch.device("cpu"),
            )
        return pool_cls(buffer_info=info, **common, **kwargs)
    except CheckFailure:
        raise
    except Exception as exc:
        raise CheckFailure(
            "HICACHE_HOST_POOL_UNSUPPORTED",
            "host_pool",
            f"{type(exc).__name__}: {exc}",
        ) from exc


def _fill_pattern(tensor: Any, seed: int = 20260909) -> None:
    import torch

    generator = torch.Generator(device=tensor.device)
    generator.manual_seed(seed)
    if not tensor.is_floating_point():
        tensor.random_(1, 127, generator=generator)
        return
    try:
        tensor.uniform_(-1.0, 1.0, generator=generator)
    except RuntimeError:
        # Some low-precision cache dtypes do not implement uniform_.  A
        # non-zero sentinel still detects missing or partial transfers.
        tensor.fill_(0.5)


def _pool_name(value: str) -> Any:
    """Use SGLang's enum when known while tolerating future pool names."""
    from sglang.srt.mem_cache.hicache_storage import PoolName

    try:
        return PoolName(value)
    except ValueError:
        return value


def _result_for_pool(results: Any, pool_name: Any) -> Any:
    if not isinstance(results, dict):
        return None
    for key in (pool_name, str(pool_name), getattr(pool_name, "value", None)):
        if key is not None and key in results:
            return results[key]
    return None


def _require_page_results(
    results: Any, pool_names: list[Any], pages: int, code: str, stage: str
) -> None:
    for pool_name in pool_names:
        pool_results = _result_for_pool(results, pool_name)
        if (
            not isinstance(pool_results, (list, tuple))
            or len(pool_results) != pages
            or not all(pool_results)
        ):
            raise CheckFailure(
                code,
                stage,
                f"pool {pool_name}: expected {pages} successful pages, "
                f"got {pool_results!r}",
            )


def _roundtrip_v1(
    storage: Any,
    host_pool: Any,
    keys: list[str],
    config: Any,
    is_mla: bool,
) -> dict[str, Any]:
    import torch

    tokens = config.pages * host_pool.page_size
    snapshots = {}
    for page in range(config.pages):
        source = _pool_page_views(host_pool, "kv", page, is_mla)
        target = _pool_page_views(host_pool, "kv", config.pages + page, is_mla)
        snapshots[page] = []
        for index, (src, dst) in enumerate(zip(source, target)):
            _fill_pattern(src, seed=20260909 + page * 101 + index)
            snapshots[page].append(src.clone())
            dst.zero_()
    source_indices = torch.arange(0, tokens, dtype=torch.int64)
    target_indices = torch.arange(tokens, tokens * 2, dtype=torch.int64)

    written = storage.batch_set_v1(keys, source_indices)
    if len(written) != config.pages or not all(written):
        raise CheckFailure("DUMP_FAILED", "roundtrip_dump", repr(written))
    hits = storage.batch_exists(keys)
    if hits != config.pages:
        raise CheckFailure(
            "LOAD_MISS",
            "roundtrip_exists",
            f"expected {config.pages} hit pages, got {hits}",
        )
    # Verify lookup termination as well as the all-hit happy path. Unique keys
    # prevent old test objects from turning a missing suffix into a false hit.
    miss = f"sglang-model-check-missing-{uuid.uuid4().hex}"
    for probe, expected_hits in (([miss, *keys], 0), ([*keys, miss], config.pages)):
        actual = storage.batch_exists(probe)
        if actual != expected_hits:
            raise CheckFailure("LOOKUP_CONTRACT_MISMATCH", "roundtrip_exists",
                               f"expected prefix {expected_hits}, got {actual}")
    loaded = storage.batch_get_v1(keys, target_indices)
    if len(loaded) != config.pages or not all(loaded):
        raise CheckFailure("LOAD_FAILED", "roundtrip_load", repr(loaded))
    bytes_compared = 0
    for page, components in snapshots.items():
        target = _pool_page_views(host_pool, "kv", config.pages + page, is_mla)
        for index, (expected, actual) in enumerate(zip(components, target)):
            if not torch.equal(expected, actual):
                raise CheckFailure("ROUNDTRIP_MISMATCH", "roundtrip_compare",
                                   f"kv page={page} component={index} differs")
            bytes_compared += expected.numel() * expected.element_size()
    return {
        "pages_written": config.pages,
        "pages_loaded": config.pages,
        "bytes_compared": bytes_compared,
    }


def _roundtrip_v2(
    storage: Any,
    pools: dict[str, Any],
    keys: list[str],
    config: Any,
    is_mla: bool,
    pool_specs: Any,
) -> dict[str, Any]:
    """Exercise real side pools with their own page units and sparse windows."""
    import torch
    from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolTransfer
    from .pool_probe import checkpoint_pages, expected_prefixes, persisted_positions

    anchor = pools["kv"]
    specs = {spec.name: spec for spec in pool_specs}
    logical = specs["kv"].kind == "logical"
    totals = (
        {"pages_written": 0, "pages_loaded": 0, "bytes_compared": 0}
        if logical else _roundtrip_v1(storage, anchor, keys, config, is_mla)
    )
    if len(specs) == 1:
        if logical:
            raise CheckFailure("CHECKER_RUNTIME_POOL_PROBE_REQUIRED", "roundtrip_prepare",
                               "logical anchor has no physical payload pools")
        totals.update(pool_names=["kv"], storage_api="v1_integrated_payload",
                      pool_specs={"kv": specs["kv"].report()})
        return totals
    windows = {spec.name: spec.window_pages for spec in pool_specs}
    written_positions = {"kv": set(range(config.pages))}
    source_transfers = []
    target_transfers = []
    snapshots = {}

    def descriptors(count, placeholders=False):
        return [PoolTransfer(
            name=_pool_name(name),
            keys=(
                ["__placeholder__"] * min(windows[name], count)
                if placeholders and windows[name]
                else keys[max(0, count - windows[name]):count] if windows[name]
                else keys[:count]
            ),
            hit_policy=(PoolHitPolicy.TRAILING_PAGES if windows[name] else PoolHitPolicy.ALL_PAGES),
        ) for name in specs if name != "kv"]

    def check_lookup(query, transfers, expected):
        hit = storage.batch_exists_v2(query, transfers)
        actual_pages = int(getattr(hit, "kv_hit_pages", -1))
        actual_prefixes = getattr(hit, "restorable_prefix_pages", None)
        if actual_pages != (max(expected) if expected else 0) or actual_prefixes != expected:
            raise CheckFailure(
                "LOOKUP_CONTRACT_MISMATCH", "roundtrip_exists_v2",
                f"expected restorable={expected}, got {hit!r}",
            )

    # Primary KV already exists, but no side payload has been persisted.
    check_lookup(keys, descriptors(config.pages, True), [])
    checkpoints = checkpoint_pages(config.pages)
    for pool_number, (name, spec) in enumerate(specs.items()):
        if name == "kv":
            continue
        pool = pools[name]
        positions = persisted_positions(config.pages, spec.window_pages, checkpoints)
        written_positions[name] = set(positions)
        for page in positions:
            source = _pool_page_views(pool, name, page, is_mla)
            target = _pool_page_views(pool, name, config.pages + page, is_mla)
            snapshots[name, page] = []
            for component, (src, dst) in enumerate(zip(source, target)):
                _fill_pattern(src, seed=20260909 + pool_number * 100003 + page * 101 + component)
                snapshots[name, page].append(src.clone())
                dst.zero_()
        page_size = int(pool.page_size)
        source_indices = torch.tensor([
            index for page in positions
            for index in range(page * page_size, (page + 1) * page_size)
        ], dtype=torch.int64)
        transfer_keys = [keys[page] for page in positions]
        source_transfers.append(
            PoolTransfer(
                name=_pool_name(name), host_indices=source_indices,
                keys=transfer_keys,
                hit_policy=(PoolHitPolicy.TRAILING_PAGES if spec.window_pages else PoolHitPolicy.ALL_PAGES),
            )
        )
        target_transfers.append(
            PoolTransfer(
                name=_pool_name(name),
                host_indices=source_indices + config.pages * page_size,
                keys=transfer_keys,
                hit_policy=(PoolHitPolicy.TRAILING_PAGES if spec.window_pages else PoolHitPolicy.ALL_PAGES),
            )
        )

    written = storage.batch_set_v2(source_transfers)
    for transfer in source_transfers:
        _require_page_results(written, [transfer.name], len(transfer.keys),
                              "DUMP_FAILED", "roundtrip_dump_v2")
    expected = expected_prefixes(config.pages, written_positions, windows)
    check_lookup(keys, descriptors(config.pages), expected)
    check_lookup(keys, descriptors(config.pages, True), expected)
    if config.pages > 1:
        shorter = config.pages - 1
        check_lookup(keys[:shorter], descriptors(shorter, True),
                     expected_prefixes(shorter, written_positions, windows))
    check_lookup([f"missing-{uuid.uuid4().hex}" for _ in keys], descriptors(config.pages), [])
    loaded = storage.batch_get_v2(target_transfers)
    for transfer in target_transfers:
        _require_page_results(loaded, [transfer.name], len(transfer.keys),
                              "LOAD_FAILED", "roundtrip_load_v2")
    for (name, page), components in snapshots.items():
        target = _pool_page_views(pools[name], name, config.pages + page, is_mla)
        for component, (expected_data, actual) in enumerate(zip(components, target)):
            if not torch.equal(expected_data, actual):
                raise CheckFailure("ROUNDTRIP_MISMATCH", "roundtrip_compare_v2",
                                   f"{name} page={page} component={component} differs")
            totals["bytes_compared"] += expected_data.numel() * expected_data.element_size()
    totals["pool_names"] = list(pools)
    totals["storage_api"] = "v2"
    totals["logical_anchor"] = logical
    totals["pool_specs"] = {name: spec.report() for name, spec in specs.items()}
    totals["persisted_pages"] = {name: sorted(value) for name, value in written_positions.items()}
    totals["restorable_prefix_pages"] = expected_prefixes(config.pages, written_positions, windows)
    totals["checks"] = ["cold_side_miss", "exact_keys", "placeholder_keys",
                        "shorter_prefix", "missing_primary", "component_data_compare"]
    return totals


def _pool_page_views(pool: Any, name: str, page: int, is_mla: bool) -> list[Any]:
    """Writable byte views of real CPU host pages, independent of tensor layout.

    Using the production page metadata also covers split scales, packed rows
    and mixed-dtype state components. The pool owns the backing memory for the
    lifetime of these views. No copies or fabricated generic KV buffers occur.
    """
    import ctypes
    import torch
    from ucm.integration.sglang.ucm_connector import SglangUcmConnector

    if str(pool.device).split(":")[0] != "cpu":
        raise CheckFailure("CHECKER_HOST_POOL_API_UNSUPPORTED", "page_metadata",
                           "the storage probe accepts CPU host buffers only")
    start = page * pool.page_size
    indices = torch.arange(start, start + pool.page_size, dtype=torch.int64)
    pointers, sizes = pool.get_page_buffer_meta(indices)
    page_ptrs, page_sizes = SglangUcmConnector._flatten_page_meta(pointers, sizes, 1)
    views = []
    for pointer, size in zip(page_ptrs[0], page_sizes[0]):
        size, pointer = int(size), int(pointer)
        if size == 0:
            continue
        if pointer <= 0 or size < 0:
            raise CheckFailure("CHECKER_HOST_POOL_API_UNSUPPORTED", "page_metadata",
                               f"invalid {name} page pointer/size")
        array = (ctypes.c_ubyte * size).from_address(pointer)
        views.append(torch.frombuffer(array, dtype=torch.uint8))
    if not views:
        raise CheckFailure("CHECKER_HOST_POOL_API_UNSUPPORTED", "page_metadata",
                           f"{name} has no physical page payload")
    return views


def _make_mamba_host_pool(spec: Any, config: CheckConfig) -> Any:
    import torch
    from sglang.srt.mem_cache.device_pool_info import MambaStateBufferInfo
    from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost

    capacity = config.pages * 2 + 1
    shape_prefix = (len(spec.layers), capacity)
    info = MambaStateBufferInfo(
        temporal=torch.empty(shape_prefix + spec.temporal_shape,
                             dtype=spec.temporal_dtype, device="cpu"),
        conv=tuple(torch.empty(shape_prefix + shape, dtype=spec.conv_dtype,
                               device="cpu") for shape in spec.conv_shapes),
    )
    if "buffer_info" not in inspect.signature(MambaPoolHost).parameters:
        raise CheckFailure("CHECKER_HOST_POOL_API_UNSUPPORTED", "host_pool",
                           "installed MambaPoolHost has no buffer_info constructor")
    return MambaPoolHost(
        buffer_info=info, host_to_device_ratio=1.0, host_size=0,
        device_capacity=capacity, pin_memory=False, device="cpu",
        allocator_type="default", layout=config.layout,
    )


def _hybrid_specs(model_config: Any, requirements: Any, config: CheckConfig) -> Any:
    from .pool_probe import PoolProbeUnavailable, build_pool_specs
    from .sglang_compat import environment_info

    try:
        return build_pool_specs(model_config, requirements.pool_names,
                                config.page_size, environment_info()["platform"])
    except (PoolProbeUnavailable, AttributeError, ImportError, TypeError, ValueError,
            RuntimeError, NotImplementedError) as exc:
        raise CheckFailure("CHECKER_RUNTIME_POOL_PROBE_REQUIRED", "pool_description",
                           f"{type(exc).__name__}: {exc}") from exc


def _roundtrip(
    config: CheckConfig, model_config: Any, requirements: Any
) -> dict[str, Any]:
    from .native_probe import native_family, build_native_pools, ProbeMemoryLimit
    from .pool_probe import PoolProbeUnavailable
    from .sglang_compat import environment_info

    family = native_family(requirements)
    specs = (_hybrid_specs(model_config, requirements, config)
             if requirements.is_hybrid and not family else None)
    if specs is not None:
        # Sum every pool before allocating any of them; per-pool limits alone
        # do not protect a hybrid model's aggregate memory footprint.
        estimate = sum(
            (config.pages * 2 + 2)
            * spec.bytes_per_slot(model_config.dtype, config.page_size) * 8
            for spec in specs
        )
        if estimate > config.max_host_mib * 1024 * 1024:
            raise CheckFailure("CHECKER_MEMORY_LIMIT", "host_pool",
                               f"aggregate probe estimate {estimate} bytes exceeds --max-host-mib={config.max_host_mib}")

    from sglang.srt.mem_cache.hicache_storage import HiCacheStorage
    from ucm.integration.sglang.unifiedcache_store import UnifiedCacheStore

    pools = {}
    anchor_name = "kv"
    storage = None
    native_bundle = None
    try:
        capabilities = infer_capabilities(UnifiedCacheStore, HiCacheStorage)
        missing = find_missing_capabilities(requirements, capabilities)
        if missing:
            raise CheckFailure(
                "UCM_STORAGE_API_UNSUPPORTED",
                "capability_check",
                "; ".join(missing),
            )

        if family:
            platform = environment_info()["platform"]
            try:
                native_bundle = build_native_pools(
                    model_config, requirements, config, platform,
                    _native_cache_dtype(model_config, config, family, platform),
                )
            except ProbeMemoryLimit as exc:
                raise CheckFailure("CHECKER_MEMORY_LIMIT", "native_pool_construction", str(exc)) from exc
            except (PoolProbeUnavailable, ImportError, AttributeError, TypeError,
                    ValueError, RuntimeError, AssertionError) as exc:
                raise CheckFailure("CHECKER_RUNTIME_POOL_PROBE_REQUIRED", "native_pool_construction",
                                   f"{type(exc).__name__}: {exc}") from exc
            pools, specs = native_bundle.pools, native_bundle.specs
        elif specs is None:
            pools["kv"] = _make_host_pool(model_config, requirements, config)
        else:
            for spec in specs:
                if spec.kind == "logical":
                    from sglang.srt.mem_cache.memory_pool_host import LogicalHostPool

                    pools[spec.name] = LogicalHostPool(
                        (config.pages * 2 + 2) * config.page_size,
                        config.page_size, layout=config.layout,
                    )
                else:
                    pools[spec.name] = (
                        _make_mamba_host_pool(spec, config) if spec.kind == "mamba"
                        else _make_host_pool(model_config, requirements, config, spec)
                    )
        host_pool = pools[anchor_name]
        storage = UnifiedCacheStore(_make_storage_config(config, requirements))
        try:
            storage.register_mem_pool_host(host_pool)
            if specs is not None:
                for name, pool in pools.items():
                    storage.register_mem_host_pool_v2(pool, _pool_name(name))
        except (ValueError, NotImplementedError) as exc:
            raise CheckFailure("UCM_POOL_REGISTRATION_UNSUPPORTED", "pool_registration", str(exc)) from exc
        salt = uuid.uuid4().hex
        keys = [
            f"sglang-model-check-{salt}-{index}" for index in range(config.pages)
        ]
        if requirements.storage_api == "v2":
            totals = _roundtrip_v2(
                storage, pools, keys, config, requirements.is_mla, specs
            )
        else:
            totals = _roundtrip_v1(
                storage, host_pool, keys, config, requirements.is_mla
            )
            totals.update(pool_names=[anchor_name], storage_api="v1")
        totals["host_pool_classes"] = {
            name: type(pool).__name__ for name, pool in pools.items()
        }
        if native_bundle is not None:
            totals["native_provider"] = native_bundle.report
        return totals
    finally:
        try:
            if storage is not None:
                storage.close()
        finally:
            if native_bundle is not None:
                native_bundle.close()
            else:
                for pool in pools.values():
                    destroy = getattr(pool, "destroy", None)
                    if callable(destroy):
                        destroy()


def _native_cache_dtype(model_config, config, family, platform):
    import torch

    if config.kv_cache_dtype != "auto":
        name = {"fp8_e4m3": "float8_e4m3fn"}.get(config.kv_cache_dtype, config.kv_cache_dtype)
        return getattr(torch, name)
    if family == "deepseek_v4":
        return torch.bfloat16 if platform == "ascend" else torch.float8_e4m3fn
    return model_config.dtype


def run(config: CheckConfig) -> CheckResult:
    result = CheckResult(model=config.model, mode=config.mode)
    result.validation = {
        "scope": "single_rank_host_storage_contract",
        "meta_model": "not_run",
        "storage_io": "not_run",
        "runtime_pools": "not_run",
        "device_transfer": "not_run",
        "distributed_consistency": "not_run",
        "inference": "not_run",
    }
    try:
        from .sglang_compat import environment_info, refine_facts_from_model, server_args_guard
        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.mem_cache.hicache_storage import HiCacheStorage
        from ucm.integration.sglang.unifiedcache_store import UnifiedCacheStore

        model_config = ModelConfig(
            model_path=config.model,
            trust_remote_code=config.trust_remote_code,
            dtype=config.dtype,
        )
        result.environment = environment_info()
        result.environment["requested_platform"] = config.platform
        facts = _model_facts(model_config)
        result.model_info = facts

        if config.page_size == 0:
            automatic_page_size = 256 if facts.get("is_deepseek_v4") else 64
            if facts.get("is_dsa"):
                from sglang.srt.configs.model_config import get_dsa_index_kpool

                physical_page = 1 if result.environment["platform"] == "rocm" else 64
                automatic_page_size = physical_page * get_dsa_index_kpool(model_config.hf_config)
            config = replace(config, page_size=automatic_page_size)
        result.environment["probe_page_size"] = config.page_size
        result.environment["requested_kv_cache_dtype"] = config.kv_cache_dtype

        detected_platform = result.environment.get("platform")
        if config.platform != "auto" and config.platform != detected_platform:
            raise CheckFailure("ENVIRONMENT_UNSUPPORTED", "platform_check",
                               f"requested {config.platform!r}, detected {detected_platform!r}")
        if config.layout not in {"page_first", "page_first_direct"}:
            raise CheckFailure("UCM_LAYOUT_UNSUPPORTED", "capability_check",
                               "UCM requires page_first or page_first_direct")
        if facts["attention_arch"] == "MHA" and facts.get("head_dim") != facts.get("v_head_dim"):
            raise CheckFailure("UCM_POOL_GEOMETRY_UNSUPPORTED", "capability_check",
                               "UCM v1 primary K/V components require equal head dimensions")

        supported_platforms = facts.get("supported_platforms")
        if supported_platforms and detected_platform not in supported_platforms:
            result.model_info = facts
            result.fail(
                "PLATFORM_UNSUPPORTED",
                "platform_check",
                f"quantization {facts.get('quantization')!r} requires one of "
                f"{supported_platforms}, detected {detected_platform!r}",
            )
            return result

        model = None
        if not config.skip_meta_model:
            model = _check_meta_model(model_config)
            facts = refine_facts_from_model(facts, model)
            if facts.get("is_hybrid_swa"):
                # Follow the same precedence as the real ModelRunner: a model
                # hook can refine the window specified by its HF config.
                try:
                    from sglang.srt.model_executor.model_runner_components.load_model_utils import (
                        resolve_sliding_window_size,
                    )
                except ImportError:
                    resolver = getattr(model, "get_attention_sliding_window_size", None)
                    if callable(resolver):
                        model_config.sliding_window_size = resolver()
                else:
                    model_config.sliding_window_size = resolve_sliding_window_size(
                        model, model_config
                    )
                facts["resolved_sliding_window_size"] = model_config.sliding_window_size
            result.validation["meta_model"] = "passed"

        unknown = [name for name in ("has_linear_attention", "is_dsa", "is_deepseek_v4", "is_minimax_sparse")
                   if facts.get(name) is None]
        if unknown:
            result.model_info = facts
            result.fail(
                "CACHE_REQUIREMENTS_UNDETERMINED",
                "model_inspection",
                f"SGLang could not determine cache requirements: {', '.join(unknown)}",
            )
            return result
        requirements = requirements_from_facts(facts, host_layout=config.layout)
        from .native_probe import native_family

        family = native_family(requirements)
        if config.kv_cache_dtype != "auto" and not family:
            raise CheckFailure("CHECKER_RUNTIME_POOL_PROBE_REQUIRED", "cache_dtype",
                               "explicit KV dtype needs a native cache provider for this family")
        capabilities = infer_capabilities(UnifiedCacheStore, HiCacheStorage)
        result.model_info = facts
        result.cache_requirements = requirements.to_dict()
        result.ucm_capabilities = capabilities

        missing = find_missing_capabilities(requirements, capabilities)
        if missing:
            result.fail(
                "UCM_STORAGE_API_UNSUPPORTED",
                "capability_check",
                "; ".join(missing),
            )
            return result

        if model is not None:
            del model

        if config.mode == "roundtrip":
            cache_context = dict(
                page_size=config.page_size, hicache_mem_layout=config.layout,
                hicache_ratio=1.0, hicache_size=0,
                kv_cache_dtype=config.kv_cache_dtype,
            )
            if family == "deepseek_v4":
                cache_context.update(
                    dsv4_attn_backend=config.dsv4_attn_backend,
                    enable_deepseek_v4_fp4_indexer=config.fp4_indexer,
                    c128_page_size=config.c128_page_size,
                )
            with server_args_guard(**cache_context), _single_rank_model_parallel():
                result.roundtrip = _roundtrip(config, model_config, requirements)
            result.cache_requirements["pool_names"] = result.roundtrip["pool_names"]
            result.validation["storage_io"] = "passed"
            result.validation["runtime_pools"] = (
                "sglang_native_host_pools" if family else
                "config_derived_specialized_host_pools" if requirements.is_hybrid
                else "single_pool_host_only"
            )
            result.succeed("roundtrip", "UCM dump/load data matched")
        else:
            qualifier = "config-only" if config.skip_meta_model else "meta-model"
            if requirements.is_hybrid:
                if family:
                    result.validation["pool_description"] = "native_provider_available"
                    result.cache_requirements["native_provider"] = family
                else:
                    with server_args_guard(), _single_rank_model_parallel():
                        specs = _hybrid_specs(model_config, requirements, config)
                    result.cache_requirements["pool_specs"] = {
                        spec.name: spec.report() for spec in specs
                    }
                    result.validation["pool_description"] = "config_derived"
                reason = (
                    f"{qualifier} structural checks passed; specialized host pool probing is available; "
                    "pool registration and storage IO require --mode roundtrip"
                )
            else:
                reason = (
                    f"{qualifier} structural checks passed; storage IO was not probed"
                )
            if requirements.is_hybrid:
                result.fail("CHECKER_STORAGE_PROBE_REQUIRED", "inspect", reason)
            elif config.skip_meta_model:
                result.fail("CHECKER_CONFIG_ONLY", "inspect", reason)
            else:
                result.succeed("inspect", reason)
        return result
    except CheckFailure as exc:
        result.fail(exc.code, exc.stage, exc.reason)
        return result
    except Exception as exc:
        result.fail(
            "ENVIRONMENT_UNSUPPORTED",
            "startup",
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )
        return result


def main() -> int:
    config = CheckConfig.from_env()
    result = run(config)
    emit_result(result, config.output, verbose=config.verbose)
    return 0 if result.supported else 1


if __name__ == "__main__":
    raise SystemExit(main())
