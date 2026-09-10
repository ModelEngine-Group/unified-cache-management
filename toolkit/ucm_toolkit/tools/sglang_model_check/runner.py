"""Isolated implementation of the SGLang/UCM compatibility check."""

from __future__ import annotations

import contextlib
import socket
import traceback
import uuid
from types import SimpleNamespace
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
        quant_config = _get_quantization_config(model_config, load_config)
        with server_args_guard(), _single_rank_model_parallel():
            with torch.device("meta"), _meta_device_guard(torch):
                model = _initialize_model(model_config, load_config, quant_config)
            resolve_layer_indices(
                model=model,
                model_config=model_config,
                is_draft_worker=False,
                spec_algorithm=SpeculativeAlgorithm.NONE,
            )
            return model
    except Exception as exc:
        raise CheckFailure(
            "META_MODEL_INIT_UNSUPPORTED",
            "meta_model_init",
            f"{type(exc).__name__}: {exc}",
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


def _make_host_pool(model_config: Any, requirements: Any, config: CheckConfig) -> Any:
    import torch

    layer_num = int(
        max(
            getattr(model_config, "num_hidden_layers", 0),
            getattr(model_config, "num_attention_layers", 0),
        )
    )
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
        pool_cls = MHATokenToKVPoolHost

    try:
        return pool_cls(
            device_pool=device_pool,
            host_to_device_ratio=1.0,
            host_size=0,
            page_size=config.page_size,
            layout=config.layout,
            pin_memory=False,
            device="cpu",
            allocator_type="default",
        )
    except Exception as exc:
        raise CheckFailure(
            "HICACHE_HOST_POOL_UNSUPPORTED",
            "host_pool",
            f"{type(exc).__name__}: {exc}",
        ) from exc


def _page_view(host_pool: Any, start: int, stop: int, is_mla: bool) -> Any:
    layout = host_pool.layout
    if layout == "page_first":
        return (
            host_pool.kv_buffer[start:stop]
            if is_mla
            else host_pool.kv_buffer[:, start:stop]
        )
    if layout == "layer_first":
        return (
            host_pool.kv_buffer[:, start:stop]
            if is_mla
            else host_pool.kv_buffer[:, :, start:stop]
        )
    if layout == "page_first_direct":
        first_page = start // host_pool.page_size
        last_page = stop // host_pool.page_size
        return (
            host_pool.kv_buffer[first_page:last_page]
            if is_mla
            else host_pool.kv_buffer[:, first_page:last_page]
        )
    raise CheckFailure(
        "CHECKER_LAYOUT_UNSUPPORTED",
        "roundtrip_prepare",
        f"checker has no data comparison strategy for layout {layout!r}",
    )


def _fill_pattern(tensor: Any) -> None:
    import torch

    generator = torch.Generator(device=tensor.device)
    generator.manual_seed(20260909)
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

    tokens = config.pages * config.page_size
    source = _page_view(host_pool, 0, tokens, is_mla)
    target = _page_view(host_pool, tokens, tokens * 2, is_mla)
    _fill_pattern(source)
    expected = source.clone()
    target.zero_()
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
    loaded = storage.batch_get_v1(keys, target_indices)
    if len(loaded) != config.pages or not all(loaded):
        raise CheckFailure("LOAD_FAILED", "roundtrip_load", repr(loaded))
    if not torch.equal(expected, target):
        mismatch = int(torch.count_nonzero(expected != target).item())
        raise CheckFailure(
            "ROUNDTRIP_MISMATCH",
            "roundtrip_compare",
            f"kv: {mismatch} tensor elements differ",
        )
    return {
        "pages_written": config.pages,
        "pages_loaded": config.pages,
        "bytes_compared": expected.numel() * expected.element_size(),
    }


def _roundtrip_v2(
    storage: Any,
    pools: dict[str, Any],
    keys: list[str],
    config: Any,
    is_mla: bool,
) -> dict[str, Any]:
    """Exercise the same anchor-v1 + side-pool-v2 contract used by SGLang."""
    import torch
    from sglang.srt.mem_cache.hicache_storage import PoolTransfer

    names = list(pools)
    anchor_name = names[0]
    anchor = pools[anchor_name]
    totals = _roundtrip_v1(storage, anchor, keys, config, is_mla)
    side_names = names[1:]
    if not side_names:
        return totals

    tokens = config.pages * config.page_size
    source_transfers = []
    target_transfers = []
    expected_by_name = {}
    enum_names = []
    for name in side_names:
        pool = pools[name]
        source = _page_view(pool, 0, tokens, is_mla)
        target = _page_view(pool, tokens, tokens * 2, is_mla)
        _fill_pattern(source)
        expected_by_name[name] = source.clone()
        target.zero_()
        enum_name = _pool_name(name)
        enum_names.append(enum_name)
        source_transfers.append(
            PoolTransfer(
                name=enum_name,
                host_indices=torch.arange(0, tokens, dtype=torch.int64),
                keys=list(keys),
            )
        )
        target_transfers.append(
            PoolTransfer(
                name=enum_name,
                host_indices=torch.arange(tokens, tokens * 2, dtype=torch.int64),
                keys=list(keys),
            )
        )

    written = storage.batch_set_v2(source_transfers)
    _require_page_results(
        written, enum_names, config.pages, "DUMP_FAILED", "roundtrip_dump_v2"
    )
    hit = storage.batch_exists_v2(keys, source_transfers)
    if int(getattr(hit, "kv_hit_pages", -1)) != config.pages:
        raise CheckFailure(
            "LOAD_MISS",
            "roundtrip_exists_v2",
            f"expected {config.pages} restorable pages, got {hit!r}",
        )
    extra_hits = getattr(hit, "extra_pool_hit_pages", {})
    for enum_name in enum_names:
        count = _result_for_pool(extra_hits, enum_name)
        if count is None or int(count) < config.pages:
            raise CheckFailure(
                "LOAD_MISS",
                "roundtrip_exists_v2",
                f"pool {enum_name}: expected {config.pages} hit pages, got {count!r}",
            )

    loaded = storage.batch_get_v2(target_transfers)
    _require_page_results(
        loaded, enum_names, config.pages, "LOAD_FAILED", "roundtrip_load_v2"
    )
    for name in side_names:
        target = _page_view(pools[name], tokens, tokens * 2, is_mla)
        expected = expected_by_name[name]
        if not torch.equal(expected, target):
            mismatch = int(torch.count_nonzero(expected != target).item())
            raise CheckFailure(
                "ROUNDTRIP_MISMATCH",
                "roundtrip_compare_v2",
                f"{name}: {mismatch} tensor elements differ",
            )
        totals["bytes_compared"] += expected.numel() * expected.element_size()
    totals["pool_names"] = names
    totals["storage_api"] = "v2"
    return totals


def _roundtrip(
    config: CheckConfig, model_config: Any, requirements: Any
) -> dict[str, Any]:
    if requirements.is_hybrid:
        raise CheckFailure(
            "CHECKER_RUNTIME_POOL_PROBE_REQUIRED",
            "roundtrip_prepare",
            "the model uses specialized SGLang side pools; synthetic MHA/MLA "
            "pools cannot prove v2 compatibility. Inspect validates the model "
            "and UCM API surface only",
        )

    from sglang.srt.mem_cache.hicache_storage import HiCacheStorage
    from ucm.integration.sglang.unifiedcache_store import UnifiedCacheStore

    pools = {
        name: _make_host_pool(model_config, requirements, config)
        for name in requirements.pool_names
    }
    anchor_name = next(iter(pools))
    host_pool = pools[anchor_name]
    storage = None
    try:
        capabilities = infer_capabilities(UnifiedCacheStore, HiCacheStorage)
        missing = find_missing_capabilities(requirements, capabilities)
        if missing:
            raise CheckFailure(
                "UCM_STORAGE_API_UNSUPPORTED",
                "capability_check",
                "; ".join(missing),
            )

        storage = UnifiedCacheStore(_make_storage_config(config, requirements))
        storage.register_mem_pool_host(host_pool)
        salt = uuid.uuid4().hex
        keys = [
            f"sglang-model-check-{salt}-{index}" for index in range(config.pages)
        ]
        if requirements.storage_api == "v2":
            for name, pool in pools.items():
                storage.register_mem_host_pool_v2(pool, _pool_name(name))
            totals = _roundtrip_v2(
                storage, pools, keys, config, requirements.is_mla
            )
        else:
            totals = _roundtrip_v1(
                storage, host_pool, keys, config, requirements.is_mla
            )
            totals.update(pool_names=[anchor_name], storage_api="v1")
        totals["host_pool_classes"] = {
            name: type(pool).__name__ for name, pool in pools.items()
        }
        return totals
    finally:
        if storage is not None:
            storage.close()
        for pool in pools.values():
            destroy = getattr(pool, "destroy", None)
            if callable(destroy):
                destroy()


def run(config: CheckConfig) -> CheckResult:
    result = CheckResult(model=config.model, mode=config.mode)
    try:
        from .sglang_compat import environment_info, refine_facts_from_model
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

        supported_platforms = facts.get("supported_platforms")
        detected_platform = result.environment.get("platform")
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

        linear_model = facts.get("has_linear_attention")
        if linear_model is None:
            result.model_info = facts
            result.fail(
                "CACHE_REQUIREMENTS_UNDETERMINED",
                "model_inspection",
                "SGLang compatibility layer could not determine whether the "
                "model uses linear-attention state; refusing to assume a v1 KV-only model",
            )
            return result
        requirements = requirements_from_facts(facts, host_layout=config.layout)
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
            with _single_rank_model_parallel():
                result.roundtrip = _roundtrip(config, model_config, requirements)
            result.succeed("roundtrip", "UCM dump/load data matched")
        else:
            qualifier = "config-only" if config.skip_meta_model else "meta-model"
            if requirements.is_hybrid:
                qualifier += "; specialized runtime pools were not constructed"
            result.succeed(
                "inspect",
                f"{qualifier} structural checks passed; storage IO was not probed",
            )
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
    emit_result(result, config.output)
    return 0 if result.supported else 1


if __name__ == "__main__":
    raise SystemExit(main())
