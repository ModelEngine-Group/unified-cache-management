"""Device-independent UCM metadata helpers for compatibility checks."""

from __future__ import annotations

import copy
import hashlib
import inspect
import math
import time
import traceback
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

import torch


class UnsupportedEnvironment(RuntimeError):
    """Describe a vLLM/UCM version or runtime that this checker cannot support."""

    pass


NATIVE_NULL_BLOCK_ID = 0


# Shared version-tolerant helpers.
def auto_max_model_len(tokens: int, block_size: int) -> int:
    """Return a model length that contains the requested prompt plus one block."""

    return ((tokens + block_size - 1) // block_size + 1) * block_size


def log(message: str) -> None:
    """Print one checker message with a stable prefix."""

    print(f"[ucm-kv-check] {message}", flush=True)


def call_with_supported_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> Any:
    """Call a version-varying vLLM API with only accepted keyword arguments."""

    signature = inspect.signature(callable_obj)
    if any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
    ):
        return callable_obj(**kwargs)
    accepted = {
        name: value for name, value in kwargs.items() if name in signature.parameters
    }
    return callable_obj(**accepted)


def current_vllm_config_context(vllm_config: Any) -> Any:
    """Return vLLM's current-config context when the installed version has it."""

    if vllm_config is None:
        return nullcontext()
    try:
        from vllm.config import set_current_vllm_config
    except Exception:
        return nullcontext()
    return call_with_supported_kwargs(
        set_current_vllm_config,
        {"vllm_config": vllm_config, "check_compile": False},
    )


@dataclass
class CacheFixture:
    """Own the real vLLM KV-cache layout and tensors used by the checker."""

    vllm_config: Any
    kv_cache_config: Any
    kv_caches: dict[str, Any]
    layer_to_group: dict[str, int]


# Engine and UCM configuration.
def make_ucm_config(
    store_pipeline: str,
    storage_backends: str,
    use_layerwise: bool,
    enable_event_sync: bool = True,
) -> dict[str, Any]:
    """Build the fixed UCM configuration used by the compatibility check."""

    return {
        "ucm_connectors": [
            {
                "ucm_connector_name": "UcmPipelineStore",
                "ucm_connector_config": {
                    "store_pipeline": store_pipeline,
                    "storage_backends": storage_backends,
                    "io_direct": False,
                    "store_health": {"enabled": True},
                    "posix_io_engine": "psync",
                    "posix_data_trans_concurrency": 128,
                    "cache_buffer_capacity_gb": 32,
                },
            }
        ],
        "enable_event_sync": enable_event_sync,
        "enable_metrics": False,
        "wa_dump_block_wise": True,
        "use_layerwise": use_layerwise,
        "use_inference_duration_monitor": False,
        "persist_token_threshold": 0,
        "load_tokens_threshold": 0,
        "enable_record_traces": False,
    }


def make_config(
    model: str,
    tokens: int,
    block_size: int,
    dtype: str,
    kv_cache_dtype: str,
    trust_remote_code: bool,
    additional_config: dict[str, Any],
    store_pipeline: str,
    storage_backends: str,
    use_layerwise: bool,
    device: str,
) -> Any:
    """Create a one-rank vLLM configuration for the synthetic request."""

    from vllm.config import KVTransferConfig
    from vllm.engine.arg_utils import EngineArgs

    kv_transfer_config = call_with_supported_kwargs(
        KVTransferConfig,
        {
            "kv_connector": "UCMConnector",
            "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": make_ucm_config(
                store_pipeline,
                storage_backends,
                use_layerwise,
            ),
        },
    )
    max_model_len = auto_max_model_len(tokens, block_size)
    engine_args = call_with_supported_kwargs(
        EngineArgs,
        {
            "model": model,
            "tokenizer": None,
            "skip_tokenizer_init": True,
            "trust_remote_code": trust_remote_code,
            "dtype": dtype,
            "kv_cache_dtype": kv_cache_dtype,
            "block_size": block_size,
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "max_model_len": max_model_len,
            "max_num_batched_tokens": max_model_len,
            "max_num_seqs": 2,
            "enable_prefix_caching": True,
            "enforce_eager": True,
            "disable_hybrid_kv_cache_manager": False,
            "kv_transfer_config": kv_transfer_config,
            "device": device,
            "additional_config": copy.deepcopy(additional_config),
        },
    )
    vllm_config = engine_args.create_engine_config()
    model_config = vllm_config.model_config
    if bool(
        getattr(model_config, "is_multimodal_model", False)
        or getattr(model_config, "multimodal_config", None) is not None
    ):
        model_config.skip_tokenizer_init = False
    return vllm_config


# No-weight model initialization.
def _module_has_meta_tensor(module: torch.nn.Module) -> bool:
    """Return whether a module still owns at least one meta tensor."""

    return any(
        tensor.is_meta
        for tensor in list(module.parameters(recurse=True))
        + list(module.buffers(recurse=True))
    )


def _to_target_device(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> torch.device | None:
    """Extract the device argument accepted by ``Tensor.to`` or ``Module.to``."""

    device_arg = kwargs.get("device")
    if device_arg is None and args:
        first_arg = args[0]
        if isinstance(first_arg, torch.Tensor):
            device_arg = first_arg.device
        elif isinstance(first_arg, (torch.device, str, int)):
            device_arg = first_arg
    if device_arg is None:
        return None
    try:
        return torch.device(device_arg)
    except Exception:
        return None


def _redirect_to_meta_device(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Rewrite a tensor/device move so construction remains on the meta device."""

    rewritten_args = list(args)
    rewritten_kwargs = dict(kwargs)
    if rewritten_args:
        first_arg = rewritten_args[0]
        if isinstance(first_arg, torch.Tensor):
            if first_arg.device.type != "meta":
                rewritten_args[0] = torch.empty(
                    (),
                    device="meta",
                    dtype=first_arg.dtype,
                    layout=first_arg.layout,
                )
        elif isinstance(first_arg, (torch.device, str, int)):
            target_device = _to_target_device((first_arg,), {})
            if target_device is not None and target_device.type != "meta":
                rewritten_args[0] = torch.device("meta")
    target_device = _to_target_device((), rewritten_kwargs)
    if target_device is not None and target_device.type != "meta":
        rewritten_kwargs["device"] = torch.device("meta")
    return tuple(rewritten_args), rewritten_kwargs


@contextmanager
def no_real_device_move_from_meta_context(
    factory_kwargs_redirect: Callable[[dict[str, Any]], dict[str, Any]],
) -> Any:
    """Keep model-construction device moves and factories on meta tensors."""

    original_module_to = torch.nn.Module.to
    original_tensor_to = torch.Tensor.to
    factory_names = (
        "arange",
        "empty",
        "empty_like",
        "empty_strided",
        "eye",
        "full",
        "full_like",
        "linspace",
        "ones",
        "ones_like",
        "rand",
        "rand_like",
        "randn",
        "randn_like",
        "tensor",
        "zeros",
        "zeros_like",
    )
    original_factories = {
        name: getattr(torch, name) for name in factory_names if hasattr(torch, name)
    }

    def checker_meta_safe_to(
        module: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module:
        """Skip moving a meta-backed module to a real accelerator."""

        target_device = _to_target_device(args, kwargs)
        if (
            target_device is not None
            and target_device.type != "meta"
            and _module_has_meta_tensor(module)
        ):
            return module
        return original_module_to(module, *args, **kwargs)

    def checker_meta_safe_tensor_to(
        tensor: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        """Keep unsafe tensor moves on meta while model construction runs."""

        target_device = _to_target_device(args, kwargs)
        if target_device is not None and (
            (tensor.is_meta and target_device.type != "meta")
            or (not tensor.is_meta and target_device.type not in ("cpu", "meta"))
        ):
            args, kwargs = _redirect_to_meta_device(args, kwargs)
        return original_tensor_to(tensor, *args, **kwargs)

    def make_meta_safe_factory(factory: Any) -> Any:
        """Wrap one torch factory to redirect its explicit device argument."""

        def checker_meta_safe_factory(*args: Any, **kwargs: Any) -> torch.Tensor:
            """Call the factory after applying the platform redirect callback."""

            return factory(*args, **factory_kwargs_redirect(kwargs))

        return checker_meta_safe_factory

    torch.nn.Module.to = checker_meta_safe_to  # type: ignore[assignment]
    torch.Tensor.to = checker_meta_safe_tensor_to  # type: ignore[assignment]
    for name, factory in original_factories.items():
        setattr(torch, name, make_meta_safe_factory(factory))
    try:
        yield
    finally:
        for name, factory in original_factories.items():
            setattr(torch, name, factory)
        torch.Tensor.to = original_tensor_to  # type: ignore[assignment]
        torch.nn.Module.to = original_module_to  # type: ignore[assignment]


# Native KV-cache layout and allocation.
def get_kv_specs(runner: Any, vllm_config: Any) -> dict[str, Any]:
    """Get KV specs through the production ModelRunner API."""

    try:
        with current_vllm_config_context(vllm_config):
            return runner.get_kv_cache_spec()
    except Exception as exc:
        raise UnsupportedEnvironment(
            "Production ModelRunner.get_kv_cache_spec() failed. This means the "
            "checker could not obtain the model's real KVCacheSpec without "
            f"loading weights. Original error: {type(exc).__name__}: {exc}"
        ) from exc


def resolve_layout(vllm_config: Any, kv_cache_specs: dict[str, Any]) -> None:
    """Resolve the model-wide KV layout when required by this vLLM version."""

    cache_config = vllm_config.cache_config
    if not callable(getattr(cache_config, "get_resolved_kv_cache_layout", None)):
        # Older vLLM versions resolve tensor shapes inside their worker path and
        # have no model-wide layout-resolution phase.
        return
    if getattr(cache_config, "kv_cache_layout", None) is not None:
        return

    try:
        from vllm.distributed.kv_transfer.kv_connector.utils import (
            get_current_attn_backends,
        )
        from vllm.v1.attention.backends.utils import (
            get_supported_kv_cache_layouts,
            resolve_kv_cache_layout,
        )
    except (ImportError, AttributeError) as exc:
        raise UnsupportedEnvironment(
            "This vLLM requires a resolved KV-cache layout, but does not expose "
            "the EngineCore layout-resolution helpers needed by the standalone "
            "checker."
        ) from exc

    with current_vllm_config_context(vllm_config):
        backends = call_with_supported_kwargs(
            get_current_attn_backends,
            {"vllm_config": vllm_config, "layer_names": None},
        )
        layouts = get_supported_kv_cache_layouts(backends)
        supported_layouts = [
            [getattr(layout, "name", str(layout)) for layout in layouts]
        ]
        call_with_supported_kwargs(
            resolve_kv_cache_layout,
            {
                "vllm_config": vllm_config,
                "supported_layouts": supported_layouts,
                "kv_cache_specs": list(kv_cache_specs.values()),
            },
        )

    if getattr(cache_config, "kv_cache_layout", None) is None:
        raise UnsupportedEnvironment(
            "vLLM's KV-cache layout resolver returned without recording a layout."
        )


def make_layout(
    vllm_config: Any,
    kv_cache_specs: dict[str, Any],
    num_blocks: int,
) -> Any:
    """Use vLLM's own global grouping/config path; never reconstruct a group."""

    import vllm.v1.core.kv_cache_utils as kv_utils

    vllm_config.cache_config.num_gpu_blocks_override = num_blocks
    if hasattr(vllm_config.cache_config, "num_gpu_blocks"):
        vllm_config.cache_config.num_gpu_blocks = num_blocks
    if hasattr(kv_utils, "get_kv_cache_configs"):
        with current_vllm_config_context(vllm_config):
            configs = kv_utils.get_kv_cache_configs(
                vllm_config, [kv_cache_specs], [1 << 50]
            )
        if len(configs) != 1:
            raise UnsupportedEnvironment(
                f"Expected one worker KV config, got {len(configs)}"
            )
        return configs[0]

    if not hasattr(kv_utils, "get_kv_cache_groups"):
        raise UnsupportedEnvironment(
            "Installed vLLM exposes neither get_kv_cache_configs nor "
            "get_kv_cache_groups; exact native grouping cannot be guaranteed."
        )
    with current_vllm_config_context(vllm_config):
        groups = kv_utils.get_kv_cache_groups(vllm_config, kv_cache_specs)
    build_config = kv_utils.get_kv_cache_config_from_groups
    kwargs: dict[str, Any] = {
        "vllm_config": vllm_config,
        "kv_cache_groups": groups,
        "available_memory": 1 << 50,
    }
    if "kv_cache_specs" in inspect.signature(build_config).parameters:
        kwargs["kv_cache_specs"] = kv_cache_specs
    with current_vllm_config_context(vllm_config):
        return build_config(**kwargs)


def plan_blocks(
    vllm_config: Any,
    kv_cache_specs: dict[str, Any],
    tokens: int,
    block_size: int,
) -> int:
    """Return three times the minimum pool accepted by this vLLM version."""

    import vllm.v1.core.kv_cache_utils as kv_utils

    if not hasattr(kv_utils, "get_kv_cache_configs"):
        return 3 * max(4, (tokens + block_size - 1) // block_size + 1)

    last_capacity_error: ValueError | None = None

    def fits(num_blocks: int) -> bool:
        """Return whether vLLM accepts this candidate KV pool size."""

        nonlocal last_capacity_error
        try:
            make_layout(vllm_config, kv_cache_specs, num_blocks)
        except ValueError as exc:
            message = str(exc).lower()
            capacity_error = "no available memory for the cache blocks" in message or (
                "kv cache is needed" in message
                and "available kv cache memory" in message
            )
            if not capacity_error:
                raise UnsupportedEnvironment(
                    "vLLM rejected the KV-cache layout for a reason unrelated "
                    f"to pool capacity at num_blocks={num_blocks}. Increasing "
                    "num_blocks cannot resolve this error. "
                    f"Original error: {type(exc).__name__}: {exc}"
                ) from exc
            last_capacity_error = exc
            return False
        return True

    low = 0
    high = max(1, (tokens + block_size - 1) // block_size + 1)
    max_blocks = high * 4096
    while not fits(high):
        if high >= max_blocks:
            raise UnsupportedEnvironment(
                "KV block planning could not satisfy vLLM's capacity check: "
                f"tokens={tokens}, block_size={block_size}, "
                f"tried_up_to={max_blocks} blocks. Continuing to double the "
                "pool was stopped to avoid numeric overflow. "
                f"Last vLLM error: {type(last_capacity_error).__name__}: "
                f"{last_capacity_error}"
            ) from last_capacity_error
        low, high = high, min(high * 2, max_blocks)
    while high - low > 1:
        middle = (low + high) // 2
        if fits(middle):
            high = middle
        else:
            low = middle
    return 3 * high


def collect_bound_kv_caches_from_runner(runner: Any) -> dict[str, Any]:
    """Fall back to KV tensors bound in the production runner context."""

    static_forward_context = getattr(
        getattr(runner, "compilation_config", None), "static_forward_context", {}
    )
    kv_caches: dict[str, Any] = {}
    for layer_name, layer in static_forward_context.items():
        if not hasattr(layer, "kv_cache"):
            continue
        kv_cache = layer.kv_cache
        if isinstance(kv_cache, dict) and 0 in kv_cache:
            kv_cache = kv_cache[0]
        elif (
            isinstance(kv_cache, list)
            and len(kv_cache) == 1
            and isinstance(kv_cache[0], (torch.Tensor, tuple, list))
        ):
            kv_cache = kv_cache[0]
        if isinstance(kv_cache, (torch.Tensor, tuple, list)):
            kv_caches[layer_name] = kv_cache
    return kv_caches


def init_kv(
    vllm_config: Any,
    runner: Any,
    kv_cache_config: Any,
    device: torch.device,
    synchronize: Callable[[torch.device], None],
) -> dict[str, Any]:
    """Allocate and reshape KV tensors through ModelRunner.initialize_kv_cache()."""

    captured: dict[str, Any] = {}
    original_initializer = getattr(runner, "initialize_kv_cache_tensors", None)

    def capture_initialize_kv_cache_tensors(*args: Any, **kwargs: Any) -> Any:
        """Record the tensors allocated by ModelRunner's native initializer."""

        kv_caches = original_initializer(*args, **kwargs)
        captured["kv_caches"] = kv_caches
        return kv_caches

    if original_initializer is not None:
        setattr(
            runner,
            "initialize_kv_cache_tensors",
            capture_initialize_kv_cache_tensors,
        )
    try:
        synchronize(device)
        with current_vllm_config_context(vllm_config):
            call_with_supported_kwargs(
                runner.initialize_kv_cache,
                {"kv_cache_config": kv_cache_config},
            )
    except Exception as exc:
        raise UnsupportedEnvironment(
            "Production ModelRunner.initialize_kv_cache() failed. This is the "
            "same layer that applies vLLM KV group post-processing, backend "
            "selection, shared-indexer handling, Mamba/SWA handling and tensor "
            "reshape. The checker will not fall back to function-level helpers. "
            f"Original error: {type(exc).__name__}: {exc}\n"
            f"Original traceback:\n{traceback.format_exc()}"
        ) from exc
    finally:
        if original_initializer is not None:
            setattr(runner, "initialize_kv_cache_tensors", original_initializer)

    kv_caches = captured.get("kv_caches")
    if kv_caches is None:
        kv_caches = collect_bound_kv_caches_from_runner(runner)
    if not kv_caches:
        raise UnsupportedEnvironment(
            "ModelRunner.initialize_kv_cache() completed but no KV-cache tensor "
            "dict was captured from initialize_kv_cache_tensors or bound into "
            "static_forward_context."
        )
    return kv_caches


def init_dist(vllm_config: Any, backend: str) -> None:
    """Create vLLM's mandatory one-rank model-parallel group once."""

    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )

    if not torch.distributed.is_initialized():
        with current_vllm_config_context(vllm_config):
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method="tcp://127.0.0.1:29500",
                local_rank=0,
                backend=backend,
            )
    if not model_parallel_is_initialized():
        with current_vllm_config_context(vllm_config):
            initialize_model_parallel(1, 1, 1, 1, backend=backend)


def make_cache(
    vllm_config: Any,
    active_device: torch.device,
    tokens: int,
    block_size: int,
    backend: str,
    synchronize: Callable[[torch.device], None],
    make_model: Callable[[Any], tuple[Any, Any]],
    make_runner: Callable[[Any, Any, torch.device], Any],
) -> CacheFixture:
    """Build the native KV layout, tensors, and layer-to-group mapping."""

    init_dist(vllm_config, backend)
    model, kv_vllm_config = make_model(vllm_config)
    runner = make_runner(kv_vllm_config, model, active_device)

    # The production executor resolves the backend-dependent block size after
    # model construction and before asking the runner for its KV-cache specs.
    # Keep the call optional for older vLLM versions that predate this hook.
    from vllm.platforms import current_platform

    update_block_size = getattr(current_platform, "update_block_size_for_backend", None)
    if callable(update_block_size):
        update_block_size(kv_vllm_config)

    kv_cache_specs = get_kv_specs(runner, kv_vllm_config)
    if not kv_cache_specs:
        raise UnsupportedEnvironment("The real model returned an empty KVCacheSpec")

    resolve_layout(kv_vllm_config, kv_cache_specs)

    kv_num_blocks = plan_blocks(
        kv_vllm_config,
        kv_cache_specs,
        tokens,
        block_size,
    )
    kv_cache_config = make_layout(
        kv_vllm_config,
        kv_cache_specs,
        kv_num_blocks,
    )
    if int(kv_cache_config.num_blocks) != kv_num_blocks:
        raise AssertionError(
            f"vLLM ignored num_gpu_blocks_override: "
            f"expected={kv_num_blocks}, actual={kv_cache_config.num_blocks}"
        )

    kv_caches = init_kv(
        kv_vllm_config,
        runner,
        kv_cache_config,
        active_device,
        synchronize,
    )
    kv_cache_config = getattr(runner, "kv_cache_config", kv_cache_config)
    if int(kv_cache_config.num_blocks) != kv_num_blocks:
        raise AssertionError(
            f"vLLM ignored num_gpu_blocks_override after ModelRunner init: "
            f"expected={kv_num_blocks}, actual={kv_cache_config.num_blocks}"
        )
    layer_to_group = {
        layer_name: group_id
        for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
        for layer_name in group.layer_names
    }
    missing = set(kv_caches) - set(layer_to_group)
    if missing:
        raise AssertionError(
            f"KV tensors are missing native group assignments: {sorted(missing)}"
        )
    return CacheFixture(
        vllm_config=kv_vllm_config,
        kv_cache_config=kv_cache_config,
        kv_caches=kv_caches,
        layer_to_group=layer_to_group,
    )


def make_worker(fixture: CacheFixture) -> Any:
    """Create the worker-side UCM connector and bind the real KV tensors."""

    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

    from ucm.integration.vllm.ucm_connector import UCMConnector

    connector = UCMConnector(
        fixture.vllm_config,
        KVConnectorRole.WORKER,
        fixture.kv_cache_config,
    )
    connector.register_kv_caches(fixture.kv_caches)
    return connector


# Scheduler request construction and dispatch.
@dataclass
class RequestDispatchFixture:
    """Keep source scheduling state required for the following load request."""

    request_id: str
    prompt_token_ids: list[int]
    scheduler_block_size: int
    hash_block_size: int
    block_ids: tuple[list[int], ...]
    request: Any
    scheduler: Any
    scheduler_output: Any


@dataclass
class SavedBlock:
    """Hold one deterministic source KV block copy for later comparison."""

    layer: str
    part: int
    group: int
    block_id: int
    data: torch.Tensor


@dataclass(frozen=True)
class FAWASegment:
    """Describe the token range UCM transfers within one FAWA physical block."""

    block_id: int
    token_offset: int
    token_count: int


@dataclass
class ScheduledRequestFixture:
    """Keep vLLM scheduling results for one synthetic request."""

    scheduler: Any
    scheduler_output: Any
    request: Any
    block_ids: tuple[list[int], ...]
    scheduler_block_size: int
    hash_block_size: int


def get_scheduler_and_hash_block_size(
    vllm_config: Any, kv_cache_config: Any
) -> tuple[int, int]:
    """Resolve the installed vLLM scheduler and prefix-hash block sizes."""

    try:
        from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes

        with current_vllm_config_context(vllm_config):
            return tuple(
                int(v)
                for v in call_with_supported_kwargs(
                    resolve_kv_cache_block_sizes,
                    {
                        "vllm_config": vllm_config,
                        "kv_cache_config": kv_cache_config,
                    },
                )
            )
    except Exception as exc:
        raise UnsupportedEnvironment(
            "Installed vLLM does not expose resolve_kv_cache_block_sizes(), so "
            "the checker cannot derive scheduler/hash block sizes from the "
            f"active implementation. Original error: {type(exc).__name__}: {exc}"
        ) from exc


def make_prompt_tokens(num_tokens: int, salt: int) -> list[int]:
    """Create deterministic, non-special token ids without a tokenizer."""

    return [
        1000 + (idx * 31 + ((salt >> ((idx % 8) * 8)) & 0xFF)) % 997
        for idx in range(num_tokens)
    ]


def source_prompt_tokens(
    prompt_token_ids: list[int], kv_cache_config: Any, hash_block_size: int
) -> list[int]:
    """Keep the source prefix at a complete multi-group KV boundary."""

    groups = kv_cache_config.kv_cache_groups
    if len(groups) <= 1:
        return prompt_token_ids

    block_sizes = []
    for group in groups:
        spec = group.kv_cache_spec
        specs = getattr(spec, "kv_cache_specs", None)
        if specs:
            spec = next(iter(specs.values()))
        block_sizes.append(int(spec.block_size))
    lcm_block_size = math.lcm(*block_sizes)
    source_tokens = len(prompt_token_ids) - hash_block_size
    source_tokens = source_tokens // lcm_block_size * lcm_block_size
    if source_tokens <= 0:
        raise UnsupportedEnvironment(
            "tokens are too short to create an aligned multi-group source prefix"
        )
    return prompt_token_ids[:source_tokens]


def make_fake_sampling_params() -> Any:
    """Provide the minimal sampling-parameter surface required by ``Request``."""

    class FakeSamplingParams:
        def __init__(self) -> None:
            """Set only the fields accessed by the installed Request class."""

            self.max_tokens = 1
            self.extra_args = None
            self.skip_reading_prefix_cache = False
            self.structured_outputs = None

    return FakeSamplingParams()


def make_vllm_request(
    request_id: str,
    prompt_token_ids: list[int],
    hash_block_size: int,
) -> Any:
    """Create a vLLM request whose prefix hashes are deterministic."""

    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.request import Request

    try:
        from vllm.utils.hashing import sha256_cbor
    except Exception:
        sha256_cbor = lambda obj: hashlib.sha256(repr(obj).encode("utf-8")).digest()

    init_none_hash(sha256_cbor)
    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=make_fake_sampling_params(),
        pooling_params=None,
        block_hasher=get_request_block_hasher(hash_block_size, sha256_cbor),
    )


def make_scheduler_cache_config(worker_kv_cache_config: Any) -> Any:
    """Convert worker KV config to the scheduler/KVCacheManager view."""

    import vllm.v1.core.kv_cache_utils as kv_utils

    if hasattr(kv_utils, "get_scheduler_kv_cache_config"):
        return kv_utils.get_scheduler_kv_cache_config([worker_kv_cache_config])
    scheduler_config = copy.deepcopy(worker_kv_cache_config)
    try:
        from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
    except Exception:
        return scheduler_config
    for group in scheduler_config.kv_cache_groups:
        group_spec = getattr(group, "kv_cache_spec", None)
        if isinstance(group_spec, UniformTypeKVCacheSpecs):
            group.kv_cache_spec = next(iter(group_spec.kv_cache_specs.values()))
    return scheduler_config


def make_scheduler(
    fixture: Any,
    scheduler_block_size: int,
    hash_block_size: int,
    patch_groups: Callable[[Any], None] | None,
) -> Any:
    """Construct the installed vLLM Scheduler and its UCM connector."""

    if patch_groups is not None:
        patch_groups(fixture)
    scheduler_kv_cache_config = make_scheduler_cache_config(fixture.kv_cache_config)
    from vllm.v1.structured_output import StructuredOutputManager

    scheduler_cls = fixture.vllm_config.scheduler_config.get_scheduler_cls()
    with current_vllm_config_context(fixture.vllm_config):
        scheduler = call_with_supported_kwargs(
            scheduler_cls,
            {
                "vllm_config": fixture.vllm_config,
                "kv_cache_config": scheduler_kv_cache_config,
                "structured_output_manager": StructuredOutputManager(
                    fixture.vllm_config
                ),
                "block_size": scheduler_block_size,
                "hash_block_size": hash_block_size,
                "include_finished_set": False,
                "log_stats": False,
            },
        )
    if getattr(scheduler, "connector", None) is None:
        raise UnsupportedEnvironment(
            "vLLM Scheduler did not construct the UCM KV connector. Check "
            "KVTransferConfig / UCM configuration."
        )
    return scheduler


# UCM dump/load and tensor comparison.
def build_forward_context(kv_caches: dict[str, Any]) -> Any:
    """Build the minimal forward context needed by UCM worker hooks."""

    from vllm.forward_context import ForwardContext

    return call_with_supported_kwargs(
        ForwardContext,
        {
            "no_compile_layers": {
                name: SimpleNamespace(kv_cache={0: value})
                for name, value in kv_caches.items()
            },
            "attn_metadata": {},
            "slot_mapping": {},
        },
    )


def call_worker_layer_hooks(worker: Any, fixture: Any, save: bool) -> None:
    """Run UCM's per-layer load wait and optional save hook for every KV cache."""

    for layer_name, kv_cache in fixture.kv_caches.items():
        worker.wait_for_layer_load(layer_name)
        if save:
            worker.save_kv_layer(layer_name, kv_cache, None)


def tensor_components(value: Any) -> list[torch.Tensor]:
    """Normalize vLLM's tensor, tuple, and combined-KV cache containers."""

    if isinstance(value, torch.Tensor):
        if value.dim() == 5 and value.shape[0] == 2:
            return [value[0], value[1]]
        return [value]
    if isinstance(value, (tuple, list)) and all(
        isinstance(item, torch.Tensor) for item in value
    ):
        return list(value)
    raise UnsupportedEnvironment(
        f"Unsupported vLLM KV-cache container: {type(value)!r}"
    )


def block_view(
    tensor: torch.Tensor, block_id: int, num_blocks: int | None = None
) -> torch.Tensor:
    """Return one native block, or one KVCacheManager page when requested."""

    if tensor.ndim == 0:
        raise UnsupportedEnvironment("KV tensor has no block axis.")
    if num_blocks is None:
        if block_id < 0 or block_id >= tensor.shape[0]:
            raise IndexError(
                f"block id {block_id} is outside tensor block range "
                f"[0, {tensor.shape[0]})"
            )
        return tensor[block_id]
    if num_blocks <= 0 or tensor.shape[0] % num_blocks:
        raise UnsupportedEnvironment(
            "KV tensor block axis is incompatible with KVCacheManager blocks: "
            f"shape={tuple(tensor.shape)}, num_blocks={num_blocks}"
        )
    blocks_per_page = tensor.shape[0] // num_blocks
    if block_id < 0 or block_id >= num_blocks:
        raise IndexError(
            f"block id {block_id} is outside KVCacheManager block range "
            f"[0, {num_blocks})"
        )
    return tensor.narrow(0, block_id * blocks_per_page, blocks_per_page)


def finish_async_dumps(worker: Any, request_id: str) -> None:
    """Collect an asynchronous connector dump when that worker API exists."""

    if hasattr(worker, "get_finished"):
        worker.get_finished({request_id})


def metadata_key_count(metadata: Any, phase: str) -> int:
    """Count UCM keys in ordinary or FAWA dispatch metadata."""

    total = 0
    for request_meta in getattr(metadata, "request_meta", {}).values():
        native = f"{phase}_block_ids"
        if hasattr(request_meta, native):
            total += len(getattr(request_meta, native)[0])
        elif hasattr(request_meta, f"{phase}_keys"):
            total += len(getattr(request_meta, f"{phase}_keys"))
    return total


def _fawa_segments(
    metadata: Any, phase: str, connector: Any
) -> dict[Any, FAWASegment] | None:
    """Map each FAWA key to the logical segment UCM transfers for it."""

    request_metas = tuple(getattr(metadata, "request_meta", {}).values())
    if not request_metas or not any(
        hasattr(request_meta, f"{phase}_keys") for request_meta in request_metas
    ):
        return None

    group_metas = getattr(connector, "group_metas", {})
    fa_groups = set(getattr(connector, "fa_group_ids", ()))
    wa_groups = set(getattr(connector, "window_group_ids", ()))
    hash_block_size = int(getattr(connector, "hash_block_size"))
    block_wise = phase == "dump" and bool(
        getattr(connector, "wa_dump_block_wise", False)
    )
    segments: dict[Any, FAWASegment] = {}

    def add(key: Any, segment: FAWASegment) -> None:
        """Add one key-to-segment mapping while rejecting conflicting metadata."""

        previous = segments.setdefault(key, segment)
        if previous != segment:
            raise ValueError(
                f"FAWA metadata maps one key to different segments: {key!r}"
            )

    for request_meta in request_metas:
        keys = getattr(request_meta, f"{phase}_keys", ())
        grouped_ids = getattr(request_meta, f"{phase}_vllm_block_ids", ())
        if not keys:
            continue
        if len(grouped_ids) != len(group_metas):
            raise ValueError(
                "FAWA metadata has an unexpected number of KV groups: "
                f"groups={len(grouped_ids)}, expected={len(group_metas)}."
            )
        hash_start = int(getattr(request_meta, f"{phase}_hash_start"))
        for group, block_ids in enumerate(grouped_ids):
            meta = group_metas.get(group)
            if meta is None:
                raise ValueError(f"FAWA metadata refers to unknown KV group {group}.")
            if group in fa_groups:
                if len(block_ids) != len(keys):
                    raise ValueError(
                        "FAWA full-attention metadata has an unexpected number "
                        f"of block ids: group={group}, keys={len(keys)}, "
                        f"block_ids={len(block_ids)}."
                    )
                for index, (key, block_id) in enumerate(
                    zip(keys, block_ids, strict=True)
                ):
                    token_offset = (
                        (hash_start + index) * hash_block_size
                    ) % meta.token_block_size
                    if token_offset + hash_block_size > meta.token_block_size:
                        raise ValueError(
                            "FAWA hash segment crosses a physical KV block: "
                            f"group={group}, offset={token_offset}, "
                            f"tokens={hash_block_size}, "
                            f"block_tokens={meta.token_block_size}."
                        )
                    add(
                        (group, key, 0),
                        FAWASegment(block_id, token_offset, hash_block_size),
                    )
            elif group in wa_groups:
                # HMA does not store or load a window group with no tail.
                # Its dispatch metadata consequently carries an empty block-id
                # row even though tail_blocks is kept at one as a layout
                # placeholder.
                if not meta.tail_tokens:
                    continue
                tail_blocks = meta.tail_blocks
                segment_tokens = meta.tail_tokens // tail_blocks
                token_offset = (
                    meta.token_block_size - meta.tail_tokens
                    if tail_blocks == 1 and meta.token_block_size > meta.tail_tokens
                    else 0
                )
                if block_wise:
                    expected = len(keys) * tail_blocks
                    if len(block_ids) != expected:
                        raise ValueError(
                            "FAWA block-wise window metadata has an unexpected "
                            f"number of block ids: group={group}, keys={len(keys)}, "
                            f"tail_blocks={tail_blocks}, block_ids={len(block_ids)}."
                        )
                    for index, key in enumerate(keys):
                        start = index * tail_blocks
                        for tail, block_id in enumerate(
                            block_ids[start : start + tail_blocks]
                        ):
                            add(
                                (group, key, tail),
                                FAWASegment(block_id, token_offset, segment_tokens),
                            )
                else:
                    if len(block_ids) != tail_blocks:
                        raise ValueError(
                            "FAWA window metadata has an unexpected number of "
                            f"block ids: group={group}, tail_blocks={tail_blocks}, "
                            f"block_ids={len(block_ids)}."
                        )
                    for tail, block_id in enumerate(block_ids):
                        add(
                            (group, keys[-1], tail),
                            FAWASegment(block_id, token_offset, segment_tokens),
                        )
            else:
                raise ValueError(f"FAWA metadata cannot classify KV group {group}.")
    return segments


def _metadata_blocks(
    metadata: Any,
    phase: str,
) -> dict[Any, int]:
    """Map native UCM content keys to their physical vLLM block ids."""

    pairs: dict[Any, int] = {}
    native = f"{phase}_block_ids"
    for request_meta in getattr(metadata, "request_meta", {}).values():
        if not hasattr(request_meta, native):
            raise ValueError(
                f"Unsupported UCM request metadata: {type(request_meta).__name__} "
                f"has no {native}."
            )
        keys, values = getattr(request_meta, native)
        pairs.update(zip(keys, values, strict=True))
    return pairs


def loaded_block_pairs(
    source: Any,
    target: Any,
    save_metadata: Any,
    load_metadata: Any,
) -> dict[tuple[int, int], int]:
    """Return target blocks keyed by ``(KV group, source block)``."""

    connector = getattr(source.scheduler.connector, "connector", None)
    dumped = _metadata_blocks(save_metadata, "dump")
    loaded = _metadata_blocks(load_metadata, "load")
    matches = {
        key: (source_id, loaded[key])
        for key, source_id in dumped.items()
        if key in loaded
    }
    if not matches:
        raise AssertionError("UCM load metadata has no keys from the source dump")

    pairs: dict[tuple[int, int], int] = {}

    manager = getattr(connector, "group_manager", None)
    request_meta = getattr(connector, "requests_meta", {}).get(
        source.request.request_id
    )
    groups = getattr(manager, "groups_by_id", ())
    group_hashes = getattr(request_meta, "group_ucm_block_ids", None)
    if group_hashes is not None and groups:
        for group, hashes in zip(groups, group_hashes, strict=True):
            if group.is_full_attention:
                keys = hashes
            else:
                state = manager.compute_mamba_align_state_hash(
                    group, len(source.request.all_token_ids), group_hashes
                )
                keys = () if state is None else (state,)
            for key in keys:
                pair = matches.get(key)
                if pair is not None:
                    pairs[(group.group_id, pair[0])] = pair[1]
    else:
        for group, source_group in enumerate(source.block_ids):
            target_ids = set(target.block_ids[group])
            for source_id, target_id in matches.values():
                if source_id in source_group and target_id in target_ids:
                    pairs[(group, source_id)] = target_id
    return pairs


def _fawa_block_slice(
    data: torch.Tensor,
    segment: FAWASegment,
    group_tokens: int,
) -> torch.Tensor:
    """Select a FAWA logical segment from one physical KV block copy."""

    if data.ndim < 1:
        raise ValueError("FAWA KV block has no token dimension.")
    block_tokens = data.shape[0]
    start = segment.token_offset * block_tokens // group_tokens
    length = segment.token_count * block_tokens // group_tokens
    if length <= 0 or start + length > block_tokens:
        raise ValueError(
            "FAWA segment is outside the physical KV block: "
            f"offset={segment.token_offset}, tokens={segment.token_count}, "
            f"group_tokens={group_tokens}, block_tokens={block_tokens}."
        )
    return data.narrow(0, start, length)


def _fawa_group_caches(fixture: Any) -> dict[int, list[tuple[str, Any]]]:
    """Index registered KV caches by the FAWA group that owns each layer."""

    groups: dict[int, list[tuple[str, Any]]] = {}
    for layer, cache in fixture.kv_caches.items():
        groups.setdefault(fixture.layer_to_group[layer], []).append((layer, cache))
    return groups


def _compare_fawa_segment(
    group: int,
    group_tokens: int,
    group_caches: list[tuple[str, Any]],
    saved_blocks: dict[tuple[str, int, int], SavedBlock],
    source_segment: FAWASegment,
    target_segment: FAWASegment,
) -> int:
    """Compare one matched FAWA key across every tensor in its KV group."""

    if target_segment.block_id == NATIVE_NULL_BLOCK_ID:
        raise AssertionError(f"FAWA load selected a null block for KV group {group}")
    if source_segment.block_id == NATIVE_NULL_BLOCK_ID:
        return 0

    compared = 0
    for layer, cache in group_caches:
        for part, tensor in enumerate(tensor_components(cache)):
            source = saved_blocks.get((layer, part, source_segment.block_id))
            if source is None:
                raise AssertionError(
                    "FAWA dump selected a block outside the scheduled source: "
                    f"layer={layer}, block={source_segment.block_id}."
                )
            target = block_view(tensor, target_segment.block_id).detach().cpu()
            source_slice = _fawa_block_slice(source.data, source_segment, group_tokens)
            target_slice = _fawa_block_slice(target, target_segment, group_tokens)
            if not torch.equal(target_slice, source_slice):
                raise AssertionError(
                    f"dump/load mismatch: layer={layer}, part={part}, "
                    f"source_block={source_segment.block_id}, "
                    f"target_block={target_segment.block_id}, "
                    f"source_offset={source_segment.token_offset}, "
                    f"target_offset={target_segment.token_offset}, "
                    f"tokens={source_segment.token_count}"
                )
            compared += 1
    return compared


def _compare_fawa_segments(
    fixture: Any,
    saved: list[SavedBlock],
    source_segments: dict[Any, FAWASegment],
    target_segments: dict[Any, FAWASegment],
    connector: Any,
) -> int:
    """Compare exactly the FAWA slices selected by matched UCM keys."""

    matches = {
        key: (source_segment, target_segments[key])
        for key, source_segment in source_segments.items()
        if key in target_segments
    }
    if not matches:
        raise AssertionError("UCM load metadata has no keys from the source dump")

    saved_blocks = {(item.layer, item.part, item.block_id): item for item in saved}
    group_caches = _fawa_group_caches(fixture)
    compared = 0
    for (group, _key, _tail), (source_segment, target_segment) in matches.items():
        compared += _compare_fawa_segment(
            group,
            int(connector.group_metas[group].token_block_size),
            group_caches.get(group, []),
            saved_blocks,
            source_segment,
            target_segment,
        )
    return compared


def save_source(
    fixture: Any,
    source_ids: tuple[list[int], ...],
    synchronize: Callable[[], None],
    native_blocks: bool,
) -> list[SavedBlock]:
    """Fill source blocks and retain their CPU copies for comparison."""

    for layer_index, (layer, cache) in enumerate(fixture.kv_caches.items()):
        group = fixture.layer_to_group[layer]
        for part, tensor in enumerate(tensor_components(cache)):
            for source_id in source_ids[group]:
                if source_id == NATIVE_NULL_BLOCK_ID:
                    continue
                source = block_view(
                    tensor,
                    source_id,
                    None if native_blocks else fixture.kv_cache_config.num_blocks,
                )
                source.fill_(layer_index + part + source_id + 1)

    synchronize()
    saved: list[SavedBlock] = []
    for layer_index, (layer, cache) in enumerate(fixture.kv_caches.items()):
        group = fixture.layer_to_group[layer]
        for part, tensor in enumerate(tensor_components(cache)):
            for source_id in source_ids[group]:
                if source_id == NATIVE_NULL_BLOCK_ID:
                    continue
                source = block_view(
                    tensor,
                    source_id,
                    None if native_blocks else fixture.kv_cache_config.num_blocks,
                )
                saved.append(
                    SavedBlock(
                        layer,
                        part,
                        group,
                        source_id,
                        source.detach().cpu().clone(),
                    )
                )
    synchronize()
    return saved


def compare_loaded_blocks(
    fixture: Any,
    saved: list[SavedBlock],
    source: ScheduledRequestFixture,
    target: ScheduledRequestFixture,
    save_metadata: Any,
    load_metadata: Any,
    worker: Any,
) -> int:
    """Compare saved source blocks with their loaded target blocks."""

    connector = getattr(worker, "connector", None)
    source_segments = _fawa_segments(save_metadata, "dump", connector)
    target_segments = _fawa_segments(load_metadata, "load", connector)
    if source_segments is not None or target_segments is not None:
        if source_segments is None or target_segments is None:
            raise ValueError("FAWA dump and load metadata do not use the same format.")
        compared = _compare_fawa_segments(
            fixture, saved, source_segments, target_segments, connector
        )
        if not compared:
            raise AssertionError("no loaded FAWA KV segments were compared")
        return compared

    pairs = loaded_block_pairs(source, target, save_metadata, load_metadata)
    compared = 0
    for item in saved:
        target_id = pairs.get((item.group, item.block_id))
        if target_id is None:
            continue
        if target_id == NATIVE_NULL_BLOCK_ID:
            raise AssertionError(
                f"load selected a null block: layer={item.layer}, block={item.block_id}"
            )
        target_block = (
            block_view(
                tensor_components(fixture.kv_caches[item.layer])[item.part],
                target_id,
                fixture.kv_cache_config.num_blocks,
            )
            .detach()
            .cpu()
        )
        if not torch.equal(target_block, item.data):
            raise AssertionError(
                f"dump/load mismatch: layer={item.layer}, part={item.part}, "
                f"source_block={item.block_id}, target_block={target_id}"
            )
        compared += 1
    if not compared:
        raise AssertionError("no loaded KV blocks were compared")
    return compared


# Schedule the synthetic dump and load requests.
def schedule_source(
    fixture: Any,
    request_id: str,
    prompt_token_ids: list[int],
    patch_groups: Callable[[Any], None] | None,
) -> ScheduledRequestFixture:
    """Schedule the source request and obtain UCM dump metadata."""

    scheduler_block_size, hash_block_size = get_scheduler_and_hash_block_size(
        fixture.vllm_config, fixture.kv_cache_config
    )
    scheduler = make_scheduler(
        fixture, scheduler_block_size, hash_block_size, patch_groups
    )
    request = make_vllm_request(request_id, prompt_token_ids, hash_block_size)
    with current_vllm_config_context(fixture.vllm_config):
        scheduler.add_request(request)
        scheduler_output = scheduler.schedule()

    new_request = next(
        (
            item
            for item in getattr(scheduler_output, "scheduled_new_reqs", ())
            if item.req_id == request_id
        ),
        None,
    )
    if new_request is None:
        raise UnsupportedEnvironment(
            "Scheduler did not schedule the synthetic request. Reduce tokens "
            "or increase the configured batch-token limit."
        )
    if (
        scheduler.kv_cache_manager.get_blocks(request_id) is None
        or not new_request.block_ids
    ):
        raise UnsupportedEnvironment(
            "Scheduler did not allocate KVCacheManager block groups."
        )
    if getattr(scheduler_output, "kv_connector_metadata", None) is None:
        raise UnsupportedEnvironment(
            "SchedulerOutput has no kv_connector_metadata from UCM."
        )
    return ScheduledRequestFixture(
        scheduler=scheduler,
        scheduler_output=scheduler_output,
        request=request,
        block_ids=new_request.block_ids,
        scheduler_block_size=scheduler_block_size,
        hash_block_size=hash_block_size,
    )


def schedule_target(
    fixture: Any,
    scheduler: Any,
    request_id: str,
    prompt_token_ids: list[int],
    scheduler_block_size: int,
    hash_block_size: int,
) -> ScheduledRequestFixture:
    """Schedule the target request after dump to obtain UCM load metadata."""

    request = make_vllm_request(request_id, prompt_token_ids, hash_block_size)
    with current_vllm_config_context(fixture.vllm_config):
        scheduler.add_request(request)
        scheduler_output = scheduler.schedule()
    new_request = next(
        (
            item
            for item in getattr(scheduler_output, "scheduled_new_reqs", ())
            if item.req_id == request_id
        ),
        None,
    )
    if new_request is None or not new_request.block_ids:
        raise UnsupportedEnvironment(
            "Scheduler did not allocate target KV block groups."
        )
    if getattr(scheduler_output, "kv_connector_metadata", None) is None:
        raise UnsupportedEnvironment(
            "Target SchedulerOutput has no kv_connector_metadata from UCM."
        )
    if scheduler.kv_cache_manager.get_blocks(request_id) is None:
        raise UnsupportedEnvironment("Target request has no KVCacheManager blocks.")
    return ScheduledRequestFixture(
        scheduler=scheduler,
        scheduler_output=scheduler_output,
        request=request,
        block_ids=new_request.block_ids,
        scheduler_block_size=scheduler_block_size,
        hash_block_size=hash_block_size,
    )


def schedule(
    fixture: Any,
    tokens: int,
    request_token_salt: int,
    patch_groups: Callable[[Any], None] | None = None,
) -> RequestDispatchFixture:
    """Schedule the aligned source request and return its UCM dump metadata."""

    if tokens <= 0:
        raise ValueError("tokens must be positive")
    if tokens > int(fixture.vllm_config.model_config.max_model_len):
        raise ValueError(
            f"tokens={tokens} exceeds max_model_len="
            f"{fixture.vllm_config.model_config.max_model_len}"
        )

    prompt_token_ids = make_prompt_tokens(tokens, request_token_salt)
    _, hash_block_size = get_scheduler_and_hash_block_size(
        fixture.vllm_config, fixture.kv_cache_config
    )
    source_token_ids = source_prompt_tokens(
        prompt_token_ids, fixture.kv_cache_config, hash_block_size
    )
    source = schedule_source(
        fixture, "ucm-kv-shape-check-source", source_token_ids, patch_groups
    )
    return RequestDispatchFixture(
        request_id=source.request.request_id,
        prompt_token_ids=prompt_token_ids,
        scheduler_block_size=source.scheduler_block_size,
        hash_block_size=source.hash_block_size,
        block_ids=source.block_ids,
        request=source.request,
        scheduler=source.scheduler,
        scheduler_output=source.scheduler_output,
    )


# End-to-end UCM persistence validation.
def verify(
    fixture: Any,
    dispatch: RequestDispatchFixture,
    worker: Any,
    synchronize: Callable[[], None],
) -> None:
    """Use only UCM calls to validate Scheduler-produced dump/load metadata."""

    source_ids = dispatch.block_ids
    save_metadata = dispatch.scheduler_output.kv_connector_metadata
    dump_count = metadata_key_count(save_metadata, "dump")
    if dump_count == 0:
        raise UnsupportedEnvironment(
            "Scheduler/UCM metadata selected no dump blocks. Increase tokens or "
            "check UCM persist thresholds."
        )

    native_blocks = (
        _fawa_segments(save_metadata, "dump", getattr(worker, "connector", None))
        is not None
    )
    saved = save_source(fixture, source_ids, synchronize, native_blocks)
    worker.bind_connector_metadata(save_metadata)
    worker.start_load_kv(build_forward_context(fixture.kv_caches))
    call_worker_layer_hooks(worker, fixture, save=True)
    worker.wait_for_save()
    finish_async_dumps(worker, dispatch.request_id)
    worker.clear_connector_metadata()
    synchronize()
    log(f"dump completed: request_id={dispatch.request_id}, dump_keys={dump_count}")
    time.sleep(10)

    target = schedule_target(
        fixture,
        dispatch.scheduler,
        "ucm-kv-shape-check-target",
        dispatch.prompt_token_ids,
        dispatch.scheduler_block_size,
        dispatch.hash_block_size,
    )
    target_ids = target.block_ids
    for source_group, target_group in zip(source_ids, target_ids, strict=True):
        overlap = (set(source_group) & set(target_group)) - {NATIVE_NULL_BLOCK_ID}
        if overlap:
            raise UnsupportedEnvironment(
                f"Scheduler reused source physical block ids for target load: {sorted(overlap)}"
            )
    load_metadata = target.scheduler_output.kv_connector_metadata
    load_count = metadata_key_count(load_metadata, "load")
    if load_count == 0:
        raise UnsupportedEnvironment(
            "Scheduler-side UCM lookup did not hit: target "
            "kv_connector_metadata has no load_block_ids."
        )
    worker.bind_connector_metadata(load_metadata)
    worker.start_load_kv(build_forward_context(fixture.kv_caches))
    call_worker_layer_hooks(worker, fixture, save=False)
    worker.wait_for_save()
    worker.clear_connector_metadata()
    synchronize()

    compared = compare_loaded_blocks(
        fixture,
        saved,
        dispatch,
        target,
        save_metadata,
        load_metadata,
        worker,
    )
    log(f"PASS: Scheduler->UCM dump/load, compared_loaded_tensor_blocks={compared}")
