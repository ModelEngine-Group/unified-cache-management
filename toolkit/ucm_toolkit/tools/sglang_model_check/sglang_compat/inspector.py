"""Keep unstable SGLang imports out of the checker core."""

from __future__ import annotations

import importlib
import importlib.metadata
from contextlib import contextmanager
from typing import Any


def _optional_callable(module_name: str, name: str):
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    value = getattr(module, name, None)
    return value if callable(value) else None


@contextmanager
def server_args_guard():
    """Publish SGLang's minimal scoped test context when no Engine did so.

    Newer SGLang model constructors read process-wide ServerArgs even during
    weight-free initialization.  RuntimeContext.override_server_args() is the
    framework-provided reversible boundary for tests and offline tools.
    """
    runtime = importlib.import_module("sglang.srt.runtime_context")
    get_server_args = getattr(runtime, "get_server_args", None)
    if callable(get_server_args):
        try:
            get_server_args()
            yield
            return
        except ValueError as exc:
            if "server args" not in str(exc).lower():
                raise

    get_context = getattr(runtime, "get_context", None)
    context = get_context() if callable(get_context) else None
    override = getattr(context, "override_server_args", None)
    if callable(override):
        with override():
            yield
        return

    raise RuntimeError(
        "this SGLang version requires global ServerArgs during model init but "
        "does not expose RuntimeContext.override_server_args()"
    )


def _call_bool(module_name: str, name: str, argument: Any):
    function = _optional_callable(module_name, name)
    if function is None:
        return None, None, f"{module_name}.{name} is unavailable"
    try:
        return bool(function(argument)), name, None
    except Exception as exc:
        return None, name, f"{name}: {type(exc).__name__}: {exc}"


def _linear_attention(model_config: Any):
    if hasattr(model_config, "linear_attn_registry_result"):
        value = getattr(model_config, "linear_attn_registry_result")
        if value:
            return True, "linear_attn_registry_result", None

    function = _optional_callable(
        "sglang.srt.configs.hybrid_arch", "mambaish_config"
    )
    if function is not None:
        try:
            value = function(model_config)
            return value is not None and bool(value), "mambaish_config", None
        except Exception as exc:
            mamba_error = f"mambaish_config: {type(exc).__name__}: {exc}"
    else:
        mamba_error = "mambaish_config is unavailable"

    function = _optional_callable(
        "sglang.srt.configs.model_config", "uses_kda_attention"
    )
    if function is not None:
        try:
            return (
                bool(function(model_config.hf_text_config)),
                "uses_kda_attention",
                mamba_error,
            )
        except Exception as exc:
            return None, "uses_kda_attention", f"{mamba_error}; {type(exc).__name__}: {exc}"

    text_config = model_config.hf_text_config
    indicators = (
        "linear_attention_config",
        "mamba_config",
        "mamba2_cache_params",
        "hybrid_override_pattern",
        "mtp_hybrid_override_pattern",
        "kda_config",
    )
    if any(getattr(text_config, name, None) is not None for name in indicators):
        return True, "config_structure", mamba_error
    return None, "undetermined", mamba_error


def environment_info() -> dict[str, Any]:
    try:
        version = importlib.metadata.version("sglang")
    except importlib.metadata.PackageNotFoundError:
        try:
            version = str(importlib.import_module("sglang").__version__)
        except Exception:
            version = "unknown"

    platform = "unknown"
    device_name = None
    try:
        current = importlib.import_module("sglang.srt.platforms").current_platform
        platform = str(getattr(current, "device_type", None) or "unknown")
        device_name = getattr(current, "device_name", None)
        if callable(device_name):
            device_name = device_name()
        if bool(getattr(current, "is_hip", lambda: False)()):
            platform = "rocm"
    except Exception:
        pass
    return {"sglang_version": version, "platform": platform, "device_name": device_name}


def supported_platforms_for_quantization(quantization: Any) -> list[str] | None:
    """Return a conservative platform constraint for accelerator-specific formats."""
    value = str(quantization or "").lower().replace("-", "_")
    if "nvfp4" in value or value == "modelopt_fp4":
        return ["cuda"]
    return None


def collect_model_facts(model_config: Any) -> dict[str, Any]:
    hf_config = model_config.hf_config
    text_config = model_config.hf_text_config
    architectures = list(getattr(hf_config, "architectures", None) or [])
    attention_arch = getattr(model_config.attention_arch, "name", None)
    if attention_arch is None:
        attention_arch = str(model_config.attention_arch).split(".")[-1]

    linear, linear_source, linear_error = _linear_attention(model_config)
    dsa, dsa_source, dsa_error = _call_bool(
        "sglang.srt.configs.model_config", "is_deepseek_dsa", text_config
    )
    dsv4, dsv4_source, dsv4_error = _call_bool(
        "sglang.srt.configs.model_config", "is_deepseek_v4", hf_config
    )
    minimax, minimax_source, minimax_error = _call_bool(
        "sglang.srt.configs.model_config", "is_minimax_sparse", hf_config
    )
    try:
        PoolName = importlib.import_module(
            "sglang.srt.mem_cache.hicache_storage"
        ).PoolName
        pool_names = [str(item.value) for item in PoolName]
    except Exception:
        pool_names = []

    errors = [item for item in (linear_error, dsa_error, dsv4_error, minimax_error) if item]
    quantization = getattr(model_config, "quantization", None)
    return {
        "architectures": architectures,
        "model_type": getattr(hf_config, "model_type", None),
        "attention_arch": attention_arch,
        "is_hybrid_swa": bool(getattr(model_config, "is_hybrid_swa", False)),
        "has_linear_attention": linear,
        "is_dsa": dsa,
        "is_deepseek_v4": dsv4,
        "is_minimax_sparse": minimax,
        "num_layers": int(max(getattr(model_config, "num_hidden_layers", 0), getattr(model_config, "num_attention_layers", 0))),
        "dtype": str(model_config.dtype),
        "quantization": quantization,
        "supported_platforms": supported_platforms_for_quantization(quantization),
        "sglang_pool_names": pool_names,
        "detection_sources": {
            "linear_attention": linear_source,
            "dsa": dsa_source,
            "deepseek_v4": dsv4_source,
            "minimax_sparse": minimax_source,
        },
        "detection_errors": errors,
    }


def refine_facts_from_model(facts: dict[str, Any], model: Any) -> dict[str, Any]:
    """Use the instantiated meta model only when config discovery was unknown."""
    if facts.get("has_linear_attention") is not None:
        return facts
    markers = ("mamba", "linearattention", "kda", "gdn", "shortconv")
    matches = []
    for name, module in model.named_modules():
        identity = f"{type(module).__module__}.{type(module).__name__}".lower()
        if any(marker in identity for marker in markers):
            matches.append(name or "<root>")
    if matches:
        facts["has_linear_attention"] = True
        facts["detection_sources"]["linear_attention"] = "meta_model_modules"
        facts["linear_attention_modules"] = matches[:32]
    return facts


def refine_facts_from_model(facts: dict[str, Any], model: Any) -> dict[str, Any]:
    if facts.get("has_linear_attention") is not None:
        return facts
    markers = ("mamba", "linearattention", "kda", "gdn")
    matches = []
    for name, module in model.named_modules():
        identity = f"{type(module).__module__}.{type(module).__name__}".lower()
        if any(marker in identity for marker in markers):
            matches.append(name or "<root>")
    updated = dict(facts)
    if matches:
        updated["has_linear_attention"] = True
        updated["detection_sources"] = dict(updated["detection_sources"])
        updated["detection_sources"]["linear_attention"] = "meta_model_modules"
        updated["linear_attention_modules"] = matches[:32]
    return updated
