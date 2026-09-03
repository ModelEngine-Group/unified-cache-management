"""Configuration for the UCM model compatibility checker."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

MODEL_ENV = "UCM_MODEL_CHECK_MODEL"
TOKENS_ENV = "UCM_MODEL_CHECK_TOKENS"
BLOCK_SIZE_ENV = "UCM_MODEL_CHECK_BLOCK_SIZE"
USE_LAYERWISE_ENV = "UCM_MODEL_CHECK_USE_LAYERWISE"
ADDITIONAL_CONFIG_ENV = "UCM_MODEL_CHECK_ADDITIONAL_CONFIG"
STORE_PIPELINE_ENV = "UCM_MODEL_CHECK_STORE_PIPELINE"
STORAGE_BACKENDS_ENV = "UCM_MODEL_CHECK_STORAGE_BACKENDS"
DEVICE_ENV = "UCM_MODEL_CHECK_DEVICE_ID"
DTYPE_ENV = "UCM_MODEL_CHECK_DTYPE"
KV_CACHE_DTYPE_ENV = "UCM_MODEL_CHECK_KV_CACHE_DTYPE"


def _bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _dict_env(name: str) -> dict[str, Any]:
    value = os.environ.get(name)
    if value is None:
        return {}
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be a JSON object: {exc}") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{name} must be a JSON object")
    return decoded


@dataclass(frozen=True)
class ModelCheckConfig:
    """All user-configurable model-check values."""

    model: str
    tokens: int
    block_size: int
    use_layerwise: bool
    additional_config: dict[str, Any]
    store_pipeline: str
    storage_backends: str
    visible_devices: str
    dtype: str
    kv_cache_dtype: str


def load_config() -> ModelCheckConfig:
    """Load model-check configuration from the child-process environment."""
    return ModelCheckConfig(
        model=os.environ.get(MODEL_ENV, "/models/Qwen2.5-14B-Instruct"),
        tokens=_int_env(TOKENS_ENV, 4096),
        block_size=_int_env(BLOCK_SIZE_ENV, 64),
        use_layerwise=_bool_env(USE_LAYERWISE_ENV, True),
        additional_config=_dict_env(ADDITIONAL_CONFIG_ENV),
        store_pipeline=os.environ.get(STORE_PIPELINE_ENV, "Cache|Posix"),
        storage_backends=os.environ.get(STORAGE_BACKENDS_ENV, "./build/data"),
        visible_devices=os.environ.get(DEVICE_ENV, "0"),
        dtype=os.environ.get(DTYPE_ENV, "auto"),
        kv_cache_dtype=os.environ.get(KV_CACHE_DTYPE_ENV, "auto"),
    )
