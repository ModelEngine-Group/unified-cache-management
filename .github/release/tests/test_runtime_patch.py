"""Regression tests for the in-tree vLLM runtime patch dispatcher."""

from __future__ import annotations

import builtins
import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PATCH_SOURCE = ROOT / "ucm/integration/vllm/patch/apply_patch.py"


class _Logger:
    def info(self, *_args: object, **_kwargs: object) -> None:
        pass

    def warning(self, *_args: object, **_kwargs: object) -> None:
        pass

    def error(self, *_args: object, **_kwargs: object) -> None:
        pass


@pytest.fixture
def patch_module(monkeypatch: pytest.MonkeyPatch):
    ucm_package = types.ModuleType("ucm")
    ucm_package.__path__ = [str(ROOT / "ucm")]
    logger_module = types.ModuleType("ucm.logger")
    logger_module.init_logger = lambda _name: _Logger()
    monkeypatch.setitem(sys.modules, "ucm", ucm_package)
    monkeypatch.setitem(sys.modules, "ucm.logger", logger_module)
    monkeypatch.setenv("ENABLE_UCM_PATCH", "1")
    name = "runtime_patch_dispatcher_under_test"
    spec = importlib.util.spec_from_file_location(name, PATCH_SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_dispatcher_owns_version_routing_without_generated_manifest(
    patch_module,
) -> None:
    assert patch_module.get_supported_versions() == [
        "0.11.0",
        "0.17.0",
        "0.18.0",
        "0.19.1",
        "0.20.2",
        "0.21.0",
        "0.22.1",
        "0.23.0",
        "0.24.0",
        "0.25.1",
        "0.26.0",
        "0.27.0",
        "0.28.0",
    ]
    source = PATCH_SOURCE.read_text(encoding="utf-8")
    assert "match version:" in source
    assert "runtime_patch_" + "rules.json" not in source
    assert "UCM_RUNTIME_PATCH_" + "VARIANTS" not in source
    assert "importlib.resources" not in source


def test_enabled_source_dispatcher_does_not_read_generated_files(
    patch_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    modules = {
        "ucm.integration.vllm.patch.logger_patch": {"patch_logger": object()},
        "ucm.integration.vllm.patch.bind_memory_patch": {},
        "ucm.integration.vllm.patch.v0210.vllm_ascend.mamba_copy_order_patch": {},
        "ucm.integration.vllm.patch.v0270.vllm.models.kimi_k3.nvidia.kimi_k3_mla_kv_hook_patch": {},
    }
    for name, attributes in modules.items():
        fake = types.ModuleType(name)
        for attribute, value in attributes.items():
            setattr(fake, attribute, value)
        monkeypatch.setitem(sys.modules, name, fake)
    monkeypatch.setattr(patch_module, "get_vllm_version", lambda: "0.17.0")
    monkeypatch.setattr(patch_module, "get_vllm_ascend_version", lambda: None)

    patch_module.apply_all_patches()


def test_representative_ascend_dispatch_preserves_connector_and_import_order(
    patch_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = [
        "ucm.integration.vllm.patch.logger_patch",
        "ucm.integration.vllm.patch.ucm_connector_registration_patch",
        "ucm.integration.vllm.patch.load_failure_patch",
        "ucm.integration.vllm.patch.bind_memory_patch",
        "ucm.integration.vllm.patch.v0230.vllm_ascend.ascend_hybrid_cache_patch",
        "ucm.integration.vllm.patch.v0230.vllm_ascend.cpu_binding_patch",
        "ucm.integration.vllm.patch.v0230.vllm_ascend.sfa_kv_transfer_patch",
        "ucm.integration.vllm.patch.v0210.vllm_ascend.mamba_copy_order_patch",
        "ucm.integration.vllm.patch.v0270.vllm.models.kimi_k3.nvidia.kimi_k3_mla_kv_hook_patch",
    ]
    for name in expected:
        fake = types.ModuleType(name)
        if name.endswith("logger_patch"):
            fake.patch_logger = object()
        monkeypatch.setitem(sys.modules, name, fake)
    monkeypatch.setattr(patch_module, "get_vllm_version", lambda: "0.23.0")
    monkeypatch.setattr(patch_module, "get_vllm_ascend_version", lambda: "0.23.0")
    imported: list[str] = []
    real_import = builtins.__import__

    def record_import(name: str, *args: object, **kwargs: object):
        if name in expected:
            imported.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", record_import)

    patch_module.apply_all_patches()

    assert imported == expected


@pytest.mark.parametrize(
    ("raw", "normalized"),
    [
        ("0.18.0rc1+vendor", "0.18.0"),
        ("0.20.2.post1", "0.20.2"),
        ("0.26.0", "0.26.0"),
    ],
)
def test_ascend_version_normalization_keeps_existing_dispatch_contract(
    patch_module, monkeypatch: pytest.MonkeyPatch, raw: str, normalized: str
) -> None:
    monkeypatch.setattr(patch_module, "_read_vllm_ascend_version_raw", lambda: raw)

    assert patch_module.get_vllm_ascend_version_full() == raw
    assert patch_module.get_vllm_ascend_version() == normalized
