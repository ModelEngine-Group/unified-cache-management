import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest


def _load_apply_patch_module():
    logger = MagicMock()
    ucm_module = ModuleType("ucm")
    ucm_module.__path__ = []
    logger_module = ModuleType("ucm.logger")
    logger_module.init_logger = MagicMock(return_value=logger)
    module_path = (
        Path(__file__).parents[3]
        / "ucm"
        / "integration"
        / "vllm"
        / "patch"
        / "apply_patch.py"
    )
    spec = importlib.util.spec_from_file_location("ucm_apply_patch_test", module_path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {"ucm": ucm_module, "ucm.logger": logger_module},
    ):
        spec.loader.exec_module(module)
    return module, logger


def test_release_version_keeps_vllm_ascend_patch_version():
    module, logger = _load_apply_patch_module()
    with patch.object(
        module,
        "_read_vllm_ascend_version_raw",
        return_value="0.19.1rc2",
    ):
        assert module.get_vllm_ascend_patch_version("0.26.0") == "0.19.1"
    logger.warning.assert_not_called()


@pytest.mark.parametrize(
    "vllm_version,expected", [("0.26.0+empty", "0.26.0"), ("0.27.1+empty", "0.27.1")]
)
def test_development_version_uses_matching_vllm_patch_version(vllm_version, expected):
    module, logger = _load_apply_patch_module()
    with patch.object(
        module,
        "_read_vllm_ascend_version_raw",
        return_value="0.19.1rc2.dev1475",
    ):
        assert module.get_vllm_ascend_patch_version(vllm_version) == expected
    logger.warning.assert_called_once()


def test_development_version_keeps_supported_matching_version():
    module, logger = _load_apply_patch_module()
    with patch.object(
        module,
        "_read_vllm_ascend_version_raw",
        return_value="0.26.0.dev12",
    ):
        assert module.get_vllm_ascend_patch_version("0.26.0") == "0.26.0"
    logger.warning.assert_not_called()


def test_development_version_does_not_select_unsupported_vllm_version():
    module, logger = _load_apply_patch_module()
    with patch.object(
        module,
        "_read_vllm_ascend_version_raw",
        return_value="0.19.1rc2.dev1475",
    ):
        assert module.get_vllm_ascend_patch_version("0.29.0") == "0.19.1"
    logger.warning.assert_not_called()


@pytest.mark.parametrize(
    "ascend_version,expected",
    [
        (None, False),
        ("0.25.1", False),
        ("0.25.99", False),
        ("0.26.0", True),
        ("0.26.0rc1", True),
        ("0.26.0.post1+build", True),
        ("0.26.1", True),
        ("0.27.1", True),
        ("0.28.0", True),
        ("0.100.0", True),
        ("1.0.0", True),
        ("0.19.1rc2.dev1475", True),
    ],
)
@pytest.mark.parametrize("enabled", [False, True])
def test_m3_patch_routing_uses_ascend_version_range(ascend_version, expected, enabled):
    module, _ = _load_apply_patch_module()
    imported = []
    original_import = __import__

    def capture_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("ucm.integration.vllm.patch."):
            imported.append(name)
            return MagicMock()
        return original_import(name, globals, locals, fromlist, level)

    with (
        patch.object(module, "ENABLE_UCM_PATCH", enabled),
        patch.object(module, "get_vllm_version", return_value="0.27.1"),
        patch.object(
            module, "_read_vllm_ascend_version_raw", return_value=ascend_version
        ),
        patch("builtins.__import__", side_effect=capture_import),
    ):
        module.apply_all_patches()

    prefix = "ucm.integration.vllm.patch."
    assert (prefix + "v0271.vllm.minimax_m3_kv_transfer_patch" in imported) is enabled
    assert (prefix + "v0260.vllm_ascend.minimax_m3_kv_transfer_patch" in imported) is (
        enabled and expected
    )
    if ascend_version in {"0.26.1", "0.27.1", "0.28.0", "0.100.0", "1.0.0"}:
        assert prefix + "v0260.vllm_ascend.cpu_binding_patch" not in imported


def test_vllm_0271_is_an_explicitly_supported_patch_version():
    module, _ = _load_apply_patch_module()
    assert "0.27.1" in module.get_supported_versions()
