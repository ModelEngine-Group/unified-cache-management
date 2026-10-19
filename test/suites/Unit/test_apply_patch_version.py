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


@pytest.mark.parametrize(
    "raw_version,expected",
    [
        (None, None),
        ("", None),
        ("0.26.0", "0.26.0"),
        ("0.26.0rc1", "0.26.0"),
        ("0.26.0.post1+build", "0.26.0"),
        (" 0.27.1+empty ", "0.27.1"),
    ],
)
def test_installed_versions_normalize_suffixes(raw_version, expected):
    module, logger = _load_apply_patch_module()
    with (
        patch.object(module, "_read_vllm_version_raw", return_value=raw_version),
        patch.object(module, "_read_vllm_ascend_version_raw", return_value=raw_version),
    ):
        assert module.get_vllm_version() == expected
        assert module.get_vllm_ascend_version() == expected
    logger.warning.assert_not_called()


@pytest.mark.parametrize(
    "vllm_version,ascend_version,expected",
    [
        ("0.27.1", None, False),
        ("0.25.1", "0.25.1", False),
        ("0.25.99", "0.25.99", False),
        ("0.26.0", "0.26.0", True),
        ("0.26.0rc1", "0.26.0rc1", True),
        ("0.26.0.post1+build", "0.26.0.post1+build", True),
        ("0.26.1", "0.26.1", True),
        ("0.27.1", "0.27.1", True),
        ("0.28.0", "0.28.0", True),
        ("0.100.0", "0.100.0", True),
        ("1.0.0", "1.0.0", True),
        ("0.27.1", "0.19.1rc2", True),
        ("0.25.1", "0.27.1", False),
    ],
)
@pytest.mark.parametrize("enabled", [False, True])
def test_m3_patch_routing_uses_aligned_version_range(
    vllm_version, ascend_version, expected, enabled
):
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
        patch.object(module, "_read_vllm_version_raw", return_value=vllm_version),
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
    assert (prefix + "v0260.vllm_ascend.cpu_binding_patch" in imported) is (
        enabled and expected
    )


def test_vllm_0271_is_an_explicitly_supported_patch_version():
    module, _ = _load_apply_patch_module()
    assert "0.27.1" in module.get_supported_versions()


@pytest.mark.parametrize(
    "version,ascend_version,enabled,expected",
    [
        ("0.30.0", "0.30.0", True, True),
        ("0.30.0rc1", "0.30.0rc1", True, True),
        ("0.30.0", "0.29.0", True, True),
        ("0.30.0", None, True, False),
        ("0.30.0", "0.30.0", False, False),
        ("0.28.0", "0.30.0", True, False),
        ("0.29.0", "0.29.0", True, False),
        ("0.31.0", "0.31.0", True, False),
    ],
)
def test_mamba_split_patch_is_scoped_to_0300(
    version, ascend_version, enabled, expected
):
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
        patch.object(module, "_read_vllm_version_raw", return_value=version),
        patch.object(
            module, "_read_vllm_ascend_version_raw", return_value=ascend_version
        ),
        patch("builtins.__import__", side_effect=capture_import),
    ):
        module.apply_all_patches()

    target = (
        "ucm.integration.vllm.patch.v0300.vllm_ascend.mamba_block_aligned_split_patch"
    )
    assert (target in imported) is expected
    assert "0.30.0" in module.get_supported_versions()
