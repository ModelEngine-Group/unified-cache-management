"""Version and opt-in behavior without an installed SGLang package."""

import importlib
import importlib.abc
import importlib.metadata
import logging
import os
import sys
import types
import unittest
from unittest.mock import Mock, patch

from test_sglang_direct_preparation import load_source

patch_module = load_source(
    "ucm/integration/sglang/patch/apply_patch.py",
    importlib=importlib,
    os=os,
    init_logger=lambda name: logging.getLogger(name),
)
boot_module = load_source(
    "ucm/integration/sglang/patch/boot_patch.py", importlib=importlib, sys=sys
)


class PatchSelectionTest(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"ENABLE_UCM_PATCH": "1"}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_release_suffixes_select_the_base_patch(self):
        for version, expected in (
            ("0.5.20", "v0520"),
            ("0.5.21", "v0521"),
            ("0.5.22+build", "v0521"),
            ("0.5.21rc1", "v0521"),
            ("0.5.21.post1", "v0521"),
            (" 0.5.22rc1+build ", "v0521"),
            ("0.5.20.post1+build", "v0520"),
            ("0.5.19.post1", None),
            ("0.5.18", None),
        ):
            with self.subTest(version=version):
                self.assertEqual(patch_module.patch_package_for(version), expected)

    def test_disabled_hook_does_not_probe_framework(self):
        os.environ["ENABLE_UCM_PATCH"] = "0"
        with patch.object(importlib.metadata, "version") as version:
            patch_module.apply_all_patches()
        version.assert_not_called()

    def test_unsupported_version_does_not_break_import(self):
        for version in ("0.5.18", "0.5.19.post1", "0.5.99rc1"):
            with (
                self.subTest(version=version),
                patch.object(importlib.metadata, "version", return_value=version),
                patch.object(
                    patch_module, "_has_native_external_linker", return_value=False
                ),
                patch.object(importlib, "import_module") as importer,
                self.assertLogs(patch_module.logger, level="WARNING"),
                patch.dict(
                    sys.modules,
                    {"ucm.integration.sglang.patch.apply_patch": patch_module},
                ),
            ):
                framework = types.SimpleNamespace()
                loader = Mock()
                loader.exec_module.side_effect = lambda module: setattr(
                    module, "existing_backend", True
                )
                boot_module._SGLangPatchLoader(loader).exec_module(framework)
                self.assertTrue(framework.existing_backend)
                importer.assert_not_called()

    def test_supported_patch_runs(self):
        for version, package in (
            ("0.5.20", "v0520"),
            ("0.5.21", "v0521"),
            ("0.5.22", "v0521"),
            ("0.5.21rc1", "v0521"),
            ("0.5.21.post1+build", "v0521"),
        ):
            with (
                self.subTest(version=version),
                patch.object(importlib.metadata, "version", return_value=version),
                patch.object(
                    patch_module, "_has_native_external_linker", return_value=False
                ),
                patch.object(
                    importlib,
                    "import_module",
                    return_value=types.SimpleNamespace(apply=Mock()),
                ) as importer,
            ):
                patch_module.apply_all_patches()
                importer.assert_called_once_with(
                    f"ucm.integration.sglang.patch.{package}"
                )
                importer.return_value.apply.assert_called_once_with()

    def test_native_support_skips_patch(self):
        with (
            patch.object(importlib.metadata, "version", return_value="0.5.99"),
            patch.object(
                patch_module, "_has_native_external_linker", return_value=True
            ),
            patch.object(importlib, "import_module") as importer,
        ):
            patch_module.apply_all_patches()
            importer.assert_not_called()

    def test_supported_patch_failure_is_not_hidden(self):
        with (
            patch.object(importlib.metadata, "version", return_value="0.5.22"),
            patch.object(
                patch_module, "_has_native_external_linker", return_value=False
            ),
            patch.object(
                importlib, "import_module", side_effect=ValueError("API drift")
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "Failed to apply"):
                patch_module.apply_all_patches()

    def test_explicit_version_pin_still_fails(self):
        os.environ["UCM_SGLANG_VERSION"] = "0.5.22"
        with patch.object(importlib.metadata, "version", return_value="0.5.21"):
            with self.assertRaisesRegex(RuntimeError, "version mismatch"):
                patch_module.apply_all_patches()

    def test_missing_distribution_skips_patch(self):
        with (
            patch.object(
                importlib.metadata,
                "version",
                side_effect=importlib.metadata.PackageNotFoundError,
            ),
            self.assertLogs(patch_module.logger, level="WARNING"),
        ):
            patch_module.apply_all_patches()


if __name__ == "__main__":
    unittest.main()
