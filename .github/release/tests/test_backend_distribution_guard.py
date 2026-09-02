"""Backend distribution names must keep the shared ``ucm`` package exclusive."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[3]


class _Distribution:
    def __init__(self, name: str):
        self.metadata = {"Name": name}


def _load_guard(monkeypatch, names: list[str]):
    logger_patch = ModuleType("ucm.integration.vllm.patch.logger_patch")
    logger_patch.patch_logger = lambda: None
    monkeypatch.setitem(
        sys.modules, "ucm.integration.vllm.patch.logger_patch", logger_patch
    )
    monkeypatch.setattr(
        importlib.metadata,
        "distributions",
        lambda: [_Distribution(name) for name in names],
    )
    spec = importlib.util.spec_from_file_location(
        "ucm_backend_guard_under_test", ROOT / "ucm" / "__init__.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "name",
    [
        "uc-manager-cuda",
        "uc-manager-cuda-cu129",
        "uc-manager-cann-a2",
        "uc-manager-cann851-a2",
        "uc_manager_cann910_a3",
        "uc-manager-cann910-a3-mc0311post1",
        "supermarioyl-uc-manager-cuda-cu130",
        "supermarioyl_uc_manager_cann901_a2",
    ],
)
def test_versioned_backend_distribution_names_are_recognized(monkeypatch, name) -> None:
    module = _load_guard(monkeypatch, [name])

    assert module._is_backend_distribution(name)


def test_multiple_versioned_backend_distributions_are_rejected(monkeypatch) -> None:
    with pytest.raises(ImportError, match="new virtual environment"):
        _load_guard(
            monkeypatch,
            ["uc-manager-cann851-a2", "uc-manager-cuda-cu129"],
        )


def test_empty_meta_distribution_is_not_counted_as_a_backend(monkeypatch) -> None:
    module = _load_guard(
        monkeypatch,
        [
            "uc-manager",
            "uc-manager-cuda-cu130",
            "supermarioyl-uc-manager",
        ],
    )

    assert not module._is_backend_distribution("uc-manager")
    assert not module._is_backend_distribution("supermarioyl-uc-manager")


def test_prefixed_and_canonical_backends_cannot_coexist(monkeypatch) -> None:
    with pytest.raises(ImportError, match="Multiple UCM backend distributions"):
        _load_guard(
            monkeypatch,
            ["uc-manager-cuda-cu129", "supermarioyl-uc-manager-cuda-cu129"],
        )
