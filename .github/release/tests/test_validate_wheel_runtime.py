from __future__ import annotations

import importlib.util
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / ".github" / "release" / "validate_wheel_runtime.py"
SPEC = importlib.util.spec_from_file_location("validate_wheel_runtime", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
runtime_validation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime_validation)


def test_parse_ldd_dependencies_collects_resolved_sonames() -> None:
    result = runtime_validation.parse_ldd_dependencies(
        """
        libmetrics-abcd.so => /opt/venv/site-packages/ucm.libs/libmetrics-abcd.so
        libcudart.so.13 => /usr/local/cuda/lib64/libcudart.so.13
        libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6
        """
    )

    assert result["libcudart.so.13"] == {"/usr/local/cuda/lib64/libcudart.so.13"}
    assert result["libmetrics-abcd.so"] == {
        "/opt/venv/site-packages/ucm.libs/libmetrics-abcd.so"
    }


def test_parse_ldd_missing_collects_unresolved_sonames() -> None:
    result = runtime_validation.parse_ldd_missing(
        """
        libascendcl.so => /usr/local/Ascend/lib64/libascendcl.so
        libascend_hal.so => not found
        """
    )

    assert result == {"libascend_hal.so"}


def test_extension_module_name_uses_installed_package_path(tmp_path: Path) -> None:
    package_root = tmp_path / "ucm"
    extension = (
        package_root
        / "shared"
        / "metrics"
        / ("ucmmetrics.cpython-312-x86_64-linux-gnu.so")
    )

    assert runtime_validation.extension_module_name(package_root, extension) == (
        "ucm.shared.metrics.ucmmetrics"
    )
    assert (
        runtime_validation.extension_module_name(
            package_root, package_root / "store" / "libstore.so"
        )
        is None
    )


def test_distribution_validation_requires_the_expected_backend() -> None:
    try:
        runtime_validation.validate_installed_distributions(
            "uc-manager-backend-that-is-not-installed", "1.0.0", None
        )
    except RuntimeError as error:
        assert "not installed" in str(error)
    else:
        raise AssertionError("missing expected backend distribution was accepted")


def test_distribution_validation_uses_the_planned_meta_name(monkeypatch) -> None:
    backend = type("Backend", (), {"version": "1.0.0"})()
    requested: list[str] = []

    def distribution(name: str):
        requested.append(name)
        if name == "supermarioyl-uc-manager-cuda-cu130":
            return backend
        raise metadata.PackageNotFoundError(name)

    def version(name: str) -> str:
        requested.append(name)
        if name == "supermarioyl-uc-manager":
            return "1.0.0"
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(
        runtime_validation.importlib.metadata, "distribution", distribution
    )
    monkeypatch.setattr(runtime_validation.importlib.metadata, "version", version)

    assert (
        runtime_validation.validate_installed_distributions(
            "supermarioyl-uc-manager-cuda-cu130",
            "1.0.0",
            "1.0.0",
            "supermarioyl-uc-manager",
        )
        is backend
    )
    assert requested == [
        "supermarioyl-uc-manager-cuda-cu130",
        "supermarioyl-uc-manager",
    ]


def test_external_runtime_library_cannot_resolve_to_sibling_wheel_libs(
    tmp_path: Path,
) -> None:
    bundled = (tmp_path / "uc_manager_cuda_cu130.libs" / "libcudart.so.13").resolve()
    try:
        runtime_validation.validate_external_resolution(
            {"libcudart.so.13": {str(bundled)}},
            ["libcudart.so.13"],
            {bundled},
        )
    except RuntimeError as error:
        assert "inside the Wheel" in str(error)
    else:
        raise AssertionError("bundled external Runtime library was accepted")


def test_auditwheel_unresolved_transitive_library_can_be_deferred() -> None:
    runtime_validation.validate_external_resolution(
        {},
        ["libascend_hal.so"],
        set(),
        missing={"libascend_hal.so"},
        deferred={"libascend_hal.so"},
    )


def test_cuda_direct_root_must_resolve_in_the_runtime() -> None:
    try:
        runtime_validation.validate_external_resolution(
            {},
            ["libcudart.so.13"],
            set(),
            missing={"libcudart.so.13"},
            deferred=set(),
        )
    except RuntimeError as error:
        assert "not deferred" in str(error)
    else:
        raise AssertionError("non-deferred missing Runtime library was accepted")


def test_cuda_direct_root_can_resolve_from_the_runtime() -> None:
    runtime_validation.validate_external_resolution(
        {"libcudart.so.13": {"/usr/local/cuda/lib64/libcudart.so.13"}},
        ["libcudart.so.13"],
        set(),
    )


def test_deferred_library_must_be_in_the_external_closure() -> None:
    try:
        runtime_validation.validate_external_resolution(
            {},
            ["libascendcl.so"],
            set(),
            missing={"libunknown.so"},
            deferred={"libunknown.so"},
        )
    except RuntimeError as error:
        assert "must be external" in str(error)
    else:
        raise AssertionError("unknown deferred Runtime library was accepted")


def test_runtime_requirements_are_checked_without_global_pip_state(monkeypatch) -> None:
    versions = {"wrapt": "1.17.2"}
    monkeypatch.setattr(
        runtime_validation.importlib.metadata,
        "version",
        lambda name: versions[name],
    )

    runtime_validation.validate_runtime_requirements(["wrapt==1.17.2"])

    try:
        runtime_validation.validate_runtime_requirements(["wrapt==1.17.1"])
    except RuntimeError as error:
        assert "does not match" in str(error)
    else:
        raise AssertionError("wrong Runtime dependency version was accepted")
