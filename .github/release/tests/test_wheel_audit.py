from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

RELEASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE_ROOT))
wheel_audit = importlib.import_module("ucm_release.wheel_audit")


def _report(
    external: dict[str, str | None],
    *,
    direct: list[str],
    dependencies: dict[str, list[str]],
) -> str:
    tree = {
        "ucm/test-extension.so": {
            "needed": direct,
            "libraries": {
                soname: {"needed": required}
                for soname, required in dependencies.items()
            },
        }
    }
    return "\n".join(
        (
            "DEBUG:auditwheel.wheel_abi:full_elftree:",
            json.dumps(tree, sort_keys=True),
            "The following external shared libraries are required by the wheel:",
            json.dumps(external, sort_keys=True),
        )
    )


def test_cann_boundary_derives_four_roots_and_unresolved_transitives() -> None:
    external = {
        "libascendcl.so": "/usr/local/Ascend/cann-9.1.0/lib64/libascendcl.so",
        "libcann_hixl.so": "/usr/local/Ascend/cann-9.1.0/lib64/libcann_hixl.so",
        "libmetadef.so": "/usr/local/Ascend/cann-9.1.0/lib64/libmetadef.so",
        "libruntime.so": "/usr/local/Ascend/cann-9.1.0/lib64/libruntime.so",
        "libhcomm.so": "/usr/local/Ascend/cann-9.1.0/lib64/libhcomm.so",
        "libhccl_v2.so": "/usr/local/Ascend/cann-9.1.0/lib64/libhccl_v2.so",
        "libascend_hal.so": None,
        "libccl_dpu.so": None,
    }
    report = _report(
        external,
        direct=[
            "libascendcl.so",
            "libcann_hixl.so",
            "libmetadef.so",
            "libruntime.so",
        ],
        dependencies={
            "libascendcl.so": [],
            "libcann_hixl.so": ["libhcomm.so"],
            "libmetadef.so": [],
            "libruntime.so": ["libascend_hal.so"],
            "libhcomm.so": ["libhccl_v2.so"],
            "libhccl_v2.so": ["libccl_dpu.so"],
            "libascend_hal.so": [],
            "libccl_dpu.so": [],
        },
    )

    closure = wheel_audit.validate_external_library_closure(
        report,
        expected_patterns=["/usr/local/Ascend/*"],
    )

    assert closure["external_library_roots"] == [
        "libascendcl.so",
        "libcann_hixl.so",
        "libmetadef.so",
        "libruntime.so",
    ]
    assert closure["external_libraries"] == sorted(external)
    assert closure["deferred_external_libraries"] == [
        "libascend_hal.so",
        "libccl_dpu.so",
    ]


def test_closure_does_not_defer_an_unresolved_direct_root() -> None:
    report = _report(
        {"libcudart.so.13": None},
        direct=["libcudart.so.13"],
        dependencies={"libcudart.so.13": []},
    )

    closure = wheel_audit.validate_external_library_closure(
        report,
        expected_patterns=["libcudart.so.13"],
    )

    assert closure["external_library_roots"] == ["libcudart.so.13"]
    assert closure["deferred_external_libraries"] == []


@pytest.mark.parametrize(
    ("reported", "configured"),
    (("libcudart.so.12", "libcudart.so.13"), ("libcudart.so.13", "libcudart.so.12")),
)
def test_cuda_boundary_rejects_the_wrong_accelerator_major(
    reported: str, configured: str
) -> None:
    report = _report(
        {reported: None},
        direct=[reported],
        dependencies={reported: []},
    )

    with pytest.raises(ValueError, match="matched no direct external"):
        wheel_audit.validate_external_library_closure(
            report,
            expected_patterns=[configured],
        )


def test_closure_rejects_a_direct_library_outside_provider_boundary() -> None:
    report = _report(
        {
            "libroot.so": "/usr/local/Ascend/cann/lib64/libroot.so",
            "libunrelated.so": "/other/libunrelated.so",
        },
        direct=["libroot.so", "libunrelated.so"],
        dependencies={"libroot.so": [], "libunrelated.so": []},
    )

    with pytest.raises(ValueError, match="outside the provider boundary"):
        wheel_audit.validate_external_library_closure(
            report,
            expected_patterns=["/usr/local/Ascend/*"],
        )


def test_closure_rejects_a_removed_transitive_edge() -> None:
    report = _report(
        {
            "libroot.so": "/usr/local/Ascend/cann/lib64/libroot.so",
            "libchild.so": "/usr/local/Ascend/cann/lib64/libchild.so",
        },
        direct=["libroot.so"],
        dependencies={"libroot.so": [], "libchild.so": []},
    )

    with pytest.raises(ValueError, match="outside the provider closure"):
        wheel_audit.validate_external_library_closure(
            report,
            expected_patterns=["/usr/local/Ascend/*"],
        )


def test_closure_rejects_ucm_owned_metrics_library() -> None:
    report = _report(
        {
            "libroot.so": "/usr/local/Ascend/cann/lib64/libroot.so",
            "libmetrics.so": "/wheel/ucm/shared/metrics/libmetrics.so",
        },
        direct=["libroot.so"],
        dependencies={
            "libroot.so": ["libmetrics.so"],
            "libmetrics.so": [],
        },
    )

    with pytest.raises(ValueError, match="UCM-owned libmetrics"):
        wheel_audit.validate_external_library_closure(
            report,
            expected_patterns=["/usr/local/Ascend/*"],
        )


def test_closure_fails_when_provider_pattern_matches_no_direct_library() -> None:
    report = _report(
        {"libroot.so": "/opt/Ascend/lib/libroot.so"},
        direct=["libroot.so"],
        dependencies={"libroot.so": []},
    )

    with pytest.raises(ValueError, match="matched no direct external"):
        wheel_audit.validate_external_library_closure(
            report,
            expected_patterns=["/usr/local/Ascend/*"],
        )


def test_unresolved_transitive_library_is_derived_from_provider_closure() -> None:
    report = _report(
        {
            "libroot.so": "/usr/local/Ascend/cann/lib64/libroot.so",
            "libunknown.so": None,
        },
        direct=["libroot.so"],
        dependencies={"libroot.so": ["libunknown.so"], "libunknown.so": []},
    )

    closure = wheel_audit.validate_external_library_closure(
        report,
        expected_patterns=["/usr/local/Ascend/*"],
    )

    assert closure["deferred_external_libraries"] == ["libunknown.so"]


def test_derived_deferred_library_requires_a_valid_soname() -> None:
    report = _report(
        {
            "libroot.so": "/usr/local/Ascend/cann/lib64/libroot.so",
            "not-a-soname": None,
        },
        direct=["libroot.so"],
        dependencies={"libroot.so": ["not-a-soname"], "not-a-soname": []},
    )

    with pytest.raises(ValueError, match="invalid external runtime SONAME"):
        wheel_audit.validate_external_library_closure(
            report,
            expected_patterns=["/usr/local/Ascend/*"],
        )


@pytest.mark.parametrize("pattern", ["/*", "/usr/../Ascend/*", "lib*.so"])
def test_external_runtime_exclude_pattern_rejects_broad_or_relative_values(
    pattern: str,
) -> None:
    with pytest.raises(ValueError, match="invalid external runtime exclude pattern"):
        wheel_audit.validate_exclude_pattern(pattern)
