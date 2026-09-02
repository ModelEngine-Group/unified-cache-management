"""Catalog model schema-validation contract.

Only the pure schema-validation tests are retained: PEP 440 specifier
validation, resolved-snapshot structure validation before expansion,
rc/stable channel version gating, nested variant-id and catalog-reference
validation, strict OCI target-coordinate syntax, and non-string digest
rejection.  The builder-check, patch-manifest, and future-values
change-detector suites were removed per the slimming plan.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
RELEASE_ROOT = ROOT / ".github" / "release"
sys.path.insert(0, str(RELEASE_ROOT))

core = importlib.import_module("ucm_release.core")


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _builder(character: str) -> dict[str, object]:
    return {
        "root": {
            "repository": "docker.io/example/builder",
            "tag": "locked",
            "index_digest": _digest(character),
            "manifest_digest": _digest(character),
            "config_digest": _digest(character),
        },
        "sources": [],
        "copy_paths": [],
        "checks": [
            {"kind": "python", "version": "3.12", "abi": "cp312"},
            {"kind": "python-soabi", "prefix": "cpython"},
            {"kind": "command", "name": "g++"},
            {"kind": "file", "path": "/opt/toolkit/include/runtime.h"},
        ],
    }


def _profile(
    profile_id: str,
    architectures: list[str],
    *,
    accelerator: str = "cuda",
    npu_arch: str = "na",
) -> dict[str, object]:
    return {
        "id": profile_id,
        "accelerator": accelerator,
        "accelerator_runtime": "cuda-13.0" if accelerator == "cuda" else "cann-9.0.0",
        "npu_arch": [npu_arch],
        "os": ["ubuntu-22.04"],
        "cpu_arch": architectures,
        "python_version": "3.12",
        "python_abi": "cp312",
        "wheel_version": f"0.5.0rc1+{profile_id}",
        "wheel_platform": "manylinux_2_28" if accelerator == "cuda" else "linux",
        "binary_profile_id": f"release-{profile_id}",
        "allowed_dt_needed": ["libc.so.6"],
        "external_required_dependencies": [],
        "validation_targets": [profile_id],
        "required_native": ["ucmtrans"],
        "forbidden_native": ["forbidden"],
        "build": {
            "docker_target": "wheel",
            "platform_arg": accelerator,
        },
        "builders": {
            architecture: _builder(str(index + 1))
            for index, architecture in enumerate(architectures)
        },
    }


def _catalog() -> dict[str, object]:
    return {
        "kind": "release-config",
        "schema_version": 2,
        "ucm_version": "0.5.0rc1",
        "lanes": ["feature-candidate", "protected-tag"],
        "runner_map": {"amd64": "runner-x64", "arm64": "runner-arm64"},
        "python_runtime_dependencies": [
            {
                "requirement": "wrapt==1.17.2",
                "import_name": "wrapt",
                "wheel_artifacts": {
                    "cp312": {
                        "amd64": {
                            "filename": "wrapt-cp312-amd64.whl",
                            "sha256": _digest("a"),
                        },
                        "arm64": {
                            "filename": "wrapt-cp312-arm64.whl",
                            "sha256": _digest("b"),
                        },
                    }
                },
            },
            {"python_build_lock": "packaging", "import_name": "packaging"},
        ],
        "python_build_lock": {
            "packages": {
                "packaging": {
                    "version": "24.2",
                    "filename": "packaging-24.2-py3-none-any.whl",
                    "sha256": _digest("f"),
                }
            },
            "pyyaml": {
                "version": "6.0.2",
                "artifacts": {
                    "cp312": {
                        "amd64": {
                            "filename": "PyYAML-cp312-amd64.whl",
                            "sha256": _digest("c"),
                        },
                        "arm64": {
                            "filename": "PyYAML-cp312-arm64.whl",
                            "sha256": _digest("d"),
                        },
                    }
                },
            },
            "cmake": {
                "version": "3.31.6",
                "artifacts": {
                    "amd64": {
                        "filename": "cmake-amd64.whl",
                        "sha256": _digest("e"),
                    },
                    "arm64": {
                        "filename": "cmake-arm64.whl",
                        "sha256": _digest("f"),
                    },
                },
            },
        },
        "wheel_profiles": [_profile("shared", ["amd64", "arm64"])],
        "upstream_products": [
            {
                "id": "vllm",
                "runtime_product": "vllm",
                "repository": "docker.io/vllm/vllm-openai",
                "version_specifier": ">=0.20,<0.23",
                "channels": ["stable"],
                "variants": [
                    {
                        "id": "default",
                        "tag_suffix": "",
                        "npu_arch": "na",
                    }
                ],
                "required_cpu_architectures": ["amd64", "arm64"],
            }
        ],
        "compatibility": {
            "rules": [
                {
                    "id": "cuda-stable",
                    "upstream_products": ["vllm"],
                    "version_specifier": ">=0.20,<0.23",
                    "variants": ["default"],
                    "accelerator": "cuda",
                    "accelerator_runtimes": ["cuda-13.0"],
                    "npu_architectures": ["na"],
                    "operating_systems": ["ubuntu-22.04"],
                    "cpu_architectures": ["amd64", "arm64"],
                    "python_abis": ["cp312"],
                    "upstream_channels": ["stable"],
                }
            ],
            "excluded_upstream_patterns": ["nightly", "dev"],
        },
        "matrix_limits": {
            "max_wheel_tasks": 8,
            "max_image_tasks": 16,
            "max_family_tasks": 8,
        },
    }


def _snapshot(version: str, tag: str, character: str) -> dict[str, object]:
    return {
        "product_id": "vllm",
        "repository": "docker.io/vllm/vllm-openai",
        "tag": tag,
        "version": version,
        "channel": "stable",
        "variant": "default",
        "index_digest": _digest(character),
        "members": {
            "amd64": {
                "manifest_digest": _digest(character),
                "config_digest": _digest(character),
            },
            "arm64": {
                "manifest_digest": _digest(character),
                "config_digest": _digest(character),
            },
        },
        "target_repository": "ghcr.io/example/vllm",
        "target_tag": f"{tag}-ucm",
    }


def test_catalog_rejects_invalid_pep440_specifier() -> None:
    catalog = _catalog()
    catalog["compatibility"]["rules"][0]["version_specifier"] = ">0.18*"

    with pytest.raises(ValueError, match="PEP 440"):
        core.validate_catalog(catalog)


def test_runtime_channel_must_explicitly_match_stable_or_rc_version() -> None:
    catalog = _catalog()
    catalog["upstream_products"][0]["version_specifier"] = ">=0.21.0rc1,<0.22"
    catalog["compatibility"]["rules"][0]["version_specifier"] = ">=0.21.0rc1,<0.22"
    snapshot = _snapshot("0.21.0rc1", "v0.21.0rc1", "9")

    with pytest.raises(ValueError, match="channel stable requires a final version"):
        core.expand_release_plan(catalog, [snapshot], lane="feature-candidate")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda snapshot: snapshot.update(index_digest="sha256:not-a-digest"),
        lambda snapshot: snapshot["members"]["amd64"].update(unexpected="field"),
    ],
)
def test_resolved_snapshot_structure_is_validated_before_expansion(mutation) -> None:
    snapshot = _snapshot("0.21.0", "v0.21.0", "a")
    mutation(snapshot)

    with pytest.raises(ValueError, match=r"resolved_upstreams\[0\]"):
        core.expand_release_plan(_catalog(), [snapshot], lane="feature-candidate")


@pytest.mark.parametrize("version", ["0.21.0a1", "0.21.0b2", "0.21.0.dev3"])
def test_rc_channel_accepts_only_actual_rc_versions(version: str) -> None:
    catalog = _catalog()
    catalog["upstream_products"][0]["version_specifier"] = ">=0.21.0.dev1,<0.22"
    catalog["upstream_products"][0]["channels"] = ["rc"]
    rule = catalog["compatibility"]["rules"][0]
    rule["version_specifier"] = ">=0.21.0.dev1,<0.22"
    rule["upstream_channels"] = ["rc"]
    snapshot = _snapshot(version, f"v{version}", "b")
    snapshot["channel"] = "rc"

    with pytest.raises(ValueError, match="channel rc requires a plain rcN version"):
        core.expand_release_plan(catalog, [snapshot], lane="feature-candidate")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda catalog: catalog["upstream_products"][0]["variants"].append(
                {"id": "default", "npu_arch": "na"}
            ),
            "duplicate upstream variant id 'default'",
        ),
        (
            lambda catalog: catalog["compatibility"]["rules"][0].update(
                upstream_products=["missing-product"]
            ),
            "unknown upstream product 'missing-product'",
        ),
        (
            lambda catalog: catalog["compatibility"]["rules"][0].update(
                variants=["missing-variant"]
            ),
            "unknown variant 'missing-variant'",
        ),
    ],
)
def test_nested_variant_ids_and_catalog_references_are_validated(
    mutation, message: str
) -> None:
    catalog = _catalog()
    mutation(catalog)

    with pytest.raises(ValueError, match=message):
        core.validate_catalog(catalog)


@pytest.mark.parametrize("version", ["0.21.0rc1.dev1", "0.21.0rc1.post1"])
def test_rc_channel_rejects_development_or_post_rc_versions(version: str) -> None:
    catalog = _catalog()
    catalog["upstream_products"][0]["version_specifier"] = ">=0.21.0.dev1,<0.22"
    catalog["upstream_products"][0]["channels"] = ["rc"]
    rule = catalog["compatibility"]["rules"][0]
    rule["version_specifier"] = ">=0.21.0.dev1,<0.22"
    rule["upstream_channels"] = ["rc"]
    snapshot = _snapshot(version, f"v{version}", "1")
    snapshot["channel"] = "rc"

    with pytest.raises(ValueError, match="channel rc requires a plain rcN version"):
        core.expand_release_plan(catalog, [snapshot], lane="feature-candidate")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("target_repository", "GHCR.io/example/vllm"),
        ("target_repository", "ghcr.io/example//vllm"),
        ("target_tag", "-not-an-oci-tag"),
        ("target_tag", "a" * 129),
    ],
)
def test_resolved_target_coordinate_uses_strict_oci_syntax(
    field: str, value: str
) -> None:
    snapshot = _snapshot("0.21.0", "v0.21.0", "2")
    snapshot[field] = value

    with pytest.raises(ValueError, match=field):
        core.expand_release_plan(_catalog(), [snapshot], lane="feature-candidate")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda snapshot: snapshot.update(index_digest=None),
        lambda snapshot: snapshot["members"]["amd64"].update(manifest_digest=123),
    ],
)
def test_non_string_resolved_digests_raise_value_error(mutation) -> None:
    snapshot = _snapshot("0.21.0", "v0.21.0", "3")
    mutation(snapshot)

    with pytest.raises(ValueError, match="exact sha256 digest"):
        core.expand_release_plan(_catalog(), [snapshot], lane="feature-candidate")
