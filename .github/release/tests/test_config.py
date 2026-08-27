"""RED structural contract for the slim UCM release package.

This test deliberately describes the target tree.  It must stay red until the
legacy release subsystem has been replaced by the compact package.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import yaml
from yaml.tokens import AliasToken, AnchorToken

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / ".github" / "release"))
PACKAGE_DIR = REPO_ROOT / ".github" / "release" / "ucm_release"
RELEASE_ROOT = REPO_ROOT / ".github" / "release"
LEGACY_RELEASE_ROOTS = (
    REPO_ROOT / "release",
    REPO_ROOT / "scripts" / "release",
    REPO_ROOT / "docker" / "release",
)
LEGACY_POLICY_FILES = (
    RELEASE_ROOT / "builders.yaml",
    RELEASE_ROOT / "native-contract.yaml",
    RELEASE_ROOT / "toolchain.lock.yaml",
)


def _source_files(path: Path, *, excluded_parts: set[str] | None = None) -> list[Path]:
    if not path.exists():
        return []
    exclusions = excluded_parts or set()
    return [
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file()
        and not exclusions.intersection(candidate.relative_to(path).parts)
    ]


def _forbidden_release_content_paths(repo_root: Path) -> tuple[list[str], list[str]]:
    """Scan both old and new implementation roots, never the contract tests."""
    modern_root = repo_root / ".github" / "release"
    roots = (
        (modern_root, {"tests", "__pycache__"}),
        (repo_root / "release", {"__pycache__"}),
        (repo_root / "scripts" / "release", {"__pycache__"}),
        (repo_root / "docker" / "release", {"__pycache__"}),
        (repo_root / ".github" / "workflows", {"__pycache__"}),
    )
    opt_references: list[str] = []
    standalone_wrapt_paths: list[str] = []
    for source, excluded_parts in roots:
        for path in _source_files(source, excluded_parts=excluded_parts):
            relative = path.relative_to(repo_root).as_posix()
            content = path.read_text(encoding="utf-8", errors="ignore")
            if "/opt/ucm-release" in content:
                opt_references.append(relative)
            if (
                "wrapt" in path.name.lower()
                or "wrapt-bundle" in content
                or "wrapt_bundle" in content
            ):
                standalone_wrapt_paths.append(relative)
    return sorted(opt_references), sorted(standalone_wrapt_paths)


def test_release_tree_rejects_legacy_release_artifacts() -> None:
    """Retain the safety invariants without constraining future growth."""
    violations: list[str] = []

    if not PACKAGE_DIR.is_dir():
        violations.append("missing .github/release/ucm_release package")

    present_legacy_roots = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in LEGACY_RELEASE_ROOTS
        if path.exists()
    ]
    if present_legacy_roots:
        violations.append(f"legacy release roots remain: {present_legacy_roots}")

    present_legacy_policy_files = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in LEGACY_POLICY_FILES
        if path.exists()
    ]
    if present_legacy_policy_files:
        violations.append(
            f"legacy release policy files remain: {present_legacy_policy_files}"
        )

    opt_release_references, standalone_wrapt_paths = _forbidden_release_content_paths(
        REPO_ROOT
    )
    if opt_release_references:
        violations.append(
            "/opt/ucm-release is forbidden; references remain in "
            f"{sorted(opt_release_references)}"
        )

    if standalone_wrapt_paths:
        violations.append(
            "standalone wrapt release bundle remains in "
            f"{sorted(standalone_wrapt_paths)}"
        )

    failure_message = "release slimming structural contract failed:\n- " + "\n- ".join(
        violations
    )
    assert not violations, failure_message


def test_forbidden_content_scan_covers_the_new_release_tree(tmp_path: Path) -> None:
    """A post-deletion implementation cannot bypass the old-root-only scan."""
    modern_package = tmp_path / ".github" / "release" / "ucm_release"
    modern_package.mkdir(parents=True)
    (modern_package / "runner.py").write_text("tool = '/opt/ucm-release/run'\n")
    (modern_package / "wrapt_bundle.py").write_text("pass\n")
    (tmp_path / ".github" / "release" / "tests").mkdir()
    (tmp_path / ".github" / "release" / "tests" / "test_contract.py").write_text(
        "example = '/opt/ucm-release is only test text'\n"
    )

    opt_paths, wrapt_paths = _forbidden_release_content_paths(tmp_path)

    assert opt_paths == [".github/release/ucm_release/runner.py"]
    assert wrapt_paths == [".github/release/ucm_release/wrapt_bundle.py"]


def _release_policy() -> dict[str, object]:
    return yaml.safe_load((RELEASE_ROOT / "release.yaml").read_text(encoding="utf-8"))


def _platform_policy() -> dict[str, object]:
    return yaml.safe_load((RELEASE_ROOT / "platforms.yaml").read_text(encoding="utf-8"))


def test_release_policy_matches_the_schema_v5_registry_surface() -> None:
    release = _release_policy()

    assert release == {
        "kind": "ucm-release-policy",
        "schema_version": 5,
        "runners": {"amd64": "ubuntu-24.04", "arm64": "ubuntu-24.04-arm"},
        "products": [
            {
                "id": "vllm",
                "runtime_repository": "docker.io/vllm/vllm-openai",
                "target_repository": "ghcr.io/{owner}/vllm-openai",
                "minimum_version": "0.23.0",
                "channel_policy": "latest-stable-or-rc-or-nightly-per-minor",
            },
            {
                "id": "vllm-ascend",
                "runtime_repository": "quay.io/ascend/vllm-ascend",
                "target_repository": "ghcr.io/{owner}/vllm-ascend",
                "minimum_version": "0.23.0",
                "channel_policy": "latest-stable-or-rc-or-nightly-per-minor",
            },
        ],
        "publish": {
            "pypi": {"index": "https://upload.pypi.org/legacy/"},
            "ghcr": {"namespace": "ghcr.io/{owner}"},
            "dockerhub": {"namespace": "docker.io/{owner}"},
            "chart_oci": {"namespace": "ghcr.io/{owner}/charts"},
            "github_release": {},
        },
        "release_profiles": {
            "stable": {
                "max_count": -1,
                "max_minor_versions": -1,
                "publish": {
                    "pypi": False,
                    "ghcr": True,
                    "dockerhub": False,
                    "chart_oci": True,
                    "github_release": True,
                },
            },
            "prerelease": {
                "max_count": -1,
                "max_minor_versions": -1,
                "publish": {
                    "pypi": False,
                    "ghcr": True,
                    "dockerhub": False,
                    "chart_oci": True,
                    "github_release": True,
                },
            },
            "draft": {
                "max_count": 7,
                "max_minor_versions": -1,
                "publish": {
                    "pypi": False,
                    "ghcr": True,
                    "dockerhub": False,
                    "chart_oci": True,
                    "github_release": True,
                },
            },
            "nightly": {
                "max_count": 7,
                "max_minor_versions": 1,
                "publish": {
                    "pypi": False,
                    "ghcr": True,
                    "dockerhub": False,
                    "chart_oci": True,
                    "github_release": True,
                },
            },
        },
        "chart": {
            "source": "charts/unified-cache-chart",
            "smoke_values": {
                "vllm": "charts/unified-cache-chart/models/cuda/values-qwen3-0p6b-1e1.yaml",
                "vllm-ascend": "charts/unified-cache-chart/models/ascend/values-qwen3-0p6b-1e1.yaml",
            },
        },
    }


def test_release_yaml_has_expanded_commented_profiles_without_aliases() -> None:
    text = (RELEASE_ROOT / "release.yaml").read_text(encoding="utf-8")

    assert "# 四类发布配置全部展开" in text
    assert "# -1 表示无限保留 Stable 发布。" in text
    assert "# 最多保留 7 个新版 Draft。" in text
    assert "# 最多保留 7 个 Nightly。" in text
    assert "# 只构建 minimum_version 起第一个实际存在的 minor。" in text
    assert not any(
        isinstance(token, (AnchorToken, AliasToken)) for token in yaml.scan(text)
    )


@pytest.mark.parametrize("field", ["max_count", "max_minor_versions"])
@pytest.mark.parametrize("value", [-2, -1.0, 0, True, 1.5, "-1", "1", None])
def test_release_profile_limits_reject_values_other_than_minus_one_or_positive_int(
    tmp_path: Path, field: str, value: object
) -> None:
    policy = importlib.import_module("ucm_release.policy")
    release = _release_policy()
    release["release_profiles"]["nightly"][field] = value
    path = tmp_path / "release.yaml"
    path.write_text(yaml.safe_dump(release, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError):
        policy.load(path)


@pytest.mark.parametrize("value", [-1, 1, 7])
def test_release_profile_limits_accept_minus_one_and_positive_int(
    tmp_path: Path, value: int
) -> None:
    policy = importlib.import_module("ucm_release.policy")
    release = _release_policy()
    release["release_profiles"]["nightly"]["max_minor_versions"] = value
    path = tmp_path / "release.yaml"
    path.write_text(yaml.safe_dump(release, sort_keys=False), encoding="utf-8")

    loaded = policy.load(path)

    assert (
        loaded["release"]["release_profiles"]["nightly"]["max_minor_versions"] == value
    )


def test_finite_profile_with_pypi_enabled_is_valid(tmp_path: Path) -> None:
    policy = importlib.import_module("ucm_release.policy")
    release = _release_policy()
    release["release_profiles"]["draft"]["publish"]["pypi"] = True
    path = tmp_path / "release.yaml"
    path.write_text(yaml.safe_dump(release, sort_keys=False), encoding="utf-8")

    loaded = policy.load(path)

    assert loaded["release"]["release_profiles"]["draft"]["max_count"] == 7
    assert loaded["release"]["release_profiles"]["draft"]["publish"]["pypi"]


def test_platform_policy_matches_supported_and_blocked_backends() -> None:
    assert _platform_policy() == {
        "kind": "ucm-platform-policy",
        "schema_version": 1,
        "excluded_upstream_variants": {"vllm-ascend": ["310p"]},
        "builder_families": {
            "cuda": {
                "target_repository": "ghcr.io/{owner}/ucm-builder-vllm",
                "source_repositories": {
                    "amd64": "docker.io/pytorch/manylinux2_28-builder",
                    "arm64": "docker.io/pytorch/manylinuxaarch64-builder",
                },
                "manylinux": "manylinux_2_28",
                "required_commands": ["gcc", "g++", "make", "git", "nvcc"],
            },
            "ascend": {
                "target_repository": "ghcr.io/{owner}/ucm-builder-vllm-ascend",
                "source_repositories": {
                    "amd64": "quay.io/ascend/manylinux",
                    "arm64": "quay.io/ascend/manylinux",
                },
                "required_commands": ["gcc", "g++", "make", "git", "cmake"],
                "required_files": ["acl.h"],
                "variant_required_files": {"a3": ["libruntime.so"]},
            },
        },
        "backends": {
            "cuda": {
                "status": "supported",
                "platform": "cuda",
                "distribution_template": "uc-manager-cuda-{runtime_variant}",
            },
            "cann-a2": {
                "status": "supported",
                "platform": "ascend",
                "distribution_template": "uc-manager-{runtime_variant}",
            },
            "cann-a3": {
                "status": "supported",
                "platform": "ascend-a3",
                "distribution_template": "uc-manager-{runtime_variant}",
            },
            "cann-a5": {
                "status": "blocked",
                "platform": "ascend-a5",
                "distribution_template": "uc-manager-{runtime_variant}",
                "reason": "A5 requires a dedicated UCM native implementation",
            },
        },
    }


def test_formal_policy_loads_exact_requirements_without_legacy_authorities() -> None:
    policy = importlib.import_module("ucm_release.policy")

    loaded = policy.load()

    assert set(loaded) == {"release", "platforms", "requirements"}
    assert loaded["requirements"] == {
        "wheel_build": [
            "build==1.3.0",
            "cmake==3.31.6",
            "packaging==24.2",
            "pyproject-hooks==1.2.0",
            "pyyaml==6.0.2",
            "setuptools==75.8.2",
            "wheel==0.45.1",
        ],
        "wheel_runtime": ["wrapt==1.17.2"],
    }
    serialized = repr(loaded)
    for legacy_key in (
        "python_build_lock",
        "python_runtime_dependencies",
        "required_native",
        "forbidden_native",
        "allowed_dt_needed",
    ):
        assert legacy_key not in serialized


@pytest.mark.parametrize(
    ("release_type", "max_count", "max_minor_versions"),
    [
        ("stable", -1, -1),
        ("prerelease", -1, -1),
        ("draft", 7, -1),
        ("nightly", 7, 1),
    ],
)
def test_policy_resolve_selects_one_profile_and_normalizes_publication(
    release_type: str, max_count: int, max_minor_versions: int
) -> None:
    policy = importlib.import_module("ucm_release.policy")

    resolved = policy.resolve(repository="release-org/ucm", release_type=release_type)

    assert resolved["release_type"] == release_type
    assert resolved["publication_scope"] == "fork"
    assert resolved["runtime_image_tag_prefix"] == "release-org-"
    assert resolved["release_profile"] == {
        "max_count": max_count,
        "max_minor_versions": max_minor_versions,
        "publish": {
            "pypi": False,
            "ghcr": True,
            "dockerhub": False,
            "chart_oci": True,
            "github_release": True,
        },
    }
    assert resolved["publish"] == {
        "pypi": {
            "index": "https://upload.pypi.org/legacy/",
            "enabled": False,
        },
        "ghcr": {"namespace": "ghcr.io/release-org", "enabled": True},
        "dockerhub": {
            "namespace": "docker.io/release-org",
            "enabled": False,
        },
        "chart_oci": {
            "namespace": "ghcr.io/release-org/charts",
            "enabled": True,
        },
        "github_release": {"enabled": True},
    }


def test_policy_resolve_uses_only_the_selected_profile_switches(
    tmp_path: Path,
) -> None:
    policy = importlib.import_module("ucm_release.policy")
    release = _release_policy()
    release["release_profiles"]["stable"]["publish"]["ghcr"] = False
    release["release_profiles"]["draft"]["publish"]["pypi"] = True
    path = tmp_path / "release.yaml"
    path.write_text(yaml.safe_dump(release, sort_keys=False), encoding="utf-8")

    stable = policy.resolve(
        path, repository=policy.OFFICIAL_REPOSITORY, release_type="stable"
    )
    draft = policy.resolve(
        path, repository=policy.OFFICIAL_REPOSITORY, release_type="draft"
    )

    assert stable["publish"]["ghcr"]["enabled"] is False
    assert stable["publish"]["pypi"]["enabled"] is False
    assert draft["publish"]["ghcr"]["enabled"] is True
    assert draft["publish"]["pypi"]["enabled"] is True
    assert draft["publication_scope"] == "official"
    assert draft["runtime_image_tag_prefix"] == ""


def test_fork_scope_disables_shared_external_channels_only(tmp_path: Path) -> None:
    policy = importlib.import_module("ucm_release.policy")
    release = _release_policy()
    release["release_profiles"]["stable"]["publish"].update(
        {"pypi": True, "dockerhub": True}
    )
    path = tmp_path / "release.yaml"
    path.write_text(yaml.safe_dump(release, sort_keys=False), encoding="utf-8")

    official = policy.resolve(
        path, repository=policy.OFFICIAL_REPOSITORY, release_type="stable"
    )
    fork = policy.resolve(
        path,
        repository="SuperMarioYL/unified-cache-management",
        release_type="stable",
    )

    assert official["publish"]["pypi"]["enabled"] is True
    assert official["publish"]["dockerhub"]["enabled"] is True
    assert fork["publication_scope"] == "fork"
    assert fork["runtime_image_tag_prefix"] == "supermarioyl-"
    assert fork["release_profile"]["publish"]["pypi"] is False
    assert fork["release_profile"]["publish"]["dockerhub"] is False
    assert fork["publish"]["pypi"]["enabled"] is False
    assert fork["publish"]["dockerhub"]["enabled"] is False
    assert fork["publish"]["ghcr"]["enabled"] is True
    assert fork["publish"]["chart_oci"]["enabled"] is True
    assert fork["publish"]["github_release"]["enabled"] is True


def test_official_repository_identity_is_case_insensitive() -> None:
    policy = importlib.import_module("ucm_release.policy")

    assert policy.publication_identity(
        "modelengine-group/UNIFIED-CACHE-MANAGEMENT"
    ) == ("official", "")


def test_builder_discovery_defaults_to_the_platform_policy() -> None:
    cli = importlib.import_module("ucm_release.cli")
    policy = importlib.import_module("ucm_release.policy")

    args = cli.build_parser().parse_args(
        [
            "builders",
            "discover",
            "--selection",
            "selection.json",
            "--output",
            "catalog.json",
        ]
    )

    assert args.config == policy.DEFAULT_PLATFORMS


@pytest.mark.parametrize(
    ("arguments", "release_type"),
    [
        (
            [
                "upstreams",
                "candidates",
                "--release-type",
                "nightly",
                "--output",
                "candidates.json",
            ],
            "nightly",
        ),
        (
            [
                "upstreams",
                "resolve",
                "--release-type",
                "draft",
                "--candidates",
                "candidates.json",
                "--runtime-probe",
                "probe.json",
                "--output",
                "selection.json",
            ],
            "draft",
        ),
        (
            [
                "compact",
                "plan",
                "--release-type",
                "prerelease",
                "--builder-catalog",
                "builders.json",
                "--runtime-selection",
                "selection.json",
                "--route",
                "release",
                "--output",
                "plan.json",
            ],
            "prerelease",
        ),
    ],
)
def test_release_commands_accept_explicit_release_type(
    arguments: list[str], release_type: str
) -> None:
    cli = importlib.import_module("ucm_release.cli")

    args = cli.build_parser().parse_args(arguments)

    assert args.release_type == release_type


def test_core_projects_v5_for_release_consumers() -> None:
    core = importlib.import_module("ucm_release.core")

    catalog = core.load_catalog(version_override="0.7.59rc7")

    assert catalog["kind"] == "release-config"
    assert catalog["schema_version"] == 3
    assert catalog["runner_map"] == _release_policy()["runners"]
    assert {
        product["target_tag_suffix"] for product in catalog["upstream_products"]
    } == {"-ucm-0.7.59rc7"}
    assert catalog["wheel_build_requirements"] == [
        "build==1.3.0",
        "cmake==3.31.6",
        "packaging==24.2",
        "pyproject-hooks==1.2.0",
        "pyyaml==6.0.2",
        "setuptools==75.8.2",
        "wheel==0.45.1",
    ]
    assert core.python_runtime_requirements(catalog) == ["wrapt==1.17.2"]
    assert [
        record["requirement"]
        for record in core.build_tool_dependency_records(catalog, "cp312", "amd64")
    ] == catalog["wheel_build_requirements"]
    assert "python_build_lock" not in catalog
    assert "python_runtime_dependencies" not in catalog
    assert catalog["backend_contracts"]["cann-a5"]["status"] == "blocked"
    assert all(
        not contract["required_native"]
        and not contract["forbidden_native"]
        and not contract["allowed_dt_needed"]
        for contract in catalog["backend_contracts"].values()
    )


def test_release_yaml_is_the_exact_publication_authority() -> None:
    release = _release_policy()

    assert release["publish"] == {
        "pypi": {"index": "https://upload.pypi.org/legacy/"},
        "ghcr": {"namespace": "ghcr.io/{owner}"},
        "dockerhub": {"namespace": "docker.io/{owner}"},
        "chart_oci": {"namespace": "ghcr.io/{owner}/charts"},
        "github_release": {},
    }
    assert release["release_profiles"] == {
        "stable": {
            "max_count": -1,
            "max_minor_versions": -1,
            "publish": {
                "pypi": False,
                "ghcr": True,
                "dockerhub": False,
                "chart_oci": True,
                "github_release": True,
            },
        },
        "prerelease": {
            "max_count": -1,
            "max_minor_versions": -1,
            "publish": {
                "pypi": False,
                "ghcr": True,
                "dockerhub": False,
                "chart_oci": True,
                "github_release": True,
            },
        },
        "draft": {
            "max_count": 7,
            "max_minor_versions": -1,
            "publish": {
                "pypi": False,
                "ghcr": True,
                "dockerhub": False,
                "chart_oci": True,
                "github_release": True,
            },
        },
        "nightly": {
            "max_count": 7,
            "max_minor_versions": 1,
            "publish": {
                "pypi": False,
                "ghcr": True,
                "dockerhub": False,
                "chart_oci": True,
                "github_release": True,
            },
        },
    }
    assert set(release) == {
        "kind",
        "schema_version",
        "runners",
        "products",
        "publish",
        "release_profiles",
        "chart",
    }
    assert set(release["chart"]) == {"source", "smoke_values"}


def test_publish_plan_is_the_normalized_config_without_runtime_layers() -> None:
    core = importlib.import_module("ucm_release.core")

    plan = core.compute_publish_plan(core.load_catalog())

    assert plan == {
        "pypi": {
            "enabled": False,
            "index": "https://upload.pypi.org/legacy/",
        },
        "ghcr": {
            "enabled": True,
            "namespace": "ghcr.io/release-org",
        },
        "dockerhub": {
            "enabled": False,
            "namespace": "docker.io/release-org",
        },
        "chart_oci": {
            "enabled": True,
            "namespace": "ghcr.io/release-org/charts",
        },
        "github_release": {"enabled": True},
    }


@pytest.mark.parametrize("enabled", [False, True])
def test_ghcr_publication_accepts_both_boolean_modes(
    tmp_path: Path, enabled: bool
) -> None:
    core = importlib.import_module("ucm_release.core")
    release = _release_policy()
    release["release_profiles"]["stable"]["publish"]["ghcr"] = enabled
    path = tmp_path / "release.yaml"
    path.write_text(yaml.safe_dump(release, sort_keys=False), encoding="utf-8")

    catalog = core.load_catalog(path, repository="release-org/ucm")
    plan = core.compute_publish_plan(catalog)

    assert plan["ghcr"]["enabled"] is enabled


@pytest.mark.parametrize("release_type", ["stable", "prerelease", "draft", "nightly"])
def test_each_profile_requires_github_release_draft_barrier(
    tmp_path: Path, release_type: str
) -> None:
    core = importlib.import_module("ucm_release.core")
    release = yaml.safe_load(
        (REPO_ROOT / ".github" / "release" / "release.yaml").read_text(encoding="utf-8")
    )
    release["release_profiles"][release_type]["publish"]["github_release"] = False
    path = tmp_path / "release.yaml"
    path.write_text(yaml.safe_dump(release, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="Draft barrier"):
        core.load_catalog(path)


@pytest.mark.parametrize("release_type", ["stable", "prerelease", "draft", "nightly"])
def test_each_profile_dockerhub_requires_ghcr_source_channel(
    tmp_path: Path, release_type: str
) -> None:
    core = importlib.import_module("ucm_release.core")
    release = yaml.safe_load(
        (REPO_ROOT / ".github" / "release" / "release.yaml").read_text(encoding="utf-8")
    )
    release["release_profiles"][release_type]["publish"]["dockerhub"] = True
    release["release_profiles"][release_type]["publish"]["ghcr"] = False
    path = tmp_path / "release.yaml"
    path.write_text(yaml.safe_dump(release, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="Docker Hub publication requires GHCR"):
        core.load_catalog(path)


@pytest.mark.parametrize(
    ("channel", "extra"),
    [
        ("pypi", {"namespace": "invalid"}),
        ("ghcr", {"index": "invalid"}),
        ("dockerhub", {"dists": ["invalid"]}),
        ("chart_oci", {"index": "invalid"}),
        ("github_release", {"namespace": "invalid"}),
    ],
)
def test_publish_channel_shapes_are_exact(
    tmp_path: Path, channel: str, extra: dict[str, object]
) -> None:
    core = importlib.import_module("ucm_release.core")
    release = yaml.safe_load(
        (REPO_ROOT / ".github" / "release" / "release.yaml").read_text(encoding="utf-8")
    )
    release["publish"][channel].update(extra)
    path = tmp_path / "release.yaml"
    path.write_text(yaml.safe_dump(release, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="Additional properties are not allowed"):
        core.load_catalog(path)


def test_catalog_validate_cli_uses_v5_policy_contract(capsys) -> None:
    cli = importlib.import_module("ucm_release.cli")

    assert cli.main(["catalog", "validate"]) == 0

    output = capsys.readouterr().out
    assert '"products":2' in output
    assert '"backends":4' in output


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("0.7.62", "0.7.62"),
        ("0.7.62rc1", "0.7.62-rc.1"),
        ("0.7.62.dev0", "0.7.62-draft.0"),
        ("0.7.62.dev4", "0.7.62-draft.4"),
    ],
)
def test_chart_version_preserves_formal_and_draft_coordinates(
    version: str, expected: str
) -> None:
    core = importlib.import_module("ucm_release.core")

    assert core.derive_chart_version(version) == expected
