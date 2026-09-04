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


def test_release_policy_matches_the_schema_v6_registry_surface() -> None:
    policy = importlib.import_module("ucm_release.policy")

    release = policy.load()["release"]

    assert release["kind"] == "ucm-release-policy"
    assert release["schema_version"] == 6
    assert set(release) == {
        "kind",
        "schema_version",
        "runners",
        "products",
        "publish",
        "release_profiles",
        "chart",
    }
    assert set(release["publish"]) == set(policy.PUBLISH_CHANNELS)
    assert release["publish"]["dockerhub"] == {}
    assert set(release["release_profiles"]) == set(policy.RELEASE_TYPES)
    for profile in release["release_profiles"].values():
        assert set(profile) == {"max_count", "publish"}
        assert set(profile["publish"]) == set(policy.PUBLISH_CHANNELS)
        assert all(isinstance(value, bool) for value in profile["publish"].values())


def test_release_yaml_has_expanded_commented_profiles_without_aliases() -> None:
    text = (RELEASE_ROOT / "release.yaml").read_text(encoding="utf-8")

    assert "# 四类发布配置全部展开" in text
    assert "# -1 表示无限保留 Stable 发布。" in text
    assert "# 最多保留 7 个新版 Draft。" in text
    assert "# 最多保留 7 个 Nightly。" in text
    assert "recent_minor_versions" not in text
    assert "max_minor_versions" not in text
    assert not any(
        isinstance(token, (AnchorToken, AliasToken)) for token in yaml.scan(text)
    )


def test_readme_documents_official_and_fork_publication_setup() -> None:
    text = (RELEASE_ROOT / "README.md").read_text(encoding="utf-8")

    for value in (
        "## Official release setup",
        "release-production",
        "PYPI_API_TOKEN",
        "gh secret set PYPI_API_TOKEN --repo",
        "## Fork preview setup",
        "fork-preview",
        "TEST_PYPI_API_TOKEN",
        "DOCKERHUB_USERNAME",
        "DOCKERHUB_TOKEN",
        "DOCKERHUB_NAMESPACE",
        "gh secret set TEST_PYPI_API_TOKEN --repo",
        "gh variable set DOCKERHUB_NAMESPACE --repo",
        "Read and write permissions",
        "https://test.pypi.org/simple/",
        "scope-skipped",
        "selected Release Profile",
        "git push origin",
    ):
        assert value in text
    assert "FORK_DOCKERHUB_NAMESPACE" not in text
    assert "copy it into the Fork or `fork-preview`" in text
    assert "not the empty meta Wheel" in text


@pytest.mark.parametrize(
    ("location", "field", "value"),
    [
        ("product", "minimum_version", "0.24.0"),
        ("product", "maximum_version", "0.27.0"),
        ("product", "recent_minor_versions", 3),
        ("product", "channel_policy", "latest-stable-or-rc-or-nightly-per-minor"),
        ("profile", "max_minor_versions", 1),
    ],
)
def test_release_yaml_rejects_runtime_version_selection_fields(
    tmp_path: Path, location: str, field: str, value: object
) -> None:
    policy = importlib.import_module("ucm_release.policy")
    release = _release_policy()
    target = (
        release["products"][0]
        if location == "product"
        else release["release_profiles"]["nightly"]
    )
    target[field] = value
    path = tmp_path / "release.yaml"
    path.write_text(yaml.safe_dump(release, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="Additional properties"):
        policy.load(path)


@pytest.mark.parametrize("value", [-2, -1.0, 0, True, 1.5, "-1", "1", None])
def test_release_profile_limits_reject_invalid_values(
    tmp_path: Path, value: object
) -> None:
    policy = importlib.import_module("ucm_release.policy")
    release = _release_policy()
    release["release_profiles"]["nightly"]["max_count"] = value
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
    release["release_profiles"]["nightly"]["max_count"] = value
    path = tmp_path / "release.yaml"
    path.write_text(yaml.safe_dump(release, sort_keys=False), encoding="utf-8")

    loaded = policy.load(path)

    assert loaded["release"]["release_profiles"]["nightly"]["max_count"] == value


@pytest.mark.parametrize("enabled", [False, True])
def test_release_profile_publication_switches_are_policy_driven(
    tmp_path: Path, enabled: bool
) -> None:
    policy = importlib.import_module("ucm_release.policy")
    release = _release_policy()
    switches = {channel: enabled for channel in policy.PUBLISH_CHANNELS}
    release["release_profiles"]["draft"]["publish"] = switches
    path = tmp_path / "release.yaml"
    path.write_text(yaml.safe_dump(release, sort_keys=False), encoding="utf-8")

    loaded = policy.load(path)

    assert loaded["release"]["release_profiles"]["draft"]["publish"] == switches


def test_version_ini_is_the_runtime_selector_authority() -> None:
    policy = importlib.import_module("ucm_release.policy")

    resolved = policy.resolve(repository=policy.OFFICIAL_REPOSITORY)

    assert {
        product_id: [selector["raw"] for selector in selectors]
        for product_id, selectors in resolved["runtime_selectors"].items()
    } == {
        "vllm": ["0.26", "0.27", "0.28"],
        "vllm-ascend": ["0.23", "0.24", "0.25", "0.26"],
    }
    assert {
        product["id"]: product["runtime_selectors"] for product in resolved["products"]
    } == resolved["runtime_selectors"]
    assert resolved["ucm_base_version"] == "0.7.0"
    assert resolved["version_authority_sha256"].startswith("sha256:")


def test_platform_policy_matches_supported_and_blocked_backends() -> None:
    assert _platform_policy() == {
        "kind": "ucm-platform-policy",
        "schema_version": 3,
        "excluded_upstream_variants": {"vllm-ascend": ["310p"]},
        "builder_families": {
            "cuda": {
                "target_repository": "ghcr.io/{owner}/ucm-builder-vllm",
                "source_repositories": {
                    "amd64": "docker.io/pytorch/manylinux2_28-builder",
                    "arm64": "docker.io/pytorch/manylinuxaarch64-builder",
                },
                "manylinux": "manylinux_2_28",
                "required_commands": [
                    "gcc",
                    "g++",
                    "make",
                    "git",
                    "nvcc",
                    "patchelf",
                ],
            },
            "ascend": {
                "target_repository": "ghcr.io/{owner}/ucm-builder-vllm-ascend",
                "source_repositories": {
                    "amd64": "quay.io/ascend/manylinux",
                    "arm64": "quay.io/ascend/manylinux",
                },
                "manylinux": "manylinux_2_34",
                "required_commands": [
                    "gcc",
                    "g++",
                    "make",
                    "git",
                    "cmake",
                    "patchelf",
                ],
                "required_files": ["acl.h"],
                "variant_required_files": {"a3": ["libruntime.so"]},
            },
        },
        "backends": {
            "cuda": {
                "status": "supported",
                "platform": "cuda",
                "distribution_template": "uc-manager-cuda-{runtime_variant}",
                "external_runtime_exclude_patterns": [
                    "libcudart.so.{accelerator_major}"
                ],
            },
            "cann-a2": {
                "status": "supported",
                "platform": "ascend",
                "distribution_template": "uc-manager-{runtime_variant}",
                "external_runtime_exclude_patterns": ["/usr/local/Ascend/*"],
            },
            "cann-a3": {
                "status": "supported",
                "platform": "ascend-a3",
                "distribution_template": "uc-manager-{runtime_variant}",
                "external_runtime_exclude_patterns": ["/usr/local/Ascend/*"],
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
            "auditwheel==6.7.0",
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


@pytest.mark.parametrize("release_type", ("stable", "prerelease", "draft", "nightly"))
def test_policy_resolve_selects_one_profile_and_normalizes_publication(
    release_type: str,
) -> None:
    policy = importlib.import_module("ucm_release.policy")
    source = _release_policy()
    selected_profile = source["release_profiles"][release_type]

    resolved = policy.resolve(repository="release-org/ucm", release_type=release_type)

    assert resolved["release_type"] == release_type
    assert resolved["publication_scope"] == "fork"
    assert resolved["runtime_image_tag_prefix"] == "release-org-"
    assert resolved["release_profile"] == selected_profile
    for channel, requested in selected_profile["publish"].items():
        publication = resolved["publish"][channel]
        assert publication["requested"] is requested
        if channel in {"pypi", "dockerhub"}:
            assert publication["enabled"] is False
            assert publication["disposition"] == (
                "scope-skipped" if requested else "disabled"
            )
        else:
            assert publication["enabled"] is requested
            assert publication["disposition"] == (
                "publish" if requested else "disabled"
            )
    assert resolved["publish"]["pypi"]["target"] == "testpypi"
    assert resolved["publish"]["pypi"]["distribution_prefix"] == "release-org-"
    for channel in ("ghcr", "chart_oci"):
        expected_namespace = (
            source["publish"][channel]["namespace"]
            .replace("{owner}", "release-org")
            .replace("{repo}", "ucm")
        )
        assert resolved["publish"][channel]["namespace"] == expected_namespace
    assert "namespace" not in resolved["publish"]["dockerhub"]


def test_policy_resolve_uses_only_the_selected_profile_switches(
    tmp_path: Path,
) -> None:
    policy = importlib.import_module("ucm_release.policy")
    release = _release_policy()
    # Synthetic inputs exercise profile isolation; they are not repository defaults.
    stable_switches = {
        "pypi": True,
        "ghcr": False,
        "dockerhub": False,
        "chart_oci": False,
        "github_release": True,
    }
    draft_switches = {
        "pypi": False,
        "ghcr": True,
        "dockerhub": True,
        "chart_oci": True,
        "github_release": True,
    }
    release["release_profiles"]["stable"]["publish"] = stable_switches
    release["release_profiles"]["draft"]["publish"] = draft_switches
    path = tmp_path / "release.yaml"
    path.write_text(yaml.safe_dump(release, sort_keys=False), encoding="utf-8")

    stable = policy.resolve(
        path, repository=policy.OFFICIAL_REPOSITORY, release_type="stable"
    )
    draft = policy.resolve(
        path,
        repository=policy.OFFICIAL_REPOSITORY,
        release_type="draft",
        dockerhub_namespace="docker.io/ucm-debug",
    )

    assert stable["release_profile"]["publish"] == stable_switches
    assert draft["release_profile"]["publish"] == draft_switches
    assert {
        channel: publication["requested"]
        for channel, publication in stable["publish"].items()
    } == stable_switches
    assert {
        channel: publication["requested"]
        for channel, publication in draft["publish"].items()
    } == draft_switches
    assert {
        channel: publication["enabled"]
        for channel, publication in stable["publish"].items()
    } == stable_switches
    assert {
        channel: publication["enabled"]
        for channel, publication in draft["publish"].items()
    } == draft_switches
    assert draft["publication_scope"] == "official"
    assert draft["runtime_image_tag_prefix"] == ""


def test_fork_scope_only_skips_requested_pypi(tmp_path: Path) -> None:
    policy = importlib.import_module("ucm_release.policy")
    release = _release_policy()
    # Give both external channels an explicit request before testing Fork scope.
    switches = {channel: True for channel in policy.PUBLISH_CHANNELS}
    release["release_profiles"]["draft"]["publish"] = switches
    path = tmp_path / "release.yaml"
    path.write_text(yaml.safe_dump(release, sort_keys=False), encoding="utf-8")

    official = policy.resolve(
        path,
        repository=policy.OFFICIAL_REPOSITORY,
        release_type="draft",
        dockerhub_namespace="docker.io/ucm-debug",
    )
    fork = policy.resolve(
        path,
        repository="SuperMarioYL/unified-cache-management",
        release_type="draft",
        dockerhub_namespace="docker.io/ucm-debug",
    )

    assert (
        official["publish"]["pypi"]["requested"],
        official["publish"]["pypi"]["enabled"],
        official["publish"]["pypi"]["disposition"],
    ) == (True, True, "publish")
    assert (
        fork["publish"]["pypi"]["requested"],
        fork["publish"]["pypi"]["enabled"],
        fork["publish"]["pypi"]["disposition"],
    ) == (True, False, "scope-skipped")
    assert (
        official["publish"]["dockerhub"]
        == fork["publish"]["dockerhub"]
        == {
            "namespace": "docker.io/ucm-debug",
            "requested": True,
            "enabled": True,
            "disposition": "publish",
        }
    )
    assert fork["publication_scope"] == "fork"
    assert fork["runtime_image_tag_prefix"] == "supermarioyl-"
    assert fork["release_profile"]["publish"] == switches
    assert fork["publish"]["pypi"]["distribution_prefix"] == "supermarioyl-"
    for channel in ("ghcr", "chart_oci", "github_release"):
        assert (
            fork["publish"][channel]["requested"],
            fork["publish"][channel]["enabled"],
            fork["publish"][channel]["disposition"],
        ) == (True, True, "publish")


def test_external_configuration_enables_only_profile_requested_channels(
    tmp_path: Path,
) -> None:
    policy = importlib.import_module("ucm_release.policy")
    release = _release_policy()
    # Opposite requests prove that credentials cannot override either channel.
    release["release_profiles"]["prerelease"]["publish"] = {
        "pypi": True,
        "ghcr": True,
        "dockerhub": False,
        "chart_oci": False,
        "github_release": True,
    }
    release["release_profiles"]["draft"]["publish"] = {
        "pypi": False,
        "ghcr": True,
        "dockerhub": True,
        "chart_oci": False,
        "github_release": True,
    }
    path = tmp_path / "release.yaml"
    path.write_text(yaml.safe_dump(release, sort_keys=False), encoding="utf-8")

    prerelease = policy.resolve(
        path,
        repository="SuperMarioYL/unified-cache-management",
        release_type="prerelease",
        fork_test_pypi=True,
        dockerhub_namespace="docker.io/ucm-debug",
    )
    draft = policy.resolve(
        path,
        repository="SuperMarioYL/unified-cache-management",
        release_type="draft",
        fork_test_pypi=True,
        dockerhub_namespace="docker.io/ucm-debug",
    )

    assert prerelease["publish"]["pypi"]["target"] == "testpypi"
    assert prerelease["publish"]["pypi"]["distribution_prefix"] == "supermarioyl-"
    assert (
        prerelease["publish"]["pypi"]["enabled"],
        prerelease["publish"]["pypi"]["disposition"],
    ) == (True, "publish")
    assert (
        prerelease["publish"]["dockerhub"]["enabled"],
        prerelease["publish"]["dockerhub"]["disposition"],
    ) == (False, "disabled")
    assert (
        draft["publish"]["pypi"]["enabled"],
        draft["publish"]["pypi"]["disposition"],
    ) == (False, "disabled")
    assert draft["publish"]["dockerhub"] == {
        "namespace": "docker.io/ucm-debug",
        "requested": True,
        "enabled": True,
        "disposition": "publish",
    }


@pytest.mark.parametrize(
    "repository",
    (
        "ModelEngine-Group/unified-cache-management",
        "SuperMarioYL/unified-cache-management",
    ),
)
def test_requested_dockerhub_namespace_is_scope_independent(
    tmp_path: Path, repository: str
) -> None:
    policy = importlib.import_module("ucm_release.policy")
    release = _release_policy()
    release["release_profiles"]["draft"]["publish"] = {
        "pypi": False,
        "ghcr": True,
        "dockerhub": True,
        "chart_oci": False,
        "github_release": True,
    }
    path = tmp_path / "release.yaml"
    path.write_text(yaml.safe_dump(release, sort_keys=False), encoding="utf-8")

    missing = policy.resolve(path, repository=repository, release_type="draft")
    configured = policy.resolve(
        path,
        repository=repository,
        release_type="draft",
        dockerhub_namespace="docker.io/ucm-debug",
    )

    assert missing["publish"]["dockerhub"] == {
        "requested": True,
        "enabled": False,
        "disposition": "scope-skipped",
    }
    assert configured["publish"]["dockerhub"] == {
        "namespace": "docker.io/ucm-debug",
        "requested": True,
        "enabled": True,
        "disposition": "publish",
    }

    with pytest.raises(ValueError, match="Docker Hub namespace"):
        policy.resolve(
            path,
            repository=repository,
            release_type="draft",
            dockerhub_namespace="ghcr.io/not-dockerhub",
        )


@pytest.mark.parametrize(
    "repository",
    (
        "ModelEngine-Group/unified-cache-management",
        "SuperMarioYL/unified-cache-management",
    ),
)
def test_disabled_dockerhub_ignores_runtime_configuration(
    tmp_path: Path, repository: str
) -> None:
    policy = importlib.import_module("ucm_release.policy")
    release = _release_policy()
    release["release_profiles"]["draft"]["publish"] = {
        channel: False for channel in policy.PUBLISH_CHANNELS
    }
    path = tmp_path / "release.yaml"
    path.write_text(yaml.safe_dump(release, sort_keys=False), encoding="utf-8")

    resolved = policy.resolve(
        path,
        repository=repository,
        release_type="draft",
        dockerhub_namespace="not-a-docker-namespace",
    )

    assert resolved["publish"]["dockerhub"] == {
        "requested": False,
        "enabled": False,
        "disposition": "disabled",
    }


def test_publication_context_uses_shared_dockerhub_namespace(tmp_path: Path) -> None:
    cli = importlib.import_module("ucm_release.cli")
    path = tmp_path / "publication-context.json"
    path.write_text(
        '{"fork_test_pypi":false,' '"dockerhub_namespace":"docker.io/ucm-debug"}',
        encoding="utf-8",
    )

    assert cli._publication_context(path) == {  # noqa: SLF001
        "fork_test_pypi": False,
        "dockerhub_namespace": "docker.io/ucm-debug",
    }


def test_official_repository_identity_is_case_insensitive() -> None:
    policy = importlib.import_module("ucm_release.policy")

    assert policy.publication_identity(
        "modelengine-group/UNIFIED-CACHE-MANAGEMENT"
    ) == ("official", "")
    assert (
        policy.pypi_distribution_prefix("modelengine-group/UNIFIED-CACHE-MANAGEMENT")
        == ""
    )


@pytest.mark.parametrize(
    "repository",
    [
        "owner/repo/extra",
        "foo_bar/repo",
        "foo.bar/repo",
        "foo--bar/repo",
        "foo-uc-manager-cuda/repo",
        "foo-uc-manager-cann901-a2/repo",
        "é/repo",
        "owner\nname/repo",
        "owner/repo\nname",
        f"{'a' * 40}/repo",
    ],
)
def test_python_distribution_prefix_rejects_lossy_or_invalid_owners(
    repository: str,
) -> None:
    policy = importlib.import_module("ucm_release.policy")

    with pytest.raises(ValueError):
        policy.pypi_distribution_prefix(repository)


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


def test_core_projects_v6_for_release_consumers() -> None:
    core = importlib.import_module("ucm_release.core")

    catalog = core.load_catalog(version_override="0.7.59rc7")

    assert catalog["kind"] == "release-config"
    assert catalog["schema_version"] == 4
    assert catalog["runner_map"] == _release_policy()["runners"]
    assert {
        product["target_tag_suffix"] for product in catalog["upstream_products"]
    } == {"-ucm-0.7.59rc7"}
    assert all(
        product["version_specifier"] == ">=0"
        and "minimum_version" not in product
        and "maximum_version" not in product
        for product in catalog["upstream_products"]
    )
    assert catalog["wheel_build_requirements"] == [
        "auditwheel==6.7.0",
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


def test_publish_plan_is_the_normalized_config_without_runtime_layers() -> None:
    core = importlib.import_module("ucm_release.core")
    catalog = core.load_catalog()
    expected = {
        channel: dict(publication)
        for channel, publication in catalog["publish"].items()
    }

    plan = core.compute_publish_plan(catalog)

    assert plan == expected
    assert plan is not catalog["publish"]
    assert all(plan[channel] is not catalog["publish"][channel] for channel in plan)


def test_publish_plan_requires_namespace_only_for_enabled_dockerhub() -> None:
    core = importlib.import_module("ucm_release.core")
    catalog = core.load_catalog()

    assert "namespace" not in catalog["publish"]["dockerhub"]
    catalog["publish"]["dockerhub"] = {
        "requested": True,
        "enabled": True,
        "disposition": "publish",
    }

    with pytest.raises(ValueError, match="Docker Hub namespace"):
        core.compute_publish_plan(catalog)

    catalog["publish"]["dockerhub"]["namespace"] = "docker.io/ucm-debug"

    assert core.compute_publish_plan(catalog)["dockerhub"]["namespace"] == (
        "docker.io/ucm-debug"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (("requested", False), ("enabled", True), ("disposition", "publish")),
)
def test_publish_plan_rejects_inconsistent_channel_decisions(
    field: str, value: object
) -> None:
    core = importlib.import_module("ucm_release.core")
    catalog = core.load_catalog()
    catalog["publish"]["pypi"].update(
        requested=True,
        enabled=False,
        disposition="scope-skipped",
    )
    catalog["publish"]["pypi"][field] = value

    with pytest.raises(ValueError, match="invalid publication decision"):
        core.compute_publish_plan(catalog)


@pytest.mark.parametrize("enabled", [False, True])
def test_ghcr_publication_accepts_both_boolean_modes(
    tmp_path: Path, enabled: bool
) -> None:
    core = importlib.import_module("ucm_release.core")
    release = _release_policy()
    release["release_profiles"]["stable"]["publish"] = {
        "pypi": False,
        "ghcr": enabled,
        "dockerhub": False,
        "chart_oci": False,
        "github_release": enabled,
    }
    path = tmp_path / "release.yaml"
    path.write_text(yaml.safe_dump(release, sort_keys=False), encoding="utf-8")

    catalog = core.load_catalog(path, repository="release-org/ucm")
    plan = core.compute_publish_plan(catalog)

    assert plan["ghcr"]["requested"] is enabled
    assert plan["ghcr"]["enabled"] is enabled


@pytest.mark.parametrize("release_type", ["stable", "prerelease", "draft", "nightly"])
def test_each_profile_requires_github_release_draft_barrier(
    tmp_path: Path, release_type: str
) -> None:
    core = importlib.import_module("ucm_release.core")
    release = yaml.safe_load(
        (REPO_ROOT / ".github" / "release" / "release.yaml").read_text(encoding="utf-8")
    )
    release["release_profiles"][release_type]["publish"] = {
        "pypi": True,
        "ghcr": False,
        "dockerhub": False,
        "chart_oci": False,
        "github_release": False,
    }
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
    release["release_profiles"][release_type]["publish"] = {
        "pypi": False,
        "ghcr": False,
        "dockerhub": True,
        "chart_oci": False,
        "github_release": True,
    }
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
