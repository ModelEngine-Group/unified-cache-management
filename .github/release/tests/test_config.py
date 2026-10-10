"""Validate current version, release, platform and publication policy."""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from email.parser import Parser
from pathlib import Path

import build
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / ".github" / "release"))
RELEASE_ROOT = REPO_ROOT / ".github" / "release"


def _release_policy() -> dict[str, object]:
    return yaml.safe_load((RELEASE_ROOT / "release.yaml").read_text(encoding="utf-8"))


def test_package_metadata_and_release_plan_read_the_same_dependencies(tmp_path):
    policy = importlib.import_module("ucm_release.policy")
    source = tmp_path / "source"
    source.mkdir()
    for filename in (
        "setup.py",
        "pyproject.toml",
        "version.ini",
        "README.md",
        "LICENSE",
    ):
        shutil.copy2(REPO_ROOT / filename, source / filename)
    project = source / "pyproject.toml"
    project.write_text(project.read_text().replace("wrapt==1.17.2", "wrapt==1.17.1"))
    bundle = policy.load(project_path=project)
    metadata_dir = build.ProjectBuilder(str(source)).prepare(
        "wheel", str(tmp_path / "metadata")
    )
    metadata = Parser().parsestr((Path(metadata_dir) / "METADATA").read_text())
    assert metadata.get_all("Requires-Dist") == bundle["requirements"]["wheel_runtime"]
    assert bundle["requirements"]["wheel_runtime"] == ["wrapt==1.17.1"]


def test_builder_pins_must_satisfy_the_package_build_requirements(tmp_path):
    policy = importlib.import_module("ucm_release.policy")
    project = tmp_path / "pyproject.toml"
    project.write_text(
        (REPO_ROOT / "pyproject.toml").read_text().replace("cmake>=3.18", "cmake>=99")
    )
    with pytest.raises(ValueError, match="Builder pin does not satisfy.*cmake>=99"):
        policy.load(project_path=project)


@pytest.mark.parametrize("value", [-2, 0, True])
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


@pytest.mark.parametrize(
    "repository",
    [
        "owner/repo/extra",
        "foo_bar/repo",
        "foo-uc-manager-cuda/repo",
        f"{'a' * 40}/repo",
    ],
)
def test_python_distribution_prefix_rejects_lossy_or_invalid_owners(
    repository: str,
) -> None:
    policy = importlib.import_module("ucm_release.policy")

    with pytest.raises(ValueError):
        policy.pypi_distribution_prefix(repository)


def test_importing_release_package_does_not_dispatch_cli():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.argv = ['-m']; import ucm_release; print('imported')",
        ],
        cwd=RELEASE_ROOT,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "imported"
