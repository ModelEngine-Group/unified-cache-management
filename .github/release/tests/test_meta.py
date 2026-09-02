"""Contract tests for the empty public PyPI meta package."""

from __future__ import annotations

import base64
import copy
import csv
import hashlib
import importlib
import importlib.metadata
import io
import os
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
RELEASE_ROOT = ROOT / ".github" / "release"
sys.path.insert(0, str(RELEASE_ROOT))

meta = importlib.import_module("ucm_release.meta")

VERSION = "0.9.1"


def _plan() -> dict[str, object]:
    return {
        "kind": "ucm-release-plan",
        "repository": "ModelEngine-Group/unified-cache-management",
        "publication_scope": "official",
        "version": VERSION,
        "publish": {"pypi": {"distribution_prefix": ""}},
        "meta_package": {
            "distribution": "uc-manager",
            "version": VERSION,
            "extras": {
                "cu130": f"uc-manager-cuda-cu130=={VERSION}",
                "cann901-a2": f"uc-manager-cann901-a2=={VERSION}",
            },
        },
        "wheels": [
            {
                "id": "cuda130-cp312-amd64",
                "dist_name": "uc-manager-cuda-cu130",
                "wheel_version": VERSION,
                "runtime_variant": "cu130",
            },
            {
                "id": "cuda130-cp312-arm64",
                "dist_name": "uc-manager-cuda-cu130",
                "wheel_version": VERSION,
                "runtime_variant": "cu130",
            },
            {
                "id": "cann901-a2-cp312-arm64",
                "dist_name": "uc-manager-cann901-a2",
                "wheel_version": VERSION,
                "runtime_variant": "cann901-a2",
            },
        ],
    }


def _fork_plan() -> dict[str, object]:
    plan = copy.deepcopy(_plan())
    plan["repository"] = "SuperMarioYL/unified-cache-management"
    plan["publication_scope"] = "fork"
    plan["publish"]["pypi"]["distribution_prefix"] = "supermarioyl-"
    for wheel in plan["wheels"]:
        wheel["dist_name"] = f"supermarioyl-{wheel['dist_name']}"
    meta_package = plan["meta_package"]
    meta_package["distribution"] = "supermarioyl-uc-manager"
    meta_package["extras"] = {
        extra: f"supermarioyl-{requirement}"
        for extra, requirement in meta_package["extras"].items()
    }
    return plan


def _record(members: dict[str, bytes], record_name: str) -> bytes:
    rows: list[list[str]] = []
    for name in sorted(members):
        content = members[name]
        digest = (
            base64.urlsafe_b64encode(hashlib.sha256(content).digest())
            .decode("ascii")
            .rstrip("=")
        )
        rows.append([name, f"sha256={digest}", str(len(content))])
    rows.append([record_name, "", ""])
    buffer = io.StringIO(newline="")
    csv.writer(buffer, lineterminator="\n").writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _write_wheel(
    path: Path,
    *,
    version: str = VERSION,
    extras: dict[str, str] | None = None,
    payload: dict[str, bytes] | None = None,
    distribution: str = "uc-manager",
) -> None:
    extras = extras or {
        "cann901-a2": f"uc-manager-cann901-a2=={version}",
        "cu130": f"uc-manager-cuda-cu130=={version}",
    }
    filename_distribution = distribution.replace("-", "_")
    dist_info = f"{filename_distribution}-{version}.dist-info"
    metadata = "\n".join(
        (
            "Metadata-Version: 2.4",
            f"Name: {distribution}",
            f"Version: {version}",
            "Requires-Python: >=3.10",
            *(f"Provides-Extra: {extra}" for extra in sorted(extras)),
            *(
                f'Requires-Dist: {requirement}; extra == "{extra}"'
                for extra, requirement in sorted(extras.items())
            ),
            "",
        )
    ).encode("utf-8")
    wheel_metadata = "\n".join(
        (
            "Wheel-Version: 1.0",
            "Generator: ucm-release-test",
            "Root-Is-Purelib: true",
            "Tag: py3-none-any",
            "",
        )
    ).encode("utf-8")
    members = {
        f"{dist_info}/METADATA": metadata,
        f"{dist_info}/WHEEL": wheel_metadata,
        f"{dist_info}/top_level.txt": b"\n",
        **(payload or {}),
    }
    record_name = f"{dist_info}/RECORD"
    members[record_name] = _record(members, record_name)
    with zipfile.ZipFile(path, "w") as archive:
        for name in sorted(members):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, members[name])


def _build_wheel(source: Path, wheelhouse: Path) -> Path:
    script = (
        "from setuptools.build_meta import build_wheel; "
        f"print(build_wheel({str(wheelhouse)!r}))"
    )
    environment = os.environ.copy()
    environment["SOURCE_DATE_EPOCH"] = str(meta.META_SOURCE_DATE_EPOCH)
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    wheels = list(wheelhouse.glob("*.whl"))
    assert len(wheels) == 1
    return wheels[0]


def test_materialized_source_builds_and_records_empty_meta_wheel(
    tmp_path: Path,
) -> None:
    plan = _plan()
    source = tmp_path / "source"
    project_path = meta.materialize_meta_source(plan, source)

    assert [path.name for path in source.iterdir()] == ["pyproject.toml"]
    document = tomllib.loads(project_path.read_text(encoding="utf-8"))
    assert document["project"] == {
        "name": "uc-manager",
        "version": VERSION,
        "description": "Unified Cache Management backend selector",
        "requires-python": ">=3.10",
        "dependencies": [],
        "optional-dependencies": {
            "cann901-a2": [f"uc-manager-cann901-a2=={VERSION}"],
            "cu130": [f"uc-manager-cuda-cu130=={VERSION}"],
        },
    }
    assert document["tool"]["setuptools"] == {"packages": []}

    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    wheel_path = _build_wheel(source, wheelhouse)
    result = meta.record_meta_wheel(plan, wheel_path)

    assert result == {
        "kind": "ucm-meta-result",
        "schema_version": 1,
        "distribution": "uc-manager",
        "version": VERSION,
        "filename": f"uc_manager-{VERSION}-py3-none-any.whl",
        "sha256": "sha256:" + hashlib.sha256(wheel_path.read_bytes()).hexdigest(),
        "size": wheel_path.stat().st_size,
        "tags": ["py3-none-any"],
        "extras": {
            "cann901-a2": f"uc-manager-cann901-a2=={VERSION}",
            "cu130": f"uc-manager-cuda-cu130=={VERSION}",
        },
        "requires_dist": [
            f'uc-manager-cann901-a2=={VERSION}; extra == "cann901-a2"',
            f'uc-manager-cuda-cu130=={VERSION}; extra == "cu130"',
        ],
    }


def test_materialized_source_build_is_reproducible(tmp_path: Path) -> None:
    wheels: list[Path] = []
    for name in ("first", "second"):
        source = tmp_path / name / "source"
        wheelhouse = tmp_path / name / "wheelhouse"
        wheelhouse.mkdir(parents=True)
        meta.materialize_meta_source(_plan(), source)
        wheels.append(_build_wheel(source, wheelhouse))

    assert wheels[0].read_bytes() == wheels[1].read_bytes()


def test_fork_meta_source_and_wheel_use_the_owner_namespace(tmp_path: Path) -> None:
    plan = _fork_plan()
    source = tmp_path / "source"
    project_path = meta.materialize_meta_source(plan, source)
    document = tomllib.loads(project_path.read_text(encoding="utf-8"))

    assert document["project"]["name"] == "supermarioyl-uc-manager"
    assert document["project"]["optional-dependencies"] == {
        "cann901-a2": [f"supermarioyl-uc-manager-cann901-a2=={VERSION}"],
        "cu130": [f"supermarioyl-uc-manager-cuda-cu130=={VERSION}"],
    }

    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    wheel_path = _build_wheel(source, wheelhouse)
    result = meta.record_meta_wheel(plan, wheel_path)

    assert wheel_path.name == (f"supermarioyl_uc_manager-{VERSION}-py3-none-any.whl")
    assert result["distribution"] == "supermarioyl-uc-manager"
    assert all(
        requirement.startswith("supermarioyl-uc-manager-")
        for requirement in result["requires_dist"]
    )


def test_fork_meta_extra_is_satisfied_by_exact_local_wheels(tmp_path: Path) -> None:
    plan = _fork_plan()
    meta_source = tmp_path / "meta-source"
    meta_wheelhouse = tmp_path / "meta-wheelhouse"
    meta_wheelhouse.mkdir()
    meta.materialize_meta_source(plan, meta_source)
    meta_wheel = _build_wheel(meta_source, meta_wheelhouse)

    backend_source = tmp_path / "backend-source"
    backend_source.mkdir()
    (backend_source / "pyproject.toml").write_text(
        "\n".join(
            (
                "[build-system]",
                'requires = ["setuptools==75.8.2", "wheel==0.45.1"]',
                'build-backend = "setuptools.build_meta"',
                "",
                "[project]",
                'name = "supermarioyl-uc-manager-cuda-cu130"',
                f'version = "{VERSION}"',
                "",
                "[tool.setuptools]",
                "packages = []",
                "",
            )
        ),
        encoding="utf-8",
    )
    backend_wheelhouse = tmp_path / "backend-wheelhouse"
    backend_wheelhouse.mkdir()
    backend_wheel = _build_wheel(backend_source, backend_wheelhouse)
    target = tmp_path / "installed"
    installer_venv = tmp_path / "installer-venv"
    subprocess.run(
        [sys.executable, "-m", "venv", str(installer_venv)],
        check=True,
        capture_output=True,
        text=True,
    )

    subprocess.run(
        [
            str(installer_venv / "bin" / "python"),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-index",
            "--target",
            str(target),
            f"{meta_wheel}[cu130]",
            str(backend_wheel),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    installed = {
        distribution.metadata["Name"]: distribution.version
        for distribution in importlib.metadata.distributions(path=[str(target)])
    }
    assert installed == {
        "supermarioyl-uc-manager": VERSION,
        "supermarioyl-uc-manager-cuda-cu130": VERSION,
    }


def test_release_workflow_uses_canonical_meta_source_date_epoch() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release-ucm.yml").read_text(
        encoding="utf-8"
    )
    build_meta = workflow.split("\n  build-meta:", 1)[1].split("\n  package-chart:", 1)[
        0
    ]

    assert f'SOURCE_DATE_EPOCH: "{meta.META_SOURCE_DATE_EPOCH}"' in build_meta


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda plan: (
                plan.update(version="0.9.1+cann901.a2"),
                plan["meta_package"].update(version="0.9.1+cann901.a2"),
            ),
            "must not contain a local version",
        ),
        (
            lambda plan: plan["meta_package"]["extras"].update(
                {"CANN901_A2": plan["meta_package"]["extras"].pop("cann901-a2")}
            ),
            "invalid canonical meta package extra",
        ),
        (
            lambda plan: plan["meta_package"]["extras"].pop("cu130"),
            "must exactly match planned runtime variants",
        ),
        (
            lambda plan: plan["meta_package"]["extras"].update(
                {"cu130": f"uc-manager-cuda-cu130>={VERSION}"}
            ),
            "must pin one distribution with ==",
        ),
    ),
)
def test_meta_plan_rejects_invalid_version_or_extras(mutate, message: str) -> None:
    plan = copy.deepcopy(_plan())
    mutate(plan)

    with pytest.raises(ValueError, match=message):
        meta.validate_meta_package(plan)


def test_meta_plan_rejects_swapped_extra_backends() -> None:
    plan = _plan()
    extras = plan["meta_package"]["extras"]
    extras["cu130"], extras["cann901-a2"] = (
        extras["cann901-a2"],
        extras["cu130"],
    )

    with pytest.raises(ValueError, match="must exactly match planned runtime variants"):
        meta.validate_meta_package(plan)


def test_meta_plan_rejects_repository_prefix_drift() -> None:
    plan = _fork_plan()
    plan["publish"]["pypi"]["distribution_prefix"] = "other-owner-"

    with pytest.raises(ValueError, match="prefix differs"):
        meta.validate_meta_package(plan)


def test_meta_wheel_rejects_package_payload(tmp_path: Path) -> None:
    wheel_path = tmp_path / f"uc_manager-{VERSION}-py3-none-any.whl"
    _write_wheel(wheel_path, payload={"ucm/__init__.py": b""})

    with pytest.raises(ValueError, match="must not contain package payload"):
        meta.record_meta_wheel(_plan(), wheel_path)


def test_meta_wheel_rejects_wrong_version(tmp_path: Path) -> None:
    wheel_path = tmp_path / "uc_manager-0.9.2-py3-none-any.whl"
    _write_wheel(wheel_path, version="0.9.2")

    with pytest.raises(ValueError, match="filename version does not match"):
        meta.record_meta_wheel(_plan(), wheel_path)


def test_meta_wheel_rejects_wrong_extras(tmp_path: Path) -> None:
    wheel_path = tmp_path / f"uc_manager-{VERSION}-py3-none-any.whl"
    _write_wheel(
        wheel_path,
        extras={"cann901-a2": f"uc-manager-cann901-a2=={VERSION}"},
    )

    with pytest.raises(ValueError, match="Provides-Extra does not match"):
        meta.record_meta_wheel(_plan(), wheel_path)
