from __future__ import annotations

import copy
import hashlib
import importlib
import json
import sys
from pathlib import Path
from urllib.parse import quote

import pytest

ROOT = Path(__file__).resolve().parents[3]
RELEASE_ROOT = ROOT / ".github" / "release"
sys.path.insert(0, str(RELEASE_ROOT))
release = importlib.import_module("ucm_release.release")
public_manifest = importlib.import_module("ucm_release.manifest")
cleanup_records = importlib.import_module("ucm_release.cleanup_records")


def _plan() -> dict[str, object]:
    return {
        "git_tag": "v0.7.62rc1",
        "release_type": "prerelease",
        "release_kind": "publish",
        "version": "0.7.62rc1",
        "publish": {
            "pypi": {
                "requested": False,
                "enabled": False,
                "disposition": "disabled",
                "index": "https://upload.pypi.org/legacy/",
                "target": "pypi",
                "simple_index": "https://pypi.org/simple/",
            },
            "ghcr": {
                "requested": True,
                "enabled": True,
                "disposition": "publish",
                "namespace": "ghcr.io/example",
            },
            "dockerhub": {
                "requested": False,
                "enabled": False,
                "disposition": "disabled",
                "namespace": "docker.io/example",
            },
            "chart_oci": {
                "requested": True,
                "enabled": True,
                "disposition": "publish",
                "namespace": "ghcr.io/example/charts",
            },
            "github_release": {
                "requested": True,
                "enabled": True,
                "disposition": "publish",
            },
        },
        "chart": {
            "name": "unified-cache-chart",
            "version": "0.7.62-rc.1",
            "app_version": "0.7.62rc1",
        },
        "wheels": [
            {
                "id": "cuda129-cp312-amd64",
                "dist_name": "uc-manager-cuda-cu129",
                "wheel_version": "0.7.62rc1",
                "python_abi": "cp312",
                "cpu_arch": "amd64",
                "backend": "cuda",
                "runtime_variant": "cu129",
                "manylinux": "manylinux_2_28",
                "target_platform_tag": "manylinux_2_28_x86_64",
                "external_runtime_exclude_patterns": ["libcudart.so.12"],
                "runtime_requirements": ["wrapt==1.17.2"],
                "repair": {
                    "tool": "auditwheel",
                    "version": "6.7.0",
                    "target_platform": "manylinux_2_28_x86_64",
                    "excluded_patterns": ["libcudart.so.12"],
                },
                "builder": {
                    "repository": "ghcr.io/example/ucm-builder",
                    "tag": "cuda129-cp312-amd64",
                    "digest": "sha256:" + "c" * 64,
                    "source_image": "docker.io/pytorch/manylinux2_28-builder:cuda12.9",
                    "source_image_digest": "sha256:builder",
                },
            }
        ],
        "images": [
            {
                "id": "vllm-v023-amd64",
                "family_id": "vllm-v023",
                "wheel_id": "cuda129-cp312-amd64",
                "cpu_arch": "amd64",
                "runtime": {
                    "repository": "docker.io/vllm/vllm-openai",
                    "tag": "v0.23.0",
                    "python_abi": "cp312",
                    "accelerator_runtime": "cuda-12.9",
                },
            }
        ],
        "families": [
            {
                "id": "vllm-v023",
                "runtime": {
                    "repository": "docker.io/vllm/vllm-openai",
                    "tag": "v0.23.0",
                    "accelerator_runtime": "cuda-12.9",
                },
                "members": [
                    {
                        "image_id": "vllm-v023-amd64",
                        "cpu_arch": "amd64",
                        "reference": "ghcr.io/example/vllm:v0.23.0-ucm-amd64",
                    }
                ],
                "published_reference": "ghcr.io/example/vllm:v0.23.0-ucm-amd64",
                "create_index": False,
            }
        ],
    }


def _write_artifact_inputs(
    tmp_path: Path,
) -> tuple[Path, Path, Path, str]:
    wheels = tmp_path / "wheels" / "one"
    wheels.mkdir(parents=True)
    filename = (
        "uc_manager_cuda_cu129-0.7.62rc1-cp312-cp312-" "manylinux_2_28_x86_64.whl"
    )
    (wheels / filename).write_bytes(b"wheel")
    report_text = "\n".join(
        (
            "DEBUG:auditwheel.wheel_abi:full_elftree:",
            json.dumps(
                {
                    "ucm/test-extension.so": {
                        "needed": ["libcudart.so.12"],
                        "libraries": {
                            "libcudart.so.12": {"needed": ["libdriver.so"]},
                            "libdriver.so": {"needed": []},
                        },
                    }
                },
                sort_keys=True,
            ),
            f"{filename} is consistent with the following platform tag: "
            '"linux_x86_64".',
            "",
            "The wheel references external versioned symbols in these system-provided "
            "shared libraries: libc.so.6 with versions {'GLIBC_2.2.5', 'GLIBC_2.17'}",
            "",
            'This constrains the platform tag to "manylinux_2_27_x86_64".',
            "",
            "The following external shared libraries are required by the wheel:",
            json.dumps({"libcudart.so.12": None, "libdriver.so": None}, indent=4),
            "",
        )
    )
    report_path = wheels / "auditwheel-show.txt"
    report_path.write_text(report_text, encoding="utf-8")
    result_path = wheels / "wheel-result.json"
    result_path.write_text(
        json.dumps(
            {
                "kind": "ucm-wheel-result",
                "schema_version": 5,
                "task_id": "cuda129-cp312-amd64",
                "distribution": "uc-manager-cuda-cu129",
                "version": "0.7.62rc1",
                "python_abi": "cp312",
                "cpu_arch": "amd64",
                "filename": filename,
                "sha256": hashlib.sha256(b"wheel").hexdigest(),
                "platform_tags": ["manylinux_2_28_x86_64"],
                "auditwheel_platform_tag": "linux_x86_64",
                "abi_compatible_platform_tag": "manylinux_2_27_x86_64",
                "glibc_versions": ["GLIBC_2.2.5", "GLIBC_2.17"],
                "glibc_floor": "GLIBC_2.17",
                "external_library_roots": ["libcudart.so.12"],
                "external_libraries": ["libcudart.so.12", "libdriver.so"],
                "deferred_external_libraries": ["libdriver.so"],
                "repair": {
                    "tool": "auditwheel",
                    "version": "6.7.0",
                    "target_platform": "manylinux_2_28_x86_64",
                    "excluded_patterns": ["libcudart.so.12"],
                },
                "dependencies": ["wrapt==1.17.2"],
                "auditwheel_report": {
                    "filename": report_path.name,
                    "sha256": hashlib.sha256(report_text.encode()).hexdigest(),
                    "text": report_text,
                },
            }
        ),
        encoding="utf-8",
    )
    chart = tmp_path / "chart"
    chart.mkdir()
    (chart / "unified-cache-chart-0.7.62-rc.1.tgz").write_bytes(b"chart")
    return tmp_path / "wheels", chart, result_path, filename


def _asset_urls(manifest: dict[str, object]) -> dict[str, str]:
    filenames = {
        str(item["filename"]) for item in manifest["wheels"]  # type: ignore[index]
    }
    if manifest.get("chart") is not None:
        filenames.add(str(manifest["chart"]["filename"]))  # type: ignore[index]
    return {
        filename: f"https://github.com/example/ucm/releases/download/v1/{filename}"
        for filename in filenames
    }


def _single_family_release_notes_manifest(
    *,
    expected_targets: dict[str, str],
    targets: list[dict[str, str]],
    release_status: str = "complete",
    family_status: str = "published",
) -> dict[str, object]:
    return {
        "release": {"git_tag": "v1.0.0rc1", "status": release_status},
        "chart": {"filename": "unified-cache-chart-1.0.0-rc.1.tgz"},
        "wheels": [
            {
                "id": "cuda-amd64",
                "filename": "uc_manager_cuda-amd64.whl",
                "backend": "cuda",
                "runtime_variant": "cu130",
                "python_abi": "cp312",
                "cpu_arch": "amd64",
            }
        ],
        "images": [
            {
                "family_id": "openai-v1",
                "wheel_id": "cuda-amd64",
                "cpu_arch": "amd64",
            }
        ],
        "families": [
            {
                "id": "openai-v1",
                "runtime": {
                    "repository": "docker.io/vllm/vllm-openai",
                    "tag": "v1.0.0",
                    "accelerator_runtime": "cuda-13.0",
                },
                "expected_targets": expected_targets,
                "status": family_status,
                "targets": targets,
            }
        ],
    }


def _write_meta_artifact(tmp_path: Path, plan: dict[str, object]) -> Path:
    meta_root = tmp_path / "meta"
    meta_root.mkdir()
    filename = "uc_manager-0.7.62rc1-py3-none-any.whl"
    wheel = meta_root / filename
    wheel.write_bytes(b"meta")
    meta_package = plan["meta_package"]
    assert isinstance(meta_package, dict)
    (meta_root / "meta-result.json").write_text(
        json.dumps(
            {
                "kind": "ucm-meta-result",
                "schema_version": 1,
                "distribution": "uc-manager",
                "version": "0.7.62rc1",
                "filename": filename,
                "sha256": "sha256:" + hashlib.sha256(b"meta").hexdigest(),
                "size": 4,
                "tags": ["py3-none-any"],
                "extras": meta_package["extras"],
                "requires_dist": [],
            }
        ),
        encoding="utf-8",
    )
    return meta_root


@pytest.mark.parametrize(
    ("prefix", "target", "repository_url"),
    [
        ("", "pypi", "https://upload.pypi.org/legacy/"),
        ("supermarioyl-", "testpypi", "https://test.pypi.org/legacy/"),
    ],
)
def test_enabled_pypi_requires_and_records_complete_receipt(
    tmp_path: Path, prefix: str, target: str, repository_url: str
) -> None:
    filename_prefix = prefix.replace("-", "_")
    meta_distribution = f"{prefix}uc-manager"
    backend_distribution = f"{prefix}uc-manager-cuda-cu129"
    meta_filename = f"{filename_prefix}uc_manager-0.7.62rc1-py3-none-any.whl"
    backend_filename = (
        f"{filename_prefix}uc_manager_cuda_cu129-0.7.62rc1-"
        "cp312-cp312-manylinux_2_28_x86_64.whl"
    )
    extra_requirement = f"{backend_distribution}==0.7.62rc1"
    manifest = {
        "publish": {
            "pypi": {
                "enabled": True,
                "target": target,
                "index": repository_url,
            }
        },
        "release": {
            "git_tag": "v0.7.62rc1",
            "version": "0.7.62rc1",
            "status": "artifacts-ready",
        },
        "meta_package": {
            "distribution": meta_distribution,
            "version": "0.7.62rc1",
            "filename": meta_filename,
            "sha256": "sha256:" + "b" * 64,
            "extras": {"cu129": extra_requirement},
        },
        "wheels": [
            {
                "distribution": backend_distribution,
                "version": "0.7.62rc1",
                "filename": backend_filename,
                "sha256": "a" * 64,
            }
        ],
        "images": [
            {
                "id": "image",
                "family_id": "family",
                "expected_targets": {},
                "status": "not-requested",
                "targets": [],
            }
        ],
        "families": [
            {
                "id": "family",
                "create_index": False,
                "expected_targets": {},
                "status": "not-requested",
                "targets": [],
            }
        ],
    }
    with pytest.raises(ValueError, match="no complete receipt"):
        release.finalize_release_state(
            manifest,
            tmp_path,
            build_outcome="skipped",
            member_outcome="skipped",
            index_outcome="skipped",
            pypi_outcome="success",
            pypi_install_outcome="success",
        )

    receipt = {
        "kind": "ucm-pypi-receipt",
        "schema_version": 2,
        "status": "complete",
        "version": "0.7.62rc1",
        "target": target,
        "repository_url": repository_url,
        "projects": [
            {
                "project": backend_distribution,
                "version": "0.7.62rc1",
                "role": "backend",
                "files": [
                    {
                        "filename": backend_filename,
                        "sha256": "sha256:" + "a" * 64,
                    }
                ],
            },
            {
                "project": meta_distribution,
                "version": "0.7.62rc1",
                "role": "meta",
                "files": [
                    {
                        "filename": meta_filename,
                        "sha256": "sha256:" + "b" * 64,
                    }
                ],
            },
        ],
        "extras": {"cu129": extra_requirement},
    }
    (tmp_path / "pypi-receipt.json").write_text(json.dumps(receipt), encoding="utf-8")

    result = release.finalize_release_state(
        manifest,
        tmp_path,
        build_outcome="skipped",
        member_outcome="skipped",
        index_outcome="skipped",
        pypi_outcome="success",
        pypi_install_outcome="success",
    )

    assert result["release"]["status"] == "complete"
    assert result["pypi"] == receipt

    receipt["projects"][0]["files"][0]["sha256"] = "sha256:" + "c" * 64
    (tmp_path / "pypi-receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="do not match release artifacts"):
        release.finalize_release_state(
            manifest,
            tmp_path,
            build_outcome="skipped",
            member_outcome="skipped",
            index_outcome="skipped",
            pypi_outcome="success",
            pypi_install_outcome="success",
        )


def test_disabled_pypi_requires_both_jobs_to_be_skipped(tmp_path: Path) -> None:
    manifest = {
        "publish": {"pypi": {"enabled": False}},
        "release": {"version": "0.9.1.dev1"},
        "images": [],
        "families": [],
    }

    with pytest.raises(ValueError, match="jobs must be skipped"):
        release.finalize_release_state(
            manifest,
            tmp_path,
            build_outcome="skipped",
            member_outcome="skipped",
            index_outcome="skipped",
            pypi_outcome="success",
            pypi_install_outcome="skipped",
        )


def test_artifacts_and_image_receipts_form_one_mapping(tmp_path: Path) -> None:
    wheels, chart, _, filename = _write_artifact_inputs(tmp_path)

    manifest, checksums = release.build_release_state(
        _plan(), wheels, chart, actions_run_id=123
    )
    assert manifest["kind"] == "ucm-release-state"
    assert manifest["schema_version"] == 3
    assert manifest["release"]["release_type"] == "prerelease"
    assert manifest["release"]["actions_run_id"] == 123
    assert manifest["chart"]["oci_reference"] == (
        "ghcr.io/example/charts/unified-cache-chart:0.7.62-rc.1"
    )
    assert manifest["release"]["status"] == "artifacts-ready"
    assert manifest["publish"]["pypi"]["disposition"] == "disabled"
    assert manifest["wheels"][0]["platform_tags"] == ["manylinux_2_28_x86_64"]
    assert manifest["wheels"][0]["auditwheel_platform_tag"] == "linux_x86_64"
    assert manifest["wheels"][0]["builder"]["source_image_digest"] == ("sha256:builder")
    assert manifest["wheels"][0]["builder"]["digest"] == "sha256:" + "c" * 64
    assert manifest["images"][0]["wheel_id"] == "cuda129-cp312-amd64"
    asset_urls = _asset_urls(manifest)
    notes = release.render_notes(
        manifest, repository="example/ucm", asset_urls=asset_urls
    )
    assert "Upstream Runtime tags<br>docker.io/vllm/vllm-openai：" in notes
    assert "Runtime tags<br>GHCR: ghcr.io/example/vllm" in notes
    assert "`v0.23.0`" in notes
    assert f"[x86_64]({asset_urls[filename]})" in notes
    assert "amd64=" not in notes
    assert notes.startswith("Status: `artifacts-ready`")
    assert "# UCM" not in notes
    assert "Checksums:" not in notes
    assert "SHA256SUMS" not in notes
    assert "release-manifest.json" not in notes
    assert {name for _, name in checksums} == {
        filename,
        "unified-cache-chart-0.7.62-rc.1.tgz",
    }

    receipts = tmp_path / "receipts"
    receipts.mkdir()
    (receipts / "member.json").write_text(
        json.dumps(
            {
                "kind": "ucm-image-member-receipt",
                "schema_version": 1,
                "id": "vllm-v023-amd64",
                "status": "published",
                "targets": [
                    {
                        "channel": "ghcr",
                        "reference": "ghcr.io/example/vllm:v0.23.0-ucm-amd64",
                        "digest": "sha256:" + "a" * 64,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    final = release.finalize_release_state(
        manifest,
        receipts,
        build_outcome="success",
        member_outcome="success",
        index_outcome="skipped",
    )
    assert final["release"]["status"] == "complete"
    assert final["images"][0]["targets"][0]["digest"] == "sha256:" + "a" * 64
    assert final["families"][0]["targets"] == final["images"][0]["targets"]
    final_notes = release.render_notes(
        final, repository="example/ucm", asset_urls=asset_urls
    )
    assert "https://github.com/example/ucm/pkgs/container/vllm" in final_notes
    assert "sha256:" not in final_notes

    state_path = tmp_path / "release-state.json"
    release_path = tmp_path / "github-release.json"
    notes_output = tmp_path / "notes-output"
    notes_output.mkdir()
    state_path.write_text(json.dumps(final), encoding="utf-8")
    release_path.write_text(
        json.dumps(
            {
                "tag_name": "v0.7.62rc1",
                "body": "## 更新说明\n\n保留人工维护的变更记录。\n",
                "assets": [
                    {"name": name, "browser_download_url": url}
                    for name, url in asset_urls.items()
                ],
            }
        ),
        encoding="utf-8",
    )
    command = release.build_parser().parse_args(
        [
            "notes",
            "--manifest",
            str(state_path),
            "--release",
            str(release_path),
            "--repository",
            "example/ucm",
            "--output",
            str(notes_output),
        ]
    )
    command.func(command)
    body = (notes_output / "release-notes.md").read_text(encoding="utf-8")
    assert body == (
        "## 更新说明\n\n保留人工维护的变更记录。\n\n"
        "<!-- ucm-release:begin -->\n"
        f"{final_notes.rstrip()}\n"
        "<!-- ucm-release:end -->\n"
    )
    release_document = json.loads(release_path.read_text(encoding="utf-8"))
    # The workflow's shell command substitution removes trailing newlines.
    release_document["body"] = body.rstrip("\n")
    release_path.write_text(json.dumps(release_document), encoding="utf-8")
    command.func(command)
    assert (notes_output / "release-notes.md").read_text(encoding="utf-8") == body


def test_disabled_image_publication_completes_with_wheels_and_chart(
    tmp_path: Path,
) -> None:
    wheels, chart, _, filename = _write_artifact_inputs(tmp_path)
    plan = _plan()
    publish = plan["publish"]
    assert isinstance(publish, dict)
    ghcr = publish["ghcr"]
    assert isinstance(ghcr, dict)
    ghcr["enabled"] = False

    manifest, _ = release.build_release_state(plan, wheels, chart, actions_run_id=123)

    assert manifest["release"]["status"] == "artifacts-ready"
    assert manifest["images"][0]["expected_targets"] == {}
    assert manifest["images"][0]["status"] == "not-requested"
    assert manifest["families"][0]["expected_targets"] == {}
    assert manifest["families"][0]["status"] == "not-requested"

    with pytest.raises(ValueError, match="must all be skipped"):
        release.finalize_release_state(
            manifest,
            tmp_path / "missing-receipts",
            build_outcome="success",
            member_outcome="skipped",
            index_outcome="skipped",
        )

    final = release.finalize_release_state(
        manifest,
        tmp_path / "missing-receipts",
        build_outcome="skipped",
        member_outcome="skipped",
        index_outcome="skipped",
    )

    assert final["release"]["status"] == "complete"
    assert all(item["status"] == "not-requested" for item in final["images"])
    assert all(item["status"] == "not-requested" for item in final["families"])
    assert all(item["targets"] == [] for item in [*final["images"], *final["families"]])
    asset_urls = _asset_urls(final)
    notes = release.render_notes(final, repository="example/ucm", asset_urls=asset_urls)
    assert f"[x86_64]({asset_urls[filename]})" in notes
    assert "| — |" in notes
    assert "Images are still building" not in notes
    assert "Images:" not in notes
    assert "pkgs/container" not in notes

    skipped_chart = release.finalize_release_state(
        manifest,
        tmp_path / "missing-receipts",
        build_outcome="skipped",
        member_outcome="skipped",
        index_outcome="skipped",
        chart_oci_outcome="skipped",
    )
    assert skipped_chart["release"]["status"] == "publication-failed"


@pytest.mark.parametrize(
    "release_type,draft_placeholder",
    [("prerelease", False), ("nightly", False), ("nightly", True)],
)
def test_disabled_channels_finalize_with_wheels_only_regardless_of_release_type(
    tmp_path: Path,
    release_type: str,
    draft_placeholder: bool,
) -> None:
    wheels, _, _, filename = _write_artifact_inputs(tmp_path)
    plan = _plan()
    plan["release_type"] = release_type
    plan["images"][0]["runtime"].update(
        product_id="vllm", variant="default", soc_version="na"
    )
    if release_type == "nightly":
        plan["git_tag"] = "nightly/v0.7.62-20260917-1"
    for channel in ("ghcr", "chart_oci"):
        plan["publish"][channel].update(
            requested=False, enabled=False, disposition="disabled"
        )

    plan["meta_package"] = {
        "distribution": "uc-manager",
        "version": plan["version"],
        "extras": {"cu129": f"uc-manager-cuda-cu129=={plan['version']}"},
    }
    meta_root = _write_meta_artifact(tmp_path, plan)
    state, checksums = release.build_release_state(
        plan, wheels, tmp_path / "missing-chart", meta_root, actions_run_id=123
    )
    assert state["chart"] is None
    assert {name for _, name in checksums} == {
        filename,
        state["meta_package"]["filename"],
    }
    final = release.finalize_release_state(
        state,
        tmp_path / "missing-receipts",
        build_outcome="skipped",
        member_outcome="skipped",
        index_outcome="skipped",
        chart_oci_outcome="skipped",
    )
    assert final["release"]["status"] == "complete"
    asset_urls = _asset_urls(final)
    release_document = {
        "tag_name": plan["git_tag"],
        "html_url": f"https://github.com/example/ucm/releases/tag/{plan['git_tag']}",
        "assets": [
            {"name": name, "browser_download_url": url}
            for name, url in asset_urls.items()
        ],
    }
    if draft_placeholder:
        release_document["draft"] = True
        release_document["html_url"] = (
            "https://github.com/example/ucm/releases/tag/untagged-123"
        )
        for asset in release_document["assets"]:
            asset["browser_download_url"] = asset["browser_download_url"].replace(
                "/download/v1/", "/download/untagged-123/"
            )
        asset_urls = {
            name: f"https://github.com/example/ucm/releases/download/{plan['git_tag']}/{name}"
            for name in asset_urls
        }
    assert public_manifest.asset_urls(final, release_document) == asset_urls
    notes = release.render_notes(final, repository="example/ucm", asset_urls=asset_urls)
    assert asset_urls[filename] in notes
    public = public_manifest.build_manifest(final, release_document)
    assert (
        public["release"]["url"]
        == f"https://github.com/example/ucm/releases/tag/{plan['git_tag']}"
    )
    assert public["wheels"][0]["url"] == asset_urls[filename]
    assert public["schema_version"] == 10
    assert public["chart"] is None
    assert public["images"] == []
    from ucm_release import cleanup_records

    assert cleanup_records.registry_resources(public) == []
    assert public["github_release_assets"] == sorted(
        [filename, "release-manifest.json"]
    )


@pytest.fixture
def published_release(tmp_path):
    wheels, chart, _, _ = _write_artifact_inputs(tmp_path)
    plan = _plan()
    plan["meta_package"] = {
        "distribution": "uc-manager",
        "version": "0.7.62rc1",
        "extras": {"cu129": "uc-manager-cuda-cu129==0.7.62rc1"},
    }
    plan["images"][0]["runtime"].update(  # type: ignore[index]
        {
            "product_id": "vllm",
            "version": "0.23.0",
            "channel": "stable",
            "variant": "default",
            "soc_version": "na",
            "os_id": "ubuntu",
            "os_version": "22.04",
        }
    )
    plan["families"][0]["runtime"].update(  # type: ignore[index]
        {
            "version": "0.23.0",
            "channel": "stable",
            "variant": "default",
            "soc_version": "na",
            "os_id": "ubuntu",
            "os_version": "22.04",
        }
    )
    meta_root = _write_meta_artifact(tmp_path, plan)
    state, _ = release.build_release_state(
        plan, wheels, chart, meta_root, actions_run_id=987654
    )
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    (receipts / "member.json").write_text(
        json.dumps(
            {
                "kind": "ucm-image-member-receipt",
                "schema_version": 1,
                "id": "vllm-v023-amd64",
                "status": "published",
                "targets": [
                    {
                        "channel": "ghcr",
                        "reference": "ghcr.io/example/vllm:v0.23.0-ucm-amd64",
                        "digest": "sha256:" + "d" * 64,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    state = release.finalize_release_state(
        state,
        receipts,
        build_outcome="success",
        member_outcome="success",
        index_outcome="skipped",
    )
    asset_urls = _asset_urls(state)
    document: dict[str, object] = {
        "tag_name": "v0.7.62rc1",
        "html_url": "https://github.com/example/ucm/releases/tag/v0.7.62rc1",
        "assets": [
            {"name": name, "browser_download_url": url}
            for name, url in asset_urls.items()
        ],
    }
    document["assets"].append(  # type: ignore[union-attr]
        {
            "name": "ucm_config_example.yaml",
            "browser_download_url": (
                "https://github.com/example/ucm/releases/download/"
                "v0.7.62rc1/ucm_config_example.yaml"
            ),
        }
    )
    return state, document


def test_public_manifest_is_exact_schema_v10_and_uses_published_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    published_release,
) -> None:
    state, document = published_release
    asset_urls = _asset_urls(state)
    manifest = public_manifest.build_manifest(state, document)

    assert manifest["schema_version"] == 10
    assert manifest["release"]["tag"] == state["release"]["git_tag"]
    assert manifest["release"]["actions_run_id"] == 987654
    assert manifest["python"]["extras"] == {"cu129": "uc-manager-cuda-cu129"}
    assert manifest["python"]["pypi"] is None
    wheel = manifest["wheels"][0]
    assert wheel["url"] == asset_urls[wheel["filename"]]
    assert wheel["sha256"] == hashlib.sha256(b"wheel").hexdigest()
    assert manifest["images"][0]["publications"]["ghcr"]["pull"] == (
        "ghcr.io/example/vllm:v0.23.0-ucm-amd64"
    )
    assert set(manifest["github_release_assets"]) == {
        "release-manifest.json",
        *(asset["name"] for asset in document["assets"]),
    }

    pypi_state = json.loads(json.dumps(state))
    pypi_state["publish"]["pypi"]["enabled"] = True
    pypi_state["pypi"] = {
        "kind": "ucm-pypi-receipt",
        "schema_version": 2,
        "target": "pypi",
        "repository_url": "https://upload.pypi.org/legacy/",
        "status": "complete",
        "version": pypi_state["release"]["version"],
        "projects": public_manifest.expected_pypi_projects(pypi_state),
        "extras": pypi_state["meta_package"]["extras"],
    }
    pypi_manifest = public_manifest.build_manifest(pypi_state, document)
    assert pypi_manifest["python"]["pypi"] == {
        "index_url": "https://pypi.org/simple",
        "project_url": "https://pypi.org/project/uc-manager/0.7.62rc1/",
    }
    assert "pypi-receipt.json" not in pypi_manifest["github_release_assets"]
    assert not any(
        "py3-none-any" in name for name in pypi_manifest["github_release_assets"]
    )
    from ucm_release import cleanup_records

    assert public_manifest.validate_manifest(pypi_manifest) == pypi_manifest
    assert cleanup_records.registry_resources(pypi_manifest)
    sys.path.insert(0, str(ROOT / "docs/docs-site/tools"))
    import release_manifest as docs_manifest

    published = json.loads(json.dumps(document))
    published["prerelease"] = True
    published["assets"].append(
        {
            "name": "release-manifest.json",
            "browser_download_url": "https://github.com/example/ucm/releases/download/v0.7.62rc1/release-manifest.json",
        }
    )
    monkeypatch.setattr(docs_manifest, "fetch_json", lambda _url: pypi_manifest)
    assert docs_manifest._release_manifest("example/ucm", published) == pypi_manifest

    state_path = tmp_path / "public-state.json"
    release_path = tmp_path / "public-release.json"
    output = tmp_path / "public-output"
    output.mkdir()
    state_path.write_text(json.dumps(state), encoding="utf-8")
    release_path.write_text(json.dumps(document), encoding="utf-8")
    command = release.build_parser().parse_args(
        [
            "manifest",
            "--state",
            str(state_path),
            "--release",
            str(release_path),
            "--output",
            str(output),
        ]
    )
    command.func(command)
    assert json.loads((output / "release-manifest.json").read_text()) == manifest


@pytest.fixture
def custom_draft_publication(published_release):
    state, document = json.loads(
        json.dumps(published_release)
        .replace("0.7.62rc1", "0.7.62.dev0+draft.test")
        .replace("0.7.62-rc.1", "0.7.62-draft.test")
        .replace("v0.23.0-ucm-amd64", "nightly-ucm-0.7.62.dev0.draft.test")
    )
    tag = "draft/v0.7.62-test"
    state["release"].update(git_tag=tag, release_type="draft", release_kind="draft")
    for item in [*state["images"], *state["families"]]:
        item["runtime"].update(tag="nightly", version="nightly", channel="pinned")
    document.update(
        tag_name=tag,
        draft=True,
        html_url=f"https://github.com/example/ucm/releases/tag/{tag}",
    )
    return state, document


def test_custom_draft_manifest_keeps_local_version_and_opaque_runtime(
    custom_draft_publication,
):
    state, document = custom_draft_publication
    manifest = public_manifest.build_manifest(state, document)
    version = "0.7.62.dev0+draft.test"
    wheel = manifest["wheels"][0]
    assert {
        manifest["release"]["version"],
        manifest["python"]["version"],
        wheel["version"],
    } == {version}
    assert manifest["images"][0]["upstream"] == {
        "version": "nightly",
        "channel": "pinned",
    }
    assert quote(version, safe="") in wheel["url"]
    assert cleanup_records.record_from_manifest(manifest, "example/ucm")[
        "versions"
    ] == [version]


def test_custom_draft_manifest_generation_rejects_a_lost_label(
    custom_draft_publication,
):
    state, document = custom_draft_publication
    # Keep every package and filename consistent, so the Tag binding must detect the loss.
    state, document = json.loads(
        json.dumps((state, document)).replace("0.7.62.dev0+draft.test", "0.7.62.dev0")
    )
    with pytest.raises(ValueError, match="version differs from release state identity"):
        public_manifest.build_manifest(state, document)


def test_wheel_capabilities_include_every_linked_image_runtime() -> None:
    from ucm_release.manifest import _wheel_capabilities

    images = [
        {
            "wheel_id": "shared-wheel",
            "runtime": {
                "product_id": "sglang",
                "accelerator_runtime": runtime,
                "variant": "default",
                "soc_version": "ascend910",
            },
        }
        for runtime in ("cann-9.0", "cann-9.1")
    ]

    assert _wheel_capabilities("shared-wheel", images) == [
        {
            "product": "sglang",
            "accelerator": {
                "runtime": "cann-9.0",
                "variant": "default",
                "soc_version": "ascend910",
            },
        },
        {
            "product": "sglang",
            "accelerator": {
                "runtime": "cann-9.1",
                "variant": "default",
                "soc_version": "ascend910",
            },
        },
    ]


def test_public_manifest_rejects_incomplete_release() -> None:
    with pytest.raises(ValueError, match="complete publication"):
        public_manifest.build_manifest(
            {
                "kind": "ucm-release-state",
                "schema_version": 3,
                "release": {
                    "git_tag": "nightly/v0.8.1-20260826-1",
                    "status": "images-failed",
                },
            },
            {"tag_name": "nightly/v0.8.1-20260826-1"},
        )


@pytest.mark.parametrize("receipt_status", [None, "uploading"])
def test_release_notes_hide_pypi_until_a_complete_receipt(
    receipt_status: str | None,
) -> None:
    manifest = _single_family_release_notes_manifest(
        expected_targets={}, targets=[], release_status="artifacts-ready"
    )
    manifest["publish"] = {
        "pypi": {
            "enabled": True,
            "target": "testpypi",
            "simple_index": "https://test.pypi.org/simple/",
            "dependency_index": "https://pypi.org/simple/",
        }
    }
    if receipt_status is not None:
        manifest["pypi"] = {"status": receipt_status}

    notes = release.render_notes(
        manifest, repository="example/ucm", asset_urls=_asset_urls(manifest)
    )

    assert "## PyPI" not in notes
    assert "## TestPyPI" not in notes
    assert "pip install" not in notes
    assert "pip download" not in notes
    assert "[x86_64](" in notes


@pytest.mark.parametrize("version", ["1.0.0", "2.4.0rc3"])
def test_release_notes_show_pypi_installation_from_published_receipt(
    version: str,
) -> None:
    manifest = _single_family_release_notes_manifest(
        expected_targets={},
        targets=[],
        release_status="images-failed",
        family_status="failed",
    )
    simple_index = "https://pypi.org/simple/"
    manifest["publish"] = {
        "pypi": {
            "enabled": True,
            "simple_index": simple_index,
            "dependency_index": simple_index,
        }
    }
    meta_project = "uc-manager"
    extra = "cu130"
    backend_project = f"uc-manager-{extra}"
    manifest["pypi"] = {
        "status": "complete",
        "target": "pypi",
        "version": version,
        "extras": {extra: f"{backend_project}=={version}"},
        "projects": [
            {"project": backend_project, "role": "backend"},
            {"project": meta_project, "role": "meta"},
        ],
    }

    notes = release.render_notes(
        manifest, repository="example/ucm", asset_urls=_asset_urls(manifest)
    )

    assert "Status: `images-failed`" in notes
    assert "## PyPI" not in notes
    assert "## TestPyPI" not in notes
    assert f"https://pypi.org/project/{meta_project}/{version}/" in notes
    assert f'pip install "{meta_project}[{extra}]=={version}"' in notes
    assert "python -m" not in notes
    assert "--index-url" not in notes
    assert "uc_manager_cuda-amd64.whl" not in notes
    assert "pip download" not in notes


def test_release_notes_show_testpypi_installation_in_the_wheel_column() -> None:
    manifest = _single_family_release_notes_manifest(expected_targets={}, targets=[])
    simple_index = "https://test.pypi.org/simple/"
    manifest["publish"] = {
        "pypi": {
            "enabled": True,
            "simple_index": simple_index,
        }
    }
    meta_project = "another-fork-uc-manager"
    version = "2.4.0rc3"
    extra = "cu130"
    backend_project = f"{meta_project}-cuda-{extra}"
    manifest["pypi"] = {
        "status": "complete",
        "target": "testpypi",
        "version": version,
        "extras": {extra: f"{backend_project}=={version}"},
        "projects": [
            {"project": backend_project, "role": "backend"},
            {"project": meta_project, "role": "meta"},
        ],
    }

    notes = release.render_notes(
        manifest, repository="example/ucm", asset_urls=_asset_urls(manifest)
    )

    assert "## PyPI" not in notes
    assert "## TestPyPI" not in notes
    assert f"https://test.pypi.org/project/{meta_project}/{version}/" in notes
    assert (
        f"pip install --index-url {simple_index} "
        "--extra-index-url https://pypi.org/simple/ "
        f'"{meta_project}[{extra}]=={version}"'
    ) in notes
    assert "python -m" not in notes
    assert "uc_manager_cuda-amd64.whl" not in notes
    assert "<details>" not in notes
    assert "mktemp" not in notes
    assert "pip download" not in notes


def test_release_notes_reject_published_packages_without_the_runtime_extra() -> None:
    manifest = _single_family_release_notes_manifest(expected_targets={}, targets=[])
    manifest["publish"] = {
        "pypi": {
            "enabled": True,
            "simple_index": "https://pypi.org/simple/",
        }
    }
    manifest["pypi"] = {
        "status": "complete",
        "target": "pypi",
        "version": "1.0.0",
        "extras": {"cu129": "uc-manager-cuda-cu129==1.0.0"},
        "projects": [{"project": "uc-manager", "role": "meta"}],
    }

    with pytest.raises(ValueError, match="no extra for 'cu130'"):
        release.render_notes(
            manifest, repository="example/ucm", asset_urls=_asset_urls(manifest)
        )


def test_release_notes_show_dockerhub_only_after_a_published_target() -> None:
    ghcr_reference = "ghcr.io/example/vllm-openai:release-tag"
    dockerhub_reference = "docker.io/example/vllm-openai:release-tag"
    configured_manifest = _single_family_release_notes_manifest(
        expected_targets={
            "ghcr": ghcr_reference,
            "dockerhub": dockerhub_reference,
        },
        targets=[],
        release_status="artifacts-ready",
        family_status="building",
    )

    before_dockerhub_push = release.render_notes(
        configured_manifest,
        repository="example/ucm",
        asset_urls=_asset_urls(configured_manifest),
    )

    assert "ghcr.io/example/vllm-openai" in before_dockerhub_push
    assert "https://github.com/example/ucm/pkgs/container/vllm-openai" in (
        before_dockerhub_push
    )
    assert "docker.io/example/vllm-openai" not in before_dockerhub_push
    assert "https://hub.docker.com/r/example/vllm-openai" not in before_dockerhub_push

    published_manifest = _single_family_release_notes_manifest(
        expected_targets={
            "ghcr": ghcr_reference,
            "dockerhub": dockerhub_reference,
        },
        targets=[
            {"channel": "ghcr", "reference": ghcr_reference},
            {"channel": "dockerhub", "reference": dockerhub_reference},
        ],
    )
    after_dockerhub_push = release.render_notes(
        published_manifest,
        repository="example/ucm",
        asset_urls=_asset_urls(published_manifest),
    )
    header = next(
        line
        for line in after_dockerhub_push.splitlines()
        if line.startswith("| Runtime capability |")
    )

    assert (
        "[`ghcr.io/example/vllm-openai`]"
        "(https://github.com/example/ucm/pkgs/container/vllm-openai)"
    ) in after_dockerhub_push
    assert (
        "[`docker.io/example/vllm-openai`]"
        "(https://hub.docker.com/r/example/vllm-openai)"
    ) in after_dockerhub_push
    assert "ghcr.io/example/vllm-openai" in header
    assert "docker.io/example/vllm-openai" in header


def test_release_notes_use_dockerhub_target_tag_without_a_ghcr_target() -> None:
    dockerhub_reference = "docker.io/example/vllm-openai:dockerhub-only-tag"
    manifest = _single_family_release_notes_manifest(
        expected_targets={"dockerhub": dockerhub_reference},
        targets=[{"channel": "dockerhub", "reference": dockerhub_reference}],
    )

    notes = release.render_notes(
        manifest,
        repository="example/ucm",
        asset_urls=_asset_urls(manifest),
    )
    header = next(
        line for line in notes.splitlines() if line.startswith("| Runtime capability |")
    )

    assert "docker.io/example/vllm-openai" in header
    assert "ghcr.io/example/vllm-openai" not in notes
    assert "`dockerhub-only-tag`" in notes
    assert "https://hub.docker.com/r/example/vllm-openai" in notes


def test_artifact_manifest_recomputes_deferred_libraries_from_report(
    tmp_path: Path,
) -> None:
    wheels, chart, result_path, _ = _write_artifact_inputs(tmp_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["deferred_external_libraries"] = []
    result_path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(ValueError, match="deferred libraries do not match auditwheel"):
        release.build_release_state(_plan(), wheels, chart, actions_run_id=123)


def test_artifact_manifest_rejects_changed_auditwheel_report(tmp_path: Path) -> None:
    wheels, chart, result_path, _ = _write_artifact_inputs(tmp_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    report_path = result_path.parent / result["auditwheel_report"]["filename"]
    report_path.write_text(
        "changed after wheel-result was recorded\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="report digest does not match"):
        release.build_release_state(_plan(), wheels, chart, actions_run_id=123)


def test_artifact_manifest_requires_immutable_builder_digest(tmp_path: Path) -> None:
    wheels, chart, _, _ = _write_artifact_inputs(tmp_path)
    plan = _plan()
    del plan["wheels"][0]["builder"]["digest"]

    with pytest.raises(ValueError, match="immutable Builder digest"):
        release.build_release_state(plan, wheels, chart, actions_run_id=123)


def test_missing_receipt_keeps_artifacts_available_and_marks_images_failed() -> None:
    manifest = {
        "release": {"git_tag": "v0.7.62rc1", "status": "artifacts-ready"},
        "images": [
            {
                "id": "image-amd64",
                "family_id": "family",
                "status": "building",
                "targets": [],
            }
        ],
        "families": [
            {
                "id": "family",
                "create_index": False,
                "status": "building",
                "targets": [],
            }
        ],
    }
    result = release.finalize_release_state(
        manifest,
        Path("/does/not/exist"),
        build_outcome="failure",
        member_outcome="skipped",
        index_outcome="skipped",
    )
    assert result["release"]["status"] == "images-failed"
    assert result["images"][0]["status"] == "failed"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 999, "invalid contract"),
        ("digest", "garbage", "digest is invalid"),
        ("reference", "ghcr.io/example/wrong:tag", "planned reference"),
    ],
)
def test_published_receipt_must_match_its_schema_and_planned_target(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    wheels, chart, _, _ = _write_artifact_inputs(tmp_path)
    manifest, _ = release.build_release_state(
        _plan(), wheels, chart, actions_run_id=123
    )
    receipt = {
        "kind": "ucm-image-member-receipt",
        "schema_version": 1,
        "id": "vllm-v023-amd64",
        "status": "published",
        "targets": [
            {
                "channel": "ghcr",
                "reference": "ghcr.io/example/vllm:v0.23.0-ucm-amd64",
                "digest": "sha256:" + "b" * 64,
            }
        ],
    }
    if field == "schema_version":
        receipt[field] = value
    else:
        receipt["targets"][0][field] = value
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    (receipts / "member.json").write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        release.finalize_release_state(
            manifest,
            receipts,
            build_outcome="success",
            member_outcome="success",
            index_outcome="skipped",
        )


def test_schema9_accepts_optional_toolkit_and_rejects_wrong_version():
    manifest = json.loads(
        (Path(__file__).parent / "fixtures/release-manifest.json").read_text()
    )
    public_manifest.validate_manifest(manifest)
    version = manifest["release"]["version"]
    distribution = (
        manifest["python"]["distribution"].removesuffix("uc-manager") + "ucm-toolkit"
    )
    filename = f"{distribution.replace('-', '_')}-{version}-py3-none-any.whl"
    manifest["toolkit"] = {
        "distribution": distribution,
        "version": version,
        "filename": filename,
        "sha256": "e" * 64,
        "url": manifest["release"]["url"].replace(
            "/releases/tag/", "/releases/download/"
        )
        + "/"
        + filename,
    }
    manifest["python"]["extras"]["toolkit"] = distribution
    manifest["github_release_assets"].append(filename)
    public_manifest.validate_manifest(manifest)
    manifest["toolkit"]["version"] = "0.0.1"
    with pytest.raises(public_manifest.ManifestError, match="toolkit"):
        public_manifest.validate_manifest(manifest)


class TestCleanupRecords:
    """Generic release records, retention, and destructive inventory contracts."""

    REPOSITORY = "release-org/unified-cache-management"
    TAG = "nightly/v0.9.0-20261009-1"
    VERSION = "0.9.0.dev20261009001"
    IMAGE = f"ghcr.io/release-org/vllm-openai:v1-ucm-{VERSION}"
    CHART = "ghcr.io/release-org/charts/unified-cache-chart:0.9.0-nightly.20261009.1"

    @pytest.fixture(autouse=True, params=["nightly", "draft"])
    def publication_type(self, request):
        if request.param == "nightly":
            self.TAG = "nightly/v0.9.0-20261009-1"
            self.VERSION = "0.9.0.dev20261009001"
            chart_version = "0.9.0-nightly.20261009.1"
        else:
            self.TAG = "draft/v0.9.0-1"
            self.VERSION = "0.9.0.dev1"
            chart_version = "0.9.0-draft.1"
        self.IMAGE = f"ghcr.io/release-org/vllm-openai:v1-ucm-{self.VERSION}"
        self.CHART = "ghcr.io/release-org/charts/unified-cache-chart:" + chart_version

    def _record(self, tag=None, *, run_id=1, attempt=1, release_id=None):
        tag = self.TAG if tag is None else tag
        return cleanup_records.record_for_tag(
            self.REPOSITORY, tag, "a" * 40, run_id, attempt, release_id
        )

    def _plan(self, *, multi_arch=True, ghcr=True, dockerhub=False, chart=True):
        members = (
            [
                {
                    "image_id": "x-amd64",
                    "cpu_arch": "amd64",
                    "reference": self.IMAGE + "-amd64",
                },
                {
                    "image_id": "x-arm64",
                    "cpu_arch": "arm64",
                    "reference": self.IMAGE + "-arm64",
                },
            ]
            if multi_arch
            else [{"image_id": "x-amd64", "cpu_arch": "amd64", "reference": self.IMAGE}]
        )
        return {
            "kind": "ucm-release-plan",
            "repository": self.REPOSITORY,
            "route": "release",
            "release_type": cleanup_records._classification(self.TAG)["release_type"],
            "git_tag": self.TAG,
            "version": self.VERSION,
            "image_version": self.VERSION,
            "publish": {
                "ghcr": {"enabled": ghcr},
                "dockerhub": {
                    "enabled": dockerhub,
                    "namespace": "docker.io/other-owner",
                },
                "chart_oci": {
                    "enabled": chart,
                    "namespace": "ghcr.io/release-org/charts",
                },
            },
            "chart": {
                "name": "unified-cache-chart",
                "version": self.CHART.rsplit(":", 1)[1],
            },
            "families": [
                {
                    "create_index": multi_arch,
                    "published_reference": self.IMAGE,
                    "members": members,
                }
            ],
            "wheels": [
                {
                    "builder": {
                        "repository": "ghcr.io/release-org/builders",
                        "tag": "shared",
                    }
                }
            ],
            "images": [
                {"runtime": {"image_reference": "ghcr.io/release-org/base:latest"}}
            ],
        }

    def _legacy_manifest(self):
        return {
            "kind": "ucm-release-manifest",
            "schema_version": 6,
            "tag": self.TAG,
            "release_type": cleanup_records._classification(self.TAG)["release_type"],
            "actions_run_id": 123,
            "chart_oci": self.CHART,
            "github_release_assets": ["release-manifest.json", "wheel.whl"],
            "runtime_images": {
                "ghcr": {
                    "indexes": [self.IMAGE],
                    "members": [self.IMAGE + "-amd64", self.IMAGE + "-arm64"],
                },
                "dockerhub": {"indexes": [], "members": []},
            },
        }

    def _public_manifest(self, schema_version=9):
        document = json.loads(
            (RELEASE_ROOT / "tests/fixtures/release-manifest.json").read_text()
        )
        text = (
            json.dumps(document)
            .replace("0.9.3", self.VERSION)
            .replace("example", "release-org")
        )
        document = json.loads(text)
        document["release"].update(
            tag=self.TAG,
            type=cleanup_records._classification(self.TAG)["release_type"],
            url=f"https://github.com/{self.REPOSITORY}/releases/tag/{self.TAG}",
        )
        publication = document["images"][0]["publications"]["ghcr"]
        publication["pull"] = self.IMAGE
        publication["members"][0]["reference"] = self.IMAGE + "-amd64"
        publication["members"][1]["reference"] = self.IMAGE + "-arm64"
        document["chart"]["oci"] = self.CHART
        if schema_version == 10:
            document["schema_version"] = schema_version
            for wheel in document["wheels"]:
                wheel["capabilities"] = [
                    {
                        "product": wheel.pop("product"),
                        "accelerator": wheel.pop("accelerator"),
                    }
                ]
        return document

    def test_plan_inventory_contains_enabled_targets_and_excludes_shared_inputs(self):
        record = cleanup_records.record_from_plan(
            self._record(), self._plan(dockerhub=True)
        )
        assert [(item["kind"], item["reference"]) for item in record["resources"]] == [
            ("chart-oci", self.CHART),
            ("ghcr-index", self.IMAGE),
            ("ghcr-member", self.IMAGE + "-amd64"),
            ("ghcr-member", self.IMAGE + "-arm64"),
            (
                "dockerhub-index",
                self.IMAGE.replace("ghcr.io/release-org", "docker.io/other-owner"),
            ),
            (
                "dockerhub-member",
                (self.IMAGE + "-amd64").replace(
                    "ghcr.io/release-org", "docker.io/other-owner"
                ),
            ),
            (
                "dockerhub-member",
                (self.IMAGE + "-arm64").replace(
                    "ghcr.io/release-org", "docker.io/other-owner"
                ),
            ),
        ]

    def test_single_architecture_pull_is_only_one_member(self):
        record = cleanup_records.record_from_plan(
            self._record(), self._plan(multi_arch=False, chart=False)
        )
        assert record["resources"] == [{"kind": "ghcr-member", "reference": self.IMAGE}]

    def test_disabled_publication_has_a_legitimate_empty_inventory(self):
        record = cleanup_records.record_from_plan(
            self._record(), self._plan(ghcr=False, chart=False)
        )
        assert record["resources"] == []

    @pytest.mark.parametrize(
        "field,value",
        [
            ("repository", "other/repo"),
            ("git_tag", "nightly/v0.9.0-20261008-1"),
            ("version", "0.9.0"),
            ("image_version", "0.9.0"),
            ("route", "pr"),
        ],
    )
    def test_plan_identity_cannot_be_grafted_onto_a_different_binding(
        self, field, value
    ):
        plan = self._plan()
        plan[field] = value
        with pytest.raises(
            cleanup_records.CleanupRecordError, match="identity differs"
        ):
            cleanup_records.record_from_plan(self._record(), plan)

    @pytest.mark.parametrize(
        "kind,reference",
        [
            ("ghcr-index", IMAGE.replace("release-org", "other-owner")),
            ("ghcr-index", "ghcr.io/release-org/vllm-openai:v1"),
            ("ghcr-member", IMAGE.replace("20261009001", "20261008001")),
            ("ghcr-member", IMAGE.replace("ghcr.io", "docker.io")),
            ("chart-oci", CHART.replace("20261009", "20261008")),
            ("chart-oci", CHART.replace("release-org", "other-owner")),
            ("pypi", "uc-manager==" + VERSION),
            ("ghcr-member", " " + IMAGE),
            ("ghcr-member", IMAGE + "@sha256:" + "b" * 64),
        ],
    )
    def test_inventory_rejects_cross_version_or_shared_resource_deletion(
        self, kind, reference
    ):
        snapshot = cleanup_records.new_record(self.REPOSITORY, self.TAG, "a" * 40, 1, 1)
        snapshot["resources"] = [{"kind": kind, "reference": reference}]
        with pytest.raises(cleanup_records.CleanupRecordError):
            cleanup_records.decode_snapshot(
                snapshot, self.REPOSITORY, "release-cleanup-1-1-planned.json"
            )

    def test_legacy_projection_does_not_widen_public_manifest_contract(self):
        legacy = self._legacy_manifest()
        record = cleanup_records.record_from_manifest(
            legacy, self.REPOSITORY, release_id=456
        )
        assert record["source_sha"] is None
        assert record["release_ids"] == [456]
        assert record["runs"] == [{"id": 123, "attempt": 1}]
        assert len(record["resources"]) == 4
        with pytest.raises(
            public_manifest.ManifestError, match="schema_version must be 9"
        ):
            public_manifest.validate_manifest(legacy)

    @pytest.mark.parametrize("schema_version", [9, 10])
    def test_current_public_manifest_projects_the_same_internal_inventory(
        self, schema_version
    ):
        record = cleanup_records.record_from_manifest(
            self._public_manifest(schema_version), self.REPOSITORY, 456
        )
        assert (
            record["resources"]
            == cleanup_records.record_from_manifest(
                self._legacy_manifest(), self.REPOSITORY
            )["resources"]
        )

    def test_schema10_sglang_manifest_preserves_exact_release_coordinates(self):
        manifest = json.loads(
            json.dumps(self._public_manifest(10))
            .replace("vllm-openai", "sglang")
            .replace("vllm", "sglang")
        )
        record = cleanup_records.record_from_manifest(manifest, self.REPOSITORY, 456)
        assert record["release_ids"] == [456]
        assert record["versions"] == [self.VERSION]
        assert any("/sglang:" in item["reference"] for item in record["resources"])
        assert record["resources"] == [
            {"kind": resource.kind, "reference": resource.reference}
            for resource in cleanup_records.registry_resources(manifest)
        ]
        if self.TAG.startswith("nightly/"):
            manifest["release"]["tag"] = "nightly/v0.9.0-20261008-1"
            with pytest.raises(
                cleanup_records.CleanupRecordError, match="version differs"
            ):
                cleanup_records.record_from_manifest(manifest, self.REPOSITORY)

    @pytest.mark.parametrize(
        "mutation", ["schema", "fields", "kind", "run", "ownership"]
    )
    def test_legacy_projection_is_strict(self, mutation):
        manifest = self._legacy_manifest()
        if mutation == "schema":
            manifest["schema_version"] = 7
        elif mutation == "fields":
            manifest["untrusted"] = True
        elif mutation == "kind":
            manifest["kind"] = "other"
        elif mutation == "run":
            manifest["actions_run_id"] = True
        else:
            manifest["runtime_images"]["ghcr"]["indexes"] = [
                self.IMAGE.replace("release-org", "someone")
            ]
        with pytest.raises(cleanup_records.CleanupRecordError):
            cleanup_records.record_from_manifest(manifest, self.REPOSITORY)

    def test_duplicate_releases_and_attempts_union_one_version_without_mutating_inputs(
        self,
    ):
        first = self._record(release_id=10)
        second = cleanup_records.record_from_plan(
            self._record(attempt=2, release_id=11), self._plan()
        )
        original = copy.deepcopy([first, second])
        merged = cleanup_records.merge_records([first, second, second], self.REPOSITORY)
        assert len(merged) == 1
        assert merged[0]["release_ids"] == [10, 11]
        assert merged[0]["runs"] == [{"id": 1, "attempt": 1}, {"id": 1, "attempt": 2}]
        assert len(merged[0]["resources"]) == 4
        assert [first, second] == original

    def test_recovered_identity_gains_known_source_but_rejects_conflicting_sources(
        self,
    ):
        legacy = self._record()
        legacy.update(source_sha=None, runs=[], attempt_sources=[])
        merged = cleanup_records.merge_records(
            [legacy, self._record()], self.REPOSITORY
        )
        assert merged[0]["source_sha"] == "a" * 40
        conflicting = self._record()
        conflicting["source_sha"] = "b" * 40
        with pytest.raises(
            cleanup_records.CleanupRecordError, match="conflicting source SHAs"
        ):
            cleanup_records.merge_records(
                [self._record(), conflicting], self.REPOSITORY
            )

    def test_retention_counts_all_versions_once_without_reserving_an_imaginary_current_slot(
        self,
    ):
        items = [
            self._record(f"nightly/v0.9.0-2026100{day}-1", run_id=day)
            for day in range(1, 5)
        ]
        items[0].update(
            source_sha=None, runs=[], attempt_sources=[]
        )  # Recovered manifestless failed Draft.
        items += [
            self._record("nightly/v0.9.0-20261004-1", run_id=4, attempt=2, release_id=9)
        ]
        selected = cleanup_records.select_retention(items, 3)
        assert [item["tag"] for item in selected.kept] == [
            "nightly/v0.9.0-20261004-1",
            "nightly/v0.9.0-20261003-1",
            "nightly/v0.9.0-20261002-1",
        ]
        assert [item["tag"] for item in selected.candidates] == [
            "nightly/v0.9.0-20261001-1"
        ]

    def test_sort_uses_tag_date_then_numeric_sequence_then_numeric_base(self):
        tags = [
            "nightly/v0.9.0-20261009-2",
            "nightly/v0.8.0-20261009-10",
            "nightly/v0.10.0-20261009-10",
            "nightly/v0.1.0-20261010-1",
        ]
        selected = cleanup_records.select_retention(
            [self._record(tag) for tag in tags], 2
        )
        assert [item["tag"] for item in selected.kept] == [tags[3], tags[2]]
        assert [item["tag"] for item in selected.candidates] == [tags[1], tags[0]]

    def test_active_excess_versions_count_toward_quota_and_are_reported_as_deferred(
        self,
    ):
        tags = [f"nightly/v0.9.0-2026100{day}-1" for day in range(1, 5)]
        selected = cleanup_records.select_retention(
            [self._record(tag) for tag in tags], 2, {tags[1]}
        )
        assert [item["tag"] for item in selected.kept] == [tags[3], tags[2]]
        assert [item["tag"] for item in selected.deferred] == [tags[1]]
        assert [item["tag"] for item in selected.candidates] == [tags[0]]

    def test_unlimited_retention_disables_deletion_even_for_old_active_versions(self):
        older = (
            "nightly/v0.9.0-20261008-1"
            if self.TAG.startswith("nightly/")
            else "draft/v0.9.0"
        )
        items = [self._record(), self._record(older)]
        selected = cleanup_records.select_retention(items, -1, {self.TAG})
        assert len(selected.kept) == 2
        assert selected.candidates == selected.deferred == ()

    @pytest.mark.parametrize("count", [0, -2, True, "7"])
    def test_invalid_quota_is_rejected(self, count):
        with pytest.raises(cleanup_records.CleanupRecordError, match="max_count"):
            cleanup_records.select_retention([], count)

    @pytest.mark.parametrize(
        "tag",
        [
            "v0.9",
            "draft/v0.9.0-0",
            "nightly/v0.9.0-20260230-1",
            "nightly/v0.9.0-20261009-01",
        ],
    )
    def test_noncanonical_or_invalid_dates_never_enter_inventory(self, tag):
        with pytest.raises(cleanup_records.CleanupRecordError):
            self._record(tag)

    def test_resource_kind_conflict_and_implicit_schema_extensions_are_rejected(self):
        record = self._record()
        record["resources"] = [
            {"kind": "ghcr-index", "reference": self.IMAGE},
            {"kind": "ghcr-member", "reference": self.IMAGE},
        ]
        with pytest.raises(
            cleanup_records.CleanupRecordError, match="conflicting resource kinds"
        ):
            cleanup_records.validate_record(record, self.REPOSITORY)
        record = self._record()
        record["complete"] = True
        with pytest.raises(cleanup_records.CleanupRecordError, match="fields differ"):
            cleanup_records.validate_record(record, self.REPOSITORY)

    def test_snapshot_roundtrip_preserves_filename_context_and_keeps_wire_schema(self):
        snapshot = cleanup_records.new_record(
            self.REPOSITORY, self.TAG, "a" * 40, 19, 2, plan=self._plan()
        )
        assert set(snapshot) == {
            "kind",
            "schema_version",
            "repository",
            "tag",
            "release_type",
            "run_id",
            "source_sha",
            "version",
            "resources",
        }
        name = cleanup_records.record_basename(snapshot, 2, planned=True)
        decoded = cleanup_records.decode_record(
            snapshot, self.REPOSITORY, asset_name=name, release_id=23
        )
        assert decoded["release_ids"] == [23]
        assert decoded["runs"] == [{"id": 19, "attempt": 2}]
        assert decoded["attempt_sources"] == [
            {"id": 19, "attempt": 2, "source_sha": "a" * 40}
        ]
        assert (
            decoded["resources"]
            == cleanup_records.record_from_plan(self._record(), self._plan())[
                "resources"
            ]
        )

    def test_planned_empty_snapshot_keeps_its_distinct_filename_stage(self):
        snapshot = cleanup_records.new_record(
            self.REPOSITORY,
            self.TAG,
            "a" * 40,
            19,
            2,
            plan=self._plan(ghcr=False, chart=False),
        )
        assert snapshot["resources"] == []
        assert cleanup_records.record_basename(snapshot, 2, planned=True).endswith(
            "-planned.json"
        )
        decoded = cleanup_records.decode_snapshot(
            snapshot,
            self.REPOSITORY,
            cleanup_records.record_basename(snapshot, 2, planned=True),
        )
        assert decoded["runs"] == [{"id": 19, "attempt": 2}]

    def test_moved_draft_tag_keeps_each_attempt_source_and_nightly_rejects_conflict(
        self,
    ):
        first = self._record(run_id=1)
        second = cleanup_records.record_for_tag(
            self.REPOSITORY, self.TAG, "b" * 40, 2, 3
        )
        if self.TAG.startswith("nightly/"):
            with pytest.raises(
                cleanup_records.CleanupRecordError, match="conflicting source SHAs"
            ):
                cleanup_records.merge_records([first, second], self.REPOSITORY)
        else:
            third = cleanup_records.record_for_tag(
                self.REPOSITORY, self.TAG, "c" * 40, 3, 1
            )
            merged = cleanup_records.merge_records(
                [first, second, third], self.REPOSITORY
            )[0]
            assert merged["source_sha"] is None
            assert merged["attempt_sources"] == [
                {"id": 1, "attempt": 1, "source_sha": "a" * 40},
                {"id": 2, "attempt": 3, "source_sha": "b" * 40},
                {"id": 3, "attempt": 1, "source_sha": "c" * 40},
            ]

    def test_snapshot_decoder_rejects_wire_extensions_and_wrong_filename_bindings(self):
        snapshot = cleanup_records.new_record(
            self.REPOSITORY, self.TAG, "a" * 40, 19, 2
        )
        with pytest.raises(cleanup_records.CleanupRecordError, match="provenance"):
            cleanup_records.decode_snapshot(
                snapshot, self.REPOSITORY, "release-cleanup-20-2-opened.json"
            )
        snapshot["attempt_sources"] = []
        with pytest.raises(cleanup_records.CleanupRecordError, match="fields differ"):
            cleanup_records.decode_snapshot(
                snapshot, self.REPOSITORY, "release-cleanup-19-2-opened.json"
            )

    @pytest.mark.parametrize(
        "tag,version",
        [
            ("draft/v0.8.0", "0.8.0.dev0+gabcdef"),
            ("draft/v0.8.0-1", "0.8.0.dev1+gabcdef"),
        ],
    )
    def test_generic_snapshot_preserves_local_version_and_configured_image_repository(
        self, tag, version
    ):
        reference = "ghcr.io/release-org/vllm:v0.23.0-ucm-" + version.replace("+", ".")
        plan = {
            "kind": "ucm-release-plan",
            "route": "release",
            "repository": self.REPOSITORY,
            "git_tag": tag,
            "release_type": "draft",
            "version": version,
            "publish": {
                "ghcr": {"enabled": True},
                "dockerhub": {"enabled": False},
                "chart_oci": {"enabled": False},
            },
            "families": [
                {
                    "create_index": False,
                    "published_reference": reference,
                    "members": [{"reference": reference}],
                }
            ],
        }
        snapshot = cleanup_records.new_record(
            self.REPOSITORY, tag, "a" * 40, 1, 1, plan=plan
        )
        decoded = cleanup_records.decode_snapshot(
            snapshot, self.REPOSITORY, "release-cleanup-1-1-planned.json"
        )
        assert decoded["versions"] == [version]
        assert decoded["resources"] == [{"kind": "ghcr-member", "reference": reference}]

    def test_legacy_nightly_wire_keeps_its_original_resource_and_field_contract(self):
        tag = "nightly/v0.9.0-20261009-1"
        wire = cleanup_records.record_for_tag(self.REPOSITORY, tag, "a" * 40, 1, 1)
        wire = {
            key: value
            for key, value in wire.items()
            if key not in {"attempt_sources", "versions"}
        }
        decoded = cleanup_records.decode_record(wire, self.REPOSITORY)
        assert decoded["runs"] == [{"id": 1, "attempt": 1}]
        wire["resources"] = [
            {
                "kind": "ghcr-member",
                "reference": "ghcr.io/release-org/vllm:v1-ucm-0.9.0.dev20261009001",
            }
        ]
        with pytest.raises(
            cleanup_records.CleanupRecordError, match="publication target"
        ):
            cleanup_records.decode_record(wire, self.REPOSITORY)
        wire["resources"] = [
            {
                "kind": "ghcr-index",
                "reference": "ghcr.io/release-org/vllm-openai:v1-ucm-0.9.0.dev20261009001-amd64",
            }
        ]
        with pytest.raises(cleanup_records.CleanupRecordError, match="Nightly version"):
            cleanup_records.decode_record(wire, self.REPOSITORY)
        wire["resources"] = []
        wire["attempt_sources"] = []
        with pytest.raises(cleanup_records.CleanupRecordError, match="fields differ"):
            cleanup_records.decode_record(wire, self.REPOSITORY)

    def test_generic_retention_protects_current_without_replacing_old_deletion_boundary(
        self,
    ):
        tags = [f"draft/v0.8.0-{number}" for number in range(1, 5)]
        items = [
            cleanup_records.record_for_tag(
                self.REPOSITORY, tag, "a" * 40, index, 1, index
            )
            for index, tag in enumerate(tags, 1)
        ]
        selected = cleanup_records.select_retention(
            items, 3, release_type="draft", current_tag=tags[0]
        )
        assert selected.candidates == ()
        selected = cleanup_records.select_retention(
            items[:3], 3, release_type="draft", current_tag=tags[3]
        )
        assert [record["tag"] for record in selected.candidates] == [tags[0]]

    def test_schema9_ordinary_manifest_keeps_its_authoritative_exact_coordinates(self):
        manifest = json.loads(
            (RELEASE_ROOT / "tests/fixtures/release-manifest.json").read_text()
        )
        manifest = json.loads(json.dumps(manifest).replace("example", "release-org"))
        manifest["release"].update(
            tag="draft/v0.8.0-1",
            type="draft",
            url=f"https://github.com/{self.REPOSITORY}/releases/tag/untagged-old-display",
        )
        decoded = cleanup_records.record_from_manifest(
            manifest, self.REPOSITORY, release_id=17
        )
        assert decoded["tag"] == "draft/v0.8.0-1"
        assert decoded["versions"] == ["0.9.3"]
        expected = cleanup_records.registry_resources(manifest)
        assert {(item["kind"], item["reference"]) for item in decoded["resources"]} == {
            (item.kind, item.reference) for item in expected
        }
        manifest["release"]["url"] = "https://github.com/other/repo/releases/tag/v0.9.3"
        with pytest.raises(
            cleanup_records.CleanupRecordError, match="another repository"
        ):
            cleanup_records.record_from_manifest(manifest, self.REPOSITORY)

    def test_schema6_reader_preserves_each_historical_asset_name_contract(self):
        manifest = self._legacy_manifest()
        manifest["github_release_assets"].append("nested/wheel.whl")
        if self.TAG.startswith("nightly/"):
            with pytest.raises(
                cleanup_records.CleanupRecordError, match="Release assets"
            ):
                cleanup_records.record_from_manifest(manifest, self.REPOSITORY)
        else:
            assert (
                cleanup_records.record_from_manifest(manifest, self.REPOSITORY)["tag"]
                == self.TAG
            )
        manifest["github_release_assets"] = [
            "release-manifest.json",
            "wheel.whl",
            "wheel.whl",
        ]
        if self.TAG.startswith("nightly/"):
            assert (
                cleanup_records.record_from_manifest(manifest, self.REPOSITORY)["tag"]
                == self.TAG
            )
        else:
            with pytest.raises(
                cleanup_records.CleanupRecordError, match="Release assets"
            ):
                cleanup_records.record_from_manifest(manifest, self.REPOSITORY)
