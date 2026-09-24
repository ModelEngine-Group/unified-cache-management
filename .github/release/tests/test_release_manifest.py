from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
RELEASE_ROOT = ROOT / ".github" / "release"
sys.path.insert(0, str(RELEASE_ROOT))
release = importlib.import_module("ucm_release.release")
public_manifest = importlib.import_module("ucm_release.manifest")


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
    from ucm_release import cleanup

    assert cleanup.registry_resources(public) == []
    assert public["github_release_assets"] == sorted(
        [filename, "release-manifest.json"]
    )


def test_public_manifest_is_exact_schema_v10_and_uses_published_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    from ucm_release import cleanup

    assert cleanup.validate_manifest(pypi_manifest) == pypi_manifest
    assert cleanup.registry_resources(pypi_manifest)
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
