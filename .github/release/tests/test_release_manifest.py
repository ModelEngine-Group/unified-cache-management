from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
MODULE = ROOT / ".github" / "release" / "ucm_release" / "release.py"
SPEC = importlib.util.spec_from_file_location("ucm_release_manifest", MODULE)
assert SPEC is not None and SPEC.loader is not None
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


def _plan() -> dict[str, object]:
    return {
        "git_tag": "v0.7.62rc1",
        "release_kind": "publish",
        "version": "0.7.62rc1",
        "publish": {
            "ghcr": {"enabled": True},
            "dockerhub": {
                "enabled": False,
                "namespace": "docker.io/example",
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
    filename = "uc_manager_cuda_cu129-0.7.62rc1-cp312-cp312-linux_x86_64.whl"
    (wheels / filename).write_bytes(b"wheel")
    report_text = "\n".join(
        (
            f"{filename} is consistent with the following platform tag: "
            '"linux_x86_64".',
            "",
            "The wheel references external versioned symbols in these system-provided "
            "shared libraries: libc.so.6 with versions {'GLIBC_2.2.5', 'GLIBC_2.17'}",
            "",
            'This constrains the platform tag to "manylinux_2_27_x86_64".',
            "",
            "The following external shared libraries are required by the wheel:",
            json.dumps({"libcudart.so.13": None, "libmetrics.so": None}, indent=4),
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
                "schema_version": 2,
                "task_id": "cuda129-cp312-amd64",
                "distribution": "uc-manager-cuda-cu129",
                "version": "0.7.62rc1",
                "python_abi": "cp312",
                "cpu_arch": "amd64",
                "filename": filename,
                "platform_tags": ["linux_x86_64"],
                "auditwheel_platform_tag": "linux_x86_64",
                "glibc_versions": ["GLIBC_2.2.5", "GLIBC_2.17"],
                "glibc_floor": "GLIBC_2.17",
                "external_libraries": ["libcudart.so.13", "libmetrics.so"],
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
    filenames.add(str(manifest["chart"]["filename"]))  # type: ignore[index]
    return {
        filename: f"https://github.com/example/ucm/releases/download/v1/{filename}"
        for filename in filenames
    }


def test_artifacts_and_image_receipts_form_one_mapping(tmp_path: Path) -> None:
    wheels, chart, _, filename = _write_artifact_inputs(tmp_path)

    manifest, checksums = release.build_artifacts_manifest(_plan(), wheels, chart)
    assert manifest["release"]["status"] == "artifacts-ready"
    assert manifest["wheels"][0]["platform_tags"] == ["linux_x86_64"]
    assert manifest["wheels"][0]["auditwheel_platform_tag"] == "linux_x86_64"
    assert manifest["wheels"][0]["builder"]["source_image_digest"] == ("sha256:builder")
    assert manifest["wheels"][0]["builder"]["digest"] == "sha256:" + "c" * 64
    assert manifest["images"][0]["wheel_id"] == "cuda129-cp312-amd64"
    asset_urls = _asset_urls(manifest)
    notes = release.render_notes(
        manifest, repository="example/ucm", asset_urls=asset_urls
    )
    assert "Runtime: `docker.io/vllm/vllm-openai`" in notes
    assert "`v0.23.0`" in notes
    assert f"[{filename}]({asset_urls[filename]})" in notes
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
    final = release.finalize_manifest(
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


def test_release_commands_write_internal_state_without_public_metadata_assets(
    tmp_path: Path,
) -> None:
    wheels, chart, _, _ = _write_artifact_inputs(tmp_path)
    plan_path = tmp_path / "release-plan.json"
    plan_path.write_text(json.dumps(_plan()), encoding="utf-8")
    artifacts_output = tmp_path / "artifacts-output"
    artifacts_output.mkdir()
    artifacts = release.build_parser().parse_args(
        [
            "artifacts",
            "--plan",
            str(plan_path),
            "--wheels",
            str(wheels),
            "--chart",
            str(chart),
            "--output",
            str(artifacts_output),
        ]
    )

    artifacts.func(artifacts)

    assert {path.name for path in artifacts_output.iterdir()} == {"release-state.json"}
    state_path = artifacts_output / "release-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["chart"]["name"] == "unified-cache-chart"
    assert state["chart"]["sha256"] == hashlib.sha256(b"chart").hexdigest()
    assert state["wheels"][0]["sha256"] == hashlib.sha256(b"wheel").hexdigest()

    final_output = tmp_path / "final-output"
    final_output.mkdir()
    finalize = release.build_parser().parse_args(
        [
            "finalize",
            "--manifest",
            str(state_path),
            "--receipts",
            str(tmp_path / "missing-receipts"),
            "--build-outcome",
            "failure",
            "--member-outcome",
            "skipped",
            "--index-outcome",
            "skipped",
            "--output",
            str(final_output),
        ]
    )

    finalize.func(finalize)

    assert {path.name for path in final_output.iterdir()} == {"release-state.json"}


def test_release_notes_split_products_and_aggregate_wheel_capabilities() -> None:
    manifest = {
        "release": {"git_tag": "v1.0.0rc1", "status": "complete"},
        "chart": {"filename": "unified-cache-chart-1.0.0-rc.1.tgz"},
        "wheels": [
            {
                "id": "cuda-amd64",
                "filename": "uc_manager_cuda-amd64.whl",
                "backend": "cuda",
                "runtime_variant": "cu130",
                "python_abi": "cp312",
                "cpu_arch": "amd64",
            },
            {
                "id": "cuda-arm64",
                "filename": "uc_manager_cuda-arm64.whl",
                "backend": "cuda",
                "runtime_variant": "cu130",
                "python_abi": "cp312",
                "cpu_arch": "arm64",
            },
            {
                "id": "cann-amd64",
                "filename": "uc_manager_cann910_a2-amd64.whl",
                "backend": "cann-a2",
                "runtime_variant": "cann910-a2",
                "python_abi": "cp312",
                "cpu_arch": "amd64",
            },
            {
                "id": "cann-arm64",
                "filename": "uc_manager_cann910_a2-arm64.whl",
                "backend": "cann-a2",
                "runtime_variant": "cann910-a2",
                "python_abi": "cp312",
                "cpu_arch": "arm64",
            },
        ],
        "images": [
            {
                "family_id": "openai-v1",
                "wheel_id": "cuda-amd64",
                "cpu_arch": "amd64",
            },
            {
                "family_id": "openai-v1",
                "wheel_id": "cuda-arm64",
                "cpu_arch": "arm64",
            },
            {
                "family_id": "openai-v2",
                "wheel_id": "cuda-amd64",
                "cpu_arch": "amd64",
            },
            {
                "family_id": "openai-v2",
                "wheel_id": "cuda-arm64",
                "cpu_arch": "arm64",
            },
            {
                "family_id": "ascend-v1",
                "wheel_id": "cann-arm64",
                "cpu_arch": "arm64",
            },
            {
                "family_id": "ascend-v2",
                "wheel_id": "cann-amd64",
                "cpu_arch": "amd64",
            },
            {
                "family_id": "ascend-v2",
                "wheel_id": "cann-arm64",
                "cpu_arch": "arm64",
            },
        ],
        "families": [
            {
                "id": "openai-v1",
                "runtime": {
                    "repository": "docker.io/vllm/vllm-openai",
                    "tag": "v1.0.0",
                    "accelerator_runtime": "cuda-13.0",
                },
                "expected_targets": {"ghcr": "ghcr.io/example/vllm-openai:v1.0.0-ucm"},
                "status": "published",
                "targets": [{"channel": "ghcr"}],
            },
            {
                "id": "openai-v2",
                "runtime": {
                    "repository": "docker.io/vllm/vllm-openai",
                    "tag": "v1.1.0",
                    "accelerator_runtime": "cuda-13.0",
                },
                "expected_targets": {"ghcr": "ghcr.io/example/vllm-openai:v1.1.0-ucm"},
                "status": "published",
                "targets": [{"channel": "ghcr"}],
            },
            {
                "id": "ascend-v1",
                "runtime": {
                    "repository": "quay.io/ascend/vllm-ascend",
                    "tag": "v1.0.0",
                    "accelerator_runtime": "cann-9.1.0",
                },
                "expected_targets": {"ghcr": "ghcr.io/example/vllm-ascend:v1.0.0-ucm"},
                "status": "published",
                "targets": [{"channel": "ghcr"}],
            },
            {
                "id": "ascend-v2",
                "runtime": {
                    "repository": "quay.io/ascend/vllm-ascend",
                    "tag": "v1.1.0",
                    "accelerator_runtime": "cann-9.1.0",
                },
                "expected_targets": {"ghcr": "ghcr.io/example/vllm-ascend:v1.1.0-ucm"},
                "status": "published",
                "targets": [{"channel": "ghcr"}],
            },
        ],
    }
    asset_urls = _asset_urls(manifest)

    notes = release.render_notes(
        manifest, repository="example/ucm", asset_urls=asset_urls
    )

    assert notes.count("## vLLM OpenAI") == 1
    assert notes.count("## vLLM-Ascend") == 1
    assert notes.count("| Runtime capability |") == 2
    assert notes.count("uc_manager_cuda-amd64.whl") == 2
    assert "CANN 9.1.0 / A2" in notes
    assert "`v1.0.0` (aarch64 only)" in notes
    assert "pkgs/container/vllm-openai" in notes
    assert "pkgs/container/vllm-ascend" in notes
    assert "2 image families / 4 architecture members" in notes
    assert "2 image families / 3 architecture members" in notes
    assert " tags / " not in notes


def test_github_asset_urls_require_only_wheel_and_chart() -> None:
    manifest = {
        "release": {"git_tag": "draft/v1.0.0-1"},
        "chart": {"filename": "unified-cache-chart.tgz"},
        "wheels": [{"filename": "ucm.whl"}],
    }
    base = "https://github.com/example/ucm/releases/download/untagged-1234567890abcdef"
    release_document = {
        "tag_name": "draft/v1.0.0-1",
        "assets": [
            {"name": name, "browser_download_url": f"{base}/{name}"}
            for name in (
                "ucm.whl",
                "unified-cache-chart.tgz",
            )
        ],
    }

    urls = release._github_asset_urls(manifest, release_document)

    assert urls["ucm.whl"] == f"{base}/ucm.whl"
    assert urls["unified-cache-chart.tgz"] == (f"{base}/unified-cache-chart.tgz")
    assert "draft%2F" not in urls["ucm.whl"]


@pytest.mark.parametrize(
    ("field", "value"),
    (("kind", "wrong-wheel-result"), ("schema_version", 1)),
)
def test_artifact_manifest_rejects_wrong_wheel_result_contract(
    tmp_path: Path, field: str, value: object
) -> None:
    wheels, chart, result_path, _ = _write_artifact_inputs(tmp_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result[field] = value
    result_path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(ValueError, match="ucm-wheel-result schema 2"):
        release.build_artifacts_manifest(_plan(), wheels, chart)


@pytest.mark.parametrize(
    "field",
    (
        "platform_tags",
        "auditwheel_platform_tag",
        "glibc_versions",
        "glibc_floor",
        "external_libraries",
        "auditwheel_report",
    ),
)
def test_artifact_manifest_requires_wheel_audit_fields(
    tmp_path: Path, field: str
) -> None:
    wheels, chart, result_path, _ = _write_artifact_inputs(tmp_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    del result[field]
    result_path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(ValueError, match="missing audit fields"):
        release.build_artifacts_manifest(_plan(), wheels, chart)


def test_artifact_manifest_rejects_changed_auditwheel_report(tmp_path: Path) -> None:
    wheels, chart, result_path, _ = _write_artifact_inputs(tmp_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    report_path = result_path.parent / result["auditwheel_report"]["filename"]
    report_path.write_text(
        "changed after wheel-result was recorded\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="report digest does not match"):
        release.build_artifacts_manifest(_plan(), wheels, chart)


def test_artifact_manifest_requires_immutable_builder_digest(tmp_path: Path) -> None:
    wheels, chart, _, _ = _write_artifact_inputs(tmp_path)
    plan = _plan()
    del plan["wheels"][0]["builder"]["digest"]

    with pytest.raises(ValueError, match="immutable Builder digest"):
        release.build_artifacts_manifest(plan, wheels, chart)


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
    result = release.finalize_manifest(
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
    manifest, _ = release.build_artifacts_manifest(_plan(), wheels, chart)
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
        release.finalize_manifest(
            manifest,
            receipts,
            build_outcome="success",
            member_outcome="success",
            index_outcome="skipped",
        )
