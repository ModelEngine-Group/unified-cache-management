from __future__ import annotations

import copy
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
plan = importlib.import_module("ucm_release.plan")


def _plan():
    wheel = {
        "id": "cann920-a2-cp312-amd64",
        "dist_name": "uc-manager-cann920-a2",
        "build": {"platform_arg": "ascend"},
        "builder": {
            "source_image": "quay.io/ascend/manylinux:9.2.0-beta.2-910b-manylinux_2_34-py3.12"
        },
    }
    image = {
        "id": "ascend-ubuntu-amd64",
        "label": "A2 Ubuntu amd64",
        "runner": "ubuntu-24.04",
        "cpu_arch": "amd64",
        "wheel_id": wheel["id"],
        "runtime": {
            "accelerator_runtime": "cann-9.2.0",
            "image_reference": "quay.io/ascend/vllm-ascend@sha256:" + "a" * 64,
        },
    }
    return {"kind": "ucm-release-plan", "wheels": [wheel], "images": [image]}


def test_matrix_covers_all_runtime_members_for_shared_wheel():
    document = _plan()
    second = copy.deepcopy(document["images"][0])
    second["id"] = "ascend-openeuler-amd64"
    second["label"] = "A2 openEuler amd64"
    second["runtime"]["image_reference"] = (
        "quay.io/ascend/vllm-ascend@sha256:" + "b" * 64
    )
    document["images"].append(second)
    selected = plan.fallback_runtime_matrix(document)["include"]
    assert len(selected) == len(document["images"])
    assert {item["runtime_image"] for item in selected} == {
        image["runtime"]["image_reference"] for image in document["images"]
    }
    assert all(item["runtime_cann_version"] == "9.2.0" for item in selected)
    assert all(item["builder_cann_version"] == "9.2.0-beta.2" for item in selected)
    assert all(item["wheel_id"] == document["wheels"][0]["id"] for item in selected)


@pytest.mark.parametrize("version", ["9.2.0", "9.2.0-beta.2"])
def test_exact_cann_version_needs_no_fallback_validation(version):
    document = _plan()
    document["images"][0]["runtime"]["accelerator_runtime"] = "cann-" + version
    document["wheels"][0]["builder"][
        "source_image"
    ] = f"quay.io/ascend/manylinux:{version}-910b-manylinux_2_34-py3.12"
    assert plan.fallback_runtime_matrix(document) == {"include": []}


def test_cuda_needs_no_cann_fallback_validation():
    document = _plan()
    document["images"][0]["runtime"]["accelerator_runtime"] = "cuda-13.0"
    assert plan.fallback_runtime_matrix(document) == {"include": []}


@pytest.mark.parametrize("version", ["9.1.0", "9.2.0-beta.1"])
def test_matrix_rejects_plan_with_unapproved_version_substitution(version):
    document = _plan()
    document["images"][0]["runtime"]["accelerator_runtime"] = "cann-" + version
    with pytest.raises(ValueError, match="unsupported CANN Builder substitution"):
        plan.fallback_runtime_matrix(document)
