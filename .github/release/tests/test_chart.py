"""Release image defaults and alternatives survive actual Helm packaging."""

from __future__ import annotations

import copy
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / ".github/release"))
chart = importlib.import_module("ucm_release.chart")
runtime = importlib.import_module("ucm_release.runtime")
CHART = ROOT / "charts/unified-cache-chart"


def _family(
    product, version, accelerator, *, suffix="", architectures=("amd64", "arm64")
):
    upstream_tag = f"v{version}{suffix}"
    repository = f"ghcr.io/test-fork/{'vllm-openai' if product == 'vllm' else product}"
    reference = f"{repository}:test-fork-{upstream_tag}-ucm-0.7.0rc1"
    return {
        "product_id": product,
        "variant": "cuda" if product == "vllm" else "a3",
        "label": f"{product} {version} · {accelerator} · Ubuntu 22.04",
        "runtime": {
            "version": version,
            "tag": upstream_tag,
            "channel": "rc" if "rc" in version else "stable",
            "accelerator_runtime": accelerator,
            "os_version": "22.04",
        },
        "published_reference": reference,
        "members": [
            {
                "cpu_arch": architecture,
                "reference": reference
                + (f"-{architecture}" if len(architectures) > 1 else ""),
            }
            for architecture in architectures
        ],
    }


@pytest.fixture
def plan():
    return {
        "route": "release",
        "publish": {
            "ghcr": {"enabled": True},
            "dockerhub": {"enabled": True, "namespace": "docker.io/release-images"},
        },
        "chart": {
            "source": "charts/unified-cache-chart",
            "name": "unified-cache-chart",
            "version": "0.7.0-rc.1",
            "app_version": "0.7.0rc1",
        },
        "families": [
            _family("vllm", "0.25.0", "cuda-13.0"),
            _family("vllm", "0.26.0", "cuda-12.9"),
            _family("vllm", "0.26.0", "cuda-13.0", suffix="-cu130"),
            _family("vllm", "0.27.0rc1", "cuda-13.0"),
            _family(
                "vllm-ascend",
                "0.26.0",
                "cann-9.0.1",
                suffix="-a3",
                architectures=("arm64",),
            ),
        ],
    }


@pytest.mark.parametrize("channel", ["dockerhub", "ghcr"])
def test_default_and_all_alternatives_use_preferred_publication(plan, channel):
    plan["publish"]["dockerhub"]["enabled"] = channel == "dockerhub"
    source = (CHART / "values.yaml").read_text()
    rendered = chart.render_values(plan, source)
    default = runtime.image_publication_targets(
        plan, plan["families"][1]["published_reference"]
    )[channel]
    before, after = yaml.safe_load(source), yaml.safe_load(rendered)
    assert after["images"]["image"] == default
    after["images"]["image"] = before["images"]["image"]
    assert after == before
    assert all(
        line in rendered
        for line in source.splitlines()
        if line.lstrip().startswith("#")
    )
    alternatives = re.findall(r'^  # image: "([^"]+)"$', rendered, re.M)
    expected = {
        runtime.image_publication_targets(plan, reference)[channel]
        for family in plan["families"]
        for reference in [
            family["published_reference"],
            *(m["reference"] for m in family["members"]),
        ]
    }
    assert set(alternatives) == expected - {default}
    assert len(alternatives) == len(set(alternatives))
    assert "arm64" in rendered and "amd64" in rendered
    plan["families"].reverse()
    assert chart.render_values(plan, source) == rendered


def test_default_without_upstream_default_uses_highest_cuda_and_os(plan):
    plan["families"].pop(1)
    plan["families"][1]["runtime"]["os_version"] = "unreported"
    newest_os = copy.deepcopy(plan["families"][1])
    newest_os["runtime"]["os_version"] = "24.04"
    newest_os["published_reference"] += "-ubuntu2404"
    newest_os["members"] = [
        {"cpu_arch": "amd64", "reference": newest_os["published_reference"]}
    ]
    plan["families"].append(newest_os)
    rendered = chart.render_values(plan, (CHART / "values.yaml").read_text())
    assert yaml.safe_load(rendered)["images"]["image"].endswith("-ubuntu2404")


def test_no_stable_cuda_keeps_alternatives_without_default(plan):
    plan["families"] = plan["families"][-2:]
    rendered = chart.render_values(plan, (CHART / "values.yaml").read_text())
    assert yaml.safe_load(rendered)["images"]["image"] == ""
    assert "没有稳定 CUDA" in rendered
    assert "# image:" in rendered


@pytest.mark.parametrize("case", ["pr", "images-disabled"])
def test_non_release_or_disabled_images_leave_values_unchanged(plan, case):
    if case == "pr":
        plan["route"] = "pr"
    else:
        for channel in plan["publish"].values():
            channel["enabled"] = False
    source = (CHART / "values.yaml").read_text()
    assert chart.render_values(plan, source) == source


def _helm(*args, check=True):
    return subprocess.run(
        ["helm", *map(str, args)], text=True, capture_output=True, check=check
    )


def _images(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "image" and isinstance(child, str):
                yield child
            else:
                yield from _images(child)
    elif isinstance(value, list):
        for child in value:
            yield from _images(child)


@pytest.mark.skipif(shutil.which("helm") is None, reason="Helm is required")
def test_packaged_chart_defaults_overrides_and_ascend_profiles(plan, tmp_path):
    source_before = (CHART / "values.yaml").read_bytes()
    plan_path = tmp_path / "release-plan.json"
    plan_path.write_text(json.dumps(plan))
    staged = tmp_path / "chart"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ucm_release",
            "chart",
            "prepare",
            "--plan",
            str(plan_path),
            "--output",
            str(staged),
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / ".github/release")},
        check=True,
        capture_output=True,
        text=True,
    )
    assert (CHART / "values.yaml").read_bytes() == source_before
    _helm("lint", staged)
    _helm(
        "package",
        staged,
        "--version",
        plan["chart"]["version"],
        "--app-version",
        plan["chart"]["app_version"],
        "--destination",
        tmp_path,
    )
    package = next(tmp_path.glob("*.tgz"))
    with tarfile.open(package) as archive:
        values = archive.extractfile("unified-cache-chart/values.yaml").read()
        metadata = yaml.safe_load(archive.extractfile("unified-cache-chart/Chart.yaml"))
    assert values == (staged / "values.yaml").read_bytes()
    assert metadata["version"] == plan["chart"]["version"]
    assert metadata["appVersion"] == plan["chart"]["app_version"]
    assert _helm("show", "values", package).stdout.rstrip() == values.decode().rstrip()
    default = yaml.safe_load(values)["images"]["image"]
    assert '# image: "docker.io/' in values.decode()
    for name in ["values-qwen3-0p6b-1e1.yaml", "values-qwen3-0p6b-1p1-1d1.yaml"]:
        profile = staged / "models/cuda" / name
        rendered = list(
            yaml.safe_load_all(_helm("template", "qwen", package, "-f", profile).stdout)
        )
        assert set(_images(rendered)) == {default}
        assert len(list(_images(rendered))) == (1 if "1e1" in name else 3)
    cuda = staged / "models/cuda/values-qwen3-0p6b-1e1.yaml"
    for overrides, expected in [
        (
            ["--set-string", "images.image=example.com/custom:v1"],
            "example.com/custom:v1",
        ),
        (
            [
                "--set-string",
                "images.image=example.com/custom:v1",
                "--set-string",
                "servingEngineSpec.modelSpec.image=example.com/model:v2",
            ],
            "example.com/model:v2",
        ),
    ]:
        rendered = list(
            yaml.safe_load_all(
                _helm("template", "qwen", package, "-f", cuda, *overrides).stdout
            )
        )
        assert set(_images(rendered)) == {expected}
    for profile in sorted((staged / "models/ascend").glob("*.yaml")):
        missing = _helm("template", "qwen", package, "-f", profile, check=False)
        assert missing.returncode != 0 and "images.image" in missing.stderr
        rendered = list(
            yaml.safe_load_all(
                _helm(
                    "template",
                    "qwen",
                    package,
                    "-f",
                    profile,
                    "--set-string",
                    "images.image=example.com/ascend:v1",
                ).stdout
            )
        )
        assert set(_images(rendered)) == {"example.com/ascend:v1"}


def test_unreported_os_versions_do_not_block_chart_packaging(plan):
    for family in plan["families"]:
        family["runtime"]["os_version"] = "unreported"
    rendered = chart.render_values(plan, (CHART / "values.yaml").read_text())
    expected = runtime.image_publication_targets(
        plan, plan["families"][1]["published_reference"]
    )["dockerhub"]
    assert yaml.safe_load(rendered)["images"]["image"] == expected
