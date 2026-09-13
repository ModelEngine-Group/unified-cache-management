"""Prepare the release Chart with its default and alternative runtime images."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from packaging.version import Version

from . import runtime, serialization

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _preferred_reference(plan: dict[str, Any], reference: str) -> str:
    targets = runtime.image_publication_targets(plan, reference)
    return targets.get("dockerhub") or targets.get("ghcr", "")


def _default_family(families: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        family
        for family in families
        if family["product_id"] == "vllm"
        and family["runtime"]["channel"] == "stable"
        and family["runtime"]["accelerator_runtime"].startswith("cuda-")
    ]
    return max(
        candidates,
        key=lambda family: (
            Version(family["runtime"]["version"]),
            family["runtime"]["tag"] == f"v{family['runtime']['version']}",
            Version(family["runtime"]["accelerator_runtime"].removeprefix("cuda-")),
            # Missing OCI OS labels remain valid candidates, below reported OS
            # versions only when the other default-selection criteria tie.
            (
                ()
                if family["runtime"]["os_version"] == "unreported"
                else Version(family["runtime"]["os_version"]).release
            ),
            family["published_reference"],
        ),
        default=None,
    )


def render_values(plan: dict[str, Any], source: str) -> str:
    """Fill only the image setting, preserving all other values and comments."""
    if plan["route"] != "release":
        return source
    families = sorted(
        plan["families"],
        key=lambda family: (
            family["product_id"],
            Version(family["runtime"]["version"]),
            family["runtime"]["accelerator_runtime"],
            family["variant"],
            family["label"],
            family["published_reference"],
        ),
    )
    choices: dict[str, str] = {}
    for family in families:
        members = sorted(family["members"], key=lambda member: member["cpu_arch"])
        references = [
            (family["published_reference"], ", ".join(m["cpu_arch"] for m in members)),
            *((member["reference"], member["cpu_arch"]) for member in members),
        ]
        for reference, architectures in references:
            preferred = _preferred_reference(plan, reference)
            if preferred:
                choices.setdefault(preferred, f"{family['label']} · {architectures}")
    if not choices:
        return source

    default = _default_family(families)
    default_image = (
        _preferred_reference(plan, default["published_reference"]) if default else ""
    )
    default_label = choices.pop(default_image, None)
    lines = [
        (
            f"  # 默认：{default_label}"
            if default_label
            else "  # 本次发布没有稳定 CUDA 默认镜像，请从下列候选指定。"
        ),
        f"  image: {json.dumps(default_image)}",
    ]
    if choices:
        lines.extend(["", "  # 本次发布的其他运行时镜像，复制地址替换上面的 image："])
        for reference, label in choices.items():
            lines.extend([f"  # {label}", f"  # image: {json.dumps(reference)}"])
        lines.append("")

    # The source Chart deliberately leaves this scalar empty. Replacing only
    # this line keeps its hand-written configuration documentation intact.
    result, count = re.subn(
        r'^  image: ""$', lambda _: "\n".join(lines), source, flags=re.M
    )
    if count != 1:
        raise ValueError(
            "source Chart must contain exactly one empty images.image setting"
        )
    return result


def prepare_chart(plan: dict[str, Any], output: Path) -> dict[str, str]:
    """Copy the source Chart and materialize image choices in the package copy."""
    source = REPOSITORY_ROOT / plan["chart"]["source"]
    shutil.copytree(source, output)
    values_path = output / "values.yaml"
    values_path.write_text(
        render_values(plan, values_path.read_text(encoding="utf-8")), encoding="utf-8"
    )
    return {
        "source": str(output),
        "image": serialization.load_yaml(values_path)["images"]["image"],
    }
