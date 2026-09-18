"""Verify published Chart bytes, rendered images and Registry readback."""

import argparse
import hashlib
import json
import re
import subprocess
import tarfile
import tempfile
from pathlib import Path

import yaml
from ucm_release import chart, runtime


def command(*args, check=True):
    return subprocess.run(
        list(map(str, args)), text=True, capture_output=True, check=check
    )


def images(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "image" and isinstance(child, str):
                yield child
            else:
                yield from images(child)
    elif isinstance(value, list):
        for child in value:
            yield from images(child)


def verify(plan, state, package):
    if state["release"]["status"] != "complete":
        raise ValueError("Chart delivery requires a completed release")
    digest = hashlib.sha256(package.read_bytes()).hexdigest()
    if digest != state["chart"]["sha256"]:
        raise ValueError("published Chart bytes differ from the built Chart")
    with tempfile.TemporaryDirectory(prefix="ucm-chart-readback-") as directory:
        with tarfile.open(package) as archive:
            archive.extractall(directory, filter="data")
        root = Path(directory) / plan["chart"]["name"]
        metadata = yaml.safe_load((root / "Chart.yaml").read_text())
        if (metadata["version"], metadata["appVersion"]) != (
            plan["chart"]["version"],
            plan["chart"]["app_version"],
        ):
            raise ValueError("published Chart versions differ from the release plan")
        values_text = (root / "values.yaml").read_text()
        expected = chart.render_values(
            plan,
            (
                chart.REPOSITORY_ROOT / plan["chart"]["source"] / "values.yaml"
            ).read_text(),
        )
        if values_text != expected:
            raise ValueError(
                "Chart default image or candidate comments differ from the release plan"
            )
        default = yaml.safe_load(values_text)["images"]["image"]
        references = {
            json.loads(match)
            for match in re.findall(r'^  (?:# )?image: (".*")$', values_text, re.M)
        } - {""}
        targets = {
            target["reference"]: target["digest"]
            for item in [*state["images"], *state["families"]]
            for target in item["targets"]
        }
        architectures = {}
        ascend_image = None
        for family in plan["families"]:
            preferred = runtime.image_publication_targets(
                plan, family["published_reference"]
            )
            pull = preferred.get("dockerhub") or preferred.get("ghcr")
            if not pull:
                continue
            architectures[pull] = {member["cpu_arch"] for member in family["members"]}
            if family["product_id"] == "vllm-ascend":
                ascend_image = ascend_image or pull
            for member in family["members"]:
                preferred = runtime.image_publication_targets(plan, member["reference"])
                reference = preferred.get("dockerhub") or preferred.get("ghcr")
                architectures[reference] = {member["cpu_arch"]}
        if references != architectures.keys():
            raise ValueError(
                "Chart image candidates do not cover the published deployment images"
            )
        verified = []
        for reference in sorted(references):
            actual = command("crane", "digest", reference).stdout.strip()
            if actual != targets.get(reference):
                raise ValueError(
                    f"Registry digest differs from publication receipt: {reference}"
                )
            document = json.loads(command("crane", "manifest", reference).stdout)
            if "manifests" in document:
                actual_arches = {
                    item["platform"]["architecture"]
                    for item in document["manifests"]
                    if item.get("platform", {}).get("os") == "linux"
                }
            else:
                config = json.loads(command("crane", "config", reference).stdout)
                actual_arches = (
                    {config["architecture"]} if config["os"] == "linux" else set()
                )
            if actual_arches != architectures[reference]:
                raise ValueError(
                    f"Registry image architectures differ from the Chart: {reference}"
                )
            verified.append(
                {
                    "reference": reference,
                    "digest": actual,
                    "architectures": sorted(actual_arches),
                }
            )
        command("helm", "lint", root)

        def render(profile, expected_image, *overrides):
            result = command(
                "helm", "template", "ucm", package, "-f", root / profile, *overrides
            )
            actual = set(images(list(yaml.safe_load_all(result.stdout))))
            if actual != {expected_image}:
                raise ValueError(f"rendered images differ for {profile}: {actual}")

        cuda = "models/cuda/values-qwen3-0p6b-1e1.yaml"
        if default:
            render(cuda, default)
            render("models/cuda/values-qwen3-0p6b-1p1-1d1.yaml", default)
        render(
            cuda,
            "example.com/override:v1",
            "--set-string",
            "images.image=example.com/override:v1",
        )
        render(
            cuda,
            "example.com/model:v2",
            "--set-string",
            "images.image=example.com/override:v1",
            "--set-string",
            "servingEngineSpec.modelSpec.image=example.com/model:v2",
        )
        if ascend_image:
            for profile in sorted((root / "models/ascend").glob("*.yaml")):
                result = command(
                    "helm", "template", "ucm", package, "-f", profile, check=False
                )
                if result.returncode == 0 or "images.image" not in result.stderr:
                    raise ValueError(
                        f"Ascend profile inherited a CUDA default: {profile.name}"
                    )
                render(
                    profile.relative_to(root),
                    ascend_image,
                    "--set-string",
                    f"images.image={ascend_image}",
                )
        return {
            "status": "complete",
            "chart": package.name,
            "sha256": digest,
            "images": verified,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--chart", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(
        json.loads(args.plan.read_text()),
        json.loads(args.state.read_text()),
        args.chart,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(
        f"Verified published Chart and {len(result['images'])} deployment image references"
    )


if __name__ == "__main__":
    main()
