"""Expose the release policy's standard image repositories to the docs UI."""

from pathlib import Path

import yaml


def on_config(config):
    policy_path = Path(__file__).resolve().parents[3] / ".github/release/release.yaml"
    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    config.extra["runtime_repositories"] = {
        product["id"]: product["runtime_repository"] for product in policy["products"]
    }
    return config
