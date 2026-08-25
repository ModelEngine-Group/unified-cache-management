from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "materialize_version.py"
SPEC = importlib.util.spec_from_file_location("materialize_version", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
materialize = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(materialize)


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("v0.7.59rc5", "0.7.59rc5"),
        ("v0.7.59", "0.7.59"),
        ("draft/v0.6.0", "0.6.0.dev0"),
        ("draft/v0.6.0-13", "0.6.0.dev13"),
    ],
)
def test_version_from_tag(tag: str, expected: str) -> None:
    assert materialize.version_from_tag(tag) == expected


@pytest.mark.parametrize(
    "value",
    [
        "v0.7.59RC5",
        "v0.7",
        "v01.7.59",
        "draft/v0.6.0-0",
        "draft/v0.6.0-01",
        "draft/v0.6.0rc1-2",
        "0.7.59",
    ],
)
def test_invalid_or_noncanonical_tags_are_rejected(value: str) -> None:
    with pytest.raises(ValueError):
        materialize.version_from_tag(value)


def test_cli_writes_one_line_and_prints_the_version(tmp_path: Path) -> None:
    output = tmp_path / "version.ini"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--tag",
            "draft/v0.6.0-13",
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=True,
    )

    assert completed.stdout == "0.6.0.dev13\n"
    assert output.read_text(encoding="utf-8") == "VLLM_UC_VERSION=0.6.0.dev13\n"


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        (
            "v0.7.62",
            {
                "git_tag": "v0.7.62",
                "release_kind": "publish",
                "version": "0.7.62",
                "chart_version": "0.7.62",
                "image_version": "0.7.62",
                "is_prerelease": False,
            },
        ),
        (
            "v0.7.62rc3",
            {
                "git_tag": "v0.7.62rc3",
                "release_kind": "publish",
                "version": "0.7.62rc3",
                "chart_version": "0.7.62-rc.3",
                "image_version": "0.7.62rc3",
                "is_prerelease": True,
            },
        ),
        (
            "draft/v0.7.62",
            {
                "git_tag": "draft/v0.7.62",
                "release_kind": "draft",
                "version": "0.7.62.dev0",
                "chart_version": "0.7.62-draft.0",
                "image_version": "0.7.62.dev0",
                "is_prerelease": True,
            },
        ),
        (
            "draft/v0.7.62-4",
            {
                "git_tag": "draft/v0.7.62-4",
                "release_kind": "draft",
                "version": "0.7.62.dev4",
                "chart_version": "0.7.62-draft.4",
                "image_version": "0.7.62.dev4",
                "is_prerelease": True,
            },
        ),
    ],
)
def test_classify_tag(tag: str, expected: dict[str, object]) -> None:
    assert materialize.classify_tag(tag) == expected


def test_classify_cli_does_not_materialize_version(tmp_path: Path) -> None:
    output = tmp_path / "version.ini"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--tag",
            "draft/v0.7.62",
            "--classify",
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=True,
    )

    assert json.loads(completed.stdout) == materialize.classify_tag("draft/v0.7.62")
    assert not output.exists()


def test_version_option_requires_canonical_pep440(tmp_path: Path) -> None:
    output = tmp_path / "version.ini"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--version",
            "0.7.59RC5",
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "canonical PEP 440" in completed.stderr
    assert not output.exists()
