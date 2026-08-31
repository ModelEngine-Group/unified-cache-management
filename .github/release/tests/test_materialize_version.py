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

VERSION_CONFIG = (
    "VLLM_UC_VERSION=0.7.62\n"
    "UCM_SUPPORTED_VLLM_VERSIONS=0.27.1\n"
    "UCM_SUPPORTED_VLLM_ASCEND_VERSIONS=0.26.0rc\n"
)


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("v0.7.59rc5", "0.7.59rc5"),
        ("v0.7.59", "0.7.59"),
        ("draft/v0.6.0", "0.6.0.dev0"),
        ("draft/v0.6.0-13", "0.6.0.dev13"),
        ("nightly/v0.7.62-20260826-1", "0.7.62.dev20260826001"),
        ("nightly/v0.7.62-20260826-1234", "0.7.62.dev202608261234"),
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
        "nightly/v0.7.62-20260826-0",
        "nightly/v0.7.62-20260826-01",
        "nightly/v0.7.62rc1-20260826-1",
        "nightly/v0.7.62-20260230-1",
        "nightly/v0.7.62-2026-08-26-1",
        "nightly/v0.7.62-20260826",
        "0.7.59",
    ],
)
def test_invalid_or_noncanonical_tags_are_rejected(value: str) -> None:
    with pytest.raises(ValueError):
        materialize.version_from_tag(value)


def test_cli_replaces_only_ucm_version_and_prints_it(tmp_path: Path) -> None:
    output = tmp_path / "version.ini"
    output.write_text(VERSION_CONFIG, encoding="utf-8")
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
    assert output.read_text(encoding="utf-8") == VERSION_CONFIG.replace(
        "VLLM_UC_VERSION=0.7.62", "VLLM_UC_VERSION=0.6.0.dev13"
    )


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        (
            "v0.7.62",
            {
                "git_tag": "v0.7.62",
                "release_type": "stable",
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
                "release_type": "prerelease",
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
                "release_type": "draft",
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
                "release_type": "draft",
                "release_kind": "draft",
                "version": "0.7.62.dev4",
                "chart_version": "0.7.62-draft.4",
                "image_version": "0.7.62.dev4",
                "is_prerelease": True,
            },
        ),
        (
            "nightly/v0.7.62-20260826-1",
            {
                "git_tag": "nightly/v0.7.62-20260826-1",
                "release_type": "nightly",
                "release_kind": "publish",
                "version": "0.7.62.dev20260826001",
                "chart_version": "0.7.62-nightly.20260826.1",
                "image_version": "0.7.62.dev20260826001",
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


def test_next_patch_uses_only_exact_stable_tags() -> None:
    assert (
        materialize.next_patch_version(
            [
                "v0.7.60",
                "v0.7.61rc9",
                "draft/v9.0.0-1",
                "nightly/v8.0.0-20260826-1",
                "v0.7.61",
                "v0.7.61-extra",
            ]
        )
        == "0.7.62"
    )


def test_next_patch_requires_a_strict_stable_tag() -> None:
    with pytest.raises(ValueError, match="strict Stable"):
        materialize.next_patch_version(
            ["v0.7.61rc1", "draft/v0.7.61-1", "nightly/v0.7.61-20260826-1"]
        )


def test_next_nightly_sequence_is_scoped_to_base_and_date() -> None:
    tags = [
        "nightly/v0.7.62-20260826-1",
        "nightly/v0.7.62-20260826-3",
        "nightly/v0.7.62-20260825-9",
        "nightly/v0.7.63-20260826-8",
        "nightly/v0.7.62-20260826-04",
    ]

    assert (
        materialize.next_nightly_sequence(
            tags, base_version="0.7.62", release_date="20260826"
        )
        == 4
    )
    assert (
        materialize.next_nightly_sequence(
            tags, base_version="0.7.62", release_date="20260827"
        )
        == 1
    )


def test_next_nightly_cli_outputs_classification_without_materializing(
    tmp_path: Path,
) -> None:
    tags_file = tmp_path / "tags.txt"
    tags_file.write_text(
        "\n".join(
            [
                "v0.7.61",
                "v0.8.0rc1",
                "nightly/v0.7.62-20260826-1",
                "nightly/v0.7.62-20260826-2",
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "version.ini"
    config = tmp_path / "source-version.ini"
    config.write_text(VERSION_CONFIG, encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--next-nightly",
            "--tags-file",
            str(tags_file),
            "--date",
            "20260826",
            "--version-config",
            str(config),
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=True,
    )

    assert json.loads(completed.stdout) == materialize.classify_tag(
        "nightly/v0.7.62-20260826-3"
    )
    assert not output.exists()


@pytest.mark.parametrize(
    "arguments",
    [
        ["--next-nightly", "--date", "20260826"],
        ["--next-nightly", "--tags-file", "tags.txt"],
        ["--tag", "v0.7.61", "--date", "20260826"],
    ],
)
def test_next_nightly_cli_requires_its_exact_inputs(
    tmp_path: Path, arguments: list[str]
) -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), *arguments, "--output", str(tmp_path / "out")],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0


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


@pytest.mark.parametrize(
    "tag",
    [
        "v0.7.62",
        "v0.7.62rc3",
        "draft/v0.7.62-4",
        "nightly/v0.7.62-20260826-1",
    ],
)
def test_tag_base_must_match_version_config(tmp_path: Path, tag: str) -> None:
    config = tmp_path / "version.ini"
    config.write_text(VERSION_CONFIG, encoding="utf-8")

    assert materialize.validate_tag_against_config(tag, config)["git_tag"] == tag
    with pytest.raises(ValueError, match="base version differs"):
        materialize.validate_tag_against_config(tag.replace("0.7.62", "0.7.63"), config)


def test_version_config_supports_keywords_and_explicit_tags() -> None:
    parsed = materialize.version_config.parse(
        "VLLM_UC_VERSION=0.9.3\n"
        "UCM_SUPPORTED_VLLM_VERSIONS=0.27.1,latest\n"
        "UCM_SUPPORTED_VLLM_ASCEND_VERSIONS=0.25.1rc@nightly-releases-v0.25.1rc\n"
    )

    assert parsed["supported_runtimes"]["vllm"] == [
        {"raw": "0.27.1", "keyword": "0.27.1", "tag": None},
        {"raw": "latest", "keyword": "latest", "tag": None},
    ]
    assert parsed["supported_runtimes"]["vllm-ascend"] == [
        {
            "raw": "0.25.1rc@nightly-releases-v0.25.1rc",
            "keyword": "0.25.1rc",
            "tag": "nightly-releases-v0.25.1rc",
        }
    ]


@pytest.mark.parametrize(
    "text",
    [
        "VLLM_UC_VERSION=0.9.3\nUCM_SUPPORTED_VLLM_VERSIONS=0.27.1\n",
        "VLLM_UC_VERSION=0.9.3\nUCM_SUPPORTED_VLLM_VERSIONS=0.27.1,0.27.1\nUCM_SUPPORTED_VLLM_ASCEND_VERSIONS=0.26.0rc\n",
        "VLLM_UC_VERSION=0.9.3\nUCM_SUPPORTED_VLLM_VERSIONS=bad keyword\nUCM_SUPPORTED_VLLM_ASCEND_VERSIONS=0.26.0rc\n",
        "VLLM_UC_VERSION=0.9.3\nUCM_SUPPORTED_VLLM_VERSIONS=0.27.1@bad/tag\nUCM_SUPPORTED_VLLM_ASCEND_VERSIONS=0.26.0rc\n",
    ],
)
def test_version_config_rejects_missing_duplicate_or_invalid_selectors(
    text: str,
) -> None:
    with pytest.raises(ValueError):
        materialize.version_config.parse(text)
