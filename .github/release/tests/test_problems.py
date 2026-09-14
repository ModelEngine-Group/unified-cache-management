"""Contracts for formal problem validation and deterministic presentation."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

RELEASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE_ROOT))

problems = importlib.import_module("ucm_release.problems")


def _problem(
    *,
    capability: str = "CANN 9.2 / A5 / cp312 / amd64 / openEuler 24.03",
    runtime_tag: str = "nightly-releases-v0.23.0-a5-openeuler",
    reason: str = "A5 requires a dedicated UCM native implementation",
) -> dict[str, object]:
    return {
        "backend": "cann-a5",
        "capability": capability,
        "reason": reason,
        "runtime": {
            "repository": "quay.io/ascend/vllm-ascend",
            "tag": runtime_tag,
        },
    }


def test_validate_preserves_structure_and_sorts_without_mutating_input() -> None:
    later = _problem(capability="CANN 9.2 / A5 / cp312 / arm64 / Ubuntu 22.04")
    earlier = _problem(capability="CANN 9.2 / A5 / cp312 / amd64 / Ubuntu 22.04")
    raw = [later, earlier]

    validated = problems.validate_formal_problems(raw)

    assert [item["capability"] for item in validated] == [
        earlier["capability"],
        later["capability"],
    ]
    assert validated[0] == earlier
    assert validated[0] is not earlier
    assert validated[0]["runtime"] is not earlier["runtime"]
    assert raw == [later, earlier]


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({}, "must be a list"),
        ([{"backend": "cann-a5"}], "invalid keys"),
        ([{**_problem(), "extra": "value"}], "invalid keys"),
        (
            [{**_problem(), "runtime": {"repository": "repo", "tag": "tag"}}],
            "runtime.repository is malformed",
        ),
        ([{**_problem(), "backend": "CANN A5"}], "backend is malformed"),
        ([{**_problem(), "capability": ""}], "capability must be"),
        ([{**_problem(), "reason": " blocked"}], "reason must be"),
        (
            [
                {
                    **_problem(),
                    "runtime": {
                        "repository": "quay.io/ascend/vllm-ascend",
                        "tag": "bad tag",
                    },
                }
            ],
            "runtime.tag is malformed",
        ),
    ],
)
def test_validate_rejects_malformed_records(value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        problems.validate_formal_problems(value)


def test_duplicate_identity_is_the_full_normalized_problem() -> None:
    first = _problem()
    with pytest.raises(ValueError, match="duplicate formal problem"):
        problems.validate_formal_problems([first, _problem()])

    different_reason = _problem(reason="Native A5 support is blocked")
    assert len(problems.validate_formal_problems([first, different_reason])) == 2


def test_actions_summary_is_deterministic_and_empty_state_is_explicit() -> None:
    arm = _problem(capability="CANN 9.2 / A5 / cp312 / arm64 / Ubuntu 22.04")
    amd = _problem(capability="CANN 9.2 / A5 / cp312 / amd64 / Ubuntu 22.04")

    rendered = problems.render_actions_summary([arm, amd])

    assert rendered == (
        "## Upstream capability problems\n"
        "\n"
        "2 blocked upstream capability problems detected.\n"
        "\n"
        "| Backend | Capability | Reason | Runtime |\n"
        "| --- | --- | --- | --- |\n"
        "| <code>cann-a5</code> | CANN 9.2 / A5 / cp312 / amd64 / Ubuntu 22.04 | "
        "A5 requires a dedicated UCM native implementation | "
        "<code>quay.io/ascend/vllm-ascend:"
        "nightly-releases-v0.23.0-a5-openeuler</code> |\n"
        "| <code>cann-a5</code> | CANN 9.2 / A5 / cp312 / arm64 / Ubuntu 22.04 | "
        "A5 requires a dedicated UCM native implementation | "
        "<code>quay.io/ascend/vllm-ascend:"
        "nightly-releases-v0.23.0-a5-openeuler</code> |\n"
    )
    assert problems.render_actions_summary([amd, arm]) == rendered
    assert problems.render_actions_summary([]) == (
        "## Upstream capability problems\n"
        "\n"
        "No blocked upstream capabilities were detected.\n"
    )


def test_rolling_issue_has_one_stable_identity_and_generated_body() -> None:
    rendered = problems.render_rolling_issue([_problem()])

    assert rendered["title"] == problems.ROLLING_ISSUE_TITLE
    assert rendered["body"].startswith(
        problems.ROLLING_ISSUE_MARKER + "\n## Blocked upstream capabilities\n"
    )
    assert "<code>cann-a5</code>" in rendered["body"]
    assert "quay.io/ascend/vllm-ascend" in rendered["body"]
    assert problems.render_rolling_issue([_problem()]) == rendered

    empty = problems.render_rolling_issue([])
    assert empty["title"] == rendered["title"]
    assert "No blocked upstream capabilities remain." in empty["body"]


def test_markdown_table_escapes_problem_text() -> None:
    problem = _problem(
        capability="CANN 9.2 | A5 <blocked>",
        reason="Native support | pending <implementation>",
    )

    rendered = problems.render_actions_summary([problem])

    assert "CANN 9.2 &#124; A5 &lt;blocked&gt;" in rendered
    assert "Native support &#124; pending &lt;implementation&gt;" in rendered


def test_rolling_issue_action_follows_problem_presence() -> None:
    assert problems.decide_rolling_issue_action([_problem()]) == "open_or_update"
    assert problems.decide_rolling_issue_action([]) == "close"
