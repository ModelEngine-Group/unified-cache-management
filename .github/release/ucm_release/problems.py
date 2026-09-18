"""Validation and deterministic presentation for formal release problems."""

from __future__ import annotations

import html
import re
from typing import Literal, TypedDict

__all__ = [
    "FormalProblem",
    "IssueAction",
    "ROLLING_ISSUE_MARKER",
    "ROLLING_ISSUE_TITLE",
    "RuntimeReference",
    "decide_rolling_issue_action",
    "render_actions_summary",
    "render_rolling_issue",
    "validate_formal_problems",
]


class RuntimeReference(TypedDict):
    repository: str
    tag: str


class FormalProblem(TypedDict):
    backend: str
    capability: str
    reason: str
    runtime: RuntimeReference


IssueAction = Literal["open_or_update", "close"]

ROLLING_ISSUE_TITLE = "UCM release: blocked upstream capabilities"
ROLLING_ISSUE_MARKER = "<!-- ucm-release:blocked-upstream-capabilities -->"

_PROBLEM_KEYS = {"backend", "capability", "reason", "runtime"}
_RUNTIME_KEYS = {"repository", "tag"}
_BACKEND_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_RUNTIME_REPOSITORY_PATTERN = re.compile(
    r"^[a-z0-9]+(?:[.-][a-z0-9]+)*(?::[0-9]+)?"
    r"(?:/[a-z0-9]+(?:(?:[._]|-+)[a-z0-9]+)*)+$"
)
_OCI_TAG_PATTERN = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")


def _exact_mapping(value: object, keys: set[str], context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{context} has non-string keys")
    actual = set(value)
    if actual != keys:
        missing = sorted(keys - actual)
        unknown = sorted(actual - keys)
        raise ValueError(
            f"{context} has invalid keys; missing={missing}, unknown={unknown}"
        )
    return value


def _text(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{context} must be a non-empty trimmed string")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{context} contains control characters")
    return value


def _problem_sort_key(problem: FormalProblem) -> tuple[str, ...]:
    return (
        problem["backend"],
        problem["capability"],
        problem["runtime"]["repository"],
        problem["runtime"]["tag"],
        problem["reason"],
    )


def validate_formal_problems(value: object) -> list[FormalProblem]:
    """Validate, copy, and deterministically order formal ``problems[]``."""

    if not isinstance(value, list):
        raise ValueError("formal problems must be a list")

    normalized: list[FormalProblem] = []
    seen: set[tuple[str, ...]] = set()
    for index, raw_problem in enumerate(value):
        context = f"formal problems[{index}]"
        problem = _exact_mapping(raw_problem, _PROBLEM_KEYS, context)
        runtime = _exact_mapping(
            problem["runtime"], _RUNTIME_KEYS, f"{context}.runtime"
        )

        backend = _text(problem["backend"], f"{context}.backend")
        capability = _text(problem["capability"], f"{context}.capability")
        reason = _text(problem["reason"], f"{context}.reason")
        runtime_repository = _text(
            runtime["repository"], f"{context}.runtime.repository"
        )
        runtime_tag = _text(runtime["tag"], f"{context}.runtime.tag")

        if _BACKEND_PATTERN.fullmatch(backend) is None:
            raise ValueError(f"{context}.backend is malformed")
        if _RUNTIME_REPOSITORY_PATTERN.fullmatch(runtime_repository) is None:
            raise ValueError(f"{context}.runtime.repository is malformed")
        if _OCI_TAG_PATTERN.fullmatch(runtime_tag) is None:
            raise ValueError(f"{context}.runtime.tag is malformed")

        item: FormalProblem = {
            "backend": backend,
            "capability": capability,
            "reason": reason,
            "runtime": {"repository": runtime_repository, "tag": runtime_tag},
        }
        identity = _problem_sort_key(item)
        if identity in seen:
            raise ValueError(f"duplicate formal problem at index {index}")
        seen.add(identity)
        normalized.append(item)

    return sorted(normalized, key=_problem_sort_key)


def _cell(value: str) -> str:
    return html.escape(value, quote=True).replace("|", "&#124;")


def _code(value: str) -> str:
    return f"<code>{_cell(value)}</code>"


def _table(problems: list[FormalProblem]) -> str:
    lines = [
        "| Backend | Capability | Reason | Runtime |",
        "| --- | --- | --- | --- |",
    ]
    for problem in problems:
        runtime = f"{problem['runtime']['repository']}:{problem['runtime']['tag']}"
        lines.append(
            "| "
            + " | ".join(
                (
                    _code(problem["backend"]),
                    _cell(problem["capability"]),
                    _cell(problem["reason"]),
                    _code(runtime),
                )
            )
            + " |"
        )
    return "\n".join(lines)


def render_actions_summary(value: object) -> str:
    """Render the formal problem section written to ``GITHUB_STEP_SUMMARY``."""

    problems = validate_formal_problems(value)
    lines = ["## Upstream capability problems", ""]
    if not problems:
        lines.append("No blocked upstream capabilities were detected.")
    else:
        noun = "problem" if len(problems) == 1 else "problems"
        lines.extend(
            (
                f"{len(problems)} blocked upstream capability {noun} detected.",
                "",
                _table(problems),
            )
        )
    return "\n".join(lines) + "\n"


def render_rolling_issue(value: object) -> dict[str, str]:
    """Render the stable title and generated body for the one rolling Issue."""

    problems = validate_formal_problems(value)
    lines = [
        ROLLING_ISSUE_MARKER,
        "## Blocked upstream capabilities",
        "",
        "This rolling issue is maintained by the formal UCM release pipeline.",
        "",
    ]
    if problems:
        lines.extend((_table(problems), ""))
    else:
        lines.extend(("No blocked upstream capabilities remain.", ""))
    lines.append(
        "The generated content above reflects the latest formal Runtime selection."
    )
    return {
        "title": ROLLING_ISSUE_TITLE,
        "body": "\n".join(lines) + "\n",
    }


def decide_rolling_issue_action(value: object) -> IssueAction:
    """Choose the sole remote action without performing any network operation."""

    return "open_or_update" if validate_formal_problems(value) else "close"
