"""Check result model and output rendering (bordered tables + JSON)."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from typing import List

# Severity levels, lowest -> highest.
INFO = "INFO"  # display only, never affects pass/fail
WARN = "WARN"  # warning if below/above a soft threshold
FAIL = "FAIL"  # hard constraint violation

SEVERITIES = (INFO, WARN, FAIL)

# Status shown in the table (decoupled from severity so an INFO check can be
# "OK", a WARN check can be "PASS"/"WARN"/"SKIP", etc.).
STATUS_PASS = "PASS"
STATUS_WARN = "WARN"
STATUS_FAIL = "FAIL"
STATUS_SKIP = "SKIP"
STATUS_OK = "OK"
STATUS_INFO = "INFO"


@dataclass
class CheckResult:
    """Outcome of a single pre-check item.

    ``severity`` declares the check's class (INFO/WARN/FAIL); ``status`` is the
    concrete verdict. ``value`` is the measured/printed value; ``threshold`` is
    the comparison basis (for WARN/FAIL checks); ``detail`` is human-readable
    context.
    """

    name: str
    severity: str = INFO
    status: str = STATUS_INFO
    value: str = ""
    threshold: str = ""
    detail: str = ""
    remediation: str = ""
    raw: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"invalid severity {self.severity!r}")


def _status_icon(status: str) -> str:
    return {
        STATUS_PASS: "[PASS]",
        STATUS_OK: "[OK]",
        STATUS_WARN: "[WARN]",
        STATUS_FAIL: "[FAIL]",
        STATUS_SKIP: "[SKIP]",
        STATUS_INFO: "[INFO]",
    }.get(status, f"[{status}]")


# ANSI color codes (disabled via --no-color / non-tty).
_RESET = "\033[0m"
_RED = "\033[31m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_DIM = "\033[2m"

_NAME_WIDTH = 20
_RULE = "\u2500" * 78


def _color(text: str, code: str, enabled: bool) -> str:
    return f"{code}{text}{_RESET}" if enabled else text


def _status_color(status: str, enabled: bool) -> str:
    if status in (STATUS_PASS, STATUS_OK):
        return _color(_status_icon(status), _GREEN, enabled)
    if status in (STATUS_WARN,):
        return _color(_status_icon(status), _YELLOW, enabled)
    if status in (STATUS_FAIL,):
        return _color(_status_icon(status), _RED, enabled)
    return _color(_status_icon(status), _DIM, enabled)


def render_text(results: List[CheckResult], color: bool = True) -> str:
    """Render results as a readable per-check block with fix advice.

    Each check prints a status line (icon + name + value) followed by indented
    threshold and ``fix:`` lines. The ``fix`` line appears only for FAIL/WARN
    outcomes that carry remediation advice (per RFC #1208).
    """
    lines = [
        _color("UCM environment pre-check", _CYAN, color),
        _color(_RULE, _DIM, color),
    ]
    for r in results:
        icon = _status_color(r.status, color) if color else _status_icon(r.status)
        lines.append(f"{icon} {r.name.ljust(_NAME_WIDTH)} {r.value}".rstrip())
        if r.threshold:
            lines.append(_color(f"{'':8}threshold: {r.threshold}", _DIM, color))
        if r.detail and r.status in (STATUS_WARN, STATUS_FAIL, STATUS_SKIP):
            lines.append(_color(f"{'':8}{r.detail}", _DIM, color))
        if r.remediation and r.status in (STATUS_WARN, STATUS_FAIL, STATUS_SKIP):
            lines.append(_color(f"{'':8}fix: {r.remediation}", _YELLOW, color))
    lines.append(_color(_RULE, _DIM, color))

    n_fail = sum(1 for r in results if r.status == STATUS_FAIL)
    n_warn = sum(1 for r in results if r.status == STATUS_WARN)
    n_skip = sum(1 for r in results if r.status == STATUS_SKIP)
    n_info = sum(1 for r in results if r.severity == INFO and r.status != STATUS_SKIP)
    n_pass = sum(
        1
        for r in results
        if r.status in (STATUS_PASS, STATUS_OK) and r.severity != INFO
    )

    if n_fail:
        overall = _color("FAILED", _RED, color)
    elif n_warn:
        overall = _color("PASSED (with warnings)", _YELLOW, color)
    else:
        overall = _color("PASSED", _GREEN, color)

    lines.append(
        f"Result: {overall}  |  "
        f"{n_pass} pass, {n_warn} warn, {n_fail} fail, "
        f"{n_skip} skip, {n_info} info"
    )
    return "\n".join(lines)


def render_json(results: List[CheckResult]) -> str:
    return json.dumps(
        {
            "checks": [asdict(r) for r in results],
            "failed": any(r.status == STATUS_FAIL for r in results),
            "warned": any(r.status == STATUS_WARN for r in results),
        },
        indent=2,
    )


def overall_failed(results: List[CheckResult], strict: bool = False) -> bool:
    """True if the run should exit non-zero (FAIL, or WARN when strict)."""
    if any(r.status == STATUS_FAIL for r in results):
        return True
    if strict and any(r.status == STATUS_WARN for r in results):
        return True
    return False


def eprint(msg: str) -> None:
    print(msg, file=sys.stderr)
