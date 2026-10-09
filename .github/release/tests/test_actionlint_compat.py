"""Keep queue compatibility narrow without hiding actionlint diagnostics."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
WRAPPER = ROOT / "scripts/actionlint_compat.py"
spec = importlib.util.spec_from_file_location("actionlint_compat", WRAPPER)
compat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compat)


@pytest.fixture
def run_wrapper(tmp_path):
    directory = tmp_path / ".github/workflows"
    directory.mkdir(parents=True)
    binary_directory = tmp_path / "bin"
    binary_directory.mkdir()
    binary = binary_directory / "actionlint"
    binary.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys, yaml\n"
        "log = pathlib.Path(os.environ['ACTIONLINT_CALLS'])\n"
        "with log.open('a') as output: output.write(json.dumps(sys.argv[1:])+'\\n')\n"
        "failed = False\n"
        "for argument in sys.argv[1:]:\n"
        "    path = pathlib.Path(argument)\n"
        "    if not path.is_file() or path.suffix != '.yml': continue\n"
        "    source = path.read_text()\n"
        "    if 'UNRELATED_ERROR' in source:\n"
        "        line = next(i+1 for i,line in enumerate(source.splitlines()) if 'UNRELATED_ERROR' in line)\n"
        "        print(f'{argument}:{line}:3: unrelated syntax error', file=sys.stderr)\n"
        "        failed = True\n"
        "    value = yaml.safe_load(source)\n"
        "    groups = [value.get('concurrency', {})] + [job.get('concurrency', {}) for job in value.get('jobs', {}).values()]\n"
        "    if any(isinstance(group,dict) and 'queue' in group for group in groups):\n"
        "        print(f'{argument}: unsupported queue', file=sys.stderr)\n"
        "        failed = True\n"
        "raise SystemExit(1 if failed else 0)\n",
        encoding="utf-8",
    )
    binary.chmod(0o755)
    calls = tmp_path / "calls.jsonl"

    def run(source, *options):
        path = directory / "test.yml"
        path.write_text(source, encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(WRAPPER), *options, ".github/workflows/test.yml"],
            cwd=tmp_path,
            env={
                **os.environ,
                "PATH": f"{binary_directory}{os.pathsep}{os.environ['PATH']}",
                "ACTIONLINT_CALLS": str(calls),
            },
            capture_output=True,
            text=True,
            check=False,
        )
        return (
            result,
            [json.loads(line) for line in calls.read_text().splitlines()],
            path,
        )

    return run


def _workflow(queue="max", cancel="false", extra=""):
    return (
        "name: Test\non: push\nconcurrency:\n"
        f"  group: test\n  queue: {queue}\n  cancel-in-progress: {cancel}\n"
        "jobs:\n  test:\n    runs-on: ubuntu-latest\n"
        f"    steps:\n      - run: echo ok {extra}\n"
    )


@pytest.mark.parametrize(
    "queue,cancel,expected",
    [
        ("latest", "false", "literal max or single"),
        ("${{ inputs.queue }}", "false", "literal max or single"),
        ("max", "true", "cannot combine"),
    ],
)
def test_invalid_queue_policy_fails_before_normalized_lint(
    run_wrapper, queue, cancel, expected
):
    result, calls, _ = run_wrapper(_workflow(queue, cancel))
    assert result.returncode == 1
    assert expected in result.stderr
    assert len(calls) == 1


@pytest.mark.parametrize("queue,cancel", [("max", "false"), ("single", "true")])
def test_supported_queue_is_checked_then_only_normalized_in_temporary_file(
    run_wrapper, queue, cancel
):
    source = _workflow(queue, cancel)
    result, calls, path = run_wrapper(source, "-shellcheck=")
    assert result.returncode == 0, result.stderr
    assert len(calls) == 2
    assert calls[0] == ["-shellcheck=", ".github/workflows/test.yml"]
    assert calls[1][0] == "-shellcheck="
    assert calls[1][-1] != calls[0][-1]
    assert path.read_text() == source


def test_unrelated_actionlint_error_and_original_location_survive_normalization(
    run_wrapper,
):
    source = _workflow(extra="UNRELATED_ERROR")
    result, calls, path = run_wrapper(source)
    assert result.returncode == 1
    assert len(calls) == 2
    line = next(
        i + 1 for i, text in enumerate(source.splitlines()) if "UNRELATED_ERROR" in text
    )
    assert (
        f".github/workflows/test.yml:{line}:3: unrelated syntax error" in result.stderr
    )
    assert "ucm-actionlint-" not in result.stderr
    assert path.read_text() == source


def test_clean_original_is_linted_once_without_normalization(run_wrapper):
    source = "name: Test\non: push\njobs: {}\n"
    result, calls, _ = run_wrapper(source)
    assert result.returncode == 0
    assert len(calls) == 1


def test_non_queue_errors_keep_original_actionlint_result(run_wrapper):
    source = "name: Test\non: push\njobs: {}\n# UNRELATED_ERROR\n"
    result, calls, _ = run_wrapper(source)
    assert result.returncode == 1
    assert len(calls) == 1
    assert "unrelated syntax error" in result.stderr


def test_only_actual_concurrency_queue_fields_are_removed():
    source = (
        "name: Test\non: push\nconcurrency: {group: root, queue: max, cancel-in-progress: false}\n"
        "jobs:\n  test:\n    concurrency:\n      group: job\n      queue: single\n"
        "    steps:\n      - run: |\n          queue: keep-this-shell-text\n"
    )
    normalized = compat.without_queue(source, "test.yml")
    assert normalized is not None
    assert "queue: keep-this-shell-text" in normalized
    assert "group: root" in normalized and "cancel-in-progress: false" in normalized
    assert "group: job" in normalized
    assert normalized.count("\n") == source.count("\n")


def test_duplicate_queue_or_unknown_concurrency_field_cannot_disappear():
    for source in (
        "concurrency:\n  group: test\n  queue: max\n  queue: single\n",
        "concurrency:\n  group: test\n  queue: max\n  unknown: true\n",
    ):
        with pytest.raises(compat.QueueError):
            compat.without_queue(source, "test.yml")
