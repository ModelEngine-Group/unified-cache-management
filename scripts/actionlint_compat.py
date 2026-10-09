#!/usr/bin/env python3
"""Validate Actions queue settings while actionlint lacks concurrency.queue support.

The original files always reach actionlint first. Only verified queue fields are
blanked in temporary copies for the second pass; every other rule remains owned
by actionlint. Remove this adapter when the pinned actionlint supports queue.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml
from yaml.nodes import MappingNode, ScalarNode


class QueueError(ValueError):
    """A queue setting cannot be checked by this narrow compatibility adapter."""


def _concurrency_mappings(document: MappingNode):
    for key, value in document.value:
        if key.value == "concurrency" and isinstance(value, MappingNode):
            yield value
        elif key.value == "jobs" and isinstance(value, MappingNode):
            for _, job in value.value:
                if isinstance(job, MappingNode):
                    for job_key, job_value in job.value:
                        if job_key.value == "concurrency" and isinstance(
                            job_value, MappingNode
                        ):
                            yield job_value


def without_queue(source: str, path: str) -> str | None:
    """Remove actual root/job concurrency.queue fields, preserving line numbers.

    Invalid YAML stays with actionlint. Mapping keys and literal queue values are
    checked before normalization so no syntax error can disappear in the adapter.
    """
    try:
        document = yaml.compose(source)
    except yaml.YAMLError:
        return None
    if not isinstance(document, MappingNode):
        return None
    removals: list[tuple[int, int]] = []
    for mapping in _concurrency_mappings(document):
        if not any(
            isinstance(key, ScalarNode) and key.value == "queue"
            for key, _ in mapping.value
        ):
            continue
        entries = {}
        for key, value in mapping.value:
            if not isinstance(key, ScalarNode) or key.value in entries:
                raise QueueError(
                    f"{path}:{key.start_mark.line + 1}: invalid concurrency key"
                )
            entries[key.value] = (key, value)
        if "queue" not in entries:
            continue
        unknown = set(entries) - {"group", "queue", "cancel-in-progress"}
        key, value = entries["queue"]
        location = f"{path}:{key.start_mark.line + 1}:{key.start_mark.column + 1}"
        if unknown:
            raise QueueError(
                f"{location}: unsupported concurrency fields: {', '.join(sorted(unknown))}"
            )
        if (
            not isinstance(value, ScalarNode)
            or value.tag != "tag:yaml.org,2002:str"
            or value.value not in {"max", "single"}
            or value.style not in {None, "'", '"'}
            or value.start_mark.index < key.end_mark.index
            or key.start_mark.index < mapping.start_mark.index
            or source[value.start_mark.index : value.end_mark.index].startswith("&")
        ):
            raise QueueError(
                f"{location}: concurrency.queue must be literal max or single"
            )
        cancel = entries.get("cancel-in-progress")
        if value.value == "max" and cancel is not None:
            cancel_value = cancel[1]
            if (
                isinstance(cancel_value, ScalarNode)
                and cancel_value.tag == "tag:yaml.org,2002:bool"
                and cancel_value.value.lower() in {"true", "yes", "on"}
            ):
                raise QueueError(
                    f"{location}: queue: max cannot combine with cancel-in-progress: true"
                )
        if mapping.flow_style:
            start, end = key.start_mark.index, value.end_mark.index
            following = end
            while following < mapping.end_mark.index and source[following].isspace():
                following += 1
            if source[following : following + 1] == ",":
                end = following + 1
            else:
                preceding = start - 1
                while (
                    preceding > mapping.start_mark.index and source[preceding].isspace()
                ):
                    preceding -= 1
                if source[preceding : preceding + 1] == ",":
                    start = preceding
        else:
            start = source.rfind("\n", 0, key.start_mark.index) + 1
            end = source.find("\n", value.end_mark.index)
            end = len(source) if end == -1 else end
        removals.append((start, end))
    if not removals:
        return None
    for start, end in sorted(removals, reverse=True):
        source = (
            source[:start]
            + "".join("\n" if char == "\n" else " " for char in source[start:end])
            + source[end:]
        )
    return source


def _file_arguments(arguments: list[str]) -> list[int]:
    # These actionlint flags consume a separate value; everything after the
    # first workflow path is a filename under Go's flag parser contract.
    value_flags = {"-config-file", "-format", "-ignore", "-pyflakes", "-shellcheck"}
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--":
            return list(range(index + 1, len(arguments)))
        if not argument.startswith("-") or argument == "-":
            return list(range(index, len(arguments)))
        index += 2 if argument in value_flags else 1
    return []


def _emit(result: subprocess.CompletedProcess[str], replacements=()) -> None:
    for stream, output in ((sys.stdout, result.stdout), (sys.stderr, result.stderr)):
        for temporary, original in replacements:
            output = output.replace(temporary, original)
        stream.write(output)


def main(arguments: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    original = subprocess.run(
        ["actionlint", *arguments], capture_output=True, text=True, check=False
    )
    if any(flag in arguments for flag in ("-version", "-help", "-h", "-init-config")):
        _emit(original)
        return original.returncode
    root = Path.cwd()
    indexes = _file_arguments(arguments)
    paths = (
        [Path(arguments[index]) for index in indexes]
        if indexes
        else sorted(
            [
                *(root / ".github/workflows").glob("*.yml"),
                *(root / ".github/workflows").glob("*.yaml"),
            ]
        )
    )
    normalized: dict[Path, str] = {}
    try:
        for path in paths:
            if not path.is_file():
                continue
            rewritten = without_queue(path.read_text(encoding="utf-8"), str(path))
            if rewritten is not None:
                normalized[path.resolve()] = rewritten
    except QueueError as error:
        print(str(error), file=sys.stderr)
        return 1
    if original.returncode == 0 or not normalized:
        _emit(original)
        return original.returncode
    with tempfile.TemporaryDirectory(prefix="ucm-actionlint-") as temporary:
        staging = Path(temporary)
        (staging / ".git").mkdir()
        workflow_directory = root / ".github/workflows"
        if workflow_directory.is_dir():
            shutil.copytree(workflow_directory, staging / ".github/workflows")
        actions = root / ".github/actions"
        if actions.is_dir():
            shutil.copytree(actions, staging / ".github/actions")
        config = root / ".github/actionlint.yaml"
        if config.is_file():
            target = staging / ".github/actionlint.yaml"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(config, target)
        mapped: dict[Path, Path] = {}
        replacements = []
        for index, path in enumerate(paths):
            if not path.is_file():
                continue
            resolved = path.resolve()
            try:
                relative = resolved.relative_to(root)
            except ValueError:
                relative = Path("inputs") / str(index) / path.name
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                normalized.get(resolved, path.read_text(encoding="utf-8")),
                encoding="utf-8",
            )
            mapped[resolved] = target
            replacements.append((str(target), str(path)))
        rerun = list(arguments)
        if indexes:
            for index in indexes:
                path = Path(arguments[index]).resolve()
                if path in mapped:
                    rerun[index] = str(mapped[path])
        else:
            rerun.extend(
                str(mapped[path.resolve()])
                for path in paths
                if path.resolve() in mapped
            )
        result = subprocess.run(
            ["actionlint", *rerun], capture_output=True, text=True, check=False
        )
        _emit(result, replacements)
        return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
