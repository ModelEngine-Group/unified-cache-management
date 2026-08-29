"""Validate auditwheel external libraries against direct vendor roots."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from fnmatch import fnmatchcase
from typing import Any

AUDITWHEEL_EXTERNAL_MARKER = (
    "The following external shared libraries are required by the wheel:"
)
AUDITWHEEL_ELF_TREE_MARKER = "DEBUG:auditwheel.wheel_abi:full_elftree:\n"
SONAME_PATTERN = r"lib[a-zA-Z0-9_.+-]+\.so(?:\.[0-9]+)*"


def validate_external_soname(soname: str) -> None:
    if re.fullmatch(SONAME_PATTERN, soname) is None or soname == "libmetrics.so":
        raise ValueError(f"invalid external runtime SONAME: {soname!r}")


def validate_exclude_pattern(pattern: str) -> None:
    """Accept an exact SONAME or a constrained absolute provider-path glob."""
    if re.fullmatch(SONAME_PATTERN, pattern):
        validate_external_soname(pattern)
        return
    parts = pattern.split("/")
    if (
        not pattern.startswith("/")
        or re.fullmatch(r"/[a-zA-Z0-9_./*+-]+", pattern) is None
        or len(parts) < 4
        or any(part in {".", ".."} for part in parts)
        or any("*" in part for part in parts[1:3])
    ):
        raise ValueError(f"invalid external runtime exclude pattern: {pattern!r}")


def parse_external_libraries(report_text: str) -> dict[str, str | None]:
    """Return auditwheel's complete external SONAME-to-path mapping."""
    if AUDITWHEEL_EXTERNAL_MARKER in report_text:
        payload = report_text.split(AUDITWHEEL_EXTERNAL_MARKER, 1)[1].lstrip()
        try:
            raw, _ = json.JSONDecoder().raw_decode(payload)
        except json.JSONDecodeError as error:
            raise ValueError("auditwheel external-library report is invalid") from error
        if not isinstance(raw, dict):
            raise ValueError("auditwheel external-library report must be a mapping")
        libraries: dict[str, str | None] = {}
        for soname, path in raw.items():
            if not isinstance(soname, str) or not soname:
                raise ValueError("auditwheel reported an invalid external SONAME")
            if path is not None and (not isinstance(path, str) or not path):
                raise ValueError(
                    f"auditwheel reported an invalid path for external library {soname}"
                )
            libraries[soname] = path
        return {name: libraries[name] for name in sorted(libraries)}
    if "The wheel requires no external shared libraries" in report_text:
        return {}
    raise ValueError("auditwheel report has no external-library result")


def _elf_dependency_graph(report_text: str) -> tuple[set[str], dict[str, set[str]]]:
    if report_text.count(AUDITWHEEL_ELF_TREE_MARKER) != 1:
        raise ValueError("auditwheel report has no unique ELF dependency graph")
    payload = report_text.split(AUDITWHEEL_ELF_TREE_MARKER, 1)[1]
    try:
        raw_trees, _ = json.JSONDecoder().raw_decode(payload)
    except json.JSONDecodeError as error:
        raise ValueError("auditwheel ELF dependency graph is invalid") from error
    if not isinstance(raw_trees, dict) or not raw_trees:
        raise ValueError("auditwheel ELF dependency graph must be a non-empty mapping")

    direct: set[str] = set()
    dependencies: dict[str, set[str]] = {}
    for raw_tree in raw_trees.values():
        if not isinstance(raw_tree, dict):
            raise ValueError("auditwheel ELF dependency tree is invalid")
        needed = raw_tree.get("needed")
        libraries = raw_tree.get("libraries")
        if not isinstance(needed, list) or not isinstance(libraries, dict):
            raise ValueError("auditwheel ELF dependency tree is incomplete")
        if any(not isinstance(item, str) or not item for item in needed):
            raise ValueError("auditwheel ELF dependency tree has an invalid edge")
        direct.update(needed)
        for soname, raw_library in libraries.items():
            if not isinstance(soname, str) or not isinstance(raw_library, dict):
                raise ValueError("auditwheel ELF dependency library is invalid")
            library_needed = raw_library.get("needed")
            if not isinstance(library_needed, list) or any(
                not isinstance(item, str) or not item for item in library_needed
            ):
                raise ValueError(
                    f"auditwheel ELF dependencies are invalid for {soname}"
                )
            dependencies.setdefault(soname, set()).update(library_needed)
    return direct, dependencies


def _reachable_libraries(
    roots: Sequence[str], dependencies: Mapping[str, set[str]]
) -> set[str]:
    reachable: set[str] = set()
    pending = list(roots)
    while pending:
        library = pending.pop()
        if library in reachable:
            continue
        reachable.add(library)
        pending.extend(dependencies.get(library, ()))
    return reachable


def validate_external_library_closure(
    report_text: str,
    *,
    expected_patterns: Sequence[str],
) -> dict[str, Any]:
    """Require every auditwheel external to descend from a matched provider root."""
    patterns = sorted(set(expected_patterns))
    for pattern in patterns:
        validate_exclude_pattern(pattern)
    external = parse_external_libraries(report_text)
    if "libmetrics.so" in external:
        raise ValueError("UCM-owned libmetrics.so remains external after repair")

    direct, dependencies = _elf_dependency_graph(report_text)
    direct_external = set(external) & direct

    def matches_boundary(soname: str) -> bool:
        path = external[soname]
        return any(
            fnmatchcase(soname, pattern)
            or (path is not None and fnmatchcase(path, pattern))
            for pattern in patterns
        )

    roots = sorted(soname for soname in direct_external if matches_boundary(soname))
    if not roots:
        raise ValueError(
            "auditwheel exclude patterns matched no direct external libraries"
        )
    unmatched_direct = sorted(direct_external - set(roots))
    if unmatched_direct:
        raise ValueError(
            "auditwheel found direct external libraries outside the provider boundary: "
            f"{unmatched_direct}"
        )
    unrooted = sorted(set(external) - _reachable_libraries(roots, dependencies))
    if unrooted:
        raise ValueError(
            "auditwheel found external libraries outside the provider closure: "
            f"{unrooted}"
        )
    unresolved_non_roots = sorted(
        soname
        for soname, path in external.items()
        if path is None and soname not in roots
    )
    for soname in unresolved_non_roots:
        validate_external_soname(soname)
    return {
        "external_library_roots": roots,
        "external_libraries": sorted(external),
        "deferred_external_libraries": unresolved_non_roots,
    }
