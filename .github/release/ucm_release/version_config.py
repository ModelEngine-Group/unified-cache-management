"""Strict parser for the repository-owned ``version.ini`` release authority."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from packaging.version import InvalidVersion, Version

UCM_VERSION_KEY = "VLLM_UC_VERSION"
SUPPORTED_VERSION_KEYS = {
    "vllm": "UCM_SUPPORTED_VLLM_VERSIONS",
    "vllm-ascend": "UCM_SUPPORTED_VLLM_ASCEND_VERSIONS",
}
VERSION_KEYS = (UCM_VERSION_KEY, *SUPPORTED_VERSION_KEYS.values())
OCI_TAG_PATTERN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", re.ASCII)


def _canonical_version(value: str, context: str) -> str:
    try:
        parsed = Version(value)
    except InvalidVersion as error:
        raise ValueError(f"{context} must be a valid PEP 440 version") from error
    if str(parsed) != value:
        raise ValueError(f"{context} must use canonical PEP 440 spelling")
    if parsed.epoch != 0 or parsed.local is not None:
        raise ValueError(f"{context} must be public and non-local")
    if len(parsed.release) != 3:
        raise ValueError(f"{context} must use an X.Y.Z release tuple")
    return value


def runtime_keyword(value: object, context: str) -> str:
    """Validate one literal Registry-tag keyword without version normalization."""

    if not isinstance(value, str) or OCI_TAG_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{context} must be a valid literal OCI tag keyword")
    return value


def _assignments(text: str, source: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, raw_value = line.partition("=")
        if (
            not separator
            or not key
            or key.strip() != key
            or raw_value.strip() != raw_value
        ):
            raise ValueError(f"{source}:{line_number}: invalid version assignment")
        if key not in VERSION_KEYS:
            raise ValueError(f"{source}:{line_number}: unsupported version key {key!r}")
        if key in values:
            raise ValueError(f"{source}:{line_number}: duplicate version key {key!r}")
        if not raw_value:
            raise ValueError(f"{source}:{line_number}: {key} must not be empty")
        values[key] = raw_value
    missing = sorted(set(VERSION_KEYS) - set(values))
    if missing:
        raise ValueError(f"{source}: missing version keys: {missing}")
    return values


def _selectors(value: str, product_id: str) -> list[dict[str, str | None]]:
    result: list[dict[str, str | None]] = []
    seen: set[tuple[str, str | None]] = set()
    for index, token in enumerate(value.split(","), start=1):
        context = f"{SUPPORTED_VERSION_KEYS[product_id]}[{index}]"
        if not token or token.strip() != token:
            raise ValueError(f"{context} must be a non-empty selector without spaces")
        raw_keyword, separator, raw_tag = token.partition("@")
        keyword = runtime_keyword(raw_keyword, f"{context} keyword")
        tag: str | None = None
        if separator:
            if (
                not raw_tag
                or "@" in raw_tag
                or OCI_TAG_PATTERN.fullmatch(raw_tag) is None
            ):
                raise ValueError(f"{context} has an invalid OCI tag")
            tag = raw_tag
        identity = (keyword, tag)
        if identity in seen:
            raise ValueError(f"{context} duplicates an earlier selector")
        seen.add(identity)
        result.append(
            {
                "raw": keyword if tag is None else f"{keyword}@{tag}",
                "keyword": keyword,
                "tag": tag,
            }
        )
    if not result:
        raise ValueError(f"{SUPPORTED_VERSION_KEYS[product_id]} must not be empty")
    return result


def parse(text: str, *, source: str = "version.ini") -> dict[str, Any]:
    """Parse and normalize one complete version authority document."""

    assignments = _assignments(text, source)
    ucm_version = _canonical_version(assignments[UCM_VERSION_KEY], UCM_VERSION_KEY)
    supported = {
        product_id: _selectors(assignments[key], product_id)
        for product_id, key in SUPPORTED_VERSION_KEYS.items()
    }
    authority = {
        "ucm_base_version": Version(ucm_version).base_version,
        "supported_runtimes": supported,
    }
    authority_sha256 = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                authority, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
        ).hexdigest()
    )
    return {
        "ucm_version": ucm_version,
        **authority,
        "authority_sha256": authority_sha256,
    }


def load(path: Path) -> dict[str, Any]:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"cannot read version authority {path}") from error
    return parse(text, source=str(path))


def render(config: dict[str, Any], *, ucm_version: str | None = None) -> str:
    """Render a shell-sourceable canonical document, optionally changing UCM version."""

    resolved_version = _canonical_version(
        ucm_version or str(config["ucm_version"]), UCM_VERSION_KEY
    )
    lines = [f"{UCM_VERSION_KEY}={resolved_version}"]
    supported = config.get("supported_runtimes")
    if not isinstance(supported, dict):
        raise ValueError("version authority has no supported runtime selectors")
    for product_id, key in SUPPORTED_VERSION_KEYS.items():
        selectors = supported.get(product_id)
        if not isinstance(selectors, list) or not selectors:
            raise ValueError(f"version authority has no selectors for {product_id}")
        values = [str(selector["raw"]) for selector in selectors]
        lines.append(f"{key}={','.join(values)}")
    return "\n".join(lines) + "\n"


def materialize_bytes(text: str, version: str, *, source: str = "version.ini") -> bytes:
    return render(parse(text, source=source), ucm_version=version).encode("utf-8")
