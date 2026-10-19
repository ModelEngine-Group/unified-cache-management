"""Strict parser for the repository-owned ``version.ini`` release authority."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from packaging.version import InvalidVersion, Version

UCM_VERSION_KEY = "UCM_VERSION"
SUPPORTED_VERSION_KEYS = {
    "vllm": "UCM_SUPPORTED_VLLM_VERSIONS",
    "vllm-ascend": "UCM_SUPPORTED_VLLM_ASCEND_VERSIONS",
    "sglang": "UCM_SUPPORTED_SGLANG_VERSIONS",
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


def runtime_selector_version(value: object, context: str) -> str:
    """Validate one canonical Runtime minor or patch selector."""

    if not isinstance(value, str):
        raise ValueError(f"{context} must be a canonical X.Y or X.Y.Z version")
    try:
        parsed = Version(value)
    except InvalidVersion as error:
        raise ValueError(
            f"{context} must be a canonical X.Y or X.Y.Z version"
        ) from error
    if (
        str(parsed) != value
        or parsed.epoch != 0
        or parsed.pre is not None
        or parsed.post is not None
        or parsed.dev is not None
        or parsed.local is not None
        or len(parsed.release) not in {2, 3}
    ):
        raise ValueError(f"{context} must be a canonical X.Y or X.Y.Z version")
    return value


def _selector_ranges_overlap(left: Version, right: Version) -> bool:
    prefix_length = min(len(left.release), len(right.release))
    return left.release[:prefix_length] == right.release[:prefix_length]


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
    selected_ranges: list[Version] = []
    for index, token in enumerate(value.split(","), start=1):
        context = f"{SUPPORTED_VERSION_KEYS[product_id]}[{index}]"
        if not token or token.strip() != token:
            raise ValueError(f"{context} must be a non-empty selector without spaces")
        raw_version, separator, raw_tag = token.partition("@")
        version = runtime_selector_version(raw_version, f"{context} version")
        tag: str | None = None
        if separator:
            if (
                not raw_tag
                or "@" in raw_tag
                or OCI_TAG_PATTERN.fullmatch(raw_tag) is None
            ):
                raise ValueError(f"{context} has an invalid OCI tag")
            tag = raw_tag
        parsed_version = Version(version)
        if any(
            _selector_ranges_overlap(parsed_version, earlier)
            for earlier in selected_ranges
        ):
            raise ValueError(f"{context} overlaps an earlier selector")
        selected_ranges.append(parsed_version)
        result.append(
            {
                "raw": version if tag is None else f"{version}@{tag}",
                "version": version,
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


def read_version(path: Path | None = None) -> str:
    version_path = path or (Path(__file__).resolve().parents[3] / "version.ini")
    return str(load(version_path)["ucm_version"])


def derive_chart_version(version: str) -> str:
    parsed = Version(_canonical_version(version, "UCM release version"))
    if parsed.dev is not None:
        if (
            parsed.pre is not None
            or parsed.post is not None
            or parsed.local is not None
        ):
            raise ValueError(
                f"unsupported UCM draft version for Chart SemVer: {version}"
            )
        return f"{parsed.base_version}-draft.{parsed.dev}"
    public = version
    match = re.fullmatch(r"([0-9]+\.[0-9]+\.[0-9]+)rc([0-9]+)", public)
    if match is None:
        if re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", public):
            return public
        raise ValueError(f"unsupported UCM release version for Chart SemVer: {version}")
    return f"{match.group(1)}-rc.{match.group(2)}"


DEFAULT_OUTPUT = Path(__file__).resolve().parents[3] / "version.ini"
VERSION_TRIPLE_PATTERN = (
    r"(?:0|[1-9][0-9]*)\." r"(?:0|[1-9][0-9]*)\." r"(?:0|[1-9][0-9]*)"
)
FORMAL_TAG = re.compile(
    r"v(?P<version>" + VERSION_TRIPLE_PATTERN + r"(?:rc(?:0|[1-9][0-9]*))?)",
    re.ASCII,
)
STABLE_TAG = re.compile(
    r"v(?P<version>" + VERSION_TRIPLE_PATTERN + r")",
    re.ASCII,
)
DRAFT_TAG = re.compile(
    r"draft/v(?P<base>" + VERSION_TRIPLE_PATTERN + r")(?:-(?P<number>[1-9][0-9]*))?",
    re.ASCII,
)
NIGHTLY_TAG = re.compile(
    r"nightly/v(?P<base>"
    + VERSION_TRIPLE_PATTERN
    + r")-(?P<date>[0-9]{8})-(?P<number>[1-9][0-9]*)",
    re.ASCII,
)


def canonical_version(value: str) -> str:
    """Return *value* only when it is already canonical PEP 440."""
    try:
        parsed = Version(value)
    except InvalidVersion as error:
        raise ValueError(f"invalid PEP 440 version: {value!r}") from error
    canonical = str(parsed)
    if value != canonical:
        raise ValueError(
            f"version must use canonical PEP 440 spelling: {value!r} != {canonical!r}"
        )
    return canonical


def _validate_nightly_date(value: str) -> str:
    try:
        parsed = datetime.strptime(value, "%Y%m%d")
    except ValueError as error:
        raise ValueError(f"invalid Nightly date: {value!r}") from error
    if parsed.strftime("%Y%m%d") != value:
        raise ValueError(f"invalid Nightly date: {value!r}")
    return value


def version_from_tag(tag: str) -> str:
    """Translate an exact supported release Tag to its Wheel version."""
    draft = DRAFT_TAG.fullmatch(tag)
    if draft is not None:
        number = draft.group("number") or "0"
        return canonical_version(f"{draft.group('base')}.dev{number}")
    nightly = NIGHTLY_TAG.fullmatch(tag)
    if nightly is not None:
        release_date = _validate_nightly_date(nightly.group("date"))
        number = int(nightly.group("number"))
        return canonical_version(
            f"{nightly.group('base')}.dev{release_date}{number:03d}"
        )
    formal = FORMAL_TAG.fullmatch(tag)
    if formal is not None:
        return canonical_version(formal.group("version"))
    raise ValueError(f"unsupported UCM release tag: {tag!r}")


def classify_tag(tag: str) -> dict[str, object]:
    """Return the immutable artifact coordinates and GitHub Release mode for *tag*."""
    version = version_from_tag(tag)
    draft = DRAFT_TAG.fullmatch(tag)
    if draft is not None:
        number = int(draft.group("number") or "0")
        return {
            "git_tag": tag,
            "release_type": "draft",
            "release_kind": "draft",
            "version": version,
            "chart_version": f"{draft.group('base')}-draft.{number}",
            "image_version": version,
            "is_prerelease": True,
        }

    nightly = NIGHTLY_TAG.fullmatch(tag)
    if nightly is not None:
        release_date = _validate_nightly_date(nightly.group("date"))
        number = int(nightly.group("number"))
        return {
            "git_tag": tag,
            "release_type": "nightly",
            "release_kind": "publish",
            "version": version,
            "chart_version": (
                f"{nightly.group('base')}-nightly.{release_date}.{number}"
            ),
            "image_version": version,
            "is_prerelease": True,
        }

    parsed = Version(version)
    chart_version = parsed.base_version
    if parsed.pre is not None:
        label, number = parsed.pre
        if label != "rc":
            raise ValueError(f"unsupported formal prerelease tag: {tag!r}")
        chart_version = f"{parsed.base_version}-rc.{number}"
    return {
        "git_tag": tag,
        "release_type": "prerelease" if parsed.is_prerelease else "stable",
        "release_kind": "publish",
        "version": version,
        "chart_version": chart_version,
        "image_version": version,
        "is_prerelease": parsed.is_prerelease,
    }


def next_nightly_sequence(
    tags: Iterable[str], *, base_version: str, release_date: str
) -> int:
    """Return the next sequence for one exact Nightly base and date."""
    if STABLE_TAG.fullmatch(f"v{base_version}") is None:
        raise ValueError(f"invalid Nightly base version: {base_version!r}")
    _validate_nightly_date(release_date)
    sequences = [
        int(match.group("number"))
        for tag in tags
        if (match := NIGHTLY_TAG.fullmatch(tag)) is not None
        and match.group("base") == base_version
        and match.group("date") == release_date
    ]
    return max(sequences, default=0) + 1


def next_nightly_classification(
    tags: Iterable[str], *, base_version: str, release_date: str
) -> dict[str, object]:
    """Classify the next Nightly for the repository-owned base version."""
    known_tags = tuple(tags)
    sequence = next_nightly_sequence(
        known_tags, base_version=base_version, release_date=release_date
    )
    return classify_tag(f"nightly/v{base_version}-{release_date}-{sequence}")


def validate_tag_against_config(tag: str, config_path: Path) -> dict[str, object]:
    classification = classify_tag(tag)
    config = load(config_path)
    tag_base = Version(str(classification["version"])).base_version
    if tag_base != config["ucm_base_version"]:
        raise ValueError(
            "release Tag base version differs from version.ini: "
            f"{tag_base} != {config['ucm_base_version']}"
        )
    return classification


def materialize_version(version: str, output: Path = DEFAULT_OUTPUT) -> str:
    """Atomically replace only UCM_VERSION in the version authority."""
    canonical = canonical_version(version)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = target.stat().st_mode & 0o777 if target.exists() else 0o644
    source = target if target.exists() else DEFAULT_OUTPUT
    rendered = render(load(source), ucm_version=canonical)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            delete=False,
        ) as temporary:
            temporary.write(rendered)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_name = temporary.name
        os.chmod(temporary_name, mode)
        os.replace(temporary_name, target)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return canonical
