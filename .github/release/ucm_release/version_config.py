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

from .runtime import oci_tag_version

UCM_VERSION_KEY = "UCM_VERSION"
SUPPORTED_VERSION_KEYS = {
    "vllm": "UCM_SUPPORTED_VLLM_VERSIONS",
    "vllm-ascend": "UCM_SUPPORTED_VLLM_ASCEND_VERSIONS",
    "sglang": "UCM_SUPPORTED_SGLANG_VERSIONS",
}
VERSION_KEYS = (UCM_VERSION_KEY, *SUPPORTED_VERSION_KEYS.values())
OCI_TAG_PATTERN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", re.ASCII)


def _draft_label(value: str, context: str) -> str:
    """Validate a named Draft suffix without normalizing distinct inputs."""
    if (
        re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", value, re.ASCII) is None
        or re.search(r"[a-z]", value) is None
        or any(
            part.isdigit() and len(part) > 1 and part.startswith("0")
            for part in value.split("-")
        )
    ):
        raise ValueError(
            f"{context} must use lowercase letters, digits and single hyphens, "
            "include a letter, and have no leading zeros in numeric segments"
        )
    return value


def release_version(value: object, context: str = "UCM release version") -> str:
    """Validate a UCM version, allowing local identity only for named Drafts."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty version string")
    try:
        parsed = Version(value)
    except InvalidVersion as error:
        raise ValueError(f"{context} must be a valid PEP 440 version") from error
    if str(parsed) != value:
        raise ValueError(f"{context} must use canonical PEP 440 spelling")
    if parsed.epoch != 0:
        raise ValueError(f"{context} must not use an epoch")
    if len(parsed.release) != 3:
        raise ValueError(f"{context} must use an X.Y.Z release tuple")
    if parsed.local is not None:
        if (
            parsed.dev != 0
            or parsed.pre is not None
            or parsed.post is not None
            or not parsed.local.startswith("draft.")
        ):
            raise ValueError(
                f"{context} local version must use X.Y.Z.dev0+draft.<label>"
            )
        _draft_label(parsed.local.removeprefix("draft.").replace(".", "-"), context)
    return value


def custom_draft_tag(version: str) -> str | None:
    """Recover a named Draft Tag; ordinary release versions return None."""
    parsed = Version(release_version(version))
    if parsed.local is None:
        return None
    suffix = parsed.local.removeprefix("draft.").replace(".", "-")
    return f"draft/v{parsed.base_version}-{suffix}"


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
        if key == UCM_VERSION_KEY and not raw_value:
            raise ValueError(f"{source}:{line_number}: {key} must not be empty")
        values[key] = raw_value
    if UCM_VERSION_KEY not in values:
        raise ValueError(f"{source}: missing version key: {UCM_VERSION_KEY}")
    return values


def _selectors(value: str, product_id: str) -> list[dict[str, str | None]]:
    result: list[dict[str, str | None]] = []
    selected_ranges: list[Version] = []
    selected_tags: set[str] = set()
    if not value:
        return result
    for index, token in enumerate(value.split(","), start=1):
        context = f"{SUPPORTED_VERSION_KEYS[product_id]}[{index}]"
        if not token or token.strip() != token:
            raise ValueError(f"{context} must be a non-empty selector without spaces")
        version: str | None = None
        tag: str | None = None
        if token.startswith("@"):
            tag = token[1:]
            if OCI_TAG_PATTERN.fullmatch(tag) is None:
                raise ValueError(f"{context} has an invalid OCI tag")
            if tag in selected_tags:
                raise ValueError(f"{context} duplicates an earlier OCI tag")
            selected_tags.add(tag)
        else:
            if "@" in token:
                raise ValueError(
                    f"{context}: version@Tag is no longer supported; use @Tag instead"
                )
            version = runtime_selector_version(token, f"{context} version")
            parsed_version = Version(version)
            if any(
                _selector_ranges_overlap(parsed_version, earlier)
                for earlier in selected_ranges
            ):
                raise ValueError(f"{context} overlaps an earlier selector")
            selected_ranges.append(parsed_version)
        result.append(
            {
                "raw": token,
                "version": version,
                "tag": tag,
            }
        )
    return result


def parse(text: str, *, source: str = "version.ini") -> dict[str, Any]:
    """Parse version authority; absent or empty Runtime keys disable that product."""

    assignments = _assignments(text, source)
    ucm_version = release_version(assignments[UCM_VERSION_KEY], UCM_VERSION_KEY)
    supported = {
        product_id: _selectors(assignments.get(key, ""), product_id)
        for product_id, key in SUPPORTED_VERSION_KEYS.items()
    }
    if not any(supported.values()):
        raise ValueError(f"{source}: at least one Runtime product must be enabled")
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

    resolved_version = release_version(
        ucm_version or str(config["ucm_version"]), UCM_VERSION_KEY
    )
    lines = [f"{UCM_VERSION_KEY}={resolved_version}"]
    supported = config.get("supported_runtimes")
    if not isinstance(supported, dict):
        raise ValueError("version authority has no supported runtime selectors")
    for product_id, key in SUPPORTED_VERSION_KEYS.items():
        selectors = supported.get(product_id)
        if not isinstance(selectors, list):
            raise ValueError(f"version authority has no selectors for {product_id}")
        if not selectors:
            continue
        values = [str(selector["raw"]) for selector in selectors]
        lines.append(f"{key}={','.join(values)}")
    if len(lines) == 1:
        raise ValueError(
            "version authority: at least one Runtime product must be enabled"
        )
    return "\n".join(lines) + "\n"


def materialize_bytes(text: str, version: str, *, source: str = "version.ini") -> bytes:
    return render(parse(text, source=source), ucm_version=version).encode("utf-8")


def read_version(path: Path | None = None) -> str:
    version_path = path or (Path(__file__).resolve().parents[3] / "version.ini")
    return str(load(version_path)["ucm_version"])


def derive_chart_version(version: str) -> str:
    parsed = Version(release_version(version))
    if parsed.local is not None:
        return f"{parsed.base_version}-draft.{parsed.local.removeprefix('draft.')}"
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
    r"draft/v(?P<base>"
    + VERSION_TRIPLE_PATTERN
    + r")(?:-(?P<suffix>[a-z0-9]+(?:-[a-z0-9]+)*))?",
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
        suffix = draft.group("suffix")
        if suffix is None:
            return release_version(f"{draft.group('base')}.dev0")
        if re.fullmatch(r"[1-9][0-9]*", suffix) is not None:
            return release_version(f"{draft.group('base')}.dev{suffix}")
        label = _draft_label(suffix, f"Draft Tag {tag!r}").replace("-", ".")
        return release_version(f"{draft.group('base')}.dev0+draft.{label}")
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
        return {
            "git_tag": tag,
            "release_type": "draft",
            "release_kind": "draft",
            "version": version,
            "chart_version": derive_chart_version(version),
            "image_version": oci_tag_version(version),
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


def validate_custom_draft_identity(
    version: str,
    git_tag: object,
    release_type: object,
    *,
    context: str = "release",
) -> dict[str, object] | None:
    """Bind named Draft identities without tightening ordinary release contracts."""
    expected_tag = custom_draft_tag(version)
    coordinates = (
        classify_tag(git_tag)
        if isinstance(git_tag, str) and git_tag.startswith("draft/")
        else None
    )
    if expected_tag is None and (
        coordinates is None or custom_draft_tag(str(coordinates["version"])) is None
    ):
        return None
    if (
        coordinates is None
        or git_tag != expected_tag
        or coordinates["version"] != version
        or coordinates["release_type"] != release_type
    ):
        raise ValueError(
            f"custom Draft package version differs from {context} identity"
        )
    return coordinates


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
