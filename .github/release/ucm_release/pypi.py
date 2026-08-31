"""Build and execute the immutable backend-first public PyPI publication."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import InvalidVersion, Version

from . import policy as release_policy

PUBLICATION_KIND = "ucm-pypi-publication"
RECEIPT_KIND = "ucm-pypi-receipt"
SCHEMA_VERSION = 2

FetchRelease = Callable[[str, str], Mapping[str, Any] | None]
UploadFile = Callable[[str], object]
Sleep = Callable[[float], object]


class PyPIError(RuntimeError):
    """Base error for a terminal PyPI publication failure."""


class PyPIConflictError(PyPIError):
    """An immutable filename exists with different bytes."""


class PyPIReadbackError(PyPIError):
    """PyPI returned an unexpected release document."""


class PyPIReadbackTimeout(PyPIError):
    """An uploaded file did not become visible before the deadline."""


class PyPIUploadError(PyPIError):
    """A local upload artifact cannot be resolved or uploaded."""


def _mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    return dict(value)


def _version(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a version")
    try:
        parsed = Version(value)
    except InvalidVersion as error:
        raise ValueError(f"{context} must be a version") from error
    if str(parsed) != value or parsed.local is not None:
        raise ValueError(f"{context} must be canonical and non-local")
    return value


def _sha256(value: object, context: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a SHA256 digest")
    digest = value.removeprefix("sha256:")
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{context} must be a SHA256 digest")
    return f"sha256:{digest}"


def _wheel_coordinates(filename: str, context: str) -> tuple[str, str]:
    try:
        distribution, version, _build, _tags = parse_wheel_filename(filename)
    except ValueError as error:
        raise ValueError(f"{context} is not a valid Wheel filename") from error
    return canonicalize_name(str(distribution)), str(version)


def _validate_extras(
    value: object, backend_projects: set[str], version: str
) -> dict[str, str]:
    extras = _mapping(value, "meta extras")
    referenced: set[str] = set()
    normalized: dict[str, str] = {}
    for extra, raw_requirement in sorted(extras.items()):
        if (
            not isinstance(extra, str)
            or not extra
            or not isinstance(raw_requirement, str)
        ):
            raise ValueError("meta extras are invalid")
        try:
            requirement = Requirement(raw_requirement)
        except InvalidRequirement as error:
            raise ValueError(f"meta extra {extra!r} is invalid") from error
        specifiers = list(requirement.specifier)
        project = canonicalize_name(requirement.name)
        if (
            canonicalize_name(extra) != extra
            or requirement.url is not None
            or requirement.marker is not None
            or requirement.extras
            or len(specifiers) != 1
            or specifiers[0].operator != "=="
            or specifiers[0].version != version
            or project not in backend_projects
            or project in referenced
            or raw_requirement != f"{project}=={version}"
        ):
            raise ValueError(f"meta extra {extra!r} does not pin one planned backend")
        referenced.add(project)
        normalized[extra] = raw_requirement
    if referenced != backend_projects:
        raise ValueError("meta extras do not exactly cover backend projects")
    return normalized


def _authorized_target(plan: Mapping[str, Any]) -> dict[str, str]:
    publish = _mapping(plan.get("publish"), "release plan publish policy")
    pypi_policy = _mapping(publish.get("pypi"), "release plan PyPI policy")
    repository = plan.get("repository")
    scope = plan.get("publication_scope")
    target = pypi_policy.get("target")
    if not isinstance(repository, str) or pypi_policy.get("enabled") is not True:
        raise ValueError("Python index publication requires an enabled release plan")
    authorized = (
        scope == "official"
        and repository.casefold() == release_policy.OFFICIAL_REPOSITORY.casefold()
        and target == "pypi"
    ) or (
        scope == "fork"
        and repository.casefold() != release_policy.OFFICIAL_REPOSITORY.casefold()
        and target == "testpypi"
    )
    if not authorized:
        raise ValueError("Python index target does not match publication scope")
    expected = release_policy.PYPI_TARGETS[str(target)]
    observed = {field: pypi_policy.get(field) for field in expected}
    if observed != expected:
        raise ValueError("Python index endpoints differ from the authorized target")
    return {"target": str(target), **expected}


def build_publication(
    release_plan: Mapping[str, Any],
    backend_results: Sequence[Mapping[str, Any]],
    meta_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind exact built files into one backend-first publication contract."""
    plan = _mapping(release_plan, "release plan")
    if plan.get("kind") != "ucm-release-plan":
        raise ValueError("PyPI publication requires a release plan")
    target = _authorized_target(plan)
    version = _version(plan.get("version"), "release version")
    raw_tasks = plan.get("wheels")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("release plan has no backend Wheels")
    tasks: dict[str, dict[str, Any]] = {}
    for raw_task in raw_tasks:
        task = _mapping(raw_task, "Wheel task")
        task_id = task.get("id")
        if not isinstance(task_id, str) or not task_id or task_id in tasks:
            raise ValueError("release plan has an invalid Wheel task ID")
        if (
            _version(task.get("wheel_version"), f"Wheel task {task_id} version")
            != version
        ):
            raise ValueError(f"Wheel task {task_id!r} version differs from release")
        tasks[task_id] = task

    results: dict[str, dict[str, Any]] = {}
    for raw_result in backend_results:
        result = _mapping(raw_result, "backend Wheel result")
        task_id = result.get("task_id")
        if (
            result.get("kind") != "ucm-wheel-result"
            or result.get("schema_version") != 5
            or not isinstance(task_id, str)
            or task_id not in tasks
            or task_id in results
        ):
            raise ValueError("backend Wheel results do not match the release plan")
        results[task_id] = result
    if set(results) != set(tasks):
        raise ValueError("backend Wheel results do not cover the release plan")

    grouped: dict[str, dict[str, Any]] = {}
    filenames: set[str] = set()
    for task_id in sorted(tasks):
        task = tasks[task_id]
        result = results[task_id]
        project = canonicalize_name(str(task.get("dist_name", "")))
        filename = result.get("filename")
        if not project.startswith("uc-manager-") or not isinstance(filename, str):
            raise ValueError(f"Wheel task {task_id!r} has invalid coordinates")
        file_project, file_version = _wheel_coordinates(filename, f"Wheel {task_id}")
        if (
            canonicalize_name(str(result.get("distribution", ""))) != project
            or _version(result.get("version"), f"Wheel {task_id} version") != version
            or file_project != project
            or file_version != version
            or filename in filenames
        ):
            raise ValueError(f"Wheel result {task_id!r} has unexpected coordinates")
        filenames.add(filename)
        record = grouped.setdefault(
            project,
            {
                "project": project,
                "version": version,
                "role": "backend",
                "files": [],
            },
        )
        record["files"].append(
            {
                "filename": filename,
                "sha256": _sha256(result.get("sha256"), filename),
            }
        )
    backends = sorted(grouped.values(), key=lambda item: item["project"])
    for backend in backends:
        backend["files"].sort(key=lambda item: item["filename"])

    planned_meta = _mapping(plan.get("meta_package"), "release meta package")
    meta = _mapping(meta_result, "meta Wheel result")
    meta_filename = meta.get("filename")
    if not isinstance(meta_filename, str):
        raise ValueError("meta Wheel result has no filename")
    meta_project, meta_version = _wheel_coordinates(meta_filename, "meta Wheel")
    if (
        meta.get("kind") != "ucm-meta-result"
        or meta.get("schema_version") != 1
        or planned_meta.get("distribution") != "uc-manager"
        or planned_meta.get("version") != version
        or meta.get("distribution") != "uc-manager"
        or meta.get("version") != version
        or meta_project != "uc-manager"
        or meta_version != version
    ):
        raise ValueError("meta Wheel result has unexpected coordinates")
    extras = _validate_extras(
        planned_meta.get("extras"), {item["project"] for item in backends}, version
    )
    if meta.get("extras") != extras:
        raise ValueError("meta Wheel extras differ from the release plan")
    meta_project_record = {
        "project": "uc-manager",
        "version": version,
        "role": "meta",
        "files": [
            {
                "filename": meta_filename,
                "sha256": _sha256(meta.get("sha256"), meta_filename),
            }
        ],
    }
    return {
        "kind": PUBLICATION_KIND,
        "schema_version": SCHEMA_VERSION,
        "version": version,
        "target": target["target"],
        "repository_url": target["index"],
        "simple_index": target["simple_index"],
        "json_api": target["json_api"],
        "dependency_index": target["dependency_index"],
        "backends": backends,
        "meta": meta_project_record,
        "extras": extras,
    }


def fetch_version_json(
    project: str,
    version: str,
    *,
    attempts: int = 4,
    retry_interval: float = 1.0,
    timeout: float = 30.0,
    json_api_url: str = "https://pypi.org/pypi/",
    open_url: Callable[..., Any] = urlopen,
    sleep: Sleep = time.sleep,
) -> dict[str, Any] | None:
    """Fetch one version document with cache busting and bounded retry."""
    if not json_api_url.startswith("https://") or not json_api_url.endswith("/"):
        raise ValueError("Python index JSON API must be an HTTPS base URL")
    for attempt in range(1, attempts + 1):
        query = urlencode({"ucm_readback": uuid.uuid4().hex})
        url = (
            f"{json_api_url}{quote(project, safe='')}/"
            f"{quote(version, safe='')}/json?{query}"
        )
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
                "User-Agent": "ucm-release-pypi-readback/1",
            },
        )
        try:
            response = open_url(request, timeout=timeout)
        except HTTPError as error:
            if error.code == 404:
                return None
            retryable = error.code in {408, 429} or 500 <= error.code <= 599
            if retryable and attempt < attempts:
                sleep(retry_interval * attempt)
                continue
            raise PyPIReadbackError(
                f"PyPI HTTP {error.code} for {project} {version}"
            ) from error
        except (OSError, URLError) as error:
            if attempt < attempts:
                sleep(retry_interval * attempt)
                continue
            raise PyPIReadbackError(
                f"PyPI request failed for {project} {version}"
            ) from error
        try:
            payload = response.read()
            status = getattr(response, "status", 200)
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()
        if status != 200:
            raise PyPIReadbackError(f"PyPI HTTP {status} for {project} {version}")
        try:
            document = json.loads(payload)
        except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PyPIReadbackError(
                f"PyPI returned malformed JSON for {project}"
            ) from error
        if not isinstance(document, dict):
            raise PyPIReadbackError(f"PyPI returned malformed JSON for {project}")
        return document
    raise AssertionError("unreachable PyPI retry state")


def _classify_project(
    expected: Mapping[str, Any], document: Mapping[str, Any] | None
) -> list[str]:
    expected_files = {item["filename"]: item["sha256"] for item in expected["files"]}
    if document is None:
        return sorted(expected_files)
    info = _mapping(document.get("info"), "PyPI release info")
    project = str(expected["project"])
    version = str(expected["version"])
    if (
        canonicalize_name(str(info.get("name", ""))) != project
        or str(info.get("version", "")) != version
        or info.get("yanked") is True
    ):
        raise PyPIReadbackError(f"PyPI returned unexpected coordinates for {project}")
    urls = document.get("urls")
    if not isinstance(urls, list):
        raise PyPIReadbackError(f"PyPI returned no file list for {project}")
    observed: dict[str, str] = {}
    for raw_file in urls:
        file = _mapping(raw_file, "PyPI release file")
        filename = file.get("filename")
        digest = file.get("digests")
        if (
            not isinstance(filename, str)
            or filename in observed
            or file.get("packagetype") != "bdist_wheel"
            or file.get("yanked") is not False
            or not isinstance(digest, Mapping)
        ):
            raise PyPIReadbackError(f"PyPI returned an invalid file for {project}")
        if filename not in expected_files:
            raise PyPIReadbackError(f"PyPI returned unexpected file {filename!r}")
        observed[filename] = _sha256(digest.get("sha256"), filename)
    missing: list[str] = []
    for filename, expected_digest in expected_files.items():
        if filename not in observed:
            missing.append(filename)
        elif observed[filename] != expected_digest:
            raise PyPIConflictError(f"PyPI immutable filename conflict: {filename}")
    return sorted(missing)


def _validate_meta_metadata(
    publication: Mapping[str, Any], document: Mapping[str, Any]
) -> None:
    info = _mapping(document.get("info"), "meta PyPI release info")
    extras = publication["extras"]
    provided = info.get("provides_extra")
    if not isinstance(provided, list) or sorted(
        canonicalize_name(str(item)) for item in provided
    ) != sorted(extras):
        raise PyPIReadbackError("meta PyPI Provides-Extra differs from the plan")
    requirements = info.get("requires_dist")
    if not isinstance(requirements, list):
        raise PyPIReadbackError("meta PyPI Requires-Dist is missing")
    observed: dict[str, str] = {}
    for raw in requirements:
        if not isinstance(raw, str):
            raise PyPIReadbackError("meta PyPI Requires-Dist is invalid")
        try:
            requirement = Requirement(raw)
        except InvalidRequirement as error:
            raise PyPIReadbackError("meta PyPI Requires-Dist is invalid") from error
        marker = str(requirement.marker) if requirement.marker is not None else ""
        matched = next(
            (extra for extra in extras if marker == f'extra == "{extra}"'), None
        )
        requirement.marker = None
        if (
            matched is None
            or str(requirement) != extras[matched]
            or matched in observed
        ):
            raise PyPIReadbackError("meta PyPI Requires-Dist differs from the plan")
        observed[matched] = str(requirement)
    if observed != extras:
        raise PyPIReadbackError("meta PyPI Requires-Dist differs from the plan")


def _read_phase(
    projects: Sequence[Mapping[str, Any]], fetch: FetchRelease
) -> tuple[dict[str, Mapping[str, Any] | None], list[str]]:
    documents: dict[str, Mapping[str, Any] | None] = {}
    missing: list[str] = []
    for project in projects:
        name = str(project["project"])
        document = fetch(name, str(project["version"]))
        documents[name] = document
        missing.extend(_classify_project(project, document))
    return documents, sorted(missing)


def _poll_phase(
    projects: Sequence[Mapping[str, Any]],
    *,
    fetch: FetchRelease,
    sleep: Sleep,
    attempts: int,
    interval: float,
) -> dict[str, Mapping[str, Any] | None]:
    for attempt in range(1, attempts + 1):
        documents, missing = _read_phase(projects, fetch)
        if not missing:
            return documents
        if attempt == attempts:
            raise PyPIReadbackTimeout(
                "PyPI files remained missing: " + ", ".join(missing)
            )
        sleep(interval)
    raise AssertionError("unreachable PyPI polling state")


def publish(
    publication: Mapping[str, Any],
    *,
    uploader: UploadFile,
    fetch: FetchRelease = fetch_version_json,
    sleep: Sleep = time.sleep,
    attempts: int = 12,
    interval: float = 5.0,
) -> dict[str, Any]:
    """Upload missing backends, read them back, then upload and verify meta."""
    if (
        publication.get("kind") != PUBLICATION_KIND
        or publication.get("schema_version") != SCHEMA_VERSION
        or not isinstance(publication.get("repository_url"), str)
    ):
        raise ValueError("invalid PyPI publication contract")
    backends = publication["backends"]
    meta = publication["meta"]
    if not isinstance(backends, list) or not backends or not isinstance(meta, Mapping):
        raise ValueError("invalid PyPI publication contract")

    _backend_documents, missing_backends = _read_phase(backends, fetch)
    meta_document = fetch(str(meta["project"]), str(meta["version"]))
    missing_meta = _classify_project(meta, meta_document)
    if not missing_meta and meta_document is not None:
        _validate_meta_metadata(publication, meta_document)
        if missing_backends:
            raise PyPIReadbackError(
                "meta Wheel is public while planned backend Wheels are missing"
            )

    for filename in missing_backends:
        uploader(filename)
    _poll_phase(
        backends,
        fetch=fetch,
        sleep=sleep,
        attempts=attempts,
        interval=interval,
    )

    meta_document = fetch(str(meta["project"]), str(meta["version"]))
    missing_meta = _classify_project(meta, meta_document)
    for filename in missing_meta:
        uploader(filename)
    final_documents = _poll_phase(
        [*backends, meta],
        fetch=fetch,
        sleep=sleep,
        attempts=attempts,
        interval=interval,
    )
    final_meta = final_documents[str(meta["project"])]
    if final_meta is None:
        raise PyPIReadbackTimeout("meta PyPI release remained missing")
    _validate_meta_metadata(publication, final_meta)
    return {
        "kind": RECEIPT_KIND,
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "version": publication["version"],
        "target": publication["target"],
        "repository_url": publication["repository_url"],
        "projects": copy.deepcopy([*backends, meta]),
        "extras": copy.deepcopy(publication["extras"]),
    }


def _exact_file(filename: str, roots: Sequence[Path]) -> Path:
    matches = {
        path.resolve(): path
        for root in roots
        for path in ([root] if root.is_file() else root.rglob("*"))
        if path.is_file() and path.name == filename
    }
    if len(matches) != 1:
        raise PyPIUploadError(f"upload file {filename!r} resolved {len(matches)} times")
    return next(iter(matches.values()))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def make_twine_uploader(
    *,
    roots: Sequence[Path],
    expected_sha256: Mapping[str, str],
    repository_url: str,
    token: str,
) -> UploadFile:
    """Create the exact-file Twine uploader used by the workflow CLI."""
    if not repository_url.startswith("https://") or not token:
        raise ValueError("PyPI repository and token are required")

    def upload(filename: str) -> Path:
        path = _exact_file(filename, roots)
        if _file_sha256(path) != expected_sha256.get(filename):
            raise PyPIUploadError(f"upload file {filename!r} differs from its result")
        environment = os.environ.copy()
        environment.update({"TWINE_USERNAME": "__token__", "TWINE_PASSWORD": token})
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "twine",
                    "upload",
                    "--non-interactive",
                    "--repository-url",
                    repository_url,
                    str(path),
                ],
                check=True,
                env=environment,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise PyPIUploadError(f"Twine upload failed for {filename!r}") from error
        return path

    return upload
