"""Read immutable OCI registry facts through Crane."""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence


def read(operation: str, reference: str) -> str:
    completed = None
    last_error = ""
    for attempt in range(1, 4):
        try:
            completed = subprocess.run(
                ["crane", operation, reference],
                text=True,
                capture_output=True,
                check=False,
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            completed = None
            last_error = "timed out after 60 seconds"
        if completed is None:
            if attempt < 3:
                time.sleep(2**attempt)
            continue
        if completed.returncode == 0:
            return completed.stdout
        last_error = completed.stderr.strip() or str(completed.returncode)
        if attempt < 3:
            time.sleep(2**attempt)
    raise ValueError(
        f"crane {operation} failed for {reference}: {last_error or 'unknown error'}"
    )


def _fixture_tags(
    fixture: Mapping[str, object] | None, repository: str
) -> list[str] | None:
    if fixture is None:
        return None
    repositories = fixture.get("repositories")
    if not isinstance(repositories, Mapping):
        return None
    raw_repository = repositories.get(repository)
    if not isinstance(raw_repository, Mapping):
        return None
    pages = raw_repository.get("pages")
    if not isinstance(pages, list):
        raise ValueError(f"Registry fixture {repository}: pages must be a list")
    tags: list[str] = []
    for index, raw_page in enumerate(pages):
        if not isinstance(raw_page, Mapping):
            raise ValueError(
                f"Registry fixture {repository}.pages[{index}] must be a mapping"
            )
        page = raw_page
        values = page.get("tags")
        if not isinstance(values, list) or not all(
            isinstance(tag, str) for tag in values
        ):
            raise ValueError(
                f"Registry fixture {repository}.pages[{index}].tags is invalid"
            )
        tags.extend(values)
    return sorted(set(tags))


def repository_tags(
    repository: str,
    *,
    tag_fixture: Mapping[str, object] | None,
    tag_loader: Callable[[str], Sequence[str]] | None,
) -> list[str]:
    fixture = _fixture_tags(tag_fixture, repository)
    if fixture is not None:
        return fixture
    values = (
        tag_loader(repository)
        if tag_loader is not None
        else read("ls", repository).splitlines()
    )
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(
            f"Registry tag loader for {repository} returned an invalid value"
        )
    return sorted(set(str(tag) for tag in values if str(tag)))


def read_json(operation: str, reference: str) -> object:
    try:
        return json.loads(read(operation, reference))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"crane {operation} returned malformed JSON for {reference}"
        ) from error
