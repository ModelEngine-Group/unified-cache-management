"""Shared test fixtures for the release-pipeline test suite.

The catalog loader infers the running repository from ``GITHUB_REPOSITORY``
(or the ``origin`` git remote as a local fallback).  In CI that env var is
always set; in the test runner it is absent, so without a default the loader
would fall back to the developer's fork remote and resolve ``{owner}``
templates to that fork's owner.  Pinning a synthetic owner here keeps the
tests deterministic and owner-agnostic: no real organisation name is ever
asserted.
"""

from __future__ import annotations

import pytest

TEST_REPOSITORY = "release-org/unified-cache-management"


@pytest.fixture(autouse=True)
def _default_github_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a default ``GITHUB_REPOSITORY`` for every test.

    Tests that need a different identity (e.g. forge-mismatch cases) override
    this with their own ``monkeypatch.setenv`` call, which takes precedence.
    """
    monkeypatch.setenv("GITHUB_REPOSITORY", TEST_REPOSITORY)
