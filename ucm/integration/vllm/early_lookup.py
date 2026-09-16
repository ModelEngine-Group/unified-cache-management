"""Call-local prepared results used by Early overrides of native methods."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class LookupContext:
    connector: object
    request: object
    job: object | None
    hbm_hit_hashes: int
    hbm_only: bool = False


# The native method and the existing Early Store see the same call context.
# It owns no jobs, resources, or worker state, and is reset in a finally block.
current_lookup: ContextVar[LookupContext | None] = ContextVar(
    "ucm_early_lookup", default=None
)
