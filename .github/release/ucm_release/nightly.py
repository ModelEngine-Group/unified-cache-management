"""Branch selection, tag spelling and wheel coordinates for nightly releases."""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from packaging.version import Version

NUMBER = r"(?:0|[1-9][0-9]*)"
RELEASE_BRANCH = rf"0\.{NUMBER}\.{NUMBER}-release"
BRANCH = rf"(?:develop|{RELEASE_BRANCH})"
TAG = re.compile(
    rf"nightly/(?P<branch>{BRANCH})-(?P<date>[0-9]{{8}})(?:-(?P<n>[1-9][0-9]*))?"
)
LEGACY = re.compile(
    rf"nightly/v(?P<base>{NUMBER}\.{NUMBER}\.{NUMBER})-(?P<date>[0-9]{{8}})-(?P<n>[1-9][0-9]*)"
)


def latest_release_branch(branches: Iterable[str]) -> str | None:
    versions = (b for b in branches if re.fullmatch(RELEASE_BRANCH, b))
    return max(
        versions,
        key=lambda b: tuple(map(int, b.removesuffix("-release").split("."))),
        default=None,
    )


@dataclass(frozen=True)
class Nightly:
    branch: str
    date: str
    sequence: int
    legacy_base: str | None = None

    def __post_init__(self) -> None:
        if not re.fullmatch(BRANCH, self.branch):
            raise ValueError(f"invalid Nightly branch: {self.branch!r}")
        if not re.fullmatch(r"[0-9]{8}", self.date):
            raise ValueError(f"invalid Nightly date: {self.date!r}")
        try:
            datetime.strptime(self.date, "%Y%m%d")
        except ValueError as error:
            raise ValueError(f"invalid Nightly date: {self.date!r}") from error
        if self.sequence < 1:
            raise ValueError("Nightly sequence must be positive")

    @classmethod
    def parse(cls, tag: str) -> Nightly | None:
        match = TAG.fullmatch(tag)
        if match:
            if match["n"] == "1":
                raise ValueError("the first branch Nightly has no sequence suffix")
            return cls(match["branch"], match["date"], int(match["n"] or 1))
        match = LEGACY.fullmatch(tag)
        if match:
            return cls("develop", match["date"], int(match["n"]), match["base"])
        return None

    @property
    def tag(self) -> str:
        if self.legacy_base:
            return f"nightly/v{self.legacy_base}-{self.date}-{self.sequence}"
        suffix = "" if self.sequence == 1 else f"-{self.sequence}"
        return f"nightly/{self.branch}-{self.date}{suffix}"

    @property
    def label(self) -> str:
        if self.branch == "develop":
            return "develop"
        return "release." + self.branch.removesuffix("-release")

    def version(self, source_base: str | None) -> str:
        base = self.legacy_base or source_base
        if base is None or not re.fullmatch(rf"{NUMBER}\.{NUMBER}\.{NUMBER}", base):
            raise ValueError(
                f"{self.tag} requires the base version from the tagged source"
            )
        suffix = "" if self.legacy_base else "+" + self.label
        return f"{base}.dev{self.date}{self.sequence:03d}{suffix}"

    @classmethod
    def from_version(cls, version: Version) -> Nightly | None:
        label = version.local
        if not label or label.startswith("draft."):
            return None
        if label == "develop":
            branch = "develop"
        elif label.startswith("release."):
            branch = label.removeprefix("release.") + "-release"
        else:
            return None
        digits = str(version.dev)
        if (
            version.pre
            or version.post is not None
            or not re.fullmatch(r"[0-9]{11,}", digits)
        ):
            raise ValueError(f"invalid branch Nightly version: {version}")
        result = cls(branch, digits[:8], int(digits[8:]))
        if result.version(version.base_version) != str(version):
            raise ValueError(f"invalid branch Nightly version: {version}")
        return result


if __name__ == "__main__":
    # Both the scheduler and retention pass a complete, successfully fetched branch list.
    latest = latest_release_branch(line.strip() for line in sys.stdin)
    if latest:
        print(latest)
