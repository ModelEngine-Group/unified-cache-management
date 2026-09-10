"""Machine-readable result types for SGLang model checks."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CheckResult:
    model: str
    mode: str
    supported: bool = False
    code: str = "INTERNAL_ERROR"
    stage: str = "startup"
    reason: str = ""
    model_info: dict[str, Any] = field(default_factory=dict)
    cache_requirements: dict[str, Any] = field(default_factory=dict)
    ucm_capabilities: dict[str, Any] = field(default_factory=dict)
    roundtrip: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 2
    framework: str = "sglang"

    def succeed(self, stage: str, reason: str = "") -> None:
        self.supported = True
        self.code = "PASS"
        self.stage = stage
        self.reason = reason

    def fail(self, code: str, stage: str, reason: str) -> None:
        self.supported = False
        self.code = code
        self.stage = stage
        self.reason = reason

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def emit_result(result: CheckResult, output: str | None = None) -> None:
    text = json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if output:
        Path(output).write_text(text + "\n", encoding="utf-8")
