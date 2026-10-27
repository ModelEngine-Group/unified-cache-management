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
    status: str = "inconclusive"
    code: str = "INTERNAL_ERROR"
    stage: str = "startup"
    reason: str = ""
    model_info: dict[str, Any] = field(default_factory=dict)
    cache_requirements: dict[str, Any] = field(default_factory=dict)
    ucm_capabilities: dict[str, Any] = field(default_factory=dict)
    roundtrip: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 3
    framework: str = "sglang"

    def succeed(self, stage: str, reason: str = "") -> None:
        self.supported = True
        self.status = "compatible"
        self.code = "PASS"
        self.stage = stage
        self.reason = reason

    def fail(self, code: str, stage: str, reason: str) -> None:
        self.supported = False
        self.status = (
            "inconclusive"
            if code in {
                "CHECKER_RUNTIME_POOL_PROBE_REQUIRED",
                "CHECKER_STORAGE_PROBE_REQUIRED",
                "CHECKER_HOST_POOL_API_UNSUPPORTED",
                "CHECKER_MEMORY_LIMIT",
                "CHECKER_CONFIG_ONLY",
                "META_MODEL_INIT_UNSUPPORTED",
                "ENVIRONMENT_UNSUPPORTED",
                "CACHE_REQUIREMENTS_UNDETERMINED",
                "HICACHE_HOST_POOL_UNSUPPORTED",
            }
            else "incompatible"
        )
        self.code = code
        self.stage = stage
        self.reason = reason

    def to_dict(self, verbose: bool = False) -> dict[str, Any]:
        data = asdict(self)
        if verbose:
            data["detail_level"] = "verbose"
            return data

        summary = {
            "model": self.model,
            "mode": self.mode,
            "status": self.status,
            "summary": (
                ("Model structure and Host/UCM storage roundtrip passed; "
                 if self.validation.get("meta_model") == "passed"
                 else "Host/UCM storage roundtrip passed (model structure not validated); ") +
                "inference, device transfers and multi-rank execution are not validated."
                if self.supported and self.mode == "roundtrip"
                else "Model structural checks passed; storage IO and inference are not validated."
                if self.supported
                else self.reason.splitlines()[0] if self.reason else "Check did not pass."
            ),
        }
        if not self.supported:
            summary["code"] = self.code
            suggestions = {
                "CHECKER_MEMORY_LIMIT": "Increase --max-host-mib and retry; use --verbose for details.",
                "CHECKER_STORAGE_PROBE_REQUIRED": "Run --mode roundtrip with a dedicated --storage-backends directory.",
                "CHECKER_CONFIG_ONLY": "Retry without --skip-meta-model to validate model construction.",
            }
            summary["suggestion"] = suggestions.get(
                self.code, "Use --verbose to inspect the failed stage and diagnostics."
            )
        return summary


def emit_result(
    result: CheckResult, output: str | None = None, verbose: bool = False
) -> None:
    text = json.dumps(
        result.to_dict(verbose=verbose), ensure_ascii=False, indent=2, sort_keys=True
    )
    print(text)
    if output:
        Path(output).write_text(text + "\n", encoding="utf-8")
