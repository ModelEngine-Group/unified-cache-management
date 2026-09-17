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
    schema_version: int = 2
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
            if code == "CHECKER_RUNTIME_POOL_PROBE_REQUIRED"
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

        model_keys = (
            "architectures",
            "model_type",
            "attention_arch",
            "dtype",
            "quantization",
            "supported_platforms",
        )
        requirement_keys = (
            "pool_names",
            "storage_api",
            "host_layout",
            "probe_fidelity",
        )
        capability_keys = ("api_versions", "detection")
        environment_keys = ("sglang_version", "platform", "requested_platform")

        data["model_info"] = {
            key: self.model_info[key]
            for key in model_keys
            if key in self.model_info and self.model_info[key] is not None
        }
        data["cache_requirements"] = {
            key: self.cache_requirements[key]
            for key in requirement_keys
            if key in self.cache_requirements
        }
        data["ucm_capabilities"] = {
            key: self.ucm_capabilities[key]
            for key in capability_keys
            if key in self.ucm_capabilities
        }
        data["environment"] = {
            key: self.environment[key]
            for key in environment_keys
            if key in self.environment and self.environment[key] is not None
        }
        if "\n" in data["reason"]:
            data["reason"] = data["reason"].splitlines()[0]
        data["detail_level"] = "summary"
        for key in ("model_info", "cache_requirements", "ucm_capabilities", "environment", "roundtrip"):
            if not data[key]:
                data.pop(key)
        return data


def emit_result(
    result: CheckResult, output: str | None = None, verbose: bool = False
) -> None:
    text = json.dumps(
        result.to_dict(verbose=verbose), ensure_ascii=False, indent=2, sort_keys=True
    )
    print(text)
    if output:
        Path(output).write_text(text + "\n", encoding="utf-8")
