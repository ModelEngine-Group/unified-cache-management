"""Pure compatibility rules, intentionally importable without SGLang/Torch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class ModelRequirements:
    attention_arch: str
    pool_names: tuple[str, ...]
    storage_api: str
    host_layout: str
    is_mla: bool
    is_hybrid: bool
    probe_fidelity: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "attention_arch": self.attention_arch,
            "pool_names": list(self.pool_names),
            "storage_api": self.storage_api,
            "host_layout": self.host_layout,
            "is_mla": self.is_mla,
            "is_hybrid": self.is_hybrid,
            "probe_fidelity": self.probe_fidelity,
        }


def requirements_from_facts(
    facts: dict[str, Any], host_layout: str = "page_first"
) -> ModelRequirements:
    """Convert facts produced by SGLang ModelConfig into storage requirements."""
    attention_arch = str(facts.get("attention_arch", "UNKNOWN")).upper()
    pools: list[str] = ["kv"]

    if facts.get("is_deepseek_v4"):
        # Reuse names exported by the installed SGLang instead of maintaining
        # a second DeepSeek-V4 list in this tool. This picks up additions such
        # as the FP4 indexer scale pool.
        pools = sorted(
            name
            for name in _strings(facts.get("sglang_pool_names"))
            if name.startswith("deepseek_v4_")
        )
        if not pools:
            pools = ["deepseek_v4_runtime_pools_unknown"]
    else:
        if facts.get("is_minimax_sparse") or facts.get("is_dsa"):
            pools.append("indexer")
        if facts.get("has_linear_attention"):
            pools.append("mamba")
        if facts.get("is_hybrid_swa"):
            pools.append("swa")

    pools = list(dict.fromkeys(pools))
    is_hybrid = len(pools) > 1 or pools[0] != "kv"
    return ModelRequirements(
        attention_arch=attention_arch,
        pool_names=tuple(pools),
        storage_api="v2" if is_hybrid else "v1",
        host_layout=host_layout,
        is_mla=attention_arch == "MLA",
        is_hybrid=is_hybrid,
        probe_fidelity="runtime_pool_required" if is_hybrid else "real_host_pool",
    )


def normalize_capabilities(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": int(raw.get("schema_version", 1)),
        "api_versions": sorted(_strings(raw.get("api_versions"))),
        "detection": str(raw.get("detection", "method_override")),
    }


def _strings(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value}
    if isinstance(value, Iterable):
        return {str(item) for item in value}
    return {str(value)}


def find_missing_capabilities(
    requirements: ModelRequirements, capabilities: dict[str, Any]
) -> list[str]:
    caps = normalize_capabilities(capabilities)
    missing: list[str] = []
    # SGLang's hybrid path stores the anchor pool through v1 and side pools
    # through v2, so a v2 model requires both contracts.
    required_apis = ("v1", "v2") if requirements.storage_api == "v2" else ("v1",)
    for api in required_apis:
        if api not in caps["api_versions"]:
            missing.append(f"storage API {api}")
    return missing


def infer_capabilities(storage_cls: type[Any], base_cls: type[Any]) -> dict[str, Any]:
    """Inspect the production class without requiring a capability declaration."""

    def overridden(name: str) -> bool:
        return getattr(storage_cls, name, None) is not getattr(base_cls, name, None)

    versions = []
    if overridden("batch_get_v1") and overridden("batch_set_v1"):
        versions.append("v1")
    if all(
        overridden(name)
        for name in ("batch_exists_v2", "batch_get_v2", "batch_set_v2")
    ):
        versions.append("v2")
    return normalize_capabilities(
        {
            "api_versions": versions,
            "detection": "method_override",
        }
    )
