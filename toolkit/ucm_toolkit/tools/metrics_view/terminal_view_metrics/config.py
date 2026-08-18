from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path
from typing import Iterator

from .promql import metric_names_in_expr

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
DEFAULT_HISTOGRAM_QUANTILES = [0.5, 0.9, 0.99]


def load_config(path: str | Path) -> dict:
    config_path = resolve_config_path(path)
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "YAML config requires PyYAML; use JSON otherwise"
            ) from exc
        config = yaml.safe_load(text)
    else:
        config = json.loads(text)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be an object: {config_path}")
    return config


def apply_config_param_overrides(config: dict, items: list[str]) -> dict:
    if not items:
        return config
    params = config.get("params", {})
    if not isinstance(params, dict):
        raise ValueError("Config params must be an object")
    updated_params = dict(params)
    for item in items:
        if "=" not in item:
            raise ValueError(f"--config-param expects KEY=VALUE, got: {item}")
        key, value = (part.strip() for part in item.split("=", 1))
        if not key or not value:
            raise ValueError(f"--config-param expects KEY=VALUE, got: {item}")
        if key not in updated_params:
            raise ValueError(f"Config parameter not found: {key}")
        updated_params[key] = value
    updated_config = dict(config)
    updated_config["params"] = updated_params
    return updated_config


def resolve_config_path(path_or_name: str | Path) -> Path:
    path = Path(path_or_name)
    if path.exists():
        return path
    candidates = [
        CONFIG_DIR / str(path_or_name),
        CONFIG_DIR / f"{path_or_name}.json",
        CONFIG_DIR / f"{path_or_name}.yaml",
        CONFIG_DIR / f"{path_or_name}.yml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Config not found: {path_or_name}")


def metric_specs(config: dict) -> list[dict]:
    prefix = str(config.get("metric_prefix", ""))
    params = _config_params(config)
    specs: list[dict] = []
    groups = config.get("groups")
    if isinstance(groups, list):
        for group in groups:
            for metric in group.get("metrics", []):
                specs.append(_normalize_metric(metric, prefix, params))
    for metric in config.get("metrics", []):
        specs.append(_normalize_metric(metric, prefix, params))
    return specs


def metric_names_for_scrape(config: dict) -> set[str]:
    names: set[str] = set()
    for spec in metric_specs(config):
        if "expr" in spec:
            names.update(metric_names_in_expr(spec["expr"]))
            continue
        name = spec.get("source", spec["name"])
        metric_type = spec.get("type", "gauge")
        if metric_type == "histogram":
            names.update({f"{name}_bucket", f"{name}_sum", f"{name}_count"})
        else:
            names.add(name)
    return names


def list_preset_configs() -> list[Path]:
    if not CONFIG_DIR.exists():
        return []
    return sorted(CONFIG_DIR.glob("*.json")) + sorted(CONFIG_DIR.glob("*.yaml"))


def parse_duration_seconds(value: str | int | float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = value.strip()
    match = re.fullmatch(r"(\d+(?:\.\d+)?)(ms|s|m|h|d)?", text)
    if not match:
        raise ValueError(f"Invalid duration: {value}")
    amount = float(match.group(1))
    unit = match.group(2) or "s"
    factors = {"ms": 0.001, "s": 1, "m": 60, "h": 3600, "d": 86400}
    return amount * factors[unit]


def parse_time_ms(value: str | int | float) -> int:
    if isinstance(value, (int, float)):
        return _numeric_time_ms(float(value))
    text = value.strip()
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return _numeric_time_ms(float(text))
    parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    return int(parsed.timestamp() * 1000)


def iter_grouped_specs(config: dict) -> Iterator[tuple[str, dict]]:
    prefix = str(config.get("metric_prefix", ""))
    params = _config_params(config)
    for group in config.get("groups", []):
        group_name = group.get("name", "")
        for metric in group.get("metrics", []):
            yield group_name, _normalize_metric(metric, prefix, params)
    for metric in config.get("metrics", []):
        yield "", _normalize_metric(metric, prefix, params)


def _config_params(config: dict) -> dict[str, str]:
    params = config.get("params", {})
    if params is None:
        return {}
    if not isinstance(params, dict):
        raise ValueError("Config params must be an object")
    return {str(key): str(value) for key, value in params.items()}


def _normalize_metric(metric: dict, prefix: str, params: dict[str, str]) -> dict:
    spec = dict(metric)
    name = spec["name"]
    if prefix and ":" not in name and not name.startswith(prefix):
        name = f"{prefix}{name}"
    spec["name"] = name
    if "expr" in spec:
        spec["expr"] = _expand_params(str(spec["expr"]), params)
    aggregate = spec.get("aggregate")
    if aggregate not in {"sum", "avg", "max"}:
        raise ValueError(
            f"Metric {name} requires aggregate to be 'sum', 'avg', or 'max'"
        )
    if spec.get("type") == "histogram":
        spec["avg"] = True
        spec["quantiles"] = list(DEFAULT_HISTOGRAM_QUANTILES)
    return spec


def _expand_params(expr: str, params: dict[str, str]) -> str:
    return re.sub(
        r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}",
        lambda match: params.get(match.group(1), match.group(0)),
        expr,
    )


def _numeric_time_ms(value: float) -> int:
    return int(value if value >= 10_000_000_000 else value * 1000)
