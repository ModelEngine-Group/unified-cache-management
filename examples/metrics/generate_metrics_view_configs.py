from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "toolkit/ucm_toolkit/tools/metrics_view/configs"
DASHBOARDS = {
    "vllm": "grafana_vllm.json",
    "connector": "grafana_connector.json",
    "store": "grafana_store.json",
}


def panels(items: list[dict]):
    for item in items:
        if item.get("type") == "row":
            yield from panels(item.get("panels", []))
        else:
            yield item


def clean_legend(legend: str) -> str:
    if legend == "__auto":
        return ""
    return re.sub(r"\s*\{\{[^}]+\}\}", "", legend).strip()


def result_labels(expr: str) -> list[str]:
    labels: list[str] = []
    normalized = expr.replace("${perWorker:raw}", "model_name")
    for match in re.finditer(r"\b(?:sum|avg|max|count)\s+by\s*\(([^)]*)\)", normalized):
        for label in match.group(1).split(","):
            label = label.strip()
            if label and label not in labels:
                labels.append(label)
    for label in re.findall(
        r',\s*"([A-Za-z_][A-Za-z0-9_]*)"\s*,\s*"[^"]*"\s*,'
        r'\s*"[^"]+"\s*,\s*"[^"]*"\s*\)',
        normalized,
    ):
        if label not in labels:
            labels.append(label)
    if "histogram_quantile(" in normalized and "le" in labels:
        labels.remove("le")
    return [label for label in labels if label != "__name__"]


def metric_name(
    title: str, targets: list[dict], target: dict, labels: list[str]
) -> str:
    legend = target.get("legendFormat", "")
    if len(targets) > 1:
        return f"{title}: {clean_legend(legend) or target['refId']}"
    placeholders = re.findall(r"\{\{([^}]+)\}\}", legend)
    dynamic = [label for label in placeholders if label in labels]
    if legend == "__auto":
        dynamic = [label for label in labels if label != "model_name"]
    if dynamic:
        suffix = " ".join(f"{{{label}}}" for label in dynamic)
        return f"{title}: {suffix}"
    return title


def build_config(name: str, dashboard_file: str) -> dict:
    dashboard = json.loads(
        (Path(__file__).parent / dashboard_file).read_text(encoding="utf-8")
    )
    metrics = []
    for panel in panels(dashboard.get("panels", [])):
        targets = [
            target
            for target in panel.get("targets", [])
            if target.get("expr") and not target.get("hide", False)
        ]
        for target in targets:
            expr = target["expr"].replace("${perWorker:raw}", "model_name")
            labels = result_labels(expr)
            metric = {
                "name": metric_name(panel["title"], targets, target, labels),
                "type": "promql",
                "expr": expr,
                "value": "value",
                "aggregate": "sum",
            }
            if labels:
                metric["group_by"] = labels
            unit = panel.get("fieldConfig", {}).get("defaults", {}).get("unit")
            if unit:
                metric["unit"] = unit
            metrics.append(metric)
    return {"title": f"{name.title()} Dashboard", "metrics": metrics}


def main() -> None:
    for name, dashboard_file in DASHBOARDS.items():
        config = build_config(name, dashboard_file)
        path = CONFIG_DIR / f"{name}.json"
        path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
