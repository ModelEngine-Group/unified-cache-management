import glob
import json
import os
from typing import Any, Dict, List, Optional

import evalscope
from common.capture_utils import export_vars


class EvalScopeRunner:
    """
    Encapsulate the logic for running evalscope tasks and collecting results.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def run(self, task_cfg: evalscope.config.TaskConfig) -> None:
        evalscope.run_task(task_cfg=task_cfg)

    @staticmethod
    def _get_latest_run_dir(output_dir: str) -> Optional[str]:
        if not os.path.exists(output_dir):
            return None
        subdirs = [
            d
            for d in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, d))
        ]
        if not subdirs:
            return None
        subdirs.sort(
            reverse=True
        )  # The timestamp directory can be sorted in descending order by string
        return os.path.join(output_dir, subdirs[0])

    @staticmethod
    def _collect_report_json_files(run_dir: str) -> List[str]:
        reports_root = os.path.join(run_dir, "reports")
        if not os.path.exists(reports_root):
            return []

        json_files = []
        for model_dir in os.listdir(reports_root):
            model_path = os.path.join(reports_root, model_dir)
            if os.path.isdir(model_path):
                json_files.extend(glob.glob(os.path.join(model_path, "*.json")))
        return json_files

    @staticmethod
    def _parse_metrics_from_json(json_path: str) -> Dict[str, Any]:
        """Parse a single JSON report file and return a structured metrics dictionary"""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        dataset_name = data.get(
            "dataset_name", os.path.splitext(os.path.basename(json_path))[0]
        )
        model_name = data.get("model_name", "")

        metrics = []
        for m in data.get("metrics", []):
            categories = [
                {
                    "name": c.get("name"),
                    "score": c.get("score", 0.0),
                    "macro_score": c.get("macro_score", 0.0),
                    "num": c.get("num", 0),
                    "subsets": c.get("subsets", []),
                }
                for c in m.get("categories", [])
            ]
            metrics.append(
                {
                    "name": m.get("name"),
                    "score": m.get("score", 0.0),
                    "macro_score": m.get("macro_score", 0.0),
                    "num": m.get("num", 0),
                    "categories": categories,
                }
            )

        return {
            "dataset_name": dataset_name,
            "model_name": model_name,
            "pretty_name": data.get("dataset_pretty_name", dataset_name),
            "score": data.get("score", 0.0),
            "metrics": metrics,
            "analysis": data.get("analysis", "N/A"),
        }

    @export_vars
    def collect_results(self) -> Dict[str, Any]:
        latest_run = self._get_latest_run_dir(self.output_dir)
        if not latest_run:
            return {"_name": "eval_scope", "_proj": {}}

        json_files = self._collect_report_json_files(latest_run)
        if not json_files:
            return {"_name": "eval_scope", "_proj": {}}

        all_metrics = {}
        extracted_model_name = ""

        for json_path in json_files:
            try:
                parsed = self._parse_metrics_from_json(json_path)
            except (json.JSONDecodeError, KeyError):
                continue

            if not extracted_model_name:
                extracted_model_name = parsed["model_name"]

            dataset_name = parsed["dataset_name"]
            all_metrics[dataset_name] = {
                "pretty_name": parsed["pretty_name"],
                "model": parsed["model_name"],
                "score": parsed["score"],
                "metrics": parsed["metrics"],
                "analysis": parsed["analysis"],
            }
            # The total score is presented in a flat format, facilitating quick access by external parties
            all_metrics[f"{dataset_name}.score"] = parsed["score"]

        if extracted_model_name:
            all_metrics["model_name"] = extracted_model_name

        return {"_name": "eval_scope", "_proj": all_metrics}
