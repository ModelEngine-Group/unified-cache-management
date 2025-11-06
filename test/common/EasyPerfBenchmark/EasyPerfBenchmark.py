import datetime
import json
from pathlib import Path

from common.db_utils import write_to_db


class EasyPerfBenchmark:
    """EasyPerf Benchmark Demo"""

    def __init__(self, config: dict):
        self.api_cfg = config["api"]
        self.model_cfg = config["model"]
        self.save_db = config.get("save_to_db", False)
        self.save_file_cfg = config.get("save_to_file", {})
        self.experiments = config.get("experiments", [])

        self.output_path = None
        if self.save_file_cfg.get("enabled", False):
            filename = self.save_file_cfg.get("filename", "easyperf.jsonl")
            self.output_path = Path(filename).resolve()
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def _run_experiment(self, exp_cfg: dict):
        result = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input_length": exp_cfg["input_length"],
            "output_length": exp_cfg["output_length"],
            "concurrency": exp_cfg["single_concurrency"],
            "rounds": exp_cfg["request_rounds"],
            "ttft": 0,
            "tpot": 0,
            "avg_tps": 0,
        }
        if self.output_path:
            with self.output_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        write_to_db("easyperf_benchmark", result)
        return result

    def run_all(self):
        """Synchronous entry point."""
        results = []
        for exp_cfg in self.experiments:
            results.append(self._run_experiment(exp_cfg))
        return results
