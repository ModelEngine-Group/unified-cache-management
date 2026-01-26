import hashlib
import json
import os
from typing import Any, Dict, List, Optional

import pytest
from common.capture_utils import export_vars
from common.config_utils import config_utils as config_instance
from common.db_utils import read_from_db
from common.llmperf.run_inference import inference_results
from common.uc_eval.task import DocQaEvalTask
from common.uc_eval.utils.data_class import EvalConfig, ModelConfig

# Global test configuration constants
DATA_FILE_PATH = "common/uc_eval/utils/multifieldqa_zh.jsonl"
MEAN_INPUT_TOKENS = 8000
MEAN_OUTPUT_TOKENS = 200
MAX_NUM_COMPLETED_REQUESTS = 8
CONCURRENT_REQUESTS = 8
ADDITIONAL_SAMPLING_PARAMS = "{}"  # JSON string for sampling parameters
SERVER_KEY = os.getenv("EX_INFO") or config_instance.get_nested_config(
    "llm_connection.ex_info"
)


def _make_config_key() -> str:
    """Generate a unique config key based on inference parameters for result caching."""
    key_str = (
        f"{SERVER_KEY}|{MEAN_INPUT_TOKENS}|{MEAN_OUTPUT_TOKENS}|"
        f"{MAX_NUM_COMPLETED_REQUESTS}|{CONCURRENT_REQUESTS}|{ADDITIONAL_SAMPLING_PARAMS}"
    )
    return hashlib.md5(key_str.encode()).hexdigest()


class TestModelValidator:
    """Pytest-compatible test suite for model validation (performance + accuracy)."""

    @pytest.mark.feature("model_validate_test_naive")
    @pytest.mark.stage(2)
    @export_vars
    def test_model_validate_naive(self, model_config: ModelConfig) -> Dict[str, Any]:
        perf_metrics = self._run_perf_test(hit_rates=[0])
        accuracy = self._run_accuracy_test(model_config, "Naive")["metric_dict"][
            "f1-score"
        ]

        self._print_perf_summary(perf_metrics)
        self._print_accuracy_comparison(None, accuracy, "Naive")

        result = {
            "accuracy_naive": accuracy,
            **perf_metrics[0],
            "config_key": _make_config_key(),
        }
        if SERVER_KEY and not self._fetch_naive_result():
            return {"_name": "model_validate_test", "_data": result}

    @pytest.mark.feature("model_validate_test_pc")
    @pytest.mark.stage(2)
    def test_model_validate_pc(self, model_config: ModelConfig) -> None:
        """Validate performance under various cache hit rates (e.g., 0, 20, ..., 100)."""
        hit_rates = [0, 20, 50, 80, 100]
        perf_metrics = self._run_perf_test(hit_rates=hit_rates)

        self._run_accuracy_test(model_config, "PC")
        accuracy = self._run_accuracy_test(model_config, "PC")["metric_dict"][
            "f1-score"
        ]

        # Fetch baseline (naive) results for comparison
        naive_dict = self._fetch_naive_result()
        if naive_dict:
            perf_metrics.insert(
                0,
                {
                    "hit_rate": "naive",
                    "ttft_mean": float(naive_dict["ttft_mean"]),
                    "tpot_mean": float(naive_dict["tpot_mean"]),
                    "e2e_mean": float(naive_dict["e2e_mean"]),
                },
            )

        self._print_perf_summary(perf_metrics)
        naive_acc = float(naive_dict["accuracy_naive"]) if naive_dict else None
        self._print_accuracy_comparison(naive_acc, accuracy, "PC")

    @pytest.mark.feature("model_validate_test_sparse")
    @pytest.mark.stage(2)
    def test_model_validate_sparse(self, model_config: ModelConfig) -> None:
        """Validate sparse retrieval scenario (treated as hit_rate=0 like naive)."""
        perf_metrics = self._run_perf_test(hit_rates=[0])
        accuracy = self._run_accuracy_test(model_config, "Sparse")["metric_dict"][
            "f1-score"
        ]

        naive_dict = self._fetch_naive_result()
        if naive_dict:
            perf_metrics.insert(
                0,
                {
                    "hit_rate": "naive",
                    "ttft_mean": float(naive_dict["ttft_mean"]),
                    "tpot_mean": float(naive_dict["tpot_mean"]),
                    "e2e_mean": float(naive_dict["e2e_mean"]),
                },
            )

        self._print_perf_summary(perf_metrics)
        naive_acc = float(naive_dict["accuracy_naive"]) if naive_dict else None
        self._print_accuracy_comparison(naive_acc, accuracy, "Sparse")

    def _run_perf_test(self, hit_rates: List[int]) -> List[Dict[str, Any]]:
        """Run inference under specified cache hit rates and extract performance metrics."""
        n = len(hit_rates)
        all_summaries = inference_results(
            mean_input_tokens=[MEAN_INPUT_TOKENS] * n,
            mean_output_tokens=[MEAN_OUTPUT_TOKENS] * n,
            max_num_completed_requests=[MAX_NUM_COMPLETED_REQUESTS] * n,
            concurrent_requests=[CONCURRENT_REQUESTS] * n,
            additional_sampling_params=[ADDITIONAL_SAMPLING_PARAMS] * n,
            hit_rate=hit_rates,
        )
        return self._extract_perf_metrics(all_summaries, hit_rates)

    def _fetch_naive_result(self) -> Optional[Dict[str, Any]]:
        records = read_from_db(
            "model_validate_test", {"config_key": _make_config_key()}, limit=1
        )
        return records[0] if records else None

    def _extract_perf_metrics(
        self, summaries: List[Dict], hit_rates: List[int]
    ) -> List[Dict[str, Any]]:
        results = []
        for summary, hr in zip(summaries, hit_rates):
            res = summary["results"]
            results.append(
                {
                    "hit_rate": hr,
                    "ttft_mean": res["ttft_s"]["quantiles"]["p50"],
                    "tpot_mean": res["inter_token_latency_s"]["quantiles"]["p50"],
                    "e2e_mean": res["end_to_end_latency_s"]["quantiles"]["p50"],
                }
            )
        return results

    def _run_accuracy_test(
        self, model_config: ModelConfig, test_id: str
    ) -> Dict[str, Any]:
        eval_config = EvalConfig(
            data_type="doc_qa",
            dataset_file_path=DATA_FILE_PATH,
            parallel_num=CONCURRENT_REQUESTS,
            benchmark_mode="evaluate",
            metrics=["f1-score"],
            eval_class="common.uc_eval.utils.metric:Includes",
        )
        file_save_path = config_instance.get_config("reports").get("base_dir")
        task = DocQaEvalTask(model_config, eval_config, file_save_path)
        result = task.run()
        print(f"\n[Accuracy Test] {test_id}")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result

    def _print_perf_summary(self, results: List[Dict[str, Any]]) -> None:
        if not results:
            return
        HIGHLIGHT = "\033[1;96m"
        RESET = "\033[0m"
        print(f"\n{HIGHLIGHT}{'=' * 110}{RESET}")
        print(
            f"{HIGHLIGHT}{'Hit Rate (%)':<12} {'Input Tokens':<15} {'Output Tokens':<15} {'Concurrency':<12} {'ttft_mean [s]':<20} {'tpot_mean [s]':<20} {'e2e_mean [s]':<20}{RESET}"
        )
        print(f"{HIGHLIGHT}{'-' * 110}{RESET}")
        for r in results:
            print(
                f"{HIGHLIGHT}{r['hit_rate']!s:<12} {MEAN_INPUT_TOKENS:<15} {MEAN_OUTPUT_TOKENS:<15} {CONCURRENT_REQUESTS:<12} {r['ttft_mean']:<20.4f} {r['tpot_mean']:<20.4f} {r['e2e_mean']:<20.4f}{RESET}"
            )
        print(f"{HIGHLIGHT}{'=' * 110}{RESET}")

    def _print_accuracy_comparison(
        self, naive_val: Optional[float], accuracy_val: float, test_id: str
    ) -> None:
        HIGHLIGHT = "\033[1;96m"
        RESET = "\033[0m"
        print(f"\n{HIGHLIGHT}{'=' * 40}{RESET}")
        print(f"{HIGHLIGHT}{'Test':<15} {'f1-score'}{RESET}")
        print(f"{HIGHLIGHT}{'-' * 40}{RESET}")
        if naive_val is not None:
            print(f"{HIGHLIGHT}{'naive':<15} {naive_val:<12.4f}{RESET}")
        print(f"{HIGHLIGHT}{test_id:<15} {accuracy_val:<12.4f}{RESET}")
        print(f"{HIGHLIGHT}{'=' * 40}{RESET}")
