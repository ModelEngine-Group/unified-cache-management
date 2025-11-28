import dataclasses

import pytest
from common.capture_utils import export_vars
from common.config_utils import config_utils as config_instance
from common.llmperf.run_inference import inference_results
from common.uc_eval.task import (
    DocQaPerfTask,
    MultiTurnDialogPerfTask,
    SyntheticPerfTask,
)
from common.uc_eval.utils.data_class import ModelConfig, PerfConfig


@pytest.mark.parametrize("mean_input_tokens", [[2000, 3000]])
@pytest.mark.parametrize("mean_output_tokens", [[200, 500]])
@pytest.mark.parametrize("max_num_completed_requests", [[8, 4]])
@pytest.mark.parametrize("concurrent_requests", [[8, 4]])
@pytest.mark.parametrize("additional_sampling_params", [["{}", "{}"]])
@pytest.mark.parametrize("hit_rate", [[0, 50]])
@pytest.mark.feature("uc_performance_test")
@export_vars
def test_performance(
    mean_input_tokens,
    mean_output_tokens,
    max_num_completed_requests,
    concurrent_requests,
    additional_sampling_params,
    hit_rate,
):
    all_summaries = inference_results(
        mean_input_tokens,
        mean_output_tokens,
        max_num_completed_requests,
        concurrent_requests,
        additional_sampling_params,
        hit_rate,
    )
    failed_cases = []

    value_lists = {
        "mean_input_tokens": [],
        "mean_output_tokens": [],
        "results_inter_token_latency_s_quantiles_p50": [],
        "results_inter_token_latency_s_quantiles_p90": [],
        "results_inter_token_latency_s_quantiles_p99": [],
        "results_inter_token_latency_s_mean": [],
        "results_ttft_s_quantiles_p50": [],
        "results_ttft_s_quantiles_p90": [],
        "results_ttft_s_quantiles_p99": [],
        "results_ttft_s_mean": [],
        "results_end_to_end_latency_s_quantiles_p50": [],
        "results_end_to_end_latency_s_quantiles_p90": [],
        "results_end_to_end_latency_s_quantiles_p99": [],
        "results_end_to_end_latency_s_mean": [],
        "num_completed_requests": [],
        "elapsed_time": [],
        "incremental_time_delay": [],
        "total_throughput": [],
        "incremental_throughput": [],
    }

    for i, summary in enumerate(all_summaries):
        mean_input_tokens = summary["mean_input_tokens"]
        mean_output_tokens = summary["mean_output_tokens"]

        results_inter_token_latency_s_quantiles_p50 = summary["results"][
            "inter_token_latency_s"
        ]["quantiles"]["p50"]
        results_inter_token_latency_s_quantiles_p90 = summary["results"][
            "inter_token_latency_s"
        ]["quantiles"]["p90"]
        results_inter_token_latency_s_quantiles_p99 = summary["results"][
            "inter_token_latency_s"
        ]["quantiles"]["p99"]
        results_inter_token_latency_s_mean = summary["results"][
            "inter_token_latency_s"
        ]["mean"]

        results_ttft_s_quantiles_p50 = summary["results"]["ttft_s"]["quantiles"]["p50"]
        results_ttft_s_quantiles_p90 = summary["results"]["ttft_s"]["quantiles"]["p90"]
        results_ttft_s_quantiles_p99 = summary["results"]["ttft_s"]["quantiles"]["p99"]
        results_ttft_s_mean = summary["results"]["ttft_s"]["mean"]

        results_end_to_end_latency_s_quantiles_p50 = summary["results"][
            "end_to_end_latency_s"
        ]["quantiles"]["p50"]
        results_end_to_end_latency_s_quantiles_p90 = summary["results"][
            "end_to_end_latency_s"
        ]["quantiles"]["p90"]
        results_end_to_end_latency_s_quantiles_p99 = summary["results"][
            "end_to_end_latency_s"
        ]["quantiles"]["p99"]
        results_end_to_end_latency_s_mean = summary["results"]["end_to_end_latency_s"][
            "mean"
        ]

        num_completed_requests = summary["num_completed_requests"]
        elapsed_time = summary["elapsed_time"]
        incremental_time_delay = summary["incremental_time_delay"]
        total_throughput = summary["total_throughput"]
        incremental_throughput = summary["incremental_throughput"]

        values = [
            mean_input_tokens,
            mean_output_tokens,
            results_inter_token_latency_s_quantiles_p50,
            results_inter_token_latency_s_quantiles_p90,
            results_inter_token_latency_s_quantiles_p99,
            results_inter_token_latency_s_mean,
            results_ttft_s_quantiles_p50,
            results_ttft_s_quantiles_p90,
            results_ttft_s_quantiles_p99,
            results_ttft_s_mean,
            results_end_to_end_latency_s_quantiles_p50,
            results_end_to_end_latency_s_quantiles_p90,
            results_end_to_end_latency_s_quantiles_p99,
            results_end_to_end_latency_s_mean,
            num_completed_requests,
            elapsed_time,
            incremental_time_delay,
            total_throughput,
            incremental_throughput,
        ]

        for var_name, val in zip(
            [
                "mean_input_tokens",
                "mean_output_tokens",
                "results_inter_token_latency_s_quantiles_p50",
                "results_inter_token_latency_s_quantiles_p90",
                "results_inter_token_latency_s_quantiles_p99",
                "results_inter_token_latency_s_mean",
                "results_ttft_s_quantiles_p50",
                "results_ttft_s_quantiles_p90",
                "results_ttft_s_quantiles_p99",
                "results_ttft_s_mean",
                "results_end_to_end_latency_s_quantiles_p50",
                "results_end_to_end_latency_s_quantiles_p90",
                "results_end_to_end_latency_s_quantiles_p99",
                "results_end_to_end_latency_s_mean",
                "num_completed_requests",
                "elapsed_time",
                "incremental_time_delay",
                "total_throughput",
                "incremental_throughput",
            ],
            values,
        ):
            value_lists[var_name].append(val)
            if val is None:
                failed_cases.append((i, var_name, "missing"))

            try:
                assert val > 0, f"value <= 0"
            except AssertionError as e:
                failed_cases.append((i, var_name, str(e)))

    # Output final result
    if failed_cases:
        print(f"\n[WARNING] Assertion failed: {len(failed_cases)} abnormal cases found")
        for i, key, reason in failed_cases:
            print(f"   Iteration={i + 1}, key='{key}' -> {reason}")
    else:
        print("\n[INFO] All values are greater than 0. Assertion passed!")

    return {"_name": "llmperf", "_data": value_lists}


@pytest.fixture(scope="session")
def model_config() -> ModelConfig:
    cfg = config_instance.get_config("models") or {}
    field_name = [field.name for field in dataclasses.fields(ModelConfig)]
    kwargs = {k: v for k, v in cfg.items() if k in field_name and v is not None}
    return ModelConfig(**kwargs)


sync_perf_cases = [
    pytest.param(
        PerfConfig(
            data_type="synthetic",
            enable_prefix_cache=False,
            parallel_num=[1, 4, 8],
            prompt_tokens=[4000, 8000],
            output_tokens=[1000, 1000],
            benchmark_mode="default-perf",
        ),
        id="benchmark-complete-recalculate-default-perf",
    ),
    pytest.param(
        PerfConfig(
            data_type="synthetic",
            enable_prefix_cache=True,
            parallel_num=[1, 4, 8],
            prompt_tokens=[4000, 8000],
            output_tokens=[1000, 1000],
            prefix_cache_num=[0.8, 0.8],
            benchmark_mode="stable-perf",
        ),
        id="benchmark-prefix-cache-stable-perf",
    ),
]


@pytest.mark.feature("perf_test")
@pytest.mark.parametrize("perf_config", sync_perf_cases)
@export_vars
def test_sync_perf(
    perf_config: PerfConfig, model_config: ModelConfig, request: pytest.FixtureRequest
):
    file_save_path = config_instance.get_config("reports").get("base_dir")
    task = SyntheticPerfTask(model_config, perf_config, file_save_path)
    result = task.run()
    return {"_name": request.node.callspec.id, "_data": result}


multiturn_dialogue_perf_cases = [
    pytest.param(
        PerfConfig(
            data_type="multi_turn_dialogue",
            dataset_file_path="common/uc_eval/datasets/multi_turn_dialogues/multiturndialog.json",
            enable_prefix_cache=False,
            parallel_num=1,
            benchmark_mode="default-perf",
        ),
        id="multiturn-dialogue-complete-recalculate-default-perf",
    )
]


@pytest.mark.feature("perf_test")
@pytest.mark.parametrize("perf_config", multiturn_dialogue_perf_cases)
@export_vars
def test_multiturn_dialogue_perf(
    perf_config: PerfConfig, model_config: ModelConfig, request: pytest.FixtureRequest
):
    file_save_path = config_instance.get_config("reports").get("base_dir")
    task = MultiTurnDialogPerfTask(model_config, perf_config, file_save_path)
    result = task.run()
    return {"_name": request.node.callspec.id, "_data": result}


doc_qa_perf_cases = [
    pytest.param(
        PerfConfig(
            data_type="doc_qa",
            dataset_file_path="common/uc_eval/datasets/doc_qa/demo.jsonl",
            enable_prefix_cache=False,
            parallel_num=1,
            benchmark_mode="default-perf",
        ),
        id="doc-qa-complete-recalculate-default-perf",
    )
]


@pytest.mark.feature("perf_test")
@pytest.mark.parametrize("perf_config", doc_qa_perf_cases)
@export_vars
def test_doc_qa_perf(
    perf_config: PerfConfig, model_config: ModelConfig, request: pytest.FixtureRequest
):
    file_save_path = config_instance.get_config("reports").get("base_dir")
    task = DocQaPerfTask(model_config, perf_config, file_save_path)
    result = task.run()
    return {"_name": request.node.callspec.id, "_data": result}
