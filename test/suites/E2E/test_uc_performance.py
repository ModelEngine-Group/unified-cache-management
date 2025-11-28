import pytest
from common.capture_utils import export_vars
from common.llmperf.run_inference import inference_results


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
