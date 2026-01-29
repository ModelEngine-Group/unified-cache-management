import dataclasses
import os

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

perf_scenarios = [
    # (mean_in, mean_out, max_req, concurrent, sampling, hit_rate)
    (1000, 1024, 1, 1, "{}", 0),
    (4000, 500, 1, 1, "{}", 0),
]
scenario_ids = [f"in_{s[0]}-out_{s[1]}-con_{s[3]}" for s in perf_scenarios]
TOTAL_COUNTER = len(perf_scenarios)
ROUND_COUNTER = 1


@pytest.mark.stage(2)
@pytest.mark.feature("uc_performance_test")
@pytest.mark.parametrize(
    "in_tokens, out_tokens, max_req, concurrent, sampling, hit_rate",
    perf_scenarios,
    ids=scenario_ids,
)
@export_vars
def test_performance(in_tokens, out_tokens, max_req, concurrent, sampling, hit_rate):
    global TOTAL_COUNTER
    global ROUND_COUNTER
    summary = inference_results(
        [in_tokens],
        [out_tokens],
        [max_req],
        [concurrent],
        [sampling],
        [hit_rate],
        TOTAL_COUNTER,
        ROUND_COUNTER,
    )
    ROUND_COUNTER += 1
    results = summary.get("results", {})

    # 构造扁平化的结果字典，方便后续分析和看板展示
    metrics = {
        # 输入指标
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
        "concurrent": concurrent,
        "sum_requests": max_req,
        "hit_rate": hit_rate,
        "ttft_mean": results.get("ttft_s", {}).get("mean"),
        "tpot_mean": results.get("inter_token_latency_s", {}).get("mean"),
        "total_throughput": summary.get("total_throughput"),
        "e2e_mean": results.get("end_to_end_latency_s", {}).get("mean"),
        "extra_info": os.getenv("TEST_EXTRA_INFO")
        or config_instance.get_nested_config("llm_connection.extra_info"),
        "mean_input_tokens": summary.get("mean_input_tokens"),
        "mean_output_tokens": summary.get("mean_output_tokens"),
        # Latency ITL
        "itl_p50": results.get("inter_token_latency_s", {})
        .get("quantiles", {})
        .get("p50"),
        "itl_p90": results.get("inter_token_latency_s", {})
        .get("quantiles", {})
        .get("p90"),
        "itl_p99": results.get("inter_token_latency_s", {})
        .get("quantiles", {})
        .get("p99"),
        # TTFT
        "ttft_p50": results.get("ttft_s", {}).get("quantiles", {}).get("p50"),
        "ttft_p90": results.get("ttft_s", {}).get("quantiles", {}).get("p90"),
        "ttft_p99": results.get("ttft_s", {}).get("quantiles", {}).get("p99"),
        # End to End
        "e2e_p50": results.get("end_to_end_latency_s", {})
        .get("quantiles", {})
        .get("p50"),
        "e2e_p90": results.get("end_to_end_latency_s", {})
        .get("quantiles", {})
        .get("p90"),
        "e2e_p99": results.get("end_to_end_latency_s", {})
        .get("quantiles", {})
        .get("p99"),
        # Throughput & Stats
        "num_completed_requests": summary.get("num_completed_requests"),
        "elapsed_time": summary.get("elapsed_time"),
        "incremental_throughput": summary.get("incremental_throughput"),
    }

    for key, val in metrics.items():
        assert val is not None, f"Metric '{key}' is missing"

    return {"_name": "llmperf", "_proj": metrics}


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
@pytest.mark.stage(2)
@pytest.mark.parametrize("perf_config", sync_perf_cases)
@export_vars
def test_sync_perf(
    perf_config: PerfConfig, model_config: ModelConfig, request: pytest.FixtureRequest
):
    file_save_path = config_instance.get_config("reports").get("base_dir")
    task = SyntheticPerfTask(model_config, perf_config, file_save_path)
    result = task.run()
    return {"_name": request.node.callspec.id, "_proj": result}


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
@pytest.mark.stage(2)
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
@pytest.mark.stage(2)
@pytest.mark.parametrize("perf_config", doc_qa_perf_cases)
@export_vars
def test_doc_qa_perf(
    perf_config: PerfConfig, model_config: ModelConfig, request: pytest.FixtureRequest
):
    file_save_path = config_instance.get_config("reports").get("base_dir")
    task = DocQaPerfTask(model_config, perf_config, file_save_path)
    result = task.run()
    return {"_name": request.node.callspec.id, "_data": result}
