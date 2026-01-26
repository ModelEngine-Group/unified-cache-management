import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List

import yaml
from common.llmperf.utils.token_benchmark import run_token_benchmark
from common.llmperf.utils.utils import reset_prefill_cache


def run_test_cases(
    llm_api,
    model,
    timeout,
    max_num_completed_requests,
    concurrent_requests,
    mean_input_tokens,
    stddev_input,
    mean_output_tokens,
    stddev_output,
    additional_sampling_params,
    timestamp_dir,
    server_url,
    tokenizer_path,
    hit_rate,
    TOTAL_COUNTER,
    ROUND_COUNTER,
):
    print(f"[INFO] Total {TOTAL_COUNTER} test cases to be executed")
    all_summaries = []
    failed_case = []

    # Clear proxy environment variables
    env = os.environ.copy()
    env.pop("http_proxy", None)
    env.pop("https_proxy", None)

    for i, (
        mean_input,
        mean_output,
        max_completed,
        concurrent,
        additional_sampling_params,
        hit_rate_val,
    ) in enumerate(
        zip(
            mean_input_tokens,
            mean_output_tokens,
            max_num_completed_requests,
            concurrent_requests,
            additional_sampling_params,
            hit_rate,
        ),
        start=1,
    ):
        # for i, case in enumerate(mean_input_tokens):
        print(f"\n>>> Executing test case {ROUND_COUNTER} <<<")
        reset_prefill_cache(env, server_url)
        # Use a fixed random_seed for each test to control PC hit_rate
        random_seed = random.randint(1, 100000)

        try:
            # Determine if two runs are needed (PC hit_rate test)
            if hit_rate_val == 0:
                summary = run_token_benchmark(
                    llm_api=llm_api,
                    model=model,
                    test_timeout_s=timeout,
                    max_num_completed_requests=max_completed,
                    concurrent_requests=concurrent,
                    mean_input_tokens=mean_input,
                    stddev_input_tokens=stddev_input,
                    mean_output_tokens=mean_output,
                    stddev_output_tokens=stddev_output,
                    additional_sampling_params=additional_sampling_params,
                    results_dir=str(timestamp_dir),
                    random_seed=random_seed,
                    openai_api_base=server_url + "/v1",
                    tokenizer_path=tokenizer_path,
                    user_metadata={"case_idx": ROUND_COUNTER, "phase": "normal"},
                )
            else:
                print(
                    f"[INFO] hit_rate > 0 detected, entering prefill mode, PC hit rate: {hit_rate_val} %"
                )
                # hit_rate > 0: first prefill mode
                prefill_mean_input = int(mean_input * hit_rate_val / 100)
                print(
                    f"[INFO] Prefill execution: mean_input_tokens={prefill_mean_input}"
                )
                run_token_benchmark(
                    llm_api=llm_api,
                    model=model,
                    test_timeout_s=timeout,
                    max_num_completed_requests=max_completed,
                    concurrent_requests=concurrent,
                    mean_input_tokens=prefill_mean_input,
                    stddev_input_tokens=stddev_input,
                    mean_output_tokens=2,
                    stddev_output_tokens=stddev_output,
                    additional_sampling_params=additional_sampling_params,
                    results_dir=str(timestamp_dir),
                    random_seed=random_seed,
                    openai_api_base=server_url + "/v1",
                    tokenizer_path=tokenizer_path,
                    user_metadata={"case_idx": ROUND_COUNTER, "phase": "prefill"},
                )
                import time

                time.sleep(10)
                reset_prefill_cache(env, server_url)
                # Then run normal mode
                print("[INFO] Prefill completed, switching to normal mode execution")
                summary = run_token_benchmark(
                    llm_api=llm_api,
                    model=model,
                    test_timeout_s=timeout,
                    max_num_completed_requests=max_completed,
                    concurrent_requests=concurrent,
                    mean_input_tokens=mean_input,
                    stddev_input_tokens=stddev_input,
                    mean_output_tokens=mean_output,
                    stddev_output_tokens=stddev_output,
                    additional_sampling_params=additional_sampling_params,
                    results_dir=str(timestamp_dir),
                    random_seed=random_seed,
                    openai_api_base=server_url + "/v1",
                    tokenizer_path=tokenizer_path,
                    user_metadata={"case_idx": ROUND_COUNTER, "phase": "normal"},
                )
        except Exception as e:
            print(f"[Warning] {e}")
            failed_case.append(ROUND_COUNTER)

    return summary, failed_case


def inference_results(
    mean_input_tokens,
    mean_output_tokens,
    max_num_completed_requests,
    concurrent_requests,
    additional_sampling_params,
    hit_rate,
    TOTAL_COUNTER,
    ROUND_COUNTER,
):
    config_file = Path(__file__).parent.parent.parent / "config.yaml"
    print("[INFO] Initialization complete, starting main process")
    print(f"[INFO] Reading configuration file: {config_file}")
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        llm_api = config.get("llm_connection", {}).get("llm_api", "openai")
        model = config.get("llm_connection", {}).get("model", "")
        test_timeout_s = config.get("llm_connection", {}).get("test_timeout_s", 60000)
        stddev_input_tokens = config.get("llm_connection", {}).get(
            "stddev_input_tokens", 0
        )
        stddev_output_tokens = config.get("llm_connection", {}).get(
            "stddev_output_tokens", 0
        )
        timestamp_dir = Path("results")
        timestamp_dir.mkdir(parents=True, exist_ok=True)
        server_url = config.get("llm_connection", {}).get("server_url", "")
        tokenizer_path = config.get("llm_connection", {}).get("tokenizer_path", "")
        print(f"[INFO] Created results directory: {timestamp_dir}")

        summary, failed_cases = run_test_cases(
            llm_api,
            model,
            test_timeout_s,
            max_num_completed_requests,
            concurrent_requests,
            mean_input_tokens,
            stddev_input_tokens,
            mean_output_tokens,
            stddev_output_tokens,
            additional_sampling_params,
            timestamp_dir,
            server_url,
            tokenizer_path,
            hit_rate,
            TOTAL_COUNTER,
            ROUND_COUNTER,
        )

        if failed_cases:
            print(f"[WARN] Failed case indices: {failed_cases}")
    return summary
