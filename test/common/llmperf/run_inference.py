import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List

import yaml
from common.config_utils import config_utils
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
    timestamp_dir,
    server_url,
    tokenizer_path,
    random_seed,
    hit_rate,
    TOTAL_COUNTER,
    ROUND_COUNTER,
):
    print(f"[INFO] Total {TOTAL_COUNTER} test cases to be executed")
    failed_case = []

    # clean proxy env
    env = os.environ.copy()
    env.pop("http_proxy", None)
    env.pop("https_proxy", None)

    llm_type = config_utils.get_nested_config("llm_connection.llm_type", "")
    import time

    def run_request(
        phase: str,
        mean_input: int,
        mean_output: int,
        max_completed: int,
        concurrent: int,
        seed,
    ):
        return run_token_benchmark(
            llm_api=llm_api,
            model=model,
            test_timeout_s=timeout,
            max_num_completed_requests=max_completed,
            concurrent_requests=concurrent,
            mean_input_tokens=mean_input,
            stddev_input_tokens=stddev_input,
            mean_output_tokens=mean_output,
            stddev_output_tokens=stddev_output,
            results_dir=str(timestamp_dir),
            random_seed=seed,
            openai_api_base=server_url + "/v1",
            tokenizer_path=tokenizer_path,
            user_metadata={
                "case_idx": ROUND_COUNTER,
                "phase": phase,
            },
        )

    def run_normal_case(args):
        return run_request(phase="normal", **args)

    def run_vllm(args, prefill_input):
        print("[INFO] vLLM mode: prefill -> normal")

        reset_prefill_cache(env, server_url, llm_type)

        prefill_args = dict(args)
        prefill_args["mean_input"] = prefill_input
        prefill_args["mean_output"] = 2

        run_request(
            phase="prefill",
            **prefill_args,
        )

        time.sleep(2)

        reset_prefill_cache(env, server_url, llm_type)

        print("[INFO] Prefill completed, switching to normal mode")
        return run_request(
            phase="normal",
            **args,
        )

    def run_sglang(args, prefill_input):
        print("[INFO] sglang mode: prefill_1 -> prefill_2 -> normal")

        reset_prefill_cache(env, server_url, llm_type)

        # ---------- prefill_1 ----------
        prefill1_args = dict(args)
        prefill1_args["mean_input"] = prefill_input
        prefill1_args["mean_output"] = 2

        run_request(
            phase="prefill_1",
            **prefill1_args,
        )
        time.sleep(2)

        # ---------- prefill_2 ----------
        prefill2_args = dict(args)
        prefill2_args["mean_input"] = prefill_input
        prefill2_args["mean_output"] = 2

        run_request(
            phase="prefill_2",
            **prefill2_args,
        )

        reset_prefill_cache(env, server_url, llm_type)
        time.sleep(2)

        # ---------- normal ----------
        print("[INFO] Prefill completed, switching to normal mode")
        return run_request(
            phase="normal",
            **args,
        )

    def run_mindie(args, prefill_input):
        print("[INFO] mindie mode: normal only (use prefill mean_input)")

        mindie_args = dict(args)
        mindie_args["mean_input"] = prefill_input

        return run_request(
            phase="normal",
            **mindie_args,
        )

    for (
        mean_input,
        mean_output,
        max_completed,
        concurrent,
        seed,
        hit_rate_val,
    ) in zip(
        mean_input_tokens,
        mean_output_tokens,
        max_num_completed_requests,
        concurrent_requests,
        random_seed,
        hit_rate,
    ):
        print(f"\n>>> Executing test case {ROUND_COUNTER} <<<")
        if seed == 0:
            seed = random.randint(1, 100000)

        args = dict(
            mean_input=mean_input,
            mean_output=mean_output,
            max_completed=max_completed,
            concurrent=concurrent,
            seed=seed,
        )

        try:
            if hit_rate_val == 0:
                summary = run_normal_case(args)
                continue

            prefill_mean_input = int(mean_input * hit_rate_val / 100)
            print(
                f"[INFO] hit_rate={hit_rate_val}%, prefill_mean_input={prefill_mean_input}"
            )

            if llm_type == "vllm":
                summary = run_vllm(args, prefill_mean_input)
            elif llm_type == "sglang":
                summary = run_sglang(args, prefill_mean_input)
            elif llm_type == "mindie":
                summary = run_mindie(args, prefill_mean_input)
            else:
                raise ValueError(f"Unsupported llm_type: {llm_type}")

        except Exception as e:
            print(f"[Warning] Case {ROUND_COUNTER} failed: {e}")
            failed_case.append(ROUND_COUNTER)

    return summary, failed_case


def inference_results(
    mean_input_tokens,
    mean_output_tokens,
    max_num_completed_requests,
    concurrent_requests,
    random_seed,
    hit_rate,
    TOTAL_COUNTER,
    ROUND_COUNTER,
):
    config_file = Path(__file__).parent.parent.parent / "config.yaml"

    llm_api = config_utils.get_nested_config("llm_connection.server_url", "")
    model = config_utils.get_nested_config("llm_connection.model", "")
    test_timeout_s = config_utils.get_nested_config(
        "llm_connection.test_timeout_s", 60000
    )
    stddev_input_tokens = config_utils.get_nested_config(
        "llm_connection.stddev_input_tokens", 0
    )
    stddev_output_tokens = config_utils.get_nested_config(
        "llm_connection.stddev_output_tokens", 0
    )
    timestamp_dir = Path("results")
    timestamp_dir.mkdir(parents=True, exist_ok=True)
    server_url = config_utils.get_nested_config("llm_connection.server_url", "")
    tokenizer_path = config_utils.get_nested_config("llm_connection.tokenizer_path", "")
    llm_config = {
        "server_url": config_utils.get_nested_config("llm_connection.server_url", ""),
        "model": config_utils.get_nested_config("llm_connection.model", ""),
        "test_timeout_s": config_utils.get_nested_config(
            "llm_connection.test_timeout_s", 60000
        ),
        "stddev_input_tokens": config_utils.get_nested_config(
            "llm_connection.stddev_input_tokens", 0
        ),
        "stddev_output_tokens": config_utils.get_nested_config(
            "llm_connection.stddev_output_tokens", 0
        ),
        "tokenizer_path": config_utils.get_nested_config(
            "llm_connection.tokenizer_path", ""
        ),
        "llm_type": config_utils.get_nested_config("llm_connection.llm_type", ""),
    }

    timestamp_dir = Path("results")
    timestamp_dir.mkdir(parents=True, exist_ok=True)

    print(
        "\n\n============================== Loaded LLM Config =============================="
    )
    for k, v in llm_config.items():
        print(f"{k:22}: {v}")
    print(f"{'results_dir':22}: {timestamp_dir.resolve()}")
    print(f"{'mean_in':22}: {mean_input_tokens[0]}")
    print(f"{'mean_out':22}: {mean_output_tokens[0]}")
    print(f"{'max_req':22}: {max_num_completed_requests[0]}")
    print(f"{'concurrent':22}: {concurrent_requests[0]}")
    print(f"{'random_seed':22}: {random_seed[0]}")
    print(f"{'hit_rate':22}: {hit_rate[0]}")

    print(
        "===============================================================================\n"
    )

    print("[INFO] Initialization complete, starting main process")
    print(f"[INFO] Reading configuration file: {config_file}")
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
        timestamp_dir,
        server_url,
        tokenizer_path,
        random_seed,
        hit_rate,
        TOTAL_COUNTER,
        ROUND_COUNTER,
    )

    if failed_cases:
        print(f"[WARN] Failed case indices: {failed_cases}")
    return summary
