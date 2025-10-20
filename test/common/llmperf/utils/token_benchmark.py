import json
import logging
import random
import re
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from common.llmperf.utils import common_metrics
from common.llmperf.utils.models import RequestConfig
from common.llmperf.utils.openai_chat_completions_client import (
    OpenAIChatCompletionsClient,
)
from common.llmperf.utils.utils import (
    LLMPerfResults,
    randomly_sample_sonnet_lines_prompt,
    sample_random_positive_int,
)
from transformers import AutoTokenizer


def get_token_throughput_latencies(
    model: str,
    mean_input_tokens: int,
    stddev_input_tokens: int,
    mean_output_tokens: int,
    stddev_output_tokens: int,
    additional_sampling_params: Optional[Dict[str, Any]] = None,
    concurrent_requests: int = 1,
    max_num_completed_requests: int = 500,
    test_timeout_s=90,
    llm_api="openai",
    random_seed: int = None,
    openai_api_base: str = "",
    tokenizer_path: str = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], float, float]:
    """Get the token throughput and latencies for the given model.

    Args:
        model: The name of the model to query.
        mean_input_tokens: The mean number of tokens to send in the prompt for the request.
        stddev_input_tokens: The standard deviation of the number of tokens to send in the prompt for the request.
        mean_output_tokens: The mean number of tokens to generate per request.
        stddev_output_tokens: The standard deviation of the number of tokens to generate per request.
        additional_sampling_params: Additional sampling parameters to send with the request.
            For more information see the LLM APIs documentation for the completions
        concurrent_requests: The number of concurrent requests to make. Increase
            this to increase the amount of load and vice versa.
        test_timeout_s: The amount of time to run the test for before reporting results.
        llm_api: The name of the llm api to use. Either "openai" or "litellm".

    Returns:
        A summary of the performance metrics collected across all completed requests
        (e.g. throughput, latencies, etc.)
        The individual metrics for each request.
    """
    random.seed(random_seed)

    print(f"Using tokenizer:{tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    get_token_length = lambda text: len(tokenizer.encode(text))

    if not additional_sampling_params:
        additional_sampling_params = {}

        # 1. create prompts
        prompts: List[Tuple[str, int]] = []
        num_output_tokens_list: List[int] = []
        for i in range(max_num_completed_requests):
            num_output = sample_random_positive_int(
                mean_output_tokens, stddev_output_tokens
            )
            num_output_tokens_list.append(num_output)
            prompts.append(
                randomly_sample_sonnet_lines_prompt(
                    prompt_tokens_mean=mean_input_tokens,
                    prompt_tokens_stddev=stddev_input_tokens,
                    tokenizer=tokenizer,
                )
            )
        start_time = time.monotonic()
        completed_requests: List[Dict[str, Any]] = []
        incremental_time_delay = 0.0
        client = OpenAIChatCompletionsClient()
        futures = []

        # 2. Submitting tasks using a thread pool
        with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            for idx in range(max_num_completed_requests):
                sampling = {"max_tokens": num_output_tokens_list[idx]}
                sampling.update(additional_sampling_params)
                cfg = RequestConfig(
                    model=model,
                    prompt=prompts[idx],
                    sampling_params=sampling,
                    llm_api=llm_api,
                    openai_api_base=openai_api_base,
                )
                futures.append(executor.submit(client.llm_request, cfg))
            # 3. Waiting for completion or timeout
            for future in as_completed(futures, timeout=test_timeout_s):
                try:
                    metrics, gen_text, req_cfg = future.result()
                except Exception as e:
                    logging.warning(f"[WARN] Future raised exception: {e}")
                    continue
                num_output_tokens = get_token_length(gen_text)
                if num_output_tokens:
                    metrics[common_metrics.INTER_TOKEN_LAT] /= (
                        (metrics[common_metrics.NUM_OUTPUT_TOKENS] - 1)
                        if (metrics[common_metrics.NUM_OUTPUT_TOKENS] - 1)
                        else 1
                    )
                    metrics[common_metrics.NUM_OUTPUT_TOKENS] = num_output_tokens
                    metrics[common_metrics.NUM_TOTAL_TOKENS] = (
                        metrics[common_metrics.NUM_INPUT_TOKENS] + num_output_tokens
                    )
                    try:
                        metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = (
                            num_output_tokens / metrics[common_metrics.E2E_LAT]
                        )
                    except ZeroDivisionError:
                        logging.error("Division by zero in throughput calculation.")

                completed_requests.append(metrics)

                incremental_time_delay += metrics.get(
                    common_metrics.INTER_TOKEN_LAT, 0.0
                )

        end_time = time.monotonic()

    print(f"Results for token benchmark for {model} queried with the {llm_api} api.\n")
    if mean_output_tokens == 2:
        print(f"[INFO] First token sending pre-embedding completed\n")
        return {}, [], 0.0, 0.0

    ret = metrics_summary(completed_requests, start_time, end_time)

    metadata = {
        "model": model,
        "mean_input_tokens": mean_input_tokens,
        "stddev_input_tokens": stddev_input_tokens,
        "mean_output_tokens": mean_output_tokens,
        "stddev_output_tokens": stddev_output_tokens,
        "concurrent_requests": concurrent_requests,
        "additional_sampling_params": additional_sampling_params,
    }

    metadata["results"] = ret
    elapsed_time = end_time - start_time
    return metadata, completed_requests, elapsed_time, incremental_time_delay


def compute_throughput(
    summary: Dict[str, Any],
    completed_requests: List[Dict[str, Any]],
    elapsed_time: float,
    incremental_time_delay: float,
) -> Tuple[float, float]:
    """
    Compute total_throughput (token/s) based on the metrics in summary.

    Formula: (mean_output_tokens * num_completed_requests) / total_e2e_latency_s

    Args:
        summary (Dict[str, Any]): A dictionary containing performance metrics.

    Returns:
        float: The computed total throughput in tokens per second. Returns 0.0 if latency is zero.
    """
    mean_output_tokens = summary.get("mean_output_tokens", 0)

    total_throughput = (
        (mean_output_tokens * len(completed_requests)) / elapsed_time
        if elapsed_time > 0
        else 0.0
    )
    incremental_throughput = (
        (mean_output_tokens * len(completed_requests)) / incremental_time_delay
        if incremental_time_delay > 0
        else 0.0
    )
    return round(total_throughput, 4), round(incremental_throughput, 4)


def metrics_summary(
    metrics: List[Dict[str, Any]], start_time: int, end_time: int
) -> Dict[str, Any]:
    """Generate a summary over metrics generated from potentially multiple instances of this client.

    Args:
        metrics: The metrics to summarize.
        start_time: The time the test started.
        end_time: The time the test ended.

    Returns:
        A summary with the following information:
            - Overall throughput (generated tokens / total test time)
            - Number of completed requests
            - Error rate
            - Error code frequency
            - Quantiles (p25-p99) for the following metrics:
                - Inter token latency
                - Time to first token
                - User total request time
                - Number of tokens processed per request
                - Number of tokens generated per request
                - User throughput (tokens / s)
    """
    ret = {}

    def flatten(item):
        for sub_item in item:
            if isinstance(sub_item, Iterable) and not isinstance(sub_item, str):
                yield from flatten(sub_item)
            else:
                yield sub_item

    df = pd.DataFrame(metrics)
    df_without_errored_req = df[df[common_metrics.ERROR_CODE].isna()]

    for key in [
        common_metrics.INTER_TOKEN_LAT,
        common_metrics.TTFT,
        common_metrics.E2E_LAT,
        common_metrics.REQ_OUTPUT_THROUGHPUT,
        common_metrics.NUM_INPUT_TOKENS,
        common_metrics.NUM_OUTPUT_TOKENS,
    ]:
        print(key)
        ret[key] = {}
        series = pd.Series(list(flatten(df_without_errored_req[key]))).dropna()
        series = series[series > 0]  # Calculate non-zero values
        quantiles = series.quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_dict()
        quantiles_reformatted_keys = {}
        for quantile, value in quantiles.items():
            reformatted_key = f"p{int(quantile * 100)}"
            print(f"    {reformatted_key} = {value}")
            quantiles_reformatted_keys[reformatted_key] = value
        ret[key]["quantiles"] = quantiles_reformatted_keys
        mean = series.mean()
        print(f"    mean = {mean}")
        ret[key]["mean"] = mean
        print(f"    min = {series.min()}")
        ret[key]["min"] = series.min()
        print(f"    max = {series.max()}")
        ret[key]["max"] = series.max()
        print(f"    stddev = {series.std()}")
        ret[key]["stddev"] = series.std()

    ret[common_metrics.NUM_REQ_STARTED] = len(metrics)

    error_codes = df[common_metrics.ERROR_CODE].dropna()
    num_errors = len(error_codes)
    ret[common_metrics.ERROR_RATE] = num_errors / len(metrics) if len(metrics) else 0
    ret[common_metrics.NUM_ERRORS] = num_errors
    print(f"Number Of Errored Requests: {num_errors}")
    error_code_frequency = dict(error_codes.value_counts())
    if num_errors:
        error_code_frequency = dict(error_codes.value_counts())
        print("Error Code Frequency")
        print(error_code_frequency)
    ret[common_metrics.ERROR_CODE_FREQ] = str(error_code_frequency)

    overall_output_throughput = df_without_errored_req[
        common_metrics.NUM_OUTPUT_TOKENS
    ].sum() / (end_time - start_time)

    print(f"Overall Output Throughput: {overall_output_throughput}")
    ret[common_metrics.OUTPUT_THROUGHPUT] = overall_output_throughput

    num_completed_requests = len(df_without_errored_req)
    num_completed_requests_per_min = (
        num_completed_requests / (end_time - start_time) * 60
    )
    print(f"Number Of Completed Requests: {num_completed_requests}")
    print(f"Completed Requests Per Minute: {num_completed_requests_per_min}")

    ret[common_metrics.NUM_COMPLETED_REQUESTS] = num_completed_requests
    ret[common_metrics.COMPLETED_REQUESTS_PER_MIN] = num_completed_requests_per_min

    return ret


def run_token_benchmark(
    llm_api: str,
    model: str,
    test_timeout_s: int,
    max_num_completed_requests: int,
    concurrent_requests: int,
    mean_input_tokens: int,
    stddev_input_tokens: int,
    mean_output_tokens: int,
    stddev_output_tokens: int,
    additional_sampling_params: str,
    results_dir: str,
    random_seed: int,
    openai_api_base: str,
    tokenizer_path: str,
    user_metadata: Dict[str, Any],
):
    """
    Args:
        llm_api: The name of the llm api to use.
        model: The name of the model to query.
        max_num_completed_requests: The number of requests to complete before finishing the test.
        test_timeout_s: The amount of time to run the test for before reporting results.
        concurrent_requests: The number of concurrent requests to make. Increase
            this to increase the amount of load and vice versa.
        mean_input_tokens: The mean number of tokens to send in the prompt for the request.
        stddev_input_tokens: The standard deviation of the number of tokens to send in the prompt for the request.
        mean_output_tokens: The mean number of tokens to generate per request.
        stddev_output_tokens: The standard deviation of the number of tokens to generate per request.
        additional_sampling_params: Additional sampling parameters to send with the request.
            For more information see the LLM APIs documentation for the completions.
        results_dir: The directory to save the results to.
        user_metadata: Additional metadata to include in the results.
    """
    if mean_input_tokens < 40:
        print(
            "the minimum number of input tokens that will be sent is 41"
            " because of the prompting logic right now"
        )

    summary, completed_requests, elapsed_time, incremental_time_delay = (
        get_token_throughput_latencies(
            model=model,
            llm_api=llm_api,
            test_timeout_s=test_timeout_s,
            max_num_completed_requests=max_num_completed_requests,
            mean_input_tokens=mean_input_tokens,
            stddev_input_tokens=stddev_input_tokens,
            mean_output_tokens=mean_output_tokens,
            stddev_output_tokens=stddev_output_tokens,
            concurrent_requests=concurrent_requests,
            additional_sampling_params=json.loads(additional_sampling_params),
            random_seed=random_seed,
            openai_api_base=openai_api_base,
            tokenizer_path=tokenizer_path,
        )
    )
    if mean_output_tokens == 2:
        return summary, completed_requests, elapsed_time, incremental_time_delay

    timestamp = int(time.time() * 1000)
    if results_dir:
        filename = f"{model}_{mean_input_tokens}_{mean_output_tokens}_{timestamp}"
        filename = re.sub(r"[^\w\d-]+", "-", filename)
        filename = re.sub(r"-{2,}", "-", filename)
        summary_filename = f"{filename}_summary"

        # Update to metadata.
        summary.update(user_metadata)
        total_tp, req_tp = compute_throughput(
            summary, completed_requests, elapsed_time, incremental_time_delay
        )
        summary["num_completed_requests"] = len(completed_requests)
        summary["elapsed_time"] = elapsed_time
        summary["incremental_time_delay"] = incremental_time_delay
        summary["total_throughput"] = total_tp
        summary["incremental_throughput"] = req_tp

        results = LLMPerfResults(name=summary_filename, metadata=summary)
        results_dir = Path(results_dir)
        if not results_dir.exists():
            results_dir.mkdir(parents=True)
        elif not results_dir.is_dir():
            raise ValueError(f"{results_dir} is not a directory")

        llmperf_dir = results_dir / "llmperf"
        if not llmperf_dir.exists():
            llmperf_dir.mkdir(parents=True)
        elif not llmperf_dir.is_dir():
            raise ValueError(f"{llmperf_dir} is not a directory")

        try:
            with open(llmperf_dir / f"{summary_filename}.json", "w") as f:
                json.dump(results.to_dict(), f, indent=4, default=str)
        except Exception as e:
            print(results.to_dict())
            raise e
    return summary
