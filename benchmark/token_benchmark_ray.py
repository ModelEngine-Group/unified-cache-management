import argparse
import json
import os
import random
import re
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import ray
from llmperf import common_metrics
from llmperf.common import SUPPORTED_APIS, construct_clients
from llmperf.models import RequestConfig
from llmperf.requests_launcher import RequestsLauncher
from llmperf.utils import (  # randomly_sample_sonnet_lines_prompt,
    LLMPerfResults,
    get_accuracy,
    get_messages_from_dataset_files,
    get_messages_from_multi_turn_dataset_files,
    get_prompts_from_dataset_files,
    sample_random_positive_int,
)
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaTokenizerFast


def get_token_throughput_latencies(
    round: int,
    past_llm_output: str,
    past_message: Tuple[List[Dict], int],
    model: str,
    model_path: str,
    mean_input_tokens: int,
    stddev_input_tokens: int,
    mean_output_tokens: int,
    stddev_output_tokens: int,
    additional_sampling_params: Optional[Dict[str, Any]] = None,
    num_concurrent_requests: int = 1,
    max_num_completed_requests: int = 500,
    test_timeout_s=90,
    llm_api="openai",
    dataset_file_names="",
    scenario="",
    context_length=4,
    delimiter: str = " # # ",
    use_delimiter: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Get the token throughput and latencies for the given model.

    Args:
        model: The name of the model to query.
        model_path: Path of the model source file.
        mean_input_tokens: The mean number of tokens to send in the prompt for the request.
        stddev_input_tokens: The standard deviation of the number of tokens to send in the prompt for the request.
        mean_output_tokens: The mean number of tokens to generate per request.
        stddev_output_tokens: The standard deviation of the number of tokens to generate per request.
        additional_sampling_params: Additional sampling parameters to send with the request.
            For more information see the LLM APIs documentation for the completions
        num_concurrent_requests: The number of concurrent requests to make. Increase
            this to increase the amount of load and vice versa.
        test_timeout_s: The amount of time to run the test for before reporting results.
        llm_api: The name of the llm api to use. Either "openai" or "litellm".

    Returns:
        A summary of the performance metrics collected across all completed requests
        (e.g. throughput, latencies, etc.)
        The individual metrics for each request.
    """
    random.seed(11111)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    get_token_length = lambda text: len(tokenizer.encode(text))

    if not additional_sampling_params:
        additional_sampling_params = {}

    completed_requests_lock = threading.Lock()
    completed_requests = []
    num_completed_requests = 0

    num_output_tokens_list = []
    messages = []

    # for i in range(max_num_completed_requests):
    #     num_output_tokens = (sample_random_positive_int(
    #         mean_output_tokens, stddev_output_tokens
    #     ))
    #     num_output_tokens_list.append(num_output_tokens)

    #     prompts.append(randomly_sample_sonnet_lines_prompt(
    #         prompt_tokens_mean=mean_input_tokens,
    #         prompt_tokens_stddev=stddev_input_tokens,
    #         expect_output_tokens=num_output_tokens,
    #         tokenizer=tokenizer
    #     ))

    if scenario == "doc-qa":
        messages, num_output_tokens_list, ground_truths_list, questions_list = (
            get_messages_from_dataset_files(
                dataset_file_names=dataset_file_names,
                tokenizer=tokenizer,
                mean_output_tokens=mean_output_tokens,
                stddev_output_tokens=stddev_output_tokens,
                context_length=context_length,
                delimiter=delimiter,
                use_delimiter=use_delimiter,
            )
        )
    else:
        messages, num_output_tokens_list = get_messages_from_multi_turn_dataset_files(
            round=round,
            past_llm_output=past_llm_output,
            past_message=past_message,
            dataset_file_names=dataset_file_names,
            tokenizer=tokenizer,
            mean_output_tokens=mean_output_tokens,
            stddev_output_tokens=stddev_output_tokens,
        )

    max_num_completed_requests = len(messages)

    start_time = time.monotonic()
    pbar = tqdm(total=max_num_completed_requests)

    llm_outputs = []

    def launch_request(thread_index):
        nonlocal num_completed_requests
        clients = construct_clients(llm_api=llm_api, num_clients=1)
        req_launcher = RequestsLauncher(clients)
        request_index = thread_index % max_num_completed_requests

        while (
            time.monotonic() - start_time < test_timeout_s
            and num_completed_requests < max_num_completed_requests
        ):

            default_sampling_params = {
                "max_tokens": num_output_tokens_list[request_index]
            }
            default_sampling_params.update(additional_sampling_params)
            request_config = RequestConfig(
                model=model,
                # prompt=prompts[request_index],
                message=messages[request_index],
                sampling_params=default_sampling_params,
                llm_api=llm_api,
            )
            req_launcher.launch_requests(request_config)

            outs = req_launcher.get_next_ready()
            all_metrics = []
            for out in outs:
                request_metrics, gen_text, _ = out
                llm_outputs.append(gen_text)
                num_output_tokens = get_token_length(gen_text)
                with completed_requests_lock:
                    if num_completed_requests < max_num_completed_requests:
                        if num_output_tokens:
                            request_metrics[
                                common_metrics.INTER_TOKEN_LAT
                            ] /= request_metrics[common_metrics.NUM_OUTPUT_TOKENS]
                        else:
                            request_metrics[common_metrics.INTER_TOKEN_LAT] = 0
                        request_metrics[common_metrics.NUM_OUTPUT_TOKENS] = (
                            num_output_tokens
                        )
                        request_metrics[common_metrics.NUM_TOTAL_TOKENS] = (
                            request_metrics[common_metrics.NUM_INPUT_TOKENS]
                            + num_output_tokens
                        )
                        request_metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = (
                            num_output_tokens / request_metrics[common_metrics.E2E_LAT]
                        )
                        all_metrics.append(request_metrics)
                        completed_requests.extend(all_metrics)
                        pbar.update(len(all_metrics))
                        num_completed_requests += len(all_metrics)
                        request_index = (
                            request_index + num_concurrent_requests
                        ) % max_num_completed_requests

    threads = []
    for i in range(num_concurrent_requests):
        thread = threading.Thread(target=launch_request, args=(i,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    pbar.close()
    end_time = time.monotonic()
    if end_time - start_time >= test_timeout_s:
        print("Test timed out before all requests could be completed.")

    # check one last time that there are no remaining results to collect.
    clients = construct_clients(llm_api=llm_api, num_clients=1)
    req_launcher = RequestsLauncher(clients)
    outs = req_launcher.get_next_ready()
    all_metrics = []
    for out in outs:
        request_metrics, gen_text, _ = out
        llm_outputs.append(gen_text)
        num_output_tokens = get_token_length(gen_text)
        with completed_requests_lock:
            if num_completed_requests < max_num_completed_requests:
                if num_output_tokens:
                    request_metrics[common_metrics.INTER_TOKEN_LAT] /= num_output_tokens
                else:
                    request_metrics[common_metrics.INTER_TOKEN_LAT] = 0
                request_metrics[common_metrics.NUM_OUTPUT_TOKENS] = num_output_tokens
                request_metrics[common_metrics.NUM_TOTAL_TOKENS] = (
                    request_metrics[common_metrics.NUM_INPUT_TOKENS] + num_output_tokens
                )
                request_metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = (
                    num_output_tokens / request_metrics[common_metrics.E2E_LAT]
                )
                completed_requests.extend(request_metrics)

    print(f"\Results for token benchmark for {model} queried with the {llm_api} api.\n")

    ret = metrics_summary(
        completed_requests,
        start_time,
        end_time,
        llm_outputs,
        ground_truths_list,
        questions_list,
    )

    metadata = {
        "model": model,
        "mean_input_tokens": mean_input_tokens,
        "stddev_input_tokens": stddev_input_tokens,
        "mean_output_tokens": mean_output_tokens,
        "stddev_output_tokens": stddev_output_tokens,
        "num_concurrent_requests": num_concurrent_requests,
        "additional_sampling_params": additional_sampling_params,
    }

    metadata["results"] = ret

    return metadata, completed_requests, messages, llm_outputs


def metrics_summary(
    metrics: List[Dict[str, Any]],
    start_time: int,
    end_time: int,
    llm_outputs: list[str],
    ground_truths_list: list[list[str]],
    questions_list: list[str],
) -> Dict[str, Any]:
    """Generate a summary over metrics generated from potentially multiple instances of this client.

    Args:
        metrics: The metrics to summarize.
        start_time: The time the test started.
        end_time: The time the test ended.
        llm_outputs: Answers that LLM gives to the questions
        ground_truths_list: Groud-Truth answers of the questions
        questions_list: Contents of the questions

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
            - Precision
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
        # qs = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        qs = []
        quantiles = series.quantile(qs).to_dict()
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

    ret[common_metrics.ACCURACY] = get_accuracy(
        questions=questions_list,
        ground_truths=ground_truths_list,
        llm_outputs=llm_outputs,
    )
    print(f"Mean accuracy: {ret[common_metrics.ACCURACY]}")

    return ret


def run_token_benchmark(
    llm_api: str,
    model: str,
    model_path: str,
    test_timeout_s: int,
    max_num_completed_requests: int,
    num_concurrent_requests: int,
    mean_input_tokens: int,
    stddev_input_tokens: int,
    mean_output_tokens: int,
    stddev_output_tokens: int,
    additional_sampling_params: str,
    results_dir: str,
    user_metadata: Dict[str, Any],
    dataset_file_names: str,
    scenario: str,
    max_round: int,
    context_length: int,
    connector: str,
    device: str,
    delimiter: str,
    use_delimiter: bool,
):
    """
    Args:
        llm_api: The name of the llm api to use.
        model: The name of the model to query.
        max_num_completed_requests: The number of requests to complete before finishing the test.
        test_timeout_s: The amount of time to run the test for before reporting results.
        num_concurrent_requests: The number of concurrent requests to make. Increase
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

    llm_outputs = [""]
    messages = [([], 0)]

    rounds = []
    ttft_means = []

    if scenario == "doc-qa":
        max_round = 1

    for rnd in range(max_round):
        summary, individual_responses, messages, llm_outputs = (
            get_token_throughput_latencies(
                round=rnd,
                past_llm_output=llm_outputs[0],
                past_message=messages[0],
                model=model,
                model_path=model_path,
                llm_api=llm_api,
                test_timeout_s=test_timeout_s,
                max_num_completed_requests=max_num_completed_requests,
                mean_input_tokens=mean_input_tokens,
                stddev_input_tokens=stddev_input_tokens,
                mean_output_tokens=mean_output_tokens,
                stddev_output_tokens=stddev_output_tokens,
                num_concurrent_requests=num_concurrent_requests,
                additional_sampling_params=json.loads(additional_sampling_params),
                dataset_file_names=dataset_file_names,
                scenario=scenario,
                context_length=context_length,
                delimiter=delimiter,
                use_delimiter=use_delimiter,
            )
        )
        rounds.append(rnd)
        ttft_means.append(summary["results"][common_metrics.TTFT]["mean"])

    if scenario == "doc-qa":
        result_to_save = [{"mean ttft": ttft_means[0]}]
        with open(
            results_dir
            + "/docqa_TTFT_"
            + str(context_length)
            + "k_"
            + os.path.basename(model)
            + "_"
            + connector
            + "_connector_"
            + device
            + ".jsonl",
            "a",
            encoding="utf-8",
        ) as file:
            json.dump(result_to_save, file, ensure_ascii=False)
            file.write("\n")

    if scenario == "multi-turn-dialogue":

        import matplotlib.pyplot as plt

        plt.plot(rounds, ttft_means, marker="o")
        plt.title("TTFT_curve_multi_turn_dialogue")
        plt.xlabel("round")
        plt.ylabel("TTFT")
        plt.grid(True)
        plt.legend(["stub_12_0802"])
        plt.savefig(
            results_dir
            + "/multi_turn_dialogue_TTFT_"
            + os.path.basename(model)
            + "_"
            + connector
            + "_connector_"
            + device
            + ".png"
        )


args = argparse.ArgumentParser(
    description="Run a token throughput and latency benchmark."
)

args.add_argument(
    "--model", type=str, required=True, help="The model to use for this load test."
)

args.add_argument(
    "--model-path", type=str, required=True, help="The path of the model source flie."
)

args.add_argument(
    "--mean-input-tokens",
    type=int,
    default=550,
    help=(
        "The mean number of tokens to send in the prompt for the request. "
        " (default: %(default)s)"
    ),
)
args.add_argument(
    "--stddev-input-tokens",
    type=int,
    default=150,
    help=(
        "The standard deviation of number of tokens to send in the prompt for the request. "
        "(default: %(default)s)"
    ),
)
args.add_argument(
    "--mean-output-tokens",
    type=int,
    default=150,
    help=(
        "The mean number of tokens to generate from each llm request. This is the max_tokens param "
        "for the completions API. Note that this is not always the number of tokens returned. "
        "(default: %(default)s)"
    ),
)
args.add_argument(
    "--stddev-output-tokens",
    type=int,
    default=80,
    help=(
        "The stdandard deviation on the number of tokens to generate per llm request. "
        "(default: %(default)s)"
    ),
)
args.add_argument(
    "--num-concurrent-requests",
    type=int,
    default=10,
    help=("The number of concurrent requests to send (default: %(default)s)"),
)
args.add_argument(
    "--timeout",
    type=int,
    default=90,
    help="The amount of time to run the load test for. (default: %(default)s)",
)
args.add_argument(
    "--max-num-completed-requests",
    type=int,
    default=10,
    help=(
        "The number of requests to complete before finishing the test. Note "
        "that its possible for the test to timeout first. (default: %(default)s)"
    ),
)
args.add_argument(
    "--additional-sampling-params",
    type=str,
    default="{}",
    help=(
        "Additional sampling params to send with the each request to the LLM API. "
        "(default: %(default)s) No additional sampling params are sent."
    ),
)
args.add_argument(
    "--results-dir",
    type=str,
    default="",
    help=(
        "The directory to save the results to. "
        "(`default: %(default)s`) No results are saved)"
    ),
)
args.add_argument(
    "--llm-api",
    type=str,
    default="openai",
    help=(
        f"The name of the llm api to use. Can select from {SUPPORTED_APIS}"
        " (default: %(default)s)"
    ),
)
args.add_argument(
    "--metadata",
    type=str,
    default="",
    help=(
        "A comma separated list of metadata to include in the results, e.g. "
        "name=foo,bar=1. These will be added to the metadata field of the results. "
    ),
)

args.add_argument(
    "--dataset-file-names",
    type=str,
    default="",
    help=("Names of the dataset files. Use commas to separate multiple names. "),
)

args.add_argument(
    "--scenario",
    type=str,
    default="",
    help=(
        "Scenario for current test. Only supports 'multi-turn-dialogue' and 'doc-qa'. "
    ),
)

args.add_argument(
    "--max-round",
    type=int,
    default=1,
    help=(
        "Maximum number of rounds in the scenario of multi-turnl dialogue. Can be not set if scenario==doc-qa. "
    ),
)

args.add_argument(
    "--context-length",
    type=int,
    default=4,
    help=(
        "Context length to measure the performance. Unit: K tokens. Only supports 2/4/8/16/32/64 now. "
    ),
)

args.add_argument("--connector", type=str, required=True, help=("Connector to use"))

args.add_argument(
    "--device", type=str, required=True, help=("Compute Device to use: GPU/NPU")
)

args.add_argument(
    "--use-delimiter",
    action="store_true",
    help=("Use delimiter to split input context into 'chunks'. "),
)

args.add_argument(
    "--delimiter",
    type=str,
    default=" # # ",
    help=(
        "Delimiter to split text into 'chunks'. Should align with CacheBlend setting if --use-delimiter flag is set. "
    ),
)

if __name__ == "__main__":
    env_vars = dict(os.environ)
    ray.init(runtime_env={"env_vars": env_vars})
    args = args.parse_args()

    assert args.scenario == "multi-turn-dialogue" or args.scenario == "doc-qa"

    # Parse user metadata.
    user_metadata = {}
    if args.metadata:
        for item in args.metadata.split(","):
            key, value = item.split("=")
            user_metadata[key] = value

    run_token_benchmark(
        llm_api=args.llm_api,
        model=args.model,
        model_path=args.model_path,
        test_timeout_s=args.timeout,
        max_num_completed_requests=args.max_num_completed_requests,
        mean_input_tokens=args.mean_input_tokens,
        stddev_input_tokens=args.stddev_input_tokens,
        mean_output_tokens=args.mean_output_tokens,
        stddev_output_tokens=args.stddev_output_tokens,
        num_concurrent_requests=args.num_concurrent_requests,
        additional_sampling_params=args.additional_sampling_params,
        results_dir=args.results_dir,
        user_metadata=user_metadata,
        dataset_file_names=args.dataset_file_names,
        scenario=args.scenario,
        max_round=args.max_round,
        context_length=args.context_length,
        connector=args.connector,
        device=args.device,
        delimiter=args.delimiter,
        use_delimiter=args.use_delimiter,
    )
