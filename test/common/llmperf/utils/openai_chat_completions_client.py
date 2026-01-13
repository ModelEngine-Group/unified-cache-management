import json
import os
import time
from asyncio import timeout
from pathlib import Path
from typing import Any, Dict, Tuple

import requests
import yaml
from common.config_utils import config_utils as config_instance
from common.llmperf.utils import common_metrics
from common.llmperf.utils.models import RequestConfig

llm_conn = config_instance.get_config("llm_connection")
stream = llm_conn["stream"]
ignore_eos = llm_conn["ignore_eos"]
timeout = llm_conn["timeout"]


class OpenAIChatCompletionsClient:
    """
    used for sending HTTP requests, receiving token streams, measuring latency, etc.
    """

    def llm_request(
        self, request_config: RequestConfig
    ) -> Tuple[Dict[str, Any], str, RequestConfig]:
        prompt, prompt_len = request_config.prompt

        message = [
            {"role": "user", "content": prompt},
        ]
        model = request_config.model
        body = {
            "model": model,
            "messages": message,
            "stream": stream,
            "ignore_eos": ignore_eos,
        }
        sampling_params = request_config.sampling_params
        body.update(sampling_params or {})

        time_to_next_token = []
        tokens_received = 0
        ttft = 0.0
        error_response_code = None
        generated_text = ""
        error_msg = ""
        output_throughput = 0.0
        total_request_time = 0.0
        flag = False

        metrics: Dict[str, Any] = {}

        metrics[common_metrics.ERROR_CODE] = None
        metrics[common_metrics.ERROR_MSG] = ""

        start_time = time.monotonic()
        most_recent_received_token_time = start_time

        address = request_config.openai_api_base

        if not address:
            raise ValueError("the environment variable OPENAI_API_BASE must be set.")
        key = os.environ.get("OPENAI_API_KEY", "secret_abcdefg")
        if not key:
            raise ValueError("the environment variable OPENAI_API_KEY must be set.")
        headers = {"Authorization": f"Bearer {key}"}
        if not address.endswith("/"):
            address = address + "/"
        address += "chat/completions"
        try:
            with requests.post(
                address,
                json=body,
                stream=stream,
                timeout=timeout,
                headers=headers,
            ) as response:
                if response.status_code != 200:
                    error_msg = response.text
                    error_response_code = response.status_code
                    response.raise_for_status()

                for chunk in response.iter_lines(chunk_size=None):
                    if not chunk:
                        continue
                    stem = b"data: "
                    if chunk.startswith(stem):
                        chunk = chunk[len(stem) :]
                    # Data might already be bytes or str
                    if isinstance(chunk, bytes):
                        chunk = chunk.decode("utf-8", errors="ignore")
                    if chunk.strip() == "[DONE]":
                        continue
                    tokens_received += 1
                    data = json.loads(chunk)
                    if "error" in data:
                        error_msg = data["error"]["message"]
                        error_response_code = data["error"]["code"]
                        raise RuntimeError(error_msg)
                    delta = data["choices"][0]["delta"]
                    content = delta.get("content", None) or delta.get(
                        "reasoning_content", ""
                    )
                    if content:
                        if tokens_received != 0 and flag == False:
                            ttft = time.monotonic() - start_time
                            flag = True
                        else:
                            time_to_next_token.append(
                                time.monotonic() - most_recent_received_token_time
                            )
                        most_recent_received_token_time = time.monotonic()
                        generated_text += content

            total_request_time = time.monotonic() - start_time
            if total_request_time > 0:
                output_throughput = tokens_received / total_request_time

        except Exception as e:
            metrics[common_metrics.ERROR_MSG] = error_msg
            metrics[common_metrics.ERROR_CODE] = error_response_code
            print(f"Warning Or Error: {e}")
            print(error_response_code)

        metrics[common_metrics.INTER_TOKEN_LAT] = sum(time_to_next_token)
        metrics[common_metrics.TTFT] = ttft
        metrics[common_metrics.E2E_LAT] = total_request_time
        metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = output_throughput
        metrics[common_metrics.NUM_TOTAL_TOKENS] = tokens_received + prompt_len
        metrics[common_metrics.NUM_OUTPUT_TOKENS] = tokens_received
        metrics[common_metrics.NUM_INPUT_TOKENS] = prompt_len

        return metrics, generated_text, request_config
