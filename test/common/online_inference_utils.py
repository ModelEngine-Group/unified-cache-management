"""
Online Inference Utilities for E2E Tests.

This module provides utilities for testing online inference with UCM (Unified Cache Management).
Unlike offline inference which loads the model directly, online inference connects to a running
inference server via OpenAI-compatible API.

USAGE EXAMPLE:
    # Use VLLMServerManager to manage vLLM server lifecycle
    from common.llm_connection.openai_connector import OpenAIConn
    from common.llm_connection.token_counter import HuggingFaceTokenizer
    from common.llm_connection.LLMBase import LLMRequest

    with VLLMServerManager(
        model_path="/home/models/Qwen2.5-1.5B-Instruct",
        port=8000,
        ucm_config={
            "ucm_connectors": [
                {
                    "ucm_connector_name": "UcmNfsStore",
                    "ucm_connector_config": {"storage_backends": ["/tmp/ucm_cache"]}
                }
            ]
        },
    ) as server:
        tokenizer = HuggingFaceTokenizer("/home/models/Qwen2.5-1.5B-Instruct")
        client = OpenAIConn(
            base_url=server.url,
            tokenizer=tokenizer,
            model="Qwen2.5-1.5B-Instruct",
        )
        req = LLMRequest(messages=[{"role": "user", "content": "Hello"}], max_tokens=100)
        response = client.chat(req)
        print(response.text)
"""

import json
import logging
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import requests
from common.common_inference_utils import (
    match_any_answer,
)
from common.llm_connection.LLMBase import LLMRequest, LLMResponse

logger = logging.getLogger(__name__)


class VLLMServerManager:
    """
    Manages vLLM server lifecycle for testing.

    This class handles starting and stopping a vLLM server with UCM configuration,
    making it easy to run online inference tests that require a live server.

    Example:
        with VLLMServerManager(
            model_path="/home/models/Qwen2.5-1.5B-Instruct",
            port=8000,
            ucm_config={
                "ucm_connectors": [
                    {
                        "ucm_connector_name": "UcmNfsStore",
                        "ucm_connector_config": {"storage_backends": ["/tmp/ucm_cache"]}
                    }
                ]
            },
        ) as server:
            client = OnlineInferenceClient(
                server_url=server.url,
                model_name="Qwen2.5-1.5B-Instruct",
                tokenizer_path="/home/models/Qwen2.5-1.5B-Instruct",
            )
            response = client.chat([{"role": "user", "content": "Hello"}])
    """

    def __init__(
        self,
        model_path: str,
        port: int = 8000,
        host: str = "0.0.0.0",
        ucm_config: Optional[Dict[str, Any]] = None,
        enable_prefix_caching: bool = False,
        max_model_len: int = 12000,
        max_num_batched_tokens: Optional[int] = None,
        additional_args: Optional[List[str]] = None,
        startup_timeout: float = 300.0,
        served_model_name: str = "",
        pipeline_parallel_size: int = 1,
    ):
        """Initialize the VLLMServerManager.

        Args:
            model_path: Path to the model weights
            port: Port to run the server on
            host: Host address to bind to
            ucm_config: UCM connector configuration dict. Should include:
                - ucm_connectors: List of connector configurations
                  Example: [{"ucm_connector_name": "UcmNfsStore",
                             "ucm_connector_config": {"storage_backends": ["/tmp/ucm_cache"]}}]
            enable_prefix_caching: Whether to enable vLLM prefix caching (HBM cache)
            max_model_len: Maximum model context length
            max_num_batched_tokens: Maximum number of batched tokens (default: 2047)
            additional_args: Additional arguments to pass to vllm serve
            startup_timeout: Timeout in seconds for server startup
            served_model_name: Optional model name to expose via the API (defaults to model_path)
            pipeline_parallel_size: Pipeline parallel size (default: 1)
        """

        gpu_memory_utilization = float(
            os.getenv("E2E_TEST_GPU_MEMORY_UTILIZATION", "0.1")
        )
        logging.info(
            "run offline inference with gpu memory utilization: %.4f",
            gpu_memory_utilization,
        )

        self.model_path = model_path
        self.port = port
        self.host = host
        self.ucm_config = ucm_config or {}
        self.enable_prefix_caching = enable_prefix_caching
        self.max_model_len = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens
        self.gpu_memory_utilization = gpu_memory_utilization
        self.additional_args = additional_args or []
        self.startup_timeout = startup_timeout
        self.served_model_name = served_model_name
        self.pipeline_parallel_size = pipeline_parallel_size

        self._process: Optional[subprocess.Popen] = None
        self._url = f"http://{host}:{port}"

    @property
    def url(self) -> str:
        """Get the server URL."""
        return self._url

    def _build_kv_transfer_config(self) -> Dict[str, Any]:
        """Build the kv-transfer-config for UCM.

        The full ucm_config is passed as kv_connector_extra_config (mirrors offline
        inference), including ucm_connectors, use_layerwise, enable_event_sync, etc.
        """
        kv_config = {
            "kv_connector": "UCMConnector",
            "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": self.ucm_config,
        }
        return kv_config

    def _build_command(self) -> List[str]:
        """Build the vllm serve command."""
        cmd = [
            "vllm",
            "serve",
            self.model_path,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--max-model-len",
            str(self.max_model_len),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--pipeline-parallel-size",
            str(self.pipeline_parallel_size),
        ]

        # Add UCM kv-transfer-config
        if self.ucm_config:
            kv_config = self._build_kv_transfer_config()
            cmd.extend(["--kv-transfer-config", json.dumps(kv_config)])

        # Add prefix caching if enabled
        if self.enable_prefix_caching:
            cmd.append("--enable-prefix-caching")

        # Add max_num_batched_tokens if specified
        if self.max_num_batched_tokens is not None:
            cmd.extend(["--max-num-batched-tokens", str(self.max_num_batched_tokens)])

        # Add served model name if specified
        if self.served_model_name:
            cmd.extend(["--served-model-name", self.served_model_name])

        # Add additional arguments
        cmd.extend(self.additional_args)

        return cmd

    def start(self) -> None:
        """Start the vLLM server."""
        if self._process is not None:
            raise RuntimeError("Server is already running")

        cmd = self._build_command()
        cmd_str = " ".join(shlex.quote(arg) for arg in cmd)
        logger.info(f"Starting vLLM server: {cmd_str}")

        # Start the process with stdout/stderr redirected to current stdout
        self._process = subprocess.Popen(
            cmd,
            stdout=sys.stdout,
            stderr=sys.stdout,
            text=True,
            bufsize=1,  # Line buffered
        )

        logger.info(f"vLLM server started with PID {self._process.pid}")

    def stop(self) -> None:
        """Stop the vLLM server."""
        if self._process is None:
            return

        logger.info(f"Stopping vLLM server (PID {self._process.pid})")

        try:
            # Try graceful shutdown first
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
                logger.info("vLLM server stopped gracefully")
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown fails
                logger.warning("vLLM server did not stop gracefully, forcing...")
                self._process.kill()
                self._process.wait(timeout=5)
                logger.info("vLLM server killed")
        except Exception as e:
            logger.error(f"Error stopping vLLM server: {e}")
        finally:
            self._process = None

    def wait_for_ready(self, timeout: Optional[float] = None) -> bool:
        """Wait for the server to be ready.

        Args:
            timeout: Maximum time to wait in seconds (default: self.startup_timeout)

        Returns:
            True if server is ready, False if timeout

        Raises:
            RuntimeError: If the server process exits unexpectedly
        """
        if timeout is None:
            timeout = self.startup_timeout

        if self._process is None:
            raise RuntimeError("Server process not started")

        start_time = time.time()
        health_url = f"{self._url}/health"

        logger.info(f"Waiting for vLLM server to be ready at {health_url}")

        while time.time() - start_time < timeout:
            # Check if process is still alive
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"vLLM server process exited unexpectedly with code {self._process.returncode}"
                )

            # Try to connect
            try:
                response = requests.get(health_url, timeout=5)
                if response.status_code == 200:
                    logger.info(
                        f"vLLM server is ready after {time.time() - start_time:.1f}s"
                    )
                    return True
            except requests.exceptions.RequestException:
                pass

            time.sleep(1)

        raise TimeoutError(f"vLLM server did not become ready within {timeout}s")

    def __enter__(self) -> "VLLMServerManager":
        """Context manager entry."""
        self.start()
        self.wait_for_ready()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()


def batch_chat(
    client,
    requests: List[LLMRequest],
    max_workers: Optional[int] = None,
) -> List[LLMResponse]:
    """Send multiple requests to the LLM server in parallel and return all responses.

    This function sends multiple requests concurrently using a thread pool,
    waits for all requests to complete, and returns the results in the same order as the input requests.

    Args:
        client: An LLM client that implements the LLMConnection protocol (e.g., OpenAIConn)
        requests: List of LLMRequest objects to send
        max_workers: Maximum number of worker threads (default: number of requests)

    Returns:
        List of LLMResponse objects in the same order as input requests

    Example:
        from common.llm_connection.openai_connector import OpenAIConn
        from common.llm_connection.LLMRequest import LLMRequest

        client = OpenAIConn(base_url="http://localhost:8000", model="qwen")
        requests = [
            LLMRequest(messages=[{"role": "user", "content": "Hello"}], max_tokens=100),
            LLMRequest(messages=[{"role": "user", "content": "Hi"}], max_tokens=100),
        ]
        responses = batch_chat(client, requests)
        for resp in responses:
            print(resp.text)
    """
    if not requests:
        return []

    if max_workers is None:
        max_workers = len(requests)

    results: List[Optional[LLMResponse]] = [None] * len(requests)

    def _send_request(index: int, request: LLMRequest) -> tuple[int, LLMResponse]:
        """Send a single request and return the index with response."""
        response = client.chat(request)
        return index, response

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all requests
        future_to_index = {
            executor.submit(_send_request, i, req): i for i, req in enumerate(requests)
        }

        # Collect results as they complete
        for future in as_completed(future_to_index):
            index, response = future.result()
            results[index] = response

    # Ensure all results are filled (should not happen if no exception)
    return [resp for resp in results if resp is not None]


def hbm_ssd_mixed_test(
    model_name: str,
    tokenizer_path: str,
    max_tokens: int,
    prompt_split_ratio: float,
    ucm_config: Dict[str, Any],
    vllm_server_startup_args: Dict[str, Any],
) -> None:
    """Test HBM + SSD mixed hit accuracy via online inference.

    This function implements  2-phasekt test flow:
    1. Phase: Start vLLM (prefix caching OFF), send full prompt twice
       -> KV cache saved to SSD, then loaded from SSD
    2. Phase: Start vLLM (prefix caching ON), send partial prompt (warm HBM),
       then send full prompt (hits both HBM and SSD) -> verify accuracy

    Args:
        model_name: Name of model (used for served_model_name and model_path lookup)
        tokenizer_path: Path to tokenizer for prompt processing
        max_tokens: Maximum tokens to generate
        prompt_split_ratio: Ratio to split prompt for Phase (0.5 = split in half)
        ucm_config: UCM connector configuration
        vllm_server_startup_args: All kwargs for VLLMServerManager initialization
    """
    import os
    from typing import List

    import pytest
    import yaml
    from common.common_inference_utils import (
        ensure_storage_dir,
        load_prompt_from_file,
        split_prompt_by_tokens,
    )
    from common.llm_connection.LLMBase import LLMRequest
    from common.llm_connection.openai_connector import OpenAIConn
    from common.llm_connection.token_counter import HuggingFaceTokenizer
    from common.path_utils import get_path_relative_to_test_root, get_path_to_model

    # Load configuration
    config_file = get_path_relative_to_test_root("config.yaml")
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ucm_storage_dir = "/tmp/ucm_cache"
    ensure_storage_dir(ucm_storage_dir, clear_existing=True)

    served_model_name = model_name
    model_path = get_path_to_model(model_name, config)

    # Load test prompt and standard answers
    test_prompt, standard_answers = load_prompt_from_file(
        get_path_relative_to_test_root("suites/E2E/prompts/test_offline_inference.json")
    )
    if not standard_answers:
        pytest.fail("No standard answers found in prompt.json")

    print(f"Standard answers: {standard_answers}")

    # Initialize tokenizer for prompt splitting
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_chat_template=True)

    # Split prompt for Phase
    prompt_first_part, _ = split_prompt_by_tokens(
        test_prompt, tokenizer, split_ratio=prompt_split_ratio
    )

    # Prepare messages
    system_content = "先读问题，再根据下面的文章内容回答问题，不要进行分析，不要重复问题，用简短的语句给出答案。\n\n例如：\u201c全国美国文学研究会的第十八届年会在哪所大学举办的？\u201d\n回答应该为：\u201cxx大学\u201d。\n\n"

    phase1_messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": test_prompt},
    ]
    phase2_partial_messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt_first_part},
    ]

    print(f"\n===== Online HBM + SSD Mixed Accuracy Test =====")
    print(f"Model: {model_path}")
    print(f"Full prompt length: {len(test_prompt)} chars")
    print(f"Max tokens: {max_tokens}")
    print(f"Prompt split ratio: {prompt_split_ratio}")

    # ===== Phase: Disable HBM PC, save KV cache to SSD and load (baseline) =====
    print(f"\n===== Phase: Save KV Cache to SSD And Load (Baseline) =====")
    print(f"Starting vLLM server with enable_prefix_caching=False")

    with VLLMServerManager(
        **vllm_server_startup_args,
        enable_prefix_caching=False,
    ) as server:
        client = OpenAIConn(
            base_url=server.url,
            tokenizer=HuggingFaceTokenizer(tokenizer_path),
            model=served_model_name,
        )
        assert client.health_check()

        print(f"server models: {client.list_models()}")

        # Phase.1: Send full prompt -> KV cache saved to SSD
        phase1_1_output = client.chat(
            LLMRequest(messages=phase1_messages, max_tokens=max_tokens, temperature=0.0)
        ).text
        print(f'Phase.1 output: "{phase1_1_output}"')

        # Phase.2: Send same prompt again -> KV cache loaded from SSD
        phase1_2_output = client.chat(
            LLMRequest(messages=phase1_messages, max_tokens=max_tokens, temperature=0.0)
        ).text
        print(f'Phase.2 output: "{phase1_2_output}"')
        client.close()

    print("Phase vLLM server stopped.")

    # ===== Phase: Enable HBM PC, test HBM + SSD mixed hit =====
    print(f"\n===== Phase: HBM + SSD Mixed Hit Test =====")
    print(f"Starting vLLM server with enable_prefix_caching=True")

    with VLLMServerManager(
        **vllm_server_startup_args,
        enable_prefix_caching=True,
    ) as server:
        client = OpenAIConn(
            base_url=server.url,
            tokenizer=HuggingFaceTokenizer(tokenizer_path),
            model=served_model_name,
        )
        assert client.health_check()

        print(f"server models: {client.list_models()}")

        # Phase.1: Send partial prompt -> warm HBM prefix cache
        phase2_partial_output = client.chat(
            LLMRequest(
                messages=phase2_partial_messages,
                max_tokens=max_tokens,
                temperature=0.0,
            )
        ).text
        print(f"[INFO] Phase.1 output (partial prompt): {phase2_partial_output}")

        # Phase.2: Send full prompt -> hits HBM (prefix) + SSD (suffix)
        phase2_full_output = client.chat(
            LLMRequest(messages=phase1_messages, max_tokens=max_tokens, temperature=0.0)
        ).text
        print(f"[INFO] Phase.2 output (full prompt): {phase2_full_output}")
        client.close()

    print("Phase vLLM server stopped.")

    # ===== Accuracy Test Results =====
    print(f"\n[INFO] ===== Accuracy Test Results =====")

    # Phase accuracy check
    phase1_correct = match_any_answer(
        phase1_1_output, standard_answers
    ) and match_any_answer(phase1_2_output, standard_answers)
    if not phase1_correct:
        print(f"\n===== Phase: SSD Load Accuracy Test (Exact Match) =====")
        print(f"Incorrect answer in Phase.1 (SSD save) or Phase.2 (SSD load)!")
        print(f"Phase.1 output:\n{phase1_1_output}")
        print(f"Phase.2 output:\n{phase1_2_output}")
        print(f"Standard answers:\n{standard_answers}")
        pytest.fail("SSD Load Accuracy Test Failed!")

    # Phase accuracy check
    phase2_correct = match_any_answer(phase2_full_output, standard_answers)
    if not phase2_correct:
        print(f"\n===== Phase: HBM + SSD Mixed Accuracy Test (Exact Match) =====")
        print(f"Incorrect answer in Phase.2 (HBM + SSD mixed)!")
        print(f"Phase.2 output:\n{phase2_full_output}")
        print(f"Standard answers:\n{standard_answers}")
        pytest.fail("HBM + SSD Mixed Accuracy Test Failed!")

    print("\n===== All Tests Passed! =====")
