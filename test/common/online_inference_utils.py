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
from typing import Any, Dict, List, Optional

import requests

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
