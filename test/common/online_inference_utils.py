"""
Online Inference Utilities for E2E Tests.

This module provides utilities for testing online inference with UCM (Unified Cache Management).
Unlike offline inference which loads the model directly, online inference connects to a running
inference server via OpenAI-compatible API.

USAGE EXAMPLE:
    # Create an online inference client
    client = OnlineInferenceClient(
        server_url="http://localhost:8000",
        model_name="Qwen2.5-1.5B-Instruct",
        tokenizer_path="/path/to/tokenizer"
    )

    # Send a request
    response = client.chat("What is the capital of France?")

    # Clear HBM cache
    client.clear_hbm()

    # Use VLLMServerManager to manage vLLM server lifecycle
    with VLLMServerManager(
        model_path="/home/models/Qwen2.5-1.5B-Instruct",
        port=8000,
        ucm_config={"storage_backends": "/tmp/ucm_cache"},
    ) as server:
        client = OnlineInferenceClient(
            server_url=server.url,
            model_name="Qwen2.5-1.5B-Instruct",
            tokenizer_path="/home/models/Qwen2.5-1.5B-Instruct",
        )
        response = client.chat([{"role": "user", "content": "Hello"}])
"""

import json
import logging
import os
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class OnlineInferenceConfig:
    """Configuration for online inference client."""

    server_url: str = "http://localhost:8000"
    model_name: str = "default"
    tokenizer_path: str = ""
    api_key: str = ""
    timeout: float = 600.0
    enable_prefix_caching: bool = False


@dataclass
class ChatResponse:
    """Response from a chat completion request."""

    text: str
    finish_reason: str
    total_tokens: int
    request_id: str = ""
    prefill_latency: float = 0.0
    decode_latency: float = 0.0


class OnlineInferenceClient:
    """
    Client for online inference via OpenAI-compatible API.

    This client connects to a running inference server (vLLM, sglang, etc.)
    and sends requests via the /v1/chat/completions endpoint.
    """

    def __init__(
        self,
        server_url: str,
        model_name: str,
        tokenizer_path: str,
        api_key: str = "",
        timeout: float = 600.0,
    ):
        """Initialize the online inference client.

        Args:
            server_url: Base URL of the inference server (e.g., "http://localhost:8000")
            model_name: Name of the model to use
            tokenizer_path: Path to the tokenizer for token counting
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
        """
        self.server_url = server_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.timeout = timeout

        # API endpoints
        self.chat_url = f"{self.server_url}/v1/chat/completions"
        self.reset_cache_url = f"{self.server_url}/reset_prefix_cache"

        # Load tokenizer
        if tokenizer_path and os.path.exists(tokenizer_path):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            self.tokenizer = None
            logger.warning(f"Tokenizer path not found: {tokenizer_path}")

        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "User-Agent": "UCM-Online-Inference-Client",
            }
        )
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stream: bool = False,
        **kwargs,
    ) -> ChatResponse:
        """Send a chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stream: Whether to stream the response
            **kwargs: Additional parameters to pass to the API

        Returns:
            ChatResponse with the generated text
        """
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }
        payload.update(kwargs)

        if stream:
            return self._stream_chat(payload)
        else:
            return self._sync_chat(payload)

    def _sync_chat(self, payload: Dict[str, Any]) -> ChatResponse:
        """Send a synchronous chat request."""
        start_time = time.time()

        response = self.session.post(
            self.chat_url,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()

        data = response.json()
        choice = data["choices"][0]
        text = choice["message"]["content"]
        finish_reason = choice.get("finish_reason", "stop")
        request_id = data.get("id", "")

        # Count tokens
        total_tokens = 0
        if self.tokenizer:
            total_tokens = len(self.tokenizer.encode(text))

        latency = time.time() - start_time

        return ChatResponse(
            text=text,
            finish_reason=finish_reason,
            total_tokens=total_tokens,
            request_id=request_id,
            decode_latency=latency,
        )

    def _stream_chat(self, payload: Dict[str, Any]) -> ChatResponse:
        """Send a streaming chat request and collect the full response."""
        payload["stream"] = True

        start_time = time.time()
        first_token_time = None
        all_text = ""
        finish_reason = "stop"
        request_id = ""

        with self.session.post(
            self.chat_url,
            json=payload,
            timeout=self.timeout,
            stream=True,
        ) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue

                line = line.decode("utf-8")
                if not line.startswith("data:"):
                    continue

                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                    request_id = data.get("id", request_id)

                    choice = data.get("choices", [{}])[0]
                    delta = choice.get("delta", {})
                    content = delta.get("content", "")
                    finish_reason = choice.get("finish_reason") or finish_reason

                    if content:
                        if first_token_time is None:
                            first_token_time = time.time()
                        all_text += content
                except json.JSONDecodeError:
                    continue

        end_time = time.time()
        prefill_latency = (first_token_time - start_time) if first_token_time else 0
        decode_latency = end_time - (first_token_time or start_time)

        total_tokens = 0
        if self.tokenizer:
            total_tokens = len(self.tokenizer.encode(all_text))

        return ChatResponse(
            text=all_text,
            finish_reason=finish_reason,
            total_tokens=total_tokens,
            request_id=request_id,
            prefill_latency=prefill_latency,
            decode_latency=decode_latency,
        )

    def clear_hbm(self) -> bool:
        """Clear the HBM prefix cache on the server.

        This calls the /reset_prefix_cache endpoint on vLLM servers.

        Returns:
            True if successful, False otherwise
        """
        try:
            response = self.session.post(
                self.reset_cache_url,
                json={},
                timeout=30,
            )
            if response.status_code == 200:
                logger.info("HBM cache cleared successfully")
                return True
            else:
                logger.warning(f"Failed to clear HBM cache: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error clearing HBM cache: {e}")
            return False

    def health_check(self) -> bool:
        """Check if the server is healthy.

        Returns:
            True if server is responding, False otherwise
        """
        try:
            response = self.session.get(
                f"{self.server_url}/health",
                timeout=10,
            )
            return response.status_code == 200
        except Exception:
            return False

    def close(self):
        """Close the client session."""
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def run_online_inference(
    server_url: str,
    model_name: str,
    tokenizer_path: str,
    prompts: List[str],
    max_tokens: int = 256,
    temperature: float = 0.0,
    stream: bool = False,
    clear_hbm_before: bool = False,
    system_prompt: Optional[str] = None,
    **kwargs,
) -> List[str]:
    """Run online inference on a list of prompts.

    This is a convenience function for running multiple prompts sequentially.

    Args:
        server_url: Base URL of the inference server
        model_name: Name of the model
        tokenizer_path: Path to the tokenizer
        prompts: List of prompts to send
        max_tokens: Maximum tokens to generate per prompt
        temperature: Sampling temperature
        stream: Whether to stream responses
        clear_hbm_before: Whether to clear HBM cache before inference
        system_prompt: Optional system prompt to prepend
        **kwargs: Additional parameters for the chat API

    Returns:
        List of generated text strings
    """
    with OnlineInferenceClient(
        server_url=server_url,
        model_name=model_name,
        tokenizer_path=tokenizer_path,
    ) as client:
        if clear_hbm_before:
            client.clear_hbm()
            time.sleep(2)  # Wait for cache to clear

        results = []
        for prompt in prompts:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = client.chat(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream,
                **kwargs,
            )
            results.append(response.text)

        return results


def load_prompt_from_file(prompt_file: str) -> Tuple[str, List[str]]:
    """Load prompt and answers from JSON file (LongBench format).

    This is a re-export from offline_inference_utils for convenience.

    Args:
        prompt_file: Path to the prompt JSON file

    Returns:
        Tuple of (combined_prompt_string, answers_list)
    """
    from common.offline_inference_utils import load_prompt_from_file as _load

    return _load(prompt_file)


def split_prompt_by_tokens(
    prompt: str, tokenizer: AutoTokenizer, split_ratio: float = 0.5
) -> Tuple[str, str]:
    """Split a prompt into two parts by token ratio.

    This is a re-export from offline_inference_utils for convenience.

    Args:
        prompt: The prompt to split
        tokenizer: Tokenizer to use for splitting
        split_ratio: Ratio to split (0.5 = split in half)

    Returns:
        Tuple of (first_part, second_part)
    """
    from common.offline_inference_utils import split_prompt_by_tokens as _split

    return _split(prompt, tokenizer, split_ratio)


class VLLMServerManager:
    """
    Manages vLLM server lifecycle for testing.

    This class handles starting and stopping a vLLM server with UCM configuration,
    making it easy to run online inference tests that require a live server.

    Example:
        with VLLMServerManager(
            model_path="/home/models/Qwen2.5-1.5B-Instruct",
            port=8000,
            ucm_config={"storage_backends": "/tmp/ucm_cache"},
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
        gpu_memory_utilization: float = 0.9,
        additional_args: Optional[List[str]] = None,
        startup_timeout: float = 300.0,
    ):
        """Initialize the VLLMServerManager.

        Args:
            model_path: Path to the model weights
            port: Port to run the server on
            host: Host address to bind to
            ucm_config: UCM connector configuration dict with keys:
                - storage_backends: Path or list of paths for SSD cache storage
                - ucm_connector_name: Connector name (default: "UcmNfsStore")
                - Additional connector-specific config
            enable_prefix_caching: Whether to enable vLLM prefix caching (HBM cache)
            max_model_len: Maximum model context length
            gpu_memory_utilization: GPU memory utilization fraction
            additional_args: Additional arguments to pass to vllm serve
            startup_timeout: Timeout in seconds for server startup
        """
        self.model_path = model_path
        self.port = port
        self.host = host
        self.ucm_config = ucm_config or {}
        self.enable_prefix_caching = enable_prefix_caching
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.additional_args = additional_args or []
        self.startup_timeout = startup_timeout

        self._process: Optional[subprocess.Popen] = None
        self._url = f"http://{host}:{port}"

    @property
    def url(self) -> str:
        """Get the server URL."""
        return self._url

    def _build_kv_transfer_config(self) -> Dict[str, Any]:
        """Build the kv-transfer-config for UCM."""
        storage_backends = self.ucm_config.get("storage_backends", "/tmp/ucm_cache")
        if isinstance(storage_backends, str):
            storage_backends = [storage_backends]

        connector_name = self.ucm_config.get("ucm_connector_name", "UcmNfsStore")

        kv_config = {
            "kv_connector": "UCMConnector",
            "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "ucm_connectors": [
                    {
                        "ucm_connector_name": connector_name,
                        "ucm_connector_config": {"storage_backends": storage_backends},
                    }
                ]
            },
        }

        # Add any extra UCM config
        extra_config = self.ucm_config.get("extra_config", {})
        if extra_config:
            kv_config["kv_connector_extra_config"]["ucm_connectors"][0][
                "ucm_connector_config"
            ].update(extra_config)

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

        # Start the process
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
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
                stdout, stderr = self._process.communicate()
                raise RuntimeError(
                    f"vLLM server process exited unexpectedly with code {self._process.returncode}\n"
                    f"stdout: {stdout}\n"
                    f"stderr: {stderr}"
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
