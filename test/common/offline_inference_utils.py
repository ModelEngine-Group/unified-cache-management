"""
MULTIPROCESS FRAMEWORK:
======================
This module provides a `run_in_spawn_subprocess` function to simplify running functions in
subprocess while handling GPU memory cleanup automatically.

NOTE: Each offline inference test case should run with multiprocessing spawn mode to ensure GPU memory
is fully released after each test. This prevents memory accumulation across test cases.

USAGE EXAMPLE:
    # Define your test function that contains the core test logic
    def my_test_logic(model_path, config, params):
        # Your test logic here - no need to handle multiprocessing or GPU cleanup
        with build_llm_with_uc(model_path, config) as llm:
            results = llm.generate(...)
        return results

    # Run it in subprocess using the framework
    results = run_in_spawn_subprocess(
        my_test_logic,
        model_path,
        config,
        params,
        timeout=180  # optional, default 180 seconds
    )
"""

import contextlib
import gc
import json
import multiprocessing
import os
import time
from dataclasses import asdict
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from common.capture_utils import export_vars
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.engine.arg_utils import EngineArgs

from ucm.logger import init_logger

_test_functions = {}
logger = init_logger(__name__)


def run_in_spawn_subprocess(func, *args, timeout: int = 180, **kwargs):
    """Run a function in a subprocess.

    Args:
        func: The function to run in subprocess
        *args: Positional arguments to pass to func
        timeout: Timeout in seconds (default 180)
        **kwargs: Keyword arguments to pass to func

    Returns:
        The return value from func

    Raises:
        RuntimeError: If subprocess times out or fails
        Exception: Any exception raised by func in the subprocess
    """

    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    error_queue = ctx.Queue()

    def _wrapper(result_queue, error_queue):
        try:
            result = func(*args, **kwargs)
            result_queue.put(result)
        except Exception as e:
            import traceback

            error_info = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            error_queue.put(RuntimeError(error_info))

    process = ctx.Process(target=_wrapper, args=(result_queue, error_queue))
    process.start()
    process.join(timeout=timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        raise RuntimeError(f"Subprocess timed out after {timeout} seconds")

    if not error_queue.empty():
        error = error_queue.get()
        raise error

    if not result_queue.empty():
        return result_queue.get()

    if process.exitcode != 0:
        raise RuntimeError(f"Subprocess failed with exit code {process.exitcode}")


def cleanup_gpu_memory():
    """Clean up GPU/NPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.empty_cache()
        torch.npu.synchronize()
    gc.collect()


@contextlib.contextmanager
def build_llm_with_uc(
    model_path: str,
    ucm_config: Optional[Dict[str, Any]] = None,
    enable_prefix_caching: bool = False,
    max_num_batched_tokens: int = 2047,
    **llm_kwargs,
):
    module_path = "ucm.integration.vllm.ucm_connector"
    name = "UCMConnector"

    ktc = KVTransferConfig(
        kv_connector=name,
        kv_connector_module_path=module_path,
        kv_role="kv_both",
        kv_connector_extra_config=ucm_config,
    )

    tensor_parallel_size = 1

    default_args = {
        "model": model_path,
        "kv_transfer_config": ktc,
        "max_model_len": 12000,
        "gpu_memory_utilization": 0.3,  # Reduced to prevent OOM after Phase 1
        "max_num_batched_tokens": max_num_batched_tokens,
        "block_size": 128,
        "enforce_eager": llm_kwargs.get("enforce_eager", True),
        "trust_remote_code": True,
        "enable_prefix_caching": enable_prefix_caching,
        "tensor_parallel_size": tensor_parallel_size,
    }
    default_args.update(llm_kwargs)

    cleanup_gpu_memory()
    time.sleep(1)  # Ensure memory is released before building LLM

    llm_args = EngineArgs(**default_args)
    llm = LLM(**asdict(llm_args))

    try:
        yield llm
    finally:
        logger.info("LLM engine is exiting")
        del llm
        cleanup_dist_env_and_memory(shutdown_ray=False)


def run_offline_inference(
    model_path: str,
    ucm_config: Dict[str, Any],
    prompts: List[str],
    sampling_params_dict: Dict[str, Any],
    enable_prefix_caching: bool,
    enforce_eager: bool,
    phase_description: str,
    max_num_batched_tokens: int,
) -> List[str]:
    """Run a phase in the subprocess.

    This function should be called via MultiprocessSpawner.run_in_subprocess().
    It handles the actual test logic without subprocess management.

    Args:
        model_path: Path to the model
        ucm_config: UCM configuration
        prompts: List of prompts to send
        sampling_params_dict: Sampling parameters as dict (for serialization)
        enable_prefix_caching: Whether to enable HBM prefix caching
        enforce_eager: Whether to enforce eager mode
        phase_description: Description string for logging
        max_num_batched_tokens: Max number of batched tokens

    Returns:
        List of generated outputs
    """
    sampling_params = SamplingParams(**sampling_params_dict)

    with build_llm_with_uc(
        model_path=model_path,
        ucm_config=ucm_config,
        enable_prefix_caching=enable_prefix_caching,
        gpu_memory_utilization=0.3,
        max_num_batched_tokens=max_num_batched_tokens,
        enforce_eager=enforce_eager,
    ) as llm:
        outputs = llm.generate(prompts, sampling_params)

        generated_texts = [output.outputs[0].text for output in outputs]

        if phase_description:
            logger.info(f"{phase_description} completed")

        return generated_texts


def split_prompt_by_tokens(
    prompt: str, tokenizer: AutoTokenizer, split_ratio: float = 0.5
) -> Tuple[str, str]:
    tokens = tokenizer.encode(prompt)
    split_idx = int(len(tokens) * split_ratio)

    first_tokens = tokens[:split_idx]
    second_tokens = tokens[split_idx:]

    first_part = tokenizer.decode(first_tokens, skip_special_tokens=False)
    second_part = tokenizer.decode(second_tokens, skip_special_tokens=False)

    return first_part, second_part


def load_prompt_from_file(prompt_file: Optional[Path] = None) -> Tuple[str, List[str]]:
    """Load prompt and answers from JSON file (LongBench format).
    LongBench format structure:
    {
        "input": "任务输入/问题",
        "context": "长上下文/文档",
        "answers": ["答案列表"],
        "length": 总长度,
        "dataset": "数据集名称",
        "language": "语言",
        ...
    }
    For LongBench, the typical format is:
    - context: 长文档/上下文（放在前面）
    - input: 问题/查询（放在后面）
    - Combined format: context + "\n\n" + input
    Args:
        prompt_file: Path to the prompt JSON file. If None, uses default path.
    Returns:
        Tuple of (combined_prompt_string, answers_list).
        - combined_prompt_string: Combined prompt (context + input)
        - answers_list: List of standard answers from the file
    """
    if prompt_file is None:
        prompt_file = Path(__file__).parent / "prompt.json"

    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    with open(prompt_file, "r", encoding="utf-8") as f:
        content = f.read().strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {prompt_file}: {e}")

    if isinstance(data, list):
        if len(data) == 0:
            raise ValueError(f"Empty list in {prompt_file}")
        data = data[0]

    input_text = data.get("input", "")
    context_text = data.get("context", "")

    # LongBench standard format: context (long document) + input (question)
    # Combine context and input to form the full prompt
    if context_text and input_text:
        full_prompt = f"{context_text}\n\n{input_text}"
    elif context_text:
        full_prompt = context_text
    elif input_text:
        full_prompt = input_text
    else:
        raise ValueError(f"No input or context found in {prompt_file}")

    # Extract answers
    answers = data.get("answers", [])
    if not isinstance(answers, list):
        answers = [answers] if answers else []

    return full_prompt, answers
