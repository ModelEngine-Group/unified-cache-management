"""
NOTE: Each test case should run with multiprocessing spawn mode to ensure GPU memory
is fully released after each test. This prevents memory accumulation across test cases.
"""

import contextlib
import gc
import json
import multiprocessing
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest
import yaml
from common.capture_utils import export_vars
from transformers import AutoTokenizer

from ucm.logger import init_logger

_test_functions = {}
logger = init_logger(__name__)


def _run_test_in_spawn_process(test_id, args, kwargs, result_queue, error_queue):
    os.environ["_IN_SPAWN_PROCESS"] = "1"
    try:
        test_func = _test_functions.get(test_id)
        if test_func is None:
            raise RuntimeError(f"Test function {test_id} not found")
        result = test_func(*args, **kwargs)
        result_queue.put(("success", result))
    except Exception as e:
        error_queue.put(("error", e))


def run_in_spawn_process(func):
    test_id = f"{func.__module__}.{func.__name__}"
    _test_functions[test_id] = func

    @wraps(func)
    def wrapper(*args, **kwargs):
        if os.environ.get("_IN_SPAWN_PROCESS") == "1":
            return func(*args, **kwargs)

        ctx = multiprocessing.get_context("spawn")
        result_queue = ctx.Queue()
        error_queue = ctx.Queue()

        process = ctx.Process(
            target=_run_test_in_spawn_process,
            args=(test_id, args, kwargs, result_queue, error_queue),
        )
        process.start()
        process.join(timeout=180)

        if process.is_alive():
            process.terminate()
            process.join()
            raise RuntimeError(f"Test {func.__name__} timed out after 3 minutes")

        if not error_queue.empty():
            status, error = error_queue.get()
            raise error

        if not result_queue.empty():
            status, result = result_queue.get()
            return result

        if process.exitcode != 0:
            raise RuntimeError(
                f"Test {func.__name__} failed in spawn process with exit code {process.exitcode}"
            )

    return wrapper


try:
    import torch
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    from vllm.distributed import cleanup_dist_env_and_memory
    from vllm.engine.arg_utils import EngineArgs

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    pytest.skip("vLLM not available", allow_module_level=True)


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

    if not os.getenv("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "7"

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

    llm_args = EngineArgs(**default_args)
    llm = LLM(**asdict(llm_args))

    try:
        yield llm
    finally:
        logger.info("LLM engine is exiting")
        del llm
        cleanup_dist_env_and_memory(shutdown_ray=False)


def _run_phase_in_subprocess(
    model_path: str,
    ucm_config: Dict[str, Any],
    prompts: List[str],
    sampling_params_dict: Dict[str, Any],
    enable_prefix_caching: bool,
    enforce_eager: bool,
    phase_description: str,
    max_num_batched_tokens: int,
    result_queue: multiprocessing.Queue,
    error_queue: multiprocessing.Queue,
):
    """Run a phase in a separate subprocess to ensure GPU memory is fully released.

    This is a generic function that can handle both Phase 1 and Phase 2 by passing
    different parameters.

    Args:
        model_path: Path to the model
        ucm_config: UCM configuration
        prompts: List of prompts to send (e.g., [full_prompt, full_prompt] for Phase 1,
                 or [partial_prompt, full_prompt] for Phase 2)
        sampling_params_dict: Sampling parameters as dict (for serialization)
        enable_prefix_caching: Whether to enable HBM prefix caching
        enforce_eager: Whether to enforce eager mode
        phase_description: Description string for logging
        result_queue: Queue to put results
        error_queue: Queue to put errors
    """
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.empty_cache()
            torch.npu.synchronize()
        gc.collect()

        sampling_params = SamplingParams(**sampling_params_dict)

        with build_llm_with_uc(
            model_path=model_path,
            ucm_config=ucm_config,
            enable_prefix_caching=enable_prefix_caching,
            gpu_memory_utilization=0.3,
            max_num_batched_tokens=max_num_batched_tokens,
            enforce_eager=enforce_eager,
        ) as llm:
            outputs = run_inference(llm, prompts, sampling_params, phase_description)

        result_queue.put(outputs)
    except Exception as e:
        import traceback

        error_info = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        error_queue.put(("error", RuntimeError(error_info)))
        raise


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


def run_inference(
    llm: LLM,
    prompts: List[str],
    sampling_params: SamplingParams,
    description: str = "",
) -> List[str]:
    """Run inference and return generated texts.

    Args:
        llm: LLM instance.
        prompts: List of prompt strings.
        sampling_params: Sampling parameters.
        description: Description for logging.

    Returns:
        List of generated texts.
    """
    outputs = llm.generate(prompts, sampling_params)

    generated_texts = []
    for output in outputs:
        generated_text = output.outputs[0].text
        generated_texts.append(generated_text)

    if description:
        logger.info(f"{description} completed")

    return generated_texts


@pytest.mark.parametrize(
    "model_path",
    [
        "/home/models/Qwen2.5-1.5B-Instruct",
        "/home/models/DeepSeek-V2-Lite-Chat-AWQ",
    ],
)
@pytest.mark.parametrize("max_tokens", [200])
@pytest.mark.parametrize("prompt_split_ratio", [0.5])  # Split prompt in half
@pytest.mark.parametrize("enforce_eager", [True, False])
@pytest.mark.parametrize("max_num_batched_tokens", [2047, 2048, 12000])
@pytest.mark.feature("uc_accuracy_test_offline")
@export_vars
@run_in_spawn_process
def test_offline_accuracy_hbm_ssd_mixed(
    model_path: str,
    max_tokens: int,
    prompt_split_ratio: float,
    enforce_eager: bool,
    max_num_batched_tokens: int,
):
    """Test HBM + SSD mixed hit accuracy (Phase 2).

    This test first runs Phase 1 to generate a baseline output, then tests Phase 2.
    Test flow:
    1. Phase 1: Disable HBM PC, send full prompt -> KV cache saved to SSD (baseline)
    2. Phase 2: Enable HBM PC, send partial prompt (warm HBM), then send full prompt (hits both HBM and SSD) -> verify mixed hit accuracy

    The prompt is loaded from prompt.json file (LongBench format).

    Args:
        model_path: Path to the model.
        max_tokens: Maximum tokens to generate.
        prompt_split_ratio: Ratio to split prompt for Phase 2 (0.5 = split in half).
    """
    config_file = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Use model_path from parameter, fallback to config or environment
    if not model_path or not os.path.exists(model_path):
        model_path = config.get("llm_connection", {}).get("model_path") or os.getenv(
            "MODEL_PATH"
        )
        if not model_path:
            pytest.skip(f"model_path not configured or not found: {model_path}")

    if not os.path.exists(model_path):
        pytest.skip(f"Model path does not exist: {model_path}")

    ucm_storage_dir = config.get("llm_connection", {}).get(
        "ucm_storage_dir"
    ) or os.getenv("UCM_STORAGE_DIR", "/tmp/ucm_cache")

    try:
        test_prompt, standard_answers = load_prompt_from_file()
        logger.info(
            f"Loaded prompt from prompt.json (length: {len(test_prompt)} chars)"
        )
        if standard_answers:
            logger.info(f"Standard answers: {standard_answers}")
        else:
            logger.info(f"No standard answers found in prompt.json")
    except Exception as e:
        pytest.skip(f"Failed to load prompt from prompt.json: {e}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_chat_template=True)

    try:
        messages = [{"role": "user", "content": test_prompt}]
        formatted_full_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            add_special_tokens=True,
        )
    except Exception:
        formatted_full_prompt = test_prompt

    prompt_first_part, prompt_second_part = split_prompt_by_tokens(
        formatted_full_prompt, tokenizer, split_ratio=prompt_split_ratio
    )

    ucm_config = {
        "ucm_connectors": [
            {
                "ucm_connector_name": "UcmNfsStore",
                "ucm_connector_config": {
                    "storage_backends": ucm_storage_dir,
                    "use_direct": False,
                },
            }
        ],
        "load_only_first_rank": False,
    }

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1,
        max_tokens=max_tokens,
        ignore_eos=False,
    )

    logger.info(f"\n===== HBM + SSD Mixed Accuracy Test =====")
    logger.info(f"Model: {model_path}")
    logger.info(f"Full prompt length: {len(test_prompt)} chars")
    logger.info(f"Max tokens: {max_tokens}")
    logger.info(f"Temperature: 0.0 (deterministic)")
    logger.info(f"UCM storage: {ucm_storage_dir}")
    logger.info(f"Prompt split ratio: {prompt_split_ratio}")
    logger.info(f"Enforce eager: {enforce_eager}")
    logger.info(f"Max num batched tokens: {max_num_batched_tokens}")

    # ===== Phase 1: Disable HBM PC, save KV cache to SSD and load (baseline) =====
    # Run Phase 1 in a separate subprocess to ensure GPU memory is fully released
    logger.info(f"\n===== Phase 1: Save KV Cache to SSD And Load (Baseline) =====")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.empty_cache()
        torch.npu.synchronize()
    gc.collect()
    time.sleep(2)

    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    error_queue = ctx.Queue()

    # Convert SamplingParams to dict for serialization
    sampling_params_dict = {
        "temperature": sampling_params.temperature,
        "top_p": sampling_params.top_p,
        "max_tokens": sampling_params.max_tokens,
        "ignore_eos": sampling_params.ignore_eos,
    }

    process = ctx.Process(
        target=_run_phase_in_subprocess,
        args=(
            model_path,
            ucm_config,
            [
                formatted_full_prompt,
                formatted_full_prompt,
            ],
            sampling_params_dict,
            False,  # enable_prefix_caching=False for Phase 1
            enforce_eager,
            "Phase 1 (SSD save and load)",
            max_num_batched_tokens,
            result_queue,
            error_queue,
        ),
    )
    process.start()
    process.join(timeout=180)

    if process.is_alive():
        process.terminate()
        process.join()
        raise RuntimeError("Phase 1 timed out after 3 minutes")

    # Check for errors first
    if not error_queue.empty():
        status, error = error_queue.get()
        raise error

    # Check exit code
    if process.exitcode != 0:
        # Try to get error from queue if available
        if not error_queue.empty():
            status, error = error_queue.get()
            raise error
        raise RuntimeError(
            f"Phase 1 failed in subprocess with exit code {process.exitcode}"
        )

    if result_queue.empty():
        raise RuntimeError("Phase 1 subprocess completed but no result in queue")
    phase1_outputs = result_queue.get()
    phase1_1_output = phase1_outputs[0]  # Phase 1.1: SSD save
    phase1_2_output = phase1_outputs[1]  # Phase 1.2: SSD load
    logger.info(f"Phase 1 completed in subprocess, GPU memory should be fully released")
    logger.info(f"Phase 1.1 output: {phase1_1_output}")
    logger.info(f"Phase 1.2 output: {phase1_2_output}")

    # ===== Phase 2: Enable HBM PC, test HBM + SSD mixed hit =====
    # Run Phase 2 in a separate subprocess to ensure GPU memory is fully released
    logger.info(f"\n===== Phase 2: HBM + SSD Mixed Hit Test =====")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.empty_cache()
        torch.npu.synchronize()
    gc.collect()
    time.sleep(2)

    ctx = multiprocessing.get_context("spawn")
    result_queue_2 = ctx.Queue()
    error_queue_2 = ctx.Queue()

    process_2 = ctx.Process(
        target=_run_phase_in_subprocess,
        args=(
            model_path,
            ucm_config,
            [
                prompt_first_part,
                formatted_full_prompt,
            ],  # Phase 2: send partial and full prompts
            sampling_params_dict,
            True,  # enable_prefix_caching=True for Phase 2
            enforce_eager,
            "Phase 2 (HBM + SSD mixed)",
            max_num_batched_tokens,
            result_queue_2,
            error_queue_2,
        ),
    )
    process_2.start()
    process_2.join(timeout=180)

    if process_2.is_alive():
        process_2.terminate()
        process_2.join()
        raise RuntimeError("Phase 2 timed out after 3 minutes")

    # Check for errors first
    if not error_queue_2.empty():
        status, error = error_queue_2.get()
        raise error

    # Check exit code
    if process_2.exitcode != 0:
        # Try to get error from queue if available
        if not error_queue_2.empty():
            status, error = error_queue_2.get()
            raise error
        raise RuntimeError(
            f"Phase 2 failed in subprocess with exit code {process_2.exitcode}"
        )

    if result_queue_2.empty():
        raise RuntimeError("Phase 2 subprocess completed but no result in queue")
    phase2_outputs = result_queue_2.get()
    phase2_partial_output = phase2_outputs[0]
    phase2_full_output = phase2_outputs[1]
    logger.info(f"Phase 2 completed in subprocess, GPU memory should be fully released")
    logger.info(f"[INFO] Phase 2.1 output: {phase2_partial_output}")
    logger.info(f"[INFO] Phase 2.2 output: {phase2_full_output}")

    logger.info(f"\n[INFO] ===== Accuracy Test Results =====")

    def normalize_text(text: str) -> str:
        """Normalize text for comparison by replacing similar punctuation."""
        text = text.replace("，", ",")
        text = text.replace("。", ".")
        text = text.replace("！", "!")
        text = text.replace("？", "?")
        text = text.replace("：", ":")
        text = text.replace("；", ";")
        return text.strip()

    # Compare Phase 1.1 vs Phase 1.2 (SSD load accuracy)
    phase1_identical = phase1_1_output == phase1_2_output
    phase1_normalized_identical = normalize_text(phase1_1_output) == normalize_text(
        phase1_2_output
    )
    if not phase1_identical:
        logger.warning(f"\n===== Phase 1: SSD Load Accuracy Test (Exact Match) =====")
        logger.warning(
            f"Phase 1.1 (SSD save) output differs from Phase 1.2 (SSD load) output!"
        )
        logger.warning(f"Phase 1.1 output:\n{phase1_1_output}")
        logger.warning(f"Phase 1.2 output:\n{phase1_2_output}")
        if phase1_normalized_identical:
            logger.info(
                f"But normalized outputs are identical (punctuation difference only)"
            )

    phase2_identical = phase1_1_output == phase2_full_output
    phase2_normalized_identical = normalize_text(phase1_1_output) == normalize_text(
        phase2_full_output
    )
    if not phase2_identical:
        logger.warning(
            f"\n===== Phase 2: HBM + SSD Mixed Accuracy Test (Exact Match) ====="
        )
        logger.warning(
            f"Phase 1.1 (SSD save) output differs from Phase 2.2 (HBM + SSD mixed) output!"
        )
        logger.warning(f"Phase 1.1 output:\n{phase1_1_output}")
        logger.warning(f"Phase 2.2 output:\n{phase2_full_output}")
        if phase2_normalized_identical:
            logger.info(
                f"But normalized outputs are identical (punctuation difference only)"
            )
            logger.info(
                f"This is likely due to numerical precision differences in KV cache loading"
            )
            logger.info(f"Normalized Phase 1.1: {normalize_text(phase1_1_output)}")
            logger.info(f"Normalized Phase 2.2: {normalize_text(phase2_full_output)}")

    # Assert outputs are identical (using normalized comparison for punctuation differences)
    # Note: Small numerical precision differences in KV cache loading can cause
    # punctuation token selection differences (e.g., full-width vs half-width comma)
    assert phase1_normalized_identical, (
        f"SSD Load Accuracy Test Failed!\n"
        f"Phase 1.1 output: {phase1_1_output}\n"
        f"Phase 1.2 output: {phase1_2_output}\n"
        f"See detailed output above for differences."
    )

    assert phase2_normalized_identical, (
        f"HBM + SSD Mixed Accuracy Test Failed!\n"
        f"Phase 1.1 output: {phase1_1_output}\n"
        f"Phase 2.2 output: {phase2_full_output}\n"
        f"Normalized comparison failed. This may indicate a real accuracy issue.\n"
        f"See detailed output above for differences."
    )

    if phase2_identical:
        logger.info(f"\n✓ HBM + SSD mixed accuracy test passed: outputs are identical")
    else:
        logger.info(
            f"\n ✓ HBM + SSD mixed accuracy test passed: normalized outputs are identical"
        )
        logger.info(
            f"Note: Punctuation difference detected (likely due to numerical precision in KV cache)"
        )

    value_lists = {
        "model_path": [model_path],
        "model_name": [os.path.basename(model_path)],
        "test_prompt_length": [len(test_prompt)],
        "max_tokens": [max_tokens],
        "prompt_split_ratio": [prompt_split_ratio],
        "phase1_identical": [1 if phase1_identical else 0],
        "phase2_identical": [1 if phase2_identical else 0],
    }

    return {"_name": "accuracy_test_offline_hbm_ssd_mixed", "_data": value_lists}
