"""
NOTE: Each test case should run with multiprocessing spawn mode to ensure GPU memory
is fully released after each test. This prevents memory accumulation across test cases.
"""

import pytest
import yaml
import time
import os
import contextlib
import gc
import json
import tempfile
import multiprocessing
import subprocess
import sys
from functools import wraps
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import asdict

from common.capture_utils import export_vars
from transformers import AutoTokenizer

_test_functions = {}


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
    # 注册测试函数到全局字典，用于存储测试函数（避免 pickle 嵌套函数的问题）
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
            args=(test_id, args, kwargs, result_queue, error_queue)
        )
        process.start()
        process.join(timeout=3600)
        
        if process.is_alive():
            process.terminate()
            process.join()
            raise RuntimeError(f"Test {func.__name__} timed out after 1 hour")
        
        if not error_queue.empty():
            status, error = error_queue.get()
            raise error
        
        if not result_queue.empty():
            status, result = result_queue.get()
            return result
        
        if process.exitcode != 0:
            raise RuntimeError(f"Test {func.__name__} failed in spawn process with exit code {process.exitcode}")
    
    return wrapper



try:
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    from vllm.engine.arg_utils import EngineArgs
    from ucm.logger import init_logger
    import torch
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    pytest.skip("vLLM not available", allow_module_level=True)

logger = init_logger(__name__)


@contextlib.contextmanager
def build_llm_with_uc(
    model_path: str,
    ucm_config: Optional[Dict[str, Any]] = None,
    enable_prefix_caching: bool = False,
    **llm_kwargs
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
        os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
    
    tensor_parallel_size = 2
    
    default_args = {
        "model": model_path,
        "kv_transfer_config": ktc,
        "max_model_len": 32768,
        "gpu_memory_utilization": 0.8,
        "max_num_batched_tokens": 30000,
        "block_size": 128,
        "enforce_eager": True,
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
        gc.collect()


def split_prompt_by_tokens(
    prompt: str,
    tokenizer: AutoTokenizer,
    split_ratio: float = 0.5
) -> Tuple[str, str]:
    tokens = tokenizer.encode(prompt)
    split_idx = int(len(tokens) * split_ratio)
    
    first_tokens = tokens[:split_idx]
    second_tokens = tokens[split_idx:]
    
    first_part = tokenizer.decode(first_tokens, skip_special_tokens=False)
    second_part = tokenizer.decode(second_tokens, skip_special_tokens=False)
    
    return first_part, second_part


def create_prompt_with_token_count(
    base_prompt: str,
    tokenizer: AutoTokenizer,
    target_token_count: int
) -> str:
    """Create a prompt with approximately target token count.
    
    Args:
        base_prompt: Base prompt string to repeat/extend.
        tokenizer: Tokenizer to count tokens.
        target_token_count: Target number of tokens.
    
    Returns:
        Prompt string with approximately target_token_count tokens.
    """
    base_tokens = tokenizer.encode(base_prompt)
    base_token_count = len(base_tokens)
    
    if base_token_count >= target_token_count:
        # If base prompt is already long enough, truncate it
        tokens = base_tokens[:target_token_count]
        return tokenizer.decode(tokens, skip_special_tokens=False)
    
    # Calculate how many times to repeat
    repeat_count = (target_token_count // base_token_count) + 1
    extended_prompt = (base_prompt + " ") * repeat_count
    
    # Trim to target length
    tokens = tokenizer.encode(extended_prompt)
    tokens = tokens[:target_token_count]
    return tokenizer.decode(tokens, skip_special_tokens=False)


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
        data = data[0]  # Use first item, or you can process all items
    
    # Extract input and context from LongBench format
    input_text = data.get("input", "")
    context_text = data.get("context", "")
    
    # LongBench standard format: context (long document) + input (question)
    # Combine context and input to form the full prompt
    # Format: context + "\n\n" + input
    if context_text and input_text:
        # Standard LongBench format: context first, then input
        full_prompt = f"{context_text}\n\n{input_text}"
    elif context_text:
        # Only context available
        full_prompt = context_text
    elif input_text:
        # Only input available
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
) -> Tuple[List[str], float]:
    """Run inference and return generated texts and elapsed time.
    
    Args:
        llm: LLM instance.
        prompts: List of prompt strings.
        sampling_params: Sampling parameters.
        description: Description for logging.
    
    Returns:
        Tuple of (generated_texts, elapsed_time).
    """
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed_time = time.time() - start_time
    
    generated_texts = []
    for output in outputs:
        generated_text = output.outputs[0].text
        generated_texts.append(generated_text)
    
    if description:
        print(f"[INFO] {description} completed in {elapsed_time:.2f}s")
    
    return generated_texts, elapsed_time


@pytest.mark.parametrize("model_path", [
    "/home/models/QwQ-32B",
    "/home/models/DeepSeek-V2-Lite",
])
@pytest.mark.parametrize("max_tokens", [100])
@pytest.mark.feature("uc_accuracy_test_offline")
@export_vars
@run_in_spawn_process
def test_offline_accuracy_ssd_load(
    model_path: str,
    max_tokens: int,
):
    """Test SSD load accuracy (Phase 1).
    
    Test flow:
    1. Phase 1.1: Disable HBM PC, send full prompt -> KV cache saved to SSD
    2. Phase 1.2: Disable HBM PC, load from SSD, send full prompt -> verify SSD load accuracy
    
    The prompt is loaded from prompt.json file (LongBench format).
    
    Args:
        model_path: Path to the model.
        max_tokens: Maximum tokens to generate.
    """
    # Load configuration
    config_file = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Use model_path from parameter, fallback to config or environment
    if not model_path or not os.path.exists(model_path):
        model_path = config.get("llm_connection", {}).get("model_path") or os.getenv("MODEL_PATH")
        if not model_path:
            pytest.skip(f"model_path not configured or not found: {model_path}")
    
    if not os.path.exists(model_path):
        pytest.skip(f"Model path does not exist: {model_path}")
    
    ucm_storage_dir = config.get("llm_connection", {}).get("ucm_storage_dir") or os.getenv("UCM_STORAGE_DIR", "/tmp/ucm_cache")
    
    try:
        test_prompt, standard_answers = load_prompt_from_file()
        print(f"[INFO] Loaded prompt from prompt.json (length: {len(test_prompt)} chars)")
        if standard_answers:
            print(f"[INFO] Standard answers: {standard_answers}")
        else:
            print(f"[INFO] No standard answers found in prompt.json")
    except Exception as e:
        pytest.skip(f"Failed to load prompt from prompt.json: {e}")
    
    # Setup tokenizer
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
    )
    
    print(f"\n[INFO] ===== SSD Load Accuracy Test =====")
    print(f"[INFO] Model: {model_path}")
    print(f"[INFO] Full prompt length: {len(test_prompt)} chars")
    print(f"[INFO] Max tokens: {max_tokens}")
    print(f"[INFO] Temperature: 0.0 (deterministic)")
    print(f"[INFO] UCM storage: {ucm_storage_dir}")
    
    # ===== Phase 1.1 and 1.2: Disable HBM PC, save KV cache to SSD and load =====
    print(f"\n[INFO] ===== Phase 1.1 and 1.2: Save KV Cache to SSD And Load =====")
    print(f"[INFO] HBM Prefix Caching: DISABLED")
    
    with build_llm_with_uc(
        model_path=model_path,
        ucm_config=ucm_config,
        enable_prefix_caching=False,  # Disable HBM PC
    ) as llm:
        phase1_outputs, phase1_time = run_inference(
            llm, [formatted_full_prompt, formatted_full_prompt], sampling_params, "First save then load"
        )
        phase1_1_output = phase1_outputs[0]
        phase1_2_output = phase1_outputs[1]
    
    # ===== Compare outputs =====
    print(f"\n[INFO] ===== Accuracy Test Results =====")
    
    # Compare Phase 1.1 vs Phase 1.2 (SSD load accuracy)
    phase1_identical = phase1_1_output == phase1_2_output
    # if not phase1_identical:
    print(f"\n[INFO] ===== Phase 1: SSD Load Accuracy Test =====")
    print(f"[INFO] Phase 1.1 (SSD save) output differs from Phase 1.2 (SSD load) output!")
    print(f"[INFO] Phase 1.1 output:\n{phase1_1_output}")
    print(f"[INFO] Phase 1.2 output:\n{phase1_2_output}")
    
    # Assert outputs are identical - test fails if any difference
    assert phase1_identical, (
        f"SSD Load Accuracy Test Failed!\n"
        f"See detailed output above for differences."
    )
    
    print(f"\n[INFO] SSD load accuracy test passed: outputs are identical")
    
    value_lists = {
        "model_path": [model_path],
        "model_name": [os.path.basename(model_path)],
        "test_prompt_length": [len(test_prompt)],
        "max_tokens": [max_tokens],
        "phase1_identical": [1 if phase1_identical else 0],
        "phase1_time": [phase1_time],
    }
    
    return {"_name": "accuracy_test_offline_ssd_load", "_data": value_lists}


@pytest.mark.parametrize("model_path", [
    "/home/models/QwQ-32B",
    "/home/models/DeepSeek-V2-Lite",
])
@pytest.mark.parametrize("max_tokens", [200])
@pytest.mark.parametrize("prompt_split_ratio", [0.5])  # Split prompt in half
@pytest.mark.feature("uc_accuracy_test_offline")
@export_vars
@run_in_spawn_process
def test_offline_accuracy_hbm_ssd_mixed(
    model_path: str,
    max_tokens: int,
    prompt_split_ratio: float,
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
    # Load configuration
    config_file = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Use model_path from parameter, fallback to config or environment
    if not model_path or not os.path.exists(model_path):
        model_path = config.get("llm_connection", {}).get("model_path") or os.getenv("MODEL_PATH")
        if not model_path:
            pytest.skip(f"model_path not configured or not found: {model_path}")
    
    if not os.path.exists(model_path):
        pytest.skip(f"Model path does not exist: {model_path}")
    
    ucm_storage_dir = config.get("llm_connection", {}).get("ucm_storage_dir") or os.getenv("UCM_STORAGE_DIR", "/tmp/ucm_cache")
    
    # Load prompt and answers from prompt.json file
    try:
        test_prompt, standard_answers = load_prompt_from_file()
        print(f"[INFO] Loaded prompt from prompt.json (length: {len(test_prompt)} chars)")
        if standard_answers:
            print(f"[INFO] Standard answers: {standard_answers}")
        else:
            print(f"[INFO] No standard answers found in prompt.json")
    except Exception as e:
        pytest.skip(f"Failed to load prompt from prompt.json: {e}")
    
    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_chat_template=True)
    
    # Format prompt with chat template if available
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
    
    print(f"\n[INFO] ===== HBM + SSD Mixed Accuracy Test =====")
    print(f"[INFO] Model: {model_path}")
    print(f"[INFO] Full prompt length: {len(test_prompt)} chars")
    print(f"[INFO] Max tokens: {max_tokens}")
    print(f"[INFO] Temperature: 0.0 (deterministic)")
    print(f"[INFO] UCM storage: {ucm_storage_dir}")
    print(f"[INFO] Prompt split ratio: {prompt_split_ratio}")
    
    # ensure Phase 1 has already run and KV cache is saved to SSD
    # ===== Phase 2: Enable HBM PC, test HBM + SSD mixed hit =====
    print(f"\n[INFO] ===== Phase 2: HBM + SSD Mixed Hit Test =====")
    print(f"[INFO] HBM Prefix Caching: ENABLED")
    print(f"[INFO] Sending partial prompt and full prompt together in one batch...")
    print(f"[INFO]   - Partial prompt (first {int(prompt_split_ratio*100)}%) will warm up HBM")
    print(f"[INFO]   - Full prompt will hit HBM for prefix + SSD for suffix")
    
    with build_llm_with_uc(
        model_path=model_path,
        ucm_config=ucm_config,
        enable_prefix_caching=True,  # Enable HBM PC
    ) as llm:
        # Send both partial and full prompts together in one batch
        # This ensures that partial prompt warms up HBM, then full prompt hits both HBM and SSD
        phase2_outputs, phase2_time = run_inference(
            llm, [prompt_first_part, formatted_full_prompt], sampling_params, "Phase 2 (HBM + SSD mixed)"
        )
        phase2_partial_output = phase2_outputs[0]  # Output from partial prompt (for reference)
        phase2_full_output = phase2_outputs[1]     # Output from full prompt (this is what we compare)
    
    # ===== Compare outputs =====
    print(f"\n[INFO] ===== Accuracy Test Results =====")
    
    # Compare Phase 1.1 vs Phase 1.2 (SSD load accuracy)
    phase1_identical = phase1_1_output == phase1_2_output
    if not phase1_identical:
        print(f"\n[ERROR] ===== Phase 1: SSD Load Accuracy Test FAILED =====")
        print(f"[ERROR] Phase 1.1 (SSD save) output differs from Phase 1.2 (SSD load) output!")
        print(f"[ERROR] Phase 1.1 output:\n{phase1_1_output}")
        print(f"[ERROR] Phase 1.2 output:\n{phase1_2_output}")
    
    phase2_identical = phase1_1_output == phase2_full_output
    if not phase2_identical:
        print(f"\n[ERROR] ===== Phase 2: HBM + SSD Mixed Accuracy Test FAILED =====")
        print(f"[ERROR] Phase 1.1 (SSD save) output differs from Phase 2.2 (HBM + SSD mixed) output!")
        print(f"[ERROR] Phase 1.1 output:\n{phase1_1_output}")
        print(f"[ERROR] Phase 2.2 output:\n{phase2_full_output}")
    
    # Assert outputs are identical - test fails if any difference
    assert phase1_identical, (
        f"SSD Load Accuracy Test Failed!\n"
        f"See detailed output above for differences."
    )
    
    assert phase2_identical, (
        f"HBM + SSD Mixed Accuracy Test Failed!\n"
        f"See detailed output above for differences."
    )
    
    print(f"\n[INFO] ✓ HBM + SSD mixed accuracy test passed: outputs are identical")
    
    # Prepare data for export
    value_lists = {
        "model_path": [model_path],
        "model_name": [os.path.basename(model_path)],
        "test_prompt_length": [len(test_prompt)],
        "max_tokens": [max_tokens],
        "prompt_split_ratio": [prompt_split_ratio],
        "phase2_identical": [1 if phase2_identical else 0],
        "phase2_time": [phase2_time],
    }
    
    return {"_name": "accuracy_test_offline_hbm_ssd_mixed", "_data": value_lists}


@pytest.mark.parametrize("model_path", [
    "/home/models/QwQ-32B",
    "/home/models/DeepSeek-V2-Lite",
])
@pytest.mark.parametrize("base_prompt", [
    "This is a test prompt for chunk prefill accuracy testing. ",
])
@pytest.mark.parametrize("max_tokens", [200])
@pytest.mark.parametrize("test_scenario", [
    # (prompt_token_count, max_num_batched_tokens, block_size, description)
    # Scenario 1: prompt < max_num_matched_tokens (no chunk prefill)
    (5000, 30000, 128, "small_prompt_no_chunk"),
    # Scenario 2: prompt > max_num_batched_tokens, divisible by block_size
    (35000, 30000, 128, "large_prompt_chunk_divisible"),
    # Scenario 3: prompt > max_num_batched_tokens, not divisible by block_size
    (35000, 30001, 128, "large_prompt_chunk_not_divisible"),
    # Scenario 4: prompt > max_num_batched_tokens, max_num_batched_tokens divisible by block_size
    (40000, 30000, 128, "very_large_prompt_divisible_batch"),
    # Scenario 5: prompt > max_num_batched_tokens, max_num_batched_tokens not divisible by block_size
    (40000, 30001, 128, "very_large_prompt_not_divisible_batch"),
])
@pytest.mark.feature("uc_accuracy_test_offline")
@export_vars
@run_in_spawn_process
def test_offline_accuracy_chunk_prefill(
    model_path: str,
    base_prompt: str,
    max_tokens: int,
    test_scenario: Tuple[int, int, int, str],
):
    """Test accuracy with chunk prefill scenarios.
    
    This test covers various chunk prefill scenarios:
    1. Prompt < max_num_matched_tokens (no chunk prefill)
    2. Prompt > max_num_batched_tokens (triggers chunk prefill)
    3. max_num_batched_tokens divisible by block_size
    4. max_num_batched_tokens not divisible by block_size
    
    Args:
        base_prompt: Base prompt string to extend.
        max_tokens: Maximum tokens to generate.
        test_scenario: Tuple of (prompt_token_count, max_num_batched_tokens, block_size, description).
    """
    prompt_token_count, max_num_batched_tokens, block_size, scenario_name = test_scenario
    
    # Load configuration
    config_file = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Use model_path from parameter, fallback to config or environment
    if not model_path or not os.path.exists(model_path):
        model_path = config.get("llm_connection", {}).get("model_path") or os.getenv("MODEL_PATH")
        if not model_path:
            pytest.skip(f"model_path not configured or not found: {model_path}")
    
    if not os.path.exists(model_path):
        pytest.skip(f"Model path does not exist: {model_path}")
    
    ucm_storage_dir = config.get("llm_connection", {}).get("ucm_storage_dir") or os.getenv("UCM_STORAGE_DIR", "/tmp/ucm_cache")
    
    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_chat_template=True)
    
    # Create prompt with target token count
    test_prompt = create_prompt_with_token_count(base_prompt, tokenizer, prompt_token_count)
    actual_token_count = len(tokenizer.encode(test_prompt))
    
    # Format prompt with chat template if available
    try:
        messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": test_prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            add_special_tokens=True,
        )
    except Exception:
        formatted_prompt = test_prompt
    
    # Setup UCM config
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
    }
    
    # Create sampling params with temperature=0 for deterministic output
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=0.95,
        max_tokens=max_tokens,
        ignore_eos=False,
    )
    
    # Check if divisible
    is_divisible = (max_num_batched_tokens % block_size == 0)
    
    print(f"\n[INFO] ===== Chunk Prefill Accuracy Test: {scenario_name} =====")
    print(f"[INFO] Model: {os.path.basename(model_path)}")
    print(f"[INFO] Prompt token count: {actual_token_count} (target: {prompt_token_count})")
    print(f"[INFO] Max num batched tokens: {max_num_batched_tokens}")
    print(f"[INFO] Block size: {block_size}")
    print(f"[INFO] Divisible: {is_divisible}")
    print(f"[INFO] Will trigger chunk prefill: {actual_token_count > max_num_batched_tokens}")
    print(f"[INFO] Max tokens: {max_tokens}")
    print(f"[INFO] Temperature: 0.0 (deterministic)")
    print(f"[INFO] UCM storage: {ucm_storage_dir}")
    
    # ===== Phase 1.1: Save KV cache to SSD (with chunk prefill if needed) =====
    print(f"\n[INFO] ===== Phase 1.1: Save KV Cache to SSD =====")
    print(f"[INFO] HBM Prefix Caching: DISABLED")
    print(f"[INFO] Sending prompt (may trigger chunk prefill)...")
    
    with build_llm_with_uc(
        model_path=model_path,
        ucm_config=ucm_config,
        enable_prefix_caching=False,
        max_num_batched_tokens=max_num_batched_tokens,
        block_size=block_size,
    ) as llm:
        phase1_1_outputs, phase1_1_time = run_inference(
            llm, [formatted_prompt], sampling_params, "Phase 1.1 (SSD save, chunk prefill)"
        )
        phase1_1_output = phase1_1_outputs[0]
    
    time.sleep(1)
    
    # ===== Phase 1.2: Load from SSD (with chunk prefill if needed) =====
    print(f"\n[INFO] ===== Phase 1.2: Load from SSD (HBM PC disabled) =====")
    print(f"[INFO] HBM Prefix Caching: DISABLED")
    print(f"[INFO] Loading KV cache from SSD and sending prompt (may trigger chunk prefill)...")
    
    with build_llm_with_uc(
        model_path=model_path,
        ucm_config=ucm_config,
        enable_prefix_caching=False,
        max_num_batched_tokens=max_num_batched_tokens,
        block_size=block_size,
    ) as llm:
        phase1_2_outputs, phase1_2_time = run_inference(
            llm, [formatted_prompt], sampling_params, "Phase 1.2 (SSD load, chunk prefill)"
        )
        phase1_2_output = phase1_2_outputs[0]
    
    # ===== Compare outputs =====
    print(f"\n[INFO] ===== Accuracy Test Results =====")
    
    # Compare Phase 1.1 vs Phase 1.2 (SSD load accuracy with chunk prefill)
    phase1_identical = phase1_1_output == phase1_2_output
    phase1_1_len = len(phase1_1_output)
    phase1_2_len = len(phase1_2_output)
    
    # Calculate similarity
    if phase1_1_len > 0 and phase1_2_len > 0:
        min_len = min(phase1_1_len, phase1_2_len)
        matching_chars = sum(1 for i in range(min_len) if phase1_1_output[i] == phase1_2_output[i])
        phase1_similarity = matching_chars / max(phase1_1_len, phase1_2_len) if max(phase1_1_len, phase1_2_len) > 0 else 0.0
    else:
        phase1_similarity = 1.0 if phase1_identical else 0.0
    
    print(f"\n[INFO] --- Chunk Prefill SSD Load Accuracy ---")
    print(f"[INFO] Scenario: {scenario_name}")
    print(f"[INFO] Phase 1.1 (SSD save) output length: {phase1_1_len}")
    print(f"[INFO] Phase 1.2 (SSD load) output length: {phase1_2_len}")
    print(f"[INFO] Outputs identical: {phase1_identical}")
    print(f"[INFO] Similarity ratio: {phase1_similarity:.4f}")
    print(f"[INFO] Phase 1.1 time: {phase1_1_time:.2f}s")
    print(f"[INFO] Phase 1.2 time: {phase1_2_time:.2f}s")
    
    if not phase1_identical:
        print(f"\n[ERROR] Outputs differ! Chunk prefill SSD load accuracy issue detected.")
        print(f"[INFO] Phase 1.1 output (first 200 chars): {phase1_1_output[:200]}")
        print(f"[INFO] Phase 1.2 output (first 200 chars): {phase1_2_output[:200]}")
        
        diff_pos = next((i for i, (c1, c2) in enumerate(zip(phase1_1_output, phase1_2_output)) if c1 != c2), None)
        if diff_pos is not None:
            print(f"[INFO] First difference at position: {diff_pos}")
            context_start = max(0, diff_pos - 50)
            context_end = min(len(phase1_1_output), diff_pos + 50)
            print(f"[INFO] Context around difference:")
            print(f"[INFO]   Phase 1.1: ...{phase1_1_output[context_start:context_end]}...")
            print(f"[INFO]   Phase 1.2: ...{phase1_2_output[context_start:context_end]}...")
    
    # Assert outputs are identical
    assert phase1_identical, (
        f"Chunk Prefill SSD Load Accuracy Test Failed!\n"
        f"Scenario: {scenario_name}\n"
        f"Phase 1.1 (SSD save) output differs from Phase 1.2 (SSD load) output.\n"
        f"Similarity ratio: {phase1_similarity:.4f}\n"
        f"Prompt token count: {actual_token_count}, Max batched tokens: {max_num_batched_tokens}\n"
        f"Block size: {block_size}, Divisible: {is_divisible}\n"
        f"Phase 1.1 output (first 500 chars): {phase1_1_output[:500]}\n"
        f"Phase 1.2 output (first 500 chars): {phase1_2_output[:500]}"
    )
    
    print(f"[INFO] ✓ Chunk prefill accuracy test passed: outputs are identical")
    
    # Prepare data for export
    value_lists = {
        "model_path": [model_path],
        "model_name": [os.path.basename(model_path)],
        "scenario_name": [scenario_name],
        "prompt_token_count": [actual_token_count],
        "max_num_batched_tokens": [max_num_batched_tokens],
        "block_size": [block_size],
        "is_divisible": [1 if is_divisible else 0],
        "triggers_chunk_prefill": [1 if actual_token_count > max_num_batched_tokens else 0],
        "max_tokens": [max_tokens],
        "phase1_identical": [1 if phase1_identical else 0],
        "phase1_similarity_ratio": [phase1_similarity],
        "phase1_1_output_length": [phase1_1_len],
        "phase1_2_output_length": [phase1_2_len],
        "phase1_1_time": [phase1_1_time],
        "phase1_2_time": [phase1_2_time],
    }
    
    return {"_name": "accuracy_test_offline_chunk_prefill", "_data": value_lists}
