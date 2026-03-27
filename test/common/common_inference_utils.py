"""
Common Inference Utilities for E2E Tests.

This module provides shared utilities for both online and offline inference testing.
These functions are used by both test types to reduce code duplication.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def ensure_storage_dir(storage_path: str, clear_existing: bool = False):
    """Ensure the storage directory exists and optionally clear existing contents.

    Args:
        storage_path: Path to the storage directory
        clear_existing: If True, remove all existing files in the directory
    """
    os.makedirs(storage_path, exist_ok=True)
    if clear_existing:
        for item in os.listdir(storage_path):
            item_path = os.path.join(storage_path, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                import shutil

                shutil.rmtree(item_path)


def split_prompt_by_tokens(
    prompt: str, tokenizer: Any, split_ratio: float = 0.5
) -> Tuple[str, str]:
    """Split a prompt into two parts by token ratio.

    Args:
        prompt: The prompt to split
        tokenizer: Tokenizer to use for splitting
        split_ratio: Ratio to split (0.5 = split in half)

    Returns:
        Tuple of (first_part, second_part)
    """
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


def load_prompt_list_from_file(
    prompt_file: Optional[Path] = None,
) -> Tuple[str, List[str]]:
    """
    Returns:
        Tuple of (combined_prompt_string_list, answers_list).
        - combined_prompt_string_list: Combined prompt (context + input)
        - answers_list: List of standard answers from the file
    """
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    with open(prompt_file, "r", encoding="utf-8") as f:
        content = f.readlines()
    full_prompts = []
    full_answers = []

    for i in range(len(content)):
        try:
            data = json.loads(content[i])
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {prompt_file}: {e}")

        if isinstance(data, list):
            if len(data) == 0:
                raise ValueError(f"Empty list in {prompt_file}")
            data = data[0]

        input_text = data.get("input", "")
        context_text = data.get("context", "")

        if context_text and input_text:
            full_prompt = f"阅读以下文字并用中文简短回答：\n\n{context_text}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input_text}\n回答："
        elif context_text:
            full_prompt = context_text
        elif input_text:
            full_prompt = input_text
        else:
            raise ValueError(f"No input or context found in {prompt_file}")

        answers = data.get("answers", [])

        if not isinstance(answers, list):
            answers = [answers] if answers else []
        full_prompts.append(full_prompt)
        full_answers.extend(answers)
    return full_prompts, full_answers


def serialize_sample_params(params: str) -> Any:
    import msgspec

    json_bytes = msgspec.json.encode(params)
    json_str = json_bytes.decode("utf-8")
    return json_str


def deserialize_sample_params(json_str: str) -> Any:
    import msgspec

    json_bytes = json_str.encode("utf-8")
    return msgspec.json.decode(
        json_bytes, type=get_platform_specific_module().SamplingParams
    )


def get_platform_specific_module():
    """Get platform-specific modules for inference.

    Returns:
        SimpleNamespace with AutoTokenizer and SamplingParams
    """
    from types import SimpleNamespace

    from transformers import AutoTokenizer
    from vllm import SamplingParams

    # Create a namespace object
    modules = SimpleNamespace()
    modules.AutoTokenizer = AutoTokenizer
    modules.SamplingParams = SamplingParams

    return modules


def match_any_answer(output: str, answers: List[str]) -> bool:
    """Check if output matches any of the standard answers.

    Args:
        output: Generated output text
        answers: List of acceptable answers

    Returns:
        True if output matches any answer
    """
    for answer in answers:
        if remove_punc(extract_answers(output)) == remove_punc(answer):
            return True
    return False


def remove_punc(text: str) -> str:
    """Remove punctuation from text for comparison.

    Args:
        text: Text to remove punctuation from

    Returns:
        Text without punctuation
    """
    import string

    text = text.strip()
    if not text:
        return ""
    cn_punctuation = (
        "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—''‛"
        "„‟…‧﹏."
    )
    all_punctuation = set(string.punctuation + cn_punctuation)
    return "".join(ch for ch in text if ch not in all_punctuation)


def match_sparse_answer(sparse_output: List[str], standard_answers: List[str]) -> bool:
    """Check if sparse output matches standard answers after removing punctuation.

    Args:
        sparse_output: List of generated outputs
        standard_answers: List of expected answers

    Returns:
        True if outputs match after normalization
    """
    if not isinstance(sparse_output, list) or not isinstance(standard_answers, list):
        return False
    if not all(isinstance(item, str) for item in sparse_output) or not all(
        isinstance(item, str) for item in standard_answers
    ):
        return False

    norm_output = [remove_punc(item) for item in sparse_output]
    norm_standard = [remove_punc(item) for item in standard_answers]
    return norm_output == norm_standard


def extract_answers(generated_text_list: List[str]) -> List[str]:
    """Extract answers from generated text by removing thinking tags.

    Args:
        generated_text_list: List of generated texts

    Returns:
        List of extracted answers
    """
    results = []

    for text in generated_text_list:
        if not isinstance(text, str):
            results.append("")
            continue

        if "</think>" in text:
            answer = text.rsplit("</think>", 1)[-1].strip()
        else:
            answer = text.strip()

        answer = answer.strip("'").strip('"').strip()

        results.append(answer)

    return results
