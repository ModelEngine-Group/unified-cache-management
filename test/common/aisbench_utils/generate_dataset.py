"""
Dataset Generation Module - Supports normal and prefix-cache datasets
Supports fixed-length and variable-length modes

Data source modes:
1. GSM8K mode (default): Use GSM8K dataset as content source
2. Random token mode: Directly generate random tokens (no dataset dependency)

Improvements:
- Removed torch dependency, using random instead
- Added random token generation mode (reuse token_counter.HuggingFaceTokenizer)
- Added time-based random offset to ensure different prefixes across test runs
- Improved error handling with user-friendly messages
"""
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Optional, Tuple, List, Union

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from .data_picker import DataPicker


def _get_tokenizer(tokenizer_path: str):
    """Load tokenizer - prefer HuggingFaceTokenizer (no torch dependency)"""
    try:
        from common.llm_connection.token_counter import HuggingFaceTokenizer
        return HuggingFaceTokenizer(tokenizer_path)
    except ImportError:
        # Fallback to LightTokenizer or AutoTokenizer
        try:
            from .data_picker import LightTokenizer
            return LightTokenizer(tokenizer_path)
        except FileNotFoundError:
            try:
                from transformers import AutoTokenizer
                return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
            except Exception as e:
                raise RuntimeError(f"Failed to load tokenizer: {tokenizer_path}, error: {e}") from e
        except ImportError as e:
            try:
                from transformers import AutoTokenizer
                return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
            except Exception as inner:
                raise RuntimeError(f"Failed to load tokenizer: {tokenizer_path}, error: {inner}") from e


def create_dataset(
    tokenizer_path: str,
    input_len: int,
    number: int,
    prefix_flag: int,
    gsm8k_path: str = None
) -> Optional[List[str]]:
    """
    Create dataset with specified token length from GSM8K

    Args:
        tokenizer_path: Path to tokenizer
        input_len: Target token length
        number: Number of samples to generate
        prefix_flag: 1 for non-repeat mode, 0 for repeatable mode
        gsm8k_path: Path to GSM8K dataset, defaults to GSM8K.jsonl in project root

    Returns:
        List of generated samples, None if failed
    """
    # Get GSM8K dataset path
    if gsm8k_path is None:
        current_dir = Path(__file__).resolve()
        # aisbench_utils -> common -> test -> project_root
        project_root = current_dir.parents[3]
        gsm8k_path = project_root / "GSM8K.jsonl"

    gsm8k_path = Path(gsm8k_path)
    if not gsm8k_path.exists():
        logging.error(f"GSM8K dataset not found: {gsm8k_path}")
        return None

    logging.info(f"Loading tokenizer: {tokenizer_path}")
    tokenizer = _get_tokenizer(tokenizer_path)

    output_samples = []
    attempts = 0
    max_attempts = number * 10

    picker = DataPicker(
        str(gsm8k_path),
        prefix_flag
    )

    pbar = tqdm(total=number, desc="Generating dataset", unit="row") if tqdm else None

    while len(output_samples) < number and attempts < max_attempts:
        attempts += 1
        raw_text = picker.pick_one()

        if raw_text is None:
            if pbar:
                pbar.close()
            logging.error(
                f"Dataset generation interrupted: {len(output_samples)}/{number} samples generated\n"
                f"Reason: GSM8K data exhausted or insufficient"
            )
            if len(output_samples) > 0:
                logging.warning(f"Returning {len(output_samples)} generated samples")
            return None

        tokens = tokenizer.encode(raw_text, add_special_tokens=False)
        if len(tokens) == 0:
            continue

        # Adjust length: repeat or truncate
        if len(tokens) >= input_len:
            adjusted_tokens = tokens[:input_len]
        else:
            repeat_times = (input_len + len(tokens) - 1) // len(tokens)
            adjusted_tokens = (tokens * repeat_times)[:input_len]

        adjusted_text = tokenizer.decode(adjusted_tokens, skip_special_tokens=True)

        # Verify and fix length
        final_len = len(tokenizer.encode(adjusted_text, add_special_tokens=False))
        if final_len != input_len:
            corrected_tokens = tokenizer.encode(adjusted_text, add_special_tokens=False)
            if len(corrected_tokens) >= input_len:
                corrected_tokens = corrected_tokens[:input_len]
            else:
                corrected_tokens = (corrected_tokens * ((input_len // len(corrected_tokens)) + 1))[:input_len]
            adjusted_text = tokenizer.decode(corrected_tokens, skip_special_tokens=True)

        output_samples.append(adjusted_text)
        if pbar:
            pbar.update(1)

    if pbar:
        pbar.close()

    if len(output_samples) < number:
        logging.warning(f"Only generated {len(output_samples)}/{number} samples")
        if len(output_samples) == 0:
            return None

    return output_samples


def create_dataset_from_random_tokens(
    tokenizer_path: str,
    input_len: int,
    number: int,
    seed: int = 42,
    use_time_offset: bool = True
) -> List[str]:
    """
    Create dataset using random token generation (no dataset dependency)

    Uses HuggingFaceTokenizer.get_some_tokens() which:
    - Generates random tokens directly from tokenizer vocabulary
    - Has length calibration to prevent tokenizer merge issues
    - Seed-based reproducibility with time offset for uniqueness

    Args:
        tokenizer_path: Path to tokenizer
        input_len: Target token length for each sample
        number: Number of samples to generate
        seed: Base random seed for reproducibility
        use_time_offset: If True, add time-based offset to ensure
                         different content across test runs (default: True)

    Returns:
        List of generated samples
    """
    # Add time-based offset to ensure different prefixes across test runs
    # This prevents prefix collision between different test executions
    time_offset = int(time.time()) % 1000000 if use_time_offset else 0
    effective_seed = seed + time_offset

    logging.info(f"Generating {number} samples with {input_len} tokens each (seed={seed}, time_offset={time_offset}, effective_seed={effective_seed})")
    tokenizer = _get_tokenizer(tokenizer_path)

    output_samples = []
    pbar = tqdm(total=number, desc="Generating random tokens", unit="row") if tqdm else None

    for i in range(number):
        # Use different seed for each sample for variety
        sample_seed = effective_seed + i
        text = tokenizer.get_some_tokens(input_len, seed=sample_seed)
        output_samples.append(text)
        if pbar:
            pbar.update(1)

    if pbar:
        pbar.close()

    logging.info(f"Generated {len(output_samples)} samples using random tokens")
    return output_samples


def generate_unique_tokens(
    tokenizer_path: str,
    seed: int,
    n: int,
    number: int
) -> List[str]:
    """
    Generate n unique tokens for number rows based on tokenizer and random seed

    Note: torch dependency removed, using random instead

    Args:
        tokenizer_path: Path to tokenizer
        seed: Random seed
        n: Number of tokens per row
        number: Number of rows to generate

    Returns:
        List containing number rows of data
    """
    tokenizer = _get_tokenizer(tokenizer_path)
    vocab_size = len(tokenizer)

    if n > vocab_size:
        raise ValueError(f"Requested tokens per row {n} exceeds vocab size {vocab_size}")

    all_lines = []
    pbar = tqdm(total=number, desc="Generating unique tokens", unit="row") if tqdm else None

    for line_idx in range(number):
        if pbar:
            pbar.update(1)

        # Use Python random instead of torch
        line_seed = seed + line_idx
        rng = random.Random(line_seed)

        unique_tokens = []
        seen_tokens = set()
        max_attempts = n * 10
        attempts = 0

        while len(unique_tokens) < n and attempts < max_attempts:
            token_id = rng.randint(0, vocab_size - 1)

            if token_id in seen_tokens:
                attempts += 1
                continue

            try:
                token_text = tokenizer.decode([token_id])
                # Filter empty and special tokens
                if token_text.strip():
                    unique_tokens.append(token_text)
                    seen_tokens.add(token_id)
            except Exception:
                pass

            attempts += 1

        if len(unique_tokens) < n:
            logging.warning(f"Row {line_idx + 1} only generated {len(unique_tokens)} unique tokens")

        all_lines.append(''.join(unique_tokens))

    if pbar:
        pbar.close()

    return all_lines


def write_data(path: str, dataset: List[str], num: Optional[int] = None):
    """Write dataset to jsonl file"""
    if num is not None:
        if len(dataset) < num:
            repeats = num // len(dataset)
            remainder = num % len(dataset)
            dataset = dataset * repeats + dataset[:remainder]
        else:
            dataset = dataset[:num]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps({"question": item, "answer": "none"}, ensure_ascii=False))
            f.write("\n")

    logging.info(f"Dataset saved: {path} ({len(dataset)} samples)")


def sample_target_length(
    rng: random.Random,
    fixed_length: int,
    length_mean: Optional[int] = None,
    length_std: Optional[float] = None,
    length_min: Optional[int] = None,
    length_max: Optional[int] = None
) -> int:
    """Sample target length from Gaussian or uniform distribution"""
    fixed_length = max(1, int(fixed_length))
    has_gauss = (length_mean is not None) and (length_std is not None)
    has_range = (length_min is not None) and (length_max is not None)

    lo = 1 if length_min is None else max(1, int(length_min))
    hi = None if length_max is None else max(1, int(length_max))

    if hi is not None and lo > hi:
        lo, hi = hi, lo

    if has_gauss:
        mu = max(1, int(length_mean))
        sigma = max(0.0, float(length_std))
        val = mu if sigma == 0 else int(round(rng.gauss(mu, sigma)))
        if hi is not None:
            val = min(val, hi)
        val = max(lo, val)
        return max(1, val)

    if has_range:
        return rng.randint(lo, hi)

    return fixed_length


def _build_length_tag(
    input_len: int,
    length_mean: Optional[int],
    length_std: Optional[float],
    length_min: Optional[int],
    length_max: Optional[int]
) -> str:
    """Build length tag for file naming"""
    if (length_mean is not None) and (length_std is not None):
        tag = f"G{int(length_mean)}_{str(length_std).replace('.', 'd')}"
        if (length_min is not None) and (length_max is not None):
            tag += f"_C{int(length_min)}_{int(length_max)}"
        return tag

    if (length_min is not None) and (length_max is not None):
        return f"U{int(length_min)}_{int(length_max)}"

    return f"L{int(input_len)}"


def _truncate_or_pad_text(tokenizer, text: str, target_len: int) -> str:
    """Adjust text token length to target_len (truncate or repeat padding)"""
    tokens = tokenizer.encode(text, add_special_tokens=False)

    if len(tokens) >= target_len:
        tokens = tokens[:target_len]
    else:
        repeat_times = (target_len + len(tokens) - 1) // len(tokens)
        tokens = (tokens * repeat_times)[:target_len]

    return tokenizer.decode(tokens, skip_special_tokens=True)


def create_multi_prefix_dataset(
    tokenizer_path: str,
    input_len: int,
    number: int,
    save_path: str,
    prefix_flag: int,
    dp: int,
    repeat_rate: float,
    seed: int,
    prefix_num: int,
    length_mean: Optional[int] = None,
    length_std: Optional[float] = None,
    length_min: Optional[int] = None,
    length_max: Optional[int] = None,
    gsm8k_path: Optional[str] = None,
    use_gsm8k: bool = True
) -> Tuple[str, str]:
    """
    Create multi-prefix dataset

    Supports two data source modes:
    1. GSM8K mode (use_gsm8k=True): Use GSM8K dataset as content source
    2. Random token mode (use_gsm8k=False): Generate random tokens directly (no dataset dependency)

    Args:
        tokenizer_path: Path to tokenizer
        input_len: Input length
        number: Number of samples
        save_path: Save path
        prefix_flag: Prefix flag (0=normal, 1=prefix mode)
        dp: Number of DP domains
        repeat_rate: Prefix repeat rate
        seed: Random seed
        prefix_num: Number of prefix types
        length_mean/length_std/length_min/length_max: Variable-length parameters
        gsm8k_path: Path to GSM8K dataset (only used when use_gsm8k=True)
        use_gsm8k: Whether to use GSM8K dataset. False = generate random tokens directly

    Returns:
        (prefix_path, dataset_path) - Prefix file path and dataset file path
    """
    base_name = os.path.basename(os.path.normpath(tokenizer_path))
    use_variable_length = (
        (length_mean is not None and length_std is not None)
        or (length_min is not None and length_max is not None)
    )

    # Data source tag for file naming
    source_tag = "GSM8K" if use_gsm8k else "Random"

    # Helper function to get dataset based on mode
    def _get_dataset(length: int, count: int, prefix_flag_val: int) -> Optional[List[str]]:
        if use_gsm8k:
            return create_dataset(tokenizer_path, length, count, prefix_flag_val, gsm8k_path)
        else:
            return create_dataset_from_random_tokens(tokenizer_path, length, count, seed)

    # ========== Normal dataset (no prefix) ==========
    if prefix_flag == 0:
        if use_variable_length:
            rng = random.Random(seed)
            real_lens = [
                sample_target_length(rng, input_len, length_mean, length_std, length_min, length_max)
                for _ in range(number)
            ]
            max_len = max(real_lens)
            long_texts = _get_dataset(max_len, number, 0)

            if long_texts is None:
                logging.error("Dataset generation failed")
                return "", ""

            tokenizer = _get_tokenizer(tokenizer_path)
            dataset = []

            pbar = tqdm(total=number, desc="Truncating to variable lengths", unit="row") if tqdm else None
            for i, rl in enumerate(real_lens):
                adjusted = _truncate_or_pad_text(tokenizer, long_texts[i], rl)
                dataset.append(adjusted)
                if pbar:
                    pbar.update(1)
            if pbar:
                pbar.close()

            length_tag = _build_length_tag(input_len, length_mean, length_std, length_min, length_max)
            dataset_path = os.path.join(save_path, f'{source_tag}-{length_tag}-num{number}-{base_name}.jsonl')
            write_data(dataset_path, dataset, number)
            return "", dataset_path
        else:
            dataset = _get_dataset(input_len, number, 0)
            if dataset is None:
                return "", ""
            dataset_path = os.path.join(save_path, f'{source_tag}-in{input_len}-num{number}-{base_name}.jsonl')
            write_data(dataset_path, dataset, number)
            return "", dataset_path

    # ========== Prefix dataset ==========
    if use_variable_length:
        return _create_prefix_dataset_variable(
            tokenizer_path, input_len, number, save_path, base_name, dp,
            repeat_rate, seed, prefix_num, length_mean, length_std,
            length_min, length_max, gsm8k_path, use_gsm8k
        )

    # -------- Fixed-length prefix dataset --------
    prefix_len = int(input_len * repeat_rate)
    prefix_data = _get_dataset(prefix_len, prefix_num, 1)

    if prefix_data is None and repeat_rate > 0:
        logging.error("Prefix dataset generation failed")
        return "", ""

    prefix_dataset = []
    for i in range(prefix_num):
        for j in range(dp):
            prefix_dataset.append(prefix_data[i])

    prefix_path = os.path.join(save_path, f'prefix-{source_tag}-in{prefix_len}-num{dp*prefix_num}-{base_name}.jsonl')
    write_data(prefix_path, prefix_dataset, dp * prefix_num)

    if repeat_rate >= 1:
        dataset_path = os.path.join(save_path, f'{source_tag}-in{prefix_len}-num{number}-{base_name}-repeatRate{repeat_rate}.jsonl')
        write_data(dataset_path, prefix_dataset, number)
        return prefix_path, dataset_path

    # Insert 3 random tokens after prefix (use tokenizer.get_some_tokens for consistency)
    tokenizer = _get_tokenizer(tokenizer_path)
    uniq_token_set = []
    for i in range(number):
        uniq_token_set.append(tokenizer.get_some_tokens(3, seed=seed + i + 1000))

    suffix_len = int(input_len - prefix_len - 3)
    suffix_dataset = _get_dataset(suffix_len, number, 0)

    if suffix_dataset is None:
        logging.error("Suffix dataset generation failed")
        return "", ""

    # Stitch complete dataset
    dataset = []
    pbar = tqdm(total=number, desc="Stitching dataset", unit="row") if tqdm else None
    for data_len in range(number):
        single_data = prefix_data[data_len % prefix_num] + uniq_token_set[data_len] + suffix_dataset[data_len]
        dataset.append(single_data)
        if pbar:
            pbar.update(1)
    if pbar:
        pbar.close()

    dataset_path = os.path.join(save_path, f'{source_tag}-in{input_len}-num{number}-{base_name}-repeatRate{repeat_rate}.jsonl')
    write_data(dataset_path, dataset, number)

    return prefix_path, dataset_path


def _create_prefix_dataset_variable(
    tokenizer_path: str,
    input_len: int,
    number: int,
    save_path: str,
    base_name: str,
    dp: int,
    repeat_rate: float,
    seed: int,
    prefix_num: int,
    length_mean: Optional[int],
    length_std: Optional[float],
    length_min: Optional[int],
    length_max: Optional[int],
    gsm8k_path: Optional[str],
    use_gsm8k: bool = True
) -> Tuple[str, str]:
    """Variable-length prefix dataset generation"""
    rng = random.Random(seed)
    source_tag = "GSM8K" if use_gsm8k else "Random"

    # Helper function to get dataset based on mode
    def _get_dataset(length: int, count: int, prefix_flag_val: int) -> Optional[List[str]]:
        if use_gsm8k:
            return create_dataset(tokenizer_path, length, count, prefix_flag_val, gsm8k_path)
        else:
            return create_dataset_from_random_tokens(tokenizer_path, length, count, seed)

    real_lens = [
        sample_target_length(rng, input_len, length_mean, length_std, length_min, length_max)
        for _ in range(number)
    ]
    common_lens = [max(0, min(rl, int(round(rl * repeat_rate)))) for rl in real_lens]
    max_common_len = max(common_lens) if common_lens else 0

    prefix_data = []
    if max_common_len > 0:
        prefix_data = _get_dataset(max_common_len, prefix_num, 1)
        if prefix_data is None:
            logging.error("Prefix dataset generation failed")
            return "", ""
    else:
        prefix_data = [""] * prefix_num

    prefix_dataset = []
    for i in range(prefix_num):
        for j in range(dp):
            prefix_dataset.append(prefix_data[i])

    prefix_path = os.path.join(save_path, f'prefix-{source_tag}-in{max_common_len}-num{dp*prefix_num}-{base_name}.jsonl')
    write_data(prefix_path, prefix_dataset, dp * prefix_num)

    if repeat_rate >= 1:
        dataset_path = os.path.join(save_path, f'{source_tag}-in{max_common_len}-num{number}-{base_name}-repeatRate{repeat_rate}.jsonl')
        write_data(dataset_path, prefix_dataset, number)
        return prefix_path, dataset_path

    # Use tokenizer.get_some_tokens for unique tokens (more consistent)
    tokenizer = _get_tokenizer(tokenizer_path)
    uniq_token_set = []
    for i in range(number):
        uniq_token_set.append(tokenizer.get_some_tokens(3, seed=seed + i + 1000))

    max_suffix_len = max(rl - cl - 3 for rl, cl in zip(real_lens, common_lens))
    if max_suffix_len < 1:
        max_suffix_len = 1

    suffix_pool = _get_dataset(max_suffix_len, number, 0)
    if suffix_pool is None:
        logging.error("Suffix dataset generation failed")
        return "", ""

    dataset = []
    pbar = tqdm(total=number, desc="Stitching dataset (variable)", unit="row") if tqdm else None
    for idx in range(number):
        rl = real_lens[idx]
        cl = common_lens[idx]
        suffix_len_needed = max(0, rl - cl - 3)

        prefix_text = prefix_data[idx % prefix_num]
        if cl > 0 and prefix_text:
            prefix_text = _truncate_or_pad_text(tokenizer, prefix_text, cl)
        else:
            prefix_text = ""

        suffix_text = suffix_pool[idx]
        if suffix_len_needed > 0 and suffix_text:
            suffix_text = _truncate_or_pad_text(tokenizer, suffix_text, suffix_len_needed)
        else:
            suffix_text = ""

        single_data = prefix_text + uniq_token_set[idx] + suffix_text
        dataset.append(single_data)
        if pbar:
            pbar.update(1)
    if pbar:
        pbar.close()

    length_tag = _build_length_tag(input_len, length_mean, length_std, length_min, length_max)
    dataset_path = os.path.join(save_path, f'{source_tag}-{length_tag}-num{number}-{base_name}-repeatRate{repeat_rate}.jsonl')
    write_data(dataset_path, dataset, number)

    logging.info(f"  max_common_len={max_common_len}, max_suffix_len={max_suffix_len}")
    logging.info(f"  avg_hit_ratio={sum(c / r for c, r in zip(common_lens, real_lens)) / len(real_lens):.2%}")

    return prefix_path, dataset_path


def parse_prefix_ratio(r: str) -> float:
    """
    Parse prefix repeat rate parameter
    "50%" -> 0.5, "0.5" -> 0.5, "0.500" -> 0.5
    """
    r = str(r).strip()
    if r.endswith("%"):
        v = float(r[:-1]) / 100.0
    else:
        v = float(r)
    if not (0.0 <= v <= 1.0):
        raise ValueError("prefix-ratio must be in [0,1] range or percentage [0%,100%]")
    return v