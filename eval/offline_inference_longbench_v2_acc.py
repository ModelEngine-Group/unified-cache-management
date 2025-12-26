import contextlib
import hashlib
import json
import os
import sys
import time
import re
import argparse
from dataclasses import asdict
from typing import Optional

from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# Third Party
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs

from ucm.logger import init_logger

logger = init_logger(__name__)
model = ""
path_to_dataset = ""
data_dir = ""
tokenizer = None
ucm_sparse_config = None
max_model_len = None  # Will be set when LLM is initialized


def setup_environment_variables():
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["PYTHONHASHSEED"] = "123456"
    os.environ["ENABLE_SPARSE"] = "true"

    global model, path_to_dataset, data_dir, ucm_sparse_config, tokenizer
    model = os.getenv("MODEL_PATH", "/home/models/Qwen2.5-14B-Instruct")
    if not os.path.isdir(model):
        model = input("Enter path to model, e.g. /home/models/Qwen2.5-14B-Instruct: ")
        if not os.path.isdir(model):
            print("Exiting. Incorrect model_path")
            sys.exit(1)

    # LongBench v2 can be loaded from HuggingFace or local JSON file
    path_to_dataset = os.getenv("DATASET_FILE", None)
    if path_to_dataset and not os.path.isfile(path_to_dataset):
        path_to_dataset = None

    data_dir = os.getenv("STORAGE_BACKENDS", "/home/data/kv_cache")
    if not os.path.isdir(data_dir):
        data_dir = input(
            "Enter the directory for UCMStore to save kv cache, e.g. /home/data/kv_cache: "
        )
        if not os.path.isdir(data_dir):
            create = input(f"Directory {data_dir} does not exist. Create it? (Y/n): ")
            if create.lower() == "y":
                os.makedirs(data_dir, exist_ok=True)
            else:
                print("Exiting. Directory not created.")
                sys.exit(1)

    sparse_config_path = os.getenv("UCM_SPARSE_CONFIG", "eval/ucm_sparse_config.json")
    if not os.path.isfile(sparse_config_path):
        sparse_config_path = input(
            "Enter path to one of the sparse config json, e.g. eval/ucm_sparse_config.json: "
        )
        if not os.path.isfile(sparse_config_path):
            print("Exiting. Incorrect config json file path")
            sys.exit(1)

    with open(sparse_config_path, "r", encoding="utf-8") as f:
        ucm_sparse_config = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)


@contextlib.contextmanager
def build_llm_with_uc(module_path: str, name: str, model: str):
    ktc = KVTransferConfig(
        kv_connector=name,
        kv_connector_module_path=module_path,
        kv_role="kv_both",
        kv_connector_extra_config={
            "ucm_connectors": [
                {
                    "ucm_connector_name": "UcmNfsStore",
                    "ucm_connector_config": {
                        "storage_backends": data_dir,
                        "use_direct": False,
                    },
                }
            ],
            "ucm_sparse_config": ucm_sparse_config,
        },
    )

    llm_args = EngineArgs(
        model=model,
        # kv_transfer_config=ktc,
        max_model_len=32768,
        gpu_memory_utilization=0.6,
        max_num_batched_tokens=30000,
        block_size=128,
        enforce_eager=True,
        trust_remote_code=True,
        distributed_executor_backend="mp",
        tensor_parallel_size=1,
    )

    # Store max_model_len globally for truncation
    global max_model_len
    max_model_len = llm_args.max_model_len

    llm = LLM(**asdict(llm_args))
    try:
        yield llm
    finally:
        logger.info("LLM engine is exiting.")


def load_prompt_template(template_path: str) -> str:
    """Load prompt template from file"""
    if not os.path.isfile(template_path):
        raise FileNotFoundError(f"Template file not found: {template_path}")
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


def build_prompt(
    item: dict,
    template: str,
    cot_response: Optional[str] = None
) -> str:
    """Build prompt from template and data item"""
    prompt = template.replace("$DOC$", item["context"].strip())
    prompt = prompt.replace("$Q$", item["question"].strip())
    prompt = prompt.replace("$C_A$", item["choice_A"].strip())
    prompt = prompt.replace("$C_B$", item["choice_B"].strip())
    prompt = prompt.replace("$C_C$", item["choice_C"].strip())
    prompt = prompt.replace("$C_D$", item["choice_D"].strip())
    
    if cot_response is not None:
        prompt = prompt.replace("$COT$", cot_response)
    
    return prompt


def truncate_prompt(prompt: str) -> str:
    """Truncate prompt by taking first half and last half if it exceeds max_model_len
    
    Similar to pred.py logic: if prompt token length > max_model_len,
    take first max_model_len//2 tokens and last max_model_len//2 tokens.
    """
    global max_model_len, tokenizer
    
    if max_model_len is None or tokenizer is None:
        return prompt
    
    try:
        # Encode prompt to get token IDs
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        
        # If length exceeds max_model_len, truncate by taking first and last halves
        if len(input_ids) > max_model_len:
            first_half = max_model_len // 2
            last_half = max_model_len - first_half
            
            # Take first half and last half
            truncated_ids = input_ids[:first_half] + input_ids[-last_half:]
            
            # Decode back to text
            prompt = tokenizer.decode(truncated_ids, skip_special_tokens=True)
            
            logger.debug(
                f"Truncated prompt: original length={len(input_ids)}, "
                f"truncated length={len(truncated_ids)}, max_model_len={max_model_len}"
            )
        
        return prompt
    except Exception as e:
        logger.warning(f"Error during truncation: {e}, returning original prompt")
        return prompt


def format_prompt_for_model(prompt: str, use_chat_template: bool = True, truncate: bool = True) -> str:
    """Format prompt using tokenizer's chat template with optional truncation
    
    Args:
        prompt: The prompt to format
        use_chat_template: Whether to apply chat template
        truncate: Whether to truncate prompt if it exceeds max_model_len.
                  If True, truncate by taking first half and last half.
                  Similar to pred.py logic.
    """
    global max_model_len, tokenizer
    
    if tokenizer is None:
        raise ValueError("Tokenizer is not initialized. Make sure setup_environment_variables() is called first.")
    
    if not use_chat_template:
        # If not using chat template, truncate the raw prompt if enabled
        if truncate:
            return truncate_prompt(prompt)
        else:
            return prompt
    
    # Apply chat template first
    messages = [{"role": "user", "content": prompt}]
    try:
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            add_special_tokens=True,
        )
        
        # Truncate formatted prompt if it exceeds max_model_len and truncation is enabled
        if truncate and max_model_len is not None:
            input_ids = tokenizer.encode(formatted, add_special_tokens=False)
            if len(input_ids) > max_model_len:
                first_half = max_model_len // 2
                last_half = max_model_len - first_half
                truncated_ids = input_ids[:first_half] + input_ids[-last_half:]
                formatted = tokenizer.decode(truncated_ids, skip_special_tokens=True)
                logger.info(
                    f"Truncated formatted prompt: original length={len(input_ids)}, "
                    f"truncated length={len(truncated_ids)}, max_model_len={max_model_len}"
                )
        
        return formatted
    except Exception as e:
        logger.error(f"Error applying chat template: {e}")
        logger.error(f"Prompt length: {len(prompt)}")
        raise


def extract_answer(response: str) -> Optional[str]:
    """Extract answer (A, B, C, or D) from model response"""
    response = response.replace("*", "")
    # Try to match "The correct answer is (X)" or "The correct answer is X"
    match = re.search(r"The correct answer is\s*\(([A-D])\)", response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    match = re.search(r"The correct answer is\s+([A-D])", response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Try to match just "(A)" or "A" at the end
    match = re.search(r"\(([A-D])\)", response)
    if match:
        return match.group(1).upper()
    
    # Try to match standalone A/B/C/D
    match = re.search(r"\b([A-D])\b", response[-50:])  # Check last 50 chars
    if match:
        return match.group(1).upper()
    
    return None


def generate_output(
    llm: LLM,
    prompts: list[str],
    sampling_params: SamplingParams,
) -> list[str]:
    """Generate outputs from LLM"""
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    logger.info(f"Generation took {time.time() - start:.2f} seconds for {len(prompts)} samples")
    
    generated_texts = []
    for output in outputs:
        generated_text = output.outputs[0].text
        generated_texts.append(generated_text.strip())
    
    return generated_texts


def load_longbench_v2_data(dataset_path: Optional[str] = None):
    """Load LongBench v2 dataset from HuggingFace or local file"""
    if dataset_path and os.path.isfile(dataset_path):
        # Load from local JSON file
        logger.info(f"Loading dataset from local file: {dataset_path}")
        with open(dataset_path, "r", encoding="utf-8") as f:
            if dataset_path.endswith(".jsonl"):
                data = [json.loads(line) for line in f]
            else:
                data = json.load(f)
        return data
    else:
        # Load from HuggingFace
        logger.info("Loading dataset from HuggingFace: THUDM/LongBench-v2")
        dataset = load_dataset("THUDM/LongBench-v2", split="train")
        data = [
            {
                "_id": item["_id"],
                "domain": item["domain"],
                "sub_domain": item["sub_domain"],
                "difficulty": item["difficulty"],
                "length": item["length"],
                "question": item["question"],
                "choice_A": item["choice_A"],
                "choice_B": item["choice_B"],
                "choice_C": item["choice_C"],
                "choice_D": item["choice_D"],
                "answer": item["answer"],
                "context": item["context"],
            }
            for item in dataset
        ]
        return data


def main():
    parser = argparse.ArgumentParser(description="Offline inference for LongBench v2")
    parser.add_argument(
        "--template",
        type=str,
        default="/home/externals/wangwenxin21/caz/LongBench/prompts/0shot.txt",
        help="Path to prompt template file",
    )
    parser.add_argument(
        "--cot",
        action="store_true",
        help="Enable Chain-of-Thought evaluation",
    )
    parser.add_argument(
        "--no_context",
        action="store_true",
        help="Test without context (pure memorization)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=20,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (None for all)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Filter by domain(s). Can specify multiple domains separated by comma. "
             "Available domains: 'Long In-context Learning', 'Single-Document QA', "
             "'Multi-Document QA', 'Long-dialogue History Understanding', "
             "'Code Repository Understanding', 'Long Structured Data Understanding'",
    )
    parser.add_argument(
        "--max_context_length",
        type=int,
        default=None,
        help="Maximum context length (in tokens) to process. "
             "Samples with context longer than this will be filtered out. "
             "If not specified, all samples will be processed.",
    )
    parser.add_argument(
        "--no_truncate",
        action="store_true",
        help="Disable prompt truncation. By default, prompts longer than max_model_len "
             "will be truncated by taking first half and last half.",
    )
    args = parser.parse_args()

    module_path = "ucm.integration.vllm.ucm_connector"
    name = "UCMConnector"
    setup_environment_variables()

    # Determine template path based on args
    if args.cot:
        template_path = args.template.replace("0shot.txt", "0shot_cot.txt")
        cot_ans_template_path = args.template.replace("0shot.txt", "0shot_cot_ans.txt")
    elif args.no_context:
        template_path = args.template.replace("0shot.txt", "0shot_no_context.txt")
        cot_ans_template_path = None
    else:
        template_path = args.template
        cot_ans_template_path = None

    # Load prompt template (needed for length filtering)
    template = load_prompt_template(template_path)
    if args.cot:
        cot_ans_template = load_prompt_template(cot_ans_template_path)

    # Load dataset
    data = load_longbench_v2_data(path_to_dataset)
    original_count = len(data)
    
    # Filter by domain if specified
    if args.domain:
        domains = [d.strip() for d in args.domain.split(",")]
        data = [item for item in data if item.get("domain") in domains]
        logger.info(f"Filtered by domain(s): {domains}")
        logger.info(f"Sample count after domain filter: {original_count} -> {len(data)}")
        if len(data) == 0:
            logger.error(f"No samples found for domain(s): {domains}")
            sys.exit(1)
        # Log domain distribution
        domain_counts = {}
        for item in data:
            domain = item.get("domain", "Unknown")
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        logger.info(f"Domain distribution: {domain_counts}")
    
    # Filter by context length if specified
    if args.max_context_length is not None:
        logger.info(f"Filtering samples by context length (max: {args.max_context_length} tokens)")
        
        filtered_data = []
        skipped_count = 0
        context_lengths = []
        
        for item in tqdm(data, desc="Calculating context lengths"):
            # Build prompt to calculate token length
            prompt = build_prompt(item, template)
            # Don't truncate when calculating context length for filtering
            formatted_prompt = format_prompt_for_model(prompt, use_chat_template=True, truncate=False)
            
            # Calculate token length
            tokens = tokenizer.encode(formatted_prompt, add_special_tokens=False)
            context_length = len(tokens)
            
            context_lengths.append(context_length)
            
            if context_length <= args.max_context_length:
                filtered_data.append(item)
            else:
                skipped_count += 1
                if skipped_count <= 5:  # Log first 5 skipped samples
                    logger.debug(
                        f"Skipped sample {item.get('_id', 'unknown')}: "
                        f"context_length={context_length} > max={args.max_context_length}"
                    )
        # print(f"context_lengths is {context_lengths}")
        # logger.info(
        #             f"Context length statistics: "
        #             f"min={min(context_lengths)}, max={max(context_lengths)}, "
        #             f"mean={statistics.mean(context_lengths):.1f}, "
        #             f"median={statistics.median(context_lengths):.1f}"
        #         )


        data = filtered_data
        logger.info(f"Sample count after length filter: {len(data) + skipped_count} -> {len(data)}")
        logger.info(f"Skipped {skipped_count} samples with context length > {args.max_context_length}")
        
        if len(data) == 0:
            logger.error(f"No samples found with context length <= {args.max_context_length}")
            sys.exit(1)
        
        # Log length statistics
        if context_lengths:
            import statistics
            valid_lengths = [l for l in context_lengths if l <= args.max_context_length]
            if valid_lengths:
                logger.info(
                    f"Context length statistics (after filter): "
                    f"min={min(valid_lengths)}, max={max(valid_lengths)}, "
                    f"mean={statistics.mean(valid_lengths):.1f}, "
                    f"median={statistics.median(valid_lengths):.1f}"
                )
    
    if args.max_samples:
        data = data[: args.max_samples]
    
    logger.info(f"Loaded {len(data)} samples from dataset")

    # Determine output file
    model_name = os.path.basename(model.rstrip("/"))
    
    # Create domain suffix for filename if domain is specified
    domain_suffix = ""
    if args.domain:
        # Convert domain names to filename-friendly format
        domains = [d.strip() for d in args.domain.split(",")]
        domain_suffix = "_" + "_".join([d.replace(" ", "_").replace("-", "_") for d in domains])
        if len(domain_suffix) > 100:  # Limit length
            domain_suffix = "_" + hashlib.md5(args.domain.encode()).hexdigest()[:8]
    
    # 从环境变量获取时间戳
    timestamp = os.getenv("TIMESTAMP", "")
    timestamp_suffix = f"_{timestamp}" if timestamp else ""
    
    if args.cot:
        out_file = os.path.join(args.save_dir, f"{model_name}_longbench_v2{domain_suffix}_cot_{timestamp_suffix}.jsonl")
    elif args.no_context:
        out_file = os.path.join(args.save_dir, f"{model_name}_longbench_v2{domain_suffix}_no_context{timestamp_suffix}.jsonl")
    else:
        out_file = os.path.join(args.save_dir, f"{model_name}_longbench_v2{domain_suffix}{timestamp_suffix}.jsonl")
    
    os.makedirs(args.save_dir, exist_ok=True)

    # Resume from existing file
    processed_ids = set()
    if args.resume:
        # 如果有时间戳，查找最新的同名文件（不带时间戳的）
        if timestamp:
            # 构建不带时间戳的文件名模式
            import glob
            if args.cot:
                base_pattern = f"{model_name}_longbench_v2{domain_suffix}_cot_esa_1_2_0.3_10"
            elif args.no_context:
                base_pattern = f"{model_name}_longbench_v2{domain_suffix}_no_context"
            else:
                base_pattern = f"{model_name}_longbench_v2{domain_suffix}"
            # 查找所有匹配的文件
            pattern = os.path.join(args.save_dir, f"{base_pattern}_*.jsonl")
            matching_files = glob.glob(pattern)
            if matching_files:
                # 按修改时间排序，获取最新的文件
                resume_file = max(matching_files, key=os.path.getmtime)
                logger.info(f"Resuming from existing file (latest): {resume_file}")
                with open(resume_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            item = json.loads(line)
                            processed_ids.add(item["_id"])
                        except:
                            pass
                logger.info(f"Found {len(processed_ids)} already processed samples")
        elif os.path.exists(out_file):
            # 没有时间戳时，使用原来的逻辑
            logger.info(f"Resuming from existing file: {out_file}")
            with open(out_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        processed_ids.add(item["_id"])
                    except:
                        pass
            logger.info(f"Found {len(processed_ids)} already processed samples")

    # Filter out processed samples
    data = [item for item in data if item["_id"] not in processed_ids]
    logger.info(f"Processing {len(data)} remaining samples")

    try:
        with build_llm_with_uc(module_path, name, model) as llm:
            with open(out_file, "a", encoding="utf-8") as fout:
                for start_idx in tqdm(range(0, len(data), args.batch_size), desc="Processing batches"):
                    end_idx = min(start_idx + args.batch_size, len(data))
                    current_batch = data[start_idx:end_idx]

                    prompts = []
                    for item in current_batch:
                        try:
                            prompt = build_prompt(item, template)
                            # Use truncate flag from args (default True, False if --no_truncate is set)
                            formatted_prompt = format_prompt_for_model(prompt, truncate=not args.no_truncate)
                            prompts.append(formatted_prompt)
                        except Exception as e:
                            logger.error(f"Error processing item {item.get('_id', 'unknown')}: {e}")
                            raise

                    # First generation (CoT reasoning if enabled)
                    sampling_params = SamplingParams(
                        temperature=args.temperature,
                        top_p=0.95,
                        max_tokens=1024 if args.cot else args.max_tokens,
                        ignore_eos=False,
                    )

                    outputs = generate_output(llm, prompts, sampling_params)

                    # Process outputs
                    for item, output in zip(current_batch, outputs):
                        result_item = {
                            "_id": item["_id"],
                            "domain": item["domain"],
                            "sub_domain": item["sub_domain"],
                            "difficulty": item["difficulty"],
                            "length": item["length"],
                            "question": item["question"],
                            "choice_A": item["choice_A"],
                            "choice_B": item["choice_B"],
                            "choice_C": item["choice_C"],
                            "choice_D": item["choice_D"],
                            "answer": item["answer"],
                        }

                        if args.cot:
                            # Extract answer from CoT response
                            result_item["response_cot"] = output
                            cot_prompt = build_prompt(item, cot_ans_template, cot_response=output)
                            # Use truncate flag from args (default True, False if --no_truncate is set)
                            cot_formatted_prompt = format_prompt_for_model(cot_prompt, truncate=not args.no_truncate)
                            
                            # Second generation to get final answer
                            cot_sampling_params = SamplingParams(
                                temperature=args.temperature,
                                top_p=0.95,
                                max_tokens=args.max_tokens,
                                ignore_eos=False,
                            )
                            final_outputs = generate_output(llm, [cot_formatted_prompt], cot_sampling_params)
                            result_item["response"] = final_outputs[0] if final_outputs else ""
                        else:
                            result_item["response"] = output

                        # Extract predicted answer
                        result_item["pred"] = extract_answer(result_item["response"])
                        result_item["judge"] = result_item["pred"] == item["answer"]

                        fout.write(json.dumps(result_item, ensure_ascii=False) + "\n")
                        fout.flush()
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

    logger.info(f"Results saved to: {out_file}")


if __name__ == "__main__":
    main()


