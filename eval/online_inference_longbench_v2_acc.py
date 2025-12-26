#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import sys
import time
import re
import argparse
import hashlib
import requests
from pathlib import Path
from typing import Optional
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


def call_model_online(
    max_new_tokens: int,
    messages: list,
    model: str,
    llm_url: str,
    temperature: float = 0.0,
    timeout_s: int = 30
) -> str:
    """
    调用在线模型 API，使用流式请求
    
    Args:
        max_new_tokens: 最大生成 token 数
        messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
        model: 模型名称
        llm_url: LLM 服务的基础 URL（例如：http://127.0.0.1:7800/v1）
        temperature: 温度参数
        timeout_s: 连接超时时间（秒）
    
    Returns:
        模型生成的文本
    """
    model_url = f"{llm_url}/chat/completions"
    headers = {'Content-Type': 'application/json'}
    
    body = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": temperature,
        "max_tokens": max_new_tokens,
        "top_p": 0.95,
        "ignore_eos": False,
    }
    
    model_answer = ""
    max_tries = 5
    tries = 0
    
    while tries < max_tries:
        tries += 1
        try:
            timeout_config = (timeout_s, None)
            with requests.post(
                model_url, 
                json=body, 
                headers=headers, 
                verify=False, 
                stream=True, 
                timeout=timeout_config
            ) as response:
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if line:
                        temp = line.decode('utf-8')
                        try:
                            if temp.startswith('data: '):
                                temp = temp[6:]
                            elif temp.startswith('data:'):
                                temp = temp[5:]
                            
                            if temp.strip() == '[DONE]':
                                break
                            
                            temp_json = json.loads(temp)
                            if 'choices' in temp_json and len(temp_json['choices']) > 0:
                                delta = temp_json['choices'][0].get('delta', {})
                                if 'content' in delta and delta['content']:
                                    model_answer += delta['content']
                        except (json.JSONDecodeError, KeyError, ValueError):
                            pass
            
            return model_answer
            
        except requests.exceptions.HTTPError as e:
            # 获取详细的错误信息
            error_detail = ""
            try:
                if hasattr(e.response, 'text'):
                    error_detail = f" Response: {e.response.text[:500]}"
                elif hasattr(e.response, 'content'):
                    error_detail = f" Response: {e.response.content[:500]}"
            except:
                pass
            
            if tries < max_tries:
                print(f"Error Occurs: \"{e}\"{error_detail}        Retry ...")
                time.sleep(2)
            else:
                print(f"Max tries. Failed with error: {e}{error_detail}")
                # 打印请求信息用于调试
                print(f"Request URL: {model_url}")
                print(f"Request body (truncated): {str(body)[:500]}")
                raise e
        except requests.exceptions.Timeout as e:
            if tries < max_tries:
                print(f"Error Occurs: \"{e}\" (Connection/Read timed out). Retry ...")
                time.sleep(5)
            else:
                raise e
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            if tries < max_tries:
                print(f"Error Occurs: \"{e}\"        Retry ...")
                time.sleep(1)
            else:
                print(f"Max tries. Failed with error: {e}")
                # 打印请求信息用于调试
                print(f"Request URL: {model_url}")
                print(f"Request body (truncated): {str(body)[:500]}")
                raise e
    
    return model_answer


def query_llm(
    prompt: str,
    model: str,
    tokenizer,
    llm_url: str,
    max_new_tokens: int = 16384,
    max_model_len: int = 131072,
    temperature: float = 0.1,
    timeout_s: int = 30,
) -> str:
    if tokenizer is not None and max_model_len is not None:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        ori_ids_len= len(ids)
        if ori_ids_len > max_model_len:
            first = int(max_model_len * 0.5)
            last = max_model_len - first
            ids = ids[:first] + ids[-last:]
            prompt = tokenizer.decode(ids, skip_special_tokens=True)
            print(f"[Truncate] {ori_ids_len} → {len(ids)} tokens")

    messages = [{"role": "user", "content": prompt}]

    return call_model_online(
        max_new_tokens=max_new_tokens,
        messages=messages,
        model=model,
        llm_url=llm_url,
        temperature=temperature,
        timeout_s=timeout_s,
    )


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


def load_longbench_v2_data(dataset_path: Optional[str] = None):
    """Load LongBench v2 dataset from HuggingFace or local file"""
    if dataset_path and os.path.isfile(dataset_path):
        # Load from local JSON file
        print(f"Loading dataset from local file: {dataset_path}")
        with open(dataset_path, "r", encoding="utf-8") as f:
            if dataset_path.endswith(".jsonl"):
                data = [json.loads(line) for line in f]
            else:
                data = json.load(f)
        return data
    else:
        # Load from HuggingFace
        print("Loading dataset from HuggingFace: THUDM/LongBench-v2")
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


def process_single_item(item, args, template, tokenizer):
    """
    处理单个数据项（用于并发处理）
    
    Args:
        item: 数据项字典
        args: 命令行参数
        template: 提示词模板
        tokenizer: tokenizer 对象
    
    Returns:
        (result_item, status): 结果字典和状态字符串 ("success" 或 "error")
    """
    try:
        # Build prompt
        prompt = build_prompt(item, template)
        
        # First generation (CoT reasoning if enabled)
        if args.cot:
            output = query_llm(
                prompt=prompt,
                model=args.model,
                tokenizer=tokenizer,
                llm_url=args.llm_url,
                max_new_tokens=1024,
                max_model_len=None if args.no_truncate else args.max_model_len,
                temperature=args.temperature,
                timeout_s=args.timeout
            )
            
            # Extract answer directly from CoT response
            cot_response = output.strip()
            response = cot_response
        else:
            output = query_llm(
                prompt=prompt,
                model=args.model,
                tokenizer=tokenizer,
                llm_url=args.llm_url,
                max_new_tokens=args.max_tokens,
                max_model_len=None if args.no_truncate else args.max_model_len,
                temperature=args.temperature,
                timeout_s=args.timeout
            )
            response = output.strip()
            cot_response = None

        # Build result item
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
            result_item["response_cot"] = cot_response
        
        result_item["response"] = response

        # Extract predicted answer
        result_item["pred"] = extract_answer(result_item["response"])
        result_item["judge"] = result_item["pred"] == item["answer"]

        return result_item, "success"
        
    except Exception as e:
        return {"error": str(e), "_id": item.get("_id", "unknown")}, "error"


def main():
    parser = argparse.ArgumentParser(description="Online inference for LongBench v2")
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("MODEL", "Qwen2.5-14B-Instruct"),
        help="Model name for API",
    )
    parser.add_argument(
        "--llm_url",
        type=str,
        default=os.getenv("LLM_URL", "http://127.0.0.1:7800/v1"),
        help="LLM service base URL",
    )
    parser.add_argument(
        "--template",
        type=str,
        default=None,
        help="Path to prompt template file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset file (optional, will load from HuggingFace if not set)",
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
        "--max_tokens",
        type=int,
        default=16384,
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
        help="Filter by domain(s). Can specify multiple domains separated by comma.",
    )
    parser.add_argument(
        "--max_context_length",
        type=int,
        default=None,
        help="Maximum context length (in tokens) to process. "
             "Samples with context longer than this will be filtered out.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (None for all)",
    )
    parser.add_argument(
        "--local_tokenizer",
        type=str,
        default="",
        help="Local tokenizer path (optional)",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=131072,
        help="Model's maximum context length for truncation (default: 131072)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Connection timeout in seconds",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--no_truncate",
        action="store_true",
        help="Disable prompt truncation",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent requests (default: 1, sequential processing)",
    )
    args = parser.parse_args()

    # Determine template path
    if args.template is None:
        # Try to find template in common locations
        code_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        template_path = os.path.join(code_root, "eval", "prompts", "0shot.txt")
        if not os.path.isfile(template_path):
            template_path = "/home/externals/wangwenxin21/caz/LongBench/prompts/0shot.txt"
    else:
        template_path = args.template

    # Determine template path based on args
    if args.cot:
        template_path = template_path.replace("0shot.txt", "0shot_cot.txt")
    elif args.no_context:
        template_path = template_path.replace("0shot.txt", "0shot_no_context.txt")

    # Load prompt template
    template = load_prompt_template(template_path)

    # Load tokenizer
    tokenizer = None
    if args.local_tokenizer and args.local_tokenizer != '':
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                args.local_tokenizer, 
                trust_remote_code=True
            )
            print(f"已从本地路径加载 tokenizer: {args.local_tokenizer}")
        except Exception as e:
            print(f"警告: 无法从本地路径加载 tokenizer: {args.local_tokenizer}, 错误: {e}")
            tokenizer = None
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model, 
                trust_remote_code=True
            )
            print(f"已从模型名称加载 tokenizer: {args.model}")
        except Exception as e:
            print(f"警告: 无法从模型名称加载 tokenizer: {args.model}, 错误: {e}")
            print("将不使用 tokenizer（仅用于截断，可能影响性能）")
            tokenizer = None

    # Load dataset
    data = load_longbench_v2_data(args.dataset)
    original_count = len(data)
    
    # Filter by domain if specified
    if args.domain:
        domains = [d.strip() for d in args.domain.split(",")]
        data = [item for item in data if item.get("domain") in domains]
        print(f"Filtered by domain(s): {domains}")
        print(f"Sample count after domain filter: {original_count} -> {len(data)}")
        if len(data) == 0:
            print(f"错误: 没有找到指定 domain 的数据: {domains}")
            sys.exit(1)
        # Log domain distribution
        domain_counts = {}
        for item in data:
            domain = item.get("domain", "Unknown")
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        print(f"Domain distribution: {domain_counts}")
    
    # Filter by context length if specified
    if args.max_context_length is not None and tokenizer is not None:
        print(f"Filtering samples by context length (max: {args.max_context_length} tokens)")
        
        filtered_data = []
        skipped_count = 0
        
        for item in tqdm(data, desc="Calculating context lengths"):
            # Build prompt to calculate token length
            prompt = build_prompt(item, template)
            # Apply chat template to get actual length
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                add_special_tokens=True,
            )
            
            # Calculate token length
            tokens = tokenizer.encode(formatted_prompt, add_special_tokens=False)
            context_length = len(tokens)
            
            if context_length <= args.max_context_length:
                filtered_data.append(item)
            else:
                skipped_count += 1
        
        data = filtered_data
        print(f"Sample count after length filter: {len(data) + skipped_count} -> {len(data)}")
        print(f"Skipped {skipped_count} samples with context length > {args.max_context_length}")
        
        if len(data) == 0:
            print(f"错误: 没有找到符合长度要求的数据")
            sys.exit(1)
    
    if args.max_samples:
        data = data[: args.max_samples]
    
    print(f"Loaded {len(data)} samples from dataset")

    # Determine output file
    model_name = os.path.basename(args.model.rstrip("/"))
    
    # Create domain suffix for filename if domain is specified
    domain_suffix = ""
    if args.domain:
        domains = [d.strip() for d in args.domain.split(",")]
        domain_suffix = "_" + "_".join([d.replace(" ", "_").replace("-", "_") for d in domains])
        if len(domain_suffix) > 100:  # Limit length
            domain_suffix = "_" + hashlib.md5(args.domain.encode()).hexdigest()[:8]
    
    # 从环境变量获取时间戳
    timestamp = os.getenv("TIMESTAMP", "")
    timestamp_suffix = f"_{timestamp}" if timestamp else ""
    
    if args.cot:
        out_file = os.path.join(args.save_dir, f"{model_name}_longbench_v2{domain_suffix}_cot{timestamp_suffix}.jsonl")
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
                base_pattern = f"{model_name}_longbench_v2{domain_suffix}_cot"
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
                print(f"Resuming from existing file (latest): {resume_file}")
                with open(resume_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            item = json.loads(line)
                            processed_ids.add(item["_id"])
                        except:
                            pass
                print(f"Found {len(processed_ids)} already processed samples")
        elif os.path.exists(out_file):
            # 没有时间戳时，使用原来的逻辑
            print(f"Resuming from existing file: {out_file}")
            with open(out_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        processed_ids.add(item["_id"])
                    except:
                        pass
            print(f"Found {len(processed_ids)} already processed samples")

    # Filter out processed samples
    data = [item for item in data if item["_id"] not in processed_ids]
    print(f"Processing {len(data)} remaining samples")

    # Process data
    print(f"开始推理，结果将保存到: {out_file}")
    print(f"LLM URL: {args.llm_url}")
    print(f"模型: {args.model}")
    print(f"并发数: {args.concurrency}")
    print("-" * 50)

    # 线程锁用于文件写入
    write_lock = threading.Lock()
    
    # 打开输出文件（追加模式）
    fout = open(out_file, "a", encoding="utf-8")
    
    total = len(data)
    processed = 0
    errors = 0
    
    # 使用线程池进行并发处理
    if args.concurrency > 1:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            # 提交所有任务
            future_to_item = {
                executor.submit(process_single_item, item, args, template, tokenizer): item
                for item in data
            }
            
            # 使用 tqdm 显示进度
            try:
                with tqdm(total=total, desc="推理进度") as pbar:
                    for future in as_completed(future_to_item):
                        result_item, status = future.result()
                        
                        if status == "error":
                            errors += 1
                            error_msg = result_item.get("error", "Unknown error")
                            item_id = result_item.get("_id", "unknown")
                            print(f"\n错误: 处理数据 {item_id} 时出错: {error_msg}")
                        elif status == "success":
                            # 线程安全地写入文件
                            with write_lock:
                                fout.write(json.dumps(result_item, ensure_ascii=False) + "\n")
                                fout.flush()
                            processed += 1
                        
                        pbar.update(1)
            except KeyboardInterrupt:
                print("\n\n用户中断，正在保存已处理的结果...")
                # 取消未完成的任务
                for future in future_to_item:
                    future.cancel()
    else:
        # 顺序处理（原有逻辑）
        try:
            for item in tqdm(data, desc="推理进度"):
                result_item, status = process_single_item(item, args, template, tokenizer)
                
                if status == "error":
                    errors += 1
                    error_msg = result_item.get("error", "Unknown error")
                    item_id = result_item.get("_id", "unknown")
                    print(f"\n错误: 处理数据 {item_id} 时出错: {error_msg}")
                    print("继续处理下一条数据...")
                    continue
                elif status == "success":
                    fout.write(json.dumps(result_item, ensure_ascii=False) + "\n")
                    fout.flush()
                    processed += 1
        except KeyboardInterrupt:
            print("\n\n用户中断，正在保存已处理的结果...")
    
    fout.close()

    print("\n" + "=" * 50)
    print(f"推理完成！")
    print(f"总数据: {total}")
    print(f"已处理: {processed}")
    print(f"错误数: {errors}")
    print(f"结果文件: {out_file}")
    print("=" * 50)


if __name__ == "__main__":
    main()

