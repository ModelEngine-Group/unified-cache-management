#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import sys
import time
import argparse
import requests
from pathlib import Path
from transformers import AutoTokenizer
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
        messages: 消息列表，格式为 [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
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
            # For streaming requests, use (connect_timeout, read_timeout) tuple
            # Set read_timeout to None to disable read timeout for streaming
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
                
                # 按行读取流式数据
                for line in response.iter_lines():
                    if line:
                        temp = line.decode('utf-8')
                        try:
                            # 跳过 "data: " 前缀
                            if temp.startswith('data: '):
                                temp = temp[6:]
                            elif temp.startswith('data:'):
                                temp = temp[5:]
                            
                            # 处理 [DONE] 消息
                            if temp.strip() == '[DONE]':
                                break
                            
                            temp_json = json.loads(temp)
                            if 'choices' in temp_json and len(temp_json['choices']) > 0:
                                delta = temp_json['choices'][0].get('delta', {})
                                if 'content' in delta and delta['content']:
                                    model_answer += delta['content']
                        except json.JSONDecodeError:
                            pass
                        except KeyError:
                            pass
                        except ValueError:
                            # 处理可能的解析错误
                            pass
            
            # 如果成功获取响应，跳出重试循环
            return model_answer
            
        except requests.exceptions.Timeout as e:
            print(f"Error Occurs: \"{e}\" (Connection/Read timed out). Retry ...")
            if tries < max_tries:
                time.sleep(5)  # 超时错误等待更长时间
            else:
                raise e
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            if tries < max_tries:
                print(f"Error Occurs: \"{e}\"        Retry ...")
                time.sleep(1)
            else:
                print("Max tries. Failed.")
                raise e
    
    return model_answer


def query_llm(
    prompt: str,
    model: str,
    tokenizer,
    llm_url: str,
    max_new_tokens: int = 2048,
    max_len: int = 32768,
    temperature: float = 0.0,
    timeout_s: int = 30
) -> str:
    """
    查询 LLM，支持 prompt 截断
    
    Args:
        prompt: 输入提示词
        model: 模型名称
        tokenizer: tokenizer 对象
        llm_url: LLM 服务 URL
        max_new_tokens: 最大生成 token 数
        max_len: 最大输入长度
        temperature: 温度参数
        timeout_s: 连接超时时间
    
    Returns:
        模型生成的文本
    """
    # 截断处理
    if tokenizer is not None:
        try:
            input_ids = tokenizer.encode(prompt)
            original_len = len(input_ids)
            
            if original_len > max_len:
                # 保留开头 40%，结尾 60%
                first_part = int(max_len * 0.4)
                last_part = max_len - first_part
                input_ids = input_ids[:first_part] + input_ids[-last_part:]
                
                # 验证截断后的长度
                if len(input_ids) > max_len:
                    input_ids = input_ids[:max_len]
                
                prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
                
                # 再次验证：重新编码检查长度
                verify_ids = tokenizer.encode(prompt)
                if len(verify_ids) > max_len:
                    verify_ids = verify_ids[:max_len]
                    prompt = tokenizer.decode(verify_ids, skip_special_tokens=True)
                    final_len = len(verify_ids)
                else:
                    final_len = len(verify_ids)
                
                print(f"警告: 提示词过长 ({original_len} tokens)，已截断到 {final_len} tokens")
        except Exception as e:
            print(f"警告: tokenizer 编码失败: {e}，将使用原始 prompt")
    
    # 构建消息格式
    messages = [
        {
            "role": "system",
            "content": "先读问题，再根据下面的文章内容回答问题，不要进行分析，不要重复问题，用简短的语句给出答案。\n\n例如：\"全国美国文学研究会的第十八届年会在哪所大学举办的？\"\n回答应该为：\"xx大学\"。\n\n",
        },
        {"role": "user", "content": prompt},
    ]
    
    return call_model_online(
        max_new_tokens=max_new_tokens,
        messages=messages,
        model=model,
        llm_url=llm_url,
        temperature=temperature,
        timeout_s=timeout_s
    )


def get_prompt(context: str, question: str) -> str:
    """
    根据 offline_inference_longbench_F1.py 的格式构建 prompt
    
    Args:
        context: 文章内容
        question: 问题
    
    Returns:
        格式化后的 prompt
    """
    prompt = f"""阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{question}\n回答："""
    return prompt


def process_single_item(item, idx, args, tokenizer, processed_ids):
    """
    处理单个数据项（用于并发处理）
    
    Args:
        item: 数据项字典
        idx: 数据项索引
        args: 命令行参数
        tokenizer: tokenizer 对象
        processed_ids: 已处理的数据 ID 集合
    
    Returns:
        (result_dict, status): 结果字典和状态字符串
    """
    item_id = item.get("_id") or item.get("id") or item.get("item_id") or str(idx)
    
    # 检查是否需要跳过
    if str(item_id) in processed_ids:
        return None, "skipped"
    
    # 获取数据字段
    context = item.get("context", "")
    question = item.get("input", "") or item.get("question", "")
    answer = item.get("answers", [])
    
    if not answer:
        answer = [item.get("answer", "")]
    
    if not context or not question:
        return None, "invalid"
    
    # 构建 prompt
    prompt = get_prompt(context, question)
    
    # 调用模型
    try:
        generated_text = query_llm(
            prompt=prompt,
            model=args.model,
            tokenizer=tokenizer,
            llm_url=args.llm_url,
            max_new_tokens=args.max_tokens,
            max_len=args.max_len,
            temperature=0.0,
            timeout_s=args.timeout
        )
        
        # 清理生成的文本
        generated_text = "".join(
            [line.strip() for line in generated_text.splitlines() if line.strip()]
        )
        
        # 构建结果
        result = {
            "pred": generated_text,
            "answers": answer if isinstance(answer, list) else [answer]
        }
        
        # 保留原始数据的其他字段
        if "_id" in item:
            result["_id"] = item["_id"]
        if "id" in item:
            result["id"] = item["id"]
        
        return result, "success"
        
    except Exception as e:
        return {"error": str(e), "item_id": item_id}, "error"


def main():
    parser = argparse.ArgumentParser(description="在线推理 LongBench F1 评测")
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("MODEL_PATH", "Qwen2.5-14B-Instruct"),
        help="模型名称（用于 API 调用）"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="数据集文件路径（JSONL 格式）"
    )
    parser.add_argument(
        "--llm_url",
        type=str,
        default=os.getenv("LLM_URL", "http://127.0.0.1:7800/v1"),
        help="LLM 服务的基础 URL（例如：http://127.0.0.1:7800/v1）"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=os.getenv("SAVE_DIR", "./ucm_sparse_predictions/longbench_v2"),
        help="结果保存目录"
    )
    parser.add_argument(
        "--local_tokenizer",
        type=str,
        default="",
        help="本地 tokenizer 路径（可选，如果不指定则尝试从模型名称加载）"
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=32768,
        help="最大输入长度（用于截断）"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="最大生成 token 数"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="并发请求数量（默认：1，即顺序处理）"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="连接超时时间（秒）"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="是否从已有结果文件恢复（跳过已处理的数据）"
    )
    
    args = parser.parse_args()
    
    # 检查数据集文件
    if not os.path.isfile(args.dataset):
        print(f"错误: 数据集文件不存在: {args.dataset}")
        sys.exit(1)
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 确定输出文件名
    # 优先使用环境变量 RES_FILE（与 offline 版本保持一致）
    output_file = os.getenv("RES_FILE", "")
    timestamp = os.getenv("TIMESTAMP", "")
    if not output_file:
        # 如果没有 RES_FILE，则根据参数生成
        dataset_name = Path(args.dataset).stem
        model_name = args.model.split("/")[-1]
        if timestamp:
            output_file = os.path.join(args.save_dir, f"{model_name}_{dataset_name}_{timestamp}.jsonl")
        else:
            output_file = os.path.join(args.save_dir, f"{model_name}_{dataset_name}.jsonl")
    
    # 加载 tokenizer
    tokenizer = None
    if args.local_tokenizer and args.local_tokenizer != '':
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                args.local_tokenizer, 
                trust_remote_code=True,
                use_chat_template=False
            )
            print(f"已从本地路径加载 tokenizer: {args.local_tokenizer}")
        except Exception as e:
            print(f"警告: 无法从本地路径加载 tokenizer: {args.local_tokenizer}, 错误: {e}")
            tokenizer = None
    else:
        # 尝试从模型名称加载（可能需要网络连接）
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model, 
                trust_remote_code=True,
                use_chat_template=False
            )
            print(f"已从模型名称加载 tokenizer: {args.model}")
        except Exception as e:
            print(f"警告: 无法从模型名称加载 tokenizer: {args.model}, 错误: {e}")
            print("将不使用 tokenizer（仅用于截断，可能影响性能）")
            tokenizer = None
    
    # 读取数据集
    print(f"正在读取数据集: {args.dataset}")
    data_list = []
    with open(args.dataset, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    data_list.append(data)
                except json.JSONDecodeError as e:
                    print(f"警告: 跳过无效的 JSON 行: {e}")
                    continue
    
    print(f"共读取 {len(data_list)} 条数据")
    
    # 检查是否需要恢复
    processed_ids = set()
    if args.resume:
        import glob
        import re
        # 从输出文件名中提取基础模式（去掉时间戳部分）
        output_file_basename = os.path.basename(output_file)
        output_file_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else args.save_dir
        
        # 匹配时间戳格式：_YYYYMMDD_HHMMSS.jsonl
        pattern_match = re.match(r"(.+?)_\d{8}_\d{6}\.jsonl$", output_file_basename)
        if pattern_match:
            # 有时间戳，查找所有匹配的文件
            base_pattern = pattern_match.group(1)
            pattern = os.path.join(output_file_dir, f"{base_pattern}_*.jsonl")
            matching_files = glob.glob(pattern)
            if matching_files:
                # 按修改时间排序，获取最新的文件
                resume_file = max(matching_files, key=os.path.getmtime)
                print(f"检测到已有结果文件（最新）: {resume_file}")
                print("正在加载已处理的数据 ID...")
                with open(resume_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            try:
                                data = json.loads(line)
                                item_id = data.get("_id") or data.get("id") or data.get("item_id")
                                if item_id:
                                    processed_ids.add(str(item_id))
                            except:
                                pass
                print(f"已处理 {len(processed_ids)} 条数据，将跳过这些数据")
        elif os.path.exists(output_file):
            # 没有时间戳时，使用原来的逻辑
            print(f"检测到已有结果文件: {output_file}")
            print("正在加载已处理的数据 ID...")
            with open(output_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            item_id = data.get("_id") or data.get("id") or data.get("item_id")
                            if item_id:
                                processed_ids.add(str(item_id))
                        except:
                            pass
            print(f"已处理 {len(processed_ids)} 条数据，将跳过这些数据")
    
    # 打开输出文件（追加模式）
    fout = open(output_file, "a", encoding="utf-8")
    
    # 线程锁用于文件写入
    write_lock = threading.Lock()
    
    # 处理数据
    print(f"开始推理，结果将保存到: {output_file}")
    print(f"LLM URL: {args.llm_url}")
    print(f"模型: {args.model}")
    print(f"并发数: {args.concurrency}")
    print("-" * 50)
    
    total = len(data_list)
    skipped = 0
    processed = 0
    errors = 0
    
    # 使用线程池进行并发处理
    if args.concurrency > 1:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            # 提交所有任务
            future_to_item = {
                executor.submit(process_single_item, item, idx, args, tokenizer, processed_ids): (idx, item)
                for idx, item in enumerate(data_list)
            }
            
            # 使用 tqdm 显示进度
            try:
                with tqdm(total=total, desc="推理进度") as pbar:
                    for future in as_completed(future_to_item):
                        result, status = future.result()
                        
                        if status == "skipped":
                            skipped += 1
                        elif status == "invalid":
                            idx, item = future_to_item[future]
                            item_id = item.get("_id") or item.get("id") or item.get("item_id") or str(idx)
                            print(f"\n警告: 跳过无效数据（缺少 context 或 question）: {item_id}")
                        elif status == "error":
                            errors += 1
                            error_msg = result.get("error", "Unknown error")
                            item_id = result.get("item_id", "unknown")
                            print(f"\n错误: 处理数据 {item_id} 时出错: {error_msg}")
                        elif status == "success":
                            # 线程安全地写入文件
                            with write_lock:
                                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
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
            for idx, item in enumerate(tqdm(data_list, desc="推理进度")):
                result, status = process_single_item(item, idx, args, tokenizer, processed_ids)
                
                if status == "skipped":
                    skipped += 1
                    continue
                elif status == "invalid":
                    item_id = item.get("_id") or item.get("id") or item.get("item_id") or str(idx)
                    print(f"警告: 跳过无效数据（缺少 context 或 question）: {item_id}")
                    continue
                elif status == "error":
                    errors += 1
                    error_msg = result.get("error", "Unknown error")
                    item_id = result.get("item_id", "unknown")
                    print(f"\n错误: 处理数据 {item_id} 时出错: {error_msg}")
                    print("继续处理下一条数据...")
                    continue
                elif status == "success":
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    fout.flush()
                    processed += 1
        except KeyboardInterrupt:
            print("\n\n用户中断，正在保存已处理的结果...")
    
    fout.close()
    
    print("\n" + "=" * 50)
    print(f"推理完成！")
    print(f"总数据: {total}")
    print(f"已处理: {processed}")
    print(f"已跳过: {skipped}")
    print(f"错误数: {errors}")
    print(f"结果文件: {output_file}")
    print("=" * 50)
    
    # 提示计算 F1 score
    print(f"\n提示: 可以使用以下命令计算 F1 score:")
    dataset_param = dataset_name.split("_")[0]  # 例如 multifieldqa_zh -> multifieldqa_zh
    print(f"python3 {os.path.dirname(__file__)}/eval.py --answer {output_file} --dataset {dataset_param}")


if __name__ == "__main__":
    main()

