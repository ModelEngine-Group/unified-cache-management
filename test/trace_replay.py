import argparse
import asyncio
import json
import time
import logging
import numpy as np
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase

benchmark_path = os.environ.get("BENCHMARK_PATH")
if benchmark_path:
    sys.path.append(benchmark_path)
    print(f"Added benchmark path: {benchmark_path}")
else:
    raise EnvironmentError("BENCHMARK_PATH is not set!")

from benchmark_serving import benchmark, create_argument_parser, get_tokenizer, get_request
from benchmark_dataset import (
    AIMODataset,
    ASRDataset,
    BenchmarkDataset,
    BurstGPTDataset,
    ConversationDataset,
    CustomDataset,
    HuggingFaceDataset,
    InstructCoderDataset,
    MTBenchDataset,
    NextEditPredictionDataset,
    RandomDataset,
    SampleRequest,
    ShareGPTDataset,
    SonnetDataset,
    VisionArenaDataset,
)
from backend_request_func import (
    ASYNC_REQUEST_FUNCS,
    OPENAI_COMPATIBLE_BACKENDS,
    RequestFuncInput,
    RequestFuncOutput,
)
try:
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    from argparse import ArgumentParser as FlexibleArgumentParser

logger = logging.getLogger(__name__)
SUPPORTED_ENGINES = ["vllm"]

class TraceReplayDataset(BenchmarkDataset):
    # Default values copied from benchmark_serving.py for the random dataset.
    DEFAULT_PREFIX_LEN = 0
    DEFAULT_RANGE_RATIO = 0.0
    DEFAULT_INPUT_LEN = 1024
    DEFAULT_OUTPUT_LEN = 128
    REG_GROUPS = defaultdict(list)
    
    def load_trace(self, trace_file):
        with open(trace_file, "r", encoding="utf-8") as f:
            # Read by line
            for line in f:
                record = json.loads(line)
                self.REG_GROUPS[int(record["timestamp"]) / 1000].append(record)
        print(f"Done load trace file, time: {time.time()}")

    def __init__(
        self,
        trace_path,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.load_trace(trace_path)
        trace_directory_name = os.path.dirname(trace_path)
        trace_file_name = os.path.basename(trace_path).split('.')[0]
        self.prompts_file_name = f"{trace_directory_name}/{trace_file_name}_dataset.jsonl"

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        prefix_len: int = 0,
        **kwargs,
    ) -> dict[float, list[SampleRequest]]:
        requests = defaultdict(list)
        if os.path.exists(self.prompts_file_name):
            with open(self.prompts_file_name, "r", encoding="utf-8") as f:
            # Read by line
                for line in f:
                    record = json.loads(line)
                    timestamp = float(record["timestamp"])
                    prompt = record["prompt"]
                    output_length = record["output_length"]
                    requests[timestamp].append(
                        SampleRequest(
                            prompt=prompt,
                            prompt_len=len(prompt),
                            expected_output_len=output_length,
                        )
                    )
            print(f"Done load trace file, time: {time.time()}")
        
        assert self.REG_GROUPS is not None, ("Find no trace info!!!")
        vocab_size = tokenizer.vocab_size
        num_special_tokens = tokenizer.num_special_tokens_to_add()
        
        all_token_sequences = []
        meta_info = [] # 保存 (timestamp, input_len, output_len)
        for timestamp, record_list in self.REG_GROUPS.items():
            for req in record_list:
                hash_ids = req["hash_ids"]
                input_length = req["input_length"]
                output_length = req["output_length"]
                prefix_token_ids = (
                    [hash_ids[i % len(hash_ids)] for i in range(prefix_len)] if prefix_len > 0 else []
                )
                real_input_len = input_length - prefix_len - num_special_tokens
                if len(hash_ids) >= real_input_len:
                    inner_seq = hash_ids[:real_input_len]
                else:
                    # 如果 hash_ids 不够长，循环填充
                    offset = sum(hash_ids) % vocab_size
                    inner_seq = (
                        (offset + np.arange(real_input_len)) % vocab_size
                    ).tolist()
                    
                # 拼接前缀 + 主体
                token_sequence = prefix_token_ids + inner_seq
                
                all_token_sequences.append(token_sequence)
                meta_info.append((timestamp, output_length))
        
        decoded_prompts = tokenizer.batch_decode(all_token_sequences)
        print(f"Done decoded prompts, time: {time.time()}")
        re_encodeds = []
        for token_ids, prompt in zip(all_token_sequences, decoded_prompts):
            re_encodeds.append(tokenizer.encode(prompt, add_special_tokens=False)[:len(token_ids)])
        print(f"Done reencoded prompts, time: {time.time()}")
        decoded_prompts.clear()
        decoded_prompts = tokenizer.batch_decode(re_encodeds)
        print(f"Done redecoded prompts, time: {time.time()}")
        
        batch_size = 100  # 每100条批量写入一次
        batch_data = []
        i = 0
        for (timestamp, output_length), token_ids, prompt in zip(meta_info, all_token_sequences, decoded_prompts):
            requests[timestamp].append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=len(token_ids),
                    expected_output_len=output_length,
                )
            )
            batch_data.append({
                "timestamp": timestamp,
                "prompt": prompt,
                "output_length": output_length,
            })
            i += 1
            if len(batch_data) >= batch_size or i == len(meta_info) - 1:
                with open(self.prompts_file_name, 'a', encoding='utf-8') as f:
                    for data in batch_data:
                        f.write(json.dumps(data, ensure_ascii=False) + '\n')
                batch_data = []  # 清空批量数据

        print(f"Done sample, time: {time.time()}")
        return requests

def create_argument_trace():
    parser = create_argument_parser()
    trace_group = parser.add_argument_group("tracing parameters")
    trace_group.add_argument(
        "--trace-path",
        type=str,
        default=None,
        help="Path to trace file path.",
    )
    return parser
    
def gene_one_req(req: json, args:argparse.Namespace, num_prompts = 1):
    backend = args.backend
    tokenizer = args.tokenizer
    
    if args.dataset_name == "custom":
        dataset = CustomDataset(dataset_path=args.dataset_path)
        input_requests = dataset.sample(
            num_requests=num_prompts,
            tokenizer=tokenizer,
            output_len=req["output_length"],
            skip_chat_template=args.custom_skip_chat_template,
        )

    elif args.dataset_name == "sonnet":
        dataset = SonnetDataset(dataset_path=args.dataset_path)
        # For the "sonnet" dataset, formatting depends on the backend.
        if args.backend == "openai-chat":
            input_requests = dataset.sample(
                num_requests=num_prompts,
                input_len=req["input_length"],
                output_len=req["output_length"],
                prefix_len=args.sonnet_prefix_len,
                tokenizer=tokenizer,
                return_prompt_formatted=False,
            )
        else:
            assert tokenizer.chat_template or tokenizer.default_chat_template, (
                "Tokenizer/model must have chat template for sonnet dataset."
            )
            input_requests = dataset.sample(
                num_requests=num_prompts,
                input_len=req["input_length"],
                output_len=req["output_length"],
                prefix_len=args.sonnet_prefix_len,
                tokenizer=tokenizer,
                return_prompt_formatted=True,
            )

    elif args.dataset_name == "hf":
        # all following datasets are implemented from the
        # HuggingFaceDataset base class
        if args.dataset_path in VisionArenaDataset.SUPPORTED_DATASET_PATHS:
            dataset_class = VisionArenaDataset
            args.hf_split = "train"
            args.hf_subset = None
        elif args.dataset_path in InstructCoderDataset.SUPPORTED_DATASET_PATHS:
            dataset_class = InstructCoderDataset
            args.hf_split = "train"
        elif args.dataset_path in MTBenchDataset.SUPPORTED_DATASET_PATHS:
            dataset_class = MTBenchDataset
            args.hf_split = "train"
        elif args.dataset_path in ConversationDataset.SUPPORTED_DATASET_PATHS:
            dataset_class = ConversationDataset
        elif args.dataset_path in AIMODataset.SUPPORTED_DATASET_PATHS:
            dataset_class = AIMODataset
            args.hf_split = "train"
        elif args.dataset_path in NextEditPredictionDataset.SUPPORTED_DATASET_PATHS:  # noqa: E501
            dataset_class = NextEditPredictionDataset
            args.hf_split = "train"
        elif args.dataset_path in ASRDataset.SUPPORTED_DATASET_PATHS:
            dataset_class = ASRDataset
            args.hf_split = "train"
        else:
            supported_datasets = set(
                [
                    dataset_name
                    for cls in HuggingFaceDataset.__subclasses__()
                    for dataset_name in cls.SUPPORTED_DATASET_PATHS
                ]
            )
            raise ValueError(
                f"Unsupported dataset path: {args.dataset_path}. "
                "Huggingface dataset only supports dataset_path"
                f" from one of following: {supported_datasets}. "
                "Please consider contributing if you would "
                "like to add support for additional dataset formats."
            )

        if dataset_class.IS_MULTIMODAL and backend not in [
            "openai-chat",
            "openai-audio",
        ]:
            # multi-modal benchmark is only available on OpenAI Chat backend.
            raise ValueError(
                "Multi-modal content is only supported on 'openai-chat' and "
                "'openai-audio' backend."
            )
        input_requests = dataset_class(
            dataset_path=args.dataset_path,
            dataset_subset=args.hf_subset,
            dataset_split=args.hf_split,
            random_seed=args.seed,
        ).sample(
            num_requests=num_prompts,
            tokenizer=tokenizer,
            output_len=req["output_length"],
        )

    else:
        # For datasets that follow a similar structure, use a mapping.
        dataset_mapping = {
            "sharegpt": lambda: ShareGPTDataset(
                random_seed=args.seed, dataset_path=args.dataset_path
            ).sample(
                tokenizer=tokenizer,
                num_requests=num_prompts,
                output_len=req["output_length"],
            ),
            "burstgpt": lambda: BurstGPTDataset(
                random_seed=args.seed, dataset_path=args.dataset_path
            ).sample(tokenizer=tokenizer, num_requests=num_prompts),
            "random": lambda: RandomDataset(dataset_path=args.dataset_path).sample(
                tokenizer=tokenizer,
                num_requests=num_prompts,
                prefix_len=args.random_prefix_len,
                input_len=req["input_length"],
                output_len=req["output_length"],
                range_ratio=args.random_range_ratio,
            ),
        }
    return input_requests
    
def gene_prompts_by_dataset_name(req_groups:dict, args:argparse.Namespace) -> dict[float, list]:
    if args.dataset_name is None:
        raise ValueError(
            "Please specify '--dataset-name' and the corresponding "
            "'--dataset-path' if required."
        )
    # {float, list[json]}
    for sec, reqs in sorted(req_groups.items()):
        input_requests = defaultdict(list)
        for req in reqs:
            # Try to produce prompt by benchmark datasets
            gene_req = gene_one_req(req, args)
            input_requests[sec].extend(gene_req)
    return input_requests

async def replay_trace_by_benchmark(req_groups:dict, args:argparse.Namespace):
    backend = args.backend
    model_id = args.model
    model_name = args.served_model_name
    tokenizer = args.tokenizer
    
    if args.base_url is not None:
        api_url = f"{args.base_url}{args.endpoint}"
        base_url = f"{args.base_url}"
    else:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"
        base_url = f"http://{args.host}:{args.port}"
    
    start_time = time.time()
    print(f"Start time is {start_time}")
    tasks = []
    for sec, reqs in sorted(req_groups.items()):
        delay = sec - (time.time() - start_time)
        delay = max(0, delay)
        async def send_one_request(r=reqs, d=delay):
            sampling_params = {}
            sampling_params["temperature"] = 0.9
            await asyncio.sleep(d)  # 等到目标时间
            print(f"Sending request at {time.time() - start_time:.3f}s")
            try:
                result = await benchmark(
                    backend=backend,
                    api_url=api_url,
                    base_url=base_url,
                    model_id=model_id,
                    model_name=model_name,
                    tokenizer=tokenizer,
                    input_requests=r,
                    logprobs=None,
                    request_rate=float("inf"), #send all requests of same timestamp at once
                    burstiness=1,
                    disable_tqdm=True,
                    profile=False,
                    selected_percentile_metrics=["ttft", "tpot", "itl"],
                    selected_percentiles=[25.0, 50.0, 75.0, 99.0],
                    ignore_eos=True,
                    goodput_config_dict={},
                    max_concurrency=args.max_concurrency,
                    lora_modules=None,
                    extra_body=sampling_params,
                    ramp_up_strategy=None,
                    ramp_up_start_rps=None,
                    ramp_up_end_rps=None,
                )
            except asyncio.TimeoutError:
                print(f"请求超时: timestamp {r[0].timestamp if r else 'unknown'}")
                return None
            except Exception as e:
                print(f"请求失败: {e}")
                return None
            return result
        tasks.append(asyncio.create_task(send_one_request(reqs, delay)))
    await asyncio.gather(*tasks)
    
async def send_one_request(args:argparse.Namespace, input_requests, d, start_time):
    await asyncio.sleep(d)  # 等到目标时间
    print(f"Sending request at {time.time() - start_time:.3f}s")
    request_rate = args.request_rate
    burstiness = args.burstiness
    ramp_up_strategy = args.ramp_up_strategy
    ramp_up_start_rps = args.ramp_up_start_rps
    ramp_up_end_rps = args.ramp_up_end_rps
    try:
        async for request, current_request_rate in get_request(
            input_requests,
            request_rate,
            burstiness,
            ramp_up_strategy,
            ramp_up_start_rps,
            ramp_up_end_rps,
        ):
            # TODO
            return None
        
    except asyncio.TimeoutError:
        print(f"请求超时: timestamp {r[0].timestamp if r else 'unknown'}")
        return None
    except Exception as e:
        print(f"请求失败: {e}")
        return None
    return result

async def replay_trace_by_time(req_groups:dict, args:argparse.Namespace):
    backend = args.backend
    model_id = args.model
    model_name = args.served_model_name
    tokenizer = args.tokenizer
    
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[backend]
    else:
        raise ValueError(f"Unknown backend: {backend}")
    if args.base_url is not None:
        api_url = f"{args.base_url}{args.endpoint}"
        base_url = f"{args.base_url}"
    else:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"
        base_url = f"http://{args.host}:{args.port}"
    
    sampling_params = {}
    sampling_params["temperature"] = 0.9
    
    print("Starting initial single prompt test run...")
    # timestamp = 0, first req
    test_prompt, test_prompt_len, test_output_len, test_mm_content = (
        req_groups[0][0].prompt,
        req_groups[0][0].prompt_len,
        req_groups[0][0].expected_output_len,
        req_groups[0][0].multi_modal_data,
    )
    assert test_mm_content is None or isinstance(test_mm_content, dict)
    test_input = RequestFuncInput(
        model=model_id,
        model_name=model_name,
        prompt=test_prompt,
        api_url=api_url,
        prompt_len=test_prompt_len,
        output_len=test_output_len,
        logprobs=None,
        multi_modal_content=test_mm_content,
        ignore_eos=True,
        extra_body=sampling_params,
    )
    test_output = await request_func(request_func_input=test_input)
    if not test_output.success:
        raise ValueError(
            "Initial test run failed - Please make sure tracereplay arguments "
            f"are correctly specified. Error: {test_output.error}"
        )
    else:
        print("Initial test run completed. Starting main tracereplay run...")
    
    start_time = time.time()
    print(f"Start time is {start_time}")
    tasks = []
    for sec, reqs in sorted(req_groups.items()):
        delay = sec - (time.time() - start_time)
        delay = max(0, delay)
    
        tasks.append(asyncio.create_task(send_one_request(reqs, delay)))
    await asyncio.gather(*tasks)
    
            
def main(args: argparse.Namespace):
    print(args)
    dataset = TraceReplayDataset(args.trace_path)
    
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model
    tokenizer_mode = args.tokenizer_mode
    tokenizer = get_tokenizer(
        tokenizer_id,
        tokenizer_mode=tokenizer_mode,
        trust_remote_code=args.trust_remote_code,
    )
    
    input_requests = dataset.sample(
        tokenizer=tokenizer,
    )
    asyncio.run(replay_trace_by_benchmark(input_requests, args))
    
        
if __name__ == "__main__":
    parser = create_argument_trace()
    args = parser.parse_args()
    main(args)
    