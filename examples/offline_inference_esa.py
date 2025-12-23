import os
os.environ["ENABLE_SPARSE"] = "true"

import ucm


import contextlib
import json
import sys
import time
from dataclasses import asdict

from transformers import AutoTokenizer

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


def setup_environment_variables():
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["PYTHONHASHSEED"] = "123456"

    global model, path_to_dataset, data_dir, tokenizer
    model = os.getenv("MODEL_PATH", "/home/models/Qwen2.5-14B-Instruct")
    if not os.path.isdir(model):
        model = input("Enter path to model, e.g. /home/models/Qwen2.5-14B-Instruct: ")
        if not os.path.isdir(model):
            print("Exiting. Incorrect model_path")
            sys.exit(1)

    path_to_dataset = os.getenv(
        "DATASET_PATH", "/home/data/Longbench/data/multifieldqa_zh.jsonl"
    )
    if not os.path.isfile(path_to_dataset):
        path_to_dataset = input(
            "Enter path to one of the longbench dataset, e.g. /home/data/Longbench/data/multifieldqa_zh.jsonl: "
        )
        if not os.path.isfile(path_to_dataset):
            print("Exiting. Incorrect dataset path")
            sys.exit(1)

    data_dir = os.getenv("DATA_DIR", "/home/data/kv_cache")
    if not os.path.isdir(data_dir):
        data_dir = input(
            "Enter the directory for UCMStore to save kv cache, e.g. /home/data/kv_cache: "
        )
        create = input(f"Directory {data_dir} dose not exist. Create it? (Y/n): ")
        if create.lower() == "y":
            os.makedirs(data_dir, exist_ok=True)
        else:
            print("Exiting. Directory not created.")
            sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(model, use_chat_template=True)


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
            "ucm_sparse_config": {
                "ESA": {
                    "init_window_sz": 1,
                    "local_window_sz": 2,
                    "min_blocks": 4,
                    "sparse_ratio": 0.3,
                    "retrieval_stride": 5,
                }
            },
        },
    )

    llm_args = EngineArgs(
        model=model,
        kv_transfer_config=ktc,
        max_model_len=32768,
        gpu_memory_utilization=0.45,
        max_num_batched_tokens=10000,
        block_size=128,
        enforce_eager=True,
        # distributed_executor_backend="mp",
        tensor_parallel_size=1,
        trust_remote_code=True,
    )

    llm = LLM(**asdict(llm_args))
    return llm


def print_output(
    llm: LLM,
    prompt: list[str],
    sampling_params: SamplingParams,
    req_str: str,
):
    start = time.time()
    outputs = llm.generate(prompt, sampling_params)
    print("-" * 50)
    lines = []
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")
        lines.append(generated_text + "\n")
    print(f"ESA Generation took {time.time() - start:.2f} seconds, {req_str} request done.")
    print("-" * 50)


def main():
    import os; print(f"esa main @ {os.getpid()}")
    module_path = "ucm.integration.vllm.ucm_connector"
    name = "UCMConnector"
    setup_environment_variables()

    def get_prompt(prompt):
        messages = [
            {
                "role": "system",
                "content": "先读问题，再根据下面的文章内容回答问题，不要进行分析，不要重复问题，用简短的语句给出答案。\n\n例如：“全国美国文学研究会的第十八届年会在哪所大学举办的？”\n回答应该为：“xx大学”。\n\n",
            },
            {"role": "user", "content": prompt},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            add_special_tokens=True,
        )

    llm = build_llm_with_uc(module_path, name, model)
    prompts = []
    batch_size = 1
    assert os.path.isfile(
        path_to_dataset
    ), f"Incorrect dataset path. Please specify the dataset path by `export DATASET_PATH=/path/to/longbench/multifieldqa_zh.jsonl`"
    with open(path_to_dataset, "r") as f:
        lines = f.readlines()
    line = lines[10]
    data = json.loads(line)
    prompt = f"""阅读以下文字并用中文简短回答：\n\n{data["context"]}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{data["input"]}\n回答："""
    prompts = [get_prompt(prompt)]
    # og_prompt = get_prompt(prompt)
    # og_len = len(og_prompt)
    # prompts = [og_prompt[:og_len//3]]

    sampling_params = SamplingParams(
        temperature=0, top_p=0.95, max_tokens=256, ignore_eos=True
    )
    for _ in range(1):
        print_output(llm, prompts, sampling_params, "init")
    llm.reset_prefix_cache()

    print("========\n\n\n========")
    line = lines[0]
    data = json.loads(line)
    prompt = f"""阅读以下文字并用中文简短回答：\n\n{data["context"]}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{data["input"]}\n回答："""
    prompts = [get_prompt(prompt)]
    # og_prompt = get_prompt(prompt)
    # og_len = len(og_prompt)
    # prompts = [og_prompt[:og_len//3]]
    print_output(llm, prompts, sampling_params, "formal")


if __name__ == "__main__":
    main()
