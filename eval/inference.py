import contextlib
import json
import os
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
    os.environ["ENABLE_SPARSE"] = "true"

    global model, path_to_dataset, data_dir, ucm_sparse_config, dataset_name, tokenizer
    model = os.getenv("MODEL_PATH", "/home/models/Qwen2.5-14B-Instruct")
    if not os.path.isdir(model):
        model = input("Enter path to model, e.g. /home/models/Qwen2.5-14B-Instruct: ")
        if not os.path.isdir(model):
            print("Exiting. Incorrect model_path")
            sys.exit(1)

    path_to_dataset = os.getenv(
        "DATASET_FLIE", "/home/data/Longbench/data/multifieldqa_zh.jsonl"
    )
    if not os.path.isfile(path_to_dataset):
        path_to_dataset = input(
            "Enter path to one of the longbench dataset, e.g. /home/data/Longbench/data/multifieldqa_zh.jsonl: "
        )
        if not os.path.isfile(path_to_dataset):
            print("Exiting. Incorrect dataset file path")
            sys.exit(1)

    data_dir = os.getenv("STORAGE_BACKENDS", "/home/data/kv_cache")
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

    dataset_name = os.getenv("file_name_no_ext", "multifieldqa_zh")

    tokenizer = AutoTokenizer.from_pretrained(model, use_chat_template=False)


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
        kv_transfer_config=ktc,
        max_model_len=32768,
        gpu_memory_utilization=0.8,
        max_num_batched_tokens=30000,
        block_size=128,
        enforce_eager=True,
        trust_remote_code=True,
        distributed_executor_backend="mp",
        tensor_parallel_size=1,
    )

    llm = LLM(**asdict(llm_args))
    try:
        yield llm
    finally:
        logger.info("LLM engine is exiting.")


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
        generated_text = "".join(
            [line.strip() for line in generated_text.splitlines() if line.strip()]
        )
        lines.append(generated_text)
    print(f"Generation took {time.time() - start:.2f} seconds, {req_str} request done.")
    return lines


def main():
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

    with build_llm_with_uc(module_path, name, model) as llm:
        res_file = os.getenv("RES_FILE", "./ucm_sparse_output/multifieldqa_zh.jsonl")
        batch_size = int(os.getenv("BATCH_SIZE", 20))
        with open(path_to_dataset, "r") as f:
            lines = f.readlines()
        # lines=lines[:20]
        total_data = len(lines)
        for start_idx in range(0, total_data, batch_size):
            end_idx = min(start_idx + batch_size, total_data)
            current_batch = lines[start_idx:end_idx]
            prompts = []
            answers = []
            for line in current_batch:
                data = json.loads(line)
                answer = data["answers"][0]
                prompt = f"""阅读以下文字并用中文简短回答：\n\n{data["context"]}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{data["input"]}\n回答："""
                prompts.append(get_prompt(prompt))
                answers.append(answer)

            sampling_params = SamplingParams(
                temperature=0, top_p=0.95, max_tokens=2048, ignore_eos=False
            )

            gen_res = print_output(
                llm, prompts, sampling_params, f"{len(current_batch)}"
            )

            with open(res_file, "a", encoding="utf-8") as file:
                for generated_text, ori_answer in zip(gen_res, answers):
                    json_obj = {"pred": generated_text, "answers": [ori_answer]}
                    file.write(json.dumps(json_obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
