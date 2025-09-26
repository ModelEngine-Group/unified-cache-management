import contextlib
import json
import os
import time
from dataclasses import asdict

from transformers import AutoTokenizer

# Third Party
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs

from ucm.logger import init_logger

MODEL_PATH = "/home/models/Qwen2.5-14B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_chat_template=True)
logger = init_logger(__name__)


def setup_environment_variables():
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["PYTHONHASHSEED"] = "123456"


@contextlib.contextmanager
def build_llm_with_uc(module_path: str, name: str, model: str):
    ktc = KVTransferConfig(
        kv_connector=name,
        kv_connector_module_path=module_path,
        kv_role="kv_both",
        kv_connector_extra_config={
            "ucm_connector_name": "UcmNfsStore",
            "ucm_connector_config": {
                "storage_backends": "/home/data",
                "kv_block_size": 33554432,
            },
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
        gpu_memory_utilization=0.8,
        max_num_batched_tokens=30000,
        block_size=128,
        enforce_eager=True,
        distributed_executor_backend="mp",
        tensor_parallel_size=2,
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
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")
    print(f"Generation took {time.time() - start:.2f} seconds, {req_str} request done.")
    print("-" * 50)


def main():
    module_path = "ucm.integration.vllm.uc_connector"
    name = "UnifiedCacheConnectorV1"
    model = os.getenv("MODEL_PATH", "/home/models/Qwen2.5-14B-Instruct")

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
        prompts = []

        batch_size = 1

        with open("/home/Longbench/data/multifieldqa_zh.jsonl", "r") as f:
            for _ in range(batch_size):
                line = f.readline()
                if not line:
                    break
                data = json.loads(line)
                context = data["context"]
                question = data["input"]
                prompts.append(get_prompt(f"{context}\n\n{question}"))

        sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=100)

        print_output(llm, prompts, sampling_params, "first")
        print_output(llm, prompts, sampling_params, "second")


if __name__ == "__main__":
    main()
