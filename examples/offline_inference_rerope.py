import contextlib
import json
import os
import sys
import time
from dataclasses import asdict

from transformers import AutoTokenizer

# setting for rerope
os.environ["VLLM_USE_REROPE"] = "true"

# Third Party
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs

from ucm.logger import init_logger

logger = init_logger(__name__)


def setup_environment_variables():
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["PYTHONHASHSEED"] = "123456"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,6,7"
    os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN_VLLM_V1"
    os.environ["REROPE_WINDOW"] = "32768"
    os.environ["TRAINING_LENGTH"] = "32768"

    global data_dir
    data_dir = os.getenv("DATA_DIR", "/home/externals/wangwenxin21/wx_data")

    if not os.path.isdir(data_dir):
        create = input(f"Directory {data_dir} dose not exist. Create it? (Y/n): ")
        if create.lower() == "y":
            os.makedirs(data_dir, exist_ok=True)
        else:
            print("Exiting. Directory not created.")
            sys.exit(1)


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
        },
    )

    llm_args = EngineArgs(
        model=model,
        kv_transfer_config=ktc,
        hf_overrides={
            "max_position_embeddings": 430080,
        },
        gpu_memory_utilization=0.8,
        max_num_batched_tokens=8192,
        block_size=16,
        enforce_eager=True,
        tensor_parallel_size=4,
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
    module_path = "ucm.integration.vllm.ucm_connector"
    name = "UCMConnector"
    model = os.getenv("MODEL_PATH", "/home/wx/models/Qwen2.5-14B-Instruct")

    tokenizer = AutoTokenizer.from_pretrained(model, use_chat_template=True)
    setup_environment_variables()

    with build_llm_with_uc(module_path, name, model) as llm:

        data_all = []
        with open(
            "/home/wx/va_clean/data/multifieldqa_zh.jsonl", "r", encoding="utf-8"
        ) as f:
            for line in f:
                data_all.append(json.loads(line))

        materials = []
        questions = []
        references = []
        batch_size = 75
        num_batch = 2
        for idx in range(num_batch):
            data = data_all[idx * batch_size : (idx + 1) * batch_size]

            materials.append(
                "\n\n".join(
                    [
                        f"【语料{i+1}】\n{item.get('context', '')}"
                        for i, item in enumerate(data)
                    ]
                )
            )
            questions.append(
                "\n".join(
                    [
                        f"{i+1}. {item.get('input', '')}"
                        for i, item in enumerate(data[:15])
                    ]
                )
            )
            references.append(
                [
                    f"{i+1}. {item.get('answers', '')}"
                    for i, item in enumerate(data[:15])
                ]
            )

        system_prompt = "你是一个AI助手，请根据以下材料回答问题。"
        tokenized_inputs = []
        for material, question in zip(materials, questions):
            content = (
                "请根据以下文本内容回答后面的问题：\n\n"
                "【文本内容开始】\n"
                f"{material}\n"
                "【文本内容结束】\n\n"
                "请回答以下问题：\n"
                f"{question}"
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ]
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            tokenized_inputs.append(inputs)

        sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=2048)

        for req in range(num_batch):
            print_output(
                llm, tokenized_inputs[req], sampling_params, "request_" + str(req)
            )


if __name__ == "__main__":
    main()
    