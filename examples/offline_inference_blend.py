import contextlib
import csv
import json
import os
import random
import re
import time
from dataclasses import asdict

from tqdm import tqdm
from vllm.v1.metrics.reader import Counter, Gauge, Histogram, Vector

random.seed(0)

import sys

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import TokensPrompt

from ucm.logger import init_logger

logger = init_logger(__name__)

model = ""
data_dir = ""
path_to_dataset = ""
tokenizer = None
# 28705 is the token id for <space> char in llama model
# 151643 is the pad token id in qwen model
chunk_end_token_id = -1
chunk_pad_token_id = -1
block_size = 64


def setup_environment_variables():
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["PYTHONHASHSEED"] = "123456"

    global model, data_dir, path_to_dataset, tokenizer, chunk_end_token_id, chunk_pad_token_id
    model = os.getenv("MODEL_PATH", "/home/models/mistralai/Mistral-7B-Instruct-v0.2")
    if not os.path.isdir(model):
        model = input(
            "Enter path to model, e.g./home/models/mistralai/Mistral-7B-Instruct-v0.2: "
        )
        if not os.path.isdir(model):
            print("Exiting. Incorrect model_path")
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

    # now support wikimqa
    path_to_dataset = os.getenv(
        "BLEND_DATASET_PATH", "/home/data/Longbench/data/2wikimqa.jsonl"
    )
    if not os.path.isfile(path_to_dataset):
        path_to_dataset = input(
            "Enter path of one of 2wikimqa dataset in longbench, e.g. /home/data/Longbench/data/2wikimqa.jsonl: "
        )
        if not os.path.isfile(path_to_dataset):
            print("Exiting. Incorrect dataset path")
            sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(model, use_chat_template=True)
    # as for Qwen model, use pad_token_id for padding block
    # as for Llama model, current use unk_token for  padding block
    chunk_pad_token_id = tokenizer.encode("▁", add_special_tokens=False)[0]
    chunk_end_token_id = chunk_pad_token_id

    if tokenizer.pad_token_id is not None:
        chunk_pad_token_id = tokenizer.pad_token_id
        chunk_end_token_id = tokenizer.pad_token_id


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
            "ucm_sparse_config": {
                "Blend": {
                    "chunk_end_token_id": chunk_end_token_id,
                    "compute_meta": {
                        "model.layers.1.self_attn.attn": {
                            "ratio": 0.2,
                        },
                    },
                }
            },
        },
    )

    llm_args = EngineArgs(
        model=model,
        enforce_eager=True,
        kv_transfer_config=ktc,
        max_model_len=16384 * 2,
        max_num_batched_tokens=16384 * 2,
        gpu_memory_utilization=0.8,
        block_size=block_size,
        enable_prefix_caching=False,
        distributed_executor_backend="mp",
        tensor_parallel_size=1,
        trust_remote_code=True,
    )

    llm = LLM(**asdict(llm_args))
    try:
        yield llm
    finally:
        logger.info("LLM engine is exiting.")


def get_output(
    llm: LLM,
    prompt,
    sampling_params: SamplingParams,
):
    start = time.time()
    outputs = llm.generate(prompt, sampling_params)
    print("-" * 50)
    generated_text = None
    for output in outputs:
        generated_text = output.outputs[0].text
    e2e_time = time.time() - start
    print("-" * 50)
    return e2e_time, generated_text


def pad_rag_chunks(token_ids, block_size, pad_id, end_id):
    """
    pad token_ids with pad_id and end up with end_id
    """
    # assert pad_id != end_id
    remainder = len(token_ids) % block_size

    if remainder == 0 and token_ids[-1] in [pad_id, end_id]:
        # no need to pad
        token_ids[-1] = end_id
        return token_ids

    pad_len = block_size - remainder - 1
    padded = token_ids + [pad_id] * pad_len + [end_id]
    return padded


systemPrompt = "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n"


def main():
    module_path = "ucm.integration.vllm.blend_connector"
    name = "UCMBlendConnector"

    setup_environment_variables()

    with build_llm_with_uc(module_path, name, model) as llm:
        prefill_sampling_params = SamplingParams(
            temperature=0.0, top_p=0.95, max_tokens=1
        )
        sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=128)
        # choose one data row in LongBenchV1 (wikimqa)
        assert os.path.isfile(
            path_to_dataset
        ), f"Incorrect dataset path. Please specify the dataset path by `export DATASET_PATH=/home/data/Longbench/data/2wikimqa.jsonl`"
        with open(path_to_dataset, "r") as f:
            lines = f.readlines()
        dataset_row = json.loads(lines[0])

        passages = re.findall(
            r"Passage\s+(\d+):(.*?)(?=Passage\s+\d+:|$)", dataset_row["context"], re.S
        )
        chunks = [f"Passage {i}:{passages[i][1]}" for i in range(len(passages))]
        question = f"\n\nAnswer the question based on the given passages. Answer the question within 5 words. Do NOT repeat the question or output any other words. Question: {dataset_row["input"]}\nAnswer:"
        origin_sys_prompt_ids = tokenizer.encode(systemPrompt)
        padded_sys_prompt_ids = pad_rag_chunks(
            origin_sys_prompt_ids, block_size, chunk_pad_token_id, chunk_end_token_id
        )
        # 1. sys prompt warm up
        print(f"---------------1. sys prompt: warm up---------------")
        get_output(
            llm,
            TokensPrompt(prompt_token_ids=padded_sys_prompt_ids),
            prefill_sampling_params,
        )
        time.sleep(0.5)

        padded_contexts_ids = []
        padded_prompt_ids = padded_sys_prompt_ids
        origin_prompt_ids = origin_sys_prompt_ids
        for text_chunk in chunks:
            un_pad_ids = tokenizer.encode(text_chunk, add_special_tokens=False)
            padded_ids = pad_rag_chunks(
                un_pad_ids, block_size, chunk_pad_token_id, chunk_end_token_id
            )
            padded_prompt_ids = padded_prompt_ids + padded_ids
            origin_prompt_ids = origin_prompt_ids + un_pad_ids
            padded_contexts_ids.append(padded_ids)

        question_ids = tokenizer.encode(question, add_special_tokens=False)
        padded_prompt_ids = padded_prompt_ids + question_ids
        origin_prompt_ids = origin_prompt_ids + question_ids

        print(f"--------------- baseline with no cache blend ---------------")
        baseline_time, baseline_gen_text = get_output(
            llm, TokensPrompt(prompt_token_ids=origin_prompt_ids), sampling_params
        )
        time.sleep(0.5)

        print(f"--------------- cache rag chunks ---------------")
        llm.generate(
            [TokensPrompt(prompt_token_ids=ids) for ids in padded_contexts_ids],
            sampling_params,
        )
        time.sleep(0.5)

        print(f"--------------- warm up blend code ---------------")
        warm_up_blend_prompt_ids = padded_sys_prompt_ids
        for ids in reversed(padded_contexts_ids):
            warm_up_blend_prompt_ids = warm_up_blend_prompt_ids + ids
        warm_up_blend_prompt_ids = warm_up_blend_prompt_ids + question_ids
        llm.generate(
            TokensPrompt(prompt_token_ids=warm_up_blend_prompt_ids), sampling_params
        )
        time.sleep(0.5)

        print(f"--------------- cache blend ---------------")
        blend_time, blend_gen_text = get_output(
            llm, TokensPrompt(prompt_token_ids=padded_prompt_ids), sampling_params
        )
        time.sleep(0.5)

        print(f"--------------- prefix cache ---------------")
        pc_time, pc_gen_text = get_output(
            llm, TokensPrompt(prompt_token_ids=origin_prompt_ids), sampling_params
        )

        print(f"Baseline generated text: {baseline_gen_text!r}")
        print(f"Baseline generated cost time: {baseline_time:.2f} seconds")
        print(f"Prefix Cache generated text: {pc_gen_text!r}")
        print(f"Prefix Cache generated cost time: {pc_time:.2f} seconds")
        print(f"Blend generated text: {blend_gen_text!r}")
        print(f"Blend generated cost time: {blend_time:.2f} seconds")

        print(f"Question:{dataset_row['input']}")
        print(f"Golden answer:{dataset_row["answers"]}")


if __name__ == "__main__":
    main()
