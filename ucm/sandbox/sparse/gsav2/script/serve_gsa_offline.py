import contextlib
import json
import os
import sys
import time
import math
import faulthandler
import json
from dataclasses import asdict, fields
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs
from ucm.logger import init_logger

logger = init_logger(__name__)
model = ""
path_to_dataset = ""
data_dir = ""
tokenizer = None
faulthandler.enable()


def setup_environment_variables():
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,1"
    os.environ["PYTHONHASHSEED"] = "123456"
    os.environ["ENABLE_SPARSE"] = "true"

    global model, path_to_dataset, data_dir, tokenizer
    model = os.getenv("MODEL_PATH", "/home/models/DeepSeek-V2-Lite-Chat")
    if not os.path.isdir(model):
        model = input("Enter path to model, e.g. /home/models/DeepSeek-V2-Lite-Chat: ")
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


def create_prompt(sample) -> str:
    options_text = ""
    if sample.get("choice_A"):
        options_text = f"A) {sample.get('choice_A', '')}\nB) {sample.get('choice_B', '')}\nC) {sample.get('choice_C', '')}\nD) {sample.get('choice_D', '')}\n"
    
    prompt = f"""Please read the following context carefully and answer the question based on the it.

Context:
{sample.get('context', '')}

Question:
{sample.get('question', '')}

    """

    if options_text:
        prompt += f"Options:\n{options_text}\n"
    prompt += """Please provide a your final answer as a single letter (A, B, C, or D).

Reasoning:"""

    return prompt


def cut_prompt(prompt: str, cut_length: int) -> str:
    tokens = tokenizer.encode(prompt)  # Ensure the prompt is tokenized
    if len(tokens) < cut_length:
        return None
    tokens = tokens[:cut_length]
    return tokenizer.decode(tokens, skip_special_tokens=False)


@contextlib.contextmanager
def build_llm_with_uc(model: str, max_bs: int):
    ktc = KVTransferConfig(
        kv_connector="UCMConnector",
        kv_connector_module_path="ucm.integration.vllm.ucm_connector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "UCM_CONFIG_FILE": "./ucm/sandbox/sparse/gsav2/script/ucm_config_example.yaml"
        },
    )

    llm_args = EngineArgs(
        model=model,
        kv_transfer_config=ktc,
        max_model_len=131072,
        gpu_memory_utilization=0.90,
        max_num_batched_tokens=8192,
        block_size=128,
        max_num_seqs=max_bs,
        seed=1024,
        distributed_executor_backend="mp",
        tensor_parallel_size=2,
        trust_remote_code=True,
        async_scheduling=False,
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [4, 16, 32, 64, 96, 128],
            "pass_config": {
                "fuse_minimax_qk_norm": False
            }
        },
        additional_config={
            "pa_shape_list": [48, 64, 72, 80],
            "fuse_muls_add": True,
            "multistream_overlap_shared_expert": True,
            "ascend_compilation_config": {"enable_npugraph_ex": True},
            "enable_mlapo": True,
            "enable_sparse_sfa_c8": True,
            "enable_sparse_li_c8": True,
            "ascend_fusion_config": {"fusion_ops_gmmswigluquant": False}
        },
        enable_prefix_caching=True,
        enable_expert_parallel=True,
        quantization="ascend",
        speculative_config={"method": "mtp", "num_speculative_tokens": 3, "enforce_eager": True}
    )
    llm_kwargs = {
        field.name: getattr(llm_args, field.name)
        for field in fields(llm_args)
    }
    llm = LLM(**llm_kwargs)
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
        lines.append(generated_text + "\n")
    print(f"Generation took {time.time() - start:.2f} seconds, {req_str} request done.")


def main():
    setup_environment_variables()
    max_bs = 32
    mini_batch = 32
    cut_len = 65530

    with build_llm_with_uc(model, max_bs) as llm:
        prompts = []
        batch_size = list(range(mini_batch))
        assert os.path.isfile(
            path_to_dataset
        ), f"Incorrect dataset path. Please specify the dataset path by `export DATASET_PATH=/path/to/longbench/multifieldqa_zh.jsonl`"

        with open(path_to_dataset, "r") as f:
            lines = f.readlines()
            data = json.loads(lines[0])

        for index in batch_size:
            prompt = create_prompt(data[index])
            prompts.append(prompt)

        sampling_params = SamplingParams(
            temperature=0, top_p=0.95, max_tokens=256, ignore_eos=False
        )

        print_output(llm, prompts, sampling_params, "first")


if __name__ == "__main__":
    main()
