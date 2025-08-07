import json
import math
import pathlib
import random
import subprocess
import time
from typing import Any, Dict, Tuple

from transformers import LlamaTokenizerFast
from transformers import AutoTokenizer


RESULTS_VERSION = "2023-08-31"

MAX_NUM_OF_REQUESTS_PER_INPUT_LENGTH = 10


class LLMPerfResults:
    def __init__(
        self,
        name: str,
        metadata: Dict[str, Any] = None,
    ):
        self.name = name
        self.metadata = metadata or {}
        self.timestamp = int(time.time())
        self.metadata["timestamp"] = self.timestamp
        self.version = RESULTS_VERSION

    def to_dict(self):
        data = {
            "version": self.version,
            "name": self.name,
        }
        data.update(self.metadata)
        data = flatten_dict(data)
        return data

    def json(self):
        data = self.to_dict()
        return json.dumps(data)


def upload_to_s3(results_path: str, s3_path: str) -> None:
    """Upload the results to s3.

    Args:
        results_path: The path to the results file.
        s3_path: The s3 path to upload the results to.

    """

    command = ["aws", "s3", "sync", results_path, f"{s3_path}/"]
    result = subprocess.run(command)
    if result.returncode == 0:
        print("Files uploaded successfully!")
    else:
        print("An error occurred:")
        print(result.stderr)


# def randomly_sample_sonnet_lines_prompt(
#     prompt_tokens_mean: int = 550,
#     prompt_tokens_stddev: int = 250,
#     expect_output_tokens: int = 150,
#     tokenizer = LlamaTokenizerFast.from_pretrained(
#         "hf-internal-testing/llama-tokenizer")
# ) -> Tuple[str, int]:
#     """Generate a prompt that randomly samples lines from a the shakespeare sonnet at sonnet.txt.

#     Args:
#         prompt_length_mean: The mean length of the prompt to generate.
#         prompt_len_stddev: The standard deviation of the length of the prompt to generate.
#         expect_output_tokens: The number of tokens to expect in the output. This is used to
#         determine the length of the prompt. The prompt will be generated such that the output
#         will be approximately this many tokens.

#     Note:
#         tokens will be counted from the sonnet using the Llama tokenizer. Using one tokenizer
#         ensures a fairer comparison across different LLMs. For example, if gpt 3.5 tokenizes
#         a prompt in less tokens than Llama2, then this will be reflected in the results since
#         they will be fed identical prompts.

#     Returns:
#         A tuple of the prompt and the length of the prompt.
#     """

#     get_token_length = lambda text: len(tokenizer.encode(text))

#     prompt = (
#         "Randomly stream lines from the following text "
#         f"with {expect_output_tokens} output tokens. "
#         "Don't generate eos tokens:\n\n"
#     )
#     # get a prompt length that is at least as long as the base
#     num_prompt_tokens = sample_random_positive_int(
#         prompt_tokens_mean, prompt_tokens_stddev
#     )
#     while num_prompt_tokens < get_token_length(prompt):
#         num_prompt_tokens = sample_random_positive_int(
#             prompt_tokens_mean, prompt_tokens_stddev
#         )
#     remaining_prompt_tokens = num_prompt_tokens - get_token_length(prompt)
#     sonnet_path = pathlib.Path(__file__).parent.resolve() / "sonnet.txt"
#     with open(sonnet_path, "r") as f:
#         sonnet_lines = f.readlines()
#     random.shuffle(sonnet_lines)
#     sampling_lines = True
#     while sampling_lines:
#         for line in sonnet_lines:
#             line_to_add = line
#             if remaining_prompt_tokens - get_token_length(line_to_add) < 0:
#                 # This will cut off a line in the middle of a word, but that's ok since an
#                 # llm should be able to handle that.
#                 line_to_add = line_to_add[: int(math.ceil(remaining_prompt_tokens))]
#                 sampling_lines = False
#                 prompt += line_to_add
#                 break
#             prompt += line_to_add
#             remaining_prompt_tokens -= get_token_length(line_to_add)
#     return (prompt, num_prompt_tokens)


def sample_random_positive_int(mean: int, stddev: int) -> int:
    """Sample random numbers from a gaussian distribution until a positive number is sampled.

    Args:
        mean: The mean of the gaussian distribution to sample from.
        stddev: The standard deviation of the gaussian distribution to sample from.

    Returns:
        A random positive integer sampled from the gaussian distribution.
    """
    ret = -1
    while ret <= 0:
        ret = int(random.gauss(mean, stddev))
    return ret


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def get_prompts_from_dataset_files(datasef_file_names, tokenizer, mean_output_tokens=500, stddev_output_tokens=100):
    get_token_length = lambda text: len(tokenizer.decode(text))
    dataset_file_name_list = dataset_file_names.split(',')
    prompts = []
    num_output_tokens_list = []
    for dataset_file in dataset_file_name_list:
        local_prompts_file_path = pathlib.Path(__file__).parent.resolve() / dataset_file
        with open(local_prompts_file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        prompt = data["context"] + " Please answer the following question according to the passage above. The question is: " + data["input"]
                        prompts.append((prompt, get_token_length(prompt)))
                        num_output_tokens_list.append(sample_random_positive_int(mean_output_tokens, stddev_output_tokens))
                    except json.JsonDecodeError as e:
                        print(f"Error decoding JSON: {e}")

    return prompts, num_output_tokens_list


def get_messages_from_dataset_files(dataset_file_names, tokenizer, mean_output_tokens=500, stddev_output_tokens=100, context_length=4):
    get_token_length = lambda text: len(tokenizer.decode(text))
    dataset_file_name_list = [str(context_length) + "k.jsonl"]
    messages = []
    num_output_tokens_list = []
    for dataset_file in dataset_file_name_list:
        local_prompts_file_path = pathlib.Path(__file__).parent.resolve() / dataset_file
        with open(local_prompts_file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        prompt = data["context"] + " Please answer the following question according to the passage above. The question is: " + data["input"]
                        message = [
                            {"role": "system", "content": ""},
                            {"role": "user", "content": prompt},
                        ]
                        messages.append((message, get_token_length(prompt)))
                        num_output_tokens_list.append(sample_random_positive_int(mean_output_tokens, stddev_output_tokens))
                        if len(messages) >= MAX_NUM_OF_REQUESTS_PER_INPUT_LENGTH:
                            break
                    except json.JSONDecodeError as e:
                        print(f"Error decoding json: {e}")
                        
    return messages, num_output_tokens_list

def read_txt_line(file_path, line_number):
    if line_number < 0:
        raise ValueError("line number must start at 0")
    with open(file_path, 'r', encoding='utf-8') as file:
        current_line = 0
        for line in file:
            if current_line == line_number:
                return line.strip()
            current_line += 1
    raise ValueError(f"The {line_number}^th line does not exist in file {file_path}")


def get_messages_from_multi_turn_dataset_files(round, past_llm_output, past_message, dataset_file_names, tokenizer, mean_output_tokens=500, stddev_output_tokens=100):
    get_token_length = lambda text: len(tokenizer.decode(text))
    dataset_file_name_list = dataset_file_names.split(',')
    messages = []
    num_output_tokens_list = []
    for dataset_file in dataset_file_name_list:
        local_prompts_file_path = pathlib.Path(__file__).parent.resolve() / dataset_file
        past_inputs = past_message[0]
        past_len = past_message[1]
        past_inputs.append({"role": "system", "content": past_llm_output})
        next_user_input = read_txt_line(local_prompts_file_path, round)
        past_inputs.append({"role": "user", "content": next_user_input})
        messages.append((past_inputs, past_len + get_token_length(past_llm_output) + get_token_length(next_user_input)))
        num_output_tokens_list.append(sample_random_positive_int(mean_output_tokens, stddev_output_tokens))

    return messages, num_output_tokens_list