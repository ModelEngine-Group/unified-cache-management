import json
import random
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import numpy as np
from common.uc_eval.utils.data_class import SynthericParams
from common.uc_eval.utils.utils import PathUtil, get_logger
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer

logger = get_logger()
EPOCH_NUM = 10


class BaseDataset(ABC):
    def __init__(self, tokenizer_path: str = None, **kwargs):
        tokenizer_path = PathUtil.get_datasets_dir_path(tokenizer_path)
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path
        )

    @abstractmethod
    def prepare_data(self, param: Any):
        raise NotImplementedError

    def load_json_file(self, file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except FileNotFoundError:
            logger.error(f"JSON file not found: {file_path}")
            raise FileNotFoundError(f"JSON file not found: {file_path}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in file {file_path}: {e}")
            raise ValueError(f"Invalid JSON format in file {file_path}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error while loading JSON file {file_path}: {e}")
            raise ValueError(f"Failed to load JSON file {file_path}: {e}")

    def load_jsonl_data(self, file_path):
        try:
            data = []
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    json_line = json.loads(line)
                    data.append(json_line)
            return data
        except FileNotFoundError:
            logger.error(f"JSONL file not found: {file_path}")
            raise FileNotFoundError(f"JSONL file not found: {file_path}")
        except json.JSONDecodeError as e:
            logger.error(f"JSONL decode error in file {file_path}: {e}")
            raise ValueError(f"Invalid JSONL format in file {file_path}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error while loading JSONL file {file_path}: {e}")
            raise ValueError(f"Failed to load JSONL file {file_path}: {e}")


class SyntheticDataset(BaseDataset):
    def __init__(self, tokenizer_path: str, **kwargs):
        super().__init__(tokenizer_path, **kwargs)

    def prepare_data(self, syntheric_params: SynthericParams) -> list[str]:
        prompt_list = []
        for parallel_num in tqdm(
            range(syntheric_params.parallel_num),
            desc="Generate synthetic data",
            unit="prompt",
        ):
            random_prompt_len = max(
                0, syntheric_params.prompt_tokens - syntheric_params.prefix_cache_tokens
            )
            random_prompt = self.generate_random_str(random_prompt_len, time.time_ns())
            if syntheric_params.prefix_cache_tokens > 0:
                pc_prompt = self.generate_random_str(
                    syntheric_params.prefix_cache_tokens,
                    syntheric_params.seeds[parallel_num],
                )
            else:
                pc_prompt = ""
            final_prompt = pc_prompt + random_prompt
            prompt_list.append(final_prompt)
        return prompt_list

    def generate_random_str(self, length: int, seed: int) -> str:
        """
        Sample random tokens from the tokenizer using a seed.
        Use timestamp when cache hit is not required; otherwise use an incrementing seed.
        """
        if length <= 0:
            return ""
        vocab_size = self.tokenizer.vocab_size
        random.seed(seed)
        ids_list = random.choices(range(vocab_size // 4, vocab_size // 3), k=length)
        ids = np.array(ids_list)
        text = self.tokenizer.decode(ids)
        completion_token_ids = self.tokenizer([text]).input_ids
        logger.debug(
            f"len(completion_token_ids[0]) = {len(completion_token_ids[0])}, length = {length}"
        )

        epoch = EPOCH_NUM
        while len(completion_token_ids[0]) != length and epoch > 0:
            epoch -= 1
            while len(completion_token_ids[0]) > length:
                diff = len(completion_token_ids[0]) - length
                now_length = ids.shape[0] - diff
                ids = ids[:now_length]
                text = self.tokenizer.decode(ids)
                completion_token_ids = self.tokenizer([text]).input_ids

            while len(completion_token_ids[0]) < length:
                diff = length - len(completion_token_ids[0])
                diff_ids_list = random.choices(
                    range(vocab_size // 4, vocab_size // 3), k=diff
                )
                diff_ids = np.array(diff_ids_list)
                ids = np.append(ids, diff_ids)
                text = self.tokenizer.decode(ids)
                completion_token_ids = self.tokenizer([text]).input_ids

        if len(completion_token_ids[0]) != length:
            logger.warning(
                "The length of completion token ids is not equal to the length of input token ids"
            )
            logger.warning(
                f"Generate tokens, target: {length}, actual: {len(completion_token_ids[0])}"
            )

        return text


class MultiTurnDialogueDataset(BaseDataset):
    def __init__(self, tokenizer_path: str, **kwargs):
        super().__init__(tokenizer_path, **kwargs)

    def prepare_data(self, dataset_file_path) -> List[List[Union[str, Dict]]]:
        """
        Load a JSON file containing multi-turn dialogue dataset paths.
        :param file_path: JSON file listing multi-turn dialogue dataset paths to traverse.
        the multi-turn dataset format: {"kimi": [{"conversion": [{"from": "user", "value": "xxx"}, ...], "qa": [{"question": "xxx", "answer": "xxx"}, ...]}]}
        """
        cases = []
        # the path of multiturndialog.json
        json_path = PathUtil.get_datasets_dir_path(dataset_file_path)
        mtd_data: dict = self.load_json_file(json_path)
        for dataset_name, files_list in mtd_data.items():
            for file_name in files_list:
                case_path = PathUtil.get_dirname(json_path).joinpath(
                    dataset_name, file_name
                )
                if case_path.exists():
                    dialogues = self.load_json_file(case_path)
                    cases.extend(self.process_single_case_file(dialogues))
                else:
                    logger.warning(
                        f"JSON file {case_path} does not exist, please check the file path"
                    )
        if len(cases) == 0:
            logger.warning(
                f"The file {json_path} does not contain multi-turn dialogue data"
            )
        return cases

    def process_single_case_file(self, dialogues: dict) -> List[List[Union[str, Dict]]]:
        cases = []
        for dialogue_name, dialogue_data in dialogues.items():
            for i, dialog in enumerate(dialogue_data):
                dialog_tokens = len(
                    self.tokenizer.tokenize(str(dialog["conversations"]))
                )
                logger.info(
                    f"Current dialogue {dialogue_name}-{i} token count: {dialog_tokens}"
                )
                cases.append([f"{dialogue_name}-{i}", dialog])
        return cases


class DocQADataset(BaseDataset):
    def __init__(self, tokenizer_path: str, **kwargs):
        super().__init__(tokenizer_path, **kwargs)
        self.select_data_class = kwargs.get("select_data_class", None)

    def prepare_data(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load a JSONL file containing doc_qa data
        :param file_path: Path to the jsonl file
        :return: List of doc_qa data
        """
        file_path = PathUtil.get_datasets_dir_path(file_path)
        data_list = []
        if file_path.suffix.lower() == ".jsonl":
            data_list = self.load_jsonl_data(file_path)
        elif file_path.suffix.lower() == ".json":
            data_list = self.load_json_file(file_path)

        cases_list = []
        for data in data_list:
            extracted_data = []
            if "choice_A" in data.keys():
                extracted_data = self._get_multiple_choice_content(
                    data, self.select_data_class
                )
            else:
                extracted_data = self._get_single_answer_content(data)
            if extracted_data:
                cases_list.append(extracted_data)

        return cases_list

    def _get_single_answer_content(
        self, json_lines, select_data_class: Dict[str, Any] = {}
    ):
        """
        Get the prompt, answer, question, case_name parameters from json data
        """
        from common.uc_eval.utils.prompt_config import doc_qa_prompt

        question = json_lines.get("input", None)
        answer = json_lines.get("answers", None)
        dataset = json_lines.get("dataset", None)
        _id = json_lines.get("_id", None)

        is_match = self.match_dataset_with_select_data_class(
            json_lines, select_data_class
        )
        if not is_match:
            return []

        prompt_list = []
        for item in doc_qa_prompt:
            prompt = self.get_prompt_from_json_lines(json_lines, item)
            prompt_list.append(prompt)

        return [f"{dataset}-{_id}", prompt_list, question, answer]

    def _get_multiple_choice_content(
        self, json_lines, select_data_class: Dict[str, Any] = {}
    ):
        """
        For multiple-choice questions, after extracting "answer" and "question", also extract keys like "choice_A" to distinguish each option before building the prompt.
        """
        from common.uc_eval.utils.prompt_config import multi_answer_prompt

        question = json_lines.get("question", None)
        answer = json_lines.get("answer", None)
        domain = json_lines.get("domain", None)
        difficulty = json_lines.get("difficulty", None)
        _id = json_lines.get("_id", None)

        is_match = self.match_dataset_with_select_data_class(
            json_lines, select_data_class
        )
        if not is_match:
            return []

        format_list = [domain, difficulty, _id]
        for i, item in enumerate(format_list):
            format_list[i] = re.sub(r"\s+", "-", item.strip())
        case_name = re.sub(r"-+", "-", f"{'-'.join(format_list)}")

        prompt_list = []
        for item in multi_answer_prompt:
            prompt = self.get_prompt_from_json_lines(json_lines, item)
            prompt_list.append(prompt)

        return [case_name, prompt_list, question, answer]

    def get_prompt_from_json_lines(self, json_lines, prompt_template):
        """
        Get the json data from prompt template
        """
        keys = re.findall(r"\{(\w+)\}", prompt_template)
        mapping = {key: json_lines.get(key, None).strip() for key in keys}
        return prompt_template.format(**mapping)

    def match_dataset_with_select_data_class(
        self, json_lines, select_data_class: Dict[str, Any] = {}
    ):
        """
        Check whether the dataset meets the specified requirements
        """
        if not select_data_class:
            return True

        for item in select_data_class:
            data = json_lines.get(item, None)
            select_data = select_data_class.get(item)
            if select_data and select_data and data not in select_data:
                return False

        return True
