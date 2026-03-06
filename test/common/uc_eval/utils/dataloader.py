import json
import random
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import numpy as np
from common.uc_eval.utils.data_class import SynthericParams
from common.uc_eval.utils.utils import JsonAndJsonlLoader, PathUtil, get_logger
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
        self.json_loader = JsonAndJsonlLoader(logger)

    @abstractmethod
    def prepare_data(self, param: Any):
        raise NotImplementedError


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
        mtd_data: dict = self.json_loader.load_json_file(json_path)
        for dataset_name, files_list in mtd_data.items():
            for file_name in files_list:
                case_path = PathUtil.get_dirname(json_path).joinpath(
                    dataset_name, file_name
                )
                if case_path.exists():
                    dialogues = self.json_loader.load_json_file(case_path)
                    if isinstance(dialogues, Dict):
                        cases.extend(self.process_single_case_file(dialogues))
                    elif isinstance(dialogues, List):
                        cases.extend(self.process_multi_case_file(dialogues))
                    else:
                        logger.warning(
                            f" The dialogue format is invalid. Please check and retry."
                        )
                else:
                    logger.warning(
                        f"JSON file {case_path} does not exist, please check the file path"
                    )
        if len(cases) == 0:
            logger.warning(
                f"The file {json_path} does not contain multi-turn dialogue data"
            )
        return cases

    def process_single_case_file(self, dialogues: Dict) -> List[List[Union[str, Dict]]]:
        cases = []
        for dialogue_name, dialogue_data in dialogues.items():
            logger.info(f"There are  {len(dialogue_data)} dialogue cases.")
            for i, dialog in enumerate(dialogue_data):
                dialog_tokens = len(
                    self.tokenizer.tokenize(str(dialog["conversations"]))
                )
                logger.info(
                    f"Current dialogue {dialogue_name}-{i} token count: {dialog_tokens}"
                )
                cases.append([f"{dialogue_name}-{i}", dialog])

        return cases

    def process_multi_case_file(self, dialogues: List) -> List[List[Union[str, Dict]]]:
        cases = []
        logger.info(f"There are  {len(dialogues)} dialogue cases.")
        for conv in dialogues:
            case_name = conv["id"]
            conversations = conv["conversations"]
            dialog_tokens = len(self.tokenizer.tokenize(str(conversations)))
            logger.info(f"Current dialogue {case_name} token count: {dialog_tokens}")
            cases.append([case_name, conv])

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
            data_list = self.json_loader.load_jsonl_data(file_path)
        elif file_path.suffix.lower() == ".json":
            data_list = self.json_loader.load_json_file(file_path)

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
        from common.uc_eval.utils.prompt_config import (
            doc_qa_prompt_en,
            doc_qa_prompt_zh,
        )

        question = json_lines.get("question") or json_lines.get("input")
        answer = json_lines.get("answers") or json_lines.get("answer")
        _id = json_lines.get("_id", None)
        language = self.resolve_language(json_lines)

        is_match = self.match_dataset_with_select_data_class(
            json_lines, select_data_class
        )
        if not is_match:
            return []

        prompt_list = []
        prompt_tmp = doc_qa_prompt_zh if language == "zh" else doc_qa_prompt_en
        for item in prompt_tmp:
            prompt = self.get_prompt_from_json_lines(json_lines, item)
            prompt_list.append(prompt)

        return [_id, prompt_list, question, answer]

    def _get_multiple_choice_content(
        self, json_lines, select_data_class: Dict[str, Any] = {}
    ):
        """
        For multiple-choice questions, after extracting "answer" and "question", also extract keys like "choice_A" to distinguish each option before building the prompt.
        """
        from common.uc_eval.utils.prompt_config import COT_KEY, multi_answer_prompt

        question = json_lines.get("question") or json_lines.get("input")
        answer = json_lines.get("answers") or json_lines.get("answer")
        domain = json_lines.get("domain", None)
        difficulty = json_lines.get("difficulty", None)
        _id = json_lines.get("_id", None)
        language = self.resolve_language(json_lines)

        is_match = self.match_dataset_with_select_data_class(
            json_lines, select_data_class
        )
        if not is_match:
            return []

        prompt_list = []
        for item in multi_answer_prompt:
            prompt, cot_string = self.get_prompt_from_json_lines(
                json_lines, item, COT_KEY
            )
            prompt_list.append([prompt, cot_string])

        return [_id, prompt_list, question, answer]

    def resolve_language(self, json_lines) -> str:
        """
        Extract language identifier from data, use default configuration if not specified
        """
        from common.uc_eval.utils.prompt_config import DEFAULT_LANGUAGE

        language = json_lines.get("language") or DEFAULT_LANGUAGE
        return language or "None"

    def get_prompt_from_json_lines(self, json_lines, prompt_template, cot_key="COT"):
        """
        Get the json data from prompt template
        """
        keys = list(re.finditer(r"\{([^}]+)\}", prompt_template))
        mapping = {}
        cot_identifier = None

        for i, match in enumerate(keys):
            key = match.group(1)
            if isinstance(key, str) and key.upper() == cot_key:
                if i == 0:
                    start_pos = 0
                else:
                    start_pos = keys[i - 1].end()

                end_pos = match.end()
                cot_identifier = prompt_template[start_pos:end_pos]

            else:
                value = json_lines.get(key)
                if value is None:
                    logger.error(f"Missing key {key} in json lines")
                    mapping[key] = ""
                else:
                    mapping[key] = str(value).strip()

        filled_prompt = prompt_template
        for key, value in mapping.items():
            filled_prompt = filled_prompt.replace(f"{{{key}}}", value)

        return filled_prompt, cot_identifier

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
