import random
import time
import json
import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm
from typing import List, Dict, Any, Union
from transformers import AutoTokenizer, PreTrainedTokenizer
from common.uc_eval.utils.data_class import SynthericParams
from common.uc_eval.utils.utils import get_logger, PathUtil


logger = get_logger()
EPOCH_NUM = 10


class BaseDataset(ABC):
    def __init__(
        self,
        tokenizer_path: str = None,
    ):
        tokenizer_path = PathUtil.get_datasets_dir_path(tokenizer_path)
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    @abstractmethod
    def prepare_data(self, param: Any):
        raise NotImplementedError


class SyntheticDataset(BaseDataset):
    def __init__(self, tokenizer_path: str):
        super().__init__(tokenizer_path)

    def prepare_data(self, syntheric_params: SynthericParams) -> list[str]:
        prompt_list = []
        for parallel_num in tqdm(
            range(syntheric_params.parallel_num), desc="Generate synthetic data", unit="prompt"
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
                diff_ids_list = random.choices(range(vocab_size // 4, vocab_size // 3), k=diff)
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
    def __init__(self, tokenizer_path: str):
        super().__init__(tokenizer_path)

    def prepare_data(self, dataset_file_path) -> List[List[Union[str, Dict]]]:
        """
        Load a JSON file containing multi-turn dialogue dataset paths.
        :param file_path: JSON file listing multi-turn dialogue dataset paths to traverse.
        the multi-turn dataset format: {"kimi": [{"conversion": [{"role": "user", "content": "xxx"}, ...], "qa": [{"question": "xxx", "answer": "xxx"}, ...]}]}
        """
        cases = []
        # the path of multiturndialog.json
        json_path = PathUtil.get_datasets_dir_path(dataset_file_path)
        mtd_data: dict = self.load_json_file(json_path)
        for dataset_name, files_list in mtd_data.items():
            for file_name in files_list:
                case_path = PathUtil.get_dirname(json_path).joinpath(dataset_name, file_name)
                if case_path.exists():
                    dialogues = self.load_json_file(case_path)
                    cases.extend(self.process_single_case_file(dialogues))
                else:
                    logger.warning(
                        f"JSON file {case_path} does not exist, please check the file path"
                    )
        if len(cases) == 0:
            logger.warning(f"The file {json_path} does not contain multi-turn dialogue data")
        return cases

    def process_single_case_file(self, dialogues: dict) -> List[List[Union[str, Dict]]]:
        cases = []
        for dialogue_name, dialogue_data in dialogues.items():
            for i, dialog in enumerate(dialogue_data):
                dialog_tokens = len(self.tokenizer.tokenize(str(dialog["conversations"])))
                logger.info(f"Current dialogue {dialogue_name}-{i} token count: {dialog_tokens}")
                cases.append([f"{dialogue_name}-{i}", dialog])
        return cases

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


class DocQADataset(BaseDataset):
    def __init__(self, tokenizer_path: str):
        super().__init__(tokenizer_path)

    def prepare_data(self, dataset_file_path) -> List[Union[str, str, str]]:
        cases_list = []
        case_data = self._load_jsonl_file(dataset_file_path)
        for case in case_data:
            context = case.get("context")
            question = case.get("question")
            answer = case.get("answers")[0]
            case_name = case.get("dataset") + "_" + case.get("_id")
            cases_list.append([case_name, context, question, answer])
        return cases_list

    def _load_jsonl_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load a JSONL file containing doc_qa data
        :param file_path: Path to the jsonl file
        :return: List of doc_qa data
        """
        case_data = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    # In doc_qa, one line per sample; each sample contains: question, context, answer, etc.
                    json_line = json.loads(line)
                    extracted_data = {
                        "question": json_line.get("input", None),
                        "context": json_line.get("context", None),
                        "answers": json_line.get("answers", None),
                        "length": json_line.get("length", None),
                        "dataset": json_line.get("dataset", None),
                        "language": json_line.get("language", None),
                        "all_classes": json_line.get("all_classes", None),
                        "_id": json_line.get("_id", None),
                    }
                    case_data.append(extracted_data)
            return case_data
        except FileNotFoundError:
            logger.error(f"JSONL file not found: {file_path}")
            raise FileNotFoundError(f"JSONL file not found: {file_path}")
        except json.JSONDecodeError as e:
            logger.error(f"JSONL decode error in file {file_path}: {e}")
            raise ValueError(f"Invalid JSONL format in file {file_path}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error while loading JSONL file {file_path}: {e}")
            raise ValueError(f"Failed to load JSONL file {file_path}: {e}")
