import json
import logging
import logging.handlers
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd
from transformers import AutoConfig, AutoTokenizer

current_dir = os.path.dirname(os.path.abspath(__file__))


def get_current_time() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))


class PathUtil(object):

    @staticmethod
    def get_dirname(file_path: str | Path):
        return Path(os.path.dirname(file_path))

    @staticmethod
    def get_root_dir_path() -> Path:
        root_path = Path(current_dir).parent
        return root_path

    @staticmethod
    def get_other_dir_path(other: str) -> Path:
        root_path = PathUtil.get_root_dir_path()
        other_path = Path.joinpath(root_path, other)
        if not other_path.is_file():
            other_path.mkdir(parents=True, exist_ok=True)
        return other_path

    @staticmethod
    def _default_datasets_path() -> Path:
        return PathUtil.get_other_dir_path("UC-Eval-datasets")

    @staticmethod
    def get_datasets_dir_path(in_file_path: str) -> Path:
        if not in_file_path or in_file_path == "":
            return PathUtil._default_datasets_path()
        input_path = Path(in_file_path)
        if input_path.is_absolute():
            return Path(in_file_path)
        else:
            return PathUtil.get_other_dir_path(in_file_path)


class JsonAndJsonlLoader(object):

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def load_json_file(self, file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except FileNotFoundError:
            self.logger.error(f"JSON file not found: {file_path}")
            raise FileNotFoundError(f"JSON file not found: {file_path}")
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error in file {file_path}: {e}")
            raise ValueError(f"Invalid JSON format in file {file_path}: {e}")
        except Exception as e:
            self.logger.error(
                f"Unexpected error while loading JSON file {file_path}: {e}"
            )
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
            self.logger.error(f"JSONL file not found: {file_path}")
            raise FileNotFoundError(f"JSONL file not found: {file_path}")
        except json.JSONDecodeError as e:
            self.logger.error(f"JSONL decode error in file {file_path}: {e}")
            raise ValueError(f"Invalid JSONL format in file {file_path}: {e}")
        except Exception as e:
            self.logger.error(
                f"Unexpected error while loading JSONL file {file_path}: {e}"
            )
            raise ValueError(f"Failed to load JSONL file {file_path}: {e}")

    def save_jsonl_file(self, save_file_path: str, data_list: List[Dict]):
        self.logger.info(f"Begin save jsonl to {save_file_path}")
        save_file_path = Path(save_file_path)
        save_file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_file_path, "w", encoding="utf-8") as f:
            for json_line in data_list:
                f.write(json.dumps(json_line, ensure_ascii=False) + "\n")


class FileUtil(object):

    _ILLEGAL_CHARS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
    _MAX_CELL_LEN = 32768

    @staticmethod
    def _sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
        """
        Iterate over every cell, remove illegal control characters, and truncate if too long.
        """

        def _clean_cell(cell):
            if isinstance(cell, str):
                cell = FileUtil._ILLEGAL_CHARS.sub("", cell)
                if len(cell) > FileUtil._MAX_CELL_LEN:
                    cell = cell[: FileUtil._MAX_CELL_LEN - 3] + "..."
            return cell

        for col in df.columns:
            df[col] = df[col].map(_clean_cell)
        return df

    @staticmethod
    def save_excel(
        file_path: Path,
        data: List[Any],
        headers: List[str] = None,
        sheet_name: str = "Sheet1",
    ):
        """
        Write test results to excel, one List[Any] represents one row of data
        """
        df = (
            pd.DataFrame(data=data, columns=headers)
            if headers
            else pd.DataFrame(data=data)
        )
        df = FileUtil._sanitize_df(df)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if file_path.exists():
            with pd.ExcelWriter(
                file_path, mode="a", engine="openpyxl", if_sheet_exists="overlay"
            ) as writer:
                workbook = writer.book
                # If the excel and sheet exist, append write
                if sheet_name in workbook.sheetnames:
                    existing_df = pd.read_excel(file_path, sheet_name=sheet_name)
                    start_now = existing_df.shape[0] + 1
                    df.to_excel(
                        writer,
                        sheet_name=sheet_name,
                        index=False,
                        startrow=start_now,
                        header=False if start_now > 0 else True,
                    )
                else:
                    # If the excel exists but the sheet does not, create a new sheet and write
                    df.to_excel(
                        writer,
                        sheet_name=sheet_name,
                        index=False,
                        header=(headers is not None),
                    )
        else:
            # if the excel does not exist, create a new excel and sheet
            with pd.ExcelWriter(file_path, mode="w", engine="openpyxl") as writer:
                df.to_excel(
                    writer,
                    sheet_name=sheet_name,
                    index=False,
                    header=(headers is not None),
                )


class LoggerHandler(logging.Logger):
    def __init__(
        self, name: str, level: int = logging.INFO, log_path: str = None
    ) -> None:
        super().__init__(name, level)
        # format of the log message
        fmt = "%(asctime)s.%(msecs)03d %(levelname)s [pid:%(process)d] [%(threadName)s] [tid:%(thread)d] [%(filename)s:%(lineno)d %(funcName)s] %(message)s"
        data_fmt = "%Y-%m-%d %H:%M:%S"
        formatter = logging.Formatter(fmt, data_fmt)

        # using file handler to log to file
        if log_path is not None:
            file_handler = logging.handlers.RotatingFileHandler(
                filename=log_path,
                maxBytes=1024 * 1024 * 10,
                backupCount=20,
                delay=True,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(self.level)
            self.addHandler(file_handler)

        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(self.level)
        self.addHandler(console_handler)

    def setLevel(self, level) -> None:
        super().setLevel(level)
        for handler in self.handlers:
            handler.setLevel(level)


def _get_level_from_env() -> int:
    """
    Get the log level from environment variable
    """
    level = os.environ.get("UC_LOG_LEVEL", "INFO")
    level = level.upper()
    return getattr(logging, level, logging.INFO)


# the global dictionary to store all the logger instances
_logger_instances: Dict[str, LoggerHandler] = {}
_DEFAULT_LOG_LEVEL = _get_level_from_env()
_LOGGER_FILE_PATH = Path(current_dir).parent.joinpath("uc_log", "log.log")


def get_logger(
    name: str = "evals", level: int = logging.INFO, log_file: str = None
) -> logging.Logger:
    level = _DEFAULT_LOG_LEVEL or level
    if name in _logger_instances:
        log = _logger_instances[name]
        log.setLevel(level)
        return log

    log_file = log_file or _LOGGER_FILE_PATH
    if not log_file.parent.exists():
        log_file.parent.mkdir(parents=True, exist_ok=True)

    # create a new logger instance
    logger = LoggerHandler(name, level, log_file)
    _logger_instances[name] = logger
    return logger


COL_WIDTHS = [
    40,
    40,
    15,
    15,
    15,
    15,
]  # Category, Sub_Category, Total, Correct, Accuracy
VALID_DIFFICULTY = ("easy", "hard")
VALID_LENGTH = ("short", "medium", "long")
ORDER_DIFFICULTY = {"easy": 0, "hard": 1}
ORDER_LENGTH = {"short": 0, "medium": 1, "long": 2}
EXCEL_HEADERS = ["Domain", "Sub-domain", "Total", "Correct", "Accuracy"]


class JsonlDataToExcel:
    def __init__(self, logger, jsonl_file_path):
        self.logger = logger
        self.save_jsonl_path = Path(jsonl_file_path)
        self.save_evcel_path = Path(str(jsonl_file_path).replace(".jsonl", ".xlsx"))

    @staticmethod
    def normalize_data(value: str, valid_set: tuple, default: str = "Unknown") -> str:
        """
        Normalize the length and difficulty of the data
        """
        if not isinstance(value, str):
            return default
        value = value.strip().lower()
        return value if value in valid_set else default

    @staticmethod
    def calculate_accuracy(stat: dict) -> float:
        """
        Get the accuracy of stat dict
        """
        return (
            stat.get("correct", 0) / stat.get("total") if stat.get("total") > 0 else 0.0
        )

    def get_data_list_from_jsonl(self, save_jsonl_file_path: str):
        """
        Get valid data from jsonl file
        """
        data_list, data_ids, duplicate_id_list = [], set(), []
        with open(save_jsonl_file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    _id = data.get("_id")
                    if _id in data_ids:
                        duplicate_id_list.append(_id)
                        continue
                    else:
                        data_ids.add(_id)
                        data_list.append(data)
                except Exception as e:
                    self.logger.error(f"line_num: {line_num}, line: {line}, error: {e}")
                    continue

        self.logger.info(
            f"There are {len(data_list)} valid data in {save_jsonl_file_path}, of which {len(duplicate_id_list)} are duplicated."
        )

        return data_list

    @staticmethod
    def get_status_dict_from_data_list(data_list: list[dict]):
        # Initialize statistics
        Stats = lambda: {"total": 0, "correct": 0}
        domain_stats = defaultdict(Stats)
        domain_sub_stats = defaultdict(Stats)
        difficulty_stats = defaultdict(Stats)
        length_stats = defaultdict(Stats)
        correct_num = 0

        if not data_list or len(data_list) == 0:
            return (
                domain_stats,
                domain_sub_stats,
                difficulty_stats,
                length_stats,
                correct_num,
            )

        # Traverse the data and calculate the accuracy information of different domains, lengths, and difficulties
        for data in data_list:
            domain = data.get("domain", "Unknown")
            sub_domain = data.get("sub_domain", "Unknown")
            difficulty = JsonlDataToExcel.normalize_data(
                data.get("difficulty", "Unknown"), ("easy", "medium", "hard")
            )
            length = JsonlDataToExcel.normalize_data(
                data.get("length", "Unknown"), ("short", "medium", "long")
            )
            is_match = data.get("is_match", False)
            correct_num += int(is_match)

            for stats_dict, key in [
                (domain_stats, domain),
                (domain_sub_stats, (domain, sub_domain)),
                (difficulty_stats, difficulty),
                (length_stats, length),
            ]:
                stats_dict[key]["total"] += 1
                if is_match:
                    stats_dict[key]["correct"] += 1

        return (
            domain_stats,
            domain_sub_stats,
            difficulty_stats,
            length_stats,
            correct_num,
        )

    @staticmethod
    def build_domain_data_list(
        domain_stats: dict, sub_domain_stats: dict
    ) -> pd.DataFrame:
        """
        Build domain DataFrame
        """
        rows = []
        for domain in sorted(domain_stats.keys()):
            total, correct = domain_stats[domain].get("total", 0), domain_stats[
                domain
            ].get("correct", 0)
            rows.append(
                [
                    domain,
                    "-",
                    total,
                    correct,
                    JsonlDataToExcel.calculate_accuracy(domain_stats[domain]),
                ]
            )

            # sub domain data
            sub_domains = sorted(s for d, s in sub_domain_stats if d == domain)
            for sub in sub_domains:
                sub_data = sub_domain_stats[(domain, sub)]
                sub_total, sub_correct = sub_data.get("total", 0), sub_data.get(
                    "correct", 0
                )
                rows.append(
                    [
                        domain,
                        sub,
                        sub_total,
                        sub_correct,
                        JsonlDataToExcel.calculate_accuracy(
                            sub_domain_stats[(domain, sub)]
                        ),
                    ]
                )

        return rows

    @staticmethod
    def build_category_data_list(
        stats_dict: dict, order_map: dict[str, int] = None
    ) -> pd.DataFrame:
        """
        General statistical dict to DataFrame, including difficulty, length
        :param stats_dict: the statistics dict, {key: {"total": x, "correct": y}}
        :param: order_map: the order of the output columns, {key_name: index}
        """
        items = stats_dict.items()
        if order_map:
            items = sorted(
                items, key=lambda x: (order_map.get(x[0], len(order_map)), x[0])
            )
        else:
            items = sorted(items)

        rows = []
        for key, value in items:
            total = value.get("total", 0)
            correct = value.get("correct", 0)
            rows.append(
                [
                    key,
                    "-",
                    total,
                    correct,
                    JsonlDataToExcel.calculate_accuracy(stats_dict[key]),
                ]
            )

        return rows

    def trans_jsonl_to_excel(self):
        results = []
        data_list = self.get_data_list_from_jsonl(self.save_jsonl_path)
        domain_stats, domain_sub_stats, difficulty_stats, length_stats, correct_data = (
            JsonlDataToExcel.get_status_dict_from_data_list(data_list)
        )
        results.append(
            [
                "overall",
                "-",
                len(data_list),
                correct_data,
                correct_data / len(data_list) if len(data_list) > 0 else 0.0,
            ]
        )

        difficult_data_list = JsonlDataToExcel.build_category_data_list(
            difficulty_stats, ORDER_DIFFICULTY
        )
        length_data_list = JsonlDataToExcel.build_category_data_list(
            length_stats, ORDER_LENGTH
        )
        domain_data_list = JsonlDataToExcel.build_domain_data_list(
            domain_stats, domain_sub_stats
        )
        data_to_excel = [difficult_data_list, length_data_list, domain_data_list]
        for data in data_to_excel:
            if data:
                keys = set(data[0] for data in data)
                if len(keys) == 1 and keys.pop() == "Unknown":
                    continue
                results.extend(data)

        FileUtil.save_excel(
            self.save_evcel_path, results, EXCEL_HEADERS, "Evaluation Result"
        )
        self.logger.info(
            f"Successfully transform jsonl to excel, file name: {self.save_evcel_path}"
        )


class ModelMemoryCalculator:
    def __init__(self, model_path: Union[Path, str]):
        if isinstance(model_path, str):
            model_path = PathUtil.get_datasets_dir_path(model_path)
        self.config = AutoConfig.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.dtype_bytes_map = {"fp16": 2, "bf16": 2, "fp32": 4, "int8": 1}

    def _get_model_info(self):
        """
        Get model architecture information
        """
        hidden_size = getattr(self.config, "hidden_size", None)
        num_layers = getattr(self.config, "num_hidden_layers", None)
        num_attention_heads = getattr(self.config, "num_attention_heads", None)
        num_kv_heads = getattr(self.config, "num_key_value_heads", num_attention_heads)
        qk_rope_head_dim = getattr(self.config, "qk_rope_head_dim", None)
        kv_lora_rank = getattr(self.config, "kv_lora_rank", None)

        head_dim = self._calculate_head_dimension(
            hidden_size, num_attention_heads, qk_rope_head_dim, kv_lora_rank
        )

        return {
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_attention_heads": num_attention_heads,
            "num_kv_heads": num_kv_heads,
            "qk_rope_head_dim": qk_rope_head_dim,
            "kv_lora_rank": kv_lora_rank,
            "head_dim": head_dim,
            "model_type": self.config.model_type,
            "element_calculate_type": 1 if qk_rope_head_dim and kv_lora_rank else 0,
        }

    def _calculate_head_dimension(
        self, hidden_size, num_attention_heads, qk_rope_head_dim, kv_lora_rank
    ):
        """
        Calculate head dimension
        """
        # First, check if both qk_rope_head_dim and kv_lora_rank parameters exist; if so, use these two parameters for calculation.
        if qk_rope_head_dim is not None and kv_lora_rank is not None:
            return qk_rope_head_dim + kv_lora_rank

        # Then, check if there is a head_dim parameter available and use it if present.
        head_dim = getattr(self.config, "head_dim", None)
        if head_dim is not None:
            return head_dim

        # Next, check if both hidden_size and num_attention_heads parameters exist; if so, use these two parameters for calculation.
        if hidden_size is not None and num_attention_heads is not None:
            if num_attention_heads == 0:
                raise ValueError("num_attention_heads cannot be zero")
            return hidden_size // num_attention_heads

        # If none of the above exist, raise an error.
        raise ValueError(
            "Unable to calculate head dimension with current model configuration. "
            "Please check if the model configuration contains required parameters."
        )

    def calculate_kv_cache_memory(self, sequence_length, batch_size=1, dtype="fp16"):
        """
        Calculate KV Cache memory usage:
        For models like DeepSeek-R1: batch_size * sequence_length * num_hidden_layers * head_dim * bytes_per_element
        For models like Qwen3-32B: 2 * batch_size * sequence_length * num_hidden_layers * num_kv_heads * head_dim * bytes_per_element
        :param sequence_length: Sequence length (number of tokens)
        :param batch_size: Batch size
        :param dtype: Data type ('fp16', 'bf16', 'fp32', 'int8')
        """
        model_info = self._get_model_info()

        # Check required parameters
        required_params = ["num_layers", "head_dim"] + (
            [] if model_info["element_calculate_type"] else ["num_attention_heads"]
        )
        for param in required_params:
            if model_info[param] is None:
                raise ValueError(f"Cannot retrieve {param} from configuration file")

        # Round up any input sequence_length to the nearest multiple of 128
        sequence_length = math.ceil(sequence_length / 128) * 128
        bytes_per_element = self.dtype_bytes_map.get(dtype, 2)

        if model_info["element_calculate_type"]:
            total_elements = (
                batch_size
                * sequence_length
                * model_info["num_layers"]
                * model_info["head_dim"]
            )
        else:
            # Use KV heads count from configuration, if not available use attention heads count
            num_kv_heads = (
                model_info["num_kv_heads"] or model_info["num_attention_heads"]
            )
            total_elements = (
                batch_size
                * sequence_length
                * model_info["num_layers"]
                * num_kv_heads
                * model_info["head_dim"]
                * 2  # key + value
            )

        memory_bytes = total_elements * bytes_per_element
        memory_gb = memory_bytes / (1024**3)

        return total_elements, round(memory_gb, 4)
