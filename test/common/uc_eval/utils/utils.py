import logging
import logging.handlers
import math
import os
import sys
import time
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
        root_path = Path(current_dir).parent.parent
        return root_path

    @staticmethod
    def get_other_dir_path(other: str) -> Path:
        root_path = PathUtil.get_root_dir_path()
        other_path = Path.joinpath(root_path, other)
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


class FileUtil(object):
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


# the global dictionary to store all the logger instances
_logger_instances: Dict[str, LoggerHandler] = {}


def get_logger(
    name: str = "evals", level: int = logging.INFO, log_file: str = None
) -> logging.Logger:
    if name in _logger_instances:
        return _logger_instances[name]

    # create a new logger instance
    logger = LoggerHandler(name, level, log_file)
    _logger_instances[name] = logger
    return logger


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
