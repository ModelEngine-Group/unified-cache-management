import os
import sys
import time
import logging
from typing import Dict
from pathlib import Path
import logging.handlers

current_dir = os.path.dirname(os.path.abspath(__file__))


def get_current_time() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


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


class LoggerHandler(logging.Logger):
    def __init__(self, name: str, level: int = logging.INFO, log_path: str = None) -> None:
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
