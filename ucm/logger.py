#
# MIT License
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

import atexit
import inspect
import logging
import os

import yaml

from ucm.shared.infra import ucmlogger

LevelMap = {
    logging.DEBUG: ucmlogger.Level.DEBUG,
    logging.INFO: ucmlogger.Level.INFO,
    logging.WARNING: ucmlogger.Level.WARNING,
    logging.ERROR: ucmlogger.Level.ERROR,
}


class Logger:
    def __init__(self, name: str = "UC", config_file: str = None):
        self.name = name
        log_config = {}
        if config_file:
            config = self.load_config(config_file)
            if config:
                log_config = config.get("log_config", {})
        path = log_config.get("path", "log/ucm.log")
        max_files = log_config.get("max_files", 3)
        max_size = log_config.get("max_size", 5)
        ucmlogger.setup(path, max_files, max_size)

    def load_config(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        return config

    def log(self, levelno, message, *args):
        level = LevelMap[levelno]
        frame = inspect.currentframe()
        caller_frame = frame.f_back.f_back
        file = os.path.basename(caller_frame.f_code.co_filename)
        line = caller_frame.f_lineno
        func = caller_frame.f_code.co_name
        msg = ucmlogger.format(message, args)
        ucmlogger.log(level, file, func, line, msg)

    def info(self, message: str, *args):
        self.log(logging.INFO, message, *args)

    def debug(self, message: str, *args):
        self.log(logging.DEBUG, message, *args)

    def warning(self, message: str, *args):
        self.log(logging.WARNING, message, *args)

    def error(self, message: str, *args):
        self.log(logging.ERROR, message, *args)

    def warning_once(self, message: str, *args):
        self.log(logging.WARNING, message, *args)

    def flush(self):
        ucmlogger.flush()


def init_logger(name: str = "UC", config_file: str = None) -> Logger:
    return Logger(name, config_file)


def _shutdown_logger_on_exit():
    """Shutdown the logger when the program exits to avoid segfault during static destruction."""
    try:
        ucmlogger.flush()
    except Exception:
        pass  # Ignore errors during exit


# Register shutdown function to be called at program exit
atexit.register(_shutdown_logger_on_exit)

if __name__ == "__main__":
    os.environ["UNIFIED_CACHE_LOG_LEVEL"] = "DEBUG"
    logger = init_logger()
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
