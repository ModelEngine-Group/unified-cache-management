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
import collections
import inspect
import io
import logging
import os
import sys
import traceback
from collections.abc import Hashable
from functools import lru_cache

from ucm.shared.infra import ucmlogger

LevelMap = {
    logging.DEBUG: ucmlogger.Level.DEBUG,
    logging.INFO: ucmlogger.Level.INFO,
    logging.WARNING: ucmlogger.Level.WARNING,
    logging.ERROR: ucmlogger.Level.ERROR,
    logging.CRITICAL: ucmlogger.Level.CRITICAL,
}


def add_log_methods(cls):
    LOG_LEVELS = {
        "info": logging.INFO,
        "debug": logging.DEBUG,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    def _create_log_method(level):
        def log_method(self, message: str, *args, **kwargs):
            self.log(level, message, *args, **kwargs)

        return log_method

    for method_name, level in LOG_LEVELS.items():
        setattr(cls, method_name, _create_log_method(level))
    return cls


@add_log_methods
class Logger(logging.Logger):
    def __init__(self, name: str = "UC"):
        self.name = name
        log_path, log_max_files, log_max_size = self._get_log_config()
        ucmlogger.setup(log_path, log_max_files, log_max_size)
        atexit.register(ucmlogger.flush)

    def isEnabledFor(self, levelno: int) -> bool:
        return ucmlogger.isEnabledFor(LevelMap[levelno])

    @staticmethod
    def _get_log_config():
        """Get log configuration from environment variables or CLI arguments."""
        log_path = os.getenv("UCM_LOG_PATH", "log")
        try:
            log_max_files = int(os.getenv("UCM_LOG_MAX_FILES", "10"))
        except (ValueError, TypeError):
            log_max_files = 10
        try:
            log_max_size = int(os.getenv("UCM_LOG_MAX_SIZE", "5"))
        except (ValueError, TypeError):
            log_max_size = 5
        return log_path, log_max_files, log_max_size

    @staticmethod
    def format_log_msg(msg, *args) -> str:

        if not isinstance(msg, str):
            msg = str(msg)

        if args:
            if (
                len(args) == 1
                and args[0]
                and isinstance(args[0], collections.abc.Mapping)
            ):
                args = args[0]
            return msg % args
        return msg

    def log(self, levelno, message, *args, exc_info=None, scope=None):
        level = LevelMap[levelno]
        frame = inspect.currentframe()
        caller_frame = frame.f_back.f_back
        file = os.path.basename(caller_frame.f_code.co_filename)
        line = caller_frame.f_lineno
        func = caller_frame.f_code.co_name
        msg = self.format_log_msg(message, *args)
        if exc_info:
            exc_text = self.format_exception(exc_info)
            msg = msg + "\n" + exc_text
        ucmlogger.log(level, file, func, line, msg)

    @staticmethod
    def format_exception(e):
        if isinstance(e, BaseException):
            e = (type(e), e, e.__traceback__)
        elif not isinstance(e, tuple):
            e = sys.exc_info()
        sio = io.StringIO()
        tb = e[2]
        traceback.print_exception(e[0], e[1], tb, None, sio)
        s = sio.getvalue()
        sio.close()
        if s[-1:] == "\n":
            s = s[:-1]
        return s

    @lru_cache
    def info_once(self, message: str, *args: Hashable, **kwargs: Hashable):
        self.log(logging.INFO, message, *args, **kwargs)

    @lru_cache
    def warning_once(self, message: str, *args: Hashable, **kwargs: Hashable):
        self.log(logging.WARNING, message, *args, **kwargs)

    @lru_cache
    def debug_once(self, message: str, *args: Hashable, **kwargs: Hashable):
        self.log(logging.DEBUG, message, *args, **kwargs)

    def exception(self, message: str, *args: Hashable, **kwargs: Hashable):
        self.log(logging.ERROR, message, *args, **kwargs, exc_info=True)


def init_logger(name: str = "UC") -> Logger:
    return Logger(name)


def current_formatter_type(lgr):
    return None


if __name__ == "__main__":
    logger = init_logger()
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
