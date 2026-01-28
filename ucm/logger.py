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

from ucm.shared.infra import ucmlogger

LevelMap = {
    logging.DEBUG: ucmlogger.Level.DEBUG,
    logging.INFO: ucmlogger.Level.INFO,
    logging.WARNING: ucmlogger.Level.WARNING,
    logging.ERROR: ucmlogger.Level.ERROR,
}


class Logger:
    def __init__(self, name: str = "UC", log: dict = {}):
        self.name = name
        directory = log.get("directory", "log")
        max_files = log.get("max_files", 3)
        max_size = log.get("max_size", 5)
        ucmlogger.setup(directory, max_files, max_size)
        atexit.register(ucmlogger.flush)

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


def init_logger(name: str = "UC", log: dict = None) -> Logger:
    if log is None:
        log = {"directory": "log", "max_files": 3, "max_size": 5}
    return Logger(name, log)


if __name__ == "__main__":
    os.environ["UNIFIED_CACHE_LOG_LEVEL"] = "DEBUG"
    logger = init_logger()
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
