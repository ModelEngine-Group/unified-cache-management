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

import os
import inspect

from ucm.shared.infra import spdlog_logger
from ucm.shared.infra import source_location


class Logger:
    def __init__(self):
        self.logger = spdlog_logger.Logger()

    def get_source_location(self):
        """Helper function to print the current file, function, and line number."""
        frame = inspect.currentframe()
        caller_frame = frame.f_back.f_back
        filename = os.path.basename(caller_frame.f_code.co_filename)
        lineno = caller_frame.f_lineno
        func_name = caller_frame.f_code.co_name
        return source_location.SourceLocation(filename, func_name, lineno)
    
    def debug(self, message: str, *args):
        self.logger.debug(message, self.get_source_location(), *args)

    def info(self, message: str, *args):
        self.logger.info(message, self.get_source_location(), *args)

    def warning(self, message: str, *args):
        self.logger.warning(message, self.get_source_location(), *args)

    def error(self, message: str, *args):
        self.logger.error(message, self.get_source_location(), *args)

def init_logger(name: str = "UNIFIED_CACHE")->Logger:
    return Logger()

def test_logger():
    logger = init_logger()
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.info("info message with format: {} {}", "test", "test2")

def test_logger_multi_thread():
    import threading
    for i in range(10):
        logger = init_logger()
        threading.Thread(target=logger.debug, args=("debug message", i))
        threading.Thread(target=logger.info, args=("info message", i))
        threading.Thread(target=logger.warning, args=("warning message", i))
        threading.Thread(target=logger.error, args=("error message", i))
        threading.Thread(target=logger.info, args=("[Thread %d]info message with format: %s %s ", i, "test", "test2"))
    

if __name__ == "__main__":
    os.environ["UNIFIED_CACHE_LOG_LEVEL"] = "DEBUG"
    test_logger_multi_thread()
