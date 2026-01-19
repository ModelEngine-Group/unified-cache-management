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
import sys


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
    return {"directory": log_path, "max_files": log_max_files, "max_size": log_max_size}


_log_config = _get_log_config()

from ucm.logger import init_logger

_ucm_logger = init_logger("UC", _log_config)
_ucm_logger.info(f"Logger initialized successfully")
