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

"""
UCM Logger Module

Environment Variables:
    UCM_LOG_LEVEL: Log level for both the Python and C++ side
                   (debug/info/warning/error/critical, default: info).
                   Falls back to legacy ``UC_LOGGER_LEVEL`` if unset.
    UCM_LOG_PATH: Log file directory (default: "log")
    UCM_LOG_MAX_FILES: Max rotated files per process (default: 10)
    UCM_LOG_MAX_SIZE: Max size in MiB per file (default: 5)
    UCM_LOG_TO_FILE: Enable the per-process log file (default: true)
    UCM_CAPTURE_VLLM_LOG: Also write vLLM's logs to <dir>/vllm-<pid>.log
                          (default: true, see logger_patch)

    UCM_LOG_RATE_LIMIT_ENABLE: Enable/disable rate limiting (default: true)
    UCM_LOG_RATE_LIMIT_WINDOW_MS: Time window in milliseconds (default: 60000 = 60s)
    UCM_LOG_RATE_LIMIT_MAX_LOGS: Max logs per window (default: 3, max: 3)

Usage:
    logger = init_logger(__name__)

    # Rate-limited logging (60s window, max 3 logs per location)
    logger.info_limit("Processing request %s", req_id)

    # One-time logging (cached by lru_cache)
    logger.info_once("Cache hit rate: %.2f", rate)
"""

import atexit
import logging
import os
from functools import lru_cache
from types import MethodType
from typing import Optional

from ucm.shared.infra import ucmlogger

_LEVEL_MAP = {
    logging.DEBUG: ucmlogger.Level.DEBUG,
    logging.INFO: ucmlogger.Level.INFO,
    logging.WARNING: ucmlogger.Level.WARNING,
    logging.ERROR: ucmlogger.Level.ERROR,
    logging.CRITICAL: ucmlogger.Level.CRITICAL,
}

_ROOT_NAME = "ucm"

_EXC_FORMATTER = logging.Formatter()


def _to_ucm_level(levelno: int):
    exact = _LEVEL_MAP.get(levelno)
    if exact is not None:
        return exact
    if levelno >= logging.CRITICAL:
        return ucmlogger.Level.CRITICAL
    if levelno >= logging.ERROR:
        return ucmlogger.Level.ERROR
    if levelno >= logging.WARNING:
        return ucmlogger.Level.WARNING
    if levelno >= logging.INFO:
        return ucmlogger.Level.INFO
    return ucmlogger.Level.DEBUG


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _resolve_level() -> int:
    raw = os.getenv("UCM_LOG_LEVEL") or os.getenv("UC_LOGGER_LEVEL") or "info"
    raw = raw.strip().lower()
    aliases = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warn": logging.WARNING,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "err": logging.ERROR,
        "critical": logging.CRITICAL,
        "off": logging.CRITICAL + 10,
    }
    return aliases.get(raw, logging.INFO)


class UcmBridgeHandler(logging.Handler):
    def __init__(self, level: int = logging.NOTSET, file_only: bool = False):
        super().__init__(level)
        self._log_file_only = getattr(ucmlogger, "log_file_only", None)
        self._file_only = file_only and self._log_file_only is not None

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
            if record.exc_info:
                msg = f"{msg}\n{_EXC_FORMATTER.formatException(record.exc_info)}"
            if record.stack_info:
                msg = f"{msg}\n{record.stack_info}"
            level = _to_ucm_level(record.levelno)
            file = os.path.basename(record.pathname)
            if getattr(record, "ucm_rate_limit", False):
                ucmlogger.log_rate_limit(
                    level, file, record.funcName, record.lineno, msg
                )
            elif self._file_only:
                self._log_file_only(level, file, record.funcName, record.lineno, msg)
            else:
                ucmlogger.log(level, file, record.funcName, record.lineno, msg)
        except Exception:
            self.handleError(record)


@lru_cache
def _print_debug_once(logger: logging.Logger, msg: str, *args) -> None:
    logger.debug(msg, *args, stacklevel=3)


@lru_cache
def _print_info_once(logger: logging.Logger, msg: str, *args) -> None:
    logger.info(msg, *args, stacklevel=3)


@lru_cache
def _print_warning_once(logger: logging.Logger, msg: str, *args) -> None:
    logger.warning(msg, *args, stacklevel=3)


_RATE_LIMIT_EXTRA = {"ucm_rate_limit": True}


def _debug_limit(logger: logging.Logger, msg: str, *args) -> None:
    logger.debug(msg, *args, stacklevel=3, extra=_RATE_LIMIT_EXTRA)


def _info_limit(logger: logging.Logger, msg: str, *args) -> None:
    logger.info(msg, *args, stacklevel=3, extra=_RATE_LIMIT_EXTRA)


def _warning_limit(logger: logging.Logger, msg: str, *args) -> None:
    logger.warning(msg, *args, stacklevel=3, extra=_RATE_LIMIT_EXTRA)


def _error_limit(logger: logging.Logger, msg: str, *args) -> None:
    logger.error(msg, *args, stacklevel=3, extra=_RATE_LIMIT_EXTRA)


def _get_log_config():
    """Read log file configuration from environment variables."""
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


_initialized = False


def _initialize_backend() -> None:
    global _initialized
    if _initialized:
        return
    _initialized = True

    log_path, log_max_files, log_max_size = _get_log_config()
    ucmlogger.setup(log_path, log_max_files, log_max_size)
    atexit.register(ucmlogger.flush)

    root = logging.getLogger(_ROOT_NAME)
    root.setLevel(_resolve_level())
    root.propagate = False
    if not any(isinstance(h, UcmBridgeHandler) for h in root.handlers):
        root.addHandler(UcmBridgeHandler())


def _bind_convenience_methods(logger: logging.Logger) -> logging.Logger:
    if getattr(logger, "_ucm_methods_bound", False):
        return logger
    logger.debug_once = MethodType(_print_debug_once, logger)
    logger.info_once = MethodType(_print_info_once, logger)
    logger.warning_once = MethodType(_print_warning_once, logger)
    logger.debug_limit = MethodType(_debug_limit, logger)
    logger.info_limit = MethodType(_info_limit, logger)
    logger.warning_limit = MethodType(_warning_limit, logger)
    logger.error_limit = MethodType(_error_limit, logger)
    logger._ucm_methods_bound = True
    return logger


def init_logger(name: Optional[str] = None) -> logging.Logger:
    _initialize_backend()
    if not name or name == "UC":
        full_name = _ROOT_NAME
    elif name == _ROOT_NAME or name.startswith(_ROOT_NAME + "."):
        full_name = name
    else:
        full_name = f"{_ROOT_NAME}.{name}"
    return _bind_convenience_methods(logging.getLogger(full_name))


def get_vllm_capture_handler() -> Optional[UcmBridgeHandler]:
    _initialize_backend()
    return UcmBridgeHandler(file_only=True)


def current_formatter_type(lgr):
    return None


if __name__ == "__main__":
    logger = init_logger()
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
