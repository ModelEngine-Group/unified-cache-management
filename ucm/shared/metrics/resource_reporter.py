#
# MIT License
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
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
import hashlib
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

from ucm.logger import init_logger
from ucm.shared.metrics import ucmmetrics

logger = init_logger(__name__)

MAX_RECORD_BYTES = 1024 * 1024


def counter_deltas(
    current: dict[str, int | float], previous: dict[str, int | float] | None
) -> dict[str, int | float]:
    counter_zero = 0
    if previous is None:
        return {name: counter_zero for name in current}
    return {
        name: (
            value - previous.get(name, counter_zero)
            if value >= previous.get(name, counter_zero)
            else value
        )
        for name, value in current.items()
    }


class FileResourceMetricsReporter:
    """Poll a snapshot log from one elected process on each host."""

    def __init__(
        self,
        log_path: str,
        reporter_name: str,
        identity: str,
        interval_sec: float = 15.0,
        shared_memory_dir: str = "/dev/shm",
    ):
        self.log_path = Path(log_path)
        self.reporter_name = reporter_name
        self.reporter_key = reporter_name.lower()
        self.interval_sec = max(float(interval_sec), 1.0)
        shared_dir = Path(shared_memory_dir)
        if not shared_dir.is_dir():
            shared_dir = Path(tempfile.gettempdir())
        identity = hashlib.sha256(identity.encode()).hexdigest()[:24]
        self.lock_path = shared_dir / f"ucm_{self.reporter_key}_metrics_{identity}.lock"
        self.state_path = (
            shared_dir / f"ucm_{self.reporter_key}_metrics_{identity}.json"
        )
        self._stop_event = threading.Event()
        self._lock_file = None
        self._thread = threading.Thread(
            target=self._run,
            name=f"{self.reporter_key}-resource-reporter",
            daemon=True,
        )
        atexit.register(self.stop)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=min(self.interval_sec + 1.0, 5.0))
        self._release_leadership()

    def _run(self) -> None:
        if self._stop_event.is_set():
            return
        try:
            if not self._try_become_leader():
                return
        except Exception as error:
            logger.warning(
                f"Failed to elect {self.reporter_name} resource reporter: {error}"
            )
            ucmmetrics.update_stats(
                {f"{self.reporter_key}_resource_log_read_errors_total": 1.0}
            )
            return

        while not self._stop_event.is_set():
            try:
                self._collect_once()
            except Exception as error:
                logger.warning(
                    f"Failed to collect {self.reporter_name} resource metrics: {error}"
                )
                ucmmetrics.update_stats(
                    {f"{self.reporter_key}_resource_log_read_errors_total": 1.0}
                )
            self._stop_event.wait(self.interval_sec)

    def _try_become_leader(self) -> bool:
        if self._lock_file is not None:
            return True
        try:
            import fcntl
        except ImportError:
            logger.warning(
                f"{self.reporter_name} resource reporter requires fcntl for host election"
            )
            self._stop_event.set()
            return False

        lock_file = open(self.lock_path, "a+")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock_file.close()
            return False
        self._lock_file = lock_file
        logger.info(
            f"Became {self.reporter_name} resource metrics reporter for {self.log_path}"
        )
        return True

    def _release_leadership(self) -> None:
        if self._lock_file is None:
            return
        try:
            import fcntl

            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
        except (ImportError, OSError):
            pass
        self._lock_file.close()
        self._lock_file = None

    def _collect_once(self) -> None:
        raise NotImplementedError

    def _read_latest_complete_line(self) -> str:
        with open(self.log_path, "rb") as log_file:
            log_file.seek(0, os.SEEK_END)
            end = log_file.tell()
            if end == 0:
                raise ValueError(f"{self.reporter_name} resource log is empty")
            position = end
            data = b""
            while position > 0:
                chunk_size = min(position, 64 * 1024)
                position -= chunk_size
                log_file.seek(position)
                data = log_file.read(chunk_size) + data
                ends_with_newline = data.endswith((b"\n", b"\r"))
                complete = data.splitlines()
                if not ends_with_newline and complete:
                    complete.pop()
                if complete:
                    for candidate in reversed(complete):
                        if candidate.strip():
                            return candidate.decode("utf-8")
            raise ValueError(
                f"{self.reporter_name} resource log has no complete JSON record"
            )

    def _read_previous_state(self) -> dict[str, Any] | None:
        try:
            with open(self.state_path, "r", encoding="utf-8") as state_file:
                return json.load(state_file)
        except FileNotFoundError:
            return None
        except (OSError, ValueError, TypeError) as error:
            logger.warning(
                f"Ignoring invalid {self.reporter_name} reporter state: {error}"
            )
            return None

    def _write_previous_state(self, state: dict[str, Any]) -> None:
        temporary_path = self.state_path.with_suffix(f".{os.getpid()}.tmp")
        with open(temporary_path, "w", encoding="utf-8") as state_file:
            json.dump(state, state_file)
        os.replace(temporary_path, self.state_path)
