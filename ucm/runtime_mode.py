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

"""Online runtime-mode control for UCM.

A UCM-enabled engine is deployed once with ``kv_transfer_config``; this module
allows operators to toggle the effective UCM behavior *at runtime* without
restarting the service:

- ``ENABLED``  - full behavior: external lookup + KV load/dump against the store.
- ``DISABLED`` - bypass: external lookups return a miss immediately and no
  dump/load is planned; the engine keeps its original compute path.
- ``LITE``     - shadow: hashing and metadata lookups still run (so hit rates
  against the real store are recorded), but no KV data is loaded or dumped;
  the engine still recomputes everything.

The mode is read from a small control file that every connector process polls
(throttled by mtime). The file format is either a bare mode string::

    echo disabled > /tmp/ucm_runtime_control_<engine_id>.json

or a JSON object with a ``mode`` key::

    {"mode": "lite"}

Control-file content is validated before it is applied; unparseable or missing
files keep the current mode, so a half-written file can never corrupt engine
behavior. Mode changes only affect *new* scheduling decisions: loads and dumps
that were already planned by the scheduler are still executed by the worker,
which keeps in-flight requests correct.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from ucm.logger import init_logger

logger = init_logger(__name__)

__all__ = [
    "UCMRuntimeMode",
    "RuntimeControlSettings",
    "RuntimeModeController",
    "mode_gauge_value",
    "parse_runtime_mode",
    "parse_runtime_control_content",
    "resolve_runtime_control_settings",
    "default_runtime_control_file",
    "write_runtime_control_file",
]


class UCMRuntimeMode(str, Enum):
    """Effective UCM behavior at runtime.

    The enum value is the canonical string used in configuration files and in
    the control file; ``mode_gauge_value`` maps it onto the numeric gauge.
    """

    ENABLED = "enabled"
    LITE = "lite"
    DISABLED = "disabled"


_MODE_GAUGE_VALUES: dict[UCMRuntimeMode, float] = {
    UCMRuntimeMode.ENABLED: 0.0,
    UCMRuntimeMode.LITE: 1.0,
    UCMRuntimeMode.DISABLED: 2.0,
}


def mode_gauge_value(mode: UCMRuntimeMode) -> float:
    return _MODE_GAUGE_VALUES[mode]


_MODE_ALIASES: dict[str, UCMRuntimeMode] = {
    "enabled": UCMRuntimeMode.ENABLED,
    "enable": UCMRuntimeMode.ENABLED,
    "on": UCMRuntimeMode.ENABLED,
    "true": UCMRuntimeMode.ENABLED,
    "1": UCMRuntimeMode.ENABLED,
    "full": UCMRuntimeMode.ENABLED,
    "lite": UCMRuntimeMode.LITE,
    "shadow": UCMRuntimeMode.LITE,
    "trace": UCMRuntimeMode.LITE,
    "disabled": UCMRuntimeMode.DISABLED,
    "disable": UCMRuntimeMode.DISABLED,
    "off": UCMRuntimeMode.DISABLED,
    "false": UCMRuntimeMode.DISABLED,
    "0": UCMRuntimeMode.DISABLED,
    "bypass": UCMRuntimeMode.DISABLED,
}

_READ_ERROR_LOG_INTERVAL_S = 60.0
_UNSET = object()


def parse_runtime_mode(value: Any) -> Optional[UCMRuntimeMode]:
    """Return the mode for ``value`` or ``None`` when it is not recognized."""
    if isinstance(value, UCMRuntimeMode):
        return value
    if not isinstance(value, str):
        return None
    return _MODE_ALIASES.get(value.strip().lower())


def parse_runtime_control_content(content: Any) -> Optional[UCMRuntimeMode]:
    """Parse control-file content (bare string or JSON ``{"mode": ...}``)."""
    if not isinstance(content, str):
        return None
    text = content.strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        payload = None
    if isinstance(payload, dict):
        return parse_runtime_mode(payload.get("mode"))
    if isinstance(payload, str):
        return parse_runtime_mode(payload)
    return parse_runtime_mode(text)


def default_runtime_control_file(engine_id: str = "") -> str:
    """Default per-engine control-file path in the system temp directory."""
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", str(engine_id or "").strip())
    safe = safe or "ucm"
    return os.path.join(tempfile.gettempdir(), f"ucm_runtime_control_{safe}.json")


@dataclass(frozen=True)
class RuntimeControlSettings:
    """Resolved runtime-mode control configuration."""

    initial_mode: UCMRuntimeMode = UCMRuntimeMode.ENABLED
    # ``None`` means polling is disabled and the mode stays ``initial_mode``.
    control_file: Optional[str] = None
    poll_interval_s: float = 1.0


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def resolve_runtime_control_settings(
    launch_config: Optional[dict[str, Any]],
    engine_id: str = "",
) -> RuntimeControlSettings:
    """Resolve runtime-mode settings from the UCM launch config.

    Precedence for the control file:
    1. ``runtime_control_file`` in the launch config (empty string disables
       polling);
    2. ``UCM_RUNTIME_CONTROL_FILE`` environment variable;
    3. default per-engine path in the system temp directory.

    ``runtime_control_enabled: false`` (or an empty ``runtime_control_file``)
    pins the mode to the ``runtime_mode`` value given at startup.
    """
    launch_config = launch_config or {}
    raw_mode = launch_config.get("runtime_mode")
    initial_mode = (
        parse_runtime_mode(raw_mode) if raw_mode is not None else UCMRuntimeMode.ENABLED
    )
    if initial_mode is None:
        raise ValueError(
            "Invalid 'runtime_mode' value "
            f"{launch_config.get('runtime_mode')!r}; expected one of "
            f"{[mode.value for mode in UCMRuntimeMode]}."
        )

    if not _as_bool(launch_config.get("runtime_control_enabled"), True):
        return RuntimeControlSettings(initial_mode=initial_mode, control_file=None)

    control_file = launch_config.get("runtime_control_file")
    if control_file is None:
        control_file = os.getenv("UCM_RUNTIME_CONTROL_FILE") or None
    if control_file is None:
        control_file = default_runtime_control_file(engine_id)
    control_file = str(control_file).strip()
    if not control_file:
        return RuntimeControlSettings(initial_mode=initial_mode, control_file=None)

    try:
        poll_interval_s = float(
            launch_config.get("runtime_control_poll_interval_s", 1.0)
        )
    except (TypeError, ValueError):
        logger.warning(
            "Invalid runtime_control_poll_interval_s %r; falling back to 1.0s.",
            launch_config.get("runtime_control_poll_interval_s"),
        )
        poll_interval_s = 1.0
    if not math.isfinite(poll_interval_s) or poll_interval_s < 0:
        poll_interval_s = 1.0

    return RuntimeControlSettings(
        initial_mode=initial_mode,
        control_file=control_file,
        poll_interval_s=poll_interval_s,
    )


def _update_stats(stats: dict[str, float]) -> None:
    """Record metrics lazily; the metrics extension is an optional dependency."""
    try:
        from ucm.shared.metrics import ucmmetrics
    except ImportError:
        return
    ucmmetrics.update_stats(stats)


class RuntimeModeController:
    """Thread-safe, throttled poller for the online runtime mode.

    One controller instance is owned by each connector (scheduler and worker
    side); every instance reads the same shared control file, so all processes
    observe mode changes within one poll interval. The controller never
    raises: any read or parse failure keeps the last known good mode.
    """

    def __init__(
        self,
        initial_mode: UCMRuntimeMode | str = UCMRuntimeMode.ENABLED,
        control_file: Optional[str] = None,
        poll_interval_s: float = 1.0,
    ):
        if not isinstance(initial_mode, UCMRuntimeMode):
            parsed = parse_runtime_mode(initial_mode)
            if parsed is None:
                raise ValueError(f"Invalid initial runtime mode: {initial_mode!r}")
            initial_mode = parsed
        self._mode: UCMRuntimeMode = initial_mode
        self._control_file = control_file
        self._poll_interval_s = max(0.0, float(poll_interval_s))
        self._lock = threading.Lock()
        self._next_poll_after = -math.inf
        self._last_signature: Any = _UNSET
        self._file_existed = False
        self._last_error_log_after = -math.inf
        self._gauge_reported = False

    @classmethod
    def from_launch_config(
        cls,
        launch_config: Optional[dict[str, Any]],
        engine_id: str = "",
    ) -> "RuntimeModeController":
        settings = resolve_runtime_control_settings(launch_config, engine_id)
        return cls(
            initial_mode=settings.initial_mode,
            control_file=settings.control_file,
            poll_interval_s=settings.poll_interval_s,
        )

    @property
    def mode(self) -> UCMRuntimeMode:
        return self._mode

    @property
    def control_file(self) -> Optional[str]:
        return self._control_file

    @property
    def poll_interval_s(self) -> float:
        return self._poll_interval_s

    def refresh(
        self, now: Optional[float] = None, force: bool = False
    ) -> UCMRuntimeMode:
        """Return the current mode, polling the control file when due.

        ``now`` (monotonic seconds) and ``force`` exist for tooling and tests;
        normal callers use the defaults.
        """
        if not self._gauge_reported:
            self._gauge_reported = True
            self._report_mode_gauge()

        if self._control_file is None:
            return self._mode

        now = time.monotonic() if now is None else now
        if not force and now < self._next_poll_after:
            return self._mode

        with self._lock:
            if not force and now < self._next_poll_after:
                return self._mode
            self._next_poll_after = now + self._poll_interval_s
            self._poll_control_file()
            return self._mode

    # -- internals ---------------------------------------------------------

    def _poll_control_file(self) -> None:
        path = self._control_file
        try:
            stat = os.stat(path)
        except FileNotFoundError:
            if self._file_existed:
                self._file_existed = False
                self._last_signature = None
                logger.warning(
                    "UCM runtime control file %s disappeared; keeping mode %s.",
                    path,
                    self._mode.name,
                )
            return
        except OSError as exc:
            self._record_read_error(f"stat failed: {exc}")
            return

        self._file_existed = True
        signature = (stat.st_mtime_ns, stat.st_size)
        if signature == self._last_signature:
            return

        mode = self._read_mode(path)
        if mode is None:
            # Keep the last signature so the file is retried on the next tick;
            # a partially written file only delays the mode switch.
            self._record_read_error("unreadable or unparseable content")
            return

        self._last_signature = signature
        self._apply_mode(mode)

    def _read_mode(self, path: str) -> Optional[UCMRuntimeMode]:
        for _attempt in range(2):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
            except OSError:
                return None
            mode = parse_runtime_control_content(content)
            if mode is not None:
                return mode
        return None

    def _apply_mode(self, new_mode: UCMRuntimeMode) -> None:
        if new_mode is self._mode:
            return
        old_mode = self._mode
        self._mode = new_mode
        self._report_mode_gauge()
        logger.info(
            "UCM runtime mode switched %s -> %s (control file: %s).",
            old_mode.name,
            new_mode.name,
            self._control_file,
        )
        _update_stats({"connector_runtime_mode_transitions_total": 1.0})

    def _report_mode_gauge(self) -> None:
        _update_stats({"connector_runtime_mode": mode_gauge_value(self._mode)})

    def _record_read_error(self, detail: str) -> None:
        _update_stats({"connector_runtime_mode_control_read_errors_total": 1.0})
        now = time.monotonic()
        if now - self._last_error_log_after >= _READ_ERROR_LOG_INTERVAL_S:
            self._last_error_log_after = now
            logger.warning(
                "Failed to read UCM runtime control file %s (%s); keeping mode %s.",
                self._control_file,
                detail,
                self._mode.name,
            )


def write_runtime_control_file(path: str, mode: UCMRuntimeMode | str) -> UCMRuntimeMode:
    """Atomically write a control file; used by tooling and tests."""
    parsed = parse_runtime_mode(mode)
    if parsed is None:
        raise ValueError(f"Invalid runtime mode: {mode!r}")
    payload = json.dumps({"mode": parsed.value})
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(payload)
    os.replace(tmp_path, path)
    return parsed
