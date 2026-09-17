"""Unit tests for online runtime-mode control.

Covers two layers:

1. ``ucm/runtime_mode.py`` - mode parsing, control-file resolution, the
   throttled ``RuntimeModeController`` poller and its failure tolerance.
2. ``UCMDirectConnector`` gating - ``get_num_new_matched_tokens`` answering a
   full miss in DISABLED mode, recording shadow hits without planning any
   load/dump in LITE mode, and normal behavior in ENABLED mode.

The vLLM-heavy stubs are reused from ``test_ucm_connector_metrics`` so both
files share one canonical stub set regardless of import order.
"""

import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "test") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "test"))

import test_ucm_connector_metrics as stubs  # noqa: E402 (installs stubs)

ucm_connector_module = stubs.ucm_connector_module
fake_ucmmetrics = stubs.fake_ucmmetrics

import ucm.runtime_mode as runtime_mode_module  # noqa: E402
from ucm.default_metrics_config import DEFAULT_METRICS_CONFIG  # noqa: E402
from ucm.integration.vllm.ucm_connector import (  # noqa: E402
    RequestMeta,
    UCMConnector,
    UCMCPConnector,
    UCMDirectConnector,
    UCMLayerWiseConnector,
    UCMLiteConnector,
    UCMMockConnector,
)
from ucm.runtime_mode import (  # noqa: E402
    RuntimeModeController,
    UCMRuntimeMode,
    default_runtime_control_file,
    mode_gauge_value,
    parse_runtime_control_content,
    parse_runtime_mode,
    resolve_runtime_control_settings,
    write_runtime_control_file,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class RecordingLogger:
    """Drop-in replacement for the connector module logger."""

    def __init__(self):
        self.messages = []

    def _record(self, level, msg, *args, **kwargs):
        if args:
            try:
                msg = msg % args
            except TypeError:
                pass
        self.messages.append((level, str(msg)))

    def info(self, msg, *args, **kwargs):
        self._record("info", msg, *args, **kwargs)

    def info_once(self, msg, *args, **kwargs):
        self._record("info_once", msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self._record("debug", msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._record("warning", msg, *args, **kwargs)

    def warning_once(self, msg, *args, **kwargs):
        self._record("warning_once", msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._record("error", msg, *args, **kwargs)

    def error_limit(self, msg, *args, **kwargs):
        self._record("error_limit", msg, *args, **kwargs)

    def has_message(self, level, substring):
        return any(
            level == recorded_level and substring in message
            for recorded_level, message in self.messages
        )


class FakeRankConsistency:
    def __init__(self, prefix_hits=0):
        # lookup_on_prefix returns the index of the last hit hash; the
        # connector adds +1 to obtain the hit-hash count.
        self.prefix_hits = prefix_hits
        self.lookup_calls = []
        self.finished_dumps = []

    def lookup_on_prefix(self, store, block_ids):
        self.lookup_calls.append(list(block_ids))
        return self.prefix_hits

    def finish_dump(self, req_ids):
        self.finished_dumps.append(set(req_ids))

    def check_load(self, task):
        return True

    def wait_load(self, task):
        pass


class FakeStore:
    def __init__(self):
        self.prefetch_calls = []

    def prefetch(self, block_ids):
        self.prefetch_calls.append(list(block_ids))


BLOCK_SIZE = 16
NUM_TOKENS = 64
NUM_BLOCKS = NUM_TOKENS // BLOCK_SIZE  # 4 blocks


def make_request(request_id="req-1", num_tokens=NUM_TOKENS):
    return SimpleNamespace(
        request_id=request_id,
        all_token_ids=list(range(num_tokens)),
        num_tokens=num_tokens,
        num_prompt_tokens=num_tokens,
        max_tokens=16,
    )


def make_connector(controller, prefix_hits=NUM_BLOCKS - 2):
    """Build a scheduler-side UCMDirectConnector without running __init__."""
    connector = object.__new__(UCMDirectConnector)
    connector._runtime_mode_controller = controller
    connector._pending_async_load_dispatches = {}
    connector._async_load_req_ids = set()
    connector.requests_meta = {}
    connector.block_size = BLOCK_SIZE
    connector.cp_world_size = 1
    connector.hash_block_size = BLOCK_SIZE
    connector.request_block_hasher = lambda request: [
        bytes([index]) for index in range(1, NUM_BLOCKS + 1)
    ]
    connector.enable_record_traces = False
    connector.persist_token_threshold = 0
    connector.use_request_async_load = False
    connector.use_layerwise = True
    connector._other_rank_hashers = []
    connector.store = FakeStore()
    connector._rank_consistency = FakeRankConsistency(prefix_hits=prefix_hits)
    connector._prefetch_direct_hit_key_hotness = lambda hbm_ids, all_ids: None
    connector._get_full_hit_recompute_tokens = lambda: 2
    connector._async_dump_req_ids = set()
    connector._pending_dump_tasks = []
    connector._pending_load_tasks = {}
    connector._finished_async_load_req_ids = set()
    return connector


def make_scheduler_output(
    request_id="req-1", new_block_ids=None, num_scheduled_tokens=None
):
    if new_block_ids is None:
        new_block_ids = list(range(10, 10 + NUM_BLOCKS))
    if num_scheduled_tokens is None:
        num_scheduled_tokens = {request_id: BLOCK_SIZE}
    return SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(
                request_id=request_id, req_id=request_id, block_ids=[new_block_ids]
            )
        ],
        scheduled_cached_reqs=[],
        num_scheduled_tokens=num_scheduled_tokens,
        finished_req_ids=set(),
        preempted_req_ids=set(),
    )


@pytest.fixture(autouse=True)
def _reset_fake_metrics():
    fake_ucmmetrics.updated.clear()
    yield
    fake_ucmmetrics.updated.clear()


@pytest.fixture
def connector_logger(monkeypatch):
    recording = RecordingLogger()
    monkeypatch.setattr(ucm_connector_module, "logger", recording)
    return recording


def recorded_update_names():
    return {name for update in fake_ucmmetrics.updated for name in update}


# ---------------------------------------------------------------------------
# Parsing and settings resolution
# ---------------------------------------------------------------------------


class TestRuntimeModeParsing:
    def test_parse_runtime_mode_aliases(self):
        for text in ("enabled", "ENABLED", " enable ", "on", "true", "1", "full"):
            assert parse_runtime_mode(text) is UCMRuntimeMode.ENABLED
        for text in ("lite", "Lite", " LITE ", "shadow", "trace"):
            assert parse_runtime_mode(text) is UCMRuntimeMode.LITE
        for text in ("disabled", "DISABLE", " off ", "false", "0", "bypass"):
            assert parse_runtime_mode(text) is UCMRuntimeMode.DISABLED
        for value in (None, "", "bogus", 3, {"mode": "lite"}):
            assert parse_runtime_mode(value) is None
        assert parse_runtime_mode(UCMRuntimeMode.LITE) is UCMRuntimeMode.LITE

    def test_parse_runtime_control_content(self):
        assert (
            parse_runtime_control_content('{"mode": "disabled"}')
            is UCMRuntimeMode.DISABLED
        )
        assert (
            parse_runtime_control_content('{"mode": "enabled", "extra": 1}')
            is UCMRuntimeMode.ENABLED
        )
        assert parse_runtime_control_content('"lite"\n') is UCMRuntimeMode.LITE
        assert parse_runtime_control_content("lite\n") is UCMRuntimeMode.LITE
        assert parse_runtime_control_content("  disabled  ") is UCMRuntimeMode.DISABLED
        for content in ("", "   \n ", "garbage", '{"mode": "bogus"}', "{}", "42"):
            assert parse_runtime_control_content(content) is None

    def test_invalid_initial_mode_raises(self):
        with pytest.raises(ValueError, match="runtime_mode"):
            resolve_runtime_control_settings({"runtime_mode": "bogus"})

    def test_default_settings_enable_control_with_auto_file(self):
        settings = resolve_runtime_control_settings(None, "engine-0")
        assert settings.initial_mode is UCMRuntimeMode.ENABLED
        assert settings.control_file == default_runtime_control_file("engine-0")
        assert settings.poll_interval_s == 1.0

    def test_explicit_config_beats_env_var(self, monkeypatch, tmp_path):
        explicit = str(tmp_path / "explicit.json")
        monkeypatch.setenv("UCM_RUNTIME_CONTROL_FILE", str(tmp_path / "env.json"))
        settings = resolve_runtime_control_settings(
            {"runtime_control_file": explicit}, "engine-0"
        )
        assert settings.control_file == explicit

    def test_env_var_used_when_config_omits_file(self, monkeypatch, tmp_path):
        env_file = str(tmp_path / "env.json")
        monkeypatch.setenv("UCM_RUNTIME_CONTROL_FILE", env_file)
        settings = resolve_runtime_control_settings({}, "engine-0")
        assert settings.control_file == env_file

    def test_control_can_be_disabled(self, monkeypatch, tmp_path):
        monkeypatch.setenv("UCM_RUNTIME_CONTROL_FILE", str(tmp_path / "env.json"))
        for launch_config in (
            {"runtime_control_enabled": False, "runtime_control_file": "/x/y.json"},
            {"runtime_control_file": ""},
            {"runtime_control_enabled": False},
        ):
            settings = resolve_runtime_control_settings(launch_config, "engine-0")
            assert settings.control_file is None

    def test_initial_mode_and_interval_from_config(self, tmp_path):
        settings = resolve_runtime_control_settings(
            {
                "runtime_mode": "lite",
                "runtime_control_file": str(tmp_path / "ctl.json"),
                "runtime_control_poll_interval_s": 0.5,
            },
            "engine-0",
        )
        assert settings.initial_mode is UCMRuntimeMode.LITE
        assert settings.poll_interval_s == 0.5

    def test_default_runtime_control_file_sanitizes_engine_id(self):
        path = default_runtime_control_file("engine 0/a")
        assert "engine_0_a" in path
        assert path.endswith(".json")
        assert "ucm_runtime_control_" in path


# ---------------------------------------------------------------------------
# RuntimeModeController
# ---------------------------------------------------------------------------


class TestRuntimeModeController:
    def test_missing_file_keeps_initial_mode(self, tmp_path):
        controller = RuntimeModeController(
            initial_mode=UCMRuntimeMode.ENABLED,
            control_file=str(tmp_path / "missing.json"),
        )
        assert controller.refresh(now=0.0) is UCMRuntimeMode.ENABLED

    def test_mode_switch_observed_after_poll_interval(self, tmp_path):
        control_file = tmp_path / "ctl.json"
        controller = RuntimeModeController(
            initial_mode=UCMRuntimeMode.ENABLED,
            control_file=str(control_file),
            poll_interval_s=1.0,
        )

        assert controller.refresh(now=0.0) is UCMRuntimeMode.ENABLED
        write_runtime_control_file(str(control_file), "disabled")
        # Within the poll interval: throttled, no change yet.
        assert controller.refresh(now=0.5) is UCMRuntimeMode.ENABLED
        # After the poll interval: the switch is observed.
        assert controller.refresh(now=1.5) is UCMRuntimeMode.DISABLED

        assert {"connector_runtime_mode_transitions_total": 1.0} in (
            fake_ucmmetrics.updated
        )
        assert {
            "connector_runtime_mode": mode_gauge_value(UCMRuntimeMode.DISABLED)
        } in (fake_ucmmetrics.updated)

    def test_unchanged_file_is_not_reread(self, tmp_path, monkeypatch):
        control_file = tmp_path / "ctl.json"
        write_runtime_control_file(str(control_file), "lite")
        controller = RuntimeModeController(
            control_file=str(control_file), poll_interval_s=1.0
        )

        reads = []
        original_read = controller._read_mode

        def counting_read(path):
            reads.append(path)
            return original_read(path)

        monkeypatch.setattr(controller, "_read_mode", counting_read)

        assert controller.refresh(now=0.0, force=True) is UCMRuntimeMode.LITE
        assert controller.refresh(now=1.0, force=True) is UCMRuntimeMode.LITE
        assert controller.refresh(now=2.0, force=True) is UCMRuntimeMode.LITE
        assert len(reads) == 1  # (mtime, size) unchanged: no re-read

    def test_invalid_content_keeps_mode_and_retries(self, tmp_path):
        control_file = tmp_path / "ctl.json"
        control_file.write_text("not-json {{{", encoding="utf-8")
        controller = RuntimeModeController(
            initial_mode=UCMRuntimeMode.ENABLED,
            control_file=str(control_file),
        )

        assert controller.refresh(now=0.0, force=True) is UCMRuntimeMode.ENABLED
        assert "connector_runtime_mode_control_read_errors_total" in (
            recorded_update_names()
        )

        # The failed signature is not cached: fixing the file applies at once.
        write_runtime_control_file(str(control_file), "disabled")
        assert controller.refresh(now=1.0, force=True) is UCMRuntimeMode.DISABLED

    def test_file_appearance_and_disappearance(self, tmp_path):
        control_file = tmp_path / "ctl.json"
        controller = RuntimeModeController(
            initial_mode=UCMRuntimeMode.ENABLED,
            control_file=str(control_file),
        )

        assert controller.refresh(now=0.0, force=True) is UCMRuntimeMode.ENABLED
        write_runtime_control_file(str(control_file), "disabled")
        assert controller.refresh(now=1.0, force=True) is UCMRuntimeMode.DISABLED

        control_file.unlink()
        # Disappearing keeps the last observed mode.
        assert controller.refresh(now=2.0, force=True) is UCMRuntimeMode.DISABLED

        write_runtime_control_file(str(control_file), "lite")
        assert controller.refresh(now=3.0, force=True) is UCMRuntimeMode.LITE

    def test_pinned_mode_never_touches_filesystem(self, monkeypatch, tmp_path):
        controller = RuntimeModeController(
            initial_mode=UCMRuntimeMode.LITE, control_file=None
        )
        monkeypatch.setattr(
            runtime_mode_module.os, "stat", lambda *args, **kwargs: pytest.fail("stat")
        )
        assert controller.refresh() is UCMRuntimeMode.LITE
        assert controller.refresh(now=100.0) is UCMRuntimeMode.LITE

    def test_bare_string_content_is_accepted(self, tmp_path):
        control_file = tmp_path / "ctl.json"
        control_file.write_text("disabled\n", encoding="utf-8")
        controller = RuntimeModeController(control_file=str(control_file))
        assert controller.refresh(now=0.0, force=True) is UCMRuntimeMode.DISABLED

    def test_write_runtime_control_file_is_atomic_json(self, tmp_path):
        control_file = tmp_path / "nested" / "ctl.json"
        mode = write_runtime_control_file(str(control_file), UCMRuntimeMode.DISABLED)
        assert mode is UCMRuntimeMode.DISABLED
        payload = json.loads(control_file.read_text(encoding="utf-8"))
        assert payload == {"mode": "disabled"}
        assert list(tmp_path.glob("**/*.tmp.*")) == []

        with pytest.raises(ValueError):
            write_runtime_control_file(str(control_file), "bogus")

    def test_concurrent_refresh_is_safe(self, tmp_path):
        control_file = tmp_path / "ctl.json"
        write_runtime_control_file(str(control_file), "enabled")
        controller = RuntimeModeController(
            control_file=str(control_file), poll_interval_s=0.0
        )
        errors = []

        def worker():
            try:
                for _ in range(50):
                    controller.refresh(force=True)
            except Exception as exc:  # pragma: no cover - failure reporting
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert not errors
        assert controller.mode in set(UCMRuntimeMode)


# ---------------------------------------------------------------------------
# UCMDirectConnector gating
# ---------------------------------------------------------------------------


class TestConnectorRuntimeModeGating:
    def test_enabled_returns_external_hits(self, connector_logger):
        connector = make_connector(RuntimeModeController(control_file=None))
        request = make_request()

        assert connector.get_num_new_matched_tokens(request, 0) == (
            (NUM_BLOCKS - 1) * BLOCK_SIZE,
            False,
        )
        assert len(connector._rank_consistency.lookup_calls) == 1
        req_meta = connector.requests_meta[request.request_id]
        assert req_meta.total_hit_block_num == NUM_BLOCKS - 1

    def test_disabled_answers_miss_without_lookup(self, tmp_path, connector_logger):
        control_file = tmp_path / "ctl.json"
        write_runtime_control_file(str(control_file), "disabled")
        connector = make_connector(
            RuntimeModeController(control_file=str(control_file), poll_interval_s=0.0)
        )
        # Stale scheduler-side plans must be dropped together with the answer.
        connector.requests_meta["req-1"] = RequestMeta(ucm_block_ids=[b"\x01"])
        connector._pending_async_load_dispatches["req-1"] = object()
        connector._async_load_req_ids.add("req-1")

        request = make_request()
        assert connector.get_num_new_matched_tokens(request, 0) == (0, False)

        assert connector._rank_consistency.lookup_calls == []
        assert "req-1" not in connector.requests_meta
        assert "req-1" not in connector._pending_async_load_dispatches
        assert "req-1" not in connector._async_load_req_ids
        assert {"connector_runtime_mode_bypassed_requests_total": 1.0} in (
            fake_ucmmetrics.updated
        )

    def test_lite_records_shadow_hit_and_plans_nothing(
        self, tmp_path, connector_logger
    ):
        control_file = tmp_path / "ctl.json"
        write_runtime_control_file(str(control_file), "lite")
        connector = make_connector(
            RuntimeModeController(control_file=str(control_file), poll_interval_s=0.0)
        )
        request = make_request()

        assert connector.get_num_new_matched_tokens(request, 0) == (0, False)

        # The metadata lookup still ran against the store...
        assert len(connector._rank_consistency.lookup_calls) == 1
        # ...and the shadow hit was recorded...
        lite_updates = [
            update
            for update in fake_ucmmetrics.updated
            if "connector_runtime_mode_lite_requests_total" in update
        ]
        assert lite_updates == [
            {
                "connector_runtime_mode_lite_requests_total": 1.0,
                "lite_shadow_lookup_blocks_total": float(NUM_BLOCKS),
                "lite_shadow_external_hit_tokens_total": float(
                    (NUM_BLOCKS - 1) * BLOCK_SIZE
                ),
            }
        ]
        # ...but no load/dump is planned for the request.
        req_meta = connector.requests_meta[request.request_id]
        assert req_meta.total_hit_block_num == req_meta.hbm_hit_block_num
        assert req_meta.token_processed == req_meta.num_token_ids

        metadata = connector.build_connector_meta(make_scheduler_output())
        dispatch = metadata.request_meta[request.request_id]
        assert dispatch.load_block_ids == ([], [])
        assert dispatch.dump_block_ids == ([], [])
        assert connector._async_dump_req_ids == set()

    def test_mode_transition_roundtrip(self, tmp_path, connector_logger):
        control_file = tmp_path / "ctl.json"
        connector = make_connector(
            RuntimeModeController(control_file=str(control_file), poll_interval_s=0.0)
        )
        request = make_request()
        expected_hits = ((NUM_BLOCKS - 1) * BLOCK_SIZE, False)

        assert connector.get_num_new_matched_tokens(request, 0) == expected_hits

        write_runtime_control_file(str(control_file), "disabled")
        assert connector.get_num_new_matched_tokens(request, 0) == (0, False)

        write_runtime_control_file(str(control_file), "lite")
        assert connector.get_num_new_matched_tokens(request, 0) == (0, False)

        write_runtime_control_file(str(control_file), "enabled")
        assert connector.get_num_new_matched_tokens(request, 0) == expected_hits

    def test_get_finished_refreshes_mode_in_all_modes(self, tmp_path, connector_logger):
        control_file = tmp_path / "ctl.json"
        connector = make_connector(
            RuntimeModeController(control_file=str(control_file), poll_interval_s=0.0)
        )

        for mode in ("enabled", "disabled", "lite"):
            write_runtime_control_file(str(control_file), mode)
            assert connector.get_finished(set()) == (None, None)

    def test_request_finished_semantics_unchanged(self, connector_logger):
        connector = make_connector(RuntimeModeController(control_file=None))
        request = make_request()

        connector._async_dump_req_ids.add(request.request_id)
        assert connector.request_finished(request, [1, 2]) == (True, None)
        assert connector.request_finished(request, [1, 2]) == (False, None)

    def test_no_controller_falls_back_to_enabled(self, connector_logger):
        connector = make_connector(RuntimeModeController(control_file=None))
        del connector._runtime_mode_controller
        request = make_request()

        assert connector.get_num_new_matched_tokens(request, 0) == (
            (NUM_BLOCKS - 1) * BLOCK_SIZE,
            False,
        )


# ---------------------------------------------------------------------------
# Facade support declaration
# ---------------------------------------------------------------------------


class TestFacadeRuntimeModeSupport:
    def _facade(self, inner):
        facade = object.__new__(UCMConnector)
        facade.connector = inner
        facade.launch_config = {}
        facade.engine_id = "engine-0"
        return facade

    def test_supported_inner_connector_does_not_warn(
        self, connector_logger, monkeypatch
    ):
        monkeypatch.delenv("UCM_RUNTIME_CONTROL_FILE", raising=False)
        inner = SimpleNamespace(supports_runtime_mode_control=True)
        self._facade(inner)._log_runtime_mode_control_support()
        assert not connector_logger.has_message("warning", "runtime-mode control")

    def test_unsupported_inner_connector_warns(self, connector_logger, monkeypatch):
        monkeypatch.delenv("UCM_RUNTIME_CONTROL_FILE", raising=False)
        inner = SimpleNamespace()
        self._facade(inner)._log_runtime_mode_control_support()
        assert connector_logger.has_message("warning", "Online runtime-mode control")
        assert connector_logger.has_message("warning", "Namespace")

    def test_no_warning_when_control_disabled(self, connector_logger):
        facade = self._facade(SimpleNamespace())
        facade.launch_config = {"runtime_control_enabled": False}
        facade._log_runtime_mode_control_support()
        assert not connector_logger.has_message("warning", "Online runtime-mode")

    def test_invalid_runtime_mode_warns(self, connector_logger):
        facade = self._facade(SimpleNamespace())
        facade.launch_config = {"runtime_mode": "bogus"}
        facade._log_runtime_mode_control_support()
        assert connector_logger.has_message("warning", "Invalid runtime-mode")

    def test_connector_family_flags(self):
        assert UCMDirectConnector.supports_runtime_mode_control is True
        assert UCMLayerWiseConnector.supports_runtime_mode_control is True
        assert UCMCPConnector.supports_runtime_mode_control is True
        assert UCMMockConnector.supports_runtime_mode_control is True
        assert getattr(UCMLiteConnector, "supports_runtime_mode_control", False) is (
            False
        )


# ---------------------------------------------------------------------------
# Metrics registration
# ---------------------------------------------------------------------------


class TestRuntimeModeMetricsRegistration:
    def test_runtime_mode_metrics_are_in_default_config(self):
        counters = {metric["name"] for metric in DEFAULT_METRICS_CONFIG["counter"]}
        gauges = {metric["name"] for metric in DEFAULT_METRICS_CONFIG["gauge"]}

        assert {
            "connector_runtime_mode_bypassed_requests_total",
            "connector_runtime_mode_lite_requests_total",
            "connector_runtime_mode_transitions_total",
            "connector_runtime_mode_control_read_errors_total",
            "lite_shadow_lookup_blocks_total",
            "lite_shadow_external_hit_tokens_total",
        } <= counters
        assert "connector_runtime_mode" in gauges
