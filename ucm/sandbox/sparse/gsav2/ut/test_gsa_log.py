# -*- coding: utf-8 -*-
"""ucm/sandbox/sparse/gsav2/gsa_log.py 单元测试。

覆盖目标：Logger 与 GSALogger 的全部方法、级别过滤、时间格式、
四种日志类型（performance/accuracy/normal/all）的路由与 enabled 属性。
"""
import os

import pytest

import ucm.sandbox.sparse.gsav2.gsa_log as gsa_log


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
class TestLogger:
    def _logger(self, tmp_path, **kw):
        log_file = os.path.join(str(tmp_path), "sub", "app.log")
        return gsa_log.Logger(log_file=log_file, **kw), log_file

    def test_init_creates_log_dir(self, tmp_path):
        lg, log_file = self._logger(tmp_path)
        assert os.path.isdir(os.path.dirname(log_file))
        assert lg.enabled is True
        assert lg.level == gsa_log.Logger.LEVEL_INFO
        assert lg.show_time is True

    def test_init_show_time_false(self, tmp_path):
        lg, _ = self._logger(tmp_path, show_time=False)
        assert lg.show_time is False

    def test_setters(self, tmp_path):
        lg, _ = self._logger(tmp_path)
        lg.set_enabled(False)
        lg.set_level(gsa_log.Logger.LEVEL_ERROR)
        lg.set_show_time(False)
        assert lg.enabled is False
        assert lg.level == gsa_log.Logger.LEVEL_ERROR
        assert lg.show_time is False

    def test_log_writes_line(self, tmp_path):
        lg, log_file = self._logger(tmp_path, show_time=False, enabled=True)
        lg.log(gsa_log.Logger.LEVEL_INFO, "hello")
        content = open(log_file, encoding="utf-8").read()
        assert "[INFO] hello" in content

    def test_log_disabled_no_write(self, tmp_path):
        lg, log_file = self._logger(tmp_path, enabled=False)
        lg.log(gsa_log.Logger.LEVEL_ERROR, "nope")
        assert not os.path.exists(log_file)

    def test_log_level_filtered(self, tmp_path):
        lg, log_file = self._logger(tmp_path, level=gsa_log.Logger.LEVEL_WARNING)
        lg.log(gsa_log.Logger.LEVEL_INFO, "filtered")
        assert not os.path.exists(log_file)
        lg.log(gsa_log.Logger.LEVEL_WARNING, "kept")
        assert "[WARNING] kept" in open(log_file, encoding="utf-8").read()

    def test_log_with_time(self, tmp_path):
        lg, log_file = self._logger(tmp_path, show_time=True)
        lg.log(gsa_log.Logger.LEVEL_ERROR, "timed")
        line = open(log_file, encoding="utf-8").read()
        assert "[ERROR] timed" in line
        assert "[20" in line  # 时间戳前缀

    def test_log_unknown_level_name(self, tmp_path):
        lg, log_file = self._logger(
            tmp_path, show_time=False, level=gsa_log.Logger.LEVEL_DEBUG
        )
        lg.log(15, "weird level")  # 非 DEBUG/INFO/WARNING/ERROR，且高于最低级别
        assert "[UNKNOWN] weird level" in open(log_file, encoding="utf-8").read()

    def test_is_debug_enabled_true(self, tmp_path):
        lg, _ = self._logger(tmp_path, level=gsa_log.Logger.LEVEL_DEBUG)
        assert lg.is_debug_enabled is True

    def test_is_debug_enabled_level_too_high(self, tmp_path):
        lg, _ = self._logger(tmp_path, level=gsa_log.Logger.LEVEL_INFO)
        assert lg.is_debug_enabled is False

    def test_is_debug_enabled_disabled(self, tmp_path):
        lg, _ = self._logger(tmp_path, level=gsa_log.Logger.LEVEL_DEBUG, enabled=False)
        assert lg.is_debug_enabled is False

    def test_is_info_enabled_true(self, tmp_path):
        lg, _ = self._logger(tmp_path, level=gsa_log.Logger.LEVEL_INFO)
        assert lg.is_info_enabled is True

    def test_is_info_enabled_false(self, tmp_path):
        lg, _ = self._logger(tmp_path, level=gsa_log.Logger.LEVEL_WARNING)
        assert lg.is_info_enabled is False

    def test_debug_writes_when_enabled(self, tmp_path):
        lg, log_file = self._logger(tmp_path, level=gsa_log.Logger.LEVEL_DEBUG)
        lg.debug("dbg msg")
        assert "[DEBUG] dbg msg" in open(log_file, encoding="utf-8").read()

    def test_debug_skipped_when_filtered(self, tmp_path):
        lg, log_file = self._logger(tmp_path, level=gsa_log.Logger.LEVEL_INFO)
        lg.debug("dbg msg")
        assert not os.path.exists(log_file)

    def test_info_default_message(self, tmp_path):
        lg, log_file = self._logger(tmp_path, show_time=False)
        lg.info()
        assert "[INFO] " in open(log_file, encoding="utf-8").read()

    def test_warning_filtered_by_level(self, tmp_path):
        lg, log_file = self._logger(tmp_path, level=gsa_log.Logger.LEVEL_ERROR)
        lg.warning("warn msg")
        assert not os.path.exists(log_file)

    def test_warning_writes(self, tmp_path):
        lg, log_file = self._logger(tmp_path, show_time=False)
        lg.warning("warn msg")
        assert "[WARNING] warn msg" in open(log_file, encoding="utf-8").read()

    def test_error_writes(self, tmp_path):
        lg, log_file = self._logger(tmp_path, show_time=False)
        lg.error("err msg")
        assert "[ERROR] err msg" in open(log_file, encoding="utf-8").read()

    def test_warning_disabled(self, tmp_path):
        lg, log_file = self._logger(tmp_path, enabled=False)
        lg.warning("warn")
        lg.error("err")
        assert not os.path.exists(log_file)


# ---------------------------------------------------------------------------
# GSALogger
# ---------------------------------------------------------------------------
class TestGSALogger:
    def _read(self, path):
        with open(path, encoding="utf-8") as f:
            return f.read()

    def test_init_all_creates_three_loggers(self, tmp_path):
        gl = gsa_log.GSALogger(
            log_dir=str(tmp_path),
            dp_rank=0,
            tp_rank=1,
            log_type=gsa_log.GSALogger.LOG_TYPE_ALL,
        )
        assert set(gl.loggers) == {"performance", "accuracy", "normal"}
        for t in ("performance", "accuracy", "normal"):
            assert isinstance(gl.loggers[t], gsa_log.Logger)
            expected = os.path.join(str(tmp_path), f"log_for_0_tp_1_{t}.log")
            assert gl.loggers[t].log_file == expected

    @pytest.mark.parametrize("log_type", ["performance", "accuracy", "normal"])
    def test_init_single_type(self, tmp_path, log_type):
        gl = gsa_log.GSALogger(log_dir=str(tmp_path), log_type=log_type)
        assert list(gl.loggers) == [log_type]

    def test_create_logger(self, tmp_path):
        gl = gsa_log.GSALogger(log_dir=str(tmp_path))
        lg = gl._create_logger(os.path.join(str(tmp_path), "x.log"))
        assert lg.enabled is True

    def test_get_level_name(self, tmp_path):
        gl = gsa_log.GSALogger(log_dir=str(tmp_path))
        assert gl._get_level_name(10) == "DEBUG"
        assert gl._get_level_name(999) == "UNKNOWN"

    def test_format_message_with_time_and_level(self, tmp_path):
        gl = gsa_log.GSALogger(log_dir=str(tmp_path), show_time=True)
        msg = gl._format_message("m", gsa_log.GSALogger.LEVEL_INFO)
        assert "[INFO] m" in msg and "[20" in msg

    def test_format_message_no_time_no_level(self, tmp_path):
        gl = gsa_log.GSALogger(log_dir=str(tmp_path), show_time=False)
        assert gl._format_message("m") == "[] m"

    def test_write_log_level_filtered(self, tmp_path):
        gl = gsa_log.GSALogger(
            log_dir=str(tmp_path), level=gsa_log.GSALogger.LEVEL_ERROR
        )
        gl.performance_error("kept")
        gl.performance_warning("dropped")
        perf = self._read(os.path.join(str(tmp_path), "log_for_0_tp_0_performance.log"))
        assert "kept" in perf and "dropped" not in perf

    def test_write_log_missing_logger_is_noop(self, tmp_path):
        # normal 模式下 performance 日志文件不存在 → 静默跳过
        gl = gsa_log.GSALogger(log_dir=str(tmp_path), log_type="normal")
        gl.performance_info("no target")
        assert not os.path.exists(
            os.path.join(str(tmp_path), "log_for_0_tp_0_performance.log")
        )

    def test_performance_is_info(self, tmp_path):
        gl_all = gsa_log.GSALogger(log_dir=str(tmp_path))
        assert gl_all.performance_is_info() is True
        assert gl_all.performance_is_debug() is False  # level=INFO > DEBUG
        gl_one = gsa_log.GSALogger(log_dir=str(tmp_path), log_type="normal")
        assert gl_one.performance_is_info() is False
        assert gl_one.performance_is_debug() is False

    def test_accuracy_is_info_debug(self, tmp_path):
        gl_all = gsa_log.GSALogger(log_dir=str(tmp_path))
        assert gl_all.accuracy_is_info() is True
        assert gl_all.accuracy_is_debug() is False
        gl_one = gsa_log.GSALogger(log_dir=str(tmp_path), log_type="performance")
        assert gl_one.accuracy_is_info() is False
        assert gl_one.accuracy_is_debug() is False

    def test_normal_is_info_debug(self, tmp_path):
        gl_all = gsa_log.GSALogger(log_dir=str(tmp_path))
        assert gl_all.normal_is_info() is True
        assert gl_all.normal_is_debug() is False
        gl_dbg = gsa_log.GSALogger(
            log_dir=str(tmp_path), level=gsa_log.GSALogger.LEVEL_DEBUG
        )
        assert gl_dbg.normal_is_debug() is True
        gl_one = gsa_log.GSALogger(log_dir=str(tmp_path), log_type="accuracy")
        assert gl_one.normal_is_info() is False

    @pytest.mark.parametrize(
        "level_method,tag",
        [
            ("performance_debug", "performance"),
            ("performance_info", "performance"),
            ("performance_warning", "performance"),
            ("performance_error", "performance"),
            ("accuracy_debug", "accuracy"),
            ("accuracy_info", "accuracy"),
            ("accuracy_warning", "accuracy"),
            ("accuracy_error", "accuracy"),
            ("normal_debug", "normal"),
            ("normal_info", "normal"),
            ("normal_warning", "normal"),
            ("normal_error", "normal"),
        ],
    )
    def test_type_level_routing(self, tmp_path, level_method, tag):
        gl = gsa_log.GSALogger(
            log_dir=str(tmp_path), show_time=False, level=gsa_log.GSALogger.LEVEL_DEBUG
        )
        getattr(gl, level_method)(f"msg-{level_method}")
        target = os.path.join(str(tmp_path), f"log_for_0_tp_0_{tag}.log")
        others = [t for t in ("performance", "accuracy", "normal") if t != tag]
        assert f"msg-{level_method}" in self._read(target)
        for t in others:
            other = os.path.join(str(tmp_path), f"log_for_0_tp_0_{t}.log")
            # 消息不得串写；其他类型文件若存在则不应包含该消息
            if os.path.exists(other):
                assert f"msg-{level_method}" not in self._read(other)

    def test_compat_interface_routes_to_normal(self, tmp_path):
        gl = gsa_log.GSALogger(log_dir=str(tmp_path), show_time=False)
        gl.debug("d")
        gl.info("i")
        gl.warning("w")
        gl.error("e")
        normal = self._read(os.path.join(str(tmp_path), "log_for_0_tp_0_normal.log"))
        assert "[INFO] i" in normal and "[ERROR] e" in normal
        # level=INFO 时 debug 被过滤
        assert "[DEBUG] d" not in normal
        assert "[WARNING] w" in normal

    def test_is_debug_enabled_property(self, tmp_path):
        gl = gsa_log.GSALogger(log_dir=str(tmp_path))
        assert gl.is_debug_enabled is False  # INFO 级别
        gl_dbg = gsa_log.GSALogger(
            log_dir=str(tmp_path), level=gsa_log.GSALogger.LEVEL_DEBUG
        )
        assert gl_dbg.is_debug_enabled is True
        gl_one = gsa_log.GSALogger(log_dir=str(tmp_path), log_type="performance")
        assert gl_one.is_debug_enabled is False  # normal logger 不存在

    def test_is_info_enabled_property(self, tmp_path):
        gl = gsa_log.GSALogger(log_dir=str(tmp_path))
        assert gl.is_info_enabled is True
        gl_warn = gsa_log.GSALogger(
            log_dir=str(tmp_path), level=gsa_log.GSALogger.LEVEL_WARNING
        )
        assert gl_warn.is_info_enabled is False
        gl_one = gsa_log.GSALogger(log_dir=str(tmp_path), log_type="accuracy")
        assert gl_one.is_info_enabled is False
