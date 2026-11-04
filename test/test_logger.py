import ast
import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_logger_module(monkeypatch):
    backend = ModuleType("ucmlogger")
    backend.Level = SimpleNamespace(
        DEBUG=10, INFO=20, WARNING=30, ERROR=40, CRITICAL=50
    )
    backend.rate_limited = []
    backend.setup = lambda *args: None
    backend.flush = lambda: None
    backend.log = lambda *args: None
    backend.log_rate_limit = lambda *args: backend.rate_limited.append(args)

    ucm = ModuleType("ucm")
    shared = ModuleType("ucm.shared")
    infra = ModuleType("ucm.shared.infra")
    infra.ucmlogger = backend
    monkeypatch.setitem(sys.modules, "ucm", ucm)
    monkeypatch.setitem(sys.modules, "ucm.shared", shared)
    monkeypatch.setitem(sys.modules, "ucm.shared.infra", infra)
    monkeypatch.setattr(logging.getLogger("ucm"), "handlers", [])

    module_name = "ucm_logger_under_test"
    spec = importlib.util.spec_from_file_location(
        module_name, REPO_ROOT / "ucm/logger.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module, backend


def test_error_limit_keeps_error_level_and_uses_rate_limiter(monkeypatch):
    module, backend = _load_logger_module(monkeypatch)
    logger = module.init_logger("error-limit-test")

    logger.error_limit("wait failed: %s", "disk error")

    assert len(backend.rate_limited) == 1
    level, _, _, _, message = backend.rate_limited[0]
    assert level == backend.Level.ERROR
    assert message == "wait failed: disk error"


@pytest.mark.parametrize(
    "relative_path",
    [
        "ucm/integration/vllm/ucm_connector.py",
        "ucm/integration/vllm/hla_connector.py",
    ],
)
def test_dump_wait_failures_use_rate_limited_error_logging(relative_path):
    source = (REPO_ROOT / relative_path).read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    matching_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        text = ast.get_source_segment(source, node)
        if text and "wait for dump kv cache failed" in text:
            matching_calls.append(node.func.attr)

    assert matching_calls
    assert set(matching_calls) == {"error_limit"}


def test_default_rate_limit_window_is_sixty_seconds():
    header = (REPO_ROOT / "ucm/shared/infra/logger/cc/spdlog_logger.h").read_text(
        encoding="utf-8"
    )
    logger_doc = (REPO_ROOT / "ucm/logger.py").read_text(encoding="utf-8")

    assert "kDefaultRateLimitWindowMs = 60000" in header
    assert "default: 60000 = 60s" in logger_doc
