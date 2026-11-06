"""Post-match logging preserves results and observes synchronized token counts."""

import logging
import types
from functools import wraps
from unittest.mock import Mock

import pytest
from common.sglang_utils import load_source


@pytest.fixture
def setup_patch(caplog):
    class Linker:
        _cache_match_log_rank = True
        _dp_rank = 1
        _attn_dp_rank = 2
        _tp_rank = 0

    original = Mock()

    class Wrapper:
        def __init__(self):
            self.cache_linker = Linker()

        def match(self, key, req, result):
            return original(self, key, req, result)

    importer = Mock(
        return_value=types.SimpleNamespace(UnifiedCacheLinkerWrapper=Wrapper)
    )
    module = load_source(
        "ucm/integration/sglang/patch/cache_match_logging.py",
        importlib=types.SimpleNamespace(import_module=importer),
        wraps=wraps,
    )
    module.logger.warning_once = Mock()
    caplog.set_level(logging.INFO, logger=module.logger.name)
    module.install_cache_match_logging(Linker)
    req = types.SimpleNamespace(
        rid="request-a", origin_input_ids=range(4031), output_ids=[]
    )
    return module, Linker, Wrapper, original, req


@pytest.mark.parametrize("hbm, external", [(2048, 1792), (0, 0), (4030, 0)])
def test_summary_uses_returned_match_not_local_lookup(
    setup_patch, caplog, hbm, external
):
    module, _, Wrapper, original, req = setup_patch
    before = types.SimpleNamespace(device_prefix_len=hbm, host_hit_length=0)
    after = types.SimpleNamespace(device_prefix_len=hbm, host_hit_length=external)
    original.return_value = after if external else before
    wrapper = Wrapper()
    key = object()
    assert wrapper.match(key, req, before) is original.return_value
    original.assert_called_once_with(wrapper, key, req, before)
    assert len(caplog.records) == 1
    assert caplog.records[0].getMessage() == (
        f"[CacheMatch] request_id=request-a total_tokens=4031 "
        f"hit_hbm_tokens={hbm} hit_external_tokens={external} "
        "dp_rank=1 attn_dp_rank=2 tp_rank=0"
    )


def test_existing_host_hits_are_not_reported_as_external(setup_patch, caplog):
    _, _, Wrapper, original, req = setup_patch
    before = types.SimpleNamespace(device_prefix_len=64, host_hit_length=128)
    original.return_value = before
    assert Wrapper().match(None, req, before) is before
    assert "hit_external_tokens=0" in caplog.text


def test_total_is_input_length_even_with_generated_tokens(setup_patch, caplog):
    _, _, Wrapper, original, req = setup_patch
    req.output_ids = range(100)
    result = types.SimpleNamespace(device_prefix_len=4096)
    original.return_value = result
    Wrapper().match(None, req, result)
    assert "total_tokens=4031 hit_hbm_tokens=4096" in caplog.text


@pytest.mark.parametrize("skip", ["rank", "backend", "level"])
def test_other_ranks_backends_and_disabled_info_are_unaffected(
    setup_patch, caplog, skip
):
    module, _, Wrapper, original, _ = setup_patch
    wrapper = Wrapper()
    if skip == "rank":
        wrapper.cache_linker._cache_match_log_rank = False
    elif skip == "backend":
        wrapper.cache_linker = object()
    else:
        caplog.set_level(logging.WARNING, logger=module.logger.name)
    # No request fields may be read when logging is disabled for this call.
    assert wrapper.match(None, object(), object()) is original.return_value
    assert not caplog.records


def test_install_is_idempotent_and_only_first_match_is_info(setup_patch, caplog):
    module, Linker, Wrapper, original, req = setup_patch
    installed = Wrapper.match
    module.install_cache_match_logging(Linker)
    assert Wrapper.match is installed
    result = types.SimpleNamespace(device_prefix_len=0)
    original.return_value = result
    wrapper = Wrapper()
    for _ in range(2):
        wrapper.match(None, req, result)
    assert original.call_count == 2
    assert len(caplog.records) == 1


@pytest.mark.parametrize("level", [logging.INFO, logging.DEBUG])
def test_loaded_hbm_and_generated_tokens_do_not_replace_first_match(
    setup_patch, caplog, level
):
    module, _, Wrapper, original, req = setup_patch
    caplog.set_level(level, logger=module.logger.name)
    wrapper = Wrapper()
    before = types.SimpleNamespace(device_prefix_len=0, host_hit_length=0)
    loaded = types.SimpleNamespace(device_prefix_len=0, host_hit_length=3840)
    original.return_value = loaded
    assert wrapper.match(None, req, before) is loaded
    for generated, hbm in ((1, 3840), (189, 4096)):
        req.output_ids = range(generated)
        result = types.SimpleNamespace(device_prefix_len=hbm, host_hit_length=0)
        original.return_value = result
        assert wrapper.match(None, req, result) is result
    info = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert len(info) == 1
    assert "total_tokens=4031 hit_hbm_tokens=0 hit_external_tokens=3840" in info[0]
    debug = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
    assert len(debug) == (2 if level == logging.DEBUG else 0)
    if debug:
        assert all(message.startswith("[CacheRematch]") for message in debug)
        assert "sequence_tokens=4032" in debug[0]
        assert "sequence_tokens=4220" in debug[1]
    assert original.call_count == 3


def test_new_request_with_reused_id_is_not_suppressed(setup_patch, caplog):
    _, _, Wrapper, original, req = setup_patch
    wrapper = Wrapper()
    result = types.SimpleNamespace(device_prefix_len=0)
    original.return_value = result
    wrapper.match(None, req, result)
    replacement = types.SimpleNamespace(
        rid=req.rid, origin_input_ids=range(128), output_ids=[]
    )
    wrapper.match(None, replacement, result)
    assert len(caplog.records) == 2
    assert "total_tokens=128" in caplog.records[1].getMessage()


def test_match_errors_propagate_without_success_log(setup_patch, caplog):
    _, _, Wrapper, original, req = setup_patch
    original.side_effect = RuntimeError("lookup failed")
    with pytest.raises(RuntimeError, match="lookup failed"):
        Wrapper().match(None, req, object())
    assert not caplog.records


def test_missing_fields_do_not_break_match(setup_patch):
    module, _, Wrapper, original, _ = setup_patch
    assert Wrapper().match(None, object(), object()) is original.return_value
    module.logger.warning_once.assert_called_once()


def test_missing_wrapper_skips_install(setup_patch, caplog):
    module, Linker, _, _, _ = setup_patch
    module.importlib.import_module.return_value = types.SimpleNamespace()
    module.install_cache_match_logging(Linker)
    assert "cache-match logging skipped" in caplog.text
