import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def patched_ascend():
    hooks = {}
    utils = ModuleType("ucm.integration.vllm.patch.utils")

    def when_imported(name):
        def register(callback):
            hooks[name] = callback
            return callback

        return register

    utils.when_imported = when_imported
    utils.patch_or_inject = setattr
    logger = ModuleType("ucm.logger")
    logger.init_logger = MagicMock(return_value=MagicMock())
    path = (
        Path(__file__).parents[3]
        / "ucm/integration/vllm/patch/v0300/vllm_ascend"
        / "mamba_block_aligned_split_patch.py"
    )
    spec = importlib.util.spec_from_file_location("ucm_mamba_split_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {utils.__name__: utils, logger.__name__: logger}):
        spec.loader.exec_module(module)

    def old_split(self, request, num_new_tokens, nlc=0, nec=0):
        return num_new_tokens

    ascend = SimpleNamespace(
        Scheduler=type("Scheduler", (), {"_mamba_block_aligned_split": old_split}),
        _mamba_block_aligned_split=old_split,
        _original_mamba_block_aligned_split=MagicMock(return_value=17),
        _get_sparse_index_kpool=lambda config: 4 if config.sparse else None,
        _skips_eagle_block_drop=lambda config: config is None
        or (config.is_kv_producer and not config.is_kv_consumer),
    )
    hook = hooks["vllm_ascend.patch.platform.patch_mamba_block_aligned_split"]
    hook(ascend)
    assert (
        ascend.Scheduler._mamba_block_aligned_split is ascend._mamba_block_aligned_split
    )
    return ascend, hook


def _scheduler(role="kv_both", sparse=True):
    config = (
        None
        if role is None
        else SimpleNamespace(
            is_kv_consumer=role in ("kv_both", "kv_consumer"),
            is_kv_producer=role in ("kv_both", "kv_producer"),
        )
    )
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            kv_transfer_config=config,
            model_config=SimpleNamespace(sparse=sparse),
        ),
        block_size=640,
        use_eagle=False,
        use_eagle_block_drop=False,
    )


def _request(computed=640, prompt=128359, total=128359):
    return SimpleNamespace(
        num_computed_tokens=computed,
        num_prompt_tokens=prompt,
        num_tokens=total,
    )


@pytest.mark.parametrize(
    "computed,local,external", [(640, 0, 0), (0, 640, 0), (0, 0, 640)]
)
@pytest.mark.parametrize("prompt", [128359, 640])
def test_continue_alignment_after_chunk_or_partial_hit(
    patched_ascend, computed, local, external, prompt
):
    ascend, _ = patched_ascend
    result = ascend.Scheduler._mamba_block_aligned_split(
        _scheduler(), _request(computed, prompt), 4096, local, external
    )
    assert result == 3840
    assert (computed + local + external + result) % 640 == 0


@pytest.mark.parametrize(
    "computed,local,external", [(1279, 0, 0), (0, 1279, 0), (0, 0, 1279), (1280, 0, 0)]
)
def test_preserve_final_token_verifier_and_decode(
    patched_ascend, computed, local, external
):
    ascend, _ = patched_ascend
    result = ascend.Scheduler._mamba_block_aligned_split(
        _scheduler(), _request(computed, 1280, 1280), 8, local, external
    )
    assert result == 8
    ascend._original_mamba_block_aligned_split.assert_not_called()


@pytest.mark.parametrize("role", ["kv_both", "kv_consumer", "kv_producer", None])
def test_nonsparse_prefill_preserves_upstream_flags(patched_ascend, role):
    ascend, _ = patched_ascend
    scheduler = _scheduler(role, sparse=False)
    scheduler.use_eagle = scheduler.use_eagle_block_drop = True
    observed = []

    def upstream(*args):
        observed.append((scheduler.use_eagle, scheduler.use_eagle_block_drop))
        return 17

    ascend._original_mamba_block_aligned_split.side_effect = upstream
    assert ascend._mamba_block_aligned_split(scheduler, _request(), 4096) == 17
    expected = role in ("kv_both", "kv_consumer")
    assert observed == [(expected, expected)]
    assert scheduler.use_eagle and scheduler.use_eagle_block_drop


def test_restore_drop_flags_on_upstream_error(patched_ascend):
    ascend, _ = patched_ascend
    scheduler = _scheduler("kv_producer", sparse=False)
    scheduler.use_eagle = scheduler.use_eagle_block_drop = True
    ascend._original_mamba_block_aligned_split.side_effect = RuntimeError("test error")
    with pytest.raises(RuntimeError, match="test error"):
        ascend._mamba_block_aligned_split(scheduler, _request(), 4096)
    assert scheduler.use_eagle and scheduler.use_eagle_block_drop


def test_registration_is_idempotent(patched_ascend):
    ascend, hook = patched_ascend
    original = ascend.Scheduler._mamba_block_aligned_split
    hook(ascend)
    assert ascend.Scheduler._mamba_block_aligned_split is original
