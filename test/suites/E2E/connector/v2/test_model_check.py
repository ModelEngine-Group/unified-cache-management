"""Opt-in E2E case. The checker declares hardware only in its own process."""

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

RUNNER = Path(__file__).parent / "model_check" / "run.py"


@pytest.mark.stage(2)
@pytest.mark.feature("connector")
def test_model_check():
    args = os.environ.get("UCM_V2_MODEL_CHECK_ARGS")
    if not args:
        pytest.skip("Set UCM_V2_MODEL_CHECK_ARGS to the run.py arguments")
    subprocess.run([sys.executable, str(RUNNER), *shlex.split(args)], check=True)


@pytest.mark.stage(2)
@pytest.mark.feature("connector")
@pytest.mark.parametrize(
    ("repo_id", "revision", "policies"),
    [
        pytest.param(
            "deepseek-ai/DeepSeek-V4-Flash",
            "60d8d70770c6776ff598c94bb586a859a38244f1",
            ("FA", "WA"),
            id="deepseek-v4-flash",
        ),
        pytest.param(
            "deepseek-ai/DeepSeek-V4.1-Flash",
            "314113f8a7dadb388215396ba5fbee2c610a04d6",
            ("FA",),
            id="deepseek-v4.1-flash",
        ),
        pytest.param(
            "zai-org/GLM-5.3-Flash",
            "46a050bec13002ad29ee795c16e61cf4fa3c4a9c",
            ("FA", "STATE"),
            id="glm-5.3-flash",
        ),
    ],
)
@pytest.mark.parametrize("hit", [0, 90], ids=["cold", "hit90"])
def test_hybrid_model(repo_id, revision, policies, hit):
    args = os.environ.get("UCM_V2_HYBRID_ARGS")
    if not args:
        pytest.skip("Set UCM_V2_HYBRID_ARGS to the target platform and topology")

    from huggingface_hub import snapshot_download

    model = snapshot_download(
        repo_id,
        revision=revision,
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "processor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
        ],
    )
    print(f"Model: {repo_id}@{revision} ({model})", flush=True)
    subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            *shlex.split(args),
            "--model",
            model,
            "--assume-hit-percent",
            str(hit),
            "--expect-policy",
            *policies,
        ],
        check=True,
    )
