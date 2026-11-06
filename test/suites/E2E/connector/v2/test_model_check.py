"""Opt-in E2E case. The checker declares hardware only in its own process."""

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.stage(2)
@pytest.mark.feature("connector")
def test_model_check():
    args = os.environ.get("UCM_V2_MODEL_CHECK_ARGS")
    if not args:
        pytest.skip("Set UCM_V2_MODEL_CHECK_ARGS to the run.py arguments")
    runner = Path(__file__).parent / "model_check" / "run.py"
    subprocess.run([sys.executable, str(runner), *shlex.split(args)], check=True)
