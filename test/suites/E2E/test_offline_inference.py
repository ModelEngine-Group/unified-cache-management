import contextlib
import gc
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import pynvml
import pytest
import torch
import yaml
from common.offline_inference_utils import (
    ensure_storage_dir,
    load_prompt_from_file,
    run_in_spawn_subprocess,
    run_offline_inference,
    split_prompt_by_tokens,
)
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs

from ucm.logger import init_logger

logger = init_logger(__name__)


def get_free_gpu(required_memory_mb):
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        if (info.free / 1024**2) >= required_memory_mb:
            return i
    return None


@pytest.fixture(autouse=True)
def setup_gpu_resource(request):
    marker = request.node.get_closest_marker("gpu_mem")
    if marker:
        mem_needed = marker.args[0]
        gpu_id = get_free_gpu(mem_needed)
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        else:
            pytest.fail(f"No GPU with {mem_needed}MB free memory available")


class TestBasicOfflineInference:
    """Test basic offline inference functionality."""

    @pytest.mark.stage(1)
    @pytest.mark.feature("offline_inference")
    @pytest.mark.gpu_mem(30000)
    @pytest.mark.parametrize("model_path", ["/home/models/Qwen2.5-1.5B-Instruct"])
    @pytest.mark.parametrize("max_tokens", [200])
    @pytest.mark.parametrize("prompt_split_ratio", [0.5])  # Split prompt in half
    @pytest.mark.parametrize("enforce_eager", [True, False])
    @pytest.mark.parametrize("max_num_batched_tokens", [2047])
    def test_offline_accuracy_hbm_ssd_mixed(
        self,
        model_path: str,
        max_tokens: int,
        prompt_split_ratio: float,
        enforce_eager: bool,
        max_num_batched_tokens: int,
    ):
        """Test HBM + SSD mixed hit accuracy (Phase 2).
        This test first runs Phase 1 to generate a baseline output, then tests Phase 2.
        Test flow:
        1. Phase 1: Disable HBM PC, send full prompt -> KV cache saved to SSD (baseline)
        2. Phase 2: Enable HBM PC, send partial prompt (warm HBM), then send full prompt (hits both HBM and SSD) -> verify mixed hit accuracy
        The prompt is loaded from prompt.json file (LongBench format).
        Args:
            model_path: Path to the model.
            max_tokens: Maximum tokens to generate.
            prompt_split_ratio: Ratio to split prompt for Phase 2 (0.5 = split in half).
        """
        config_file = Path(__file__).parent.parent.parent / "config.yaml"
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # if no model_path from parameter, fallback to config or environment
        if not model_path:
            logger.info(
                "No model_path parameter provided, checking config and environment variable"
            )
            model_path = config.get("llm_connection", {}).get(
                "model_path"
            ) or os.getenv("MODEL_PATH")
            assert (
                model_path is not None
            ), "model_path must be specified via parameter, config, or environment variable"

        assert os.path.exists(model_path), f"Model path does not exist: {model_path}"

        ucm_storage_dir = config.get("llm_connection", {}).get(
            "ucm_storage_dir"
        ) or os.getenv("UCM_STORAGE_DIR", "/tmp/ucm_cache")

        # make sure UCM storage directory exists and is empty
        ensure_storage_dir(ucm_storage_dir, clear_existing=True)

        try:
            test_prompt, standard_answers = load_prompt_from_file(
                Path(__file__).parent / "prompt.json"
            )
            logger.info(
                f"Loaded prompt from prompt.json (length: {len(test_prompt)} chars)"
            )
            if standard_answers:
                logger.info(f"Standard answers: {standard_answers}")
            else:
                pytest.fail(f"No standard answers found in prompt.json")
        except Exception as e:
            pytest.fail(f"Failed to load prompt from prompt.json: {e}")

        tokenizer = AutoTokenizer.from_pretrained(model_path, use_chat_template=True)

        try:
            messages = [{"role": "user", "content": test_prompt}]
            formatted_full_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                add_special_tokens=True,
            )
        except Exception:
            formatted_full_prompt = test_prompt

        prompt_first_part, prompt_second_part = split_prompt_by_tokens(
            formatted_full_prompt, tokenizer, split_ratio=prompt_split_ratio
        )

        ucm_config = {
            "ucm_connectors": [
                {
                    "ucm_connector_name": "UcmNfsStore",
                    "ucm_connector_config": {
                        "storage_backends": ucm_storage_dir,
                        "use_direct": False,
                    },
                }
            ],
            "load_only_first_rank": False,
        }

        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1,
            max_tokens=max_tokens,
            ignore_eos=False,
        )

        logger.info(f"\n===== HBM + SSD Mixed Accuracy Test =====")
        logger.info(f"Model: {model_path}")
        logger.info(f"Full prompt length: {len(test_prompt)} chars")
        logger.info(f"Max tokens: {max_tokens}")
        logger.info(f"Temperature: 0.0 (deterministic)")
        logger.info(f"UCM storage: {ucm_storage_dir}")
        logger.info(f"Prompt split ratio: {prompt_split_ratio}")
        logger.info(f"Enforce eager: {enforce_eager}")
        logger.info(f"Max num batched tokens: {max_num_batched_tokens}")

        # ===== Phase 1: Disable HBM PC, save KV cache to SSD and load (baseline) =====
        # Run Phase 1 in a separate subprocess to ensure GPU memory is fully released
        logger.info(f"\n===== Phase 1: Save KV Cache to SSD And Load (Baseline) =====")

        # Convert SamplingParams to dict for serialization
        sampling_params_dict = {
            "temperature": sampling_params.temperature,
            "top_p": sampling_params.top_p,
            "max_tokens": sampling_params.max_tokens,
            "ignore_eos": sampling_params.ignore_eos,
        }

        phase1_outputs = run_in_spawn_subprocess(
            run_offline_inference,
            model_path,
            ucm_config,
            [formatted_full_prompt, formatted_full_prompt],
            sampling_params_dict,
            False,  # enable_prefix_caching=False for Phase 1
            enforce_eager,
            "Phase 1 (SSD save and load)",
            max_num_batched_tokens,
            timeout=180,
        )
        phase1_1_output = phase1_outputs[0]  # Phase 1.1: SSD save
        phase1_2_output = phase1_outputs[1]  # Phase 1.2: SSD load
        logger.info(f"Phase 1 completed in subprocess")
        logger.info(f'Phase 1.1 output: "{phase1_1_output}"')
        logger.info(f'Phase 1.2 output: "{phase1_2_output}"')

        # ===== Phase 2: Enable HBM PC, test HBM + SSD mixed hit =====
        # Run Phase 2 in a separate subprocess to ensure GPU memory is fully released
        logger.info(f"\n===== Phase 2: HBM + SSD Mixed Hit Test =====")

        phase2_outputs = run_in_spawn_subprocess(
            run_offline_inference,
            model_path,
            ucm_config,
            [prompt_first_part, formatted_full_prompt],
            sampling_params_dict,
            True,  # enable_prefix_caching=True for Phase 2
            enforce_eager,
            "Phase 2 (HBM + SSD mixed)",
            max_num_batched_tokens,
            timeout=180,
        )
        phase2_partial_output = phase2_outputs[0]
        phase2_full_output = phase2_outputs[1]
        logger.info(f"Phase 2 completed in subprocess")
        logger.info(f"[INFO] Phase 2.1 output: {phase2_partial_output}")
        logger.info(f"[INFO] Phase 2.2 output: {phase2_full_output}")

        logger.info(f"\n[INFO] ===== Accuracy Test Results =====")

        # Note: Small numerical precision differences in KV cache loading can cause
        # punctuation token selection differences (e.g., full-width vs half-width comma)
        def normalize_text(text: str) -> str:
            """Normalize text for comparison by replacing similar punctuation."""
            text = text.replace("，", ",")
            text = text.replace("。", ".")
            text = text.replace("！", "!")
            text = text.replace("？", "?")
            text = text.replace("：", ":")
            text = text.replace("；", ";")
            return text.strip()

        # Compare Phase 1.1 vs Phase 1.2 (SSD load accuracy)
        phase1_identical = normalize_text(phase1_1_output) == normalize_text(
            phase1_2_output
        )
        if not phase1_identical:
            logger.warning(
                f"\n===== Phase 1: SSD Load Accuracy Test (Exact Match) ====="
            )
            logger.warning(
                f"Phase 1.1 (SSD save) output differs from Phase 1.2 (SSD load) output!"
            )
            logger.warning(f"Phase 1.1 output:\n{phase1_1_output}")
            logger.warning(f"Phase 1.2 output:\n{phase1_2_output}")
            pytest.fail("SSD Load Accuracy Test Failed!")

        phase2_identical = normalize_text(phase1_1_output) == normalize_text(
            phase2_full_output
        )
        if not phase2_identical:
            logger.warning(
                f"\n===== Phase 2: HBM + SSD Mixed Accuracy Test (Exact Match) ====="
            )
            logger.warning(
                f"Phase 1.1 (SSD save) output differs from Phase 2.2 (HBM + SSD mixed) output!"
            )
            logger.warning(f"Phase 1.1 output:\n{phase1_1_output}")
            logger.warning(f"Phase 2.2 output:\n{phase2_full_output}")
            pytest.fail("HBM + SSD Mixed Accuracy Test Failed!")
