import os
from pathlib import Path

import pytest
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

from ucm.logger import init_logger

logger = init_logger(__name__)


class TestBasicOfflineInferenceSparse:
    """Test basic offline inference functionality."""

    @pytest.mark.stage(1)
    @pytest.mark.feature("offline_inference_sparse")
    @pytest.mark.gpu_mem(30000)
    @pytest.mark.parametrize("model_path", ["/home/models/Qwen2.5-1.5B-Instruct"])
    @pytest.mark.parametrize("max_tokens", [200])
    @pytest.mark.parametrize("prompt_split_ratio", [0.5])  # Split prompt in half
    @pytest.mark.parametrize("enforce_eager", [True, False])
    @pytest.mark.parametrize("max_num_batched_tokens", [2047])
    def test_offline_accuracy_no_sparse(
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
                Path(__file__).parent / "prompts" / "test_offline_inference.json"
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
            messages = [
                {
                    "role": "system",
                    "content": "先读问题，再根据下面的文章内容回答问题，不要进行分析，不要重复问题，用简短的语句给出答案。\n\n例如：“全国美国文学研究会的第十八届年会在哪所大学举办的？”\n回答应该为：“xx大学”。\n\n",
                },
                {"role": "user", "content": test_prompt},
            ]
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

        def match_any_answer(output: str, answers: list[str]) -> bool:
            """Check if output matches any of the standard answers."""
            for answer in answers:
                if normalize_text(output) == normalize_text(answer):
                    return True
            return False

        # Compare Phase 1.1 vs Phase 1.2 (SSD load accuracy)
        phase1_correct = match_any_answer(
            phase1_1_output, standard_answers
        ) and match_any_answer(phase1_2_output, standard_answers)
        if not phase1_correct:
            logger.warning(
                f"\n===== Phase 1: SSD Load Accuracy Test (Exact Match) ====="
            )
            logger.warning(
                f"Incorrect answer in Phase 1.1 (SSD save) or Phase 1.2 (SSD load) output!"
            )
            logger.warning(f"Phase 1.1 output:\n{phase1_1_output}")
            logger.warning(f"Phase 1.2 output:\n{phase1_2_output}")
            logger.warning(f"Standard answers:\n{standard_answers}")
            pytest.fail("SSD Load Accuracy Test Failed!")

        # Phase 2.1 should be skipped from accuracy check since it's only partial prompt
        phase2_correct = match_any_answer(phase2_full_output, standard_answers)
        if not phase2_correct:
            logger.warning(
                f"\n===== Phase 2: HBM + SSD Mixed Accuracy Test (Exact Match) ====="
            )
            logger.warning(f"Incorrect answer in Phase 2.2 (HBM + SSD mixed) output!")
            logger.warning(f"Phase 2.2 output:\n{phase2_full_output}")
            logger.warning(f"Standard answers:\n{standard_answers}")
            pytest.fail("HBM + SSD Mixed Accuracy Test Failed!")
