"""
Online Inference E2E Tests for UCM (Unified Cache Management).

This module contains tests for online inference with UCM, which connects to
a running inference server via OpenAI-compatible API.

The tests verify:
1. SSD cache save and load accuracy (Phase 1 - prefix caching disabled)
2. HBM + SSD mixed cache hit accuracy (Phase 2 - prefix caching enabled)

Test flow mirrors test_offline_inference.py:
- Phase 1: Start vLLM WITHOUT prefix caching -> send full prompt twice -> KV saved to SSD
- Phase 2: Start vLLM WITH prefix caching -> send partial prompt (warm HBM),
           then full prompt (hits HBM + SSD) -> verify accuracy
"""

import os
from typing import List

import pytest
import yaml
from common.common_inference_utils import (
    extract_answers,
    load_prompt_from_file,
    match_any_answer,
)
from common.llm_connection.LLMBase import LLMRequest
from common.llm_connection.openai_connector import OpenAIConn
from common.llm_connection.token_counter import HuggingFaceTokenizer
from common.path_utils import get_path_relative_to_test_root, get_path_to_model


class TestBasicOnlineInference:
    """Test basic online inference functionality."""

    @pytest.mark.stage(1)
    @pytest.mark.feature("fvt_test")
    @pytest.mark.parametrize("max_tokens", [200])
    @pytest.mark.parametrize("prompt_split_ratio", [0.5])
    def test_online_accuracy_hbm_ssd_mixed(
        self,
        max_tokens: int,
        prompt_split_ratio: float,
    ):
        # Load configuration
        config_file = get_path_relative_to_test_root("config.yaml")
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Get server_url from config
        server_url = config.get("llm_connection", {}).get("server_url", "")
        tokenizer_path = config.get("llm_connection", {}).get("tokenizer_path", "")

        if not server_url:
            pytest.fail("server_url not found in config.yaml")

        # Load test prompt and standard answers
        test_prompt, standard_answers = load_prompt_from_file(
            get_path_relative_to_test_root(
                "suites/E2E/prompts/test_offline_inference.json"
            )
        )
        if not standard_answers:
            pytest.fail("No standard answers found in prompt.json")

        print(f"Standard answers: {standard_answers}")

        # Split prompt for Phase 2
        prompt_first_part = test_prompt[: int(len(test_prompt) * prompt_split_ratio)]

        system_content = "先读问题，再根据下面的文章内容回答问题，不要进行分析，不要重复问题，用简短的语句给出答案。\n\n例如：\u201c全国美国文学研究会的第十八届年会在哪所大学举办的？\u201d\n回答应该为：\u201cxx大学\u201d。\n\n"
        # Prepare messages
        phase1_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": test_prompt},
        ]
        phase2_partial_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt_first_part},
        ]

        print(f"\n===== Online HBM + SSD Mixed Accuracy Test =====")
        print(f"Server URL: {server_url}")
        print(f"Full prompt length: {len(test_prompt)} chars")
        print(f"Max tokens: {max_tokens}")
        print(f"Prompt split ratio: {prompt_split_ratio}")

        # Connect to existing server
        client = OpenAIConn(
            base_url=server_url,
            tokenizer=HuggingFaceTokenizer(tokenizer_path),
            model="",  # Will be set after list_models
        )
        assert client.health_check()

        # Get the first model from list_models
        models = client.list_models()
        print(f"server models: {models}")
        if not models:
            pytest.fail("No models available on the server")
        served_model_name = models[0]
        client.model = served_model_name
        print(f"Using model: {served_model_name}")

        # ===== Phase 1: Disable HBM PC, save KV cache to SSD and load (baseline) =====
        print(f"\n===== Phase 1: Save KV Cache to SSD And Load (Baseline) =====")

        print(f"clear hbm before Phase 1")
        client.clear_hbm()

        # Phase 1.1: Send full prompt -> KV cache saved to SSD
        phase1_1_output = client.chat(
            LLMRequest(messages=phase1_messages, max_tokens=max_tokens, temperature=0.0)
        ).text
        print(f'Phase 1.1 output: "{phase1_1_output}"')

        # Clear HBM after Phase 1.1
        print(f"clear hbm after Phase 1.1")
        client.clear_hbm()

        # Phase 1.2: Send same prompt again -> KV cache loaded from SSD
        phase1_2_output = client.chat(
            LLMRequest(messages=phase1_messages, max_tokens=max_tokens, temperature=0.0)
        ).text
        print(f'Phase 1.2 output: "{phase1_2_output}"')

        # Clear HBM after Phase 1.2
        print(f"clear hbm after Phase 1.2")
        client.clear_hbm()

        print("Phase 1 completed.")

        # ===== Phase 2: Enable HBM PC, test HBM + SSD mixed hit =====
        print(f"\n===== Phase 2: HBM + SSD Mixed Hit Test =====")
        print(f"Using existing server with enable_prefix_caching=True")

        # Phase 2.1: Send partial prompt -> warm HBM prefix cache
        phase2_partial_output = client.chat(
            LLMRequest(
                messages=phase2_partial_messages,
                max_tokens=max_tokens,
                temperature=0.0,
            )
        ).text
        print(f"[INFO] Phase 2.1 output (partial prompt): {phase2_partial_output}")

        # Phase 2.2: Send full prompt -> hits HBM (prefix) + SSD (suffix)
        phase2_full_output = client.chat(
            LLMRequest(messages=phase1_messages, max_tokens=max_tokens, temperature=0.0)
        ).text
        print(f"[INFO] Phase 2.2 output (full prompt): {phase2_full_output}")

        client.close()

        print("Phase 2 completed.")

        # ===== Accuracy Test Results =====
        print(f"\n[INFO] ===== Accuracy Test Results =====")

        # Phase 1 accuracy check
        phase1_correct = match_any_answer(
            phase1_1_output, standard_answers
        ) and match_any_answer(phase1_2_output, standard_answers)
        if not phase1_correct:
            print(f"\n===== Phase 1: SSD Load Accuracy Test (Exact Match) =====")
            print(f"Incorrect answer in Phase 1.1 (SSD save) or Phase 1.2 (SSD load)!")
            print(f"Phase 1.1 output:\n{phase1_1_output}")
            print(f"Phase 1.2 output:\n{phase1_2_output}")
            print(f"Standard answers:\n{standard_answers}")
            pytest.fail("SSD Load Accuracy Test Failed!")

        # Phase 2 accuracy check
        phase2_correct = match_any_answer(phase2_full_output, standard_answers)
        if not phase2_correct:
            print(f"\n===== Phase 2: HBM + SSD Mixed Accuracy Test (Exact Match) =====")
            print(f"Incorrect answer in Phase 2.2 (HBM + SSD mixed)!")
            print(f"Phase 2.2 output:\n{phase2_full_output}")
            print(f"Standard answers:\n{standard_answers}")
            pytest.fail("HBM + SSD Mixed Accuracy Test Failed!")

        print("\n===== All Tests Passed! =====")
