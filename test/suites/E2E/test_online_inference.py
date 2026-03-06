"""
Online Inference E2E Tests for UCM (Unified Cache Management).

This module contains tests for online inference with UCM, which connects to
a running inference server via OpenAI-compatible API.

The tests verify:
1. SSD cache save and load accuracy (Phase 1)
2. HBM + SSD mixed cache hit accuracy (Phase 2)

Unlike offline inference tests, these tests require:
- A running inference server with UCM configured
- The server must have prefix caching support
- The server must expose /reset_prefix_cache endpoint for HBM clearing

The tests can be run in two modes:
1. With VLLMServerManager (default): Automatically starts/stops vLLM server
2. External server: Set SERVER_URL environment variable to use existing server

Prerequisites for VLLMServerManager mode:
- vLLM installed with OpenAI API server support
- GPU available for inference
- Model weights accessible at /home/models/{model_name}
"""

import os
from pathlib import Path
from typing import List

import pytest
import yaml
from common.online_inference_utils import (
    OnlineInferenceClient,
    VLLMServerManager,
    load_prompt_from_file,
    run_online_inference,
    split_prompt_by_tokens,
)
from common.path_utils import get_path_relative_to_test_root, get_path_to_model


class TestBasicOnlineInference:
    """Test basic online inference functionality."""

    @pytest.mark.stage(1)
    @pytest.mark.feature("online_inference")
    @pytest.mark.parametrize("model_name", ["Qwen2.5-1.5B-Instruct"])
    @pytest.mark.parametrize("max_tokens", [200])
    @pytest.mark.parametrize("prompt_split_ratio", [0.5])
    def test_online_accuracy_hbm_ssd_mixed(
        self,
        model_name: str,
        max_tokens: int,
        prompt_split_ratio: float,
    ):
        """Test HBM + SSD mixed hit accuracy via online inference.

        This test verifies that the UCM system correctly handles:
        1. Phase 1: KV cache saved to SSD (with HBM prefix cache disabled)
        2. Phase 2: HBM + SSD mixed cache hit (with HBM prefix cache enabled)

        Test flow:
        1. Phase 1: Clear HBM, send full prompt -> KV cache saved to SSD
        2. Phase 2: Clear HBM, send partial prompt (warm HBM), then send full prompt
           (hits both HBM and SSD) -> verify mixed hit accuracy

        The test can run in two modes:
        - With VLLMServerManager (default): Automatically starts/stops vLLM server
        - External server: Set SERVER_URL environment variable to use existing server

        Args:
            model_name: Name of model (used to determine tokenizer path)
            max_tokens: Maximum tokens to generate
            prompt_split_ratio: Ratio to split prompt for Phase 2 (0.5 = split in half)
        """
        # Load configuration
        config_file = get_path_relative_to_test_root("config.yaml")
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Get server configuration from config or environment
        external_server_url = os.getenv("SERVER_URL")
        served_model_name = os.getenv(
            "MODEL_NAME", config.get("models", {}).get("served_model_name", model_name)
        )
        tokenizer_path = config.get("models", {}).get(
            "tokenizer_path", f"/home/models/{model_name}"
        )
        model_path = get_path_to_model(model_name, config)

        # Load test prompt and standard answers
        try:
            test_prompt, standard_answers = load_prompt_from_file(
                get_path_relative_to_test_root(
                    "suites/E2E/prompts/test_offline_inference.json"
                )
            )
            if not standard_answers:
                pytest.fail("No standard answers found in prompt.json")
        except Exception as e:
            pytest.fail(f"Failed to load prompt from prompt.json: {e}")

        print(f"Standard answers: {standard_answers}")

        # Initialize tokenizer for prompt splitting
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, use_chat_template=True
        )

        # Format prompt with chat template
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

        # Split prompt for Phase 2
        prompt_first_part, prompt_second_part = split_prompt_by_tokens(
            formatted_full_prompt, tokenizer, split_ratio=prompt_split_ratio
        )

        # Build UCM config for vLLM server
        ucm_config = {
            "storage_backends": "/tmp/ucm_cache",
        }

        def run_test_with_client(client: OnlineInferenceClient):
            """Run the actual test with the given client."""
            try:
                # ===== Phase 1: Save KV Cache to SSD (Baseline) =====
                print(
                    f"\n===== Phase 1: Save KV Cache to SSD And Load (Baseline) ====="
                )

                # Clear HBM to ensure clean state
                client.clear_hbm()
                print("HBM cache cleared for Phase 1")

                # Phase 1.1: Send full prompt to save to SSD
                phase1_messages = [
                    {
                        "role": "system",
                        "content": '先读问题，再根据下面的文章内容回答问题，不要进行分析，不要重复问题，用简短的语句给出答案。\n\n例如："全国美国文学研究会的第十八届年会在哪所大学举办的？"\n回答应该为："xx大学"。\n\n',
                    },
                    {"role": "user", "content": test_prompt},
                ]

                phase1_1_response = client.chat(
                    messages=phase1_messages,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    stream=True,
                )
                phase1_1_output = phase1_1_response.text
                print(f'Phase 1.1 output: "{phase1_1_output}"')

                # Phase 1.2: Send same prompt again to load from SSD
                # Note: In online inference, we rely on server-side caching
                phase1_2_response = client.chat(
                    messages=phase1_messages,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    stream=True,
                )
                phase1_2_output = phase1_2_response.text
                print(f'Phase 1.2 output: "{phase1_2_output}"')

                # ===== Phase 2: HBM + SSD Mixed Hit Test =====
                print(f"\n===== Phase 2: HBM + SSD Mixed Hit Test =====")

                # Clear HBM before Phase 2
                client.clear_hbm()
                print("HBM cache cleared for Phase 2")

                # Phase 2.1: Send partial prompt to warm HBM
                phase2_partial_messages = [
                    {
                        "role": "system",
                        "content": '先读问题，再根据下面的文章内容回答问题，不要进行分析，不要重复问题，用简短的语句给出答案。\n\n例如："全国美国文学研究会的第十八届年会在哪所大学举办的？"\n回答应该为："xx大学"。\n\n',
                    },
                    {"role": "user", "content": prompt_first_part},  # Only first part
                ]

                phase2_partial_response = client.chat(
                    messages=phase2_partial_messages,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    stream=True,
                )
                phase2_partial_output = phase2_partial_response.text
                print(
                    f"[INFO] Phase 2.1 output (partial prompt): {phase2_partial_output}"
                )

                # Phase 2.2: Send full prompt - should hit both HBM (prefix) and SSD (suffix)
                phase2_full_response = client.chat(
                    messages=phase1_messages,  # Full prompt
                    max_tokens=max_tokens,
                    temperature=0.0,
                    stream=True,
                )
                phase2_full_output = phase2_full_response.text
                print(f"[INFO] Phase 2.2 output (full prompt): {phase2_full_output}")

                # ===== Accuracy Test Results =====
                print(f"\n[INFO] ===== Accuracy Test Results =====")

                # Normalize text for comparison
                def normalize_text(text: str) -> str:
                    """Normalize text for comparison by replacing similar punctuation."""
                    text = text.replace("，", ",")
                    text = text.replace("。", ".")
                    text = text.replace("！", "!")
                    text = text.replace("？", "?")
                    text = text.replace("：", ":")
                    text = text.replace("；", ";")
                    return text.strip()

                def match_any_answer(output: str, answers: List[str]) -> bool:
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
                    print(
                        f"\n===== Phase 1: SSD Load Accuracy Test (Exact Match) ====="
                    )
                    print(
                        f"Incorrect answer in Phase 1.1 (SSD save) or Phase 1.2 (SSD load) output!"
                    )
                    print(f"Phase 1.1 output:\n{phase1_1_output}")
                    print(f"Phase 1.2 output:\n{phase1_2_output}")
                    print(f"Standard answers:\n{standard_answers}")
                    pytest.fail("SSD Load Accuracy Test Failed!")

                # Phase 2.2 should match standard answers
                phase2_correct = match_any_answer(phase2_full_output, standard_answers)
                if not phase2_correct:
                    print(
                        f"\n===== Phase 2: HBM + SSD Mixed Accuracy Test (Exact Match) ====="
                    )
                    print(f"Incorrect answer in Phase 2.2 (HBM + SSD mixed) output!")
                    print(f"Phase 2.2 output:\n{phase2_full_output}")
                    print(f"Standard answers:\n{standard_answers}")
                    pytest.fail("HBM + SSD Mixed Accuracy Test Failed!")

                print("\n===== All Tests Passed! =====")

            finally:
                client.close()

        # Check if using external server or starting our own
        if external_server_url:
            print(f"\n===== Using external server at {external_server_url} =====")
            print(f"Model: {served_model_name}")
            print(f"Full prompt length: {len(test_prompt)} chars")
            print(f"Max tokens: {max_tokens}")
            print(f"Temperature: 0.0 (deterministic)")
            print(f"Prompt split ratio: {prompt_split_ratio}")

            client = OnlineInferenceClient(
                server_url=external_server_url,
                model_name=served_model_name,
                tokenizer_path=tokenizer_path,
            )
            run_test_with_client(client)
        else:
            print(f"\n===== Starting vLLM server with VLLMServerManager =====")
            print(f"Model path: {model_path}")
            print(f"Port: 8000")
            print(f"UCM config: {ucm_config}")
            print(f"Enable prefix caching: False (Phase 1)")
            print(f"Full prompt length: {len(test_prompt)} chars")
            print(f"Max tokens: {max_tokens}")
            print(f"Temperature: 0.0 (deterministic)")
            print(f"Prompt split ratio: {prompt_split_ratio}")

            # Use VLLMServerManager to start vLLM server
            with VLLMServerManager(
                model_path=model_path,
                port=8000,
                ucm_config=ucm_config,
                enable_prefix_caching=False,  # Phase 1: SSD only
                max_model_len=12000,
                gpu_memory_utilization=0.9,
            ) as server:
                client = OnlineInferenceClient(
                    server_url=server.url,
                    model_name=served_model_name,
                    tokenizer_path=tokenizer_path,
                )
                run_test_with_client(client)
