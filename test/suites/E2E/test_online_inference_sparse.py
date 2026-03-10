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

from typing import List

import pytest
import yaml
from common.llm_connection.LLMBase import LLMRequest
from common.llm_connection.openai_connector import OpenAIConn
from common.llm_connection.token_counter import HuggingFaceTokenizer
from common.offline_inference_utils import ensure_storage_dir
from common.online_inference_utils import (
    VLLMServerManager,
    load_prompt_from_file,
    split_prompt_by_tokens,
)
from common.path_utils import get_path_relative_to_test_root, get_path_to_model


class TestBasicOnlineInference:
    """Test basic online inference functionality."""

    @pytest.mark.stage(1)
    @pytest.mark.gpu_mem(6000)
    @pytest.mark.feature("online_inference_sparse")
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

        Mirrors test_offline_inference.py flow:
        1. Phase 1: Start vLLM (prefix caching OFF), send full prompt twice
           -> KV cache saved to SSD, then loaded from SSD
        2. Phase 2: Start vLLM (prefix caching ON), send partial prompt (warm HBM),
           then send full prompt (hits both HBM and SSD) -> verify accuracy

        Args:
            model_name: Name of model (used to determine tokenizer path)
            max_tokens: Maximum tokens to generate
            prompt_split_ratio: Ratio to split prompt for Phase 2 (0.5 = split in half)
        """
        # Load configuration
        config_file = get_path_relative_to_test_root("config.yaml")
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        ucm_storage_dir = "/tmp/ucm_cache"
        ensure_storage_dir(ucm_storage_dir, clear_existing=True)

        served_model_name = model_name
        tokenizer_path = f"/home/models/{model_name}"
        model_path = get_path_to_model(model_name, config)

        # Load test prompt and standard answers
        test_prompt, standard_answers = load_prompt_from_file(
            get_path_relative_to_test_root(
                "suites/E2E/prompts/test_offline_inference.json"
            )
        )
        if not standard_answers:
            pytest.fail("No standard answers found in prompt.json")

        print(f"Standard answers: {standard_answers}")

        # Initialize tokenizer for prompt splitting
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, use_chat_template=True
        )

        # Format prompt with chat template
        system_content = "先读问题，再根据下面的文章内容回答问题，不要进行分析，不要重复问题，用简短的语句给出答案。\n\n例如：\u201c全国美国文学研究会的第十八届年会在哪所大学举办的？\u201d\n回答应该为：\u201cxx大学\u201d。\n\n"
        try:
            messages = [
                {"role": "system", "content": system_content},
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
        prompt_first_part, _ = split_prompt_by_tokens(
            formatted_full_prompt, tokenizer, split_ratio=prompt_split_ratio
        )

        # Build UCM config
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
        }

        # Common VLLMServerManager kwargs
        server_common_kwargs = dict(
            model_path=model_path,
            port=8000,
            ucm_config=ucm_config,
            max_model_len=12000,
            served_model_name=served_model_name,
        )

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
        print(f"Model: {model_path}")
        print(f"Full prompt length: {len(test_prompt)} chars")
        print(f"Max tokens: {max_tokens}")
        print(f"Prompt split ratio: {prompt_split_ratio}")

        # ===== Phase 1: Disable HBM PC, save KV cache to SSD and load (baseline) =====
        print(f"\n===== Phase 1: Save KV Cache to SSD And Load (Baseline) =====")
        print(f"Starting vLLM server with enable_prefix_caching=False")

        with VLLMServerManager(
            **server_common_kwargs,
            enable_prefix_caching=False,
        ) as server:
            client = OpenAIConn(
                base_url=server.url,
                tokenizer=HuggingFaceTokenizer(tokenizer_path),
                model=served_model_name,
            )
            assert client.health_check()

            print(f"server models: {client.list_models()}")

            # Phase 1.1: Send full prompt -> KV cache saved to SSD
            phase1_1_output = client.chat(
                LLMRequest(
                    messages=phase1_messages, max_tokens=max_tokens, temperature=0.0
                )
            ).text
            print(f'Phase 1.1 output: "{phase1_1_output}"')

            # Phase 1.2: Send same prompt again -> KV cache loaded from SSD
            phase1_2_output = client.chat(
                LLMRequest(
                    messages=phase1_messages, max_tokens=max_tokens, temperature=0.0
                )
            ).text
            print(f'Phase 1.2 output: "{phase1_2_output}"')
            client.close()

        print("Phase 1 vLLM server stopped.")

        # ===== Phase 2: Enable HBM PC, test HBM + SSD mixed hit =====
        print(f"\n===== Phase 2: HBM + SSD Mixed Hit Test =====")
        print(f"Starting vLLM server with enable_prefix_caching=True")

        with VLLMServerManager(
            **server_common_kwargs,
            enable_prefix_caching=True,
        ) as server:
            client = OpenAIConn(
                base_url=server.url,
                tokenizer=HuggingFaceTokenizer(tokenizer_path),
                model=served_model_name,
            )
            assert client.health_check()

            print(f"server models: {client.list_models()}")

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
                LLMRequest(
                    messages=phase1_messages, max_tokens=max_tokens, temperature=0.0
                )
            ).text
            print(f"[INFO] Phase 2.2 output (full prompt): {phase2_full_output}")
            client.close()

        print("Phase 2 vLLM server stopped.")

        # ===== Accuracy Test Results =====
        print(f"\n[INFO] ===== Accuracy Test Results =====")

        def normalize_text(text: str) -> str:
            text = text.replace("\uff0c", ",")
            text = text.replace("\u3002", ".")
            text = text.replace("\uff01", "!")
            text = text.replace("\uff1f", "?")
            text = text.replace("\uff1a", ":")
            text = text.replace("\uff1b", ";")
            return text.strip()

        def match_any_answer(output: str, answers: List[str]) -> bool:
            for answer in answers:
                if normalize_text(output) == normalize_text(answer):
                    return True
            return False

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

    @pytest.mark.skip(reason="refine this code and re-enable later")
    @pytest.mark.stage(1)
    @pytest.mark.gpu_mem(10000)
    @pytest.mark.feature("online_inference_sparse")
    @pytest.mark.parametrize("model_name", ["Qwen3-4B"])
    @pytest.mark.parametrize("max_tokens", [200])
    def test_online_gsa(
        self,
        model_name: str,
        max_tokens: int,
    ):
        """Test GSA sparse attention via online inference.

        Mirrors test_offline_inference_sparse.py::test_offline_gsa.
        Starts vLLM with GSA sparse config, sends full prompt twice,
        verifies SSD save/load works.
        """
        config_file = get_path_relative_to_test_root("config.yaml")
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        ucm_storage_dir = "/tmp/ucm_cache"
        ensure_storage_dir(ucm_storage_dir, clear_existing=True)

        served_model_name = model_name
        tokenizer_path = f"/home/models/{model_name}"
        model_path = get_path_to_model(model_name, config)

        test_prompt, _ = load_prompt_from_file(
            get_path_relative_to_test_root(
                "suites/E2E/prompts/test_offline_inference.json"
            )
        )

        system_content = "先读问题，再根据下面的文章内容回答问题，不要进行分析，不要重复问题，用简短的语句给出答案。\n\n例如：\u201c全国美国文学研究会的第十八届年会在哪所大学举办的？\u201d\n回答应该为：\u201cxx大学\u201d。\n\n"

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
            "ucm_sparse_config": {"GSAOnDevice": {}},
        }

        phase1_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": test_prompt},
        ]

        print(f"\n===== Online GSA Sparse Test =====")
        print(f"Model: {model_path}")
        print(f"Starting vLLM server with GSA sparse config")

        with VLLMServerManager(
            model_path=model_path,
            port=8000,
            ucm_config=ucm_config,
            max_model_len=12000,
            served_model_name=served_model_name,
            enable_prefix_caching=False,
        ) as server:
            client = OpenAIConn(
                base_url=server.url,
                tokenizer=HuggingFaceTokenizer(tokenizer_path),
                model=served_model_name,
            )
            assert client.health_check()

            print(f"server models: {client.list_models()}")

            # Phase 1.1: SSD save
            phase1_1_output = client.chat(
                LLMRequest(
                    messages=phase1_messages, max_tokens=max_tokens, temperature=0.0
                )
            ).text
            print(f'Phase 1.1 output: "{phase1_1_output}"')

            # Phase 1.2: SSD load
            phase1_2_output = client.chat(
                LLMRequest(
                    messages=phase1_messages, max_tokens=max_tokens, temperature=0.0
                )
            ).text
            print(f'Phase 1.2 output: "{phase1_2_output}"')
            client.close()

        print("GSA inference completed.")

    @pytest.mark.skip(reason="refine this code and re-enable later")
    @pytest.mark.stage(1)
    @pytest.mark.gpu_mem(6000)
    @pytest.mark.feature("online_inference_sparse")
    @pytest.mark.parametrize("model_name", ["Qwen2.5-1.5B-Instruct"])
    @pytest.mark.parametrize("max_tokens", [200])
    def test_online_esa(
        self,
        model_name: str,
        max_tokens: int,
    ):
        """Test ESA sparse attention via online inference.

        Mirrors test_offline_inference_sparse.py::test_offline_esa.
        Starts vLLM with ESA sparse config, sends full prompt twice,
        verifies SSD save/load works.
        """
        config_file = get_path_relative_to_test_root("config.yaml")
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        ucm_storage_dir = "/tmp/ucm_cache"
        ensure_storage_dir(ucm_storage_dir, clear_existing=True)

        served_model_name = model_name
        tokenizer_path = f"/home/models/{model_name}"
        model_path = get_path_to_model(model_name, config)

        test_prompt, _ = load_prompt_from_file(
            get_path_relative_to_test_root(
                "suites/E2E/prompts/test_offline_inference.json"
            )
        )

        system_content = "先读问题，再根据下面的文章内容回答问题，不要进行分析，不要重复问题，用简短的语句给出答案。\n\n例如：\u201c全国美国文学研究会的第十八届年会在哪所大学举办的？\u201d\n回答应该为：\u201cxx大学\u201d。\n\n"

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
            "ucm_sparse_config": {
                "ESA": {
                    "init_window_sz": 1,
                    "local_window_sz": 2,
                    "min_blocks": 4,
                    "sparse_ratio": 0.3,
                    "retrieval_stride": 5,
                }
            },
        }

        phase1_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": test_prompt},
        ]

        print(f"\n===== Online ESA Sparse Test =====")
        print(f"Model: {model_path}")
        print(f"Starting vLLM server with ESA sparse config")

        with VLLMServerManager(
            model_path=model_path,
            port=8000,
            ucm_config=ucm_config,
            max_model_len=12000,
            served_model_name=served_model_name,
            enable_prefix_caching=False,
        ) as server:
            client = OpenAIConn(
                base_url=server.url,
                tokenizer=HuggingFaceTokenizer(tokenizer_path),
                model=served_model_name,
            )
            assert client.health_check()

            print(f"server models: {client.list_models()}")

            # Phase 1.1: SSD save
            phase1_1_output = client.chat(
                LLMRequest(
                    messages=phase1_messages, max_tokens=max_tokens, temperature=0.0
                )
            ).text
            print(f'Phase 1.1 output: "{phase1_1_output}"')

            # Phase 1.2: SSD load
            phase1_2_output = client.chat(
                LLMRequest(
                    messages=phase1_messages, max_tokens=max_tokens, temperature=0.0
                )
            ).text
            print(f'Phase 1.2 output: "{phase1_2_output}"')
            client.close()

        print("ESA inference completed.")
