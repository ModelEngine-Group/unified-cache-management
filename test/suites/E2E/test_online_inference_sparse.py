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
    ensure_storage_dir,
    extract_answers,
    load_prompt_from_file,
    load_prompt_list_from_file,
    match_any_answer,
    match_sparse_answer,
    normalize_text,
    split_prompt_by_tokens,
)
from common.llm_connection.LLMBase import LLMRequest
from common.llm_connection.openai_connector import OpenAIConn
from common.llm_connection.token_counter import HuggingFaceTokenizer
from common.online_inference_utils import (
    VLLMServerManager,
    batch_chat,
    hbm_ssd_mixed_test,
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
        os.environ["ENABLE_SPARSE"] = "0"
        os.environ["VLLM_HASH_ATTENTION"] = "0"

        # Load configuration
        config_file = get_path_relative_to_test_root("config.yaml")
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        ucm_storage_dir = "/tmp/ucm_cache"
        served_model_name = model_name
        tokenizer_path = f"/home/models/{model_name}"
        model_path = get_path_to_model(model_name, config)

        # Build UCM config
        ucm_config = {
            "ucm_connectors": [
                {
                    "ucm_connector_name": "UcmPipelineStore",
                    "ucm_connector_config": {
                        "store_pipeline": "Cache|Posix",
                        "storage_backends": ucm_storage_dir,
                        "use_direct": False,
                        "cache_buffer_capacity_gb": 32,
                    },
                }
            ],
        }

        # Build vllm_server_server_startup_args
        vllm_server_startup_args = dict(
            model_path=model_path,
            port=8000,
            ucm_config=ucm_config,
            max_model_len=12000,
            max_num_batched_tokens=2047,
            served_model_name=served_model_name,
        )

        hbm_ssd_mixed_test(
            model_name,
            tokenizer_path,
            max_tokens,
            prompt_split_ratio,
            ucm_config,
            vllm_server_startup_args,
        )

    @pytest.mark.stage(1)
    @pytest.mark.gpu_mem(70000)
    @pytest.mark.feature("online_inference_sparse")
    @pytest.mark.parametrize("model_name", ["DeepSeek-V2-Lite-Chat"])
    @pytest.mark.parametrize("max_tokens", [16])
    def test_online_gsa_mla(
        self,
        model_name: str,
        max_tokens: int,
    ):
        """Test GSA sparse attention via online inference for MLA-based model.

        Mirrors test_offline_inference_sparse.py::test_offline_gsa_mla.
        Loads prompts from test_offline_gsaondevice_inference.json,
        sends them in parallel using batch_chat, verifies using match_sparse_answer.
        """
        os.environ["ENABLE_SPARSE"] = "1"
        os.environ["VLLM_HASH_ATTENTION"] = "1"

        config_file = get_path_relative_to_test_root("config.yaml")
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        model_path = get_path_to_model(model_name, config)
        tokenizer_path = f"/home/models/{model_name}"
        served_model_name = model_name

        # Load prompts and answers
        try:
            test_prompts, standard_answers = load_prompt_list_from_file(
                get_path_relative_to_test_root(
                    "suites/E2E/prompts/test_offline_gsaondevice_inference.json"
                )
            )
            if not standard_answers:
                pytest.fail(f"No standard answers found in prompt.json")
        except Exception as e:
            pytest.fail(f"Failed to load prompt from prompt.json: {e}")

        print(f"Standard answers: {standard_answers}")

        tokenizer = HuggingFaceTokenizer(tokenizer_path)

        system_content = "先读问题，再根据下面的文章内容回答问题，不要进行分析，不要重复问题，用简短的语句给出答案。\n\n例如：\u201c全国美国文学研究会的第十八届年会在哪所大学举办的？\u201d\n回答应该为：\u201cxx大学\u201d。\n\n"

        # Create LLMRequest list
        requests = [
            LLMRequest(
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            for prompt in test_prompts
        ]

        # UCM config with UcmPipelineStore
        ucm_config = {
            "ucm_connectors": [
                {
                    "ucm_connector_name": "UcmPipelineStore",
                    "ucm_connector_config": {
                        "store_pipeline": "Empty",
                        "share_buffer_enable": True,
                    },
                }
            ],
            "ucm_sparse_config": {"GSAOnDevice": {}},
        }

        print(f"\n===== Online GSA MLA Sparse Test =====")
        print(f"Model: {model_path}")
        print(f"Starting vLLM server with GSA sparse config")

        with VLLMServerManager(
            model_path=model_path,
            port=8000,
            ucm_config=ucm_config,
            max_model_len=70000,
            served_model_name=served_model_name,
            enable_prefix_caching=False,
        ) as server:
            client = OpenAIConn(
                base_url=server.url,
                tokenizer=tokenizer,
                model=served_model_name,
            )
            assert client.health_check()

            print(f"server models: {client.list_models()}")

            # Send requests in parallel using batch_chat
            responses = batch_chat(client, requests)
            outputs = [resp.text for resp in responses]

            print(f"GSA MLA online inference completed.")
            print(f'GSA MLA output: "{outputs}"')
            print(f'Standard answers: "{standard_answers}"')

            # Verify
            phase_sparse_correct = match_sparse_answer(outputs, standard_answers)

            if not phase_sparse_correct:
                print(f"Incorrect answer in GSA MLA online inference output!")
                print(f"GSA MLA output:\n{outputs}")
                print(f"Standard answers:\n{standard_answers}")
                pytest.fail("GSA MLA Online Test Failed!")

            client.close()

        print("GSA MLA online inference completed.")

    @pytest.mark.stage(1)
    @pytest.mark.gpu_mem(30000)
    @pytest.mark.feature("online_inference_sparse")
    @pytest.mark.parametrize("model_name", ["Qwen3-4B"])
    @pytest.mark.parametrize("max_tokens", [2048])
    def test_online_gsa_gqa(
        self,
        model_name: str,
        max_tokens: int,
    ):
        """Test GSA sparse attention via online inference.

        Mirrors test_offline_inference_sparse.py::test_offline_gsa_gqa.
        Loads prompts from test_offline_gsaondevice_inference.json,
        sends them in parallel using batch_chat, verifies using match_sparse_answer.
        """
        os.environ["ENABLE_SPARSE"] = "1"
        os.environ["VLLM_HASH_ATTENTION"] = "1"

        config_file = get_path_relative_to_test_root("config.yaml")
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        model_path = get_path_to_model(model_name, config)
        tokenizer_path = f"/home/models/{model_name}"
        served_model_name = model_name

        # Load prompts and answers
        try:
            test_prompts, standard_answers = load_prompt_list_from_file(
                get_path_relative_to_test_root(
                    "suites/E2E/prompts/test_offline_gsaondevice_inference.json"
                )
            )
            if not standard_answers:
                pytest.fail(f"No standard answers found in prompt.json")
        except Exception as e:
            pytest.fail(f"Failed to load prompt from prompt.json: {e}")

        print(f"Standard answers: {standard_answers}")

        tokenizer = HuggingFaceTokenizer(tokenizer_path)

        system_content = "先读问题，再根据下面的文章内容回答问题，不要进行分析，不要重复问题，用简短的语句给出答案。\n\n例如：\u201c全国美国文学研究会的第十八届年会在哪所大学举办的？\u201d\n回答应该为：\u201cxx大学\u201d。\n\n"

        # Create LLMRequest list
        requests = [
            LLMRequest(
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            for prompt in test_prompts
        ]

        # UCM config with UcmPipelineStore
        ucm_config = {
            "ucm_connectors": [
                {
                    "ucm_connector_name": "UcmPipelineStore",
                    "ucm_connector_config": {
                        "store_pipeline": "Empty",
                        "share_buffer_enable": True,
                    },
                }
            ],
            "ucm_sparse_config": {"GSAOnDevice": {}},
        }

        print(f"\n===== Online GSA Sparse Test =====")
        print(f"Model: {model_path}")
        print(f"Starting vLLM server with GSA sparse config")

        with VLLMServerManager(
            model_path=model_path,
            port=8000,
            ucm_config=ucm_config,
            max_model_len=30000,
            served_model_name=served_model_name,
            enable_prefix_caching=False,
        ) as server:
            client = OpenAIConn(
                base_url=server.url,
                tokenizer=tokenizer,
                model=served_model_name,
            )
            assert client.health_check()

            print(f"server models: {client.list_models()}")

            # Send requests in parallel using batch_chat
            responses = batch_chat(client, requests)
            outputs = [resp.text for resp in responses]

            print(f"GSA online inference completed.")
            print(f'GSA output: "{outputs}"')
            print(f'Standard answers: "{standard_answers}"')

            # Extract answers and verify
            outputs = extract_answers(outputs)
            phase_sparse_correct = match_sparse_answer(outputs, standard_answers)

            if not phase_sparse_correct:
                print(f"Incorrect answer in GSA online inference output!")
                print(f"GSA output:\n{outputs}")
                print(f"Standard answers:\n{standard_answers}")
                pytest.fail("GSA Online Test Failed!")

            client.close()

        print("GSA online inference completed.")
