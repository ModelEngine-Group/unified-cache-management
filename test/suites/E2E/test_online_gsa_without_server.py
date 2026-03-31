"""
Online GSA Sparse Inference Tests (Without Server Management).

This module contains tests for GSA (Generalized Sparse Attention) that connect
to an already-running vLLM server started by the Jenkins pipeline.

Unlike test_online_inference_sparse.py which starts its own server via
VLLMServerManager, these tests assume the server is pre-configured with
GSA sparse config and environment variables (ENABLE_SPARSE=1, VLLM_HASH_ATTENTION=1).

The tests verify GSA sparse attention accuracy for:
1. MLA-based model (DeepSeek-V2-Lite-Chat)
2. GQA-based model (Qwen3-4B)
"""

import pytest
import yaml
from common.common_inference_utils import (
    extract_answers,
    load_prompt_list_from_file,
    match_sparse_answer,
)
from common.llm_connection.LLMBase import LLMRequest
from common.llm_connection.openai_connector import OpenAIConn
from common.llm_connection.token_counter import HuggingFaceTokenizer
from common.online_inference_utils import batch_chat
from common.path_utils import get_path_relative_to_test_root


def _load_gsa_test_data():
    """Load prompts, answers, and build LLMRequest list for GSA tests."""
    test_prompts, standard_answers = load_prompt_list_from_file(
        get_path_relative_to_test_root(
            "suites/E2E/prompts/test_offline_gsaondevice_inference.json"
        )
    )
    if not standard_answers:
        pytest.fail("No standard answers found in prompt file")

    system_content = (
        "先读问题，再根据下面的文章内容回答问题，不要进行分析，不要重复问题，"
        "用简短的语句给出答案。\n\n"
        "例如：\u201c全国美国文学研究会的第十八届年会在哪所大学举办的？\u201d\n"
        "回答应该为：\u201cxx大学\u201d。\n\n"
    )

    return test_prompts, standard_answers, system_content


def _connect_to_server(config):
    """Connect to the pre-running server from config."""
    server_url = config.get("llm_connection", {}).get("server_url", "")
    tokenizer_path = config.get("llm_connection", {}).get("tokenizer_path", "")

    if not server_url:
        pytest.fail("server_url not found in config.yaml")

    client = OpenAIConn(
        base_url=server_url,
        tokenizer=HuggingFaceTokenizer(tokenizer_path),
        model="",
    )
    assert client.health_check()

    models = client.list_models()
    print(f"server models: {models}")
    if not models:
        pytest.fail("No models available on the server")
    client.model = models[0]
    print(f"Using model: {client.model}")

    return client


class TestGSASparseWithoutServer:
    """GSA sparse attention tests that connect to a pre-running server."""

    @pytest.mark.stage(1)
    @pytest.mark.feature("fvt_gsa_test")
    @pytest.mark.parametrize("max_tokens", [16])
    def test_online_gsa_mla(self, max_tokens: int):
        """Test GSA sparse attention for MLA-based model (DeepSeek-V2-Lite-Chat).

        Connects to a pre-running server configured with GSA sparse config.
        Loads prompts, sends them via batch_chat, verifies with match_sparse_answer.
        """
        config_file = get_path_relative_to_test_root("config.yaml")
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        test_prompts, standard_answers, system_content = _load_gsa_test_data()
        print(f"Standard answers: {standard_answers}")

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

        print(f"\n===== Online GSA MLA Sparse Test (Without Server) =====")

        client = _connect_to_server(config)

        responses = batch_chat(client, requests)
        outputs = [resp.text for resp in responses]

        print(f"GSA MLA online inference completed.")
        print(f'GSA MLA output: "{outputs}"')
        print(f'Standard answers: "{standard_answers}"')

        phase_sparse_correct = match_sparse_answer(outputs, standard_answers)

        if not phase_sparse_correct:
            print(f"Incorrect answer in GSA MLA online inference output!")
            print(f"GSA MLA output:\n{outputs}")
            print(f"Standard answers:\n{standard_answers}")
            pytest.fail("GSA MLA Online Test Failed!")

        client.close()
        print("GSA MLA online inference completed.")

    @pytest.mark.stage(1)
    @pytest.mark.feature("fvt_gsa_test")
    @pytest.mark.parametrize("max_tokens", [2048])
    def test_online_gsa_gqa(self, max_tokens: int):
        """Test GSA sparse attention for GQA-based model (Qwen3-4B).

        Connects to a pre-running server configured with GSA sparse config.
        Loads prompts, sends them via batch_chat, verifies with match_sparse_answer
        after extracting answers.
        """
        config_file = get_path_relative_to_test_root("config.yaml")
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        test_prompts, standard_answers, system_content = _load_gsa_test_data()
        print(f"Standard answers: {standard_answers}")

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

        print(f"\n===== Online GSA GQA Sparse Test (Without Server) =====")

        client = _connect_to_server(config)

        responses = batch_chat(client, requests)
        outputs = [resp.text for resp in responses]

        print(f"GSA GQA online inference completed.")
        print(f'GSA GQA output: "{outputs}"')
        print(f'Standard answers: "{standard_answers}"')

        # GQA model needs answer extraction (removes thinking tags)
        outputs = extract_answers(outputs)
        phase_sparse_correct = match_sparse_answer(outputs, standard_answers)

        if not phase_sparse_correct:
            print(f"Incorrect answer in GSA GQA online inference output!")
            print(f"GSA GQA output:\n{outputs}")
            print(f"Standard answers:\n{standard_answers}")
            pytest.fail("GSA GQA Online Test Failed!")

        client.close()
        print("GSA GQA online inference completed.")
