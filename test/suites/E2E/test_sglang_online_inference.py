import os
from pathlib import Path

import pytest
from common.common_inference_utils import (
    ensure_storage_dir,
    load_prompt_from_file,
    match_any_answer,
)
from common.llm_connection.LLMBase import LLMRequest
from common.llm_connection.openai_connector import OpenAIConn
from common.llm_connection.token_counter import HuggingFaceTokenizer
from common.online_inference_utils import SGLangServerManager
from common.path_utils import get_path_relative_to_test_root


def _request_document_answer(
    server_url: str,
    tokenizer_path: str,
    messages: list[dict[str, str]],
) -> str:
    client = OpenAIConn(
        base_url=server_url,
        tokenizer=HuggingFaceTokenizer(tokenizer_path),
        model="",
    )
    try:
        assert client.health_check()
        models = client.list_models()
        assert models, "SGLang did not expose a model through the OpenAI API"
        client.model = models[0]
        return client.chat(
            LLMRequest(messages=messages, max_tokens=200, temperature=0.0)
        ).text
    finally:
        client.close()


@pytest.mark.stage(1)
@pytest.mark.platform("npu")
@pytest.mark.feature("sglang_online_inference")
def test_sglang_online_accuracy_from_persisted_cache_a2():
    """Verify SGLang output before and after restoring UCM cache on Ascend A2."""
    import torch_npu

    assert torch_npu.npu.is_available(), "SGLang A2 E2E requires an Ascend NPU"
    device_name = torch_npu.npu.get_device_name(0)
    assert "910B" in device_name.upper() or "A2" in device_name.upper(), (
        f"Expected an Ascend A2/910B device, got {device_name!r}"
    )

    model_path = os.getenv(
        "SGLANG_E2E_MODEL_PATH", "/home/models/Qwen3-0.6B"
    )
    storage_path = os.getenv(
        "SGLANG_E2E_STORAGE_PATH", "/tmp/ucm_sglang_e2e_cache"
    )
    ensure_storage_dir(storage_path, clear_existing=True)

    prompt, standard_answers = load_prompt_from_file(
        get_path_relative_to_test_root(
            "suites/E2E/prompts/test_offline_inference.json"
        )
    )
    assert standard_answers, "The SGLang E2E prompt has no standard answers"
    messages = [
        {
            "role": "system",
            "content": (
                "先读问题，再根据下面的文章内容回答问题，不要进行分析，"
                "不要重复问题，用简短的语句给出答案。"
            ),
        },
        {"role": "user", "content": prompt},
    ]
    server_args = {
        "model_path": model_path,
        "storage_path": storage_path,
        "port": int(os.getenv("SGLANG_E2E_PORT", "30000")),
        "startup_timeout": float(os.getenv("SGLANG_E2E_STARTUP_TIMEOUT", "600")),
        "additional_args": ["--device", "npu", "--attention-backend", "ascend"],
    }

    with SGLangServerManager(**server_args) as server:
        first_output = _request_document_answer(server.url, model_path, messages)

    persisted_files = [path for path in Path(storage_path).rglob("*") if path.is_file()]
    assert persisted_files, (
        "SGLang completed inference but UCM persisted no cache files"
    )

    # Restarting SGLang clears its process-local radix/HBM cache. The second
    # request can therefore exercise the persistent UCM HiCache path.
    with SGLangServerManager(**server_args) as server:
        restored_output = _request_document_answer(server.url, model_path, messages)

    assert match_any_answer(first_output, standard_answers), (
        f"SGLang baseline output is incorrect: {first_output!r}"
    )
    assert match_any_answer(restored_output, standard_answers), (
        f"SGLang output after UCM cache restore is incorrect: {restored_output!r}"
    )
