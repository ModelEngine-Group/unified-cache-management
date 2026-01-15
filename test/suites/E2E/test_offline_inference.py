import contextlib
import os
import time
from dataclasses import asdict
from typing import Dict, Any, Optional
import pytest

from transformers import AutoTokenizer

# Third Party
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs

from ucm.logger import init_logger

logger = init_logger(__name__)


@contextlib.contextmanager
def build_llm_without_uc(
    model: str,
    engine_args_override: Optional[Dict[str, Any]] = None,
):
    """Build LLM without UCM connector.
    
    Args:
        model: Model path or identifier
        engine_args_override: Optional overrides for EngineArgs
    """

    llm_args = {
        "model": model,
        "max_model_len": 5000,
        "gpu_memory_utilization": 0.8,
        "max_num_batched_tokens": 30000,
        "block_size": 128,
        "enforce_eager": True,
        "trust_remote_code": True,
        "enable_prefix_caching": False,
    }
    
    # Apply overrides if provided
    if engine_args_override:
        llm_args.update(engine_args_override)

    llm = LLM(**llm_args)
    try:
        yield llm
    finally:
        logger.info("LLM engine is exiting.")


@contextlib.contextmanager
def build_llm_with_uc(
    module_path: str,
    name: str,
    model: str,
    kv_connector_config: Optional[Dict[str, Any]] = None,
    engine_args_override: Optional[Dict[str, Any]] = None,
):
    """Build LLM with UCM connector.
    
    Args:
        module_path: Path to the UCM connector module
        name: Name of the connector
        model: Model path or identifier
        kv_connector_config: Optional custom KV connector config
        engine_args_override: Optional overrides for EngineArgs
    """
    # if kv_connector_config is None:
    #     kv_connector_config = {"UCM_CONFIG_FILE": "./ucm_config_example.yaml"}

    ktc = KVTransferConfig(
        kv_connector=name,
        kv_connector_module_path=module_path,
        kv_role="kv_both",
        kv_connector_extra_config=kv_connector_config,
    )

    llm_args = {
        "model": model,
        "kv_transfer_config": ktc,
        "max_model_len": 5000,
        "gpu_memory_utilization": 0.8,
        "max_num_batched_tokens": 30000,
        "block_size": 128,
        "enforce_eager": True,
        "trust_remote_code": True,
        "enable_prefix_caching": False,
    }
    
    # Apply overrides if provided
    if engine_args_override:
        llm_args.update(engine_args_override)

    llm = LLM(**llm_args)
    try:
        yield llm
    finally:
        logger.info("LLM engine is exiting.")


def get_output(
    llm: LLM,
    prompt: str,
    sampling_params: SamplingParams,
    req_str: str,
) -> str:
    """Generate and print output from LLM.
    
    Returns:
        Generated text
    """
    start = time.time()
    outputs = llm.generate(prompt, sampling_params)
    elapsed = time.time() - start
    
    print("-" * 50)
    generated_text = "".join(output.outputs[0].text for output in outputs)
    print(f"Generated text: {generated_text!r}")
    print(f"Generation took {elapsed:.2f} seconds, {req_str} request done.")
    print("-" * 50)
    
    return {
        "generated_text": generated_text,
        "elapsed_time": elapsed,
    }


# Fixture for model path
@pytest.fixture
def model_path():
    """Get model path from environment or use default."""
    return os.getenv("MODEL_PATH", "/home/models/DeepSeek-V2-Lite")


@pytest.fixture
def sampling_params():
    """Create standard sampling parameters for testing."""
    return SamplingParams(temperature=0, top_p=0.95, max_tokens=100)


class TestBasicOfflineInference:
    """Test basic offline inference functionality."""
    @pytest.mark.stage(1)
    def test_simple_offline_inference(self, model_path, sampling_params):
        """Test single inference request."""
        module_path = "ucm.integration.vllm.ucm_connector"
        name = "UCMConnector"
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_chat_template=True)
        
        messages = [
            {
                "role": "system",
                "content": "You are a highly specialized assistant whose mission is to faithfully reproduce English "
                "literary texts verbatim, without any deviation, paraphrasing, or omission. Your primary "
                "responsibility is accuracy: every word, every punctuation mark, and every line must "
                "appear exactly as in the original source. Core Principles: Verbatim Reproduction: If the "
                "user asks for a passage, you must output the text word-for-word. Do not alter spelling, "
                "punctuation, capitalization, or line breaks. Do not paraphrase, summarize, modernize, "
                "or “improve” the language. Consistency: The same input must always yield the same output. "
                "Do not generate alternative versions or interpretations. Clarity of Scope: Your role is "
                "not to explain, interpret, or critique. You are not a storyteller or commentator, "
                "but a faithful copyist of English literary and cultural texts. Recognizability: Because "
                "texts must be reproduced exactly, they will carry their own cultural recognition. You "
                "should not add labels, introductions, or explanations before or after the text. Coverage: "
                "You must handle passages from classic literature, poetry, speeches, or cultural texts. "
                "Regardless of tone—solemn, visionary, poetic, persuasive—you must preserve the original "
                "form, structure, and rhythm by reproducing it precisely. Success Criteria: A human reader "
                "should be able to compare your output directly with the original and find zero "
                "differences. The measure of success is absolute textual fidelity. Your function can be "
                "summarized as follows: verbatim reproduction only, no paraphrase, no commentary, "
                "no embellishment, no omission.",
            },
            {
                "role": "user",
                "content": "Please reproduce verbatim the opening sentence of the United States Declaration of "
                "Independence (1776), starting with 'When in the Course of human events' and continuing "
                "word-for-word without paraphrasing.",
            },
        ]
        
        result1 = None
        result2 = None
        # get result from pure vllm
        with build_llm_without_uc(model_path) as llm:
            prompts = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            result1 = get_output(llm, prompts, sampling_params, "without UCM")

        # get result from vllm with ucm
        with build_llm_with_uc(module_path, name, model_path) as llm:
            prompts = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            result2 = get_output(llm, prompts, sampling_params, "with UCM")
        
        assert result1["generated_text"] == result2["generated_text"]

