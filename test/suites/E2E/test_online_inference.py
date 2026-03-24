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
from common.online_inference_utils import hbm_ssd_mixed_test
from common.path_utils import get_path_relative_to_test_root, get_path_to_model


class TestBasicOnlineInference:
    """Test basic online inference functionality."""

    @pytest.mark.stage(1)
    @pytest.mark.gpu_mem(6000)
    @pytest.mark.feature("online_inference")
    @pytest.mark.parametrize("model_name", ["Qwen2.5-1.5B-Instruct"])
    @pytest.mark.parametrize("max_tokens", [200])
    @pytest.mark.parametrize("prompt_split_ratio", [0.5])
    @pytest.mark.parametrize("ucm_connector_name", ["UcmNfsStore", "UcmPipelineStore"])
    @pytest.mark.parametrize("use_layerwise", [True, False])
    @pytest.mark.parametrize("max_num_batched_tokens", [2047])
    def test_online_accuracy_hbm_ssd_mixed(
        self,
        model_name: str,
        max_tokens: int,
        prompt_split_ratio: float,
        ucm_connector_name: str,
        use_layerwise: bool,
        max_num_batched_tokens: int,
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
            prompt_split_ratio: Ratio to split prompt for Phase 2 (0.5 = split in)
            ucm_connector_name: Name of UCM store.
            use_layerwise: Whether to use layerwise mode.
            max_num_batched_tokens: Maximum number of batched tokens.
        """
        if use_layerwise is True and ucm_connector_name == "UcmNfsStore":
            pytest.skip("Skipping: UcmNfsStore does NOT support use_layerwise=True")

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
            "use_layerwise": (
                use_layerwise if ucm_connector_name != "UcmNfsStore" else False
            ),
            "ucm_connectors": [
                {
                    "ucm_connector_name": ucm_connector_name,
                    "ucm_connector_config": {
                        "store_pipeline": "Cache|Posix",
                        "storage_backends": ucm_storage_dir,
                        "use_direct": False,
                        "cache_buffer_capacity_gb": 32,
                    },
                }
            ],
            **(
                {"enable_event_sync": False}
                if ucm_connector_name == "UcmNfsStore"
                else {}
            ),
        }

        # Build vllm_server_startup_args
        vllm_server_startup_args = dict(
            model_path=model_path,
            port=8000,
            ucm_config=ucm_config,
            max_model_len=12000,
            max_num_batched_tokens=max_num_batched_tokens,
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
    @pytest.mark.gpu_mem(6000)
    @pytest.mark.gpu_count(2)
    @pytest.mark.feature("online_inference")
    @pytest.mark.parametrize("model_name", ["Qwen2.5-1.5B-Instruct"])
    @pytest.mark.parametrize("max_tokens", [200])
    @pytest.mark.parametrize("prompt_split_ratio", [0.5])
    @pytest.mark.parametrize("ucm_connector_name", ["UcmPipelineStore"])
    @pytest.mark.parametrize("use_layerwise", [True])
    @pytest.mark.parametrize("max_num_batched_tokens", [2047])
    def test_online_accuracy_hbm_ssd_mixed_pp(
        self,
        model_name: str,
        max_tokens: int,
        prompt_split_ratio: float,
        ucm_connector_name: str,
        use_layerwise: bool,
        max_num_batched_tokens: int,
    ):
        """Test HBM + SSD mixed hit accuracy via online inference with pipeline parallel.

        Mirrors test_offline_inference.py flow:
        1. Phase 1: Start vLLM (prefix caching OFF), send full prompt twice
           -> KV cache saved to SSD, then loaded from SSD
        2. Phase 2: Start vLLM (prefix caching ON), send partial prompt (warm HBM),
           then send full prompt (hits both HBM and SSD) -> verify accuracy

        Args:
            model_name: Name of model (used to determine tokenizer path)
            max_tokens: Maximum tokens to generate
            prompt_split_ratio: Ratio to split prompt for Phase 2 (0.5 = split in half)
            ucm_connector_name: Name of UCM store.
            use_layerwise: Whether to use layerwise mode.
            max_num_batched_tokens: Maximum number of batched tokens.
        """
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
            "use_layerwise": (
                use_layerwise if ucm_connector_name != "UcmNfsStore" else False
            ),
            "ucm_connectors": [
                {
                    "ucm_connector_name": ucm_connector_name,
                    "ucm_connector_config": {
                        "store_pipeline": "Cache|Posix",
                        "storage_backends": ucm_storage_dir,
                        "use_direct": False,
                        "cache_buffer_capacity_gb": 32,
                    },
                }
            ],
            **(
                {"enable_event_sync": False}
                if ucm_connector_name == "UcmNfsStore"
                else {}
            ),
        }

        # Build vllm_server_startup_args with pipeline parallel size
        vllm_server_startup_args = dict(
            model_path=model_path,
            port=8000,
            ucm_config=ucm_config,
            max_model_len=12000,
            max_num_batched_tokens=max_num_batched_tokens,
            served_model_name=served_model_name,
            pipeline_parallel_size=2,
        )

        hbm_ssd_mixed_test(
            model_name,
            tokenizer_path,
            max_tokens,
            prompt_split_ratio,
            ucm_config,
            vllm_server_startup_args,
        )
