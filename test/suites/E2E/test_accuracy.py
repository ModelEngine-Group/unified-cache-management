import dataclasses
import json

import pytest
from common.capture_utils import export_vars
from common.config_utils import config_utils as config_instance
from common.uc_eval.task import DocQaEvalTask
from common.uc_eval.utils.data_class import EvalConfig, ModelConfig


@pytest.fixture(scope="session")
def model_config() -> ModelConfig:
    cfg = config_instance.get_config("models") or {}
    field_name = [field.name for field in dataclasses.fields(ModelConfig)]
    kwargs = {k: v for k, v in cfg.items() if k in field_name and v is not None}
    if "payload" in kwargs and isinstance(kwargs["payload"], str):
        try:
            kwargs["payload"] = json.loads(kwargs["payload"])
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid payload JSON format: {e}")
    return ModelConfig(**kwargs)


_DOC_QA_BASE_CONFIG = {
    "data_type": "doc_qa",
    "dataset_file_path": "../../common/uc_eval/datasets/doc_qa/prompts.jsonl",
    "enable_prefix_cache": True,
    "parallel_num": 1,
    "benchmark_mode": "evaluate",
    "metrics": ["accuracy", "bootstrap-accuracy", "f1-score"],
    "eval_class": "common.uc_eval.utils.metric:Includes",
}

doc_qa_eval_cases = [
    pytest.param(
        EvalConfig(
            **{
                **_DOC_QA_BASE_CONFIG,
                "prompt_split_ratio": None,
                "enable_warmup": False,
                "round": 1,
            }
        ),
        id="doc-qa-full-prompt-warmup-evaluate",
    ),
    pytest.param(
        EvalConfig(
            **{**_DOC_QA_BASE_CONFIG, "prompt_split_ratio": 0.5, "enable_warmup": False}
        ),
        id="doc-qa-full-prompt-no-warmup-evaluate",
    ),
    pytest.param(
        EvalConfig(
            **{
                **_DOC_QA_BASE_CONFIG,
                "prompt_split_ratio": None,
                "enable_warmup": False,
                "round": 2,
            }
        ),
        id="doc-qa-half-prompt-warmup-evaluate",
    ),
]

test_configs = [
    pytest.param(
        {"max_tokens": 1024, "ignore_eos": True, "temperature": 0.7},
        False,  # enable_clear_hbm
        id="max_tokens_2048_clear_hbm_true",
    ),
]


@pytest.mark.feature("accu_test")
@pytest.mark.stage(2)
@pytest.mark.parametrize("eval_config", doc_qa_eval_cases)
@pytest.mark.parametrize("payload_updates,enable_clear_hbm", test_configs)
@export_vars
def test_doc_qa_perf(
    eval_config: EvalConfig,
    model_config: ModelConfig,
    payload_updates: dict,
    enable_clear_hbm: bool,
    request: pytest.FixtureRequest,
):
    file_save_path = config_instance.get_config("reports").get("base_dir")
    if isinstance(model_config.payload, str):
        model_config.payload = json.loads(model_config.payload)

    model_config.payload.update(payload_updates)

    if eval_config.prompt_split_ratio is None:
        model_config.enable_clear_hbm = True
    else:
        model_config.enable_clear_hbm = enable_clear_hbm

    task = DocQaEvalTask(model_config, eval_config, file_save_path)
    result = task.run()
    return {"_name": request.node.callspec.id, "_data": result}
