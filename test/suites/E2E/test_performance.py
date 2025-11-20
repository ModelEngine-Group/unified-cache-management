import pytest
import dataclasses
from common.config_utils import config_utils as config_instance
from common.capture_utils import export_vars
from common.uc_eval.utils.data_class import ModelConfig, PerfConfig
from common.uc_eval.task import SyntheticPerfTask, MultiPerfTask, DocQaPerfTask


@pytest.fixture(scope="session")
def model_config() -> ModelConfig:
    cfg = config_instance.get_config("models") or {}
    field_name = [field.name for field in dataclasses.fields(ModelConfig)]
    kwargs = {k: v for k, v in cfg.items() if k in field_name and v is not None}
    return ModelConfig(**kwargs)


sync_perf_cases = [
    pytest.param(
        PerfConfig(
            data_type="synthetic",
            enable_prefix_cache=False,
            parallel_num=[1, 4, 8],
            prompt_tokens=[4000, 8000],
            output_tokens=[1000, 1000],
            benchmark_mode="default-perf",
        ),
        id="benchmark-complete-recalculate-default-perf",
    ),
    pytest.param(
        PerfConfig(
            data_type="synthetic",
            enable_prefix_cache=True,
            parallel_num=[1, 4, 8],
            prompt_tokens=[4000, 8000],
            output_tokens=[1000, 1000],
            prefix_cache_num=[0.8, 0.8],
            benchmark_mode="stable-perf",
        ),
        id="benchmark-prefix-cache-stable-perf",
    ),
]


@pytest.mark.feature("perf_test")
@pytest.mark.parametrize("perf_config", sync_perf_cases)
@export_vars
def test_sync_perf(perf_config: PerfConfig, model_config: ModelConfig, request: pytest.FixtureRequest):
    task = SyntheticPerfTask(model_config, perf_config)
    result = task.process()
    return {
        "_name": request.node.callspec.id,
        "_data": result
    }


multiturn_dialogue_perf_cases = [
    pytest.param(
        PerfConfig(
            data_type="multi_turn_dialogue",
            dataset_file_path="test/uc_eval /datasets/multi_turn_dialogues/multiturndialog.json",
            enable_prefix_cache=False,
            parallel_num=1,
            benchmark_mode="default-perf",
        ),
        id="multiturn-dialogue-complete-recalculate-default-perf",
    )
]


@pytest.mark.feature("perf_test")
@pytest.mark.parametrize("perf_config", multiturn_dialogue_perf_cases)
@export_vars
def test_multiturn_dialogue_perf(perf_config: PerfConfig, model_config: ModelConfig, request: pytest.FixtureRequest):
    task = MultiPerfTask(model_config, perf_config)
    result = task.process()
    return {
        "_name": request.node.callspec.id,
        "_data": result
    }

doc_qa_perf_cases = [
    pytest.param(
        PerfConfig(
            data_type="doc_qa",
            dataset_file_path="/home/externals/zhuzy/pyproject/uc-eval-new/datasets/doc_qa/demo.jsonl",
            enable_prefix_cache=False,
            parallel_num=1,
            benchmark_mode="default-perf",
        ),
        id="doc-qa-complete-recalculate-default-perf",
    )
]


@pytest.mark.feature("perf_test")
@pytest.mark.parametrize("perf_config", doc_qa_perf_cases)
@export_vars
def test_doc_qa_perf(perf_config: PerfConfig, model_config: ModelConfig, request: pytest.FixtureRequest):
    task = DocQaPerfTask(model_config, perf_config)
    result = task.process()
    return {
        "_name": request.node.callspec.id,
        "_data": result
    }
