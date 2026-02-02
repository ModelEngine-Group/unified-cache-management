import pytest
from common.capture_utils import export_vars
from common.config_utils import config_utils as config_instance
from common.uc_eval.task import DocQaEvalTask
from common.uc_eval.utils.data_class import EvalConfig, ModelConfig

doc_qa_eval_cases = [
    # longbench v2
    pytest.param(
        EvalConfig(
            data_type="doc_qa",
            dataset_file_path="datasets/doc_qa/demo_2.json",
            enable_prefix_cache=False,
            parallel_num=1,
            benchmark_mode="evaluate",
            metrics=["accuracy"],
            eval_class="common.uc_eval.utils.metric:MatchPatterns",
            select_data_class={"domain": ["Single-Document QA"]},
            test_name="longbench v2 and no prefix cache",
        ),
    ),
    # longbench
    pytest.param(
        EvalConfig(
            data_type="doc_qa",
            dataset_file_path="datasets/doc_qa/demo.jsonl",
            enable_prefix_cache=False,
            parallel_num=1,
            benchmark_mode="evaluate",
            metrics=["accuracy", "bootstrap-accuracy", "f1-score"],
            eval_class="common.uc_eval.utils.metric:FuzzyMatch",
            test_name="longbench and no prefix cache",
        ),
    ),
]


@pytest.mark.feature("qa_eval_test")
@pytest.mark.stage(2)
@pytest.mark.parametrize("eval_config", doc_qa_eval_cases)
@export_vars
def test_doc_qa_perf(eval_config: EvalConfig, model_config: ModelConfig):
    file_save_path = config_instance.get_config("reports").get("base_dir")
    task = DocQaEvalTask(model_config, eval_config, file_save_path)
    result = task.run()
    return {"_name": eval_config.test_name, "_data": result}
