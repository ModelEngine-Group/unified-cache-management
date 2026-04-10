import os

import evalscope
import pytest
from common.config_utils import config_utils as config_instance
from common.evalscope_utils import EvalScopeRunner

# ---------- Default Var Value ----------
DEFAULT_DATASET_ROOT = ""
# DEFAULT_TASK_LIST = ["aime24","aime25","aime26","ceval","cmmlu","gsm8k","humaneval","longbench_v2","mmlu","mmlu_pro","mmmlu"]
DEFAULT_TASK_LIST = ["gsm8k"]
DEFAULT_NEEDLE_MIN = 1000
DEFAULT_NEEDLE_MAX = 32000


def _build_general_task_config(
    model: str,
    api_url: str,
    api_key: str,
    datasets: list,
    dataset_root: str,
    output_dir: str,
) -> evalscope.config.TaskConfig:
    """Build a general evaluation task configuration"""
    dataset_args = {}
    if dataset_root:
        dataset_args = {ds: {"dataset_id": f"{dataset_root}/{ds}/"} for ds in datasets}
    return evalscope.config.TaskConfig(
        model=model,
        api_url=api_url,
        api_key=api_key,
        datasets=datasets,
        dataset_args=dataset_args,
        work_dir=output_dir,
        no_timestamp=False,
    )


def _build_needle_task_config(
    model: str,
    api_url: str,
    api_key: str,
    tokenizer_path: str,
    dataset_root: str,
    output_dir: str,
) -> evalscope.config.TaskConfig:
    """Constructing the task configuration for finding a needle in a haystack"""
    needle_min = int(os.getenv("SCOPE_NEEDLE_MIN", DEFAULT_NEEDLE_MIN))
    needle_max = int(os.getenv("SCOPE_NEEDLE_MAX", DEFAULT_NEEDLE_MAX))

    return evalscope.config.TaskConfig(
        model=model,
        api_url=api_url,
        api_key=api_key,
        eval_type="openai_api",
        datasets=["needle_haystack"],
        eval_batch_size=5,
        dataset_args={
            "needle_haystack": {
                "subset_list": ["chinese", "english"],
                "dataset_id": f"{dataset_root}/Needle-in-a-Haystack-Corpus/",
                "extra_params": {
                    "retrieval_question": "According to the text, what is the unique name of the award-winning purple vegetable grown on the floating farm?",
                    "needles": [
                        "\nIn a surprising turn of events recorded only in this document, the award for 'Most Exotic Produce of 2077' was given to a bioluminescent purple vegetable called the 'Voidlight Yam', which is cultivated exclusively on a floating aeroponic farm barge named 'The Drifting Mandrake'.\n"
                    ],
                    "context_lengths_min": needle_min,
                    "context_lengths_max": needle_max,
                    "context_lengths_num_intervals": 20,
                    "document_depth_percent_min": 0,
                    "document_depth_percent_max": 100,
                    "document_depth_percent_intervals": 10,
                    "tokenizer_path": tokenizer_path,
                    "show_score": True,
                },
            }
        },
        generation_config={"max_tokens": 512},
        # If necessary, other models can be selected as the judge model
        judge_model_args={
            "model_id": model,
            "api_url": api_url,
            "api_key": api_key,
        },
        work_dir=output_dir,
        no_timestamp=False,
    )


@pytest.mark.feature("eval_scope")
def test_eval_accuracy():
    env_list = os.getenv("SCOPE_TEST_LIST")
    if env_list:
        task_list = [x.strip() for x in env_list.split(",") if x.strip()]
    else:
        task_list = DEFAULT_TASK_LIST

    llm_cfg = config_instance.get_nested_config("llm_connection")
    base_url = llm_cfg.get("server_url", "").rstrip("/")
    model = llm_cfg.get("model")
    api_url = f"{base_url}/v1/chat/completions"
    api_key = "EMPTY_TOKEN"

    dataset_root = os.getenv("SCOPE_DATASET_ROOT") or DEFAULT_DATASET_ROOT
    dataset_root = dataset_root.rstrip("/") + "/" if dataset_root else ""

    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "results",
        "evalscope_outputs",
    )

    task_cfg = _build_general_task_config(
        model=model,
        api_url=api_url,
        api_key=api_key,
        datasets=task_list,
        dataset_root=dataset_root,
        output_dir=output_dir,
    )

    runner = EvalScopeRunner(output_dir)
    runner.run(task_cfg)
    runner.collect_results()

    assert True


@pytest.mark.feature("eval_scope")
def test_needle_task():
    """Haystack Needle Test (using oneself as the referee model)"""
    llm_cfg = config_instance.get_nested_config("llm_connection")
    base_url = llm_cfg.get("server_url", "").rstrip("/")
    model = llm_cfg.get("model")
    api_url = f"{base_url}/v1/chat/completions"
    api_key = "EMPTY_TOKEN"
    tokenizer_path = llm_cfg.get("tokenizer_path")

    dataset_root = os.getenv("SCOPE_DATASET_ROOT") or DEFAULT_DATASET_ROOT
    dataset_root = dataset_root.rstrip("/") + "/" if dataset_root else ""

    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "results",
        "evalscope_outputs",
    )

    task_cfg = _build_needle_task_config(
        model=model,
        api_url=api_url,
        api_key=api_key,
        tokenizer_path=tokenizer_path,
        dataset_root=dataset_root,
        output_dir=output_dir,
    )

    runner = EvalScopeRunner(output_dir)
    runner.run(task_cfg)
    runner.collect_results()

    assert True
