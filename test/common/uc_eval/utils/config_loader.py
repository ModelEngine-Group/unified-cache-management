import dataclasses
import functools
import importlib
import json
from typing import Any, Optional, Tuple

from common.uc_eval.utils.benchmark import (
    BenchmarkBase,
    EvaluatorBenchmark,
    PerformanceBenchmark,
)
from common.uc_eval.utils.client import BaseClient, DocQaClient, MultiDialogClient
from common.uc_eval.utils.data_class import (
    BenchmarkModeType,
    DatasetType,
    EvalConfig,
    ModelConfig,
    PerfConfig,
)
from common.uc_eval.utils.dataloader import (
    BaseDataset,
    DocQADataset,
    MultiTurnDialogueDataset,
    SyntheticDataset,
)
from common.uc_eval.utils.utils import get_logger

logger = get_logger()


def make_object(object_ref: str, *args: Any, **kwargs: Any) -> Any:
    """create object based on class name"""
    modname, qualname_separator, qualname = object_ref.partition(":")
    obj = importlib.import_module(modname)
    if qualname_separator:
        for attr in qualname.split("."):
            obj = getattr(obj, attr)
    return functools.partial(obj, *args, **kwargs)


class ConfigLoader:
    def __init__(
        self,
        model_config: ModelConfig,
        perf_config: PerfConfig = None,
        eval_config: EvalConfig = None,
    ):

        self.model_config = model_config
        self.perf_config = perf_config
        self.eval_config = eval_config
        self._valid_config()

    def _valid_config(self) -> bool:
        logger.info("Validating config...")
        if self.perf_config is not None and self.eval_config is not None:
            raise ValueError(
                "perf_config and eval_config are mutually exclusive – one must be None."
            )
        if self.perf_config is None and self.eval_config is None:
            raise ValueError(
                "At least one of perf_config or eval_config must be provided."
            )

        result = (
            self._valid_model_config() and self._valid_perf_config()
            if self.perf_config is not None
            else self._valid_eval_config()
        )
        logger.info("Complete validation...")
        return result

    def _valid_model_config(self) -> bool:
        payload = self.model_config.payload
        if isinstance(payload, str):
            try:
                self.model_config.payload = json.loads(payload)
            except Exception as e:
                raise ValueError(f"Invalid payload JSON format: {e}")

        empty_fields = []
        field_names = [field.name for field in dataclasses.fields(ModelConfig)]
        for field_name in field_names:
            value = getattr(self.model_config, field_name)
            if value is None or (isinstance(value, str) and not value.strip()):
                empty_fields.append(field_name)

        if empty_fields:
            raise ValueError(
                f"The following model config fields can't be empty: {', '.join(empty_fields)}"
            )

        return True

    def _valid_perf_config(self) -> bool:
        data_type = self.perf_config.data_type
        benchmark_mode = self.perf_config.benchmark_mode
        if benchmark_mode not in [
            BenchmarkModeType.DEFAULT_PERF,
            BenchmarkModeType.STABLE_PREF,
        ]:
            raise ValueError(
                f"Invalid benchmark mode: {benchmark_mode}. Valid modes are: {BenchmarkModeType.DEFAULT_PERF}, {BenchmarkModeType.STABLE_PREF}"
            )
        prompt_fields = ["prompt_tokens", "output_tokens"] + (
            ["prefix_cache_num"] if self.perf_config.enable_prefix_cache else []
        )
        if data_type == DatasetType.SYNTHETIC:
            invalid_fields = []
            for field in prompt_fields:
                value = getattr(self.perf_config, field)
                if not isinstance(value, list) or not value:
                    invalid_fields.append(field)
            if invalid_fields:
                raise ValueError(
                    f"The following dataset config fields must be non-empty list for synthetic data: {', '.join(invalid_fields)}"
                )

            length = {
                field: len(getattr(self.perf_config, field)) for field in prompt_fields
            }
            if len(set(length.values())) > 1:
                raise ValueError(
                    f"The following dataset config is not matched: {', '.join(length.keys())}"
                )
        else:
            if self.perf_config.dataset_file_path is None:
                raise ValueError(
                    f"dataset_file_path is required for {data_type} data type"
                )
            if not isinstance(self.perf_config.parallel_num, int):
                raise TypeError(
                    f"parallel_num must be an integer for {data_type} data type"
                )
            not_empty_fields = [
                field for field in prompt_fields if getattr(self.perf_config, field)
            ]
            if not_empty_fields:
                raise ValueError(
                    f"The following dataset fields should be None for {data_type} data type: {not_empty_fields}"
                )

        return True

    def _valid_eval_config(self) -> bool:
        data_type = self.perf_config.data_type
        dataset_file_path = self.perf_config.dataset_file_path
        benchmark_mode = self.perf_config.benchmark_mode
        if benchmark_mode != BenchmarkModeType.EVAL:
            raise ValueError(
                f"Invalid benchmark mode: {benchmark_mode}. Valid modes are: {BenchmarkModeType.EVAL}"
            )
        if data_type == DatasetType.SYNTHETIC or dataset_file_path is None:
            raise ValueError(
                f"Invalid dataset type: {data_type} or Invalid dataset file path: {dataset_file_path}"
            )
        # TODO: add more validations
        return True


class TaskFactory:
    _dataset: BaseDataset = {
        DatasetType.SYNTHETIC: SyntheticDataset,
        DatasetType.MULTI_DIALOGUE: MultiTurnDialogueDataset,
        DatasetType.DOC_QA: DocQADataset,
    }
    _client: BaseClient = {
        DatasetType.SYNTHETIC: BaseClient,
        DatasetType.MULTI_DIALOGUE: MultiDialogClient,
        DatasetType.DOC_QA: DocQaClient,
    }
    _benchmark: BenchmarkBase = {
        BenchmarkModeType.EVAL: EvaluatorBenchmark,
        BenchmarkModeType.STABLE_PREF: PerformanceBenchmark,
        BenchmarkModeType.DEFAULT_PERF: PerformanceBenchmark,
    }

    @classmethod
    def create_task(
        cls,
        model_config: ModelConfig,
        perf_config: Optional[PerfConfig],
        eval_config: Optional[EvalConfig],
    ) -> Tuple[BaseDataset, BaseClient, BenchmarkBase]:
        data_type = (perf_config or eval_config).data_type
        tokenizer_path = model_config.tokenizer_path
        enable_prefix_cache = perf_config.enable_prefix_cache
        benchmark_mode = (perf_config or eval_config).benchmark_mode
        stable = benchmark_mode == BenchmarkModeType.STABLE_PREF
        if benchmark_mode in [
            BenchmarkModeType.STABLE_PREF,
            BenchmarkModeType.DEFAULT_PERF,
        ]:
            stream = True
        return (
            cls._dataset[data_type](tokenizer_path),
            cls._client[data_type](model_config, stream, enable_prefix_cache),
            cls._benchmark[benchmark_mode](stable if perf_config else eval_config),
        )
