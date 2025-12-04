import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

from common.uc_eval.utils.config_loader import ConfigLoader, TaskFactory
from common.uc_eval.utils.data_class import (
    BenchmarkModeType,
    EvalConfig,
    LatencyStatistics,
    ModelConfig,
    MultiTurnDialogRecord,
    PerfConfig,
    RequestRecord,
    SynthericParams,
)
from common.uc_eval.utils.utils import FileUtil, PathUtil, get_current_time, get_logger

MS_SCALE = 1000
BAD_COMPLETION_TOKENS_THR = 20
logger = get_logger()
PERF_CSV_HEADER = [
    "Test Time",
    "Total Cases",
    "Parallel Num",
    "Prefix Cache",
    "Total Latency(ms)",
    "E2E TPS(tokens/s)",
    "Per Request TPS(tokens/s)",
    "TTFT P50(ms)",
    "TTFT P90(ms)",
    "TTFT P99(ms)",
    "MAX TTFT(ms)",
    "Average TTFT(ms)",
    "TBT P50(ms)",
    "TBT P90(ms)",
    "TBT P99(ms)",
    "TBT MAX(ms)",
    "TBT Average(ms)",
]

SYNC_PERF_CSV_HEADER = [
    "Test Time",
    "Parallel Num",
    "Input Tokens",
    "Output Tokens",
    "Parallel Num",
    "Prefix Cache",
    "Hit Rate",
    "Total Latency(ms)",
    "E2E TPS(tokens/s)",
    "Per Request TPS(tokens/s)",
    "TTFT P50(ms)",
    "TTFT P90(ms)",
    "TTFT P99(ms)",
    "MAX TTFT(ms)",
    "Average TTFT(ms)",
    "TBT P50(ms)",
    "TBT P90(ms)",
    "TBT P99(ms)",
    "TBT MAX(ms)",
    "TBT Average(ms)",
]

CASE_PERF_CSV_HEADER = [
    "Test Time",
    "Prefix Cache",
    "Total Cases",
    "Current Case",
    "Case Name",
    "Input Tokens",
    "Output Tokens",
    "Latency(ms)",
    "TTFT(ms)",
    "TBT(ms)",
]

CASE_EVAL_CSV_HEADER = [
    "Test Time",
    "Prefix Cache",
    "Total Cases",
    "Current Case",
    "Case Name",
    "Input Tokens",
    "Output Tokens",
    "Input Text",
    "Question",
    "Expected Output",
    "Real Output",
    "Is Match",
    "Match Class",
]


class BaseTask(ABC):
    def __init__(
        self,
        model_config: ModelConfig,
        perf_config: PerfConfig = None,
        eval_config: EvalConfig = None,
        save_to_excel: bool = True,
        file_save_path: str = None,
    ):
        ConfigLoader(model_config, perf_config, eval_config)
        self.current_time = get_current_time()
        self.model_config = model_config
        self.perf_config = perf_config
        self.eval_config = eval_config
        common_config = perf_config if perf_config else eval_config
        self.data_type = common_config.data_type
        self.parallel_num = common_config.parallel_num
        self.enable_prefix_cache = common_config.enable_prefix_cache
        self.benchmark_mode = common_config.benchmark_mode
        self.save_to_excel = save_to_excel
        self.file_save_path = PathUtil.get_datasets_dir_path(file_save_path).joinpath(
            self.benchmark_mode, f"{self.data_type}_latency.xlsx"
        )

        self.dataset, self.client, self.benchmark = TaskFactory.create_task(
            model_config, perf_config, eval_config
        )

    def run(self):
        logger.info("-----------------------------------------------------------")
        logger.info(
            f"Begin test, the data type: {self.data_type}, the benchmark mode: {self.benchmark_mode}"
        )
        latency_results, case_len = self.process()
        result_to_pytest = self.pytest_result(latency_results, case_len)
        return result_to_pytest

    @abstractmethod
    def process(self) -> Any:
        raise NotImplementedError

    def pytest_result(
        self, records: Union[LatencyStatistics, List[Dict]], case_len: int
    ):
        if isinstance(records, list):
            # If records is a list, it indicates the result of SyntheticPerfTask which has been processed before
            return records

        data_dict = self.update_single_record(records, case_len)
        data = list(data_dict.values())
        if self.perf_config and self.save_to_excel:
            logger.info(
                f"Begin save latency data to excel, file name: {self.file_save_path}"
            )
            FileUtil.save_excel(
                self.file_save_path, [data], PERF_CSV_HEADER, "Overall Performance"
            )
        return data_dict

    def update_single_record(self, record: LatencyStatistics, case_len: int):
        logger.info(f"There are {case_len} cases to save to the database.")
        data_dict = {
            "current_time": self.current_time,
            "total_case_num": case_len,
            "parallel_num": self.parallel_num,
            "enable_prefix_cache": self.enable_prefix_cache,
        }
        record_dict = record.to_dict()
        metric_key = list(record_dict.keys())[-1]
        latency_key = list(record_dict.keys())[:-1]
        if self.perf_config:
            data_dict.update({k: record_dict[k] for k in latency_key})
        else:
            data_dict.update({metric_key: record_dict[metric_key]})

        return data_dict

    def save_perf_cases_excel(
        self, records: List[RequestRecord | MultiTurnDialogRecord]
    ):
        save_data = []
        common_columns = [self.current_time, self.enable_prefix_cache]
        for idx, record in enumerate(records):
            if isinstance(record, MultiTurnDialogRecord):
                columns = common_columns + [record.total_turns, record.turn_id]
            elif isinstance(record, RequestRecord):
                columns = common_columns + [len(records), idx]
            columns += [
                record.case_name,
                record.input_tokens,
                record.output_tokens,
                round(record.req_cost * MS_SCALE, 3),
                round(record.prefill_latency * MS_SCALE, 3),
                round(record.tbt_latency * MS_SCALE, 3),
            ]
            save_data.append(columns)
        FileUtil.save_excel(
            self.file_save_path,
            save_data,
            CASE_PERF_CSV_HEADER,
            "Single Case Performance",
        )

    def save_eval_cases_excel(
        self, records: List[RequestRecord | MultiTurnDialogRecord], match_cls: str
    ):
        save_data = []
        common_columns = [self.current_time, self.enable_prefix_cache]
        for idx, record in enumerate(records):
            if isinstance(record, MultiTurnDialogRecord):
                columns = common_columns + [record.total_turns, record.turn_id]
            elif isinstance(record, RequestRecord):
                columns = common_columns + [len(records), idx]
            columns += [
                record.case_name,
                record.input_tokens,
                record.output_tokens,
                record.input_data,
                record.question,
                record.expected_output,
                record.output_data,
                record.is_match,
                match_cls,
            ]
            save_data.append(columns)
        FileUtil.save_excel(
            self.file_save_path,
            save_data,
            CASE_EVAL_CSV_HEADER,
            "Single Case Evaluation",
        )


class SyntheticPerfTask(BaseTask):
    def __init__(
        self,
        model_config: ModelConfig,
        perf_config: PerfConfig,
        file_save_path: str,
        stable_rate: int = 5,
    ):
        super().__init__(
            model_config=model_config,
            perf_config=perf_config,
            file_save_path=file_save_path,
        )
        self.enable_clear_hbm = model_config.enable_clear_hbm
        self.prompt_tokens = perf_config.prompt_tokens
        self.output_tokens = perf_config.output_tokens
        self.prefix_cache_num = perf_config.prefix_cache_num
        self.prompt_seed = 0 if self.enable_prefix_cache else -1
        self.stable_perf = self.benchmark_mode == BenchmarkModeType.STABLE_PREF
        self.stable_rate = stable_rate

    def process(self):
        result = []
        for parallel_num in self.parallel_num:
            for idx in range(len(self.prompt_tokens)):
                syntheric_params = SynthericParams()
                syntheric_params.parallel_num = parallel_num
                if self.stable_perf:
                    syntheric_params.parallel_num *= self.stable_rate
                if self.enable_prefix_cache:
                    syntheric_params.seeds = [
                        self.prompt_seed + i
                        for i in range(syntheric_params.parallel_num)
                    ]
                    self.prompt_seed += syntheric_params.parallel_num
                else:
                    syntheric_params.seeds = [
                        self.prompt_seed
                    ] * syntheric_params.parallel_num
                syntheric_params.prompt_tokens = self.prompt_tokens[idx]
                syntheric_params.prefix_cache_tokens = (
                    int(self.prefix_cache_num[idx] * syntheric_params.prompt_tokens)
                    if self.enable_prefix_cache
                    else 0
                )
                logger.info(
                    f"Performance benchmark running with: enable prefix cache: ({self.enable_prefix_cache}), {syntheric_params=}"
                )
                if self.enable_prefix_cache and self.prefix_cache_num[idx] > 0:
                    logger.info(f"Begin build kvcache...")
                    input_data = self.dataset.prepare_data(syntheric_params)
                    self.client.handle_requests_with_pool(
                        input_data, parallel_num, BAD_COMPLETION_TOKENS_THR
                    )
                    logger.info(
                        "To ensure thal all kvcache is offload2ssd, sleep for 10 seconds"
                    )
                    time.sleep(10)

                if self.enable_clear_hbm:
                    self.client.clear_hbm()

                logger.info(f"Begin post cases...")
                input_data = self.dataset.prepare_data(syntheric_params)
                records: List[RequestRecord] = self.client.handle_requests_with_pool(
                    input_data, parallel_num, self.output_tokens[idx]
                )
                latency_statistics: LatencyStatistics = self.benchmark.perf_show(
                    records, parallel_num
                )
                # Make sure to store the data after each test is completed, to prevent data loss after a request fails
                data_dict = {
                    "current_time": self.current_time,
                    "total_case_num": syntheric_params.parallel_num,
                    "input_tokens": self.prompt_tokens[idx],
                    "output_tokens": self.output_tokens[idx],
                    "parallel_num": parallel_num,
                    "enable_prefix_cache": self.enable_prefix_cache,
                    "prefix_cache_num": (
                        self.prefix_cache_num[idx] if self.enable_prefix_cache else 0
                    ),
                }
                latency_dict = latency_statistics.to_dict()
                data_dict.update(dict(list(latency_dict.items())[:-1]))
                data = data_dict.values()
                if self.save_to_excel:
                    logger.info(
                        f"Begin save latency data to excel, file name: {self.file_save_path}"
                    )
                    FileUtil.save_excel(
                        self.file_save_path,
                        [data],
                        SYNC_PERF_CSV_HEADER,
                        "Overall Performance",
                    )

                result.append(data_dict)

        return result, len(result)


class MultiTurnDialogPerfTask(BaseTask):
    def __init__(
        self, model_config: ModelConfig, perf_config: PerfConfig, file_save_path: str
    ):
        super().__init__(
            model_config=model_config,
            perf_config=perf_config,
            file_save_path=file_save_path,
        )
        self.dataset_file_path = perf_config.dataset_file_path

    def process(self):
        cases = self.dataset.prepare_data(self.dataset_file_path)
        records: List[List[MultiTurnDialogRecord]] = (
            self.client.handle_requests_with_pool(cases, self.parallel_num)
        )
        for record in records:
            self.save_perf_cases_excel(record)
        all_records = [r for record in records for r in record]
        latency_statistics = self.benchmark.perf_show(all_records, self.parallel_num)
        return latency_statistics, len(records)


class DocQaPerfTask(BaseTask):
    def __init__(
        self, model_config: ModelConfig, perf_config: PerfConfig, file_save_path: str
    ):
        super().__init__(
            model_config=model_config,
            perf_config=perf_config,
            file_save_path=file_save_path,
        )
        self.dataset_file_path = perf_config.dataset_file_path
        self.max_tokens = model_config.payload.get("max_tokens")

    def process(self):
        cases_list = self.dataset.prepare_data(self.dataset_file_path)
        if self.enable_prefix_cache:
            logger.info("Begin build kvcache...")
            self.client.handle_requests_with_pool(
                cases_list, self.parallel_num, BAD_COMPLETION_TOKENS_THR
            )

        logger.info("Begin post cases...")
        records: List[RequestRecord] = self.client.handle_requests_with_pool(
            cases_list, self.parallel_num, self.max_tokens
        )
        self.save_perf_cases_excel(records)
        latency_statistics = self.benchmark.perf_show(records, self.parallel_num)
        return latency_statistics, len(records)


class DocQaEvalTask(BaseTask):
    def __init__(
        self, model_config: ModelConfig, eval_config: EvalConfig, file_save_path: str
    ):
        super().__init__(
            model_config=model_config,
            eval_config=eval_config,
            file_save_path=file_save_path,
        )
        self.dataset_file_path = eval_config.dataset_file_path
        self.max_tokens = model_config.payload.get("max_tokens")
        self.eval_cls = eval_config.eval_class

    def process(self):
        cases_list = self.dataset.prepare_data(self.dataset_file_path)
        if self.enable_prefix_cache:
            logger.info("Begin build kvcache...")
            self.client.handle_requests_with_pool(
                cases_list, self.parallel_num, BAD_COMPLETION_TOKENS_THR
            )

        logger.info("Begin post cases...")
        records: List[RequestRecord] = self.client.handle_requests_with_pool(
            cases_list, self.parallel_num, self.max_tokens
        )
        metric_result, match_record_list = self.benchmark.perf_show(
            records, self.parallel_num
        )
        self.save_eval_cases_excel(match_record_list, self.eval_cls)
        return metric_result, len(records)
