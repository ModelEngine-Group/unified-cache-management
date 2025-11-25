import time
from abc import ABC, abstractmethod
from typing import Any, List, Union

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
    "测试日期",
    "总测试用例数",
    "并发数",
    "是否开启prefix cache",
    "所有请求耗时(ms)",
    "E2E吞吐(tokens/s)",
    "单请求平均吞吐(tokens/s)",
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
    "测试日期",
    "总测试用例数",
    "输入长度",
    "输出长度",
    "并发数",
    "是否开启prefix cache",
    "命中率",
    "所有请求耗时(ms)",
    "E2E吞吐(tokens/s)",
    "单请求平均吞吐(tokens/s)",
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
    "测试日期",
    "是否开启prefix cache",
    "总测试用例数",
    "当前轮数",
    "测试用例名",
    "输入tokens数",
    "输出tokens数",
    "请求耗时(ms)",
    "TTFT(ms)",
    "TBT(ms)",
]

CASE_EVAL_CSV_HEADER = [
    "测试日期",
    "是否开启prefix cache",
    "总测试用例数",
    "当前轮数",
    "测试用例名",
    "输入tokens数",
    "输出tokens数",
    "模型输入",
    "问题",
    "正确答案",
    "模型实际输出",
    "准确率",
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
        self.data_type = perf_config.data_type
        self.parallel_num = perf_config.parallel_num
        self.enable_prefix_cache = perf_config.enable_prefix_cache
        self.benchmark_mode = perf_config.benchmark_mode
        self.save_to_excel = save_to_excel
        self.file_save_path = PathUtil.get_datasets_dir_path(file_save_path).joinpath(
            perf_config.benchmark_mode, f"{perf_config.data_type}_latency.xlsx"
        )

        self.dataset, self.client, self.benchmark = TaskFactory.create_task(
            model_config, perf_config, eval_config
        )

    def run(self):
        logger.info("-----------------------------------------------------------")
        logger.info(
            "Begin test, the data type: {self.data_type}, the benchmark mode: {self.benchmark_mode}"
        )
        latency_results, case_len = self.process()
        result_to_pytest = self.pytest_result(latency_results, case_len)
        return result_to_pytest

    @abstractmethod
    def process(self) -> Any:
        raise NotImplementedError

    def pytest_result(
        self, records: Union[LatencyStatistics, List[LatencyStatistics]], case_len: int
    ):
        # If records is list, it means that the data has already been processed and saved in process.
        if isinstance(records, list):
            return records
        logger.info(f"There are {case_len} cases to save to the database.")
        data = [
            self.current_time,
            case_len,
            self.parallel_num,
            self.enable_prefix_cache,
        ]
        data.extend(list(records.to_dict().values()))
        if self.save_to_excel:
            logger.info(
                f"Begin save latency data to excel, file name: {self.file_save_path}"
            )
            FileUtil.save_excel(
                self.file_save_path, [data], PERF_CSV_HEADER, "Overall Performance"
            )
        return data

    def save_cases_excel(self, records: List[RequestRecord | MultiTurnDialogRecord]):
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
        logger.info(
            "-------------------------------------------------------------------"
        )
        logger.info(
            f"Starting synthetic performance benchmark, the benchmark mode is {self.benchmark_mode}"
        )
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
                latency_statistics = self.benchmark.perf_show(records, parallel_num)
                # Make sure to store the data after each test is completed, to prevent data loss after a request fails
                data = [
                    self.current_time,
                    syntheric_params.parallel_num,
                    self.prompt_tokens[idx],
                    self.output_tokens[idx],
                    parallel_num,
                    self.enable_prefix_cache,
                    self.prefix_cache_num[idx] if self.enable_prefix_cache else 0,
                ]
                data.extend(list(latency_statistics.to_dict().values()))
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

                result.append(data)

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
        logger.info(
            f"Begin test, the data type: {self.data_type}, the benchmark mode: {self.benchmark_mode}"
        )
        cases = self.dataset.prepare_data(self.dataset_file_path)
        records: List[List[MultiTurnDialogRecord]] = (
            self.client.handle_requests_with_pool(cases, self.parallel_num)
        )
        for record in records:
            self.save_cases_excel(record)
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
        logger.info(
            f"Begin test, the data type: {self.data_type}, the benchmark mode: {self.benchmark_mode}"
        )
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
        self.save_cases_excel(records)
        latency_statistics = self.benchmark.perf_show(records, self.parallel_num)
        return latency_statistics, len(records)
