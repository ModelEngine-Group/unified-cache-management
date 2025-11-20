import time
from typing import Any
from abc import ABC, abstractmethod
from common.uc_eval.utils.utils import get_logger
from common.uc_eval.utils.config_loader import ConfigLoader, TaskFactory
from common.uc_eval.utils.data_class import (
    SynthericParams,
    BenchmarkModeType,
    ModelConfig,
    PerfConfig,
    EvalConfig,
)

BAD_COMPLETION_TOKENS_THR = 20
logger = get_logger()


class BaseTask(ABC):
    def __init__(
        self,
        model_config: ModelConfig,
        perf_config: PerfConfig = None,
        eval_config: EvalConfig = None,
    ):
        ConfigLoader(model_config, perf_config, eval_config)
        self.model_config = model_config
        self.perf_config = perf_config
        self.eval_config = eval_config

        self.dataset, self.client, self.benchmark = TaskFactory.create_task(
            model_config, perf_config, eval_config
        )

    @abstractmethod
    def process(self) -> Any:
        raise NotImplementedError


class SyntheticPerfTask(BaseTask):
    def __init__(self, model_config: ModelConfig, perf_config: PerfConfig):
        super().__init__(model_config, perf_config)
        self.enable_clear_hbm = model_config.enable_clear_hbm
        self.enable_prefix_cache = perf_config.enable_prefix_cache
        self.parallel_num = perf_config.parallel_num
        self.prompt_tokens = perf_config.prompt_tokens
        self.output_tokens = perf_config.output_tokens
        self.prefix_cache_num = perf_config.prefix_cache_num
        self.benchmark_mode = perf_config.benchmark_mode
        self.stable_perf = perf_config.benchmark_mode == BenchmarkModeType.STABLE_PREF
        self.prompt_seed = 0 if self.enable_prefix_cache else -1

    def process(self):
        logger.info("-------------------------------------------------------------------")
        logger.info(
            f"Starting synthetic performance benchmark, the benchmark mode is {self.benchmark_mode}"
        )
        result = []
        for parallel_num in self.parallel_num:
            for idx in range(len(self.prompt_tokens)):
                syntheric_params = SynthericParams()
                syntheric_params.parallel_num = parallel_num
                if self.stable_perf:
                    syntheric_params.parallel_num *= 5
                if self.enable_prefix_cache:
                    syntheric_params.seeds = [
                        self.prompt_seed + i for i in range(syntheric_params.parallel_num)
                    ]
                    self.prompt_seed += syntheric_params.parallel_num
                else:
                    syntheric_params.seeds = [self.prompt_seed] * syntheric_params.parallel_num
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
                    logger.info("To ensure thal all kvcache is offload2ssd, sleep for 10 seconds")
                    time.sleep(10)

                if self.enable_clear_hbm:
                    self.client.clear_hbm()

                logger.info(f"Begin post cases...")
                input_data = self.dataset.prepare_data(syntheric_params)
                request_records = self.client.handle_requests_with_pool(
                    input_data, parallel_num, self.output_tokens[idx]
                )
                latency_statistics = self.benchmark.perf_show(request_records, parallel_num)
                result.append(latency_statistics)
        return result


class MultiPerfTask(BaseTask):
    def __init__(self, model_config: ModelConfig, perf_config: PerfConfig):
        super().__init__(model_config, perf_config)
        self.data_type = perf_config.data_type
        self.dataset_file_path = perf_config.dataset_file_path
        self.benchmark_mode = perf_config.benchmark_mode
        self.parallel_num = perf_config.parallel_num

    def process(self):
        logger.info(
            f"Begin test, the data type: {self.data_type}, the benchmark mode: {self.benchmark_mode}"
        )
        cases = self.dataset.prepare_data(self.dataset_file_path)
        records = self.client.handle_requests_with_pool(cases, self.parallel_num)
        all_records = [r for record in records for r in record]
        latency_statistics = self.benchmark.perf_show(all_records, self.parallel_num)
        return latency_statistics


class DocQaPerfTask(BaseTask):
    def __init__(self, model_config: ModelConfig, perf_config: PerfConfig):
        super().__init__(model_config, perf_config)
        self.data_type = perf_config.data_type
        self.dataset_file_path = perf_config.dataset_file_path
        self.enable_prefix_cache = perf_config.enable_prefix_cache
        self.parallel_num = perf_config.parallel_num
        self.max_tokens = model_config.payload.get("max_tokens")
        self.benchmark_mode = perf_config.benchmark_mode

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
        record = self.client.handle_requests_with_pool(
            cases_list, self.parallel_num, self.max_tokens
        )
        latency_statistics = self.benchmark.perf_show(record, self.parallel_num)
        return latency_statistics
