from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
from common.uc_eval.utils.data_class import (
    EvalConfig,
    LatencyStatistics,
    MultiTurnDialogRecord,
    RequestRecord,
)
from common.uc_eval.utils.utils import get_logger
from tqdm import tqdm

logger = get_logger()
MS_SCALE = 1000
# the max wave rate for stable perf
MAX_WAVE_RATE = 0.05


class BenchmarkBase(ABC):
    def __init__(self, eval_config: Optional[EvalConfig], stable_perf: bool = False):
        self.stable_perf = stable_perf

    def get_success_request(self, data: List[RequestRecord | MultiTurnDialogRecord]):
        """
        Get the successful request from the record
        """
        success_request = []
        for request in data:
            if request.is_success:
                success_request.append(request)
        if len(success_request) == 0:
            logger.warning(f"No success request found, please check the result")
        return success_request

    def result_to_column_dict(
        self, data: List[RequestRecord | MultiTurnDialogRecord]
    ) -> Dict[str, List[Any]]:
        """
        format: list[dict] ---> dict[list]
        """
        if not data:
            return {}
        keys = list(data[0].to_dict().keys())
        result = {key: [] for key in keys}
        for item in data:
            for key in keys:
                result[key].append(item.to_dict()[key])
        return result

    @abstractmethod
    def perf_show(self, records: Any, parallel_num: int = 1):
        raise NotImplementedError


class EvaluatorBenchmark(BenchmarkBase):
    def __init__(self, stable_perf: bool, eval_class: str):
        self.stable_perf = stable_perf
        self.metric_method = eval_class

    def perf_show(self, records: List[RequestRecord | MultiTurnDialogRecord]):
        pass


class PerformanceBenchmark(BenchmarkBase):
    def __init__(self, stable_perf: bool):
        super().__init__(stable_perf)
        self.stable_perf = stable_perf
        self.stable_work_time = [0, 0]

    def perf_show(
        self,
        input_data_lists: List[RequestRecord | MultiTurnDialogRecord],
        parallel_num: int,
    ) -> LatencyStatistics:
        logger.info(f"Begin calculate latency...")
        success_request = self.get_success_request(input_data_lists)
        request_record_dict = self.result_to_column_dict(success_request)
        if self.stable_perf:
            request_ids = self._get_stable_request_id(request_record_dict, parallel_num)
        else:
            request_ids = request_record_dict.get("request_id")
        records = [
            record for record in input_data_lists if record.request_id in request_ids
        ]
        perf_result = self._get_performance_data(records)
        return perf_result

    def _get_performance_data(
        self, record_list: List[RequestRecord | MultiTurnDialogRecord]
    ) -> LatencyStatistics:
        """
        After all requests are completed, get the performance data
        """
        if len(record_list) == 0:
            logger.warning(f"there is no request_id in the record_list, please check")
        latency = LatencyStatistics()
        record_dict = self.result_to_column_dict(record_list)

        e2e_latency_all = (
            max(record_dict["end_time"]) - min(record_dict["start_time"])
        ) * MS_SCALE
        latency.e2e_latency_all = round(e2e_latency_all, 2)
        logger.debug("All request latencies: %.4f ms", e2e_latency_all)

        total_output_tokens = sum(record_dict["output_tokens"])
        output_token_throughput = total_output_tokens / e2e_latency_all * MS_SCALE
        latency.output_token_throughput = round(output_token_throughput, 2)
        logger.debug(
            "Total output token throughput: %.4f tokens/s", output_token_throughput
        )

        throughputs = []
        for tokens, cost in zip(record_dict["output_tokens"], record_dict["req_cost"]):
            if cost > 0:
                throughputs.append(tokens / cost)
        if throughputs:
            token_throughput_per_request = np.mean(throughputs)
            latency.token_throughput_per_request = round(
                token_throughput_per_request, 2
            )
            logger.debug(
                "Average per-request throughput: %.4f tokens/s",
                token_throughput_per_request,
            )
        else:
            logger.warning("No valid requests for throughput calculation")

        prefill_latency_list = [record_dict["prefill_latency"]]
        p50_prefill_latency = np.percentile(prefill_latency_list, 50) * MS_SCALE
        latency.p50_prefill_latency = round(p50_prefill_latency, 2)
        logger.debug("Time to First token latency P50: %.4f ms", p50_prefill_latency)

        p90_prefill_latency = np.percentile(prefill_latency_list, 90) * MS_SCALE
        latency.p90_prefill_latency = round(p90_prefill_latency, 2)
        logger.debug("Time to First token latency TP90: %.4f ms", p90_prefill_latency)

        p99_prefill_latency = np.percentile(prefill_latency_list, 99) * MS_SCALE
        latency.p99_prefill_latency = round(p99_prefill_latency, 2)
        logger.debug("Time to First token latency TP99: %.4f ms", p99_prefill_latency)

        max_prefill_latency = np.max(prefill_latency_list) * MS_SCALE
        latency.max_prefill_latency = round(max_prefill_latency, 2)
        logger.debug(
            "Maximum time to first token latency: %.4f ms", max_prefill_latency
        )

        avg_prefill_latency = np.mean(prefill_latency_list) * MS_SCALE
        latency.avg_prefill_latency = round(avg_prefill_latency, 2)
        logger.debug(
            "Average time to first token latency: %.4f ms", avg_prefill_latency
        )

        # 这里是list[list[float]]，要求的是这里所有float的平均（这里可能也会要求list[float]中去除前两个元素后，取所有请求per decode time的均值）
        decode_latency_list = []
        for tmp_list in record_dict["tbt_list"]:
            decode_latency_list.extend(tmp_list[2:])

        p50_decode_latency = np.percentile(decode_latency_list, 50) * MS_SCALE
        latency.p50_decode_latency = round(p50_decode_latency, 2)
        logger.debug("Tokens Per Second latency TP50: %.4f ms", p50_decode_latency)

        p90_decode_latency = np.percentile(decode_latency_list, 90) * MS_SCALE
        latency.p90_decode_latency = round(p90_decode_latency, 2)
        logger.debug("Tokens Per Second latency TP90: %.4f ms", p90_decode_latency)

        p99_decode_latency = np.percentile(decode_latency_list, 99) * MS_SCALE
        latency.p99_decode_latency = round(p99_decode_latency, 2)
        logger.debug("Tokens Per Second latency TP99: %.4f ms", p99_decode_latency)

        max_decode_latency = np.max(decode_latency_list) * MS_SCALE
        latency.max_decode_latency = round(max_decode_latency, 2)
        logger.debug("Maximum tokens per second latency: %.4f ms", max_decode_latency)

        avg_decode_latency = np.mean(decode_latency_list) * MS_SCALE
        latency.avg_decode_latency = round(avg_decode_latency, 2)
        logger.debug("Average tokens per second latency: %.4f ms", avg_decode_latency)

        return latency.to_dict()

    def _get_stable_request_id(
        self, result: Dict[str, List[Any]], target_concurrency: int
    ):
        """
        Get steady-state request ids via start_time vs. end_time delta
        """
        # the number of concurrent requests at each request start and end
        request_num = len(result.get("request_id", []))
        concurrent_levels = [0] * 2 * request_num
        request_events = []
        for idx in range(request_num):
            request_events.append(
                {
                    "request_id": result.get("request_id", [])[idx],
                    "event_type": "start",
                    "timestamp": result.get("start_time", [])[idx],
                }
            )
            request_events.append(
                {
                    "request_id": result.get("request_id", [])[idx],
                    "event_type": "end",
                    "timestamp": result.get("end_time", [])[idx],
                }
            )
        sorted_events = sorted(request_events, key=lambda x: x["timestamp"])
        stable_stage_requests = []
        logger.info("Start calculating stable request id")
        used_request_num = 0
        for idx, item in enumerate(
            tqdm(sorted_events, desc="search stable request id")
        ):
            if item["event_type"] == "start":
                used_request_num += 1
                concurrent_levels[idx] = (
                    concurrent_levels[idx - 1] + 1 if idx > 0 else 1
                )
            else:
                concurrent_levels[idx] = concurrent_levels[idx - 1] - 1
            if (
                item["event_type"] == "start"
                and concurrent_levels[idx] == target_concurrency
            ):
                stable_stage_requests.append(item["request_id"])
                if len(stable_stage_requests) == 2:
                    self.stable_work_time[0] = item["timestamp"]
            elif (
                item["event_type"] == "start"
                and concurrent_levels[idx]
                >= int(target_concurrency * (1 - MAX_WAVE_RATE))
                and len(stable_stage_requests) > 2
            ):
                stable_stage_requests.append(item["request_id"])
            elif used_request_num == request_num and item["event_type"] == "end":
                self.stable_work_time[1] = item["timestamp"]
                break
            elif (
                len(stable_stage_requests) > 1
                and item["event_type"] == "end"
                and concurrent_levels[idx]
                < int(target_concurrency * (1 - MAX_WAVE_RATE))
            ):
                self.stable_work_time[1] = item["timestamp"]
                break

        if len(stable_stage_requests) > 1:
            # igore first request
            stable_stage_requests.pop(0)
        if len(stable_stage_requests) == 0:
            logger.error("cannot find stable stage, please check your settings")
            raise ValueError("cannot find stable stage, please check your settings")
        logger.info(f"stable request id list: {stable_stage_requests=}")
        return stable_stage_requests
