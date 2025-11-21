# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import os
import threading
import time

# Third Party
from prometheus_client import REGISTRY
import prometheus_client
from vllm.config import VllmConfig

# First Party
from ucm.logger import init_logger
def thread_safe(func):
    def wrapper(*args, **kwargs):
        with threading.Lock():
            result = func(*args, **kwargs)
        return result

    return wrapper

logger = init_logger(__name__)


@dataclass
class UCMStats:
    # Counter (Note that these are incremental values,
    # which will accumulate over time in Counter)
    interval_retrieve_requests: int
    interval_store_requests: int
    interval_lookup_requests: int
    interval_requested_tokens: int
    interval_hit_tokens: int
    interval_stored_tokens: int
    interval_lookup_tokens: int
    interval_lookup_hits: int
    interval_vllm_hit_tokens: int
    interval_prompt_tokens: int

    # Real time value measurements (will be reset after each log)
    retrieve_hit_rate: float
    lookup_hit_rate: float

    # Distribution measurements
    time_to_retrieve: List[float]
    time_to_store: List[float]
    retrieve_speed: List[float]  # Tokens per second
    store_speed: List[float]  # Tokens per second

    # request lookup hit rates
    interval_lookup_hit_rates: List[float]


@dataclass
class LookupRequestStats:
    num_tokens: int
    hit_tokens: int
    is_finished: bool

    def hit_rate(self):
        if self.num_tokens == 0:
            return 0
        return self.hit_tokens / self.num_tokens


@dataclass
class RetrieveRequestStats:
    num_tokens: int
    local_hit_tokens: int
    remote_hit_tokens: int  # Not used for now
    start_time: float
    end_time: float

    def time_to_retrieve(self):
        if self.end_time == 0:
            return 0
        return self.end_time - self.start_time

    def retrieve_speed(self):
        if self.time_to_retrieve() == 0:
            return 0
        return (
            self.local_hit_tokens + self.remote_hit_tokens
        ) / self.time_to_retrieve()


@dataclass
class StoreRequestStats:
    num_tokens: int
    start_time: float
    end_time: float

    def time_to_store(self):
        if self.end_time == 0:
            return 0
        return self.end_time - self.start_time

    def store_speed(self):
        if self.time_to_store() == 0:
            return 0
        return self.num_tokens / self.time_to_store()


class UCMStatsMonitor:
    def __init__(self):
        # Interval metrics that will be reset after each log
        # Accumulate incremental values in the Prometheus Counter
        self.interval_retrieve_requests = 0
        self.interval_store_requests = 0
        self.interval_lookup_requests = 0
        self.interval_requested_tokens = 0  # total requested tokens retrieve
        self.interval_hit_tokens = 0  # total hit tokens retrieve
        self.interval_stored_tokens = 0  # total tokens tored in LMCache
        self.interval_lookup_tokens = 0  # total requested tokens lookup
        self.interval_lookup_hits = 0  # total hit tokens lookup
        self.interval_vllm_hit_tokens = 0  # total hit tokens in vllm
        self.interval_prompt_tokens = 0  # total prompt tokens

        self.retrieve_requests: Dict[int, RetrieveRequestStats] = {}
        self.store_requests: Dict[int, StoreRequestStats] = {}
        self.lookup_requests: Dict[int, LookupRequestStats] = {}

        self.retrieve_request_id = 0
        self.store_request_id = 0
        self.lookup_request_id = 0

    @thread_safe
    def on_lookup_request(self, num_tokens: int) -> int:
        """
        This function is called when a lookup request is sent to the cache.
        It will record the number of tokens requested.
        """
        lookup_stats = LookupRequestStats(
            num_tokens=num_tokens,
            hit_tokens=0,
            is_finished=False,
        )
        self.interval_lookup_requests += 1
        self.interval_lookup_tokens += num_tokens
        self.lookup_requests[self.lookup_request_id] = lookup_stats
        self.lookup_request_id += 1
        return self.lookup_request_id - 1

    @thread_safe
    def on_lookup_finished(self, request_id: int, num_hit_tokens: int):
        """
        This function is called when a lookup request is finished.
        It will record the number of tokens hit.
        """
        assert request_id in self.lookup_requests
        lookup_stats = self.lookup_requests[request_id]
        lookup_stats.hit_tokens = num_hit_tokens
        lookup_stats.is_finished = True
        self.interval_lookup_hits += num_hit_tokens

    @thread_safe
    def on_retrieve_request(self, num_tokens: int) -> int:
        """
        Returns the internal "request id" that will be used in
        on_retrieve_finished
        """
        curr_time = time.time()
        retrieve_stats = RetrieveRequestStats(
            num_tokens=num_tokens,
            local_hit_tokens=0,
            remote_hit_tokens=0,
            start_time=curr_time,
            end_time=0,
        )
        self.interval_requested_tokens += num_tokens
        self.interval_retrieve_requests += 1
        self.retrieve_requests[self.retrieve_request_id] = retrieve_stats
        self.retrieve_request_id += 1
        return self.retrieve_request_id - 1

    @thread_safe
    def on_retrieve_finished(self, request_id: int, retrieved_tokens: int):
        curr_time = time.time()
        assert request_id in self.retrieve_requests
        retrieve_stats = self.retrieve_requests[request_id]
        retrieve_stats.local_hit_tokens = retrieved_tokens
        retrieve_stats.end_time = curr_time
        self.interval_hit_tokens += retrieved_tokens

    @thread_safe
    def on_store_request(self, num_tokens: int) -> int:
        """
        Returns the internal "request id" that will be used in on_store_finished
        """
        curr_time = time.time()
        store_stats = StoreRequestStats(
            num_tokens=num_tokens, start_time=curr_time, end_time=0
        )
        self.interval_store_requests += 1
        self.interval_stored_tokens += num_tokens
        self.store_requests[self.store_request_id] = store_stats
        self.store_request_id += 1
        return self.store_request_id - 1

    @thread_safe
    def on_store_finished(self, request_id: int, num_tokens: int = -1):
        curr_time = time.time()
        assert request_id in self.store_requests
        store_stats = self.store_requests[request_id]
        store_stats.end_time = curr_time
        if num_tokens >= 0:
            store_stats.num_tokens = num_tokens


    @thread_safe
    def update_interval_vllm_hit_tokens(self, delta: int):
        self.interval_vllm_hit_tokens += delta

    @thread_safe
    def update_interval_prompt_tokens(self, delta: int):
        self.interval_prompt_tokens += delta

    def _clear(self):
        """
        Clear all the distribution stats
        """
        self.interval_retrieve_requests = 0
        self.interval_store_requests = 0
        self.interval_lookup_requests = 0

        self.interval_requested_tokens = 0
        self.interval_hit_tokens = 0
        self.interval_stored_tokens = 0
        self.interval_lookup_tokens = 0
        self.interval_lookup_hits = 0
        self.interval_vllm_hit_tokens = 0
        self.interval_prompt_tokens = 0

        new_retrieve_requests = {}
        for request_id, retrieve_stats in self.retrieve_requests.items():
            if retrieve_stats.end_time == 0:
                new_retrieve_requests[request_id] = retrieve_stats
        self.retrieve_requests = new_retrieve_requests

        new_store_requests = {}
        for request_id, store_stats in self.store_requests.items():
            if store_stats.end_time == 0:
                new_store_requests[request_id] = store_stats
        self.store_requests = new_store_requests

        new_lookup_requests = {}
        for request_id, lookup_stats in self.lookup_requests.items():
            if not lookup_stats.is_finished:
                new_lookup_requests[request_id] = lookup_stats
        self.lookup_requests = new_lookup_requests

    @thread_safe
    def get_stats_and_clear(self) -> UCMStats:
        """
        This function should be called with by prometheus adapter with
        a specific interval.
        The function will return the latest states between the current
        call and the previous call.
        """
        retrieve_hit_rate = (
            0
            if self.interval_requested_tokens == 0
            else self.interval_hit_tokens / self.interval_requested_tokens
        )

        lookup_hit_rate = (
            0
            if self.interval_lookup_tokens == 0
            else self.interval_lookup_hits / self.interval_lookup_tokens
        )

        def filter_out_invalid(stats: List[float]):
            return [x for x in stats if x != 0]

        time_to_retrieve = filter_out_invalid(
            [stats.time_to_retrieve() for stats in self.retrieve_requests.values()]
        )

        time_to_store = filter_out_invalid(
            [stats.time_to_store() for stats in self.store_requests.values()]
        )

        retrieve_speed = filter_out_invalid(
            [stats.retrieve_speed() for stats in self.retrieve_requests.values()]
        )

        store_speed = filter_out_invalid(
            [stats.store_speed() for stats in self.store_requests.values()]
        )

        request_lookup_hit_rates = [
            stats.hit_rate()
            for stats in self.lookup_requests.values()
            if stats.is_finished
        ]

        ret = UCMStats(
            interval_retrieve_requests=self.interval_retrieve_requests,
            interval_store_requests=self.interval_store_requests,
            interval_lookup_requests=self.interval_lookup_requests,
            interval_requested_tokens=self.interval_requested_tokens,
            interval_hit_tokens=self.interval_hit_tokens,
            interval_stored_tokens=self.interval_stored_tokens,
            interval_lookup_tokens=self.interval_lookup_tokens,
            interval_lookup_hits=self.interval_lookup_hits,
            retrieve_hit_rate=retrieve_hit_rate,
            lookup_hit_rate=lookup_hit_rate,
            time_to_retrieve=time_to_retrieve,
            time_to_store=time_to_store,
            retrieve_speed=retrieve_speed,
            store_speed=store_speed,
            interval_vllm_hit_tokens=self.interval_vllm_hit_tokens,
            interval_lookup_hit_rates=request_lookup_hit_rates,
            interval_prompt_tokens=self.interval_prompt_tokens,
        )
        self._clear()
        return ret

    _instance = None

    @staticmethod
    def GetOrCreate() -> "UCMStatsMonitor":
        if UCMStatsMonitor._instance is None:
            UCMStatsMonitor._instance = UCMStatsMonitor()
        return UCMStatsMonitor._instance

    @staticmethod
    def DestroyInstance():
        UCMStatsMonitor._instance = None

    @staticmethod
    def unregister_all_metrics():
        collectors = list(REGISTRY._collector_to_names.keys())
        for collector in collectors:
            try:
                REGISTRY.unregister(collector)
            except KeyError:
                pass


class PrometheusLogger:
    _gauge_cls = prometheus_client.Gauge
    _counter_cls = prometheus_client.Counter
    _histogram_cls = prometheus_client.Histogram

    def __init__(self, vllm_config: "VllmConfig", engine_index: int = 0):
        # Ensure PROMETHEUS_MULTIPROC_DIR is set before any metric registration
        if "PROMETHEUS_MULTIPROC_DIR" not in os.environ:
            default_dir = "/tmp/ucm_prometheus"
            os.environ["PROMETHEUS_MULTIPROC_DIR"] = default_dir
            if not os.path.exists(default_dir):
                os.makedirs(default_dir, exist_ok=True)

        labelnames = ["model_name", "engine"]

        self.labels =  {
            "model_name": vllm_config.model_config.served_model_name,
            "engine": str(engine_index),
        }

        self.counter_num_retrieve_requests = self._counter_cls(
            name="ucm:num_retrieve_requests",
            documentation="Total number of retrieve requests sent to lmcache",
            labelnames=labelnames,
        )

        self.counter_num_store_requests = self._counter_cls(
            name="ucm:num_store_requests",
            documentation="Total number of store requests sent to lmcache",
            labelnames=labelnames,
        )

        self.counter_num_lookup_requests = self._counter_cls(
            name="ucm:num_lookup_requests",
            documentation="Total number of lookup requests sent to lmcache",
            labelnames=labelnames,
        )

        self.counter_num_requested_tokens = self._counter_cls(
            name="ucm:num_requested_tokens",
            documentation="Total number of tokens requested from lmcache",
            labelnames=labelnames,
        )

        self.counter_num_hit_tokens = self._counter_cls(
            name="ucm:num_hit_tokens",
            documentation="Total number of tokens hit in lmcache",
            labelnames=labelnames,
        )

        self.counter_num_stored_tokens = self._counter_cls(
            name="ucm:num_stored_tokens",
            documentation=(
                "Total number of tokens stored in lmcache including evicted ones"
            ),
            labelnames=labelnames,
        )

        self.counter_num_lookup_tokens = self._counter_cls(
            name="ucm:num_lookup_tokens",
            documentation="Total number of tokens requested in lookup from lmcache",
            labelnames=labelnames,
        )

        self.counter_num_lookup_hits = self._counter_cls(
            name="ucm:num_lookup_hits",
            documentation="Total number of tokens hit in lookup from lmcache",
            labelnames=labelnames,
        )

        self.counter_num_vllm_hit_tokens = self._counter_cls(
            name="ucm:num_vllm_hit_tokens",
            documentation="Number of hit tokens in vllm",
            labelnames=labelnames,
        )

        self.counter_num_prompt_tokens = self._counter_cls(
            name="ucm:num_prompt_tokens",
            documentation="Number of prompt tokens in lmcache",
            labelnames=labelnames,
        )


        self.gauge_retrieve_hit_rate = self._gauge_cls(
            name="ucm:retrieve_hit_rate",
            documentation="Hit rate of lmcache retrieve requests since last log",
            labelnames=labelnames,
            multiprocess_mode="livemostrecent",
        )

        self.gauge_lookup_hit_rate = self._gauge_cls(
            name="ucm:lookup_hit_rate",
            documentation="Hit rate of lmcache lookup requests since last log",
            labelnames=labelnames,
            multiprocess_mode="livemostrecent",
        )


        time_to_retrieve_buckets = [
            0.001,
            0.005,
            0.01,
            0.02,
            0.04,
            0.06,
            0.08,
            0.1,
            0.25,
            0.5,
            0.75,
            1.0,
            2.5,
            5.0,
            7.5,
            10.0,
        ]
        self.histogram_time_to_retrieve = self._histogram_cls(
            name="ucm:time_to_retrieve",
            documentation="Time to retrieve from lmcache (seconds)",
            labelnames=labelnames,
            buckets=time_to_retrieve_buckets,
        )

        time_to_store_buckets = [
            0.001,
            0.005,
            0.01,
            0.02,
            0.04,
            0.06,
            0.08,
            0.1,
            0.25,
            0.5,
            0.75,
            1.0,
            2.5,
            5.0,
            7.5,
            10.0,
        ]
        self.histogram_time_to_store = self._histogram_cls(
            name="ucm:time_to_store",
            documentation="Time to store to lmcache (seconds)",
            labelnames=labelnames,
            buckets=time_to_store_buckets,
        )

        retrieve_speed_buckets = [
            1,
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            1024,
            2048,
            4096,
            8192,
            16384,
            32768,
            65536,
        ]
        self.histogram_retrieve_speed = self._histogram_cls(
            name="ucm:retrieve_speed",
            documentation="Retrieve speed of lmcache (tokens per second)",
            labelnames=labelnames,
            buckets=retrieve_speed_buckets,
        )

        store_speed_buckets = [
            1,
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            1024,
            2048,
            4096,
            8192,
            16384,
            32768,
            65536,
        ]
        self.histogram_store_speed = self._histogram_cls(
            name="ucm:store_speed",
            documentation="Store speed of lmcache (tokens per second)",
            labelnames=labelnames,
            buckets=store_speed_buckets,
        )

        request_cache_hit_rate = [
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
        ]
        self.histogram_request_cache_hit_rate = self._histogram_cls(
            name="ucm:request_cache_hit_rate",
            documentation="Request cache hit rate",
            labelnames=labelnames,
            buckets=request_cache_hit_rate,
        )


    def _log_gauge(self, gauge, data: Union[int, float]) -> None:
        # Convenience function for logging to gauge.
        gauge.labels(**self.labels).set(data)

    def _log_counter(self, counter, data: Union[int, float]) -> None:
        # Convenience function for logging to counter.
        # Prevent ValueError from negative increment
        if data < 0:
            return
        counter.labels(**self.labels).inc(data)

    def _log_histogram(self, histogram, data: Union[List[int], List[float]]) -> None:
        # Convenience function for logging to histogram.
        for value in data:
            histogram.labels(**self.labels).observe(value)

    def log_prometheus(self, stats: UCMStats):
        self._log_counter(
            self.counter_num_retrieve_requests, stats.interval_retrieve_requests
        )
        self._log_counter(
            self.counter_num_store_requests, stats.interval_store_requests
        )
        self._log_counter(
            self.counter_num_lookup_requests, stats.interval_lookup_requests
        )

        self._log_counter(
            self.counter_num_requested_tokens, stats.interval_requested_tokens
        )
        self._log_counter(self.counter_num_hit_tokens, stats.interval_hit_tokens)
        self._log_counter(self.counter_num_stored_tokens, stats.interval_stored_tokens)
        self._log_counter(self.counter_num_lookup_tokens, stats.interval_lookup_tokens)
        self._log_counter(self.counter_num_lookup_hits, stats.interval_lookup_hits)
        self._log_counter(self.counter_num_prompt_tokens, stats.interval_prompt_tokens)
        self._log_counter(
            self.counter_num_vllm_hit_tokens, stats.interval_vllm_hit_tokens
        )

        self._log_gauge(self.gauge_retrieve_hit_rate, stats.retrieve_hit_rate)

        self._log_gauge(self.gauge_lookup_hit_rate, stats.lookup_hit_rate)

        self._log_histogram(self.histogram_time_to_retrieve, stats.time_to_retrieve)

        self._log_histogram(self.histogram_time_to_store, stats.time_to_store)

        self._log_histogram(self.histogram_retrieve_speed, stats.retrieve_speed)

        self._log_histogram(self.histogram_store_speed, stats.store_speed)

        self._log_histogram(
            self.histogram_request_cache_hit_rate, stats.interval_lookup_hit_rates
        )

    _instance = None

    @staticmethod
    def GetOrCreate(vllm_config: "VllmConfig") -> "PrometheusLogger":
        if PrometheusLogger._instance is None:
            PrometheusLogger._instance = PrometheusLogger(vllm_config)
        return PrometheusLogger._instance

    @staticmethod
    def GetInstance() -> "PrometheusLogger":
        assert PrometheusLogger._instance is not None, (
            "PrometheusLogger instance not created yet"
        )
        return PrometheusLogger._instance

    @staticmethod
    def GetInstanceOrNone() -> Optional["PrometheusLogger"]:
        """
        Returns the singleton instance of PrometheusLogger if it exists,
        otherwise returns None.
        """
        return PrometheusLogger._instance


class UCMStatsLogger:
    def __init__(self, vllm_config: "VllmConfig", log_interval: int):
        self.log_interval = log_interval
        self.monitor = UCMStatsMonitor.GetOrCreate()
        self.prometheus_logger = PrometheusLogger.GetOrCreate(vllm_config)
        self.is_running = True

        self.thread = threading.Thread(target=self.log_worker, daemon=True)
        self.thread.start()

    def log_worker(self):
        while self.is_running:
            current_pid = os.getpid()
            logger.info(f"The current pid is {current_pid} for UCMStatsLogger")
            stats = self.monitor.get_stats_and_clear()
            self.prometheus_logger.log_prometheus(stats)
            #TODO continous usage context
            time.sleep(self.log_interval)

    def shutdown(self):
        self.is_running = False
        self.thread.join()
