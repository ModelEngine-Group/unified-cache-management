"""
AISBench Prefix Cache Performance Test Cases

Test model performance, prefix cache performance, dataset generation, etc.

Parameter passing methods:
1. Default config: Use default test_scenarios array
2. Environment variable: Set AISBENCH_TEST_CASE env var with JSON format test config
3. Config file: Configure in config.yaml aisbench_prefix_cache.test_scenarios

Example:
# Pass multiple test configs via environment variable
export AISBENCH_TEST_CASE='[
    {"input_len": 2048, "output_len": 2048, "data_num": 160, "concurrency": 40, "test_name": "2k_perf"},
    {"input_len": 4096, "output_len": 1024, "data_num": 80, "concurrency": 32, "dataset_type": "prefix_cache", "repeat_rate": "50%", "prefix_test": true, "test_name": "prefix_50pct"}
]'
pytest --feature=aisbench_prefix_cache
"""

import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml
from common.aisbench_utils import (
    AisbenchConfig,
    AisbenchResult,
    DataPicker,
    cal_prefix_hit_info,
    create_multi_prefix_dataset,
    generate_api_config,
    get_data,
    get_pod_metrics_info,
    parse_prefix_ratio,
    save_csv,
    save_log,
    symlink_force,
)
from common.capture_utils import export_vars, set_test_info
from common.config_utils import config_utils

# Get project root directory
PRJ_ROOT = Path(__file__).resolve().parents[2]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# ========== Test Scenario Configuration ==========

# Default test scenarios
default_test_scenarios: List[Dict[str, Any]] = [
    {
        "input_len": 512,
        "output_len": 512,
        "data_num": 10,
        "concurrency": 10,
        "request_rate": 10,  # int: 10 req/s (also accepts "10", 0=unlimited)
        "dataset_type": "prefix_cache",
        "repeat_rate": 0.5,  # float: 0.5 (also accepts 50, "50%", "0.5")
        "prefix_num": 1,
        "prefix_test": True,
        "dp": 1,
        "test_name": "prefix_50pct_dp1",
    }
]

# Load test config from environment variable
test_scenarios = default_test_scenarios.copy()

# Method 1: Environment variable AISBENCH_TEST_CASE
env_test_case_str = os.getenv("AISBENCH_TEST_CASE")
if env_test_case_str:
    try:
        parsed = json.loads(env_test_case_str)
        if isinstance(parsed, list) and len(parsed) > 0:
            # Validate and convert config
            valid_configs = []
            for item in parsed:
                if isinstance(item, dict):
                    try:
                        config = AisbenchConfig(**item)
                        valid_configs.append(config.to_dict())
                    except TypeError as e:
                        logging.warning(f"Invalid config item: {item}, error: {e}")
                        continue

            if valid_configs:
                test_scenarios = valid_configs
                logging.info(
                    f"Loaded {len(test_scenarios)} test configs from AISBENCH_TEST_CASE env var"
                )
            else:
                logging.warning(
                    "Environment variable parse failed, using default config"
                )
        else:
            logging.warning("Environment variable format invalid, using default config")
    except json.JSONDecodeError as e:
        logging.warning(f"JSON parse failed: {e}, using default config")
    except Exception as e:
        logging.warning(f"Parse error: {e}, using default config")
else:
    logging.info("AISBENCH_TEST_CASE env var not set, using default config")

# Method 2: Load from config.yaml
config_scenarios = config_utils.get_nested_config(
    "aisbench_prefix_cache.test_scenarios", []
)
if config_scenarios and isinstance(config_scenarios, list):
    try:
        valid_configs = []
        for item in config_scenarios:
            if isinstance(item, dict):
                config = AisbenchConfig(**item)
                valid_configs.append(config.to_dict())

        if valid_configs:
            test_scenarios = valid_configs
            logging.info(f"Loaded {len(test_scenarios)} test configs from config.yaml")
    except Exception as e:
        logging.warning(f"Config file load failed: {e}")

logging.info(f"Final test scenario count: {len(test_scenarios)}")
for i, scenario in enumerate(test_scenarios):
    logging.info(
        f"  [{i+1}] {scenario.get('test_name', 'unnamed')}: input={scenario.get('input_len')}, output={scenario.get('output_len')}, type={scenario.get('dataset_type')}"
    )


# Generate test IDs
scenario_ids = [
    s.get(
        "test_name",
        f"in_{s['input_len']}-out_{s['output_len']}-type_{s['dataset_type']}",
    )
    for s in test_scenarios
]


# ========== Test Runner Class ==========


class AisbenchTestRunner:
    """AISBench test runner"""

    def __init__(self, config: AisbenchConfig):
        self.config = config
        self.global_config = config_utils.read_config()

        # Get base config from global config
        aisbench_config = self.global_config.get("aisbench_prefix_cache", {})
        self.model_path = aisbench_config.get("model", {}).get("path", "")
        self.model_name = aisbench_config.get("model", {}).get("name", "")
        self.host_ip = aisbench_config.get("server", {}).get("host_ip", "")
        self.host_port = aisbench_config.get("server", {}).get("host_port", "")
        self.url = aisbench_config.get("server", {}).get(
            "url", ""
        )  # Support custom URL for domain-based connection
        self.work_path = aisbench_config.get("aisbench", {}).get(
            "work_path", "/home/benchmark"
        )
        self.dataset_path = aisbench_config.get("dataset", {}).get(
            "base_path", "/home/dataset"
        )
        self.gsm8k_path = aisbench_config.get("dataset", {}).get(
            "gsm8k_source", str(PRJ_ROOT.parent / "GSM8K.jsonl")
        )
        self.use_gsm8k = aisbench_config.get("dataset", {}).get(
            "use_gsm8k", True
        )  # Data source mode
        self.output_dir = aisbench_config.get("aisbench", {}).get(
            "output_dir", "./outputs/default"
        )
        self.pod_info = aisbench_config.get("pod_info", [])
        self.default_perf = aisbench_config.get("test", {}).get(
            "default_perf", "default_perf"
        )

    def generate_aisbench_command(self) -> str:
        """Generate AISBench test command"""
        if self.config.test_accuracy:
            return f"ais_bench --models vllm_api_chat_temp --datasets gsm8k_gen_0_shot_cot_str_perf --work-dir {self.output_dir} --dump-eval-details"
        else:
            base_cmd = f"ais_bench --models vllm_api_chat_temp --datasets gsm8k_gen_0_shot_cot_str_perf --mode perf --summarizer {self.default_perf} --work-dir {self.output_dir} --debug --num-warmups 0"
            if sys.platform == "win32":
                # Windows: redirect stdout+stderr to log file (tee not available on cmd.exe)
                return f"{base_cmd} > aisbench.log 2>&1"
            else:
                # Linux: use tee to output to both terminal and log file
                return f"{base_cmd} 2>&1 | tee aisbench.log"

    def setup_dataset_dir(self) -> str:
        """Setup dataset directory"""
        dst_dir = os.path.normpath(
            os.path.join(self.work_path, "ais_bench/datasets/gsm8k")
        )
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        train_dataset = os.path.join(dst_dir, "train.jsonl")
        if not os.path.exists(train_dataset):
            with open(train_dataset, "w") as f:
                pass

        return dst_dir

    def generate_dataset(self, dst_dir: str) -> tuple:
        """Generate test dataset"""
        repeat_rate = parse_prefix_ratio(self.config.repeat_rate)
        prefix_flag = 1 if self.config.dataset_type == "prefix_cache" else 0

        src_file_prefix, src_file_data = create_multi_prefix_dataset(
            tokenizer_path=self.model_path,
            input_len=self.config.input_len,
            number=self.config.data_num,
            save_path=self.dataset_path,
            prefix_flag=prefix_flag,
            dp=self.config.dp,
            repeat_rate=repeat_rate,
            seed=self.config.seed,
            prefix_num=self.config.prefix_num,
            length_mean=self.config.length_mean,
            length_std=self.config.length_std,
            length_min=self.config.length_min,
            length_max=self.config.length_max,
            gsm8k_path=self.gsm8k_path,
            use_gsm8k=self.use_gsm8k,
        )

        # Create symlink
        if self.config.dataset_type == "prefix_cache" and self.config.prefix_test:
            # Prefix warmup phase
            prefix_dst = os.path.join(dst_dir, "test.jsonl")
            symlink_force(src_file_prefix, prefix_dst)

        # Full dataset
        data_dst = os.path.join(dst_dir, "test.jsonl")
        symlink_force(src_file_data, data_dst)

        return src_file_prefix, src_file_data

    def run_prefix_warmup(
        self, src_file_prefix: str, dst_dir: str, ais_bench_cmd: str
    ) -> dict:
        """Run prefix warmup test"""
        # Build pod_info: prefer url if available, otherwise use host_ip:host_port
        if self.url:
            # Extract host from url for pod_info
            import re

            url_match = re.search(r"https?://([^:/]+):(\d+)", self.url)
            if url_match:
                pod_info = [f"{url_match.group(1)}:{url_match.group(2)}"]
            else:
                pod_info = [self.url]
        else:
            pod_info = (
                self.pod_info if self.pod_info else [f"{self.host_ip}:{self.host_port}"]
            )

        logging.info(f"[Start] Prefix warmup test")

        # Generate API config (dp concurrency, output 1 token)
        generate_api_config(
            model_path=self.model_path,
            model_name=self.model_name,
            concurrency=self.config.dp,
            output_len=1,
            request_rate=self.config.request_rate,
            host_ip=self.host_ip,
            host_port=self.host_port,
            url=self.url,
            test_type=self.config.test_type,
            enable_think=self.config.enable_think,
            test_accuracy=self.config.test_accuracy,
            work_path=self.work_path,
        )

        # Create symlink
        symlink_force(src_file_prefix, os.path.join(dst_dir, "test.jsonl"))

        # Get metrics before test
        query_tokens, query_tokens_external, hit_tokens, hit_tokens_external = (
            get_pod_metrics_info(pod_info)
        )

        # Execute test
        logging.info(f"Running prefix warmup test with command: {ais_bench_cmd}")
        os.system(ais_bench_cmd)

        # Get metrics after test
        (
            query_tokens_new,
            query_tokens_external_new,
            hit_tokens_new,
            hit_tokens_external_new,
        ) = get_pod_metrics_info(pod_info)
        hit_info = cal_prefix_hit_info(
            query_tokens,
            query_tokens_external,
            hit_tokens,
            hit_tokens_external,
            query_tokens_new,
            query_tokens_external_new,
            hit_tokens_new,
            hit_tokens_external_new,
        )

        logging.info(f"[Done] Prefix warmup test")

        return hit_info

    def run_full_test(
        self, src_file_data: str, dst_dir: str, ais_bench_cmd: str
    ) -> tuple:
        """Run full dataset test"""
        # Build pod_info: prefer url available, else use host_ip/host_port
        if self.url:
            # Extract host for pod_info from URL (e.g., http://api.example.com:8080 -> api.example.com:8080)
            import re

            url_match = re.search(r"https?://([^/:]+):(\d+)", self.url)
            if url_match:
                pod_info = [f"{url_match.group(1)}:{url_match.group(2)}"]
            else:
                pod_info = [self.url]
        else:
            pod_info = (
                self.pod_info if self.pod_info else [f"{self.host_ip}:{self.host_port}"]
            )

        logging.info(f"[Start] Full dataset test")

        # Get metrics before test
        query_tokens, query_tokens_external, hit_tokens, hit_tokens_external = (
            get_pod_metrics_info(pod_info)
        )

        # Generate API config
        generate_api_config(
            model_path=self.model_path,
            model_name=self.model_name,
            concurrency=self.config.concurrency,
            output_len=self.config.output_len,
            request_rate=self.config.request_rate,
            host_ip=self.host_ip,
            host_port=self.host_port,
            url=self.url,
            test_type=self.config.test_type,
            enable_think=self.config.enable_think,
            test_accuracy=self.config.test_accuracy,
            work_path=self.work_path,
        )

        # Create symlink
        symlink_force(src_file_data, os.path.join(dst_dir, "test.jsonl"))

        # Execute test
        logging.info(f"Executing AISBench command: {ais_bench_cmd}")
        os.system(ais_bench_cmd)

        # Get metrics after test
        (
            query_tokens_new,
            query_tokens_external_new,
            hit_tokens_new,
            hit_tokens_external_new,
        ) = get_pod_metrics_info(pod_info)
        hit_info = cal_prefix_hit_info(
            query_tokens,
            query_tokens_external,
            hit_tokens,
            hit_tokens_external,
            query_tokens_new,
            query_tokens_external_new,
            hit_tokens_new,
            hit_tokens_external_new,
        )

        # Parse results
        perf_result, log_dir = get_data(
            "aisbench.log", self.config.request_rate, self.config.npu_num
        )
        save_log("aisbench.log", log_dir)
        save_csv(perf_result, "aisbench_result.csv")

        logging.info(f"[Done] Full dataset test")

        return perf_result, hit_info

    def run(self) -> AisbenchResult:
        """Execute full test flow"""
        result = AisbenchResult(
            test_name=self.config.test_name,
            input_len=self.config.input_len,
            output_len=self.config.output_len,
            data_num=self.config.data_num,
            concurrency=self.config.concurrency,
            request_rate=self.config.request_rate,
            dataset_type=self.config.dataset_type,
            repeat_rate=self.config.repeat_rate,
        )

        try:
            dst_dir = self.setup_dataset_dir()
            ais_bench_cmd = self.generate_aisbench_command()

            src_file_prefix, src_file_data = self.generate_dataset(dst_dir)

            # Prefix Cache mode with warmup
            if self.config.dataset_type == "prefix_cache" and self.config.prefix_test:
                self.run_prefix_warmup(src_file_prefix, dst_dir, ais_bench_cmd)

            # Run full test
            perf_result, hit_info = self.run_full_test(
                src_file_data, dst_dir, ais_bench_cmd
            )

            # Fill results
            if perf_result and len(perf_result) >= 20:
                result.ttft_avg = perf_result[7]
                result.ttft_p90 = perf_result[8]
                result.tpot_avg = perf_result[9]
                result.tpot_p90 = perf_result[10]
                result.total_time = perf_result[11]
                result.output_throughput = perf_result[12]
                result.single_output_throughput = perf_result[13]
                result.e2e_throughput = perf_result[14]
                result.single_e2e_throughput = perf_result[15]
                result.qps = perf_result[16]
                result.qpm = perf_result[17]
                result.input_token_throughput = perf_result[18]
                result.prefill_throughput = perf_result[19]
                result.total_input_tokens = perf_result[1]
                result.total_output_tokens = perf_result[2]
                result.total_requests = perf_result[3]

            # Fill hit rate info
            if hit_info:
                for pod, pod_data in hit_info.items():
                    if pod_data.get("engines"):
                        engine = pod_data["engines"][0]
                        result.hbm_hit_rate = engine.get("hbm_hit_rate", "")
                        result.hbm_hits = engine.get("hbm_hits", 0)
                        result.hbm_queries = engine.get("hbm_queries", 0)
                        result.external_hit_rate = engine.get("external_hit_rate", "")
                        result.external_hits = engine.get("external_hits", 0)
                        result.external_queries = engine.get("external_queries", 0)
                        break

            result.status = "pass"

        except Exception as e:
            logging.error(f"Test execution failed: {e}")
            result.status = "failed"

        return result


# ========== Test Cases ==========


@pytest.mark.stage(2)
@pytest.mark.feature("aisbench_prefix_cache")
@pytest.mark.platform("npu")
@pytest.mark.parametrize("test_config", test_scenarios, ids=scenario_ids)
@export_vars
def test_aisbench_prefix_cache(test_config: Dict[str, Any]):
    """
    AISBench Prefix Cache Performance Test

    Supports multiple test scenarios:
    - Normal dataset performance test
    - Prefix Cache performance test (with warmup)
    - Variable-length dataset test
    - Accuracy test

    Parameter passing methods:
    1. Default config
    2. Environment variable AISBENCH_TEST_CASE (JSON format)
    3. config.yaml configuration file

    Args:
        test_config: Test configuration dict containing all test parameters
    """
    logging.info(
        f"========== Start Test: {test_config.get('test_name', 'unnamed')} ========== "
    )
    logging.info(f"Config params: {json.dumps(test_config, indent=2)}")

    # Create config object
    config = AisbenchConfig(**test_config)

    # Execute test
    runner = AisbenchTestRunner(config)
    result = runner.run()

    logging.info(
        f"========== Test Done: {result.test_name}, Status: {result.status} ========== "
    )

    # Return results for data export
    return {"_name": "aisbench_prefix_cache_result", "_data": result.to_dict()}
