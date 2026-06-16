"""
Prefix Cache hit rate calculation module
Query vLLM metrics API to get prefix cache hit statistics
"""

import logging
import re
import subprocess
from typing import Dict, List, Tuple

logging.getLogger().setLevel(logging.INFO)


def get_prefix_queries_total(
    ip_address: str, port: str
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Get query token count

    Args:
        ip_address: vLLM service IP address
        port: vLLM service port

    Returns:
        (normal_stats, external_stats) - Dict in {engine: tokens} format
        normal_stats: HBM prefix cache query count
        external_stats: External prefix cache query count
    """
    try:
        url = f"http://{ip_address}:{port}/metrics"
        command = f"unset http_proxy && unset https_proxy && sleep 3s && curl -s {url} | grep 'prefix_cache_queries_total' | grep 'model_name'"

        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=30
        )

        if result.returncode != 0 or not result.stdout.strip():
            return {}, {}

        lines = result.stdout.strip().split("\n")
        normal_stats = {}
        external_stats = {}

        for line in lines:
            engine_match = re.search(r'engine="(\d+)"', line)
            if not engine_match:
                continue

            engine = engine_match.group(1)

            parts = line.split()
            if len(parts) < 2:
                continue

            value_str = parts[-1]
            try:
                value = float(value_str)
                if value.is_integer():
                    value = int(value)
            except ValueError:
                continue

            if "external_prefix_cache_queries_total" in line:
                external_stats[int(engine)] = value
            elif "vllm:prefix_cache_queries_total" in line:
                normal_stats[int(engine)] = value

        logging.info(f"HBM queries: {normal_stats}, External queries: {external_stats}")
        return normal_stats, external_stats

    except Exception as e:
        logging.error(f"Error getting query token count: {e}")
        return {}, {}


def get_prefix_hits_total(
    ip_address: str, port: str
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Get hit token count

    Args:
        ip_address: vLLM service IP address
        port: vLLM service port

    Returns:
        (normal_stats, external_stats) - Dict in {engine: tokens} format
        normal_stats: HBM prefix cache hit count
        external_stats: External prefix cache hit count
    """
    try:
        url = f"http://{ip_address}:{port}/metrics"
        command = f"unset http_proxy && unset https_proxy && sleep 3s && curl -s {url} | grep 'prefix_cache_hits_total' | grep 'model_name'"

        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=30
        )

        if result.returncode != 0 or not result.stdout.strip():
            return {}, {}

        lines = result.stdout.strip().split("\n")
        normal_stats = {}
        external_stats = {}

        for line in lines:
            engine_match = re.search(r'engine="(\d+)"', line)
            if not engine_match:
                continue

            engine = engine_match.group(1)

            parts = line.split()
            if len(parts) < 2:
                continue

            value_str = parts[-1]
            try:
                value = float(value_str)
                if value.is_integer():
                    value = int(value)
            except ValueError:
                continue

            if "external_prefix_cache_hits_total" in line:
                external_stats[int(engine)] = value
            elif "vllm:prefix_cache_hits_total" in line:
                normal_stats[int(engine)] = value

        logging.info(f"HBM hits: {normal_stats}, External hits: {external_stats}")
        return normal_stats, external_stats

    except Exception as e:
        logging.error(f"Error getting hit token count: {e}")
        return {}, {}


def get_pod_metrics_info(pod_info: List[str]) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Get metrics info for all POD nodes

    Args:
        pod_info: POD list in ["ip:port", ...] format

    Returns:
        (query_tokens, query_tokens_external, hit_tokens, hit_tokens_external)
        Each is nested dict in {pod: {engine: tokens}} format
    """
    query_tokens = {}
    query_tokens_external = {}
    hit_tokens = {}
    hit_tokens_external = {}

    for pod in pod_info:
        ip, port = pod.split(":")
        query_tokens[pod], query_tokens_external[pod] = get_prefix_queries_total(
            ip, port
        )
        hit_tokens[pod], hit_tokens_external[pod] = get_prefix_hits_total(ip, port)

    return query_tokens, query_tokens_external, hit_tokens, hit_tokens_external


def cal_prefix_hit_info(
    query_tokens: Dict,
    query_tokens_external: Dict,
    hit_tokens: Dict,
    hit_tokens_external: Dict,
    query_tokens_new: Dict,
    query_tokens_external_new: Dict,
    hit_tokens_new: Dict,
    hit_tokens_external_new: Dict,
) -> Dict:
    """
    Calculate and print prefix cache hit rate info

    Args:
        Metrics data before and after test

    Returns:
        Hit rate statistics result dict
    """
    if (
        not query_tokens
        or not query_tokens_external
        or not hit_tokens
        or not hit_tokens_external
    ):
        return {}

    result = {}

    # Define column widths
    col1_width = 15
    col2_width = 20
    col3_width = 20
    col4_width = 20
    col5_width = 20

    total_width = col1_width + col2_width + col3_width + col4_width + col5_width + 8

    for pod, engines in sorted(query_tokens.items()):
        pod_result = {"pod": pod, "engines": []}

        print("\n" + "=" * total_width)
        print(f"POD: {pod}")
        print("=" * total_width)

        headers = [
            "engine_id",
            "hbm_hit_rate",
            "hbm(hit/query)",
            "external_hit_rate",
            "external(hit/query)",
        ]
        print(
            f"{headers[0]:<{col1_width}} {headers[1]:<{col2_width}} {headers[2]:<{col3_width}} {headers[3]:<{col4_width}} {headers[4]:<{col5_width}}"
        )
        print("-" * total_width)

        for engine_id, token in sorted(engines.items()):
            query_hbm = query_tokens_new[pod][engine_id] - query_tokens[pod][engine_id]
            hits_hbm = hit_tokens_new[pod][engine_id] - hit_tokens[pod][engine_id]
            query_ex = (
                query_tokens_external_new[pod][engine_id]
                - query_tokens_external[pod][engine_id]
            )
            hits_ex = (
                hit_tokens_external_new[pod][engine_id]
                - hit_tokens_external[pod][engine_id]
            )

            if query_hbm == 0:
                hit_rate_str = "0%"
                hit_detail = "0/0"
            else:
                hit_rate_str = format(hits_hbm / query_hbm, ".2%")
                hit_detail = f"{hits_hbm}/{query_hbm}"

            if query_ex == 0:
                hit_rate_ex_str = "0%"
                hit_ex_detail = "0/0"
            else:
                hit_rate_ex_str = format(hits_ex / query_ex, ".2%")
                hit_ex_detail = f"{hits_ex}/{query_ex}"

            engine_result = {
                "engine_id": str(engine_id),
                "hbm_hit_rate": hit_rate_str,
                "hbm_hits": hits_hbm,
                "hbm_queries": query_hbm,
                "external_hit_rate": hit_rate_ex_str,
                "external_hits": hits_ex,
                "external_queries": query_ex,
            }
            pod_result["engines"].append(engine_result)

            print(
                f"{engine_result['engine_id']:<{col1_width}} {hit_rate_str:<{col2_width}} {hit_detail:<{col3_width}} {hit_rate_ex_str:<{col4_width}} {hit_ex_detail:<{col5_width}}"
            )

        print("=" * total_width)
        result[pod] = pod_result

    return result
