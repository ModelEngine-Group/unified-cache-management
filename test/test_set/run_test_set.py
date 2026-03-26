#!/usr/bin/env python3
"""
Test Set Runner - 通过 Jenkins Pipeline 执行测试套件。

独立工具，不依赖项目其他模块。
依赖: pyyaml, requests

Usage:
    python run_test_set.py                          # 使用默认 run_config.yaml
    python run_test_set.py -c my_config.yaml        # 指定配置文件
    python run_test_set.py --dry-run                # 仅打印参数，不触发
    python run_test_set.py --list                   # 列出所有可用测试
    python run_test_set.py -t vllm_cuda_qwen3_pc    # 只运行指定测试
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import yaml

# Import from co-located jenkins_sdk.py
from jenkins_sdk import (
    BuildInfo,
    JenkinsPipelineClient,
    PipelineParameters,
)

# ---------------------------------------------------------------------------
# Platform mapping: Docker suffix (test_set) → Jenkins PLATFORM
# ---------------------------------------------------------------------------
PLATFORM_MAP: Dict[str, str] = {
    "vllm_gpu": "vllm-cuda",
    "vllm_npu": "vllm-ascend",
    "mindie_llm": "mindie",
    "sglang_gpu": "sglang-cuda",
}


# ---------------------------------------------------------------------------
# Image URL resolution
# ---------------------------------------------------------------------------


def get_image_url(platform: str, filter_string: str, harbor_cfg: dict) -> str:
    """从 Harbor 查询匹配的镜像 URL。

    Args:
        platform: Docker 后缀形式的平台名
        filter_string: 用于过滤 tag 的字符串（大小写不敏感包含匹配）
        harbor_cfg: Harbor 配置字典，包含 url, auth_token, project
    """
    harbor_url = harbor_cfg["url"]
    auth_token = harbor_cfg["auth_token"]
    project = harbor_cfg["project"]
    repo = "ucm-" + platform

    api_url = (
        f"{harbor_url}/api/v2.0/projects/{project}/repositories/{repo}"
        f"/artifacts?with_tag=true&page_size=100"
    )

    headers = {
        "Authorization": f"Basic {auth_token}",
        "Accept": "application/json",
    }

    response = requests.get(
        api_url,
        headers=headers,
        verify=False,
        timeout=(5, 10),
    )
    response.raise_for_status()

    artifacts = response.json()

    image_urls = []
    registry_host = harbor_url.replace("https://", "").replace("http://", "")

    for artifact in artifacts:
        tags = artifact.get("tags")
        if isinstance(tags, list):
            for tag in tags:
                tag_name = tag.get("name") if tag else None
                if tag_name:
                    tag_name = str(tag_name)
                    if not filter_string or filter_string.lower() in tag_name.lower():
                        full_url = f"{registry_host}/{project}/{platform}:{tag_name}"
                        image_urls.append(full_url)

    if not image_urls:
        raise ValueError(
            f"No matching image found for platform {platform} with filter {filter_string}"
        )

    return image_urls[0]


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_run_config(config_path: str) -> dict:
    """加载 run_config.yaml 并返回字典。"""
    if not os.path.isfile(config_path):
        print(f"Error: config file not found: {config_path}")
        sys.exit(1)
    return load_yaml(config_path)


def load_test_sets(test_set_config_path: str, base_dir: str) -> List[dict]:
    """加载 test_set 配置文件，返回 test_sets 列表。"""
    if not os.path.isabs(test_set_config_path):
        test_set_config_path = os.path.join(base_dir, test_set_config_path)
    if not os.path.isfile(test_set_config_path):
        print(f"Error: test set config not found: {test_set_config_path}")
        sys.exit(1)
    data = load_yaml(test_set_config_path)
    return data.get("test_sets", [])


# ---------------------------------------------------------------------------
# Build PipelineParameters from test_set config
# ---------------------------------------------------------------------------
def build_pipeline_params(
    test_set: dict,
    jenkins_platform: str,
    override_image_url: str,
) -> PipelineParameters:
    """将 test_set 配置项转换为 PipelineParameters。"""
    server_cfg = test_set.get("server_start_config", {})
    pytest_cfg = test_set.get("pytest_config", {})

    node_count = server_cfg.get("node_count", 1)
    deploy_mode = "multi" if node_count > 1 else "single"

    return PipelineParameters(
        BUILD_NAME=test_set.get("name", ""),
        PLATFORM=jenkins_platform,
        OVERRIDE_IMAGE=override_image_url,
        GPU_COUNT=str(server_cfg.get("gpu_count", "1")),
        DEPLOY_MODE=deploy_mode,
        SERVER_PORT=str(server_cfg.get("server_port", "9527")),
        VLLM_COMMAND_MASTER=server_cfg.get("master_start_command", ""),
        VLLM_COMMAND_WORKER=server_cfg.get("slave_start_command", ""),
        UCM_CONFIG_YAML=server_cfg.get("ucm_config", ""),
        API_MODEL_NAME=pytest_cfg.get("api_model_name", ""),
        MODEL_FOLDER_NAME=pytest_cfg.get("api_model_name", ""),
        TEST_PARAMS=pytest_cfg.get("test_params", ""),
    )


# ---------------------------------------------------------------------------
# Determine image tag for a test set
# ---------------------------------------------------------------------------
def get_override_image(
    platform: str,
    override_images: Dict[str, str],
    package_name: str,
    harbor_cfg: dict,
) -> str:
    """确定某个平台的完整镜像 URL。

    优先使用 override_image 配置，否则通过 get_image_url 从 Harbor 查询。
    """
    if platform in override_images:
        return override_images[platform]
    if not package_name:
        print(f"Error: no package_name or override_image for platform '{platform}'")
        sys.exit(1)
    return get_image_url(platform, package_name, harbor_cfg)


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------
def format_duration(ms: int) -> str:
    """将毫秒转换为可读的时间格式。"""
    seconds = ms // 1000
    minutes = seconds // 60
    seconds = seconds % 60
    if minutes > 0:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def print_results(results: List[Tuple[str, str, BuildInfo]]) -> int:
    """打印测试结果汇总表。返回失败数量。"""
    if not results:
        print("No test results.")
        return 0

    # Calculate column widths
    name_width = max(len(r[0]) for r in results)
    name_width = max(name_width, len("Test Set"))
    plat_width = max(len(r[1]) for r in results)
    plat_width = max(plat_width, len("Platform"))

    header = f"  {'Test Set':<{name_width}}  {'Platform':<{plat_width}}  {'Status':<10}  {'Duration':<10}"
    sep = (
        "  "
        + "-" * name_width
        + "  "
        + "-" * plat_width
        + "  "
        + "-" * 10
        + "  "
        + "-" * 10
    )

    print()
    print(header)
    print(sep)

    passed = 0
    for name, platform, info in results:
        status = info.status.value
        duration = format_duration(info.duration_ms)
        marker = "✓" if info.is_successful else "✗"
        print(
            f"  {name:<{name_width}}  {platform:<{plat_width}}  {marker} {status:<8}  {duration:<10}"
        )
        if info.is_successful:
            passed += 1

    print(sep)
    total = len(results)
    failed = total - passed
    print(f"  Result: {passed}/{total} PASSED", end="")
    if failed > 0:
        print(f" ({failed} FAILED)")
    else:
        print()
    print()

    return failed


def print_trigger_results(triggered: List[Tuple[str, str, int]]):
    """打印触发结果（--no-wait 模式）。"""
    print()
    print(f"  {'Test Set':<30}  {'Platform':<15}  {'Build #':<10}")
    print(f"  {'-'*30}  {'-'*15}  {'-'*10}")
    for name, platform, build_num in triggered:
        print(f"  {name:<30}  {platform:<15}  #{build_num}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test Set Runner - 通过 Jenkins Pipeline 执行测试套件",
    )
    parser.add_argument(
        "-c",
        "--config",
        default="run_config.yaml",
        help="run_config.yaml 路径 (默认: run_config.yaml)",
    )
    parser.add_argument(
        "--test-set-config",
        default=None,
        help="覆盖 test_set_config 路径",
    )
    parser.add_argument(
        "--package-name",
        default=None,
        help="覆盖 package_name",
    )
    parser.add_argument(
        "--override-image",
        action="append",
        default=[],
        metavar="PLATFORM=IMAGE",
        help="覆盖指定平台的镜像，如 vllm_gpu=abc.io/vllm:v2 (可多次指定)",
    )
    parser.add_argument(
        "-t",
        "--test",
        action="append",
        default=[],
        metavar="NAME",
        help="只运行指定名称的测试 (可多次指定)",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="串行执行测试",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="只触发不等待结果",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印将要执行的参数，不实际触发",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_tests",
        help="列出所有可用的测试名称",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # Resolve config path
    config_path = os.path.abspath(args.config)
    base_dir = os.path.dirname(config_path)

    # Load run config
    run_config = load_run_config(config_path)

    # Determine test_set_config path
    test_set_config_path = args.test_set_config or run_config.get(
        "test_set_config", "test_set_config/test_set.yaml"
    )

    # Load test sets
    test_sets = load_test_sets(test_set_config_path, base_dir)

    # --list: just show available tests and exit
    if args.list_tests:
        print("Available test sets:")
        for ts in test_sets:
            platform = ts.get("platform", "unknown")
            jenkins_plat = PLATFORM_MAP.get(platform, f"UNKNOWN({platform})")
            print(f"  - {ts['name']}  (platform: {platform} → {jenkins_plat})")
        return

    # Filter by --test if specified
    if args.test:
        selected_names = set(args.test)
        available_names = {ts["name"] for ts in test_sets}
        unknown = selected_names - available_names
        if unknown:
            print(f"Error: unknown test set(s): {', '.join(sorted(unknown))}")
            print(f"Available: {', '.join(sorted(available_names))}")
            sys.exit(1)
        test_sets = [ts for ts in test_sets if ts["name"] in selected_names]

    if not test_sets:
        print("No test sets to run.")
        return

    # Merge CLI overrides into config
    test_build = run_config.get("test_build", {})
    package_name = args.package_name or test_build.get("package_name", "")

    override_images: Dict[str, str] = test_build.get("override_image", {}) or {}
    for item in args.override_image:
        if "=" not in item:
            print(
                f"Error: invalid --override-image format: {item} (expected PLATFORM=IMAGE)"
            )
            sys.exit(1)
        plat, img = item.split("=", 1)
        override_images[plat] = img

    parallel = run_config.get("parallel", True)
    if args.no_parallel:
        parallel = False

    harbor_cfg = run_config.get("harbor", {})

    # Build parameters for each test set
    jobs: List[Tuple[str, str, PipelineParameters]] = []
    for ts in test_sets:
        platform = ts.get("platform", "")
        jenkins_platform = PLATFORM_MAP.get(platform)
        if jenkins_platform is None:
            print(f"Error: unknown platform '{platform}' in test set '{ts['name']}'")
            print(f"Known platforms: {', '.join(PLATFORM_MAP.keys())}")
            sys.exit(1)

        override_image_url = get_override_image(
            platform, override_images, package_name, harbor_cfg
        )
        params = build_pipeline_params(ts, jenkins_platform, override_image_url)
        jobs.append((ts["name"], jenkins_platform, params))

    # --dry-run: print parameters and exit
    if args.dry_run:
        print(f"Dry run: {len(jobs)} test(s) would be triggered\n")
        for name, platform, params in jobs:
            print(f"--- {name} (platform: {platform}) ---")
            for k, v in asdict(params).items():
                display_v = v if len(str(v)) <= 80 else str(v)[:77] + "..."
                print(f"  {k}: {display_v}")
            print()
        return

    # Validate Jenkins config
    jenkins_cfg = run_config.get("jenkins", {})
    required_keys = ["url", "username", "api_token", "job_name"]
    missing = [k for k in required_keys if not jenkins_cfg.get(k)]
    if missing:
        print(f"Error: missing Jenkins config keys: {', '.join(missing)}")
        print("Please configure 'jenkins' section in run_config.yaml")
        sys.exit(1)

    # Create Jenkins client
    client = JenkinsPipelineClient(
        jenkins_url=jenkins_cfg["url"],
        username=jenkins_cfg["username"],
        api_token=jenkins_cfg["api_token"],
        job_name=jenkins_cfg["job_name"],
        branch=jenkins_cfg.get("branch", "jenkins-dev-yhq"),
    )

    # Trigger builds
    print(f"Triggering {len(jobs)} test(s)...\n")
    triggered: List[Tuple[str, str, int]] = []

    def trigger_one(
        name: str, platform: str, params: PipelineParameters
    ) -> Tuple[str, str, int]:
        print(f"  Triggering: {name} ({platform})")
        build_number = client.trigger(params)
        return (name, platform, build_number)

    if parallel and len(jobs) > 1:
        with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
            futures = {
                executor.submit(trigger_one, name, plat, params): name
                for name, plat, params in jobs
            }
            for future in as_completed(futures):
                triggered.append(future.result())
    else:
        for name, plat, params in jobs:
            triggered.append(trigger_one(name, plat, params))

    # --no-wait: print build numbers and exit
    if args.no_wait:
        print_trigger_results(triggered)
        print("Builds triggered. Use Jenkins UI to monitor progress.")
        return

    # Wait for all builds and collect results
    print(f"\nWaiting for {len(triggered)} build(s) to complete...\n")
    results: List[Tuple[str, str, BuildInfo]] = []

    def wait_one(
        name: str, platform: str, build_number: int
    ) -> Tuple[str, str, BuildInfo]:
        info = client.wait_for_completion(build_number)
        return (name, platform, info)

    if parallel and len(triggered) > 1:
        with ThreadPoolExecutor(max_workers=len(triggered)) as executor:
            futures = {
                executor.submit(wait_one, name, plat, bnum): name
                for name, plat, bnum in triggered
            }
            for future in as_completed(futures):
                result = future.result()
                print(f"  Completed: {result[0]} → {result[2].status.value}")
                results.append(result)
    else:
        for name, plat, bnum in triggered:
            result = wait_one(name, plat, bnum)
            print(f"  Completed: {result[0]} → {result[2].status.value}")
            results.append(result)

    # Print summary
    failed_count = print_results(results)
    sys.exit(1 if failed_count > 0 else 0)


if __name__ == "__main__":
    main()
