"""
AISBench API configuration generation module
Generate AISBench test API configuration file based on template
"""

import errno
import logging
import os
import re
import shutil
import sys
from typing import Optional

logging.getLogger().setLevel(logging.INFO)


def symlink_force(target: str, link_name: str):
    """
    Force create symlink or copy on Windows (delete existing target first if needed)

    On Linux: uses os.symlink for symbolic links.
    On Windows: uses shutil.copy2 because os.symlink requires admin privileges
                or developer mode, and may fail with WinError 4392.

    Args:
        target: Target path (source file)
        link_name: Link name (destination path)
    """
    if sys.platform == "win32":
        # Windows: use file copy instead of symlink
        logging.info(f"copy file: {link_name} ==> {target}")
        if os.path.exists(link_name):
            os.remove(link_name)
        # Ensure destination directory exists
        os.makedirs(os.path.dirname(link_name), exist_ok=True)
        shutil.copy2(target, link_name)
    else:
        # Linux: use symlink
        logging.info(f"make symlink: {link_name} ==> {target}")
        try:
            os.symlink(target, link_name)
        except OSError as e:
            if e.errno == errno.EEXIST:
                os.remove(link_name)
                os.symlink(target, link_name)
            else:
                raise e


def generate_api_config(
    model_path: str,
    model_name: str,
    concurrency: int,
    output_len: int,
    request_rate: str,
    host_ip: str = "",
    host_port: str = "",
    url: str = "",
    test_type: str = "stream",
    enable_think: bool = False,
    test_accuracy: bool = False,
    work_path: str = "/home/benchmark",
    test_abbr: Optional[str] = None,
) -> str:
    """
    Generate AISBench API configuration file

    Supports two connection modes:
    1. host_ip + host_port: For IP address based connection (validated as IPv4/IPv6)
    2. url: For domain name or custom URL path (host_ip/host_port will be ignored)

    Args:
        model_path: Model weights path
        model_name: Service model name
        concurrency: Maximum concurrency
        output_len: Output token length
        request_rate: Request rate
        host_ip: Service IP address (optional if url is provided)
        host_port: Service port (optional if url is provided)
        url: Custom URL path for domain-based connection (e.g., "http://api.example.com:8080")
             When url is set, host_ip and host_port will be ignored
        test_type: Test type (stream/text)
        enable_think: Whether to enable thinking mode (DeepSeek V3.1)
        test_accuracy: Whether to test accuracy
        work_path: AISBench work path
        test_abbr: Test abbreviation name

    Returns:
        Generated temporary config file path
    """
    # Distinguish stream and non-stream
    if test_type == "text":
        api_test_type = "VLLMCustomAPIChat"
        api_test_abbr = test_abbr or "vllm-api-general-chat"
    elif test_type == "stream":
        api_test_type = "VLLMCustomAPIChatStream"
        api_test_abbr = test_abbr or "vllm-api-stream-chat"
    else:
        api_test_type = "VLLMCustomAPIChatStream"
        api_test_abbr = test_abbr or "vllm-api-stream-chat"

    # Generate generation_kwargs
    if test_accuracy:
        generation_kwargs = "temperature=0.6,\n\t\t\ttop_p = 0.95"
    else:
        generation_kwargs = "temperature=0,\n\t\t\tignore_eos=True"

    if enable_think:
        generation_kwargs = (
            generation_kwargs
            + ',\n\t\t\tchat_template_kwargs={"enable_thinking": True}'
        )

    # Build host configuration based on url or host_ip/host_port
    if url:
        # Use custom URL (supports domain names)
        host_config = f'url="{url}"'
        logging.info(f"Using custom URL: {url}")
    else:
        # Use host_ip + host_port (must be valid IP address)
        host_config = f'host_ip="{host_ip}",\n        host_port={host_port}'
        logging.info(f"Using host_ip: {host_ip}, host_port: {host_port}")

    # API config template content
    template_content = f"""from ais_bench.benchmark.models import {api_test_type}

models = [
    dict(
        attr="service",
        type={api_test_type},
        abbr='{api_test_abbr}',
        path="{model_path}",
        model="{model_name}",
        request_rate={request_rate},
        retry=2,
        {host_config},
        max_out_len={output_len},
        batch_size={concurrency},
        generation_kwargs=dict(
            {generation_kwargs.expandtabs(4)}
        )
    )
]
"""

    # Write to temporary config file
    temp_api_path = os.path.join(os.getcwd(), "temp_api.py")
    with open(temp_api_path, "w", encoding="utf-8") as f:
        f.write(template_content)

    logging.info(f"API config file generated: {temp_api_path}")
    logging.info(f"Model path: {model_path}")
    logging.info(f"Model name: {model_name}")
    logging.info(f"Concurrency: {concurrency}")
    logging.info(f"Output len: {output_len}")

    # Create symlink to AISBench config directory
    target_path = os.path.normpath(
        os.path.join(
            work_path,
            "ais_bench/benchmark/configs/models/vllm_api/vllm_api_chat_temp.py",
        )
    )
    symlink_force(temp_api_path, target_path)

    return temp_api_path


def modify_aisbench_api_from_template(
    template_path: str,
    model_path: str,
    model_name: str,
    concurrency: int,
    output_len: int,
    request_rate: str,
    host_ip: str = "",
    host_port: str = "",
    url: str = "",
    test_type: str = "stream",
    enable_think: bool = False,
    test_accuracy: bool = False,
    work_path: str = "/home/benchmark",
    test_abbr: Optional[str] = None,
) -> str:
    """
    Generate AISBench API config from template file (using regex substitution)

    Supports two connection modes:
    1. host_ip + host_port: For IP address based connection
    2. url: For domain name or custom URL path (host_ip/host_port will be ignored)

    Args:
        template_path: Template file path
        Other args same as generate_api_config

    Returns:
        Generated temporary config file path
    """
    # Distinguish stream and non-stream
    if test_type == "text":
        api_test_type = "VLLMCustomAPIChat"
        api_test_abbr = test_abbr or "vllm-api-general-chat"
    elif test_type == "stream":
        api_test_type = "VLLMCustomAPIChatStream"
        api_test_abbr = test_abbr or "vllm-api-stream-chat"
    else:
        api_test_type = "VLLMCustomAPIChatStream"
        api_test_abbr = test_abbr or "vllm-api-stream-chat"

    # Read template file
    temp_api_path = os.path.join(os.getcwd(), "temp_api.py")

    with open(template_path, "r", encoding="utf-8") as file_default:
        content = file_default.read()

    # Use regex substitution
    content = re.sub("model_path_for_replace", model_path, content)
    content = re.sub("model_name_for_replace", model_name, content)
    content = re.sub("rr_for_replace", request_rate, content)
    content = re.sub("test_type_for_replace", api_test_type, content)
    content = re.sub("test_abbr_for_replace", api_test_abbr, content)

    # Handle host configuration based on url or host_ip/host_port
    if url:
        # Use url parameter (supports domain names)
        content = re.sub("url_for_replace", url, content)
        # Remove host_ip/host_port lines if present in template
        content = re.sub(
            r'host_ip="ip_for_replace",\s*\n\s*host_port=port_for_replace,',
            'url="url_for_replace",',
            content,
        )
        logging.info(f"Using custom URL: {url}")
    else:
        # Use host_ip + host_port
        content = re.sub("ip_for_replace", host_ip, content)
        content = re.sub("port_for_replace", host_port, content)
        # Remove url line if present in template
        content = re.sub(
            r'url="url_for_replace",',
            f'host_ip="{host_ip}",\n        host_port={host_port},',
            content,
        )
        logging.info(f"Using host_ip: {host_ip}, host_port: {host_port}")

    content = re.sub("outputlen_for_replace", str(output_len), content)
    content = re.sub("concurrency_for_replace", str(concurrency), content)

    # Generate generation_kwargs
    if test_accuracy:
        generation_kwargs = "temperature=0.6,\n\t\t\ttop_p = 0.95"
    else:
        generation_kwargs = "temperature=0,\n\t\t\tignore_eos=True"

    if enable_think:
        generation_kwargs = (
            generation_kwargs
            + ',\n\t\t\tchat_template_kwargs={"enable_thinking": True}'
        )

    content = re.sub(
        "generation_kwargs_for_replace", generation_kwargs.expandtabs(4), content
    )

    # Write to temporary file
    with open(temp_api_path, "w", encoding="utf-8") as file_temp:
        file_temp.write(content)

    logging.info(f"API config file generated from template: {temp_api_path}")

    # Create symlink to AISBench config directory
    target_path = os.path.normpath(
        os.path.join(
            work_path,
            "ais_bench/benchmark/configs/models/vllm_api/vllm_api_chat_temp.py",
        )
    )
    symlink_force(temp_api_path, target_path)

    return temp_api_path
