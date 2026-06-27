"""
AISBench result parser module
Parse aisbench log file, extract performance metrics and save results
"""

import logging
import os
import re
import shutil
import traceback
from datetime import datetime
from typing import List, Tuple

import pandas as pd

logging.getLogger().setLevel(logging.INFO)


def get_data(aisbench_log: str, req_rate: str, npu_num: int) -> Tuple[List, str]:
    """
    Parse performance data from aisbench log file

    Args:
        aisbench_log: aisbench log file path
        req_rate: Request rate
        npu_num: Number of NPU cards

    Returns:
        (performance data list, log directory)
    """
    log_dir = ""
    default_values = [99999] * 20
    default_values[5] = 0  # req_rate default to 0

    try:
        with open(aisbench_log, "r") as f_streaming:
            txt = f_streaming.readlines()

            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            Total_InputTokens = 99999
            Total_GeneratedTokens = 99999
            Total_requests = 99999
            max_Concurrency = 99999
            Concurrency = 99999
            request_rate = float(req_rate) if req_rate else 0
            AVG_first_token_time = 99999
            slo_p90_first_token_time = 99999
            AVG_token_time = 99999
            slo_p90_token_time = 99999
            total_time = 99999
            GenerateSpeed = 99999
            single_generatespeed = 9999
            e2e_throughput = 9999
            single_e2e_throughput = 9999
            qps = 9999
            qpm = 9999
            input_token_throughput = 9999
            prefill_throughput = 9999

            for i in range(len(txt)):
                line = txt[i]

                if "Current exp folder" in line:
                    matches = re.findall(r"[\w']+", line)
                    log_dir = "/".join(matches[-3:])

                if "TTFT" in line:
                    matches = re.findall(r"(\d+\.\d+)", line)
                    if len(matches) >= 6:
                        slo_p90_first_token_time = float(matches[5])
                        AVG_first_token_time = float(matches[0])

                if "TPOT" in line:
                    matches = re.findall(r"(\d+\.\d+)", line)
                    if len(matches) >= 6:
                        slo_p90_token_time = float(matches[5])
                        AVG_token_time = float(matches[0])

                if "Benchmark Duration" in line:
                    matches = re.findall(r"(\d+\.\d+)", line)
                    if matches:
                        total_time = float(matches[0]) / 1000

                if "Concurrency" in line:
                    matches = re.findall(r"(\d+\.\d+)", line)
                    if matches:
                        Concurrency = float(matches[0])

                if "Max Concurrency" in line:
                    matches = re.findall(r"[\w']+", line)
                    max_Concurrency = matches[-1]

                if "Output Token Throughput" in line:
                    matches = re.findall(r"(\d+\.\d+)", line)
                    if matches:
                        GenerateSpeed = float(matches[0])
                        single_generatespeed = GenerateSpeed / npu_num

                if "Input Token Throughput" in line:
                    matches = re.findall(r"(\d+\.\d+)", line)
                    if matches:
                        input_token_throughput = float(matches[0])

                if "Total Token Throughput" in line:
                    matches = re.findall(r"(\d+\.\d+)", line)
                    if matches:
                        e2e_throughput = float(matches[0])
                        single_e2e_throughput = e2e_throughput / npu_num

                if "InputTokens" in line:
                    matches = re.findall(r"(\d+\.?\d*)", line)
                    if matches:
                        Total_InputTokens = float(matches[0])

                if "OutputTokens" in line:
                    matches = re.findall(r"(\d+\.?\d*)", line)
                    if matches:
                        Total_GeneratedTokens = float(matches[0])

                if "Total Requests" in line:
                    matches = re.findall(r"(\d+\.?\d*)", line)
                    if matches:
                        Total_requests = float(matches[0])

                if "Request Throughput" in line:
                    matches = re.findall(r"(\d+\.\d+)", line)
                    if matches:
                        qps = float(matches[0])
                        qpm = qps * 60

                if "Prefill Token Throughput" in line:
                    matches = re.findall(r"(\d+\.\d+)", line)
                    if matches:
                        prefill_throughput = float(matches[0])

            perf_result = [
                current_time,
                Total_InputTokens,
                Total_GeneratedTokens,
                Total_requests,
                max_Concurrency,
                Concurrency,
                request_rate,
                AVG_first_token_time,
                slo_p90_first_token_time,
                AVG_token_time,
                slo_p90_token_time,
                total_time,
                GenerateSpeed,
                single_generatespeed,
                e2e_throughput,
                single_e2e_throughput,
                qps,
                qpm,
                input_token_throughput,
                prefill_throughput,
            ]

    except Exception as e:
        logging.warning(traceback.format_exc())
        perf_result = default_values

    return perf_result, log_dir


def save_log(aisbench_log: str, log_dir: str):
    """
    Save aisbench log file

    Args:
        aisbench_log: aisbench log file path
        log_dir: Target directory
    """
    if not log_dir:
        logging.warning("log_dir is empty, skipping log save")
        return

    shutil.copy2(aisbench_log, log_dir)

    source_file = aisbench_log
    target_file = "aisbench_all.log"

    try:
        with open(source_file, "r", encoding="utf-8") as src:
            content = src.read()

        with open(target_file, "a", encoding="utf-8") as tgt:
            tgt.write(f"\n\n{'='*50}\n")
            tgt.write(f"{'='*50}\n\n")
            tgt.write(content)
            tgt.write(f"\n\n{'='*50}\n")
            tgt.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            tgt.write(f"{'='*50}\n")

        logging.info(f"Successfully appended {source_file} content to {target_file}")

    except FileNotFoundError:
        logging.error(f"Error: File not found, please check file path")
    except Exception as e:
        logging.error(f"Error occurred: {e}")


def save_csv(perf_result: List, filename: str):
    """
    Save performance data to CSV file

    Args:
        perf_result: Performance data list
        filename: Output CSV filename
    """
    headers = [
        "current_time",
        "input_len",
        "output_len",
        "total_req",
        "max_cc",
        "cc",
        "rr",
        "TTFT avg",
        "TTFT P90",
        "TPOT avg",
        "TPOT SLO_P90",
        "E2E_time",
        "output_throughput",
        "single_output_throughput",
        "E2E_throughput",
        "single_E2E_throughput",
        "qps",
        "qpm",
        "input_token_throughput",
        "prefill_token_throughput",
    ]

    file_exists = os.path.exists(filename)

    try:
        if file_exists:
            df_existing = pd.read_csv(filename)
            logging.info("File exists, reading existing data")
            new_row = pd.DataFrame([perf_result], columns=headers)
            df_updated = pd.concat([df_existing, new_row], ignore_index=True)
            df_updated.to_csv(filename, index=False)
            logging.info("Successfully appended new row")
        else:
            df_new = pd.DataFrame([perf_result], columns=headers)
            df_new.to_csv(filename, index=False)
            logging.info("Created new file and wrote data")

    except Exception as e:
        logging.error(f"Operation failed: {e}")
