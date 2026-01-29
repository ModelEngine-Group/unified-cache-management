import argparse
import logging
import os
import random
import sys
import threading
import time

# Configuration
LOG_FILE = "app.log"
MAX_BYTES = 1048576 * 5
BACKUP_COUNT = 3  # Keep up to 5 backup logs (app.log.1 to app.log.5)
# Add project root to Python path to enable importing ucm modules
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def get_python_file_logger():
    from logging.handlers import RotatingFileHandler

    logger = logging.getLogger("python_logger_perf")
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(
        LOG_FILE,
        mode="a",  # Append mode
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding=None,
        delay=False,
    )
    handler.setLevel(logging.INFO)  # Set the handler's specific level

    formatter = logging.Formatter(
        "[%(asctime)s.%(msecs)06d][%(name)s][%(levelname).1s] %(message)s [%(process)d,%(thread)d][%(filename)s:%(lineno)d]",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger


def get_python_logger():
    logger = logging.getLogger("python_logger_perf")
    logger.setLevel(logging.INFO)
    # Ensure we don't pay for I/O; NullHandler drops all records.
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


def get_pybind_logger():
    try:
        from ucm.logger import init_logger  # type: ignore

        logger = init_logger()
    except Exception as exc:  # pragma: no cover - import guard
        print(f"Failed to import 'spdlog_logger' pybind module: {exc}", file=sys.stderr)
        sys.exit(1)

    return logger


def run_log(logger, n: int, num_threads: int = 4):
    # Random log level distribution: 89% info, 10% warning, 1% error
    levels = ["info", "warning", "error"]
    weights = [0.89, 0.10, 0.01]
    msg = "test log message"

    def log_worker(thread_id: int, iterations: int):
        """Worker function that performs logging in a thread"""
        for _ in range(iterations):
            selected_level = random.choices(levels, weights=weights)[0]
            if selected_level == "info":
                logger.info(msg)
            elif selected_level == "warning":
                logger.warning(msg)
            elif selected_level == "error":
                logger.error(msg)

    # Create and start threads
    threads = []
    start = time.perf_counter()

    for i in range(num_threads):
        # Distribute remainder iterations across first few threads
        thread_iterations = n // num_threads
        thread = threading.Thread(target=log_worker, args=(i, thread_iterations))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    end = time.perf_counter()

    return end - start


def bench_python_logging(n: int, num_threads: int = 4) -> float:
    logger = get_python_logger()
    return run_log(logger, n, num_threads)


def bench_python_file_logging(n: int, num_threads: int = 4) -> float:
    logger = get_python_file_logger()
    return run_log(logger, n, num_threads)


def bench_pybind_logging(n: int, num_threads: int = 4) -> float:
    logger = get_pybind_logger()
    return run_log(logger, n, num_threads)


def run_benchmark(n: int, itr: int, warmup: int = 10_000) -> None:
    # Warm up both paths to reduce first‑call overhead / JIT effects.
    _ = bench_python_logging(warmup, 1)
    _ = bench_pybind_logging(warmup, 1)

    py_time = bench_python_logging(n, 4)
    cpp_time = bench_pybind_logging(n, 4)

    py_per_call_ns = py_time / n * 1e9
    cpp_per_call_ns = cpp_time / n * 1e9

    overhead = cpp_per_call_ns - py_per_call_ns
    overhead_pct = (
        (cpp_per_call_ns / py_per_call_ns - 1.0) * 100.0
        if py_per_call_ns > 0
        else float("inf")
    )
    print(f"Test run: {itr+1}, iterations: {n}")
    print(
        f"  pybind11 (spdlog_logger): {cpp_time:.6f}s total, {cpp_per_call_ns:.1f} ns / call"
    )
    print(f"  Python logging: {py_time:.6f}s total, {py_per_call_ns:.1f} ns / call")
    print(f"  Difference: {overhead:.1f} ns / call ({overhead_pct:.2f} % overhead)\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare performance of Python logging vs pybind11-based spdlog_logger."
    )
    parser.add_argument(
        "-n",
        "--iterations",
        type=int,
        default=100_000,
        help="Number of log calls per test case (default: 10000).",
    )
    parser.add_argument(
        "-l",
        "--level",
        choices=["debug", "info", "warning", "error", "all"],
        default="all",
        help="Log level to benchmark (default: all).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10_000,
        help="Number of warmup iterations for each logger (default: 1000).",
    )
    args = parser.parse_args()

    levels = (
        ["debug", "info", "warning", "error"] if args.level == "all" else [args.level]
    )

    print("Benchmarking python logging(null handler) vs pybind11 spdlog_logger.\n")
    print(f"Iterations per level: {args.iterations}, warmup: {args.warmup}\n")

    for itr in range(5):
        run_benchmark(args.iterations, itr, warmup=args.warmup)


if __name__ == "__main__":
    main()
