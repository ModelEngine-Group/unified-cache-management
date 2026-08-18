from pathlib import Path
from typing import Any


def stack_yuanrong_store(config: dict[str, object], pipeline: Any) -> None:
    pipeline.Stack(
        "YuanRong",
        str(Path(__file__).resolve().parent / "libyuanrongstore.so"),
        config,
    )

    log_path = config.get("yuanrong_resource_log_path", "")
    enabled = bool(config.get("yuanrong_resource_metrics_enable", bool(log_path)))
    if not enabled or not log_path or int(config.get("device_id", -1)) >= 0:
        return

    from .resource_reporter import start_yuanrong_resource_reporter

    start_yuanrong_resource_reporter(config)
