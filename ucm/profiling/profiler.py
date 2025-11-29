import time
from typing import Any
import json
import queue
import threading
from ucm.logger import init_logger

logger = init_logger(__name__)

class Profiler:
    def __init__(
        self,
        launch_config: dict[str, Any],
        block_size: int,
        rank: int,
    ) -> None:
        self.block_size = block_size
        self.rank = rank

        self.record_config = launch_config.get("record_config", {})
        self.enable_record: bool = self.record_config.get("enable", False) and self.rank == 0
        if self.enable_record:
            self.write_thread = threading.Thread(
                target=self._async_record_loop, daemon=True
            )
            self.write_thread.start()

    def log_operation(self, operation_data: dict[str, Any]) -> None:
        """Record operation log (non-blocking)"""
        if not self.enable_record:
            return

        default_data = {
            "timestamp": time.time(),
            "op_type": "None",
            "block_size": self.block_size,
        }
        log_entry = {**default_data, **operation_data}

        try:
            self.log_queue.put_nowait(log_entry)
        except queue.Full:
            logger.error(
                f"Log queue is full, dropping one log: {log_entry.get('request_id')}"
            )

    def _async_record_loop(self):
        self.log_queue = queue.Queue(maxsize=10000)  # Max cache: 10000 entries
        log_path = self.record_config.get(
            "log_path", "/vllm-workspace/ucm_logs/ucm_ops.log"
        )
        flush_size = self.record_config.get("flush_size", 100)
        flush_interval = self.record_config.get("flush_interval", 5.0)
        batch_buffer = []
        last_flush_time = time.time()
        while True:
            try:
                # Get log from queue (1 second timeout)
                is_flush = False
                current_time = time.time()
                log_entry = self.log_queue.get(timeout=1.0)
                batch_buffer.append(log_entry)

                # Flush if conditions are met
                if (
                    len(batch_buffer) >= flush_size
                    or (current_time - last_flush_time) >= flush_interval
                ):
                    is_flush = True
                    last_flush_time = current_time
                self.log_queue.task_done()
            except queue.Empty:
                if (current_time - last_flush_time) >= flush_interval:
                    last_flush_time = current_time
            except Exception as e:
                logger.error(f"Log thread exception: {str(e)}")

            if is_flush:
                with open(log_path, "a", encoding="utf-8") as f:
                    for log_entry in batch_buffer:
                        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                    batch_buffer.clear()