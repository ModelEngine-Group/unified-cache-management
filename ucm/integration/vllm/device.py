# -*- coding: utf-8 -*-
"""
Event-based sync between Python compute stream and C++ cache stream.

When dump_data is called, the cache's C++ stream does D2H from device memory.
We must ensure the Python compute stream has finished writing KVCache before
the cache reads. Event sync: record event on compute stream, pass to C++,
cache stream waits for event before D2H. This avoids blocking the CPU.
"""
from abc import ABC, abstractmethod
from typing import Optional

import torch
from vllm.platforms import current_platform

from ucm.logger import init_logger

logger = init_logger(__name__)


class Device(ABC):
    def __init__(self):
        self.events = []

    @abstractmethod
    def get_event_handle(self) -> int:
        """Return event handle for stream sync. 0 means no event (use synchronize instead)."""
        pass

    @abstractmethod
    def synchronize(self):
        pass

    @abstractmethod
    def destroy_event_handles(self):
        pass


class CudaDevice(Device):
    def __init__(self):
        super().__init__()

    def get_event_handle(self) -> int:
        try:
            cuda_event = torch.cuda.Event(enable_timing=False)
            stream = torch.cuda.current_stream()
            cuda_event.record(stream)
            handle = int(cuda_event.cuda_event)
            if handle is None or handle == 0:
                return 0
            self.events.append(cuda_event)
            return handle
        except Exception as e:
            logger.error(f"get cuda event handle failed. {e}")
            return 0

    def synchronize(self):
        torch.cuda.current_stream().synchronize()

    def destroy_event_handles(self):
        self.events.clear()


class NpuDevice(Device):
    def __init__(self):
        super().__init__()

    def get_event_handle(self) -> int:
        import acl
        import torch_npu

        try:
            stream = torch_npu.npu.current_stream().npu_stream
            event, ret = acl.rt.create_event()
            if ret != 0:
                logger.error(f"acl create_event failed: {ret}")
                return 0
            self.events.append(event)
            ret = acl.rt.record_event(event, stream)
            if ret != 0:
                logger.error(f"acl record_event failed: {ret}")
                return 0
            handle = int(event)
            if not handle:
                return 0
            return handle
        except Exception as e:
            logger.error(f"get npu event handle failed. {e}")
            return 0

    def synchronize(self):
        torch.npu.current_stream().synchronize()

    def destroy_event_handles(self):
        import acl

        for event in self.events:
            try:
                acl.rt.destroy_event(event)
            except Exception as e:
                logger.error(f"destroy npu event failed. {e}")
                continue
        self.events.clear()


def create_device() -> Optional[Device]:
    if current_platform.is_cuda_alike():
        return CudaDevice()

    if current_platform.device_type == "npu":
        return NpuDevice()

    return None
