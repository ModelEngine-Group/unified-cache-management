# -*- coding: utf-8 -*-
"""
Event-based sync between Python compute stream and C++ cache stream.

When dump_data is called, the cache's C++ stream does D2H from device memory.
We must ensure the Python compute stream has finished writing KVCache before
the cache reads. Event sync: record event on compute stream, pass to C++,
cache stream waits for event before D2H. This avoids blocking the CPU.
"""
import os
import re
import subprocess
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

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

    @abstractmethod
    def get_cpu_affinity(self, local_rank: int) -> Optional[str]:
        """
        Return CPU affinity as a cpulist string, e.g. "0-43,88-131".
        """
        pass

    def split_cores(self, local_rank: int) -> Tuple[List[int], List[int]]:
        """
        Shared split logic for both CUDA and NPU.
        Split each cpulist segment evenly and keep at least one core for worker.
        """
        worker_cores, store_cores = [], []
        cpu_affinity = self.get_cpu_affinity(local_rank)

        if not cpu_affinity:
            return worker_cores, store_cores

        try:
            for part in cpu_affinity.split(","):
                part = part.strip()
                if not part:
                    continue

                if "-" in part:
                    a, b = map(int, part.split("-", 1))
                    if a > b:
                        a, b = b, a
                    seg = list(range(a, b + 1))
                else:
                    seg = [int(part)]

                mid = max(1, len(seg) // 2)
                worker_cores.extend(seg[:mid])
                store_cores.extend(seg[mid:])

            if not worker_cores:
                cores = sorted(os.sched_getaffinity(0))
                if cores:
                    worker_cores = [cores[0]]
                    store_cores = cores[1:]

        except Exception as e:
            logger.error(f"split cores failed, cpu_affinity={cpu_affinity}: {e}")
            return [], []

        logger.info(
            f"[CPU Affinity] rank={local_rank}, cpu_affinity={cpu_affinity}\n"
            f"[worker_cores]={worker_cores}\n"
            f"[store_cores]={store_cores}"
        )
        return worker_cores, store_cores


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

    def get_cpu_affinity(self, local_rank: int) -> Optional[str]:
        """
        CUDA path:
        1. GPU -> PCI -> NUMA -> cpulist
        2. fallback: split current allowed CPUs by local_rank
        """
        try:
            prop = torch.cuda.get_device_properties(local_rank)
            pci_bus_id = (
                f"{prop.pci_domain_id:04x}:"
                f"{prop.pci_bus_id:02x}:"
                f"{prop.pci_device_id:02x}.0"
            )

            numa_path = f"/sys/bus/pci/devices/{pci_bus_id}/numa_node"
            if os.path.exists(numa_path):
                with open(numa_path) as f:
                    numa_node = int(f.read().strip())

                if numa_node >= 0:
                    cpu_list_path = f"/sys/devices/system/node/node{numa_node}/cpulist"
                    if os.path.exists(cpu_list_path):
                        with open(cpu_list_path) as f:
                            return f.read().strip()
        except Exception as e:
            logger.warning(f"get cuda cpu affinity from numa failed: {e}")

        try:
            cores = sorted(os.sched_getaffinity(0))
            if not cores:
                return None

            visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            total_devices = (
                len([x.strip() for x in visible.split(",") if x.strip()])
                if visible
                else torch.cuda.device_count()
            )

            if total_devices <= 0 or local_rank < 0 or local_rank >= total_devices:
                logger.warning(
                    f"[CPU Affinity] invalid cuda fallback split: "
                    f"local_rank={local_rank}, total_devices={total_devices}"
                )
                return None

            base = len(cores) // total_devices
            extra = len(cores) % total_devices
            start = local_rank * base + min(local_rank, extra)
            length = base + (1 if local_rank < extra else 0)
            sliced = cores[start : start + length]

            if not sliced:
                return None

            parts = []
            s = e = sliced[0]
            for c in sliced[1:]:
                if c == e + 1:
                    e = c
                else:
                    parts.append(f"{s}-{e}" if s != e else str(s))
                    s = e = c
            parts.append(f"{s}-{e}" if s != e else str(s))

            cpu_affinity = ",".join(parts)
            logger.warning(
                f"[CPU Affinity] fallback to sliced allowed CPUs for cuda rank={local_rank}: "
                f"{cpu_affinity}"
            )
            return cpu_affinity

        except Exception as e:
            logger.error(f"get cuda cpu affinity fallback failed: {e}")
            return None


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
        self.events.clear()

    def _get_device_id(self, local_rank: int) -> int:
        visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES") or os.environ.get(
            "ASCEND_VISIBLE_DEVICES"
        )
        if not visible:
            return local_rank

        dev_list = [int(x.strip()) for x in visible.split(",") if x.strip()]
        return dev_list[local_rank] if local_rank < len(dev_list) else local_rank

    def get_cpu_affinity(self, local_rank: int) -> Optional[str]:
        """
        NPU path:
        1. try `npu-smi info -t topo`
        2. fallback: split current allowed CPUs by local_rank
        """
        device_id = self._get_device_id(local_rank)

        try:
            result = subprocess.run(
                ["npu-smi", "info", "-t", "topo"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )

            pattern = (
                rf"^\s*NPU{device_id}\s+.*?((?:\d+(?:-\d+)?)(?:,\d+(?:-\d+)?)*)\s*$"
            )
            for line in result.stdout.splitlines():
                m = re.match(pattern, line)
                if m:
                    return m.group(1)
        except Exception as e:
            logger.warning(f"get npu cpu affinity from topo failed: {e}")

        try:
            cores = sorted(os.sched_getaffinity(0))
            if not cores:
                return None

            visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES") or os.environ.get(
                "ASCEND_VISIBLE_DEVICES"
            )
            total_devices = (
                len([x.strip() for x in visible.split(",") if x.strip()])
                if visible
                else torch.npu.device_count()
            )

            if total_devices <= 0 or local_rank < 0 or local_rank >= total_devices:
                logger.warning(
                    f"[CPU Affinity] invalid npu fallback split: "
                    f"local_rank={local_rank}, total_devices={total_devices}"
                )
                return None

            base = len(cores) // total_devices
            extra = len(cores) % total_devices
            start = local_rank * base + min(local_rank, extra)
            length = base + (1 if local_rank < extra else 0)
            sliced = cores[start : start + length]

            if not sliced:
                return None

            parts = []
            s = e = sliced[0]
            for c in sliced[1:]:
                if c == e + 1:
                    e = c
                else:
                    parts.append(f"{s}-{e}" if s != e else str(s))
                    s = e = c
            parts.append(f"{s}-{e}" if s != e else str(s))

            cpu_affinity = ",".join(parts)
            logger.warning(
                f"[CPU Affinity] fallback to sliced allowed CPUs for npu rank={local_rank}: "
                f"{cpu_affinity}"
            )
            return cpu_affinity

        except Exception as e:
            logger.error(f"get npu cpu affinity fallback failed: {e}")
            return None


def create_device() -> Optional[Device]:
    if current_platform.is_cuda_alike():
        return CudaDevice()

    if current_platform.device_type == "npu":
        return NpuDevice()

    return None
