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
from dataclasses import dataclass
from itertools import accumulate
from typing import Dict, List, Optional, Tuple, Union

import torch
from vllm.platforms import current_platform

from ucm.logger import init_logger

logger = init_logger(__name__)


def execute_command(cmd_list):
    with subprocess.Popen(
        cmd_list, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as p:
        out, err = p.communicate(timeout=1000)
    res = out.decode()
    return res


@dataclass
class DeviceInfo:
    _info_line: str = ""
    npu_id: int = 0
    chip_id: int = 0
    chip_logic_id: Union[int, str] = 0
    chip_name: str = ""

    def __post_init__(self):
        self.npu_id, self.chip_id, self.chip_logic_id, self.chip_name = (
            self._info_line.strip().split(None, 3)
        )
        self.npu_id = int(self.npu_id)
        self.chip_id = int(self.chip_id)
        if self.chip_logic_id.isnumeric():
            self.chip_logic_id = int(self.chip_logic_id)


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

    def _get_visible_devices(self) -> List[int]:
        visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES") or os.environ.get(
            "ASCEND_VISIBLE_DEVICES"
        )
        if visible:
            return [int(x.strip()) for x in visible.split(",") if x.strip()]

        try:
            return sorted(list(self._get_device_map_info().keys()))
        except Exception:
            return list(range(torch.npu.device_count()))

    def _get_device_map_info(self) -> Dict[int, DeviceInfo]:
        device_map_info = {}
        device_map = execute_command(["npu-smi", "info", "-m"]).strip().split("\n")[1:]
        for line in device_map:
            line = line.strip()
            if not line:
                continue
            try:
                device_info = DeviceInfo(line)
                if isinstance(device_info.chip_logic_id, int):
                    device_map_info[device_info.chip_logic_id] = device_info
            except (ValueError, IndexError):
                continue
        return device_map_info

    def _get_pcie_info(
        self, devices: List[int], keyword: str = "PCIeBusInfo"
    ) -> Dict[int, str]:
        device_map_info = self._get_device_map_info()
        device_pcie_tbl = {}

        for device in devices:
            device_info = device_map_info.get(device)
            if not device_info:
                warn_msg = (
                    f"cannot get device info for device {device}, "
                    f"skipping PCIe binding for this device."
                )
                logger.warning(warn_msg)
                raise RuntimeError(warn_msg)

            pcie_info = (
                execute_command(
                    [
                        "npu-smi",
                        "info",
                        "-t",
                        "board",
                        "-i",
                        f"{device_info.npu_id}",
                        "-c",
                        f"{device_info.chip_id}",
                    ]
                )
                .strip()
                .split("\n")
            )

            for line_raw in pcie_info:
                line = "".join(line_raw.split())
                if line.startswith(keyword):
                    device_pcie_tbl[device] = line[len(keyword) + 1 :]
                    break

        return device_pcie_tbl

    def _get_numa_info(
        self, pcie_tbl: Dict[int, str]
    ) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
        device_numa_tbl: Dict[int, int] = {}
        numa_devices_tbl: Dict[int, List[int]] = {}

        for device, pcie_no in pcie_tbl.items():
            # 1. try sysfs first
            numa_path = f"/sys/bus/pci/devices/{pcie_no}/numa_node"
            try:
                if os.path.exists(numa_path):
                    with open(numa_path) as f:
                        numa_id = int(f.read().strip())
                    if numa_id >= 0:
                        device_numa_tbl[device] = numa_id
                        numa_devices_tbl.setdefault(numa_id, []).append(device)
                        continue
            except Exception as e:
                logger.warning(
                    f"[NUMA] failed to read sysfs NUMA node for device {device}, PCI {pcie_no}: {e}"
                )

            # 2. optional lspci fallback
            try:
                out = execute_command(["/usr/bin/lspci", "-s", pcie_no, "-vvv"])
                m = re.search(r"NUMA\s*node\s*:\s*(\d+)", out, re.IGNORECASE)
                if m:
                    numa_id = int(m.group(1))
                    device_numa_tbl[device] = numa_id
                    numa_devices_tbl.setdefault(numa_id, []).append(device)
                    continue
            except Exception:
                pass

        return device_numa_tbl, numa_devices_tbl

    def _get_numa_info_v2(
        self, devices: List[int], keyword: str = "NUMAnode(s)"
    ) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
        """
        Fallback when real NPU->NUMA mapping is unavailable:
        distribute visible devices evenly across NUMA nodes.
        """
        numa_nodes = 1
        numa_info = execute_command(["lscpu"]).split("\n")
        for line_raw in numa_info:
            line = "".join(line_raw.split())
            if keyword not in line:
                continue
            try:
                numa_nodes = int(line.split(":")[-1])
            except Exception:
                numa_nodes = 1
            break

        device_per_numa, tail_device = divmod(len(devices), numa_nodes)
        device_count_per_numa_list = [
            device_per_numa + (i < tail_device) for i in range(numa_nodes)
        ]

        ends = list(accumulate(device_count_per_numa_list))
        starts = [0] + ends[:-1]

        numa_devices_tbl = {
            ind: devices[start:end]
            for ind, (start, end) in enumerate(zip(starts, ends))
        }

        device_numa_tbl = {
            device: numa
            for numa, _devices in numa_devices_tbl.items()
            for device in _devices
        }

        return device_numa_tbl, numa_devices_tbl

    def _get_cpu_info(
        self, numa_ids: List[int], keyword1: str = "NUMAnode", keyword2: str = "CPU(s)"
    ) -> Dict[int, List[int]]:
        cpu_idx_tbl: Dict[int, List[int]] = {}
        numa_keywords = [f"{keyword1}{idx}{keyword2}" for idx in numa_ids]
        cpu_info = execute_command(["lscpu"]).split("\n")

        for line_raw in cpu_info:
            line = "".join(line_raw.split())
            if not any(line.startswith(word) for word in numa_keywords):
                continue

            split_info = line.split(":")
            cpu_id_ranges = split_info[-1].split(",")

            ranges: List[int] = []
            for range_str in cpu_id_ranges:
                endpoints = range_str.split("-")
                if len(endpoints) == 2:
                    start, end = map(int, endpoints)
                    if start > end:
                        start, end = end, start
                    ranges.extend(range(start, end + 1))
                elif len(endpoints) == 1 and endpoints[0] != "":
                    ranges.append(int(endpoints[0]))
                else:
                    warn_msg = (
                        "cannot obtain CPU range for NUMA while executing `lscpu`."
                    )
                    logger.warning(warn_msg)
                    raise RuntimeError(warn_msg)

            numa_id = int(split_info[0].replace(keyword1, "").replace(keyword2, ""))
            cpu_idx_tbl[numa_id] = ranges

        return cpu_idx_tbl

    def _to_cpulist_str(self, cores: List[int]) -> Optional[str]:
        if not cores:
            return None

        cores = sorted(set(cores))
        parts = []
        s = e = cores[0]

        for c in cores[1:]:
            if c == e + 1:
                e = c
            else:
                parts.append(f"{s}-{e}" if s != e else str(s))
                s = e = c
        parts.append(f"{s}-{e}" if s != e else str(s))
        return ",".join(parts)

    def _fallback_cpu_affinity(self, local_rank: int) -> Optional[str]:
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

            cpu_affinity = self._to_cpulist_str(sliced)
            logger.warning(
                f"[CPU Affinity] fallback to sliced allowed CPUs for npu rank={local_rank}: "
                f"{cpu_affinity}"
            )
            return cpu_affinity

        except Exception as e:
            logger.error(f"get npu cpu affinity fallback failed: {e}")
            return None

    def get_cpu_affinity(self, local_rank: int) -> Optional[str]:
        """
        NPU path:
        1. NPU -> PCIe -> NUMA -> cpulist
        2. fallback: split current allowed CPUs by local_rank
        """
        device_id = self._get_device_id(local_rank)

        try:
            devices = self._get_visible_devices()

            # NPU -> PCIe
            device_pcie_tbl = self._get_pcie_info(devices)

            # PCIe -> NUMA
            device_numa_tbl, numa_devices_tbl = self._get_numa_info(device_pcie_tbl)
            if not device_numa_tbl or not numa_devices_tbl:
                logger.warning(
                    "[CPU Affinity] failed to get real NPU->NUMA mapping, "
                    "fallback to evenly distributed NUMA mapping."
                )
                device_numa_tbl, numa_devices_tbl = self._get_numa_info_v2(devices)

            numa_id = device_numa_tbl.get(device_id)
            if numa_id is None:
                logger.warning(
                    f"[CPU Affinity] cannot find NUMA node for NPU device {device_id}"
                )
                return self._fallback_cpu_affinity(local_rank)

            # NUMA -> CPU list
            cpu_idx_tbl = self._get_cpu_info(list(numa_devices_tbl.keys()))
            cores = cpu_idx_tbl.get(numa_id)
            if not cores:
                logger.warning(
                    f"[CPU Affinity] cannot find CPU list for NUMA node {numa_id}"
                )
                return self._fallback_cpu_affinity(local_rank)

            cpu_affinity = self._to_cpulist_str(cores)
            logger.info(
                f"[CPU Affinity] NPU device={device_id}, numa_id={numa_id}, "
                f"cpu_affinity={cpu_affinity}"
            )
            return cpu_affinity

        except Exception as e:
            logger.warning(f"get npu cpu affinity from numa failed: {e}")
            return self._fallback_cpu_affinity(local_rank)


def create_device() -> Optional[Device]:
    if current_platform.is_cuda_alike():
        return CudaDevice()

    if current_platform.device_type == "npu":
        return NpuDevice()

    return None
